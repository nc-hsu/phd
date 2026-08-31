"""One-off manifest re-stamps for the AvgSA_03 record-selection stripes.

Both migrations here exist for the same reason: a refactor changed *what a stripe's
fingerprint names* without changing a single byte of the data a stripe was selected
from. Left alone that marks every already-selected stripe stale and triggers hours of
GCIM + selection for nothing. So the manifests are re-stamped in place instead --
each migration only where it can prove the underlying data is unchanged.

``--to shards`` (historical, completed)
    ``disagg_data_file`` (the 2.3 GB monolith) -> ``disagg_site_shard`` (that site's
    38 MB shard), and ``disagg_shards_digest`` for the batch artifacts. Honest
    because it refuses any manifest whose recorded ``disagg_data_file`` hash differs
    from the monolith sha256 recorded in ``_split_provenance.json``.

``--to content-hashes`` (current, the default)
    ``disagg_site_shard`` -> ``disagg_stripe_data`` and ``disagg_stats_file`` ->
    ``disagg_stats_row``: whole-file hashes become content hashes of the single
    ``(site, iml)`` DataFrame and the single stats row that stripe actually depends
    on. Sharding by site was not enough on its own -- an OpenQuake ``iml_disagg``
    run covers all 60 sites at ONE iml, so adding one MSA stripe rewrites every
    shard and appends rows to the shared stats pickle, marking all ~500 stripes
    stale. Honest because it refuses any manifest whose recorded
    ``disagg_site_shard`` / ``disagg_stats_file`` hashes do not match the files
    currently on disk: only then is that manifest known to describe *this* data, so
    hashes derived from it are true. A manifest that fails the check is genuinely
    stale and is left for the pipeline to recompute.

    Run it while the shard set the stripes were selected from is still on disk --
    i.e. BEFORE pulling a rebuilt one -- and after backfilling its
    ``_content_hashes.json`` (``disagg_shards.write_content_hashes``).

The stripe pickles and their ``.manifest.json`` sidecars are git-tracked, so
``git diff`` shows exactly what changed and ``git checkout`` undoes it.

    python -m phd_project.scripts.WP1_ground_motion_set.migrate_stripe_manifests
    python -m phd_project.scripts.WP1_ground_motion_set.migrate_stripe_manifests --write
    python -m phd_project.scripts.WP1_ground_motion_set.migrate_stripe_manifests --to shards
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pickle

from phd_project.config.config import load_config
from phd_project.scripts import disagg_shards
from phd_project.scripts.cache_utils import (
    _manifest_path,
    file_hash,
    fingerprint,
)
from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    stats_row,
    stripe_pickle_path,
)
from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
    DISAGG_IMT,
    wanted_stripe_keys,
)

STRIPE_RE = re.compile(r"^site_(\d+)__stripe_iml_.*__gm_selection\.pickle\.manifest\.json$")

# Skipped / untouched / migrated classifications
MIGRATE_STRIPE = "stripe"
MIGRATE_BATCH = "batch"
NO_DISAGG_KEY = "no disagg_data_file entry (nothing to migrate)"
ALREADY_DONE = "already on the sharded scheme"
HASH_MISMATCH = "disagg_data_file hash != split source -- genuinely stale, left alone"

# --- content-hash migration ------------------------------------------------
CH_DONE = "already on the per-stripe content-hash scheme"
CH_NOT_STRIPE = "not a stripe manifest (batch artifact -- left on whole-set hashes)"
CH_NO_KEYS = "no disagg_site_shard/disagg_stats_file entries (nothing to migrate)"
CH_SHARD_MISMATCH = ("disagg_site_shard hash != the shard on disk -- genuinely stale, "
                     "left alone")
CH_STATS_MISMATCH = ("disagg_stats_file hash != the stats pickle on disk -- genuinely "
                     "stale, left alone")
CH_NO_ROW = "no disagg_stats row for this (site, iml) -- left alone"
CH_MIGRATE = "stripe"


def _classify(mf: Path, inputs: dict, source_sha: str) -> tuple[str, int | None]:
    if "disagg_site_shard" in inputs or "disagg_shards_digest" in inputs:
        return ALREADY_DONE, None
    entry = inputs.get("disagg_data_file")
    if entry is None:
        return NO_DISAGG_KEY, None
    if entry.get("hash") != source_sha:
        return HASH_MISMATCH, None
    m = STRIPE_RE.match(mf.name)
    if m:
        return MIGRATE_STRIPE, int(m.group(1))
    return MIGRATE_BATCH, None


def migrate(*, write: bool = False) -> int:
    cfg = load_config()
    shard_dir = cfg["proc_data"]["AvgSA_03_disagg_data_shards"]

    prov = disagg_shards.read_split_provenance(shard_dir)
    if prov is None:
        print(f"ERROR: no {disagg_shards.SPLIT_PROVENANCE_NAME} in {shard_dir}.\n"
              f"       Run shard_disagg_data.py --write first.")
        return 1
    source_sha = prov["source_sha256"]
    print(f"shard dir : {shard_dir}")
    print(f"split from monolith sha256 {source_sha}\n")

    # The whole-set digest is shared by every batch artifact; compute it once.
    batch_entry = fingerprint(
        disagg_shards_digest=disagg_shards.shards_digest(shard_dir))["disagg_shards_digest"]

    roots = [cfg["results"]["AvgSA_03_record_selection"], cfg["proc_data"]["gm_selection"]]
    counts: dict[str, int] = {}
    changed = []

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for mf in sorted(root.glob("*.manifest.json")):
            manifest = json.loads(mf.read_text())
            inputs = manifest.get("inputs", {})
            kind, site = _classify(mf, inputs, source_sha)
            counts[kind] = counts.get(kind, 0) + 1

            if kind == MIGRATE_STRIPE:
                new = fingerprint(
                    disagg_site_shard=disagg_shards.shard_path(shard_dir, site))
                entry = {"disagg_site_shard": new["disagg_site_shard"]}
            elif kind == MIGRATE_BATCH:
                entry = {"disagg_shards_digest": batch_entry}
            else:
                if kind == HASH_MISMATCH:
                    print(f"  [left alone] {mf.name}: {HASH_MISMATCH}")
                continue

            # Rebuild inputs preserving key order, swapping the disagg entry in place.
            new_inputs = {}
            for k, v in inputs.items():
                if k == "disagg_data_file":
                    new_inputs.update(entry)
                else:
                    new_inputs[k] = v
            manifest["inputs"] = new_inputs
            manifest.setdefault("_meta", {}).update({
                "migrated_at": datetime.now(timezone.utc).isoformat(),
                "migrated_from": "disagg_data_file (monolith) -> per-site shards; "
                                 f"monolith sha256 {source_sha}",
            })
            changed.append((mf, manifest))

    print("\nclassification:")
    for k, n in sorted(counts.items()):
        print(f"  {n:>4}  {k}")
    print(f"\n{len(changed)} manifest(s) to re-stamp.")

    if not write:
        print("\n-- dry run; nothing written. Re-run with --write to apply.")
        return 0

    for mf, manifest in changed:
        mf.write_text(json.dumps(manifest, indent=2))
    print(f"re-stamped {len(changed)} manifest(s).")
    if counts.get(HASH_MISMATCH):
        print(f"NOTE: {counts[HASH_MISMATCH]} manifest(s) were left alone and will "
              f"recompute -- see the [left alone] lines above.")
    return 0


def _content_hash_reason(inputs: dict, shard_sha: str, stats_sha: str) -> str:
    """Why this manifest can (or cannot) be re-stamped onto the content-hash scheme.

    The guard: re-stamp only when the manifest's recorded whole-file hashes still
    match the files on disk. That is what makes the swap honest -- it proves the
    manifest describes *this* shard set and *this* stats pickle, so the per-stripe
    hashes derived from them are the hashes of the data the stripe was really
    selected from. Anything else is genuinely stale and must recompute.
    """
    if "disagg_stripe_data" in inputs or "disagg_stats_row" in inputs:
        return CH_DONE
    if "disagg_site_shard" not in inputs and "disagg_stats_file" not in inputs:
        return CH_NO_KEYS
    if inputs.get("disagg_site_shard", {}).get("hash") != shard_sha:
        return CH_SHARD_MISMATCH
    if inputs.get("disagg_stats_file", {}).get("hash") != stats_sha:
        return CH_STATS_MISMATCH
    return CH_MIGRATE


def migrate_to_content_hashes(*, write: bool = False) -> int:
    """Re-stamp stripe manifests from whole-file hashes onto per-stripe content hashes.

    Iterates the wanted ``(site, iml)`` key set -- the same set the notebooks check
    -- rather than globbing filenames, because the stripe filename tag is rounded to
    3 dp and cannot recover the native iml the fingerprint needs.
    """
    cfg = load_config()
    shard_dir = Path(cfg["proc_data"]["AvgSA_03_disagg_data_shards"])
    stats_fp = Path(cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"])
    result_folder = Path(cfg["results"]["AvgSA_03_record_selection"])

    if not disagg_shards.content_hashes_path(shard_dir).is_file():
        print(f"ERROR: no {disagg_shards.CONTENT_HASHES_NAME} in {shard_dir}.\n"
              f"       Backfill it first:\n"
              f"         from phd_project.scripts import disagg_shards\n"
              f"         disagg_shards.write_content_hashes(shard_dir)")
        return 1

    print(f"shard dir  : {shard_dir}")
    print(f"stats file : {stats_fp.name}")
    print(f"results    : {result_folder}\n")

    with open(stats_fp, "rb") as f:
        disagg_stats = pickle.load(f)
    stats_sha = file_hash(stats_fp)

    wanted = wanted_stripe_keys()
    counts: dict[str, int] = {}
    changed = []
    missing_pickle = 0

    for site, iml in wanted:
        stripe_fp = stripe_pickle_path(result_folder, site, iml)
        mf = _manifest_path(stripe_fp)
        if not stripe_fp.is_file() or not mf.is_file():
            missing_pickle += 1
            continue

        manifest = json.loads(mf.read_text())
        inputs = manifest.get("inputs", {})
        shard_sha = file_hash(disagg_shards.shard_path(shard_dir, site))
        kind = _content_hash_reason(inputs, shard_sha, stats_sha)

        if kind == CH_MIGRATE:
            row = stats_row(disagg_stats, site, DISAGG_IMT, iml)
            if row.empty:
                kind = CH_NO_ROW
        counts[kind] = counts.get(kind, 0) + 1
        if kind != CH_MIGRATE:
            if kind in (CH_SHARD_MISMATCH, CH_STATS_MISMATCH, CH_NO_ROW):
                print(f"  [left alone] {mf.name}: {kind}")
            continue

        # Route through fingerprint() so the entries are byte-identical to what
        # stripe_input_fingerprint will compute on the next run.
        new = fingerprint(
            disagg_stripe_data=disagg_shards.stripe_content_hash(
                shard_dir, site, DISAGG_IMT, iml),
            disagg_stats_row=row,
        )

        # Rebuild inputs preserving key order, swapping each entry in place.
        new_inputs = {}
        for k, v in inputs.items():
            if k == "disagg_site_shard":
                new_inputs["disagg_stripe_data"] = new["disagg_stripe_data"]
            elif k == "disagg_stats_file":
                new_inputs["disagg_stats_row"] = new["disagg_stats_row"]
            else:
                new_inputs[k] = v
        manifest["inputs"] = new_inputs
        manifest.setdefault("_meta", {}).update({
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "migrated_from": "whole-file disagg hashes -> per-stripe content hashes; "
                             f"shard sha256 {shard_sha}, stats sha256 {stats_sha}",
        })
        changed.append((mf, manifest))

    print(f"\n{len(wanted)} wanted (site, iml); {missing_pickle} have no stripe pickle "
          f"yet (nothing to migrate -- they are new work).")
    print("\nclassification:")
    for k, n in sorted(counts.items()):
        print(f"  {n:>4}  {k}")
    print(f"\n{len(changed)} manifest(s) to re-stamp.")

    if not write:
        print("\n-- dry run; nothing written. Re-run with --write to apply.")
        return 0

    for mf, manifest in changed:
        mf.write_text(json.dumps(manifest, indent=2))
    print(f"re-stamped {len(changed)} manifest(s).")
    left = sum(counts.get(k, 0) for k in (CH_SHARD_MISMATCH, CH_STATS_MISMATCH, CH_NO_ROW))
    if left:
        print(f"NOTE: {left} manifest(s) were left alone and will recompute -- see the "
              f"[left alone] lines above.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="actually rewrite the manifests")
    ap.add_argument("--to", choices=("content-hashes", "shards"), default="content-hashes",
                    help="which re-stamp to run (default: content-hashes)")
    args = ap.parse_args(argv)
    if args.to == "shards":
        return migrate(write=args.write)
    return migrate_to_content_hashes(write=args.write)


if __name__ == "__main__":
    sys.exit(main())
