"""One-off: re-stamp existing manifests for the sharded disagg layout.

Splitting the disagg monolith into per-site shards changes what a stripe's
fingerprint names -- ``disagg_data_file`` (the 2.3 GB monolith) becomes
``disagg_site_shard`` (that site's 38 MB shard), and the batch artifacts get
``disagg_shards_digest``. Left alone, that would mark all ~450 already-selected
stripes stale and trigger hours of GCIM + selection for data that has not changed
by a single byte.

So the manifests are re-stamped in place instead. This is only honest because the
shards are provably the same bytes: the script refuses to touch any manifest whose
recorded ``disagg_data_file`` hash differs from the monolith sha256 that
``shard_disagg_data.py`` recorded in ``_split_provenance.json``. A manifest that
fails that check is genuinely stale and is left for the pipeline to recompute.

The stripe pickles and their ``.manifest.json`` sidecars are git-tracked, so
``git diff`` shows exactly what changed and ``git checkout`` undoes it.

    python -m phd_project.scripts.WP1_ground_motion_set.migrate_stripe_manifests
    python -m phd_project.scripts.WP1_ground_motion_set.migrate_stripe_manifests --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from phd_project.config.config import load_config
from phd_project.scripts import disagg_shards
from phd_project.scripts.cache_utils import fingerprint

STRIPE_RE = re.compile(r"^site_(\d+)__stripe_iml_.*__gm_selection\.pickle\.manifest\.json$")

# Skipped / untouched / migrated classifications
MIGRATE_STRIPE = "stripe"
MIGRATE_BATCH = "batch"
NO_DISAGG_KEY = "no disagg_data_file entry (nothing to migrate)"
ALREADY_DONE = "already on the sharded scheme"
HASH_MISMATCH = "disagg_data_file hash != split source -- genuinely stale, left alone"


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="actually rewrite the manifests")
    return migrate(write=ap.parse_args(argv).write)


if __name__ == "__main__":
    sys.exit(main())
