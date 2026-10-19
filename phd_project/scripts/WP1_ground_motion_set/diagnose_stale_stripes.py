"""Explain *why* the AvgSA_03 stale check wants to recompute a stripe.

``find_stale_stripes`` only answers yes/no, which is unhelpful when a notebook
recomputes work you believe was just done. This prints the reason per stripe, and
aggregates the reasons so a systematic cause stands out immediately.

Two very different things get confused here, so it names them explicitly:

- **Nothing marks a stripe valid until nb 033 runs.** The per-stripe
  ``.manifest.json`` is written by nb 033, not 031 or 032. Running 031 twice will
  therefore report the *same* stale set both times -- it computed the gcim, but
  nothing recorded a stripe as selected. That is the pipeline working as designed,
  not a cache miss.
- **A genuinely mismatched input**, in which case the differing fingerprint entry
  is named. ``disagg_stripe_data`` means this stripe's own disaggregation
  DataFrame changed; ``disagg_stats_row`` means its own stats row did. Both are
  per-stripe content hashes, so such a mismatch is real -- unlike the whole-file
  hashes they replaced, which went stale for all ~500 stripes whenever a single
  new iml was disaggregated across the 60 sites.
- **A stripe that contains a newly excluded record.** The stale check is
  versioned by exclusion list (``UNAVAILABLE_RECORDS``): a stripe whose manifest
  was written under an older version of the list is still valid, unless it
  contains a record that was excluded after that version. Such a stripe is
  reported with the version it was selected under and the record(s) it contains.
  The summary also counts the valid stripes by the version they were selected
  under.

    python -m phd_project.scripts.WP1_ground_motion_set.diagnose_stale_stripes
    python -m phd_project.scripts.WP1_ground_motion_set.diagnose_stale_stripes --show 20
    python -m phd_project.scripts.WP1_ground_motion_set.diagnose_stale_stripes --selection optimised
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from phd_project.config.config import load_config
from phd_project.scripts import disagg_shards
from phd_project.scripts.cache_utils import _manifest_path

from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    drop_unavailable_records,
    exclusion_records_added_after,
    manifest_exclusion_version,
    stripe_pickle_path,
)
from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
    EXCLUSION_LIST_VERSION,
    SELECTION_CONFIG,
    SELECTION_CONFIG_OPTIMISED,
    UNAVAILABLE_RECORDS,
    setup_AvgSA03_gcim_gm_selection,
    stripe_fingerprints,
    stripe_source_fps,
    wanted_stripe_keys,
)

# Each selection's result folder (cfg["results"] key), selection config, and the
# notebook that would (re)compute its stale stripes.
SELECTIONS = {
    "original": ("AvgSA_03_record_selection", SELECTION_CONFIG, "031 / 032"),
    "optimised": ("AvgSA_03_record_selection_optimised", SELECTION_CONFIG_OPTIMISED,
                  "032-optimised"),
}

NO_PICKLE = "no stripe pickle on disk (never selected, or pruned)"
NO_MANIFEST = "stripe pickle present but no .manifest.json (unmanaged)"


def _diff(cached: dict, fp: dict) -> list[str]:
    """Names of the fingerprint entries whose hash the manifest does not match.

    Same comparison as cache_utils.manifest_matches: only the entries of ``fp``
    count, and an entry whose hash is None (the version-0 "no exclusion list"
    marker) matches only a manifest that lacks it."""
    return [name for name, cur in fp.items()
            if cached.get(name, {}).get("hash") != cur["hash"]]


def _contained(stripe_fp: Path, version: int) -> list[str]:
    """Labels of the records excluded after ``version`` that this stripe contains."""
    with open(stripe_fp, "rb") as f:
        recs = pickle.load(f)["recs"]
    return [rec.get("label", str(rec["identity"]))
            for rec in exclusion_records_added_after(UNAVAILABLE_RECORDS, version)
            if len(drop_unavailable_records(recs, [rec], verbose=False)) < len(recs)]


def _reason(stripe_fp: Path, site, iml, fp_fn, superseded) -> tuple[str, list[str], int | None]:
    """(reason, differing input names, version) for one stripe.

    ``version`` is the exclusion-list version the manifest was written under, for a
    stripe that is valid ("MATCHES") or stale only because it contains a record
    excluded since; ``None`` otherwise."""
    if not stripe_fp.is_file():
        return NO_PICKLE, [], None
    mp = _manifest_path(stripe_fp)
    if not mp.is_file():
        return NO_MANIFEST, [], None

    version = manifest_exclusion_version(stripe_fp, site, iml, fp_fn, superseded,
                                         EXCLUSION_LIST_VERSION)
    if version is not None:
        if version == EXCLUSION_LIST_VERSION:
            return "MATCHES (not stale)", [], version
        contained = _contained(stripe_fp, version)
        if not contained:
            return "MATCHES (not stale)", [], version
        return (f"selected under exclusion list v{version}, contains record(s) excluded "
                f"since: {'; '.join(contained)}"), [], version

    # Matches no version: some other input differs. Report the differences against
    # the version the manifest is closest to, so the db_exclusions entry is not
    # listed merely because the stripe was selected under an older list.
    cached = json.loads(mp.read_text()).get("inputs", {})
    candidates = [(EXCLUSION_LIST_VERSION, fp_fn)] + [
        (EXCLUSION_LIST_VERSION - 1 - k, f) for k, (f, _) in enumerate(superseded)]
    v, diff = min(((v, _diff(cached, f(site, iml))) for v, f in candidates),
                  key=lambda t: len(t[1]))
    return f"input mismatch (vs exclusion list v{v})", diff, None


def diagnose(show: int = 10, selection: str = "original") -> int:
    cfg = load_config()
    results_key, selection_config, recompute_nb = SELECTIONS[selection]
    result_folder = Path(cfg["results"][results_key])
    shard_dir = Path(cfg["proc_data"]["AvgSA_03_disagg_data_shards"])

    # --- environment first: a missing shard set makes EVERY stripe stale -----
    print("=" * 72)
    print(f"INPUTS  (selection: {selection}, exclusion list v{EXCLUSION_LIST_VERSION})")
    print("=" * 72)
    shards = disagg_shards.shard_paths(shard_dir) if shard_dir.is_dir() else []
    index_ok = (shard_dir / disagg_shards.INDEX_NAME).is_file()
    print(f"  shard dir     : {shard_dir}")
    print(f"                  exists={shard_dir.is_dir()}  shards={len(shards)}  "
          f"_index.json={index_ok}")
    hashes_ok = disagg_shards.content_hashes_path(shard_dir).is_file()
    print(f"                  {disagg_shards.CONTENT_HASHES_NAME}={hashes_ok}")
    if not shards or not index_ok:
        print("\n  *** The disagg shards are missing or incomplete. ***")
        print("      Every stripe fingerprint would hash the shard PATH STRING")
        print("      instead of file bytes, so nothing can ever match.")
        print("      Fix: dvc pull  (then re-run this).")
        return 1
    if not hashes_ok:
        print(f"\n  *** No {disagg_shards.CONTENT_HASHES_NAME} in the shard dir. ***")
        print("      Per-stripe fingerprints are read from it, so the stale check")
        print("      cannot run at all. This is a shard set written before the")
        print("      per-stripe index existed.")
        print("      Fix: disagg_shards.write_content_hashes(shard_dir)  (one pass")
        print("           over the shards; changes no shard byte).")
        return 1
    print(f"  results folder: {result_folder}")
    print(f"                  {len(list(result_folder.glob('*__stripe_iml_*.pickle')))} "
          f"stripe pickles, "
          f"{len(list(result_folder.glob('*__stripe_iml_*.manifest.json')))} manifests")

    # --- the stale check, exactly as the notebooks run it -------------------
    wanted = wanted_stripe_keys()
    _, disagg_stats, _, ctx, _ = setup_AvgSA03_gcim_gm_selection(sites=())
    fp_fn, superseded = stripe_fingerprints(
        selection_config, stripe_source_fps(), ctx, disagg_stats)

    reasons, diff_counter, examples = Counter(), Counter(), {}
    valid_by_version = Counter()
    stale = []
    for (site, iml) in wanted:
        fp = stripe_pickle_path(result_folder, site, iml)
        reason, diff, version = _reason(fp, site, iml, fp_fn, superseded)
        if reason == "MATCHES (not stale)":
            valid_by_version[version] += 1
            continue
        stale.append((site, iml))
        key = reason if not diff else f"input mismatch: {', '.join(sorted(diff))}"
        reasons[key] += 1
        for d in diff:
            diff_counter[d] += 1
        examples.setdefault(key, (site, iml, fp))

    print()
    print("=" * 72)
    print(f"STALE: {len(stale)} of {len(wanted)} wanted stripes")
    print("=" * 72)
    print("  valid stripes by the exclusion-list version they were selected under:")
    for v in sorted(valid_by_version):
        print(f"    v{v}: {valid_by_version[v]:>4}"
              + ("  (current)" if v == EXCLUSION_LIST_VERSION else ""))
    if not stale:
        print(f"\n  Nothing stale. nb {recompute_nb} would compute nothing.")
        return 0

    for reason, n in reasons.most_common():
        site, iml, fp = examples[reason]
        print(f"\n  {n:>4} stripe(s): {reason}")
        print(f"        e.g. site {site}, iml {iml}  ->  {fp.name}")

    print("\n" + "-" * 72)
    if reasons and all(r.startswith("selected under exclusion list") for r in reasons):
        print("DIAGNOSIS: every stale stripe contains a record excluded after the")
        print("  exclusion-list version it was selected under. That is the versioned")
        print("  exclusion working as designed: these, and only these, are reselected.")
    elif reasons and all(r == NO_PICKLE for r in reasons):
        print("DIAGNOSIS: every stale stripe simply has no stripe pickle yet.")
        print()
        print("  Nb 031 computes the GCIM but NEVER writes a stripe pickle or")
        print("  manifest -- nb 033 does. So re-running 031 alone will report the")
        print("  same stale set forever, however many times you run it.")
        print()
        print("  Run 031 -> 032 -> 033. After 033 these become valid and 031 will")
        print("  report them as already valid.")
    else:
        print("DIAGNOSIS: at least some stripes differ on a tracked input.")
        print("  Most frequently differing fingerprint entries:")
        for name, n in diff_counter.most_common(8):
            print(f"    {n:>4}x  {name}")
        print()
        print("  An entry differing on EVERY stale stripe points at a shared input")
        print("  (gm database, site model, gmm logic tree, selection config).")
        print("  'disagg_site_shard' differing on only some points at those sites'")
        print("  disaggregation having been rebuilt.")

    print("\n  Sample stripes (first %d):" % show)
    for (site, iml) in stale[:show]:
        print(f"    site {site:>3}, iml {iml}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", type=int, default=10,
                    help="how many stale stripes to list (default 10)")
    ap.add_argument("--selection", choices=sorted(SELECTIONS), default="original",
                    help="which selection's result folder to check (default: original)")
    args = ap.parse_args(argv)
    return diagnose(args.show, args.selection)


if __name__ == "__main__":
    sys.exit(main())
