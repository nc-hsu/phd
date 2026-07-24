"""Summarise which site-specific MSA record sets are fully available on disk.

For the site-specific Multiple-Stripe Analysis (MSA), ground motions were
selected per conditioning IM -- ``AvgSA([0, 3])`` and ``AvgSA([0, 6])`` -- for
60 sites x 6 stripes, giving 30 records per (site, IM, stripe). Those selections
live in ``<gm_selection>/AvgSA_{03,06}_final_ensembles.pickle`` as a dict keyed
by ``(site_index, stripe_poe)`` whose ``['recs']`` value is a DataFrame of the
selected records (ESM and NGA-Sub mixed).

Only some of the required NGA-Sub time histories have been downloaded, so MSA can
only run at a site once *all* its records are present on the HSU file share. A
record counts as available if it is on the share (JSON conversion happens later):
    * NGA-Sub -> a folder ``NGASub_RSN_<rsn>`` exists under ``ngasub_folder``.
    * ESM     -> a file ``<event>__<station>__<loc>.h5`` exists under ``esm_folder``.

This script writes three CSVs into the gm-selection folder:
    1. ``records_availability_summary.csv`` -- one row per (IM, site, stripe) with
       required/available/missing counts (split by database) and completeness
       flags (``stripe_complete`` and ``site_im_complete``).
    2. ``complete_site_im_sets.csv`` -- simple ``site, im`` list of the (site, IM)
       units whose 6 stripes are all fully available (ready to run MSA now).
    3. ``ngasub_rsns_to_download_by_site_completion_batched.csv`` -- the still
       missing NGA-Sub RSNs, re-ordered so that downloading them completes whole
       (site, IM) record sets as early as possible, in the same batched layout as
       ``ngasub_rsns_to_download_batched.csv``.

Usage:
    python -m phd_project.scripts.data_handling.summarise_record_availability
        [--gm-selection-dir DIR] [--ngasub-folder DIR] [--esm-folder DIR]
        [--site-model CSV] [--out-dir DIR]
"""

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

from phd_project.config.config import load_config

# Console messages use emoji; make sure they survive a non-UTF-8 stdout (e.g.
# the default Windows cp1252 console) instead of crashing the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# The two conditioning-IM selection campaigns and their final-ensemble pickles.
_IMS = {
    "AvgSA_03": "AvgSA_03_final_ensembles.pickle",
    "AvgSA_06": "AvgSA_06_final_ensembles.pickle",
}

_RSN_FOLDER_RE = re.compile(r"NGASub_RSN_(\d+)", re.IGNORECASE)

# PEER NGA-Sub portal limits: 30 RSNs per sub-batch, 200 RSNs per download session.
BATCH_SIZE = 30
SESSION_LIMIT = 200


def _as_code(value):
    """Normalise a metadata code to a clean string (drops a trailing ``.0``)."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


# ---------------------------------------------------------------------------
# Step 1 -- inventory what is on the share
# ---------------------------------------------------------------------------
def inventory_ngasub(folder):
    """Return the set of NGA-Sub RSNs present as ``NGASub_RSN_<rsn>`` folders."""
    folder = Path(folder)
    rsns = set()
    for child in folder.glob("NGASub_RSN_*"):
        match = _RSN_FOLDER_RE.match(child.name)
        if match:
            rsns.add(int(match.group(1)))
    return rsns


def inventory_esm(folder):
    """Return the set of ESM record keys (``event__station__loc``) present as .h5."""
    folder = Path(folder)
    keys = set()
    for h5 in folder.glob("*.h5"):
        keys.add(h5.stem)
    return keys


def _loc_variants(loc):
    """Location-code spellings to try (disk zero-pads to 2 digits, e.g. ``00``)."""
    loc = _as_code(loc)
    variants = {loc, loc.zfill(2), loc.lstrip("0") or "0"}
    return variants


def esm_available(event, station, loc, available_esm):
    """True if any spelling of the ESM record key is present on the share."""
    for variant in _loc_variants(loc):
        if f"{event}__{station}__{variant}" in available_esm:
            return True
    return False


# ---------------------------------------------------------------------------
# Step 2 -- required records per (IM, site, stripe), with availability
# ---------------------------------------------------------------------------
def collect_requirements(im, pickle_path, available_ngasub, available_esm):
    """Extract per (site, stripe) required records and their availability.

    Returns a list of dicts, one per (site, stripe), each carrying the counts and
    the set of *missing* NGA-Sub RSNs for that (site, stripe).
    """
    with open(pickle_path, "rb") as f:
        ensembles = pickle.load(f)

    rows = []
    for (site, level), value in ensembles.items():
        recs = value["recs"]
        db = recs[("metadata", "database")]
        idx = recs[("metadata", "index")]
        event = recs[("metadata", "event_id")]
        station = recs[("metadata", "station_code")]
        location = recs[("metadata", "location_code")]

        req_ngasub = set()          # unique RSNs required here
        req_esm = set()             # unique ESM keys required here
        for d, i, ev, st, lo in zip(db, idx, event, station, location):
            if "NGA" in str(d).upper():
                req_ngasub.add(int(i))
            else:
                req_esm.add((_as_code(ev), _as_code(st), _as_code(lo)))

        missing_ngasub = {r for r in req_ngasub if r not in available_ngasub}
        missing_esm = {
            k for k in req_esm if not esm_available(k[0], k[1], k[2], available_esm)
        }

        rows.append(
            {
                "im": im,
                "site": int(site),
                "stripe_poe": float(level),
                "n_required_ngasub": len(req_ngasub),
                "n_required_esm": len(req_esm),
                "n_missing_ngasub": len(missing_ngasub),
                "n_missing_esm": len(missing_esm),
                "missing_ngasub": missing_ngasub,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Step 4 -- greedy ordering of missing NGA-Sub RSNs by (site, IM) completion
# ---------------------------------------------------------------------------
def order_missing_rsns(unit_missing, im_priority=None):
    """Order missing RSNs so early downloads complete the most (site, IM) units.

    ``unit_missing`` maps each (im, site) unit to its set of missing NGA-Sub RSNs.
    ``im_priority`` maps an IM label to a rank; units of a lower-rank IM are fully
    completed before higher-rank ones (so, e.g., AvgSA_03 sites finish before
    AvgSA_06). Within an IM the greedy picks the unit needing the fewest remaining
    RSNs (ties -> the unit whose RSNs are shared by the most other units), queues
    its RSNs (shared-most first), marks them downloaded, and recomputes. Returns a
    de-duplicated list covering every missing RSN. Because RSNs are shared across
    IMs, completing the priority IM also chips away at the other.
    """
    im_priority = im_priority or {}

    # How many units need each RSN (drives tie-breaking and inner ordering).
    rsn_unit_count = {}
    for missing in unit_missing.values():
        for rsn in missing:
            rsn_unit_count[rsn] = rsn_unit_count.get(rsn, 0) + 1

    remaining = {unit: set(missing) for unit, missing in unit_missing.items() if missing}
    ordered = []
    queued = set()

    while any(remaining.values()):
        # Priority IM first; then unit closest to completion; then most-shared.
        unit = min(
            (u for u, s in remaining.items() if s),
            key=lambda u: (
                im_priority.get(u[0], len(im_priority)),
                len(remaining[u]),
                -sum(rsn_unit_count[r] for r in remaining[u]),
                u,
            ),
        )
        for rsn in sorted(remaining[unit], key=lambda r: (-rsn_unit_count[r], r)):
            if rsn not in queued:
                queued.add(rsn)
                ordered.append(rsn)
        # Everything queued so far is "downloaded" -- drop from every unit.
        for other in remaining:
            remaining[other] -= queued
    return ordered


def completion_curve(ordered, unit_missing):
    """Map each (im, site) unit to the number of ordered RSNs downloaded before it
    completes (its last missing RSN's 1-based position); units with no missing
    RSNs complete at 0."""
    pos = {rsn: i + 1 for i, rsn in enumerate(ordered)}
    return {
        unit: max((pos[r] for r in missing), default=0)
        for unit, missing in unit_missing.items()
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_summary(path, rows, site_region, site_im_complete):
    """Write the per (IM, site, stripe) availability summary CSV."""
    fieldnames = [
        "im", "site", "region", "stripe_poe", "stripe_return_period",
        "n_required", "n_required_ngasub", "n_required_esm",
        "n_available", "n_missing", "n_missing_ngasub", "n_missing_esm",
        "stripe_complete", "site_im_complete",
    ]
    ordered_rows = sorted(rows, key=lambda r: (r["im"], r["site"], -r["stripe_poe"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ordered_rows:
            n_required = r["n_required_ngasub"] + r["n_required_esm"]
            n_missing = r["n_missing_ngasub"] + r["n_missing_esm"]
            writer.writerow(
                {
                    "im": r["im"],
                    "site": r["site"],
                    "region": site_region.get(r["site"], ""),
                    "stripe_poe": r["stripe_poe"],
                    "stripe_return_period": round(1.0 / r["stripe_poe"]),
                    "n_required": n_required,
                    "n_required_ngasub": r["n_required_ngasub"],
                    "n_required_esm": r["n_required_esm"],
                    "n_available": n_required - n_missing,
                    "n_missing": n_missing,
                    "n_missing_ngasub": r["n_missing_ngasub"],
                    "n_missing_esm": r["n_missing_esm"],
                    "stripe_complete": n_missing == 0,
                    "site_im_complete": site_im_complete[(r["im"], r["site"])],
                }
            )


def write_complete_units(path, site_im_complete):
    """Write the simple ``site, im`` list of fully-available (site, IM) units."""
    complete = sorted(
        (unit for unit, ok in site_im_complete.items() if ok),
        key=lambda u: (u[0], u[1]),
    )
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site", "im"])
        for im, site in complete:
            writer.writerow([site, im])
    return len(complete)


def write_batched(path, ordered):
    """Write ordered RSNs in the PEER batched layout.

    RSNs are grouped into download sessions of ``SESSION_LIMIT`` (200); within a
    session they are sub-batched by ``BATCH_SIZE`` (30) -- i.e. 6 full batches of
    30 plus a trailing batch of 20. A blank row separates sessions; batch numbers
    stay continuous, matching ``ngasub_rsns_to_download_batched.csv``.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        batch_no = 0
        for s_start in range(0, len(ordered), SESSION_LIMIT):
            if s_start > 0:
                writer.writerow([])  # blank spacer between download sessions
            session = ordered[s_start:s_start + SESSION_LIMIT]
            for b_start in range(0, len(session), BATCH_SIZE):
                batch_no += 1
                writer.writerow([batch_no, *session[b_start:b_start + BATCH_SIZE]])


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gm-selection-dir", type=Path, default=None)
    parser.add_argument("--ngasub-folder", type=Path, default=None)
    parser.add_argument("--esm-folder", type=Path, default=None)
    parser.add_argument("--site-model", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    gm_selection_dir = args.gm_selection_dir or cfg["proc_data"]["gm_selection"]
    ngasub_folder = args.ngasub_folder or cfg["raw_data"]["ngasub_folder"]
    esm_folder = args.esm_folder or cfg["raw_data"]["esm_hdf5_folder"]
    site_model = args.site_model or cfg["hazard_models"]["eshm20_wp1_site_model"]
    out_dir = args.out_dir or gm_selection_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Inventorying share:\n  NGA-Sub: {ngasub_folder}\n  ESM:     {esm_folder}")
    available_ngasub = inventory_ngasub(ngasub_folder)
    available_esm = inventory_esm(esm_folder)
    print(f"  found {len(available_ngasub)} NGA-Sub RSNs, {len(available_esm)} ESM files")

    site_region = {}
    try:
        site_df = pd.read_csv(site_model)
        if "region" in site_df.columns:
            site_region = {i: site_df["region"].iloc[i] for i in range(len(site_df))}
    except Exception as exc:  # region is informational only
        print(f"⚠️  could not read site model {site_model}: {exc}")

    # Step 2: gather requirements for both IMs.
    all_rows = []
    for im, filename in _IMS.items():
        pickle_path = Path(gm_selection_dir) / filename
        print(f"ℹ️  reading {pickle_path.name} ...")
        all_rows.extend(
            collect_requirements(im, pickle_path, available_ngasub, available_esm)
        )

    # Completeness per (site, IM) unit = every stripe fully available.
    unit_stripe_rows = {}
    for r in all_rows:
        unit_stripe_rows.setdefault((r["im"], r["site"]), []).append(r)
    site_im_complete = {
        unit: all(
            (r["n_missing_ngasub"] + r["n_missing_esm"]) == 0 for r in rows
        )
        for unit, rows in unit_stripe_rows.items()
    }

    # Missing NGA-Sub RSNs aggregated to (site, IM) units, for greedy ordering.
    unit_missing = {}
    for unit, rows in unit_stripe_rows.items():
        missing = set()
        for r in rows:
            missing |= r["missing_ngasub"]
        unit_missing[unit] = missing

    # Prioritise completing AvgSA_03 units, then AvgSA_06 (order of _IMS).
    im_priority = {im: rank for rank, im in enumerate(_IMS)}
    ordered = order_missing_rsns(unit_missing, im_priority)

    # Write outputs.
    summary_path = out_dir / "records_availability_summary.csv"
    complete_path = out_dir / "complete_site_im_sets.csv"
    batched_path = out_dir / "ngasub_rsns_to_download_by_site_completion_batched.csv"

    write_summary(summary_path, all_rows, site_region, site_im_complete)
    n_complete = write_complete_units(complete_path, site_im_complete)
    write_batched(batched_path, ordered)

    # Console summary.
    for im in _IMS:
        done = sum(1 for (u_im, _), ok in site_im_complete.items() if u_im == im and ok)
        total = sum(1 for (u_im, _) in site_im_complete if u_im == im)
        print(f"✅ {im}: {done}/{total} site record sets complete")
    print(f"✅ {n_complete}/{len(site_im_complete)} (site, IM) units fully available")
    print(f"ℹ️  {len(ordered)} NGA-Sub RSNs still to download")

    # Ordering sanity: how many units complete after each download session,
    # broken down by IM (AvgSA_03 is completed before AvgSA_06).
    curve = completion_curve(ordered, unit_missing)
    im_totals = {im: sum(1 for u in site_im_complete if u[0] == im) for im in _IMS}
    print("   completed (site, IM) units vs RSNs downloaded:")
    for n in range(0, len(ordered) + SESSION_LIMIT, SESSION_LIMIT):
        by_im = " ".join(
            f"{im}={sum(1 for u, c in curve.items() if u[0] == im and c <= n)}/{im_totals[im]}"
            for im in _IMS
        )
        print(f"     after {min(n, len(ordered)):4d} RSNs: {by_im}")
        if n >= len(ordered):
            break

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {complete_path}")
    print(f"  {batched_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
