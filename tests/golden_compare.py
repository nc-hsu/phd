"""Semantic comparison helpers for the gm-selection golden-master regression.

Pickles are compared by their *meaningful payload* (the set of selected record
ids per ``(site, poe)``), not by raw bytes -- serialization/dtype noise makes
byte comparison give false failures. CSVs are compared as row-sets (order
ignored).

Run as a script to compare the *current* pipeline outputs against the snapshot in
``tests/golden/<campaign>/``::

    python tests/golden_compare.py              # compares every snapshotted campaign
    python tests/golden_compare.py AvgSA_03     # just one campaign

Exits non-zero if anything diverges.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = REPO / "tests" / "golden"

INDEX_COL = ("metadata", "index")

# campaign -> config key for the per-(site,poe) result pickles dir
RESULT_CFG_KEY = {
    "AvgSA_03": "AvgSA_03_record_selection",
    "AvgSA_06": "AvgSA_06_record_selection",
}


def _load(fp: Path):
    with open(fp, "rb") as f:
        return pickle.load(f)


def _selected_ids(ensemble) -> list | None:
    """Sorted selected-record ids for one ensemble, or None if no ensemble."""
    if ensemble is None:
        return None
    return sorted(ensemble["recs"][INDEX_COL].tolist())


def compare_ensemble_dicts(got: dict, exp: dict, label: str) -> list[str]:
    """Compare two ``{(site, poe): ensemble}`` dicts. Returns mismatch messages."""
    problems = []
    if set(got) != set(exp):
        only_got = set(got) - set(exp)
        only_exp = set(exp) - set(got)
        if only_got:
            problems.append(f"[{label}] extra keys: {sorted(only_got)[:5]}")
        if only_exp:
            problems.append(f"[{label}] missing keys: {sorted(only_exp)[:5]}")
    for key in sorted(set(got) & set(exp)):
        g, e = _selected_ids(got[key]), _selected_ids(exp[key])
        if g != e:
            problems.append(f"[{label}] {key}: selected records differ "
                            f"(got {len(g or [])}, exp {len(e or [])})")
    return problems


def compare_csv_rowset(got_fp: Path, exp_fp: Path, label: str) -> list[str]:
    """Compare two CSVs as unordered row-sets."""
    g = pd.read_csv(got_fp).astype(str)
    e = pd.read_csv(exp_fp).astype(str)
    if list(g.columns) != list(e.columns):
        return [f"[{label}] columns differ: {list(g.columns)} vs {list(e.columns)}"]
    g_rows = sorted(map(tuple, g.values.tolist()))
    e_rows = sorted(map(tuple, e.values.tolist()))
    if g_rows != e_rows:
        return [f"[{label}] row-sets differ (got {len(g_rows)} rows, exp {len(e_rows)} rows)"]
    return []


def compare_campaign(campaign: str, current_06: Path, current_results: Path) -> list[str]:
    """Compare a campaign's current outputs against its golden snapshot.

    Stage pickles, CSVs and the 360 result pickles are discovered from the golden
    snapshot dir, so AvgSA_06's extra ``_rd03`` stage is handled automatically.
    The ``gcim_dist_*`` pickle in the snapshot is a selection *input*, not an
    output, and is skipped.
    """
    golden = GOLDEN_ROOT / campaign
    if not golden.is_dir():
        return [f"[{campaign}] no golden snapshot at {golden}"]
    problems = []

    # stage pickles: every *.pickle in the snapshot except the gcim input.
    for gfp in sorted(golden.glob("*.pickle")):
        if gfp.name.startswith("gcim_dist_"):
            continue
        cur = current_06 / gfp.name
        if not cur.is_file():
            problems.append(f"[{gfp.name}] current file missing: {cur}")
            continue
        problems += compare_ensemble_dicts(_load(cur), _load(gfp), gfp.name)

    # download / convert CSVs.
    for gfp in sorted(golden.glob("*.csv")):
        cur = current_06 / gfp.name
        if not cur.is_file():
            problems.append(f"[{gfp.name}] current file missing: {cur}")
            continue
        problems += compare_csv_rowset(cur, gfp, gfp.name)

    # 360 per-(site, poe) result pickles.
    for gfp in sorted((golden / "results").glob("*__gm_selection.pickle")):
        cfp = current_results / gfp.name
        if not cfp.is_file():
            problems.append(f"[results] missing {gfp.name}")
            continue
        if _selected_ids(_load(cfp)) != _selected_ids(_load(gfp)):
            problems.append(f"[results] {gfp.name}: selected records differ")
    return problems


def main(argv: list[str]) -> int:
    from phd_project.config.config import load_config
    cfg = load_config()

    campaigns = argv or sorted(p.name for p in GOLDEN_ROOT.iterdir() if p.is_dir())
    if not campaigns:
        print(f"No golden snapshots found under {GOLDEN_ROOT}")
        return 1

    total = 0
    for campaign in campaigns:
        problems = compare_campaign(
            campaign,
            current_06=cfg["proc_data"]["gm_selection"],
            current_results=cfg["results"][RESULT_CFG_KEY[campaign]],
        )
        if problems:
            total += len(problems)
            print(f"\n{campaign}: FAILED ({len(problems)} differences):")
            for p in problems[:50]:
                print("  " + p)
        else:
            print(f"\n{campaign}: PASSED -- all outputs match the snapshot.")

    if total:
        print(f"\nGOLDEN COMPARISON FAILED ({total} total differences).")
        return 1
    print("\nGOLDEN COMPARISON PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
