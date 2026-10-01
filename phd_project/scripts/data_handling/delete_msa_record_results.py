"""Delete the MSA results of individual (site, structure, stripe, record) slots.

Why this exists
---------------
The MSA launcher (``run_batch_msa_per_record.py``) resumes per record: a slot whose
``record_{k}_log.json`` exists, and whose logged record name and scale factor match
the stripe selection, is skipped. That covers a record being *swapped* in a stripe
pickle. It cannot cover a record whose JSON time history was *re-generated under the
same name* -- e.g. the ESM U/V-swap fix, after which every affected file has the same
name and scale factor but different content. Those slots look finished and would
never be re-run.

This script removes the results of exactly the slots listed in a CSV, so the next
ordinary launch of each building re-runs only those records.

Input CSV
---------
One row per slot to delete, with columns (as written by nb 043, e.g.
``data_processed/07_gm_records/swapped_esm_records_msa_usage_AvgSA_03.csv``)::

    site, structure, stripe_iml, record_index[, record_tag]

    site          site index, e.g. 41
    structure     storey folder, "3s" / "5s"
    stripe_iml    stripe file-name token, e.g. "00pt650" (kept as text, zero padding intact)
    record_index  0-based slot of the record within the stripe (= the MSA record tag)
    record_tag    optional; the record's JSON stem. When present it is used as a safety
                  check: a slot whose log names a DIFFERENT record is skipped, not deleted,
                  because the slot already holds a result for some other (valid) record.

What is deleted
---------------
Per row, every file matching ``record_{k}_*`` in

    <root>/site_{site}/{structure}/mdof/msa_{im}/stripe_{stripe_iml}/

i.e. the five per-record artefacts written by ``standes.analysis.msa.msa_one_record``
(``_log.json``, ``_recorders.pickle``, ``_collapse_recorders.pickle``,
``_timearray.pickle``, ``_injection_function_returns.pickle``). Nothing else: the stripe
log ``msa_log_stripe_*.json``, the collated ``msa_log.json`` and
``collapse_fragility.json`` are left alone -- the next launch rebuilds them from the
record logs on disk. Until that re-run they still describe the OLD results, and once it
is done nb 060 must be re-run to refresh the processed collapse flags / fragilities.

ONE-SHOT per CSV: after the re-analysis the re-run slots carry the same record names as
before, so running this script again on the same CSV would delete the fresh results.

Usage
-----
    python -m phd_project.scripts.data_handling.delete_msa_record_results CSV --dry-run
    python -m phd_project.scripts.data_handling.delete_msa_record_results CSV
        [--root DIR] [--im AvgSA_03]

Always look at the ``--dry-run`` report first.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["site", "structure", "stripe_iml", "record_index"]

# per-row outcomes
DELETE = "delete"               # files found and (if checkable) the log names the right record
MISMATCH = "name mismatch"      # the slot's log names a different record -> left alone
NOTHING = "nothing to delete"   # no stripe folder or no record_k_* files (never run / already gone)


@dataclass
class SlotPlan:
    """What will happen to one CSV row's slot, plus the evidence for it."""
    site: str
    structure: str
    stripe_iml: str
    record_index: int
    record_tag: str | None
    stripe_dir: Path
    status: str
    files: list[Path] = field(default_factory=list)
    note: str = ""


def read_slots(csv_path: Path) -> pd.DataFrame:
    """Read and validate the slot CSV.

    Everything is read as text (``dtype=str``) so ``stripe_iml`` keeps its zero-padded
    token ("00pt310" must not become a number or lose its leading zeros) and ``site``
    builds the folder name verbatim. Duplicate rows are dropped: the same slot listed
    twice is still one set of files.
    """
    df = pd.read_csv(csv_path, dtype=str).apply(lambda c: c.str.strip())
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing column(s) {missing}; "
                         f"required: {REQUIRED_COLUMNS}")
    n_before = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(df) < n_before:
        print(f"note: dropped {n_before - len(df)} duplicate row(s)")
    return df


def plan_slot(row, root: Path, im: str) -> SlotPlan:
    """Locate one slot's result files and decide whether they may be deleted."""
    k = int(row["record_index"])
    record_tag = row.get("record_tag")
    record_tag = record_tag if isinstance(record_tag, str) and record_tag else None

    stripe_dir = (root / f"site_{row['site']}" / row["structure"] / "mdof"
                  / f"msa_{im}" / f"stripe_{row['stripe_iml']}")
    plan = SlotPlan(site=row["site"], structure=row["structure"],
                    stripe_iml=row["stripe_iml"], record_index=k, record_tag=record_tag,
                    stripe_dir=stripe_dir, status=NOTHING)

    if not stripe_dir.is_dir():
        plan.note = "stripe folder not found"
        return plan

    # The trailing underscore is what keeps this to ONE slot: "record_2_*" cannot match
    # "record_29_log.json", whose prefix is "record_29".
    plan.files = sorted(stripe_dir.glob(f"record_{k}_*"))
    if not plan.files:
        plan.note = "no record files (already deleted, or never run)"
        return plan

    # Safety check: if we know which record the CSV expects, make sure the slot's log
    # was written for it. A different name means the slot was re-selected and re-run
    # with another record since the CSV was made -- that result is valid, keep it.
    log_fp = stripe_dir / f"record_{k}_log.json"
    if record_tag is not None and log_fp.is_file():
        try:
            with open(log_fp) as file:
                logged = json.load(file).get("record_name")
        except (json.JSONDecodeError, OSError):
            logged = None       # unreadable log: an unfinished result, fine to delete
        if logged is not None and logged != f"{record_tag}.json":
            plan.status = MISMATCH
            plan.note = f"log has {logged}, CSV expects {record_tag}.json"
            return plan

    plan.status = DELETE
    return plan


def report(plans: list[SlotPlan], dry_run: bool) -> None:
    """Print one line per slot, then the summary counts and any problem slots."""
    to_delete = [p for p in plans if p.status == DELETE]
    mismatched = [p for p in plans if p.status == MISMATCH]
    nothing = [p for p in plans if p.status == NOTHING]

    verb = "would delete" if dry_run else "deleting"
    print(f"\n{'DRY RUN - nothing is deleted' if dry_run else 'DELETING'}\n")
    for p in to_delete:
        print(f"  {verb} {len(p.files)} file(s): {p.stripe_dir / f'record_{p.record_index}_*'}")

    n_files = sum(len(p.files) for p in to_delete)
    by_structure = pd.Series([p.structure for p in to_delete], dtype=object).value_counts()
    print("\nSummary")
    print(f"  CSV rows (unique slots):     {len(plans)}")
    print(f"  records to delete:           {len(to_delete)}  "
          f"({', '.join(f'{s}: {n}' for s, n in by_structure.sort_index().items()) or '-'})")
    print(f"  files to delete:             {n_files}")
    print(f"  sites affected:              {len({p.site for p in to_delete})}")
    print(f"  structure-stripes affected:  "
          f"{len({(p.site, p.structure, p.stripe_iml) for p in to_delete})}")
    # a record normally has 5 files; anything else is worth a look before deleting
    odd = [p for p in to_delete if len(p.files) != 5]
    if odd:
        print(f"  !! {len(odd)} slot(s) without the usual 5 files:")
        for p in odd:
            print(f"       {p.stripe_dir} record {p.record_index}: "
                  f"{[f.name for f in p.files]}")

    print(f"  skipped (name mismatch):     {len(mismatched)}")
    for p in mismatched:
        print(f"       {p.stripe_dir} record {p.record_index}: {p.note}")
    print(f"  nothing to delete:           {len(nothing)}")
    for p in nothing:
        print(f"       {p.stripe_dir} record {p.record_index}: {p.note}")


def run(csv_path: Path, root: Path, im: str = "AvgSA_03", dry_run: bool = False) -> list[SlotPlan]:
    """Plan every slot, print the report and (unless ``dry_run``) delete the files.

    Planning is done for ALL rows before anything is deleted, so the report a real run
    prints is the same one ``--dry-run`` shows.
    """
    if not root.is_dir():
        # most likely the analysis drive is not mounted on this machine
        raise FileNotFoundError(f"results root {root} does not exist (drive not mounted?)")

    slots = read_slots(csv_path)
    plans = [plan_slot(row, root, im) for _, row in slots.iterrows()]
    report(plans, dry_run)

    if not dry_run:
        n = 0
        for p in plans:
            if p.status != DELETE:
                continue
            for f in p.files:
                f.unlink()
                n += 1
        print(f"\ndeleted {n} file(s). Re-launch the affected buildings as usual; "
              f"then re-run nb 060.")
    return plans


def _default_root() -> Path:
    """The MSA results root from the project config (analysis_data.wp1_casestudy_sites)."""
    from phd_project.config.config import load_config
    return Path(load_config()["analysis_data"]["wp1_casestudy_sites"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete the MSA results of the (site, structure, stripe, record) "
                    "slots listed in a CSV, so the next launch re-runs only those.")
    parser.add_argument("csv", type=Path,
                        help="CSV with columns site, structure, stripe_iml, record_index "
                             "[, record_tag]")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be deleted, delete nothing")
    parser.add_argument("--root", type=Path, default=None,
                        help="results root holding site_*/ folders (default: config "
                             "analysis_data.wp1_casestudy_sites)")
    parser.add_argument("--im", default="AvgSA_03",
                        help="intensity measure; selects the msa_{im} results folder "
                             "(default: AvgSA_03)")
    args = parser.parse_args()

    try:
        run(args.csv, args.root if args.root is not None else _default_root(),
            im=args.im, dry_run=args.dry_run)
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"error: {err}")
