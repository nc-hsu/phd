"""Convert selected NGA-Subduction AT2 records to the project's JSON format.

NGA-Sub records live (after ``extract_ngasub_records.py``) under
``<src>/NGASub_RSN_<rsn>/NGAsubRSN<rsn>_<station><chan>.AT2`` -- three AT2 files
(E/N/Z) per RSN. Given a selection CSV of ``(record_identifier, component)``
rows (``record_identifier`` is the RSN for NGA-Sub), this script finds the
correct horizontal-component file, parses it, and writes one JSON record per row
matching the existing ESM-style schema (with ``record`` last).

Which physical channel is H1 vs H2 is taken from the filename map flatfile
(``accFilePathH1`` / ``accFilePathH2``). Event/station codes are taken from the
metadata flatfile (``NGAsubEQID`` / ``NGAsubSSN``).

Usage:
    python convert_ngasub_records_to_json.py <src> <selection_csv> <dst>
        --filename-map NGASub_SA_H1_H2_filenamemap.csv
        [--metadata NGASub_Metadata_SA_rotD50.csv]
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

from standes.groundmotion import record_json_filename, ngasub_record_identifier

# Console messages use emoji; make sure they survive a non-UTF-8 stdout (e.g.
# the default Windows cp1252 console) instead of crashing the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Matches integers, decimals, and scientific notation (e.g. -3.8367193E-08).
_FLOAT_RE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?')


def _normalise(text):
    """Lowercase and strip whitespace (for fuzzy header/value matching)."""
    return re.sub(r"\s+", "", str(text)).lower()


def _find_column(fieldnames, *aliases):
    """Return the actual header matching one of ``aliases`` (case-insensitive)."""
    wanted = {_normalise(a) for a in aliases}
    for name in fieldnames:
        if _normalise(name) in wanted:
            return name
    return None


def load_filename_map(path):
    """Map each RSN to its H1/H2 AT2 basenames.

    Returns ``{rsn: {"H1": basename, "H2": basename}}`` where ``rsn`` is a
    string and basenames are like ``ATKABNE.AT2``.
    """
    path = Path(path)
    mapping = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rsn_col = _find_column(reader.fieldnames, "NGAsubRSN", "RSN")
        h1_col = _find_column(reader.fieldnames, "accFilePathH1")
        h2_col = _find_column(reader.fieldnames, "accFilePathH2")
        if not (rsn_col and h1_col and h2_col):
            raise ValueError(
                f"{path.name}: expected columns NGAsubRSN, accFilePathH1, "
                f"accFilePathH2 (found {reader.fieldnames})."
            )
        for row in reader:
            rsn = str(row[rsn_col]).strip()
            if not rsn:
                continue
            mapping[rsn] = {
                "H1": Path(str(row[h1_col]).strip().replace("\\", "/")).name,
                "H2": Path(str(row[h2_col]).strip().replace("\\", "/")).name,
            }
    return mapping


def load_metadata(path):
    """Map each RSN to its event/station codes.

    Returns ``{rsn: {"event_id": NGAsubEQID, "station_code": NGAsubSSN}}``.
    """
    path = Path(path)
    mapping = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rsn_col = _find_column(reader.fieldnames, "NGAsubRSN", "RSN")
        eq_col = _find_column(reader.fieldnames, "NGAsubEQID")
        ssn_col = _find_column(reader.fieldnames, "NGAsubSSN")
        if not (rsn_col and eq_col and ssn_col):
            raise ValueError(
                f"{path.name}: expected columns NGAsubRSN, NGAsubEQID, "
                f"NGAsubSSN (found {reader.fieldnames})."
            )
        for row in reader:
            rsn = str(row[rsn_col]).strip()
            if not rsn:
                continue
            mapping[rsn] = {
                "event_id": str(row[eq_col]).strip(),
                "station_code": str(row[ssn_col]).strip(),
            }
    return mapping


def parse_at2(path):
    """Parse an NGA-Sub AT2 file.

    Returns ``(dt, record)``. Raises ``ValueError`` if units are not g or the
    file is malformed.
    """
    path = Path(path)
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError("file too short to contain header + data")

    if "UNITS OF G" not in lines[2].upper():
        raise ValueError("units are not 'g'")

    dt_nums = [float(m.group(0)) for m in _FLOAT_RE.finditer(lines[3])]
    if len(dt_nums) < 2:
        raise ValueError(f"could not parse dt from header line: {lines[3].strip()!r}")
    dt = dt_nums[1]  # NPTS=..., DT=...

    record = []
    for line in lines[4:]:
        if not line.strip():
            continue
        record.extend(float(m.group(0)) for m in _FLOAT_RE.finditer(line))

    if not record:
        raise ValueError("no acceleration data found")

    return dt, record


def resolve_at2_path(src, rsn, basename):
    """Locate the AT2 file for ``rsn`` whose original basename is ``basename``.

    Returns the resolved ``Path`` or ``None`` if not found.
    """
    folder = Path(src) / f"NGASub_RSN_{rsn}"
    if not folder.is_dir():
        return None

    expected = folder / f"NGAsubRSN{rsn}_{basename}"
    if expected.is_file():
        return expected

    # Fallback: any file in the folder ending with the mapped basename.
    matches = list(folder.glob(f"*{basename}"))
    return matches[0] if matches else None


def convert_record(rsn, component, src, filename_map, metadata, dst):
    """Convert one (RSN, component) selection into a JSON file.

    Returns None if a file was written, or a short reason string if the record
    was skipped.
    """
    fm = filename_map.get(rsn)
    if fm is None:
        return "not in filename map"

    basename = fm.get(component)
    if not basename:
        return f"no {component} filename in map"

    at2_path = resolve_at2_path(src, rsn, basename)
    if at2_path is None:
        return "AT2 file not found"

    try:
        dt, record = parse_at2(at2_path)
    except ValueError as e:
        return str(e)

    meta = metadata.get(rsn) if metadata is not None else None
    event_id = meta["event_id"] if meta else ""
    station_code = meta["station_code"] if meta else ""

    record_dict = {
        "eq_index": rsn,
        "database": "ngasub",
        "dt": dt,
        "duration": dt * len(record),
        "units": "g",
        "record_type": "acc",
        "component": component,
        "event_id": event_id,
        "station_code": station_code,
        "location_code": "0",
        "record": record,
    }

    out_path = Path(dst) / record_json_filename(ngasub_record_identifier(rsn), component)
    with open(out_path, "w") as f:
        json.dump(record_dict, f, indent=4)

    return None


def read_selection(path):
    """Yield ``(rsn, component)`` tuples from the selection CSV.

    The ``record_identifier`` column holds the RSN for NGA-Sub records.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rsn_col = _find_column(reader.fieldnames, "record_identifier", "RSN", "NGAsubRSN")
        comp_col = _find_column(reader.fieldnames, "component", "comp")
        if not (rsn_col and comp_col):
            raise ValueError(
                f"{path.name}: expected columns record_identifier, component "
                f"(found {reader.fieldnames})."
            )
        for row in reader:
            rsn = str(row[rsn_col]).strip()
            component = str(row[comp_col]).strip().upper()
            if not rsn:
                continue
            yield rsn, component


def convert_ngasub_records(src, selection_csv, dst, filename_map, metadata=None,
                           max_workers=None, overwrite=False):
    """Convert all ``(record_identifier, component)`` rows of the selection CSV.

    ``record_identifier`` is the RSN. ``filename_map`` and ``metadata`` are paths
    (loaded once here); ``metadata`` may be ``None``, in which case event_id and
    station_code are left empty. Records are read from the (often
    network-mounted) source in parallel using a thread pool, since the work is
    I/O-bound; ``max_workers`` defaults to ``max(cpu_count - 3, 1)``. Rows whose
    target JSON already exists in ``dst`` are skipped without reading the source,
    unless ``overwrite`` is True. Non-interactive. Returns
    ``(converted_count, skipped_records)`` where ``skipped_records`` is a list of
    ``{"record_identifier", "component", "reason"}`` dicts.
    """
    if max_workers is None:
        max_workers = max((os.cpu_count() or 1) - 3, 1)

    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    fmap = load_filename_map(filename_map)

    meta = None
    if metadata is not None:
        if Path(metadata).is_file():
            meta = load_metadata(metadata)
        else:
            print(f"⚠️  Metadata flatfile not found: {metadata} — "
                  f"event_id/station_code left empty.")

    converted = 0
    skipped_existing = 0
    skipped_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for rsn, component in read_selection(selection_csv):
            if component not in ("H1", "H2"):
                skipped_records.append({
                    "record_identifier": rsn, "component": component,
                    "reason": "unexpected component",
                })
                continue
            out_path = dst / record_json_filename(ngasub_record_identifier(rsn), component)
            if out_path.exists() and not overwrite:
                skipped_existing += 1
                continue
            futures.append((rsn, component,
                            executor.submit(convert_record, rsn, component,
                                            src, fmap, meta, dst)))
        for rsn, component, future in tqdm(
                futures, total=len(futures), desc="Converting NGA-Sub records"):
            reason = future.result()
            if reason is None:
                converted += 1
            else:
                skipped_records.append({
                    "record_identifier": rsn, "component": component, "reason": reason,
                })

    print(f"\nDone: {converted} converted, {skipped_existing} already present "
          f"(skipped), {len(skipped_records)} failed (max_workers={max_workers}).")
    if skipped_records:
        print("Skipped records:")
        for s in skipped_records:
            print(f"  {s['record_identifier']} {s['component']}: {s['reason']}")
    return converted, skipped_records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert selected NGA-Sub AT2 records to project JSON format."
    )
    parser.add_argument("src", help="Root folder containing NGASub_RSN_<rsn>/ subfolders.")
    parser.add_argument("selection_csv", help="CSV with record_identifier, component columns.")
    parser.add_argument("dst", help="Output folder for JSON files.")
    parser.add_argument("--filename-map", required=True,
                        help="Path to NGASub_SA_H1_H2_filenamemap.csv.")
    parser.add_argument("--metadata", default=None,
                        help="Path to NGASub_Metadata_SA_rotD50.csv (optional).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers (default: max(cpu_count - 3, 1)).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert records even if their JSON already exists in dst.")
    args = parser.parse_args(argv)

    # CLI nicety: if no usable metadata flatfile, confirm before proceeding with
    # empty event_id/station_code (the importable function stays non-interactive).
    if not (args.metadata and Path(args.metadata).is_file()):
        if args.metadata:
            print(f"⚠️  Metadata flatfile not found: {args.metadata}")
        else:
            print("⚠️  No --metadata flatfile provided.")
        answer = input(
            "Continue with event_id and station_code set to empty strings? [y/N]: "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    convert_ngasub_records(args.src, args.selection_csv, args.dst,
                           args.filename_map, args.metadata,
                           max_workers=args.max_workers, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
