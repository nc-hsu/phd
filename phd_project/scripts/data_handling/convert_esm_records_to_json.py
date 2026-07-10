"""Convert selected ESM records (ASDF/HDF5) to the project's JSON format.

ESM records are stored one HDF5 file per record on the share, named
``{event_id}__{station_code}__{location_code}.h5`` (ASDF format). Each file's
``/Waveforms/{NET}.{STA}/`` group holds three acceleration datasets sorted
E/N/Z; in key-sorted order the 1st is component U, 2nd V, 3rd W. The waveform
data is in cm/s^2 and is converted to g on output.

Given a selection CSV of ``(record_identifier, component)`` rows -- where
``record_identifier`` is ``{event_id}_{station_code}_{location_code}`` -- this
script writes one JSON record per row, using the same schema and filename
convention as the NGA-Sub converter (``record`` field last;
``{record_identifier}__{component}.json``).

Usage:
    python convert_esm_records_to_json.py <src> <selection_csv> <dst>

    src           Folder containing the ESM ``*.h5`` files.
    selection_csv CSV with ``record_identifier, component`` columns.
    dst           Output folder for JSON files.
"""

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

# Console messages use emoji; make sure they survive a non-UTF-8 stdout (e.g.
# the default Windows cp1252 console) instead of crashing the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

G0 = 9.80665  # m/s^2 per g

# Component label -> dataset index within the key-sorted station group.
_COMPONENT_INDEX = {"U": 0, "V": 1, "W": 2}


def _normalise(text):
    """Lowercase and strip whitespace (for fuzzy header matching)."""
    return re.sub(r"\s+", "", str(text)).lower()


def _find_column(fieldnames, *aliases):
    """Return the actual header matching one of ``aliases`` (case-insensitive)."""
    wanted = {_normalise(a) for a in aliases}
    for name in fieldnames or []:
        if _normalise(name) in wanted:
            return name
    return None


def _to_g(values, units):
    """Convert an acceleration array to units of g."""
    u = (units or "").strip().lower()
    if u in ("g", "grav", "gravity", "gravities"):
        return values.astype(float)
    if "cm/s" in u or "gal" in u:
        return values.astype(float) / (100.0 * G0)
    if "m/s" in u:
        return values.astype(float) / G0
    return values.astype(float)


def read_selection(path):
    """Yield ``(record_identifier, component)`` tuples from the selection CSV.

    Detects the header row containing ``record_identifier`` so that both a
    plain single-header CSV and the 2-row MultiIndex header produced by
    ``DataFrame.to_csv`` (columns like ``("metadata", "record_identifier")``)
    are handled.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        norm = {_normalise(c) for c in row}
        if "record_identifier" in norm and ("component" in norm or "comp" in norm):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"{path.name}: could not find a header row with "
            f"'record_identifier' and 'component'."
        )

    header = rows[header_idx]
    id_col = next(i for i, c in enumerate(header)
                  if _normalise(c) == "record_identifier")
    comp_col = next(i for i, c in enumerate(header)
                    if _normalise(c) in ("component", "comp"))

    for row in rows[header_idx + 1:]:
        if len(row) <= max(id_col, comp_col):
            continue
        record_identifier = str(row[id_col]).strip()
        component = str(row[comp_col]).strip().upper()
        if not record_identifier:
            continue
        yield record_identifier, component


def parse_record_identifier(record_identifier):
    """Split ``{event_id}_{station_code}_{location_code}`` from the right.

    ``event_id`` may itself contain underscores (e.g. ``EMSC-20040223_0000011``)
    while station/location codes do not, so split on the last two underscores.
    """
    parts = record_identifier.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(
            f"record_identifier '{record_identifier}' does not have the form "
            f"event_id_station_code_location_code."
        )
    return parts[0], parts[1], parts[2]


def resolve_hdf5_path(src, event_id, station_code, location_code):
    """Locate the HDF5 file for a record, tolerating a zero-padded location.

    Returns the resolved ``Path`` or ``None`` if not found / ambiguous.
    """
    src = Path(src)
    exact = src / f"{event_id}__{station_code}__{location_code}.h5"
    if exact.is_file():
        return exact

    matches = sorted(src.glob(f"{event_id}__{station_code}__*.h5"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"⚠️  {event_id}/{station_code}: {len(matches)} location matches "
              f"{[m.name for m in matches]} — skipped.")
        return None
    return None


def _find_station_group(h5, station_code):
    """Return the /Waveforms station group (single group, or matched by code)."""
    if "Waveforms" not in h5:
        raise KeyError("HDF5 missing /Waveforms")
    wf = h5["Waveforms"]
    keys = list(wf.keys())
    if len(keys) == 1:
        return wf[keys[0]]
    matches = [k for k in keys
               if f".{station_code}" in str(k) or str(k).endswith(station_code)]
    if matches:
        matches.sort(key=lambda x: len(str(x)))
        return wf[matches[0]]
    raise KeyError(
        f"station group not found for station_code={station_code}; "
        f"Waveforms keys={keys[:10]}"
    )


def convert_record(record_identifier, component, src, dst):
    """Convert one (record_identifier, component) selection into a JSON file.

    Returns None if a file was written, or a short reason string if the record
    was skipped.
    """
    try:
        event_id, station_code, location_code = parse_record_identifier(record_identifier)
    except ValueError as e:
        print(f"⚠️  {e} — skipped.")
        return "bad record_identifier"

    if component not in _COMPONENT_INDEX:
        print(f"⚠️  {record_identifier}: unknown component '{component}' "
              f"(expected U/V/W) — skipped.")
        return "unknown component"

    h5_path = resolve_hdf5_path(src, event_id, station_code, location_code)
    if h5_path is None:
        print(f"⚠️  {record_identifier} {component}: HDF5 file not found — skipped.")
        return "HDF5 file not found"

    try:
        with h5py.File(h5_path, "r") as h5:
            stgrp = _find_station_group(h5, station_code)
            datasets = [stgrp[k] for k in sorted(stgrp.keys())
                        if isinstance(stgrp[k], h5py.Dataset)]
            idx = _COMPONENT_INDEX[component]
            if idx >= len(datasets):
                print(f"⚠️  {h5_path.name}: only {len(datasets)} datasets, "
                      f"need index {idx} for {component} — skipped.")
                return "too few datasets"
            ds = datasets[idx]
            sr = ds.attrs.get("sampling_rate", None)
            if sr is None:
                print(f"⚠️  {h5_path.name}: sampling_rate missing — skipped.")
                return "sampling_rate missing"
            dt = 1.0 / float(sr)
            record = _to_g(np.array(ds[()], dtype=float).ravel(), "cm/s^2").tolist()
    except (KeyError, OSError, ValueError) as e:
        print(f"⚠️  {h5_path.name}: {e} — skipped.")
        return str(e)

    record_dict = {
        "eq_index": record_identifier,
        "database": "esm",
        "dt": dt,
        "duration": dt * len(record),
        "units": "g",
        "record_type": "acc",
        "component": component,
        "event_id": event_id,
        "station_code": station_code,
        "location_code": location_code,
        "record": record,
    }

    out_path = Path(dst) / f"{record_identifier}__{component}.json"
    with open(out_path, "w") as f:
        json.dump(record_dict, f, indent=4)

    print(f"✅ {h5_path.name} [{component}] -> {out_path.name}")
    return None


def convert_esm_records(src, selection_csv, dst, max_workers=None, overwrite=False):
    """Convert all ``(record_identifier, component)`` rows of the selection CSV.

    ``src``, ``selection_csv`` and ``dst`` may be paths or strings. Records are
    read from the (often network-mounted) source in parallel using a thread
    pool, since the work is I/O-bound. ``max_workers`` defaults to
    ``max(cpu_count - 3, 1)``. Rows whose target JSON already exists in ``dst``
    are skipped without reading the source, unless ``overwrite`` is True.
    Returns ``(converted_count, skipped_records)`` where ``skipped_records`` is a
    list of ``{"record_identifier", "component", "reason"}`` dicts for every row
    that could not be converted.
    """
    if max_workers is None:
        max_workers = max((os.cpu_count() or 1) - 3, 1)

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    rows = list(read_selection(selection_csv))
    converted = 0
    skipped_existing = 0
    skipped_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for record_identifier, component in rows:
            out_path = dst / f"{record_identifier}__{component}.json"
            if out_path.exists() and not overwrite:
                skipped_existing += 1
                continue
            futures.append((record_identifier, component,
                            executor.submit(convert_record, record_identifier,
                                            component, src, dst)))
        for record_identifier, component, future in futures:
            reason = future.result()
            if reason is None:
                converted += 1
            else:
                skipped_records.append({
                    "record_identifier": record_identifier,
                    "component": component,
                    "reason": reason,
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
        description="Convert selected ESM records (HDF5) to project JSON format."
    )
    parser.add_argument("src", help="Folder containing the ESM *.h5 files.")
    parser.add_argument("selection_csv", help="CSV with record_identifier, component columns.")
    parser.add_argument("dst", help="Output folder for JSON files.")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers (default: max(cpu_count - 3, 1)).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert records even if their JSON already exists in dst.")
    args = parser.parse_args(argv)

    convert_esm_records(args.src, args.selection_csv, args.dst,
                        max_workers=args.max_workers, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
