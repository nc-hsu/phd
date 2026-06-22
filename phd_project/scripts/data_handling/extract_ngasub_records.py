"""Extract and reorganise NGA-Subduction ground-motion record downloads.

Records are downloaded from OneDrive as a zip-of-zips. Each inner zip
(``NGAsubductionGMdatabaseDownload_*.zip``) holds record files for many sites,
named ``NGAsubRSN<rsn>_<station>.<AT2|AVD>`` (6 files per RSN). This script
splits every site's files into its own ``NGASub_RSN_<rsn>`` folder and,
optionally, updates a download-manifest CSV.

Usage:
    python extract_ngasub_records.py <src> <dst> [manifest_csv]

    src          OneDrive .zip, an inner .zip, or a folder containing them.
    dst          Folder under which the NGASub_RSN_<rsn> subfolders are created.
    manifest_csv (optional) CSV with at least an RSN column and a "Status"
                 column; extracted RSNs are marked "downloaded".
"""

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

# RSN embedded in record filenames, e.g. "NGAsubRSN1002974_K215360.AT2".
_RSN_RE = re.compile(r"NGAsubRSN(\d+)", re.IGNORECASE)

# Metadata-report files bundled with the records; silently ignored.
_IGNORE_RE = re.compile(r"NGAsubWebPortal_MetadataReport", re.IGNORECASE)

# Normalised header names accepted for the manifest RSN column.
_RSN_COLUMN_ALIASES = {"rsn", "rsnid"}

_STATUS_DOWNLOADED = "downloaded"


def _normalise(text):
    """Lowercase and strip all whitespace (for fuzzy header/RSN matching)."""
    return re.sub(r"\s+", "", str(text)).lower()


def _extract_from_zip(zip_bytes, dst, extracted_rsns, skipped):
    """Recursively walk a zip (given as bytes), extracting record files.

    Nested zips are read in memory so they never hit disk.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue
            name = entry.filename
            if name.lower().endswith(".zip"):
                _extract_from_zip(zf.read(name), dst, extracted_rsns, skipped)
            else:
                _write_record(name, lambda n=name: zf.read(n), dst,
                              extracted_rsns, skipped)


def _write_record(name, read_bytes, dst, extracted_rsns, skipped):
    """Write a single record file into its NGASub_RSN_<rsn> folder."""
    basename = Path(name).name
    if _IGNORE_RE.search(basename):
        return
    match = _RSN_RE.search(basename)
    if not match:
        if basename not in skipped:
            print(f"⚠️  Skipped (no RSN in name): {basename}")
            skipped.add(basename)
        return

    rsn = match.group(1)
    out_dir = dst / f"NGASub_RSN_{rsn}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / basename).write_bytes(read_bytes())
    extracted_rsns.add(rsn)


def extract_records(src, dst):
    """Extract all NGA-Sub record files found under ``src`` into ``dst``.

    ``src`` may be a .zip file (including a zip-of-zips) or a folder containing
    zips and/or loose record files. Returns the set of extracted RSNs.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    extracted_rsns = set()
    skipped = set()

    if src.is_file() and src.suffix.lower() == ".zip":
        _extract_from_zip(src.read_bytes(), dst, extracted_rsns, skipped)
    elif src.is_dir():
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".zip":
                _extract_from_zip(path.read_bytes(), dst, extracted_rsns, skipped)
            else:
                _write_record(path.name, path.read_bytes, dst,
                              extracted_rsns, skipped)
    else:
        raise SystemExit(f"src is not a .zip file or a folder: {src}")

    return extracted_rsns


def _find_rsn_column(fieldnames):
    """Return the manifest header that holds RSN numbers, or None."""
    for name in fieldnames:
        if _normalise(name) in _RSN_COLUMN_ALIASES:
            return name
    return None


def update_manifest(manifest_path, extracted_rsns):
    """Mark extracted RSNs as 'downloaded' in the manifest CSV.

    Existing rows have their Status set to 'downloaded'; previously unseen RSNs
    are appended with the RSN column set, Status='downloaded', and every other
    column left blank. Prints the matched RSN column name and the count of rows
    that are still not 'downloaded' afterwards.
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rsn_col = _find_rsn_column(fieldnames)
    if rsn_col is None:
        raise SystemExit(
            f"No RSN column found in manifest {manifest_path.name}; "
            f"headers were: {fieldnames}"
        )
    print(f"Using manifest RSN column: '{rsn_col}'")

    if "Status" not in fieldnames:
        raise SystemExit(
            f"No 'Status' column found in manifest {manifest_path.name}; "
            f"headers were: {fieldnames}"
        )

    rows_by_rsn = {_normalise(row.get(rsn_col, "")): row for row in rows}

    for rsn in sorted(extracted_rsns):
        key = _normalise(rsn)
        if key in rows_by_rsn:
            rows_by_rsn[key]["Status"] = _STATUS_DOWNLOADED
        else:
            new_row = {name: "" for name in fieldnames}
            new_row[rsn_col] = rsn
            new_row["Status"] = _STATUS_DOWNLOADED
            rows.append(new_row)
            rows_by_rsn[key] = new_row

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    not_downloaded = sum(
        1 for row in rows
        if _normalise(row.get("Status", "")) != _STATUS_DOWNLOADED
    )
    print(f"{not_downloaded} manifest record(s) not '{_STATUS_DOWNLOADED}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract NGA-Sub ground-motion records into per-RSN folders."
    )
    parser.add_argument("src", help="OneDrive .zip, inner .zip, or folder of zips.")
    parser.add_argument("dst", help="Folder to hold the NGASub_RSN_<rsn> subfolders.")
    parser.add_argument("manifest", nargs="?", default=None,
                        help="Optional download-manifest CSV to update.")
    args = parser.parse_args()

    # Allow emoji/status output on consoles with a legacy code page (e.g. Windows cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    extracted_rsns = extract_records(args.src, args.dst)
    print(f"✅ Extracted {len(extracted_rsns)} records (RSNs).")

    if args.manifest:
        update_manifest(args.manifest, extracted_rsns)


if __name__ == "__main__":
    main()
