import json
import re
from pathlib import Path

# Regex patterns
_FLOAT_RE = re.compile(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?')
_DATE_RE = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{2})\b')  # matches mm/dd/yy


def extract_metadata(lines):
    """Extract metadata lines (top or bottom 4 lines)."""
    if len(lines) < 4:
        raise ValueError("File too short to contain metadata lines.")

    top_meta = [line.strip() for line in lines[:4]]
    bottom_meta = [line.strip() for line in lines[-4:]]

    if any("ACCELERATION TIME HISTORY" in line.upper() for line in top_meta):
        return top_meta, 4, len(lines)
    elif any("ACCELERATION TIME HISTORY" in line.upper() for line in bottom_meta):
        return bottom_meta, 0, len(lines) - 4
    else:
        raise ValueError("Metadata not found in expected locations.")


def parse_floats_from_line(line):
    """Return list of floats parsed from a line."""
    return [float(m.group(0)) for m in _FLOAT_RE.finditer(line)]


def extract_date_from_description(description):
    """Find mm/dd/yy in description and return reformatted as dd/mm/yy."""
    m = _DATE_RE.search(description)
    if not m:
        return None
    mm, dd, yy = m.groups()
    return f"{int(dd):02d}/{int(mm):02d}/{yy}"


def read_existing_metadata(json_path):
    """Read Record Database, Record Description, and dt from an existing JSON file."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return {
                "Record Database": data.get("Record Database"),
                "Record Description": data.get("Record Description"),
                "dt": data.get("dt")
            }
    except Exception:
        return None


def metadata_matches(meta1, meta2):
    """Return True if two metadata dicts are identical (ignoring case/whitespace)."""
    if not meta1 or not meta2:
        return False
    return all(
        str(meta1.get(k, "")).strip().lower() == str(meta2.get(k, "")).strip().lower()
        for k in ["Record Database", "Record Description", "dt"]
    )


def determine_unique_filename(base_path, record_meta, output_dir, prepend, append):
    """
    Check for duplicate filenames and determine correct filename.
    If metadata matches → overwrite.
    If different → append incremental '_r0i' suffix.
    """
    output_dir = Path(output_dir)
    base_name = f"{prepend}{base_path.stem}{append}"
    json_path = output_dir / f"{base_name}.json"

    if not json_path.exists():
        return json_path  # first occurrence

    # Compare metadata with existing file(s)
    existing_meta = read_existing_metadata(json_path)
    if metadata_matches(existing_meta, record_meta):
        print(f"ℹ️  {json_path.name}: same metadata found — overwriting existing file.")
        return json_path

    # Different metadata → find next _r0i index
    counter = 2
    while True:
        alt_name = f"{base_name}_r0{counter}.json"
        alt_path = output_dir / alt_name
        if not alt_path.exists():
            return alt_path
        existing_meta = read_existing_metadata(alt_path)
        if metadata_matches(existing_meta, record_meta):
            print(f"ℹ️  {alt_name}: same metadata found — overwriting existing file.")
            return alt_path
        counter += 1


def convert_at2_to_json(at2_path, output_dir, prepend_str="", append_str=""):
    """Convert a single .at2 file to JSON according to the custom format."""
    at2_path = Path(at2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(at2_path, "r", errors="replace") as f:
        lines = f.readlines()

    # --- Extract metadata ---
    try:
        metadata_lines, data_start, data_end = extract_metadata(lines)
    except ValueError as e:
        print(f"⚠️  {at2_path.name}: {e} — skipped.")
        return

    record_database = metadata_lines[0]
    record_description = metadata_lines[1]
    third_line = metadata_lines[2].upper()

    # Check units
    if "ACCELERATION TIME HISTORY IN UNITS OF G" not in third_line:
        print(f"⚠️  {at2_path.name}: Units not 'g' — skipped.")
        return

    # Extract dt (second float)
    last_line = metadata_lines[3]
    nums = parse_floats_from_line(last_line)
    dt = nums[1] if len(nums) >= 2 else None

    # --- Extract acceleration data ---
    record = []
    for line in lines[data_start:data_end]:
        if not line.strip():
            continue
        record.extend(parse_floats_from_line(line))

    if not record:
        print(f"⚠️  {at2_path.name}: No acceleration data found — skipped.")
        return

    # --- Extract date and append to eq_index ---
    date_str = extract_date_from_description(record_description)
    eq_index_base = at2_path.stem
    eq_index = f"{eq_index_base}_{date_str}" if date_str else eq_index_base

    # --- Metadata dictionary for duplicate checking ---
    record_meta = {
        "Record Database": record_database,
        "Record Description": record_description,
        "dt": dt
    }

    # --- Determine unique filename (duplicate handling) ---
    json_path = determine_unique_filename(at2_path, record_meta, output_dir, prepend_str, append_str)

    # --- Build JSON object ---
    record_dict = {
        "eq_index": eq_index,
        "Record Database": record_database,
        "Record Description": record_description,
        "dt": dt,
        "normalisation_factor": 1.0,
        "units": "g",
        "record_type": "acc",
        "record": record
    }

    # --- Write JSON ---
    with open(json_path, "w") as f:
        json.dump(record_dict, f, indent=4)

    print(f"✅ Converted {at2_path.name} -> {json_path.name}")


def convert_folder_recursively(folder_path, output_dir, prepend_str="", append_str=""):
    """Recursively convert all .at2 files in folder and subfolders."""
    folder_path = Path(folder_path)
    files = list(folder_path.rglob("*.at2"))
    if not files:
        print(f"No .at2 files found in {folder_path}")
        return

    for f in files:
        convert_at2_to_json(f, output_dir, prepend_str, append_str)


if __name__ == "__main__":
    # ---------------- USER SETTINGS ----------------
    input_path = Path(r"C:\Users\clemettn\Documents\phd\data\innoseis_records\original_at2_sets\medium_record_set")  # ← change this
    output_folder = Path(r"C:\Users\clemettn\Documents\phd\data\innoseis_records\medium_seismicity")                 # output directory

    # Optional prepend/append to output filenames
    prepend = ""      # e.g. "NGAW2_"
    append = ""       # e.g. "_norm"
    # ------------------------------------------------

    if input_path.is_file() and input_path.suffix.lower() == ".at2":
        convert_at2_to_json(input_path, output_folder, prepend, append)
    elif input_path.is_dir():
        convert_folder_recursively(input_path, output_folder, prepend, append)
    else:
        print("Input path is not a file or folder, or file extension is not .at2. Please update `input_path` in the script.")