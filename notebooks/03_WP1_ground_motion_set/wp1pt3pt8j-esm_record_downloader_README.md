
# ESM HDF5 → JSON Pipeline

This project provides a **robust, restart‑safe pipeline** to download earthquake waveform records from the **European Strong Motion (ESM) database**, store them in **HDF5 format**, and convert them into **JSON files (one per component)**.

The pipeline is designed for **large-scale datasets (thousands of records)** and includes:

- Concurrent downloads
- Batch processing
- Crash‑safe resume capability
- Progress monitoring
- Robust dataframe normalization
- Per‑component JSON output

---

# Features

### 1. Concurrent Downloads

Records are downloaded from the ESM API:

```
https://esm-db.eu/esmws/eventdata/1/query
```

Parameters used:

- `eventid`
- `station`
- `data-type=ACC`

Downloads run using **ThreadPoolExecutor (concurrency = 4)**.

---

### 2. Resume‑Safe Manifest

Every download attempt is logged in:

```
out_hdf5/download_manifest.csv
```

This file stores:

| column | description |
|------|-------------|
record_id | event_id__station_code__location_code |
status | downloaded / skipped / failed |
http_status | HTTP response code |
error_type | exception type |
error_message | explanation |
timestamp | UTC timestamp |

If the program crashes, **re‑running the pipeline resumes automatically**.

---

### 3. Batch Processing

Large datasets are processed in **batches**:

```
batch_size = 500
```

This prevents:

- memory spikes
- server overload
- connection instability

---

### 4. Progress Monitoring

Progress is shown using **tqdm**:

```
Downloading:  35%|█████
Converting:   78%|██████████
```

If tqdm is unavailable, simple counters are printed.

---

# Input Data Format

Example metadata CSV:

| event_id | event_time | station_code | location_code |
|--------|------------|-------------|--------------|
EMSC-20160824_0000006 | 8/24/2016 1:36 | AQG | 00 |

The loader automatically handles **two‑row header CSV files**.

The dataframe is normalized to ensure:

```
event_id
station_code
location_code
```

---

# Output Structure

```
project/
│
├── example_df.csv
│
├── out_hdf5/
│   ├── EMSC-20160824_0000006__AQG__00.h5
│   ├── download_manifest.csv
│
├── out_json/
│   ├── EMSC-20160824_0000006__AQG__00_U.json
│   ├── EMSC-20160824_0000006__AQG__00_V.json
│   ├── EMSC-20160824_0000006__AQG__00_W.json
│
└── out_json_subset/
```

---

# JSON Output Format

Each waveform component becomes its own JSON file.

Filename format:

```
event_id__station_code__location_code_component.json
```

Example:

```
EMSC-20160824_0000006__AQG__00_U.json
```

Example JSON:

```json
{
  "eq_index": "EMSC-20160824_0000006__AQG__00",
  "database": "esm",
  "dt": 0.01,
  "units": "g",
  "record_type": "acc",
  "component": "U",
  "record": [0.0012, -0.0009]
}
```

Notes:

- **U/V/W correspond to the 1st, 2nd, 3rd datasets in the HDF5 file**
- They **do NOT necessarily correspond to E/N/Z **

---

# Pipeline Workflow

```
CSV Metadata
      │
      ▼
Normalize dataframe
      │
      ▼
Download HDF5 files
      │
      ▼
Save manifest + resume capability
      │
      ▼
Convert HDF5 → JSON
      │
      ▼
Optional subset copy
```

---

# Running the Pipeline

Example runner:

```python
CSV_PATH = Path("example_df.csv")

df_raw = load_example_df_robust(CSV_PATH)
df_norm = normalize_metadata_df(df_raw)

download_records_to_hdf5(
    df=df_norm,
    dst_dir=Path("out_hdf5"),
    batch_size=500,
    max_workers=4
)

for row in df_norm.itertuples():
    hdf5_record_to_json(
        hdf5_fp=Path("out_hdf5") / f"{row.event_id}__{row.station_code}__{row.location_code}.h5",
        dst_dir=Path("out_json"),
        metadata=row._asdict()
    )
```

---

# Key Functions

### download_records_to_hdf5()

Downloads waveform records from ESM.

Features:

- concurrency = 4
- retries + backoff
- batch processing
- resume-safe manifest logging

---

### hdf5_record_to_json()

Reads HDF5 waveform data and writes **one JSON per component**.

Mapping:

```
U → dataset[0]
V → dataset[1]
W → dataset[2]
```

Acceleration values are converted to **g**.

---

### copy_records()

Copies a subset of JSON records into another folder.

Useful for:

- quick testing
- model training subsets
- debugging

---

# Dependencies

Install required libraries:

```bash
pip install pandas numpy requests h5py tqdm
```

---

# Performance

Typical performance:

| dataset size | runtime |
|--------------|--------|
100 records | ~3-4 minutes |
1000 records | ~10–15 minutes |
10000 records | depends on network speed |

---

# Failure Handling

Failures are automatically logged.

Examples:

| status | explanation |
|------|-------------|
failed | HTTP error |
invalid_metadata | missing event or station |
skipped | already downloaded |

Conversion errors are logged to:

```
out_json/conversion_errors.csv
```

---

# Restarting After Crash

Simply rerun the pipeline.

The downloader reads:

```
download_manifest.csv
```

and skips previously downloaded files.

---


