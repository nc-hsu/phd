"""Write and read a fragility data container as plain arrays plus a JSON manifest.

Why this exists
---------------
covmar rebuilds contrasts, V, bootstrap replicates and bias corrections from a
complete data container, but it may not import the phd loader
(``load_fragility_data``). So the content of each phd ``FragilityData`` is
frozen here in a form anything with numpy and pandas can read back:

    containers/<quantity>.npz    every array of the container (np.savez_compressed)
    containers/<quantity>.json   the manifest: names, dtypes, dict layouts, and a
                                 SHA-256 per array

This module imports only numpy, pandas and the standard library, never any phd
code. It deals in the container's PARTS (a dict of the eight dataclass fields:
quantity, im_tag, rows, groups, reps, estimates, estimates_bc, bias), so the
same reader serves both a phd ``FragilityData(**parts)`` and a covmar
``FragilityData(**parts, schema=...)``.

Encoding rules
--------------
* Floats, ints and bools are stored as their raw numpy arrays, so floats
  round-trip bit for bit.
* Object columns are allowed only when every element is a Python ``str``. They
  are stored as a fixed-width unicode array and turned back into object dtype
  on reading. Anything else in an object column refuses to be written, because
  it would not round-trip.
* No pickle (``allow_pickle=False`` on reading), no CSV.
* An index is stored one array per level, by level POSITION (names can be
  None); the manifest holds the names, the dtypes and whether it is a
  MultiIndex.

npz key layout
--------------
    rows/columns/<j>             column j of ``rows`` (order in the manifest)
    rows/index/<i>               level i of the ``rows`` index (site, n_storeys)
    groups/index/0               the design_group_id values in canonical order
    reps/<sym>/values            (n or J, k) replicate matrix of one symbol
    reps/<sym>/index/<i>         its row index, one array per level
    reps/<sym>/columns/0         its column labels (the replicate ids)
    <field>/<sym>/values         estimates / estimates_bc / bias: one Series
    <field>/<sym>/index/<i>      and its index
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Version of this file layout, written into every manifest so a reader can
# refuse a layout it does not know.
FORMAT_VERSION: int = 1

# The four dict-valued fields of the container, in dataclass order. ``reps``
# holds DataFrames; the other three hold Series.
DICT_FIELDS: tuple[str, ...] = ("reps", "estimates", "estimates_bc", "bias")

# Element kind written in the manifest for each stored array.
#   "native"      stored as is (float64, int64, bool, ...)
#   "str_object"  an object column of Python str, stored as fixed-width unicode
KIND_NATIVE: str = "native"
KIND_STR_OBJECT: str = "str_object"


# =============================================================================
# Hashing
# =============================================================================

def array_sha256(arr: np.ndarray) -> str:
    """SHA-256 of an array's raw bytes in C order.

    The dtype and shape are recorded beside the hash in the manifest, so two
    arrays with the same bytes but different dtype or shape are still told
    apart.
    """
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# =============================================================================
# Writing
# =============================================================================

class _Writer:
    """Collects arrays for one npz file and their manifest entries."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.entries: dict[str, dict[str, Any]] = {}

    def put(self, key: str, values: Any) -> dict[str, Any]:
        """Store one 1-D or 2-D array under ``key``; return its dtype description.

        Object arrays must hold only Python ``str``; they are converted to a
        unicode array and marked ``str_object`` so the reader restores object
        dtype. Any other object content raises, as it would not round-trip.
        """
        arr = np.asarray(values)
        if arr.dtype == object:
            bad = {type(v).__name__ for v in arr.ravel() if type(v) is not str}
            if bad:
                raise TypeError(f"{key}: object array holds {sorted(bad)}, only str can be "
                                "stored without pickle")
            stored = np.array(arr.tolist(), dtype=str)
            # An empty-string-only column would give dtype <U1 holding "" -- fine;
            # a length check guards against numpy truncating anything.
            if stored.shape != arr.shape:
                raise ValueError(f"{key}: shape changed while converting strings")
            kind, pandas_dtype = KIND_STR_OBJECT, "object"
        elif arr.dtype.kind in "biuf":
            stored, kind, pandas_dtype = arr, KIND_NATIVE, str(arr.dtype)
        else:
            raise TypeError(f"{key}: dtype {arr.dtype} is not supported")
        if key in self.arrays:
            raise KeyError(f"duplicate npz key {key}")
        self.arrays[key] = stored
        self.entries[key] = {"dtype": stored.dtype.str, "shape": list(stored.shape),
                             "kind": kind, "sha256": array_sha256(stored)}
        return {"dtype": pandas_dtype, "kind": kind}

    def put_index(self, prefix: str, index: pd.Index) -> dict[str, Any]:
        """Store an Index or MultiIndex one level at a time; return its description."""
        levels = []
        for i in range(index.nlevels):
            vals = index.get_level_values(i)
            levels.append(self.put(f"{prefix}/{i}", vals.to_numpy()))
        return {"type": "MultiIndex" if isinstance(index, pd.MultiIndex) else "Index",
                "names": list(index.names), "levels": levels, "length": len(index)}


def container_parts(data: Any) -> dict[str, Any]:
    """The eight stored fields of a container object, as a dict."""
    return {"quantity": data.quantity, "im_tag": data.im_tag, "rows": data.rows,
            "groups": data.groups, **{f: getattr(data, f) for f in DICT_FIELDS}}


def write_container(parts: dict[str, Any], folder: Path) -> tuple[Path, Path]:
    """Write ``<folder>/<quantity>.npz`` and ``<folder>/<quantity>.json``.

    Parameters
    ----------
    parts : dict
        The container fields (see :func:`container_parts`).
    folder : Path
        Destination folder, created if needed.

    Returns
    -------
    (npz path, manifest path)
    """
    w = _Writer()
    rows: pd.DataFrame = parts["rows"]
    if not rows.columns.is_unique:
        raise ValueError("rows has duplicate column names")

    # -- rows: every column by position, plus its index levels ------------------
    columns = [w.put(f"rows/columns/{j}", rows.iloc[:, j].to_numpy())
               for j in range(rows.shape[1])]
    rows_desc = {
        "column_names": [str(c) for c in rows.columns],
        "column_names_dtype": str(rows.columns.dtype),
        "columns_name": rows.columns.name,
        "columns": columns,
        "index": w.put_index("rows/index", rows.index),
    }

    # -- groups --------------------------------------------------------------------
    groups_desc = w.put_index("groups/index", parts["groups"])

    # -- the four dict fields --------------------------------------------------------
    dict_desc: dict[str, Any] = {}
    for fld in DICT_FIELDS:
        d: dict[str, Any] = parts[fld]
        entry: dict[str, Any] = {"order": list(d), "present": [], "none": [], "items": {}}
        for sym, obj in d.items():
            if obj is None:
                entry["none"].append(sym)
                continue
            entry["present"].append(sym)
            base = f"{fld}/{sym}"
            if isinstance(obj, pd.DataFrame):
                if not obj.dtypes.eq(obj.dtypes.iloc[0]).all():
                    raise TypeError(f"{base}: mixed column dtypes, cannot store as one matrix")
                item = {"type": "DataFrame",
                        "values": w.put(f"{base}/values", obj.to_numpy()),
                        "shape": list(obj.shape),
                        "index": w.put_index(f"{base}/index", obj.index),
                        "columns": w.put_index(f"{base}/columns", obj.columns)}
            elif isinstance(obj, pd.Series):
                item = {"type": "Series", "name": obj.name,
                        "values": w.put(f"{base}/values", obj.to_numpy()),
                        "index": w.put_index(f"{base}/index", obj.index)}
            else:
                raise TypeError(f"{base}: {type(obj).__name__} is not a DataFrame or Series")
            entry["items"][sym] = item
        dict_desc[fld] = entry

    manifest = {
        "format_version": FORMAT_VERSION,
        "quantity": parts["quantity"],
        "im_tag": parts["im_tag"],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rows": rows_desc,
        "groups": groups_desc,
        "fields": dict_desc,
        "arrays": w.entries,
    }

    folder.mkdir(parents=True, exist_ok=True)
    npz_path = folder / f"{parts['quantity']}.npz"
    json_path = folder / f"{parts['quantity']}.json"
    np.savez_compressed(npz_path, **w.arrays)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return npz_path, json_path


# =============================================================================
# Reading
# =============================================================================

class _Reader:
    """Reads arrays from one npz, checking each against its manifest entry."""

    def __init__(self, npz: Any, entries: dict[str, dict[str, Any]]) -> None:
        self.npz = npz
        self.entries = entries
        self.used: set[str] = set()

    def get(self, key: str) -> np.ndarray:
        """The stored array, hash-checked, with object dtype restored for strings."""
        e = self.entries[key]
        arr = self.npz[key]
        if arr.dtype.str != e["dtype"] or list(arr.shape) != e["shape"]:
            raise ValueError(f"{key}: stored {arr.dtype.str}{list(arr.shape)} does not "
                             f"match the manifest {e['dtype']}{e['shape']}")
        if array_sha256(arr) != e["sha256"]:
            raise ValueError(f"{key}: SHA-256 does not match the manifest")
        self.used.add(key)
        if e["kind"] == KIND_STR_OBJECT:
            # astype(object) turns numpy str_ elements into Python str.
            return arr.astype(object)
        return arr

    def index(self, prefix: str, desc: dict[str, Any]) -> pd.Index:
        """Rebuild an Index / MultiIndex from its levels and description."""
        arrays = [pd.Index(self.get(f"{prefix}/{i}"), dtype=lev["dtype"])
                  for i, lev in enumerate(desc["levels"])]
        if desc["type"] == "MultiIndex":
            return pd.MultiIndex.from_arrays(arrays, names=desc["names"])
        return pd.Index(arrays[0], dtype=desc["levels"][0]["dtype"], name=desc["names"][0])


def read_container(folder: Path, quantity: str) -> dict[str, Any]:
    """Read ``<folder>/<quantity>.*`` back into the container's parts.

    Every array is checked against its dtype, shape and SHA-256 in the manifest,
    and every array in the npz must be used, so a partial or tampered file
    fails loudly instead of loading short.

    Returns
    -------
    dict
        quantity, im_tag, rows, groups, reps, estimates, estimates_bc, bias --
        the keyword arguments of the container dataclass.
    """
    manifest = json.loads((folder / f"{quantity}.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unknown container format {manifest['format_version']}")
    with np.load(folder / f"{quantity}.npz", allow_pickle=False) as npz:
        r = _Reader(npz, manifest["arrays"])
        extra = set(npz.files) ^ set(manifest["arrays"])
        if extra:
            raise ValueError(f"npz and manifest disagree on keys: {sorted(extra)}")

        # -- rows ----------------------------------------------------------------------
        # Each column comes back as a numpy array of its own dtype (object for
        # strings), which the DataFrame keeps as the column dtype.
        rd = manifest["rows"]
        index = r.index("rows/index", rd["index"])
        rows = pd.DataFrame({name: r.get(f"rows/columns/{j}")
                             for j, name in enumerate(rd["column_names"])}, index=index)
        rows.columns = pd.Index(rd["column_names"], dtype=rd["column_names_dtype"],
                                name=rd["columns_name"])

        # -- groups -------------------------------------------------------------------
        groups = r.index("groups/index", manifest["groups"])

        # -- dict fields: rebuilt in the stored key order, None kept as None -------------
        fields: dict[str, dict[str, Any]] = {}
        for fld in DICT_FIELDS:
            fd = manifest["fields"][fld]
            out: dict[str, Any] = {}
            for sym in fd["order"]:
                if sym in fd["none"]:
                    out[sym] = None
                    continue
                item, base = fd["items"][sym], f"{fld}/{sym}"
                values = r.get(f"{base}/values")
                idx = r.index(f"{base}/index", item["index"])
                if item["type"] == "DataFrame":
                    out[sym] = pd.DataFrame(values, index=idx,
                                            columns=r.index(f"{base}/columns", item["columns"]))
                else:
                    out[sym] = pd.Series(values, index=idx, name=item["name"],
                                         dtype=item["values"]["dtype"])
            fields[fld] = out

        unused = set(npz.files) - r.used
        if unused:
            raise ValueError(f"arrays in the npz that the manifest never placed: {sorted(unused)}")

    return {"quantity": manifest["quantity"], "im_tag": manifest["im_tag"],
            "rows": rows, "groups": groups, **fields}
