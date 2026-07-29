"""Provenance-aware caching for the ground-motion selection pipeline.

The selection pipeline caches expensive intermediate results to pickle files.
Historically those files were reloaded on a fixed filename with *no* check that
the inputs that produced them still matched — which let a changed input
ground-motion database silently feed stale cached ensembles into a new run.

This module adds a lightweight provenance guard:

- :func:`fingerprint` reduces a set of inputs to a content-based ``{name: {hash,
  summary}}`` dict. Hashes are content-based (not memory addresses), so they are
  stable across machines and OSes.
- :func:`write_manifest` writes that dict to a ``<artifact>.manifest.json``
  sidecar next to the cached pickle.
- :func:`load_or_compute` is the single entry point used by the compute
  functions: it loads the cache only when the stored manifest matches the
  current inputs, raises :class:`StaleCacheError` when a *managed* artifact's
  inputs have changed, and otherwise computes, pickles and writes the manifest.

Only the ``hash`` of each named input is compared. Environment metadata (git
commit, timestamp) is recorded for diagnostics but never compared. The
``pickagm`` package version *is* compared, because it can change outputs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


class StaleCacheError(Exception):
    """Raised when a cached artifact's manifest no longer matches its inputs."""


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fingerprint_one(name: str, value) -> dict:
    """Return ``{"hash": str, "summary": str}`` for a single input value.

    Dispatches on type so that DataFrames hash by content, file paths hash by
    their bytes, and everything else falls back to a deterministic pickle hash.
    """
    # pandas objects: hash the values + index, summarise the shape.
    if isinstance(value, (pd.DataFrame, pd.Series)):
        h = _hash_bytes(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        shape = value.shape if isinstance(value, pd.DataFrame) else (len(value),)
        return {"hash": h, "summary": f"pandas shape={tuple(shape)}"}

    # A path to an existing file: hash the file bytes (cheap, stable).
    if isinstance(value, (str, Path)):
        p = Path(value)
        if p.is_file():
            h = _hash_bytes(p.read_bytes())
            return {"hash": h, "summary": f"file {p.name} ({p.stat().st_size} bytes)"}
        # A plain string that is not a file path: hash its text.
        h = _hash_bytes(str(value).encode())
        return {"hash": h, "summary": f"str {value!r}"}

    # numpy arrays: hash the raw buffer + dtype/shape.
    if isinstance(value, np.ndarray):
        h = _hash_bytes(value.tobytes() + str((value.dtype, value.shape)).encode())
        return {"hash": h, "summary": f"ndarray shape={value.shape} dtype={value.dtype}"}

    # Plain scalars: hash a stable repr.
    if value is None or isinstance(value, (int, float, bool, str)):
        h = _hash_bytes(repr(value).encode())
        return {"hash": h, "summary": f"{type(value).__name__}={value!r}"}

    # Fallback: deterministic pickle hash (dicts, lists, IMT objects, ...).
    h = _hash_bytes(pickle.dumps(value, protocol=4))
    return {"hash": h, "summary": f"{type(value).__name__}"}


def fingerprint(**inputs) -> dict:
    """Build a comparable fingerprint dict from named inputs.

    Each keyword becomes ``{name: {"hash", "summary"}}``. The installed
    ``pickagm`` version is appended automatically (it affects outputs, so it is
    compared). Use the returned dict as ``fp_dict`` for :func:`load_or_compute`.
    """
    fp = {name: _fingerprint_one(name, value) for name, value in inputs.items()}
    fp["pickagm_version"] = {"hash": _pickagm_version(), "summary": "pickagm version"}
    return fp


def _pickagm_version() -> str:
    try:
        return version("pickagm")
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _manifest_path(artifact_fp: Path) -> Path:
    # Sidecar must end in .json (NOT .manifest — that is gitignored).
    return artifact_fp.with_name(artifact_fp.name + ".manifest.json")


def write_manifest(artifact_fp: Path, fp_dict: dict) -> Path:
    """Write the ``<artifact>.manifest.json`` sidecar next to ``artifact_fp``."""
    artifact_fp = Path(artifact_fp)
    manifest = {
        "inputs": fp_dict,
        "_meta": {
            "artifact": artifact_fp.name,
            "git_commit": _git_commit(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    mp = _manifest_path(artifact_fp)
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)
    return mp


def _diff_inputs(cached: dict, current: dict) -> list[str]:
    """Return human-readable descriptions of every input whose hash changed."""
    changes = []
    for name in sorted(set(cached) | set(current)):
        c = cached.get(name, {}).get("hash")
        n = current.get(name, {}).get("hash")
        if c != n:
            c_sum = cached.get(name, {}).get("summary", "<absent>")
            n_sum = current.get(name, {}).get("summary", "<absent>")
            changes.append(f"  - {name}: {c_sum} [{c}] -> {n_sum} [{n}]")
    return changes


def load_or_compute(
    artifact_fp: Path,
    fp_dict: dict,
    compute_fn: Callable[[], object],
    *,
    force_recompute: bool = False,
):
    """Load a cached artifact when its inputs match, else compute and cache it.

    Behaviour:

    - ``force_recompute`` → always run ``compute_fn``, pickle, write manifest.
    - artifact absent → compute (the "fresh clone / no cache" case; quiet).
    - artifact present but **no manifest** → unmanaged/legacy artifact whose
      provenance cannot be verified → compute and adopt it (writes a manifest).
    - artifact + manifest present and inputs **match** → load and return.
    - artifact + manifest present and inputs **differ** → print the artifact
      name and location, then recompute and overwrite it.
    """
    artifact_fp = Path(artifact_fp)
    manifest_fp = _manifest_path(artifact_fp)

    def _compute_and_save():
        result = compute_fn()
        artifact_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_fp, "wb") as f:
            pickle.dump(result, f)
        write_manifest(artifact_fp, fp_dict)
        return result

    if force_recompute:
        return _compute_and_save()

    if not artifact_fp.is_file():
        return _compute_and_save()

    if not manifest_fp.is_file():
        # Artifact exists but is unmanaged (e.g. pre-refactor pickle). We cannot
        # trust it without a manifest, so recompute and write one rather than
        # silently reusing data of unknown provenance.
        print(f"[cache] '{artifact_fp.name}' has no manifest; recomputing to establish provenance.")
        return _compute_and_save()

    with open(manifest_fp) as f:
        cached_inputs = json.load(f).get("inputs", {})

    changes = _diff_inputs(cached_inputs, fp_dict)
    if changes:
        # Inputs changed since the cache was written. Rather than refusing to
        # proceed, recompute automatically and overwrite the stale artifact.
        print(f"{artifact_fp.name} ({artifact_fp.parent})")
        return _compute_and_save()

    print(f"[cache] '{artifact_fp.name}' loaded (inputs match).")
    with open(artifact_fp, "rb") as f:
        return pickle.load(f)


def load_manifest(artifact_fp: Path) -> dict:
    """Return the parsed ``<artifact>.manifest.json`` (or raise if absent)."""
    mp = _manifest_path(Path(artifact_fp))
    if not mp.is_file():
        raise StaleCacheError(f"No manifest found for '{Path(artifact_fp).name}'.")
    with open(mp) as f:
        return json.load(f)


def manifest_matches(artifact_fp: Path, fp_dict: dict) -> bool:
    """Return True iff ``artifact_fp`` has a manifest matching ``fp_dict``.

    Non-raising sibling of :func:`verify`: the artifact is considered a valid
    cache when its ``<artifact>.manifest.json`` sidecar exists and every key in
    ``fp_dict`` has the same hash there. A missing sidecar (or any mismatch)
    returns False rather than raising. Used by the per-stripe incremental cache
    to decide whether a ``(site, iml)`` stripe still needs recomputing.
    """
    try:
        cached = load_manifest(artifact_fp).get("inputs", {})
    except StaleCacheError:
        return False
    # Subset semantics (like verify): only the keys in fp_dict must match.
    return all(
        cached.get(name, {}).get("hash") == cur.get("hash")
        for name, cur in fp_dict.items()
    )


def verify(artifact_fp: Path, fp_dict: dict) -> None:
    """Assert a cached artifact's manifest still matches the given inputs.

    Only the keys present in ``fp_dict`` are compared, so a *subset* fingerprint
    (e.g. just the source-file hashes) can cheaply confirm an artifact is fresh
    without rebuilding the full input set. Raises :class:`StaleCacheError` on any
    mismatch. Used by the post-processing stage to refuse plotting a stale
    artifact.
    """
    cached = load_manifest(artifact_fp).get("inputs", {})
    changes = []
    for name, cur in fp_dict.items():
        c = cached.get(name, {}).get("hash")
        if c != cur.get("hash"):
            c_sum = cached.get(name, {}).get("summary", "<absent>")
            changes.append(f"  - {name}: {c_sum} [{c}] -> {cur['summary']} [{cur['hash']}]")
    if changes:
        raise StaleCacheError(
            f"Artifact '{Path(artifact_fp).name}' is stale relative to current inputs:\n"
            + "\n".join(changes)
            + "\n\nRe-run the Stage-1 (compute) notebook to rebuild it."
        )
