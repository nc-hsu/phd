"""Provenance-aware caching for the ground-motion selection pipeline.

The selection pipeline caches expensive intermediate results to pickle files.
Historically those files were reloaded on a fixed filename with *no* check that
the inputs that produced them still matched — which let a changed input
ground-motion database silently feed stale cached ensembles into a new run.

This module adds a lightweight provenance guard:

- :func:`fingerprint` reduces a set of inputs to a content-based ``{name: {hash,
  summary}}`` dict. Hashes are content-based (not memory addresses), so they are
  stable across machines and OSes. File inputs are hashed in a stream (never
  slurped) and memoised per ``(path, size, mtime_ns)`` for the life of the
  process, so fingerprinting hundreds of artifacts against the same few source
  files reads each of them once -- see :data:`_FILE_HASH_CACHE` and
  :func:`clear_file_hash_cache`.
- :func:`write_manifest` writes that dict to a ``<artifact>.manifest.json``
  sidecar next to the cached pickle.
- :func:`load_or_compute` is the single entry point used by the compute
  functions: it loads the cache only when the stored manifest matches the
  current inputs, raises :class:`StaleCacheError` when a *managed* artifact's
  inputs have changed, and otherwise computes, pickles and writes the manifest.
- :func:`json_load_or_compute` is the JSON sibling of the above, used by the
  fragility-curve notebooks: same provenance rules, but it reads/writes JSON and
  also returns whether the artifact was reused or rebuilt.

Only the ``hash`` of each named input is compared. Environment metadata (git
commit, timestamp) is recorded for diagnostics but never compared. The
``pickagm`` package version *is* compared, because it can change outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
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


def _hash_file(p: Path) -> str:
    """SHA-256 of a file's bytes, streamed.

    Identical digest to ``_hash_bytes(p.read_bytes())`` but without holding the
    whole file on the heap -- the disagg data and the gm database are GB-scale.
    Mirrors the chunked style of ``oq_runner._source_models_digest``.
    """
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# Content hashes of input files, keyed by (path, size, mtime_ns). The stripe stale
# check fingerprints every (site, iml) against the SAME handful of source files, so
# without this the 2.3 GB disagg input was read and hashed once per stripe -- 454
# stripes x 2.4 GB, ~53 min per notebook.
#
# This is an in-process memo ONLY: nothing here is persisted, and the hash stored in
# a manifest stays a pure content hash. Note the deliberate contrast with
# ``oq_runner._source_models_digest``, which rejected mtime *because it was
# persisted* and a git checkout made it lie. Here mtime only decides whether to
# re-hash within one kernel: a spurious mtime change costs one extra hash (the safe
# direction), and a real rewrite always moves size or mtime_ns (NTFS mtime
# granularity is 100 ns).
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def file_hash(p: Path, st: os.stat_result | None = None) -> str:
    """Memoised content SHA-256 of a file.

    ``st`` lets a caller that has already stat-ed the file pass it in rather than
    paying a second syscall.
    """
    p = Path(p)
    st = st or p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    h = _FILE_HASH_CACHE.get(key)
    if h is None:
        h = _FILE_HASH_CACHE[key] = _hash_file(p)
    return h


def clear_file_hash_cache() -> None:
    """Drop the in-process file-hash memo, forcing the next fingerprint to re-read."""
    _FILE_HASH_CACHE.clear()


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

    # A path to an existing file: hash the file bytes (stable, and memoised on
    # (path, size, mtime_ns) -- see _FILE_HASH_CACHE).
    if isinstance(value, (str, Path)):
        p = Path(value)
        if p.is_file():
            st = p.stat()
            return {"hash": file_hash(p, st),
                    "summary": f"file {p.name} ({st.st_size} bytes)"}
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


def _np_default(o):
    """``json.dump`` default: make numpy arrays and scalars serialisable."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _changed_inputs(cached: dict, current: dict) -> list[str]:
    """Names of inputs whose content hash differs between two fingerprints."""
    return [name for name in sorted(set(cached) | set(current))
            if cached.get(name, {}).get("hash") != current.get(name, {}).get("hash")]


def json_load_or_compute(artifact_fp: Path, fp_dict: dict, compute_fn,
                         force: bool = False, input_paths: dict | None = None):
    """JSON sibling of :func:`load_or_compute`, returning the cache status too.

    Behaves like :func:`load_or_compute` but reads/writes JSON (numpy-aware, via
    :func:`_np_default`) instead of pickle, and returns ``(result, status)`` where
    ``status`` is ``"cached"`` or ``"computed"`` so a caller looping over many
    artifacts can report how many were reused.

    - ``force`` → always run ``compute_fn``, write the JSON and its manifest.
    - artifact or manifest absent → compute (quiet).
    - artifact + manifest present and inputs **match** → load and return ``"cached"``.
    - artifact + manifest present and inputs **differ** → print a ``[stale]`` line
      naming each changed input (its path, when given in ``input_paths``), then
      recompute and overwrite.

    ``input_paths`` maps fingerprint input names to the file they came from; it is
    used only to make the ``[stale]`` message point at the file that changed.
    """
    artifact_fp = Path(artifact_fp)
    manifest_fp = _manifest_path(artifact_fp)
    input_paths = input_paths or {}

    if not force and artifact_fp.is_file() and manifest_fp.is_file():
        with open(manifest_fp) as f:
            cached = json.load(f).get("inputs", {})
        changed = _changed_inputs(cached, fp_dict)
        if not changed:
            with open(artifact_fp) as f:
                return json.load(f), "cached"
        for name in changed:
            where = input_paths.get(name)
            where = str(where) if where is not None else f"input '{name}' (no file path)"
            print(f"[stale] {artifact_fp.name}: recomputing -- changed input: {where}")

    result = compute_fn()
    artifact_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_fp, "w") as f:
        json.dump(result, f, indent=2, default=_np_default)
    write_manifest(artifact_fp, fp_dict)
    return result, "computed"


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
