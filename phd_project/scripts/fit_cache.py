"""On-disk cache for the model fits of ``fragility_data_models``.

Why this module exists
----------------------
A reportable fit refits the model inside every one of ~5000 bootstrap
replicates, which takes minutes per model and hours for the whole ladder. The
full-sample fit is cheap; the replicate loop is not. So when a notebook is
reopened we want to LOAD a fit whose inputs have not changed and FIT (then save)
only the ones whose inputs have. :class:`ModelRegistry` and
:class:`JointModelRegistry` do that when given a ``cache_dir``; this module holds
the logic they share, so neither registry carries its own copy.

The analogy is a lab notebook with a label on every jar: the jar (the ``.pkl``)
holds the result, the label (the ``.pkl.manifest.json`` sidecar, the
convention of :mod:`cache_utils`) says exactly what went into it. A jar is
used only if its label matches what we would put in today; otherwise it is
emptied and refilled, and we are told which ingredient changed.

What goes on the label -- the fingerprint
-----------------------------------------
A fit is a deterministic function of exactly five things, and the fingerprint
holds one entry for each (all must match for a file to be used):

``contrast``
    The contrast the model is fitted to: ``y_hat``, the replicate matrix ``y``,
    ``V``, the factor table ``rows``, and every label. This is everything a fit
    ever reads from the data, so it is hashed INSTEAD of the whole
    ``FragilityData`` container (user decision Q1, 2026-10-19): re-running nb 072
    to change an arm a contrast does not use must not refit it. A whole-container
    hash is still written to the manifest's ``_meta``, for diagnostics only.
``spec``
    Every ``ModelSpec`` / ``JointModelSpec`` field except ``note`` (edited
    often, changes no number) and, for a univariate spec, ``estimators`` (the
    estimator is in the FILE NAME, so adding DL to a REML-only spec must not
    invalidate the REML file).
``kw``
    The keyword arguments passed through to the full-sample REML fit
    (``n_starts``, ``method`` ...), minus ``verbose``. Absent for DL, which
    ignores them.
``boot_seed``
    Only for a hybrid (``_hb``) spec: a record-only bootstrap has nothing random
    in it, so its seed is not an input.
``code``
    A hash of the source of every function on the path that produces the saved
    numbers -- see "The code hash" below.

``pickagm_version`` is appended by :func:`cache_utils.fingerprint`; harmless.

``n_boot`` is deliberately NOT in the fingerprint: see "5000 serves 500".

5000 serves 500
---------------
Replicate r of a bootstrap does not depend on how many replicates were asked
for. The record scheme refits column r of the cloud on its own; the hybrid
scheme fills its pseudo-data column by column from one seeded random stream, so
a shorter run is the same stream stopped earlier (like dealing the first 15
cards of one shuffled deck). Both are documented in ``bootstrap_fits`` and
verified bit-for-bit in ``tests/test_fit_cache.py``. Therefore a file saved with
N replicates answers any request for ``n <= N`` by slicing its first n rows; a
request for MORE refits. ``n_boot=None`` (no bootstrap) is answered by any file,
with the replicates dropped. The comparison uses the EFFECTIVE count
``min(n_boot, k)``, because ``bootstrap_fits`` never runs more than the cloud has.

The code hash
-------------
The user chose an automatic hash over a hand-bumped version number. Starting
from a short, named list of entry points (``UNIVARIATE_FIT_CODE_ROOTS`` and
``JOINT_FIT_CODE_ROOTS`` in ``fragility_data_models``), :func:`code_hash` parses
each function, follows every name it references that resolves to a function in
one of the three project modules (:data:`PROJECT_MODULES`), and recurses. Each
function is hashed as its syntax tree with docstrings removed, so

- editing a comment or a docstring does NOT invalidate the cache (the AST holds
  neither);
- editing logic does, however deep in the call tree it sits;
- the VALUES of module constants a function refers to are hashed too, because
  the tree holds only the name: ``zero_tol=_REML_ZERO_TOL`` would otherwise let a
  changed tolerance slip through.

Classes are not followed (their accessors -- ``ci``, ``summary``, the forest
plot -- run after loading, from the saved numbers), and neither are numpy,
scipy or pandas (their versions are recorded in the manifest, never compared).
Methods reached as ``con.<attr>`` cannot be resolved statically, which is why
the few ``Contrast`` methods on the fit path are listed among the roots by hand.

Atomic writes
-------------
:func:`atomic_save` removes the old manifest, writes the pickle to a temporary
file in the same folder, ``os.replace``-s it into place and writes the new
manifest LAST. A kernel killed at any point leaves either no manifest (an
untrusted file, refitted next time) or a complete, matching pair. As a second
guard the pickle carries a digest of its own fingerprint, so a manifest from one
run paired with a pickle from another (two kernels writing at once) is caught.

Folder layout
-------------
``<cache_dir>/<im_tag>/<subset>[__<cache_tag>]/<quantity>/<model_id>_<estimator>.pkl``
for univariate fits and ``.../joint/<model_id>.pkl`` for joint ones, each with
its ``.pkl.manifest.json``. ``<subset>`` is derived from the container --
``n120_3s5s`` for the full study, ``n60_3s_provisional`` for the 3-storey subset
-- so provisional results can never overwrite full-study ones. ``cache_tag``
separates two containers that would otherwise share every path component, such
as nb 034's ``theta`` and its shifted copy ``theta_star``.

The cache is disposable by definition: deleting the folder is always safe.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import pickle
import platform
import re
import tempfile
import textwrap
import types
import warnings
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import phd_project.scripts.cache_utils as cu

# The modules whose functions the code hash follows. Compared on the LAST
# component of ``__module__``, so the hash is the same whether a notebook
# imported ``phd_project.scripts.metaregression_analysis`` or put the scripts
# folder on sys.path and imported ``metaregression_analysis``.
PROJECT_MODULES: frozenset[str] = frozenset(
    {"fragility_data_models", "metaregression_analysis", "data_simulation"})

# The full study, against which a container's row count is labelled. Anything
# smaller is a subset and its folder is suffixed "_provisional" (CLAUDE.md
# section 1: the 60-row 3-storey subset is provisional and must say so).
FULL_STUDY_ROWS = 120

# Keyword arguments that never change a number and are therefore dropped from
# the ``kw`` fingerprint entry.
_KW_NOT_HASHED = frozenset({"verbose"})

# Characters allowed in a cache_tag: it becomes part of a folder name.
_CACHE_TAG_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha(text: str | bytes) -> str:
    """SHA-256 hex digest of a string or bytes."""
    if isinstance(text, str):
        text = text.encode()
    return hashlib.sha256(text).hexdigest()


# =============================================================================
# The code hash
# =============================================================================

# Memo of code_hash, keyed by the tuple of root keys. The walk costs a few
# milliseconds, but a notebook fits dozens of models, and the source cannot
# change under a running kernel without a reload -- after which (or after a
# monkeypatch in a test) call clear_code_hash_cache().
_CODE_HASH_CACHE: dict[tuple[str, ...], tuple[str, dict[str, str]]] = {}


def clear_code_hash_cache() -> None:
    """Forget every memoised code hash, so the next fit re-reads the source."""
    _CODE_HASH_CACHE.clear()


def _short_module(name: str) -> str:
    """``"phd_project.scripts.metaregression_analysis"`` -> ``"metaregression_analysis"``."""
    return name.rsplit(".", 1)[-1]


def _unwrap(obj):
    """The plain function behind a property, cached_property, classmethod or method.

    The roots list ``Contrast.v`` (a cached_property) and ``Contrast.n`` (a
    property) alongside plain functions; this turns each into the function whose
    source is to be hashed.
    """
    if isinstance(obj, property):
        return obj.fget
    if isinstance(obj, cached_property):
        return obj.func
    if isinstance(obj, (classmethod, staticmethod)):
        return obj.__func__
    if isinstance(obj, types.MethodType):
        return obj.__func__
    return obj


def _is_project_function(obj) -> bool:
    """True for a plain Python function defined in one of :data:`PROJECT_MODULES`."""
    return (isinstance(obj, types.FunctionType)
            and _short_module(getattr(obj, "__module__", "") or "") in PROJECT_MODULES)


def _function_key(fn) -> str:
    """Inventory label of a function, e.g. ``"metaregression_analysis.fit_reml"``."""
    return f"{_short_module(fn.__module__)}.{fn.__qualname__}"


def _strip_docstrings(tree: ast.AST) -> None:
    """Remove every docstring from a parsed tree, in place.

    A docstring is the first statement of a module, class or function body when
    that statement is a bare string. Removing it is what makes a documentation
    edit leave the hash alone. A body left empty gets a ``pass`` so the tree is
    still well-formed. Comments need no handling: the parser drops them.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())


def _canonical_constant(value) -> str | None:
    """A stable text form of a plain module constant, or None if it is not one.

    Only plain data counts as a constant: numbers, strings, None, numpy scalars
    and arrays, and tuples / lists / sets / dicts of those. Modules, classes and
    functions return None (functions are followed instead; the rest is not part
    of the hash). Sets and dicts are sorted, because a set's iteration order
    depends on string-hash randomisation and would otherwise differ between two
    processes. Anything containing a non-plain item (e.g. a dict of ModelSpec
    objects, whose content is hashed through ``spec`` instead) returns None.
    """
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes,
                                           np.generic)):
        return repr(value)
    if isinstance(value, np.ndarray):
        return f"ndarray[{value.dtype},{value.shape}]:" + _sha(
            np.ascontiguousarray(value).tobytes())
    if isinstance(value, (tuple, list)):
        parts = [_canonical_constant(v) for v in value]
        if any(p is None for p in parts):
            return None
        return f"{type(value).__name__}(" + ",".join(parts) + ")"
    if isinstance(value, (set, frozenset)):
        parts = [_canonical_constant(v) for v in value]
        if any(p is None for p in parts):
            return None
        return "set(" + ",".join(sorted(parts)) + ")"
    if isinstance(value, dict):
        items = [(_canonical_constant(k), _canonical_constant(v))
                 for k, v in value.items()]
        if any(k is None or v is None for k, v in items):
            return None
        return "dict(" + ",".join(f"{k}:{v}" for k, v in sorted(items)) + ")"
    return None


def _references(tree: ast.AST, globals_: dict):
    """Yield ``(name, object, owner_module)`` for every global a function refers to.

    Two forms are resolved:

    - a bare name, ``fit_dl`` or ``_REML_ZERO_TOL``, looked up in the function's
      own module globals;
    - ``module.attr`` where ``module`` is a project module, as in
      ``mra.fit_reml`` or ``dsim.hybrid_pseudo_data``.

    Locals that happen to share a global's name are resolved too. That can only
    ADD an entry to the inventory, i.e. err towards refitting, never towards a
    stale load.
    """
    owner = _short_module(globals_.get("__name__", ""))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in globals_:
            yield node.id, globals_[node.id], owner
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id in globals_):
            mod = globals_[node.value.id]
            if (isinstance(mod, types.ModuleType)
                    and _short_module(mod.__name__) in PROJECT_MODULES
                    and hasattr(mod, node.attr)):
                yield node.attr, getattr(mod, node.attr), _short_module(mod.__name__)


def code_inventory(roots) -> dict[str, str]:
    """``{label: sha256}`` for every function and constant reachable from ``roots``.

    The walk described in the module docstring. Labels are
    ``"<module>.<qualname>"`` for functions and ``"<module>.<NAME>"`` for
    constants, so a stale-cache message can say exactly which one changed.
    """
    inventory: dict[str, str] = {}
    todo = [_unwrap(r) for r in roots]
    seen: set[str] = set()
    while todo:
        fn = todo.pop()
        if not isinstance(fn, types.FunctionType):
            raise TypeError(f"code-hash root {fn!r} is not a Python function")
        key = _function_key(fn)
        if key in seen:
            continue
        seen.add(key)
        # dedent: methods and nested functions are indented in their file, and
        # ast.parse refuses indented top-level code.
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        _strip_docstrings(tree)
        # include_attributes=False (the default) leaves out line and column
        # numbers, so moving a function within its file changes nothing.
        inventory[key] = _sha(ast.dump(tree))
        for name, obj, owner in _references(tree, fn.__globals__):
            obj = _unwrap(obj)
            if _is_project_function(obj):
                todo.append(obj)
                continue
            text = _canonical_constant(obj)
            if text is not None:
                inventory[f"{owner}.{name}"] = _sha(text)
    return dict(sorted(inventory.items()))


def code_hash(roots) -> tuple[str, dict[str, str]]:
    """``(digest, inventory)`` of the code reachable from ``roots``. Memoised per process."""
    key = tuple(_function_key(_unwrap(r)) for r in roots)
    if key not in _CODE_HASH_CACHE:
        inventory = code_inventory(roots)
        _CODE_HASH_CACHE[key] = (_sha(json.dumps(inventory, sort_keys=True)), inventory)
    return _CODE_HASH_CACHE[key]


# =============================================================================
# Content hashes of the data
# =============================================================================

def _labels_hash(index: pd.Index) -> str:
    """Hash of an index's labels and names, in order.

    Needed because :func:`cache_utils.dataframe_content_hash` deliberately DROPS
    the index: without this, relabelled rows with identical values would pass.
    """
    return _sha(repr((list(index.names), index.tolist())))


def frame_hash(obj: pd.DataFrame | pd.Series | None) -> str:
    """Values (via ``dataframe_content_hash``) plus index and column labels."""
    if obj is None:
        return "none"
    parts = [cu.dataframe_content_hash(obj), _labels_hash(obj.index)]
    if isinstance(obj, pd.DataFrame):
        parts.append(_labels_hash(obj.columns))
    return _sha("|".join(parts))


def array_hash(a: np.ndarray) -> str:
    """Bytes plus dtype and shape of an array."""
    a = np.ascontiguousarray(a)
    return _sha(a.tobytes() + repr((a.dtype.str, a.shape)).encode())


def contrast_entry(con) -> dict:
    """Fingerprint entry ``{"hash", "summary"}`` of a univariate ``Contrast``.

    Everything the fit reads: ``y_hat``, ``y``, ``V``, the factor table (which
    carries the covariates, the design groups and the sites) and the
    identifying attributes.
    """
    parts = [frame_hash(con.y_hat), frame_hash(con.y), array_hash(con.V),
             frame_hash(con.rows),
             repr((con.name, con.quantity, con.scope, tuple(con.terms),
                   bool(con.corrected)))]
    return {"hash": _sha("|".join(parts)),
            "summary": (f"contrast {con.name} ({con.quantity}, n={con.n}, k={con.k}, "
                        f"corrected={con.corrected})")}


def joint_contrast_entry(jcon) -> dict:
    """Fingerprint entry of a ``JointContrast``: the stacked arrays and both halves."""
    parts = [frame_hash(jcon.y_hat), frame_hash(jcon.y), array_hash(jcon.V),
             contrast_entry(jcon.theta)["hash"], contrast_entry(jcon.beta)["hash"]]
    return {"hash": _sha("|".join(parts)),
            "summary": (f"joint contrast {jcon.theta.name} (n={jcon.theta.n}, "
                        f"k={jcon.theta.k}, beta corrected={jcon.beta.corrected})")}


def container_hash(data) -> str:
    """Hash of a whole ``FragilityData``. Recorded in ``_meta`` only, never compared."""
    parts = [data.quantity, data.im_tag, frame_hash(data.rows), _labels_hash(data.groups)]
    for name in ("reps", "estimates", "estimates_bc", "bias"):
        d = getattr(data, name) or {}
        parts += [f"{name}.{sym}={frame_hash(d[sym])}" for sym in sorted(d)]
    return _sha("|".join(parts))


# =============================================================================
# The fingerprint
# =============================================================================

def spec_items(spec, exclude: frozenset[str]) -> tuple:
    """``(class name, ((field, value), ...))`` for every spec field not excluded."""
    import dataclasses

    return (type(spec).__name__,
            tuple((f.name, getattr(spec, f.name)) for f in dataclasses.fields(spec)
                  if f.name not in exclude))


def build_fingerprint(*, contrast: dict, spec_tuple: tuple, code_digest: str,
                      n_code: int, kw: dict | None, boot_seed: int | None) -> dict:
    """Assemble the fingerprint dict in :func:`cache_utils.fingerprint` format.

    ``kw=None`` leaves the ``kw`` entry out (DL); ``boot_seed=None`` leaves the
    seed out (record-only scheme). Built through ``cache_utils.fingerprint`` so
    the manifests have the same shape as every other cache in the project; the
    tuples hash via its deterministic-pickle fallback, and their summaries are
    then replaced with readable ones for the stale message.
    """
    inputs = {"spec": spec_tuple}
    kw_clean = None
    if kw is not None:
        kw_clean = {k: v for k, v in sorted(kw.items()) if k not in _KW_NOT_HASHED}
        inputs["kw"] = tuple(kw_clean.items())
    if boot_seed is not None:
        inputs["boot_seed"] = int(boot_seed)
    fp = cu.fingerprint(**inputs)
    fp["spec"]["summary"] = _short_repr(dict(spec_tuple[1]), f"{spec_tuple[0]} ")
    if kw_clean is not None:
        fp["kw"]["summary"] = _short_repr(kw_clean, "kw ")
    fp["contrast"] = contrast
    fp["code"] = {"hash": code_digest, "summary": f"{n_code} hashed functions/constants"}
    return fp


def _short_repr(obj, prefix: str = "", limit: int = 300) -> str:
    text = prefix + repr(obj)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _pair_digest(fp: dict, n_saved: int | None) -> str:
    """Digest tying a pickle to the manifest written with it."""
    return _sha(json.dumps(fp, sort_keys=True) + f"|n_boot_saved={n_saved!r}")


def library_versions() -> dict[str, str]:
    """numpy / scipy / pandas / Python versions. Recorded, never compared."""
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


# =============================================================================
# Paths
# =============================================================================

def check_cache_tag(cache_tag: str | None) -> str | None:
    """Validate a cache_tag; ``None`` and ``""`` both mean "no tag" and return None."""
    if cache_tag is None or cache_tag == "":
        return None
    if not isinstance(cache_tag, str) or not _CACHE_TAG_RE.match(cache_tag):
        raise ValueError(f"cache_tag must be letters, digits, '_' or '-', got {cache_tag!r}: "
                         "it becomes part of a folder name")
    return cache_tag


def subset_label(data) -> str:
    """``"n120_3s5s"`` for the full study; a smaller study is marked ``_provisional``.

    Derived from the container itself (row count and the storey level of the
    row index), not passed in, so the label cannot disagree with the data.
    """
    n = len(data.rows)
    storeys = sorted(set(data.rows.index.get_level_values("n_storeys")))
    label = f"n{n}_" + "".join(f"{int(s)}s" for s in storeys)
    if n < FULL_STUDY_ROWS:
        label += "_provisional"
    return label


def cache_folder(cache_dir: Path, data, cache_tag: str | None) -> Path:
    """``<cache_dir>/<im_tag>/<subset>[__<cache_tag>]``: everything but the quantity."""
    sub = subset_label(data) + (f"__{cache_tag}" if cache_tag else "")
    return Path(cache_dir) / data.im_tag / sub


# =============================================================================
# Loading and saving
# =============================================================================

def replicate_count_ok(want: int | None, have: int | None) -> str | None:
    """None if a file saved with ``have`` replicates answers a request for ``want``.

    Otherwise the reason it does not. See "5000 serves 500".
    """
    if want is None:
        return None
    if have is None:
        return f"saved without a bootstrap, {want} replicates requested"
    if want > have:
        return f"saved {have} replicates, {want} requested"
    return None


def _describe_changes(cached: dict, current: dict, cached_inv: dict | None,
                      current_inv: dict) -> str:
    """Name every fingerprint entry that changed; for ``code``, name the functions."""
    changed = cu._changed_inputs(cached, current)
    words = []
    for name in changed:
        if name == "code" and cached_inv is not None:
            diff = sorted(k for k in set(cached_inv) | set(current_inv)
                          if cached_inv.get(k) != current_inv.get(k))
            shown = ", ".join(diff[:6]) + (f", +{len(diff) - 6} more" if len(diff) > 6 else "")
            words.append(f"code ({shown})" if diff else "code")
        else:
            words.append(name)
    return "changed: " + ", ".join(words)


def try_load(path: Path, fp: dict, want: int | None,
             inventory: dict[str, str]) -> tuple[dict | None, str]:
    """``(payload, "")`` if ``path`` holds a valid fit for ``fp``, else ``(None, reason)``.

    A file is refused, with the reason returned, when: it does not exist; its
    manifest is missing or unreadable (a crash mid-write, or a file of unknown
    provenance); any fingerprint entry differs; it holds too few replicates;
    the pickle itself cannot be read; or the pickle's own digest does not match
    the manifest (the two were written by different runs). Reasons are phrased
    for the one-line ``[stale]`` message the registry prints.
    """
    path = Path(path)
    if not path.is_file():
        return None, "no cached file"
    mp = cu._manifest_path(path)
    if not mp.is_file():
        return None, "cached file has no manifest (untrusted)"
    try:
        with open(mp) as f:
            manifest = json.load(f)
        cached = manifest["inputs"]
        meta = manifest.get("_meta", {})
    except (OSError, ValueError, KeyError, TypeError):
        return None, "manifest unreadable"
    if cu._changed_inputs(cached, fp):
        return None, _describe_changes(cached, fp, meta.get("code_inventory"), inventory)
    have = meta.get("n_boot_saved")
    why = replicate_count_ok(want, have)
    if why:
        return None, why
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:  # truncated or corrupt: refit rather than crash
        return None, "cached file unreadable"
    if (not isinstance(payload, dict)
            or payload.get("pair_digest") != _pair_digest(fp, have)
            or payload.get("n_boot_saved") != have):
        return None, "cached file and manifest come from different runs"
    return payload, ""


def atomic_save(path: Path, payload: dict, fp: dict, meta: dict,
                n_saved: int | None) -> None:
    """Write ``payload`` and its manifest so a crash never leaves a trusted half-file.

    Order matters (see "Atomic writes" in the module docstring): old manifest
    out first, pickle in via a temp file and ``os.replace``, new manifest last.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mp = cu._manifest_path(path)
    # 1. Withdraw trust from whatever is there now.
    mp.unlink(missing_ok=True)
    payload = {**payload, "n_boot_saved": n_saved, "pair_digest": _pair_digest(fp, n_saved)}
    # 2. The pickle, written in full to a temp file in the SAME folder (so the
    #    rename cannot cross a filesystem) and only then moved into place.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    # 3. The manifest last: its presence is what makes the file trusted.
    cu.write_manifest(path, fp, extra_meta={**meta, "n_boot_saved": n_saved})


def slice_replicates(payload: dict, want: int | None):
    """``(replicates, boot)`` for a loaded fit asked for ``want`` replicates.

    ``want=None`` gives ``(None, {})``, exactly what a fresh no-bootstrap fit
    holds. Otherwise the first ``want`` rows, with the frame's ``attrs``
    (``bootstrap_scheme``, ``seed``) restored from the payload.
    """
    if want is None or payload.get("replicates") is None:
        return None, {}
    reps = payload["replicates"].iloc[:want].copy()
    reps.attrs = dict(payload.get("replicates_attrs") or {})
    return reps, dict(payload.get("boot") or {})


def _count(n: int | None) -> str:
    return "no bootstrap" if n is None else f"{n}"


# =============================================================================
# The part each registry inherits
# =============================================================================

class FitCacheMixin:
    """Cache plumbing shared by ``ModelRegistry`` and ``JointModelRegistry``.

    The registries own their fits, so the load-or-fit decision lives on them;
    this mixin is the part that is identical between the two. A registry calls
    :meth:`_init_fit_cache` in its constructor and :meth:`_load_or_fit` once per
    ``(model_id, estimator)`` when ``self.cache_dir`` is set.
    """

    cache_dir: Path | None
    cache_tag: str | None
    cache_log: dict[tuple[str, str], str]

    def _init_fit_cache(self, cache_dir, cache_tag) -> None:
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.cache_tag = check_cache_tag(cache_tag)
        # (model_id, estimator) -> "loaded (...)" | "fitted (...)", the record
        # of what this registry did, read by cache_summary() and the tests.
        self.cache_log = {}

    def _load_or_fit(self, key: tuple[str, str], path: Path, fp: dict, meta: dict,
                     inventory: dict[str, str], want: int | None, refit: bool,
                     fit_fn: Callable[[], object],
                     to_payload: Callable[[object], dict],
                     from_payload: Callable[[dict], object]):
        """Load ``path`` if it is valid for ``fp`` and ``want``; else fit and save.

        Prints one line either way, so a refit is never silent and the reason is
        always given.
        """
        label = f"{key[0]} [{key[1]}]"
        if refit:
            reason = "refit=True"
        else:
            payload, reason = try_load(path, fp, want, inventory)
            if payload is not None:
                obj = from_payload(payload)
                status = (f"loaded ({_count(payload['n_boot_saved'])} saved, "
                          f"using {_count(want)})")
                self.cache_log[key] = status
                print(f"[cache] {label}: {status}")
                return obj

        obj = fit_fn()
        reps = getattr(obj, "replicates", None)
        n_saved = None if reps is None else len(reps)
        saved = "saved"
        try:
            atomic_save(path, to_payload(obj), fp,
                        {**meta, "code_inventory": inventory}, n_saved)
        except Exception as exc:
            # The fit itself is good and goes into memory; only the cache copy
            # failed (a locked file on Windows, a full disk). Say so loudly, but
            # do not throw away minutes of bootstrap over it.
            saved = "NOT saved"
            warnings.warn(f"{label}: fitted, but saving it to the cache failed "
                          f"({exc!r}); it will be refitted next time.",
                          RuntimeWarning, stacklevel=3)
        status = f"fitted and {saved} ({reason})"
        self.cache_log[key] = status
        tag = "[cache]" if reason == "no cached file" else "[stale]"
        print(f"{tag} {label}: {status}")
        return obj

    def cache_summary(self, verbose: bool = True) -> str:
        """"N loaded, M fitted", then one line per refit with its reason. Printed and returned."""
        loaded = [k for k, s in self.cache_log.items() if s.startswith("loaded")]
        fitted = [(k, s) for k, s in self.cache_log.items() if not s.startswith("loaded")]
        lines = [f"cache: {len(loaded)} loaded, {len(fitted)} fitted"]
        lines += [f"  {m} [{e}]: {s}" for (m, e), s in fitted]
        text = "\n".join(lines)
        if verbose:
            print(text)
        return text
