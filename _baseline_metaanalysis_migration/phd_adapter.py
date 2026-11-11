"""Freeze the covmar baseline from the UNMODIFIED phd fitting code.

What this script does
---------------------
For each of the 34 cases in ``cases.json`` it builds the inputs from the real
120-row study data with the existing phd functions, fits them with the existing
phd functions, and writes:

    inputs/<case_id>.npz                what the fit consumes (y, X, V, codes, ...)
    expected/<case_id>.json             the canonical result (covmar roadmap,
                                        "Adapter and harness")
    expected/<case_id>_replicates.npz   bootstrap cases only, for diagnosis
    calibration/report.json             row-permutation noise floors, findings
    provenance.json                     commit, tree state, library versions

Nothing here changes what the phd code does. Every number written comes from a
phd function called with the settings the case names. The only code in this file
is (a) translation between the case description and the phd call signatures and
(b) translation of the phd result into covmar's canonical names.

Two check levels
----------------
``engine``  the frozen arrays go straight into ``mra.fit_reml`` /
            ``mra.fit_reml_mv``. The variance-component patterns G are rebuilt
            from the SAVED integer codes, so this level depends on nothing but
            the arrays in ``inputs/<case_id>.npz``.
``model``   the same fit through the phd model layer: ``fm.ModelRegistry`` /
            ``fm.JointModelRegistry`` with a spec built from the case (data
            container -> contrast -> spec -> fit), including the outer
            bootstrap for the bootstrap cases.
A case run at both levels must give the same answer at both, within
``tolerances.json``; if not, the run stops, because that would mean the engine
and the model layer disagree in the original code.

Run it from anywhere with the phd virtual environment::

    C:\\Users\\clemettn\\documents\\phd\\.venv\\Scripts\\python.exe phd_adapter.py

then run ``add_tolerance_overrides.py``: this script writes the expected JSONs
without per-case ``tolerance_override`` blocks, and that one adds the
user-approved overrides and their re-checked margins.

The rules for this folder (new files only, never touch the phd code or covmar,
read-only git) are in ``CLAUDE.md`` next to this file.
"""

from __future__ import annotations

import sys

# Do not write .pyc files into the phd source tree: this script must leave the
# repository outside this folder exactly as it found it.
sys.dont_write_bytecode = True

import dataclasses
import datetime as _dt
import hashlib
import json
import platform
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

# The phd repository root is the parent of this folder. Putting it on the path
# lets `phd_project` be imported exactly as the notebooks import it.
HERE: Path = Path(__file__).resolve().parent
PHD_ROOT: Path = HERE.parent
if str(PHD_ROOT) not in sys.path:
    sys.path.insert(0, str(PHD_ROOT))

from phd_project.config import config  # noqa: E402
import phd_project.scripts.data_simulation as dsim  # noqa: E402
import phd_project.scripts.fragility_data_models as fm  # noqa: E402
import phd_project.scripts.metaregression_analysis as mra  # noqa: E402


# =============================================================================
# Constants: folders, name maps, seeds
# =============================================================================

INPUTS_DIR: Path = HERE / "inputs"
EXPECTED_DIR: Path = HERE / "expected"
CALIBRATION_DIR: Path = HERE / "calibration"

# The contrast every case uses (MSA-SS minus IDA-FX at the geometric-average
# collapse IML). Checked against each case rather than assumed.
CONTRAST: str = "y0_avg"
IM_TAG: str = "AvgSA_03"

# Canonical level name -> phd level name. The phd univariate layer calls the
# components "design" / "site" in specs and tau_d2 / tau_s2 / tau_u2 in results;
# the joint layer uses "design" / "site" / "row" throughout.
CANON_TO_PHD_LEVEL: dict[str, str] = {"factor1": "design", "factor2": "site", "row": "row"}
PHD_LEVEL_TO_CANON: dict[str, str] = {v: k for k, v in CANON_TO_PHD_LEVEL.items()}
# phd univariate component name -> canonical level name.
PHD_COMPONENT_TO_CANON: dict[str, str] = {"tau_d2": "factor1", "tau_s2": "factor2",
                                          "tau_u2": "row"}
CANON_TO_PHD_COMPONENT: dict[str, str] = {v: k for k, v in PHD_COMPONENT_TO_CANON.items()}

# Seed of the one row permutation used for calibration. The same permutation of
# the 120 rows is applied to every case, so all cases are calibrated against the
# same reordering.
CALIBRATION_SEED: int = 20261110

# How the frozen log-likelihood is defined, written into every expected JSON so
# covmar reproduces the same quantity rather than a differently-normalised one.
# All three phd objectives (reml_nll, reml_nll_chol, the multivariate
# _reml_mv_value_and_grad) return 0.5 * (log|Sigma| + log|X' Sigma^-1 X| + r'
# Sigma^-1 r), i.e. the same constant is dropped from each.
LOGLIK_DEFINITION: str = (
    "loglik = -nll, where nll is the minimised objective returned by the phd "
    "fitter: 0.5 * (log|Sigma| + log|X' Sigma^-1 X| + r' Sigma^-1 r), the REML "
    "negative log-likelihood with the constant (N - p)/2 * log(2 pi) dropped. "
    "Evaluated at the optimiser's final point, BEFORE variance components below "
    "zero_tol are snapped to zero (beta and se_beta are computed after snapping).")

# Quantile method used for every bootstrap summary (numpy's default).
QUANTILE_METHOD: str = "linear"


class StopRun(RuntimeError):
    """Raised when the folder rules say to stop and report, not to carry on."""


# =============================================================================
# Small helpers: JSON, hashing, flattening
# =============================================================================

def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas values into plain JSON types, NaN -> None.

    Python's ``json`` writes floats with the shortest repr that reads back
    bit-identical, so converting numpy floats with ``float()`` keeps full
    precision. NaN and inf are not valid JSON; NaN becomes ``null`` as the
    roadmap specifies, and an infinity (never expected) raises so it cannot
    slip into a fixture unnoticed.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if np.isnan(x):
            return None
        if np.isinf(x):
            raise ValueError("an infinite value cannot be written to a baseline fixture")
        return x
    return obj


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON. ``allow_nan=False`` guards the NaN rule."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, allow_nan=False)
        fh.write("\n")


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, so provenance pins the exact spec files used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(d: Any, prefix: tuple = ()) -> dict[tuple, Any]:
    """Nested dicts / lists -> ``{path tuple: leaf}``.

    Used by the comparison and calibration code, so both treat every output
    by its NAME path (e.g. ``("T", "factor1", "var", 0)``) and never by position
    in some array.
    """
    out: dict[tuple, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out |= flatten(v, prefix + (k,))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out |= flatten(v, prefix + (i,))
    else:
        out[prefix] = d
    return out


def path_str(path: tuple) -> str:
    """``("T", "row", "var", 0)`` -> ``"T.row.var[0]"``, for readable reports."""
    s = ""
    for p in path:
        s += f"[{p}]" if isinstance(p, int) else (f".{p}" if s else str(p))
    return s


# =============================================================================
# Provenance
# =============================================================================

def git(*args: str) -> str:
    """Run a READ-ONLY git command in the phd repo (rev-parse / status only)."""
    if args[0] not in ("rev-parse", "status"):
        raise ValueError("only read-only git commands are allowed here")
    res = subprocess.run(["git", "-C", str(PHD_ROOT), *args], capture_output=True,
                         text=True, check=True)
    return res.stdout


def build_provenance(study_name: str, roots: dict) -> dict:
    """Commit, tree state, versions, date and data source of this baseline.

    The working tree is judged dirty only by changes OUTSIDE this folder: the
    baseline's own new files are necessarily untracked until the user commits
    them, and that does not make the phd code that produced them uncommitted.
    Both the raw status and the in-folder lines are recorded anyway.
    """
    folder = HERE.name + "/"
    status_lines = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]

    def entry_path(line: str) -> str:
        # Porcelain v1: two status characters, a space, then the path (quoted
        # when it holds special characters; renames are "old -> new").
        return line[3:].strip().strip('"').split(" -> ")[-1]

    outside = [ln for ln in status_lines if not entry_path(ln).startswith(folder)]
    inside = [ln for ln in status_lines if entry_path(ln).startswith(folder)]
    return {
        "code": "phd",
        "commit": git("rev-parse", "HEAD").strip(),
        "tree_clean_outside_baseline_folder": not outside,
        "status_outside_baseline_folder": outside,
        "status_inside_baseline_folder": inside,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "study": study_name,
        "im_tag": IM_TAG,
        "roots": {k: str(v) for k, v in roots.items()},
        "cases_json_sha256": sha256_of(HERE / "cases.json"),
        "tolerances_json_sha256": sha256_of(HERE / "tolerances.json"),
        "producer": "phd_adapter.py",
    }


# =============================================================================
# Case -> phd spec
# =============================================================================

def is_multivariate(case: dict) -> bool:
    """True for the bivariate (theta, beta) cases."""
    return case["fit"]["engine"] == "multivariate"


def check_case(case: dict) -> None:
    """Refuse a case that names something this adapter cannot build as specified.

    The folder rules say: if a case cannot be built as specified, stop and
    report which case and why -- never adapt the case to make it run.
    """
    cid = case["case_id"]
    resp = case["response"]
    problems = []
    if resp["contrast"] != CONTRAST:
        problems.append(f"contrast {resp['contrast']!r}, adapter builds {CONTRAST!r}")
    if resp["im_tag"] != IM_TAG:
        problems.append(f"im_tag {resp['im_tag']!r}, adapter loads {IM_TAG!r}")
    if is_multivariate(case):
        if resp["outcomes"] != ["theta", "beta"]:
            problems.append(f"multivariate outcomes {resp['outcomes']}, expected theta, beta")
    elif len(resp["outcomes"]) != 1 or resp["outcomes"][0] not in fm.QUANTITIES:
        problems.append(f"univariate outcomes {resp['outcomes']}")
    if case["random"][-1] != "row" or any(l not in CANON_TO_PHD_LEVEL for l in case["random"]):
        problems.append(f"random levels {case['random']} (row must be last)")
    for lev, col in case.get("factors", {}).items():
        expected_col = {"factor1": "design_group_id", "factor2": "site"}[lev]
        if col != expected_col:
            problems.append(f"{lev} -> {col!r}, the phd code fixes it to {expected_col!r}")
    if case["V"] not in ("dense", "diag", "blockdiag"):
        problems.append(f"V transform {case['V']!r}")
    if case["V"] == "diag" and is_multivariate(case):
        problems.append("diag V on a multivariate case")
    if case["V"] == "blockdiag" and not is_multivariate(case):
        problems.append("blockdiag V on a univariate case")
    if problems:
        raise StopRun(f"case {cid!r} cannot be built as specified:\n  " + "\n  ".join(problems))


def base_id(case: dict) -> str:
    """The case id with a trailing ``_bc`` removed: the root of its spec family.

    The phd naming rules (fm._check_model_specs) read the suffixes off the id:
    an id ends in ``_bc`` exactly when ``corrected`` is True and in ``_hb``
    exactly when the scheme is hybrid, in the order base[_bc][_hb].
    """
    return case["case_id"].removesuffix("_bc")


def phd_components(case: dict) -> tuple[str, ...]:
    """Spec components beyond the row level, in the case's order (design, site)."""
    return tuple(CANON_TO_PHD_LEVEL[l] for l in case["random"] if l != "row")


def uni_spec(case: dict, model_id: str, *, corrected: bool, scheme: str) -> fm.ModelSpec:
    """One univariate phd ModelSpec for a case.

    ``estimators=("REML",)`` explicitly: the ModelSpec default also asks for
    DerSimonian-Laird, which is not part of the baseline (and refuses extra
    components anyway). ``diagonal_V`` stays False because the model level is
    only run for dense-V cases.
    """
    return fm.ModelSpec(model_id, CONTRAST, estimators=("REML",),
                        components=phd_components(case),
                        covariates=tuple(case["fixed"]["0"]),
                        diagonal_V=False, note=f"covmar baseline case {case['case_id']}",
                        corrected=corrected, bootstrap_scheme=scheme)


def joint_spec(case: dict, model_id: str, *, corrected: bool,
               scheme: str) -> fm.JointModelSpec:
    """One phd JointModelSpec for a multivariate case.

    ``correlated`` lists the levels whose theta-beta cross term is estimated,
    translated to phd level names; covariates are set per outcome.
    """
    return fm.JointModelSpec(model_id, CONTRAST, components=phd_components(case),
                             correlated=tuple(CANON_TO_PHD_LEVEL[l]
                                              for l in case["correlated"]),
                             covariates_theta=tuple(case["fixed"]["0"]),
                             covariates_beta=tuple(case["fixed"]["1"]),
                             estimators=("REML",),
                             note=f"covmar baseline case {case['case_id']}",
                             corrected=corrected, bootstrap_scheme=scheme)


def spec_family(case: dict) -> tuple[dict, str, str | None]:
    """All the specs a case needs, plus the ids to fit.

    The family is built so it passes the phd naming guard as it stands: for a
    corrected bootstrap case it holds ``base``, ``base_hb``, ``base_bc`` and
    ``base_bc_hb``, identical except in model_id, corrected, bootstrap_scheme
    and note. The guard is called on the family here, explicitly, because the
    registries themselves only run it on the built-in tables at import.

    Returns
    -------
    (specs, record_id, hybrid_id)
        ``record_id`` is the id of the record-scheme spec (also the one fitted
        for a non-bootstrap case); ``hybrid_id`` is None without a bootstrap.
    """
    corrected = bool(case["response"]["corrected"])
    has_boot = case.get("bootstrap") is not None
    make = joint_spec if is_multivariate(case) else uni_spec
    base = base_id(case)

    specs: dict = {}
    for bc in ([False, True] if corrected else [False]):
        for scheme in (["record", "hybrid"] if has_boot else ["record"]):
            mid = base + ("_bc" if bc else "") + ("_hb" if scheme == "hybrid" else "")
            specs[mid] = make(case, mid, corrected=bc, scheme=scheme)

    if is_multivariate(case):
        fm._check_joint_model_specs(specs, {})
    else:
        fm._check_model_specs(specs)

    record_id = base + ("_bc" if corrected else "")
    hybrid_id = record_id + "_hb" if has_boot else None
    return specs, record_id, hybrid_id


# =============================================================================
# Inputs
# =============================================================================

def transform_V(V: np.ndarray, how: str, n: int) -> np.ndarray:
    """Apply the case's V transform to the full sampling covariance.

    ``dense``      unchanged.
    ``diag``       ``np.diag(np.diag(V))`` -- only the sampling variances.
    ``blockdiag``  each outcome's own n x n block kept, the between-outcome
                   blocks (theta-beta sampling covariance) zeroed.
    """
    V = np.array(V, dtype=float, copy=True)
    if how == "dense":
        return V
    if how == "diag":
        return np.diag(np.diag(V))
    if how == "blockdiag":
        V[:n, n:] = 0.0
        V[n:, :n] = 0.0
        return V
    raise ValueError(f"unknown V transform {how!r}")


def factor_labels(con: fm.Contrast) -> dict[str, list[str]]:
    """The level labels behind ``con.g`` and ``con.site_codes``, in code order.

    The phd Contrast builds both code vectors from the SORTED unique labels
    (``mra.build_codes``), so code j is the j-th sorted label. Stored as strings
    so the npz needs no pickling.
    """
    groups = sorted(con.rows["design_group_id"].unique())
    sites = sorted(con.rows.index.get_level_values("site").unique())
    return {"factor1": [str(x) for x in groups], "factor2": [str(x) for x in sites]}


def build_inputs(case: dict, theta: fm.FragilityData,
                 beta: fm.FragilityData) -> dict[str, Any]:
    """Build what the fit consumes, with phd functions only.

    The contrast comes from ``fm.make_contrast`` (whose ``V`` is the full sample
    covariance of the contrast replicates), X from ``fm.build_X`` /
    ``fm.build_X_joint`` (covariates centred and scaled to z-scores over the
    rows, ddof=1, intercept unscaled), and the factor codes from the Contrast's
    own ``g`` / ``site_codes``. The spec used for X is the case's record spec;
    every member of a family shares the same covariates.
    """
    specs, record_id, _ = spec_family(case)
    spec = specs[record_id]
    corrected = bool(case["response"]["corrected"])

    if is_multivariate(case):
        # theta is never bias-corrected in the phd code; a joint `_bc` corrects
        # beta only, which is how JointModelRegistry.contrast builds the pair.
        con_t = fm.make_contrast(theta, CONTRAST, corrected=False)
        con_b = fm.make_contrast(beta, CONTRAST, corrected=corrected)
        jcon = fm.JointContrast.from_pair(con_t, con_b)
        X, x_names, scaling = fm.build_X_joint(spec, jcon)
        n = jcon.n
        y = jcon.y_hat.to_numpy(dtype=float)
        V = transform_V(jcon.V, case["V"], n)
        con_rows = jcon.theta
        fixed_names = [nm for _, nm in x_names]
        fixed_outcome = [fm.OUTCOMES.index(q) for q, _ in x_names]
        scaling_out = {str(fm.OUTCOMES.index(q)): sc for q, sc in scaling.items()}
    else:
        data = {"theta": theta, "beta": beta}[case["response"]["outcomes"][0]]
        con = fm.make_contrast(data, CONTRAST, corrected=corrected)
        X, x_names, scaling = fm.build_X(spec, con)
        n = con.n
        y = con.y_hat.to_numpy(dtype=float)
        V = transform_V(con.V, case["V"], n)
        con_rows = con
        fixed_names = list(x_names)
        fixed_outcome = [0] * len(x_names)
        scaling_out = scaling

    labels = factor_labels(con_rows)
    row_labels = [f"{site}|{ns}" for site, ns in con_rows.y_hat.index]
    return {
        "y": y, "X": np.asarray(X, dtype=float), "V": V, "n": n,
        "codes_factor1": np.asarray(con_rows.g, dtype=int),
        "codes_factor2": np.asarray(con_rows.site_codes, dtype=int),
        "levels_factor1": labels["factor1"], "levels_factor2": labels["factor2"],
        "row_labels": row_labels,
        "fixed_names": fixed_names, "fixed_outcome": fixed_outcome,
        "outcome_order": (["theta", "beta"] if is_multivariate(case)
                          else list(case["response"]["outcomes"])),
        "random_levels": list(case["random"]),
        "mean_diag_V": float(np.mean(np.diag(V))),
        "covariate_scaling": scaling_out,
    }


def save_inputs(case_id: str, inp: dict[str, Any]) -> None:
    """Write ``inputs/<case_id>.npz``. Strings are stored as unicode arrays.

    The covariate scaling is flattened into three parallel arrays (key, mean,
    sd) so the file needs no pickle to read; for a multivariate case the key is
    ``"<outcome>:<covariate>"``.
    """
    keys, means, sds = [], [], []
    sc = inp["covariate_scaling"]
    nested = bool(sc) and all(isinstance(v, dict) and "mean" not in v for v in sc.values())
    items = ([(f"{q}:{c}", v) for q, d in sc.items() for c, v in d.items()]
             if nested else list(sc.items()))
    for k, v in items:
        keys.append(k)
        means.append(v["mean"])
        sds.append(v["sd"])
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(INPUTS_DIR / f"{case_id}.npz",
             y=inp["y"], X=inp["X"], V=inp["V"],
             codes_factor1=inp["codes_factor1"], codes_factor2=inp["codes_factor2"],
             levels_factor1=np.array(inp["levels_factor1"], dtype=str),
             levels_factor2=np.array(inp["levels_factor2"], dtype=str),
             row_labels=np.array(inp["row_labels"], dtype=str),
             fixed_names=np.array(inp["fixed_names"], dtype=str),
             fixed_outcome=np.array(inp["fixed_outcome"], dtype=int),
             outcome_order=np.array(inp["outcome_order"], dtype=str),
             random_levels=np.array(inp["random_levels"], dtype=str),
             mean_diag_V=np.array(inp["mean_diag_V"]),
             scaling_key=np.array(keys, dtype=str),
             scaling_mean=np.array(means, dtype=float),
             scaling_sd=np.array(sds, dtype=float))


# =============================================================================
# Engine level: frozen arrays straight into the phd fitters
# =============================================================================

def engine_gs(case: dict, inp: dict[str, Any]) -> tuple[list[np.ndarray], list[str]]:
    """Variance-component patterns G = M M' rebuilt from the SAVED codes.

    ``mra.make_indicator`` gives the n x K indicator M of a factor; the
    n x n pattern G = M M' is formed by ``dsim.build_component_gs``, the same
    function the phd model layer uses, so both levels form G identically. The
    row level is the identity. ``n_levels`` is passed so an unused level could
    never silently drop a column. Names are the phd univariate ones
    (tau_d2 / tau_s2 / tau_u2), in the case's level order.
    """
    n = inp["n"]
    comps: dict[str, tuple] = {}
    for lev in case["random"]:
        if lev == "row":
            M = np.eye(n)
        else:
            M = mra.make_indicator(inp[f"codes_{lev}"], n_levels=len(inp[f"levels_{lev}"]))
        comps[CANON_TO_PHD_COMPONENT[lev]] = (M, 0.0)
    return dsim.build_component_gs(comps)


def fit_engine(case: dict, inp: dict[str, Any]) -> dict:
    """One engine-level fit of a case's arrays; returns the raw phd dict."""
    Gs, names = engine_gs(case, inp)
    f = case["fit"]
    if is_multivariate(case):
        level_Gs = {fm._COMPONENT_NAMES[nm]: G for nm, G in zip(names, Gs)}
        correlated = tuple(CANON_TO_PHD_LEVEL[l] for l in case["correlated"])
        return mra.fit_reml_mv(inp["y"], inp["X"], inp["V"], level_Gs, correlated,
                               n_starts=f["n_starts"], seed=f["seed"],
                               method=f["optimiser"])
    return mra.fit_reml(inp["y"], inp["X"], inp["V"], Gs, names=names,
                        n_starts=f["n_starts"], seed=f["seed"], method=f["objective"])


# =============================================================================
# Canonical results
# =============================================================================

def canonical_outputs(case: dict, raw: dict, fixed_names: list[str],
                      fixed_outcome: list[int]) -> dict:
    """phd raw fit dict -> the canonical ``outputs`` block, keyed by name.

    Univariate: ``tau2`` and ``at_zero_boundary`` per canonical level.
    Multivariate: ``beta`` / ``se_beta`` nested by outcome index ("0" theta,
    "1" beta); ``T[level] = {var: [theta, beta], cov, corr}`` where corr is the
    phd ``rho`` (NaN, written null, when either variance is zero); and
    ``at_zero_boundary[level] = [theta_flag, beta_flag]``.
    """
    beta = np.atleast_1d(raw["beta"])
    se = np.atleast_1d(raw["se_beta"])
    out: dict[str, Any] = {"loglik": -float(raw["nll"])}
    if is_multivariate(case):
        out["beta"] = {"0": {}, "1": {}}
        out["se_beta"] = {"0": {}, "1": {}}
        for b, s, nm, q in zip(beta, se, fixed_names, fixed_outcome):
            out["beta"][str(q)][nm] = float(b)
            out["se_beta"][str(q)][nm] = float(s)
        out["T"], out["at_zero_boundary"] = {}, {}
        for lev in raw["levels"]:
            T = np.asarray(raw["T"][lev], dtype=float)
            canon = PHD_LEVEL_TO_CANON[lev]
            out["T"][canon] = {"var": [float(T[0, 0]), float(T[1, 1])],
                               "cov": float(T[0, 1]),
                               "corr": float(raw["rho"][lev])}
            out["at_zero_boundary"][canon] = [bool(z) for z in raw["at_zero"][lev]]
    else:
        out["beta"] = {nm: float(b) for nm, b in zip(fixed_names, beta)}
        out["se_beta"] = {nm: float(s) for nm, s in zip(fixed_names, se)}
        out["tau2"] = {PHD_COMPONENT_TO_CANON[nm]: float(t)
                       for nm, t in zip(raw["names"], raw["tau2"])}
        out["at_zero_boundary"] = {PHD_COMPONENT_TO_CANON[nm]: bool(z)
                                   for nm, z in zip(raw["names"], raw["at_zero_boundary"])}
    # Converted to JSON types here, so every comparison sees exactly what the
    # fixture holds (in particular an undefined correlation is None, never NaN,
    # and NaN can never pass a numeric comparison silently).
    return to_jsonable(out)


def summary_stats(x: np.ndarray) -> dict[str, Any]:
    """Mean, SD (ddof=1) and 2.5 / 50 / 97.5 % quantiles of one replicate column.

    NaN replicates (a correlation is undefined when a variance is zero) are
    left out, and ``n_valid`` says how many were used, so a summary over fewer
    replicates is visible as such.
    """
    x = np.asarray(x, dtype=float)
    v = x[~np.isnan(x)]
    if v.size == 0:
        return {"mean": None, "sd": None, "q2.5": None, "q50": None, "q97.5": None,
                "n_valid": 0}
    q = np.quantile(v, [0.025, 0.5, 0.975], method=QUANTILE_METHOD)
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else None,
            "q2.5": float(q[0]), "q50": float(q[1]), "q97.5": float(q[2]),
            "n_valid": int(v.size)}


def bootstrap_summary(case: dict, reps: pd.DataFrame, fixed_names: list[str],
                      fixed_outcome: list[int]) -> dict:
    """Replicate frame of one scheme -> canonical per-parameter summaries.

    Univariate frame columns (fm.bootstrap_fits): ``beta_<name>`` and one
    column per phd component (``tau_d2`` ...). Joint frame columns
    (fm.bootstrap_joint_fits): ``beta_<outcome>_<name>``, ``T_<level>_tt/_bb/_tb``
    and ``rho_<level>``. The canonical layout mirrors ``outputs``.
    """
    if is_multivariate(case):
        out: dict[str, Any] = {"beta": {"0": {}, "1": {}}, "T": {}}
        for nm, q in zip(fixed_names, fixed_outcome):
            col = f"beta_{fm.OUTCOMES[q]}_{nm}"
            out["beta"][str(q)][nm] = summary_stats(reps[col])
        for canon in case["random"]:
            lev = CANON_TO_PHD_LEVEL[canon]
            out["T"][canon] = {"var": [summary_stats(reps[f"T_{lev}_tt"]),
                                       summary_stats(reps[f"T_{lev}_bb"])],
                               "cov": summary_stats(reps[f"T_{lev}_tb"]),
                               "corr": summary_stats(reps[f"rho_{lev}"])}
        return to_jsonable(out)
    out = {"beta": {nm: summary_stats(reps[fm.beta_column(nm)]) for nm in fixed_names},
           "tau2": {}}
    for canon in case["random"]:
        out["tau2"][canon] = summary_stats(reps[CANON_TO_PHD_COMPONENT[canon]])
    return to_jsonable(out)


# =============================================================================
# Comparison against tolerances
# =============================================================================

def rule_for(path: tuple, tol: dict) -> dict:
    """The tolerance rule an output path falls under (by its top-level key)."""
    return tol["rules"][path[0]]


def allowed_diff(expected: float, rule: dict, mean_diag_V: float) -> float:
    """``atol + rtol * |expected|``, with atol scaled by mean(diag V) where asked."""
    atol = rule["atol"] * (mean_diag_V if rule.get("atol_scale") == "mean_diag_V" else 1.0)
    return atol + rule["rtol"] * abs(expected)


def corr_comparable(path: tuple, flat_expected: dict, band_hi: float) -> bool:
    """Whether a correlation entry is compared at all.

    The roadmap: ``corr`` is compared only when both variances of that level
    are above the near-threshold band, since a correlation between two
    near-zero effects is meaningless. Applies to ``outputs`` (path
    ``T.<lev>.corr``) and to bootstrap summaries (``...T.<lev>.corr.<stat>``),
    judged on the full-sample variances.
    """
    if "corr" not in path:
        return True
    i = path.index("corr")
    lev = path[i - 1]
    v0 = flat_expected.get(("T", lev, "var", 0))
    v1 = flat_expected.get(("T", lev, "var", 1))
    return v0 is not None and v1 is not None and v0 > band_hi and v1 > band_hi


def compare(expected: dict, actual: dict, tol: dict, mean_diag_V: float,
            *, bootstrap: bool = False,
            full_sample: dict | None = None) -> list[str]:
    """All mismatches between two canonical blocks, as readable strings.

    The same semantics covmar's ``compare()`` is to have: missing or extra keys
    fail; booleans compare exactly; ``None`` (NaN) matches only ``None``;
    numbers pass when ``|a - e| <= atol + rtol |e|``. With ``bootstrap=True``
    every number uses the ``bootstrap`` rule. ``full_sample`` (the case's
    ``outputs``) decides which correlations are comparable.
    """
    fe, fa = flatten(expected), flatten(actual)
    ref = flatten(full_sample if full_sample is not None else expected)
    band_hi = tol["near_threshold_warning"]["band"][1]
    problems = []
    for p in sorted(set(fe) | set(fa), key=str):
        if p not in fe or p not in fa:
            problems.append(f"{path_str(p)}: key only in {'expected' if p in fe else 'actual'}")
            continue
        e, a = fe[p], fa[p]
        if not corr_comparable(p, ref, band_hi):
            continue
        if isinstance(e, bool) or isinstance(a, bool):
            if e != a:
                problems.append(f"{path_str(p)}: flag {e} vs {a}")
            continue
        if e is None or a is None:
            if not (e is None and a is None):
                problems.append(f"{path_str(p)}: {e} vs {a} (null mismatch)")
            continue
        if path_str(p).endswith("n_valid"):
            if e != a:
                problems.append(f"{path_str(p)}: n_valid {e} vs {a}")
            continue
        rule = tol["rules"]["bootstrap"] if bootstrap else rule_for(p, tol)
        d = abs(a - e)
        lim = allowed_diff(e, rule, mean_diag_V)
        if d > lim:
            problems.append(f"{path_str(p)}: expected {e!r}, actual {a!r}, "
                            f"|diff| {d:.3e} > allowed {lim:.3e}")
    return problems


def noise_floor(expected: dict, actual: dict, tol: dict, mean_diag_V: float,
                *, bootstrap: bool = False,
                full_sample: dict | None = None) -> dict[str, dict]:
    """Per output group: the largest differences and the tolerance margin.

    For each comparable numeric entry: ``|diff|``, ``|diff| / |expected|`` and
    the margin ``allowed / |diff|`` (infinite when the refit is bit-identical).
    Groups are the top-level keys (loglik, beta, se_beta, tau2, T) for
    ``outputs``, and ``bootstrap.<key>`` for summaries. ``flag_flips`` counts
    zero-boundary flags that changed.
    """
    fe, fa = flatten(expected), flatten(actual)
    ref = flatten(full_sample if full_sample is not None else expected)
    band_hi = tol["near_threshold_warning"]["band"][1]
    groups: dict[str, dict] = {}
    for p, e in fe.items():
        a = fa.get(p)
        if not corr_comparable(p, ref, band_hi):
            continue
        g = ("bootstrap." if bootstrap else "") + str(p[0])
        rec = groups.setdefault(g, {"max_abs_diff": 0.0, "max_rel_diff": 0.0,
                                    "min_margin": float("inf"), "worst": None,
                                    "flag_flips": 0, "n_compared": 0})
        if isinstance(e, bool):
            rec["flag_flips"] += int(e != a)
            continue
        if e is None or a is None or path_str(p).endswith("n_valid"):
            continue
        rule = tol["rules"]["bootstrap"] if bootstrap else rule_for(p, tol)
        d = abs(a - e)
        rel = d / abs(e) if e != 0 else (0.0 if d == 0 else float("inf"))
        margin = allowed_diff(e, rule, mean_diag_V) / d if d > 0 else float("inf")
        rec["n_compared"] += 1
        rec["max_abs_diff"] = max(rec["max_abs_diff"], d)
        rec["max_rel_diff"] = max(rec["max_rel_diff"], rel)
        if margin < rec["min_margin"]:
            rec["min_margin"], rec["worst"] = margin, path_str(p)
    return groups


def json_margin(rec: dict) -> dict:
    """An infinite margin (bit-identical refit) is written as null, with a flag."""
    out = dict(rec)
    out["bit_identical"] = rec["min_margin"] == float("inf")
    if out["bit_identical"]:
        out["min_margin"] = None
    if out["max_rel_diff"] == float("inf"):
        out["max_rel_diff"] = None
    return out


# =============================================================================
# Model level
# =============================================================================

def fit_model(case: dict, theta: fm.FragilityData, beta: fm.FragilityData
              ) -> dict[str, Any]:
    """Fit a case through the phd model layer, with its bootstrap if it has one.

    Registries are created WITHOUT ``cache_dir``, so the phd fit cache is
    neither read nor written: every number is fitted fresh by the current code.
    ``n_starts``, ``seed`` and the objective / optimiser go through ``**kw`` to
    the full-sample fit, as in the notebooks. The replicates are refitted the
    way the phd code always refits them (univariate: cholesky, one start;
    joint: one start, warm-started from the full-sample T).

    Returns the raw full-sample dict, the replicate frames per scheme, the
    contrast object used, the registry and the spec family, and any warnings.
    """
    specs, record_id, hybrid_id = spec_family(case)
    f = case["fit"]
    boot = case.get("bootstrap")
    n_boot = boot["n_boot"] if boot else None
    boot_seed = boot["seed"] if boot else fm.BOOT_SEED
    method = f["optimiser"] if is_multivariate(case) else f["objective"]
    kw = {"n_starts": f["n_starts"], "seed": f["seed"], "method": method}

    ids = [record_id] + ([hybrid_id] if hybrid_id else [])
    raws, reps, caught = {}, {}, []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if is_multivariate(case):
            reg = fm.JointModelRegistry(theta, beta, specs=specs)
            for mid in ids:
                fit = reg.fit(mid, n_boot=n_boot, boot_seed=boot_seed, **kw)
                raws[mid], reps[mid] = fit.raw, fit.replicates
            con = reg.contrast(CONTRAST, bool(case["response"]["corrected"]))
        else:
            data = {"theta": theta, "beta": beta}[case["response"]["outcomes"][0]]
            reg = fm.ModelRegistry(data, specs=specs)
            for mid in ids:
                fit = reg.fit(mid, n_boot=n_boot, boot_seed=boot_seed, **kw)[0]
                raws[mid], reps[mid] = fit.raw, fit.replicates
            con = reg.contrast(CONTRAST, bool(case["response"]["corrected"]))
        caught = [f"{x.category.__name__}: {x.message}" for x in w]

    # Both schemes refit the SAME full-sample model (the specs differ only in
    # the scheme), so their full-sample fits must be identical to the bit.
    if hybrid_id is not None:
        a, b = raws[record_id], raws[hybrid_id]
        if not (np.array_equal(a["beta"], b["beta"]) and a["nll"] == b["nll"]):
            raise StopRun(f"{case['case_id']}: the record and hybrid specs gave different "
                          "full-sample fits; they should be the same model")

    return {"raw": raws[record_id], "record_id": record_id, "hybrid_id": hybrid_id,
            "replicates": {"record": reps[record_id],
                           **({"hybrid": reps[hybrid_id]} if hybrid_id else {})},
            "contrast": con, "specs": specs, "warnings": caught}


def save_replicates(case_id: str, reps: dict[str, pd.DataFrame]) -> None:
    """``expected/<case_id>_replicates.npz``: every column of every scheme's frame.

    Keys are ``<scheme>/<column>``, plus ``<scheme>/__k__`` holding the
    replicate labels (which columns of the cloud were used). For diagnosis only;
    covmar's tests compare the summaries in the JSON, not these arrays.
    """
    arrays: dict[str, np.ndarray] = {}
    for scheme, frame in reps.items():
        if frame is None:
            continue
        arrays[f"{scheme}/__k__"] = np.asarray(frame.index.to_numpy())
        for col in frame.columns:
            arrays[f"{scheme}/{col}"] = np.asarray(frame[col].to_numpy())
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EXPECTED_DIR / f"{case_id}_replicates.npz", **arrays)


# =============================================================================
# Calibration: one row permutation per case
# =============================================================================

def permutation(n: int) -> np.ndarray:
    """The one fixed row permutation used for every case."""
    return np.random.default_rng(CALIBRATION_SEED).permutation(n)


def permute_inputs(case: dict, inp: dict[str, Any], p: np.ndarray) -> dict[str, Any]:
    """Reorder the rows of a case's arrays consistently.

    Univariate: y, X rows, V rows and columns, codes. Multivariate: the same
    permutation is applied inside BOTH outcome blocks (index ``[p, p + n]``),
    so theta row i and beta row i stay the same building and the outcome
    stacking order is kept. The level labels do not change -- only which row
    points at which level.
    """
    n = inp["n"]
    P = np.concatenate([p, p + n]) if is_multivariate(case) else p
    out = dict(inp)
    out["y"] = inp["y"][P]
    out["X"] = inp["X"][P]
    out["V"] = inp["V"][np.ix_(P, P)]
    out["codes_factor1"] = inp["codes_factor1"][p]
    out["codes_factor2"] = inp["codes_factor2"][p]
    out["row_labels"] = [inp["row_labels"][i] for i in p]
    return out


def permute_contrast(con: fm.Contrast, p: np.ndarray) -> fm.Contrast:
    """A Contrast with its rows reordered: y_hat, replicates, factor table, V.

    ``dataclasses.replace`` makes a NEW object, so the Contrast's cached
    properties (``g``, ``site_codes``, ``v``, ``labels``) are recomputed from the
    permuted ``rows`` and ``V`` rather than carried over in the old order. Those
    are the only derived quantities the Contrast holds.
    """
    return dataclasses.replace(con, y_hat=con.y_hat.iloc[p], y=con.y.iloc[p],
                               rows=con.rows.iloc[p], V=con.V[np.ix_(p, p)])


def calibrate_case(case: dict, inp: dict[str, Any], frozen: dict, tol: dict,
                   model: dict | None) -> dict:
    """Noise floors for one case from one row-permuted refit.

    Full-sample fit: engine-level refit of the permuted arrays, compared with
    the frozen ``outputs``.

    Bootstrap cases additionally: the permuted Contrast is built from the
    model-layer contrast (rows, replicates and V reordered together). GATE:
    its full-sample fit through the phd model functions must equal the
    permuted engine fit within tolerance, which shows the permuted Contrast is
    the same model rather than a scrambled one. Only then are its record-scheme
    replicate refits (``fm.bootstrap_fits`` / ``fm.bootstrap_joint_fits``, the
    functions the registries call) compared with the frozen record summaries.
    The hybrid summaries cannot be permutation-calibrated -- the hybrid scheme
    draws new effects by row position, so a permutation changes the pseudo-data
    itself -- and are given the record floor of the same case, marked
    ``calibrated_by: record_proxy`` (CLAUDE.md section 6).
    """
    p = permutation(inp["n"])
    inp_p = permute_inputs(case, inp, p)
    mdv = inp["mean_diag_V"]
    raw_p = fit_engine(case, inp_p)
    out_p = canonical_outputs(case, raw_p, inp["fixed_names"], inp["fixed_outcome"])
    report: dict[str, Any] = {
        "permutation_seed": CALIBRATION_SEED,
        "outputs": {g: json_margin(r) | {"calibrated_by": "permutation"}
                    for g, r in noise_floor(frozen["outputs"], out_p, tol, mdv).items()},
    }

    if model is None:
        return report

    f = case["fit"]
    boot = case["bootstrap"]
    spec = model["specs"][model["record_id"]]
    con = model["contrast"]
    if is_multivariate(case):
        con_p = fm.JointContrast.from_pair(permute_contrast(con.theta, p),
                                           permute_contrast(con.beta, p))
        raw_gate, _, _ = fm.fit_reml_joint(con_p, spec, n_starts=f["n_starts"],
                                           seed=f["seed"], method=f["optimiser"])
    else:
        con_p = permute_contrast(con, p)
        *_, raw_gate = fm.fit_reml(con_p, spec, n_starts=f["n_starts"], seed=f["seed"],
                                   method=f["objective"])
    gate = canonical_outputs(case, raw_gate, inp["fixed_names"], inp["fixed_outcome"])
    gate_problems = compare(out_p, gate, tol, mdv)
    report["permuted_contrast_gate"] = {"passed": not gate_problems,
                                        "problems": gate_problems}
    if gate_problems:
        raise StopRun(f"{case['case_id']}: the permuted Contrast does not reproduce the "
                      "permuted engine fit, so its replicate refits would not measure "
                      "numerical noise:\n  " + "\n  ".join(gate_problems))

    if is_multivariate(case):
        reps_p = fm.bootstrap_joint_fits(con_p, spec, n_boot=boot["n_boot"],
                                         T0=raw_gate["T"], full_fit=raw_gate,
                                         boot_seed=boot["seed"])
    else:
        tau2 = {nm: float(t) for nm, t in zip(raw_gate["names"], raw_gate["tau2"])}
        reps_p = fm.bootstrap_fits(con_p, spec, estimator="REML", n_boot=boot["n_boot"],
                                   full_fit={"beta": raw_gate["beta"], "tau2": tau2},
                                   boot_seed=boot["seed"])
    summ_p = bootstrap_summary(case, reps_p, inp["fixed_names"], inp["fixed_outcome"])
    rec_floor = noise_floor(frozen["bootstrap"]["record"], summ_p, tol, mdv,
                            bootstrap=True, full_sample=frozen["outputs"])
    report["bootstrap"] = {
        "record": {g: json_margin(r) | {"calibrated_by": "permutation"}
                   for g, r in rec_floor.items()},
        "hybrid": {g: json_margin(r) | {"calibrated_by": "record_proxy"}
                   for g, r in rec_floor.items()},
    }
    return report


# =============================================================================
# Driver
# =============================================================================

def base_meta(case: dict, inp: dict[str, Any], provenance: dict) -> dict:
    """The ``meta`` block common to every result JSON."""
    mv = is_multivariate(case)
    fixed: Any
    if mv:
        fixed = {"0": [], "1": []}
        for nm, q in zip(inp["fixed_names"], inp["fixed_outcome"]):
            fixed[str(q)].append(nm)
    else:
        fixed = list(inp["fixed_names"])
    return {
        "n_rows": inp["n"],
        "n_outcomes": 2 if mv else 1,
        "outcomes": inp["outcome_order"],
        "levels": list(case["random"]),
        "fixed": fixed,
        "mean_diag_V": inp["mean_diag_V"],
        "covariate_scaling": inp["covariate_scaling"],
        "covariate_scaling_rule": ("z-score over the rows of the contrast: "
                                   "(x - mean) / sd, sd with ddof=1; intercept unscaled "
                                   "(fm.build_X / fm.build_X_joint)"),
        "levels_run": list(case["levels_run"]),
        "V": case["V"],
        "fit": dict(case["fit"]),
        "corrected": bool(case["response"]["corrected"]),
        "loglik_definition": LOGLIK_DEFINITION,
        "produced_by": provenance,
    }


def bootstrap_meta(case: dict, con: Any) -> dict:
    """Exactly which replicates the bootstrap used, so covmar uses the same ones.

    Record scheme: the first ``n_boot`` columns of the replicate cloud, in its
    stored order (``list(con.y.columns)[:n_boot]`` in fm.bootstrap_fits /
    bootstrap_joint_fits). Hybrid scheme: the same columns as noise, centred on
    the mean of ALL k replicates (dsim._hybrid_noise), plus new random effects
    drawn with ``seed``.
    """
    boot = case["bootstrap"]
    cols = list(con.y.columns)[:boot["n_boot"]]
    mv = is_multivariate(case)
    return {
        "schemes": list(boot["schemes"]),
        "k_total": int(con.y.shape[1]),
        "n_boot": int(boot["n_boot"]),
        "columns_used": [to_jsonable(c) for c in cols],
        "columns_rule": "the first n_boot replicate columns of the stored cloud, in order",
        "seed": int(boot["seed"]),
        "seed_applies_to": "hybrid scheme effect draws only (record scheme has no randomness)",
        "hybrid_noise_centring": "replicate deviations from the mean over all k_total replicates",
        "replicate_refit": ("mra.fit_reml_mv, L-BFGS-B, n_starts=1, warm-started from the "
                            "full-sample T" if mv else
                            "mra.fit_reml, method='cholesky', n_starts=1"),
        "quantile_method": QUANTILE_METHOD,
        "sd_ddof": 1,
    }


def run_case(case: dict, theta: fm.FragilityData, beta: fm.FragilityData,
             tol: dict, provenance: dict) -> tuple[dict, dict, dict | None]:
    """Build, fit and write one case. Returns (result, inputs, model-fit info)."""
    cid = case["case_id"]
    check_case(case)
    inp = build_inputs(case, theta, beta)
    save_inputs(cid, inp)

    result: dict[str, Any] = {"schema_version": 1, "case_id": cid,
                              "meta": base_meta(case, inp, provenance)}
    if case.get("note"):
        result["meta"]["note"] = case["note"]

    engine_out = None
    if "engine" in case["levels_run"]:
        raw = fit_engine(case, inp)
        engine_out = canonical_outputs(case, raw, inp["fixed_names"], inp["fixed_outcome"])
        result["meta"]["engine_diagnostics"] = {
            "converged": bool(raw["converged"]), "start_spread": raw["start_spread"]}

    model = None
    if "model" in case["levels_run"]:
        model = fit_model(case, theta, beta)
        model_out = canonical_outputs(case, model["raw"], inp["fixed_names"],
                                      inp["fixed_outcome"])
        result["meta"]["model_diagnostics"] = {
            "converged": bool(model["raw"]["converged"]),
            "start_spread": model["raw"]["start_spread"],
            "spec_ids": list(model["specs"]), "warnings": model["warnings"]}
        if engine_out is not None:
            problems = compare(engine_out, model_out, tol, inp["mean_diag_V"])
            if problems:
                write_json(EXPECTED_DIR / f"{cid}.json",
                           result | {"outputs": engine_out, "outputs_model": model_out})
                raise StopRun(f"{cid}: engine and model layer disagree in the phd code:\n  "
                              + "\n  ".join(problems))
            result["outputs"] = engine_out
            result["outputs_model"] = model_out
            result["engine_vs_model"] = {"agree": True}
        else:
            result["outputs"] = model_out
        if case.get("bootstrap"):
            result["meta"]["bootstrap"] = bootstrap_meta(case, model["contrast"])
            result["bootstrap"] = {
                scheme: bootstrap_summary(case, frame, inp["fixed_names"],
                                          inp["fixed_outcome"])
                for scheme, frame in model["replicates"].items()}
            save_replicates(cid, model["replicates"])
    else:
        result["outputs"] = engine_out

    write_json(EXPECTED_DIR / f"{cid}.json", result)
    return result, inp, model


def near_threshold(results: dict[str, dict], tol: dict) -> list[dict]:
    """Every frozen tau^2 / T diagonal entry inside the near-threshold band."""
    lo, hi = tol["near_threshold_warning"]["band"]
    hits = []
    for cid, res in results.items():
        out = res["outputs"]
        entries = ([(f"tau2.{k}", v) for k, v in out["tau2"].items()] if "tau2" in out else
                   [(f"T.{lev}.var[{i}]", v) for lev, d in out["T"].items()
                    for i, v in enumerate(d["var"])])
        hits += [{"case_id": cid, "entry": e, "value": v}
                 for e, v in entries if v is not None and lo <= v <= hi]
    return hits


def any_zero_flag(outputs: dict) -> bool:
    """True if any variance component of a fit sits at the zero boundary."""
    return any(bool(v) for v in flatten(outputs["at_zero_boundary"]).values())


def main() -> int:
    cases_doc = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    tol = json.loads((HERE / "tolerances.json").read_text(encoding="utf-8"))
    cases = cases_doc["cases"]
    if len(cases) != 34:
        raise StopRun(f"cases.json holds {len(cases)} cases, the specification says 34")

    cfg = config.load_config()
    study_name = config.study(cfg)["name"]
    roots = config.bootstrap_roots(cfg)
    provenance = build_provenance(study_name, roots)
    write_json(HERE / "provenance.json", provenance)
    if not provenance["tree_clean_outside_baseline_folder"]:
        print("!" * 78)
        print("WARNING: the phd working tree is DIRTY outside the baseline folder. A "
              "baseline from uncommitted code cannot be reproduced.")
        print("\n".join(provenance["status_outside_baseline_folder"]))
        print("!" * 78)

    print(f"study {study_name}, commit {provenance['commit'][:10]}")
    theta = fm.load_fragility_data(roots, IM_TAG, quantity="theta", verbose=False)
    beta = fm.load_fragility_data(roots, IM_TAG, quantity="beta", verbose=False)
    print(f"loaded theta / beta: n={theta.n}, groups={theta.J}, sites={theta.n_sites}, "
          f"k={theta.k}")

    # -- 1. every case except the zero-boundary placeholder ----------------------
    results: dict[str, dict] = {}
    inputs: dict[str, dict] = {}
    models: dict[str, dict | None] = {}
    by_id = {c["case_id"]: c for c in cases}
    for case in cases:
        cid = case["case_id"]
        if cid == "zero_boundary":
            continue
        t0 = _dt.datetime.now()
        results[cid], inputs[cid], models[cid] = run_case(case, theta, beta, tol, provenance)
        print(f"  {cid:36s} written ({(_dt.datetime.now() - t0).total_seconds():6.1f} s)",
              flush=True)

    # -- 2. explicit objective must match cholesky -------------------------------
    explicit_check = {}
    for cid, case in by_id.items():
        if case.get("fit", {}).get("objective") == "explicit":
            partner = cid.removesuffix("_explicit")
            probs = compare(results[partner]["outputs"], results[cid]["outputs"], tol,
                            inputs[cid]["mean_diag_V"])
            explicit_check[cid] = {"against": partner, "agree": not probs,
                                   "problems": probs}

    # -- 3. zero-boundary resolution ---------------------------------------------
    zb_from = next((c["case_id"] for c in cases
                    if c.get("group") == "grid" and any_zero_flag(results[c["case_id"]]["outputs"])),
                   None)
    zb_case = by_id["zero_boundary"]
    if zb_from is None:
        zb = {"schema_version": 1, "case_id": "zero_boundary", "covered": False,
              "meta": {"produced_by": provenance, "note": zb_case.get("note", "")},
              "note": "No grid case has a variance component at the zero boundary; not "
                      "covered by the baseline, left to the synthetic tests."}
    else:
        zb = dict(results[zb_from]) | {"case_id": "zero_boundary", "covered": True,
                                      "resolved_from": zb_from}
        shutil.copyfile(INPUTS_DIR / f"{zb_from}.npz", INPUTS_DIR / "zero_boundary.npz")
    write_json(EXPECTED_DIR / "zero_boundary.json", zb)

    # -- 4. calibration ----------------------------------------------------------
    calib: dict[str, Any] = {}
    for cid in results:
        calib[cid] = calibrate_case(by_id[cid], inputs[cid], results[cid], tol,
                                    models[cid] if by_id[cid].get("bootstrap") else None)
        print(f"  calibrated {cid}", flush=True)
    if zb_from is not None:
        calib["zero_boundary"] = {"resolved_from": zb_from, **calib[zb_from]}

    min_margin = tol["calibration"]["min_margin"]
    tight = []
    for cid, rep in calib.items():
        blocks = [("outputs", rep["outputs"])]
        if "bootstrap" in rep:
            blocks += [(f"bootstrap.{s}", rep["bootstrap"][s]) for s in ("record", "hybrid")]
        for where, groups in blocks:
            for g, r in groups.items():
                if r["min_margin"] is not None and r["min_margin"] < min_margin:
                    tight.append({"case_id": cid, "block": where, "output": g,
                                  "min_margin": r["min_margin"], "worst": r["worst"],
                                  "calibrated_by": r["calibrated_by"]})
                if r["flag_flips"]:
                    tight.append({"case_id": cid, "block": where, "output": g,
                                  "flag_flips": r["flag_flips"]})

    nt = near_threshold(results, tol)
    write_json(CALIBRATION_DIR / "report.json", {
        "schema_version": 1,
        "permutation_seed": CALIBRATION_SEED,
        "min_margin_required": min_margin,
        "margin_definition": "allowed / |diff| per entry, minimum over the output group; "
                             "null = refit bit-identical",
        "cases": calib,
        "findings": {
            "tolerance_below_min_margin": tight,
            "near_threshold": nt,
            "zero_boundary": {"resolved_from": zb_from, "covered": zb_from is not None},
            "explicit_vs_cholesky": explicit_check,
            "engine_vs_model": {cid: res.get("engine_vs_model", {}).get("agree")
                                for cid, res in results.items()
                                if "outputs_model" in res},
        },
        "produced_by": provenance,
    })

    # -- 5. console report -------------------------------------------------------
    n_written = len(list(EXPECTED_DIR.glob("*.json")))
    print("\n" + "=" * 78)
    print(f"expected JSON files written: {n_written} (should be 34)")
    both = [cid for cid, r in results.items() if "outputs_model" in r]
    print(f"engine vs model: {len(both)} cases run at both levels, all agree")
    for cid, chk in explicit_check.items():
        print(f"explicit vs cholesky ({cid} vs {chk['against']}): "
              f"{'agree' if chk['agree'] else 'DISAGREE'}")
        for p in chk["problems"]:
            print(f"    {p}")
    print(f"zero boundary: {'resolved from ' + zb_from if zb_from else 'not covered'}")
    print(f"near-threshold entries: {len(nt)}")
    for h in nt:
        print(f"    {h['case_id']}: {h['entry']} = {h['value']:.3e}")
    print(f"tolerances under {min_margin}x the noise floor (or flag flips): {len(tight)}")
    for t in tight:
        print(f"    {t}")
    for cid, m in models.items():
        if m and m["warnings"]:
            print(f"warnings in {cid}:")
            for wmsg in m["warnings"]:
                print(f"    {wmsg}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StopRun as exc:
        print("\nSTOPPED:", exc, file=sys.stderr)
        sys.exit(2)
