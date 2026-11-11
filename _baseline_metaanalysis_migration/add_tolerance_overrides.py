"""Add the user-approved per-case tolerance overrides to the frozen baseline.

Why this exists
---------------
``phd_adapter.py`` freezes the baseline and calibrates it with one row-permuted
refit per case. For a few multivariate fits the permutation noise is within a
factor of 10 of the default tolerance in ``tolerances.json``: their optimum is
flat because the theta-beta correlations at the design and row levels sit at
+-1, so a harmless reordering of the rows moves beta (and the bootstrap T
summaries) by more than the default allows with a safe margin.

The roadmap's rule is that tolerances are never loosened globally. A case whose
noise floor is too close gets an explicit override in ITS OWN expected JSON,
under ``tolerance_override``, with the measured floor recorded beside it. The
overrides below were decided by the user; this script writes them and
re-checks the margin under each one.

What it does
------------
For every case in :data:`OVERRIDES`:

1. Rebuilds the case's inputs with the phd functions (``phd_adapter.build_inputs``)
   and asserts they are identical to the frozen ``inputs/<case_id>.npz``.
2. Repeats the calibration refit exactly as ``phd_adapter.calibrate_case`` does
   (same permutation seed; for the bootstrap case, the permuted Contrast and
   its record-scheme replicate refits). The refit is deterministic, so it
   reproduces the measured noise floor; this is asserted against
   ``calibration/report.json``.
3. Computes the margin (allowed / |diff|, minimum over the entries of the
   output group) under the default rtol and under the override rtol. atol is
   never changed. Every entry is re-judged, not only the previous worst one,
   because a larger rtol can change which entry is the tightest.
4. Writes ``tolerance_override`` into ``expected/<case_id>.json`` (that key
   only; everything else in the file is left as it is) and a pointer to the
   overrides into ``calibration/report.json`` under
   ``findings.tolerance_overrides``.

Hybrid-scheme summaries cannot be permutation-calibrated (the hybrid scheme
draws effects by row position); their override is judged on the record-scheme
floor of the same case, marked ``calibrated_by: record_proxy``, as in
``CLAUDE.md`` section 6.

Run it after ``phd_adapter.py``, with the phd virtual environment::

    C:\\Users\\clemettn\\documents\\phd\\.venv\\Scripts\\python.exe add_tolerance_overrides.py

Running it again gives the same files: it overwrites the override blocks
with the same content.
"""

from __future__ import annotations

import sys

# Leave no .pyc files in the phd source tree (see phd_adapter.py).
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

# The adapter lives next to this file; importing it also puts the phd
# repository root on sys.path, so the phd modules import as in the notebooks.
HERE: Path = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phd_adapter as A  # noqa: E402
from phd_project.config import config  # noqa: E402
import phd_project.scripts.fragility_data_models as fm  # noqa: E402


# =============================================================================
# The overrides, as decided by the user
# =============================================================================
#
# case_id -> list of (override key, tolerances.json rule it overrides, new rtol).
#
# Override keys name an output group as the calibration report does:
#   "beta"                 the full-sample fixed effects (rule "beta")
#   "bootstrap.<scheme>.T" the T summaries of one bootstrap scheme
#                          (rule "bootstrap", restricted to the T entries)
OVERRIDES: dict[str, list[tuple[str, str, float]]] = {
    "mv_re2x_int_dense_all":      [("beta", "beta", 1e-5)],
    "mv_re2x_cov2_dense_all":     [("beta", "beta", 1e-5)],
    "mv_re2x_int_dense_mixed":    [("beta", "beta", 1e-5)],
    "boot_mv_re2x_int_dense_all": [("beta", "beta", 1e-5),
                                   ("bootstrap.record.T", "bootstrap", 5e-4),
                                   ("bootstrap.hybrid.T", "bootstrap", 5e-4)],
}


# =============================================================================
# Reproducing the calibration refit
# =============================================================================

def permuted_bootstrap_summary(case: dict, inp: dict[str, Any],
                               theta: fm.FragilityData,
                               beta: fm.FragilityData) -> dict | None:
    """Record-scheme summaries of the row-permuted bootstrap, or None.

    Built exactly as ``phd_adapter.calibrate_case`` builds them: the contrast
    from a phd registry (no cache), its rows permuted with
    ``phd_adapter.permute_contrast``, the full-sample fit redone on the
    permuted contrast to give the warm start, and the replicates refitted with
    the phd bootstrap function the registries call.
    """
    if not case.get("bootstrap"):
        return None
    specs, record_id, _ = A.spec_family(case)
    spec, f, b = specs[record_id], case["fit"], case["bootstrap"]
    corrected = bool(case["response"]["corrected"])
    p = A.permutation(inp["n"])

    if A.is_multivariate(case):
        jcon = fm.JointModelRegistry(theta, beta, specs=specs).contrast(A.CONTRAST, corrected)
        con_p = fm.JointContrast.from_pair(A.permute_contrast(jcon.theta, p),
                                           A.permute_contrast(jcon.beta, p))
        raw, _, _ = fm.fit_reml_joint(con_p, spec, n_starts=f["n_starts"], seed=f["seed"],
                                      method=f["optimiser"])
        reps = fm.bootstrap_joint_fits(con_p, spec, n_boot=b["n_boot"], T0=raw["T"],
                                       full_fit=raw, boot_seed=b["seed"])
    else:
        data = {"theta": theta, "beta": beta}[case["response"]["outcomes"][0]]
        con_p = A.permute_contrast(
            fm.ModelRegistry(data, specs=specs).contrast(A.CONTRAST, corrected), p)
        *_, raw = fm.fit_reml(con_p, spec, n_starts=f["n_starts"], seed=f["seed"],
                              method=f["objective"])
        tau2 = {nm: float(t) for nm, t in zip(raw["names"], raw["tau2"])}
        reps = fm.bootstrap_fits(con_p, spec, estimator="REML", n_boot=b["n_boot"],
                                 full_fit={"beta": raw["beta"], "tau2": tau2},
                                 boot_seed=b["seed"])
    return A.bootstrap_summary(case, reps, inp["fixed_names"], inp["fixed_outcome"])


def floors(case: dict, expected: dict, inp: dict[str, Any], tol: dict,
           boot_permuted: dict | None) -> dict[str, dict]:
    """Noise-floor groups of the permuted refit, judged under ``tol``.

    ``outputs`` holds the full-sample groups (engine-level permuted refit, as in
    calibration); ``bootstrap`` the record-scheme summary groups, when the case
    has a bootstrap.
    """
    p = A.permutation(inp["n"])
    raw_p = A.fit_engine(case, A.permute_inputs(case, inp, p))
    out_p = A.canonical_outputs(case, raw_p, inp["fixed_names"], inp["fixed_outcome"])
    res = {"outputs": A.noise_floor(expected["outputs"], out_p, tol, inp["mean_diag_V"])}
    if boot_permuted is not None:
        res["bootstrap"] = A.noise_floor(expected["bootstrap"]["record"], boot_permuted,
                                         tol, inp["mean_diag_V"], bootstrap=True,
                                         full_sample=expected["outputs"])
    return res


def pick(groups: dict[str, dict], key: str) -> dict:
    """The noise-floor group an override key refers to."""
    return groups["outputs"]["beta"] if key == "beta" else groups["bootstrap"]["bootstrap.T"]


def reported(report: dict, case_id: str, key: str) -> dict:
    """The same group as recorded in ``calibration/report.json``."""
    rep = report["cases"][case_id]
    if key == "beta":
        return rep["outputs"]["beta"]
    return rep["bootstrap"][key.split(".")[1]]["bootstrap.T"]


# =============================================================================
# Driver
# =============================================================================

def main() -> int:
    tol = json.loads((HERE / "tolerances.json").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in
             json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]}
    report_path = A.CALIBRATION_DIR / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    min_margin = tol["calibration"]["min_margin"]

    cfg = config.load_config()
    roots = config.bootstrap_roots(cfg)
    theta = fm.load_fragility_data(roots, A.IM_TAG, quantity="theta", verbose=False)
    beta = fm.load_fragility_data(roots, A.IM_TAG, quantity="beta", verbose=False)

    pointer: list[dict] = []
    for case_id, overrides in OVERRIDES.items():
        case = cases[case_id]
        path = A.EXPECTED_DIR / f"{case_id}.json"
        expected = json.loads(path.read_text(encoding="utf-8"))

        # 1. The inputs must be the frozen ones, or the margins below would be
        #    measured on something other than the baseline.
        inp = A.build_inputs(case, theta, beta)
        saved = np.load(A.INPUTS_DIR / f"{case_id}.npz")
        for k in ("y", "X", "V"):
            if not np.array_equal(inp[k], saved[k]):
                raise A.StopRun(f"{case_id}: rebuilt {k} differs from the frozen input")

        # 2. The permuted refit, once; judged under each tolerance set below.
        boot_p = permuted_bootstrap_summary(case, inp, theta, beta)
        default = floors(case, expected, inp, tol, boot_p)

        block: dict[str, dict] = {}
        for key, rule_name, rtol in overrides:
            rule = tol["rules"][rule_name]
            tol_o = copy.deepcopy(tol)
            tol_o["rules"][rule_name]["rtol"] = rtol
            before = pick(default, key)
            after = pick(floors(case, expected, inp, tol_o, boot_p), key)
            rep = reported(report, case_id, key)

            # The repeated refit must be the one calibration measured.
            if not np.isclose(before["min_margin"], rep["min_margin"], rtol=1e-12):
                raise A.StopRun(f"{case_id} {key}: reproduced margin {before['min_margin']} "
                                f"differs from the calibration report {rep['min_margin']}")

            block[key] = {
                "rtol": rtol,
                "atol": rule["atol"],
                **({"atol_scale": rule["atol_scale"]} if "atol_scale" in rule else {}),
                "default_rtol": rule["rtol"],
                "noise_floor": {"max_abs_diff": before["max_abs_diff"],
                                "max_rel_diff": before["max_rel_diff"],
                                "worst": rep["worst"],
                                "n_compared": before["n_compared"]},
                "calibrated_by": rep["calibrated_by"],
                "margin_default": before["min_margin"],
                "margin_override": after["min_margin"],
                "margin_override_worst": after["worst"],
                "decided_by": "user",
            }
            pointer.append({"case_id": case_id, "output": key, "rtol": rtol,
                            "default_rtol": rule["rtol"],
                            "margin_default": before["min_margin"],
                            "margin_override": after["min_margin"],
                            "meets_min_margin": after["min_margin"] >= min_margin,
                            "where": f"expected/{case_id}.json: tolerance_override.{key}"})

        # 3. Add only the override key; the rest of the file is written back
        #    as it was read.
        expected["tolerance_override"] = block
        A.write_json(path, expected)

    # 4. Pointer in the calibration report. `tolerance_below_min_margin` stays
    #    as measured against the default tolerances; this entry says which of
    #    those findings are handled by a per-case override, and the margin
    #    under it.
    report["findings"]["tolerance_overrides"] = {
        "note": ("Findings in tolerance_below_min_margin that are covered by a per-case "
                 "tolerance_override in the expected JSON. The margins in "
                 "tolerance_below_min_margin are against the default tolerances; "
                 "margin_override is against the override."),
        "overrides": pointer,
    }
    A.write_json(report_path, report)

    print(f"{'case':30s} {'override':20s} {'before':>8s} {'after':>8s}")
    for e in pointer:
        status = "OK" if e["meets_min_margin"] else f"STILL < {min_margin}x"
        print(f"{e['case_id']:30s} {e['output']:20s} {e['margin_default']:8.2f} "
              f"{e['margin_override']:8.2f}  {status}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except A.StopRun as exc:
        print("\nSTOPPED:", exc, file=sys.stderr)
        sys.exit(2)
