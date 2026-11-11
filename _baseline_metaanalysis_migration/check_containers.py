"""Check that the frozen containers and specs are complete and sufficient.

The three gates of ``task_20261110_freeze_baseline_containers.md``:

1. **Round trip (phd).** A phd ``FragilityData`` rebuilt from
   ``containers/<quantity>.*`` alone equals the one ``load_fragility_data``
   gives, field by field, exactly (values, dtypes, index and column types,
   names, dict key order, ``None`` entries).
2. **Refit gate (phd).** Every model-level case, refitted through the phd
   model layer with the REBUILT containers and the specs read from
   ``specs/<case_id>.json``, gives results bit-identical to
   ``expected/<case_id>.json`` (full-sample outputs and bootstrap summaries)
   and replicate frames bit-identical to ``expected/<case_id>_replicates.npz``.
   Same data and same code, so any difference would mean the export is
   incomplete.
3. **Rebuild gate (covmar, read-only).** In a separate process that never
   imports phd code: a covmar ``ArmSchema`` from ``arm_schema.json`` and a
   covmar ``FragilityData`` from each container; covmar's ``y0_avg`` contrast
   must have the same y, V, factor codes and row order as
   ``inputs/uni_re2x_int_dense.npz``, bit for bit. Nothing is fitted with covmar
   and nothing is written to covmar (bytecode writing is switched off).

It writes nothing. Run with the phd virtual environment::

    python check_containers.py            all three gates (gate 3 in a subprocess)
    python check_containers.py phd        gates 1 and 2 only
    python check_containers.py covmar     gate 3 only
"""

from __future__ import annotations

import os
import sys

# No .pyc files in the phd tree, and none in the covmar repository either.
sys.dont_write_bytecode = True

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pdt

HERE: Path = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import container_io as cio  # noqa: E402  (numpy / pandas only, no phd code)

CONTAINERS_DIR: Path = HERE / "containers"
SPECS_DIR: Path = HERE / "specs"
INPUTS_DIR: Path = HERE / "inputs"
EXPECTED_DIR: Path = HERE / "expected"
QUANTITIES: tuple[str, ...] = ("theta", "beta")

# The case whose frozen inputs gate 3 compares covmar's contrast against, and
# the container / correction it was built from (cases.json: theta, uncorrected).
GATE3_CASE: str = "uni_re2x_int_dense"


class GateFailed(AssertionError):
    """A gate found a difference."""


# =============================================================================
# Exact comparison of container parts
# =============================================================================

def assert_same_index(a: pd.Index, b: pd.Index, what: str) -> None:
    """Index equality including type, dtype per level, names and exact values."""
    pdt.assert_index_equal(a, b, exact=True, check_names=True, check_exact=True,
                           check_order=True, obj=what)
    if isinstance(a, pd.MultiIndex):
        if list(a.dtypes) != list(b.dtypes):
            raise GateFailed(f"{what}: level dtypes {list(a.dtypes)} != {list(b.dtypes)}")


def assert_same_parts(loaded: Any, rebuilt: Any, quantity: str) -> list[str]:
    """Compare every stored field of two containers; return what was checked."""
    checked = []
    for f in ("quantity", "im_tag"):
        if getattr(loaded, f) != getattr(rebuilt, f):
            raise GateFailed(f"{quantity}.{f}: {getattr(loaded, f)!r} != {getattr(rebuilt, f)!r}")
    pdt.assert_frame_equal(loaded.rows, rebuilt.rows, check_exact=True, check_dtype=True,
                           check_index_type=True, check_column_type=True,
                           check_names=True, check_flags=True, obj=f"{quantity}.rows")
    assert_same_index(loaded.rows.index, rebuilt.rows.index, f"{quantity}.rows.index")
    checked.append(f"rows {loaded.rows.shape}")
    assert_same_index(loaded.groups, rebuilt.groups, f"{quantity}.groups")
    checked.append(f"groups ({len(loaded.groups)})")
    for fld in cio.DICT_FIELDS:
        a, b = getattr(loaded, fld), getattr(rebuilt, fld)
        if list(a) != list(b):
            raise GateFailed(f"{quantity}.{fld}: keys {list(a)} != {list(b)}")
        for sym in a:
            what = f"{quantity}.{fld}[{sym!r}]"
            if a[sym] is None or b[sym] is None:
                if not (a[sym] is None and b[sym] is None):
                    raise GateFailed(f"{what}: None on one side only")
                continue
            if isinstance(a[sym], pd.DataFrame):
                pdt.assert_frame_equal(a[sym], b[sym], check_exact=True, check_dtype=True,
                                       check_index_type=True, check_column_type=True,
                                       check_names=True, obj=what)
                assert_same_index(a[sym].columns, b[sym].columns, what + ".columns")
            else:
                pdt.assert_series_equal(a[sym], b[sym], check_exact=True, check_dtype=True,
                                        check_index_type=True, check_names=True, obj=what)
            assert_same_index(a[sym].index, b[sym].index, what + ".index")
        checked.append(f"{fld} {list(a)}")
    return checked


# =============================================================================
# Gates 1 and 2 (phd code)
# =============================================================================

def gate1_round_trip(theta: Any, beta: Any) -> dict[str, Any]:
    """Rebuild phd containers from the files alone and compare to the loaded ones."""
    import dataclasses
    import phd_project.scripts.fragility_data_models as fm

    stored = {"quantity", "im_tag", "rows", "groups", *cio.DICT_FIELDS}
    fields = {f.name for f in dataclasses.fields(fm.FragilityData)}
    if fields != stored:
        raise GateFailed(f"phd FragilityData fields {sorted(fields)} are not the stored "
                         f"set {sorted(stored)}")
    rebuilt, report = {}, {}
    for q, loaded in (("theta", theta), ("beta", beta)):
        rebuilt[q] = fm.FragilityData(**cio.read_container(CONTAINERS_DIR, q))
        report[q] = assert_same_parts(loaded, rebuilt[q], q)
    return {"rebuilt": rebuilt, "report": report}


def tuplify(d: dict[str, Any]) -> dict[str, Any]:
    """JSON lists back into the tuples the phd spec dataclasses hold."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def normalised(obj: Any) -> Any:
    """What ``obj`` looks like after being written to and read from a fixture."""
    import phd_adapter as A
    return json.loads(json.dumps(A.to_jsonable(obj), allow_nan=False))


def gate2_refit(theta_r: Any, beta_r: Any) -> dict[str, Any]:
    """Refit every model-level case from the rebuilt containers and frozen specs."""
    import warnings
    import phd_adapter as A
    import phd_project.scripts.fragility_data_models as fm

    cases = {c["case_id"]: c for c in
             json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]}
    model_cases = [cid for cid, c in cases.items() if "model" in c["levels_run"]]
    spec_files = sorted(p.stem for p in SPECS_DIR.glob("*.json"))
    if sorted(model_cases) != spec_files:
        raise GateFailed(f"specs/ holds {spec_files}, model-level cases are {sorted(model_cases)}")

    report: dict[str, Any] = {}
    for cid in model_cases:
        case = cases[cid]
        sp = json.loads((SPECS_DIR / f"{cid}.json").read_text(encoding="utf-8"))
        cls = getattr(fm, sp["spec_class"])
        specs = {mid: cls(**tuplify(d)) for mid, d in sp["specs"].items()}

        # The frozen specs must be the adapter's own, object for object.
        adapter_specs = A.spec_family(case)[0]
        if specs != adapter_specs:
            raise GateFailed(f"{cid}: specs/ differs from phd_adapter.spec_family")
        if A.is_multivariate(case):
            fm._check_joint_model_specs(specs, {})
            reg = fm.JointModelRegistry(theta_r, beta_r, specs=specs)
        else:
            fm._check_model_specs(specs)
            reg = fm.ModelRegistry({"theta": theta_r, "beta": beta_r}[sp["container"][0]],
                                   specs=specs)

        kw = sp["fit_kwargs"]
        fits = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mid in sp["fitted_ids"]:
                fit = reg.fit(mid, n_boot=kw["n_boot"], boot_seed=kw["boot_seed"],
                              n_starts=kw["n_starts"], seed=kw["seed"], method=kw["method"])
                fits[mid] = fit if A.is_multivariate(case) else fit[0]

        expected = json.loads((EXPECTED_DIR / f"{cid}.json").read_text(encoding="utf-8"))
        inp = np.load(INPUTS_DIR / f"{cid}.npz")
        fixed_names = [str(x) for x in inp["fixed_names"]]
        fixed_outcome = [int(x) for x in inp["fixed_outcome"]]

        diffs: list[str] = []
        for where, mid in sp["results_from"].items():
            if where.startswith("bootstrap."):
                scheme = where.split(".")[1]
                got = A.bootstrap_summary(case, fits[mid].replicates, fixed_names, fixed_outcome)
                if normalised(got) != expected["bootstrap"][scheme]:
                    diffs.append(where)
            else:
                got = A.canonical_outputs(case, fits[mid].raw, fixed_names, fixed_outcome)
                if normalised(got) != expected[where]:
                    diffs.append(where)

        # Replicate frames, column by column, against the diagnosis npz.
        n_rep_cols = 0
        if case.get("bootstrap"):
            saved = np.load(EXPECTED_DIR / f"{cid}_replicates.npz")
            for where, mid in sp["results_from"].items():
                if not where.startswith("bootstrap."):
                    continue
                scheme, frame = where.split(".")[1], fits[mid].replicates
                keys = [k for k in saved.files if k.startswith(scheme + "/")]
                if sorted(keys) != sorted([f"{scheme}/__k__"] +
                                          [f"{scheme}/{c}" for c in frame.columns]):
                    diffs.append(f"{scheme} replicate columns")
                    continue
                if not np.array_equal(saved[f"{scheme}/__k__"], frame.index.to_numpy()):
                    diffs.append(f"{scheme} replicate labels")
                for c in frame.columns:
                    if not np.array_equal(saved[f"{scheme}/{c}"], frame[c].to_numpy(),
                                          equal_nan=True):
                        diffs.append(f"{scheme} replicates {c}")
                    n_rep_cols += 1

        if diffs:
            raise GateFailed(f"{cid}: refit from the frozen containers differs in {diffs}")
        report[cid] = {"compared": list(sp["results_from"]),
                       "replicate_columns": n_rep_cols, "bit_identical": True}
        print(f"  gate 2  {cid:36s} bit-identical"
              + (f" (+ {n_rep_cols} replicate columns)" if n_rep_cols else ""), flush=True)
    return report


def run_phd() -> int:
    """Gates 1 and 2 with the phd code."""
    import phd_adapter as A
    from phd_project.config import config
    import phd_project.scripts.fragility_data_models as fm

    roots = config.bootstrap_roots(config.load_config())
    theta = fm.load_fragility_data(roots, A.IM_TAG, quantity="theta", verbose=False)
    beta = fm.load_fragility_data(roots, A.IM_TAG, quantity="beta", verbose=False)

    g1 = gate1_round_trip(theta, beta)
    for q, items in g1["report"].items():
        print(f"gate 1  {q}: identical - " + "; ".join(items))
    print("gate 1  PASSED")

    # Gate 2 uses ONLY the rebuilt containers; the loaded ones are dropped so
    # they cannot be used by mistake.
    del theta, beta
    g2 = gate2_refit(g1["rebuilt"]["theta"], g1["rebuilt"]["beta"])
    print(f"gate 2  PASSED: {len(g2)} model-level cases bit-identical")
    return 0


# =============================================================================
# Gate 3 (covmar, no phd code)
# =============================================================================

def run_covmar() -> int:
    """Build covmar objects from the frozen files and compare its y0_avg contrast."""
    from covmar.model import models as cm

    sch = json.loads((CONTAINERS_DIR / "arm_schema.json").read_text(encoding="utf-8"))
    schema = cm.ArmSchema(symbols=sch["symbols"], scope_of_arm=sch["scope_of_arm"],
                          contrast_specs={k: tuple(v) for k, v in sch["contrast_specs"].items()})
    data = {q: cm.FragilityData(**cio.read_container(CONTAINERS_DIR, q), schema=schema)
            for q in QUANTITIES}

    # The whole point of the gate: covmar and these files, nothing from phd.
    leaked = sorted(m for m in sys.modules if m.startswith("phd_project") or m == "phd_adapter")
    if leaked:
        raise GateFailed(f"phd modules were imported in the covmar gate: {leaked}")

    saved = np.load(INPUTS_DIR / f"{GATE3_CASE}.npz")
    con = cm.make_contrast(data["theta"], "y0_avg", corrected=False)
    checks = {
        "y": np.array_equal(con.y_hat.to_numpy(dtype=float), saved["y"]),
        "V": np.array_equal(np.asarray(con.V), saved["V"]),
        "codes_factor1": np.array_equal(np.asarray(con.g), saved["codes_factor1"]),
        "codes_factor2": np.array_equal(np.asarray(con.site_codes), saved["codes_factor2"]),
        "row_order": [f"{s}|{n}" for s, n in con.y_hat.index] == [str(x) for x in saved["row_labels"]],
    }
    for k, ok in checks.items():
        print(f"gate 3  {k:14s} {'bit-identical' if ok else 'DIFFERS'}")
    print(f"gate 3  covmar FragilityData theta/beta built: n={data['theta'].n}, "
          f"J={data['theta'].J}, k={data['theta'].k}, symbols={list(data['theta'].available)}")
    if not all(checks.values()):
        raise GateFailed("covmar's y0_avg contrast differs from the frozen inputs")
    print("gate 3  PASSED")
    return 0


# =============================================================================
# Driver
# =============================================================================

def folder_size(folder: Path) -> int:
    """Total bytes of the files in a folder."""
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "all"
    if which == "covmar":
        return run_covmar()
    if which not in ("phd", "all"):
        raise SystemExit(f"unknown argument {which!r}: use phd, covmar or nothing")
    # Gate 3 first, in a fresh interpreter, so it is fast to see and certainly
    # free of phd imports; PYTHONDONTWRITEBYTECODE keeps covmar's tree untouched.
    if which == "all":
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run([sys.executable, str(Path(__file__).resolve()), "covmar"],
                             env=env, cwd=str(HERE))
        if res.returncode != 0:
            print("gate 3  FAILED", file=sys.stderr)
            return res.returncode
    run_phd()
    size = folder_size(CONTAINERS_DIR)
    print(f"containers/ total size: {size} bytes ({size / 2**20:.1f} MiB); "
          f"specs/: {folder_size(SPECS_DIR)} bytes")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except GateFailed as exc:
        print("\nGATE FAILED:", exc, file=sys.stderr)
        sys.exit(2)
