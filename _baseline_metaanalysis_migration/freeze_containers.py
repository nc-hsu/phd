"""Freeze the full data containers, the arm tables and the model-level specs.

Why this exists
---------------
``inputs/<case_id>.npz`` holds only what the ENGINE consumes (y, X, V, factor
codes). The model-level and bootstrap cases need a complete data container,
because covmar's ``FragilityData`` rebuilds contrasts, V, bootstrap replicates
and bias corrections itself. covmar may not import the phd loader or the phd
arm tables, so their content is frozen here (task
``dump/tasks_and_plans/task_20261110_freeze_baseline_containers.md``):

    containers/<quantity>.npz / .json   theta and beta containers + manifest
                                        (format: container_io.py)
    containers/arm_schema.json          phd SYMBOLS, SCOPE_OF_ARM, _CONTRAST_SPECS
    specs/<case_id>.json                every spec phd_adapter.py registered for
                                        a model-level case, and the fit arguments
    provenance.json                     a ``containers`` block is added; nothing
                                        else in the file changes

The containers are loaded exactly as ``phd_adapter.py`` loads them, and the
specs come from ``phd_adapter.spec_family`` itself, so they are the objects
the frozen results were fitted with, not a re-description of them.

Run it with the phd virtual environment, then ``check_containers.py``::

    C:\\Users\\clemettn\\documents\\phd\\.venv\\Scripts\\python.exe freeze_containers.py
"""

from __future__ import annotations

import sys

# Leave no .pyc files in the phd source tree (see phd_adapter.py).
sys.dont_write_bytecode = True

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any

HERE: Path = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Importing the adapter also puts the phd repository root on sys.path.
import phd_adapter as A  # noqa: E402
import container_io as cio  # noqa: E402
from phd_project.config import config  # noqa: E402
import phd_project.scripts.fragility_data_models as fm  # noqa: E402

CONTAINERS_DIR: Path = HERE / "containers"
SPECS_DIR: Path = HERE / "specs"
QUANTITIES: tuple[str, ...] = ("theta", "beta")


# =============================================================================
# Arm tables
# =============================================================================

def arm_schema() -> dict[str, Any]:
    """The phd arm tables in the shape covmar's ``ArmSchema`` takes.

    Copied in table order (JSON objects keep insertion order); the contrast
    tuples become lists. covmar turns them back into tuples when it builds the
    schema.
    """
    return {
        "symbols": dict(fm.SYMBOLS),
        "scope_of_arm": dict(fm.SCOPE_OF_ARM),
        "contrast_specs": {name: list(spec) for name, spec in fm._CONTRAST_SPECS.items()},
        "source": "phd fm.SYMBOLS, fm.SCOPE_OF_ARM, fm._CONTRAST_SPECS",
    }


# =============================================================================
# Specs of the model-level cases
# =============================================================================

def results_key(case: dict) -> str:
    """Where a case's model-level full-sample result sits in its expected JSON.

    A case run at both levels keeps the engine result in ``outputs`` and the
    model result in ``outputs_model``; a model-only case keeps it in ``outputs``.
    """
    return "outputs_model" if "engine" in case["levels_run"] else "outputs"


def case_specs(case: dict) -> dict[str, Any]:
    """The exact specs and fit arguments ``phd_adapter.fit_model`` used for a case.

    The spec family comes from ``phd_adapter.spec_family`` (which also runs the
    phd naming guard on it), and the keyword arguments are assembled the same
    way ``fit_model`` assembles them: ``n_starts``, ``seed`` and ``method`` go
    to the full-sample fit, ``n_boot`` and ``boot_seed`` to the bootstrap
    (``n_boot`` is None and ``boot_seed`` the phd default for a case with no
    bootstrap, exactly as passed).
    """
    specs, record_id, hybrid_id = A.spec_family(case)
    f, boot = case["fit"], case.get("bootstrap")
    mv = A.is_multivariate(case)
    fitted = [record_id] + ([hybrid_id] if hybrid_id else [])
    from_ids: dict[str, str] = {results_key(case): record_id}
    if boot:
        from_ids["bootstrap.record"] = record_id
        from_ids["bootstrap.hybrid"] = hybrid_id
    return {
        "case_id": case["case_id"],
        "spec_class": "JointModelSpec" if mv else "ModelSpec",
        "registry": ("fm.JointModelRegistry(theta, beta, specs=specs)" if mv else
                     f"fm.ModelRegistry({case['response']['outcomes'][0]}, specs=specs)"),
        "container": ["theta", "beta"] if mv else list(case["response"]["outcomes"]),
        "specs": {mid: dataclasses.asdict(s) for mid, s in specs.items()},
        "record_id": record_id,
        "hybrid_id": hybrid_id,
        "fitted_ids": fitted,
        "results_from": from_ids,
        "fit_kwargs": {
            "n_starts": f["n_starts"], "seed": f["seed"],
            "method": f["optimiser"] if mv else f["objective"],
            "n_boot": boot["n_boot"] if boot else None,
            "boot_seed": boot["seed"] if boot else fm.BOOT_SEED,
        },
        "fit_call": "registry.fit(model_id, n_boot=n_boot, boot_seed=boot_seed, "
                    "n_starts=n_starts, seed=seed, method=method)",
        "replicate_refit": ("one start, warm-started from the full-sample T (phd default)"
                            if mv else "method='cholesky', one start (phd default)"),
    }


# =============================================================================
# Driver
# =============================================================================

def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    cfg = config.load_config()
    roots = config.bootstrap_roots(cfg)

    written: dict[str, str] = {}

    # -- 1. containers, loaded exactly as phd_adapter.main loads them ------------
    for q in QUANTITIES:
        data = fm.load_fragility_data(roots, A.IM_TAG, quantity=q, verbose=False)
        for p in cio.write_container(cio.container_parts(data), CONTAINERS_DIR):
            written[p.relative_to(HERE).as_posix()] = A.sha256_of(p)
        print(f"container {q}: n={data.n}, groups={data.J}, k={data.k}, "
              f"symbols={list(data.reps)}")

    # -- 2. arm tables -------------------------------------------------------------
    p = CONTAINERS_DIR / "arm_schema.json"
    A.write_json(p, arm_schema())
    written[p.relative_to(HERE).as_posix()] = A.sha256_of(p)

    # -- 3. specs of every model-level case -----------------------------------------
    n_specs = 0
    for case in cases:
        if "model" not in case["levels_run"]:
            continue
        p = SPECS_DIR / f"{case['case_id']}.json"
        A.write_json(p, case_specs(case))
        written[p.relative_to(HERE).as_posix()] = A.sha256_of(p)
        n_specs += 1
    print(f"specs written for {n_specs} model-level cases")

    # -- 4. provenance: add the containers block, leave everything else ----------
    prov_path = HERE / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["containers"] = {
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": A.git("rev-parse", "HEAD").strip(),
        "producer": "freeze_containers.py",
        "loaded_with": f"fm.load_fragility_data(config.bootstrap_roots(cfg), "
                       f"{A.IM_TAG!r}, quantity=q, verbose=False)",
        "sha256": written,
    }
    A.write_json(prov_path, prov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
