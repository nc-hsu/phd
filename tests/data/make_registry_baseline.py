"""Write the frozen ``ModelRegistry`` baseline that the no-regression test compares against.

Why this exists
---------------
The ``corrected`` flag (bias-corrected point estimates, see
``dump/task_corrected_flag_model_registry.md``) touches the code path every fit runs
through: ``ModelSpec``, ``make_contrast`` and ``ModelRegistry.fit``. The promise made when
it was added is that ``corrected=False`` reproduces every earlier result EXACTLY. The
only way to show that is to keep numbers produced by the code as it was BEFORE the
change, and compare the new code against them. This script writes those numbers.

It was run once, on the unmodified module, on 2026-10-08. Re-running it after the change
would overwrite the baseline with the new code's output, and the test would then compare
the new code with itself, so it would prove nothing. Re-run it only for a deliberate,
reviewed change to the uncorrected numbers, for example a new nb 070/072 run that changes
the dataset.

What is stored
--------------
For every ``MODEL_SPECS`` entry that fits, on both quantities, for every estimator:
``beta``, ``se_beta``, ``tau2`` and the outer-bootstrap ``replicates`` frame. Those are
exactly the fields the task's acceptance criterion 1 names. ``N_BOOT`` is small (20) on
purpose. The test re-runs every fit, and the crossed 120-row REML refits are the slow
part. Twenty replicates still exercise the whole replicate path (column alignment,
refits, slopes), which is what a regression would break; they do not need to be enough
for a usable interval.

A model that fails to fit (for example a covariate missing from the dataset) is recorded
together with its error message, not skipped silently. The test then checks that it
still fails the same way, rather than accidentally starting to pass with different
numbers.

Usage::

    .venv/Scripts/python.exe tests/data/make_registry_baseline.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

# Run from anywhere: put the repo root on the path so the package imports resolve.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import phd_project.scripts.fragility_data_models as fm  # noqa: E402
from phd_project.config import config  # noqa: E402

IM_TAG = "AvgSA_03"
N_BOOT = 20
OUT = Path(__file__).with_name(f"registry_baseline_{IM_TAG}.pkl")


def main() -> None:
    root = config.load_config()["proc_data"]["bootstrapping"]
    baseline: dict = {"im_tag": IM_TAG, "n_boot": N_BOOT, "fits": {}, "errors": {}}

    for quantity in fm.QUANTITIES:
        data = fm.load_fragility_data(root, IM_TAG, quantity=quantity,
                                      verbose=False, check=False)
        reg = fm.ModelRegistry(data)
        # Iterate over the dict KEYS, the ids a notebook passes to fit(). Not over
        # spec.model_id, which was wrong for A3b/A3c when this baseline was taken.
        for model_id in fm.MODEL_SPECS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fits = reg.fit(model_id, n_boot=N_BOOT)
            except Exception as exc:  # recorded, not swallowed: see module docstring
                baseline["errors"][(quantity, model_id)] = f"{type(exc).__name__}: {exc}"
                print(f"  {quantity:5s} {model_id:8s} FAILED  {type(exc).__name__}")
                continue
            for f in fits:
                baseline["fits"][(quantity, model_id, f.estimator)] = {
                    "beta": f.beta.copy(),
                    "se_beta": f.se_beta.copy(),
                    "tau2": dict(f.tau2),
                    "replicates": f.replicates.copy(),
                }
                print(f"  {quantity:5s} {model_id:8s} {f.estimator:4s} m={f.m:+.6f}")

    with open(OUT, "wb") as fh:
        pickle.dump(baseline, fh)
    print(f"wrote {OUT}  ({len(baseline['fits'])} fits, "
          f"{len(baseline['errors'])} recorded failures)")


if __name__ == "__main__":
    main()
