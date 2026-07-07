"""Process-isolated modal-analysis runner for the per-site MDOF case studies.

Each generated site folder contains a ``config_modal.py`` that does
``from structural_model import model_init``. Running several folders' configs in
one interpreter would collide on the cached ``structural_model`` module, so every
modal analysis must run in its own fresh process. ``run_modal_analyses_parallel``
uses a ``ProcessPoolExecutor`` with ``max_tasks_per_child=1`` (Python 3.11+) so
each task gets a clean interpreter.
"""

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_modal_for_folder(folder) -> dict:
    """Run the modal analysis defined in ``folder/config_modal.py``.

    Imports the folder's ``run_modal.py`` and ``config_modal.py`` via
    ``standes.utils.import_from_path`` (which puts the folder on ``sys.path``) and
    executes the analysis, writing the ``modal/`` results folder. Intended to run
    in a fresh worker process.

    Returns a picklable status dict ``{"folder", "success", "error"}``. Any
    exception (notably ``opensees.OpenSeesError``, which cannot itself be pickled
    back to the parent process) is caught here and its traceback returned as a
    plain string, so one bad folder surfaces the real error instead of crashing
    the whole pool with an opaque ``PicklingError``.
    """
    from standes.utils import import_from_path

    folder = Path(folder)
    try:
        run_modal = import_from_path(folder / "run_modal.py")
        run_modal.run(folder / "config_modal.py")
        return {"folder": str(folder), "success": True, "error": None}
    except Exception:
        return {"folder": str(folder), "success": False,
                "error": traceback.format_exc()}


def default_n_workers() -> int:
    return max((os.cpu_count() or 1) - 3, 1)


def run_modal_analyses_parallel(folders, *, n_workers=None, raise_on_error=True):
    """Run modal analyses for many site folders in parallel, one per process.

    Returns a list of status dicts ``{"folder", "success", "error"}`` (order not
    guaranteed). Progress is reported with ``tqdm``. If any folder failed, the
    aggregated tracebacks are raised as a ``RuntimeError`` (set
    ``raise_on_error=False`` to only print them and keep the results instead).
    """
    from tqdm.auto import tqdm

    if n_workers is None:
        n_workers = default_n_workers()

    folders = [Path(f) for f in folders]
    results = []
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=1) as ex:
        futures = {ex.submit(run_modal_for_folder, f): f for f in folders}
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Modal analyses"
        ):
            results.append(fut.result())

    failures = [r for r in results if not r["success"]]
    if failures:
        msg = f"{len(failures)}/{len(results)} modal analyses failed:"
        for r in failures:
            msg += f"\n\n--- {r['folder']} ---\n{r['error']}"
        if raise_on_error:
            raise RuntimeError(msg)
        print(msg)
    return results
