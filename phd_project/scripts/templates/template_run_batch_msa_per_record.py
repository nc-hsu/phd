import argparse
from datetime import datetime
from pathlib import Path

from standes.parallel import run_records_in_subprocesses
from standes.analysis.msa import (
    find_stripe_pickles, stripe_id, stripe_record_tags, record_is_complete,
    collate_stripe, collate_msa_results)
from standes.utils import import_from_path


## launcher / coordinator: run an MSA for a SINGLE building, parallelised across
## EVERY (stripe, record) pair at once (one CPU core per pair) for maximum
## concurrency. Every pair is driven by the SAME config file -- only the
## "{stripe_id}:{record}" tag differs -- so there is no per-record config file.
##
## Resumable: a pair whose record log already exists is skipped, so re-running the
## batch (e.g. after dropping in an extra stripe pickle) only analyses the missing
## pairs. Set `resume = False` in the config to force a full re-run. Stripe result
## folders are named `stripe_{stripe_id}` (the stripe's intensity tag), so adding a
## lower- or higher-intensity stripe never renumbers the existing ones.
##
## After every worker finishes, each stripe log is assembled, the collated
## msa_log.json is written and (optionally) post-processed into a collapse fragility.
def run(config_data: str | Path,
        max_workers: int | None = None,
        use_semaphore: bool = True,
        worker_script: str | Path | None = None,
        show_worker_windows: bool = True):

    config_path = Path(config_data)
    if not (config_path.exists() and config_path.is_file()):
        raise ValueError(f"Invalid path for config_data: {config_path}")

    config: dict = import_from_path(config_path).config

    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    if worker_script is None:
        worker_script = Path(__file__).parent / "run_msa_per_record.py"

    stripe_pickles = find_stripe_pickles(
        config["gm_selection_src"], ascending=config["stripe_order_ascending"])
    if not stripe_pickles:
        raise FileNotFoundError(
            f"No stripe selection pickles found in {config['gm_selection_src']}")

    resume = config.get("resume", True)

    # build the full flattened list of "{stripe_id}:{record}" tags, skipping any
    # (stripe, record) pair whose result already exists when resuming
    tags: list[str] = []
    n_total = 0
    for stripe_pickle in stripe_pickles:
        sid = stripe_id(stripe_pickle)
        stripe_folder = output_folder / f"stripe_{sid}"
        for record_tag in stripe_record_tags(stripe_pickle, config["max_n_records"]):
            n_total += 1
            if resume and record_is_complete(stripe_folder, record_tag):
                continue
            tags.append(f"{sid}:{record_tag}")

    semaphore_module = _load_semaphore() if use_semaphore else None

    print(f"Launch Time: {datetime.now()}")
    if resume and n_total - len(tags):
        print(f"resuming: {n_total - len(tags)}/{n_total} pair(s) already complete, "
              f"dispatching {len(tags)}")
    print(f"Dispatching {len(tags)} (stripe, record) job(s) across "
          f"{len(stripe_pickles)} stripe(s)")
    start_time = datetime.now()

    if tags:
        run_records_in_subprocesses(
            worker_script=worker_script,
            config_path=config_path,
            record_tags=tags,
            max_workers=max_workers,
            semaphore_module=semaphore_module,
            show_worker_windows=show_worker_windows,
            log_dir=output_folder / "worker_logs")
    else:
        print("Nothing to dispatch -- every (stripe, record) pair is already complete.")

    elapsed_time = datetime.now() - start_time
    print(f"All jobs complete. Time elapsed: {elapsed_time}")

    ## COLLATE per-record logs -> per-stripe logs -> msa_log.json (all stripes, so
    ## the collated log reflects both freshly-run and previously-cached pairs)
    for stripe_pickle in stripe_pickles:
        collate_stripe(output_folder, stripe_id(stripe_pickle), stripe_pickle)
    collate_msa_results(output_folder)

    ## POST PROCESSING
    if config["post_process"]:
        from standes.analysis.msa import process_msa_results
        process_msa_results(
            output_folder,
            edp_idxs=config["edp_idxs"],
            edp_tags=config["edp_tags"],
            collapse_fragility=config["calculate_collapse_fragility"])


def _load_semaphore():
    import importlib
    # phd_project is installed (editable), so the package-qualified name works
    # without PYTHONPATH fiddling; fall back to the bare name for setups that put
    # the folder containing process_semaphore on PYTHONPATH directly.
    for module_name in ("phd_project.process_semaphore.process_semaphore",
                        "process_semaphore.process_semaphore"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "use_semaphore=True but the process_semaphore module could not be imported "
        "as 'phd_project.process_semaphore.process_semaphore' or "
        "'process_semaphore.process_semaphore'. Install phd_project or add the folder "
        "containing process_semaphore to PYTHONPATH, or run with use_semaphore=False "
        "(max_workers).")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_batch_msa_per_record.py config_msa.py --max-workers 4
    parser = argparse.ArgumentParser(description="Run an MSA for a single building, parallelised across (stripe, record) pairs.")
    parser.add_argument("config", nargs="?", default="config_msa.py",
                        help="config file (relative to this folder, or an absolute path)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="max subprocesses to run at once (default: physical cores - 3)")
    parser.add_argument("--use-semaphore", action="store_true",
                        help="use the machine-global process_semaphore instead of --max-workers")
    parser.add_argument("--quiet", action="store_true",
                        help="no worker windows; stream worker output to worker_logs/ instead")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path, max_workers=args.max_workers,
        use_semaphore=args.use_semaphore,
        show_worker_windows=not args.quiet)
