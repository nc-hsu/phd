import argparse
from datetime import datetime
from pathlib import Path

from standes.parallel import run_records_in_subprocesses
from standes.analysis.msa import (
    find_stripe_pickles, stripe_record_tags, collate_stripe, collate_msa_results)
from standes.utils import import_from_path


## launcher / coordinator (run mode 2): run an MSA parallelised across the records
## of a stripe (one CPU core per record), looping the stripes serially. Every
## record is driven by the SAME config file -- only the "{stripe}:{record}" tag
## differs -- so there is no need for a separate config file per record.
##
## After all records of a stripe finish, the stripe log is assembled; after all
## stripes finish, the collated msa_log.json is written and (optionally)
## post-processed into a collapse fragility.
def run(config_data: str | Path,
        max_workers: int | None = None,
        use_semaphore: bool = False,
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

    semaphore_module = _load_semaphore() if use_semaphore else None

    print(f"Launch Time: {datetime.now()}")
    start_time = datetime.now()

    # one stripe at a time; within a stripe fire one subprocess per record
    for stripe_idx, stripe_pickle in enumerate(stripe_pickles, start=1):
        record_tags = stripe_record_tags(stripe_pickle, config["max_n_records"])
        tags = [f"{stripe_idx}:{record_tag}" for record_tag in record_tags]

        print(f"\nStripe {stripe_idx} ({stripe_pickle.name}): {len(tags)} record(s)")
        run_records_in_subprocesses(
            worker_script=worker_script,
            config_path=config_path,
            record_tags=tags,
            max_workers=max_workers,
            semaphore_module=semaphore_module,
            show_worker_windows=show_worker_windows,
            log_dir=output_folder / "worker_logs")

        # assemble this stripe's log from the per-record logs just written
        collate_stripe(output_folder, stripe_idx, stripe_pickle)

    elapsed_time = datetime.now() - start_time
    print(f"All stripes complete. Time elapsed: {elapsed_time}")

    ## COLLATE per-stripe logs -> msa_log.json
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
    try:
        return importlib.import_module("process_semaphore.process_semaphore")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "use_semaphore=True but 'process_semaphore' is not importable. "
            "Add the folder containing the process_semaphore package to "
            "PYTHONPATH, or run with use_semaphore=False (max_workers).") from exc


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_batch_msa_per_record.py config_msa.py --max-workers 4
    parser = argparse.ArgumentParser(description="Run an MSA for a single building, parallelised across records.")
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
