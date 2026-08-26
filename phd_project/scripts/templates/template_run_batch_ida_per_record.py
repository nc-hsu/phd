import argparse
from datetime import datetime
from pathlib import Path

from standes.parallel import run_records_in_subprocesses
from standes.analysis.ida import collate_ida_results, process_ida_results
from standes.groundmotion import load_ground_motion_from_json
from standes.utils import import_from_path


## launcher / coordinator: run an IDA for a SINGLE building, parallelised across
## ground motion records (one CPU core per record). Every record is driven by the
## SAME config file (config_data) -- only the record tag differs -- so there is no
## need for a separate config file per record.
##
## After all per-record subprocesses finish, the per-record results are collated
## into ida_results.pickle / record_logs.json and (optionally) post-processed into
## fragility curves, exactly as the serial run_ida_htf_multiple_records.py does.
def run(config_data: str | Path,
        max_workers: int | None = None,
        use_semaphore: bool = False,
        worker_script: str | Path | None = None,
        show_worker_windows: bool = True,
        n_cores: int | None = None):

    config_path = Path(config_data)
    if not (config_path.exists() and config_path.is_file()):
        raise ValueError(f"Invalid path for config_data: {config_path}")

    # load the config to read output folder, record tags and post-processing opts
    config: dict = import_from_path(config_path).config

    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    # the worker that each subprocess runs (sibling script by default)
    if worker_script is None:
        worker_script = Path(__file__).parent / "run_ida_htf_per_record.py"

    # record tags come straight from the existing config -- no per-record config
    record_tags = list(config["gm_json_files"].keys())

    # optionally use the shared, machine-global semaphore so this can coexist
    # with a multi-building batch; otherwise cap with a local max_workers count
    semaphore_module = None
    if use_semaphore:
        import importlib
        # phd_project is installed (editable), so the package-qualified name works
        # without PYTHONPATH fiddling; fall back to the bare name for setups that
        # put the folder containing process_semaphore on PYTHONPATH directly.
        for module_name in ("phd_project.process_semaphore.process_semaphore",
                            "process_semaphore.process_semaphore"):
            try:
                semaphore_module = importlib.import_module(module_name)
                break
            except ModuleNotFoundError:
                continue
        if semaphore_module is None:
            raise ModuleNotFoundError(
                "use_semaphore=True but the process_semaphore module could not be "
                "imported as 'phd_project.process_semaphore.process_semaphore' or "
                "'process_semaphore.process_semaphore'. Install phd_project or add "
                "the folder containing process_semaphore to PYTHONPATH, or run with "
                "use_semaphore=False (max_workers).")
        # n_cores is None when this coordinator was spawned by a *_buildings
        # launcher -- that launcher has already set the machine-global cap, and we
        # must not overwrite it. Only a direct CLI run sets it here.
        if n_cores is not None:
            semaphore_module.set_max_concurrent(n_cores, set_by=Path(__file__).name)

    print(f"Launch Time: {datetime.now()}"
          + (f" | {semaphore_module.describe_max_concurrent()}" if semaphore_module else ""))
    start_time = datetime.now()

    # fire one subprocess per record and wait for them all
    run_records_in_subprocesses(
        worker_script=worker_script,
        config_path=config_path,
        record_tags=record_tags,
        max_workers=max_workers,
        semaphore_module=semaphore_module,
        show_worker_windows=show_worker_windows,
        log_dir=output_folder / "worker_logs")

    elapsed_time = datetime.now() - start_time
    print(f"All records complete. Time elapsed: {elapsed_time}")

    ## COLLATE per-record results -> ida_results.pickle + record_logs.json
    ida_results, _ = collate_ida_results(output_folder)

    ## POST PROCESSING
    if config["post_process"]:
        # only load records when collapse-IM conversions are requested (else skip I/O).
        # build gm_record_files from the same config the workers used, keyed by the
        # record tags that collate_ida_results returns.
        convert_collapse_to_ims = config.get("convert_collapse_to_ims")
        gm_records = None
        if convert_collapse_to_ims:
            gm_record_files = {k: Path(config["gm_json_src"]) / v
                               for k, v in config["gm_json_files"].items()}
            gm_records = {tag: load_ground_motion_from_json(gm_record_files[tag])
                          for tag in ida_results.keys()}

        process_ida_results(
            ida_results.values(),
            output_folder,
            edp_idxs=config["edp_idxs"],
            edp_tags=config["edp_tags"],
            ida_fractiles=config["ida_fractiles"],
            collapse_fragility=config["calculate_collapse_fragility"],
            record_ids=ida_results.keys(),
            intensity_measure=config["ida_parameters"]["intensity_measure"],
            gm_records=gm_records,
            convert_collapse_to_ims=convert_collapse_to_ims,
            gravity_factor=config["ida_parameters"]["gravity_factor"])


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_batch_ida_per_record.py config_ida_htf_femap695_set.py --max-workers 4
    #   python run_batch_ida_per_record.py config_ida_htf_femap695_set.py --use-semaphore --n-cores 20
    parser = argparse.ArgumentParser(
        description="Run an IDA for a single building, parallelised across ground motion records.")
    parser.add_argument("config", nargs="?", default="config_ida_htf.py",
                        help="IDA config file (relative to this folder, or an absolute path)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="max records to run at once for THIS coordinator "
                             "(default: physical cores - 4)")
    parser.add_argument("--use-semaphore", action="store_true",
                        help="use the machine-global process_semaphore instead of --max-workers")
    parser.add_argument("--n-cores", type=int, default=None,
                        help="with --use-semaphore: the machine-global cap on simultaneous "
                             "workers across ALL launchers on this machine (default policy: "
                             "physical cores - 4). Sticky until the next --n-cores.")
    parser.add_argument("--quiet", action="store_true",
                        help="no worker windows; stream worker output to worker_logs/ instead")
    args = parser.parse_args()

    if args.use_semaphore and args.n_cores is None:
        parser.error("--n-cores is required with --use-semaphore; drop --use-semaphore "
                     "and pass --max-workers N to run with a fixed local worker count instead")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path, max_workers=args.max_workers,
        use_semaphore=args.use_semaphore,
        show_worker_windows=not args.quiet,
        n_cores=args.n_cores)
