import argparse
from datetime import datetime
import json
from pathlib import Path

from standes.analysis.ida import ida_htf_one_record
from standes.utils import import_from_path


## worker: run the hunt-trace-fill IDA for a SINGLE record of the config.
## Dispatched one-per-process by run_batch_ida_per_record.py so that all records
## of a building can be analysed concurrently (one CPU core per record). The
## record is selected by `record_tag`, a key of config["gm_json_files"], so no
## per-record config file is needed.
def run(config_data: str | Path | dict, record_tag: str):
    # loading the configuration data from the provided file path
    if isinstance(config_data, str):
        config_data = Path(config_data)
        if not (config_data.exists() and config_data.is_file()):
            raise ValueError(f"Invalid path for config_data: {config_data}")

    if isinstance(config_data, Path):
        config_module = import_from_path(config_data)
        config: dict = config_module.config

    elif not isinstance(config_data, dict):
        raise ValueError

    else:
        config = config_data

    record_tag = str(record_tag)
    print(f"Launch Time: {datetime.now()} | record {record_tag}")

    # initialise output folder (the record_{tag} subfolder is made by standes)
    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    # resolve the single record path for this tag
    if record_tag not in config["gm_json_files"]:
        raise KeyError(
            f"record_tag '{record_tag}' not in config['gm_json_files'] "
            f"(available: {list(config['gm_json_files'].keys())})")
    record_path = Path(config["gm_json_src"]) / config["gm_json_files"][record_tag]

    # do the ida for just this record
    start_time = datetime.now()

    ida_result = ida_htf_one_record(
        ida_output_folder=output_folder,
        record_tag=record_tag,
        record_path=record_path,
        ida_parameters=config["ida_parameters"],
        use_interim_htf_result=config["use_interim_htf_result"],
        injection_functions=config["injection_functions"])

    elapsed_time = datetime.now() - start_time

    # record the elapsed time in this record's log file
    record_log = ida_result[-1]
    record_log["elapsed_time"] = str(elapsed_time)
    record_log_path = output_folder / f"record_{record_tag}" / f"ida_log_record_{record_tag}.json"
    with open(record_log_path, "w") as file:
        json.dump(record_log, file, indent=4)

    print(f"Time elapsed (record {record_tag}): {elapsed_time}")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_ida_htf_per_record.py config_ida_htf.py 0
    parser = argparse.ArgumentParser(description="Run the hunt-trace-fill IDA for a SINGLE record of the config.")
    parser.add_argument("config", nargs="?", default="config_ida_htf.py",
                        help="config file (relative to this folder, or an absolute path)")
    parser.add_argument("tag", nargs="?", default="0",
                        help="the single record/stripe tag to run")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path, args.tag)
