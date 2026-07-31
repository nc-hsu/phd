import argparse
from datetime import datetime
import json
import pickle
from pathlib import Path

from standes.analysis.ida import (
    ida_htf_multiple_records,
    process_ida_results,
    )

from standes.groundmotion import load_ground_motion_from_json
from standes.utils import import_from_path

## model to run a nonlinear time history analysis:
def run(config_data: str|Path|dict):
    # loading the configuration data from the provided file path
    if isinstance(config_data, str):
        config_data = Path(config_data)
        if not (config_data.exists() and config_data.is_file()):
            raise ValueError(f"Invalid path for config_data: {config_data}")

    if isinstance(config_data, Path):
        # import the configuration data from module path
        config_module = import_from_path(config_data)
        config: dict = config_module.config
    
    elif not isinstance(config_data, dict):
        raise ValueError
    
    else:
        config = config_data

    # acutally starting the structural analysis stuff
    print(f"Launch Time: {datetime.now()}")

    # initialise output folder
    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    # get the record file paths
    gm_record_files = {k: Path(config["gm_json_src"]) / v for k,v in config["gm_json_files"].items()}

    injection_functions = config["injection_functions"]

    ## Actually RUN IDA
    # partially initialise the model function

    # do an ida
    start_time = datetime.now()

    ida_results, record_logs = ida_htf_multiple_records(
            ida_output_folder=output_folder,
            gm_record_files=gm_record_files,
            ida_parameters=config["ida_parameters"],
            use_interim_htf_result=config["use_interim_htf_result"],
            injection_functions=injection_functions)

    # add some final stuff to the log file
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    record_logs["elapsed_time"] = str(elapsed_time)

    # save the results and logs
    with open(output_folder / "ida_results.pickle", "wb") as file:
        pickle.dump(ida_results, file)

    with open(output_folder / "record_logs.json", "w") as file:
        json.dump(record_logs, file, indent=4)

    print(f"Time elapsed: {elapsed_time}")

    ## POST PROCESSING
    if config["post_process"]:
        # only load records when collapse-IM conversions are requested (else skip I/O)
        convert_collapse_to_ims = config.get("convert_collapse_to_ims")
        gm_records = None
        if convert_collapse_to_ims:
            gm_records = {tag: load_ground_motion_from_json(path)
                          for tag, path in gm_record_files.items()}

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
    #   python run_ida_htf_multiple_records.py config_ida_htf.py
    parser = argparse.ArgumentParser(description="Run an IDA for a single building, serially over all records.")
    parser.add_argument("config", nargs="?", default="config_ida_htf.py",
                        help="config file (relative to this folder, or an absolute path)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path)
