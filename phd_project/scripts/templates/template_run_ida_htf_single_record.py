from datetime import datetime
import json
import pickle
from pathlib import Path

from standes.analysis.ida import (
    ida_htf_single_record, 
    process_ida_results,
    load_newest_interim_htf_result
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

    # get the record file --> just take the first one specified in the config
    gm_record_file = Path(config["gm_json_src"]) / list(config["gm_json_files"].values())[0]
    gm_record = load_ground_motion_from_json(gm_record_file)

    injection_functions = config["injection_functions"]

    # determine if the interim result should be used
    if config["use_interim_htf_result"]:
        interim_htf_result = load_newest_interim_htf_result(output_folder)
    else: 
        interim_htf_result = ()

    ## Actually RUN IDA
    # partially initialise the model function

    # do an ida
    start_time = datetime.now()

    ida_result = ida_htf_single_record(
            gm_record=gm_record,
            record_filepath=gm_record_file,
            record_results_folder=output_folder,
            interim_htf_result=interim_htf_result,
            injection_functions=injection_functions,
            **config["ida_parameters"])

    # add some final stuff to the log file
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    record_log = ida_result[-1]
    record_log["elapsed_time"] = f"{elapsed_time}"

    # save the results and logs
    with open(output_folder / "ida_results.pickle", "wb") as file:
        pickle.dump(ida_result, file)

    with open(output_folder / "record_log.json", "w") as file:
        json.dump(record_log, file, indent=4)

    print(f"Time elapsed: {elapsed_time}")

    ## POST PROCESSING 
    if config["post_process"]:
        process_ida_results(
            [ida_result], 
            output_folder, 
            edp_idxs=config["edp_idxs"],
            edp_tags=config["edp_tags"],
            ida_fractiles=[],           # not possible with 1 record
            collapse_fragility=False)   # not possible with 1 record
        

if __name__ == "__main__":
    # example usage
    config_path = Path(__file__).parent / "config_ida_htf_single_record.py"
    run(config_path)