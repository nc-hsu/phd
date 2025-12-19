import json
import pickle
from tqdm import tqdm
from pathlib import Path
import openseespy.opensees as ops

from standes.analysis.nltha import set_analysis_objects, do_analysis_step, limit_state_collapse
from standes.analysis.recorders import update_recorders
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
    # load the record
    gm = load_ground_motion_from_json(config["gm_json_src"] / config["gm_json_file"])

    # create the model
    recorders, collapse_recorders = config["model_init"]()

    # create loading
    # create new excitation pattern
    ops.timeSeries("Path", config["tseries_tag"], 
                "-values", *gm[1], 
                "-time", *gm[0], 
                "-factor", config["gravity_factor"]) # scale for puting record in correct units of gravity

    ops.pattern("UniformExcitation", 
                config["pattern_tag"], config["excitation_dof"], 
                "-accel", config["tseries_tag"],
                "-factor", config["scale_factor"])    # scale for im level

    set_analysis_objects(config["analysis_parameters"])  # intialises the analysis in OpenSees

    # determine the maximum record time
    if config["max_record_time"] is None:
        max_time = gm[0][-1]
    else:
        max_time = config["max_record_time"]

    # record initial state at t=0
    time = [ops.getTime()]
    update_recorders(recorders)      

    with tqdm(total=max_time, desc="Performing NLTHA") as pbar:
        while time[-1] < max_time:

            collapsed = do_analysis_step(config["analysis_parameters"], recorders, 
                                            collapse_recorders)
            if collapsed: break
            time.append(ops.getTime())
            pbar.update(time[-1] - time[-2])

    ls_collapse = limit_state_collapse(collapse_recorders)        

    # save the recorders 
    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(output_folder / f"recorders.pickle", "wb") as file:
            pickle.dump(recorders, file)

    with open(output_folder / f"collapserecorders.pickle", "wb") as file:
        pickle.dump(collapse_recorders, file)

    with open(output_folder / f"timearray.pickle", "wb") as file:
        pickle.dump(time, file)

    results = {
        "collapsed": collapsed,
        "ls_collapse": ls_collapse
        }

    with open(output_folder / "collapse.json", "w") as file:
        json.dump(results, file)

