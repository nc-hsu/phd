import pickle
import numpy as np
from pathlib import Path
import openseespy.opensees as ops

from standes.analysis.pushover import pushover_analysis
from standes.utils import import_from_path

## model to run a nonlinear static pushover analysis:
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

    # actually starting the structural analysis stuff
    # Determine the maximum displacement and the step size:
    if config_data["U_max"][0] == "drift":
        max_drift = config_data["U_max"][1]
        U_max = ops.nodeCoord(config_data["ctrl_node"])[1] * max_drift / 100
        dU = U_max * config_data["dU"] / max_drift

    analysis_parameters = config_data["analysis_parameters"](config_data["ctrl_node"], U_max, dU, 
                                                             config_data["excitation_dof"])

    # create the model
    recorders, ls_recorders = config["model_init"]()

    # create loading
    # create new load pattern
    ops.timeSeries("Linear", config["tseries_tag"]) 
    ops.pattern("Plain", config["pattern_tag"], config["tseries_tag"])
    _ = config_data["load_pattern"](config_data["excitation_dof"], 1)

    lf, disps = pushover_analysis(analysis_parameters, recorders, ls_recorders, 
                                    config_data["allow_negative_load_factor"])

    po_curve = np.hstack([disps, lf])

    # save the recorders 
    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(output_folder / f"recorders.pickle", "wb") as file:
            pickle.dump(recorders, file)

    with open(output_folder / f"lsrecorders.pickle", "wb") as file:
        pickle.dump(ls_recorders, file)

    with open(output_folder / f"po_curve.pickle", "wb") as file:
        pickle.dump(po_curve, file)