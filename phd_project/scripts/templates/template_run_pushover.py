import argparse
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
        
    if isinstance(config_data, Path):
        if not (config_data.exists() and config_data.is_file()):
            raise ValueError(f"Invalid path for config_data: {config_data}")
        
        # import the configuration data from module path
        config_module = import_from_path(config_data)
        config: dict = config_module.config
    
    elif not isinstance(config_data, dict):
        raise ValueError
    
    else:
        config = config_data

    # create the opensees model
    recorders, ls_recorders = config["model_init"]()

    # actually starting the structural analysis stuff
    # Determine the maximum displacement and the step size:
    U_max = config["U_max"]
    dU = config["dU"]

    if config["displacement_type"] == "drift":
        coord = ops.nodeCoord(config["ctrl_node"])
        if len(coord) == 2:
            height = coord[1]
        elif len(coord) == 3:
            height = coord[2]

        # convert the drifts to displacements
        U_max = height * U_max / 100
        dU = height * dU / 100

    elif config["displacement_type"] != "disp":
        raise ValueError("Invalid displacement type: either 'drift', or 'disp'")

    analysis_parameters = config["analysis_parameters"](config["ctrl_node"], U_max, dU, 
                                                        config["excitation_dof"])
    
    # create loading
    # create new load pattern
    ops.timeSeries("Linear", config["tseries_tag"]) 
    ops.pattern("Plain", config["pattern_tag"], config["tseries_tag"])
    _ = config["load_pattern"](config["excitation_dof"], 1)

    lf, disps = pushover_analysis(analysis_parameters, recorders, ls_recorders, 
                                    config["allow_negative_load_factor"])

    po_curve = np.column_stack([disps, lf])

    # save the recorders 
    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(output_folder / f"recorders.pickle", "wb") as file:
            pickle.dump(recorders, file)

    with open(output_folder / f"lsrecorders.pickle", "wb") as file:
        pickle.dump(ls_recorders, file)

    np.savetxt(output_folder / f"po_curve.csv", po_curve, delimiter=",")

if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_pushover.py config_pushover.py
    parser = argparse.ArgumentParser(description="Run a pushover analysis.")
    parser.add_argument("config", nargs="?", default="config_pushover.py",
                        help="config file (relative to this folder, or an absolute path)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path)
