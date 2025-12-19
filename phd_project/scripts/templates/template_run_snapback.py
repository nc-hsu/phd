import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import openseespy.opensees as ops
from scipy.signal import argrelextrema

from standes.analysis.nltha import set_analysis_objects
from standes.analysis.recorders import update_recorders
from standes.utils import import_from_path


## model to run a snapback analysis to check damping:
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

    # create the model
    recorders, _ = config["model_init"]()
    # remove the acceleration recorders
    recorders = [r for r in recorders if r.recorder_type != "node_acceleration"]

    # impose displacement at the roof by adding a force
    F0 = config["F_0"]
    dof = config["dof"]
    ctrl_node = config["ctrl_node"]

    ops.timeSeries("Linear", config["tseries_tag"])
    ops.pattern("Plain", config["pattern_tag"], config["tseries_tag"])

    for n in config["roof_nodes"]:
        ops.load(n, F0, 0, 0, 0, 0, 0)

    # setup static analysis
    analysis_parameters = config["static_analysis_parameters"]
    ops.wipeAnalysis()
    ops.constraints(analysis_parameters.constraints)
    ops.numberer(analysis_parameters.numberer)
    ops.system(analysis_parameters.system)
    ops.integrator(*analysis_parameters.integrator)
    ops.test(*analysis_parameters.test)
    ops.algorithm(*analysis_parameters.algorithm)
    ops.analysis(analysis_parameters.analysis)

    # run static analysis
    ops.analyze(analysis_parameters.n_steps)

    print(f"Initial Displacement: {ops.nodeDisp(ctrl_node)[0]:.1f} mm")

    ops.wipeAnalysis()
    ops.remove("timeSeries", 2)
    ops.remove("pattern", 2)

    ops.setTime(0)

    analysis_parameters = config["dynamic_analysis_parameters"]
    set_analysis_objects(analysis_parameters)

    t_final = config["t_final"]
    dt = analysis_parameters.dt
    Nsteps = int(t_final/dt)

    # set up some recorders for damping forces
    damping_recorders_l2 = {
        101010200 : [0],
        102010200 : [0],
        103010200 : [0],
        104010200 : [0],
    }

    damping_recorders_l3 = {
        101010300 : [0],
        102010300 : [0],
        103010300 : [0],
        104010300 : [0],
    }

    damping_recorders_l4 = {
        101010400 : [0],
        102010400 : [0],
        103010400 : [0],
        104010400 : [0],
    }

    roof_disp = [ops.nodeDisp(ctrl_node)[0]]
    time = [ops.getTime()]

    for i in tqdm(range(Nsteps), desc="Performing Snapback Analysis"):
        ops.analyze(1,analysis_parameters.dt)
        roof_disp.append(ops.nodeDisp(ctrl_node)[0])
        time.append(ops.getTime())

        update_recorders(recorders)

        for k,v in damping_recorders_l2.items():
            v.append(ops.nodeResponse(k, dof, 8))

        for k,v in damping_recorders_l3.items():
            v.append(ops.nodeResponse(k, dof, 8))

        for k,v in damping_recorders_l4.items():
            v.append(ops.nodeResponse(k, dof, 8))


    # calculate the damping ratio
    roof_disp = np.array(roof_disp)
    local_maxs = argrelextrema(roof_disp, np.greater)
    n_peaks = len(local_maxs[0])

    U1 = roof_disp[local_maxs[0][1]]
    Un = roof_disp[local_maxs[0][n_peaks-1]]

    delta = 1 / n_peaks * np.log(U1 / Un)
    observed_ksi = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2) 

    # save output
    result_dst: Path = config["result_dst"]
    result_dst.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "delta": float(delta),
        "ksi": float(observed_ksi),
        "time": time,
        "roof_disp": roof_disp.tolist(),
        "local_maxs": local_maxs[0].tolist()
    }

    with open(result_dst / "results.json", "w") as file:
        json.dump(result_dict, file, indent=4)

    with open(result_dst / "damping_recorders_l2.pickle", "wb") as file:
        pickle.dump(damping_recorders_l2, file)

    with open(result_dst / "damping_recorders_l3.pickle", "wb") as file:
        pickle.dump(damping_recorders_l3, file)

    with open(result_dst / "damping_recorders_l4.pickle", "wb") as file:
        pickle.dump(damping_recorders_l4, file)

    with open(result_dst / "recorders.pickle", "wb") as file:
        pickle.dump(recorders, file)





























