import json
import pickle
from pathlib import Path
from functools import partial

import numpy as np
from scipy.optimize import differential_evolution
import openseespy.opensees as ops

from standes.analysis.pushover import cyclic_pushover_analysis, CyclicSpoParameters

from phd_project.scripts.standardise_responses import standardise_responses


def _build_steel01_sdof_model(Fy, E0, b, du, a1, a2) -> tuple[int, int, int, float]:

    # Model parameters
    support_node: int = 1
    free_node: int = 2
    ele_tags = [1]

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    
    ops.node(support_node, 0, 0)
    ops.node(free_node, 0, 0)
    
    ops.fix(support_node, 1, 1, 1)
    ops.fix(free_node, 0, 1, 1)

    # spring 1 - frame
    a3 = a1
    a4 = a2

    ops.uniaxialMaterial("Steel01", 1, Fy, E0, b, a1, a2, a3, a4)
    ops.element("zeroLength", ele_tags[0], support_node, free_node, 
                "-mat", 1, "-dir", 1)
    
    return free_node, support_node, ele_tags


def _build_steel02_sdof_model(Fy, E0, b, du, R0, cR1, cR2, a1, a2) -> tuple[int, int, int, float]:

    # Model parameters
    support_node: int = 1
    free_node: int = 2
    ele_tags = [1]

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    
    ops.node(support_node, 0, 0)
    ops.node(free_node, 0, 0)
    
    ops.fix(support_node, 1, 1, 1)
    ops.fix(free_node, 0, 1, 1)

    # spring 1 - frame
    a3 = a1
    a4 = a2

    transition_params = [R0, cR1, cR2]

    ops.uniaxialMaterial("Steel02", 1, Fy, E0, b, *transition_params, a1, a2, a3, a4)
    ops.element("zeroLength", ele_tags[0], support_node, free_node, 
                "-mat", 1, "-dir", 1)
    
    return free_node, support_node, ele_tags


def _build_steel02_and_hysteretic_sdof_model(
        Fy, E0, b, du, R0, cR1, cR2, a1, a2,
        s1p, e1p, s2p, e2p, s3p, e3p, s1n, e1n, s2n, e2n, s3n, e3n, 
        pinch_x, pinch_y, damage_1, damage_2, beta, mass):
    
    # Model parameters
    support_node: int = 1
    free_node: int = 2
    ele_tags = [1, 2]

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    
    ops.node(support_node, 0, 0)
    ops.node(free_node, 0, 0)
    
    ops.mass(free_node, mass, 0, 0)

    ops.fix(support_node, 1, 1, 1)
    ops.fix(free_node, 0, 1, 1)

    # spring 1 - frame
    a3 = a1
    a4 = a2

    transition_params = [R0, cR1, cR2]

    ops.uniaxialMaterial("Steel02", 1, Fy, E0, b, *transition_params, a1, a2, a3, a4)
    ops.element("zeroLength", ele_tags[0], support_node, free_node, 
                "-mat", 1, "-dir", 1)
    
    # spring 2 - braces
    ops.uniaxialMaterial("Hysteretic", 2, 
                         s1p, e1p, s2p, e2p, s3p, e3p, 
                         s1n, e1n, s2n, e2n, s3n, e3n, 
                         pinch_x, pinch_y, damage_1, damage_2, beta)
    ops.element("zeroLength", ele_tags[1], support_node, free_node, 
                "-mat", 2, "-dir", 1)
    
    return free_node, support_node, ele_tags
    ...


def initialise_steel01_model(a1, a2, Fy, E0, b, du):
    return _build_steel01_sdof_model(Fy, E0, b, du, a1, a2)


def initialise_steel02_model(a1, a2, R0, cR1, cR2, Fy, E0, b, du):
    return _build_steel02_sdof_model(Fy, E0, b, du, R0, cR1, cR2, a1, a2)


def initialise_steel02_hysteretic_model(
        a1, a2, R0, cR1, cR2, pinch_x, pinch_y, damage_1, damage_2, beta, 
        steel02_bb_params, hysteretic_bb_params, mass):
    
    return _build_steel02_and_hysteretic_sdof_model(
        *steel02_bb_params, R0, cR1, cR2, a1, a2,
        *hysteretic_bb_params,
        pinch_x, pinch_y, damage_1, damage_2, beta, 
        mass)
    

def run_cpo(ctrl_node, displacements, dof, dU_max, cpo_parameters, print_progress=False):
    # actually starting the structural analysis stuff
    # Determine the displacement path:
    analysis_parameters = cpo_parameters(
        ctrl_node, 
        displacements, 
        dof,
        dU_max)

    # create loading
    # create new load pattern
    ops.timeSeries("Linear", 1) 
    ops.pattern("Plain", 1, 1)
    ops.load(ctrl_node, *[1, 0, 0])       # unit force on free node

    lf, disps = cyclic_pushover_analysis(analysis_parameters, [], [], print_progress=print_progress)

    po_curve = np.column_stack([disps, lf])

    return po_curve


def objective_function(params, model_init, target_disps, test_data, dU_max):
    """_summary_

    Args:
        params (list[float]): values of the parameters being optimised.
        model_init : a function for initialising the opensees model. Should be callled with "params"
        disp_hists : a list of arrays containing the maximum displacements that should be reached in each cycle
        test_hists : a list of Nx2 arrays containing the F-D path of the test structure
        dU_max : the maximum displacement step for the cpo analysis. This should be set to a value that is smaller than the smallest step in "disp_hists" to ensure the target displacements are reached.
    """
    # create SDOF model with all the required parameters
    free_node, *_ = model_init(*params)

    # run model for the target displacements
    sim_data = run_cpo(free_node, target_disps, 1, dU_max, CyclicSpoParameters)

    # standardise the displacement histories
    target_response, sim_response, _ = standardise_responses(test_data, sim_data)
        
    # calculate SSE for each displacement history. Difference in force for the same displacement
    score = np.sqrt(np.sum((target_response[:, 1] /1000 - sim_response[:, 1] / 1000) ** 2))
    
    return score


""" 
Example of a config for the optimisation of one building:

    config = {
        "results_folder": Path,
        "building_tag": str,
        "sdof_params": Path,             <--- path to json containing the constant params for the springs
        "test_data": Path,        <--- path to po_curve.csv for each mdof test
        "displacements": Path,        <--- path to displacements.csv containing the target displacements for the cpo
        "dU_max": float,
        "param_bounds": list[tuple],
        "initialise_model_func": function
    }

Use a list of configs to optimise several structures
"""

def run_optimisation(config: dict):
    
    # load the test data
    test_data =np.loadtxt(config["test_data"], delimiter=",")

    # load the displacements
    target_disps = np.loadtxt(config["displacements"], delimiter=",")

    # load the sdof params
    with open(config["sdof_params"], "r") as file:
        sdof_const = json.load(file)

    # load the bounds of the optimisation parameters
    param_bounds = np.loadtxt(config["param_bounds"], delimiter=",").tolist()

    # initialise the model
    func = MODEL_INIT_FUNCTIONS[config["initialise_model_func"]]
    model_init = partial(func, **sdof_const)

    args = (model_init, target_disps, test_data, config["dU_max"])

    result = differential_evolution(
        objective_function, 
        param_bounds, 
        args, 
        popsize=config["popsize"], 
        updating="deferred", 
        workers=config["cores"], 
        rng=config["seed"],
        tol=config["tol"], 
        init="latinhypercube", 
        mutation=config["mutation"], 
        recombination=config["recombination"])

    with open(config["results_folder"] / f"{config["building_tag"]}{config['result_name']}", "wb") as file:
        pickle.dump(result, file)

    return result


def validate_optimised_parameters(config, result):
    # load the test data
    test_data =np.loadtxt(config["test_data"], delimiter=",")

    # load the displacements
    target_disps = np.loadtxt(config["displacements"], delimiter=",")

    # load the sdof params
    with open(config["sdof_params"], "r") as file:
        sdof_const = json.load(file)

    # initialise the model
    func = MODEL_INIT_FUNCTIONS[config["initialise_model_func"]]
    model_init = partial(func, **sdof_const)

    free_node, *_ = model_init(*result.x)

    sim_data = run_cpo(free_node, target_disps, 1, config["dU_max"], CyclicSpoParameters)

    target_response, sim_response, _ = standardise_responses(test_data, sim_data)
    
    # return target_response, sim_data
    return target_response, sim_response


MODEL_INIT_FUNCTIONS = {
    "steel01": initialise_steel01_model,
    "steel02": initialise_steel02_model,
    "steel02_and_hysteretic": initialise_steel02_hysteretic_model,
}

if __name__ == "__main__":

    import time

    config = {
        "results_folder": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\optimisation_results"),
        "result_name": "_test_full_optimisation.pickle",
        "building_tag": "3s_cbf_dc2_41",
        "sdof_params": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\sdof_parameters\3s_cbf_dc2_41_sdof_parameters.json"),             
        "test_data": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\test_data\3s_cbf_dc2_41_eq_sdof_cpo.csv"),        
        "displacements": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\target_displacements\3s_cbf_dc2_41_target_displacements.csv"),
        "dU_max": 2.0,
        "param_bounds": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\parameter_bounds\3s_cbf_dc2_41_steel02_hysteretic_parameter_bounds.csv"),
        "initialise_model_func": "steel02_and_hysteretic",
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-8,
        "mutation": (0.5, 1),
        "recombination": 0.7
    }

    print("Running optimisation...")
    tic = time.time()
    result = run_optimisation(config)
    toc = time.time()
    print(f"DONE!!   --  Time taken: {toc - tic:.2f} seconds")
    
    target_response, sim_response = validate_optimised_parameters(config, result)
    print(result.x)

    #### Stuff for testing purposes
    # # load the test data
    # test_data =np.loadtxt(config["test_data"], delimiter=",")

    # # load the displacements
    # target_disps = np.loadtxt(config["displacements"], delimiter=",")

    # # load the sdof params
    # with open(config["sdof_params"], "r") as file:
    #     sdof_const = json.load(file)

    # # initialise the model
    # func = MODEL_INIT_FUNCTIONS[config["initialise_model_func"]]
    # model_init = partial(func, **sdof_const)

    # x = [0.01, 1.0, 20, 0.925, 0.25, 0, 0, 0, 0, 0]

    # free_node, *_ = model_init(*x)

    # sim_data = run_cpo(free_node, target_disps, 1, config["dU_max"], CyclicSpoParameters, print_progress=True)

    # target_response, sim_response, _ = standardise_responses(test_data, sim_data)

    
    ### Plot Model
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(target_response[:, 0], target_response[:, 1], color="k", label="Target")
    ax.plot(sim_response[:, 0], sim_response[:, 1], color="b", ls="-.", label="Optimised SDOF")
    ax.legend()
    plt.show()
    pass