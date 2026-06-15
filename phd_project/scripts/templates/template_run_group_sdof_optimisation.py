import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from phd_project.scripts.WP1_ground_motion_set.optimise_sdof_parameters_differential_evolution import run_optimisation

running_processes = []  # List of dicts: {proc, script, config, start_time}

# Paths to your scripts and the configurations

configs = [
    {
        "results_folder": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\optimisation_results"),
        "result_name": "_mechanism_steel02_optimisation.pickle",
        "building_tag": "3s_cbf_dc2_41",
        "sdof_params": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\sdof_parameters\3s_cbf_dc2_41_mechanism_sdof_parameters.json"),             
        "test_data": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\test_data\3s_cbf_dc2_41_mechanism_eq_sdof_cpo.csv"),        
        "displacements": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\target_displacements\3s_cbf_dc2_41_target_displacements.csv"),
        "dU_max": 1.0,
        "param_bounds": Path(r"C:\Users\clemettn\Documents\phd\data_processed\09_sdof_fitting\parameter_bounds\3s_cbf_dc2_41_steel_02_parameter_bounds.csv"),
        "initialise_model_func": "initialise_steel02_model",
        "popsize": 15,
        "cores": 30,
        "seed": 1,
        "tol": 1e-8,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
]

def main(configs):
    """
    Runs the optimization for each configuration with a progress bar.
    """
    # Wrap the iterator with tqdm for the progress bar
    for config in tqdm(configs, desc="Optimizing configurations"):
        _ = run_optimisation(config)

if __name__ == "__main__":
    # Pass the configs list defined globally
    main(configs)