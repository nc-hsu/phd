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
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_10',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_10_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_10_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_10_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_10_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_20',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_20_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_20_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_20_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_20_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_31',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_31_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_31_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_31_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_31_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_41',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_41_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_41_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_41_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_41_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_51',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_51_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_51_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_51_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_51_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_61',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_61_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_61_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_61_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_61_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_71',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_71_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_71_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_71_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_71_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_102',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_102_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_102_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_102_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_102_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    },
    {
        "results_folder": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/optimisation_results'),
        "result_name": '_steel02_and_hysteretic_optimisation.pickle',
        "building_tag": '3s_cbf_dc2_122',
        "sdof_params": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/sdof_parameters/3s_cbf_dc2_122_sdof_parameters.json'),
        "test_data": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/test_data/3s_cbf_dc2_122_eq_sdof_cpo.csv'),
        "displacements": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/target_displacements/3s_cbf_dc2_122_target_displacements.csv'),
        "dU_max": 2.0,
        "param_bounds": Path(r'C:/Users/clemettn/Documents/phd/data_processed/09_sdof_fitting/parameter_bounds/3s_cbf_dc2_122_steel02_hysteretic_parameter_bounds.csv'),
        "initialise_model_func": 'steel02_and_hysteretic',
        "popsize": 15,
        "cores": 50,
        "seed": 1,
        "tol": 1e-08,
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