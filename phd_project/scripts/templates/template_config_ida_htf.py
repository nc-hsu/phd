from pathlib import Path
from standes.analysis.nltha import NlthaParameters
from structural_model import model_init# type: ignore
from injection_functions import injection_functions # type: ignore
from config_im import im # type: ignore
from ida_process_recorders import process_recorder_func, edp_tags, edp_idxs # type: ignore
""" 
No comments can be placeed inside the config ditionary as this messing with the
copy and edit functions that are used to create new config files from this template.

Place all comments here:
--------------------------
- results_folder_name should be the first variable in this file, as the copy and 
edit functions expect this when creating new config files from this template.

"""

results_folder_name = "ida"
gm_json_src_str = 'E:/gm_records_p695'
record_filenames = ['fema_p695_120111.json']
gravity = 9810          # mm/s²
iml_0_factor = 0.05
iml_step_factor = 0.15
max_runs = 13
excitation_dof = 1
max_record_time = None
print_run_to_screen = True
skip_non_ls_collapse = False
trace_delta = 1/3
trace_tolerance = 0.0
trace_max_resolution = 0.1
min_runs = 8
dt = 0.005      
convergece_test = ("NormDispIncr", 1e-6, 50)
nlth_analysis_parameters = NlthaParameters(test=convergece_test, dt=dt)

config = {
    "result_dst": Path(__file__).parent / f"{results_folder_name}",
    'gm_json_src': Path(gm_json_src_str),
    'gm_json_files': {str(ii): record for ii, record in enumerate(record_filenames)},
    "injection_functions": injection_functions,
    'use_interim_htf_result': True,
    'post_process': True,
    'edp_tags': edp_tags,
    'edp_idxs': edp_idxs,
    'ida_fractiles': [0.16, 0.5, 0.84],
    'calculate_collapse_fragility': True,
    "ida_parameters": {
        "initialise_model_func": model_init,
        "proc_recorder_func": process_recorder_func,
        "intensity_measure": im,
        "iml_0": iml_0_factor * gravity,
        "iml_step": iml_step_factor * gravity,
        "max_runs": max_runs,
        "gravity_factor": gravity,
        "tseries_tag": 2,
        "pattern_tag": 2,
        "excitation_dof": excitation_dof,
        "nlth_analysis_parameters": nlth_analysis_parameters,
        "max_record_time": max_record_time,
        "print_run_to_screen": print_run_to_screen, 
        "skip_non_ls_collapse": skip_non_ls_collapse,
        "trace_delta": trace_delta,
        "trace_tolerance": trace_tolerance, 
        "trace_max_resolution": trace_max_resolution,
        "min_runs": min_runs
    },

}