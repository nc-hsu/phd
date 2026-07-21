from pathlib import Path
from standes.analysis.nltha import NlthaParameters
from standes.utils import import_from_path
from msa_process_recorders import process_recorder_func, edp_tags, edp_idxs # type: ignore
"""
No comments can be placed inside the config dictionary as this messes with the
copy and edit functions that are used to create new config files from this template.

Place all comments here:
--------------------------
- results_folder_name should be the first variable in this file, as the copy and
edit functions expect this when creating new config files from this template.

- model_file_name and init_function follow results_folder_name and name the structural
model file (in this folder) and the model-building callable imported from it. This config
is the single place that imports the model; it passes the model module to the
injection-functions factory make_injection_functions(model, **params) named by injection_file.
injection_function_params (default {}) holds any extra keyword parameters passed to it.

- the stripe selection pickles are discovered automatically from gm_selection_src:
  any *.pickle whose name contains both "stripe" and "gm_selection" is used. They
  are ordered by the token following "stripe" in the filename (numerically when it
  is an integer). stripe_order_ascending toggles ascending/descending order.

- the return period, probability of exceedance, intensity measure and intensity
  level of each stripe are read directly from its selection pickle.

- max_n_records caps how many records are run per stripe. None runs all records;
  an integer smaller than a stripe's record count runs only the first that many.

"""

results_folder_name = "msa"
model_file_name = "structural_model.py"
init_function = "model_init"
injection_file = "injection_functions.py"
injection_function_params = {}
gm_selection_src_str = 'C:/Users/clemettn/Desktop/test_msa'
record_src_str = 'C:/Users/clemettn/Documents/phd/data_processed/07_gm_records'
stripe_order_ascending = True
max_n_records = None
gravity = 9810          # mm/s²
excitation_dof = 1
max_record_time = None
print_run_to_screen = True
dt = 0.005
convergece_test = ("NormDispIncr", 1e-6, 50)
nlth_analysis_parameters = NlthaParameters(test=convergece_test, dt=dt)

_model = import_from_path(Path(__file__).parent / model_file_name)
model_init = getattr(_model, init_function)
injection_functions = import_from_path(Path(__file__).parent / injection_file).make_injection_functions(_model, **injection_function_params)

config = {
    "result_dst": Path(__file__).parent / f"{results_folder_name}",
    "gm_selection_src": Path(gm_selection_src_str),
    "stripe_order_ascending": stripe_order_ascending,
    "max_n_records": max_n_records,
    "record_src": Path(record_src_str),
    "injection_functions": injection_functions,
    "post_process": True,
    "edp_tags": edp_tags,
    "edp_idxs": edp_idxs,
    "calculate_collapse_fragility": True,
    "msa_parameters": {
        "initialise_model_func": model_init,
        "proc_recorder_func": process_recorder_func,
        "gravity_factor": gravity,
        "tseries_tag": 2,
        "pattern_tag": 2,
        "excitation_dof": excitation_dof,
        "nlth_analysis_parameters": nlth_analysis_parameters,
        "max_record_time": max_record_time,
        "print_run_to_screen": print_run_to_screen,
    },

}
