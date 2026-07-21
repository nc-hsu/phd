from pathlib import Path
from standes.analysis.pushover import CyclicSpoParameters
from standes.analysis.load_patterns import ec8_triangular_load_pattern
from standes.utils import import_from_path


"""
No comments or spaces can be placeed inside the config ditionary after the last comma
as this messes with the
copy and edit functions that are used to create new config files from this template.

Place all comments here:
--------------------------
results_folder_name should be the first variable in this file, as the copy and
edit functions expect this when creating new config files from this template.

model_file_name and init_function follow results_folder_name and name the structural
model file (in this folder) and the model-building callable imported from it.

The following parameters are used to create a multilinear cyclic ramp function
for the cyclic pushover.
"displacement_type": 'drift' or 'displacement'      # drifts will be converted to displacements in the cpo algorithm
"dU": 0.5,              # the basic dispalcement step
"""

results_folder_name = "cyclic_pushover"
model_file_name = "structural_model.py"
init_function = "model_init"

model_init = getattr(import_from_path(Path(__file__).parent / model_file_name), init_function)


config = {
    "result_dst": Path(__file__).parent / f'{results_folder_name}',
    "model_init": model_init,
    "load_pattern": ec8_triangular_load_pattern,
    "ctrl_node": 101010400,
    "analysis_parameters": CyclicSpoParameters,
    "displacement_type": "displacement",
    "dU": 1.0,
    "displacements": [],
    'tseries_tag': 2,
    'pattern_tag': 2,
    'excitation_dof': 1,
    
}
