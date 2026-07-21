from pathlib import Path
from standes.analysis.nltha import NlthaParameters
from standes.utils import import_from_path

"""
No comments can be placeed inside the config ditionary as this messing with the
copy and edit functions that are used to create new config files from this template.

Place all comments here:
--------------------------
- results_folder_name should be the first variable in this file, as the copy and
edit functions expect this when creating new config files from this template.

- model_file_name and init_function follow results_folder_name and name the structural
model file (in this folder) and the model-building callable imported from it. The damping
model pieces are read from the same model file.

- damping_config: configuration of damping model needed for updating modal damping model in nltha
- damping_model: is a function that sets the damping model in opensees, needed for updating modal damping model in nltha
"""

results_folder_name = ""
model_file_name = "structural_model.py"
init_function = "model_init"
gm_json_src_str = 'E:/gm_records_p695'

_model = import_from_path(Path(__file__).parent / model_file_name)
model_init = getattr(_model, init_function)
damping_config = _model.damping_config
damping_model = _model.damping_model


config = {
    "result_dst": Path(__file__).parent / f"{results_folder_name}",
    "model_init": model_init,
    "damping_config": damping_config,
    "damping_model": damping_model,
    'gm_json_src': Path(gm_json_src_str),
    'gm_json_file': 'fema_p695_120111.json',
    "analysis_parameters": NlthaParameters(),
    'tseries_tag': 2,
    'pattern_tag': 2,
    'excitation_dof': 1,
    "scale_factor": 1.667,
    "max_record_time": None,
    "gravity_factor": 9810

}