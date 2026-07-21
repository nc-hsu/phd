from pathlib import Path
from standes.analysis.nltha import NlthaParameters
from standes.analysis.gravity import NonlinearParameters
from standes.utils import import_from_path

# model_file_name and init_function name the structural model file (in this folder)
# and the model-building callable imported from it.
model_file_name = "structural_model.py"
init_function = "model_init"

model_init = getattr(import_from_path(Path(__file__).parent / model_file_name), init_function)

config = {
    "result_dst": Path(__file__).parent / f'snapback',
    "model_init": model_init,
    "F_0": 25e3,
    "dof": 1,
    "ctrl_node": 101010400,
    "roof_nodes": [101010400, 102010400, 103010400, 104010400],
    "tseries_tag": 2,
    "pattern_tag": 2,
    "t_final": 10,
    "static_analysis_parameters": NonlinearParameters(algorithm = ("KrylovNewton",)),
    "dynamic_analysis_parameters": NlthaParameters(dt=0.01)
}