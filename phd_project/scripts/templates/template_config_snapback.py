from pathlib import Path
from standes.analysis.nltha import NlthaParameters
from standes.analysis.gravity import NonlinearParameters
from structural_model import model_init # type: ignore

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