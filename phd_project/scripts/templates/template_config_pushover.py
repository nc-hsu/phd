from pathlib import Path
from standes.analysis.pushover import SpoParameters
from standes.analysis.load_patterns import ec8_triangular_load_pattern
from structural_model import model_init # type: ignore


config = {
    "result_dst": Path(__file__).parent / f'pushover',
    "model_init": model_init,
    "load_pattern": ec8_triangular_load_pattern,
    "ctrl_node": 101010400,
    "U_max": ["drift", 6.0],
    "dU": 0.01,                 # step: same units as U_max
    "analysis_parameters": SpoParameters,
    'tseries_tag': 2,
    'pattern_tag': 2,
    'excitation_dof': 1,
    "allow_negative_load_factor": False
}