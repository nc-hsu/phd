from pathlib import Path
from standes.analysis.nltha import NlthaParameters
from structural_model import model_init # type: ignore

config = {
    "result_dst": Path(__file__).parent / f'nltha_120111_sf_1667',
    "model_init": model_init,
    'gm_json_src': Path('E:/gm_records_p695'),
    'gm_json_file': 'fema_p695_120111.json',
    "analysis_parameters": NlthaParameters(),
    'tseries_tag': 2,
    'pattern_tag': 2,
    'excitation_dof': 1,
    "scale_factor": 1.667,
    "max_record_time": None,
    "gravity_factor": 9810

}