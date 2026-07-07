from pathlib import Path
from structural_model import model_init # type: ignore


""" 
No comments or spaces can be placeed inside the config ditionary after the last comma
as this messes with the
copy and edit functions that are used to create new config files from this template.

Place all comments here:
--------------------------
results_folder_name should be the first variable in this file, as the copy and 
edit functions expect this when creating new config files from this template.

"""
results_folder_name = ""


config = {
    "result_dst": Path(__file__).parent / f'{results_folder_name}',
    "model_init": model_init,
    "n_modes": 1,
    "solver":"-genBandArpack",
}