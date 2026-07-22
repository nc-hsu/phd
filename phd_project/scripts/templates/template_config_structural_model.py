import json
from pathlib import Path

from standes.opsmodels.nlcbf_3d_01 import build_model_nlcbf_3d_01
from standes.analysis.damping import set_modal_damping

# =============================================================================
# Structural-model config
# =============================================================================
# Defines *what the model is*: the design file it is built from, the OpenSees
# model configuration(s), the damping configuration, and thin wrappers around the
# standes model-building calls.
#
# It carries no recorder logic and no `model_init` -- that lives in
# `initialise_model.py`, which imports this module by path (see
# `config_structural_model_file_name` there) and selects which of the model
# configs below to use (see `model_config_name` there).
# =============================================================================


design_json = "3s_cbf_dc2_41_out.json"

ROOT = Path(__file__).parent
with open(ROOT / design_json, "r") as file: # assumes the design file is in the same folder
    structure_data = json.load(file)["structure"]

# configuration parameters for the model
ops_model_config: dict = {
    "gravity_analysis": True,
    "print_output": True,
    "mass_dofs": [1,3],
    "z_mass_reduction": 1e-4,           # used to reduce the vertical mass so that the horizontal modeshapes all have the longest periods
    "leaning_column_transf": "Corotational",
    "gravity_ts_pat_tag": 1,
    "brace_material": "Fatigue",
    "brace_oos": 0.001,
    "brace_oos_dof": 2,
    "brace_oos_shape": "parabolic",
    "k_brace": 1,
    "brace_element_type": "dispBeamColumn",
    "n_brace_elements": 8,
    "brace_transf": "Corotational",

    "gusset_stiffness": "normal",
    "gusset_material": "Steel02",
    "gusset_nonlinearity": "nonlinear",
    "lock_gusset_nodes": False,

    "column_material": "Fatigue",
    "column_oos": 0.001,
    "column_oos_dof": 2,
    "column_oos_shape": "parabolic",
    "k_column": 1,
    "column_element_type": "dispBeamColumn",
    "n_column_elements": 8,
    "column_transf": "Corotational",

    "beam_material": "Elastic",
    "beam_nonlinearity": "linear",
    "beam_transf": "Linear",

    "splice_stiffness": "normal",
    "splice_material": "IMKPinching",
    "splice_nonlinearity": "nonlinear",
    "lock_splice_nodes": False,
    "splice_theta_ult": 0.15,

    "bcj_stiffness": "normal",
    "bcj_material": "IMKPinching",
    "bcj_nonlinearity": "nonlinear",
    "lock_bcj_nodes": False,
    "bcj_theta_ult": 0.15,
}

ops_model_config_no_G = ops_model_config.copy()
ops_model_config_no_G["gravity_analysis"] = False

damping_config = {  # dependent on the damping model that is being used
    "n_modes": 1,
    "damping_ratio": 0.05,
    "fullgen": False,
}


def damping_model(damping_config: dict):
    set_modal_damping(**damping_config)


def build_model(structure_data: dict, **model_config):
    # thin wrapper around the standes model builder; returns recorder_tag_lists
    return build_model_nlcbf_3d_01(structure_data, **model_config)
