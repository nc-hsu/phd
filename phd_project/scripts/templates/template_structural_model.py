import json
from pathlib import Path
from functools import partial

import numpy as np
import openseespy.opensees as ops

from standes.opsmodels.nlcbf_3d_01 import build_model_nlcbf_3d_01, nltha_recorders
from standes.analysis.recorders import (get_recorders, Recorder, DriftLimitStateRecorder3D, 
                                        ZeroLengthForceLimitStateRecorder, RemoveElementRecorder)
from standes.analysis.damping import set_modal_damping
from standes.geometries.bolts import bolt_group_from_dict
from standes.verification.bolts import bolt_group_concentric_block_failure_capacity_y
from standes import utils, get, config 


design_json = "3s_cbf_dc2_41_out.json"

ROOT = Path(__file__).parent
with open(ROOT / design_json, "r") as file: # assumes the design file is in the same folder
    structure_data = json.load(file)["structure"]

# configuration parameters for the model
ops_model_config: dict = {
    "gravity_analysis": True,
    "print_output": True,
    "mass_dofs": [1,3],
    "z_mass_reduction": 1e-4,
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
    "bcj_theta_ult": 0.15
}

ops_model_config_no_G = ops_model_config
ops_model_config_no_G["gravity_analysis"] = False

recorder_config = {
    "drift_limit": 0.2,
    "incl_remove_recorders": False
}

damping_config = {  # dependent on the damping model that is being used
    "n_modes": 1,
    "damping_ratio": 0.05,
    "fullgen": False
}
  

def damping_model(damping_config: dict):
    set_modal_damping(**damping_config)


# function for initialising the model and recorders at the start of each nltha
def initialise_model(structure_data: dict, model_config: dict, recorder_config: dict, 
                     damping_model, damping_config: dict 
                     ) -> tuple[list[Recorder], list[Recorder]]:
    drift_limit = recorder_config["drift_limit"]
    recorder_tag_lists = build_model_nlcbf_3d_01(structure_data, **model_config)

    # apply damping
    damping_model(damping_config)

    # define recorders
    recorders = nltha_recorders(recorder_tag_lists, acc_dofs=[1], load_pattern_tags=[2])
    add_limit_state_recorders(recorder_tag_lists, recorders, structure_data, ops_model_config["n_brace_elements"], 
                          drift_limit, recorder_config["incl_remove_recorders"])
    collapse_recorders = [r for r in get_recorders(recorders, "drift_limit_state").values()]
    
    return recorders, collapse_recorders

model_init = partial(initialise_model, structure_data, ops_model_config, recorder_config, 
                     damping_model, damping_config)

model_init_no_g = partial(initialise_model, structure_data, ops_model_config_no_G, recorder_config,
                          damping_model, damping_config)


def add_limit_state_recorders(recorder_tag_lists: dict, recorders: list[Recorder], structure_data: dict, 
                          n_brace_elements: int, drift_limit: float, incl_remove_recorders: bool,
                          F_limit:float|None=None):
    # create limit state recorders for drifts and the gusset bolts moment/axial capacity
    gusset_tags = recorder_tag_lists["gusset_elements"]
    drift_recorders = list(get_recorders(recorders, "drift").values())
    gusset_f_recorders = {k: r for k, r in get_recorders(recorders, "zerolength_force").items()
                        if k in gusset_tags}
    gusset_d_recorders = {k: r for k, r in get_recorders(recorders, "zerolength_deformation").items()
                        if k in gusset_tags}
    section_force_recorders = get_recorders(recorders, "section_force")
    section_deformation_recorders = get_recorders(recorders, "section_deformation")
    global_force_recorders = get_recorders(recorders, "BeamColumn_global_force")
    mid_node_recorders = get_recorders(recorders, "node_displacement")

    ### drifts
    drift_ls_recorders = [DriftLimitStateRecorder3D(r, 1, drift_limit, "gt", absolute_value=True) for r in drift_recorders]    # 10% absolute drift

    ### gussets 
    gusset_moment_ls_recorders = []
    gusset_tearout_ls_recorders = []
    gusset_boltshear_ls_recorders = []
    gusset_remove_recorders = []

    omega_rm = get.material_randomness_factor(structure_data["material"]["fy"])
    f_y = omega_rm * structure_data["material"]["fy"]
    f_u = omega_rm * get.ultimate_steel_strength(structure_data["material"]["fy"])

    for tag in gusset_tags:
        x_idx = utils.get_x_idx(tag)
        y_idx = utils.get_y_idx(tag)
        z_idx = utils.get_z_idx(tag)
        q_point = utils.get_q_point(tag)

        gusset_f_recorder = gusset_f_recorders[tag]
        
        gusset_dict = [g for g in structure_data["gussets"] 
                    if g["grid"] == x_idx and g["level"] == z_idx and g["q_point"] == q_point][0]
        bolt_dict = gusset_dict["brace_connection"]
        bolt_group = bolt_group_from_dict(bolt_dict)

        # get the opposite gusset tag, the connected brace tag and the associated node tags
        if q_point == 2:
            brace_tag = utils.generate_type_1_tag(config.BR_ID, x_idx, y_idx, z_idx, 0, 0)
            opposite_gusset_tag = utils.generate_type_1_tag(config.GUS_ID, x_idx+1, y_idx, z_idx+1, 6, 0)
        elif q_point == 4:
            brace_tag = utils.generate_type_1_tag(config.BR_ID, x_idx, y_idx, z_idx-1, 0, 0)
            opposite_gusset_tag = utils.generate_type_1_tag(config.GUS_ID, x_idx+1, y_idx, z_idx-1, 8, 0)
        elif q_point == 6:
            brace_tag = utils.generate_type_1_tag(config.BR_ID, x_idx-1, y_idx, z_idx-1, 0, 0)  
            opposite_gusset_tag = utils.generate_type_1_tag(config.GUS_ID, x_idx-1, y_idx, z_idx-1, 2, 0)
        elif q_point == 8:
            brace_tag = utils.generate_type_1_tag(config.BR_ID, x_idx-1, y_idx, z_idx, 0, 0)
            opposite_gusset_tag = utils.generate_type_1_tag(config.GUS_ID, x_idx-1, y_idx, z_idx+1, 4, 0)

        brace_ele_tags = [brace_tag + (ii+1) for ii in range(n_brace_elements)]
        brace_nodes = list(set([n for br in brace_ele_tags for n in ops.eleNodes(br)]))

        # moment
        M_rk = bolt_group.moment_capacity 
        dof = 5
        ls_recorder_tag = f"{gusset_f_recorder.tag}_{dof}_boltmoment_{M_rk:.0f}"
        moment_ls_recorder = ZeroLengthForceLimitStateRecorder(gusset_f_recorder, dof, M_rk, "gt", absolute_value=True, tag=ls_recorder_tag)
        gusset_moment_ls_recorders.append(moment_ls_recorder)

        # tearout
        if F_limit == None:
            F_tearout = bolt_group_concentric_block_failure_capacity_y(bolt_group, gusset_dict["thickness"], f_y, f_u)
        else:
            F_tearout = F_limit
        
        dof = 1
        ls_recorder_tag = f"{gusset_f_recorder.tag}_{dof}_tearout_{F_tearout:.0f}"
        F_tearout_ls_recorder = ZeroLengthForceLimitStateRecorder(gusset_f_recorder, dof, F_tearout, "gt", tag=ls_recorder_tag)  # gt because tension is +ve for a section recorder
        gusset_tearout_ls_recorders.append(F_tearout_ls_recorder)
        
        # bolt shear
        if F_limit == None:
            F_boltshear = sum([bolt.F_v_Rk for bolt in bolt_group.bolts])
        else:
            F_boltshear = F_limit
        dof = 1
        ls_recorder_tag = f"{gusset_f_recorder.tag}_{dof}_boltshear_{F_boltshear:.0f}"
        F_boltshear_ls_recorder = ZeroLengthForceLimitStateRecorder(gusset_f_recorder, dof, F_boltshear, "gt", absolute_value=True, tag=ls_recorder_tag)  # gt because tension is +ve for a section recorder
        gusset_boltshear_ls_recorders.append(F_boltshear_ls_recorder)

        # get all the recorders that will be affected by the removal of these elements
        brace_sf_recorders = [v for k,v in section_force_recorders.items() if str(brace_tag)[:8] == k[:8]]
        brace_sd_recorders = [v for k,v in section_deformation_recorders.items() if str(brace_tag)[:8] == k[:8]]
        brace_gf_recorders = [v for k,v in global_force_recorders.items() if str(brace_tag)[:8] == str(k)[:8]]
        opposite_gusset_f_recorder = [gusset_f_recorders[opposite_gusset_tag]]
        opposite_gusset_d_recorder = [gusset_d_recorders[opposite_gusset_tag]]
        mid_node_recorder = [v for k,v in mid_node_recorders.items() if str(brace_tag)[:8] == str(k)[:8]]

        other_recorders = brace_sf_recorders + brace_sd_recorders + brace_gf_recorders + opposite_gusset_f_recorder + opposite_gusset_d_recorder + mid_node_recorder

        # create the remove element recorders
        elements_to_remove = [tag] + [opposite_gusset_tag] + brace_ele_tags
        nodes_to_remove = brace_nodes

        remove_recorder = RemoveElementRecorder(elements_to_remove, [moment_ls_recorder, F_tearout_ls_recorder, F_boltshear_ls_recorder], 
                                                node_tags=nodes_to_remove, other_recorders=other_recorders)
        gusset_remove_recorders.append(remove_recorder)

    
    recorders.extend(drift_ls_recorders)
    recorders.extend(gusset_moment_ls_recorders)
    recorders.extend(gusset_tearout_ls_recorders)
    recorders.extend(gusset_boltshear_ls_recorders)

    if incl_remove_recorders:
        recorders.extend(gusset_remove_recorders)

    return
