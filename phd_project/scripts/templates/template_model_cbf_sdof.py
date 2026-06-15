from functools import partial

import openseespy.opensees as ops

from standes.analysis.recorders import (
    get_recorders, Recorder, 
    NodeDisplacementRecorder3D,
    NodeAccelerationRecorder3D,
    ZeroLengthForceRecorder, 
    DisplacementLimitStateRecorder3D
    )

from standes.analysis.damping import set_modal_damping


# configuration parameters for the model
ops_model_config: dict = {
    "steel02_bb_params": [132304.19749064848, 1909.185674440267, -0.3246827261343593, 213.43531141947088],
    "steel02_hyst_params": [0.058899073529551434, 1.899472066814619, 18.194887568580384, 0.7678689477226374, 0.5477645899254575],
    "hysteretic_bb_params": [1239408.3424872202, 24.89779616304721, 1116064.0455594708, 49.48498722953466, 0.0, 98.90894610047322, -1239408.3424872202, -24.89779616304721, -1116064.0455594708, -49.48498722953466, -0.0, -98.90894610047322],
    "hysteretic_hyst_params": [0.2154939503107748, 0.4805287173966139, 0.15992730604422367, 2.5535246597796033e-09, 0.393951285805807],
    "mass": 199.69,
}

recorder_config = {
    "disp_limit": 700,
}

# dependent on the damping model that is being used
damping_config = {  
    "n_modes": 1,
    "damping_ratio": 0.05,
    "fullgen": True,
}
  
def damping_model(damping_config: dict):
    set_modal_damping(**damping_config)


def _build_steel02_and_hysteretic_sdof_model(
        steel02_bb_params, 
        steel02_hyst_params, 
        hysteretic_bb_params, 
        hysteretic_hyst_params,
        mass 
        ):
    
    Fy, E0, b, du = steel02_bb_params
    a1, a2, R0, cR1, cR2 = steel02_hyst_params
    s1p, e1p, s2p, e2p, s3p, e3p, s1n, e1n, s2n, e2n, s3n, e3n = hysteretic_bb_params
    pinch_x, pinch_y, damage_1, damage_2, beta = hysteretic_hyst_params


    # Model parameters
    support_node: int = 1
    free_node: int = 2
    ele_tags = [1, 2]

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    
    ops.node(support_node, 0, 0)
    ops.node(free_node, 0, 0)
    
    ops.mass(free_node, mass, 0, 0)

    ops.fix(support_node, 1, 1, 1)
    ops.fix(free_node, 0, 1, 1)

    # spring 1 - frame
    a3 = a1
    a4 = a2

    transition_params = [R0, cR1, cR2]

    ops.uniaxialMaterial("Steel02", 1, Fy, E0, b, *transition_params, a1, a2, a3, a4)
    ops.element("zeroLength", ele_tags[0], support_node, free_node, 
                "-mat", 1, "-dir", 1)
      
    # spring 2 - braces
    ops.uniaxialMaterial("Hysteretic", 2, 
                         s1p, e1p, s2p, e2p, s3p, e3p, 
                         s1n, e1n, s2n, e2n, s3n, e3n, 
                         pinch_x, pinch_y, damage_1, damage_2, beta)
    ops.element("zeroLength", ele_tags[1], support_node, free_node, 
                "-mat", 2, "-dir", 1)
    
    return free_node, support_node, ele_tags


# function for initialising the model and recorders at the start of each nltha
def initialise_model(model_config: dict, recorder_config: dict, 
                     damping_model, damping_config: dict 
                     ) -> tuple[list[Recorder], list[Recorder]]:
    
    free_node, _, ele_tags = _build_steel02_and_hysteretic_sdof_model(**model_config)

    # apply damping
    damping_model(damping_config)

    # define recorders
    recorders = [
        NodeDisplacementRecorder3D(free_node),
        ZeroLengthForceRecorder(ele_tags[0], dof_tags=[1]),
        ZeroLengthForceRecorder(ele_tags[1], dof_tags=[1]),  
    ]
    
    disp_limit = recorder_config["disp_limit"]
    ls_recorders = [
        DisplacementLimitStateRecorder3D(
        recorders[0], dof=1, limit=disp_limit, 
        comparison="gt", absolute_value=True)]
    
    recorders.extend(ls_recorders)

    collapse_recorders = [r for r in get_recorders(recorders, "node_displacement_limit_state").values()]
    
    return recorders, collapse_recorders

model_init = partial(initialise_model, ops_model_config, recorder_config, 
                     damping_model, damping_config)


