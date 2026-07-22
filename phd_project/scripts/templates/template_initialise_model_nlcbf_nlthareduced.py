import openseespy.opensees as ops
from pathlib import Path

from standes.opsmodels.nlcbf_3d_01 import nltha_recorders
from standes.analysis.recorders import (get_recorders, Recorder, DriftLimitStateRecorder3D,
                                        ZeroLengthForceLimitStateRecorder, RemoveElementRecorder)
from standes.geometries.bolts import bolt_group_from_dict
from standes.verification.bolts import bolt_group_concentric_block_failure_capacity_y
from standes import utils, get, config
from standes.utils import import_from_path, get_series_id, get_q_point

# =============================================================================
# initialise_model -- REDUCED recorder set (for IDA / MSA)
# =============================================================================
# Identical to `template_initialise_model_nlcbf_nltha.py` except that, after the
# full recorder list has been built, `generate_recorders` passes it through
# `_reduce_recorders(...)`: a pure SUBTRACTION that keeps only the recorders
# needed to detect collapse or to compute the IDA/MSA EDP (roof / storey drift),
# plus the storey-line accelerations. On a 3-storey CBF the kept recorders are
# ~3% of the samples, i.e. an expected ~97% reduction in the recorder-pickle size.
#
# The model itself (design file, ops config, damping) lives in
# `config_structural_model.py`, imported here BY PATH so its filename can be
# swapped via `config_structural_model_file_name`; the model config used is pinned
# by `model_config_name`.
#
# NOTE: `add_limit_state_recorders` below is shared verbatim with
# `template_initialise_model_nlcbf_nltha.py`. If you change the recorder-
# construction logic, edit BOTH templates.
# =============================================================================


config_structural_model_file_name = "config_structural_model.py"
model_config_name = "ops_model_config"

_sm = import_from_path(Path(__file__).parent / config_structural_model_file_name)
structure_data = _sm.structure_data
model_config = getattr(_sm, model_config_name)      # the single model config this file is pinned to
damping_config = _sm.damping_config                 # re-exported for make_injection_functions
damping_model = _sm.damping_model                   # re-exported for make_injection_functions
build_model = _sm.build_model

recorder_config = {
    "drift_limit": 0.2,
    "incl_remove_recorders": True,
}


# function for initialising the model and recorders at the start of each nltha
def initialise_model() -> tuple[list[Recorder], list[Recorder]]:
    recorder_tag_lists = build_model(structure_data, **model_config)

    # apply damping
    damping_model(damping_config)

    # define recorders and extract the collapse (drift limit state) recorders
    return generate_recorders(recorder_tag_lists)


model_init = initialise_model   # zero-arg callable, returns (recorders, collapse_recorders)


def generate_recorders(recorder_tag_lists: dict) -> tuple[list[Recorder], list[Recorder]]:
    # build the full recorder set exactly as the original monolithic template did
    recorders = nltha_recorders(recorder_tag_lists, acc_dofs=[1], load_pattern_tags=[2])
    add_limit_state_recorders(recorder_tag_lists, recorders, structure_data,
                          model_config["n_brace_elements"],
                          recorder_config["drift_limit"], recorder_config["incl_remove_recorders"])

    # ---- SUBTRACTION STEP: keep only the recorders needed for collapse + EDP ----
    recorders = _reduce_recorders(recorders, recorder_tag_lists)

    collapse_recorders = [r for r in get_recorders(recorders, "drift_limit_state").values()]

    return recorders, collapse_recorders


def _reduce_recorders(recorders: list[Recorder], recorder_tag_lists: dict) -> list[Recorder]:
    """Subtract every recorder that is not required for collapse detection or for the
    IDA/MSA EDPs (roof / storey drift), returning the reduced list.

    This is a pure filter over the list already produced by `nltha_recorders` +
    `add_limit_state_recorders`; it does not create, alter or reorder any recorder.

    KEPT
    ----
    - "drift" (DriftRecorder3D)
        Storey drifts. Also the underlying recorders wrapped by the collapse
        (drift_limit_state) recorders, so they must stay in the list and keep
        being updated.
    - "node_displacement" of the storey-line nodes (series id 1, q_point 0)
        These are the only recorders the roof-drift / x-displacement process
        functions read (they select the top storey node). Also give storey
        displacements. Mid-span brace/column node displacements are excluded by
        the series/q_point test.
    - gusset "zerolength_force"
        Drive the gusset moment / tear-out / bolt-shear limit states and hence
        the RemoveElementRecorder that physically removes failed braces+gussets
        from the analysis. Dropping them would change the modelled behaviour.
    - all limit-state / element-removal recorders ("drift_limit_state",
      "zerolength_force_limit_state", "multi_limit_state", "remove_element")
        Needed for collapse detection and brace/gusset removal. They store no
        time history, so they cost effectively nothing.
    - "node_acceleration" of the storey-line nodes (series id 1, q_point 0)
        Not needed for the drift EDP, but kept so peak floor accelerations are
        available for acceleration-based fragility curves later. The full time
        history and max/min are retained.

    DROPPED (full per-step time histories, ~97% of the data)
    --------------------------------------------------------
    - "section_force" / "section_deformation"  (columns + braces)
    - "BeamColumn_local_force"                  (columns + beams)
    - "BeamColumn_global_force" + "storey_shear"
    - splice/bcj "zerolength_force", all "zerolength_deformation"
    - mid-span brace/column "node_displacement"

    None of the dropped recorders is read to decide collapse or to compute the
    roof/storey-drift EDP. The RemoveElementRecorder keeps its own references to
    any dropped recorders it needs to disable on element removal, so removing
    them from this list is safe; and the section/node recorders on removed
    elements self-disable via their own try/except in `update()`.
    """
    gusset_tags = set(recorder_tag_lists["gusset_elements"])
    keep_limit_state_types = {
        "drift_limit_state",
        "zerolength_force_limit_state",
        "multi_limit_state",
        "remove_elements",     # RemoveElementRecorder._type is plural; must be kept to preserve collapse behaviour
    }

    reduced: list[Recorder] = []
    for r in recorders:
        rtype = r.recorder_type

        if rtype == "drift":
            reduced.append(r)

        elif rtype in keep_limit_state_types:
            reduced.append(r)

        elif rtype == "zerolength_force" and r.tag in gusset_tags:
            reduced.append(r)

        elif rtype == "node_displacement" and get_series_id(r.tag) == 1 and get_q_point(r.tag) == 0:
            reduced.append(r)

        elif rtype == "node_acceleration" and get_series_id(r.tag) == 1 and get_q_point(r.tag) == 0:
            reduced.append(r)

        # everything else is subtracted (not appended)

    return reduced


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
