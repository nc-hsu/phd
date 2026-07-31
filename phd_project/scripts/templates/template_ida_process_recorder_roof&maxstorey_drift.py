import openseespy.opensees as ops

from standes.analysis.recorders import get_recorders
from standes.utils import get_z_idx, get_series_id, get_q_point

def process_recorder_func(recorders) -> tuple[float]:
    # this function gets called at the end of each nltha to obtain the EDP value(s) used to plot the ida curves
    # in this case, we want to obtain the roof drift, so we need to get the maximum roof displacement and divide it by the roof height
    return _storey_and_roof_drift(recorders)

edp_tags = ["max_storey_drift", "roof_drift"]
edp_idxs = [0, 1]

def _storey_and_roof_drift(recorders) -> tuple[float]:
    # gets called at the end of each nltha to obtained the EDP value(s) used to plot the ida curves
    # uses the resultant storey drift

    dof = 1 # x-direction
    # roof drift
    # get all node displacement recorders
    displacement_recorders = get_recorders(recorders, "node_displacement")
    
    # get the recorder with the maximum z-value
    main_node_heights = {tag: get_z_idx(tag) for tag in displacement_recorders.keys() 
          if get_series_id(tag) == 1 and get_q_point(tag) == 0} # only consider recorders for roof nodes (series 1) and in the x-direction (q=1)
      
    z_max = max(main_node_heights.values())
    roof_displacement_recorders = {tag: recorder for tag, recorder in displacement_recorders.items() 
                                   if tag in main_node_heights.keys() and main_node_heights[tag] == z_max}

    max_roof_disp = max([max(r.max_values[dof], abs(r.min_values[dof])) for r in roof_displacement_recorders.values()])
    
    roof_node = list(roof_displacement_recorders.keys())[0]
    roof_height = ops.nodeCoord(roof_node)[-1]
    max_roof_drift = max_roof_disp / roof_height

    # drift
    drift_recorders = get_recorders(recorders, "drift")
    max_storey_drift = max([dr.max["res"] for dr in drift_recorders.values()])

    return (max_storey_drift, max_roof_drift,)