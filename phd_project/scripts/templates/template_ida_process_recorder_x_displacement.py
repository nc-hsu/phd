import openseespy.opensees as ops

from standes.analysis.recorders import get_recorders
from standes.utils import get_z_idx, get_series_id, get_q_point

def process_recorder_func(recorders) -> tuple[float]:
    # this function gets called at the end of each nltha to obtain the EDP value(s) used to plot the ida curves
    # in this case, we want to obtain the roof displacement
    return _x_displacement(recorders)

edp_tags = ["roof_displacement"]
edp_idxs = [0]

def _x_displacement(recorders) -> tuple[float]:
    # gets called at the end of each nltha to obtained the EDP value(s) used to plot the ida curves
    
    dof = 1 # x-direction
    displacement_recorders = get_recorders(recorders, "node_displacement")
    
    max_disp = max([max(r.max_values[dof], abs(r.min_values[dof])) for r in displacement_recorders.values()])
    
    return (max_disp,)