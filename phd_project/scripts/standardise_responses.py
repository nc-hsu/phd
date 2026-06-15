import numpy as np

def standardise_responses(curve_a, curve_b):
    """
    Harmonises two hysteresis curves using the intersection of 
    rounded cumulative displacement values.
    """
    def get_rounded_cum_disp(curve):
        # Round displacement to 3 decimal places as requested
        d_rounded = np.round(curve[:, 0], 3)
        delta_d = np.abs(np.diff(d_rounded))
        # Use a small tolerance or rounding on the sum to avoid 
        # floating point drift during intersection
        s = np.concatenate(([0], np.cumsum(delta_d)))
        return np.round(s, 6), d_rounded, curve[:, 1]

    # Process both curves
    s_a, d_a, f_a = get_rounded_cum_disp(curve_a)
    s_b, d_b, f_b = get_rounded_cum_disp(curve_b)
    
    # Find the intersection of cumulative displacement points
    # This automatically limits the range to the shorter path
    s_union = np.union1d(s_a, s_b)
    
    # limit s_union to values less than the maximum of either curve
    mask = s_union <= min(s_a[-1], s_b[-1])
    s_union = s_union[mask]
    
    # Interpolate Force and Displacement for both curves onto s_common
    # Using np.interp ensures values are found even if indices vary
    res_a = np.column_stack((
        np.interp(s_union, s_a, d_a),
        np.interp(s_union, s_a, f_a)
    ))

    res_b = np.column_stack((
        np.interp(s_union, s_b, d_b),
        np.interp(s_union, s_b, f_b)
    ))

    return res_a, res_b, s_union