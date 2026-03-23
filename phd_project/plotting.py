import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def custom_log_formatter(x, pos):
    """
    Formats the tick label:
    - If x < 1, display as a decimal (e.g., 0.1, 0.01)
    - If x >= 1, display as an integer (e.g., 1, 10, 100)
    """
    # Check if the number is less than 1 (or 0.99999... for float safety)
    if x < 1.0:
        # Use a general format for floating point numbers
        # The .2f or similar might be too restrictive, so we check for magnitude.
        # Format based on the magnitude to avoid excessive trailing zeros.
        if x < 0.001:
            # If very small, revert to scientific notation for readability
            return f"{x:.0e}"
        else:
            # Use general float format, stripping insignificant zeros
            return f"{x:g}"
    else:
        # Use integer format for values >= 1
        return f"{int(round(x))}"
    

def plot_im_ecdf_vs_target(
        ax, im_ecdf: np.ndarray, ks_bounds: np.ndarray,
        style={
            "lb": {"ls": "--", "color":"0.8"},
            "cdf": {"ls": "-", "color":"0.8"},
            "ub": {"ls": "--", "color":"0.8"},
            "ecdf": {"ls": "-", "color":"b"},
        }):
    
    ax.plot(ks_bounds[:,0], ks_bounds[:,1], **style["lb"])
    ax.plot(ks_bounds[:,0], ks_bounds[:,2], **style["cdf"])
    ax.plot(ks_bounds[:,0], ks_bounds[:,3], **style["ub"])
    ax.plot(im_ecdf[:,0], im_ecdf[:,1], **style["ecdf"])

    ax.set_xlim(0, 1.1*max(im_ecdf[:,0]))
    ax.set_ylim(0, 1)
    return ax