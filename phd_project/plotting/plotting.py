import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    

def apply_log_axis_format(ax, axis:str="x"):
    formatter = ticker.FuncFormatter(custom_log_formatter)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    return ax


def apply_grid_lines(ax):
    ax.grid(True, which="both", ls="-.", color="0.8")
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', left=False)


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


def _get_SA_periods_SA_strings(ims: list[str]):
    # Parse Periods from column headers
    pattern = re.compile(r"SA\(([\d.]+)\)")
    periods = np.array([float(pattern.search(im).group(1)) for im in ims if im.startswith("SA")])
    return periods


def plot_conditional_distribution(
        ax: plt.Axes, gcim_cdf: np.ndarray, with_ks_bounds:bool=False,
        take_exp=True,
        style={
            "lb": {"ls": "--", "color":"0.8"},
            "cdf": {"ls": "-", "color":"0.8"},
            "ub": {"ls": "--", "color":"0.8"}
        }):
    
    if take_exp:
        data = np.exp(gcim_cdf)
    else:
        data = gcim_cdf

    if with_ks_bounds:
        ax.plot(data[:,0], data[:,1], **style["lb"])
        ax.plot(data[:,0], data[:,2], **style["cdf"])
        ax.plot(data[:,0], data[:,3], **style["ub"])
    else:
        ax.plot(data[:,0], data[:,1], **style["cdf"])

    ax.set_xlim(0)
    ax.set_ylim(0, 1)
    return ax


def plot_conditional_spectrum(
        ax:plt.Axes, gcim_stats: pd.DataFrame,
        style={
            "p16": {"ls": "--", "color":"0.3"},
            "p50": {"ls": "-", "color":"k"},
            "p84": {"ls": "--", "color":"0.3"}
        }, take_exp: bool=False):
    # get all the ims that are SA
    ims = [im for im in gcim_stats.index if im.startswith("SA")]
    periods = _get_SA_periods_SA_strings(ims)

    if take_exp:
        ax.loglog(periods, np.exp(gcim_stats.loc[ims, "p16"].to_numpy()), **style["p16"])
        ax.loglog(periods, np.exp(gcim_stats.loc[ims, "p50"].to_numpy()), **style["p50"])
        ax.loglog(periods, np.exp(gcim_stats.loc[ims, "p84"].to_numpy()), **style["p84"])
    else:
        ax.loglog(periods, gcim_stats.loc[ims, "p16"].to_numpy(), **style["p16"])
        ax.loglog(periods, gcim_stats.loc[ims, "p50"].to_numpy(), **style["p50"])
        ax.loglog(periods, gcim_stats.loc[ims, "p84"].to_numpy(), **style["p84"])

    return ax