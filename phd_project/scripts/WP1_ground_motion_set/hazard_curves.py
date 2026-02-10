import numpy as np
import pandas as pd
from phd_project.scripts.oqhelpers import (
    get_hc_metadata, get_imtls, get_hcurves_from_dstore)


def group_hazard_curves(groups: list[tuple], metadata, dstore,
                        calculate_mean_curves:bool=True):

    """
    Groups the hazard curves stored in an openquake datastore according to
    groups and metadata.
    The mean curves for each groupd are calculated
    Returns a dictionary with the following levels
    return_dict[group][site][imt][stat] = hazard curve np.ndarray shape (:, 2)
    """

    hc_metadata = get_hc_metadata(dstore)
    imtls = get_imtls(dstore)
    hcs = get_hcurves_from_dstore(dstore)

    grouped_hcs = {}
    for s, r in groups:
        mask = get_mask(["seismicity", "region"], [s, r], metadata)
        sites = metadata[mask]
        
        group_hcs = {idx : hcs[idx] for idx in sites.index}
        temp = {}

        if not calculate_mean_curves:
            grouped_hcs[(s, r)] = group_hcs
            continue

        for imt in imtls.keys():
            temp[imt] = {}
            for stat in hc_metadata["stat"]:
                mean_curve = np.mean(
                    np.vstack([shc[imt][stat][:,1] 
                               for shc in group_hcs.values()]), axis=0)
                temp[imt][stat] = np.vstack([np.array(imtls[imt]),
                                                    mean_curve]).T

        group_hcs["mean_site"] = temp
        grouped_hcs[(s, r)] = group_hcs
    return grouped_hcs
        


def get_mask(keys: list, targets: list, df: pd.DataFrame):
    """
    Returns a mask where all specified columns match 
    their respective target values.

    param: cols list: names of columns being considered
    param: targets list: list of target values for each category
    param: df pd.Dataframe: the df we create the mask from
    """
    # Create a list of boolean Series for each condition
    conditions = [df[c] == t for c, t in zip(keys, targets)]
    
    # Use reduce or concat to find where all conditions are True
    mask = pd.concat(conditions, axis=1).all(axis=1)
    
    return mask



    ...