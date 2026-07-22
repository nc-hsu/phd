import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import geopandas as gpd
from phd_project.scripts.oqhelpers import (
    get_hc_metadata, get_imtls, get_hcurves_from_dstore,
    get_disagg_from_datastore)


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


def group_disagg_data(groups: list[tuple], metadata, dstore,
                      disagg_type, traditional, occurence=False):
    """
    Groups the dissagregation data stored in an openquake datastore according to
    groups and metadata.
    Returns a dictionary with the following levels
    return_dict[group][site][imt][poe] = pd.DataFrame representing the
    disaggregation results.
    """

    disagg_data = get_disagg_from_datastore(dstore, disagg_type, traditional, occurence)

    grouped_data = {}
    for s, r in groups:
        mask = get_mask(["seismicity", "region"], [s, r], metadata)
        sites = metadata[mask]
        group = {idx : disagg_data[idx] for idx in sites.index}
        grouped_data[(s, r)] = group
    return grouped_data


def _get_disagg_stats(df):

    trt_proportions = df.groupby("TRT")["P(m|X>x)"].sum()
    stats = {f"{i} [%]": np.round(trt_proportions.loc[i] * 100, 2) 
             for i in trt_proportions.index}
    stats["Mag_mean"] = round((df["Mag"] * df["P(m|X>x)"]).sum(),2)
    stats["Dist_mean"] = round((df["Dist"] * df["P(m|X>x)"]).sum(),2)
    # stats["Eps_mean"] = round((df["Eps"] * df["P(m|X>x)"]).sum(),2)
    return stats


def get_disagg_stats_from_groups(groups_disagg, site_metadata, hmaps,
                                 stat_idx=0, geodf:bool=False):
    all_stats = []
    for (s, r), vi in groups_disagg.items():
        for site_id, vj in vi.items():
            for imt_idx, (imt, vk) in enumerate(vj.items()):
                for poe_idx, (poe, vl) in enumerate(vk.items()):
                    stats = {"site_id": site_id,
                             "lat": site_metadata.loc[site_id, "lat"],
                             "lon": site_metadata.loc[site_id, "lon"],
                             "seismicity": s,
                             "region": r,
                             "imt": imt,
                             "poe": poe,
                             "imtl": hmaps[site_id, stat_idx, imt_idx, poe_idx]
                    } 
                    all_stats.append(stats | _get_disagg_stats(vl))

    df = pd.DataFrame.from_records(all_stats)
    if geodf:
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    return df


if __name__ == "__main__":
    ...