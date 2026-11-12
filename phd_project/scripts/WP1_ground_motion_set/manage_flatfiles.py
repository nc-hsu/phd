# -*- coding: utf-8 -*-
"""
Module for obtaining the flatfiles for each tectonic regime in ESHM20 and 
populating any required data that is missing.
"""
import re
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

import pickagm.eqdbases as eqdb
import pickagm.dfops as dfops
from pickagm.avgSA import compute_ln_AvgSA

from openquake.hazardlib.imt import IMT
from phd_project.config.config import load_config


cfg = load_config() 

flatfile_folder = cfg["raw_data"]["gm_flatfiles"]
esm_flatfile_fp = flatfile_folder / "ESM_flatfile_SA.csv"
site_model_fp = cfg["hazard_models"]["eshm20_site_model"]

def load_esm_flatfile(file_path) -> pd.DataFrame:
    # load all columns as strings to prevent any unintended formatting changes
    # e.g. station code 0703 being loaded as int 703...
    # the numerical columns can be converted to floats ad hoc
    return pd.read_csv(file_path, sep=";", dtype=str)


def dataset_is_subset_of_esm(database: dict, esm: pd.DataFrame):

    df_B = _convert_dict_to_df(database)
    df_B = df_B.astype(str)
    esm["sensor_depth_m"] = esm["sensor_depth_m"].astype(np.float64)
    df_B["sensor_depth_m"] = df_B["sensor_depth_m"].astype(np.float64)
    keys = ["event_id", "network_code", "station_code", "sensor_depth_m"]

    # perform a "merge" based on the the three columns of "keys".
    # Left join from B to A. All the rows in B are retained.
    # If the indicator is "both" for all rows then the same rows appear in
    # both DataFrames.
    check_merge = df_B[keys].merge(
        esm[keys], on=keys, how="left", indicator=True)
    
    return (check_merge["_merge"] == "both").all()


def rows_not_in_esm(database: dict, esm: pd.DataFrame):

    df_B = _convert_dict_to_df(database)
    df_B = df_B.astype(str)
    esm["sensor_depth_m"] = esm["sensor_depth_m"].astype(np.float64)
    df_B["sensor_depth_m"] = df_B["sensor_depth_m"].astype(np.float64)
    keys = ["event_id", "network_code", "station_code", "sensor_depth_m"]

    # perform a "merge" based on the the three columns of "keys".
    # Left join from B to A. All the rows in B are retained.
    # If the indicator is "both" for all rows then the same rows appear in
    # both DataFrames.
    check_merge = df_B[keys].merge(
        esm[keys], on=keys, how="left", indicator=True)
    
    idxs = check_merge[check_merge["_merge"] == "both"].index
    return idxs


def _convert_dict_to_df(database: dict):
    event_id = [r["event"]["id"] for r in database["records"]]
    network_code = [r["site"]["network_code"] for r in database["records"]]
    station_code = [r["site"]["code"] for r in database["records"]]
    sensor_depth = [r["site"]["sensor_depth"] for r in database["records"]]

    df = pd.DataFrame.from_dict({"event_id": event_id,
                                 "network_code": network_code,
                                 "station_code": station_code,
                                 "sensor_depth_m": sensor_depth},
                                 orient="columns")
    return df


def extract_flatfile_for_dataset(database: dict, esm: pd.DataFrame):
    df_B = _convert_dict_to_df(database)
    df_B = df_B.astype(str)
    esm["sensor_depth_m"] = esm["sensor_depth_m"].astype(np.float64)
    df_B["sensor_depth_m"] = df_B["sensor_depth_m"].astype(np.float64)
    keys = ["event_id", "network_code", "station_code", "sensor_depth_m"]

    # perform a "merge" based on the the four columns of "keys".
    # inner join from A to B. All the rows from A are retained that are
    # also in B.
    new_df = esm.merge(
        df_B[keys], on=keys, how="inner")

    return new_df


# ---------------------------------------------------------------------------------------------
# Regional datasets defined by ESM record ID lists
# ---------------------------------------------------------------------------------------------
# The ESHM20 subduction (in-slab, interface) and Vrancea datasets are subsets of the ESM
# flatfile. Each record in them is identified in ESM by the four columns below, the same
# columns the JSON-based functions above match on. Publishing a list of these four values per
# dataset is therefore enough to rebuild each regional flatfile from the ESM flatfile.

# The columns that identify one ESM record (one event recorded by one sensor). The sensor
# depth is needed because some stations have several sensors (e.g. surface and borehole).
ESM_RECORD_KEYS = ["event_id", "network_code", "station_code", "sensor_depth_m"]


def record_ids_from_dataset(database: dict) -> pd.DataFrame:
    """The ESM record IDs of a dataset in the GFZ JSON format, one row per record.

    Uses the same conversion as ``extract_flatfile_for_dataset``: the three code columns as
    strings and the sensor depth as float64 (a missing depth becomes NaN). Rows are de-duplicated
    and sorted on the key columns, so the list is a canonical description of the dataset that
    does not depend on the order of the records in the JSON.

    Parameters
    ----------
    database : dict
        A dataset as loaded from one of the GFZ JSON files.

    Returns
    -------
    pd.DataFrame
        Columns ``ESM_RECORD_KEYS``, a fresh RangeIndex.
    """
    ids = _convert_dict_to_df(database).astype(str)
    ids["sensor_depth_m"] = ids["sensor_depth_m"].astype(np.float64)
    ids = ids[ESM_RECORD_KEYS].drop_duplicates()
    return ids.sort_values(ESM_RECORD_KEYS).reset_index(drop=True)


def load_esm_record_ids(file_path) -> pd.DataFrame:
    """Load an ESM record ID list written by nb 002c.

    The code columns are read as literal text, so codes such as station "0703" keep their
    leading zero and codes that pandas would otherwise read as missing (e.g. "NA") stay strings.
    Only an empty sensor-depth cell is read as missing (NaN); the depth is then float64, the
    type ``extract_flatfile_for_dataset`` matches on.

    Parameters
    ----------
    file_path : str or Path
        CSV file with the columns ``ESM_RECORD_KEYS``.

    Returns
    -------
    pd.DataFrame
        Columns ``ESM_RECORD_KEYS``.
    """
    ids = pd.read_csv(file_path, sep=",", dtype=str,
                      keep_default_na=False, na_values={"sensor_depth_m": [""]})
    ids["sensor_depth_m"] = ids["sensor_depth_m"].astype(np.float64)
    return ids[ESM_RECORD_KEYS]


def record_ids_found_in_esm(record_ids: pd.DataFrame, esm: pd.DataFrame) -> pd.Series:
    """For each record ID, whether the ESM flatfile has a record with those four keys.

    ``esm`` is not modified; its sensor depth is converted to float64 on a copy of the key
    columns only.

    Returns
    -------
    pd.Series
        Boolean, aligned with ``record_ids``.
    """
    esm_keys = esm[ESM_RECORD_KEYS].astype({"sensor_depth_m": np.float64}).drop_duplicates()
    check = record_ids[ESM_RECORD_KEYS].merge(esm_keys, on=ESM_RECORD_KEYS,
                                              how="left", indicator=True)
    return pd.Series((check["_merge"] == "both").to_numpy(), index=record_ids.index)


def extract_flatfile_for_record_ids(record_ids: pd.DataFrame, esm: pd.DataFrame) -> pd.DataFrame:
    """The ESM records listed in an ESM record ID list, as a flatfile.

    Gives the same result as ``extract_flatfile_for_dataset`` does for the GFZ dataset the list
    was made from: an inner merge of ``esm`` with the key columns, so the rows keep the ESM
    order and all ESM columns, with a fresh RangeIndex, and ``sensor_depth_m`` is float64.
    Unlike ``extract_flatfile_for_dataset``, ``esm`` itself is not modified.

    Parameters
    ----------
    record_ids : pd.DataFrame
        Columns ``ESM_RECORD_KEYS`` (e.g. from ``load_esm_record_ids``).
    esm : pd.DataFrame
        The ESM flatfile (any extra columns are carried through).

    Returns
    -------
    pd.DataFrame
        The matching ESM rows.
    """
    # convert the depth on a new frame so the caller's ESM frame is left as it is
    esm = esm.assign(sensor_depth_m=esm["sensor_depth_m"].astype(np.float64))
    return esm.merge(record_ids[ESM_RECORD_KEYS], on=ESM_RECORD_KEYS, how="inner")


def add_site_rup_ctx_columns(esm_db: pd.DataFrame, rake_angle: float):
    """ adds new columns with the names used in the openquake site_rup context
    objects.
    
    Most of the data is provided / can be calculated from data already in the flatfiles
    with the exception of the rake angle of the ruptures. The given rake angle 
    provided as an input is is applied to all records that have an oblique or undefined
    fault type.
    rake_angle = 90 "reverse fault"  
    rake_angle = 0 "strike slip fault"
    rake_angle = -90 "normal fault" 

    xvf: the distance to the volcanic front in km
    approximated for each recording using  the neareast site from the ESHM20 site
    model
    """
    n_ctx_columns_added = 19
    
    ####### Some precalculations and data conversions
    # add some metadata columns to the esm_db
    esm_db_mod = esm_db.copy()

    # calculate the hypocentral distance for every record
    esm_db_mod["epi_dist"] = pd.to_numeric(esm_db_mod["epi_dist"])
    esm_db_mod["ev_depth_km"] = pd.to_numeric(esm_db_mod["ev_depth_km"])
    hypo_dist = np.sqrt(esm_db_mod["epi_dist"] ** 2 + esm_db_mod["ev_depth_km"] ** 2)

    idx = np.where(esm_db_mod.columns == "epi_dist")[0][0]
    try:
        esm_db_mod.insert(loc=idx+1, column="hypo_dist", value=hypo_dist)
    except ValueError:
        esm_db_mod["hypo_dist"] = hypo_dist

    # convert some other columns to numeric
    esm_db_mod["es_dip"] = pd.to_numeric(esm_db_mod["es_dip"])
    esm_db_mod["es_width"] = pd.to_numeric(esm_db_mod["es_width"])
    esm_db_mod["es_z_top"] = pd.to_numeric(esm_db_mod["es_z_top"])
    esm_db_mod["es_rake"] = pd.to_numeric(esm_db_mod["es_rake"])
    
    # calculate the geometric mean of the 5%-95% significant duration (T90)
    esm_db_mod["U_T90"] = pd.to_numeric(esm_db_mod["U_T90"])
    esm_db_mod["V_T90"] = pd.to_numeric(esm_db_mod["V_T90"])
    gm_T90 = np.round((esm_db_mod["U_T90"] * esm_db_mod["V_T90"]) ** (1 / 2), 3)

    idx = np.where(esm_db_mod.columns == "W_T90")[0][0]
    try:
        esm_db_mod.insert(loc=idx+1, column="GM_T90", value=gm_T90)
    except ValueError:
        esm_db_mod["GM_T90"] = gm_T90

    # calculate the geometric mean of the Arias Intensity
    esm_db_mod["U_ia"] = pd.to_numeric(esm_db_mod["U_ia"])
    esm_db_mod["V_ia"] = pd.to_numeric(esm_db_mod["V_ia"])
    gm_ia = (esm_db_mod["U_ia"] * esm_db_mod["V_ia"]) ** (1 / 2)

    idx = np.where(esm_db_mod.columns == "W_ia")[0][0]
    try:
        esm_db_mod.insert(loc=idx+1, column="GM_ia", value=gm_ia)
    except ValueError:
        esm_db_mod["GM_ia"] = gm_ia

    # some data conversions:
    esm_db_mod["EMEC_Mw"] = pd.to_numeric(esm_db["EMEC_Mw"])
    
    ####### Start the creation of columns specifically for the site_rup_ctxs
    # xvf: the distance to the volcanic front in km
    # approximated for each recording using  the neareast site from the ESHM20 site
    # model
    # load the site model 
    site_model = pd.read_csv(site_model_fp, sep=",")
    site_model = gpd.GeoDataFrame(site_model, 
                                geometry=gpd.points_from_xy(
                                site_model.lon, site_model.lat),
                                crs="EPSG:4326")

    # match each esm_db site to the nearest site in the site model
    esm_gdf = gpd.GeoDataFrame(esm_db, 
                                geometry=gpd.points_from_xy(
                                    esm_db.st_longitude, esm_db.st_latitude),
                                crs="EPSG:4326")

    nearest_sites = site_model.sindex.nearest(esm_gdf.geometry, return_all=False)
    xvf_values = site_model.iloc[nearest_sites[1]]["xvf"].values
    z1pt0_values = site_model.iloc[nearest_sites[1]]["z1pt0"].values
    region_values = site_model.iloc[nearest_sites[1]]["region"].values

    # add a column indicating the maximum usable period based on the highpass filter
    # frequency of the record
    esm_db_mod["max_usable_T"] = 0.8 / esm_db_mod[["U_hp", "V_hp"]].apply(pd.to_numeric).min(axis=1)

    # add columns needed for the openquake context array
    # magnitude "mag": where possible used the EMEC_Mw, otherwise fall back on the
    # default Mw
    esm_db_mod["mag"] = esm_db_mod["EMEC_Mw"].combine_first(esm_db_mod["Mw"])
    esm_db_mod["mag"] = pd.to_numeric(esm_db_mod["mag"])

    esm_db_mod["hypo_depth"] = esm_db_mod["ev_depth_km"]
    # rake: where possible use the "es_rake" otherwise base it on the fm_type_code
    esm_db_mod["rake"] = esm_db_mod["es_rake"].combine_first(
        _estimate_rake_angle(esm_db_mod["fm_type_code"], rake_angle))
    
    esm_db_mod["fault_dip"] = esm_db_mod["es_dip"].combine_first(
        pd.Series(_estimate_dip_angle(esm_db_mod["rake"]), index=esm_db_mod["es_dip"].index))

    esm_db_mod["rup_width"] = esm_db_mod["es_width"].combine_first(
        pd.Series(_estimate_fault_width(esm_db_mod["rake"].values, esm_db_mod["mag"].values), index=esm_db_mod["es_width"].index))
    
    esm_db_mod["ztor"] = esm_db_mod["es_z_top"].combine_first(
        pd.Series(_estimate_ztor(esm_db_mod["hypo_depth"], esm_db_mod["rup_width"], esm_db_mod["fault_dip"]), index=esm_db_mod["es_z_top"].index))

    esm_db_mod["rhypo"] = esm_db_mod["hypo_dist"]
    # distance from rupture plane "rrup": where possible use rup_dist, otherwise
    # fall back on hypo_dist
    esm_db_mod["rrup"] = esm_db_mod["rup_dist"].combine_first(esm_db_mod["hypo_dist"])
    esm_db_mod["rjb"] = esm_db_mod["JB_dist"].combine_first(esm_db_mod["epi_dist"])
    esm_db_mod["xvf"] = xvf_values

    esm_db_mod["vs30"] = esm_db_mod["vs30_m_sec"].combine_first(esm_db_mod["vs30_m_sec_WA"])
    # differentiate between vs30 from measurements or from slope
    esm_db_mod["vs30measured"] = esm_db_mod["vs30_m_sec"].notna()
    esm_db_mod["z1pt0"] = z1pt0_values
    esm_db_mod["region"] = region_values
    # station location
    esm_db_mod["lat"] = pd.to_numeric(esm_db_mod["st_latitude"])
    esm_db_mod["lon"] = pd.to_numeric(esm_db_mod["st_longitude"])
    # hypocentre location
    esm_db_mod["hypo_lat"] = pd.to_numeric(esm_db_mod["ev_latitude"])
    esm_db_mod["hypo_lon"] = pd.to_numeric(esm_db_mod["ev_latitude"])

    # reorder the dataframe columns for easier viewing
    cols = esm_db_mod.columns.tolist()
    cols_before = cols[:72]
    cols_after = cols[72:-n_ctx_columns_added]
    cols_to_move = cols[-n_ctx_columns_added:]
    new_cols = cols_before + cols_to_move + cols_after
    esm_db_mod = esm_db_mod[new_cols]

    return esm_db_mod


def _estimate_dip_angle(rake_values: np.ndarray) -> np.ndarray:
    # based on Kaklamanos et al. 2011
    # Define faulting conditions based on rake
    conds = [
        (rake_values >= -135) & (rake_values <= -45), # Normal
        (rake_values >= 45) & (rake_values <= 135)    # Reverse
    ]
    
    # Define the corresponding output values
    choices = [50, 40]
    
    # Default corresponds to Strike-Slip
    return np.select(conds, choices, default=90)
    

def _estimate_fault_width(rake_values: np.ndarray, mag: np.ndarray) -> np.ndarray:
    # based on Kaklamanos et al. 2011
    # Define faulting conditions based on rake
    conds = [
        (rake_values >= -135) & (rake_values <= -45), # Normal
        (rake_values >= 45) & (rake_values <= 135)    # Reverse
    ]

    # fault width estimates for each regime
    choices = [
        10**(-1.14 + 0.35 * mag),
        10**(-1.61 + 0.41 * mag)
    ]

    # Default corresponds to Strike-Slip
    return np.select(conds, choices, default=10**(-0.76 + 0.27 * mag))


def _estimate_ztor(
        hypo_depth: np.ndarray, rup_width: np.ndarray, fault_dip: np.ndarray) -> np.ndarray:
    # based on Kaklamanos et al. 2011
    return np.maximum((hypo_depth - 0.6 * rup_width * np.sin(fault_dip)), 0)
    

def _estimate_rake_angle(fm_type_codes: pd.Series, fill_rake):
    rake_map = {
        "NF": -90,
        "TF": 90,
        "SS": 0,
        "O": fill_rake,
        "U": fill_rake
    }

    estimated_rakes = fm_type_codes.map(rake_map)
    return estimated_rakes


def drop_records_with_missing_metadata(df):
    df = df[df["mag"].notna() 
        & df["rhypo"].notna() 
        & df["hypo_depth"].notna() 
        & df["rhypo"].notna() 
        & df["vs30"].notna() 
        & df["vs30measured"].notna()
        & df["z1pt0"].notna()
        & df["rrup"].notna()
        & df["ztor"].notna()
        ]
    return df
    

def drop_records_with_missing_imdata(df, idx_im_data):
    im_df = eqdb.extract_im_data(df, idx_im_data)
    im_df = im_df.dropna(axis=0)
    return df.loc[im_df.index, :]


def add_AvgSA_columns(
        flatfile_df: pd.DataFrame, 
        components: list[str], 
        tags: list[str], 
        periods: list[np.ndarray],
        idx_im_data: int):

    idx = pd.IndexSlice

    sa_df = extract_SA_data(flatfile_df, idx_im_data)
    sa_df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in sa_df.columns])

    pga_df = extract_PGA_data(flatfile_df, idx_im_data)
    pga_df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in pga_df.columns])

    for com, tag, ts in zip(components, tags, periods):
        
        # extract the SA values for these components. interpolate if needed
        sa_com_df = sa_df.loc[:, idx[com, :, :]]
        sa_com_df = sa_com_df[com]
        
        if any(ts < 0.01): # then PGA needs to be treated like SA(0.001)
            pga_series = pga_df[com]
            sa_com_df.insert(loc=0, column=f"SA(0.0)", value=pga_series)
        
        sa_com_df = sa_com_df.apply(pd.to_numeric) # ensure numeric data
        sa_com_df = sa_com_df.dropna(axis=0)
        interped_SA_values = dfops.interpolate_for_sa(sa_com_df, ts, 
                                                      pga_to_small_T=True)
        
        # calculate the AvgSA
        sa_arr = interped_SA_values.to_numpy()
        AvgSA = np.exp(compute_ln_AvgSA(np.log(sa_arr)))
        
        # assign the avgSA values back to the original flatfile
        column_name = f"{com}_{tag}"
        flatfile_df.loc[sa_com_df.index, column_name] = AvgSA

    return flatfile_df


def add_AvgSA_columns2(
        im_df: pd.DataFrame, 
        tags: list[str], 
        periods: list[np.ndarray]):
    
    sa_df = im_df[[c for c in im_df.columns if c.startswith("SA(")]]

    if any(np.concatenate(periods) <= 0.01): # consider all periods
        pga_series = im_df["PGA"]
        sa_df.insert(loc=0, column=f"SA(0.0)", value=pga_series)
   
    sa_df = sa_df.apply(pd.to_numeric) # ensure numeric data
    sa_df = sa_df.dropna(axis=0)   # remove rows with nan
    
    for tag, ts in zip(tags, periods):
        interped_SA_values = dfops.interpolate_for_sa(sa_df, ts, 
                                                      pga_to_small_T=True)
    
        # calculate the AvgSA
        sa_arr = interped_SA_values.to_numpy()
        AvgSA = np.exp(compute_ln_AvgSA(np.log(sa_arr)))
    
        # assign the avgSA values back to the original flatfile
        im_df.loc[sa_df.index, tag] = AvgSA.copy()

    return im_df


def interpolate_SA_values(im_df: pd.DataFrame, periods):
    
    
    sa_df = im_df[[c for c in im_df.columns if c.startswith("SA(")]]

    if any(periods < 0.01): # consider all periods
        sa_df.insert(loc=0, column=f"SA(0.0)", value=im_df["PGA"])
   
    sa_df = sa_df.apply(pd.to_numeric) # ensure numeric data
    valid_sa_df = sa_df.dropna(axis=0)   # remove rows with nan

    interped_SA_values = dfops.interpolate_for_sa(valid_sa_df, periods, pga_to_small_T=True)
    result = pd.DataFrame(
            index=im_df.index, 
            columns=interped_SA_values.columns, 
            dtype=float
        )
        
    # 6. Map interpolated data back to the matching indices
    result.loc[valid_sa_df.index, :] = interped_SA_values
    
    return result



def filter_gm_database_on_imts(im_df, imts: list[IMT]):
    """ returns a dataframes with columns corresponding to the imts given.
    Values of SA are interpolated where possible.
    Assumes only the im columns are given and there is only a single level index
    consisting of the im strings.
    No error is raised if an imt is not in im_df."""

    # get the non-SA ims
    non_sa_ims = [im.string for im in imts 
                 if (not im.string.startswith("SA(")) and
                 (im.string in im_df.columns)]
    non_sa_df = im_df[non_sa_ims]
    
    # interpolate for desired SA ims
    sa_df = im_df[[c for c in im_df.columns if c.startswith("SA(")]]
    desired_periods = np.array([im.period for im in imts if im.name == "SA"])

    if any(desired_periods <= 0.01): # consider all periods
        pga_series = im_df["PGA"]
        sa_df.insert(loc=0, column=f"SA(0.0)", value=pga_series)
   
    sa_df = sa_df.apply(pd.to_numeric) # ensure numeric data
    sa_df = sa_df.dropna(axis=0)   # remove rows with nan
    
    interped_SA_df = dfops.interpolate_for_sa(
        sa_df, desired_periods, pga_to_small_T=True)
    
    # combine everything together
    new_im_df = pd.concat([non_sa_df, interped_SA_df], axis=1)

    return new_im_df


def extract_SA_data(flatfile_df: pd.DataFrame, idx_im_data: int):
    flag_for_SA = "_SA("
    df = eqdb.rename_sa_columns_esm(flatfile_df)
    im_df = eqdb.extract_im_data(df, idx_im_data)
    sa_cols = [c for c in im_df.columns if flag_for_SA in c]
    sa_df = im_df[sa_cols]
    
    return sa_df


def extract_PGA_data(flatfile_df: pd.DataFrame, idx_im_data:int):
    flag_for_SA = "_pga"
    im_df = eqdb.extract_im_data(flatfile_df, idx_im_data)
    pga_cols = [c for c in im_df.columns if flag_for_SA in c]
    pga_df = im_df[pga_cols]
    
    return pga_df


def get_active_shallow_crust_flatfile(deep_events, database_df):
    """The active shallow crust (ASC) flatfile, following section 4.2.1 of the ESHM20 report
    (Danciu et al. 2021):

    (i)   the record is not in a subduction or deep dataset: every record in ``deep_events``
          (the ESHM20 in-slab, interface and Vrancea flatfiles) is removed;
    (ii)  the reported hypocentral depth is at most 40 km;
    (iii) the event has at least 3 records;
    (iv)  only records whose high-pass filter frequency allows the period are used at that
          period. This is applied later, per IM, in the residual partitioning.

    Parameters
    ----------
    deep_events : pd.DataFrame
        The regional flatfiles concatenated; only the columns ``ESM_RECORD_KEYS`` are used.
    database_df : pd.DataFrame
        The ESM flatfile. Its ``sensor_depth_m`` must already be float64, as in nb 002.

    Returns
    -------
    pd.DataFrame
        The ASC rows of ``database_df``, in their original order and with their original
        index labels. The labels matter: nb 002 writes the flatfile with ``index=True``, and the
        residual files are keyed on that index.

    Raises
    ------
    ValueError
        If, after step (i), a remaining row still matches a ``deep_events`` record, or if the
        number of rows removed is not the number of ``deep_events`` records found in
        ``database_df``.
    """
    # ---- (i) remove the regional records ----------------------------------------------------
    # The match is on the four ESM record keys, the same keys the regional flatfiles were
    # selected with, and not on the index. The regional flatfiles come from a merge
    # (extract_flatfile_for_record_ids), which gives them a fresh 0..n-1 RangeIndex, so their
    # index labels say nothing about which ESM row a record came from.
    #
    # The keys are taken on both sides with the sensor depth as float64, so the depth is matched
    # as a number (and a missing depth, NaN, matches a missing depth, as in the merge that built
    # the regional flatfiles). The regional keys are de-duplicated so the merge below gives
    # exactly one output row per ESM row.
    db_keys = database_df[ESM_RECORD_KEYS].astype({"sensor_depth_m": np.float64})
    deep_keys = (deep_events[ESM_RECORD_KEYS].astype({"sensor_depth_m": np.float64})
                 .drop_duplicates())

    # Anti-join: a left merge with indicator=True marks each ESM row "both" (it is a regional
    # record) or "left_only" (it is not). A merge drops the index, so each row's POSITION in
    # database_df is carried through as a column; selecting by position afterwards keeps the
    # original index labels and the original row order (a left merge keeps the left order).
    flagged = (db_keys.assign(_position=np.arange(len(database_df)))
               .merge(deep_keys, on=ESM_RECORD_KEYS, how="left", indicator=True))
    if len(flagged) != len(database_df):
        raise ValueError("Matching the regional records changed the number of ESM rows.")
    keep = flagged.loc[flagged["_merge"] == "left_only", "_position"].to_numpy()
    asc_esm_df = database_df.iloc[keep]    # (i)

    # Check (i) against an independent count. No remaining row may match a regional record...
    leftover = asc_esm_df[ESM_RECORD_KEYS].astype({"sensor_depth_m": np.float64}).merge(
        deep_keys, on=ESM_RECORD_KEYS, how="inner")
    if len(leftover) > 0:
        raise ValueError(f"{len(leftover)} ASC rows still match a subduction/deep record.")
    # ...and the rows removed must be exactly the regional records present in the ESM frame
    # (each record key appears once in ESM, so one key removes one row).
    n_removed = len(database_df) - len(asc_esm_df)
    n_found = len(deep_keys.merge(db_keys.drop_duplicates(), on=ESM_RECORD_KEYS, how="inner"))
    if n_removed != n_found:
        raise ValueError(f"Step (i) removed {n_removed} rows, but {n_found} distinct regional "
                         "record keys were found in the ESM flatfile.")
    if len(deep_events) > 0 and n_removed == 0:
        raise ValueError("Step (i) removed no rows although deep_events is not empty.")

    # ---- (ii) and (iii) ---------------------------------------------------------------------
    asc_esm_df = asc_esm_df[asc_esm_df["ev_depth_km"] <= 40]    # (ii)
    asc_esm_df = asc_esm_df.groupby("event_id").filter(lambda g: len(g) >= 3)    # (iii)
    return asc_esm_df


def print_summary(df_flatfile):
    print("No. Events: ", len(df_flatfile.groupby("event_id")))
    print("No. Records: ", len(df_flatfile))
    print("No. Stations: ", len(df_flatfile["station_code"].unique()))
    print("Mw (min/max): ", 
        df_flatfile["EMEC_Mw"].min(), df_flatfile["EMEC_Mw"].max())


def set_new_rake(df: pd.DataFrame, new_rake_angle: float):
    # sets a new rake angle for all records where one is not already specified
    df["rake"] = df["es_rake"].combine_first(
        _estimate_rake_angle(df["fm_type_code"], new_rake_angle))
    return df


def label_record_trts_esm(esm_df, col_idx=None):
    """Add a column "trt" labelling each ESM record with its tectonic region type.

    Records in the ESHM20 subduction in-slab, subduction interface and Vrancea datasets
    (identified by their ESM record ID lists, see ``load_esm_record_ids``) are labelled
    "Subduction Inslab", "Subduction Interface" and "Non-Subduction Deep"; every other record
    is "Shallow Default". The column is appended, or inserted at position ``col_idx``.
    ``esm_df["sensor_depth_m"]`` is converted to float64 so it can be matched.
    """
    record_ids_folder = cfg["raw_data"]["trt_record_ids"]

    # region tag of each ID list -> label written to the "trt" column
    trt_labels = {
        "sslab": "Subduction Inslab",
        "sinter": "Subduction Interface",
        "vran": "Non-Subduction Deep",
        }

    esm_df.loc[:, "sensor_depth_m"] = esm_df["sensor_depth_m"].astype(np.float64)
    if col_idx is None:
        esm_df = esm_df.assign(trt="Shallow Default")
    else:
        esm_df.insert(col_idx, "trt", "Shallow Default")

    # keep the row labels in a column so the matched rows can be found after the merge
    df_A = esm_df.reset_index()

    for tag, label in trt_labels.items():
        ids = load_esm_record_ids(record_ids_folder / f"{tag}_esm_record_ids.csv")

        # Merge using only the stable identifier columns
        matched = df_A.merge(ids[ESM_RECORD_KEYS], on=ESM_RECORD_KEYS, how='inner')
        indices = matched['index'].tolist()  # row labels of the matched records

        # update the appropriate rows in the trt column
        esm_df.loc[indices, "trt"] = label
    return esm_df


def scale_cols(df: pd.DataFrame, conversion_map: dict) -> pd.DataFrame:
    """multiples each colum by a value according to conversion map, which
    maps the name of each column to a scale factor"""
    for tag, sf in conversion_map.items():
        mask = df.columns.str.startswith(tag)
        df.loc[:, mask] *= sf
    return df


def get_SA_periods_from_headers(df: pd.DataFrame):
    # Parse Periods from column headers
    pattern = re.compile(r"SA\(([\d.]+)\)")
    cols = [c for c in df.columns if pattern.search(c)]
    periods = np.array([float(pattern.search(c).group(1)) for c in cols])
    return periods


# ---------------------------------------------------------------------------------------------
# network_code in the selection database (added 2026-09-30)
# ---------------------------------------------------------------------------------------------
# Why this exists: ESM has 281 (event, station, location) combinations recorded by TWO networks
# with the same station code (278 GE/HL, 2 FR/RA, 1 AC/IT). Until 2026-09-30 nb 030 dropped the
# network code, so each such pair became two DB rows that differed only by `index`, and the ESM
# converter could not tell which NET.STA group of the downloaded HDF5 file the selected row was.
# nb 030 now keeps the network code as the LAST metadata column.
#
# The selection results (stripe pickles, batch pickles) were made from the database WITHOUT that
# column, and every one of their manifests stores the SHA-256 of the database file. Adding the
# column changes that hash, so the manifests have to be re-stamped. That is only honest if the
# new file is the old file plus one column: same rows, same order, same text in every other
# column. The helpers below check exactly that, byte for byte.

NETWORK_CODE_COL = ("metadata", "network_code")


def read_gm_db_as_text(fp) -> pd.DataFrame:
    """Read the selection database with every cell kept as its literal CSV text.

    ``dtype=str`` stops pandas turning e.g. location code "00" into 0, and ``na_filter=False``
    stops empty cells becoming NaN. So nothing is re-formatted, and writing the frame back with
    ``to_csv(fp, sep=",", index=False)`` reproduces the original file byte for byte (checked on
    the 2026-09-30 database: identical SHA-256). This is the basis of the comparison below.
    """
    return pd.read_csv(fp, header=[0, 1], dtype=str, keep_default_na=False, na_filter=False)


def gm_db_sha256_without_network_code(fp) -> str:
    """SHA-256 of the selection database as it would be WITHOUT the network_code column.

    The file is read as text, the network_code column is dropped (if there is one), and the rest
    is written to a temporary file in the same way nb 030 writes the database (same separator,
    no index, the platform's default line ending). The hash of that temporary file is returned.

    - For a database without the column this is simply the hash of the file itself.
    - For a database with the column it is the hash the file WOULD have had if nb 030 had not
      added the column. So if it equals the hash recorded in the selection manifests, the new
      database differs from the one the selection used only by the extra column.
    """
    import hashlib
    import tempfile

    df = read_gm_db_as_text(fp)
    if NETWORK_CODE_COL in df.columns:
        df = df.drop(columns=[NETWORK_CODE_COL])
    with tempfile.TemporaryDirectory() as tmp:
        tmp_fp = Path(tmp) / "gm_db_without_network_code.csv"
        df.to_csv(tmp_fp, sep=",", index=False)
        with open(tmp_fp, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()


def network_code_lookup(gm_db_fp, recs: pd.DataFrame) -> pd.Series:
    """The network code of each selected record, looked up from the selection database.

    ``recs`` is a stripe's (or several stripes') ``recs`` frame. Its index is the database ROW
    LABEL, the positional RangeIndex the selection works with (see
    ``gm_selection.greedy_optimise_ensemble``). This is the only thing that tells the two rows of
    a cross-network pair apart, so it is what we join on.

    The ensembles selected before 2026-09-30 carry no network_code column themselves, and they
    are deliberately NOT rewritten (their pickles feed other fingerprints). Looking the code up
    here works for old and new ensembles alike.

    As a safety check the event, station and component of every selected row must equal those of
    the database row with the same label, so a database whose rows had moved would fail loudly
    here instead of silently attaching the wrong network.

    Returns a Series aligned with ``recs`` (same index, same order). NGA-Sub rows without a
    network code are NaN.
    """
    db = read_gm_db_as_text(gm_db_fp)
    if NETWORK_CODE_COL not in db.columns:
        raise KeyError(f"{Path(gm_db_fp).name} has no network_code column: re-run nb 030")
    db_meta = db["metadata"]

    labels = recs.index.to_numpy()
    if labels.min() < 0 or labels.max() >= len(db_meta):
        raise IndexError("recs index labels fall outside the selection database")
    matched = db_meta.iloc[labels]

    # identity check: the text read of the DB and the parsed recs must describe the same record
    for col in ("database", "event_id", "station_code", "component"):
        a = matched[col].astype(str).to_numpy()
        b = recs[("metadata", col)].astype(str).to_numpy()
        bad = a != b
        if bad.any():
            raise ValueError(f"{bad.sum()} selected record(s) do not match the database row "
                             f"with the same label in column {col!r}, e.g. label "
                             f"{labels[bad][0]}: DB {a[bad][0]!r} vs selection {b[bad][0]!r}")

    # empty text means "no network code" (NGA-Sub rows whose flatfile value was -999)
    net = matched["network_code"].replace("", np.nan).to_numpy()
    return pd.Series(net, index=recs.index, name="network_code")
