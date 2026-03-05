# -*- coding: utf-8 -*-
"""
Module for obtaining the flatfiles for each tectonic regime in ESHM20 and 
populating any required data that is missing.
"""
import re
import json
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


def add_site_rup_ctx_columns(esm_db: pd.DataFrame, rake_angle: float):
    """ adds new columns with the names used in the openquake site_rup context
    objects.
    
    Most of the data is provided / can be calculated from data already in the flatfiles
    with the exception of the rake angle of the ruptures. The given rake angle 
    provided as an input is is applied to all records.
    rake_angle = 90 "reverse fault"  
    rake_angle = 0 "strike slip fault"
    rake_angle = -90 "normal fault" 

    xvf: the distance to the volcanic front in km
    approximated for each recording using  the neareast site from the ESHM20 site
    model
    """
    n_ctx_columns_added = 12
    
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

    # calculate the geometric mean of the 5%-95% significant duration (T90)
    esm_db_mod["U_T90"] = pd.to_numeric(esm_db_mod["U_T90"])
    esm_db_mod["V_T90"] = pd.to_numeric(esm_db_mod["V_T90"])
    gm_T90 = np.round((esm_db_mod["U_T90"] * esm_db_mod["V_T90"]) ** (1 / 2), 3)

    idx = np.where(esm_db_mod.columns == "W_T90")[0][0]
    try:
        esm_db_mod.insert(loc=idx+1, column="GM_T90", value=gm_T90)
    except ValueError:
        esm_db_mod["GM_T90"] = gm_T90

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
    esm_db_mod["hypo_depth"] = esm_db_mod["ev_depth_km"]
    # rake: where possible use the "es_rake" otherwise set to 90 (reverse faulting)
    esm_db_mod["rake"] = esm_db_mod["es_rake"].fillna(rake_angle)

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

    # reorder the dataframe columns for easier viewing
    cols = esm_db_mod.columns.tolist()
    cols_before = cols[:72]
    cols_after = cols[72:-n_ctx_columns_added]
    cols_to_move = cols[-n_ctx_columns_added:]
    new_cols = cols_before + cols_to_move + cols_after
    esm_db_mod = esm_db_mod[new_cols]

    return esm_db_mod


def drop_records_with_missing_metadata(df):
    df = df[df["mag"].notna() 
        & df["rhypo"].notna() 
        & df["hypo_depth"].notna() 
        & df["rhypo"].notna() 
        & df["vs30"].notna() 
        & df["vs30measured"].notna()
        & df["z1pt0"].notna()
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
    # get the active shallow crust flatfile based on the filter criteria described 
    # in section 4.2.1 of the ESHM20 report. Danciu et al. (2021)
    # i) events must be classified as non-subduction (according to the 
    # classification scheme indicated in the subsequent section), 
    # ii) reported hypocentral depth must be shallower than 40 km, 
    # iii) events must have â‰¥ 3 records, 
    # iv) only records with high-pass filter frequency ð�‘“Hp â‰¤ 0.8 ð�‘‡ are retained
    # for each period in the regression. 
    # 
    # According to the report the final database yields 18,222 records from 927 
    # events (3.1 â‰¤ ð�‘€w â‰¤ 7.4) recorded at 1829 stations (0 â‰¤ ð�‘…JB(ð�‘˜ð�‘š) â‰¤ 545).
    asc_esm_df = database_df.loc[~database_df.index.isin(deep_events)]    # (i)
    asc_esm_df = asc_esm_df[asc_esm_df["ev_depth_km"] <= 40]    # (ii)
    asc_esm_df = asc_esm_df.groupby("event_id").filter(lambda g: len(g) >= 3) # (iii)  
    #(iv) is done later when working with the data    
    return asc_esm_df


def print_summary(df_flatfile):
    print("No. Events: ", len(df_flatfile.groupby("event_id")))
    print("No. Records: ", len(df_flatfile))
    print("No. Stations: ", len(df_flatfile["station_code"].unique()))
    print("Mw (min/max): ", 
        df_flatfile["EMEC_Mw"].min(), df_flatfile["EMEC_Mw"].max())


def set_new_rake(df, new_rake_angle):
    # sets a new rake angle for all records where one is not already specified
    df["rake"] = df["es_rake"].fillna(new_rake_angle)
    return df


def label_record_trts_esm(esm_df, col_idx=None):
    
    non_asc_data_folder = cfg["raw_data"]["root"] / "eshm20_subduction_vrancea_datasets"
    non_asc_data_files = [
        "hellenic_inslab_db.json", 
        "hellenic_interface_db.json", 
        "vrancea_db.json"
        ]
    
    tags = ["Subduction Inslab", "Subduction Interface", "Non-Subduction Deep"]

    esm_df.loc[:, "sensor_depth_m"] = esm_df["sensor_depth_m"].astype(np.float64)
    if col_idx is None:
        esm_df = esm_df.assign(trt="Shallow Default") 
    else:
        esm_df.insert(col_idx, "trt", "Shallow Default")
    
    df_A = esm_df.reset_index() 
    
    for f, tag in zip(non_asc_data_files, tags):
        with open(non_asc_data_folder / f, "r") as file:
            db = json.load(file)
            df_B = _convert_dict_to_df(db)
            df_B = df_B.astype(str)
            df_B["sensor_depth_m"] = df_B["sensor_depth_m"].astype(np.float64)
            keys = ["event_id", "network_code", "station_code", "sensor_depth_m"]

            # Merge using only the stable identifier columns
            matched = df_A.merge(df_B[keys], on=keys, how='inner')
            indices = matched['index'].tolist() # Extract the list of indices

            # update the appropriate rows in the trt column
            esm_df.loc[indices, "trt"] = tag
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

if __name__ == "__main__":
    ...
    


