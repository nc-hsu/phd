"""
Functions used to calculate the correlation coefficients required for this project
 - rotD50 PGA, 
 - rotD50 SA(T), 
 - rotD50 AvgSA[0,3], 
 - rotD50 AvgSA[0,6], 
 - GeometricMean RSD595

 Functions are used in wp1pt3pt5-derivation_of_correlation_models.ipynb
"""
import re

import numpy as np
import pandas as pd
from openquake.hazardlib.gsim.base import GMPE

import pickagm.eqdbases as eqdb
import pickagm.dfops as dfops
from pickagm.interpolate import interpolate_SA_df
import pickagm.avgSA
from phd_project.scripts.WP1_ground_motion_set.correlations import (
    derive_correlation_model
    )

def load_flatfile(cfg, filename: str):
    db_fp = cfg["data"]["gm_flatfiles"] / f"{filename}"
    df = pd.read_csv(db_fp, sep=";", index_col=0, 
                    dtype={
                        "Ms_ref": str,
                        "EMEC_Mw_type": str,
                        "EMEC_Mw_ref": str,
                        "location_code": str
                        })
    return df


def organise_database(df: pd.DataFrame, ims:dict[str, str]):
    idx = pd.IndexSlice
    start_idx_of_im_data = 83  # after this index is im data. before it is metadata

    df = eqdb.rename_sa_columns_esm(df)
    metadata = df[df.columns[:start_idx_of_im_data]]
    im_df = df[df.columns[start_idx_of_im_data:]]
    im_df = im_df.dropna(axis=0)
    im_df = im_df.rename(columns={c: (c.split("_")[0], c.split("_")[1]) 
                                for c in im_df.columns})
    im_df.columns = pd.MultiIndex.from_tuples(im_df.columns)
    im_df.columns = pd.MultiIndex.from_tuples(
        [(l1, l2, "None") for l1, l2 in im_df.columns]) #  make it three level

    # for each component expand the SA ims out to three levels
    pattern = re.compile(rf"SA\(([\d.]+)\)")
    components = im_df.columns.get_level_values(0).unique()

    name_mapper = {}
    for component in components:
        if component == "GM":
            continue
        keys = dfops.extract_keys_from_labels(
            pattern, im_df[component], return_keys=True)
        name_mapper = name_mapper | {(component, v[0], v[1]): (component, "SA", k) for k,v in keys.items()} 
        
    im_df.columns = im_df.columns.to_flat_index()
    im_df = im_df.rename(columns=name_mapper)

    # remove the ims that aren't needed and rename the remaining ones
    im_df = im_df[[c for c in im_df if c[1] in ims.keys()]]
    im_df.columns = [(c[0], ims[c[1]], c[2]) for c in im_df.columns]

    # sort columns
    im_df = im_df.reindex(sorted(im_df.columns), axis=1)
    im_df.columns = pd.MultiIndex.from_tuples(
        im_df.columns, names=["component", "im", "period"])

    # perform unit conversions of the ims that remain
    unit_conv = eqdb.esm_unit_conversions
    for im, sf in unit_conv.items():
        columns_to_scale = im_df.loc[:, idx[:, im, :]].columns
        im_df.loc[:, columns_to_scale] = im_df.loc[:, columns_to_scale] * sf

    # accelerations are now in [g]

    # copy PGA to SA(0.0)
    for c in components:
        if c == "GM":
            continue
        im_df.loc[:, (c, "SA", 0.0)] = np.abs(im_df.loc[:, (c, "PGA", "None")])
    im_df = im_df.reindex(sorted(im_df.columns), axis=1)

    # remove all records with invalid metadata
    metadata = metadata[metadata["mag"].notna() 
                    & metadata["rhypo"].notna() 
                    & metadata["hypo_depth"].notna() 
                    & metadata["rhypo"].notna() 
                    & metadata["vs30"].notna() 
                    & metadata["vs30measured"].notna()
                    & metadata["z1pt0"].notna()
                    ]
    idx_2_keep =  [i for i in im_df.index if i in metadata.index]
    metadata = metadata.loc[idx_2_keep]
    im_df = im_df.loc[idx_2_keep]

    return im_df, metadata


def create_site_rup_ctx(metadata: pd.DataFrame):
    # create the rupture contexts
    ctx_columns = ["mag", "rhypo", "hypo_depth", "xvf", "vs30", "vs30measured"
                , "rake", "rrup", "z1pt0", "region", "rjb"]
    site_rup_ctxs = np.recarray(
        len(metadata), dtype=[
        ("mag", "f4"),
        ("rhypo", "f4"),
        ("hypo_depth", "f4"),
        ("xvf", "f4"),
        ("vs30", "f4"),
        ("vs30measured", "bool"),
        ("rake", "f4"),
        ("rrup", "f4"),
        ("z1pt0", "f4"),
        ("region", "f4"),
        ("rjb", "f4")
    ])

    for col in ctx_columns:
        site_rup_ctxs[col] = metadata[col].to_numpy(dtype="f4")

    return site_rup_ctxs


def get_relevant_observed_data(
        im_df: pd.DataFrame,
        periods_to_interp: list[float]):
    
    idx = pd.IndexSlice
    
    # get the observed data for gm components and ims of interest
    df_observed = pd.concat([im_df.loc[:, idx["rotD50", ["SA", "PGA", "RSD595"], :]],
                             im_df.loc[:, idx["GM", "RSD595", :]]], axis=1)

    # remove SA(0.0) and SA(0.01) because they are outside the GMM range
    # interpolate the df for the periods needed to calculate 
    df_observed = df_observed.drop(
        columns=[(c) for c in df_observed.columns
        if c[1] == "SA" and ((c[2] < 0.025) or (c[2] > 8)) ], inplace=False)

    df_observed = interpolate_SA_df(df_observed, periods_to_interp, 
                                    scheme="linear_xy", return_mode="merged")
    return df_observed


def SA_residual_correlations(
        df_observed: pd.DataFrame, 
        metadata: pd.DataFrame,
        site_rup_ctxs: pd.DataFrame,
        gmm_PGA: GMPE,
        gmm_SA: GMPE,
        gmm_RSD595: GMPE):
    # Correlations for SA PGA, RSD595

    gmm_map = {"rotD50": 
                    {"SA": gmm_PGA,
                    "PGA": gmm_SA},
                "GM": 
                    {"RSD595": gmm_RSD595}
            }
    rho = derive_correlation_model(df_observed, metadata, site_rup_ctxs, gmm_map)
    return df_observed, site_rup_ctxs, rho


def add_AvgSA_values(
        df_observed: pd.DataFrame,
        periods_0_3: list[float],
        periods_0_6: list[float]):
    
    idx = pd.IndexSlice
    # calculate the AvgSA[0,3]
    SA_data = df_observed.loc[:, idx["rotD50", ["PGA", "SA"], periods_0_3]].to_numpy()
    observed_AvgSA_03 = pickagm.avgSA.compute_ln_AvgSA(np.log(SA_data))
    df_observed[("rotD50", "AvgSA[0,3]", repr(periods_0_3))] = np.exp(observed_AvgSA_03)

    # calculate the AvgSA[0,6]
    SA_data = df_observed.loc[:, idx["rotD50", ["PGA", "SA"], periods_0_6]].to_numpy()
    observed_AvgSA_06 = pickagm.avgSA.compute_ln_AvgSA(np.log(SA_data))
    df_observed[("rotD50", "AvgSA[0,6]", repr(periods_0_6))] = np.exp(observed_AvgSA_06)


def total_residual_correlations_with_AvgSA(
        df_observed: pd.DataFrame, 
        metadata: pd.DataFrame,
        site_rup_ctxs: pd.DataFrame,
        gmm_PGA_SA: GMPE,
        gmm_RSD595: GMPE,
        rho: pd.DataFrame,
        periods_0_3: list[float],
        periods_0_6: list[float]):
    # correlations for SA PGA, RSD595, AvgSA[0,3], and AvgSA[0,6]
    idx = pd.IndexSlice

    total_residual_correlation = rho[-1]
    
    rho_for_AvgSA_03 = total_residual_correlation \
                        .loc[idx["rotD50", ["PGA", "SA"], periods_0_3], 
                        idx["rotD50", ["PGA", "SA"], periods_0_3]].to_numpy()
    rho_for_AvgSA_06 = total_residual_correlation \
                        .loc[idx["rotD50", ["PGA", "SA"], periods_0_6], 
                        idx["rotD50", ["PGA", "SA"], periods_0_6]].to_numpy()
    AvgSA_GMPE_03 = pickagm.avgSA.indirect_AvgSA_GMPE(
        gmm_PGA_SA, rho_for_AvgSA_03)
    AvgSA_GMPE_06 = pickagm.avgSA.indirect_AvgSA_GMPE(
        gmm_PGA_SA, rho_for_AvgSA_06)

    gmm_map = {
        "rotD50": {
            "SA": gmm_PGA_SA,
            "PGA": gmm_PGA_SA,
            "AvgSA[0,3]": AvgSA_GMPE_03,
            "AvgSA[0,6]": AvgSA_GMPE_06
            },
        "GM": {
            "RSD595": gmm_RSD595
            }
        }

    rhos = derive_correlation_model(
        df_observed, metadata, site_rup_ctxs, gmm_map)
    
    return rhos
