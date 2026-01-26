import re
import numpy as np
import pandas as pd
from pathlib import Path
from openquake.hazardlib.imt import SA, PGA, RSD595

import pickagm.eqdbases as eqdb
import pickagm.dfops as dfops
from phd_project.config.config import load_config
cfg = load_config()

# RESIDUALS_FOLDER = cfg["raw_data"]["residuals"]
# FLATFILE_FOLDER = cfg["data"]["gm_flatfiles"]
START_IDX_IM_DATA = 84  # after this index is im data. before it is metadata (applies for)

def calculate_residuals(flatfile_fp: Path, gmms: dict[str, dict], residuals_fp: Path):
    """_summary_

    Args:
        model_params (dict): a dict of dicts containing the paramters that are
            needed to calculate the total residuals. 
            key = model_tag (str)
            value = {file: str with the file name of the flatfile to be used,
                    gmm_PGA_SA: GMPE from openquake used to predict PGA and SA,
                    gmm_RSD595: GMPE from openquake used to predict RSD595} 
    """
    # load the database and organise it
    gmm_PGA_SA = gmms["gmm_PGA_SA"]
    gmm_RSD595 = gmms["gmm_RSD595"]
    gmm_AvgSA = gmms["gmm_AvgSA"]

    print(f"Calculating residuals:\n"
            f"    Flatfile  :  {flatfile_fp.name}\n"
            f"    Output file: {residuals_fp.name}\n"
            f"    GMM PGA/SA:  {gmm_PGA_SA}\n"
            f"    GMM RSD595:  {gmm_RSD595}\n"
            f"    GMM AvgSA :  {gmm_AvgSA}\n")

    ims = {"pga": "PGA", "T90": "RSD595", "SA":"SA", 
            "AvgSA[0,3]":"AvgSA[0,3]", "AvgSA[0,6]":"AvgSA[0,6]"}
    
    df = load_flatfile(flatfile_fp)
    im_df, metadata = organise_database_for_residual_calculations(df, ims)
    ctxs = create_site_rup_ctx(metadata)

    idx = pd.IndexSlice

    # calculate the residuals for PGA and SA between 0.025s and 8s
    periods = im_df.loc[:, idx["rotD50", "SA", :]].columns.get_level_values(2)
    periods = [t for t in periods if 0.02 <= t <= 8]

    obs = np.log(im_df.loc[:, idx["rotD50", ["PGA", "SA"], ["None"] + periods]])

    imts = [PGA()] + [SA(t) for t in periods if 0.02 <= t <= 8]

    pred = np.zeros((len(imts), len(ctxs)))
    sig = np.zeros((len(imts), len(ctxs)))
    tau = np.zeros((len(imts), len(ctxs)))
    phi = np.zeros((len(imts), len(ctxs)))
    gmm_PGA_SA.compute(ctxs, imts, pred, sig, tau, phi)

    pred = pd.DataFrame(pred.T, index=obs.index, columns=obs.columns)

    # calculate the residuals for RSD595 and append to the dataframes
    pred_rsd595 = np.zeros((1, len(ctxs)))
    sig_rsd595 = np.zeros((1, len(ctxs)))
    tau_rsd595 = np.zeros((1, len(ctxs)))
    phi_rsd595 = np.zeros((1, len(ctxs)))
    gmm_RSD595.compute(ctxs, [RSD595()], pred_rsd595, sig_rsd595, tau_rsd595, phi_rsd595)

    obs[("GM", "RSD595", "None")] = np.log(im_df.loc[:, idx["GM", "RSD595", "None"]])
    pred[("GM", "RSD595", "None")] = pred_rsd595[0,:]

    # calculate the residuals for AvgSA[0,3] and AvgSA[0,6] and append to dataframes
    AvgSA_cols = [c for c in im_df if "AvgSA" in c[1]]
    for AvgSA_col in AvgSA_cols:
        imts = [PGA()] + [SA(t) for t in np.linspace(0, AvgSA_col[2])[1:]]
        pred_AvgSA = np.zeros((1, len(ctxs)))
        gmm_AvgSA.compute(ctxs, imts, pred_AvgSA)

        obs[AvgSA_col] = np.log(im_df.loc[:, AvgSA_col])
        pred[AvgSA_col] = pred_AvgSA[0,:]

    obs = obs.reindex(sorted(obs.columns), axis=1)

    pred = pred.reindex(sorted(pred.columns), axis=1)

    # calculate all the residuals
    residuals = obs - pred

    # add some the metadata back to the residuals dataframe
    new_labels = ["_".join([str(i) for i in c]+["Delta"]) for c in residuals.columns]
    residuals["event_id"] = metadata["event_id"]
    residuals["station_code"] = metadata["station_code"]
    residuals["max_usable_T"] = metadata["max_usable_T"].round(4)
    periods = np.array([0, 0] + periods)  # 0 for PGA and RSD595
    new_labels += ["event_id", "station_code", "max_usable_T"]
    residuals.columns = new_labels

    # save the dataframes as csv files for R processing
    residuals.to_csv(residuals_fp, index=True, sep=",")


def load_flatfile(flatfile_fp: Path):
    df = pd.read_csv(flatfile_fp, sep=";", index_col=0, 
                    dtype={
                        "Ms_ref": str,
                        "EMEC_Mw_type": str,
                        "EMEC_Mw_ref": str,
                        "location_code": str,
                        "housing_code": str,
                        "installation_code": str,
                        "vs30_ref": str,
                        "vs30_meas_type": str,
                        })
    return df


def organise_database_for_residual_calculations(df: pd.DataFrame, ims:dict[str, str]):
    idx = pd.IndexSlice

    df = eqdb.rename_sa_columns_esm(df)
    metadata = eqdb.extract_metadata(df, START_IDX_IM_DATA)
    im_df = eqdb.extract_im_data(df, START_IDX_IM_DATA)
    
    im_df = im_df.dropna(axis=0) # remove records without complete IM data
    im_df = im_df.rename(columns={c: (c.split("_")[0], "_".join(c.split("_")[1:])) 
                                for c in im_df.columns})
    # im_df.columns = pd.MultiIndex.from_tuples(im_df.columns)
    im_df.columns = pd.MultiIndex.from_tuples(
        [(l1, l2, "None") for l1, l2 in im_df.columns]) #  make it three level

    # for each component expand the SA and AvgSA ims out to three levels
    SA_pattern = re.compile(rf"SA\(([\d.]+)\)")
    AvgSA_pattern = re.compile(rf"AvgSA\[\d*\.?\d+,(\d*\.?\d+)\]")
    
    new_labels = []
    for c in im_df.columns:
        if c[1].startswith("SA"):
            im = "SA"
            m = SA_pattern.search(c[1])
            t = float(m.group(1)) if m else "None"
        elif c[1].startswith("AvgSA"):
            im = c[1]
            m = AvgSA_pattern.search(c[1])
            t = float(m.group(1)) if m else "None"
        else:
            try:
                im = eqdb.esm_nonSA_im_translations[c[1]]
            except KeyError:
                im = c[1]
            t = "None"
        new_labels.append((c[0], im, t))
    
    im_df.columns = pd.MultiIndex.from_tuples(
        new_labels, names=["component", "im", "period"])

    # remove the ims that aren't needed
    im_df = im_df.loc[:, idx[["GM", "rotD50"], list(ims.values()), :]]
    im_df = im_df.drop(columns=("rotD50", "RSD595", "None"))

    # sort columns
    im_df = im_df.reindex(sorted(im_df.columns), axis=1)

    # perform unit conversions of the ims that remain
    unit_conv = eqdb.esm_unit_conversions
    unit_conv["AvgSA[0,3]"] = 1/981.0       # from cm/s2 to g
    unit_conv["AvgSA[0,6]"] = 1/981.0       # from cm/s2 to g

    for im, sf in unit_conv.items():
        columns_to_scale = im_df.loc[:, idx[:, im, :]].columns
        im_df.loc[:, columns_to_scale] = im_df.loc[:, columns_to_scale] * sf

    # accelerations are now in [g]
    return im_df, metadata


def create_site_rup_ctx(metadata: pd.DataFrame):
    # create the rupture contexts for calculating the predicted ims
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


def unmangle_AvgSA_column_names(column_name):
    # R mangles the AvgSA column names becuase [] and , are not allowed.
    # This function unmangles the _AvgSA.x.x._3.0 column name and returns it to
    # _AvgSA[x,x]_
    """
    A robust version for Earthquake Engineering data.
    Uses regex to identify numbers (including decimals) and preserves them,
    only replacing the dots that act as delimiters.
    """
    # Pattern: match prefix, then '.', then a number, then '.', 
    # then another number, then '.' followed by the rest.
    # We use (\d+\.?\d*) to capture numbers like '0' or '0.5'
    pattern = r'(.*_AvgSA)\.(\d+\.?\d*)\.(\d+\.?\d*)\.(.*)'
    
    match = re.match(pattern, column_name)
    if match:
        prefix, x1, x2, suffix = match.groups()
        return f"{prefix}[{x1},{x2}]{suffix}"
    
    return column_name



if __name__ == "__main__":  
    mangled = "rotD50_AvgSA.0.5.2.5._3.0_Delta"
    unmangled = unmangle_AvgSA_column_names(mangled)
    print(unmangled)
    ...
