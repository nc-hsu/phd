import os
import json
import pickle
import numpy as np
import pandas as pd

from openquake.hazardlib.imt import PGA, SA, RSD595, AvgSA, IA, IMT

from phd_project.config.config import load_config

from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    ESHM20SiteRupCtxBuilder,
    create_gmm_map,
    create_corr_model_map,
    get_poe_from_disaggstats,
)
import phd_project.scripts.WP1_ground_motion_set.manage_flatfiles as mf

cfg = load_config()


def setup_AvgSA03_gcim_gm_selection_w_IA(weight_rsd595=0.125, weight_ia=0.125):
    ########### set some parameters for the selection ##########################
    t_lower = 0.025     # lower SA period considered in selection
    t_upper = 3         # upper SA period considered in selection
    n_periods = 20      # number of periods to consider in selection

    conditioning_imt: IMT = AvgSA([0,3]) 
    nonSA_imts: list[IMT] = [AvgSA([0,3]), RSD595(), PGA(), IA()] 
    sa_periods = np.round(np.geomspace(t_lower, t_upper, num=n_periods), 3)
    SA_imts: list[IMT] = [SA(period) for period in sa_periods]
    selection_imts: list[IMT] = nonSA_imts[1:] + SA_imts    # not AvgSA but the others (PGA and RSD595)
    nonSA_imt_strs: list[str] = [im.string for im in nonSA_imts] # strings match the correlation matrix

    # weights of the IMs  -> assumed weights
    # remaining_weight = 1 - (weight_rsd595 + weight_ia)
    # n_other_ims = len([imt for imt in selection_imts if imt.name not in ["RSD595", "IA"]])
    # imt_weights = np.array([remaining_weight / n_other_ims if imt.name not in ["RSD595", "IA"] 
    #                         else weight_rsd595 for imt in selection_imts])
    # imt_weights /= imt_weights.sum()

    imt_weights = _im_weights(selection_imts, 
                              {"RSD595": weight_rsd595,
                               "IA": weight_ia})

    return _set_up_selection(conditioning_imt, selection_imts, imt_weights, 
                             nonSA_imt_strs, sa_periods, t_upper)


def setup_AvgSA03_gcim_gm_selection(weight_rsd595=0.25):
    ########### set some parameters for the selection ##########################
    t_lower = 0.025     # lower SA period considered in selection
    t_upper = 3         # upper SA period considered in selection
    n_periods = 20      # number of periods to consider in selection

    conditioning_imt: IMT = AvgSA([0,3]) 
    nonSA_imts: list[IMT] = [AvgSA([0,3]), RSD595(), PGA()] 
    sa_periods = np.round(np.geomspace(t_lower, t_upper, num=n_periods), 3)
    SA_imts: list[IMT] = [SA(period) for period in sa_periods]
    selection_imts: list[IMT] = nonSA_imts[1:] + SA_imts    # not AvgSA but the others (PGA and RSD595)
    nonSA_imt_strs: list[str] = [im.string for im in nonSA_imts] # strings match the correlation matrix

    imt_weights = _im_weights(selection_imts, {"RSD595": weight_rsd595})

    return _set_up_selection(conditioning_imt, selection_imts, imt_weights, 
                             nonSA_imt_strs, sa_periods, t_upper)


def _im_weights(selection_imts, spec_weights: dict[str, float]):
    """ helper function to calculate the im weights based on the specified 
    weights for certain imts and equal weighting for the rest """

    selection_im_strings = [imt.string for imt in selection_imts]
    assert all(im in selection_im_strings for im in spec_weights.keys()), \
        "Specified weights must be for IMTs in the selection IMTs"

    speced_weight = sum(spec_weights.values())
    remaining_weight = 1 - speced_weight
    n_other_ims = len(selection_imts) - len(spec_weights)
    default_weight = remaining_weight / n_other_ims if n_other_ims > 0 else 0

    imt_weights = [spec_weights.get(im.string, default_weight) for im in selection_imts]
    imt_weights = np.array(imt_weights)
    imt_weights /= imt_weights.sum()
    return imt_weights
    

def _set_up_selection(
        conditioning_imt, selection_imts, imt_weights,
        nonSA_imt_strs, sa_periods, t_upper):

    # some other things
    assumed_rake = -90  # assumed rake for RSD595 calculation
    occurence = True    # the record selection should be performed based on occurence

    ####################### Load Data ##########################################
    # Load the IML-based disaggregation (the sig4 / eps4 truncation set from nb 021).
    # disagg_data is flat: disagg_data[site][imt][iml] -> DataFrame
    # disagg_stats is a flat DataFrame, one row per (site, imt, imtl) with a poe column.
    fp = cfg["proc_data"]["AvgSA_03_disagg_data_gm_selection"]
    with open(fp, "rb") as f:
        disagg_data = pickle.load(f)

    fp = cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"]
    with open(fp, "rb") as f:
        disagg_stats = pickle.load(f)

    # Only build ensembles for a per-site subset of IMLs (one MSA stripe each), listed in the
    # imls_for_selection JSON: {site_id_str: {structure_tag: [iml, ...], ..., "union": [iml, ...]}}.
    # We use the per-site "union" (all IMLs any structure at the site requests).
    with open(cfg["proc_data"]["AvgSA_03_imls_for_selection"]) as f:
        imls_for_selection = json.load(f)

    # organise the disagg data, keyed by (site, poe). poe is looked up from the
    # stats for each requested iml so the rest of the pipeline can keep the
    # (site, poe) scheme unchanged.
    imt = conditioning_imt.name  # "AvgSA" (only works for AvgSA; see disagg_imt comment below)
    site_poe_disaggs = {}
    invalid = []  # (site, iml) requested in the JSON but not usable (no stats row / no disagg df)
    for site in sorted(disagg_data.keys()):
        site_entry = imls_for_selection.get(str(site))
        wanted = site_entry.get("union") if site_entry else None
        if wanted is None:
            continue  # site not listed in the subset JSON -> skip
        iml_keys = list(disagg_data[site][imt].keys())
        for iml in wanted:
            if iml is None:
                continue  # upper-stripe placeholder (iml above the hazard ceiling) -> skip
            poe = get_poe_from_disaggstats(disagg_stats, site, imt, iml)
            key = next((k for k in iml_keys if np.isclose(k, iml)), None)
            if poe is None or key is None:
                # e.g. a zero-hazard / excluded iml (dropped from stats in nb 021) or
                # an iml that was never disaggregated -> skip gracefully.
                invalid.append((site, iml))
                continue
            site_poe_disaggs[(site, poe)] = disagg_data[site][imt][key]
    site_poes = sorted(list(site_poe_disaggs.keys()), key=lambda x: x[0])
    site_poe_disaggs = {k: site_poe_disaggs[k] for k in site_poes}

    print(f"Built {len(site_poe_disaggs)} (site, poe) disaggregations "
          f"across {len({s for s, _ in site_poe_disaggs})} sites.")
    if invalid:
        print(f"WARNING: {len(invalid)} (site, iml) requested in imls_for_selection are not "
              f"present in disagg_stats/disagg_data and were skipped:")
        for site, iml in invalid:
            print(f"    site {site}: iml {iml}")

    # load the site file
    site_model = pd.read_csv(cfg["hazard_models"]["eshm20_wp1_site_model"])

    # load the flatfiles
    flatfile_folder = cfg["proc_data"]["corr_model"] / "reverse" / "flatfiles"
    flatfiles = {}
    for f in [f for f in os.listdir(flatfile_folder) if f.endswith(".csv")]:
        tag = f.split("_")[0]
        flatfiles[tag] = pd.read_csv(flatfile_folder / f, delimiter=";", index_col=0, low_memory=False)

    flatfiles["volcanic"] = pd.read_csv(cfg["raw_data"]["gm_flatfiles"] / "volcanic_lanzanoluzi_flatfile.csv", 
                                        delimiter=";", index_col=0)

    # load the preprocessed gm database
    gm_database = pd.read_csv(cfg["proc_data"]["gm_database"], sep=",", low_memory=False, header=[0, 1])

    # filter the gm_database so that only the selection and conditioning ims are present
    gm_db = gm_database.copy()
    updated_ims = mf.filter_gm_database_on_imts(
        gm_db["ims"], selection_imts + [conditioning_imt])
    updated_ims.columns = pd.MultiIndex.from_product([['ims'], updated_ims.columns])
    gm_db = pd.concat([gm_db.drop('ims', axis=1, level=0), updated_ims], axis=1)

    # Create the GMM Map for AvgSA by reading the median-branch logic tree
    AvgSA_03_lt_fp = cfg["hazard_models"]["eshm20_AvgSA_03_median_lt"]
    gmm_map = create_gmm_map(AvgSA_03_lt_fp)

    # get the correlation model map
    corr_map = create_corr_model_map(nonSA_imt_strs, sa_periods)

    ######################## Make some assumptions #############################
    # create the average depth map for each TRT
    average_depths = {
        "Craton": flatfiles["asc"]["ev_depth_km"].mean(),
        "Non-Subduction Deep": flatfiles["vran"]["ev_depth_km"].mean(),
        "Shallow Default": flatfiles["asc"]["ev_depth_km"].mean(),
        "Subduction Inslab": flatfiles["sinter"]["ev_depth_km"].mean(),
        "Subduction Interface": flatfiles["sinter"]["ev_depth_km"].mean(),
        "Volcanic": flatfiles["volcanic"]["ev_depth_km"].mean(),
    }

    # add the average depth to the top of the rupture
    average_ztor = {
        "Craton": flatfiles["asc"]["ztor"].mean(),
        "Non-Subduction Deep": flatfiles["vran"]["ztor"].mean(),
        "Shallow Default": flatfiles["asc"]["ztor"].mean(),
        "Subduction Inslab": flatfiles["sslab"]["ztor"].mean(),
        "Subduction Interface": flatfiles["sinter"]["ztor"].mean(),
        "Volcanic": flatfiles["asc"]["ztor"].mean(),
    }

    # the map of what sim trts are allowed to match with what record trts
    OK_TRT_MATCHES = {
        "Craton": ["Shallow Default"],
        "Non-Subduction Deep": ["Non-Subduction Deep", "Subduction Inslab", "Subduction"],
        "Shallow Default": ["Shallow Default"],
        "Subduction Inslab": ["Non-Subduction Deep", "Subduction Inslab", "Subduction"],
        "Subduction Interface": ["Subduction Interface"],
        "Volcanic": ["Shallow Default"],
    }

    ###################### set up the selection context ########################
    basic_selection_ctx = {
        "n_ensembles": 20,
        "n_samples": 30,
        "conditioning_imt": conditioning_imt ,
        "disagg_imt": conditioning_imt.name , # this only works for AvgSA. otherwise used .string 
        "selection_imts": selection_imts ,
        "imt_weights": imt_weights ,
        "ctx_builder": ESHM20SiteRupCtxBuilder ,
        "ctx_builder_params": ["average_depths", "assumed_rake", "average_ztor"] ,
        "average_depths": average_depths ,
        "assumed_rake": assumed_rake ,
        "average_ztor": average_ztor ,
        "gmm_map": gmm_map ,
        "corr_map": corr_map ,
        "m_bound_model": "tarbali_and_bradley_2016" ,
        "d_bound_model": "tarbali_and_bradley_2016" ,
        "vs30_bound_model": "tarbali_and_bradley_2016" ,
        "sf_bounds": None ,
        "usable_T": t_upper ,
        "max_n_recs": 5 ,
        "p_value": 0.05 ,
        "ok_trt_matches": OK_TRT_MATCHES ,
        "occurence": occurence ,
    }

    return site_poe_disaggs, disagg_stats, site_model, basic_selection_ctx, gm_db
