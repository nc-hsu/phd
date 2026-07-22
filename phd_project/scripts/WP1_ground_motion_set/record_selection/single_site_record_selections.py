import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from pathlib import Path
from copy import deepcopy

from openquake.hazardlib.imt import PGA, SA, RSD595, AvgSA, IMT

from pickagm.distributions import ensemble_ks_bounds

from phd_project.config.config import load_config 
from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    ESHM20SiteRupCtxBuilder,
    create_gmm_map,
    create_corr_model_map,
    get_record_ensembles_for_single_poe_distribution,
    get_best_ensemble_in_list,
    total_ks_statistic_and_failing_im_penalty,
    get_imtl_from_disaggstats,
    optimise_ground_motion_ensembles_for_sites,
    optimise_ground_motion_ensembles_for_sites_with_shuffles,
)
import phd_project.scripts.WP1_ground_motion_set.manage_flatfiles as mf
from phd_project.plotting import custom_log_formatter

cfg = load_config()


# Filepaths:
# input file paths
fp_disagg_data = cfg["proc_data"]["AvgSA_03_disagg_data_gm_selection"]
fp_disagg_stats = cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"]
fp_gcim_dists = cfg["proc_data"]["gcim_dists"] / f"gcim_dist_AvgSA_03_rake-90.pickle"
fp_site_model = cfg["hazard_models"]["eshm20_AvgSA_site_model_all"]
flatfile_folder = cfg["proc_data"]["corr_model"] / "reverse" / "flatfiles"
fp_gm_database = cfg["proc_data"]["gm_database"]
fp_gmm_logic_tree = cfg["hazard_models"]["eshm20_AvgSA_03_median_lt"]

# output file paths
preliminary_selection_fp = cfg["proc_data"]["gm_selection"] / f"AvgSA_03_prelim_selection_rake-90.pickle"

# Site and PoE:
site_id = 30
poe =  0.0001

#################### Load Data #################################################
# Load the disaggregation data
with open(fp_disagg_data, "rb") as f:
    disagg_data = pickle.load(f)

with open(fp_disagg_stats, "rb") as f:
    disagg_stats = pickle.load(f)

# load the site file
site_condition_data = pd.read_csv(fp_site_model)

# load the flatfiles
flatfiles = {}
for f in [f for f in os.listdir(flatfile_folder) if f.endswith(".csv")]:
    tag = f.split("_")[0]
    flatfiles[tag] = pd.read_csv(flatfile_folder / f, delimiter=";", index_col=0, low_memory=False)

flatfiles["volcanic"] = pd.read_csv(cfg["raw_data"]["gm_flatfiles"] / "volcanic_lanzanoluzi_flatfile.csv", 
                                    delimiter=";", index_col=0)

# load the preprocessed gm database
gm_database = pd.read_csv(fp_gm_database, sep=",", low_memory=False, header=[0, 1])

# create the average depth map for each TRT
average_depths = {
    "Craton": flatfiles["asc"]["ev_depth_km"].mean(),
    "Non-Subduction Deep": flatfiles["vran"]["ev_depth_km"].mean(),
    "Shallow Default": flatfiles["asc"]["ev_depth_km"].mean(),
    "Subduction Inslab": flatfiles["sinter"]["ev_depth_km"].mean(),
    "Subduction Interface": flatfiles["sinter"]["ev_depth_km"].mean(),
    "Volcanic": flatfiles["volcanic"]["ev_depth_km"].mean(),
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

occurence = True    # the record selection should be performed based on occurence

#################### Selection Parameters ######################################

t_lower = 0.025     # lower SA period considered in selection
t_upper = 3         # upper SA period considered in selection
n_periods = 20      # number of periods to consider in selection

conditioning_imt: IMT = AvgSA([0,3])                      
nonSA_imts: list[IMT] = [AvgSA([0,3]), RSD595(), PGA()] 
sa_periods = np.round(np.geomspace(t_lower, t_upper, num=n_periods), 3)
SA_imts: list[IMT] = [SA(period) for period in sa_periods]
selection_imts: list[IMT] = nonSA_imts[1:] + SA_imts
nonSA_imt_strs: list[str] = [im.string for im in nonSA_imts] # strings match the correlation matrix

# weights of the IMs
weight_rsd595 = 0.25
n_other_ims = len([imt for imt in selection_imts if imt.name == "SA" or imt.name == "PGA"])
imt_weights = np.array([(1-weight_rsd595) / n_other_ims if imt.name != "RSD595" 
                        else weight_rsd595 for imt in selection_imts])
imt_weights /= imt_weights.sum()

# some other things
disagg_type = "TRT_Mag_Dist_Eps"
percentiles = [0.05, 0.16, 0.5, 0.84, 0.95]     # percentiles of the gcim distribution to return  
assumed_rake = -90                              # assumed rake for RSD595 calculation

# filter the gm_database so that only the selection and conditioning ims are present
gm_db = gm_database.copy()
updated_ims = mf.filter_gm_database_on_imts(
    gm_db["ims"], selection_imts + [conditioning_imt])
updated_ims.columns = pd.MultiIndex.from_product([['ims'], updated_ims.columns])
gm_db = pd.concat([gm_db.drop('ims', axis=1, level=0), updated_ims], axis=1)
   
# Create the GMM Map for AvgSA by reading the logic tree
gmm_map = create_gmm_map(fp_gmm_logic_tree)

# get the correlation model map
corr_map = create_corr_model_map(nonSA_imt_strs, sa_periods)

# organise the disagg data (flat IML-based structure from nb 021:
# disagg_data[site][imt][iml] -> df). Pick the (site_id, poe) of interest by
# resolving each iml's poe from disagg_stats (see get_poe_from_disaggstats).
from phd_project.scripts.WP1_ground_motion_set.gm_selection import get_poe_from_disaggstats
poe_disagg = None
for iml, df in disagg_data[site_id][conditioning_imt.name].items():
    p = get_poe_from_disaggstats(disagg_stats, site_id, conditioning_imt.name, iml)
    if p is not None and np.isclose(p, poe):
        poe_disagg = df
        break
if poe_disagg is None:
    raise ValueError(f"No disagg for site {site_id} at poe {poe}; "
                     f"choose a poe present in disagg_stats for this site.")


# set up the selection context:
site_params = site_condition_data.loc[site_id,:].to_dict()
site_specific_selection_ctx = {
    "n_ensembles": 20,
    "n_samples": 30,
    "conditioning_imt": conditioning_imt ,
    "disagg_imt": conditioning_imt.name , # this only works for AvgSA. otherwise used .string 
    "selection_imts": selection_imts ,
    "imt_weights": imt_weights ,
    "site_params": site_params ,
    "ctx_builder": ESHM20SiteRupCtxBuilder ,
    "ctx_builder_params": ["average_depths", "assumed_rake"] ,
    "average_depths": average_depths ,
    "assumed_rake": assumed_rake ,
    "gmm_map": gmm_map ,
    "corr_map": corr_map ,
    "m_bound_model": "tarbali_and_bradley_2016" ,
    "d_bound_model": None,# "tarbali_and_bradley_2016" ,
    "vs30_bound_model": "tarbali_and_bradley_2016" ,
    "sf_bounds": None,#(0.25, 4),
    "usable_T": t_upper ,
    "max_n_recs": 5 ,
    "p_value": 0.05 ,
    "ok_trt_matches": OK_TRT_MATCHES ,
    "occurence": True ,
}

rng_seed = 1
LOAD_GCIM_IF_EXISTS = True
LOAD_RS_IF_EXISTS = True
LOAD_OPTIMISED_ENSEMBLES_IF_EXISTS = True


#################### GCIM Distributions ########################################

if LOAD_GCIM_IF_EXISTS: # load the gcim distributions instead of 
    no_file = False
    if fp_gcim_dists.is_file():
        with open(fp_gcim_dists, "rb") as file:
            gcim_dists = pickle.load(file)
        print("Existing GCIM distribution data loaded...")
    else:
        no_file = True
        print("No existing GCIM distribution data found...")

#################### Preliminary Selection #####################################
gcim_cdfs = gcim_dists[(site_id, poe)]["cdfs"]
imt = site_specific_selection_ctx["disagg_imt"]

iml = get_imtl_from_disaggstats(
    disagg_stats, site_id, imt, poe)

print("Start selection...")
candidate_ensembles = get_record_ensembles_for_single_poe_distribution(
            poe_disagg, iml, site_specific_selection_ctx, gm_db, 
            gcim_cdfs, rng_seed)
print("Finished selection.")

# get the best of the candidate ensembles and save
obj_func = total_ks_statistic_and_failing_im_penalty

if candidate_ensembles == []:
    # no ensemble was found
    preliminary_ensemble = None
    print(f"No preliminary ensembles found for site {site_id}, poe = {poe}")

else:

    target_cdfs = gcim_dists[(site_id, poe)]["cdfs"]
    ks_bounds = ensemble_ks_bounds(
        target_cdfs, 
        site_specific_selection_ctx["n_samples"],
        site_specific_selection_ctx["p_value"])

    obj_func_kwargs = {
        "target_cdfs_list": [np.column_stack([np.log(ksb[1:,0]), ksb[1:,2]]) 
                            for ksb in ks_bounds.values()],
        "upper_ks_bounds_list": [np.column_stack([np.log(ksb[1:,0]), ksb[1:,3]]) 
                                for ksb in ks_bounds.values()],
        "lower_ks_bounds_list": [np.column_stack([np.log(ksb[1:,0]), ksb[1:,1]]) 
                                for ksb in ks_bounds.values()],
        "n_recs": site_specific_selection_ctx["n_samples"],
        "penalty_constant": 10
    }

    preliminary_ensemble = get_best_ensemble_in_list(
        candidate_ensembles, site_specific_selection_ctx["conditioning_imt"].string, 
        obj_func, obj_func_kwargs)

if preliminary_ensemble is not None:
    if preliminary_ensemble["ks_passed"]:
        print(f"OK! - Preliminary ensemble passes for all IMs")
    else:
        print(f"Sub-Optimal! - Preliminary ensemble fails for some IMs:")
        print(preliminary_ensemble["ks_failed_ims"])

...