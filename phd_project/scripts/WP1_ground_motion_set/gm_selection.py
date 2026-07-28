import pickle
from pathlib import Path

from sympy import im
from openquake.hazardlib.imt import IMT, AvgSA
import numpy as np
from pickagm.typing import TRT 
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Callable
from collections import Counter
from openquake.hazardlib.gsim.base import registry
from openquake.hazardlib.imt import SA, RSD595, AvgSA, PGA
from tqdm.autonotebook import tqdm

from pickagm.avgSA import indirect_AvgSA_GMPE
import pickagm.eqdbases as eqdb
import pickagm.eqdbases as dfops
from pickagm.corrmodels import (
    CORR_MODELS
)
from pickagm.distributions import (
    gcim_distributions,
    gcim_simulation, 
    ensemble_ks_test, 
    ensemble_r_score,
    ensemble_ks_bounds,
    ensemble_ecdfs,
    ensemble_quantiles
    )
from pickagm.selection import (
    preselection, 
    scale_records_in_db,
    filter_on_sf,
    filter_on_m,
    filter_on_d,
    filter_on_vs30,
    filter_by_usable_T,
    calculate_sse_scores,
    pick_records,
    set_weights,
    valid_selection
    )

from pickagm.corrmodels import (
    CORR_MODELS
)

from phd_project.scripts.cache_utils import (
    fingerprint,
    load_or_compute,
    write_manifest,
    StaleCacheError,
)

import phd_project.scripts.WP1_ground_motion_set.manage_flatfiles as mf
from phd_project.scripts.oqhelpers import parse_nrml_logic_tree

from phd_project.config.config import load_config 
cfg = load_config()


type_map = {
    float: "f4",
    int: "i4",
    bool: "?",
    str: "U50"
}


class ESHM20SiteRupCtxBuilder:
    
    def __init__(self, site_params: dict[str, float|bool],
                 depths: dict[TRT, float], assumed_rake: float, 
                 average_ztor: dict[TRT, float]):
        
        self.site_params = site_params
        self.depths = depths
        self.assumed_rake = assumed_rake
        self.average_ztor = average_ztor
        

    def ctx(self, rup_params: dict) -> np.recarray:

        site_rup_params = self.site_params | rup_params | {
             "hypo_depth": self.depths[rup_params["trt"]],
             "rake": self.assumed_rake,
             "ztor": self.average_ztor[rup_params["trt"]],
             "hypo_lat": self.site_params["lat"] + 0.1, # dummy -> same as lat + 0.1°
             "hypo_lon": self.site_params["lon"] + 0.1, # dummy -> same as lon + 0.1°
        }
    
        site_rup_params["rrup"] = np.sqrt(site_rup_params["rjb"]**2 + 
                                          site_rup_params["hypo_depth"]**2) # TODO:: Document this assumption
        site_rup_params["rhypo"] = np.sqrt(site_rup_params["rjb"]**2 + 
                                           site_rup_params["hypo_depth"]**2) # TODO:: Document this assumption

        dynamic_dtype = []
        for key, value in site_rup_params.items():
            if isinstance(value, (float, np.float32, np.float64)):
                value = float(value)
            py_type = type(value)
            np_type = type_map.get(py_type, "O")  # Default to 'O' (Object) if unknown
            dynamic_dtype.append((key, np_type))

        
        # 4. Initialize the recarray
        ctx = np.recarray(1, dtype=dynamic_dtype)

        for key, value in site_rup_params.items():
            ctx[0][key] = value

        return ctx


def SA_PGA_gmm_from_logic_tree(LT: dict, trt: str):
    ignore_tags = ["avg_periods", "corr_func"]
    params = list(deepcopy(LT[trt]).values())[0]["params"]
    for tag in ignore_tags:
        params.pop(tag, None)
    gmpe_name = params.pop("gmpe_name")
    gmm = registry[gmpe_name](**params)
    return gmm


def AvgSA_gmm_from_logic_tree(LT: dict, trt: str):
        params = list(deepcopy(LT[trt]).values())[0]["params"]
        gmpe_name = params.pop("gmpe_name")
        periods = params.pop("avg_periods")
        corr_func = params.pop("corr_func")
        rho_total = CORR_MODELS[corr_func](incl_SA_Ts = periods).rho["total"]
        corr_mat = rho_total.to_numpy()
        base_gmm = registry[gmpe_name](**params)
        gmm = indirect_AvgSA_GMPE(base_gmm, corr_mat, avg_periods=periods)
        return gmm


def create_gmm_map(gmm_logic_tree_fp):
    GMM_AvgSA_LT = parse_nrml_logic_tree(gmm_logic_tree_fp)
    gmm_map = {
        "Craton": {
            # the key strings need to match the imt.name so that they be found when the imt is up to be calculated.
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Craton"),         
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Craton"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Craton"),
            "RSD595": registry["BahrampouriEtAldm2021Asc"](),
            "IA": registry["BahrampouriEtAl2021Asc"](),
            },
        "Non-Subduction Deep": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "RSD595": registry["BahrampouriEtAldm2021SSlab"](),
            "IA": registry["BahrampouriEtAl2021SSlab"]()
            },
        "Shallow Default": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "RSD595": registry["BahrampouriEtAldm2021Asc"](),
            "IA": registry["BahrampouriEtAl2021Asc"]()
            },
        "Subduction Inslab": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "RSD595": registry["BahrampouriEtAldm2021SSlab"](),
            "IA": registry["BahrampouriEtAl2021SSlab"]()
            },
        "Subduction Interface": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "RSD595": registry["BahrampouriEtAldm2021SInter"](),
            "IA": registry["BahrampouriEtAl2021SInter"]()
            },
        "Volcanic": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "RSD595": registry["BahrampouriEtAldm2021Asc"](),
            "IA": registry["BahrampouriEtAl2021Asc"]()
            }
    }
    return gmm_map


def create_corr_model_map(nonSA_imts: list[str], sa_periods):
    corr_model_map = {
        "Craton": CORR_MODELS["clemett_asc"](nonSA_imts, sa_periods).rho["total"],
        "Non-Subduction Deep": CORR_MODELS["clemett_vrancea"](nonSA_imts, sa_periods).rho["total"],
        "Shallow Default": CORR_MODELS["clemett_asc"](nonSA_imts, sa_periods).rho["total"],
        "Subduction Inslab": CORR_MODELS["clemett_sslab"](nonSA_imts, sa_periods).rho["total"],
        "Subduction Interface": CORR_MODELS["clemett_sinter"](nonSA_imts, sa_periods).rho["total"],
        "Volcanic": CORR_MODELS["clemett_asc"](nonSA_imts, sa_periods).rho["total"],
    }
    return corr_model_map


def rename_non_sa_columns_esm(df):
    columns = []
    for c in df.columns:
        try:
            new_col = (c[0], eqdb.esm_nonSA_im_translations[c[1]])
        except KeyError:
            new_col = c
        columns.append(new_col)
    df.columns = pd.MultiIndex.from_tuples(columns)
    return df


def select_ensembles(
        disagg_dst: pd.DataFrame,
        target_gcim_cdfs: dict,
        df_of_records: pd.DataFrame,
        n_ensembles: int,
        n_samples: int,
        conditioning_imt: IMT, 
        conditioning_iml: float,
        selection_imts: list[IMT],
        imt_weights: list[float],
        gmm_map: dict,
        corr_model_map: dict,
        ctx_builder: ESHM20SiteRupCtxBuilder,
        magnitude_bounds: tuple|None,
        distance_bounds: tuple|None, 
        vs30_bounds: tuple|None,
        sf_bounds: tuple|None,
        usable_T: tuple|None,
        max_records_from_one_event: int,
        p_value: float,
        ok_trt_matches: dict[str, list[str]],
        rng_seed: int):
    """ selects ensembles of records for a given hazard dissaggregation"""
    
    df = df_of_records
    occurence = True
    rup_rng = np.random.default_rng(rng_seed)
    sim_rng = np.random.default_rng(rng_seed)

    # selection stats
    filter_stats = {
        "m_bounds": magnitude_bounds,
        "d_bounds": distance_bounds,
        "vs30_bounds": vs30_bounds,
        "sf_bounds": sf_bounds,
        "usable_T_bound": usable_T,
        "N_recs": len(df),       # total number of records
    }
    
    df = filter_on_m(df, magnitude_bounds)
    filter_stats["N_recs|m"] = len(df)
    filter_stats["% N_recs|m"] = filter_stats["N_recs|m"] / filter_stats["N_recs"]

    df = filter_on_d(df, distance_bounds)
    filter_stats["N_recs|m,d"] = len(df)
    filter_stats["% N_recs|m,d"] = filter_stats["N_recs|m,d"] / filter_stats["N_recs"]
    
    df = filter_on_vs30(df, vs30_bounds)
    filter_stats["N_recs|m,d,v"] = len(df)
    filter_stats["% N_recs|m,d,v"] = filter_stats["N_recs|m,d,v"] / filter_stats["N_recs"]
    
    df = filter_by_usable_T(df, usable_T)
    filter_stats["N_recs|m,d,v,t"] = len(df)
    filter_stats["% N_recs|m,d,v,t"] = filter_stats["N_recs|m,d,v,t"] / filter_stats["N_recs"]
    
    df = scale_records_in_db(df, conditioning_imt, conditioning_iml)
    df = filter_on_sf(df, sf_bounds)
    filter_stats["N_recs|m,d,v,t,sf"] = len(df)
    filter_stats["% N_recs|m,d,v,t,sf"] = filter_stats["N_recs|m,d,v,t,sf"] / filter_stats["N_recs"]

    weights = set_weights(imt_weights, len(selection_imts))

    gcim_ensembles = []
    for ii in range(n_ensembles):
        # generate the simulations
        sims, mu, sig, ctxs, eps = gcim_simulation(
            n_samples, disagg_dst, conditioning_iml, gmm_map, selection_imts, 
            conditioning_imt, corr_model_map, ctx_builder, occurence,
            rup_rng, sim_rng)
        
        sim_trts = list(ctxs.trt)

        # get scores for each record compared to the simulations
        # here we use sse
        sse_scores = calculate_sse_scores(
            df, sims, sig, weights, conditioning_imt)
        
        # pick records for the ensemble
        df_sel = pick_records(df, sse_scores, max_records_from_one_event,
                              sim_trts, ok_trt_matches)

        # check if the ensemble is full and there are no empty rows.
        
        has_empty_rows = df_sel.isnull().all(axis=1).any()
        if has_empty_rows:
            # empty rows indicate that there were not enough records in the filtered database
            # to allow a full ensemble to be selected.
            continue # skip this ensemble attempt

        # do KS-tests to assess the fit of the data
        ks_passed, ks_results, failed_ims = ensemble_ks_test(
            df_sel["ims_scaled"], target_gcim_cdfs, p_value)       
        rscore = ensemble_r_score(ks_results, weights)
        ks_bounds = ensemble_ks_bounds(target_gcim_cdfs, n_samples, p_value)
        ecdfs = ensemble_ecdfs(df_sel["ims_scaled"])
        quantiles = ensemble_quantiles(df_sel["ims_scaled"])

        ensemble_data = {
            "sims": sims,
            "mu_cond_sims": mu,
            "sig_cond_sims": sig,
            "ctxs": ctxs,
            "eps": eps,
            "recs": df_sel,
            "ks_passed": ks_passed,
            "ks_results": ks_results,
            "ks_failed_ims": failed_ims,
            "ks_bounds": ks_bounds,
            "R-score": rscore,
            "ecdfs": ecdfs,
            "quantiles": quantiles,
            "cond_imt": conditioning_imt.string,
            "cond_iml": conditioning_iml,
            "filter_stats": filter_stats}
        
        gcim_ensembles.append(ensemble_data)

    # if and empty list is returned it means that no suitable ensembles could
    # be found. Generally this is because of a lack or records in the filtered
    # ground motion database.

    return gcim_ensembles

    
def get_ground_motion_ensembles_for_sites(
        site_poe_disaggs: dict, 
        disagg_stats: pd.DataFrame,
        gcim_dists: dict,
        ground_motion_database: pd.DataFrame,
        basic_selection_ctx: dict,
        site_model: pd.DataFrame,
        rng_seed: int,
        only: list[tuple[int, float]]=[],
        desc: str="Selecting GM Ensembles"
        ) -> dict:
    """ 
    site_poe_disaggs is a dictionary with the following levels
    - (site_id: int, poe: float)    -> tuple
       -> pd.DataFrame: exceedence probability of the disaggregations

    disagg_stats is pd.DataFrame containing with columns:
    - ["site_id", "seismicity", "region", "imt", "poe"]
    
    gcim_dists is a dictionary with the following levels
    - (site_id: int, poe: float)    -> tuple
        -> dict: statistics and distributions of the GCIMs
    
    selection_ctx is a dictionary with the following required keys:
    n_ensembles         : int           :
    n_samples           : int           :
    conditioning_imt    : IMT           : 
    disagg_imt          : str           : the string representation of the conditioing imt that is the key in the disaggregation dictionary
    selection_imts      : list[IMT]     :
    imt_weights         : np.ndarray    : weights of the imts the record selection. order corresponds to selection_imts
    sites               : pd.DataFrame  :
    gmm_map             : dict          :
    corr_map            : dict          :
    average_depths      : dict          :
    assumed_rake        : float         :
    m_bound_model       : str|None      : the tag of the model to use e.g. tarbali_and_bradley_2016
    d_bound_model       : str|None      : the tag of the model to use e.g. tarbali_and_bradley_2016
    v30_bound_model     : str|None      : the tag of the model to use e.g. tarbali_and_bradley_2016
    sf_bounds           : tuple         : upper and lower bounds on the scale factor of the records
    usable_T            : float         : all records must have reliable data up to this period
    max_n_recs          : int           : maximum number of records that can be selected from a single event per ensemble
    p_value             : float         : p-value used in the KS test to of the im distributions
    ok_trt_matches      : dict|None     : map of trt types that are allowed to be selected given a simulation has certain trt type
    occurence           : bool          : true if the dissagregation was done for "P(m|X=x)"
    ctx_builder         :                 context builder that builds the site_rup_ctx used by the openquake GMMs
    ctx_builder_params  : list[str]     : list of keys in this dict that should be passed to ctx_builder after site_params e.g. ["average_depths", "assumed_rake"]      

    returns a dictionary with tuple keys:
    - (site_id: int, poe: float)
 
    """

    imt_strings = [imt.string for imt in basic_selection_ctx["selection_imts"]]
    assert all(item in ground_motion_database["ims"].columns 
           for item in imt_strings) , "Not all selection imts are available in the ground motion database"
    imt = basic_selection_ctx["disagg_imt"]

    selections = {}

    if only != []:
        keys = only
    else:
        keys = list(site_poe_disaggs.keys())
    
    ii_loop = tqdm(range(len(keys)), desc=desc)
    for ii in ii_loop:
        site, poe = keys[ii]

        ii_loop.set_postfix(site=site, poe=f"{poe:9.6f}")
        poe_disagg = site_poe_disaggs[(site, poe)]

        site_params = site_model.loc[site,:].to_dict()
        site_selection_ctx = deepcopy(basic_selection_ctx)
        site_selection_ctx["site_params"] = site_params

        gcim_cdfs = gcim_dists[(site, poe)]["cdfs"]
        iml = get_imtl_from_disaggstats(
            disagg_stats, site, imt, poe)
        
        selections[(site, poe)] = get_record_ensembles_for_single_poe_distribution(
            poe_disagg, iml, site_selection_ctx, ground_motion_database, 
            gcim_cdfs, rng_seed
            )

    return selections


def optimise_ground_motion_ensembles_for_sites(
        preliminary_ensembles: dict,
        site_poe_disaggs: dict[tuple[int, float], pd.DataFrame],
        disagg_stats: pd.DataFrame,
        gcim_dists: dict[tuple[int, float], dict],
        ground_motion_database: pd.DataFrame,
        basic_selection_ctx: dict,
        site_model: pd.DataFrame,
        only: list[tuple[int, float]]=[],
        shuffle:bool = False,
        rng_seed: float = 1.0, 
        descr: str = "Optimising GM Ensembles",
        penalty_constant: float = 10
        ):

    obj_func = weighted_ks_statistic_and_L2norm_penalty

    optimised_ensembles = {}
    scores = {}

    if only != []:
        keys = only
    else:
        keys = list(preliminary_ensembles.keys())

    ii_loop = tqdm(range(len(keys)), desc=f"{descr}")
    for ii in ii_loop:
        site, poe = keys[ii]
        ii_loop.set_postfix(site=site, poe=f"{poe:9.6f}")
        pr_ensemble = preliminary_ensembles[(site, poe)]
    
        # if there is no preliminary ensemble it can't be optimised
        if pr_ensemble is None:
            optimised_ensembles[(site, poe)] = None
            continue

        disagg_dist = site_poe_disaggs[(site, poe)]
        target_cdfs = gcim_dists[(site, poe)]["cdfs"]

        site_params = site_model.loc[site,:].to_dict()
        site_selection_ctx = basic_selection_ctx.copy()
        site_selection_ctx["site_params"] = site_params

        obj_func_kwargs = _optimise_obj_kwargs(
            site, poe, gcim_dists, basic_selection_ctx, penalty_constant)

        # do the optimisation
        cond_iml = disagg_stats[(disagg_stats["site_id"] == site) &
                                (disagg_stats["poe"] == poe)]["imtl"].iloc[0]
        
        new_recs, score = greedy_optimise_ensemble(
            pr_ensemble, cond_iml, disagg_dist, ground_motion_database, 
            site_selection_ctx, obj_func, obj_func_kwargs, is_nested=True,
            shuffle=shuffle, rng_seed=rng_seed)

        new_ensemble = assemble_ensemble_dict(
            new_recs, target_cdfs, site_selection_ctx, 
            basic_selection_ctx["conditioning_imt"].string, cond_iml)
        
        optimised_ensembles[(site, poe)] = new_ensemble
        scores[(site, poe)] = score
    
    return optimised_ensembles, scores


def optimise_ground_motion_ensembles_for_sites_with_shuffles(
        n_shuffles: int,
        rng_seeds: list[int],
        preliminary_ensembles: dict,
        site_poe_disaggs: dict[tuple[int, float], pd.DataFrame],
        disagg_stats: pd.DataFrame,
        gcim_dists: dict[tuple[int, float], dict],
        ground_motion_database: pd.DataFrame,
        basic_selection_ctx: dict,
        site_model: pd.DataFrame,
        only: list[tuple[int, float]]=[],
    ):
    
    assert len(rng_seeds) >= n_shuffles, "The number of seeds must be >= the number of shuffles"

    ensembles = []
    scores = []
    for ii, rng_seed in enumerate(rng_seeds):
        es, ss = optimise_ground_motion_ensembles_for_sites(
            preliminary_ensembles, site_poe_disaggs, disagg_stats,
            gcim_dists, ground_motion_database, basic_selection_ctx, site_model, only, 
            shuffle=True, rng_seed=rng_seed, descr=f"Shuffle {ii+1} - Optimising Ensemble")
        ensembles.append(es)
        scores.append(ss)

    score_tuples = {
        key: tuple(d[key] for d in scores) 
        for key in scores[0].keys()
        }

    best_ensembles = {}
    best_scores = {}
    for site_poe, scores in score_tuples.items():
        best_score = min(scores)
        idx = scores.index(best_score)
        best_ensemble = ensembles[idx][site_poe]
        best_ensembles[site_poe] = best_ensemble
        best_scores[site_poe] = best_score

    return best_ensembles, best_scores


def _initialise_ctx_builder(site_selection_ctx):
    # initialise the context builder
    ctx_builder = site_selection_ctx["ctx_builder"](
        site_selection_ctx["site_params"], 
        *[site_selection_ctx[p] for p in site_selection_ctx["ctx_builder_params"]])
    return ctx_builder


def get_record_ensembles_for_single_poe_distribution(
        disagg_dst, conditioning_iml, site_selection_ctx, ground_motion_database, 
        gcim_cdfs, rng_seed):
    
    # initialise the context builder
    ctx_builder = _initialise_ctx_builder(site_selection_ctx)

    # get m_bounds, dbounds, vs30bounds
    m_bounds = get_m_bounds(
        disagg_dst, site_selection_ctx["occurence"], 
        site_selection_ctx["m_bound_model"])
    
    d_bounds = get_d_bounds(
        disagg_dst, site_selection_ctx["occurence"], 
        site_selection_ctx["d_bound_model"])

    vs30_bounds = get_vs30_bounds(
        site_selection_ctx["site_params"]["vs30"], 
        site_selection_ctx["vs30_bound_model"])

    # select the ensembles                
    poe_ensembles = select_ensembles(
        disagg_dst,
        gcim_cdfs,
        ground_motion_database,
        site_selection_ctx["n_ensembles"],
        site_selection_ctx["n_samples"], 
        site_selection_ctx["conditioning_imt"],
        conditioning_iml,
        site_selection_ctx["selection_imts"], 
        site_selection_ctx["imt_weights"],
        site_selection_ctx["gmm_map"],
        site_selection_ctx["corr_map"],
        ctx_builder,
        m_bounds, 
        d_bounds, 
        vs30_bounds,
        site_selection_ctx["sf_bounds"],
        site_selection_ctx["usable_T"],
        site_selection_ctx["max_n_recs"],
        site_selection_ctx["p_value"],
        site_selection_ctx["ok_trt_matches"],
        rng_seed)
    
    return poe_ensembles


def calculate_gcim_distributions_for_sites(
        site_poe_disaggs: dict, disagg_stats: pd.DataFrame, 
        site_model: pd.DataFrame, basic_selection_ctx: dict, 
        percentiles: list[int]) -> dict:
    """ 
    site_poe_disaggs is a dictionary with tuple keys (site_id: int, poe: float)
        -> exceedence probability of the disaggregations

    disagg_stats is pd.DataFrame containing with columns:
    - ["site_id", "seismicity", "region", "imt", "poe"]
    
    basic_selection_ctx is a dictionary with the following required keys:
    conditioning_imt    : IMT           
                        : the imt that the disaggregation is conditioned on. 
                        : this is needed to extract the correct imtl from the 
                        : disagg_stats and to calculate the gcim distributions
    
    selection_imts      : list[IMT]     
                        : the imts for which the gcim distributions should be 
                        : calculated.
    
    gmm_map             : dict[str, dict[str, GMM]] 
                        : map of gmm objects for each tectonic region and imt. 
                        : the keys should be the same as the trt types in the 
                        : disagg_stats and the imts in selection_imts
    
    corr_map            : dict[str, pd.DataFrame] 
                        : map of correlation matrices for each tectonic region. 
                        : the keys should be the same as the trt types in the 
                        : disagg_stats and the columns and index should be the 
                        : same as the imts in selection_imts
    ctx_builder         :                 
                        : context builder that builds the site_rup_ctx used by 
                        : the openquake GMMs. this is needed to calculate the 
                        : gcim distributions
    
    ctx_builder_params  : list[str]     
                        : list of keys in this dict that should be passed to 
                        : ctx_builder after site_params 
                        : e.g. ["average_depths", "assumed_rake", "average_ztor"]
    
    occurence           : bool          
                        : true if the dissagregation was done for "P(m|X=x)"
    
    returns a dictionary with the tuple keys:
    - (site_id: int, poe: float) 
        -> dict: GCIM distribution statistics and distributions"""
    
    bsc = basic_selection_ctx 

    gcim_dists = {}
    keys = list(site_poe_disaggs.keys())

    ii_loop = tqdm(range(len(keys)), desc="Calculating GCIM Distributions")
    for ii in ii_loop:
        site, poe = keys[ii]
        ii_loop.set_postfix(site=site, poe=f"{poe:9.6f}")
        
        # initialise the context builder
        site_params = site_model.loc[site,:].to_dict()
        site_selection_ctx = deepcopy(basic_selection_ctx)
        site_selection_ctx["site_params"] = site_params
        ctx_builder = _initialise_ctx_builder(site_selection_ctx)
        
        poe_disagg = site_poe_disaggs[(site, poe)]

        imtl = get_imtl_from_disaggstats(
            disagg_stats, site, bsc["conditioning_imt"].name, poe)
            
        gcim_stats, gcim_pdfs, gcim_cdfs = gcim_distributions(
            poe_disagg, imtl, bsc["gmm_map"], bsc["selection_imts"], 
            bsc["conditioning_imt"], bsc["corr_map"], ctx_builder, 
            bsc["occurence"], percentiles)
            
        gcim_dists[(site, poe)] = {
            "stats": gcim_stats, 
            "pdfs": gcim_pdfs, 
            "cdfs": gcim_cdfs}

    return gcim_dists


def get_m_bounds(disagg_dst: pd.DataFrame, occurence: bool, model="tarbali_and_bradley_2016") -> tuple:
    if model == None:
        return (None, None)
    elif model == "tarbali_and_bradley_2016":
        return m_bounds_tarbali_and_bradley_2016(disagg_dst, occurence)


def get_d_bounds(disagg_dst: pd.DataFrame, occurence: bool, model="tarbali_and_bradley_2016") -> tuple:
    if model == None:
        return None
    elif model == "tarbali_and_bradley_2016":
        return d_bounds_tarbali_and_bradley_2016(disagg_dst, occurence)


def get_vs30_bounds(site_vs30, model="tarbali_and_bradley_2016") -> tuple:
    if model == None:
        return None
    elif model == "tarbali_and_bradley_2016":
        return vs30_bounds_tarbali_and_bradley_2016(site_vs30)


def get_disagg_percentiles(disagg_dist, param_key, prob_key, targets=[0.01, 0.1, 0.9, 0.99]):
    srs = disagg_dist.groupby(param_key)[prob_key].sum()
    srs = srs.sort_index()
    # Calculate the cumulative probability
    cdf = srs.cumsum()
    # Normalize CDF if the sum isn't exactly 1.0 due to rounding
    cdf = cdf / srs.sum()
    # Interpolate distances at target cumulative probabilities
    results = np.round(np.interp(targets, cdf, srs.index.to_numpy()), 3)
    
    return dict(zip(targets, results))


def m_bounds_tarbali_and_bradley_2016(
        disagg_dst: pd.DataFrame, occurence: bool,) -> tuple[float, float]:
    
    param_key = "Mag"
    if occurence:
        prob_key = "P(m|X=x)"
    else:
        raise NotImplementedError
    
    pctiles = get_disagg_percentiles(disagg_dst, param_key, prob_key)
    lower = min(pctiles[0.01], pctiles[0.1] - 0.5)
    upper = max(pctiles[0.99], pctiles[0.90] + 0.5)

    return (lower, upper)


def d_bounds_tarbali_and_bradley_2016(
        disagg_dst: pd.DataFrame, occurence: bool,) -> tuple[float, float]:
    
    param_key = "Dist"
    if occurence:
        prob_key = "P(m|X=x)"
    else:
        raise NotImplementedError
    
    pctiles = get_disagg_percentiles(disagg_dst, param_key, prob_key)
    lower = min(pctiles[0.01], 0.5 * pctiles[0.1])
    upper = max(pctiles[0.99], 1.5 * pctiles[0.90])

    return (lower, upper)


def vs30_bounds_tarbali_and_bradley_2016(
        site_vs30: float) -> tuple[float, float]:
    return (0.5 * site_vs30, 1.5 * site_vs30)


def get_imtl_from_disaggstats(disagg_stats, site_id, imt, poe):
    imtl = disagg_stats[(disagg_stats["site_id"] == site_id) &
                        (disagg_stats["imt"] == imt) &
                        (disagg_stats["poe"] == poe)]["imtl"].values[0]
    return imtl


def get_poe_from_disaggstats(disagg_stats, site_id, imt, imtl):
    """Return the poe for (site_id, imt, imtl), or None if there is no matching
    stats row (e.g. a zero-hazard / excluded IML). Returns None rather than
    raising so a requested-but-unusable IML can be skipped gracefully by the
    caller. The imtl is matched with a tolerance (np.isclose) because the
    requested value may come from a JSON list with slightly different rounding."""
    row = disagg_stats[(disagg_stats["site_id"] == site_id) &
                       (disagg_stats["imt"] == imt) &
                       (np.isclose(disagg_stats["imtl"], imtl))]
    if row.empty:
        return None
    return float(row["poe"].values[0])


def adjust_ensemble_distributions(
        disagg_dist: pd.DataFrame, 
        target_gcim_cdfs: dict, ensemble: dict, selection_ctx: dict,
        gm_db: pd.DataFrame):
    # takes an ensemble failing the ks-test and trys to tweak the selected
    # records so that the distribution passes.
    # it is not an optimisation algorithm.
    # It will stop when the all the distributions are passing the KS-test
    # or when ... 

    # the ensemble dict must have the following keys as a minimum:
    # ...

    # the selection_ctx must have the following keys as a minimum:
    # ...

    recs = ensemble["recs"]
    current_recs = recs.copy()
    current_ks_results = ensemble["ks_results"]
    current_rscore = ensemble["R-score"]
    current_ecdfs = ensemble["ecdfs"]
    
    # calculate KS-bounds incase the number of samples or p-value has changed
    # in the seleection context
    ks_bounds = ensemble_ks_bounds(target_gcim_cdfs, selection_ctx["n_samples"],
                                   selection_ctx["p_value"])
    
    # initialise the context builder
    site_params = selection_ctx["site_params"]

    # get m_bounds, dbounds, vs30bounds
    m_bounds = get_m_bounds(
        disagg_dist, selection_ctx["occurence"], 
        selection_ctx["m_bound_model"])
    
    d_bounds = get_d_bounds(
        disagg_dist, selection_ctx["occurence"], 
        selection_ctx["d_bound_model"])

    vs30_bounds = get_vs30_bounds(
        site_params["vs30"], 
        selection_ctx["vs30_bound_model"])
    
    # get the filtered gm_database
    gm_db = preselection(gm_db, m_bounds, d_bounds, vs30_bounds, selection_ctx["usable_T"])
    gm_db = scale_records_in_db(gm_db, selection_ctx["conditioning_imt"], ensemble["cond_iml"])
    gm_db = filter_on_sf(gm_db, selection_ctx["sf_bounds"])

    # get the failing ims and order from lowest to highest p-value
    failing_ims = ensemble["ks_failed_ims"]

    # set some flags
    passing = ensemble["ks_passed"]
    hail_mary_ims = []
    updated_at_least_once = False
    update_failed = False
    failed_point = None

    while (not passing) and (not update_failed):
        updated = False
        im = failing_ims[0]
        
        # calculate determine the imls exceeding the ks bounds
        lower_exceedances = ks_lower_bound_exceedances(
            current_ecdfs[im], ks_bounds[im][:, [0,1]])
        
        upper_exceedances = ks_upper_bound_exceedances(
            current_ecdfs[im], ks_bounds[im][:, [0,3]])

        # calculate the sign of the ecdf relative to the cdf
        im_signs = _assign_signs_from_cdf_array(
            current_ecdfs[im], ks_bounds[im][:, [0,2]])
        
        if lower_exceedances != []:
            # select the iml to improve
            iml, diff = lower_exceedances[0]
            target_iml = iml + diff

            # find the records that can be replaced:
            # must have an iml higher than the iml we want to improve
            # and lie on the same side of the cdf, here 1 (right-hand side)
            ok_replacement_ims = [i for i, sign in im_signs if sign == 1 and i > target_iml]
            ok_replacement_recs = current_recs[current_recs[("ims_scaled", im)].isin(ok_replacement_ims)]\
                .copy()\
                .sort_values(("ims_scaled", im), axis=0, ascending=True)
            
            # get a mask for the imls
            # < used because we want to move an upper part of the ecdf down to 
            # the lower end
            gm_db_iml_mask = gm_db[("ims_scaled", im)] < target_iml
            
        elif upper_exceedances != []:
            # select the iml to improve
            iml, diff = upper_exceedances[0]
            target_iml = iml + diff

            # find the records that can be replaced:
            # must have an iml lower than the iml we want to improve
            # and lie on the same side of the cdf, here -1 (left-hand side)
            ok_replacement_ims = [i for i, sign in im_signs if sign == -1 and i < target_iml]
            ok_replacement_recs = current_recs[current_recs[("ims_scaled", im)].isin(ok_replacement_ims)]\
                .copy()\
                .sort_values(("ims_scaled", im), ascending=False)
            
            # get a mask for the imls
            # > used because we want to move an lower part of the cdf up to the
            # upper end
            gm_db_iml_mask = gm_db[("ims_scaled", im)] > target_iml
            
        else:
            # if we get here then neither the upper or lower ks bounds have been
            # exceeded. But the last round throught the while loop should have 
            # yielded no failing ims and passing == True. 
            # Technically we should never get here...
            raise NotImplementedError()

        # print(im, iml, diff)

        # do the loops
        for _, rec_to_replace in ok_replacement_recs.iterrows():
            rec_trt = rec_to_replace[("metadata", "trt")]

            # filter the gm_database by trt and iml:
            filtered_gm_db = gm_db[(gm_db[("metadata", "trt")] == rec_trt) &
                                    gm_db_iml_mask]
        
            # get a new record and try the fit. returns none if no improvement was possible
            res = _loop_to_improve_im_and_rscore(
                rec_to_replace, current_recs, current_rscore, filtered_gm_db,
                selection_ctx, target_gcim_cdfs)
        
            if res is not None:
                # record found that improves r-score
                (passing, current_recs, current_rscore, current_ecdfs, 
                current_ks_results, failing_ims, updated) = res

                # the ecdf has changed. we need to break out
                break
        
        if updated:
            # go back to start of the while loop and check for new exceedance
            # points
            updated_at_least_once = True
            continue

        # if we get here we weren't able to improve the rscore by replacing
        # any of the records that are allowed to be replaced.
        # Now we try allowing records to be selected that don't improve the 
        # R-Score but don't allow any new IMs to fail either 
        # (a bit of a looser requirement)
        for _, rec_to_replace in ok_replacement_recs.iterrows():
            rec_trt = rec_to_replace[("metadata", "trt")]

            # filter the gm_database by trt and iml:
            filtered_gm_db = gm_db[(gm_db[("metadata", "trt")] == rec_trt) &
                                    gm_db_iml_mask]

            res = _loop_to_improve_im_without_others_failing(
                rec_to_replace, current_recs, failing_ims, filtered_gm_db,
                selection_ctx, target_gcim_cdfs)

            if res is not None:
                # record found that improves im distribution without causing others
                # to fail
                (passing, current_recs, current_rscore, current_ecdfs, 
                current_ks_results, failing_ims, updated) = res
                # the ecdf has changed. we need to break out
                break

        if updated:
            # go back to start of the while loop and check for new exceedance
            # points
            updated_at_least_once = True
            continue

        # if we get here we weren't able to improve the im without causing
        # others to fail. 
        failed_point = (im, iml, diff) # the im iml combination that can't be made to work
        update_failed = True
    
    if not updated_at_least_once:
        return ensemble["ks_passed"], ensemble, failed_point
    
    # update new ensemble  
    new_ensemble = deepcopy(ensemble)
    new_ensemble["recs"] = current_recs
    new_ensemble["ks_passed"] = passing
    new_ensemble["ks_results"] = current_ks_results
    new_ensemble["ks_failed_ims"] = failing_ims
    new_ensemble["R-score"] = current_rscore
    new_ensemble["ecdfs"] = current_ecdfs
    quantiles = ensemble_quantiles(current_recs["ims_scaled"])
    new_ensemble["quantiles"] = quantiles
    new_ensemble["ks_bounds"] = ks_bounds

    return passing, new_ensemble, failed_point


def _loop_to_improve_im_and_rscore(
        rec_to_replace: pd.Series, current_recs: pd.DataFrame, 
        current_rscore: float, gm_db: pd.DataFrame, 
        selection_ctx: dict, target_gcim_cdfs: dict):
    
    for ii in range(len(gm_db)):
        new_rec = gm_db.iloc[ii, :] # the new record to try

        if not valid_selection(new_rec, current_recs, selection_ctx["max_n_recs"]):
            continue

        test_out = _test_new_record(
            rec_to_replace, new_rec, current_recs, selection_ctx, target_gcim_cdfs) 
        
        passing = test_out[1]

        # check that the R-score has improved:
        if test_out[4] >= current_rscore:
            continue # no improvement

        # update the failing_ims and recs
        new_recs = test_out[0]
        new_rscore = test_out[4]
        new_ecdfs = ensemble_ecdfs(new_recs["ims_scaled"])
        new_ks_results = test_out[2]
        failing_ims = test_out[3]
        updated = True
        return (passing, new_recs, new_rscore, new_ecdfs, 
                new_ks_results, failing_ims, updated)
    
    return None


def _loop_to_improve_im_without_others_failing(
        rec_to_replace: pd.Series, 
        current_recs: pd.DataFrame, 
        failing_ims: list[str],
        gm_db: pd.DataFrame, 
        selection_ctx: dict, 
        target_gcim_cdfs: dict):
    
    for ii in range(len(gm_db)):
        new_rec = gm_db.iloc[ii, :] # the new record to try

        if not valid_selection(new_rec, current_recs, selection_ctx["max_n_recs"]):
            continue

        test_out = _test_new_record(
            rec_to_replace, new_rec, current_recs, selection_ctx, target_gcim_cdfs) 
        
        passing = test_out[1]

        if _no_new_failing_ims(test_out[3], failing_ims) != []:
                # some new im failed and we dismiss this record
                continue

        # update the failing_ims and recs
        new_recs = test_out[0]
        new_rscore = test_out[4]
        new_ecdfs = ensemble_ecdfs(new_recs["ims_scaled"])
        new_ks_results = test_out[2]
        failing_ims = test_out[3]
        updated = True
        return (passing, new_recs, new_rscore, new_ecdfs, 
                new_ks_results, failing_ims, updated)
    
    return None


def _test_new_record(
        rec_to_replace: pd.Series, 
        new_rec: pd.Series, 
        new_recs: pd.DataFrame,
        selection_ctx: dict, 
        target_gcim_cdfs: dict):
    
    # temporarily replace the record 
    idx_pos = new_recs.index.get_loc(rec_to_replace.name)
    temp_recs = pd.concat([new_recs.iloc[:idx_pos], 
                            new_rec.to_frame().T, 
                            new_recs.iloc[idx_pos + 1:]])
    temp_recs = temp_recs.astype(new_recs.dtypes)

    # recompute the KS tests for all IMs 
    passing, temp_ks_results, temp_failed_ims = ensemble_ks_test(
    temp_recs["ims_scaled"], target_gcim_cdfs, selection_ctx["p_value"])

    # calculate the R-Score
    temp_rscore = ensemble_r_score(
        temp_ks_results, selection_ctx["imt_weights"])
    
    return temp_recs, passing, temp_ks_results, temp_failed_ims, temp_rscore 


def _no_new_failing_ims(list_a, list_b):
    """Returns elements in A that are not in B."""
    # Convert B to a set for O(1) lookup performance
    set_b = set(list_b)
    
    # Use a list comprehension to filter A
    return [item for item in list_a if item not in set_b]


def _assign_signs_from_cdf_array(ecdf, cdf):
    """
    Assigns -1 or 1 to ECDF x-values based on position relative to 
    a theoretical CDF provided as an array.
    """
    # 1. Filter ECDF for the 'leading edge' (min y for each unique x)
    # Sort by x ascending, then y ascending
    ecdf_sorted = ecdf[np.lexsort((ecdf[:, 1], 
                                   ecdf[:, 0]))]
    _, indices = np.unique(ecdf_sorted[:, 0], return_index=True)
    ecdf_filtered = ecdf_sorted[indices]
    
    x_ecdf = ecdf_filtered[:, 0]
    y_ecdf = ecdf_filtered[:, 1]
    
    # 2. Extract CDF array columns
    x_cdf = cdf[:, 0]
    y_cdf = cdf[:, 1]
    
    # 3. Interpolate the theoretical x for the given ecdf y-values
    # Note: np.interp(x_eval, x_data, y_data) 
    # To get x as a function of y, we swap the data columns:
    x_theoretical_at_y = np.interp(y_ecdf, y_cdf, x_cdf)
    
    # 4. Assign signs
    # x_ecdf > x_theory means the observation is larger than expected (Right)
    # Using np.where(condition, if_true, if_false)
    signs = np.where(x_ecdf >= x_theoretical_at_y, 1, -1)
    
    # Create list of tuples (x_value, sign)
    return list(zip(x_ecdf, signs))


def ks_lower_bound_exceedances(ecdf: np.ndarray, ks_bound: np.ndarray):
    # Identify each unique x and select the pair with the minimum y value. 
    # This ensures we are comparing the "leading edge" of each step against the 
    # Kolmogorov-Smirnov (KS) lower bound.
    
    # Sort by x then y to ensure the first occurrence is the minimum y
    ecdf_sorted = ecdf[np.lexsort((ecdf[:, 1], ecdf[:, 0]))]

    # Identify unique x-values and keep only the first occurrence (min y)
    _, indices = np.unique(ecdf_sorted[:, 0], return_index=True)
    ecdf_filtered = ecdf_sorted[indices]
    
    # ECDF: col 0 is x, col 1 is y
    x_ecdf = ecdf_filtered[:, 0]
    y_ecdf = ecdf_filtered[:, 1]

    # KS Bound: col 0 is x, col 1 is y
    x_ks = ks_bound[:, 0]
    y_ks = ks_bound[:, 1]

    # Interpolate KS x-values for each y-value present in the ECDF
    x_ks_interped = np.interp(y_ecdf, y_ks, x_ks)

    # if the last y_ks = 1 then we have an upper bound case. We need to fix
    # the interpolation otherwise it jumps to the last value with y = 1.0
    if np.round(y_ks[-1], 6) == 1:
        x_max = x_ks[np.where(np.round(y_ks, 6) == 1)[0][0]]
        x_ks_interped[-1] = x_max

    diffs = x_ks_interped - x_ecdf
    exceedances = [(x, d) for x, d in zip(x_ecdf, diffs) if d < 0]

    return exceedances


def ks_upper_bound_exceedances(ecdf: np.ndarray, ks_bound: np.ndarray):
    """
    Finds points where ECDF x-values exceed the KS upper bound.
    Exceedance occurs when x_ecdf < x_ks (ECDF is to the left of the bound).
    Diff is calculated as x_ks - x_ecdf.
    """
    # Sort by x ascending, then y descending to grab max y per x
    ecdf_sorted = ecdf[np.lexsort((-ecdf[:, 1], ecdf[:, 0]))]

    # Identify unique x-values and keep the first occurrence (max y)
    _, indices = np.unique(ecdf_sorted[:, 0], return_index=True)
    ecdf_filtered = ecdf_sorted[indices]
    
    x_ecdf = ecdf_filtered[:, 0]
    y_ecdf = ecdf_filtered[:, 1]

    # KS Bound: col 0 is x, col 1 is y
    x_ks = ks_bound[:, 0]
    y_ks = ks_bound[:, 1]

    # Interpolate KS x-values for the ECDF y-levels
    x_ks_interped = np.interp(y_ecdf, y_ks, x_ks)

    # Handle the boundary case where y reaches 1.0
    if np.round(y_ks[-1], 6) == 1.0:
        # Find the very first x-value where the KS bound hits 1.0
        idx_first_one = np.where(np.round(y_ks, 6) == 1.0)[0][0]
        x_max_ks = x_ks[idx_first_one]
        
        # Apply this x_max to all interpolated points where ECDF is at 1.0
        mask_one = np.round(y_ecdf, 6) == 1.0
        x_ks_interped[mask_one] = x_max_ks

    # In an upper bound context, a positive value means x_ecdf < x_ks
    diffs = x_ks_interped - x_ecdf
    
    # Return (x_ecdf, x_ks, diff) for exceedances (where diff > 0)
    exceedances = [(xe, d) for xe, d in zip(x_ecdf, diffs) if d > 0]

    return exceedances


def greedy_optimise_ensemble(
        ensemble: dict,
        iml: float,
        disagg_dist: pd.DataFrame, 
        gm_database: pd.DataFrame,
        selection_ctx: dict,
        objective_function: Callable,
        obj_func_kwargs: list,
        is_nested:bool=False, 
        shuffle:bool=False,
        rng_seed:int=1):
    
    cond_imt = selection_ctx["conditioning_imt"].string
           
    # get m_bounds, dbounds, vs30bounds
    m_bounds = get_m_bounds(
        disagg_dist, selection_ctx["occurence"], 
        selection_ctx["m_bound_model"])
    
    d_bounds = get_d_bounds(
        disagg_dist, selection_ctx["occurence"], 
        selection_ctx["d_bound_model"])

    site_params = selection_ctx["site_params"]
    vs30_bounds = get_vs30_bounds(
        site_params["vs30"], 
        selection_ctx["vs30_bound_model"])
    
    # get the filtered gm_database
    filtered_gm_db = preselection(gm_database, m_bounds, d_bounds, vs30_bounds, selection_ctx["usable_T"])
    filtered_gm_db = scale_records_in_db(filtered_gm_db, selection_ctx["conditioning_imt"], iml)
    filtered_gm_db = filter_on_sf(filtered_gm_db, selection_ctx["sf_bounds"])
    
    if shuffle:
        gm_db = filtered_gm_db.sample(frac=1, random_state=rng_seed)
    else:
        gm_db = filtered_gm_db

    rec_event_map = gm_db[("metadata", "event_id")].to_dict()
    rec_event_station_map = {
        idx: (row[("metadata", "event_id")], row[("metadata", "station_code")]) 
        for idx, row in gm_db.iterrows()
    }

    # presort the gm database by trt. Also track the index so I know what 
    # record was selected
    gm_db_trt_groups = {}
    for trt, group in gm_db.groupby(("metadata", "trt")):
        gm_db_trt_groups[trt] = {
            "ims": np.log(group["ims_scaled"].drop(columns=cond_imt).to_numpy()),
            "indices": group.index.values # Store the original index
        }

    # set up the database and some other stuff
    original_ensemble_indices = ensemble["recs"].index.tolist()

    current_ims = np.log(ensemble["recs"]["ims_scaled"].drop(columns=cond_imt).to_numpy().copy())
    current_score = objective_function(current_ims, **obj_func_kwargs)
    current_ensemble_indices = original_ensemble_indices.copy()

    # counter to track the number of records selection from each event
    current_event_counts = Counter([rec_event_map[idx] for idx in current_ensemble_indices])
    
    # counter to track the number of records selection from each event
    current_events_and_stations = [
        (row[("metadata", "event_id")], row[("metadata", "station_code")]) 
        for _, row in ensemble["recs"].iterrows()
        ]
    
    # for each record in ensemble
    ii_loop = tqdm(range(len(ensemble["recs"])), desc="Optimizing Ensemble Slots",
                   leave=not is_nested)
    for ii in ii_loop:
        # get database of gms filtered by sf and trt etc
        rec_trt = ensemble["recs"].iloc[ii][("metadata", "trt")]
        
        # get the event ID of the record to replace
        old_event_id = rec_event_map[current_ensemble_indices[ii]]

        if rec_trt not in gm_db_trt_groups:
            raise KeyError

        gm_pool_ims = gm_db_trt_groups[rec_trt]["ims"]
        gm_pool_indices = gm_db_trt_groups[rec_trt]["indices"]
        
        # current best ims for this slot
        best_ims_for_ii = current_ims[ii, :].copy()

        for jj, gm_idx in enumerate(gm_pool_indices):
            
            # skip record if a record with that event and station is already
            # in the ensemble
            if rec_event_station_map[gm_idx] in current_events_and_stations:
                continue
            
            # skip record if adding it exceeds the max count of records
            # from a single event
            new_event_id = rec_event_map[gm_idx]
            if new_event_id != old_event_id:
                if current_event_counts[new_event_id] >= selection_ctx["max_n_recs"]:
                    continue
             
            # update current ims with the new record
            current_ims[ii, :] = gm_pool_ims[jj, :]
            
            test_score = objective_function(current_ims, **obj_func_kwargs)

            if test_score < current_score:
                current_score = test_score

                # update the event counter
                current_event_counts[old_event_id] -= 1
                current_event_counts[new_event_id] += 1

                # update the event and station tracker              
                current_events_and_stations[ii] = rec_event_station_map[gm_idx]

                # Update our tracker with the index from gm_db
                current_ensemble_indices[ii] = gm_pool_indices[jj]
                best_ims_for_ii = gm_pool_ims[jj, :].copy()
                old_event_id = new_event_id
                ii_loop.set_postfix(score=f"{current_score:.4f}")
            else:
                # Replace the row with the current best IM values if no improvement
                current_ims[ii, :] = best_ims_for_ii

    
    # reconstruct the dataframe
    final_recs = gm_db.loc[current_ensemble_indices]
    
    return final_recs, current_score


def sse_point_stats(rec_ims, stat_types, target_stats, im_weights, stat_weights):
    """
    rec_ims and target_stats should be provided in log units if accurate results
    are desired.

    rec_ims: (M, N) matrix of linear-space IM values (e.g., Sa in g)
    stat_types: (X,) list of metrics to use to calculate ["mean", "sigma", quantiles as floats between 0-1]
    target_stats: (N, X) matrix of stats for the target distributions function
    im_weights: (N,) vector
    stat_weights: (X,) vector of weights corresponding to the stats in stat_types
    """   
    
    assert len(stat_types) == len(stat_weights), "stats and weights must have the same length"
    
    # Calculate sample stats in log-space
    stats = []
    for s in stat_types:
        if s == "mean":
            stats.append(np.mean(rec_ims, axis=0))
        elif s == "sigma":
            stats.append(np.std(rec_ims, ddof=1, axis=0)) # sample standard deviation
        elif isinstance(s, float):
            stats.append(np.quantile(rec_ims, s, axis=0))
   
    sample_stats = np.column_stack(stats)
    
    # Weighted SSE
    # (N, 4) matrix of squared errors
    sq_err = (target_stats - sample_stats)**2
    
    # normalise weights
    im_weights = np.array(im_weights) / sum(im_weights) 
    stat_weights = np.array(stat_weights) / sum(stat_weights)

    # Weighted sum across stats, then across IMs
    weighted_im_err = np.sum(sq_err * stat_weights, axis=1)
    total_sse = np.sum(weighted_im_err * im_weights)
    
    return total_sse


def weighted_ks_statistic(
        rec_ims, target_cdfs_list, im_weights, n_recs):
    """
    rec_ims and target_cdfs should be provided in log units if accurate results
    are desired.

    rec_ims: (n_recs, n_imts) numpy array
    target_cdfs_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    im_weights: (n_imts,) numpy array
    """
    # sort all IMTs at once along the record axis (axis 0)
    sorted_ims = np.sort(rec_ims, axis=0)
    
    # pre-calculate the ECDF steps (i/n and (i-1)/n). constant for a fixed ensemble size
    steps_up = np.linspace(1.0/n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs-1.0)/n_recs, n_recs).reshape(-1, 1)
    
    total_weighted_ks = 0.0
    
    for i in range(len(target_cdfs_list)):
        xp = target_cdfs_list[i][:,0]
        fp = target_cdfs_list[i][:,1]
        
        # interpolate to get the values of the target that match the x values of
        # the cdf. This calculates F(xi) for the whole column at once
        # assumes linear interpolation.
        f_xi = np.interp(sorted_ims[:, i], xp, fp, left=0, right=1).reshape(-1, 1)
        
        # Calculate D+ and D- (the two possible maximum distances)
        d_plus = np.max(np.abs(steps_up - f_xi))
        d_minus = np.max(np.abs(steps_down - f_xi))
        
        ks_stat = max(d_plus, d_minus)
        total_weighted_ks += im_weights[i] * ks_stat
        
    return total_weighted_ks


def total_ks_statistic(
        rec_ims, target_cdfs_list, n_recs):
    """
    rec_ims and target_cdfs should be provided in log units if accurate results
    are desired.

    rec_ims: (n_recs, n_imts) numpy array
    target_cdfs_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    im_weights: (n_imts,) numpy array
    """
    # sort all IMTs at once along the record axis (axis 0)
    sorted_ims = np.sort(rec_ims, axis=0)
    
    # pre-calculate the ECDF steps (i/n and (i-1)/n). constant for a fixed ensemble size
    steps_up = np.linspace(1.0/n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs-1.0)/n_recs, n_recs).reshape(-1, 1)
    
    total_ks = 0.0
    
    for i in range(len(target_cdfs_list)):
        xp = target_cdfs_list[i][:,0]
        fp = target_cdfs_list[i][:,1]
        
        # interpolate to get the values of the target that match the x values of
        # the cdf. This calculates F(xi) for the whole column at once
        # assumes linear interpolation.
        f_xi = np.interp(sorted_ims[:, i], xp, fp, left=0, right=1).reshape(-1, 1)
        
        # Calculate D+ and D- (the two possible maximum distances)
        d_plus = np.max(np.abs(steps_up - f_xi))
        d_minus = np.max(np.abs(steps_down - f_xi))
        
        ks_stat = max(d_plus, d_minus)
        total_ks += ks_stat
        
    return total_ks


def weighted_ks_statistic_and_failing_im_penalty(
        rec_ims, target_cdfs_list, upper_ks_bounds_list, lower_ks_bounds_list, 
        im_weights, n_recs, penalty_constant):
    """
    rec_ims and target_cdfs should be provided in log units if accurate results
    are desired.

    rec_ims: (n_recs, n_imts) numpy array
    target_cdfs_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    im_weights: (n_imts,) numpy array
    """
    # sort all IMTs at once along the record axis (axis 0)
    sorted_ims = np.sort(rec_ims, axis=0)
    
    # pre-calculate the ECDF steps (i/n and (i-1)/n). constant for a fixed ensemble size
    steps_up = np.linspace(1.0/n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs-1.0)/n_recs, n_recs).reshape(-1, 1)
    
    total_weighted_ks = 0.0
    
    failed_im_count = 0
    for i in range(len(target_cdfs_list)):
        xp = target_cdfs_list[i][:,0]
        fp = target_cdfs_list[i][:,1]
        fp_upper = upper_ks_bounds_list[i][:,1]
        fp_lower = lower_ks_bounds_list[i][:,1]
        
        # interpolate to get the values of the target that match the x values of
        # the cdf. This calculates F(xi) for the whole column at once
        # assumes linear interpolation.
        f_xi = np.interp(sorted_ims[:, i], xp, fp, left=0, right=1).reshape(-1, 1)
        f_upper_xi = np.interp(sorted_ims[:, i], xp, fp_upper).reshape(-1, 1)
        f_lower_xi = np.interp(sorted_ims[:, i], xp, fp_lower).reshape(-1, 1)
        
        # is im failing?
        is_failing = (np.any(steps_up > f_upper_xi) or 
                      np.any(steps_down < f_lower_xi))
        
        if is_failing:
            failed_im_count += 1
        # Calculate D+ and D- (the two possible maximum distances)
        d_plus = np.max(np.abs(steps_up - f_xi))
        d_minus = np.max(np.abs(steps_down - f_xi))
        
        ks_stat = max(d_plus, d_minus)

        total_weighted_ks += im_weights[i] * ks_stat

    objective_value = (failed_im_count * penalty_constant) + total_weighted_ks
        
    return objective_value


def total_ks_statistic_and_failing_im_penalty(
        rec_ims, target_cdfs_list, upper_ks_bounds_list, lower_ks_bounds_list, 
        n_recs, penalty_constant):
    """
    rec_ims and target_cdfs should be provided in log units if accurate results
    are desired.

    rec_ims: (n_recs, n_imts) numpy array
    target_cdfs_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    """
    # sort all IMTs at once along the record axis (axis 0)
    sorted_ims = np.sort(rec_ims, axis=0)
    
    # pre-calculate the ECDF steps (i/n and (i-1)/n). constant for a fixed ensemble size
    steps_up = np.linspace(1.0/n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs-1.0)/n_recs, n_recs).reshape(-1, 1)
    
    total_ks = 0.0
    
    failed_im_count = 0
    for i in range(len(target_cdfs_list)):
        xp = target_cdfs_list[i][:,0]
        fp = target_cdfs_list[i][:,1]
        fp_upper = upper_ks_bounds_list[i][:,1]
        fp_lower = lower_ks_bounds_list[i][:,1]
        
        # interpolate to get the values of the target that match the x values of
        # the cdf. This calculates F(xi) for the whole column at once
        # assumes linear interpolation.
        f_xi = np.interp(sorted_ims[:, i], xp, fp, left=0, right=1).reshape(-1, 1)
        f_upper_xi = np.interp(sorted_ims[:, i], xp, fp_upper).reshape(-1, 1)
        f_lower_xi = np.interp(sorted_ims[:, i], xp, fp_lower).reshape(-1, 1)
        
        # is im failing?
        is_failing = (np.any(steps_up > f_upper_xi) or 
                      np.any(steps_down < f_lower_xi))
        
        if is_failing:
            failed_im_count += 1
        # Calculate D+ and D- (the two possible maximum distances)
        d_plus = np.max(np.abs(steps_up - f_xi))
        d_minus = np.max(np.abs(steps_down - f_xi))
        
        ks_stat = max(d_plus, d_minus)

        total_ks += ks_stat

    objective_value = (failed_im_count * penalty_constant) + total_ks
        
    return objective_value


def weighted_ks_statistic_and_L2norm_penalty(
        rec_ims, target_cdfs_list, upper_ks_bounds_list, lower_ks_bounds_list, 
        im_weights, n_recs, penalty_constant):
    """
    This objective function scores an ensemble based on the sum weighted KS-statistics
    of each IM. A penalty based the sum of the squared distances between the ecdf
    and the ks bounds is used to prioritise swaps that reduce the number of failing
    ims. There is no penalty for IMs that are inside the ks bounds
    
    rec_ims and target_cdfs should be provided in log units if accurate results
    are desired.

    rec_ims: (n_recs, n_imts) numpy array
    target_cdfs_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    upper_ks_bounds_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    lower_ks_bounds_list: A list of 2D numpy arrays for each IMT. [xp_values, fp_values]
    im_weights: (n_imts,) numpy array,
    n_recs: int,
    penalty_constant: float
    """
    # sort all IMTs at once along the record axis (axis 0)
    sorted_ims = np.sort(rec_ims, axis=0)
    
    # pre-calculate the ECDF steps (i/n and (i-1)/n). constant for a fixed ensemble size
    steps_up = np.linspace(1.0/n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs-1.0)/n_recs, n_recs).reshape(-1, 1)
    
    total_weighted_ks = 0.0
    total_violation = 0.0
    
    for i in range(len(target_cdfs_list)):

        if im_weights[i] == 0:
            continue # skip if this im is not weighted

        xp = target_cdfs_list[i][:,0]
        fp = target_cdfs_list[i][:,1]
        fp_upper = upper_ks_bounds_list[i][:,1]
        fp_lower = lower_ks_bounds_list[i][:,1]
        
        # interpolate to get the values of the target that match the x values of
        # the cdf. This calculates F(xi) for the whole column at once
        # assumes linear interpolation.
        f_xi = np.interp(sorted_ims[:, i], xp, fp, left=0, right=1).reshape(-1, 1)
        f_upper_xi = np.interp(sorted_ims[:, i], xp, fp_upper).reshape(-1, 1)
        f_lower_xi = np.interp(sorted_ims[:, i], xp, fp_lower).reshape(-1, 1)
        
        # Calculate KS-statistics D+ and D- (the two possible maximum distances)
        d_plus = np.max(np.abs(steps_up - f_xi))
        d_minus = np.max(np.abs(steps_down - f_xi))
        ks_stat = max(d_plus, d_minus)

        upper_violation = np.sum(np.maximum(0, steps_up - f_upper_xi) ** 2)
        lower_violation = np.sum(np.maximum(0, f_lower_xi - steps_down) ** 2)
        
        total_violation += (upper_violation + lower_violation)
        total_weighted_ks += im_weights[i] * ks_stat

    objective_value = (total_violation * penalty_constant) + total_weighted_ks
        
    return objective_value


def get_best_ensemble_in_list(
        ensembles: list[dict],
        conditioning_imt: str, 
        obj_func: Callable, 
        obj_func_kwargs: dict) -> dict:
    # assumes the im values are in linear scale and converts the ensemble
    # ims to log units.

    current_ensemble = ensembles[0]
    rec_ims = current_ensemble["recs"]["ims_scaled"].drop(columns=conditioning_imt).to_numpy()
    current_score = obj_func(np.log(rec_ims), **obj_func_kwargs)

    for e in ensembles[1:]:
        rec_ims = e["recs"]["ims_scaled"].drop(columns=conditioning_imt).to_numpy()
        score = obj_func(np.log(rec_ims), **obj_func_kwargs)
        if score < current_score:
            current_score = score
            current_ensemble = e

    return current_ensemble


def assemble_ensemble_dict(new_recs, target_gcim_cdfs, site_selection_ctx,
                    cond_imt, cond_iml):
    # do a ks test
    ks_passed, ks_results, failed_ims = ensemble_ks_test(
        new_recs["ims_scaled"], target_gcim_cdfs, site_selection_ctx["p_value"])       
    ks_bounds = ensemble_ks_bounds(target_gcim_cdfs, site_selection_ctx["n_samples"], site_selection_ctx["p_value"])
    ecdfs = ensemble_ecdfs(new_recs["ims_scaled"])
    quantiles = ensemble_quantiles(new_recs["ims_scaled"], qs=[0.05, 0.16, 0.33, 0.5, 0.67, 0.84, 0.9])
    mu_lnIM_recs = pd.Series(np.exp(np.mean(np.log(new_recs["ims_scaled"]), axis=0)), index=new_recs["ims_scaled"].columns)
    sig_lnIM_recs = pd.Series(np.std(np.log(new_recs["ims_scaled"]), axis=0, ddof=1), index=new_recs["ims_scaled"].columns)

    # create the ensemble dict
    new_ensemble = {}
    new_ensemble["recs"] = new_recs
    new_ensemble["ks_passed"] = ks_passed
    new_ensemble["ks_results"] = ks_results
    new_ensemble["ks_failed_ims"] = failed_ims
    new_ensemble["ks_bounds"] = ks_bounds
    new_ensemble["ecdfs"] = ecdfs
    new_ensemble["quantiles"] = quantiles
    new_ensemble["mu_lnIM_recs"] = mu_lnIM_recs
    new_ensemble["sig_lnIM_recs"] = sig_lnIM_recs
    new_ensemble["cond_imt"] = cond_imt
    new_ensemble["cond_iml"] = cond_iml

    return new_ensemble


def preliminary_record_selection(
        site_poe_disaggs: dict,
        disagg_stats: dict,
        gcim_dists: dict,
        gm_db: pd.DataFrame,
        basic_selection_ctx: dict,
        site_model: pd.DataFrame,
        rng_seed: int,
        preliminary_selection_fp: Path|None,
        only_select: list[tuple]=[],
        load_rs_if_exists: bool=False,
        penalty_constant: float=10,
        input_fingerprint: dict|None=None,
        force_recompute: bool=False,
        desc: str|None=None,
        quiet: bool=False):

    """
    performs the preliminary record selection step of the workflow.
    This involves selecting a set of candidate ensembles for each site and poe,
    then selecting the best ensemble for each site and poe based on the
    weighted KS statistic and a penalty for failing IMs. The results are
    saved to a file if preliminary_selection_fp is provided.

    Caching: if ``input_fingerprint`` is provided, the preliminary ensembles are
    loaded/saved through :func:`cache_utils.load_or_compute`, which verifies that
    the cached file was produced by the same inputs and raises
    :class:`cache_utils.StaleCacheError` otherwise (override with
    ``force_recompute=True``). This is the preferred path. If
    ``input_fingerprint`` is None the legacy ``load_rs_if_exists`` behaviour is
    used (no provenance check).
    """

    # candidate_ensembles is only available when the selection is actually
    # computed (not when a cached result is loaded); preserve that semantics.
    candidate_ensembles = None

    def _compute_preliminary():
        nonlocal candidate_ensembles
        # select multiple candidtate ensembles for all sites
        candidate_ensembles = get_ground_motion_ensembles_for_sites(
            site_poe_disaggs, disagg_stats, gcim_dists,
            gm_db, basic_selection_ctx, site_model, rng_seed, only_select,
            desc=desc if desc is not None else "Selecting GM Ensembles")

        # get the best of the candidate ensembles for each site and poe
        obj_func = weighted_ks_statistic_and_L2norm_penalty

        preliminary_ensembles = {}

        for (site, poe), ensembles in candidate_ensembles.items():

            if ensembles == []:
                # no ensemble was found
                preliminary_ensembles[(site, poe)] = None
                continue

            obj_func_kwargs = _optimise_obj_kwargs(
                        site, poe, gcim_dists, basic_selection_ctx, penalty_constant)

            e = get_best_ensemble_in_list(
                ensembles, basic_selection_ctx["conditioning_imt"].string,
                obj_func, obj_func_kwargs)
            preliminary_ensembles[(site, poe)] = e

        return preliminary_ensembles

    if input_fingerprint is not None and preliminary_selection_fp is not None:
        # Provenance-checked cache (preferred path).
        preliminary_ensembles = load_or_compute(
            preliminary_selection_fp, input_fingerprint, _compute_preliminary,
            force_recompute=force_recompute)

    else:
        # Legacy path: load on fixed filename without provenance check.
        no_file = False
        if load_rs_if_exists:
            if preliminary_selection_fp.is_file():
                with open(preliminary_selection_fp, "rb") as file:
                    preliminary_ensembles = pickle.load(file)
                print("Existing record selection data loaded...")
            else:
                no_file = True
                print("No existing record selection data found...")

        if (not load_rs_if_exists) or no_file or preliminary_selection_fp is None:
            preliminary_ensembles = _compute_preliminary()
            # Save the results
            if preliminary_selection_fp is not None:
                with open(preliminary_selection_fp, "wb") as file:
                    pickle.dump(preliminary_ensembles, file)

    # Check if all the sites and poes found a set of suitable ground motions
    prelim_ensembles_not_passing = []
    no_preliminary_ensembles = []
    for (site, poe), ensemble in preliminary_ensembles.items():
        
        if ensemble is None:
            prelim_ensembles_not_passing.append((site, poe))
            no_preliminary_ensembles.append((site, poe))
        
        elif not ensemble["ks_passed"]:
            prelim_ensembles_not_passing.append((site, poe))

    if not quiet:
        if len(prelim_ensembles_not_passing) == 0:
            print(f"OK! - Preliminary ensembles pass for all combinations of site and poe")
        else:
            print(f"Sub-Optimal! - Preliminary ensembles do not pass for {len(prelim_ensembles_not_passing)} combinations of site and poe")

        if no_preliminary_ensembles:
            print(f"No preliminary ensembles found for {len(no_preliminary_ensembles)} combinations of site and poe")

    return preliminary_ensembles, prelim_ensembles_not_passing, no_preliminary_ensembles, candidate_ensembles


def optimise_record_selection(
        site_poe_disaggs: dict,
        disagg_stats: dict,
        gcim_dists: dict,
        gm_db: pd.DataFrame,
        basic_selection_ctx: dict,
        site_model: pd.DataFrame,
        preliminary_ensembles: int,
        optimised_selection_fp: Path|None,
        only_optimise: list[tuple]=[],
        shuffle: bool=False,
        rng_seed: int=1,
        description: str="",
        load_rs_if_exists: bool=False,
        penalty_constant: float=10,
        input_fingerprint: dict|None=None,
        force_recompute: bool=False,
        n_shuffles: int=1,
        rng_seeds: list[int]|None=None,
        quiet: bool=False):

    def _compute_optimised():
        # do the ensemble optimisation
        if shuffle and n_shuffles > 1:
            # re-optimise over several shuffled DB orderings and keep the
            # best-scoring ensemble per (site, poe) (used by the round-4 retry).
            seeds = rng_seeds if rng_seeds is not None else list(range(1, n_shuffles + 1))
            optimised_ensembles, _ = optimise_ground_motion_ensembles_for_sites_with_shuffles(
                n_shuffles, seeds, preliminary_ensembles, site_poe_disaggs,
                disagg_stats, gcim_dists, gm_db, basic_selection_ctx, site_model,
                only_optimise)
        else:
            optimised_ensembles, _ = optimise_ground_motion_ensembles_for_sites(
                preliminary_ensembles, site_poe_disaggs, disagg_stats,
                gcim_dists, gm_db, basic_selection_ctx, site_model, only_optimise,
                penalty_constant=penalty_constant, shuffle=shuffle,
                rng_seed=rng_seed, descr=description)
        return optimised_ensembles

    if input_fingerprint is not None and optimised_selection_fp is not None:
        # Provenance-checked cache (preferred path).
        optimised_ensembles = load_or_compute(
            optimised_selection_fp, input_fingerprint, _compute_optimised,
            force_recompute=force_recompute)

    else:
        # Legacy path: load on fixed filename without provenance check.
        no_file = False
        if load_rs_if_exists:
            if optimised_selection_fp.is_file():
                with open(optimised_selection_fp, "rb") as file:
                    optimised_ensembles = pickle.load(file)
                print("Existing optimised ensembles loaded...")
            else:
                no_file = True
                print("No existing optimised ensembles found...")

        if (not load_rs_if_exists) or no_file:
            optimised_ensembles = _compute_optimised()
            # Save the results
            with open(optimised_selection_fp, "wb") as file:
                pickle.dump(optimised_ensembles, file)

    # Check if all the sites and poes found a set of suitable ground motions
    optim_ensembles_not_passing = []
    for (site, poe), ensemble in optimised_ensembles.items():

        if ensemble is None:
            optim_ensembles_not_passing.append((site, poe))

        elif not ensemble["ks_passed"]:
            optim_ensembles_not_passing.append((site, poe))

    if not quiet:
        if len(optim_ensembles_not_passing) == 0:
            print(f"OK! - Optimised ensembles pass for all combinations of site and poe")
        else:
            print(f"Sub-Optimal! - Optimised ensembles do not pass for {len(optim_ensembles_not_passing)} combinations of site and poe")

    return optimised_ensembles, optim_ensembles_not_passing


def _ctx_with_unbounded(base_ctx: dict, unbounded: list[str]) -> dict:
    """Copy ``base_ctx`` and set the named causal-parameter bounds to ``None``.

    ``unbounded`` contains any of ``"m"``, ``"d"``, ``"vs30"`` — the corresponding
    ``m_bound_model`` / ``d_bound_model`` / ``vs30_bound_model`` keys are dropped
    (set to ``None``) so that round's selection/optimisation ignores that bound.
    """
    name_map = {"m": "m_bound_model", "d": "d_bound_model", "vs30": "vs30_bound_model"}
    ctx = base_ctx.copy()
    for b in unbounded:
        if b not in name_map:
            raise ValueError(f"unbounded entry must be one of {list(name_map)}, got {b!r}")
        ctx[name_map[b]] = None
    return ctx


def _ensemble_passes(ensemble) -> bool:
    """True if an ensemble exists and passed the KS test."""
    return ensemble is not None and ensemble["ks_passed"]


def _round_select_optimise_unbounded(spec) -> tuple[list[str], list[str]]:
    """Return ``(select_unbounded, optimise_unbounded)`` for one round spec.

    A ``list``/``tuple`` applies the same dropped bounds to both the selection and
    the optimisation step of that round. A ``dict`` ``{"select": [...],
    "optimise": [...]}`` lets the two steps differ — e.g. reselect with only the
    distance bound free but optimise with distance + vs30 free (the AvgSA_06
    scheme). Missing dict keys default to ``[]`` (all bounds kept).
    """
    if isinstance(spec, dict):
        return list(spec.get("select", [])), list(spec.get("optimise", []))
    return list(spec), list(spec)


def _selection_ctx_fingerprint_inputs(ctx: dict) -> dict:
    """Extract the selection-context entries that affect the selected records.

    Returned as a flat dict of plain/hashable values for :func:`fingerprint`.
    Object-valued entries (IMTs) are reduced to their stable ``.string`` form.
    """
    return {
        "n_ensembles": ctx["n_ensembles"],
        "n_samples": ctx["n_samples"],
        "p_value": ctx["p_value"],
        "usable_T": ctx["usable_T"],
        "max_n_recs": ctx["max_n_recs"],
        "m_bound_model": ctx["m_bound_model"],
        "d_bound_model": ctx["d_bound_model"],
        "vs30_bound_model": ctx["vs30_bound_model"],
        "occurence": ctx["occurence"],
        "conditioning_imt": ctx["conditioning_imt"].string,
        "selection_imts": tuple(im.string for im in ctx["selection_imts"]),
        "imt_weights": np.asarray(ctx["imt_weights"]),
    }


def _optimise_obj_kwargs(site, poe, gcim_dists, basic_selection_ctx,
                         penalty_constant: float = 10) -> dict:
    """Build the ``weighted_ks_statistic_and_L2norm_penalty`` kwargs for one
    ``(site, poe)``.

    Shared by :func:`optimise_ground_motion_ensembles_for_sites` and the engine's
    keep-best scoring (:func:`_score_ensemble`) so both evaluate an *identical*
    objective — otherwise the round-4 keep/replace comparison would be meaningless.
    """
    target_cdfs = gcim_dists[(site, poe)]["cdfs"]
    ks_bounds = ensemble_ks_bounds(
        target_cdfs,
        basic_selection_ctx["n_samples"],
        basic_selection_ctx["p_value"])
    im_strings = [im.string for im in basic_selection_ctx["selection_imts"]]
    return {
        "target_cdfs_list": [np.column_stack([np.log(ksb[1:, 0]), ksb[1:, 2]])
                             for k, ksb in ks_bounds.items() if k in im_strings],
        "upper_ks_bounds_list": [np.column_stack([np.log(ksb[1:, 0]), ksb[1:, 3]])
                                 for k, ksb in ks_bounds.items() if k in im_strings],
        "lower_ks_bounds_list": [np.column_stack([np.log(ksb[1:, 0]), ksb[1:, 1]])
                                 for k, ksb in ks_bounds.items() if k in im_strings],
        "n_recs": basic_selection_ctx["n_samples"],
        "im_weights": basic_selection_ctx["imt_weights"],
        "penalty_constant": penalty_constant,
    }


def _score_ensemble(ensemble, site, poe, gcim_dists, basic_selection_ctx) -> float:
    """Objective score for an ensemble (lower is better); ``np.inf`` if missing.

    Uses the same objective as the greedy optimiser so scores are directly
    comparable across rounds for a given ``(site, poe)`` (the target CDFs, and
    hence the objective, depend only on ``(site, poe)`` — not on the round's
    causal-parameter bounds). Used to keep the better of the incumbent and the
    shuffled round-4 result.
    """
    if ensemble is None:
        return np.inf
    kwargs = _optimise_obj_kwargs(site, poe, gcim_dists, basic_selection_ctx)
    cond = basic_selection_ctx["conditioning_imt"].string
    ims = np.log(ensemble["recs"]["ims_scaled"].drop(columns=cond).to_numpy())
    return weighted_ks_statistic_and_L2norm_penalty(ims, **kwargs)


def build_final_ensembles(
        site_poe_disaggs: dict,
        disagg_stats: dict,
        gcim_dists: dict,
        gm_db: pd.DataFrame,
        basic_selection_ctx: dict,
        site_model: pd.DataFrame,
        *,
        source_fps: dict,
        stage_fps: dict,
        output_fp: Path,
        round_unbounded: list[list[str]] = ([], ["d"], ["m", "d", "vs30"]),
        force_optimisation: list[bool] | None = None,
        shuffle: list[bool] | None = None,
        n_shuffles: int = 5,
        shuffle_rng_seeds: list[int] | None = None,
        rng_seed: int = 1,
        force_recompute: bool = False):
    """Run a configurable N-round selection + optimisation pipeline.

    This centralises the orchestration that previously lived in the AvgSA
    selection notebook so the slow compute is reproducible and provenance
    checked. Each round runs two cached steps — a *selection* (only for sites
    that don't yet have a record set) and an *optimisation* — using that round's
    causal-parameter bounds. The assembled ``final_ensembles`` is saved to
    ``output_fp`` with a ``.manifest.json`` sidecar. The returned dict is the
    *pre-post-processing* ``final_ensembles`` (the post-processing notebook adds
    ``record_identifier`` / ``trt_stats`` columns).

    Parameters
    ----------
    source_fps : dict
        Paths used for cheap, content-based input fingerprinting. Keys:
        ``gm_db_file``, ``gcim_file``, ``disagg_data_file``,
        ``disagg_stats_file``, ``site_model_file``.
    stage_fps : dict
        Output pickle paths per round, as ``{"select": [fp_r1, ...],
        "optimise": [fp_r1, ...]}`` (one entry per round).
    output_fp : Path
        Canonical ``*_final_ensembles.pickle`` path.
    round_unbounded : list
        One entry per round (length sets the number of rounds). Each entry is
        either a ``list`` of bounds to drop — any of ``"m"``, ``"d"``, ``"vs30"``,
        ``[]`` keeps all — applied to both that round's selection and
        optimisation, or a ``dict`` ``{"select": [...], "optimise": [...]}`` to
        drop different bounds in the two steps (e.g. AvgSA_06 round 2 reselects
        with distance free but optimises with distance + vs30 free).
    force_optimisation : list[bool] | None
        One bool per round (same length as ``round_unbounded``). For round ``r``:
        ``False`` (the default for every round) optimises only sites still failing
        after that round's selection (a site reselected into a passing state is
        left alone); ``True`` optimises the whole round work-set (passing +
        failing). ``None`` is treated as all ``False``.
    shuffle : list[bool] | None
        One bool per round (same length as ``round_unbounded``). ``True`` re-runs
        that round's optimisation over ``n_shuffles`` shuffled database orderings
        (via :func:`optimise_ground_motion_ensembles_for_sites_with_shuffles`)
        and, per ``(site, poe)``, keeps the best of {the shuffled result, the
        incumbent from earlier rounds} — replacing the incumbent only if the new
        ensemble passes or scores strictly better. This is the round-4 retry that
        tries to escape a bad greedy local optimum for the last stubborn sets.
        ``None`` is treated as all ``False`` (legacy behaviour, unshuffled).
    n_shuffles : int
        Number of shuffled orderings tried per shuffled round (best-of).
    shuffle_rng_seeds : list[int] | None
        Seeds for the shuffles; defaults to ``list(range(1, n_shuffles + 1))``.
    force_recompute : bool
        Recompute and overwrite every stage + the final artifact, ignoring any
        existing cache (bypasses the staleness guard).
    """
    output_fp = Path(output_fp)
    n_rounds = len(round_unbounded)
    round_specs = [_round_select_optimise_unbounded(s) for s in round_unbounded]
    if force_optimisation is None:
        force_optimisation = [False] * n_rounds
    force_optimisation = list(force_optimisation)
    if len(force_optimisation) != n_rounds:
        raise ValueError(f"force_optimisation must have one bool per round "
                         f"({n_rounds}); got {len(force_optimisation)}")

    if shuffle is None:
        shuffle = [False] * n_rounds
    shuffle = list(shuffle)
    if len(shuffle) != n_rounds:
        raise ValueError(f"shuffle must have one bool per round "
                         f"({n_rounds}); got {len(shuffle)}")
    shuffle_seeds = (shuffle_rng_seeds if shuffle_rng_seeds is not None
                     else list(range(1, n_shuffles + 1)))

    all_keys = list(site_poe_disaggs.keys())

    # Base inputs common to every stage. Source files are hashed by their bytes
    # (cheap and stable) rather than re-hashing the large in-memory objects.
    base_inputs = {
        "gm_db_file": source_fps["gm_db_file"],
        "gcim_file": source_fps["gcim_file"],
        "disagg_data_file": source_fps["disagg_data_file"],
        "disagg_stats_file": source_fps["disagg_stats_file"],
        "site_model_file": source_fps["site_model_file"],
        "rng_seed": rng_seed,
    }

    def stage_fingerprint(ctx: dict, stage: str, only: list, **extra) -> dict:
        # Base inputs fully determine every stage's result; ``stage`` + ``only``
        # + the stage's own ctx params (+ any extra, e.g. the force flag)
        # disambiguate the cached artifacts and invalidate a cache when its
        # scope/bounds/flags change.
        return fingerprint(
            **base_inputs, stage=stage, only=tuple(sorted(only)),
            **_selection_ctx_fingerprint_inputs(ctx), **extra,
        )

    def bounds_label(unbounded: list[str]) -> str:
        # Report the full m/d/vs30 universe split into which bounds are enforced
        # (bounded) and which are dropped (unbounded) this round.
        universe = ["m", "d", "vs30"]
        bounded = [b for b in universe if b not in unbounded]
        b = ", ".join(bounded) if bounded else "none"
        u = ", ".join(unbounded) if unbounded else "none"
        return f"bounded: {b} | unbounded: {u}"

    def _count(keys, dct):
        """(#passing, #failing-with-a-set, #no-set) over ``keys`` in ``dct``."""
        passing = failing = no_set = 0
        for k in keys:
            e = dct.get(k)
            if e is None:
                no_set += 1
            elif e["ks_passed"]:
                passing += 1
            else:
                failing += 1
        return passing, failing, no_set

    # Two layers, matching the legacy notebook: ``base`` holds the *selection*
    # results only (preliminary + reselections) and is what every optimise round
    # starts from; ``ensembles`` is the running best (base overwritten by each
    # round's optimised output) and drives the work-set / final result. Optimise
    # rounds are NOT chained — each re-optimises the selection base with its own
    # bounds, so a later round never optimises an earlier round's optimised
    # ensemble. For non-shuffled rounds the latest round's optimised output wins;
    # a shuffled round instead keeps whichever of {new, incumbent} scores better.
    base: dict = {}
    ensembles: dict = {}
    work_set = list(all_keys)
    sel_key_set = set()  # keys (re)selected in the current round (for reporting)

    for r, (sel_unbounded, opt_unbounded) in enumerate(round_specs):
        select_ctx = _ctx_with_unbounded(basic_selection_ctx, sel_unbounded)
        optimise_ctx = _ctx_with_unbounded(basic_selection_ctx, opt_unbounded)
        force = force_optimisation[r]
        shuf = shuffle[r]

        # ---- round header ------------------------------------------------------
        hdr = f"Round {r + 1}/{n_rounds}"
        if shuf:
            hdr += f"  (shuffled DB, best of {n_shuffles})"
        print(f"\n================ {hdr} ================")
        if sel_unbounded == opt_unbounded:
            print(f"Bounds  [{bounds_label(sel_unbounded)}]")
        else:
            print(f"Bounds  select [{bounds_label(sel_unbounded)}]  "
                  f"optimise [{bounds_label(opt_unbounded)}]")

        # ---- SELECTION phase ---------------------------------------------------
        # Round 0 selects everything; later rounds re-select only the work-set
        # sites that still have no record set at all (None selection result).
        if r == 0:
            sel_keys = list(work_set)
            print(f"Selection phase — all {len(sel_keys)} (site, poe) [first round]")
        else:
            sel_keys = [k for k in work_set if base.get(k) is None]
            if sel_keys:
                print(f"Selection phase — reselecting {len(sel_keys)} (site, poe) "
                      f"that had no record set after round {r}")
            else:
                print("Selection phase — none need reselection (every failing "
                      "(site, poe) already has a record set)")
        sel_key_set = set(sel_keys)

        if sel_keys:
            sel = preliminary_record_selection(
                site_poe_disaggs, disagg_stats, gcim_dists, gm_db,
                select_ctx, site_model, rng_seed,
                preliminary_selection_fp=stage_fps["select"][r],
                only_select=sel_keys,
                input_fingerprint=stage_fingerprint(select_ctx, f"select_r{r + 1}", sel_keys),
                desc=f"R{r + 1}/{n_rounds} select ({len(sel_keys)} sites)",
                force_recompute=force_recompute, quiet=True)[0]
            for k in sel_keys:
                base[k] = sel[k]
                ensembles[k] = sel[k]
            p, f_, n0 = _count(sel_keys, ensembles)
            note = f"  (of which {n0} could not form a record set at all)" if n0 else ""
            print(f"    -> {p} pass, {f_ + n0} fail KS{note}")

        # ---- OPTIMISATION phase ------------------------------------------------
        # Work-set computed AFTER selection so reselected sites are included:
        # whole round work-set if forced, else only sites still failing. (None
        # entries are skipped inside the optimiser.)
        if force:
            opt_keys = [k for k in work_set if ensembles.get(k) is not None]
            scope = "force: all still-failing sets that have a record set"
        else:
            opt_keys = [k for k in work_set if not _ensemble_passes(ensembles.get(k))]
            scope = "only sites still failing"
        print(f"Optimisation phase — {len(opt_keys)} (site, poe)  [{scope}]")

        if opt_keys:
            # Describe what the work-set is made of (helps read the counts).
            if r > 0:
                n_resel = sum(1 for k in opt_keys if k in sel_key_set)
                n_carried = len(opt_keys) - n_resel
                parts = []
                if n_carried:
                    parts.append(f"{n_carried} still failing from earlier rounds")
                if n_resel:
                    parts.append(f"{n_resel} just reselected this round")
                if parts:
                    print("    work-set = " + " + ".join(parts))
            n_noset = sum(1 for k in opt_keys if ensembles.get(k) is None)
            if n_noset:
                print(f"    ({n_noset} of these still have no record set -> skipped "
                      f"by the optimiser)")

            # Start from the selection ``base`` (NOT the running best), matching the
            # legacy behaviour where each optimise round re-optimises the prelim /
            # reselected ensemble. Shuffle fields enter the fingerprint only for a
            # shuffled round, so unshuffled rounds keep their existing caches.
            opt_extra = {"force_optimisation": force}
            if shuf:
                opt_extra.update(shuffle=True, n_shuffles=n_shuffles,
                                 rng_seeds=tuple(shuffle_seeds))
            opt, _ = optimise_record_selection(
                site_poe_disaggs, disagg_stats, gcim_dists, gm_db,
                optimise_ctx, site_model, base,
                optimised_selection_fp=stage_fps["optimise"][r],
                only_optimise=opt_keys, shuffle=shuf, rng_seed=rng_seed,
                n_shuffles=(n_shuffles if shuf else 1),
                rng_seeds=(shuffle_seeds if shuf else None),
                input_fingerprint=stage_fingerprint(optimise_ctx, f"optimise_r{r + 1}",
                                                    opt_keys, **opt_extra),
                description=f"R{r + 1}/{n_rounds} optimise ({len(opt_keys)} sites)",
                force_recompute=force_recompute, quiet=True)

            replaced = kept = 0
            for k in opt_keys:
                cand = opt[k]
                if not shuf:
                    ensembles[k] = cand  # latest round wins
                    continue
                # keep-best: only adopt the shuffled result if it passes, there is
                # no incumbent, or it scores strictly better than the incumbent.
                incumbent = ensembles.get(k)
                if _ensemble_passes(cand) or incumbent is None:
                    ensembles[k] = cand
                    replaced += 1
                elif (_score_ensemble(cand, *k, gcim_dists, basic_selection_ctx)
                      < _score_ensemble(incumbent, *k, gcim_dists, basic_selection_ctx)):
                    ensembles[k] = cand
                    replaced += 1
                else:
                    kept += 1

            p, f_, n0 = _count(opt_keys, ensembles)
            print(f"    -> {p} now pass, {f_ + n0} still fail")
            if shuf:
                print(f"      (keep-best: replaced {replaced}, kept incumbent for {kept})")

        # carry the sites that still don't have a passing record set
        work_set = [k for k in all_keys if not _ensemble_passes(ensembles.get(k))]
        if work_set:
            print(f"-> {len(work_set)} (site, poe) carried into the next round")
        else:
            print("-> all (site, poe) now pass")

    final_ensembles = ensembles
    print()

    if work_set:
        print(f"Sub-Optimal! - {len(work_set)} (site, poe) still without a passing "
              f"ensemble after {n_rounds} rounds")
    else:
        print(f"OK! - all (site, poe) have a passing ensemble after {n_rounds} rounds")

    # Save the canonical artifact + manifest.
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fp, "wb") as file:
        pickle.dump(final_ensembles, file)
    final_fp = fingerprint(
        **base_inputs, stage="final_ensembles",
        round_unbounded=tuple((tuple(s), tuple(o)) for s, o in round_specs),
        force_optimisation=tuple(force_optimisation),
        shuffle=tuple(shuffle),
        n_shuffles=n_shuffles,
        shuffle_rng_seeds=tuple(shuffle_seeds),
        **_selection_ctx_fingerprint_inputs(basic_selection_ctx))
    write_manifest(output_fp, final_fp)

    return final_ensembles


if __name__ == "__main__":
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from pickagm.distributions import ensemble_ks_bounds

    from phd_project.config.config import load_config 
    from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
        calculate_gcim_distributions_for_sites,
        get_ground_motion_ensembles_for_sites,
        get_best_ensemble_in_list,
        total_ks_statistic_and_failing_im_penalty,
        optimise_ground_motion_ensembles_for_sites,
        optimise_ground_motion_ensembles_for_sites_with_shuffles,
        preliminary_record_selection
    )

    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
        setup_AvgSA03_gcim_gm_selection,
        )

    cfg = load_config()

        # file paths:
    gcim_dist_fp = cfg["proc_data"]["gcim_dists"] / f"gcim_dist_AvgSA_03.pickle"
    # optimised_selection_rd1_fp = cfg["proc_data"]["gm_selection"] / f"AvgSA_06_optimised_selection_rd01.pickle"
    # optimised_selection_rd2_fp = cfg["proc_data"]["gm_selection"] / f"AvgSA_06_optimised_selection_rd02.pickle"
    # optimised_selection_rd3_fp = cfg["proc_data"]["gm_selection"] / f"AvgSA_06_optimised_selection_rd03.pickle"
    # optimised_selection_rd4_fp = cfg["proc_data"]["gm_selection"] / f"AvgSA_06_optimised_selection_rd04.pickle"

    # other stuff:
    rng_seed = 1

        
        # load the gcim distributions
    if gcim_dist_fp.is_file():
        with open(gcim_dist_fp, "rb") as file:
            gcim_dists = pickle.load(file)
        print("Existing GCIM distribution data loaded...")
    else:
        print("No existing GCIM distribution data found...")
    
    only_select = [(6, 0.002103)]

    # set up the record selection
    site_poe_disaggs, disagg_stats, site_model, basic_selection_ctx, gm_db = setup_AvgSA03_gcim_gm_selection()

    c1_ensembles, prelim_ensembles_not_passing, no_preliminary_ensembles, candidates = preliminary_record_selection(
        site_poe_disaggs, disagg_stats, gcim_dists, gm_db, basic_selection_ctx, site_model, rng_seed, 
        preliminary_selection_fp=None, only_select=only_select, load_rs_if_exists=False)
    ...
    