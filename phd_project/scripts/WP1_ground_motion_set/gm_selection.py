from openquake.hazardlib.imt import IMT, AvgSA
import numpy as np
from pickagm.typing import TRT 
import numpy as np
import pandas as pd
from copy import deepcopy

from openquake.hazardlib.gsim.base import registry
from openquake.hazardlib.imt import SA, RSD595, AvgSA, PGA

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
    ensemble_ks_bounds
    )
from pickagm.selection import (
    preselection, 
    scale_records_in_db,
    filter_on_sf,
    calculate_sse_scores,
    pick_records,
    set_weights
    )

from pickagm.corrmodels import (
    CORR_MODELS
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
                 depths: dict[TRT, float], assumed_rake: float):
        
        self.site_params = site_params
        self.depths = depths
        self.assumed_rake = assumed_rake
        

    def ctx(self, rup_params: dict) -> np.recarray:

        site_rup_params = self.site_params | rup_params | {
             "hypo_depth": self.depths[rup_params["trt"]], # TODO:: check if needs to be done better
             "rake": self.assumed_rake, # TODO:: check if needs to be done better  
        }
    
        site_rup_params["rrup"] = np.sqrt(site_rup_params["rjb"]**2 + 
                                          site_rup_params["hypo_depth"]**2)
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
            },
        "Non-Subduction Deep": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "RSD595": registry["BahrampouriEtAldm2021SSlab"]()
            },
        "Shallow Default": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "RSD595": registry["BahrampouriEtAldm2021Asc"]()
            },
        "Subduction Inslab": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "RSD595": registry["BahrampouriEtAldm2021SSlab"]()
            },
        "Subduction Interface": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "RSD595": registry["BahrampouriEtAldm2021SInter"]()
            },
        "Volcanic": {
            "AvgSA": AvgSA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "RSD595": registry["BahrampouriEtAldm2021Asc"]()
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

    df = preselection(df, magnitude_bounds, distance_bounds, vs30_bounds, usable_T)
    df = scale_records_in_db(df, conditioning_imt, conditioning_iml)
    df = filter_on_sf(df, sf_bounds)

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

        # do KS-tests to assess the fit of the data
        ks_passed, ks_results, failed_ims = ensemble_ks_test(
            df_sel["ims_scaled"], target_gcim_cdfs, p_value)
        # calculate the R-Score
        rscore = ensemble_r_score(ks_results, weights)

        ensemble_data = {
            "sims": sims,
            "mu_cond": mu,
            "sig_cond": sig,
            "ctxs": ctxs,
            "eps": eps,
            "recs": df_sel,
            "ks_passed": ks_passed,
            "ks_results": ks_results,
            "ks_failed_ims": failed_ims,
            "R-score": rscore}
        
        gcim_ensembles.append(ensemble_data)

    return gcim_ensembles


def get_best_ensemble(gcim_ensembles: list[dict], must_pass_ks=True):
    
    if must_pass_ks:
        passing_es = [ii for ii, e in enumerate(gcim_ensembles) if e["ks_passed"]]
        if passing_es == []:
            return None, None
        best_idx = passing_es[0]
        best_e = gcim_ensembles[best_idx]
    
    for ii, e in enumerate(gcim_ensembles[1:]):
        if must_pass_ks:
            if e["R-score"] < best_e["R-score"] and e["ks_passed"]:
                best_e = e
                best_idx = ii+1
        else:
            if e["R-score"] < best_e["R-score"]:
                best_e = e
                best_idx = ii+1

    return best_e, best_idx

    
def get_record_ensembles_for_sites(
        grouped_disagg_data: dict, 
        disagg_stats: pd.DataFrame,
        grouped_gcim_dists: dict,
        ground_motion_database: pd.DataFrame,
        selection_ctx: dict,
        rng_seed: int,
        only_sites:list[int]=[],
        only_poes: list[int]=[] 
        ) -> dict:
    """ 
    grouped_disagg_data is a nested dictionary with the following levels
    - (seismicity: str, region: int)    -> tuple
        - site_id: int
            - poe: float                -> exceedence probability of the disaggregations

    disagg_stats is pd.DataFrame containing with columns:
    - ["site_id", "seismicity", "region", "imt", "poe"]
    
    grouped_gcim_dists is a nested dictionary with the following levels
    - (seismicity: str, region: int)    -> tuple
        - site_id: int
            - poe: float                -> exceedence probability of the disaggregations
    
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

    returns a nested dictionary with the following levels:
    - (seismicity: str, region: int)    -> tuple
        - site_id: int
            - poe: float                -> exceedence probability of the disaggregations
                - selection_results     
    """

    imt = selection_ctx["disagg_imt"]
    
    selections = {}
    print("Selecting Record Ensembles...")
    for (s, r), group_disaggs in grouped_disagg_data.items():
        print(f"  Region {r} - {s} seismicity")
        
        selections_site = {}
        for site_id, site_dissags in group_disaggs.items():
            if site_id not in only_sites and only_sites != []:
                continue

            print(f"    site id: {site_id}")
            
            site_params = selection_ctx["sites"].loc[site_id,:].to_dict()
            
            selections_poe = {}
            for poe, poe_disagg in site_dissags[imt].items():
                if poe not in only_poes and only_poes != []:
                    continue

                print(f"        poe: {poe}")
                gcim_cdfs = grouped_gcim_dists[(s,r)][site_id][poe]["cdfs"]
                iml = get_imtl_from_disaggstats(
                    disagg_stats, site_id, s, r, imt, poe)
                
                selections_poe[poe] = get_record_ensembles_for_single_distribution(
                    poe_disagg, iml, selection_ctx, ground_motion_database, 
                    site_params, gcim_cdfs, rng_seed
                    )
                
            selections_site[site_id] = selections_poe
        selections[(s, r)] = selections_site
    
    return selections


def get_record_ensembles_for_site_and_poe(
        site_id: int,
        poe: float,
        grouped_disagg_data: dict, 
        disagg_stats: pd.DataFrame,
        grouped_gcim_dists: dict,
        ground_motion_database: pd.DataFrame,
        selection_ctx: dict,
        rng_seed: int 
    ):

    imt = selection_ctx["disagg_imt"]
    
    unique_sites = disagg_stats["site_id"].unique()
    unique_poes = disagg_stats["poe"].unique()

    if not poe in unique_poes:
        raise ValueError(f"poe {poe} is not in the disagg_stats data")
    
    if not site_id in unique_sites:
        raise ValueError(f"site_id {site_id} is not in the disagg_stats data")

    print(f"Selecting Record Ensembles for Site {site_id}, poe: {poe}")


    idx = disagg_stats[(disagg_stats["site_id"] == site_id) &
                       (disagg_stats["poe"] == poe)].index[0]
    s = disagg_stats.loc[idx, "seismicity"]
    r = disagg_stats.loc[idx, "region"]

    site_params = selection_ctx["sites"].loc[site_id,:].to_dict()
    poe_disagg = grouped_disagg_data[(s, r)][site_id][imt][poe]

    gcim_cdfs = grouped_gcim_dists[(s,r)][site_id][poe]["cdfs"]
    
    iml = get_imtl_from_disaggstats(
        disagg_stats, site_id, s, r, imt, poe)
    
    selection = get_record_ensembles_for_single_distribution(
        poe_disagg, iml, selection_ctx, ground_motion_database, 
        site_params, gcim_cdfs, rng_seed
        )
                
    return selection


def get_record_ensembles_for_single_site(
        site_id: int,
        grouped_disagg_data: dict, 
        disagg_stats: pd.DataFrame,
        site_gcim_dists: dict,
        ground_motion_database: pd.DataFrame,
        selection_ctx: dict,
        rng_seed: int
    ):

    imt = selection_ctx["disagg_imt"]
    
    unique_sites = disagg_stats["site_id"].unique()
    if not site_id in unique_sites:
        raise ValueError(f"site_id {site_id} is not in the disagg_stats data")

    print(f"Calculating Record Ensembles for Site {site_id}")

    idxs = disagg_stats[(disagg_stats["site_id"] == site_id)].index

    # initialise the context builder
    site_params = selection_ctx["sites"].loc[site_id,:].to_dict()
    
    selections = {}
    for idx in idxs:
        s = disagg_stats.loc[idx, "seismicity"]
        r = disagg_stats.loc[idx, "region"]
        poe = disagg_stats.loc[idx, "poe"]
        imtl = disagg_stats.loc[idx, "imtl"]

        print(f"poe:  {poe}")
        # get the disagg distribution
        poe_disagg = grouped_disagg_data[(s, r)][site_id][imt][poe]

        gcim_cdfs = site_gcim_dists[poe]["cdfs"]
        
        selection = get_record_ensembles_for_single_distribution(
            poe_disagg, imtl, selection_ctx, ground_motion_database, 
            site_params, gcim_cdfs, rng_seed
            )

        selections[poe] = selection    

    return selections



def get_record_ensembles_for_single_distribution(
        disagg_dst, conditioning_iml, selection_ctx, ground_motion_database, 
        site_params, gcim_cdfs, rng_seed):
    
    # initialise the context builder
    ctx_builder = selection_ctx["ctx_builder"](
        site_params, *[selection_ctx[p] for p in selection_ctx["ctx_builder_params"]])

    # get m_bounds, dbounds, vs30bounds
    m_bounds = get_m_bounds(
        disagg_dst, selection_ctx["occurence"], 
        selection_ctx["m_bound_model"])
    
    d_bounds = get_d_bounds(
        disagg_dst, selection_ctx["occurence"], 
        selection_ctx["d_bound_model"])

    vs30_bounds = get_vs30_bounds(
        site_params["vs30"], 
        selection_ctx["vs30_bound_model"])

    # select the ensembles                
    poe_ensembles = select_ensembles(
        disagg_dst,
        gcim_cdfs,
        ground_motion_database,
        selection_ctx["n_ensembles"],
        selection_ctx["n_samples"], 
        selection_ctx["conditioning_imt"],
        conditioning_iml,
        selection_ctx["selection_imts"], 
        selection_ctx["imt_weights"],
        selection_ctx["gmm_map"],
        selection_ctx["corr_map"],
        ctx_builder,
        m_bounds, 
        d_bounds, 
        vs30_bounds,
        selection_ctx["sf_bounds"],
        selection_ctx["usable_T"],
        selection_ctx["max_n_recs"],
        selection_ctx["p_value"],
        selection_ctx["ok_trt_matches"],
        rng_seed)
    
    # check that some pass, otherwise log a failed flag
    # get the best ensemble
    best_ensemble, idx = get_best_ensemble(poe_ensembles, must_pass_ks=True)
    ensemble_found = True if best_ensemble is not None else False
    
    # calculate the ks bounds for the distribution
    ks_bounds = ensemble_ks_bounds(
        gcim_cdfs, selection_ctx["n_samples"], selection_ctx["p_value"])
    
    # save the results
    selection = {
        "ensemble_found": ensemble_found,
        "best_ensemble": best_ensemble, 
        "best_ensemble_idx": idx,
        "all_ensembles": poe_ensembles, 
        "ks_bounds": ks_bounds,
        "m_bounds": m_bounds,
        "d_bounds": d_bounds,
        "vs30_bounds": vs30_bounds
    }

    return selection


def calculate_site_gcim_distributions_for_all_sites(
        grouped_disagg_data: dict, disagg_stats: pd.DataFrame, 
        conditioning_imt:IMT, selection_imts: list[IMT],
        sites: pd.DataFrame, gmm_map: dict, corr_map: dict,
        average_depths: dict, assumed_rake: float, occurence: bool, 
        percentiles: list[int]) -> dict:
    """ 
    disagg_data is a nested dictionary with the following levels
    - (seismicity: str, region: int)    -> tuple
        - site_id: int
            - conditioning_imt: string
            - poe: float                -> exceedence probability of the disaggregations

    disagg_stats is pd.DataFrame containing with columns:
    - ["site_id", "seismicity", "region", "imt", "poe"]
    
    returns a nested dictionary with the following levels:
    - (seismicity: str, region: int)    -> tuple
        - site_id: int
            - poe: float                -> exceedence probability of the disaggregations"""
    
    gcim_dists = {}
    print("Calculating GCIM Distributions...")
    for (s, r), group_disaggs in grouped_disagg_data.items():
        print(f"  Region {r} - {s} seismicity")
        gcim_site = {}
        for site_id, site_dissags in group_disaggs.items():
            print(f"    site id: {site_id}")
            # initialise the context builder
            site_params = sites.loc[site_id,:].to_dict()
            ctx_builder = ESHM20SiteRupCtxBuilder(site_params, average_depths, assumed_rake)
            gcim_poe = {}
            for poe, poe_dissag in site_dissags[conditioning_imt.name].items():
                imtl = get_imtl_from_disaggstats(
                    disagg_stats, site_id, s, r, conditioning_imt.name, poe)
                gcim_stats, gcim_pdfs, gcim_cdfs = gcim_distributions(
                    poe_dissag, imtl, gmm_map, selection_imts, 
                    conditioning_imt, corr_map, ctx_builder, occurence, percentiles)
                
                gcim_poe[poe] = {"stats": gcim_stats, "pdfs": gcim_pdfs, "cdfs": gcim_cdfs}
            gcim_site[site_id] = gcim_poe
        gcim_dists[(s, r)] = gcim_site

    return gcim_dists


def get_gcim_distributions_for_single_site_and_poe(
        site_id: int,
        poe: float,
        grouped_disagg_data: dict, 
        disagg_stats: pd.DataFrame,
        selection_ctx: dict,
        percentiles: list=[0.16, 0.5, 0.84]
    ):

    imt = selection_ctx["disagg_imt"]
    
    unique_sites = disagg_stats["site_id"].unique()
    unique_poes = disagg_stats["poe"].unique()

    if not poe in unique_poes:
        raise ValueError(f"poe {poe} is not in the disagg_stats data")
    
    if not site_id in unique_sites:
        raise ValueError(f"site_id {site_id} is not in the disagg_stats data")

    print(f"Calculating GCIM distributions for Site {site_id}, poe: {poe}")

    idx = disagg_stats[(disagg_stats["site_id"] == site_id) &
                            (disagg_stats["poe"] == poe)].index[0]
    s = disagg_stats.loc[idx, "seismicity"]
    r = disagg_stats.loc[idx, "region"]

    # initialise the context builder
    site_params = selection_ctx["sites"].loc[site_id,:].to_dict()
    ctx_builder = selection_ctx["ctx_builder"](
        site_params, *[selection_ctx[p] for p in selection_ctx["ctx_builder_params"]])
    
    # get the disagg distribution
    poe_disagg = grouped_disagg_data[(s, r)][site_id][imt][poe]

    conditioning_imt = selection_ctx["conditioning_imt"]
    selection_imts = selection_ctx["selection_imts"]

    imtl = get_imtl_from_disaggstats(
        disagg_stats, site_id, s, r, conditioning_imt.name, poe)
    
    gmm_map = selection_ctx["gmm_map"]
    corr_map = selection_ctx["corr_map"]
    occurence = selection_ctx["occurence"]
    gcim_stats, gcim_pdfs, gcim_cdfs = gcim_distributions(
        poe_disagg, imtl, gmm_map, selection_imts, 
        conditioning_imt, corr_map, ctx_builder, occurence, percentiles)
    
    gcim_out = {"stats": gcim_stats, "pdfs": gcim_pdfs, "cdfs": gcim_cdfs}
                
    return gcim_out


def get_gcim_distributions_for_single_site(
        site_id: int,
        grouped_disagg_data: dict, 
        disagg_stats: pd.DataFrame,
        selection_ctx: dict,
        percentiles: list=[0.16, 0.5, 0.84]
    ):

    imt = selection_ctx["disagg_imt"]
    
    unique_sites = disagg_stats["site_id"].unique()
    if not site_id in unique_sites:
        raise ValueError(f"site_id {site_id} is not in the disagg_stats data")

    print(f"Calculating GCIM distributions for Site {site_id}")

    idxs = disagg_stats[(disagg_stats["site_id"] == site_id)].index

    # initialise the context builder
    site_params = selection_ctx["sites"].loc[site_id,:].to_dict()
    ctx_builder = selection_ctx["ctx_builder"](
        site_params, *[selection_ctx[p] for p in selection_ctx["ctx_builder_params"]])
    
    conditioning_imt = selection_ctx["conditioning_imt"]
    selection_imts = selection_ctx["selection_imts"]
    gmm_map = selection_ctx["gmm_map"]
    corr_map = selection_ctx["corr_map"]
    occurence = selection_ctx["occurence"]

    gcim_out = {}
    for idx in idxs:
        s = disagg_stats.loc[idx, "seismicity"]
        r = disagg_stats.loc[idx, "region"]
        poe = disagg_stats.loc[idx, "poe"]
        imtl = disagg_stats.loc[idx, "imtl"]

        # get the disagg distribution
        poe_disagg = grouped_disagg_data[(s, r)][site_id][imt][poe]

        gcim_stats, gcim_pdfs, gcim_cdfs = gcim_distributions(
            poe_disagg, imtl, gmm_map, selection_imts, 
            conditioning_imt, corr_map, ctx_builder, occurence, percentiles)
        
        gcim_out[poe] = {"stats": gcim_stats, "pdfs": gcim_pdfs, "cdfs": gcim_cdfs}
                
    return gcim_out



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


def m_bounds_tarbali_and_bradley_2016(
        disagg_dst: pd.DataFrame, occurence: bool,) -> tuple[float, float]:
    if occurence:
        mask = disagg_dst["P(m|X=x)"] != 0
        df = disagg_dst.loc[mask, "Mag"]
        lower = min(df.quantile(0.01), df.quantile(0.1) - 0.5)
        upper = max(df.quantile(0.99), df.quantile(0.9) - 0.5)
    else:
        raise NotImplementedError
    return (lower, upper)


def d_bounds_tarbali_and_bradley_2016(
        disagg_dst: pd.DataFrame, occurence: bool,) -> tuple[float, float]:
    if occurence:
        mask = disagg_dst["P(m|X=x)"] != 0
        df = disagg_dst.loc[mask, "Dist"]
        lower = min(df.quantile(0.01), 0.5 * df.quantile(0.1))
        upper = max(df.quantile(0.99), 1.5 * df.quantile(0.90))
    else:
        raise NotImplementedError
    return (lower, upper)


def vs30_bounds_tarbali_and_bradley_2016(
        site_vs30: float) -> tuple[float, float]:
    return (0.5 * site_vs30, 1.5 * site_vs30)


def get_imtl_from_disaggstats(disagg_stats, site_id, seismicity, region, imt, poe):
    imtl = disagg_stats[(disagg_stats["site_id"] == site_id) &
                        (disagg_stats["seismicity"] == seismicity) &
                        (disagg_stats["region"] == region) &
                        (disagg_stats["imt"] == imt) &
                        (disagg_stats["poe"] == poe)]["imtl"].values[0]
    return imtl




if __name__ == "__main__":
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import scipy.stats as stats

    from openquake.hazardlib.imt import PGA, SA, RSD595, AvgSA, IMT

    from phd_project.config.config import load_config 
    from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
        ESHM20SiteRupCtxBuilder,
        create_gmm_map,
        create_corr_model_map,
        calculate_site_gcim_distributions_for_all_sites,
        get_record_ensembles_for_site_and_poe
    )
    import phd_project.scripts.WP1_ground_motion_set.manage_flatfiles as mf

    cfg = load_config()

    # Load the disaggregation data
    fp = cfg["proc_data"]["site_hazard"] / "AvgSA_03_disagg_data_60sites.pickle"
    with open(fp, "rb") as f:
        disagg_data_AvgSA_03 = pickle.load(f)

    fp = cfg["proc_data"]["site_hazard"] / "AvgSA_03_disagg_stats_60sites.pickle"
    with open(fp, "rb") as f:
        disagg_stats_AvgSA_03 = pickle.load(f)

    fp = cfg["proc_data"]["site_hazard"] / "AvgSA_06_disagg_data_60sites.pickle"
    with open(fp, "rb") as f:
        disagg_data_AvgSA_06 = pickle.load(f)

    fp = cfg["proc_data"]["site_hazard"] / "AvgSA_06_disagg_stats_60sites.pickle"
    with open(fp, "rb") as f:
        disagg_stats_AvgSA_06 = pickle.load(f)

    # load the site file
    sites = pd.read_csv(cfg["hazard_models"]["eshm20_AvgSA_site_model_all"])

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

    # create the average depth map for each TRT #TODO:: if this needs to be more specific
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

    # set some parameters for the selection
    t_lower = 0.025     # lower SA period considered in selection
    t_upper = 6         # upper SA period considered in selection
    n_periods = 20      # number of periods to consider in selection

    conditioning_imt: IMT = AvgSA([0,6])                      
    nonSA_imts: list[IMT] = [AvgSA([0,6]), RSD595(), PGA()] 
    sa_periods = np.round(np.geomspace(t_lower, t_upper, num=n_periods), 3)
    SA_imts: list[IMT] = [SA(period) for period in sa_periods]
    selection_imts: list[IMT] = nonSA_imts[1:] + SA_imts
    nonSA_imt_strs: list[str] = [im.string for im in nonSA_imts] # strings match the correlation matrix

    # weights of the IMs
    weight_rsd595 = 0.3
    n_other_ims = len([imt for imt in selection_imts if imt.name == "SA" or imt.name == "PGA"])
    imt_weights = np.array([(1-weight_rsd595) / n_other_ims if imt.name != "RSD595" 
                            else weight_rsd595 for imt in selection_imts])
    imt_weights /= imt_weights.sum()

    # some other things
    disagg_type = "TRT_Mag_Dist_Eps"
    percentiles = [0.05, 0.16, 0.5, 0.84, 0.95]     # percentiles of the gcim distribution to return  
    assumed_rake = 0                                # assumed rake for RSD595 calculation

    # filter the gm_database so that only the selection and conditioning ims are present
    gm_db_AvgSA_06 = gm_database.copy()
    updated_ims = mf.filter_gm_database_on_imts(
        gm_db_AvgSA_06["ims"], selection_imts + [conditioning_imt])
    updated_ims.columns = pd.MultiIndex.from_product([['ims'], updated_ims.columns])
    gm_db_AvgSA_06 = pd.concat([gm_db_AvgSA_06.drop('ims', axis=1, level=0), updated_ims], axis=1)

    # Create the GMM Map for AvgSA by reading the logic tree
    AvgSA_06_lt_fp = cfg["hazard_models"]["eshm20_AvgSA"] / "gmpe_logic_tree_AvgSA_0to6_median_branch.xml"
    AvgSA_06_gmm_map = create_gmm_map(AvgSA_06_lt_fp)

    # get the correlation model map
    AvgSA_06_corr_map = create_corr_model_map(nonSA_imt_strs, sa_periods)

    # set up the selection context:
    selection_ctx_AvgSA_06 = {
        "n_ensembles": 50,
        "n_samples": 25,
        "conditioning_imt": conditioning_imt ,
        "disagg_imt": conditioning_imt.name , # this only works for AvgSA. otherwise used .string 
        "selection_imts": selection_imts ,
        "imt_weights": imt_weights ,
        "sites": sites ,
        "ctx_builder": ESHM20SiteRupCtxBuilder ,
        "ctx_builder_params": ["average_depths", "assumed_rake"] ,
        "average_depths": average_depths ,
        "assumed_rake": assumed_rake ,
        "gmm_map": AvgSA_06_gmm_map ,
        "corr_map": AvgSA_06_corr_map ,
        "m_bound_model": "tarbali_and_bradley_2016" ,
        "d_bound_model": "tarbali_and_bradley_2016" ,
        "vs30_bound_model": "tarbali_and_bradley_2016" ,
        "sf_bounds": None,#(0.25, 4) ,
        "usable_T": t_upper ,
        "max_n_recs": 3 ,
        "p_value": 0.05 ,
        "ok_trt_matches": OK_TRT_MATCHES ,
        "occurence": True ,
    }

    site = 30
    rng_seed = 2

    site_gcim_dists_06 = get_gcim_distributions_for_single_site(
        site, disagg_data_AvgSA_06, disagg_stats_AvgSA_06, selection_ctx_AvgSA_06)

    record_selection_results_AvgSA06 = get_record_ensembles_for_single_site(
        site, disagg_data_AvgSA_06, disagg_stats_AvgSA_06, site_gcim_dists_06, gm_db_AvgSA_06, selection_ctx_AvgSA_06, rng_seed)
    
    records = record_selection_results_AvgSA06[0.0001]
    es = records["all_ensembles"]
    e = es[0]
    print(e.keys())
    for ei in es:
        print(f"{ei["ks_passed"]}, {ei["R-score"]:.4f}, {ei["ks_failed_ims"]}")
    ...