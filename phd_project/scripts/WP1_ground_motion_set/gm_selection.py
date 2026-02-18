from openquake.hazardlib.imt import IMT, AvgSA
import numpy as np
from pickagm.selection import TRT 
import numpy as np
import pandas as pd
from typing import Protocol

from openquake.hazardlib.imt import IMT
from openquake.hazardlib.gsim.base import GMPE

from pickagm.corrmodels import conditional_correlation_matrix

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


if __name__ == "__main__":

    import os
    import pickle
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from copy import deepcopy

    from pickagm.corrmodels import (
        ClemettCorrelationModelAsc,
        ClemettCorrelationModelSInter,
        ClemettCorrelationModelSSlab,
        ClemettCorrelationModelVrancea,
        CORR_MODELS
    )

    from pickagm.selection import gcim_simulation
    from pickagm.avgSA import indirect_AvgSA_GMPE

    from openquake.hazardlib.gsim.base import registry
    from openquake.hazardlib.imt import PGA, SA, RSD595, Sa_avg2, AvgSA
    from openquake.hazardlib.gsim.mgmpe.generic_gmpe_avgsa import GenericGmpeAvgSA
    from openquake.hazardlib.gsim.bahrampouri_2021_duration import (
        BahrampouriEtAldm2021Asc,
        BahrampouriEtAldm2021SInter,
        BahrampouriEtAldm2021SSlab
    )

    from phd_project.config.config import load_config 
    from phd_project.scripts.oqhelpers import parse_nrml_logic_tree
    from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
        ESHM20SiteRupCtxBuilder
    )

    cfg = load_config()

    # Load the disaggregation data
    fp = cfg["proc_data"]["site_hazard_disaggregation"] / "AvgSA_03_disagg_data_60sites_5poes.pickle"
    with open(fp, "rb") as f:
        disagg_data = pickle.load(f)

    fp = cfg["proc_data"]["site_hazard_disaggregation"] / "AvgSA_03_disagg_stats_60sites_5poes.pickle"
    with open(fp, "rb") as f:
        disagg_stats = pickle.load(f)

    # load the site file
    sites = pd.read_csv(cfg["hazard_models"]["eshm20_AvgSA_site_model_all"])

    # load the flatfiles
    flatfile_folder = cfg["proc_data"]["corr_model"] / "reverse" / "flatfiles"
    flatfiles = {}
    for f in [f for f in os.listdir(flatfile_folder) if f.endswith(".csv")]:
        tag = f.split("_")[0]
        flatfiles[tag] = pd.read_csv(flatfile_folder / f, delimiter=";", index_col=0)

    flatfiles["volcanic"] = pd.read_csv(cfg["raw_data"]["gm_flatfiles"] / "volcanic_lanzanoluzi_flatfile.csv", 
                                        delimiter=";", index_col=0)

    # a disaggregation distribution
    site_id = 31
    seismicity = "high"
    region = 0
    poe = 0.02
    imt = "AvgSA"
    sr = (seismicity, region)
    disagg_dst = disagg_data[sr][site_id][imt][poe]
    imtl = disagg_stats[(disagg_stats["site_id"] == site_id) &
                        (disagg_stats["seismicity"] == seismicity) &
                        (disagg_stats["region"] == region) &
                        (disagg_stats["imt"] == imt) &
                        (disagg_stats["poe"] == poe)]["imtl"].values[0]
    disagg_type = "TRT_Mag_Dist_Eps"
    assumed_rake = 0


    def SA_PGA_gmm_from_logic_tree(LT: dict, trt: str):
        ignore_tags = ["avg_periods", "corr_func"]
        params = list(deepcopy(LT[trt]).values())[0]["params"]
        for tag in ignore_tags:
            params.pop(tag, None)
        gmpe_name = params.pop("gmpe_name")
        gmm = registry[gmpe_name](**params)
        return gmm

    def AvgSA_ggm_from_logic_tree(LT: dict, trt: str):
        params = list(deepcopy(LT[trt]).values())[0]["params"]
        gmpe_name = params.pop("gmpe_name")
        periods = params.pop("avg_periods")
        corr_func = params.pop("corr_func")
        rho_total = CORR_MODELS[corr_func](incl_SA_Ts = periods).rho["total"]
        corr_mat = rho_total.to_numpy()
        base_gmm = registry[gmpe_name](**params)
        gmm = indirect_AvgSA_GMPE(base_gmm, corr_mat, avg_periods=periods)
        return gmm


    # set some parameters for the selection
    sa_periods = np.round(np.logspace(np.log10(0.025), np.log10(8), num=30), 3)
    nonSA_imts = [AvgSA([0,3]).string, RSD595().string, PGA().string]  # strings match the correlation matrix
    cond_imt = AvgSA([0,3])  # strings match the correlation matrix
    selection_imts = [RSD595(), PGA()] + [SA(period) for period in sa_periods]  # these need to be imt instances to work with GMMs

    # Create the GMM Map for AvgSA by reading the logic tree
    fp = cfg["hazard_models"]["eshm20_AvgSA"] / "gmpe_logic_tree_AvgSA_0to3_median_branch.xml"
    GMM_AvgSA_LT = parse_nrml_logic_tree(file_path=fp)

    GMM_MAP = {
        "Craton": {
            "AvgSA": AvgSA_ggm_from_logic_tree(GMM_AvgSA_LT, "Craton"),         # these strings need to match the imt.name so that they be found when the imt is up to be calculated.
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Craton"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Craton"),
            "RSD595": BahrampouriEtAldm2021Asc(),
            },
        "Non-Subduction Deep": {
            "AvgSA": AvgSA_ggm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Non-Subduction Deep"),
            "RSD595": BahrampouriEtAldm2021SSlab()
            },
        "Shallow Default": {
            "AvgSA": AvgSA_ggm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Shallow Default"),
            "RSD595": BahrampouriEtAldm2021Asc()
            },
        "Subduction Inslab": {
            "AvgSA": AvgSA_ggm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Inslab"),
            "RSD595": BahrampouriEtAldm2021SSlab()
            },
        "Subduction Interface": {
            "AvgSA": AvgSA_ggm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Subduction Interface"),
            "RSD595": BahrampouriEtAldm2021SInter()
            },
        "Volcanic": {
            "AvgSA": AvgSA_ggm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "PGA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "SA": SA_PGA_gmm_from_logic_tree(GMM_AvgSA_LT, "Volcanic"),
            "RSD595": BahrampouriEtAldm2021Asc()
            }
    }

    # create the average depth map for each TRT #TODO:: if this needs to be more specific
    average_depths = {
        "Craton": flatfiles["asc"]["ev_depth_km"].mean(),
        "Non-Subduction Deep": flatfiles["vran"]["ev_depth_km"].mean(),
        "Shallow Default": flatfiles["asc"]["ev_depth_km"].mean(),
        "Subduction Inslab": flatfiles["sinter"]["ev_depth_km"].mean(),
        "Subduction Interface": flatfiles["sinter"]["ev_depth_km"].mean(),
        "Volcanic": flatfiles["volcanic"]["ev_depth_km"].mean(),
    }

    # get the correlation model map
    corr_model_map = {
        "Craton": ClemettCorrelationModelAsc(nonSA_imts, sa_periods).rho["total"],
        "Non-Subduction Deep": ClemettCorrelationModelVrancea(nonSA_imts, sa_periods).rho["total"],
        "Shallow Default": ClemettCorrelationModelAsc(nonSA_imts, sa_periods).rho["total"],
        "Subduction Inslab": ClemettCorrelationModelSSlab(nonSA_imts, sa_periods).rho["total"],
        "Subduction Interface": ClemettCorrelationModelSInter(nonSA_imts, sa_periods).rho["total"],
        "Volcanic": ClemettCorrelationModelAsc(nonSA_imts, sa_periods).rho["total"],
    }


    # create the site_rup context for the sampled M-R-TRT
    site_params = sites.loc[site_id,:].to_dict()
    ctx_builder = ESHM20SiteRupCtxBuilder(site_params, average_depths, assumed_rake)
    sim_result = gcim_simulation(3, disagg_dst, disagg_type, imtl, GMM_MAP,
                                selection_imts, cond_imt, corr_model_map, ctx_builder,
                                1)
