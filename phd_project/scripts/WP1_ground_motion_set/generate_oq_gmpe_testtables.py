"""
Script for generating test tables for the Indirect AvgSA GMPES in OpenQuake
"""

import numpy as np
import pandas as pd
from pickagm.avgSA import indirect_AvgSA_GMPE
from openquake.hazardlib.imt import SA, PGA
from openquake.hazardlib.gsim.kotha_2020 import KothaEtAl2020ESHM20
from phd_project.config.config import load_config

cfg = load_config()

folder = cfg["proc_data"]["corr_model"] / "test_tables_for_openquake"

## Testtables for GenericGMPEAvgSaTablesTestCaseClemettAsc
# periods for the generic GMPE: 
ts1 = [0.3, 0.4, 0.5]
ts2 = [0.0, 0.1, 0.2]

# correlations:
rhos_ts1 = np.array(
    [[1.000, 0.943, 0.890],
     [0.943, 1.000, 0.958],
     [0.890, 0.958, 1.000]])

rhos_ts2 = np.array(
    [[1.000, 0.955, 0.943],
     [0.955, 1.000, 0.909],
     [0.943, 0.909, 1.000]])

# create the dataframe of the input parameters
parameters = {}
parameters["rup_mag"] = np.array(sorted(5 * [4.5, 6.0, 7.5]))
parameters["rup_hypo_depth"] = 10 * np.ones(15)
parameters["dist_rjb"] = np.array(3 * [0, 50, 100, 150, 200])
parameters["site_vs30"] = 800 * np.ones(15)
parameters["site_vs30measured"] = 15 * [1]
parameters["site_region"] = 15 * [0]
parameters["damping"] = 5 * np.ones(15)

testtable = pd.DataFrame.from_dict(parameters, orient="columns")

# calculate AvgSA and the total sigma
site_rup_ctxs = np.recarray(
    len(parameters["rup_mag"]), 
    dtype=[
        ("mag", "f4"),
        ("hypo_depth", "f4"),
        ("vs30", "f4"),
        ("vs30measured", "bool"),
        ("region", "f4"),
        ("rjb", "f4")
    ])
site_rup_ctxs["mag"] = parameters["rup_mag"]
site_rup_ctxs["hypo_depth"] = parameters["rup_hypo_depth"]
site_rup_ctxs["rjb"] = parameters["dist_rjb"]
site_rup_ctxs["region"] = parameters["site_region"]
site_rup_ctxs["vs30"] = parameters["site_vs30"]
site_rup_ctxs["vs30measured"] = parameters["site_vs30measured"]

## Case 1: ts1 -> just SA
imts = [SA(t) for t in ts1]
mean = np.zeros((len(imts), len(site_rup_ctxs)))
sigma = np.zeros_like(mean)
tau = np.zeros_like(mean)
phi = np.zeros_like(mean)

KothaEtAl2020ESHM20().compute(site_rup_ctxs, imts, mean, sigma, tau, phi)

avgSA_mean = np.exp(mean.mean(axis=0))
avgSA_sigma = np.sqrt(
    (rhos_ts1[:, np.newaxis, :] * np.einsum("ij, kj -> ijk", sigma, sigma)) \
    .sum(axis=(0,2))) / len(imts)

testtable_avgSA_mean = testtable.copy()
testtable_avgSA_mean.loc[:, "result_type"] = "MEAN"
testtable_avgSA_mean["AvgSA"] = avgSA_mean

testtable_avgSA_stddev = testtable.copy()
testtable_avgSA_stddev.loc[:, "result_type"] = "TOTAL_STDDEV"
testtable_avgSA_stddev["AvgSA"] = avgSA_sigma

testtable_avgSA_mean.to_csv(folder / "generic_gmpe_avgsa_clemettasc_mean.csv")
testtable_avgSA_stddev.to_csv(folder / "generic_gmpe_avgsa_clemettasc_stddev.csv")

# test my own implementation of AvgSA GMM
avgSA_gmm = indirect_AvgSA_GMPE(KothaEtAl2020ESHM20(), rho_total=rhos_ts1, avg_periods=ts1)

mean_t = np.zeros((1, len(site_rup_ctxs)))
sigma_t = np.zeros_like(mean)
tau_t = np.zeros_like(mean)
phi_t = np.zeros_like(mean)
avgSA_gmm.compute(site_rup_ctxs, None, mean_t, sigma_t, tau_t, phi_t)

##########################################################################################
## Case 2: ts2 -> SA and PGA
imts = [SA(t) if t != 0 else PGA() for t in ts2]
mean = np.zeros((len(imts), len(site_rup_ctxs)))
sigma = np.zeros_like(mean)
tau = np.zeros_like(mean)
phi = np.zeros_like(mean)

KothaEtAl2020ESHM20().compute(site_rup_ctxs, imts, mean, sigma, tau, phi)

avgSA_mean = np.exp(mean.mean(axis=0))
avgSA_sigma = np.sqrt(
    (rhos_ts2[:, np.newaxis, :] * np.einsum("ij, kj -> ijk", sigma, sigma)) \
    .sum(axis=(0,2))) / len(imts)

testtable_avgSA_mean = testtable.copy()
testtable_avgSA_mean.loc[:, "result_type"] = "MEAN"
testtable_avgSA_mean["AvgSA"] = avgSA_mean

testtable_avgSA_stddev = testtable.copy()
testtable_avgSA_stddev.loc[:, "result_type"] = "TOTAL_STDDEV"
testtable_avgSA_stddev["AvgSA"] = avgSA_sigma

testtable_avgSA_mean.to_csv(folder / "generic_gmpe_avgsa_clemettasc_w_pga_mean.csv")
testtable_avgSA_stddev.to_csv(folder / "generic_gmpe_avgsa_clemettasc_w_pga_stddev.csv")

# test my own implementation of AvgSA GMM
avgSA_gmm = indirect_AvgSA_GMPE(KothaEtAl2020ESHM20(), rho_total=rhos_ts2, avg_periods=ts2)

mean_t = np.zeros((1, len(site_rup_ctxs)))
sigma_t = np.zeros_like(mean)
tau_t = np.zeros_like(mean)
phi_t = np.zeros_like(mean)
avgSA_gmm.compute(site_rup_ctxs, None, mean_t, sigma_t, tau_t, phi_t)

...

if __name__ == "__main__":
    ...