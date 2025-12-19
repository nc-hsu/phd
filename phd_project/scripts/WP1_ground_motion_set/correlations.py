"""_summary_
Module to derive correlation models for ground motion IMs."""

import numpy as np
import pandas as pd
from openquake.hazardlib.imt import IMT, SA, PGA
from openquake.hazardlib.gsim.base import GMPE

from phd_project.scripts.oq_imt_translations import IMT_MAP


def pairwise_correlation(
        eps1: np.ndarray, eps2: np.ndarray) -> float:
    """Calculate the correlation coefficient between two sets of residuals."""
    
    mean_eps1 = np.mean(eps1)
    mean_eps2 = np.mean(eps2)
    
    eps_sub_mean1 = eps1 - mean_eps1
    eps_sub_mean2 = eps2 - mean_eps2

    numerator = np.sum(eps_sub_mean1 * eps_sub_mean2)
    denominator = np.sqrt(np.sum(eps_sub_mean1**2) * np.sum(eps_sub_mean2**2))
    rho = numerator / denominator
    return rho


def pairwise_correlation_from_dB_and_dW(
        rho_dB_ij: float, rho_dW_ij: float, tau_i: float, tau_j: float, 
        phi_i: float, phi_j: float, sig_i: float, sig_j: float):
    
    rho = (rho_dB_ij * tau_i * tau_j + 
           rho_dW_ij * phi_i * phi_j) / \
          (sig_i * sig_j)
    return rho


def _get_predicted_im_values(im: IMT, gmm: GMPE, site_rup_ctxs):
    
    if str(im).startswith("AvgSA"):
        imts = [PGA() if T == 0 else SA(T) for T in im.period]
    else:
        imts = [im]

    mean = np.zeros((1, len(site_rup_ctxs)))
    sig = np.zeros((1, len(site_rup_ctxs)))
    tau = np.zeros((1, len(site_rup_ctxs)))
    phi = np.zeros((1, len(site_rup_ctxs)))
    gmm.compute(site_rup_ctxs, imts, mean, sig, tau, phi)

    return mean.flatten(), sig.flatten(), tau.flatten(), phi.flatten()


def _calculate_delta_Bi(
        event_index: pd.Index|pd.MultiIndex, 
        Delta_ij: np.ndarray, 
        tau: np.ndarray, 
        phi: np.ndarray) -> np.ndarray:
    # calculation of the between-event residual
    # using equation B.42 from Baker, Bradley and Stafford (2021) Seismic Hazard 
    # and Risk Analysis. This equation was originally proposed by 
    # Abrahamsom and Youngs (1992)

    Delta_ij_df = pd.DataFrame(Delta_ij, index=event_index)
    # total number of recordings per event
    ni = Delta_ij_df.groupby(level=0).transform('count').to_numpy()
    # total residual per event
    sum_Delta_ij = Delta_ij_df.groupby(level=0).transform('sum').to_numpy()
    # between-event residual 
    delta_Bi = tau ** 2 * sum_Delta_ij / (ni * tau ** 2 + phi ** 2)
    return delta_Bi


def _calculate_delta_Wij(
        Delta_ij: np.ndarray, 
        delta_Bi: np.ndarray) -> np.ndarray:
    return Delta_ij - delta_Bi


def _calculate_fhp_dependent_residuals(
        event_index: pd.Index|pd.MultiIndex,
        Delta_ij: np.ndarray, 
        tau: np.ndarray,
        phi: np.ndarray,
        min_fhps: np.ndarray,
        fhp_limits: np.ndarray) -> dict[float, np.ndarray]:
    
    # min_fhps: the minimum high pass frequency for horizontal motion for each
    # record
    delta_Bis = {}
    delta_Wijs = {}
    masks = {}

    for fhp in fhp_limits:
        valid_mask = min_fhps <= fhp
        temp_event_index = event_index[valid_mask]
        temp_Delta_ij = Delta_ij[valid_mask, :]
        temp_tau = tau[valid_mask, :]
        temp_phi = phi[valid_mask, :]

        delta_Bi = _calculate_delta_Bi(
            temp_event_index, temp_Delta_ij, temp_tau, temp_phi)
        delta_Wij = _calculate_delta_Wij(temp_Delta_ij, delta_Bi)

        delta_Bis[fhp] = delta_Bi
        delta_Wijs[fhp] = delta_Wij
        masks[fhp] = valid_mask
    
    return delta_Bis, delta_Wijs, masks 


def derive_correlation_model(
        observed_df, metadata, site_rup_ctxs, gmm_map, fhp_cutoff=0.8):
    
    num_cols = observed_df.shape[1]
    cols = observed_df.columns
    
    # 1. Pre-calculate Column Frequency Limits and Predicted Values (O(N))
    f_hp_limits = []
    residuals = []
    taus = []
    phis = []
    
    for col in cols:
        _, imt, T = col   # use only the imt and period. component is not needed
        if imt == "SA":
            im = IMT_MAP[imt](T) 
        elif (imt.startswith("AvgSA")):
            im = IMT_MAP[imt](eval(T))
        else:
            im = IMT_MAP[imt]()
        
        # Calculate f_hp for this specific column
        if imt == "SA":
            f_hp = 1 / T * fhp_cutoff
        elif imt.startswith("AvgSA"):
            f_hp = 1 / max(im.period) * fhp_cutoff 
        else:
            f_hp = 100.0        # no filtering based on the high pass frequency

        f_hp_limits.append(f_hp)
        
        # Get predictions for the ENTIRE dataset once
        gmm_out = _get_predicted_im_values(im, gmm_map[imt], site_rup_ctxs)
        pred, _, tau_vec, phi_vec = gmm_out

        # Calculate residuals for all rows (we will mask them later)
        # Flattening to ensure 1D array per column
        res = np.log(observed_df[col].to_numpy()) - pred
        residuals.append(res)
        taus.append(tau_vec)
        phis.append(phi_vec)

    f_hp_limits = np.array(f_hp_limits)
    min_fhp_values = metadata[["U_hp", "V_hp"]].min(axis=1).to_numpy()
    # Convert data list to 2D arrays (Rows x Columns)
    Delta_ij = np.vstack(residuals).T
    tau = np.vstack(taus).T
    phi = np.vstack(phis).T 
    
    # get the between-event and within-event residuals:
    event_index = pd.Index(metadata["event_id"])
    
    # unlike the total residual the values of delta Bi are dependent on the
    # the number of records for each event. For each period the number of 
    # records can change depending on the high-pass filter frequency of the 
    # seismometer.

    # precalculate delta_Bi and delta_Wij for each possible high_pass limit
    delta_Bis, delta_Wijs, masks = _calculate_fhp_dependent_residuals(
        event_index, Delta_ij, tau, phi, min_fhp_values, f_hp_limits)
    # delta_Bi = _calculate_delta_Bi(event_index, Delta_ij, tau, phi)
    # delta_Wij = _calculate_delta_Wij(Delta_ij, delta_Bi)

    # Prepare the Output Arrays
    rho_total_A = np.eye(num_cols) # Diagonal is always 1.0
    rho_total_B = np.eye(num_cols) # Diagonal is always 1.0
    rho_dB = np.eye(num_cols)
    rho_dW = np.eye(num_cols)

    # We only compute the upper triangle (j > i)
    for i in range(num_cols):
        f_hp_i = f_hp_limits[i]
        
        for j in range(i + 1, num_cols):
            # Check component match once at the start or assume valid
            if cols[i][0] != cols[j][0]:
                continue
                
            f_hp_j = f_hp_limits[j]
            
            # The limit is the stricter of the two
            f_hp_limit = min(f_hp_i, f_hp_j)

            # Vectorized Masking
            valid_mask = masks[f_hp_limit]
            # valid_mask = min_fhp_values <= f_hp_limit
            # valid_mask = (metadata["U_hp"].values <= f_hp_limit) & \
            #              (metadata["V_hp"].values <= f_hp_limit)
            
            # calculate the correlations:
            # between-event: use the dB array corresponding to f_hp_limit 
            r_dB = pairwise_correlation(
                delta_Bis[f_hp_limit][:, i], delta_Bis[f_hp_limit][:, j])
            rho_dB[i, j] = r_dB
            rho_dB[j, i] = r_dB # Exploiting Symmetry
            
            # within-event
            r_dW = pairwise_correlation(
                delta_Wijs[f_hp_limit][:, i], delta_Wijs[f_hp_limit][:, i])
            rho_dW[i, j] = r_dW
            rho_dW[j, i] = r_dW # Exploiting Symmetry
            
            # total A - from between- and within-event residuals
            # event averaged values for tau and phi
            tau_bar_i = tau[valid_mask, i].mean()
            tau_bar_j = tau[valid_mask, j].mean()
            phi_bar_i = phi[valid_mask, i].mean()
            phi_bar_j = phi[valid_mask, j].mean()
            sig_i = np.sqrt(tau_bar_i ** 2 + phi_bar_i ** 2)
            sig_j = np.sqrt(tau_bar_j ** 2 + phi_bar_j ** 2)

            r_tot_A = pairwise_correlation_from_dB_and_dW(
                r_dB, r_dW, tau_bar_i, tau_bar_j, phi_bar_i, phi_bar_j, sig_i, sig_j)
            rho_total_A[i, j] = r_tot_A
            rho_total_A[j, i] = r_tot_A # Exploiting Symmetry

            # total B - from total residual
            r_tot_B = pairwise_correlation(
                Delta_ij[valid_mask, i], Delta_ij[valid_mask, j])
            rho_total_B[i, j] = r_tot_B
            rho_total_B[j, i] = r_tot_B # Exploiting Symmetry

    rho_total_A = pd.DataFrame(rho_total_A, index=cols, columns=cols)
    rho_dB = pd.DataFrame(rho_dB, index=cols, columns=cols)
    rho_dW = pd.DataFrame(rho_dW, index=cols, columns=cols)
    rho_total_B = pd.DataFrame(rho_total_B, index=cols, columns=cols)

    return rho_total_A, rho_dB, rho_dW, rho_total_B 


if __name__ == "__main__":
    import pickle
    import time
    from phd_project.config.loader import load_config
    from openquake.hazardlib.gsim.bahrampouri_2021_duration import BahrampouriEtAldm2021Asc
    from pickagm.avgSA import indirect_AvgSA_GMPE
    from openquake.hazardlib.gsim.kotha_2020 import KothaEtAl2020ESHM20
    
    cfg = load_config()
    idx = pd.IndexSlice

    with open(cfg["data"]["root"] / "test_data.pkl", "rb") as f:
        df_observed = pickle.load(f)
    with open(cfg["data"]["root"] / "test_data_ctxs.pkl", "rb") as f:
        site_rup_ctxs = pickle.load(f)
    with open(cfg["data"]["root"] / "test_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    with open(cfg["data"]["root"] / "test_cg26.pkl", "rb") as f:
        rho_cg26 = pickle.load(f)

    periods_0_3 = list(np.round(np.linspace(0.0, 3.0, 10), 3))
    periods_0_6 = list(np.round(np.linspace(0.0, 6.0, 10), 3))

    correlations_for_AvgSA_03 = rho_cg26.loc[idx["rotD50", ["PGA", "SA"], periods_0_3], 
                                    idx["rotD50", ["PGA", "SA"], periods_0_3]].to_numpy()
    correlations_for_AvgSA_06 = rho_cg26.loc[idx["rotD50", ["PGA", "SA"], periods_0_6], 
                                    idx["rotD50", ["PGA", "SA"], periods_0_6]].to_numpy()
    AvgSA_GMPE_03 = indirect_AvgSA_GMPE(KothaEtAl2020ESHM20(),
                                        correlations_for_AvgSA_03)
    AvgSA_GMPE_06 = indirect_AvgSA_GMPE(KothaEtAl2020ESHM20(),
                                        correlations_for_AvgSA_06)

    gmm_map = {"SA": KothaEtAl2020ESHM20(),
               "PGA": KothaEtAl2020ESHM20(),
               "RSD595": BahrampouriEtAldm2021Asc(),
               "AvgSA[0,3]": AvgSA_GMPE_03,
               "AvgSA[0,6]": AvgSA_GMPE_06}

    start = time.time()
    rho_out = derive_correlation_model(df_observed, metadata, site_rup_ctxs, gmm_map)
    rho, rho_dB, rho_dW, rho_tot = rho_out
    end = time.time()
    print(end-start)

    pass
