"""
Module containing functions for calculating the correlation factors for
significant duration according to 

Haung and Galasso 2019
"Ground-motion intensity measure correlations observed in
Italian strong-motion records"

and

Haung et al. 2020
"Correlation properties of integral ground-motion intensity
measures from Italian strong-motion records"

"""

import numpy as np
import pandas as pd

AL_COEFFICIENTS_RSD595_SA = {
    0 : -0.580,
    1 : -0.576,
    2 : -0.592,
    3 : -0.573,
    4 : -0.539,
    5 : -0.441,
    6 : -0.002,
    7 : 0.101,
    8 : 0.090
}


TL_COEFFICIENTS_RSD595_SA = {
    0 : 0.01,
    1 : 0.04,
    2 : 0.1,
    3 : 0.15,
    4 : 0.2,
    5 : 0.3,
    6 : 1.1,
    7 : 2.1,
    8:  4 
}


TL_COEFFICIENTS_IA_SA = {
    0 : 0.01,
    1 : 0.07,
    2 : 0.2,
    3 : 4,
}


ABCD_COEFFICIENTS_IA_SA = {
    1 : (0.958, 0.881, 0.046, 2.343), # (al, bl, cl, dl)
    2 : (0.891, 0.911, 0.121, 4.882),
    3 : (0.943, 0.481, 0.768, 1.039)
}


TL_COEFFICIENTS_PGA_SA = {
    0 : 0.01,
    1 : 0.2,
    2 : 4.0
}


PHI_COEFFICIENTS_PGA_SA = {
    1 : (1, 0.950, 0.045, 2.225), # phi1, phi2, phi3, phi4
    2 : (1, 0.344, 0.783, 0.824)
}


def RSD595_SA_correlation_function(Ti: float) -> float:
    def _rho_rsd595_sa(Ti):
        n = _get_n(Ti, TL_COEFFICIENTS_RSD595_SA)
        a_l = AL_COEFFICIENTS_RSD595_SA[n]
        a_lm1 = AL_COEFFICIENTS_RSD595_SA[n-1]
        t_l = TL_COEFFICIENTS_RSD595_SA[n]
        t_lm1 = TL_COEFFICIENTS_RSD595_SA[n-1]
        rho = a_lm1 + (np.log(Ti / t_lm1) / np.log(t_l / t_lm1)) * (a_l - a_lm1)
        return rho
    
    if isinstance(Ti, float) or isinstance(Ti, int):
        return _rho_rsd595_sa(Ti)
    return np.array([_rho_rsd595_sa(ti) for ti in Ti])


def IA_SA_correlation_function(Ti: float) -> float:
    def _rho_ia_sa(Ti):
        n = _get_n(Ti, TL_COEFFICIENTS_IA_SA)
        a_l, b_l, c_l, d_l = ABCD_COEFFICIENTS_IA_SA[n]
        rho = (a_l + b_l) / 2 - ((a_l - b_l) / 2) * np.tanh(d_l * np.log(Ti / c_l))
        return rho
    
    if isinstance(Ti, float) or isinstance(Ti, int):
        return _rho_ia_sa(Ti)
    return np.array([_rho_ia_sa(ti) for ti in Ti])


def PGA_SA_correlation(Ti: float) -> float:
    def _rho_pga_sa(Ti):
        n = _get_n(Ti, TL_COEFFICIENTS_PGA_SA)
        p1, p2, p3, p4 = PHI_COEFFICIENTS_PGA_SA[n]
        rho = (p1 + p2) / 2 - ((p1 - p2) / 2) * np.tanh(p4 * np.log(Ti/p3))
        return rho
    
    if isinstance(Ti, float) or isinstance(Ti, int):
        return _rho_pga_sa(Ti)
    return np.array([_rho_pga_sa(ti) for ti in Ti])
    

def SA_correlation_function(T1: float, T2: float) -> float:

    T_max = max(T1, T2)
    T_min = min(T1, T2)

    c1 = 1 - np.cos(np.pi/2 - 0.2351 * np.log(T_max / max(T_min, 0.1)))
    c2 = 1 - 0.0617 * (1 - 1 / (1 + np.exp(100 * T_max - 5))) * (T_max - T_min) / (T_max - 0.0099)
    c3 = c1 + 0.3131 * (np.sqrt(c1) - c1) * (1 + np.cos(np.pi * T_min / 0.1))

    if T_max <= 0.1:
        rho = c2
    elif T_min > 0.1:
        rho = c1
    elif T_max <= 0.2:
        rho = min(c2, c3)
    else:
        rho = c3
    
    return rho


def _get_n(Ti: float, tl_coefficients: dict[float, float]) -> int:
    
    if Ti < 0.01:
        raise ValueError("T < 0.01s not supported")
    if Ti > 4:
        raise ValueError("T > 4s not supported")
    for n, T in tl_coefficients.items():
        if Ti == 0.01:
            return n+1
        if Ti <= T:
            return n
        

def rho_RSD595_PGA() -> float:
    return -0.579


def rho_RSD595_IA() -> float:
    return -0.444


def rho_PGA_IA() -> float:
    return 0.958


def build_correlation_matrix(periods) -> np.ndarray:
    """
    Constructs the correlation matrix period-by-period from the
    correlation functions
    
    The correlation between  PGA and RSD595 is assumed to be zero because no
    value is given in the original paper
    """

    n = len(periods) + 3  # +3 for PGA, RSD595, and IA
    rho = np.eye(n)
   
    # add the SA-SA correlations
    for i, t1 in enumerate(periods):
        for j, t2 in enumerate(periods[i:]):
            corr = SA_correlation_function(t1, t2)
            rho[i, i + j] = corr
            rho[i + j, i] = corr

    # add the PGA correlations
    rho[-3, :len(periods)] = PGA_SA_correlation(periods)
    rho[:len(periods), -3] = PGA_SA_correlation(periods)
    
    # add the RSD595 correlations
    rho[-2, :len(periods)] = RSD595_SA_correlation_function(periods)
    rho[:len(periods), -2] = RSD595_SA_correlation_function(periods)

    # add the IA correlations
    rho[-1, :len(periods)] = IA_SA_correlation_function(periods)
    rho[:len(periods), -1] = IA_SA_correlation_function(periods)
    
    # add correlation between RSD595, PGA, IA
    rho[-3, -2] = rho[-2, -3] = rho_RSD595_PGA()  # PGA/RSD595
    rho[-3, -1] = rho[-1, -3] = rho_PGA_IA() # PGA/IA
    rho[-2, -1] = rho[-1, -2] = rho_RSD595_IA() # RSD595/IA
    
    # labels
    labels = [("rotD50", "SA", t) for t in periods] + \
    [("rotD50", "PGA", "None"), ("rotD50", "RSD595", "None"), ("rotD50", "IA", "None")]

    rho = pd.DataFrame(rho, 
                       index=pd.MultiIndex.from_tuples(labels),
                       columns=pd.MultiIndex.from_tuples(labels))
    return rho


def htg20_correlation_model(periods) -> pd.DataFrame:
    return build_correlation_matrix(periods)


if __name__ == "__main__":
    rhos = htg20_correlation_model(np.array([0.1, 0.3, 0.5, 1.0, 2, 3, 4]))
    ...
