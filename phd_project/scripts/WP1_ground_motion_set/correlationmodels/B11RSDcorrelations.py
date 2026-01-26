"""
Module containing functions for calculating the correlation factors for
significant duration according to Bradley 2011
"""

import numpy as np

A_COEFFICIENTS_RSD595 = {
    0 : -0.41,
    1 : -0.41,
    2 : -0.38,
    3 : -0.35,
    4 : -0.02,
    5 : 0.23,
    6 : 0.02
}

B_COEFFICIENTS_RSD595 = {
    0 : 0.01,
    1 : 0.04,
    2 : 0.08,
    3 : 0.26,
    4 : 1.40,
    5 : 6.00,
    6 : 10.00
}

def rho_RSD595_SA(Ti: float) -> float:
    
    n = _get_n(Ti, B_COEFFICIENTS_RSD595)
    a_n = A_COEFFICIENTS_RSD595[n]
    a_nm1 = A_COEFFICIENTS_RSD595[n-1]
    b_n = B_COEFFICIENTS_RSD595[n]
    b_nm1 = B_COEFFICIENTS_RSD595[n-1]
    rho = a_nm1 + np.log(Ti / b_nm1) / np.log(b_n / b_nm1) * (a_n - a_nm1)
    return rho


def _get_n(Ti: float, b_coefficients: dict[float, float]) -> int:
    
    if Ti < 0.01:
        raise ValueError("T < 0.01s not supported")
    if Ti > 10:
        raise ValueError("T > 10s not supported")
    for n, T in b_coefficients.items():
        if Ti < T:
            return n
        

def rho_RSD595_PGA() -> float:
    return -0.442