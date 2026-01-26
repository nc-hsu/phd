"""
Correlation Model for NGA-Subduction from:
Macdeo and Liu (2021). Ground-Motion Intensity Measure Correlations on Interface 
and Intraslab Subduction Zone Earthquakes Using the NGA-Sub Database
Bulletin of the Seismological Society of America (2021) 111 (3): 1529-1541.
"""

import numpy as np
import pandas as pd

MODEL_COEFFICIENTS = {
     "sinter_total": (0.109, 0.2, 0.277, 0.104, 662, 100, 5, 994, 0.387, 0.109),
     "sslab_total": (0.1, 0.2, 0.290, 0.095, 20500, 100, 5, 26000, 0.331, 0.083)
}

def ml21_correlation_model(which, periods) -> pd.DataFrame:
    return build_correlation_matrix(which, periods)


def SA_correlation_function(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, t1, t2) -> float:
    """
    Basic function of the Baker & Jayaram (2007) cross-correlation model
    allowing for flexibility in the coefficients of the model.

    Adapted from OpenQuake's implementation:
    hazardlib/gsim/mgmpe/generic_gmpe_avgsa.py > baker_jayaram_correlation_model_function

    :param float d1:
        Coefficient d1 (0.366 in original model)

    :param float d2:
        Coefficient d2 (0.105 in original model)

    :param float d3:
        Coefficient d3 (0.0099 in original model)

    :param float d4:
        Coefficient d4 (0.109 in the original model)

    :param float d5:
        Coeffucuent d5 ( in the original model)
    
    :param float d5:
        Coefficient d5 (0.2 in the original model)

    :param float d7:
        Coefficient d6 (0.5 in the original model)

    :

    :param float t1:
        First period of interest.

    :param float t2:
        Second period of interest.

    :return float rho:
        The predicted correlation coefficient.
    """
    t_min = min(t1, t2)
    t_max = max(t1, t2)

    c1 = 1.0 - np.cos(np.pi / 2.0 - d3 * np.log(t_max / max(t_min, d4)))

    if t_max < d2:
        c2 = 1 + d5 * (1.0 - 1.0 / (1.0 + np.exp(d6 * t_max - d7))) * \
            (t_max - t_min) / (t_max - d8)
    else:
        c2 = 0

    if t_max < d1:
        c3 = c2
    else:
        c3 = c1

    c4 = c1 + d9 * (np.sqrt(c3) - c3) * (1.0 + np.cos(np.pi * t_min / d10))

    if t_max <= d1:
        rho = c2
    elif t_min > d1:
        rho = c1
    elif t_max < d2:
        rho = min(c2, c4)
    else:
        rho = c4

    return rho



def PGA_SA_correlation(which, periods=None) -> np.ndarray:
    if which == "sinter_total":
        data = np.array(PGA_SA_RHO_SINTER)
    elif which == "sslab_total":
        data = np.array(PGA_SA_RHO_SSLAB)

    if periods is not None:
        return np.interp(periods, data[:,0], data[:,1])
    return data[:1].flatten()


def RSD595_SA_correlation_function(which, periods=None) -> np.ndarray:
    if which == "sinter_total":
        data = np.array(RSD595_SA_RHO_SINTER)
    elif which == "sslab_total":
        data = np.array(RSD595_SA_RHO_SSLAB)

    if periods is not None:
        return np.interp(periods, data[:,0], data[:,1])
    return data[:1].flatten()


def build_correlation_matrix(which, periods) -> np.ndarray:
    """
    Constructs the correlation matrix period-by-period from the
    correlation functions
    
    The correlation between  PGA and RSD595 is assumed to be zero because no
    value is given in the original paper
    """

    n = len(periods) + 2  # +2 for PGA and RSD595
    rho = np.eye(n)
   
    # add the SA-SA correlations
    for i, t1 in enumerate(periods):
        for j, t2 in enumerate(periods[i:]):
            rho[i, i + j] = SA_correlation_function(*MODEL_COEFFICIENTS[which], t1, t2)
            rho[i + j, i] = SA_correlation_function(*MODEL_COEFFICIENTS[which], t1, t2)
    
    # add the PGA correlations
    rho[-2, :len(periods)] = PGA_SA_correlation(which, periods)
    rho[:len(periods), -2] = PGA_SA_correlation(which, periods)
    
    # add the RSD595 correlations
    rho[-1, :len(periods)] = RSD595_SA_correlation_function(which, periods)
    rho[:len(periods), -1] = RSD595_SA_correlation_function(which, periods)
    
    # add correlation between RSD595 and PGA
    rho[-1, -2] = 0
    rho[-2, -1] = 0
    
    # labels
    labels = [("rotD50", "SA", t) for t in periods] + [("rotD50", "PGA", "None"), ("rotD50", "RSD595", "None")]

    rho = pd.DataFrame(rho, 
                       index=pd.MultiIndex.from_tuples(labels),
                       columns=pd.MultiIndex.from_tuples(labels))
    return rho





# SA(T)-rho pairs for correlation between PGA and SA(T) for interface events
PGA_SA_RHO_SINTER = [
    (0.01, 0.999), (0.018, 0.996), (0.023, 0.996), (0.032, 0.991), (0.046, 0.983), 
    (0.068, 0.957), (0.092, 0.94), (0.136, 0.935), (0.165, 0.935), (0.229, 0.927), 
    (0.291, 0.91), (0.377, 0.867), (0.511, 0.786), (0.677, 0.713), (0.86, 0.641), 
    (1.115, 0.581), (1.51, 0.496), (1.959, 0.436), (2.771, 0.389), (3.369, 0.363), 
    (3.754, 0.325), (4.465, 0.286), (4.869, 0.269), (5.086, 0.282), (5.792, 0.239), 
    (6.889, 0.21), (7.678, 0.218), (8.019, 0.214), (8.939, 0.244), (9.963, 0.235)
]

# SA(T)-rho pairs for correlation between RSD595 and SA(T) for interface events
RSD595_SA_RHO_SINTER  = [
    (0.01, -0.457), (0.015, -0.454), (0.024, -0.454), (0.035, -0.441), (0.052, -0.423), 
    (0.071, -0.406), (0.094, -0.397), (0.128, -0.393), (0.149, -0.406), (0.174, -0.406), 
    (0.19, -0.419), (0.217, -0.419), (0.253, -0.423), (0.308, -0.423), (0.367, -0.414), 
    (0.419, -0.397), (0.478, -0.38), (0.522, -0.367), (0.57, -0.362), (0.665, -0.349), 
    (0.811, -0.301), (0.885, -0.284), (1.032, -0.271), (1.103, -0.262), (1.258, -0.254), 
    (1.5, -0.219), (1.788, -0.184), (1.91, -0.167), (2.432, -0.154), (3.03, -0.132), 
    (3.457, -0.106), (3.859, -0.08), (4.702, -0.084), (5.484, -0.032), (6.539, 0.012), 
    (7.46, -0.001), (7.969, 0.007), (9.092, -0.028), (10.148, -0.028)
]

# SA(T)-rho pairs for correlation between PGA and SA(T) for inslab events
PGA_SA_RHO_SSLAB = [
    (0.01, 1.00), (0.012, 1.000), (0.017, 0.999), (0.022, 0.999), (0.029, 0.994), 
    (0.039, 0.981), (0.055, 0.954), (0.068, 0.936), (0.103, 0.936), (0.144, 0.941), 
    (0.182, 0.94), (0.233, 0.914), (0.301, 0.876), (0.35, 0.854), (0.435, 0.796), 
    (0.526, 0.736), (0.661, 0.668), (0.761, 0.618), (0.921, 0.573), (1.072, 0.53), 
    (1.218, 0.503), (1.474, 0.493), (1.695, 0.48), (2.105, 0.45), (2.547, 0.417), 
    (2.967, 0.407), (3.457, 0.387), (4.184, 0.34), (4.515, 0.345), (4.874, 0.309), 
    (5.606, 0.302), (7.048, 0.277), (7.511, 0.257), (8.53, 0.279), (9.206, 0.267), 
    (10.0, 0.267)
]

# SA(T)-rho pairs for correlation between RSD595 and SA(T) for inslab events
RSD595_SA_RHO_SSLAB = [
    (0.01, -0.36), (0.014, -0.357), (0.02, -0.355), (0.029, -0.347), (0.037, -0.329), 
    (0.047, -0.316), (0.055, -0.314), (0.064, -0.306), (0.084, -0.311), (0.108, -0.326), 
    (0.132, -0.329), (0.167, -0.349), (0.21, -0.361), (0.255, -0.374), (0.283, -0.377), 
    (0.321, -0.356), (0.343, -0.356), (0.395, -0.333), (0.498, -0.298), (0.552, -0.287), 
    (0.678, -0.231), (0.761, -0.191), (0.823, -0.162), (0.998, -0.122), (1.18, -0.096), 
    (1.395, -0.068), (1.607, -0.071), (2.026, -0.035), (2.458, -0.012), (3.02, -0.007), 
    (3.479, 0.011), (3.907, -0.004), (4.113, -0.002), (4.33, -0.012), (4.559, -0.009), 
    (4.8, -0.042), (5.531, 0.009), (6.623, 0.021), (7.342, 0.042), (8.034, 0.05), 
    (8.568, 0.011), (10.0, 0.039)
]


if __name__ == "__main__":

    periods = np.linspace(0.01, 10, 30)
    rhos = ml21_correlation_model("sinter_total", periods)
    pass
