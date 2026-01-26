"""
Functions used to calculate the correlation coefficients required for this project
 - rotD50 PGA, 
 - rotD50 SA(T), 
 - rotD50 AvgSA[0,3], 
 - rotD50 AvgSA[0,6], 
 - GeometricMean RSD595

 Functions are used in wp1pt3pt5-derivation_of_correlation_models.ipynb
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import IterationLimitWarning


idx = pd.IndexSlice

def is_pos_def(A):
    try:
        # Method 1: Cholesky (Fastest)
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def ensure_positive_definiteness(
        rho_df: pd.DataFrame, threshold: float=1e-15, n=100,
        print_out:bool=False) -> pd.DataFrame:
    
    A = rho_df.to_numpy()
    
    with warnings.catch_warnings():
        # Only ignore IterationLimitWarning within this block
        warnings.simplefilter("ignore", category=IterationLimitWarning)
        A_new = sm.stats.corr_nearest(A, threshold, n) # get the new PSD matrix

    # calculate some metrics to assess the effect of the changes:
    is_pd = is_pos_def(A_new)
    # 1. Relative change in the Frobenius Norm (Euclidean Norm for the matrix)
    # < 5% is ~ OK < 1% is safe
    rel_error = np.linalg.norm(A - A_new, 'fro') / np.linalg.norm(A, 'fro')

    # 2. Maximum elementwise deviation
    # checks if any individual correlation relationship is broken
    max_diff = np.max(np.abs(A - A_new))

    # 3. Error on the diagonal
    diag_error = np.max(np.abs(np.diag(A_new) - 1.0))
    
    # 3. Eigenvalues
    original_eigenvalues = np.linalg.eigvals(A)
    new_eigenvalues = np.linalg.eigvals(A_new)
    eigenvalues = np.vstack([original_eigenvalues, new_eigenvalues]).T

    rho_df_new = pd.DataFrame(A_new,
                              index=rho_df.index,
                              columns=rho_df.columns)

    if print_out:
        print("The old matrix is positive definite:", is_pos_def(A))
        print("The new matrix is positive definite:", is_pd)
        print(f"Relative error in Frobenius Norm: {rel_error:.6f}")
        print(f"Maximum absolute change of values: {max_diff:.6f}")
        print(f"Maximum error of diagonal terms: {diag_error}")

    return rho_df_new, is_pd, rel_error, max_diff, diag_error, eigenvalues


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


def total_correlation_from_components(
    residuals: list[np.ndarray], std_devs: list[np.ndarray],
    return_component_correlations: bool=False):

    # use the pandas functions because they ignore NA values
    cov_mats = [pd.DataFrame(r).cov().to_numpy() for r in residuals]
    corr_mats = [pd.DataFrame(r).corr().to_numpy() for r in residuals]
    sigma = np.sqrt(((np.vstack(std_devs)) ** 2).T.sum(axis=1))
    sig_mat = sigma.reshape(-1, 1) @ sigma.reshape(1, -1)

    total_correlation = 0
    for cm in cov_mats:
            total_correlation += cm
    total_correlation /= sig_mat

    if return_component_correlations:
            return (total_correlation, *corr_mats)
    return total_correlation


def total_sigma(std_devs: list[np.ndarray]) -> np.ndarray:
    return np.sqrt(((np.vstack(std_devs)) ** 2).T.sum(axis=1))


def total_correlation_from_total_residuals(
    im_data: pd.DataFrame, max_usable_T: pd.Series):
    
    total_residuals = im_data.loc[:, idx[:, :, :, "Delta"]].copy()
    for c in total_residuals.columns:
        if c[1] != "SA":
            t = 0.0
        else:
            t = c[2]
        
        mask = t <= max_usable_T
        total_residuals[c] = total_residuals[c].mask(~mask)
        total_residuals[c] =total_residuals[c].mask(~mask)

    im_labels = total_residuals.columns.droplevel(3)

    # Total correlation from the total residuals
    rho_Delta = total_residuals.corr()
    rho_Delta.index = im_labels
    rho_Delta.columns = im_labels
    return rho_Delta


def split_meta_and_im_data(df):
    metadata = df[["event_id", "station_code", "max_usable_T"]]
    im_data = df.drop(columns=["event_id", "station_code", "max_usable_T"])
    col_names = [c.split("_") for c in im_data.columns]
    col_names = [tuple([c[0], c[1], float(c[2]), c[3]]) if c[2] != "None" else tuple(c) for c in col_names]
    im_data.columns = pd.MultiIndex.from_tuples(col_names)

    return metadata, im_data


def get_partitioned_residuals(im_data: pd.DataFrame):
    dBe = im_data.loc[:, idx[:, :, :, "dBe"]].copy()
    dS2Ss = im_data.loc[:, idx[:, :, :, "dS2Ss"]].copy()
    dWSes = im_data.loc[:, idx[:, :, :, "dWSes"]].copy()

    return dBe, dS2Ss, dWSes


def get_standard_devs(partitioned_residuals: list[pd.DataFrame]):
    std_devs = [pr.std(axis=0) for pr in partitioned_residuals]
    sigma_B = total_sigma([s.to_numpy() for s in std_devs])
    sigma_B = pd.Series(sigma_B, index=std_devs[0].index)

    return (*std_devs, sigma_B)


def get_correlations_for_residual_dataset(residuals_filepath: Path):
    rhos = {}
    
    # load the data
    
    df = pd.read_csv(residuals_filepath, index_col=0)  



    metadata, im_data = split_meta_and_im_data(df)
    rho_Delta_B = total_correlation_from_total_residuals(im_data, metadata["max_usable_T"])
    partitioned_residuals = get_partitioned_residuals(im_data)
    std_devs = get_standard_devs(partitioned_residuals)
    
    # Correlation from the partitioned residuals
    rho_arrs = total_correlation_from_components(
        [pr.to_numpy() for pr in partitioned_residuals],
        [sd.to_numpy() for sd in std_devs[:-1]],
        return_component_correlations=True)


    resid_types = list(im_data.columns.get_level_values(3).unique())
    resid_types.remove("Delta")

    im_labels = im_data.loc[:, idx[:, :, :, resid_types[0]]].columns.droplevel(3)

    resid_types = ["total"] + resid_types
    # loop through the returned values and save
    for resid_type, r in zip(resid_types, rho_arrs):
        tag = f"rho_{resid_type}"
        rho_df = pd.DataFrame(r, index=im_labels, columns=im_labels)
        rhos[tag] = rho_df

    rhos["rho_total_B"] = rho_Delta_B

    return rhos





if __name__ == "__main__":
    from phd_project.config.config import load_config
    cfg = load_config()

    residuals_folder = cfg['data']['residuals'] 
    tr = "asc"
    partitioned_residuals_tail = "_partitioned_residuals_lmer.csv"
    residuals_filepath =  residuals_folder/ f"{tr}{partitioned_residuals_tail}"

    rhos = get_correlations_for_residual_dataset(residuals_filepath)