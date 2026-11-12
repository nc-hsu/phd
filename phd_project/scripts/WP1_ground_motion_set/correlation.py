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


# ---------------------------------------------------------------------------------------------
# Reading the correlation matrices back from the CSV files nb 002 writes
# ---------------------------------------------------------------------------------------------
# nb 002 saves every correlation matrix with ``rho_df.to_csv(fp, index=True)``. The rows and the
# columns are the same 3-level MultiIndex (component, IM, period), e.g. ("rotD50", "SA", 0.2),
# ("rotD50", "AvgSA([0, 3])", 3.0) or ("GM", "RSD595", "None"). The period level is MIXED: a
# float wherever the IM has a period, and the literal string "None" where it has none.
#
# A plain ``pd.read_csv(fp, header=[0, 1, 2], index_col=[0, 1, 2])`` does not give that back:
# pandas parses each label level as one type, so the periods come back as strings ("0.2") or the
# string "None" becomes NaN. Code downstream selects rows with labels such as
# ``idx["rotD50", "SA", 0.2]`` and ``idx[:, "RSD595", :]``, so the labels must be rebuilt with
# exactly the types the in-memory frames had. The functions below do that by reading every cell
# as text and converting the labels and the values explicitly.

# The regions and residual types nb 002 writes, in the order nb 002 builds them. The order is
# kept so that code iterating over the loaded dict (e.g. the HDF5 export in nb 002a) visits the
# entries in the same order as it did over the in-memory dict.
TECTONIC_REGIONS = ("asc", "sslab", "sinter", "vran")
RESIDUAL_TYPES = ("rho_total", "rho_dBe", "rho_dS2Ss", "rho_dWSes", "rho_total_B")

# number of label levels on each axis of the correlation matrices
_N_LABEL_LEVELS = 3


def _parse_period_label(label: str) -> float | str:
    """Convert one period label from its CSV text to the type it had in memory.

    "None" (an IM without a period, e.g. PGA or RSD595) stays the string "None"; anything else
    is a period and becomes a float. ``float()`` parses the shortest-repr text that ``to_csv``
    writes back to exactly the same float, so the labels compare equal to the originals.
    """
    return label if label == "None" else float(label)


def _labels_to_multiindex(label_rows: list[tuple[str, str, str]]) -> pd.MultiIndex:
    """Build the (component, IM, period) MultiIndex from text labels.

    The component and IM levels stay strings; only the period level is converted.
    """
    tuples = [(comp, im, _parse_period_label(period)) for comp, im, period in label_rows]
    return pd.MultiIndex.from_tuples(tuples)


def read_correlation_csv(file_path: Path) -> pd.DataFrame:
    """Read one correlation matrix saved with ``to_csv(..., index=True)``.

    Works for every matrix with (component, IM, period) labels on both axes: the raw
    (``*_{tag}.csv``) and positive-definite (``*_PD.csv``) matrices nb 002 writes, and the
    ASO24 matrix written by ``correlationmodels/export_aso24_correlations.py``.

    Parameters
    ----------
    file_path : Path
        CSV file of one correlation matrix.

    Returns
    -------
    pd.DataFrame
        The matrix with float64 values and the (component, IM, period) MultiIndex on both axes,
        with the period level holding floats and the string "None" as in nb 002.
    """
    # Read every cell as literal text. keep_default_na=False stops pandas turning the "None"
    # labels (and the empty top-left corner cells) into NaN.
    raw = pd.read_csv(file_path, header=None, dtype=str, keep_default_na=False)

    # Layout written by to_csv for a 3-level MultiIndex on both axes:
    #   rows 0-2, from column 3 on : the three column-label levels
    #   rows 3-,  columns 0-2      : the three row-label levels
    #   rows 3-,  columns 3-       : the correlation values
    n = _N_LABEL_LEVELS
    col_labels = list(zip(*[raw.iloc[lv, n:].tolist() for lv in range(n)]))
    row_labels = list(raw.iloc[n:, :n].itertuples(index=False, name=None))

    # astype(float64) parses each text value with Python's float(), which inverts the
    # shortest-repr text to_csv writes, so the values are bit-identical to the saved frame.
    values = raw.iloc[n:, n:].to_numpy().astype(np.float64)

    return pd.DataFrame(values,
                        index=_labels_to_multiindex(row_labels),
                        columns=_labels_to_multiindex(col_labels))


def load_positive_definite_correlations(
        results_folder: Path, fault_type: str,
        regions: tuple[str, ...] = TECTONIC_REGIONS,
        residual_types: tuple[str, ...] = RESIDUAL_TYPES) -> dict[str, dict[str, pd.DataFrame]]:
    """Rebuild the positive-definite correlation matrices of one fault type from their CSVs.

    nb 002 writes one file per region and residual type to
    ``{results_folder}/{fault_type}/{region}_{residual_type}_PD.csv``. This reads them back into
    the nested dict nb 002 holds in memory, ``{region: {residual_type: DataFrame}}``, with the
    same labels, label types and values (see ``read_correlation_csv``).

    Parameters
    ----------
    results_folder : Path
        The correlation-model results folder (``cfg["results"]["corr_model"]``).
    fault_type : str
        Fault-type sub-folder, e.g. "reverse".
    regions : tuple of str, optional
        Tectonic regions to load, in the order the dict should have.
    residual_types : tuple of str, optional
        Residual types to load for each region, in the order the dict should have.

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        ``{region: {residual_type: correlation matrix}}``.
    """
    folder = Path(results_folder) / fault_type
    return {tr: {tag: read_correlation_csv(folder / f"{tr}_{tag}_PD.csv")
                 for tag in residual_types}
            for tr in regions}
