"""Compare MSA- and IDA-derived collapse fragilities and test whether they differ.

For each case-study structure the collapse fragility is estimated twice: by
Multiple-Stripe Analysis (binomial maximum likelihood over the stripes) and by an
incremental dynamic analysis following FEMA P695 (method-of-moments fit over the
collapse intensities). The two estimates disagree, and this module answers whether
that disagreement is larger than the sampling noise of the two estimators.

The method is parametric resampling, following Iervolino (2022), "Estimation
uncertainty for some common seismic fragility curve fitting methods". Each fitted
curve is treated as the truth of its own arm and resampled ``k_samples`` times at
the sample size that was actually run -- binomial draws at the stripe IMLs for MSA,
lognormal draws of ``n_ida_recs`` collapse IMLs for IDA. Every replicate is refitted
by that arm's own procedure, giving a cloud of ``(ln theta, ln beta)`` per arm. The
paired difference vector ``d = v_msa - v_ida`` defines a squared Mahalanobis
distance, and the point estimate's distance from the origin is compared against the
``1 - alpha`` quantile of the bootstrap distance distribution. Degenerate MSA fits
are dropped from *both* arms so the pairing survives.

Both estimators are biased on the log scale, so the point estimate is shifted by a
bias correction before the test. The IDA dispersion has a closed-form log-scale
bias that depends only on the record count (:func:`get_ida_bias_correction`); the
MSA bias has no closed form and is estimated by a second, larger bootstrap
(:func:`compute_bias_data`) because it depends on where the stripes fall relative
to the fragility. :func:`get_stripe_coverage` relates that per-site MSA bias to the
Fisher information the stripe placement carries about ``ln beta``, and splits it
into a raw-scale part and a Jensen gap.

Each replicate is also convolved with the site hazard curve to give a mean annual
frequency of collapse, so a fragility difference can be read as a risk difference
and as a margin against the EC8 target.

Analysis parameters -- bootstrap sizes, ``alpha``, the EC8 target, the intensity
measure, the record counts -- are deliberately *not* module constants. They are
passed in by the driver notebook
``notebooks/03_WP1_ground_motion_set/061-site_msa_ida_fragility_results_analysis.ipynb``,
which also owns all file I/O.

Bias notation follows Efron & Tibshirani (1993), *An Introduction to the
Bootstrap*: the bias estimate is eq. 10.2 and the corrected estimate is eq. 10.6.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import psi
from scipy.stats import kurtosis, lognorm, norm, skew, spearmanr

from hazrisk.hazard import mafe_to_poe
from hazrisk.risk import numerical_mafe
from standes.fitting import lognorm_mle_fit, lognorm_moment_fit
from standes.fragility_curves import (
    FragilityCurve, fragility_from_ida, fragility_from_msa)
from standes.intensitymeasures import IntensityMeasure

# Chart colours are fixed so every figure in the chapter reads the same way:
# MSA blue, IDA red, and the MSA-IDA difference green.
_COLOR_MSA = "b"
_COLOR_IDA = "r"
_COLOR_DELTA = "g"

# Regressions of a pooled quantity on a hazard proxy are split by seismicity band,
# because the bands differ in both the proxy and the response: pooling them can
# produce a slope whose sign neither band shows on its own. The band colours are
# deliberately not the MSA blue / IDA red pair, which means something else.
_BAND_COL = "seismicity"
_BAND_ORDER = ("all", "lowmod", "high")
_BAND_COLORS = {"all": "k", "lowmod": "tab:purple", "high": "tab:orange"}
_BAND_LABELS = {"all": "all", "lowmod": "low/mod", "high": "high"}

# Quantities that already carry the site hazard level in their denominator. They
# must never be regressed on that same hazard level: the shared term manufactures a
# correlation out of arithmetic alone, and because the hazard at different return
# periods is correlated at r ~ 0.9 across these sites, picking a different return
# period for the abscissa does not escape it either.
DESIGN_NORMALISED_COLUMNS = ("theta_msa_norm", "theta_ida_norm")

# The two bootstrap arms must run on disjoint RNG streams, but both should still be
# reproducible from a single per-site seed, so each arm gets a fixed offset. The
# same offset belongs to the same arm everywhere, in the bias assessment and in the
# hypothesis test alike.
_SEED_OFFSET_MSA = 1000
_SEED_OFFSET_IDA = 2000

# Collapse probabilities are clipped away from {0, 1} before the Fisher information
# is formed, because p * (1 - p) appears in a denominator.
_P_CLIP = 1e-9

# A resampled MSA fit is treated as degenerate once it exceeds this multiple of the
# fit it was drawn from. The affected replicates sit many orders of magnitude away
# (theta ~ 1e21), so anything from roughly 5 to 50 selects the same ones.
_MAX_FIT_RATIO = 10.0

# The case-study sites are laid out on a fixed grid: 30 low/moderate-seismicity sites
# followed by 30 high-seismicity ones, and within each band six regional groupings of
# five consecutive sites. Sites without a usable MSA fit are simply absent from the
# results, and the cross-site charts key the x-axis on the site number so the gap they
# leave stays in the right place.
_SITES_PER_REGION = 5
_N_REGIONS = 6
_SEISMICITY_BANDS = ((0, 30, "Low / moderate seismicity"),
                     (30, 60, "High seismicity"))


# =============================================================================
# Loading
# =============================================================================

def structure_tag(site_idx: int, n_storeys: int) -> str:
    """Return the canonical identifier for a case-study structure."""
    return f"{n_storeys}s_cbf_dc2_site{site_idx}"


def load_site_fragility_curves(
    frag_root: Path,
    sites: Sequence[int],
    n_storeys: Sequence[int],
) -> tuple[dict, dict]:
    """Load the MSA and IDA collapse fragilities for every site and storey count.

    Returns ``(fragility_curves, fragility_curves_flat)``. The first is nested
    ``[site][tag][arm]``, the second is flat ``[arm][tag]``. A structure is skipped
    entirely unless *both* arms are present, so the two arms are always paired.
    """
    fragility_curves = {}
    fragility_curves_flat = {"msa": {}, "ida-femap695": {}}

    for site in sites:
        site_fcs = {}
        for n in n_storeys:
            tag = structure_tag(site, n)
            structure_fcs = {}

            for arm, suffix in [("msa", "msa"), ("ida-femap695", "ida_femap695")]:
                fc_path = (frag_root / f"site_{site}"
                           / f"{tag}_{suffix}_collapsefragility_AvgSA_03.json")
                try:
                    with open(fc_path, "r") as file:
                        fc = json.load(file)
                        structure_fcs[arm] = {k: np.array(v) if isinstance(v, list) else v
                                                for k, v in fc.items()}
                except FileNotFoundError:
                    print(f"No Fragility Curve for {arm}: site {site} and {n}s. "
                          "Skipping...")
            
            site_fcs[tag] = structure_fcs
            for arm in fragility_curves_flat:
                try:
                    fragility_curves_flat[arm][tag] = structure_fcs[arm]
                except KeyError:
                    pass

        fragility_curves[site] = site_fcs

    return fragility_curves, fragility_curves_flat


# =============================================================================
# Parametric bootstrap
# =============================================================================

def msa_fit_ok(
    thetas: np.ndarray,
    betas: np.ndarray,
    theta_ref: float | None = None,
    beta_ref: float | None = None,
    max_ratio: float = _MAX_FIT_RATIO,
) -> np.ndarray:
    """Return a boolean mask flagging the usable binomial-MLE fits.

    Both parameters must be positive and finite. Given the reference fit the
    replicates were drawn from, fits that have run away from it by more than
    ``max_ratio`` are rejected as well. When the resampled collapse counts come out
    flat or non-monotonic across the stripes the likelihood has no interior maximum:
    ``theta`` and ``beta`` diverge together and the fitted curve degenerates into a
    horizontal line through the stripe range. Those fits are positive and finite, so
    the finiteness check alone does not catch them, and because the runaway values
    are astronomically large they destroy every moment taken over the cloud.

    The same criterion is applied in :func:`msa_estimator_bias`, so the bootstrap
    cloud and the bias estimate are computed over the same set of replicates.
    """
    thetas = np.asarray(thetas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    ok = (thetas > 0) & (betas > 0) & np.isfinite(thetas) & np.isfinite(betas)
    if theta_ref is not None:
        ok &= thetas < max_ratio * theta_ref
    if beta_ref is not None:
        ok &= betas < max_ratio * beta_ref
    return ok


def bootstrap_msa(
    fc_msa: dict,
    n_tests: int,
    k_samples: int,
    im: IntensityMeasure,
    seed: int = 1,
) -> tuple[list[FragilityCurve], np.ndarray]:
    """Resample binomial collapse counts at the stripes and refit each replicate.

    Returns the replicate fragilities together with a boolean mask flagging the
    usable fits. The degenerate ones are flagged rather than removed so the caller
    can drop the matching IDA replicates and keep the two arms paired.
    """
    rng = np.random.default_rng(seed)

    collapses = []
    for iml in fc_msa["efc"][0, :]:
        p = lognorm.cdf(iml, s=fc_msa["dispersion"], scale=fc_msa["median"])

        im_collapse_samples = rng.binomial(n_tests, p, size=k_samples)
        collapses.append(im_collapse_samples)

    ims = fc_msa["efc"][0, :]
    collapses = np.column_stack(collapses)

    sim_fcs_msa = []
    for row in collapses:
        sim_fcs_msa.append(fragility_from_msa(ims, row, n_tests, im))

    ok = msa_fit_ok([fc.median for fc in sim_fcs_msa],
                    [fc.dispersion for fc in sim_fcs_msa],
                    fc_msa["median"], fc_msa["dispersion"])

    return sim_fcs_msa, ok


def bootstrap_ida(
    fc_ida: dict,
    n_recs: int,
    k_samples: int,
    im: IntensityMeasure,
    seed: int = 1,
) -> list[FragilityCurve]:
    """Resample lognormal collapse intensities and refit each replicate by moments."""
    rng = np.random.default_rng(seed)
    # sample the N_ida points randomly from this distribution for k_trials
    collapse_imls = rng.lognormal(np.log(fc_ida["median"]), fc_ida["dispersion"],
                                  size=(k_samples, n_recs))

    # fit the distribution of the median and dispersion
    sim_fcs_ida = []
    for row in collapse_imls:
        sim_fcs_ida.append(fragility_from_ida(row, im))

    return sim_fcs_ida


# =============================================================================
# Descriptive statistics
# =============================================================================

def get_stats(values) -> dict[str, float]:
    """Return the summary statistics recorded for every bootstrapped quantity."""
    return {"N_obs": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "variance": np.var(values),
            "std": np.std(values, ddof=1),
            "2.5pc": np.percentile(values, 2.5),
            "5pc": np.percentile(values, 5),
            "16pc": np.percentile(values, 16),
            "84pc": np.percentile(values, 84),
            "95pc": np.percentile(values, 95),
            "97.5pc": np.percentile(values, 97.5),
            "skewness": skew(values),
            "kurtosis": kurtosis(values)
            }


def print_stats(stat_dicts: list[dict[str, float]],
                headings: list[str] | None = None) -> None:
    """Print one or more :func:`get_stats` dictionaries as an aligned table."""
    title_string = f"{'Stat':12}"
    if headings:
        for heading in headings:
            title_string += f"{heading:>12}"
    title_string += "\n" + "-" * (len(stat_dicts) + 1) * 12

    keys = stat_dicts[0].keys()
    values = [d.values() for d in stat_dicts]

    value_string = ""
    for z in zip(keys, *values):
        value_string += f"{z[0]:12}"
        if any(np.array(z[1:]) < 0.001):
            for vi in z[1:]:
                value_string += f"{vi:12.3e}"
        else:
            for vi in z[1:]:
                value_string += f"{vi:12.3f}"
        value_string += "\n"

    print(title_string)
    print(value_string)


# =============================================================================
# Hypothesis test
# =============================================================================

def squared_mahalanobis_distance(
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the squared Mahalanobis distances of ``d`` about its own mean.

    Also returns the mean point and the covariance matrix, both of which the
    hypothesis test needs.
    """
    d_bar = np.mean(d, axis=0)    # the mean point
    Gamma = 1 / (len(d) - 1) * (d - d_bar).T @ (d - d_bar)    # the covariance matrix
    Gamma_inv = np.linalg.inv(Gamma)

    # calculate the Mahalanobis distance for each element
    D2Mis = np.sum((d - d_bar) @ Gamma_inv * (d - d_bar), axis=1)

    return D2Mis, d_bar, Gamma


def hypothesis_test(
    dis: np.ndarray,
    d_orig: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Test whether the observed difference vector is further from zero than noise.

    ``d_orig`` is the point being tested -- in practice the bias-corrected observed
    difference. Its squared Mahalanobis distance from the origin is compared against
    the ``1 - alpha`` quantile of the bootstrap distance distribution, which is
    equivalent to asking whether the origin lies inside the confidence region
    centred on ``d_orig``.

    The threshold is a plain percentile rather than a bias-corrected interval: the
    ``D2M`` distribution is bounded at zero and strongly right-skewed by
    construction, so a BC or BCa interval on it is not meaningful. The bias is
    instead removed from ``d_orig`` before it reaches this function.
    """
    # calculate difference vectors (sample statistics)
    D2Mis, d_bar, Gamma = squared_mahalanobis_distance(dis)
    D2M_test_point = ((d_orig).reshape((1, 2)) @ np.linalg.inv(Gamma)
                      @ (d_orig).reshape((2, 1)))[0][0]

    CI_distance = np.quantile(D2Mis, (1 - alpha))

    # achieved significance level
    asl = np.mean(D2Mis >= D2M_test_point)

    return D2Mis, d_bar, Gamma, D2M_test_point, CI_distance, asl


def get_CI_ellipse(d_centre: np.ndarray, CI_distance: float,
                   Gamma: np.ndarray) -> np.ndarray:
    """Return the ``(2, 200)`` confidence-region ellipse centred on ``d_centre``."""
    phi = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.vstack((np.cos(phi), np.sin(phi)))  # Shape: (2, 200)

    L = np.linalg.cholesky(Gamma)  # compute Cholesky factor L
    # scale/rotate unit circle
    CI_ellipse = d_centre[:, None] + np.sqrt(CI_distance) * (L @ unit_circle)
    return CI_ellipse


# =============================================================================
# Risk metrics
# =============================================================================

def calculate_mafcs(sim_fcs: list[FragilityCurve], haz_mafes: np.ndarray,
                    imls: np.ndarray) -> np.ndarray:
    """Convolve each replicate fragility with the hazard to get its MAF of collapse.

    ``haz_mafes`` is the interpolated mean-annual-frequency-of-exceedance column
    evaluated at ``imls``, not the two-column hazard curve.
    """
    mafcs = []
    for sim_fc in sim_fcs:
        fc = lognorm.cdf(imls, s=sim_fc.dispersion, scale=sim_fc.median)
        mafcs.append(numerical_mafe(haz_mafes, fc))
    return np.array(mafcs)


def calculate_mafc_margins(
    mafcs_msa: np.ndarray,
    mafcs_ida: np.ndarray,
    target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two EC8 margins and the MSA-relative difference in collapse rate."""
    # % differences
    MAFC_diff = (mafcs_msa - mafcs_ida)
    # relative to MSA
    mafc_rel_msa = MAFC_diff / mafcs_msa

    # relative to EC8 target
    mafc_margin_msa = (target / mafcs_msa)
    mafc_margin_ida = (target / mafcs_ida)
    return mafc_margin_msa, mafc_margin_ida, mafc_rel_msa


# =============================================================================
# Estimator bias
# =============================================================================

def _calc_bias(boot_estimates: np.ndarray,
               orig_estimate: float) -> dict[str, float]:
    """Summarise the log-scale bias of one estimator from its bootstrap replicates."""
    lboot = np.log(boot_estimates)
    out = {
        "bias": lboot.mean() - np.log(orig_estimate),      # E&T eq. 10.2, log scale
        "se_est": lboot.std(ddof=1),                       # bootstrap se of the estimator
        "mcse": lboot.std(ddof=1) / np.sqrt(len(lboot)),   # MC error on the bias itself
        "raw_rel": (boot_estimates.mean() - orig_estimate) / orig_estimate,  # raw-scale bias
        "jensen": -0.5 * lboot.var(ddof=1),
    }
    out["bias/se"] = abs(out["bias"] / out["se_est"])
    return out


def ida_estimator_bias(
    fc_ida: dict,
    b: int,
    n_ida_recs: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Bootstrap the log-scale bias of the IDA median and dispersion estimators.

    Resamples ``n_ida_recs`` lognormal collapse intensities and refits by moments.
    """
    rng = np.random.default_rng(seed)
    theta_est = fc_ida["median"]
    beta_est = fc_ida["dispersion"]
    sims = rng.lognormal(np.log(theta_est), beta_est, size=(b, n_ida_recs))
    th_i, be_i = np.array([lognorm_moment_fit(s) for s in sims]).T

    # the bias is taken on the log scale because that is what the test statistic uses
    theta_bias_summary = _calc_bias(th_i, theta_est)
    beta_bias_summary = _calc_bias(be_i, beta_est)
    return theta_bias_summary, beta_bias_summary


def msa_estimator_bias(
    fc_msa: dict,
    b: int,
    imls: np.ndarray,
    n_stripe_recs: int,
    seed: int,
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Bootstrap the log-scale bias of the MSA median and dispersion estimators.

    Resamples binomial counts at the stripes and refits by MLE. Set ``verbose`` to
    report how many replicates produced degenerate fits and were discarded.
    """
    rng = np.random.default_rng(seed)
    theta_est = fc_msa["median"]
    beta_est = fc_msa["dispersion"]
    p_true = lognorm.cdf(imls, s=beta_est, scale=theta_est)
    counts = rng.binomial(n_stripe_recs, p_true, size=(b, len(imls)))

    fits = [lognorm_mle_fit(imls, c, np.full(len(imls), n_stripe_recs))
            for c in counts]
    th_m, be_m = np.array(fits).T

    # protect against bad fits
    ok = msa_fit_ok(th_m, be_m, theta_est, beta_est)
    if verbose and not ok.all():
        print(f"degenerate MSA fits dropped: {np.sum(~ok)} / {b}")
    th_m, be_m = th_m[ok], be_m[ok]

    # the bias is taken on the log scale because that is what the test statistic uses
    theta_bias_summary = _calc_bias(th_m, theta_est)
    beta_bias_summary = _calc_bias(be_m, beta_est)
    return theta_bias_summary, beta_bias_summary


def compute_bias_data(
    fragility_curves: dict,
    n_storeys: Sequence[int],
    b: int,
    n_ida_recs: int,
    n_stripe_recs: int,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Bootstrap the bias of all four estimators at every structure.

    Returns one DataFrame per estimator, keyed ``msa_theta``, ``msa_beta``,
    ``ida_theta``, ``ida_beta`` and indexed by structure tag. Each arm is seeded
    from the site index with its own offset so the two arms stay independent.
    """
    lntheta_bias_ida = {}
    lnbeta_bias_ida = {}
    lntheta_bias_msa = {}
    lnbeta_bias_msa = {}

    for site in list(fragility_curves.keys()):
        for ns in n_storeys:
            tag = structure_tag(site, ns)
            # get the fragility curve estimates
            try:
                fc_msa = fragility_curves[site][tag]["msa"]
                fc_ida = fragility_curves[site][tag]["ida-femap695"]
            except KeyError:
                continue

            # calculate the estimator bias
            lntheta_bias_ida[tag], lnbeta_bias_ida[tag] = ida_estimator_bias(
                fc_ida, b, n_ida_recs, seed=_SEED_OFFSET_IDA + site)
            imls = fc_msa["efc"].T[:, 0]
            lntheta_bias_msa[tag], lnbeta_bias_msa[tag] = msa_estimator_bias(
                fc_msa, b, imls, n_stripe_recs, seed=_SEED_OFFSET_MSA + site,
                verbose=verbose)

    return {"msa_theta": pd.DataFrame.from_dict(lntheta_bias_msa, orient="index"),
            "msa_beta": pd.DataFrame.from_dict(lnbeta_bias_msa, orient="index"),
            "ida_theta": pd.DataFrame.from_dict(lntheta_bias_ida, orient="index"),
            "ida_beta": pd.DataFrame.from_dict(lnbeta_bias_ida, orient="index")}


# =============================================================================
# Bias corrections
# =============================================================================

def get_ida_bias_correction(n_ida_records: int) -> float:
    """Return the closed-form log-scale bias of the IDA dispersion estimator.

    The moment fit gives an unbiased ``beta ** 2``, but the square root and the log
    are both concave, so ``ln beta`` comes out low by a fixed amount that depends
    only on the record count. The result is negative.
    """
    ida_bias = 0.5 * (psi((n_ida_records - 1) / 2)
                      - np.log((n_ida_records - 1) / 2))
    return ida_bias


def get_average_biases(bias_dfs: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Return the across-site mean bias of each estimator."""
    return {k: v["bias"].mean() for k, v in bias_dfs.items()}


def add_bias_mcse_ratio(bias_df: pd.DataFrame) -> pd.Series:
    """Add ``bias/mcse`` in place: is the estimated bias resolved from zero?"""
    bias_df["bias/mcse"] = np.abs(bias_df["bias"]) / bias_df["mcse"]
    return bias_df["bias/mcse"]


def add_bias_corrected_mcse_ratio(bias_df: pd.DataFrame) -> pd.Series:
    """Add ``bias_corrected/mcse`` in place: is the residual resolved from zero?"""
    bias_df["bias_corrected/mcse"] = (np.abs(bias_df["bias_corrected"])
                                      / bias_df["mcse"])
    return bias_df["bias_corrected/mcse"]


def add_bias_correction(bias_df: pd.DataFrame,
                        correction: float | pd.Series) -> tuple[pd.Series, pd.Series]:
    """Subtract a bias correction in place and add the residual diagnostics.

    ``correction`` may be a scalar applied to every site, or a Series giving each
    site its own correction.
    """
    bias_df["bias_corrected"] = bias_df["bias"] - correction
    bias_df["bias/se_corrected"] = (np.abs(bias_df["bias_corrected"])
                                    / bias_df["se_est"])
    return bias_df["bias_corrected"].copy(), bias_df["bias/se_corrected"].copy()


def build_estimator_frame(fragility_curves_flat: dict) -> pd.DataFrame:
    """Return the log point estimates of both arms, one row per structure."""
    msa_theta_est = pd.Series({k: np.log(v["median"])
                               for k, v in fragility_curves_flat["msa"].items()})
    msa_beta_est = pd.Series({k: np.log(v["dispersion"])
                              for k, v in fragility_curves_flat["msa"].items()})
    ida_theta_est = pd.Series(
        {k: np.log(v["median"])
         for k, v in fragility_curves_flat["ida-femap695"].items()})
    ida_beta_est = pd.Series(
        {k: np.log(v["dispersion"])
         for k, v in fragility_curves_flat["ida-femap695"].items()})
    estimator_df = pd.DataFrame(
        [msa_theta_est, msa_beta_est, ida_theta_est, ida_beta_est]).T
    estimator_df.columns = ["msa_t", "msa_b", "ida_t", "ida_b"]
    return estimator_df


def apply_bias_corrections(bias_data: dict[str, pd.DataFrame],
                           n_ida_recs: int) -> dict[str, list]:
    """Correct every estimator in place and return the per-structure test-point shift.

    ``ln theta`` is left uncorrected in both arms: its bias is immaterial relative
    to the estimator's own standard error. ``ln beta`` is corrected in both arms,
    the IDA arm by the closed-form constant and the MSA arm by *its own per-site*
    bootstrap estimate -- which is why the MSA residual is identically zero by
    construction, while the IDA residual is a genuine test of the analytic result.

    The returned dict maps each structure tag to the ``[d_lntheta, d_lnbeta]``
    correction to subtract from that structure's observed difference vector.
    """
    ida_beta_corr = get_ida_bias_correction(n_ida_recs)
    # order matches bias_data: msa_theta, msa_beta, ida_theta, ida_beta
    bias_corrections = [0, bias_data["msa_beta"]["bias"], 0, ida_beta_corr]

    for df, c in zip(bias_data.values(), bias_corrections):
        add_bias_mcse_ratio(df)
        add_bias_correction(df, c)
        add_bias_corrected_mcse_ratio(df)

    return {t: [0, bias_data["msa_beta"].loc[t, "bias"] - ida_beta_corr]
            for t in bias_data["msa_beta"].index}


# =============================================================================
# Stripe placement and the MSA dispersion bias
# =============================================================================
# The Fisher information for ln(beta) from a set of independent binomial stripes is
#
#       I = sum_j  n * [phi(z_j) * z_j]^2 / (p_j * (1 - p_j)),
#       z_j = ln(IML_j / theta) / beta,   p_j = Phi(z_j)
#
# The factor phi(z)*z vanishes at z = 0, so a stripe placed at the median
# constrains theta only and carries NO information about the dispersion.
# Information about beta comes from stripes out in the tails. Sites whose stripes
# span a wide range of collapse probability therefore estimate beta precisely, and
# sites whose stripes cluster centrally do not.
#
# se_pred = 1/sqrt(I) is the asymptotic standard error implied by the stripe
# placement alone. It understates the bootstrap se by ~15% at n=30 per stripe, so
# use it for ranking sites rather than for absolute values.
#
# The bias in ln(beta) splits exactly into two parts:
#
#   E[ln b*] - ln b  =  ln( E[b*] / b )      +  ( E[ln b*] - ln E[b*] )
#                       ^^^^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^^^^^^^
#                       raw-scale bias          Jensen gap  (<= 0 always)
#
#   raw-scale bias : is the fit centred on the right dispersion, in beta units?
#                    A property of the MLE and the stripe design.
#   Jensen gap     : what taking the log does to a spread-out estimate.
#                    Depends only on the SPREAD, ~ -0.5 * var(ln b*).
# =============================================================================

def get_stripe_coverage(fc_msa: dict, n_stripe_recs: int) -> dict[str, float]:
    """Return the collapse probabilities at the stripes and their ln(beta) information."""
    imls = np.asarray(fc_msa["efc"][0, :], dtype=float)
    z = np.log(imls / fc_msa["median"]) / fc_msa["dispersion"]
    p = np.clip(norm.cdf(z), _P_CLIP, 1 - _P_CLIP)
    dp_dlnbeta = -norm.pdf(z) * z                 # d p / d ln(beta)
    fisher = np.sum(n_stripe_recs * dp_dlnbeta ** 2 / (p * (1 - p)))
    return {"p_min": p.min(),
            "p_max": p.max(),
            "p_range": np.ptp(p),
            "n_central": int(((p > 0.2) & (p < 0.8)).sum()),
            "se_pred": 1 / np.sqrt(fisher)}


def build_stripe_coverage_frame(
    bias_df: pd.DataFrame,
    fcs_flat: dict,
    n_stripe_recs: int,
) -> pd.DataFrame:
    """Join the stripe-coverage metrics to the bias and its two components."""
    cov = pd.DataFrame.from_dict(
        {tag: get_stripe_coverage(fcs_flat["msa"][tag], n_stripe_recs)
         for tag in bias_df.index},
        orient="index").loc[bias_df.index]
    cov["bias"] = bias_df["bias"]
    cov["se_boot"] = bias_df["se_est"]
    # decomposition: bias = raw-scale bias + Jensen gap
    cov["raw_scale"] = np.log1p(bias_df["raw_rel"])  # ln(1 + relative bias of beta)
    cov["jensen"] = bias_df["jensen"]                # -0.5 * var(ln beta*), <= 0
    return cov


# =============================================================================
# Plotting
# =============================================================================

def style_legend(ax, **kwargs) -> None:
    """Add a legend with the black frame used throughout the chapter."""
    leg = ax.legend(**kwargs)
    leg.get_frame().set_edgecolor("k")


def plot_fragility_curves(ax, fc_msa: dict, fc_ida: dict,
                          sim_fcs_msa: list[FragilityCurve],
                          sim_fcs_ida: list[FragilityCurve]) -> np.ndarray:
    """Plot both fitted curves with their bootstrap envelopes.

    Returns the IML grid it built, which the caller reuses for the MAFC integration.
    """
    im_max = max([lognorm.ppf(0.95, s=fc.dispersion, scale=fc.median)
                  for fc in sim_fcs_msa])
    imls = np.linspace(0, im_max, 75)  # in g

    fcs = []
    for sfc in sim_fcs_ida:
        fcs.append(lognorm.cdf(imls, s=sfc.dispersion, scale=sfc.median))
    fcs = np.column_stack(fcs)
    ida_bounds = np.column_stack([np.min(fcs, axis=1), np.max(fcs, axis=1)])

    fcs = []
    for sfc in sim_fcs_msa:
        fcs.append(lognorm.cdf(imls, s=sfc.dispersion, scale=sfc.median))
    fcs = np.column_stack(fcs)
    msa_bounds = np.column_stack([np.min(fcs, axis=1), np.max(fcs, axis=1)])

    base_ida_fc = lognorm.cdf(imls, s=fc_ida["dispersion"], scale=fc_ida["median"])
    base_msa_fc = lognorm.cdf(imls, s=fc_msa["dispersion"], scale=fc_msa["median"])
    ax.fill_between(imls, ida_bounds[:, 0], ida_bounds[:, 1], color=_COLOR_IDA,
                    alpha=0.2, label="IDA variation")
    ax.fill_between(imls, msa_bounds[:, 0], msa_bounds[:, 1], color=_COLOR_MSA,
                    alpha=0.2, label="MSA variation")
    ax.plot(imls, base_ida_fc, color=_COLOR_IDA, ls="-", lw=2, label="IDA-FEMA-P695")
    ax.plot(imls, base_msa_fc, color=_COLOR_MSA, ls="-", lw=2, label="MSA")
    ax.grid(ls="-.", color="0.8")
    ax.set_xlim(0)

    ax.set_ylabel("Probability of Collapse, P[C]")
    ax.set_xlabel("AvgSA[0,3], in [g]")

    style_legend(ax)

    return imls


def plot_theta_beta_hists(ax1, ax2, thetas_msa, betas_msa, thetas_ida, betas_ida,
                          n_bins: int = 20) -> None:
    """Histogram the bootstrapped median and dispersion of both arms."""
    ax1.hist(thetas_msa, label="MSA", facecolor=[(0.0, 0.0, 1.0, 0.5)],
             edgecolor=[(0.0, 0.0, 1.0, 1.0)], density=True, bins=n_bins)
    ax1.hist(thetas_ida, label="IDA", facecolor=[(1.0, 0.0, 0.0, 0.5)],
             edgecolor=[(1.0, 0.0, 0.0, 1.0)], density=True, bins=n_bins)

    ax2.hist(betas_msa, label="MSA", facecolor=[(0.0, 0.0, 1.0, 0.5)],
             edgecolor=[(0.0, 0.0, 1.0, 1.0)], density=True, bins=n_bins)
    ax2.hist(betas_ida, label="IDA", facecolor=[(1.0, 0.0, 0.0, 0.5)],
             edgecolor=[(1.0, 0.0, 0.0, 1.0)], density=True, bins=n_bins)

    ax1.set_xlabel(r"Median Collapse Capacity, $\theta$")
    ax2.set_xlabel(r"Dispersion, $\beta$")
    ax1.set_ylabel("PDF")

    style_legend(ax1)
    style_legend(ax2)


def plot_theta_beta_correlation(ax, thetas_msa, betas_msa,
                                thetas_ida, betas_ida) -> None:
    """Scatter dispersion against median for both arms."""
    ax.plot(thetas_msa, betas_msa, marker=".", mfc=_COLOR_MSA, mec=_COLOR_MSA,
            ls="none", label="MSA")
    ax.plot(thetas_ida, betas_ida, marker=".", mfc=_COLOR_IDA, mec=_COLOR_IDA,
            ls="none", label="IDA-FEMA-P695")

    ax.set_ylabel(r"$\beta$")
    ax.set_xlabel(r"$\theta$")
    ax.grid(ls="-.", color="0.8")
    style_legend(ax)


def visualise_hypothesis_test(ax, D2Mis, alpha: float, D2M_test_point: float,
                              CI_distance: float, n_bins: int = 30) -> None:
    """Plot the bootstrap distance distribution against the test point and threshold."""
    ax.hist(D2Mis, density=True, facecolor=[(0.0, 0.0, 1.0, 0.5)],
            edgecolor=[(0.0, 0.0, 1.0, 1.0)], bins=n_bins)
    ax.axvline(CI_distance, color='k', linewidth=1.5, ls="--",
               label=rf"$\alpha = {alpha:.2f}$ CI Limit")
    ax.axvline(D2M_test_point, color='r', linewidth=1.5, ls="--",
               label="Test Distance")
    ax.axvspan(0, CI_distance, color="g", alpha=0.3,
               label="$H_0$ cannot be rejected")

    ax.set_ylabel("PDF")
    ax.set_xlabel("$D^2_M$, [-]")

    ax.set_xlim(0)
    style_legend(ax)


def plot_msa_ida_diff(ax, dis, d_bar, d_orig, d_corr, CI_distance, Gamma) -> None:
    """Plot the difference-vector cloud with the confidence region about ``d_corr``.

    The region is centred on the corrected estimate, so the test reads off the
    figure directly: the curves differ significantly when the origin falls outside.
    """
    CI_ellipse = get_CI_ellipse(d_corr, CI_distance, Gamma)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.plot(dis[:, 0], dis[:, 1], marker=".", mfc=_COLOR_DELTA, mec=_COLOR_DELTA,
            ls="none", label="Bootstr. vals")
    ax.plot(d_bar[0], d_bar[1], marker="o", mfc="r", mec="k", ls="none", label="Mean")
    ax.plot(d_orig[0], d_orig[1], marker="o", mfc="b", mec="k", ls="none",
            label="Orig. estimate")
    ax.plot(d_corr[0], d_corr[1], marker="o", mfc="m", mec="k", ls="none",
            label="Corr. estimate")
    ax.plot(CI_ellipse[0, :], CI_ellipse[1, :], ls="-", color="k", label="Sig. limit")
    ax.set_ylabel(r"$\Delta_{\ln{\beta}} = \ln{\beta_{MSA}} - \ln{\beta_{IDA}}$")
    ax.set_xlabel(r"$\Delta_{\ln{\theta}} = \ln{\theta_{MSA}} - \ln{\theta_{IDA}}$")

    ax.grid(ls="-.", color="0.8")
    style_legend(ax)


def plot_msa_ida_ratios(ax1, ax2, theta_ratio, beta_ratio,
                        n_bins: int = 20) -> None:
    """Histogram the MSA/IDA ratios of median and dispersion."""
    ax1.hist(theta_ratio, label=r"$\theta_{MSA}/\theta_{IDA}$",
             facecolor=[(0.0, 0.0, 1.0, 0.5)], edgecolor=[(0.0, 0.0, 1.0, 1.0)],
             density=True, bins=n_bins)
    ax2.hist(beta_ratio, label=r"$\beta_{MSA}/\beta_{IDA}$",
             facecolor=[(1.0, 0.0, 0.0, 0.5)], edgecolor=[(1.0, 0.0, 0.0, 1.0)],
             density=True, bins=n_bins)

    ax1.set_xlabel(r"$\theta_{MSA}/\theta_{IDA}$")
    ax2.set_xlabel(r"$\beta_{MSA}/\beta_{IDA}$")
    ax1.set_ylabel("PDF")
    ax2.set_ylabel("PDF")

    style_legend(ax1)
    style_legend(ax2)


def plot_mafc_distributions(ax, mafcs_msa, mafcs_ida, n_bins: int = 20) -> None:
    """Histogram the mean annual frequency of collapse for both arms."""
    ax.hist(mafcs_msa, label="MSA", facecolor=[(0.0, 0.0, 1.0, 0.5)],
            edgecolor=[(0.0, 0.0, 1.0, 1.0)], density=True, bins=n_bins)
    ax.hist(mafcs_ida, label="IDA", facecolor=[(1.0, 0.0, 0.0, 0.5)],
            edgecolor=[(1.0, 0.0, 0.0, 1.0)], density=True, bins=n_bins)

    ax.set_xlabel(r"$\lambda_C$")
    ax.set_ylabel("PDF")

    style_legend(ax)


def plot_mafc_difference_distributions(ax1, ax2, mafc_rel_msa, mafc_margin_msa,
                                       mafc_margin_ida, n_bins: int = 20) -> None:
    """Histogram the relative collapse-rate difference and the two EC8 margins."""
    ax1.hist(mafc_rel_msa, label=r"$\Delta\lambda_{C,%MSA}$",
             facecolor=[(1.0, 0.0, 1.0, 0.5)], edgecolor=[(1.0, 0.0, 1.0, 1.0)],
             density=True, bins=n_bins)
    ax2.hist(mafc_margin_msa, label=r"MSA", facecolor=[(0.0, 0.0, 1.0, 0.5)],
             edgecolor=[(0.0, 0.0, 1.0, 1.0)], density=True, bins=n_bins)
    ax2.hist(mafc_margin_ida, label=r"IDA", facecolor=[(1.0, 0.0, 0.0, 0.5)],
             edgecolor=[(1.0, 0.0, 0.0, 1.0)], density=True, bins=n_bins)

    ax1.set_xlabel(r"Difference relative to MSA, $\Delta\lambda_{C,MSA}$")
    ax2.set_xlabel(r"EC8 Margin, $\lambda_{C,EC8} / \lambda_{C,x}$")
    ax1.set_ylabel("PDF")
    ax2.set_ylabel("PDF")

    style_legend(ax2)


def plot_pc_distributions(ax, pcs_msa, pcs_ida, t, n_bins: int = 20) -> None:
    """Histogram the probability of collapse in ``t`` years for both arms."""
    ax.hist(pcs_msa, label="MSA", facecolor=[(0.0, 0.0, 1.0, 0.5)],
            edgecolor=[(0.0, 0.0, 1.0, 1.0)], density=True, bins=n_bins)
    ax.hist(pcs_ida, label="IDA", facecolor=[(1.0, 0.0, 0.0, 0.5)],
            edgecolor=[(1.0, 0.0, 0.0, 1.0)], density=True, bins=n_bins)

    ax.set_xlabel(rf"P[collapse] in {t} years")
    ax.set_ylabel("PDF")

    style_legend(ax)


def _plot_biases(ax, site_labels, msa_bias, ida_bias,
                 avg_msa_bias, avg_ida_bias) -> None:
    """Scatter one estimator's per-site bias for both arms, with the site means."""
    ax.axhline(0, ls="-", color="k", lw=0.75)
    ax.plot(site_labels, msa_bias, marker=".", mfc=_COLOR_MSA, mec=_COLOR_MSA,
            ls="none", label="MSA")
    ax.plot(site_labels, ida_bias, marker=".", mfc=_COLOR_IDA, mec=_COLOR_IDA,
            ls="none", label="IDA")

    ax.axhline(avg_msa_bias, ls="--", color=_COLOR_MSA, label="Avg. MSA")
    ax.axhline(avg_ida_bias, ls="--", color=_COLOR_IDA, label="Avg. IDA")
    ax.grid(ls="-.", color="0.8")
    ax.set_xlabel("Site No.")
    ax.set_ylabel("Bias")

    style_legend(ax)


def _plot_bias_se_ratios(ax, site_labels, msa_ratios, ida_ratios) -> None:
    """Scatter bias/se against the Efron & Tibshirani 0.25 materiality threshold."""
    ax.plot(site_labels, msa_ratios, marker=".", mfc=_COLOR_MSA, mec=_COLOR_MSA,
            ls="none", label="MSA")
    ax.plot(site_labels, ida_ratios, marker=".", mfc=_COLOR_IDA, mec=_COLOR_IDA,
            ls="none", label="IDA")
    ax.set_xlabel("Site No.")
    ax.set_ylabel("Bias / S.E. [-]")
    ax.axhline(0.25, ls="--", color="k")
    ax.set_ylim(0, 0.4)

    ax.axhspan(0.25, 0.4, alpha=0.2, color="r")
    ax.axhspan(0.0, 0.25, alpha=0.2, color="g")

    style_legend(ax)


def _plot_bias_mcse_ratios(ax, site_labels, msa_ratios, ida_ratios) -> None:
    """Scatter bias/mcse against 2, i.e. whether the bias is resolved from zero."""
    ax.plot(site_labels, msa_ratios, marker=".", mfc=_COLOR_MSA, mec=_COLOR_MSA,
            ls="none", label="MSA")
    ax.plot(site_labels, ida_ratios, marker=".", mfc=_COLOR_IDA, mec=_COLOR_IDA,
            ls="none", label="IDA")
    ax.set_xlabel("Site No.")
    ax.set_ylabel("Bias / MCSE [-]")
    ax.axhline(2, ls="--", color="k")

    style_legend(ax)


def plot_bias_assessment(
    bias_dfs: dict[str, pd.DataFrame],
    estimates_df: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray, plt.Figure, plt.Axes]:
    """Chart the bias of all four estimators, before and after correction.

    The 3x3 grid is arranged as columns ``ln theta`` / ``ln beta`` / corrected
    ``ln beta``, and rows raw bias / bias-to-se / bias-to-mcse. The second figure
    shows the net bias on the test statistic itself, ``b_msa - b_ida``.

    Also prints the pooled bias-to-se ratios of the test statistic, which are the
    numbers that decide whether a correction is needed for cross-site claims.
    """
    msa_t, msa_b, ida_t, ida_b = bias_dfs.values()

    fig1, axs1 = plt.subplots(3, 3, figsize=(12, 10))

    site_labels = [int(i.split("site")[1]) for i in msa_t.index]

    bbar_msa_t = msa_t["bias"].mean()
    bbar_ida_t = ida_t["bias"].mean()
    bbar_msa_b = msa_b["bias"].mean()
    bbar_ida_b = ida_b["bias"].mean()
    bbar_msa_b_corr = msa_b["bias_corrected"].mean()
    bbar_ida_b_corr = ida_b["bias_corrected"].mean()

    bbar_t = (msa_t["bias"] - ida_t["bias"]).mean()
    bbar_b = (msa_b["bias"] - ida_b["bias"]).mean()
    bbar_b_corr = (msa_b["bias_corrected"] - ida_b["bias_corrected"]).mean()

    delta_t = estimates_df["msa_t"] - estimates_df["ida_t"]
    delta_b = estimates_df["msa_b"] - estimates_df["ida_b"]

    b_delta_b = msa_b["bias"] - ida_b["bias"]
    b_delta_b_corr = msa_b["bias_corrected"] - ida_b["bias_corrected"]

    se_p_t = delta_t.std(ddof=1) / np.sqrt(len(delta_t))
    se_p_b = delta_b.std(ddof=1) / np.sqrt(len(delta_b))

    _plot_biases(axs1[0, 0], site_labels, msa_t["bias"], ida_t["bias"],
                 bbar_msa_t, bbar_ida_t)
    _plot_biases(axs1[0, 1], site_labels, msa_b["bias"], ida_b["bias"],
                 bbar_msa_b, bbar_ida_b)
    _plot_biases(axs1[0, 2], site_labels, msa_b["bias_corrected"],
                 ida_b["bias_corrected"], bbar_msa_b_corr, bbar_ida_b_corr)

    _plot_bias_se_ratios(axs1[1, 0], site_labels, msa_t["bias/se"], ida_t["bias/se"])
    _plot_bias_se_ratios(axs1[1, 1], site_labels, msa_b["bias/se"], ida_b["bias/se"])
    _plot_bias_se_ratios(axs1[1, 2], site_labels, msa_b["bias/se_corrected"],
                         ida_b["bias/se_corrected"])

    _plot_bias_mcse_ratios(axs1[2, 0], site_labels, msa_t["bias/mcse"],
                           ida_t["bias/mcse"])
    _plot_bias_mcse_ratios(axs1[2, 1], site_labels, msa_b["bias/mcse"],
                           ida_b["bias/mcse"])
    _plot_bias_mcse_ratios(axs1[2, 2], site_labels, msa_b["bias_corrected/mcse"],
                           ida_b["bias_corrected/mcse"])

    axs1[0, 1].set_ylim(-0.035, 0.035)
    axs1[0, 2].set_ylim(-0.035, 0.035)

    axs1[0, 0].set_title(r"Bias in $\ln{\theta}$")
    axs1[0, 1].set_title(r"Bias in $\ln{\beta}$")
    axs1[0, 2].set_title(r"Corrected Bias in $\ln{\beta}$")

    axs1[0, 0].set_ylabel(r"Bias, [$\ln{\theta}$ units]")
    axs1[0, 1].set_ylabel(r"Bias, [$\ln{\beta}$ units]")
    axs1[0, 2].set_ylabel(r"Bias, [$\ln{\beta}$ units]")

    print("BIAS / SE ratios for test statistic (msa-ida) averaged over the sites:")
    print("Theta", abs(bbar_t) / se_p_t)
    print("Beta", abs(bbar_b) / se_p_b)
    print("Beta_corrected", abs(bbar_b_corr) / se_p_b)

    fig1.tight_layout()

    fig2, axs2 = plt.subplots()
    axs2.axhline(0, ls="-", color="k", lw=0.75)
    axs2.plot(site_labels, b_delta_b, ls="none", marker=".", mfc="0.5", mec="k",
              label="Uncorrected Bias")
    axs2.plot(site_labels, b_delta_b_corr, ls="none", marker=".", mfc=_COLOR_DELTA,
              mec=_COLOR_DELTA, label="Corrected Bias")
    axs2.axhline(bbar_b, ls="--", color="0.5", label="Avg. Uncorr. Bias")
    axs2.axhline(bbar_b_corr, ls="--", color=_COLOR_DELTA, label="Avg. Corr. Bias")
    axs2.grid(ls="-.", color="0.8")
    axs2.set_xlabel("Site No.")
    axs2.set_ylabel(r"Bias on $ln\beta$")
    axs2.set_title(r"Total Bias $ln\beta$")

    style_legend(axs2)

    return fig1, axs1, fig2, axs2


def _scatter_bias_panel(ax, cov: pd.DataFrame, xkey: str, xlabel: str,
                        arm: str) -> None:
    """Scatter the bias against one stripe-placement metric, with a trend line."""
    x, y = cov[xkey].values, cov["bias"].values
    r = np.corrcoef(x, y)[0, 1]
    ax.axhline(0, ls="-", color="k", lw=0.75)
    ax.plot(x, y, marker="o", ms=4, mfc="b", mec="k", mew=0.4, ls="none")
    m, c = np.polyfit(x, y, 1)                       # least-squares trend
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, m * xs + c, ls="--", color="r", lw=1.2, label=rf"fit, $r={r:+.3f}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"Bias in $\ln{{\beta}}_{{{arm.upper()}}}$, [-]")
    ax.grid(ls="-.", color="0.8")
    style_legend(ax)


def _scatter_bias_decomposition_panel(ax, cov: pd.DataFrame, xkey: str,
                                      xlabel: str) -> None:
    """Scatter the raw-scale and Jensen components of the bias, with trend lines."""
    x = cov[xkey].values
    series = [("raw_scale", "r", r"Raw-scale bias, $\ln(E[\hat{\beta}]/\beta)$"),
              ("jensen", "g", r"Jensen gap, $E[\ln\hat{\beta}] - \ln E[\hat{\beta}]$")]
    ax.axhline(0, ls="-", color="k", lw=0.75)
    for key, col, lab in series:
        y = cov[key].values
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(x, y, marker="o", ms=4, mfc=col, mec="k", mew=0.4, ls="none",
                label=rf"{lab}, $r={r:+.3f}$")
        m, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, m * xs + c, ls="--", color=col, lw=1.2)
    # net bias for reference - shows how the two components offset each other
    ax.plot(x, cov["bias"].values, marker=".", ms=4, mfc="0.45", mec="0.45",
            ls="none", label="Net bias (sum)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Contribution to bias in $\ln{\beta}$, [-]")
    ax.grid(ls="-.", color="0.8")
    style_legend(ax, fontsize=8)


def _print_stripe_coverage_summary(cov: pd.DataFrame) -> None:
    """Print the correlations linking stripe placement to the MSA dispersion bias."""
    print(f"corr(bias, p_range)   = {np.corrcoef(cov.p_range, cov.bias)[0, 1]:+.3f}")
    print(f"corr(bias, se_pred)   = {np.corrcoef(cov.se_pred, cov.bias)[0, 1]:+.3f}")
    print(f"corr(bias, n_central) = {np.corrcoef(cov.n_central, cov.bias)[0, 1]:+.3f}")
    print(f"corr(se_pred, se_boot)= {np.corrcoef(cov.se_pred, cov.se_boot)[0, 1]:+.3f}"
          f"   (validates the placement -> precision link)")
    print(f"\nse_pred mean {cov.se_pred.mean():.4f} vs se_boot mean "
          f"{cov.se_boot.mean():.4f}"
          f"  ({100 * (cov.se_pred.mean() / cov.se_boot.mean() - 1):+.1f}%)")
    print(f"\ndecomposition (means): raw-scale {cov.raw_scale.mean():+.5f}"
          f"  +  Jensen {cov.jensen.mean():+.5f}"
          f"  =  {cov.raw_scale.mean() + cov.jensen.mean():+.5f}"
          f"   (actual bias {cov.bias.mean():+.5f})")
    print(f"corr(raw_scale, p_range) = "
          f"{np.corrcoef(cov.p_range, cov.raw_scale)[0, 1]:+.3f}"
          f"   corr(jensen, p_range) = "
          f"{np.corrcoef(cov.p_range, cov.jensen)[0, 1]:+.3f}")


def plot_bias_vs_stripe_coverage(
    bias_dfs: dict[str, pd.DataFrame],
    fcs_flat: dict,
    n_stripe_recs: int,
    arm: str = "msa",
    param: str = "beta",
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Relate the per-site dispersion bias to how informative the stripes are.

    Panels: bias against the collapse-probability span, bias against the
    Fisher-predicted standard error, and the raw-scale/Jensen decomposition.
    Returns the figure, its axes, and the per-site coverage frame.
    """
    bias_df = bias_dfs[f"{arm}_{param}"]
    cov = build_stripe_coverage_frame(bias_df, fcs_flat, n_stripe_recs)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    xlab_p = r"Collapse probability spanned by the stripes, $p_{max} - p_{min}$"
    _scatter_bias_panel(axs[0], cov, "p_range", xlab_p, arm)
    _scatter_bias_panel(
        axs[1], cov, "se_pred",
        r"Fisher-predicted $se(\ln{\hat{\beta}})$ from stripe placement, [-]", arm)
    _scatter_bias_decomposition_panel(axs[2], cov, "p_range", xlab_p)

    axs[0].set_title("Bias vs. stripe coverage")
    axs[1].set_title("Bias vs. dispersion identifiability")
    axs[2].set_title("Decomposition of the bias")
    fig.tight_layout()

    _print_stripe_coverage_summary(cov)
    return fig, axs, cov


# -----------------------------------------------------------------------------
# Cross-site comparison charts
# -----------------------------------------------------------------------------

# The chart set produced by plot_site_statistic_summary(). Each entry is
# (quantities, y label, reference line, log y-axis). A two-element quantity tuple is
# the paired MSA/IDA case and is drawn blue/red; a one-element tuple is a derived
# quantity that already contains both arms, so it is drawn in the difference colour.
# The two risk metrics span orders of magnitude between the low- and high-seismicity
# bands, so they need a log axis to stay readable on a shared scale.
_SITE_CHART_SPECS = {
    "theta": (("theta_msa", "theta_ida"),
              r"Median Collapse Capacity, $\theta$, [g]", None, False),
    "beta": (("beta_msa", "beta_ida"),
             r"Dispersion, $\beta$, [-]", None, False),
    "theta_ratio": (("theta_ratio",),
                    r"$\theta_{MSA}/\theta_{IDA}$, [-]", 1.0, False),
    "beta_ratio": (("beta_ratio",),
                   r"$\beta_{MSA}/\beta_{IDA}$, [-]", 1.0, False),
    "pc50": (("pcs_msa", "pcs_ida"),
             "P[collapse] in <t> years, [%]", None, True),
    "ec8_margin": (("mafc_margin_msa", "mafc_margin_ida"),
                   r"EC8 Margin, $\lambda_{C,EC8} / \lambda_{C,x}$, [-]", 1.0, True),
}


def _site_index_from_tag(tag: str) -> int:
    """Return the site number encoded in a structure tag."""
    return int(tag.split("site")[1])


def _shade_regions(ax, site_start: int) -> None:
    """Shade and label the six regional groupings of five sites.

    The spans are drawn in site-number coordinates, so a missing site leaves its gap
    inside the correct region rather than shifting the blocks along.
    """
    for i in range(_N_REGIONS):
        lo = site_start + i * _SITES_PER_REGION
        hi = lo + _SITES_PER_REGION
        if i % 2:
            ax.axvspan(lo - 0.5, hi - 0.5, color="0.85", alpha=0.5, zorder=0)
        # Labels sit just above the frame so they clear both the bars and the legend.
        ax.text((lo + hi - 1) / 2, 1.01, f"R{i}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="0.35", zorder=1)


def _asymmetric_ci(stats: pd.DataFrame, quantity: str, tags,
                   values: np.ndarray) -> np.ndarray:
    """Return the ``(2, n)`` yerr array spanning the bootstrap 5-95% interval.

    The bounds are clipped at zero so a plotted statistic that falls outside its own
    percentiles cannot produce a whisker pointing the wrong way.
    """
    lo = stats.loc[tags, (quantity, "5pc")].to_numpy()
    hi = stats.loc[tags, (quantity, "95pc")].to_numpy()
    return np.vstack([np.clip(values - lo, 0, None), np.clip(hi - values, 0, None)])


def _bar_panel(ax, stats: pd.DataFrame, site_start: int, site_stop: int,
               quantities: Sequence[str], labels: Sequence[str],
               colors: Sequence[str], stat: str) -> None:
    """Draw one seismicity band as grouped bars with 5-95% whiskers."""
    _shade_regions(ax, site_start)

    tags = [t for t in stats.index if site_start <= _site_index_from_tag(t) < site_stop]
    sites = np.array([_site_index_from_tag(t) for t in tags], dtype=float)

    width = 0.4 if len(quantities) > 1 else 0.7
    offsets = ([-width / 2, width / 2] if len(quantities) > 1 else [0.0])

    for quantity, label, color, offset in zip(quantities, labels, colors, offsets):
        values = stats.loc[tags, (quantity, stat)].to_numpy()
        ax.bar(sites + offset, values, width=width, label=label, color=color,
               edgecolor="k", lw=0.5, zorder=2,
               yerr=_asymmetric_ci(stats, quantity, tags, values), capsize=2,
               error_kw={"lw": 0.8, "ecolor": "k", "zorder": 3})

    ax.set_xticks(range(site_start, site_stop))
    ax.set_xlim(site_start - 0.6, site_stop - 0.4)
    ax.set_xlabel("Site No.")
    ax.grid(ls="-.", color="0.8", axis="y", zorder=0)
    ax.set_axisbelow(True)

    style_legend(ax, loc="upper left")


def plot_site_statistic(
    bootstrap_stats: pd.DataFrame,
    quantities: str | Sequence[str],
    ylabel: str,
    labels: Sequence[str] | None = None,
    stat: str = "median",
    hline: float | None = None,
    log_y: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 8),
    save_fp: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Chart one bootstrapped quantity across every site, split by seismicity.

    ``quantities`` is either a single level-0 key of ``bootstrap_stats`` or a pair of
    them, in which case the two are drawn side by side per site in the MSA and IDA
    colours. Bar heights are the ``stat`` column and the whiskers span the bootstrap
    5-95% interval. The low/moderate-seismicity band is plotted above the
    high-seismicity one on a shared y-axis, with the six regional groupings shaded and
    labelled ``R0``-``R5``.

    ``save_fp`` is the full path, including the file name, to write the figure to.
    Nothing is written when it is ``None``.
    """
    if isinstance(quantities, str):
        quantities = (quantities,)
    if len(quantities) > 1:
        colors = (_COLOR_MSA, _COLOR_IDA)
        labels = labels or ("MSA", "IDA")
    else:
        # A single series is a derived quantity that already contains both arms, so
        # MSA/IDA labels would be misleading; name it after the axis instead.
        colors = (_COLOR_DELTA,)
        labels = labels or (ylabel.split(",")[0],)

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharey=True)

    for ax, (site_start, site_stop, band) in zip(axs, _SEISMICITY_BANDS):
        _bar_panel(ax, bootstrap_stats, site_start, site_stop, quantities, labels,
                   colors, stat)
        ax.set_title(band, pad=18)   # headroom for the region labels
        ax.set_ylabel(ylabel)
        if hline is not None:
            ax.axhline(hline, ls="--", color="k", lw=1, zorder=3)
        if log_y:
            ax.set_yscale("log")

    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()

    if save_fp is not None:
        fig.savefig(save_fp)

    return fig, axs


def plot_site_statistic_summary(
    bootstrap_stats: pd.DataFrame,
    stat: str = "median",
    t: int = 50,
    out_folder: Path | None = None,
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    """Build the full set of cross-site comparison charts.

    Returns ``{name: (fig, axs)}`` for the medians, dispersions, their MSA/IDA ratios,
    the probability of collapse in ``t`` years and the EC8 margin. Given an
    ``out_folder``, each figure is also written there as ``site_comparison_{name}.jpg``.
    """
    figures = {}
    for name, (quantities, ylabel, hline, log_y) in _SITE_CHART_SPECS.items():
        save_fp = (None if out_folder is None
                   else out_folder / f"site_comparison_{name}.jpg")
        figures[name] = plot_site_statistic(
            bootstrap_stats, quantities, ylabel.replace("<t>", str(t)),
            stat=stat, hline=hline, log_y=log_y, save_fp=save_fp)
    return figures


# =============================================================================
# Pooling across sites and ad-hoc proxies
# =============================================================================

def build_pooled_ratio_frame(summary_stats: pd.DataFrame) -> pd.DataFrame:
    """Return the pooled per-structure fragility parameters and their MSA/IDA ratios.

    The ratios come from ``d_corr``, the bias-corrected log-parameter difference
    vector, so ``theta_ratio = theta_MSA / theta_IDA`` and likewise for ``beta``. The
    point estimates of both arms are also carried through in their own right --
    ``theta_msa``, ``theta_ida``, ``beta_msa``, ``beta_ida`` -- so a proxy can be
    regressed on the level of a parameter and not only on the disagreement between
    the two arms, and so each structure's own fragility quantiles can be recovered.
    Indexed by structure tag and carrying ``site_id`` so the pooled values keep
    their provenance and can be joined to any per-site quantity.
    """
    d_corr = np.vstack(summary_stats["d_corr"].to_numpy())
    msa_pt = np.vstack(summary_stats["msa_pt_estimate"].to_numpy())
    ida_pt = np.vstack(summary_stats["ida_pt_estimate"].to_numpy())
    return pd.DataFrame(
        {"site_id": [_site_index_from_tag(tag) for tag in summary_stats.index],
         "theta_ratio": np.exp(d_corr[:, 0]),
         "beta_ratio": np.exp(d_corr[:, 1]),
         "theta_msa": msa_pt[:, 0],
         "theta_ida": ida_pt[:, 0],
         "beta_msa": msa_pt[:, 1],
         "beta_ida": ida_pt[:, 1]},
        index=summary_stats.index)


def add_mafc_ratios(pooled: pd.DataFrame,
                    summary_stats: pd.DataFrame,
                    hcs: dict,
                    im_grid: np.ndarray | None = None) -> pd.DataFrame:
    """Add the mean annual frequency of collapse of both arms, and their ratio.

    The point estimates in ``summary_stats`` are convolved with each site's mean
    hazard curve on ``im_grid``. Returns a copy of ``pooled`` with ``mafc_msa``,
    ``mafc_ida`` and ``mafc_ratio`` appended.
    """
    imls = np.geomspace(0.001, 2.0) if im_grid is None else im_grid

    mafcs_msa, mafcs_ida = [], []
    for tag, row in summary_stats.iterrows():
        hc = hcs[_site_index_from_tag(tag)]["AvgSA"]["mean"]
        hc_interp_mafes = np.interp(imls, hc[:, 0], hc[:, 1])

        theta_msa, beta_msa = row["msa_pt_estimate"]
        theta_ida, beta_ida = row["ida_pt_estimate"]

        mafcs_msa.append(numerical_mafe(
            hc_interp_mafes, lognorm.cdf(imls, s=beta_msa, scale=theta_msa)))
        mafcs_ida.append(numerical_mafe(
            hc_interp_mafes, lognorm.cdf(imls, s=beta_ida, scale=theta_ida)))

    pooled = pooled.copy()
    pooled["mafc_msa"] = pd.Series(mafcs_msa, index=summary_stats.index)
    pooled["mafc_ida"] = pd.Series(mafcs_ida, index=summary_stats.index)
    pooled["mafc_ratio"] = pooled["mafc_msa"] / pooled["mafc_ida"]
    return pooled

def _require_columns(disagg_stats: pd.DataFrame, proxy_col: str) -> None:
    """Fail early and legibly when the disaggregation lacks a needed column."""
    for col in (proxy_col, _BAND_COL):
        if col not in disagg_stats.columns:
            raise KeyError(f"{col!r} is not a column of disagg_stats; "
                           f"available: {list(disagg_stats.columns)}")
        
def join_ratios_to_disagg(pooled: pd.DataFrame,
                          disagg_stats: pd.DataFrame,
                          proxy_col: str) -> pd.DataFrame:
    """Join the pooled per-structure ratios to one per-``(site, imtl)`` proxy.

    ``disagg_stats`` is the disaggregation summary indexed on ``[site_id, imtl]``.
    The result is long -- one row per structure and IM level -- with the structure
    tag kept in a ``tag`` column and the site's seismicity band in ``band_col``.
    Sites without a usable MSA fit are absent from ``pooled`` and so drop out, as do
    the IM levels a site's hazard curve never reaches.
    """
    _require_columns(disagg_stats, proxy_col)

    proxy = disagg_stats.reset_index()[["site_id", "imtl", _BAND_COL, proxy_col]]
    ratio_cols = [c for c in pooled.columns if c != "site_id"]

    joined = (pooled.reset_index(names="tag")
              .merge(proxy, on="site_id", how="inner", validate="m:m"))
    return joined[["tag", "site_id", "imtl", _BAND_COL, *ratio_cols,
                   proxy_col]].dropna(subset=["imtl", proxy_col])





def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares line through one group, with its correlation coefficients.

    Groups too small or too degenerate to fit -- fewer than three points, or no
    spread in either variable -- report NaN rather than raising, because a panel at
    a high IM level can easily lose one seismicity band entirely.
    """
    fit = {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
           "spearman_rho": np.nan, "n_sites": len(x)}
    if len(x) < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return fit

    m, c = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fit.update(slope=m, intercept=c, r2=r ** 2,
               spearman_rho=spearmanr(x, y).statistic)
    return fit


def _fit_ratio_panel(ax, block: pd.DataFrame, proxy_col: str,
                     ratio_col: str) -> dict[str, dict[str, float]]:
    """Scatter one panel by seismicity band and fit each band and the pooled set.

    The bands are drawn in their own colours and fitted separately, with the pooled
    fit over the top in black. Pooling the two bands is what makes this worth
    separating: the bands differ in *both* the proxy and the response, so a pooled
    slope can take a sign that neither band shows on its own.

    Returns ``{group: fit}`` for ``"all"`` and for each band present.
    """
    fits, y_text = {}, 0.96
    for group in _BAND_ORDER:
        sub = block if group == "all" else block[block[_BAND_COL] == group]
        if sub.empty:
            continue

        x, y = sub[proxy_col].to_numpy(), sub[ratio_col].to_numpy()
        color = _BAND_COLORS[group]

        if group != "all":                     # the pooled set is the two bands
            ax.plot(x, y, marker="o", ms=4, mfc=color, mec="k", mew=0.4, ls="none",
                    label=_BAND_LABELS[group], zorder=3)

        fit = _linear_fit(x, y)
        fits[group] = fit
        if np.isfinite(fit["slope"]):
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, fit["slope"] * xs + fit["intercept"],
                    ls="--" if group == "all" else "-", color=color,
                    lw=1.6 if group == "all" else 1.2, zorder=4)
            label = (f"{_BAND_LABELS[group]}: $R^2$={fit['r2']:.3f}, "
                     f"m={fit['slope']:+.2e}, n={fit['n_sites']}")
        else:
            label = f"{_BAND_LABELS[group]}: no fit, n={fit['n_sites']}"

        ax.text(0.03, y_text, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=7, color=color, zorder=5,
                bbox={"fc": "w", "ec": "none", "alpha": 0.7, "pad": 1.0})
        y_text -= 0.075

    ax.grid(ls="-.", color="0.8")
    return fits


def _fits_to_frame(fits: dict, index_name: str) -> pd.DataFrame:
    """Stack the per-panel ``{key: {group: fit}}`` dictionaries into one frame."""
    fit_df = pd.DataFrame.from_dict(
        {(key, group): fit for key, groups in fits.items()
         for group, fit in groups.items()},
        orient="index")
    fit_df.index = pd.MultiIndex.from_tuples(fit_df.index,
                                             names=[index_name, "group"])
    fit_df["n_sites"] = fit_df["n_sites"].astype(int)
    return fit_df.sort_index(
        level="group",
        key=lambda idx: idx.map(_BAND_ORDER.index), sort_remaining=False)


def plot_ratio_vs_proxy(
    pooled: pd.DataFrame,
    disagg_stats: pd.DataFrame,
    ratio_col: str = "beta_ratio",
    proxy_col: str = "Shallow Default [%]",
    ylabel: str | None = None,
    nrows: int = 5,
    ncols: int = 3,
    log_y: bool = False,
    figsize: tuple[float, float] = (13, 16),
    save_fp: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Regress one pooled MSA/IDA quantity on a disaggregation proxy, per IM level.

    One panel per IM level present in ``disagg_stats``, each scattering the quantity
    for every site against ``proxy_col`` at that level. The two seismicity bands are
    coloured separately and fitted separately, and the pooled fit is drawn over them
    in black; all three sets of statistics are reported on the axes. Panels share
    both axes so the scatter can be compared between levels, and spare axes in the
    grid are switched off.

    Returns the figure, its axes, and a frame of ``slope``, ``intercept``, ``r2``,
    ``spearman_rho`` and ``n_sites`` indexed by ``(imtl, group)``, where ``group`` is
    ``all``, ``lowmod`` or ``high``. The Spearman coefficient is carried alongside
    :math:`R^2` because proxies such as the shallow-crustal percentage pile up at
    100%, which gives a least-squares fit very uneven leverage.
    """
    joined = join_ratios_to_disagg(pooled, disagg_stats, proxy_col)
    imtls = np.sort(joined["imtl"].unique())
    if len(imtls) > nrows * ncols:
        raise ValueError(
            f"{len(imtls)} IM levels do not fit a {nrows}x{ncols} grid")

    ylabel = ylabel or ratio_col
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)

    fits = {}
    for ax, imtl in zip(axs.flat, imtls):
        fits[imtl] = _fit_ratio_panel(
            ax, joined[joined["imtl"] == imtl], proxy_col, ratio_col)
        ax.set_title(f"IM = {imtl:g} g", fontsize=10)
    for ax in axs.flat[len(imtls):]:
        ax.set_axis_off()

    if log_y:
        axs.flat[0].set_yscale("log")
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    # The last row of the grid is short wherever there are spare axes, so the
    # lowest *populated* panel of every column carries the x-label.
    for col in range(ncols):
        populated = list(range(col, len(imtls), ncols))
        if populated:
            axs.flat[populated[-1]].set_xlabel(proxy_col)
    style_legend(axs.flat[0], loc="lower right", fontsize=8)

    fig.suptitle(f"{ylabel} vs. {proxy_col}, by IM level and seismicity band")
    fig.tight_layout()

    if save_fp is not None:
        fig.savefig(save_fp, dpi=300, bbox_inches="tight")

    return fig, axs, _fits_to_frame(fits, "imtl")


def plot_fit_trend_vs_iml(
    fit_df: pd.DataFrame,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save_fp: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Chart how the fitted slope and its :math:`R^2` evolve with IM level.

    Takes the frame returned by :func:`plot_ratio_vs_proxy`, with one line per
    seismicity band and one for the pooled fit. The upper panel is the slope against
    a zero reference line, the lower one the explained variance.
    """
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axs[0].axhline(0.0, ls="-", color="k", lw=0.75)

    for group in _BAND_ORDER:
        if group not in fit_df.index.get_level_values("group"):
            continue
        sub = fit_df.xs(group, level="group").sort_index()
        color, label = _BAND_COLORS[group], _BAND_LABELS[group]
        style = {"marker": "o", "ms": 5, "mfc": color, "mec": "k", "mew": 0.4,
                 "color": color, "lw": 1.0,
                 "ls": "--" if group == "all" else "-"}
        axs[0].plot(sub.index, sub["slope"], label=label, **style)
        axs[1].plot(sub.index, sub["r2"], label=label, **style)

    axs[0].set_ylabel("Fitted slope, [-]")
    axs[1].set_ylabel(r"$R^2$, [-]")
    axs[1].set_xlabel("IM level, AvgSA(0-3s), [g]")
    axs[1].set_ylim(bottom=0.0)

    for ax in axs:
        ax.grid(ls="-.", color="0.8")
        style_legend(ax, fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_fp is not None:
        fig.savefig(save_fp, dpi=300, bbox_inches="tight")
    return fig, axs


def select_disagg_at_fragility_quantiles(
    pooled: pd.DataFrame,
    disagg_stats: pd.DataFrame,
    proxy_col: str,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Pick the disaggregation IM level nearest each site's own fragility quantiles.

    The disaggregation is tabulated on a fixed IM grid, but the intensity that
    matters to a structure is set by its *own* fragility: a site whose median
    collapse capacity is 0.30 g and one whose median is 0.90 g are not comparable at
    a common IM level. For each ``q`` the target intensity is the MSA fit's ``q``
    quantile, ``lognorm.ppf(q, s=beta_msa, scale=theta_msa)``, and the row taken is
    the one whose ``imtl`` lies closest to it among the levels that site has.

    Returns a long frame -- one row per structure and quantile -- carrying the
    pooled ratio columns, the site's seismicity band, the chosen ``imtl``, the
    ``iml_target`` it was matched to and the residual ``iml_gap``. The grid is coarse
    and bounded (0.27 to 1.15 g), so ``iml_gap`` is the honest record of how good
    each match actually is and should be read before the fits are trusted.
    """
    _require_columns(disagg_stats, proxy_col)

    proxy = disagg_stats.reset_index()[["site_id", "imtl", _BAND_COL, proxy_col]]
    ratio_cols = [c for c in pooled.columns if c != "site_id"]

    rows = []
    for tag, rec in pooled.iterrows():
        site_levels = proxy[proxy["site_id"] == rec["site_id"]]
        if site_levels.empty:
            continue
        imtls = site_levels["imtl"].to_numpy()

        for q in quantiles:
            target = lognorm.ppf(q, s=rec["beta_msa"], scale=rec["theta_msa"])
            pick = site_levels.iloc[int(np.argmin(np.abs(imtls - target)))]
            rows.append({"tag": tag, "site_id": rec["site_id"], "quantile": q,
                         "iml_target": target, "imtl": pick["imtl"],
                         "iml_gap": pick["imtl"] - target,
                         _BAND_COL: pick[_BAND_COL],
                         **{c: rec[c] for c in ratio_cols},
                         proxy_col: pick[proxy_col]})

    return pd.DataFrame(rows).dropna(subset=[proxy_col])


def plot_ratio_vs_proxy_at_quantiles(
    pooled: pd.DataFrame,
    disagg_stats: pd.DataFrame,
    ratio_col: str = "beta_ratio",
    proxy_col: str = "Shallow Default [%]",
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ylabel: str | None = None,
    log_y: bool = False,
    figsize: tuple[float, float] = (15, 5),
    save_fp: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Regress one pooled quantity on a proxy, at each site's own fragility quantiles.

    One panel per entry in ``quantiles``, drawn side by side on shared axes and split
    by seismicity band exactly as :func:`plot_ratio_vs_proxy` does. Unlike that
    function, every site contributes its own IM level, chosen by
    :func:`select_disagg_at_fragility_quantiles` to sit closest to that quantile of
    its MSA fit -- so a panel holds all 57 structures at a comparable point on their
    own fragility curve rather than at a common absolute intensity.

    Returns the figure, its axes, and a frame indexed by ``(quantile, group)``
    holding the same fit statistics as :func:`plot_ratio_vs_proxy`, plus
    ``median_imtl`` and ``max_abs_gap`` describing which levels were selected and how
    far the coarse disaggregation grid forced the worst match to stray from its
    target intensity.
    """
    picked = select_disagg_at_fragility_quantiles(
        pooled, disagg_stats, proxy_col, quantiles)

    ylabel = ylabel or ratio_col
    fig, axs = plt.subplots(1, len(quantiles), figsize=figsize,
                            sharex=True, sharey=True, squeeze=False)
    axs = axs[0]

    fits = {}
    for ax, q in zip(axs, quantiles):
        block = picked[picked["quantile"] == q]
        panel_fits = _fit_ratio_panel(ax, block, proxy_col, ratio_col)
        for group, fit in panel_fits.items():
            sub = block if group == "all" else block[block[_BAND_COL] == group]
            fit["median_imtl"] = sub["imtl"].median()
            fit["max_abs_gap"] = sub["iml_gap"].abs().max()
        fits[q] = panel_fits

        ax.set_title(f"{q:.0%} of the MSA fragility", fontsize=10)
        ax.set_xlabel(proxy_col)

    if log_y:
        axs[0].set_yscale("log")
    axs[0].set_ylabel(ylabel)
    style_legend(axs[0], loc="lower right", fontsize=8)

    fig.suptitle(f"{ylabel} vs. {proxy_col}, at each site's own fragility "
                 f"quantiles, by seismicity band")
    fig.tight_layout()

    if save_fp is not None:
        fig.savefig(save_fp, dpi=300, bbox_inches="tight")

    return fig, axs, _fits_to_frame(fits, "quantile")


def _site_bands(disagg_stats: pd.DataFrame) -> pd.Series:
    """Return the seismicity band of every site, as a ``site_id``-indexed series."""
    if _BAND_COL not in disagg_stats.columns:
        raise KeyError(f"{_BAND_COL!r} is not a column of disagg_stats")
    return disagg_stats.reset_index().groupby("site_id")[_BAND_COL].first()


def site_hazard_at_return_periods(
    hcs: dict,
    sites: Sequence[int],
    return_periods: Sequence[float] = (475, 2475, 4975),
    im_key: str = "AvgSA",
    stat: str = "mean",
) -> pd.DataFrame:
    """Interpolate the site hazard curves to the IM level of each return period.

    ``hcs[site][im_key][stat]`` is an ``(n_iml, 2)`` array of ``[iml, mafe]``. The
    return period is inverted to a mean annual frequency of exceedance and the
    intensity read off by linear interpolation **in log-log space**, which is the
    only defensible reading of a hazard curve whose ordinate spans several orders of
    magnitude between tabulated points.

    Rows of the curve at zero (or negative) ``mafe`` -- the numerical tail beyond
    which the calculation returned nothing -- are dropped before interpolating. A
    return period that falls outside the surviving range of a site's curve yields
    NaN for that site rather than being silently clamped to an end point.

    Returns a frame indexed by ``site_id`` with one column per return period.
    """
    out = {}
    for site in sites:
        hc = np.asarray(hcs[site][im_key][stat], dtype=float)
        hc = hc[hc[:, 1] > 0.0]
        if len(hc) < 2:
            out[site] = {rtp: np.nan for rtp in return_periods}
            continue

        # np.interp needs an ascending abscissa; mafe descends down the curve.
        ln_mafe = np.log(hc[::-1, 1])
        ln_iml = np.log(hc[::-1, 0])
        lo, hi = hc[:, 1].min(), hc[:, 1].max()

        out[site] = {
            rtp: (np.exp(np.interp(np.log(1.0 / rtp), ln_mafe, ln_iml))
                  if lo <= 1.0 / rtp <= hi else np.nan)
            for rtp in return_periods}

    return pd.DataFrame.from_dict(out, orient="index").rename_axis("site_id")


def build_hazard_proxy_frame(
    pooled: pd.DataFrame,
    hcs: dict,
    disagg_stats: pd.DataFrame,
    return_periods: Sequence[float] = (475, 2475, 4975),
    proxy_col: str = "AvgSA(0-3s) [g]",
    im_key: str = "AvgSA",
    stat: str = "mean",
) -> pd.DataFrame:
    """Join the pooled per-structure quantities to the hazard level at each return period.

    Unlike the disaggregation proxies, this one is read straight off the site hazard
    curve rather than tabulated on a fixed grid, so every site contributes a value at
    every return period. Returns a long frame -- one row per structure and return
    period -- carrying the pooled columns, the site's seismicity band and the
    interpolated intensity in ``proxy_col``.
    """
    hazard = site_hazard_at_return_periods(
        hcs, sorted(pooled["site_id"].unique()), return_periods, im_key, stat)
    bands = _site_bands(disagg_stats)
    ratio_cols = [c for c in pooled.columns if c != "site_id"]

    rows = [{"tag": tag, "site_id": rec["site_id"],
             _BAND_COL: bands.get(rec["site_id"]), "return_period": rtp,
             **{c: rec[c] for c in ratio_cols},
             proxy_col: hazard.loc[rec["site_id"], rtp]}
            for tag, rec in pooled.iterrows() for rtp in return_periods]

    return pd.DataFrame(rows).dropna(subset=[proxy_col, _BAND_COL])


def plot_ratio_vs_hazard_level(
    pooled: pd.DataFrame,
    hcs: dict,
    disagg_stats: pd.DataFrame,
    ratio_col: str = "beta_ratio",
    return_periods: Sequence[float] = (475, 2475, 4975),
    proxy_col: str = "AvgSA(0-3s) [g]",
    ylabel: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    allow_shared_denominator: bool = False,
    figsize: tuple[float, float] = (15, 5),
    save_fp: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Regress one pooled quantity on the site hazard level, per return period.

    One panel per return period, split by seismicity band exactly as
    :func:`plot_ratio_vs_proxy` is. The proxy here is the intensity the site's own
    hazard curve reaches at that return period, so it is a direct measure of how
    strong the shaking is rather than of where it comes from -- and it is close to
    what the design was driven by, which makes it the natural thing to test against
    the tectonic-composition proxies.

    Quantities in :data:`DESIGN_NORMALISED_COLUMNS` already carry the hazard level in
    their denominator and are refused, because the shared term produces a strong
    correlation on its own; ``allow_shared_denominator`` overrides that deliberately.

    Returns the figure, its axes, and a frame of the same fit statistics indexed by
    ``(return_period, group)``.
    """
    if ratio_col in DESIGN_NORMALISED_COLUMNS and not allow_shared_denominator:
        raise ValueError(
            f"{ratio_col!r} is normalised by the site hazard level, which is also "
            f"this proxy, so the regression would be measuring arithmetic rather "
            f"than structural behaviour. Pass allow_shared_denominator=True to plot "
            f"it anyway.")

    picked = build_hazard_proxy_frame(
        pooled, hcs, disagg_stats, return_periods, proxy_col)

    ylabel = ylabel or ratio_col
    fig, axs = plt.subplots(1, len(return_periods), figsize=figsize,
                            sharex=True, sharey=True, squeeze=False)
    axs = axs[0]

    fits = {}
    for ax, rtp in zip(axs, return_periods):
        fits[rtp] = _fit_ratio_panel(
            ax, picked[picked["return_period"] == rtp], proxy_col, ratio_col)
        ax.set_title(f"{rtp:g}-year return period", fontsize=10)
        ax.set_xlabel(proxy_col)

    if log_x:
        axs[0].set_xscale("log")
    if log_y:
        axs[0].set_yscale("log")
    axs[0].set_ylabel(ylabel)
    style_legend(axs[0], loc="lower right", fontsize=8)

    fig.suptitle(f"{ylabel} vs. {proxy_col} at the design return periods, "
                 f"by seismicity band")
    fig.tight_layout()

    if save_fp is not None:
        fig.savefig(save_fp, dpi=300, bbox_inches="tight")

    return fig, axs, _fits_to_frame(fits, "return_period")


def add_seismicity_band(pooled: pd.DataFrame,
                        disagg_stats: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``pooled`` with each structure's seismicity band attached."""
    out = pooled.copy()
    out[_BAND_COL] = out["site_id"].map(_site_bands(disagg_stats))
    return out


def plot_theta_beta_by_band(
    pooled: pd.DataFrame,
    disagg_stats: pd.DataFrame,
    arms: Sequence[str] = ("msa", "ida"),
    titles: Sequence[str] | None = None,
    sharey: bool = True,
    figsize: tuple[float, float] = (12, 5.5),
    save_fp: Path | str | None = None,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """Scatter dispersion against median across structures, one panel per arm.

    Unlike :func:`plot_theta_beta_correlation`, which draws the *bootstrap cloud* of
    a single structure, this is one point per structure: how the fitted ``beta``
    tracks the fitted ``theta`` across the case-study set. Both panels are split by
    seismicity band and carry a trend line and statistics for each band and for the
    pooled set, drawn exactly as the proxy regressions are.

    Duplicate points are expected and are not a plotting fault: several sites share a
    design, so they share an ``(theta, beta)`` pair exactly and overplot.

    Returns the figure, its axes, and a frame of ``slope``, ``intercept``, ``r2``,
    ``spearman_rho`` and ``n_sites`` indexed by ``(arm, group)``.
    """
    banded = add_seismicity_band(pooled, disagg_stats).dropna(subset=[_BAND_COL])
    titles = titles or [a.upper() if a == "msa" else "IDA (FEMA P695)" for a in arms]

    fig, axs = plt.subplots(1, len(arms), figsize=figsize,
                            sharex=True, sharey=sharey, squeeze=False)
    axs = axs[0]

    fits = {}
    for ax, arm, title in zip(axs, arms, titles):
        fits[arm] = _fit_ratio_panel(ax, banded, f"theta_{arm}", f"beta_{arm}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(rf"$\theta_{{{arm.upper()}}}$, [g]")
        ax.set_ylabel(rf"$\beta_{{{arm.upper()}}}$, [-]")

    style_legend(axs[0], loc="lower right", fontsize=8)
    fig.suptitle("Fitted dispersion against fitted median, by seismicity band")
    fig.tight_layout()

    if save_fp is not None:
        fig.savefig(save_fp, dpi=300, bbox_inches="tight")

    return fig, axs, _fits_to_frame(fits, "arm")


def add_design_normalised_medians(
    pooled: pd.DataFrame,
    hcs: dict,
    return_period: float = 475,
    im_key: str = "AvgSA",
    stat: str = "mean",
) -> pd.DataFrame:
    """Add each arm's median collapse capacity normalised by the design intensity.

    The design intensity is the site's own hazard at ``return_period`` -- 475 years
    by default, the EC8 no-collapse requirement -- read off the mean hazard curve by
    :func:`site_hazard_at_return_periods`. The ratio ``theta / Sa_475`` is the margin
    between what the structure actually collapses at and the intensity it was
    designed for, so it strips out the order-of-magnitude difference in hazard
    between the low/moderate and high seismicity bands that dominates ``theta``
    itself.

    Returns a copy of ``pooled`` with ``sa_design``, ``theta_msa_norm`` and
    ``theta_ida_norm`` appended.

    Do not regress this against the hazard level itself: ``Sa_475`` appears both in
    the denominator of the response and on the abscissa, which induces a strong
    negative correlation even when ``theta`` is independent of the hazard.
    :func:`plot_ratio_vs_hazard_level` refuses these columns for that reason.
    """
    hazard = site_hazard_at_return_periods(
        hcs, sorted(pooled["site_id"].unique()), (return_period,), im_key, stat)

    out = pooled.copy()
    out["sa_design"] = out["site_id"].map(hazard[return_period])
    out["theta_msa_norm"] = out["theta_msa"] / out["sa_design"]
    out["theta_ida_norm"] = out["theta_ida"] / out["sa_design"]
    return out


# =============================================================================
# Orchestration
# =============================================================================

def process_structure(out_folder, site, nstoreys, fc_msa, n_recs_msa, fc_ida,
                      n_recs_ida, k_samples, bias_correction, im, haz_curve,
                      alpha=0.05, ec8_target=2e-4, n_bins=20, t=50, seed=1):
    """Bootstrap, test and chart one structure, saving both figures to disk.

    ``bias_correction`` is the ``[d_lntheta, d_lnbeta]`` shift subtracted from the
    observed difference vector before the test. Returns the per-quantity statistics
    frame and a dict of the scalar test outcomes.
    """
    # comparison between MSA and IDA
    # fragility curves + boot straps
    sim_fcs_msa, ok_msa = bootstrap_msa(fc_msa, n_recs_msa, k_samples, im,
                                        seed=_SEED_OFFSET_MSA + seed)
    sim_fcs_ida = bootstrap_ida(fc_ida, n_recs_ida, k_samples, im,
                                seed=_SEED_OFFSET_IDA + seed)

    # discard degenerate MSA fits, and the paired IDA replicates with them, so that
    # dis = v_msa - v_ida stays well defined and the cloud is filtered on the same
    # criterion as the bias estimate in msa_estimator_bias()
    if not ok_msa.all():
        print(f"site {site}: dropped {int((~ok_msa).sum())}/{k_samples} "
              "degenerate MSA fits")
    sim_fcs_msa = [fc for fc, keep in zip(sim_fcs_msa, ok_msa) if keep]
    sim_fcs_ida = [fc for fc, keep in zip(sim_fcs_ida, ok_msa) if keep]

    ### MSA
    thetas_msa = [fc.median for fc in sim_fcs_msa]
    betas_msa = [fc.dispersion for fc in sim_fcs_msa]
    v_msa = np.column_stack([np.log(thetas_msa), np.log(betas_msa)])

    ### IDA
    thetas_ida = [fc.median for fc in sim_fcs_ida]
    betas_ida = [fc.dispersion for fc in sim_fcs_ida]
    v_ida = np.column_stack([np.log(thetas_ida), np.log(betas_ida)])

    ### Statistical significance
    dis = v_msa - v_ida     # vector of differences
    d_orig = (np.log(np.array([fc_msa["median"], fc_msa["dispersion"]]))
              - np.log(np.array([fc_ida["median"], fc_ida["dispersion"]])))

    d_corr = d_orig - bias_correction

    D2Mis, d_bar, Gamma, test_distance, CI_distance, asl = hypothesis_test(
        dis, d_corr, alpha)

    ### Plotting
    fig1, axs1 = plt.subplots(3, 2, figsize=(10, 12))
    plt.close(fig1)

    imls = plot_fragility_curves(axs1[0, 0], fc_msa, fc_ida, sim_fcs_msa, sim_fcs_ida)
    plot_theta_beta_correlation(axs1[0, 1], thetas_msa, betas_msa, thetas_ida,
                                betas_ida)
    plot_theta_beta_hists(axs1[1, 0], axs1[1, 1], thetas_msa, betas_msa, thetas_ida,
                          betas_ida)
    visualise_hypothesis_test(axs1[2, 0], D2Mis, alpha, test_distance, CI_distance)
    plot_msa_ida_diff(axs1[2, 1], dis, d_bar, d_orig, d_corr, CI_distance, Gamma)

    fig1.suptitle(f"Site {site} - {nstoreys} stories - MSA vs. IDA (Chart 1/2)",
                  fontweight="bold")
    fig1.tight_layout()

    ### Other stats
    theta_ratio = np.array(thetas_msa) / np.array(thetas_ida)
    beta_ratio = np.array(betas_msa) / np.array(betas_ida)

    combined_imls = sorted(np.concat([haz_curve[:, 0].flatten(), imls]))
    hc_interp = np.interp(combined_imls, haz_curve[:, 0], haz_curve[:, 1])
    mafcs_msa = calculate_mafcs(sim_fcs_msa, hc_interp, combined_imls)
    mafcs_ida = calculate_mafcs(sim_fcs_ida, hc_interp, combined_imls)
    pcs_msa = mafe_to_poe(mafcs_msa, t) * 100        # in %
    pcs_ida = mafe_to_poe(mafcs_ida, t) * 100        # in %
    mafc_margin_msa, mafc_margin_ida, mafc_rel_msa = calculate_mafc_margins(
        mafcs_msa, mafcs_ida, ec8_target)

    fig2, axs2 = plt.subplots(3, 2, figsize=(10, 12))
    plt.close(fig2)
    plot_msa_ida_ratios(axs2[0, 0], axs2[0, 1], theta_ratio, beta_ratio, n_bins)
    plot_mafc_distributions(axs2[1, 0], mafcs_msa, mafcs_ida, n_bins)
    plot_pc_distributions(axs2[1, 1], pcs_msa, pcs_ida, t, n_bins)
    plot_mafc_difference_distributions(axs2[2, 0], axs2[2, 1], mafc_rel_msa,
                                       mafc_margin_msa, mafc_margin_ida, n_bins)

    fig2.suptitle(f"Site {site} - {nstoreys} stories - MSA vs. IDA (Chart 2/2)",
                  fontweight="bold")
    fig2.tight_layout()

    ## create stats dicts
    stats_dicts = {
        "theta_msa": get_stats(thetas_msa),
        "beta_msa": get_stats(betas_msa),
        "theta_ida": get_stats(thetas_ida),
        "beta_ida": get_stats(betas_ida),
        "D2Mis": get_stats(D2Mis),
        "theta_ratio": get_stats(theta_ratio),
        "beta_ratio": get_stats(beta_ratio),
        "mafc_msa": get_stats(mafcs_msa),
        "mafc_ida": get_stats(mafcs_ida),
        "pcs_msa": get_stats(pcs_msa),
        "pcs_ida": get_stats(pcs_ida),
        "mafc_margin_msa": get_stats(mafc_margin_msa),
        "mafc_margin_ida": get_stats(mafc_margin_ida),
        "mafc_rel_msa": get_stats(mafc_rel_msa),
    }

    bootstrap_stats = pd.DataFrame.from_dict(stats_dicts, orient="columns")

    ## dict of remaining stats
    summary_stats = {
        "statistically_different": test_distance >= CI_distance,
        "rho_msa": np.corrcoef(thetas_msa, betas_msa)[0, 1],
        "rho_ida": np.corrcoef(thetas_ida, betas_ida)[0, 1],
        "msa_pt_estimate": [fc_msa["median"], fc_msa["dispersion"]],
        "ida_pt_estimate": [fc_ida["median"], fc_ida["dispersion"]],
        "d_orig": d_orig,
        "d_corr": d_corr,
        "d_bar": d_bar,
        "covar": Gamma,
        "test_distance": test_distance,
        "CI_distance": CI_distance,
        "achieved_sig_level": asl,
    }

    ## save figures
    tag = structure_tag(site, nstoreys)
    fig1.savefig(out_folder / f"{tag}_chart1_hypothesis_test.jpg")
    fig2.savefig(out_folder / f"{tag}_chart2_risk_metrics.jpg")

    return bootstrap_stats, summary_stats


def run_hypothesis_tests(
    out_folder: Path,
    fragility_curves: dict,
    hazard_curves: dict,
    n_storeys: Sequence[int],
    bias_corrections: dict[str, list],
    im: IntensityMeasure,
    n_stripe_recs: int,
    n_ida_recs: int,
    k_samples: int,
    alpha: float = 0.05,
    ec8_target: float = 2e-4,
    sites: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`process_structure` over every structure and collate the results.

    ``sites`` restricts the run to a subset; the default is every site present in
    ``fragility_curves``. Each structure is seeded from its site index, so results
    do not depend on which subset is run.

    Returns ``(bootstrap_stats, summary_stats)`` as DataFrames.
    """
    if sites is None:
        sites = list(fragility_curves.keys())

    bootstrap_stats = {}
    summary_stats = {}

    for site in sites:
        hc = hazard_curves[site]["AvgSA"]["mean"]
        for ns in n_storeys:
            tag = structure_tag(site, ns)
            # get the fragility curve estimates
            try:
                fc_msa = fragility_curves[site][tag]["msa"]
                fc_ida = fragility_curves[site][tag]["ida-femap695"]
            except KeyError:
                continue

            bootstrap_stats[tag], summary_stats[tag] = process_structure(
                out_folder, site, ns, fc_msa, n_stripe_recs, fc_ida, n_ida_recs,
                k_samples, bias_corrections[tag], im, hc,
                alpha=alpha, ec8_target=ec8_target, seed=site)

    return (pd.concat(bootstrap_stats).unstack(),
            pd.DataFrame.from_dict(summary_stats, orient="index"))
