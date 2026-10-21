"""SaRatio moments from the GCIM distribution at a site.

WHAT THIS COMPUTES
------------------
SaRatio (Eads et al. 2016; Zhong et al. 2022 Eq. 2) is the first-mode spectral
ordinate divided by the geometric mean of the ordinates over a band around it:

    SaRatio(T1) = Sa(T1) / [ PROD_i Sa(Ti) ] ** (1/n),   Ti in [0.2 T1, 3 T1]

which in log space is a numerator minus the mean of a denominator:

    ln SaRatio(T1) = ln Sa(T1) - (1/n) SUM_i ln Sa(Ti)

:func:`expected_ln_saratio` returns its mean and standard deviation given that
the conditioning intensity measure AvgSA[0,3] takes a particular value at a
site -- the GCIM distribution of Bradley (2010). That distribution is a MIXTURE
over the disaggregated rupture scenarios: each scenario contributes a joint
lognormal for the spectrum, and the scenarios are combined with their
disaggregation probabilities. The calculation therefore has two layers, kept
visibly separate here:

    1. within one scenario  -> ln SaRatio is a linear combination of jointly
                               normal variables, so its mean and variance are
                               closed form (:func:`scenario_ln_saratio`);
    2. across scenarios     -> the law of total variance
                               (:func:`combine_scenarios`).

HOW IT IS WRITTEN
-----------------
The structure follows ``admin/rough_algorithm_for_SARatio.txt``: one visible
loop per level of the algorithm, the formulas spelled the way the literature
spells them, and every intermediate quantity in a local variable a debugger can
see. Scenarios are evaluated one at a time.

The one concession to speed is that the ground-motion models are called once per
Tectonic Region Type rather than once per rupture -- see
:func:`conditional_spectra`. Without it a single (site, structure, level) takes
~15 s instead of ~0.15 s, because the OpenQuake GMMs loop over the periods in
Python and re-read the rupture context inside that loop.

ENTRY POINTS
------------
:func:`expected_ln_saratio`
    The two numbers, plus diagnostics, for one (site, structure, level).
:func:`conditional_spectra`
    Every scenario's conditional log-spectrum, one column per rupture. This is
    what :func:`expected_ln_saratio` reduces, and what a plot of the method
    needs.
:func:`mixture_spectrum`
    The conditional spectrum itself: the mixture mean and sigma of ln Sa at each
    period, across the scenarios.

The caller supplies the inputs: the rupture disaggregation and its conditioning
level, the site model, and the selection context built by
``setup_AvgSA03_gm_selection.setup_AvgSA03_gcim_gm_selection``. Nothing here
reads a file except the correlation coefficients, which arrive through
``create_corr_model_map``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from openquake.hazardlib.imt import SA, AvgSA
from pickagm.corrmodels import conditional_correlation_matrix

from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    _initialise_ctx_builder,
    create_corr_model_map,
)


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

# SaRatio band, Zhong et al. (2022) Eq. 2 after Eads et al. (2016): the
# denominator runs from 0.2 T1 to 3 T1, sampled every 0.01 s.
SARATIO_TA, SARATIO_TB, SARATIO_DT = 0.2, 3.0, 0.01

# The conditioning intensity measure of the whole WP1 ground-motion selection.
CONDITIONING_IMT = AvgSA([0, 3])
CONDITIONING_LABEL = CONDITIONING_IMT.string  # "AvgSA([0, 3])"


# ---------------------------------------------------------------------------
# 2. The SaRatio band
# ---------------------------------------------------------------------------

def saratio_periods(T1: float) -> np.ndarray:
    """The denominator periods Ti: 0.2 T1 to 3 T1 every 0.01 s, endpoint included.

    Rounded to 3 decimal places because the correlation model labels its rows
    ``f"SA({round(T, 3)})"`` and every lookup below is by that label; an
    unrounded period would simply not be found.
    """
    stop = SARATIO_TB * T1 + SARATIO_DT / 2   # + dt/2 so the endpoint is included
    return np.round(np.arange(SARATIO_TA * T1, stop, SARATIO_DT), 3)


def sa_label(period: float) -> str:
    """The string a period is known by in the correlation tables and the GMMs."""
    return SA(float(period)).string


# ---------------------------------------------------------------------------
# 3. Correlations
# ---------------------------------------------------------------------------

_CORRELATION_CACHE: dict[tuple, tuple[dict, dict]] = {}


def correlation_tables(periods: np.ndarray) -> tuple[dict, dict]:
    """Per-tectonic-region correlations of ln Sa at ``periods``, two flavours.

    Returns ``(corr_unconditional, corr_given_iml)``.

    ``corr_unconditional[trt]`` is the ordinary correlation matrix of the total
    residuals, and it carries the conditioning IM as an extra row/column -- that
    row is where the rho used to condition each ordinate on AvgSA comes from.

    ``corr_given_iml[trt]`` is what is left of the correlation between two
    ordinates once AvgSA is known (Jayaram & Baker 2008):

        rho_ij|c = (rho_ij - rho_ic rho_jc) / sqrt((1 - rho_ic^2)(1 - rho_jc^2))

    Both are DataFrames labelled by ``sa_label``. The result is cached on the
    period tuple: these tables are ~280 x 280, they cost an HDF5 read plus a 2-D
    interpolation, and they depend only on T1 -- so there are 51 distinct ones
    across all 480 (site, structure, return period) combinations, not 480.
    """
    key = tuple(periods)
    if key in _CORRELATION_CACHE:
        return _CORRELATION_CACHE[key]

    # The project's TRT -> correlation-model dispatch (Clemett & Gundel 2026),
    # rebuilt at exactly the periods we need rather than interpolated afterwards.
    corr_unconditional = create_corr_model_map([CONDITIONING_LABEL], periods)

    labels = [sa_label(T) for T in periods]
    missing = [lab for lab in labels
               if lab not in corr_unconditional["Shallow Default"].index]
    assert not missing, (
        f"the correlation model does not label these periods as expected: "
        f"{missing[:3]} -- check the 3-decimal rounding in saratio_periods")

    corr_given_iml = {
        trt: conditional_correlation_matrix(df, CONDITIONING_LABEL).loc[labels, labels]
        for trt, df in corr_unconditional.items()}

    _CORRELATION_CACHE[key] = (corr_unconditional, corr_given_iml)
    return corr_unconditional, corr_given_iml


# ---------------------------------------------------------------------------
# 4. One rupture scenario
# ---------------------------------------------------------------------------

def scenarios_for(disagg_df: pd.DataFrame, max_scenarios: int | None = None
                  ) -> list[dict]:
    """The (TRT, Mag, Dist) rupture scenarios and their probabilities.

    Three things happen, all of which the fast code does too but inside a
    one-liner:

    1. The epsilon axis is summed away. Epsilon does not enter the conditional
       moments below -- the conditioning epsilon is recomputed from the
       conditioning level and the GMM -- so the disaggregation on epsilon carries
       no information here.
    2. Zero-probability rows are dropped. The disaggregation grid is a full
       (TRT x Mag x Dist) product and most cells are empty; keeping them would
       multiply the runtime by ~60 for no contribution.
    3. The weights are renormalised to sum to 1, so the mixture formulas in
       :func:`combine_scenarios` are the textbook ones.

    Returns a list of plain dicts so the caller's loop reads
    ``for scenario in scenarios:`` and a debugger shows one rupture at a time.
    """
    collapsed = (disagg_df
                 .groupby(["TRT", "Mag", "Dist"], as_index=False)["P(m|X=x)"]
                 .sum())
    collapsed = collapsed[collapsed["P(m|X=x)"] > 0.0].reset_index(drop=True)

    total_probability = float(collapsed["P(m|X=x)"].sum())
    scenarios = []
    for trt, mag, dist, probability in collapsed.itertuples(index=False):
        scenarios.append({"trt": trt,
                          "mag": float(mag),
                          "rjb": float(dist),
                          "eps": 0.0,      # collapsed away, see (1) above
                          "weight": float(probability) / total_probability})

    if max_scenarios is not None:
        # A truncated set no longer sums to 1, so renormalise again -- otherwise
        # the "between scenarios" variance term would be nonsense rather than
        # merely approximate.
        scenarios = scenarios[:max_scenarios]
        kept = sum(s["weight"] for s in scenarios)
        for s in scenarios:
            s["weight"] /= kept
    return scenarios


def group_scenarios_by_trt(scenarios: list[dict]) -> dict[str, list[dict]]:
    """The scenarios sorted by Tectonic Region Type, order preserved.

    The TRT picks both the ground-motion model and the correlation model, so it
    is the only axis along which scenarios can share a single GMM call. It is not
    the site's "region": scenarios of one TRT may come from several source
    regions, and that distinction does not enter this calculation.
    """
    by_trt: dict[str, list[dict]] = {}
    for scenario in scenarios:
        by_trt.setdefault(scenario["trt"], []).append(scenario)
    return by_trt


def stack_contexts(ctx_builder, scenarios: list[dict]) -> np.recarray:
    """One recarray holding every rupture of one TRT, one row each.

    Handing OpenQuake all the ruptures of a TRT in a single call is what makes
    this calculation fast. The GMMs loop over the periods in Python and read the
    context fields inside that loop, so the per-period overhead is paid once per
    call: with one rupture per call it is paid once for every rupture instead.
    """
    return np.concatenate(
        [ctx_builder.ctx({"trt": s["trt"], "mag": s["mag"],
                          "rjb": s["rjb"], "eps": s["eps"]})
         for s in scenarios]).view(np.recarray)


def gmm_mean_and_sigma(gmm, ctx: np.recarray, imts: list) -> tuple[np.ndarray, np.ndarray]:
    """Median ln IM and its total standard deviation, for a set of ruptures.

    OpenQuake GMMs write into preallocated arrays rather than returning
    anything, which is why this wrapper exists. Both outputs are shaped
    ``(len(imts), len(ctx))``: **one row per period, one column per rupture**.
    Every caller reads a single rupture out as ``mu[:, k]``.

    One call covers the whole period vector and every rupture in ``ctx``. The
    GMM is vectorised over both, so splitting either axis across calls buys
    nothing and costs a great deal.
    """
    mean = np.zeros((len(imts), len(ctx)))
    sigma = np.zeros_like(mean)
    tau = np.zeros_like(mean)
    phi = np.zeros_like(mean)
    gmm.compute(ctx, imts, mean, sigma, tau, phi)

    assert mean.shape == (len(imts), len(ctx)), (
        f"expected one value per (period, rupture), got {mean.shape} for "
        f"{len(imts)} periods and {len(ctx)} ruptures")
    return mean, sigma


def conditioning_epsilons(gmm_conditioning, ctx: np.recarray,
                          iml: float) -> np.ndarray:
    """How many sigma the conditioning level sits above each rupture's median.

        epsilon = (ln(iml) - mu_lnAvgSA) / sigma_lnAvgSA

    A rare, large level at a site dominated by small nearby events gives a large
    positive epsilon: that scenario has to be an unusually strong realisation to
    produce the level at all, and every other ordinate is pulled up with it.

    Returns one epsilon per rupture, shape ``(len(ctx),)``.
    """
    mu, sigma = gmm_mean_and_sigma(gmm_conditioning, ctx, [CONDITIONING_IMT])

    # A single conditioning IM, so the period axis has length 1 and is dropped.
    return (np.log(iml) - mu[0, :]) / sigma[0, :]


def conditional_mean(mu: np.ndarray, sigma: np.ndarray,
                     rho_sa_avgsa: np.ndarray, eps: float) -> np.ndarray:
    """E[ln Sa(Ti) | AvgSA = iml] (Bradley 2010).

        mu_i|c = mu_i + sigma_i * rho_ic * eps

    Shapes: ``mu`` and ``sigma`` are (n_periods, n_ruptures); ``rho_sa_avgsa``
    varies with period only and ``eps`` with rupture only, so the caller passes
    them as a column and a row and the product reads as "this period, this
    rupture".
    """
    return mu + sigma * rho_sa_avgsa * eps


def conditional_sigma(sigma: np.ndarray, rho_sa_avgsa: np.ndarray) -> np.ndarray:
    """SD[ln Sa(Ti) | AvgSA = iml].

        s_i = sigma_i * sqrt(1 - rho_ic^2)

    Knowing AvgSA can only remove uncertainty, never add it, so s_i <= sigma_i
    always -- a useful thing to assert on when stepping through.

    Shapes: ``sigma`` is (n_periods, n_ruptures) and ``rho_sa_avgsa`` is passed
    as a column, since it varies with period only. The conditioning level does
    not enter, so unlike the mean this is the same for every rupture of a TRT.
    """
    return sigma * np.sqrt(1.0 - rho_sa_avgsa ** 2)


def scenario_ln_saratio(mu_cond: np.ndarray, sigma_cond: np.ndarray,
                        corr_given_iml: np.ndarray,
                        i_t1: int, i_band: np.ndarray) -> tuple[float, float]:
    """Mean and variance of ln SaRatio WITHIN one rupture scenario.

    Within a scenario the log ordinates, given the conditioning IM, are jointly
    normal, and ln SaRatio is a linear combination of them:

        ln SaRatio = ln Sa(T1) - (1/n) SUM_i ln Sa(Ti)

    so it is itself normal, with

        mean = mu(T1) - (1/n) SUM_i mu(Ti)

        var  = s1^2                                        (the numerator)
             + (1/n^2) SUM_i SUM_j rho_ij|c s_i s_j        (the denominator)
             - (2/n)   SUM_i rho_1i|c s1 s_i               (their covariance)

    Note the sign of the third term: because the numerator period sits inside the
    band, SaRatio is much less variable than either Sa(T1) or the band mean on
    its own. Positive correlation between them REDUCES the variance here, which
    is the opposite of what it does for AvgSA itself.

    ``i_t1`` and ``i_band`` are positions into the unique-period axis that
    ``mu_cond``, ``sigma_cond`` and ``corr_given_iml`` share. ``i_band`` may
    contain ``i_t1`` -- when a band period lands exactly on T1 nothing special is
    needed, its correlation with T1 is just 1.0.
    """
    n = len(i_band)

    mu_t1 = mu_cond[i_t1]
    s_t1 = sigma_cond[i_t1]
    mu_band = mu_cond[i_band]
    s_band = sigma_cond[i_band]

    # mean = mu(T1) - mean of the band means. The geometric mean of the
    # denominator is the arithmetic mean of its logs, which is all this is.
    mean = float(mu_t1 - mu_band.mean())

    # Term 1: the variance the numerator contributes on its own.
    variance = s_t1 ** 2

    # Term 2: the variance of the band mean,
    #     (1/n^2) SUM_i SUM_j rho_ij|c s_i s_j
    # which written out as a loop would be
    #     for i in range(n):
    #         for j in range(n):
    #             variance += corr_band[i, j] * s_band[i] * s_band[j] / n**2
    # -- ~280 x 280 per scenario in pure Python, so it is the one sum kept in
    # matrix form. The quadratic form below is that double loop exactly.
    corr_band = corr_given_iml[np.ix_(i_band, i_band)]
    variance += float(s_band @ corr_band @ s_band) / n ** 2

    # Term 3: minus twice the covariance between numerator and band mean,
    #     (2/n) SUM_i rho_1i|c s1 s_i
    corr_t1_band = corr_given_iml[i_t1, i_band]
    variance -= 2.0 / n * s_t1 * float(corr_t1_band @ s_band)

    # The correlation table is bilinearly interpolated off a coarser grid, so it
    # is not guaranteed positive semi-definite and the quadratic form can come
    # out a hair below zero when the true variance is tiny. Clamp rather than
    # propagate a negative variance into a sqrt.
    return mean, max(variance, 0.0)


# ---------------------------------------------------------------------------
# 5. Combining scenarios
# ---------------------------------------------------------------------------

def combine_scenarios(means: list[float], variances: list[float],
                      weights: list[float]) -> tuple[float, float, float, float]:
    """Mixture moments over the disaggregated scenarios.

    The law of total variance (Ang & Tang 2007, *Probability Concepts in
    Engineering*, 2nd ed., Ch. 3, ~p. 130 -- page approximate):

        E[X]   = SUM_k w_k m_k
        Var[X] = SUM_k w_k v_k               "within":  spread inside a scenario
               + SUM_k w_k m_k^2 - E[X]^2    "between": spread of scenario means

    Returns ``(mean, sigma, var_within, var_between)``. The two contributions are
    returned separately because which one dominates is the interesting diagnostic
    -- at a site with one clear controlling source the "between" term should be
    small, and at a site with competing tectonic regions it should not be.
    """
    mean = 0.0
    for m, w in zip(means, weights):
        mean += w * m

    var_within = 0.0
    second_moment_of_means = 0.0
    for m, v, w in zip(means, variances, weights):
        var_within += w * v
        second_moment_of_means += w * m ** 2

    var_between = second_moment_of_means - mean ** 2

    # Same round-off caveat as above; "between" is a difference of two nearly
    # equal numbers when all the scenarios agree.
    var_between = max(float(var_between), 0.0)

    return (float(mean), float(np.sqrt(var_within + var_between)),
            float(var_within), var_between)


# ---------------------------------------------------------------------------
# 6. The conditional spectra
# ---------------------------------------------------------------------------

class ConditionalSpectra(NamedTuple):
    """Every rupture scenario's conditional log-spectrum, one column per rupture.

    periods     (n_T,)      the period axis, shared by every column
    mu_cond     (n_T, n_k)  E[ln Sa(Ti) | AvgSA = iml] for rupture k
    sigma_cond  (n_T, n_k)  SD[ln Sa(Ti) | AvgSA = iml] for rupture k
    weights     (n_k,)      P(rupture k | AvgSA = iml), summing to 1
    trts        (n_k,)      the Tectonic Region Type of each column
    scenarios   list[dict]  the ruptures themselves, in column order

    Columns come out grouped by TRT, because that is the order the ground-motion
    models are evaluated in. ``scenarios``, ``weights`` and ``trts`` are all in
    that same order, so column k, ``weights[k]``, ``trts[k]`` and
    ``scenarios[k]`` always describe the same rupture.
    """

    periods: np.ndarray
    mu_cond: np.ndarray
    sigma_cond: np.ndarray
    weights: np.ndarray
    trts: np.ndarray
    scenarios: list


def conditional_spectra(site: int, periods: np.ndarray, disagg_df: pd.DataFrame,
                        iml: float, site_model: pd.DataFrame,
                        selection_ctx: dict,
                        max_scenarios: int | None = None) -> ConditionalSpectra:
    """The conditional log-spectrum of every disaggregated rupture at a site.

    Steps 3.1 to 3.5 of ``admin/rough_algorithm_for_SARatio.txt``: for each
    rupture scenario, how far above its median the conditioning level sits, and
    what that implies for the mean and spread of every ordinate.

    The scenarios are taken a Tectonic Region Type at a time. The TRT selects
    both the ground-motion model and the correlation model, and nothing in the
    conditional moments couples one rupture to another, so every rupture of a TRT
    goes to the GMM in a single call. That grouping is the only reason this
    function exists rather than a loop over scenarios in the caller: it is worth
    a factor of ~70 in runtime, and it changes no number.
    """
    # The site id IS the row index of the site model (lat, lon, vs30, z1pt0...).
    site_ctx = dict(selection_ctx)
    site_ctx["site_params"] = site_model.loc[site, :].to_dict()
    ctx_builder = _initialise_ctx_builder(site_ctx)

    corr_unconditional, _ = correlation_tables(periods)
    labels = [sa_label(T) for T in periods]
    sa_imts = [SA(float(T)) for T in periods]

    scenarios = scenarios_for(disagg_df, max_scenarios)

    mu_blocks, sigma_blocks, ordered_scenarios = [], [], []
    for trt, trt_scenarios in group_scenarios_by_trt(scenarios).items():

        # the correlation of each ordinate with the conditioning IM, for this TRT
        rho_sa_avgsa = corr_unconditional[trt].loc[CONDITIONING_LABEL, labels].to_numpy()

        # the ruptures themselves, as OpenQuake wants them: one row each
        contexts = stack_contexts(ctx_builder, trt_scenarios)

        # how extreme the conditioning level is for each of these ruptures, and
        # what the GMM says about every ordinate before conditioning
        eps = conditioning_epsilons(
            selection_ctx["gmm_map"][trt]["AvgSA"], contexts, iml)
        mu, sigma = gmm_mean_and_sigma(
            selection_ctx["gmm_map"][trt]["SA"], contexts, sa_imts)

        # rho varies with period (a column), eps with rupture (a row)
        mu_blocks.append(conditional_mean(mu, sigma,
                                          rho_sa_avgsa[:, None], eps[None, :]))
        sigma_blocks.append(conditional_sigma(sigma, rho_sa_avgsa[:, None]))
        ordered_scenarios.extend(trt_scenarios)

    return ConditionalSpectra(
        periods=np.asarray(periods, dtype=float),
        mu_cond=np.hstack(mu_blocks),
        sigma_cond=np.hstack(sigma_blocks),
        weights=np.array([s["weight"] for s in ordered_scenarios]),
        trts=np.array([s["trt"] for s in ordered_scenarios]),
        scenarios=ordered_scenarios)


def mixture_spectrum(spectra: ConditionalSpectra) -> tuple[np.ndarray, np.ndarray]:
    """The conditional spectrum: mixture mean and sigma of ln Sa at each period.

    The same two-layer combination :func:`expected_ln_saratio` applies to ln
    SaRatio, applied instead to one ordinate at a time. So this is the law of
    total variance run once per period, and it reuses
    :func:`combine_scenarios` rather than restating it.

    Returns ``(ln_mean, ln_sigma)``, both (n_periods,) and both in log units.
    ``exp(ln_mean)`` is the median of the conditional spectrum.
    """
    n_periods = len(spectra.periods)
    ln_mean = np.empty(n_periods)
    ln_sigma = np.empty(n_periods)

    for i in range(n_periods):
        # at this period each rupture contributes a normal (mu, sigma); the
        # scenarios mix with their disaggregation weights
        ln_mean[i], ln_sigma[i], _, _ = combine_scenarios(
            spectra.mu_cond[i, :], spectra.sigma_cond[i, :] ** 2, spectra.weights)

    return ln_mean, ln_sigma


# ---------------------------------------------------------------------------
# 7. SaRatio for one (site, structure, conditioning level)
# ---------------------------------------------------------------------------

def expected_ln_saratio(site: int, T1: float, disagg_df: pd.DataFrame, iml: float,
                        site_model: pd.DataFrame, selection_ctx: dict,
                        max_scenarios: int | None = None) -> dict:
    """Mean and sigma of ln SaRatio(T1) at ``site`` given AvgSA[0,3] = ``iml``.

    Parameters
    ----------
    site
        Site id, which is the row index of ``site_model``.
    T1
        First-mode period of the structure. The SaRatio band is built from it.
    disagg_df
        The rupture disaggregation at the conditioning level, with columns
        ``TRT``, ``Mag``, ``Dist`` and ``P(m|X=x)``. It may still carry its
        epsilon axis; :func:`scenarios_for` sums that away.
    iml
        The conditioning level itself, in g.
    site_model, selection_ctx
        As returned by
        ``setup_AvgSA03_gm_selection.setup_AvgSA03_gcim_gm_selection``.
    max_scenarios
        Truncate the scenario set. For stepping through the calculation only --
        the answer is then meaningless.

    Returns
    -------
    dict
        ``ln_saratio_mean`` and ``ln_saratio_sigma``, plus the diagnostics
        ``var_within_scenarios`` / ``var_between_scenarios`` (the two halves of
        the law of total variance), ``n_scenarios``, ``n_band_periods`` and the
        band limits.
    """
    # --- 1/2. the periods this structure needs -----------------------------
    band = saratio_periods(T1)
    T1_rounded = float(np.round(T1, 3))   # same 3 dp as the band, for the labels

    # The GMMs and the correlation tables are addressed by period, so they only
    # ever need each DISTINCT period once; T1 may coincide with a band period.
    # The SaRatio arithmetic below still sees T1 and the full band separately --
    # this axis is only how the two are looked up.
    periods = np.unique(np.concatenate([[T1_rounded], band]))
    position_of = {float(T): i for i, T in enumerate(periods)}
    i_t1 = position_of[T1_rounded]
    i_band = np.array([position_of[float(T)] for T in band])

    _, corr_given_iml = correlation_tables(periods)

    # --- 3. every scenario's conditional spectrum --------------------------
    spectra = conditional_spectra(site, periods, disagg_df, iml,
                                  site_model, selection_ctx, max_scenarios)

    # The reduction below wants the array, not the frame. Only the TRTs that
    # actually occur at this site are materialised: the correlation model
    # covers six, a site typically sees two or three, and each conversion
    # copies an (n_periods x n_periods) block.
    corr_arrays = {trt: corr_given_iml[trt].to_numpy()
                   for trt in dict.fromkeys(spectra.trts)}

    # --- 3.6/3.7. one scenario at a time -----------------------------------
    means, variances = [], []
    for k in range(len(spectra.scenarios)):
        m, v = scenario_ln_saratio(spectra.mu_cond[:, k], spectra.sigma_cond[:, k],
                                   corr_arrays[spectra.trts[k]], i_t1, i_band)
        means.append(m)
        variances.append(v)

    # --- 4/6. mix the scenarios together ------------------------------------
    mean, sigma_total, var_within, var_between = combine_scenarios(
        means, variances, spectra.weights)

    return {"ln_saratio_mean": mean,
            "ln_saratio_sigma": sigma_total,
            "var_within_scenarios": var_within,
            "var_between_scenarios": var_between,
            "n_scenarios": len(spectra.scenarios),
            "n_band_periods": len(band),
            "band_lo": float(band[0]),
            "band_hi": float(band[-1])}
