"""A deliberately slow, readable reference calculation of the site- and
building-specific SaRatio.

WHY THIS FILE EXISTS
--------------------
Notebook 072 section 8.5 (``gcim_ln_saratio``) already computes E[ln SaRatio] and
sigma[ln SaRatio] for every (site, structure) pair. Its maths is right but it is
written for speed: ln SaRatio appears as a single contraction ``a @ mu`` against a
coefficient vector built with ``np.unique(..., return_inverse=True)`` and
``np.add.at``, and every scenario's variance is produced at once by one
``np.einsum``. Nothing in that code ever holds "the mean for scenario k" as a
number you can look at.

This script computes the same two quantities by following
``admin/rough_algorithm_for_SARatio.txt`` literally: one visible loop per level
of the algorithm, formulas written the way the literature writes them, and every
intermediate quantity sitting in a local variable where a debugger can see it. It
is a reference implementation to step through, NOT a production path -- nothing
else in the project imports it.

The file name has no ``test_`` prefix on purpose, so a plain ``pytest`` run from
the repo root does not collect it.

WHAT IS BEING COMPUTED
----------------------
SaRatio (Eads et al. 2016; Zhong et al. 2022 Eq. 2) is the first-mode spectral
ordinate divided by the geometric mean of the ordinates over a band around it:

    SaRatio(T1) = Sa(T1) / [ PROD_i Sa(Ti) ] ** (1/n),   Ti in [0.2 T1, 3 T1]

which in log space is simply a numerator minus the mean of a denominator:

    ln SaRatio(T1) = ln Sa(T1) - (1/n) SUM_i ln Sa(Ti)

We want its mean and standard deviation given that the conditioning intensity
measure AvgSA[0,3] takes a particular value at the site (the GCIM distribution,
Bradley 2010). That distribution is a MIXTURE over the disaggregated rupture
scenarios: each scenario contributes a joint lognormal for the spectrum, and the
scenarios are combined with their disaggregation probabilities. So the
calculation has exactly two layers, and they are kept visibly separate here:

    1. within one scenario  -> a linear combination of jointly normal variables,
                               so a closed-form mean and variance;
    2. across scenarios     -> the law of total variance.

HOW TO USE IT
-------------
Set the knobs in section 1 and run the file. ``SITE_STRUCTURE_PAIRS`` overrides
the full 120-row set, ``MAX_SCENARIOS`` truncates the innermost loop, and
``RETURN_PERIODS`` currently holds only the 475 yr level. A single pair with a
handful of scenarios returns in a second or two, which is the intended debugging
mode.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from openquake.hazardlib.imt import SA, AvgSA
from pickagm.corrmodels import conditional_correlation_matrix

from phd_project.config.config import load_config
from phd_project.scripts import fragility_data_models as fm
from phd_project.scripts.disagg_shards import load_shards, read_index
from phd_project.scripts.msa_ida_hypothesis_testing import site_hazard_at_return_periods
from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    _initialise_ctx_builder,
    create_corr_model_map,
)
from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
    setup_AvgSA03_gcim_gm_selection,
)

cfg = load_config()


# ---------------------------------------------------------------------------
# 1. Control knobs
# ---------------------------------------------------------------------------

# Hazard-fixed conditioning levels, in years. Only the 475 yr level is wired up
# for now; adding 2500 / 5000 / 10000 later is a change to this list alone,
# PROVIDED the disaggregation source below carries those levels.
RETURN_PERIODS = [475]

# Which rows to run. ``None`` means the full study: 60 sites x {3-storey,
# 5-storey} = 120 rows (CLAUDE.md section 1). Otherwise a list of
# ``(site, n_storeys)`` pairs, e.g. ``[(0, 3), (0, 5)]``.
SITE_STRUCTURE_PAIRS = None

# Truncate the scenario loop. ``None`` uses every disaggregated scenario (~120
# per site after the epsilon axis is collapsed and zero-probability rows are
# dropped). A small integer makes one call return almost immediately, at the cost
# of a meaningless answer -- for stepping through the machinery only.
MAX_SCENARIOS = None

# Where the disaggregation at each return period comes from:
#   "rtp_product"   - the pickle nb 072 section 8.3 writes from an OpenQuake
#                     disaggregation run AT the return-period levels. This is the
#                     correct source; it is produced on the OQ machine.
#   "nearest_stripe" - a fallback that reuses the per-site MSA stripe
#                     disaggregations already on disk, picking the stripe whose
#                     IML is closest to the site's return-period hazard level.
#                     The numbers are then NOT the return-period numbers, because
#                     the disaggregation and the conditioning level both belong to
#                     the neighbouring stripe. Use it only to exercise the code.
DISAGG_SOURCE = "rtp_product"

# SaRatio band, Zhong et al. (2022) Eq. 2 after Eads et al. (2016).
SARATIO_TA, SARATIO_TB, SARATIO_DT = 0.2, 3.0, 0.01

# The conditioning intensity measure of the whole WP1 ground-motion selection.
CONDITIONING_IMT = AvgSA([0, 3])
CONDITIONING_LABEL = CONDITIONING_IMT.string  # "AvgSA([0, 3])"

# ``None`` prints the result frame only; otherwise a path to write it to.
WRITE_CSV = None

# Full-study dimensions, asserted on load so a silently truncated input is caught
# at the top rather than showing up as a short result frame at the bottom.
N_SITES, N_ROWS, N_DESIGNS = 60, 120, 51

# The intermediate-products folder nb 072 writes into.
INT_PTH = (Path(cfg["proc_data"]["bootstrapping"]).parent
           / "regression_coefficients")
DISAGG_RTPS_FP = INT_PTH / "AvgSA_03_disagg_rtps_eps4.pickle"


# ---------------------------------------------------------------------------
# 2. Inputs
#
# One loader per input, each returning a plain object. They are called once in
# __main__ and everything downstream takes them as arguments -- no module-level
# state holding data, so a debugger session can rebuild any one of them alone.
# ---------------------------------------------------------------------------

# The first-mode period is not stored as a column anywhere; it is embedded in the
# ``intensity_measure`` string of the IDA fragility JSON, e.g. "SA(period=0.62)".
# Same pattern nb 053 and nb 072 use.
PERIOD_RE = re.compile(r"period=([0-9.eE+-]+)")


def modal_period(site: int, n_storeys: int) -> float:
    """First-mode period of the nonlinear model for one (site, structure) row.

    Read back out of the row's IDA SA fragility JSON rather than from a summary
    table, because that file is the thing the fragility was actually fitted
    against -- a summary table could drift from it.
    """
    frag_dir = Path(cfg["proc_data"]["wp1_sites_fragility_curves"])
    fp = (frag_dir / f"site_{site}"
          / f"{n_storeys}s_cbf_dc2_site{site}_ida_femap695_collapsefragility_SA.json")
    with open(fp) as f:
        match = PERIOD_RE.search(json.load(f)["intensity_measure"])
    if match is None:
        raise ValueError(f"no period= in the intensity_measure of {fp.name}")
    return float(match.group(1))


def load_rows() -> pd.DataFrame:
    """The 120 (site, structure) rows with their design group and their T1.

    A row is one (site, structure) pair; each of the 60 sites hosts a 3-storey
    and a 5-storey structure. ``design_group_id`` is the label several sites
    share when they were given the same structural design -- it is what makes T1
    repeat across sites, which is why the correlation tables below are worth
    caching.
    """
    designs = fm.build_design_factors(
        cfg["proc_data"]["unique_structural_designs_csv"],
        expect_designs=N_DESIGNS, expect_rows=N_ROWS, expect_sites=N_SITES)

    rows = designs[["site", "storeys", "design_group_id"]].copy()
    rows = rows.rename(columns={"storeys": "n_storeys"})
    rows["T1"] = [modal_period(s, n) for s, n
                  in zip(rows["site"], rows["n_storeys"])]
    rows = rows.sort_values(["site", "n_storeys"]).reset_index(drop=True)

    # T1 is a property of the DESIGN, so it must be identical everywhere a design
    # group appears. If it is not, the caching below would silently use one
    # site's band for another site's structure.
    spread = rows.groupby("design_group_id")["T1"].agg(lambda x: x.max() - x.min())
    assert (spread < 1e-12).all(), (
        f"T1 varies within a design group:\n{spread[spread >= 1e-12]}")
    return rows


def load_disaggregation(return_periods: list[int]) -> tuple[dict, dict]:
    """The rupture disaggregation and the conditioning level, keyed (site, rtp).

    Returns ``(disagg_by_key, iml_by_key)``. The DataFrames carry the columns
    ``TRT``, ``Mag``, ``Dist``, ``Eps`` and the probability columns; they are
    collapsed and normalised later, in :func:`scenarios_for`, so that what this
    loader returns is still recognisably the thing on disk.
    """
    if DISAGG_SOURCE == "rtp_product":
        return _load_disagg_rtp_product(return_periods)
    if DISAGG_SOURCE == "nearest_stripe":
        return _load_disagg_nearest_stripe(return_periods)
    raise ValueError(f"unknown DISAGG_SOURCE {DISAGG_SOURCE!r}")


def _load_disagg_rtp_product(return_periods: list[int]) -> tuple[dict, dict]:
    """The proper source: the OpenQuake run disaggregated AT the RTP levels."""
    if not DISAGG_RTPS_FP.is_file():
        raise FileNotFoundError(
            f"{DISAGG_RTPS_FP} is missing. It is written by notebook 072 section "
            f"8.3 from an OpenQuake disaggregation run, which happens on the OQ "
            f"machine; copy it here with its .manifest.json. To exercise this "
            f"script in the meantime set DISAGG_SOURCE = 'nearest_stripe'.")

    with open(DISAGG_RTPS_FP, "rb") as f:
        product = pickle.load(f)

    disagg_by_key, iml_by_key = {}, {}
    for site in range(N_SITES):
        for rtp in return_periods:
            key = (site, rtp)
            if key not in product["disagg"]:
                raise KeyError(
                    f"{DISAGG_RTPS_FP.name} has no disaggregation for {key}; it "
                    f"carries {sorted({r for _, r in product['disagg']})} yr")
            disagg_by_key[key] = product["disagg"][key]
            iml_by_key[key] = float(product["iml"][key])
    return disagg_by_key, iml_by_key


def _load_disagg_nearest_stripe(return_periods: list[int]) -> tuple[dict, dict]:
    """Fallback source: the MSA stripe disaggregations already on disk.

    The stripes were disaggregated at the IMLs the ground-motion selection
    needed, not at the return-period levels, so for each (site, rtp) we take the
    stripe whose IML is nearest the site's hazard level at that return period and
    condition on THAT stripe's IML. The conditioning level and the
    disaggregation then still belong together -- which is what matters for the
    arithmetic being exercised -- but the answer is the neighbouring stripe's
    answer, not the return period's.
    """
    shard_dir = cfg["proc_data"]["AvgSA_03_disagg_data_shards"]
    index = read_index(shard_dir)

    # The AvgSA[0,3] hazard curves, inverted at each return period.
    hazard_curves = pd.read_pickle(cfg["proc_data"]["AvgSA_03_hazard_curves_4sig"])
    targets = site_hazard_at_return_periods(
        hazard_curves, range(N_SITES), tuple(return_periods), im_key="AvgSA")

    shards = load_shards(shard_dir, set(range(N_SITES)))

    disagg_by_key, iml_by_key = {}, {}
    for site in range(N_SITES):
        available = np.asarray(index[site]["AvgSA"], dtype=float)
        for rtp in return_periods:
            target = float(targets.loc[site, rtp])
            iml = float(available[np.argmin(np.abs(available - target))])
            disagg_by_key[(site, rtp)] = shards[site]["AvgSA"][iml]
            iml_by_key[(site, rtp)] = iml
    print(f"[fallback] conditioning on the nearest MSA stripe; these are NOT "
          f"return-period numbers")
    return disagg_by_key, iml_by_key


def load_gcim_context() -> tuple[pd.DataFrame, dict]:
    """The site model and the ground-motion / context machinery of the WP1 GCIM.

    ``sites=()`` tells the setup to load no disaggregation shards at all: we only
    want the GMM map (per tectonic region, keyed by IM name), the rupture-context
    builder and its assumptions. Takes ~10 s, mostly reading flatfiles.
    """
    _, _, site_model, selection_ctx, _ = setup_AvgSA03_gcim_gm_selection(sites=())
    assert selection_ctx["occurence"], (
        "the pipeline disaggregates on occurrence, so the scenario weights must "
        "be P(m|X=x); this script assumes that")
    return site_model, selection_ctx


# ---------------------------------------------------------------------------
# 3. The SaRatio band
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
# 4. Correlations
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
# 5. One rupture scenario
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


def gmm_mean_and_sigma(gmm, ctx: np.recarray, imts: list) -> tuple[np.ndarray, np.ndarray]:
    """Median ln IM and its total standard deviation from one GMM, one rupture.

    OpenQuake GMMs write into preallocated (n_imts, n_contexts) arrays rather
    than returning anything, which is why this wrapper exists. ``ctx`` here is
    always a single rupture, so both outputs are squeezed to vectors over the
    IMs. The GMM is called once for the whole period vector: it is vectorised
    over IMs, and calling it once per period would be ~140,000 calls over a full
    run for no gain in clarity.
    """
    mean = np.zeros((len(imts), len(ctx)))
    sigma = np.zeros_like(mean)
    tau = np.zeros_like(mean)
    phi = np.zeros_like(mean)
    gmm.compute(ctx, imts, mean, sigma, tau, phi)
    return mean[:, 0], sigma[:, 0]


def conditioning_epsilon(gmm_conditioning, ctx: np.recarray, iml: float) -> float:
    """How many sigma the conditioning level sits above this scenario's median.

        epsilon = (ln(iml) - mu_lnAvgSA) / sigma_lnAvgSA

    A rare, large level at a site dominated by small nearby events gives a large
    positive epsilon: that scenario has to be an unusually strong realisation to
    produce the level at all, and every other ordinate is pulled up with it.
    """
    mu, sigma = gmm_mean_and_sigma(gmm_conditioning, ctx, [CONDITIONING_IMT])
    return float((np.log(iml) - mu[0]) / sigma[0])


def conditional_mean(mu: np.ndarray, sigma: np.ndarray,
                     rho_sa_avgsa: np.ndarray, eps: float) -> np.ndarray:
    """E[ln Sa(Ti) | AvgSA = iml] for one scenario (Bradley 2010).

        mu_i|c = mu_i + sigma_i * rho_ic * eps
    """
    return mu + sigma * rho_sa_avgsa * eps


def conditional_sigma(sigma: np.ndarray, rho_sa_avgsa: np.ndarray) -> np.ndarray:
    """SD[ln Sa(Ti) | AvgSA = iml] for one scenario.

        s_i = sigma_i * sqrt(1 - rho_ic^2)

    Knowing AvgSA can only remove uncertainty, never add it, so s_i <= sigma_i
    always -- a useful thing to assert on when stepping through.
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
# 6. Combining scenarios
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
# 7. One (site, structure, return period)
# ---------------------------------------------------------------------------

def expected_ln_saratio(site: int, T1: float, disagg_df: pd.DataFrame, iml: float,
                        site_model: pd.DataFrame, selection_ctx: dict,
                        max_scenarios: int | None = None) -> dict:
    """Mean and sigma of ln SaRatio(T1) at ``site`` given AvgSA[0,3] = ``iml``.

    This is step 3 of ``admin/rough_algorithm_for_SARatio.txt``, written out as
    the loop it describes. Returns a dict so the caller can keep the diagnostics
    (scenario count, band size, the two variance contributions) alongside the two
    numbers that matter.
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
    sa_imts = [SA(float(T)) for T in periods]
    labels = [sa_label(T) for T in periods]

    corr_unconditional, corr_given_iml = correlation_tables(periods)

    # --- the site's own context -------------------------------------------
    # The site id IS the row index of the site model (lat, lon, vs30, z1pt0, ...).
    site_ctx = deepcopy(selection_ctx)
    site_ctx["site_params"] = site_model.loc[site, :].to_dict()
    ctx_builder = _initialise_ctx_builder(site_ctx)

    # --- 3. loop over the rupture scenarios --------------------------------
    scenarios = scenarios_for(disagg_df, max_scenarios)

    means, variances, weights = [], [], []
    for scenario in scenarios:
        trt = scenario["trt"]

        # 3.1 the correlations this tectonic region uses
        rho_sa_avgsa = corr_unconditional[trt].loc[CONDITIONING_LABEL, labels].to_numpy()
        corr_given_iml_np = corr_given_iml[trt].to_numpy()

        # the rupture itself, as OpenQuake wants it
        ctx = ctx_builder.ctx({"trt": trt, "mag": scenario["mag"],
                               "rjb": scenario["rjb"], "eps": scenario["eps"]})

        # 3.2-3.5 the conditional moments of every ordinate we need
        eps = conditioning_epsilon(selection_ctx["gmm_map"][trt]["AvgSA"], ctx, iml)
        mu, sigma = gmm_mean_and_sigma(selection_ctx["gmm_map"][trt]["SA"], ctx, sa_imts)
        mu_cond = conditional_mean(mu, sigma, rho_sa_avgsa, eps)
        sigma_cond = conditional_sigma(sigma, rho_sa_avgsa)

        # 3.6-3.7 this scenario's ln SaRatio moments
        m, v = scenario_ln_saratio(mu_cond, sigma_cond, corr_given_iml_np,
                                   i_t1, i_band)
        means.append(m)
        variances.append(v)
        weights.append(scenario["weight"])

    # --- 4/6. mix the scenarios together ------------------------------------
    mean, sigma_total, var_within, var_between = combine_scenarios(
        means, variances, weights)

    return {"ln_saratio_mean": mean,
            "ln_saratio_sigma": sigma_total,
            "var_within_scenarios": var_within,
            "var_between_scenarios": var_between,
            "n_scenarios": len(scenarios),
            "n_band_periods": len(band),
            "band_lo": float(band[0]),
            "band_hi": float(band[-1])}


# ---------------------------------------------------------------------------
# 8. Run it
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    """Run every requested (site, structure, return period) and report."""
    rows = load_rows()
    if SITE_STRUCTURE_PAIRS is not None:
        wanted = {(int(s), int(n)) for s, n in SITE_STRUCTURE_PAIRS}
        rows = rows[[(s, n) in wanted for s, n
                     in zip(rows["site"], rows["n_storeys"])]]
        rows = rows.reset_index(drop=True)
        assert len(rows) == len(wanted), (
            f"{len(wanted) - len(rows)} of the requested pairs are not rows of "
            f"the study")

    print(f"{len(rows)} (site, structure) rows x {len(RETURN_PERIODS)} return "
          f"period(s) = {len(rows) * len(RETURN_PERIODS)} calculations")

    disagg_by_key, iml_by_key = load_disaggregation(RETURN_PERIODS)
    site_model, selection_ctx = load_gcim_context()

    results = []
    for row in rows.itertuples(index=False):
        for rtp in RETURN_PERIODS:
            started = time.perf_counter()
            out = expected_ln_saratio(
                site=int(row.site), T1=float(row.T1),
                disagg_df=disagg_by_key[(int(row.site), rtp)],
                iml=iml_by_key[(int(row.site), rtp)],
                site_model=site_model, selection_ctx=selection_ctx,
                max_scenarios=MAX_SCENARIOS)
            elapsed = time.perf_counter() - started

            results.append({"site": int(row.site),
                            "n_storeys": int(row.n_storeys),
                            "design_group_id": row.design_group_id,
                            "rtp": rtp,
                            "T1": float(row.T1),
                            "iml": iml_by_key[(int(row.site), rtp)],
                            **out,
                            "seconds": elapsed})
            print(f"  site {row.site:>2} {row.n_storeys}s  {rtp:>5} yr  "
                  f"T1={row.T1:.3f}  mean={out['ln_saratio_mean']:+.4f}  "
                  f"sigma={out['ln_saratio_sigma']:.4f}  "
                  f"({out['n_scenarios']} scenarios, {elapsed:.1f} s)")

    results = pd.DataFrame(results)

    print("\n" + "=" * 78)
    print(results[["ln_saratio_mean", "ln_saratio_sigma",
                   "var_within_scenarios", "var_between_scenarios"]]
          .describe().T.round(4).to_string())

    if WRITE_CSV is not None:
        Path(WRITE_CSV).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(WRITE_CSV, index=False)
        print(f"\nwritten to {WRITE_CSV}")

    return results


if __name__ == "__main__":
    main()
