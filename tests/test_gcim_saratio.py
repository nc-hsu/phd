"""Tests for the GCIM SaRatio moments used as site-side covariates in nb 072.

What is pinned here is the arithmetic, not the ground-motion science. The two
numbers this module produces -- E[ln SaRatio] and its sigma, conditioned on
AvgSA[0,3] -- feed the regression dataset as ``ss_ln_saratio_mean`` and
``ss_ln_saratio_sigma``, where a sign error or a dropped covariance term would be
invisible: the values would still look like plausible SaRatios.

So the cases below are the ones with an answer that can be written down in
advance:

- **Fast unit tests**, no data files, always run. The band definition; the
  collapse and renormalisation of the disaggregation; the conditional moments at
  rho = 0 and rho = 1; and four cases of :func:`scenario_ln_saratio` whose
  variance is exact -- a degenerate band, an uncorrelated band, a perfectly
  correlated band, and a hand-computed two-period case.
- **A regression test** against real pipeline output, skipped cleanly when the
  disaggregation product is absent (the pattern ``test_pipeline_regression.py``
  uses).

Running this file as a script instead of under pytest produces the worked
example, in ``tests/saratio_worked_example/`` (gitignored, regenerate at will):

- a figure -- the conditional spectrum with Sa(T1), the SaRatio band and its
  geometric mean marked, the per-scenario spectra behind it, and the resulting
  ln SaRatio distribution (:func:`draw_worked_example`);
- the numbers behind it as CSVs -- every scenario's conditional spectrum, the
  scenario weights, the period axis and the correlation matrices
  (:func:`write_worked_example`).

Both are built from one :func:`build_worked_example` call, so the figure and the
CSVs always describe the same calculation.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from phd_project.config.config import load_config
from phd_project.scripts import gcim_saratio as gsr

cfg = load_config()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def disagg_frame(rows) -> pd.DataFrame:
    """A disaggregation frame in the shape the OpenQuake extraction writes.

    ``rows`` are ``(TRT, Mag, Dist, Eps, probability)`` tuples, so a test can
    spell out several epsilon bins of the same rupture and check they collapse.
    """
    return pd.DataFrame(rows, columns=["TRT", "Mag", "Dist", "Eps", "P(m|X=x)"])


class FakeGMM:
    """A stand-in GMM that writes known numbers, to pin the shape contract.

    OpenQuake GMMs return nothing and write into the arrays they are handed, so
    a fake has to do the same. This one writes ``mean[i, k] = base_mean + i + k``
    and a constant sigma, which makes both axes identifiable in the output.
    """

    def __init__(self, base_mean=0.0, sigma=0.5):
        self.base_mean = base_mean
        self.sigma = sigma

    def compute(self, ctx, imts, mean, sigma, tau, phi):
        for i in range(len(imts)):
            for k in range(len(ctx)):
                mean[i, k] = self.base_mean + i + k
        sigma[:] = self.sigma


# ---------------------------------------------------------------------------
# The SaRatio band
# ---------------------------------------------------------------------------

def test_band_runs_from_a_fifth_of_T1_to_three_T1_inclusive():
    # T1 = 1.0 s, so the band is 0.2 ... 3.0 s. Both ends must be present: the
    # upper one only is if the arange stop is nudged past it.
    band = gsr.saratio_periods(1.0)
    assert band[0] == pytest.approx(0.2)
    assert band[-1] == pytest.approx(3.0)


def test_band_is_spaced_at_the_sampling_interval():
    band = gsr.saratio_periods(0.65)
    steps = np.diff(band)
    assert steps == pytest.approx(gsr.SARATIO_DT * np.ones_like(steps))


def test_band_length_matches_the_arithmetic():
    # (3.0 - 0.2) * T1 / 0.01 + 1, the count of a closed interval.
    for T1 in (0.311, 0.65, 0.911):
        expected = round((gsr.SARATIO_TB - gsr.SARATIO_TA) * T1 / gsr.SARATIO_DT) + 1
        assert len(gsr.saratio_periods(T1)) == expected


def test_band_periods_are_rounded_to_three_decimals():
    # The correlation model labels its rows f"SA({round(T, 3)})", so a period
    # carrying more digits would simply not be found in the lookup.
    band = gsr.saratio_periods(0.649888)
    assert np.all(band == np.round(band, 3))
    assert gsr.sa_label(band[0]) in {f"SA({band[0]})", f"SA({band[0]:g})"}


# ---------------------------------------------------------------------------
# Turning a disaggregation into scenarios
# ---------------------------------------------------------------------------

def test_scenarios_sum_the_epsilon_axis_away():
    # One rupture split across three epsilon bins must come back as one
    # scenario carrying the total probability.
    df = disagg_frame([("Shallow Default", 6.0, 20.0, -1.0, 0.1),
                       ("Shallow Default", 6.0, 20.0, 0.0, 0.2),
                       ("Shallow Default", 6.0, 20.0, 1.0, 0.3)])
    scenarios = gsr.scenarios_for(df)
    assert len(scenarios) == 1
    assert scenarios[0]["mag"] == 6.0
    assert scenarios[0]["weight"] == pytest.approx(1.0)


def test_scenarios_drop_zero_probability_cells():
    # The disaggregation grid is a full (TRT x Mag x Dist) product and most
    # cells are empty; carrying them would cost time and contribute nothing.
    df = disagg_frame([("Shallow Default", 5.0, 10.0, 0.0, 0.0),
                       ("Shallow Default", 6.0, 20.0, 0.0, 0.4),
                       ("Craton", 7.0, 50.0, 0.0, 0.0)])
    scenarios = gsr.scenarios_for(df)
    assert len(scenarios) == 1
    assert scenarios[0]["mag"] == 6.0


def test_scenario_weights_are_normalised():
    # The mixture formulas in combine_scenarios are the textbook ones, which
    # assume the weights sum to 1; the raw P(m|X=x) column does not.
    df = disagg_frame([("Shallow Default", 6.0, 20.0, 0.0, 0.3),
                       ("Craton", 7.0, 50.0, 0.0, 0.1)])
    scenarios = gsr.scenarios_for(df)
    weight_of = {s["trt"]: s["weight"] for s in scenarios}

    assert sum(weight_of.values()) == pytest.approx(1.0)
    assert weight_of["Shallow Default"] == pytest.approx(0.75)
    assert weight_of["Craton"] == pytest.approx(0.25)


def test_scenarios_come_back_sorted_by_trt_then_magnitude_then_distance():
    # The collapse goes through a groupby, so the order is the sorted key order
    # and NOT the order of the disaggregation file. Nothing downstream depends
    # on it -- the mixture is a sum -- but max_scenarios slices this order, so
    # it is worth knowing which scenarios a truncated run keeps.
    df = disagg_frame([("Volcanic", 5.0, 10.0, 0.0, 0.2),
                       ("Shallow Default", 7.0, 20.0, 0.0, 0.1),
                       ("Shallow Default", 6.0, 30.0, 0.0, 0.3),
                       ("Craton", 7.0, 50.0, 0.0, 0.4)])
    scenarios = gsr.scenarios_for(df)

    assert [(s["trt"], s["mag"]) for s in scenarios] == [
        ("Craton", 7.0), ("Shallow Default", 6.0),
        ("Shallow Default", 7.0), ("Volcanic", 5.0)]


def test_truncating_the_scenario_set_renormalises_it():
    # Otherwise the "between scenarios" variance would be nonsense rather than
    # merely approximate. The two kept are the first two in sorted order, i.e.
    # Craton (0.3) and Shallow Default (0.5), so the kept mass is 0.8.
    df = disagg_frame([("Shallow Default", 6.0, 20.0, 0.0, 0.5),
                       ("Craton", 7.0, 50.0, 0.0, 0.3),
                       ("Volcanic", 5.0, 10.0, 0.0, 0.2)])
    scenarios = gsr.scenarios_for(df, max_scenarios=2)

    assert [s["trt"] for s in scenarios] == ["Craton", "Shallow Default"]
    assert sum(s["weight"] for s in scenarios) == pytest.approx(1.0)
    assert scenarios[0]["weight"] == pytest.approx(0.3 / 0.8)
    assert scenarios[1]["weight"] == pytest.approx(0.5 / 0.8)


# ---------------------------------------------------------------------------
# Grouping by Tectonic Region Type
# ---------------------------------------------------------------------------

def test_grouping_by_trt_partitions_the_scenarios_exactly():
    scenarios = [{"trt": "Shallow Default", "id": 0},
                 {"trt": "Craton", "id": 1},
                 {"trt": "Shallow Default", "id": 2},
                 {"trt": "Craton", "id": 3}]
    grouped = gsr.group_scenarios_by_trt(scenarios)

    assert set(grouped) == {"Shallow Default", "Craton"}
    assert sum(len(v) for v in grouped.values()) == len(scenarios)
    # every scenario appears exactly once, and order within a TRT is preserved
    assert [s["id"] for s in grouped["Shallow Default"]] == [0, 2]
    assert [s["id"] for s in grouped["Craton"]] == [1, 3]


# ---------------------------------------------------------------------------
# The conditional moments
# ---------------------------------------------------------------------------

def test_an_uncorrelated_ordinate_is_untouched_by_the_conditioning():
    # rho = 0: knowing AvgSA says nothing about this period, so both the mean
    # and the spread come through unchanged however extreme epsilon is.
    mu = np.array([[-1.0], [-2.0]])
    sigma = np.array([[0.6], [0.7]])
    rho = np.zeros((2, 1))
    assert gsr.conditional_mean(mu, sigma, rho, np.array([[3.0]])) == pytest.approx(mu)
    assert gsr.conditional_sigma(sigma, rho) == pytest.approx(sigma)


def test_a_perfectly_correlated_ordinate_has_no_conditional_spread_left():
    # rho = 1: the ordinate is a deterministic function of the conditioning IM,
    # so once that is fixed there is nothing left to vary.
    sigma = np.array([[0.6], [0.7]])
    rho = np.ones((2, 1))
    assert gsr.conditional_sigma(sigma, rho) == pytest.approx(np.zeros((2, 1)))


def test_conditioning_never_increases_the_spread():
    sigma = np.array([[0.6], [0.7], [0.5]])
    for rho_value in (-0.9, -0.3, 0.0, 0.4, 0.95):
        rho = np.full((3, 1), rho_value)
        assert np.all(gsr.conditional_sigma(sigma, rho) <= sigma + 1e-15)


def test_conditional_moments_broadcast_period_against_rupture():
    # rho varies with period (a column), eps with rupture (a row); the result
    # must be one value per (period, rupture).
    mu = np.zeros((3, 4))
    sigma = np.ones((3, 4))
    rho = np.array([[0.1], [0.2], [0.3]])
    eps = np.array([[1.0, 2.0, 3.0, 4.0]])

    mu_cond = gsr.conditional_mean(mu, sigma, rho, eps)
    assert mu_cond.shape == (3, 4)
    # period 1 (rho = 0.2), rupture 2 (eps = 3.0) -> 0 + 1 * 0.2 * 3.0
    assert mu_cond[1, 2] == pytest.approx(0.6)


def test_the_gmm_wrapper_returns_one_row_per_period_and_one_column_per_rupture():
    ctx = np.zeros(4)                      # 4 ruptures; FakeGMM ignores content
    imts = [gsr.SA(0.2), gsr.SA(0.3), gsr.SA(0.5)]
    mean, sigma = gsr.gmm_mean_and_sigma(FakeGMM(base_mean=10.0), ctx, imts)

    assert mean.shape == (3, 4)
    assert sigma.shape == (3, 4)
    assert mean[2, 3] == pytest.approx(10.0 + 2 + 3)


def test_conditioning_epsilon_is_the_standardised_distance_to_the_level():
    # FakeGMM gives mu = 7.0 and sigma = 0.5 for a single IM, single rupture,
    # so eps = (ln(iml) - 7.0) / 0.5.
    ctx = np.zeros(1)
    iml = float(np.exp(8.0))
    eps = gsr.conditioning_epsilons(FakeGMM(base_mean=7.0, sigma=0.5), ctx, iml)

    assert eps.shape == (1,)
    assert eps[0] == pytest.approx((8.0 - 7.0) / 0.5)


def test_conditioning_epsilon_is_returned_per_rupture():
    # FakeGMM makes mu depend on the rupture index, so the two ruptures must
    # come back with different epsilons rather than one value broadcast.
    ctx = np.zeros(3)
    eps = gsr.conditioning_epsilons(FakeGMM(base_mean=0.0, sigma=1.0), ctx, np.exp(5.0))
    assert eps.shape == (3,)
    assert eps == pytest.approx([5.0, 4.0, 3.0])


# ---------------------------------------------------------------------------
# ln SaRatio within one scenario -- the load-bearing arithmetic
# ---------------------------------------------------------------------------

def test_a_band_of_only_T1_gives_a_saratio_of_exactly_one():
    # ln SaRatio = ln Sa(T1) - ln Sa(T1) = 0, identically, whatever the
    # ground motion does. This exercises all three variance terms against a
    # known answer: s1^2 + s1^2 - 2 s1^2 = 0.
    mu_cond = np.array([-1.0])
    sigma_cond = np.array([0.6])
    corr = np.array([[1.0]])

    mean, var = gsr.scenario_ln_saratio(mu_cond, sigma_cond, corr,
                                        i_t1=0, i_band=np.array([0]))
    assert mean == pytest.approx(0.0)
    assert var == pytest.approx(0.0)


def test_an_uncorrelated_band_adds_its_variance_in_quadrature():
    # T1 at index 0, band at indices 1..3, identity correlation, every sigma
    # equal to s. The cross term vanishes (rho_1i = 0) and the band mean has
    # variance s^2/n, so
    #     var = s^2 + s^2/n = 0.25 + 0.25/3
    s, n = 0.5, 3
    mu_cond = np.zeros(4)
    sigma_cond = np.full(4, s)
    corr = np.eye(4)

    mean, var = gsr.scenario_ln_saratio(mu_cond, sigma_cond, corr,
                                        i_t1=0, i_band=np.array([1, 2, 3]))
    assert mean == pytest.approx(0.0)
    assert var == pytest.approx(s ** 2 + s ** 2 / n)


def test_a_perfectly_correlated_spectrum_has_a_fixed_saratio():
    # If every ordinate moves together by the same amount, the shape of the
    # spectrum cannot change, so the ratio is deterministic:
    #     var = s^2 + (1/n^2)(n^2 s^2) - (2/n)(n s^2) = s^2 + s^2 - 2 s^2 = 0
    s = 0.4
    mu_cond = np.array([0.0, -0.5, -0.7, -1.0])
    sigma_cond = np.full(4, s)
    corr = np.ones((4, 4))

    mean, var = gsr.scenario_ln_saratio(mu_cond, sigma_cond, corr,
                                        i_t1=0, i_band=np.array([1, 2, 3]))
    # mean is still the spectral shape: 0 - (-0.5 - 0.7 - 1.0)/3
    assert mean == pytest.approx(0.0 - (-0.5 - 0.7 - 1.0) / 3)
    assert var == pytest.approx(0.0)


def test_scenario_variance_matches_a_hand_computed_two_period_case():
    # T1 at index 0 with s1 = 0.5; a one-period band at index 1 with s2 = 0.3;
    # rho_12|c = 0.4. With n = 1:
    #     term 1 = s1^2                    = 0.25
    #     term 2 = (1/1) * rho_22 s2 s2    = 0.09
    #     term 3 = (2/1) * rho_12 s1 s2    = 2 * 0.4 * 0.5 * 0.3 = 0.12
    #     var    = 0.25 + 0.09 - 0.12      = 0.22
    mu_cond = np.array([-0.8, -1.3])
    sigma_cond = np.array([0.5, 0.3])
    corr = np.array([[1.0, 0.4],
                     [0.4, 1.0]])

    mean, var = gsr.scenario_ln_saratio(mu_cond, sigma_cond, corr,
                                        i_t1=0, i_band=np.array([1]))
    assert mean == pytest.approx(-0.8 - (-1.3))
    assert var == pytest.approx(0.22)


def test_a_negative_variance_from_round_off_is_clamped_not_propagated():
    # The correlation table is interpolated, so it is not guaranteed positive
    # semi-definite and the quadratic form can land a hair below zero. That must
    # not reach a sqrt.
    mu_cond = np.array([0.0, 0.0])
    sigma_cond = np.array([1.0, 1.0])
    corr = np.array([[1.0, 1.0 + 1e-9],       # slightly inconsistent
                     [1.0 + 1e-9, 1.0]])

    _, var = gsr.scenario_ln_saratio(mu_cond, sigma_cond, corr,
                                     i_t1=0, i_band=np.array([1]))
    assert var >= 0.0


# ---------------------------------------------------------------------------
# Mixing the scenarios -- the law of total variance
# ---------------------------------------------------------------------------

def test_a_single_scenario_has_no_between_scenario_variance():
    mean, sigma, within, between = gsr.combine_scenarios([0.4], [0.09], [1.0])
    assert mean == pytest.approx(0.4)
    assert within == pytest.approx(0.09)
    assert between == pytest.approx(0.0)
    assert sigma == pytest.approx(0.3)


def test_two_scenarios_follow_the_law_of_total_variance():
    # Equal weights, means 0.2 and 0.6, variances 0.04 and 0.16:
    #     E      = 0.5*0.2 + 0.5*0.6                     = 0.4
    #     within = 0.5*0.04 + 0.5*0.16                   = 0.10
    #     between= 0.5*0.04 + 0.5*0.36 - 0.16            = 0.04
    mean, sigma, within, between = gsr.combine_scenarios(
        [0.2, 0.6], [0.04, 0.16], [0.5, 0.5])

    assert mean == pytest.approx(0.4)
    assert within == pytest.approx(0.10)
    assert between == pytest.approx(0.04)
    assert sigma == pytest.approx(np.sqrt(0.14))


def test_identical_scenarios_put_everything_in_the_within_term():
    # Nothing distinguishes the scenarios, so the spread between their means
    # must be zero however many of them there are.
    mean, sigma, within, between = gsr.combine_scenarios(
        [0.3] * 5, [0.09] * 5, [0.2] * 5)
    assert between == pytest.approx(0.0)
    assert within == pytest.approx(0.09)
    assert sigma == pytest.approx(0.3)


def test_the_two_contributions_sum_to_the_reported_variance():
    mean, sigma, within, between = gsr.combine_scenarios(
        [0.1, 0.5, 0.9], [0.02, 0.05, 0.11], [0.2, 0.3, 0.5])
    assert sigma ** 2 == pytest.approx(within + between)


# ---------------------------------------------------------------------------
# Regression against real pipeline output
# ---------------------------------------------------------------------------

# Values at the 475 yr level from the OpenQuake return-period disaggregation
# product, i.e. the source the notebook uses. They pin the whole chain, not any
# one step, so a change here means a change somewhere between the disaggregation
# and the mixture.
#
# They depend on DISAGG_SOURCE below: the "nearest_stripe" fallback conditions on
# a neighbouring stripe and gives different (still self-consistent) numbers. If
# the source changes, these have to be regenerated rather than loosened.
PINNED = {
    (0, 3): (0.374728189527, 0.255660149012),    # iml 0.0287321 g, 579 scenarios
    (0, 5): (0.435487283670, 0.249398937313),    # iml 0.0287321 g, 579 scenarios
    (17, 3): (0.286959008792, 0.243847290756),   # iml 0.0595837 g, 421 scenarios
}


@pytest.fixture(scope="module")
def pipeline_inputs():
    """The real rows, disaggregation and GCIM context, loaded once for the module.

    Module-scoped because ``load_gcim_context`` reads the flatfiles and builds
    the GMM map, which takes ~15 s -- too much to repeat per parametrised case.
    """
    try:
        rows = load_rows()
        disagg, imls = load_disaggregation([475])
        site_model, selection_ctx = load_gcim_context()
    except (FileNotFoundError, KeyError, OSError) as exc:
        pytest.skip(f"pipeline inputs not available on this machine: {exc}")
    return rows, disagg, imls, site_model, selection_ctx


@pytest.mark.parametrize("site,n_storeys", sorted(PINNED))
def test_expected_ln_saratio_reproduces_the_pinned_values(site, n_storeys,
                                                          pipeline_inputs):
    rows, disagg, imls, site_model, selection_ctx = pipeline_inputs

    T1 = float(rows[(rows["site"] == site)
                    & (rows["n_storeys"] == n_storeys)]["T1"].iloc[0])
    out = gsr.expected_ln_saratio(
        site=site, T1=T1, disagg_df=disagg[(site, 475)], iml=imls[(site, 475)],
        site_model=site_model, selection_ctx=selection_ctx)

    mean, sigma = PINNED[(site, n_storeys)]
    assert out["ln_saratio_mean"] == pytest.approx(mean, rel=1e-9)
    assert out["ln_saratio_sigma"] == pytest.approx(sigma, rel=1e-9)


# ===========================================================================
# Below here: the scaffolding for the worked example. None of it runs under
# pytest except through the regression test above -- it exists so the method
# can be driven and drawn from a single file.
# ===========================================================================

# Which row and level the worked example draws.
DEMO_SITE, DEMO_STOREYS, DEMO_RTP = 0, 3, 475

# Where the disaggregation at each return period comes from:
#   "rtp_product"    - the pickle nb 072 section 8.3 writes from an OpenQuake
#                      disaggregation run AT the return-period levels. The
#                      correct source; it is produced on the OQ machine.
#   "nearest_stripe" - reuses the per-site MSA stripe disaggregations already on
#                      disk, picking the stripe whose IML is closest to the
#                      site's return-period hazard level. The answer is then the
#                      neighbouring stripe's, not the return period's.
DISAGG_SOURCE = "rtp_product"

# Everything the worked example produces lands here: the figure and the CSVs
# behind it. It sits next to this file rather than under results/ because these
# are artefacts of running the tests, not outputs of the analysis, and it is
# gitignored for the same reason tests/golden/ is.
DEMO_DIR = Path(__file__).parent / "saratio_worked_example"
DEMO_PNG = DEMO_DIR / "gcim_saratio_method_example.png"

# Period range the demo spectrum is drawn over. Wider than the SaRatio band, so
# the band reads as a slice of the spectrum rather than as the whole picture.
# The upper limit is the correlation model's own ceiling.
PLOT_T_MIN, PLOT_T_MAX = 0.05, 3.0

N_SITES, N_ROWS, N_DESIGNS = 60, 120, 51

INT_PTH = (Path(cfg["proc_data"]["bootstrapping"]).parent
           / "regression_coefficients")
DISAGG_RTPS_FP = INT_PTH / "AvgSA_03_disagg_rtps_eps4.pickle"

# The first-mode period is not stored as a column anywhere; it is embedded in
# the ``intensity_measure`` string of the IDA fragility JSON.
PERIOD_RE = re.compile(r"period=([0-9.eE+-]+)")


def modal_period(site: int, n_storeys: int) -> float:
    """First-mode period of the nonlinear model for one (site, structure) row."""
    frag_dir = Path(cfg["proc_data"]["wp1_sites_fragility_curves"])
    fp = (frag_dir / f"site_{site}"
          / f"{n_storeys}s_cbf_dc2_site{site}_ida_femap695_collapsefragility_SA.json")
    with open(fp) as f:
        match = PERIOD_RE.search(json.load(f)["intensity_measure"])
    if match is None:
        raise ValueError(f"no period= in the intensity_measure of {fp.name}")
    return float(match.group(1))


def load_rows() -> pd.DataFrame:
    """The 120 (site, structure) rows with their design group and their T1."""
    from phd_project.scripts import fragility_data_models as fm

    designs = fm.build_design_factors(
        cfg["proc_data"]["unique_structural_designs_csv"],
        expect_designs=N_DESIGNS, expect_rows=N_ROWS, expect_sites=N_SITES)

    rows = designs[["site", "storeys", "design_group_id"]].copy()
    rows = rows.rename(columns={"storeys": "n_storeys"})
    rows["T1"] = [modal_period(s, n) for s, n
                  in zip(rows["site"], rows["n_storeys"])]
    return rows.sort_values(["site", "n_storeys"]).reset_index(drop=True)


def load_disaggregation(return_periods: list[int]) -> tuple[dict, dict]:
    """The rupture disaggregation and its conditioning level, keyed (site, rtp)."""
    if DISAGG_SOURCE == "rtp_product":
        if not DISAGG_RTPS_FP.is_file():
            raise FileNotFoundError(
                f"{DISAGG_RTPS_FP} is missing. It is written by notebook 072 "
                f"section 8.3 from an OpenQuake disaggregation run, which happens "
                f"on the OQ machine. Set DISAGG_SOURCE = 'nearest_stripe' to "
                f"drive the method from the stripe disaggregations instead.")
        with open(DISAGG_RTPS_FP, "rb") as f:
            product = pickle.load(f)
        return ({k: v for k, v in product["disagg"].items() if k[1] in return_periods},
                {k: float(v) for k, v in product["iml"].items() if k[1] in return_periods})

    if DISAGG_SOURCE == "nearest_stripe":
        return _load_disagg_nearest_stripe(return_periods)
    raise ValueError(f"unknown DISAGG_SOURCE {DISAGG_SOURCE!r}")


def _load_disagg_nearest_stripe(return_periods: list[int]) -> tuple[dict, dict]:
    """The MSA stripe disaggregations, at whichever stripe is nearest the level.

    The stripes were disaggregated at the IMLs the ground-motion selection
    needed, not at the return-period levels. The conditioning level and the
    disaggregation still belong together -- which is what the method needs --
    but the answer is the neighbouring stripe's, not the return period's.
    """
    from phd_project.scripts.disagg_shards import load_shards, read_index
    from phd_project.scripts.msa_ida_hypothesis_testing import (
        site_hazard_at_return_periods)

    shard_dir = cfg["proc_data"]["AvgSA_03_disagg_data_shards"]
    index = read_index(shard_dir)
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
    print("[fallback] conditioning on the nearest MSA stripe; these are NOT "
          "return-period numbers")
    return disagg_by_key, iml_by_key


def load_gcim_context() -> tuple[pd.DataFrame, dict]:
    """The site model and the GMM / context machinery of the WP1 GCIM."""
    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
        setup_AvgSA03_gcim_gm_selection)

    _, _, site_model, selection_ctx, _ = setup_AvgSA03_gcim_gm_selection(sites=())
    assert selection_ctx["occurence"], (
        "the pipeline disaggregates on occurrence, so the scenario weights must "
        "be P(m|X=x)")
    return site_model, selection_ctx


class WorkedExample(NamedTuple):
    """Everything one (site, structure, level) produces, ready to plot or write.

    Built once by :func:`build_worked_example` so the figure and the CSVs beside
    it are guaranteed to describe the same calculation.
    """

    site: int
    T1: float
    iml: float
    periods: np.ndarray          # (n_T,) the shared period axis
    band: np.ndarray             # (n,)   the SaRatio denominator periods
    i_t1: int                    # position of T1 in periods
    i_band: np.ndarray           # (n,)   positions of the band in periods
    spectra: object              # gsr.ConditionalSpectra
    ln_mean: np.ndarray          # (n_T,) mixture mean of ln Sa per period
    ln_sigma: np.ndarray         # (n_T,) mixture sigma of the same
    scenario_means: np.ndarray   # (n_k,) m_k, per-scenario mean of ln SaRatio
    scenario_vars: np.ndarray    # (n_k,) v_k, per-scenario variance
    mean: float                  # E[ln SaRatio]
    sigma: float                 # SD[ln SaRatio]
    within: float                # the two halves of the law of total variance
    between: float
    corr_unconditional: dict     # per TRT, DataFrame incl. the AvgSA row/column
    corr_given_iml: dict         # per TRT, DataFrame over the SA periods


def build_worked_example(site: int, T1: float, disagg_df: pd.DataFrame, iml: float,
                         site_model: pd.DataFrame,
                         selection_ctx: dict) -> WorkedExample:
    """Run the method for one row and keep every intermediate.

    :func:`gcim_saratio.expected_ln_saratio` throws the per-scenario quantities
    away once it has mixed them; here they are all kept, because they are what
    the figure draws and the CSVs record.
    """
    band = gsr.saratio_periods(T1)
    T1_rounded = float(np.round(T1, 3))

    # The calculation only ever needs T1 and the band, but then the plotted
    # period axis IS the band and the shading says nothing. So the spectrum is
    # built on a wider grid with T1 and the band unioned in. Nothing about the
    # SaRatio numbers changes: they are read off i_t1 and i_band, which are
    # positions into whatever axis is used -- asserted below.
    plot_grid = np.round(np.geomspace(PLOT_T_MIN, PLOT_T_MAX, 160), 3)
    periods = np.unique(np.concatenate([plot_grid, [T1_rounded], band]))
    position_of = {float(T): i for i, T in enumerate(periods)}
    i_t1 = position_of[T1_rounded]
    i_band = np.array([position_of[float(T)] for T in band])

    spectra = gsr.conditional_spectra(site, periods, disagg_df, iml,
                                      site_model, selection_ctx)
    ln_mean, ln_sigma = gsr.mixture_spectrum(spectra)

    corr_unconditional, corr_given_iml = gsr.correlation_tables(periods)
    corr_arrays = {trt: corr_given_iml[trt].to_numpy()
                   for trt in dict.fromkeys(spectra.trts)}

    means, variances = [], []
    for k in range(len(spectra.scenarios)):
        m, v = gsr.scenario_ln_saratio(spectra.mu_cond[:, k], spectra.sigma_cond[:, k],
                                       corr_arrays[spectra.trts[k]], i_t1, i_band)
        means.append(m)
        variances.append(v)
    mean, sigma, within, between = gsr.combine_scenarios(
        means, variances, spectra.weights)

    # The extra plotting periods must not have moved the answer.
    reference = gsr.expected_ln_saratio(site, T1, disagg_df, iml,
                                        site_model, selection_ctx)
    assert np.isclose(mean, reference["ln_saratio_mean"], rtol=1e-12)
    assert np.isclose(sigma, reference["ln_saratio_sigma"], rtol=1e-12)

    return WorkedExample(
        site=site, T1=T1, iml=iml, periods=periods, band=band,
        i_t1=i_t1, i_band=i_band, spectra=spectra,
        ln_mean=ln_mean, ln_sigma=ln_sigma,
        scenario_means=np.asarray(means), scenario_vars=np.asarray(variances),
        mean=mean, sigma=sigma, within=within, between=between,
        corr_unconditional=corr_unconditional, corr_given_iml=corr_given_iml)


def trt_slug(trt: str) -> str:
    """A Tectonic Region Type as a filename fragment: 'Shallow Default' -> 'shallow_default'."""
    return trt.lower().replace(" ", "_").replace("-", "_")


def write_worked_example(example: WorkedExample, out_dir: Path) -> list[Path]:
    """Write the worked example out as CSVs, and return what was written.

    Everything shares one period axis, so the spectra, the mixture and the
    correlation matrices can be read against each other row for row. Values are
    written to 6 significant figures: the correlation matrices are
    (n_periods x n_periods) and full precision would roughly double the files
    for digits that are interpolation artefacts anyway.
    """
    ex = example
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%.6g"
    written = []

    def save(frame: pd.DataFrame, name: str, **kw):
        fp = out_dir / name
        frame.to_csv(fp, float_format=fmt, **kw)
        written.append(fp)

    # the shared period axis
    save(pd.DataFrame({"period_s": ex.periods}), "periods.csv", index=False)

    # one row per rupture scenario: what it is, how much it counts, and the
    # ln SaRatio moments it contributed
    save(pd.DataFrame({
        "k": np.arange(len(ex.spectra.scenarios)),
        "trt": ex.spectra.trts,
        "mag": [s["mag"] for s in ex.spectra.scenarios],
        "rjb_km": [s["rjb"] for s in ex.spectra.scenarios],
        "weight": ex.spectra.weights,
        "ln_saratio_mean": ex.scenario_means,
        "ln_saratio_variance": ex.scenario_vars,
    }), "scenarios.csv", index=False)

    # the conditional spectra themselves: rows are periods, columns are
    # scenarios, matching scenarios.csv column for row
    columns = [f"k{k:04d}" for k in range(len(ex.spectra.scenarios))]
    index = pd.Index(ex.periods, name="period_s")
    save(pd.DataFrame(ex.spectra.mu_cond, index=index, columns=columns),
         "conditional_spectra_ln_mean.csv")
    save(pd.DataFrame(ex.spectra.sigma_cond, index=index, columns=columns),
         "conditional_spectra_ln_sigma.csv")

    # the mixture over those scenarios -- the conditional spectrum that is drawn
    save(pd.DataFrame({"ln_mean": ex.ln_mean, "ln_sigma": ex.ln_sigma,
                       "median_g": np.exp(ex.ln_mean)}, index=index),
         "mixture_spectrum.csv")

    # The correlation model covers six Tectonic Region Types; only those that
    # actually occur at this site are written, since the rest took no part.
    for trt in dict.fromkeys(ex.spectra.trts):
        save(ex.corr_unconditional[trt],
             f"correlation_unconditional_{trt_slug(trt)}.csv")
        save(ex.corr_given_iml[trt],
             f"correlation_given_iml_{trt_slug(trt)}.csv")

    return written


def draw_worked_example(example: WorkedExample, out_png: Path | None = None):
    """Three panels showing what the SaRatio moments are made of.

    (a) the conditional spectrum, with Sa(T1), the SaRatio band and the band's
        geometric mean marked -- the gap between the last two IS E[ln SaRatio];
    (b) the same spectrum per rupture scenario, coloured by Tectonic Region
        Type, so the mixture is visibly an average of a family;
    (c) the ln SaRatio distribution: one normal per scenario, plus the mixture,
        which is where the within/between variance split becomes visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ex = example
    site, T1, iml = ex.site, ex.T1, ex.iml
    periods, band, i_band = ex.periods, ex.band, ex.i_band
    spectra, ln_mean, ln_sigma = ex.spectra, ex.ln_mean, ex.ln_sigma
    means, variances = ex.scenario_means, ex.scenario_vars
    mean, sigma, within, between = ex.mean, ex.sigma, ex.within, ex.between

    median = np.exp(ln_mean)

    # the two quantities SaRatio is the ratio of
    sa_t1 = float(median[ex.i_t1])
    band_geomean = float(np.exp(ln_mean[i_band].mean()))

    trts = sorted(set(spectra.trts))
    colours = dict(zip(trts, plt.cm.viridis(np.linspace(0.1, 0.85, len(trts)))))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle(
        f"GCIM SaRatio at site {site}, $T_1$ = {T1:.3f} s, "
        f"conditioned on AvgSA[0,3] = {iml:.4g} g "
        f"({len(spectra.scenarios)} rupture scenarios)", y=0.99)

    # --- (a) the conditional spectrum -------------------------------------
    ax = axes[0]
    ax.axvspan(band[0], band[-1], color="0.90", zorder=0,
               label=f"SaRatio band  $0.2T_1$–$3T_1$  ({band[0]:.2f}–{band[-1]:.2f} s)")
    ax.fill_between(periods, np.exp(ln_mean - ln_sigma), np.exp(ln_mean + ln_sigma),
                    color="#4C72B0", alpha=0.20, lw=0, label=r"$\pm 1\,\sigma$")
    ax.plot(periods, median, color="#1F3B73", lw=2.0, label="conditional spectrum (median)")

    ax.axhline(band_geomean, color="#C44E52", lw=1.4, ls="--",
               label=f"band geometric mean = {band_geomean:.3f} g")
    ax.axvline(T1, color="0.35", lw=1.0, ls=":")
    ax.plot([T1], [sa_t1], "o", ms=7, color="#1F3B73", zorder=5)
    ax.annotate(f"$Sa(T_1)$ = {sa_t1:.3f} g", xy=(T1, sa_t1),
                xytext=(6, 10), textcoords="offset points", fontsize=9)

    # the gap between the two lines IS E[ln SaRatio]
    ax.annotate("", xy=(T1, sa_t1), xytext=(T1, band_geomean),
                arrowprops=dict(arrowstyle="<->", color="#C44E52", lw=1.4))
    ax.annotate(f"$E[\\ln SaRatio]$ = {mean:.3f}\n"
                f"$Sa(T_1)\\,/\\,$geomean = {np.exp(mean):.2f}",
                xy=(T1, np.sqrt(sa_t1 * band_geomean)),
                xytext=(20, -26), textcoords="offset points",
                fontsize=9, color="#C44E52",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

    ax.set(xscale="log", yscale="log", xlabel="period $T$ [s]",
           ylabel="$Sa$ [g]", title="(a) conditional spectrum")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # --- (b) the scenarios behind it --------------------------------------
    ax = axes[1]
    w_max = spectra.weights.max()
    for k in range(len(spectra.scenarios)):
        ax.plot(periods, np.exp(spectra.mu_cond[:, k]),
                color=colours[spectra.trts[k]], lw=0.8,
                alpha=float(0.12 + 0.75 * spectra.weights[k] / w_max))
    ax.plot(periods, median, color="k", lw=2.0, label="mixture median")
    ax.axvspan(band[0], band[-1], color="0.90", zorder=0)
    ax.axvline(T1, color="0.35", lw=1.0, ls=":")

    handles = [Line2D([], [], color=colours[t], lw=1.6,
                      label=f"{t}  (n={int(np.sum(spectra.trts == t))})") for t in trts]
    handles.append(Line2D([], [], color="k", lw=2.0, label="mixture median"))
    ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
    ax.set(xscale="log", yscale="log", xlabel="period $T$ [s]", ylabel="$Sa$ [g]",
           title="(b) one spectrum per rupture scenario")

    # --- (c) the resulting ln SaRatio distribution ------------------------
    # The mixture density is literally the weighted sum of the per-scenario
    # normals, so the thick line here is the sum of the thin ones. The dashed
    # line is the normal carrying the mean and sigma this module reports --
    # drawn alongside so it is visible that those are MOMENTS of the mixture,
    # not a claim that the mixture is itself normal.
    ax = axes[2]
    grid = np.linspace(mean - 4.5 * sigma, mean + 4.5 * sigma, 500)

    def normal_pdf(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    mixture_pdf = np.zeros_like(grid)
    for k in range(len(spectra.scenarios)):
        s_k = np.sqrt(variances[k])
        if s_k <= 0:
            continue
        component = spectra.weights[k] * normal_pdf(grid, means[k], s_k)
        mixture_pdf += component
        ax.plot(grid, component, color=colours[spectra.trts[k]], lw=0.8,
                alpha=float(0.15 + 0.70 * spectra.weights[k] / w_max))

    ax.plot(grid, mixture_pdf, color="k", lw=2.2,
            label="mixture = sum of the weighted scenarios")
    ax.plot(grid, normal_pdf(grid, mean, sigma), color="#C44E52", lw=1.6, ls="--",
            label=f"normal with the reported moments\n"
                  f"mean {mean:.3f},  $\\sigma$ {sigma:.3f}")
    ax.axvline(mean, color="#C44E52", lw=1.0, ls=":")

    ax.plot([], [], " ", label=f"within scenarios   {within:.4f}  "
                              f"({100*within/(within+between):.0f}% of variance)")
    ax.plot([], [], " ", label=f"between scenarios  {between:.4f}  "
                              f"({100*between/(within+between):.0f}%)")
    ax.set(xlabel=r"$\ln SaRatio(T_1)$", ylabel="probability density",
           title="(c) the mixture over rupture scenarios")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"written to {out_png}")
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rows = load_rows()
    disagg, imls = load_disaggregation([DEMO_RTP])
    site_model, selection_ctx = load_gcim_context()

    T1 = float(rows[(rows["site"] == DEMO_SITE)
                    & (rows["n_storeys"] == DEMO_STOREYS)]["T1"].iloc[0])
    key = (DEMO_SITE, DEMO_RTP)

    example = build_worked_example(
        site=DEMO_SITE, T1=T1, disagg_df=disagg[key], iml=imls[key],
        site_model=site_model, selection_ctx=selection_ctx)

    print(f"\nsite {DEMO_SITE}, {DEMO_STOREYS}-storey, T1 = {T1:.3f} s, "
          f"AvgSA[0,3] = {imls[key]:.4g} g")
    print(f"  {'ln_saratio_mean':<24} {example.mean}")
    print(f"  {'ln_saratio_sigma':<24} {example.sigma}")
    print(f"  {'var_within_scenarios':<24} {example.within}")
    print(f"  {'var_between_scenarios':<24} {example.between}")
    print(f"  {'n_scenarios':<24} {len(example.spectra.scenarios)}")
    print(f"  {'n_periods':<24} {len(example.periods)} "
          f"({len(example.band)} in the SaRatio band)")

    written = write_worked_example(example, DEMO_DIR)
    print(f"\n{len(written)} files written to {DEMO_DIR}:")
    for fp in written:
        print(f"  {fp.name:<45} {fp.stat().st_size / 1024:8.1f} kB")

    draw_worked_example(example, out_png=DEMO_PNG)
    plt.show()
