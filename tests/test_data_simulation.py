"""Verification of the fake-data simulator, and of fit_reml through it.

Two kinds of test live here.

The cheap ones check the plumbing: shapes, reproducibility, that ``Z Z'`` comes out
as the matrix written by hand in notebook 076 cell 31, and that a mis-shaped factor
raises instead of sliding through.

The expensive ones are the actual statistical verification -- can ``fit_reml``
recover a truth it was not told? These run at a small ``n_sims`` on small synthetic
structures so the suite stays fast and needs no data files, which means the
Monte-Carlo tolerances are loose. They are calibration checks, not precision
measurements: run them at full size from a scratch script when a number matters.

The test that earns its keep is ``test_identifiability_trap``. The roadmap's A1c
section warns that ``tau_d^2`` silently absorbs the shared-b_j sampling covariance
unless ``Z V_bb Z'`` is carried in ``V_known``, and that the contamination runs
entirely in favour of "the building matters more" -- the conclusion the analysis is
supposed to be testing. That claim is only demonstrable with a known truth of
``tau_d^2 = 0``, which is what a simulator is for.
"""

import numpy as np
import pandas as pd
import pytest

import phd_project.scripts.data_simulation as dsim
import phd_project.scripts.metaregression_analysis as mra

# Small enough to run in seconds, large enough that the crossed structure is real
# and the variance components are actually identified: 10 designs x 20 sites x 2
# structures per site, so 4 rows per design. Mirrors the study's shape (designs
# spanning several sites, two rows per site) at a third of its size. Going much
# smaller stops being a test of the estimator and starts being a test of how a
# boundary-constrained variance component behaves with two observations per group.
N_SITES = 20
N_DESIGNS = 10
N_ROWS = 2 * N_SITES
N_SIMS = 500
SEED = 4321

# The fit is the bottleneck, not the simulation, so every test that does not
# specifically exercise the restarts uses the fast objective and a single start.
FAST_FIT = {"method": "cholesky", "n_starts": 1}


def unbiased(row, tol: float = 0.2) -> bool:
    """Is the bias small compared with the estimator's own spread?

    Deliberately NOT ``abs(row["z"]) < 3``. ``z = bias / mcse`` asks whether this
    many simulations can *resolve* the bias from zero, and mcse shrinks as
    1/sqrt(n_sims) -- so a z-threshold gets harder to satisfy the more work you do,
    and will eventually flag a bias of 0.1% of the truth as a failure. That makes it
    a good diagnostic (it is reported in the summary table, where you read it
    alongside pct_error) and a bad regression assertion.

    What matters for a regression test is whether the bias is large enough to
    matter, which means measuring it against the estimator's own standard
    deviation. Across seeds this ratio sits below 0.1 for every fit here; 0.2 leaves
    headroom for the Monte-Carlo noise at N_SIMS (which is ~1/sqrt(N_SIMS) in these
    same units) while still catching a genuinely biased estimator, which would blow
    well past 1. If N_SIMS is cut, this tolerance has to grow with it.
    """
    return abs(row["bias"]) < tol * row["empirical_sd"]


@pytest.fixture
def structure():
    """Row -> design and row -> site indicators for the small synthetic study.

    Rows alternate 3-storey / 5-storey within a site, so ``S`` has a 2 x 2 block per
    site. Designs are assigned so that each spans more than one site -- if every
    design sat at one site, design and site would be nested rather than crossed and
    the test would not exercise the case the model is for.
    """
    site_codes = np.repeat(np.arange(N_SITES), 2)
    design_codes = np.arange(N_ROWS) % N_DESIGNS

    Z = mra.make_indicator(design_codes, n_levels=N_DESIGNS)
    S = mra.make_indicator(site_codes, n_levels=N_SITES)
    return {"Z": Z, "S": S, "I": np.eye(N_ROWS),
            "design_codes": design_codes, "site_codes": site_codes}


@pytest.fixture
def v_diag():
    """A plausible spread of per-row sampling variances (the A1a V_known)."""
    rng = np.random.default_rng(7)
    return 0.0004 + 0.002 * rng.random(N_ROWS)


@pytest.fixture
def V_bb():
    """A dense design-level sampling covariance, as the shared FX record set produces.

    Built as ``A A' / m`` so it is PSD by construction and strongly correlated
    off-diagonal -- which is the physically correct answer for one fixed record set,
    not an artefact (see ``mra.bootstrap_cov``).
    """
    rng = np.random.default_rng(11)
    m = 40
    A = rng.normal(0, 0.03, (N_DESIGNS, m)) + rng.normal(0, 0.03, (1, m))
    return A @ A.T / m


# =============================================================================
# Plumbing
# =============================================================================

def test_build_component_gs_matches_hand_calculation():
    """The 4-row example written out by hand in notebook 076 cell 31."""
    Z = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=float)
    S = np.array([[1, 0],
                  [1, 0],
                  [0, 1],
                  [0, 1]], dtype=float)

    Gs, names = dsim.build_component_gs({"tau_d": (Z, 0.1), "tau_s": (S, 0.1)})

    assert names == ["tau_d", "tau_s"]
    np.testing.assert_array_equal(Gs[0], np.array([[1, 0, 1, 0],
                                                   [0, 1, 0, 0],
                                                   [1, 0, 1, 0],
                                                   [0, 0, 0, 1]], dtype=float))
    np.testing.assert_array_equal(Gs[1], np.array([[1, 1, 0, 0],
                                                   [1, 1, 0, 0],
                                                   [0, 0, 1, 1],
                                                   [0, 0, 1, 1]], dtype=float))


def test_component_order_is_insertion_order(structure):
    """Gs, names and the reported taus must all follow the order the dict was written."""
    components = {"tau_d": (structure["Z"], 0.1),
                  "tau_s": (structure["S"], 0.1),
                  "tau_u": (structure["I"], 0.1)}
    Gs, names = dsim.build_component_gs(components)
    assert names == ["tau_d", "tau_s", "tau_u"]
    assert [G.shape for G in Gs] == [(N_ROWS, N_ROWS)] * 3


def test_mis_shaped_factor_raises(v_diag):
    """A factor with the wrong number of ROWS is the shape bug that must be loud.

    The specific mistake this guards against is building the site indicator as
    (n, n) instead of (n, n_sites) -- which makes S S' the identity, silently turning
    the site component into a second copy of tau_u.
    """
    wrong = np.eye(N_ROWS + 1)
    with pytest.raises(ValueError, match="rows but the model has n"):
        dsim.simulate_reml_fits(0.0, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                {"tau_u": (wrong, 0.1)}, n_sims=1)


def test_beta_true_length_must_match_X(v_diag, structure):
    X = np.column_stack([np.ones(N_ROWS), np.linspace(-1, 1, N_ROWS)])
    with pytest.raises(ValueError, match="one true coefficient per column"):
        dsim.simulate_reml_fits(0.5, X, np.diag(v_diag),
                                {"tau_u": (structure["I"], 0.1)}, n_sims=1)


def test_non_psd_V_known_raises_and_points_at_check_V_bb(structure):
    """A broken V_bb invalidates the real fit too, so it is raised, not jittered."""
    bad = np.full((N_ROWS, N_ROWS), 0.001)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="check_V_bb"):
        dsim.simulate_reml_fits(0.0, np.ones((N_ROWS, 1)), bad,
                                {"tau_u": (structure["I"], 0.1)}, n_sims=1)


def test_draws_shape_and_columns(v_diag, structure):
    components = {"tau_d": (structure["Z"], 0.1), "tau_u": (structure["I"], 0.05)}
    res = dsim.simulate_reml_fits(0.3, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                  components, n_sims=5, seed=SEED,
                                  fit_kwargs=FAST_FIT, progress=False)

    assert len(res.draws) == 5
    assert set(res.draws.columns) == {
        "converged", "nll", "start_spread", "beta_0", "se_beta_0",
        "tau2_tau_d", "at_zero_tau_d", "tau2_tau_u", "at_zero_tau_u"}
    assert res.n == N_ROWS and res.q == 1
    np.testing.assert_allclose(res.tau2_true, [0.01, 0.0025])


def test_same_seed_reproduces_exactly(v_diag, structure):
    components = {"tau_u": (structure["I"], 0.2)}
    kwargs = dict(n_sims=5, seed=SEED, fit_kwargs=FAST_FIT, progress=False)
    a = dsim.simulate_reml_fits(0.3, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                components, **kwargs)
    b = dsim.simulate_reml_fits(0.3, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                components, **kwargs)
    pd.testing.assert_frame_equal(a.draws, b.draws)


def test_simulate_y_parts_sum_to_y(v_diag, structure):
    """return_parts must decompose the same y it returns, not a second draw."""
    rng = np.random.default_rng(1)
    components = {"tau_d": (structure["Z"], 0.1), "tau_u": (structure["I"], 0.05)}
    X = np.ones((N_ROWS, 1))
    y, parts = dsim.simulate_y(rng, [0.4], X, np.diag(v_diag), components,
                               return_parts=True)

    rebuilt = (X @ np.array([0.4]) + parts["tau_d"]["contribution"]
               + parts["tau_u"]["contribution"] + parts["eps"])
    np.testing.assert_allclose(y, rebuilt)


def test_summarise_simulation_layout(v_diag, structure):
    res = dsim.simulate_reml_fits(0.3, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                  {"tau_u": (structure["I"], 0.2)}, n_sims=20,
                                  seed=SEED, fit_kwargs=FAST_FIT, progress=False)
    summary = dsim.summarise_simulation(res)

    assert list(summary.index) == ["beta_0", "tau2_tau_u"]
    # Variance components carry no honest closed-form SE, so those cells stay NaN
    # rather than being filled with something that looks usable.
    assert np.isnan(summary.loc["tau2_tau_u", "se_ratio"])
    assert np.isnan(summary.loc["tau2_tau_u", "cov_95"])
    assert np.isnan(summary.loc["beta_0", "frac_at_zero"])
    assert 0.0 <= summary.loc["beta_0", "cov_95"] <= 1.0


# =============================================================================
# Statistical recovery
# =============================================================================

def test_a1a_recovers_truth_and_covers(v_diag, structure):
    """Check 1: the A1a shape -- intercept plus one variance component, diagonal V."""
    m_true, tau_true = 0.98, 0.65
    res = dsim.simulate_reml_fits(m_true, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                  {"tau_u": (structure["I"], tau_true)},
                                  n_sims=N_SIMS, seed=SEED, fit_kwargs=FAST_FIT,
                                  progress=False)
    summary = dsim.summarise_simulation(res)

    assert res.frac_converged == 1.0
    assert unbiased(summary.loc["beta_0"])
    # The model-based SE tracks the actual spread. Tolerance is wide because at
    # n_sims = 200 the empirical SD itself carries ~5% Monte-Carlo error.
    assert 0.85 < summary.loc["beta_0", "se_ratio"] < 1.15
    # Mild undercoverage is expected and is not a bug: se_beta conditions on the
    # fitted tau as if it were known (fit_reml's first Caveat), and the interval
    # uses a normal rather than a t quantile. Both bite hardest at small n. The
    # realised value sits around 0.92; the bound is set to catch an SE that is
    # actually broken (which reads ~0.6), not to police a few percent.
    assert 0.87 < summary.loc["beta_0", "cov_95"] <= 1.0
    # REML should not be badly biased downward on tau2 -- that is the whole reason
    # it is preferred over DerSimonian-Laird here.
    assert summary.loc["tau2_tau_u", "mean"] > 0.7 * tau_true ** 2


def test_zero_component_hits_the_boundary(v_diag, structure):
    """Check 2: with no true heterogeneity, tau_u should be detected as absent.

    The fixed-effect SE sqrt(1 / sum(1/v)) is the right comparison because with
    tau_u = 0 the random-effects weights collapse onto the fixed-effect weights.
    """
    res = dsim.simulate_reml_fits(0.0, np.ones((N_ROWS, 1)), np.diag(v_diag),
                                  {"tau_u": (structure["I"], 0.0)},
                                  n_sims=N_SIMS, seed=SEED, fit_kwargs=FAST_FIT,
                                  progress=False)
    summary = dsim.summarise_simulation(res)

    assert summary.loc["tau2_tau_u", "frac_at_zero"] > 0.4
    assert unbiased(summary.loc["beta_0"])

    se_fe = np.sqrt(1.0 / np.sum(1.0 / v_diag))
    np.testing.assert_allclose(summary.loc["beta_0", "mean_analytic_se"], se_fe,
                               rtol=0.15)


def test_identifiability_trap(structure, V_bb):
    """Check 3: the roadmap's A1c warning, with tau_d = 0 as the known truth.

    Data are generated with NO design effect, from the TRUE covariance
    ``diag(v_a) + Z V_bb Z'`` -- so the shared-b_j correlation between rows of the
    same design is really in the data. Both arms fit the same generated data; only
    what the fit is told differs:

      * told the full V_known    -> tau_d^2 should recover ~0;
      * told only diag(v1)       -> the correlation is real but unexplained, and the
        only thing in the model with that block structure is tau_d^2 Z Z', so it
        must land there.

    Generating and fitting with the same diagonal would be a different (and empty)
    experiment -- the data would have no shared-b_j correlation to misattribute.
    That is what ``fit_V_known`` is for.

    This is the result that justifies Route 1 over a diagonal fit, and the reason
    "do not fit A1c with a diagonal V" is in CLAUDE.md.
    """
    Z, I = structure["Z"], structure["I"]
    g = structure["design_codes"]
    v_a = np.full(N_ROWS, 0.0008)

    V_full = np.diag(v_a) + Z @ V_bb @ Z.T
    # The A1a-style diagonal: v1[i] = v_a[i] + V_bb[g[i], g[i]]. Same diagonal as
    # V_full by construction -- the ONLY difference is the discarded off-diagonals.
    v1 = v_a + np.diag(V_bb)[g]
    np.testing.assert_allclose(np.diag(V_full), v1)

    components = {"tau_d": (Z, 0.0), "tau_u": (I, 0.05)}
    common = dict(n_sims=N_SIMS, seed=SEED, fit_kwargs=FAST_FIT, progress=False)
    X = np.ones((N_ROWS, 1))

    correct = dsim.simulate_reml_fits(0.0, X, V_full, components, **common)
    trapped = dsim.simulate_reml_fits(0.0, X, V_full, components,
                                      fit_V_known=np.diag(v1), **common)
    assert not correct.misspecified and trapped.misspecified

    full = dsim.summarise_simulation(correct)
    diag = dsim.summarise_simulation(trapped)
    mean_vb = float(np.mean(np.diag(V_bb)))

    # Thresholds are set from the realised behaviour, which is stable across seeds:
    # correctly specified ~0.18 x mean_vb with ~55% of fits at the boundary;
    # misspecified ~0.52 x mean_vb with ~15% at the boundary; ratio 2.6-3.0.
    #
    # The correctly specified fit lands at 0.18 x mean_vb rather than 0 because a
    # variance component cannot go negative: with a true tau_d^2 of exactly 0 the
    # estimator sits on a boundary, so its MEAN is positive however good it is. The
    # honest reading of "recovers zero" is the boundary fraction, not the mean.
    assert full.loc["tau2_tau_d", "mean"] < 0.3 * mean_vb
    assert full.loc["tau2_tau_d", "frac_at_zero"] > 0.4

    # Without the covariance, record-set noise is promoted to a design effect, and
    # the bias runs in exactly one direction -- in favour of "the building matters
    # more". It recovers about half of mean_vb rather than all of it, because this
    # V_bb is ~0.5 correlated off-diagonal and only the part of the shared block
    # that looks like tau_d^2 Z Z' can be absorbed. The roadmap's
    # "tau_d^2 ~ mean(v^b)" is an order-of-magnitude smell test, and this is
    # consistent with it.
    assert diag.loc["tau2_tau_d", "mean"] > 2.2 * full.loc["tau2_tau_d", "mean"]
    assert diag.loc["tau2_tau_d", "mean"] > 0.4 * mean_vb
    # The sharper signal: misspecified, the component stops being detected as absent.
    assert diag.loc["tau2_tau_d", "frac_at_zero"] < 0.5 * full.loc["tau2_tau_d", "frac_at_zero"]


def test_crossed_recovery(structure, V_bb):
    """Check 4: A1c -- three well-separated components, all recovered.

    Only the ordering and the order of magnitude are asserted. At this size the site
    component is estimated from 6 sites, where the relative uncertainty on tau_s is
    ~1/sqrt(2*5) = 32% before anything else, so a tight tolerance would be testing
    the random seed rather than the estimator.
    """
    Z, S, I = structure["Z"], structure["S"], structure["I"]
    v_a = np.full(N_ROWS, 0.0008)
    V_full = np.diag(v_a) + Z @ V_bb @ Z.T

    components = {"tau_d": (Z, 0.20), "tau_s": (S, 0.10), "tau_u": (I, 0.05)}
    res = dsim.simulate_reml_fits(0.1, np.ones((N_ROWS, 1)), V_full, components,
                                  n_sims=N_SIMS, seed=SEED, fit_kwargs=FAST_FIT,
                                  progress=False)
    summary = dsim.summarise_simulation(res)

    assert unbiased(summary.loc["beta_0"])
    # The dominant component is identified as dominant -- the claim A1c exists to make.
    assert (summary.loc["tau2_tau_d", "mean"] > summary.loc["tau2_tau_s", "mean"]
            > summary.loc["tau2_tau_u", "mean"])
    for name, truth in [("tau_d", 0.04), ("tau_s", 0.01), ("tau_u", 0.0025)]:
        assert 0.3 * truth < summary.loc[f"tau2_{name}", "mean"] < 3.0 * truth


def test_regression_recovers_every_coefficient(structure, V_bb):
    """Check 5: the A3 shape -- a covariate matrix on top of the A1c components."""
    Z, S, I = structure["Z"], structure["S"], structure["I"]
    v_a = np.full(N_ROWS, 0.0008)
    V_full = np.diag(v_a) + Z @ V_bb @ Z.T

    # Centred and scaled, as the roadmap requires -- the simulation should be run on
    # the same design matrix the real fit will use.
    rng = np.random.default_rng(99)
    raw = rng.normal(size=(N_ROWS, 2))
    Xc = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    X = np.column_stack([np.ones(N_ROWS), Xc])
    beta_true = np.array([0.10, 0.05, -0.03])

    components = {"tau_d": (Z, 0.10), "tau_s": (S, 0.08), "tau_u": (I, 0.05)}
    res = dsim.simulate_reml_fits(beta_true, X, V_full, components, n_sims=N_SIMS,
                                  seed=SEED, fit_kwargs=FAST_FIT, progress=False)
    summary = dsim.summarise_simulation(res)

    for k in range(3):
        assert unbiased(summary.loc[f"beta_{k}"])
        assert 0.80 < summary.loc[f"beta_{k}", "se_ratio"] < 1.20
        # Same mild undercoverage as the A1a check -- see the comment there.
        assert 0.87 < summary.loc[f"beta_{k}", "cov_95"] <= 1.0
