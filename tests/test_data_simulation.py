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


# =============================================================================
# The hybrid bootstrap: pseudo-data and the scheme-comparison study
# =============================================================================
#
# dump/plan_hybrid_bootstrap.md. The record-only outer bootstrap refits each
# record-resampling replicate as it is; that replicate carries TWO rounds of
# sampling noise while V_known describes one, so every replicate tau^2 is
# inflated by about the mean sampling variance. The hybrid bootstrap refits
# pseudo-data with one round: X beta_hat + new effects drawn from the fitted
# tau's + the centred replicate cloud. The tests below check the pieces
# (centring, the effect draw, the covariance of the pseudo-data, seeding) and
# then the claim itself, on fake data with a known truth.


def test_refactored_simulator_draws_are_bit_identical(structure, V_bb):
    """Acceptance 1: moving the effect draw into ``draw_effects`` changed no number.

    The reference below is the loop ``simulate_y`` contained before the
    refactor, written out. Exact equality, not approximate: the same generator
    calls in the same order, and the same order of floating-point additions,
    must give the same bits -- which is what keeps every seeded recovery test
    above meaning what it meant.
    """
    Z, S, I = structure["Z"], structure["S"], structure["I"]
    V = np.diag(np.full(N_ROWS, 0.001)) + Z @ V_bb @ Z.T
    X = np.column_stack([np.ones(N_ROWS), np.linspace(-1, 1, N_ROWS)])
    beta = np.array([0.1, -0.05])
    components = {"tau_d": (Z, 0.10), "tau_s": (S, 0.08), "tau_u": (I, 0.05)}

    rng = np.random.default_rng(99)
    want = X @ beta
    for M, tau in ((Z, 0.10), (S, 0.08), (I, 0.05)):
        want = want + M @ (tau * rng.standard_normal(M.shape[1]))
    want = want + dsim._known_error_sampler(V)(rng)

    got, parts = dsim.simulate_y(np.random.default_rng(99), beta, X, V, components,
                                 return_parts=True)
    assert np.array_equal(got, want)
    # return_parts still decomposes y exactly as before.
    assert set(parts) == {"tau_d", "tau_s", "tau_u", "eps"}

    # draw_effects must not write into the vector it is handed: the bootstrap
    # passes the same X beta_hat for every replicate.
    base = X @ beta
    kept = base.copy()
    dsim.draw_effects(np.random.default_rng(0), [Z, S, I], [0.1, 0.08, 0.05], base=base)
    assert np.array_equal(base, kept)


def test_centred_noise_has_zero_row_means_and_ignores_a_row_shift():
    """Acceptance 3: e = y - ybar averages zero per row, and a per-row constant cancels.

    The second half is the `_bc` case at array level: a bias-corrected contrast
    is the uncorrected one plus one constant per row, so it must give the same
    noise. In floating point the two agree to rounding, not bit for bit.
    """
    rng = np.random.default_rng(3)
    Y = 0.3 + 0.1 * rng.standard_normal((N_ROWS, 250))
    E = dsim.centre_replicates(Y)
    assert E.shape == Y.shape
    assert np.abs(E.mean(axis=1)).max() < 1e-14

    shift = rng.uniform(-0.05, 0.05, N_ROWS)
    np.testing.assert_allclose(dsim.centre_replicates(Y + shift[:, None]), E,
                               rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match="rows are studies"):
        dsim.centre_replicates(Y[:, 0])


def _entrywise_z(S_hat: np.ndarray, Sigma: np.ndarray, N: int) -> float:
    """Largest |sample - expected| covariance entry, in Monte Carlo standard errors.

    For N draws the sampling variance of a covariance entry (i, j) is about
    ``(Sigma_ii Sigma_jj + Sigma_ij^2) / N`` (the Wishart result). Scaling by it
    lets one threshold serve the large diagonal entries and the small
    off-diagonal ones alike.
    """
    d = np.diag(Sigma)
    se = np.sqrt((np.outer(d, d) + Sigma ** 2) / N)
    return float(np.max(np.abs(S_hat - Sigma) / se))


def test_hybrid_pseudo_data_has_the_fitted_covariance(structure, V_bb):
    """Acceptance 4: Var(y*) = sum_l tau_l^2 G_l + V_known, the fitted model.

    This is the whole reason the scheme works, so it is tested directly, with
    no fit involved: a 400-column cloud is cycled 50 times, each of the 20 000
    pseudo-replicates gets a FRESH effect draw, and the sample covariance of
    the result is compared entry by entry with the model's.

    "V_known" here is the covariance of the cycled cloud itself. That is the
    point of taking the noise from the cloud: whatever dependence it has (the
    dense design block below) arrives in y* without being modelled.
    """
    Z, S, I = structure["Z"], structure["S"], structure["I"]
    rng = np.random.default_rng(5)
    V = np.diag(0.002 + 0.002 * rng.random(N_ROWS)) + Z @ V_bb @ Z.T
    draw = dsim._known_error_sampler(V)
    cloud = 0.2 + np.column_stack([draw(rng) for _ in range(400)])
    cycled = np.tile(cloud, 50)
    N = cycled.shape[1]

    fitted = np.linspace(-0.1, 0.1, N_ROWS)
    taus = {"tau_d2": 0.07, "tau_s2": 0.09, "tau_u2": 0.04}
    comps = {"tau_d2": (Z, taus["tau_d2"]), "tau_s2": (S, taus["tau_s2"]),
             "tau_u2": (I, taus["tau_u2"])}
    Y_star = dsim.hybrid_pseudo_data(cycled, fitted, comps, seed=11)
    assert Y_star.shape == (N_ROWS, N)

    Gs, _ = dsim.build_component_gs(comps)
    Sigma = np.cov(cycled) + sum(t ** 2 * G for t, G in zip(taus.values(), Gs))
    z = _entrywise_z(np.cov(Y_star), Sigma, N)
    assert z < 5, z

    # ... and it is centred on the fitted values: the noise averages exactly
    # zero and the effects average zero to Monte Carlo error.
    se_mean = np.sqrt(np.diag(Sigma) / N)
    assert np.max(np.abs(Y_star.mean(axis=1) - fitted) / se_mean) < 5

    # A component at ZERO draws nothing: the covariance loses exactly that term.
    comps0 = comps | {"tau_d2": (Z, 0.0)}
    Sigma0 = Sigma - taus["tau_d2"] ** 2 * Gs[0]
    assert _entrywise_z(np.cov(dsim.hybrid_pseudo_data(cycled, fitted, comps0, seed=11)),
                        Sigma0, N) < 5


def test_hybrid_pseudo_data_is_seeded_and_does_not_depend_on_n_boot(structure):
    """Acceptance 7 at array level: same seed, same matrix; and replicate r is replicate r.

    The second property is why the noise is centred on the WHOLE cloud and the
    generator is used replicate by replicate: a 250-replicate pilot is then a
    prefix of the 5000-replicate run, not a different experiment.
    """
    Z, I = structure["Z"], structure["I"]
    rng = np.random.default_rng(8)
    cloud = rng.normal(0, 0.05, (N_ROWS, 60))
    fitted = np.full(N_ROWS, 0.1)
    comps = {"tau_d2": (Z, 0.05), "tau_u2": (I, 0.03)}

    a = dsim.hybrid_pseudo_data(cloud, fitted, comps, seed=1)
    assert np.array_equal(a, dsim.hybrid_pseudo_data(cloud, fitted, comps, seed=1))
    assert not np.array_equal(a, dsim.hybrid_pseudo_data(cloud, fitted, comps, seed=2))
    assert np.array_equal(a[:, :20],
                          dsim.hybrid_pseudo_data(cloud, fitted, comps, seed=1, n_boot=20))

    with pytest.raises(ValueError, match="n_boot must be between"):
        dsim.hybrid_pseudo_data(cloud, fitted, comps, seed=1, n_boot=61)
    with pytest.raises(ValueError, match="fitted has"):
        dsim.hybrid_pseudo_data(cloud, fitted[:-1], comps, seed=1)


def _single_level_study(n_sites: int):
    """X, V_known and components for an A0b-type model on ``2 * n_sites`` rows.

    One row-level component with tau^2 = 0.02, against sampling variances of
    about 0.01. The two are chosen close on purpose: the record-only inflation
    (one extra sampling variance) is then half the size of tau^2 itself and
    cannot be missed, while tau^2 stays far enough above zero (about three of
    its own standard errors) that the boundary plays no part in the test.
    """
    n = 2 * n_sites
    rng = np.random.default_rng(21)
    v = 0.008 + 0.004 * rng.random(n)
    return np.ones((n, 1)), np.diag(v), {"tau_u2": (np.eye(n), np.sqrt(0.02))}


def _close_to_estimate(row, truth: float) -> bool:
    """Is the replicate mean on the full-sample estimate?

    Within 3 Monte Carlo SEs, or within 10% of the true value, whichever is
    looser. The second clause is there for the same reason ``unbiased`` above
    does not use z alone: an MCSE shrinks with the size of the run, so a pure
    z-threshold eventually fails on a bias too small to matter. (A small
    positive bias is real here: a replicate variance cannot go below zero.)
    """
    return abs(row["z"]) < 3 or abs(row["excess"]) < 0.10 * truth


def test_record_only_bootstrap_inflates_tau2_and_the_hybrid_does_not():
    """Acceptance 5, univariate: the problem and the fix, on data with a known truth.

    For each simulated study the replicate-mean tau^2 is compared with that
    study's own full-sample estimate.

    * record-only: too large by about the mean sampling variance (within 25%),
      and by far more than Monte Carlo error;
    * hybrid: on the estimate.

    Fast version (30 studies x 50 replicates x 2 schemes); the coverage study
    is the slow test below.
    """
    X, V, comps = _single_level_study(n_sites=20)
    res = dsim.simulate_bootstrap_schemes(0.1, X, V, comps, n_sims=30, n_boot=50,
                                          seed=SEED, progress=False)
    s = dsim.summarise_bootstrap_schemes(res)
    print("\n" + s.to_string(float_format="%.5f"))

    rec, hyb = s.loc[("tau_u2", "record")], s.loc[("tau_u2", "hybrid")]
    assert rec["z"] > 5, rec.to_dict()
    assert rec["excess"] == pytest.approx(res.mean_v, rel=0.25), (rec.to_dict(), res.mean_v)
    assert _close_to_estimate(hyb, res.truth["tau_u2"]), hyb.to_dict()
    # The hybrid interval is the wider one: it also carries the effect redraw.
    assert hyb["mean_width"] > rec["mean_width"]

    # Seeded: the whole study is reproduced by (seed, n_sims, n_boot).
    again = dsim.simulate_bootstrap_schemes(0.1, X, V, comps, n_sims=3, n_boot=10,
                                            seed=SEED, progress=False)
    once = dsim.simulate_bootstrap_schemes(0.1, X, V, comps, n_sims=3, n_boot=10,
                                           seed=SEED, progress=False)
    pd.testing.assert_frame_equal(again.draws, once.draws)

    with pytest.raises(ValueError, match="unknown bootstrap scheme"):
        dsim.simulate_bootstrap_schemes(0.1, X, V, comps, n_sims=1, n_boot=5,
                                        schemes=("record", "parametric"))


@pytest.mark.slow
def test_hybrid_interval_covers_tau2_and_the_record_only_one_does_not():
    """Acceptance 6, univariate: coverage of the 95% percentile interval for tau^2.

    250 studies of 60 rows, 150 replicates per scheme (75 000 fits, about a
    quarter of an hour).

    WHAT "CONSISTENT WITH 0.95" MEANS HERE. A percentile interval for a
    variance is only first-order accurate, because the estimate is skewed. For
    a plain variance on nu degrees of freedom the parametric percentile
    interval covers at about 0.934 for nu = 39 and 0.946 for nu = 119 (from the
    chi-square distribution; nearly all the misses are on one side). So about
    0.94 is expected at 60 rows, before Monte Carlo error, whose binomial SE at
    250 studies is 0.014. The tolerance is therefore [0.90, 0.99].

    The record-only interval is centred one sampling variance too high, so it
    must cover far less often: at least 0.2 less.
    """
    X, V, comps = _single_level_study(n_sites=30)
    res = dsim.simulate_bootstrap_schemes(0.1, X, V, comps, n_sims=250, n_boot=150,
                                          seed=SEED)
    s = dsim.summarise_bootstrap_schemes(res)
    print("\n" + s.to_string(float_format="%.5f"))

    rec, hyb = s.loc[("tau_u2", "record")], s.loc[("tau_u2", "hybrid")]
    assert 0.90 <= hyb["coverage"] <= 0.99, hyb.to_dict()
    assert rec["coverage"] <= hyb["coverage"] - 0.2, (rec.to_dict(), hyb.to_dict())
    assert _close_to_estimate(hyb, res.truth["tau_u2"]), hyb.to_dict()


@pytest.mark.slow
def test_hybrid_bootstrap_on_a_crossed_model(structure, V_bb):
    """The same study on the model the hybrid scheme is actually used for.

    Design, site and row components with a DENSE design block in V_known, as
    in A0c / A1c. Smaller than the single-level study (100 studies x 100
    replicates), because each fit has three variance parameters.

    Asserted on the TOTAL between-row variance, which is the well-identified
    quantity; the per-component rows are printed. With 40 rows the design and
    row components are often on their zero boundary, where a bootstrap interval
    is one-sided -- that is reported (``frac_zero``), not asserted.
    """
    Z, S, I = structure["Z"], structure["S"], structure["I"]
    rng = np.random.default_rng(31)
    V = np.diag(0.002 + 0.002 * rng.random(N_ROWS)) + Z @ V_bb @ Z.T
    comps = {"tau_d2": (Z, 0.06), "tau_s2": (S, 0.10), "tau_u2": (I, 0.05)}
    res = dsim.simulate_bootstrap_schemes(0.1, np.ones((N_ROWS, 1)), V, comps,
                                          n_sims=100, n_boot=100, seed=SEED)
    s = dsim.summarise_bootstrap_schemes(res)
    print("\n" + s.to_string(float_format="%.5f"))

    rec, hyb = s.loc[("T_sq", "record")], s.loc[("T_sq", "hybrid")]
    assert rec["z"] > 5, rec.to_dict()
    assert _close_to_estimate(hyb, res.truth["T_sq"]), hyb.to_dict()
    assert 0.88 <= hyb["coverage"] <= 0.99, hyb.to_dict()
