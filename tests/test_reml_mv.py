"""Tests for the bivariate (joint theta / beta) REML: ``mra.fit_reml_mv`` and its simulator.

Pure-array tests with no data on disk, which is why the estimator lives in
``mra`` and the simulator in ``dsim`` (the layering rule of
``fragility_data_models``). They cover, in the order of the task's acceptance
criteria (dump/task_joint_theta_beta_model.md):

- separability: with no theta-beta sampling covariance and no cross terms, the
  joint fit IS two univariate fits (the joint model's DL <-> REML regression test);
- positive semi-definiteness of every fitted T, including on data built to
  force a boundary;
- the analytic gradient against finite differences;
- recovery of known T, rho and V_cross from simulated data, and the bias that
  leaving V_cross out of the fit introduces.
"""

from __future__ import annotations

import numpy as np
import pytest

import phd_project.scripts.data_simulation as dsim
import phd_project.scripts.metaregression_analysis as mra


# --------------------------------------------------------------------------- #
# A small crossed study and a V_known with a known theta-beta sampling corr.  #
# --------------------------------------------------------------------------- #

def _study(n_sites: int = 30, n_designs: int = 8, seed: int = 0):
    """Row -> design / site factors for ``2 * n_sites`` rows (a 3s and a 5s per site).

    Crossed like the real study: the two structures at a site belong to
    different designs, and every design spans several sites.
    """
    sites = np.repeat(np.arange(n_sites), 2)
    storey = np.tile([0, 1], n_sites)
    half = n_designs // 2
    design = np.where(storey == 0, sites % half, half + sites % (n_designs - half))
    Z = mra.make_indicator(design, n_designs)
    S = mra.make_indicator(sites, n_sites)
    return Z, S


def _V_known(Z: np.ndarray, r_samp: float = 0.6, sd_t: float = 0.08,
             sd_b: float = 0.12, k: int = 4000, seed: int = 1) -> np.ndarray:
    """A stacked (2n x 2n) sampling covariance with a known per-row theta-beta corr.

    Built from SYNTHETIC REPLICATES (then np.cov), so it has the same
    provenance as the real one: a per-row noise pair with correlation
    ``r_samp``, plus a design-level noise pair shared by every row of a design
    (as a shared fixed-record-set arm would be), which makes the theta-beta
    block dense rather than diagonal.
    """
    rng = np.random.default_rng(seed)
    n, J = Z.shape
    C = np.array([[1.0, r_samp], [r_samp, 1.0]])
    R = np.linalg.cholesky(C)
    row = rng.standard_normal((n, k, 2)) @ R.T             # per-row pairs
    des = rng.standard_normal((J, k, 2)) @ R.T             # per-design pairs
    noise_t = sd_t * (0.8 * row[..., 0] + 0.6 * (Z @ des[..., 0]))
    noise_b = sd_b * (0.8 * row[..., 1] + 0.6 * (Z @ des[..., 1]))
    return mra.bootstrap_cov(np.vstack([noise_t, noise_b]))


def _blockdiag(*blocks: np.ndarray) -> np.ndarray:
    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    out = np.zeros((rows, cols))
    r = c = 0
    for b in blocks:
        out[r:r + b.shape[0], c:c + b.shape[1]] = b
        r, c = r + b.shape[0], c + b.shape[1]
    return out


def _levels(Z, S, which):
    n = Z.shape[0]
    all_levels = {"design": Z, "site": S, "row": np.eye(n)}
    return {lev: all_levels[lev] for lev in which}


# --------------------------------------------------------------------------- #
# Gradient                                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("which, correlated", [
    (("row",), ("row",)),
    (("design", "site", "row"), ("site",)),
    (("design", "site", "row"), ("design", "site", "row")),
])
def test_analytic_gradient_matches_finite_differences(which, correlated):
    """The analytic REML gradient is what makes L-BFGS-B viable, so it is checked
    entry by entry against central differences of the objective, at random
    interior points, for every kind of level (correlated, independent)."""
    Z, S = _study()
    n = Z.shape[0]
    V = _V_known(Z)
    factors = _levels(Z, S, which)
    Gs = {lev: M @ M.T for lev, M in factors.items()}
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    rng = np.random.default_rng(5)
    y = rng.normal(0, 0.15, 2 * n)
    levels = list(Gs)
    for _ in range(3):
        x = rng.normal(0, 0.08, mra._mv_n_params(levels, correlated))
        g = mra.reml_grad_mv(x, y, X, V, Gs, levels, correlated)
        h = 1e-6
        fd = np.array([(mra.reml_nll_mv(x + h * e, y, X, V, Gs, levels, correlated)
                        - mra.reml_nll_mv(x - h * e, y, X, V, Gs, levels, correlated))
                       / (2 * h) for e in np.eye(x.size)])
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


# --------------------------------------------------------------------------- #
# Separability -- the joint model's regression test                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("which", [("row",), ("design", "site", "row")])
@pytest.mark.parametrize("with_covariates", [False, True])
def test_separable_problem_reproduces_two_univariate_fits(which, with_covariates):
    """Acceptance 2. With V_cross = 0 and every cross term fixed at 0, Sigma is
    block-diagonal and X is block-diagonal, so log|Sigma|, log|X'Sigma^-1 X| and
    the quadratic form all split into a theta part plus a beta part: the joint
    REML objective IS the sum of the two univariate ones. Its optimum must
    therefore be the two univariate optima, to optimiser tolerance. Run with
    DIFFERENT covariate sets per outcome, which is allowed."""
    Z, S = _study()
    n = Z.shape[0]
    V = _V_known(Z)
    V[:n, n:] = 0.0
    V[n:, :n] = 0.0
    factors = _levels(Z, S, which)
    T_true = {lev: np.diag([0.006, 0.012]) for lev in which}
    rng = np.random.default_rng(11)
    if with_covariates:
        Xt = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        Xb = np.column_stack([np.ones(n), rng.normal(size=(n, 1))])
    else:
        Xt, Xb = np.ones((n, 1)), np.ones((n, 1))
    X = _blockdiag(Xt, Xb)
    y = dsim.simulate_y_mv(rng, np.r_[0.05, np.zeros(Xt.shape[1] - 1),
                                      -0.02, np.zeros(Xb.shape[1] - 1)],
                           X, V, {lev: (factors[lev], T_true[lev]) for lev in which})

    Gs = {lev: M @ M.T for lev, M in factors.items()}
    joint = mra.fit_reml_mv(y, X, V, Gs, correlated=())
    for q, (h, Xq, cols) in enumerate(((slice(0, n), Xt, joint["theta_cols"]),
                                       (slice(n, 2 * n), Xb, joint["beta_cols"]))):
        uni = mra.fit_reml(y[h], Xq, V[h, h], list(Gs.values()), names=list(Gs))
        np.testing.assert_allclose(joint["beta"][cols], uni["beta"], atol=1e-6)
        for j, lev in enumerate(Gs):
            assert joint["T"][lev][q, q] == pytest.approx(uni["tau2"][j], rel=1e-4,
                                                          abs=1e-7)


def test_nelder_mead_and_lbfgs_reach_the_same_optimum():
    """L-BFGS-B (analytic gradient) is the default; Nelder-Mead is kept to cross-check it."""
    Z, S = _study()
    n = Z.shape[0]
    V = _V_known(Z)
    rng = np.random.default_rng(3)
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    T_u = np.array([[0.01, -0.004], [-0.004, 0.02]])
    y = dsim.simulate_y_mv(rng, [0.0, 0.0], X, V, {"row": (np.eye(n), T_u)})
    a = mra.fit_reml_mv(y, X, V, {"row": np.eye(n)}, ("row",))
    b = mra.fit_reml_mv(y, X, V, {"row": np.eye(n)}, ("row",), method="Nelder-Mead")
    assert a["nll"] == pytest.approx(b["nll"], abs=1e-6)
    np.testing.assert_allclose(a["T"]["row"], b["T"]["row"], atol=1e-5)
    assert a["converged"]


# --------------------------------------------------------------------------- #
# PSD, including on purpose-built boundaries                                   #
# --------------------------------------------------------------------------- #

def _assert_psd_and_rho_ok(fit):
    for lev, T in fit["T"].items():
        assert np.linalg.eigvalsh(T).min() >= -1e-12, (lev, T)
        r = fit["rho"][lev]
        assert np.isnan(r) or -1.0 <= r <= 1.0
    assert np.isnan(fit["rho_tot"]) or -1.0 <= fit["rho_tot"] <= 1.0


@pytest.mark.parametrize("T_site", [
    np.diag([0.02, 0.0]),                                  # beta variance at zero
    np.array([[0.02, 0.02], [0.02, 0.02]]),                # rho = +1
    np.array([[0.02, -0.03], [-0.03, 0.045]]),             # rho = -1
])
def test_fitted_T_is_psd_on_boundary_cases(T_site):
    """Acceptance 4. Every fitted T is PSD and every rho is in [-1, 1] (or NaN
    when undefined), on data whose TRUE T sits on each kind of boundary. The
    Cholesky parameterisation guarantees this; the test is that nothing
    downstream (zero-snapping, rho, rho_tot) breaks it."""
    Z, S = _study()
    n = Z.shape[0]
    V = _V_known(Z)
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    comps = {"design": (Z, np.diag([0.003, 0.003])), "site": (S, T_site),
             "row": (np.eye(n), np.diag([0.002, 0.0]))}
    Gs = {lev: M @ M.T for lev, (M, _) in comps.items()}
    rng = np.random.default_rng(21)
    for _ in range(4):
        y = dsim.simulate_y_mv(rng, [0.0, 0.0], X, V, comps)
        _assert_psd_and_rho_ok(mra.fit_reml_mv(y, X, V, Gs, tuple(Gs), n_starts=2))


def test_X_must_be_block_diagonal():
    """A column spanning both outcomes would be a shared fixed effect -- a
    different model -- so it is refused rather than misreported."""
    n = 10
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    X = np.column_stack([X, np.ones(2 * n)])
    with pytest.raises(ValueError, match="block-diagonal"):
        mra.fit_reml_mv(np.zeros(2 * n), X, np.eye(2 * n), {"row": np.eye(n)})


def test_correlated_level_must_exist():
    n = 10
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    with pytest.raises(ValueError, match="not among the levels"):
        mra.fit_reml_mv(np.zeros(2 * n), X, np.eye(2 * n), {"row": np.eye(n)},
                        correlated=("site",))


def test_simulator_refuses_a_non_psd_T():
    with pytest.raises(ValueError, match="not PSD"):
        dsim.simulate_y_mv(np.random.default_rng(0), [0, 0],
                           _blockdiag(np.ones((4, 1)), np.ones((4, 1))), np.eye(8),
                           {"row": (np.eye(4), np.array([[0.01, 0.02], [0.02, 0.01]]))})


# --------------------------------------------------------------------------- #
# Recovery, and the cost of leaving V_cross out                                #
# --------------------------------------------------------------------------- #

def test_recovery_of_T_rho_and_m_and_the_bias_from_omitting_V_cross():
    """Acceptance 5. Data simulated with a known T_u (rho = -0.4) and a STRONG
    known sampling correlation (r_samp = +0.6, comparable in size to T):

    * the correctly specified fit recovers m, both variances and rho to within
      simulation error;
    * a fit that is NOT told about V_cross (its off-diagonal blocks zeroed)
      has nowhere to put that sampling covariance except the true-correction
      covariance, so its rho is dragged towards +0.6 -- the contamination the
      A8 section of analysis_models.md warns about, made measurable.
    """
    Z, _ = _study(n_sites=30)
    n = Z.shape[0]
    V = _V_known(Z, r_samp=0.6, sd_t=0.09, sd_b=0.12)
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    T_u = np.array([[0.01, -0.4 * 0.1 * np.sqrt(0.02)],
                    [-0.4 * 0.1 * np.sqrt(0.02), 0.02]])
    comps = {"row": (np.eye(n), T_u)}
    kw = dict(beta_true=[0.03, -0.01], X=X, V_known=V, components=comps,
              correlated=("row",), n_sims=200, seed=7, progress=False,
              fit_kwargs={"n_starts": 1})
    good = dsim.simulate_reml_mv_fits(**kw)
    s = dsim.summarise_simulation_mv(good)
    for p in ("beta_0", "beta_1", "T_row_tt", "T_row_bb", "rho_row"):
        row = s.loc[p]
        tol = max(3 * row["mcse"], 0.10 * abs(row["truth"]), 0.03 if p == "rho_row" else 0)
        assert abs(row["bias"]) <= tol, (p, row.to_dict())

    V_no_cross = V.copy()
    V_no_cross[:n, n:] = 0.0
    V_no_cross[n:, :n] = 0.0
    bad = dsim.simulate_reml_mv_fits(**kw, fit_V_known=V_no_cross)
    assert bad.misspecified
    sb = dsim.summarise_simulation_mv(bad)
    shift = sb.loc["rho_row", "mean"] - s.loc["rho_row", "mean"]
    assert shift > 5 * sb.loc["rho_row", "mcse"], (shift, sb.loc["rho_row"].to_dict())


# --------------------------------------------------------------------------- #
# The hybrid bootstrap, joint version (dump/plan_hybrid_bootstrap.md)         #
# --------------------------------------------------------------------------- #
#
# The record-only bootstrap inflates every replicate T by the sampling
# covariance, which drags the replicate rho towards the sampling correlation
# and makes its interval far too narrow. The hybrid pseudo-data are
# X beta_hat + new (theta, beta) effect pairs drawn from the fitted T's + the
# centred stacked replicate cloud. Tested here at array level; the univariate
# counterparts are in test_data_simulation.py.


def test_refactored_mv_simulator_draws_are_bit_identical():
    """Moving the pair draw into ``draw_effects_mv`` changed no number.

    The reference is the loop ``simulate_y_mv`` contained before, written out,
    including a RANK-ONE T (|rho| = 1), the case the eigen-root exists for.
    """
    Z, S = _study(n_sites=10, n_designs=4)
    n = Z.shape[0]
    V = _V_known(Z, k=600)
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    beta = np.array([0.1, -0.2])
    T_site = np.array([[0.01, 0.01], [0.01, 0.01]])            # rank one
    T_row = np.array([[0.01, -0.004], [-0.004, 0.02]])
    comps = {"site": (S, T_site), "row": (np.eye(n), T_row)}

    rng = np.random.default_rng(42)
    want = X @ beta
    for M, T in ((S, T_site), (np.eye(n), T_row)):
        D = rng.standard_normal((M.shape[1], 2)) @ dsim._psd_root(T).T
        want = want + np.concatenate([M @ D[:, 0], M @ D[:, 1]])
    want = want + dsim._known_error_sampler(V)(rng)

    got, parts = dsim.simulate_y_mv(np.random.default_rng(42), beta, X, V, comps,
                                    return_parts=True)
    assert np.array_equal(got, want)
    assert set(parts) == {"site", "row", "eps"}
    # Rank one: the theta and beta site effects are the same numbers.
    np.testing.assert_allclose(parts["site"]["effects"][:, 0],
                               parts["site"]["effects"][:, 1], atol=1e-12)


def _entrywise_z(S_hat: np.ndarray, Sigma: np.ndarray, N: int) -> float:
    """Largest covariance-entry error in Monte Carlo SEs (see test_data_simulation)."""
    d = np.diag(Sigma)
    return float(np.max(np.abs(S_hat - Sigma) / np.sqrt((np.outer(d, d) + Sigma ** 2) / N)))


@pytest.mark.parametrize("T_site", [
    np.array([[0.010, -0.005], [-0.005, 0.020]]),              # interior
    np.array([[0.010, 0.010], [0.010, 0.010]]),                # rho = +1 (rank one)
    np.array([[0.010, 0.0], [0.0, 0.0]]),                      # beta variance at zero
])
def test_hybrid_pseudo_data_mv_has_the_fitted_covariance(T_site):
    """Acceptance 4, joint: Var(y*) = sum_l T_l kron G_l + V_known.

    20 000 pseudo-replicates (a 400-column stacked cloud cycled 50 times, a
    fresh effect draw each), no fits. The Kronecker layout is built here by
    hand with ``np.kron``, independently of the code under test, so a theta /
    beta mix-up in the stacking would show as a covariance in the wrong block.

    Run for an interior T and for the two kinds of boundary a fitted T can sit
    on, since those are drawn through the eigen-root.
    """
    Z, S = _study(n_sites=10, n_designs=4)
    n = Z.shape[0]
    V = _V_known(Z, k=2000)
    rng = np.random.default_rng(3)
    draw = dsim._known_error_sampler(V)
    cloud = np.column_stack([draw(rng) for _ in range(400)])
    cycled = np.tile(cloud, 50)
    N = cycled.shape[1]

    T_row = np.array([[0.012, -0.004], [-0.004, 0.008]])
    comps = {"site": (S, T_site), "row": (np.eye(n), T_row)}
    fitted = np.concatenate([np.full(n, 0.1), np.full(n, -0.2)])
    Y_star = dsim.hybrid_pseudo_data_mv(cycled, fitted, comps, seed=7)
    assert Y_star.shape == (2 * n, N)

    Sigma = np.cov(cycled) + np.kron(T_site, S @ S.T) + np.kron(T_row, np.eye(n))
    assert _entrywise_z(np.cov(Y_star), Sigma, N) < 5

    # Seeded, and replicate r does not depend on n_boot.
    assert np.array_equal(Y_star[:, :30], dsim.hybrid_pseudo_data_mv(
        cycled, fitted, comps, seed=7, n_boot=30))
    with pytest.raises(ValueError, match="stack two outcomes"):
        dsim.hybrid_pseudo_data_mv(cycled[:-1], fitted[:-1], comps, seed=7)


def _joint_study(n_sites: int):
    """X, V_known and components for an A8a-type model on ``2 * n_sites`` rows.

    A row-level T with rho = -0.4, against a sampling covariance whose per-row
    theta-beta correlation is +0.6: the two have opposite signs on purpose, so
    a replicate rho contaminated by sampling covariance moves visibly.
    """
    Z, _ = _study(n_sites=n_sites, n_designs=8)
    n = Z.shape[0]
    V = _V_known(Z, r_samp=0.6)
    X = _blockdiag(np.ones((n, 1)), np.ones((n, 1)))
    rho = -0.4
    T = np.array([[0.020, rho * np.sqrt(0.020 * 0.030)],
                  [rho * np.sqrt(0.020 * 0.030), 0.030]])
    return X, V, {"row": (np.eye(n), T)}, n


def _close(row, truth: float) -> bool:
    """On the estimate: within 3 MCSE, or within 10% of the truth, whichever is looser."""
    return abs(row["z"]) < 3 or abs(row["excess"]) < 0.10 * abs(truth)


def test_record_only_bootstrap_inflates_T_and_the_hybrid_does_not():
    """Acceptance 5, joint. Fast version: 20 studies x 40 replicates x 2 schemes.

    Record-only: each replicate T is too large, by an amount of the order of
    the mean sampling variance of that outcome (between half of it and 1.25x:
    V_known has a dense design block here, part of which a single row-level T
    cannot absorb), and its rho is pulled from -0.4 towards the +0.6 sampling
    correlation. Hybrid: the replicate T's and rho sit on the estimates.
    """
    X, V, comps, n = _joint_study(n_sites=20)
    res = dsim.simulate_bootstrap_schemes_mv([0.1, -0.2], X, V, comps, ("row",),
                                             n_sims=20, n_boot=40, seed=11,
                                             progress=False)
    s = dsim.summarise_bootstrap_schemes(res)
    print("\n" + s.to_string(float_format="%.5f"))

    v_bar = {"tt": float(np.mean(np.diag(V)[:n])), "bb": float(np.mean(np.diag(V)[n:]))}
    for key in ("tt", "bb"):
        rec, hyb = s.loc[(f"T_row_{key}", "record")], s.loc[(f"T_row_{key}", "hybrid")]
        assert rec["z"] > 5, rec.to_dict()
        assert 0.5 * v_bar[key] < rec["excess"] < 1.25 * v_bar[key], (rec.to_dict(), v_bar)
        assert _close(hyb, res.truth[f"T_row_{key}"]), hyb.to_dict()

    rec, hyb = s.loc[("rho_tot", "record")], s.loc[("rho_tot", "hybrid")]
    assert rec["excess"] > 5 * rec["mcse"], rec.to_dict()      # pulled towards +0.6
    assert abs(hyb["excess"]) < 3 * hyb["mcse"] or abs(hyb["excess"]) < 0.05, hyb.to_dict()
    assert hyb["mean_width"] > rec["mean_width"]


@pytest.mark.slow
def test_hybrid_interval_covers_T_and_rho():
    """Acceptance 6, joint: coverage of the 95% percentile intervals.

    150 studies of 60 rows, 120 replicates per scheme (36 000 fits, roughly a
    quarter of an hour). The binomial SE of a coverage near 0.95 is 0.018 at
    150 studies, and a percentile interval for a variance under-covers a
    little by construction (see the univariate slow test), so the tolerance is
    [0.88, 0.995] for the two variances and for rho_tot. The record-only
    interval for rho must do clearly worse.
    """
    X, V, comps, n = _joint_study(n_sites=30)
    res = dsim.simulate_bootstrap_schemes_mv([0.1, -0.2], X, V, comps, ("row",),
                                             n_sims=150, n_boot=120, seed=11)
    s = dsim.summarise_bootstrap_schemes(res)
    print("\n" + s.to_string(float_format="%.5f"))

    for p in ("T_row_tt", "T_row_bb", "rho_tot"):
        hyb = s.loc[(p, "hybrid")]
        assert 0.88 <= hyb["coverage"] <= 0.995, (p, hyb.to_dict())
        assert _close(hyb, res.truth[p]) or p == "rho_tot", (p, hyb.to_dict())
    assert (s.loc[("rho_tot", "record"), "coverage"]
            < s.loc[("rho_tot", "hybrid"), "coverage"] - 0.1)
