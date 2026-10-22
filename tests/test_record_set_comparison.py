"""Tests for ``record_set_comparison`` (nb 034).

Two kinds, as in ``test_fragility_data_models.py``:

- **Synthetic tests** that need nothing on disk: set matching and classification, the
  predicted-fraction average, the expected-count fragility fit, the Firth logistic fit.
- **Real-data tests**, skipped cleanly when the inputs are absent: the KS score reproduces the
  optimiser's own scoring function on a real set, and shifting the SS arm leaves V_known and
  the fixed-set arms untouched.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import phd_project.scripts.record_set_comparison as rsc

try:
    from phd_project.config import config

    _CFG = config.load_config()
    _ORIG = config.ss_dir(_CFG, "results", "record_selection", "AvgSA_03__ks")
    _GCIM = Path(_CFG["proc_data"]["gcim_dists"]) / "gcim_dist_AvgSA_03.pickle"
    # The tests check the data that exist: the original study (__ks sets vs FEMA P695),
    # pinned rather than read from record_sets.active_study.
    _BOOT = config.bootstrap_roots(_CFG, "AvgSA_03__ks__vs__femap695")
    _HAVE_SETS = _ORIG.is_dir() and _GCIM.is_file()
except Exception:  # pragma: no cover - bare checkout
    _HAVE_SETS = False


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _fake_set(filenames, alphas, iml=0.5):
    """A minimal ensemble dict: just the metadata classify_sets reads."""
    meta = pd.DataFrame({"filename": filenames, "alpha": alphas,
                         "database": ["ESM"] * len(filenames)})
    recs = pd.concat({"metadata": meta}, axis=1)
    return {"recs": recs, "cond_iml": iml, "ks_passed": True}


# -----------------------------------------------------------------------------
# 1. set matching and classification
# -----------------------------------------------------------------------------

def test_match_set_key_absorbs_filename_rounding_but_not_neighbours():
    # 0.6049 is what the MSA ran; the selection stored it as cond_iml 0.6049 but the
    # filename says 0.605 -- the tolerance must match it, and only it.
    keys = [(3, 0.55), (3, 0.6049), (3, 0.65), (4, 0.6049)]
    assert rsc.match_set_key(3, 0.605, keys) == (3, 0.6049)
    assert rsc.match_set_key(3, 0.6049, keys) == (3, 0.6049)
    with pytest.raises(KeyError):
        rsc.match_set_key(3, 0.62, keys)            # nothing within tolerance
    with pytest.raises(KeyError):
        rsc.match_set_key(3, 0.6, [(3, 0.5998), (3, 0.6002)])   # ambiguous


def test_classify_sets_counts_overlap_and_status(monkeypatch):
    monkeypatch.setattr(rsc, "N_RECORDS", 3)
    a = [f"r{i}" for i in range(3)]
    orig = {(0, 0.5): _fake_set(a, [1.0, 2.0, 3.0]),
            (0, 0.6): _fake_set(a, [1.0, 2.0, 3.0]),
            (1, 0.5): _fake_set(a, [1.0, 2.0, 3.0])}
    opt = {(0, 0.5): _fake_set(a, [1.0, 2.0, 3.0]),                # unchanged
           (0, 0.6): _fake_set(["r0", "x1", "x2"], [1.0, 9, 9]),   # partly
           (1, 0.5): _fake_set(["y0", "y1", "y2"], [5, 6, 7])}     # fully
    out = rsc.classify_sets(orig, opt).set_index(["site", "iml"])
    assert out.loc[(0, 0.5), "status"] == "unchanged"
    assert out.loc[(0, 0.6), "status"] == "partly changed"
    assert out.loc[(0, 0.6), "n_common"] == 1
    assert out.loc[(1, 0.5), "status"] == "fully changed"


def test_classify_sets_raises_on_kept_record_with_new_alpha(monkeypatch):
    monkeypatch.setattr(rsc, "N_RECORDS", 2)
    orig = {(0, 0.5): _fake_set(["r0", "r1"], [1.0, 2.0])}
    opt = {(0, 0.5): _fake_set(["r0", "r9"], [1.5, 2.0])}
    with pytest.raises(ValueError):
        rsc.classify_sets(orig, opt)


def test_classify_sets_raises_on_different_coverage():
    with pytest.raises(KeyError):
        rsc.classify_sets({(0, 0.5): _fake_set(["a"], [1])}, {(0, 0.6): _fake_set(["a"], [1])})


# -----------------------------------------------------------------------------
# 2. KS score = the optimiser's own scoring function
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_SETS, reason="record sets / GCIM file not on disk")
def test_score_set_reproduces_the_optimisers_score_function():
    """score_set's J must equal gm_selection._score_ensemble -- the function the optimiser
    uses for its keep-best comparison -- driven from nb 031's GCIM file. This also proves the
    stored ks_bounds are the ones the optimiser rebuilds."""
    import phd_project.scripts.WP1_ground_motion_set.gm_selection as gms

    with open(_GCIM, "rb") as f:
        gcim = pickle.load(f)
    fp = _ORIG / "site_0__stripe_iml_00pt270__gm_selection.pickle"
    with open(fp, "rb") as f:
        ens = pickle.load(f)

    class _Imt:                         # _score_ensemble only reads .string
        def __init__(self, s):
            self.string = s

    ctx = {"conditioning_imt": _Imt(rsc.CONDITIONING_IM), "n_samples": rsc.N_RECORDS,
           "p_value": rsc.P_VALUE, "selection_imts": rsc.SELECTION_IMTS,
           "imt_weights": rsc.selection_weights()}
    key = next(k for k in gcim if k[0] == 0 and np.isclose(k[1], 0.27))
    expected = gms._score_ensemble(ens, 0, key[1], gcim, ctx)

    got = rsc.score_set(ens, rsc.selection_weights())
    assert got["J"] == pytest.approx(expected, abs=1e-12)
    # the split is exact by construction, and the penalty is ~0 for a set that passes
    assert got["wks"] + got["penalty"] == pytest.approx(got["J"], abs=1e-12)
    assert ens["ks_passed"] and got["penalty"] == pytest.approx(0.0, abs=1e-9)


def test_selection_weights_match_the_selector():
    w = rsc.selection_weights()
    assert len(w) == 22 and w.sum() == pytest.approx(1.0)
    assert w[rsc.SELECTION_IMS.index("RSD595")] == pytest.approx(0.25)
    assert np.allclose(np.delete(w, 0), 0.75 / 21)


# -----------------------------------------------------------------------------
# 3. predicted-fraction averaging
# -----------------------------------------------------------------------------

def test_predicted_fraction_is_the_mean_probability_and_order_free():
    X = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 2.0]])
    coef = np.array([0.3, 1.2])
    hand = np.mean([expit(0.3 - 1.2), expit(0.3), expit(0.3 + 2.4)])
    assert rsc.predicted_fraction(X, coef) == pytest.approx(hand, abs=1e-15)
    assert rsc.predicted_fraction(X[::-1], coef) == pytest.approx(hand, abs=1e-15)


# -----------------------------------------------------------------------------
# 4. the expected-count fragility fit
# -----------------------------------------------------------------------------

def test_expected_count_fit_equals_standes_on_integer_counts():
    from standes.fitting import lognorm_mle_fit

    imls = [0.31, 0.36, 0.4, 0.5, 0.55]
    z = [4, 7, 14, 23, 26]
    a = lognorm_mle_fit(imls, z, 30)
    b = rsc.lognorm_mle_fit_expected(imls, z, 30)
    assert b == pytest.approx(a, rel=1e-10)


def test_expected_count_fit_is_finite_on_fractional_counts():
    th, be = rsc.lognorm_mle_fit_expected([0.31, 0.36, 0.4, 0.5, 0.55],
                                          [4.3, 7.1, 13.6, 22.9, 26.4], 30)
    assert np.isfinite(th) and np.isfinite(be) and 0.3 < th < 0.6 and 0 < be < 1


# -----------------------------------------------------------------------------
# 5. Firth logistic regression
# -----------------------------------------------------------------------------

def test_firth_matches_ml_on_a_large_well_conditioned_sample():
    rng = np.random.default_rng(1)
    n = 2000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = (rng.random(n) < expit(X @ np.array([-0.2, 1.0, -0.5]))).astype(float)
    f = rsc.firth_logit(X, y)
    m = rsc.firth_logit(X, y, firth=False)
    assert f["converged"] and m["converged"]
    assert np.allclose(f["coef"], m["coef"], rtol=0.01, atol=0.01)


def test_firth_stays_finite_under_complete_separation():
    x = np.linspace(-2, 2, 40)
    X = np.column_stack([np.ones_like(x), x])
    y = (x > 0).astype(float)                       # perfectly separated
    f = rsc.firth_logit(X, y)
    assert f["converged"] and np.all(np.isfinite(f["coef"])) and abs(f["coef"][1]) < 20
    flags = rsc.separation_flags(X, y, f)
    assert flags["separation"]


# -----------------------------------------------------------------------------
# 6. shifting the SS arm
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_SETS, reason="bootstrap data not on disk")
def test_shift_ss_arm_moves_only_a_and_keeps_every_covariance():
    import phd_project.scripts.fragility_data_models as fm
    import phd_project.scripts.metaregression_analysis as mra

    th = fm.load_fragility_data(_BOOT, "AvgSA_03", quantity="theta", verbose=False)
    delta = pd.Series(np.linspace(-0.05, 0.05, len(th.a_hat)), index=th.a_hat.index)
    st = rsc.shift_ss_arm(th, delta)

    assert np.allclose(st.a_hat - th.a_hat, delta)
    assert np.allclose(mra.bootstrap_cov(st.a.to_numpy()), mra.bootstrap_cov(th.a.to_numpy()))
    for s in ("b", "c", "c_avg"):
        assert st.reps[s] is th.reps[s] and st.estimates[s] is th.estimates[s]
    # the contrast A0c is fitted on: same V, outcome moved by exactly delta
    c0, c1 = fm.make_contrast(th, "y0_avg"), fm.make_contrast(st, "y0_avg")
    assert np.allclose(c0.V, c1.V, rtol=0, atol=1e-14)
    assert np.allclose(c1.y_hat - c0.y_hat, delta.reindex(c0.y_hat.index))
    # the original container is untouched
    assert not np.allclose(st.a_hat, th.a_hat)


# -----------------------------------------------------------------------------
# 7. example-stripe pickers and the ECDF grid (Section 2 figures)
# -----------------------------------------------------------------------------

def test_example_pickers_choose_among_run_sets_only():
    # the biggest J drop and the widest Ds5-95 spread both sit on UNRUN sets here, so the
    # pickers must skip them and return the best RUN set
    summ = pd.DataFrame({"site": [0, 1, 2], "iml": [0.3, 0.4, 0.5],
                         "d_J": [-0.05, -0.30, -0.10], "used_in_msa": [True, False, True]})
    assert rsc.largest_improvement_key(summ) == (2, 0.5)

    ch = pd.DataFrame({"site": [0, 1, 2, 0], "iml": [0.3, 0.4, 0.5, 0.3],
                       "im": [rsc.LEAD_IM] * 3 + ["PGA"],
                       "r_sigma_opt": [1.2, 3.0, 1.9, 9.0], "r_sigma_orig": [2.5, 1.0, 1.1, 1.0],
                       "used_in_msa": [True, False, True, True]})
    assert rsc.widest_lead_dispersion_key(ch) == (2, 0.5)           # PGA's 9.0 ignored
    assert rsc.widest_lead_dispersion_key(ch, version="orig") == (0, 0.3)


def test_ecdf_xy_is_a_proper_step_cdf():
    x, y = rsc._ecdf_xy([3.0, 1.0, 2.0])
    assert np.array_equal(x, [1.0, 1.0, 2.0, 3.0])
    assert np.allclose(y, [0.0, 1 / 3, 2 / 3, 1.0])


_OPT = (config.ss_dir(_CFG, "results", "record_selection", "AvgSA_03__opt_ks")
        if _HAVE_SETS else None)


@pytest.mark.skipif(not (_HAVE_SETS and _OPT is not None and _OPT.is_dir()),
                    reason="record sets not on disk")
def test_plot_ecdf_grid_on_a_real_set():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = "site_0__stripe_iml_00pt270__gm_selection.pickle"
    sets = []
    for folder in (_ORIG, _OPT):
        with open(folder / name, "rb") as f:
            raw = pickle.load(f)
        sets.append({k: raw[k] for k in rsc._KEEP})
    o, n = sets

    fig = rsc.plot_ecdf_grid(o, n, "test")
    axs = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axs) == len(rsc.SELECTION_IMS)
    # the corner text carries the optimiser's own per-IM KS distances
    tgt = rsc.objective_kwargs(o, np.ones(22))["target_cdfs_list"]
    d_o = rsc.ks_distances(rsc.set_log_ims(o), tgt)
    d_n = rsc.ks_distances(rsc.set_log_ims(n), tgt)
    txt = [t.get_text() for t in axs[0].texts]
    assert f"D {d_o[0]:.3f} → {d_n[0]:.3f}" in txt
    plt.close(fig)

    # two sets with different GCIM targets must be refused, not silently drawn
    bad = dict(n, ks_bounds={k: v * 1.01 for k, v in n["ks_bounds"].items()})
    with pytest.raises(ValueError):
        rsc.plot_ecdf_grid(o, bad, "test")


# -----------------------------------------------------------------------------
# 8. the beta check: shifting a bias-corrected SS arm
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_SETS, reason="bootstrap data not on disk")
def test_shift_ss_arm_on_corrected_beta_moves_the_corrected_contrast_by_delta():
    """A0c_bc_hb fits the CORRECTED contrast. Shifting the SS arm must move its corrected
    point estimate by exactly delta and leave V and the bias correction itself unchanged."""
    import phd_project.scripts.fragility_data_models as fm

    be = fm.load_fragility_data(_BOOT, "AvgSA_03", quantity="beta", verbose=False)
    delta = pd.Series(np.linspace(-0.05, 0.05, len(be.a_hat)), index=be.a_hat.index)
    st = rsc.shift_ss_arm(be, delta)

    spec = fm.MODEL_SPECS[rsc.A0C_BC_HB]
    assert spec.corrected and spec.bootstrap_scheme == "hybrid"
    assert np.allclose(st.replicate_shift("a"), be.replicate_shift("a"))
    c0 = fm.make_contrast(be, spec.contrast, corrected=True)
    c1 = fm.make_contrast(st, spec.contrast, corrected=True)
    assert np.allclose(c0.V, c1.V, rtol=0, atol=1e-14)
    assert np.allclose(c1.y_hat - c0.y_hat, delta.reindex(c0.y_hat.index))


# -----------------------------------------------------------------------------
# 9. the worked logistic example (fit_row_logit, logit_coef_table, plot_row_logit)
# -----------------------------------------------------------------------------

def _synthetic_row(seed=3, imls=(0.3, 0.4, 0.5, 0.6, 0.7), T_star=0.4):
    """Records of one synthetic row in the long format row_shift expects, plus a shifted
    'optimised' version (longer durations). Collapse depends on IML and duration."""
    rng = np.random.default_rng(seed)
    frames = []
    for iml in imls:
        n = rsc.N_RECORDS
        ln_ds = rng.normal(np.log(10.0), 0.5, n)
        shape = rng.normal(0.2, 0.15, n)
        df = pd.DataFrame({"set_iml": iml, "ln_iml": np.log(iml), f"ln_{rsc.LEAD_IM}": ln_ds,
                           f"ln_SA({T_star})": np.log(iml) + shape})
        eta = -1.0 + 4.0 * (np.log(iml) - np.log(0.5)) + 1.2 * (ln_ds - np.log(10.0))
        df["y"] = (rng.random(n) < expit(eta)).astype(float)
        frames.append(df)
    train = pd.concat(frames, ignore_index=True)
    opt = train.drop(columns="y").copy()
    opt[f"ln_{rsc.LEAD_IM}"] += 0.2
    return train, opt, np.array(imls), T_star


def test_fit_row_logit_is_the_model_row_shift_uses():
    train, opt, imls, T = _synthetic_row()
    lf = rsc.fit_row_logit(train, rsc.DEFAULT_MODEL, T)
    r = rsc.row_shift(train, train.drop(columns="y"), opt, imls, rsc.DEFAULT_MODEL, T)
    assert lf["names"] == r["coef_names"]
    assert np.allclose(lf["fit"]["coef"], r["coef"], rtol=0, atol=1e-12)


def test_logit_coef_table_natural_units_give_the_same_linear_predictor():
    train, _, _, T = _synthetic_row()
    lf = rsc.fit_row_logit(train, rsc.DEFAULT_MODEL, T)
    tab = rsc.logit_coef_table(lf)
    eta_std = lf["design"](lf["P"]) @ lf["fit"]["coef"]
    X_nat = np.column_stack([np.ones(len(lf["P"])), lf["P"].to_numpy()])
    eta_nat = X_nat @ tab["coef_natural"].to_numpy()
    assert np.allclose(eta_std, eta_nat, atol=1e-10)
    # SEs are the sqrt of the covariance diagonal; slopes in natural units scale by 1/sd
    assert np.allclose(tab["se_std"], np.sqrt(np.diag(lf["fit"]["cov"])))
    assert np.allclose(tab["coef_natural"].iloc[1:], tab["coef_std"].iloc[1:] / lf["sd"].to_numpy())
    # the duration effect built into the data is recovered with the right sign
    assert tab.loc["ln_ds595", "coef_natural"] > 0 and np.isnan(tab.loc["intercept", "odds_ratio_x2"])


def test_plot_row_logit_draws_one_panel_per_stripe_plus_the_fragility():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, opt, imls, T = _synthetic_row()
    lf = rsc.fit_row_logit(train, rsc.DEFAULT_MODEL, T)
    fig = rsc.plot_row_logit(lf, train, opt, imls, "test", n_draws=200)
    assert len(fig.axes) == 1 + len(imls)
    plt.close(fig)


def test_lowest_score_key_picks_the_best_fitting_run_set():
    summ = pd.DataFrame({"site": [0, 1, 2], "iml": [0.3, 0.4, 0.5],
                         "J_opt": [0.09, 0.05, 0.07], "J_orig": [0.08, 0.2, 0.06],
                         "used_in_msa": [True, False, True]})
    assert rsc.lowest_score_key(summ) == (2, 0.5)            # (1, 0.4) is lower but unrun
    assert rsc.lowest_score_key(summ, version="orig") == (2, 0.5)


def test_plot_row_logit_fragility_curves_carry_step3s_shift():
    """The lognormal curves on the left panel are refitted to the predicted fractions with the
    same MLE as row_shift, so the Delta a printed in the panel title must be Step 3's value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, opt, imls, T = _synthetic_row()
    lf = rsc.fit_row_logit(train, rsc.DEFAULT_MODEL, T)
    r = rsc.row_shift(train, train.drop(columns="y"), opt, imls, rsc.DEFAULT_MODEL, T)
    fig = rsc.plot_row_logit(lf, train, opt, imls, "test", n_draws=100, n_curve_draws=20)
    title = fig.axes[0].get_title()
    assert f"Δa = {r['d_ln_theta']:+.4f}" in title
    assert f"Δln β = {r['d_ln_beta']:+.4f}" in title
    plt.close(fig)


def test_regression_summary_adds_slopes_to_the_shift_table_and_checks():
    """A3ab is read on A0c's terms plus its slopes: each slope gets a row in shift_table (its
    Delta divided by the ORIGINAL replicate SD) and a sign / CI-includes-0 qualitative check."""
    from types import SimpleNamespace

    import phd_project.scripts.fragility_data_models as fm

    rng = np.random.default_rng(0)
    names = ["intercept", "x1"]

    def fake_fit(b0, b1):
        reps = pd.DataFrame({"M_star": b0 + 0.1 * rng.standard_normal(400),
                             fm.beta_column("x1"): b1 + 0.1 * rng.standard_normal(400),
                             # taus vary a little so their SEs are not zero
                             "tau_d2": 0.01 + 0.001 * rng.random(400),
                             "tau_s2": 0.02 + 0.001 * rng.random(400),
                             "tau_u2": 0.001 * rng.random(400)})
        return SimpleNamespace(replicates=reps, m=b0, se_boot=float(reps.M_star.std(ddof=1)),
                               ci=lambda: (b0 - 0.2, b0 + 0.2), x_names=names,
                               beta=np.array([b0, b1]),
                               se_boot_beta=reps[["M_star", fm.beta_column("x1")]].std(ddof=1).to_numpy(),
                               tau2={"tau_d2": 0.01, "tau_s2": 0.02, "tau_u2": 0.0})

    s0, s1 = rsc.regression_summary(fake_fit(0.0, 0.5)), rsc.regression_summary(fake_fit(0.0, -0.5))
    assert s0["quantities"] == ["m", "tau_d", "tau_s", "tau_u", "x1"]
    tab = rsc.shift_table(s0, s1).set_index("quantity")
    assert np.isclose(tab.loc["x1", "delta_over_se"], -1.0 / s0["se_x1"])
    assert tab.loc["x1", "exceeds_threshold"] and not tab.loc["m", "exceeds_threshold"]
    q = rsc.qualitative_flags(s0, s1).set_index("check")
    assert q.loc["sign of x1", "changed"]
    assert not q.loc["CI of x1 includes 0", "changed"]       # 0.5 +/- 0.2 and -0.5 +/- 0.2
    # an A0c summary carries no slopes, so its table and checks are unchanged
    assert rsc.a0c_summary(fake_fit(0.0, 0.5))["slopes"] == []
