"""Tests for ``fragility_data_models`` -- the A-series data model.

Two kinds of test here:

- **Structural tests** that need no data on disk, built on small synthetic
  containers. They cover the registry invariants, the scope logic, the
  label-based expansion and the refusal to overwrite a fit.
- **Equivalence tests** against the real pipeline output, skipped cleanly when
  that output is absent (the pattern ``test_pipeline_regression.py`` uses).
  These assert that the container reproduces, term for term, the hand-written
  setup of notebook 073 cells 4/5/7 -- so the refactor can be shown not to have
  moved a single number.

The old cell 4/5/7 idiom is reconstructed inline in
``test_container_matches_the_hand_written_notebook_setup``. That is the ONLY
place it is allowed to survive, and it exists precisely so it can be deleted
from the notebook with evidence rather than hope.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import phd_project.scripts.fragility_data_models as fm
import phd_project.scripts.metaregression_analysis as mra

try:
    from phd_project.config import config

    _CFG = config.load_config()
    _ROOT = _CFG["proc_data"]["bootstrapping"]
    _IM_TAG = "AvgSA_03"
    _HAVE_DATA = fm.dataset_csv_path(_ROOT, _IM_TAG).is_file()
except Exception:  # pragma: no cover - config missing in a bare checkout
    _ROOT, _IM_TAG, _HAVE_DATA = None, "AvgSA_03", False

needs_data = pytest.mark.skipif(
    not _HAVE_DATA,
    reason="needs the nb-070 bootstrap output and the nb-072 dataset")


# --------------------------------------------------------------------------- #
# Structural tests -- no data required                                        #
# --------------------------------------------------------------------------- #

def test_registry_is_self_consistent():
    """Every symbol, scope and contrast refers to an arm that actually exists.

    Cheap, but it is the check that stops a typo in ``ARMS`` from surfacing as a
    confusing KeyError three notebooks away.
    """
    assert set(fm.SYMBOLS) == set(fm.ARMS)
    assert set(fm.SCOPE_OF_ARM) == set(fm.ARMS)
    assert set(fm.CORE_ARMS) <= set(fm.ARMS)
    assert set(fm.OPTIONAL_ARMS) <= set(fm.ARMS)
    assert not (set(fm.CORE_ARMS) & set(fm.OPTIONAL_ARMS))

    for left, right, scope, _ in fm._CONTRAST_SPECS.values():
        assert left in fm.ARM_OF_SYMBOL
        assert right in fm.ARM_OF_SYMBOL
        assert scope in {"row", "group"}

    for spec in fm.MODEL_SPECS.values():
        assert spec.contrast in fm._CONTRAST_SPECS
        assert set(spec.estimators) <= {"DL", "REML"}
        assert set(spec.components) <= {"design", "site"}


def test_no_arm_prefix_shadows_another():
    """nb 072 selects dataset columns by prefix, so this must hold.

    If one prefix were a prefix of another, a ``startswith`` selection would
    match the longer name twice and the dataset would silently gain duplicate
    columns.
    """
    for p in fm.ARMS:
        for q in fm.ARMS:
            if p != q:
                assert not p.startswith(q)


def _toy_data(k: int = 200, seed: int = 0) -> fm.FragilityData:
    """A tiny but structurally honest container: 3 sites x 2 storeys, 3 designs.

    Deliberately crossed -- the 3s and 5s structures at a site belong to
    different design groups -- so that the expansion and the indicator matrices
    are exercised rather than trivially satisfied.
    """
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({
        "site": [0, 0, 1, 1, 2, 2],
        "n_storeys": [3, 5, 3, 5, 3, 5],
        "structure_id": [0, 1, 0, 2, 0, 2],
        "tag": [f"t{i}" for i in range(6)],
        "design_group_id": ["g3_00", "g5_00", "g3_00", "g5_01", "g3_00", "g5_01"],
        "n_sites_in_group": [3, 1, 3, 2, 3, 2],
        "is_representative": [True, True, False, True, False, False],
    }).set_index(["site", "n_storeys"]).sort_index()
    groups = pd.Index(sorted(rows["design_group_id"].unique()),
                      name="design_group_id")

    a = pd.DataFrame(rng.normal(0.0, 0.1, (len(rows), k)), index=rows.index)
    b = pd.DataFrame(rng.normal(0.1, 0.1, (len(groups), k)), index=groups)
    c = pd.DataFrame(rng.normal(0.2, 0.1, (len(groups), k)), index=groups)
    reps = {"a": a, "b": b, "c": c, "c_ub": None, "c_avg": None}
    estimates = {"a": a.mean(axis=1), "b": b.mean(axis=1), "c": c.mean(axis=1),
            "c_ub": None, "c_avg": None}
    return fm.FragilityData(quantity="theta", im_tag="TEST", rows=rows,
                            groups=groups, reps=reps, estimates=estimates,
                            estimates_bc=dict(estimates))


def test_scope_and_shapes():
    d = _toy_data()
    assert d.n == 6 and d.J == 3 and d.n_sites == 3
    assert d.scope_of("a") == "row"
    assert d.scope_of("b") == d.scope_of("c") == "group"
    assert d.Z.shape == (6, 3)
    assert d.S.shape == (6, 3)
    # exactly one 1 per row in each indicator
    assert np.all(d.Z.sum(axis=1) == 1)
    assert np.all(d.S.sum(axis=1) == 1)
    assert d.available == ("b", "c", "a")


def test_expand_is_label_based_not_positional():
    """Reordering the group index must not change the expanded result.

    This is the single most likely silent bug in the whole pipeline: the order
    designs appear in the by_group bootstrap need not match the order rows
    appear in the by_site one. A positional expansion would pass every shape
    check and quietly attach the wrong design to every row.
    """
    d = _toy_data()
    expanded = d.expand(d.hat("b"))
    shuffled = d.hat("b").iloc[::-1]
    np.testing.assert_allclose(d.expand(shuffled).to_numpy(),
                               expanded.to_numpy())
    # and it really did map g[i] -> the right label
    for idx, label in zip(d.rows.index, d.rows["design_group_id"]):
        assert expanded.loc[idx] == pytest.approx(d.hat("b").loc[label])


def test_roadmap_identity_holds_on_toy_data():
    """y0 == y1 + y2[g[i]], exactly, on estimates and on every replicate.

    Holds because y2 is constant within a design group. It is the sharpest
    available test that Z, the label expansion and the replicate alignment are
    all correct at once.
    """
    d = _toy_data()
    y0, y1, y2 = (fm.make_contrast(d, n) for n in ("y0", "y1", "y2"))
    np.testing.assert_allclose(y0.y_hat.to_numpy(),
                               (y1.y_hat + d.expand(y2.y_hat)).to_numpy())
    np.testing.assert_allclose(y0.y.to_numpy(),
                               (y1.y + d.expand(y2.y)).to_numpy())


def test_group_scope_contrast_stays_at_group_scope():
    """y2 has one row per design and must not be replicated up to row scope.

    Replicating it to the row count would invent information that does not
    exist and shrink SE(m2) for nothing (analysis_models.md section 1(b)).
    """
    d = _toy_data()
    y2 = fm.make_contrast(d, "y2")
    assert y2.scope == "group"
    assert y2.n == d.J == 3
    assert fm.make_contrast(d, "y0").n == d.n == 6


def test_v_is_the_diagonal_of_V():
    d = _toy_data()
    con = fm.make_contrast(d, "y1")
    np.testing.assert_allclose(con.v.to_numpy(), np.diag(con.V))
    # and V really is the sample covariance of the contrast replicates
    np.testing.assert_allclose(con.v.to_numpy(),
                               con.y.var(axis=1, ddof=1).to_numpy())


def test_missing_optional_arm_raises_a_useful_error():
    d = _toy_data()
    with pytest.raises(KeyError, match="ida_avg"):
        d.rep("c_avg")
    with pytest.raises(KeyError):
        fm.make_contrast(d, "y0_avg")


def test_registry_refuses_to_overwrite_a_fit():
    """The direct fix for the silent rebinding of nb 073."""
    d = _toy_data()
    fits = fm.ModelRegistry(d)
    fits.fit("A0a")
    with pytest.raises(ValueError, match="already been fitted"):
        fits.fit("A0a")
    fits.fit("A0a", overwrite=True)          # explicit is fine


def test_dl_refuses_multiple_variance_components():
    """DL is one moment equation and cannot identify two components."""
    d = _toy_data()
    spec = fm.ModelSpec("X", "y1", ("DL",), components=("design",))
    with pytest.raises(ValueError, match="single moment equation"):
        fm.fit_dl(fm.make_contrast(d, "y1"), spec)


def test_prediction_interval_is_wider_than_the_confidence_interval():
    """The PI answers a different question and must be the wider of the two.

    Borenstein Ch. 17: the CI covers the MEAN effect, the PI covers where a NEW
    study lands, so the PI carries tau^2 as well as Var(M).
    """
    lo, hi = fm.prediction_interval(M=0.1, T_sq=0.04, V_M=0.001, n=50)
    assert lo < 0.1 < hi
    ci_half = 1.96 * np.sqrt(0.001)
    assert (hi - lo) / 2 > ci_half


def test_prediction_interval_needs_enough_studies():
    with pytest.raises(ValueError, match="at least 3"):
        fm.prediction_interval(M=0.0, T_sq=0.1, V_M=0.01, n=2)


# --------------------------------------------------------------------------- #
# Equivalence tests against the real pipeline output                          #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_theta():
    return fm.load_fragility_data(_ROOT, _IM_TAG, quantity="theta",
                                  verbose=False, check=False)


@needs_data
def test_container_matches_the_hand_written_notebook_setup(real_theta):
    """The container reproduces nb 073 cells 4/5/7, term for term.

    The old idiom is reconstructed inline below -- the only place it is allowed
    to survive. Note ``check_like=True``: the container sorts rows by
    ``(site, n_storeys)`` whereas nb 073 took whatever order the CSV happened to
    give. DL is a sum and so is invariant to a row permutation, but REML's
    ``V_known`` and ``Gs`` are NOT invariant unless all three are permuted
    together -- which is exactly the bug class the container exists to prevent.
    So compare up to label alignment, never positionally.
    """
    d = real_theta

    # --- old cell 4 + 5 + 7, for the site-specific arm ---------------------
    at_old = np.log(mra.reformat_bootstrap_df(
        mra.load_saved_bootstrap(_ROOT, "site_msa", ("theta",), None, _IM_TAG,
                                 "by_site")["theta"]))
    pd.testing.assert_frame_equal(d.a, at_old.reindex(d.a.index),
                                  check_like=True, check_names=False)

    # --- the same for a group-scope arm -----------------------------------
    bt_old = np.log(mra.reformat_bootstrap_df(
        mra.load_saved_bootstrap(_ROOT, "msa_femap695", ("theta",), None, _IM_TAG,
                                 "by_group")["theta"])).droplevel("n_storeys")
    pd.testing.assert_frame_equal(d.b, bt_old.reindex(d.b.index),
                                  check_like=True, check_names=False)

    # --- point estimates: the dataset must agree with the estimate files ---
    # This is the audit that justifies preferring the dataset CSV.
    fm._check_against_estimate_files(d, _ROOT)


@needs_data
def test_real_data_passes_every_structural_check(real_theta):
    report = fm.check_fragility_data(real_theta, root=_ROOT)
    assert "ok" in report


@needs_data
def test_full_study_dimensions(real_theta):
    """CLAUDE.md section 1: 120 rows, 60 sites, 51 design groups."""
    d = real_theta
    assert (d.n, d.n_sites, d.J) == (120, 60, 51)
    assert sorted(d.rows.index.get_level_values("n_storeys").unique()) == [3, 5]


@needs_data
def test_V_bb_is_dense(real_theta):
    """The designs share one record set, so V_bb must NOT be diagonal.

    If the mean off-diagonal correlation were ~0 the replicates did not share
    record indices, and the whole argument for carrying a dense V_bb -- and for
    preferring the bootstrap SE over the model-based one -- would be vacuous.
    """
    stats = mra.check_V_bb(real_theta.V_bb, n_replicates=real_theta.k,
                           verbose=False)
    assert stats["mean_offdiag_corr"] > 0.05


@needs_data
def test_bootstrap_se_exceeds_model_se_for_the_DL_fit(real_theta):
    """The finding that motivates the whole V_known machinery.

    DL assumes a diagonal sampling covariance. Because the fixed-set arms share
    a record set, that assumption is false and the model-based SE comes out far
    too small. The outer bootstrap carries the covariance and must therefore
    give a materially LARGER standard error.
    """
    fits = fm.ModelRegistry(real_theta)
    fits.fit("A0a", n_boot=200)
    dl = fits["A0a", "DL"]
    assert dl.se_boot > 2 * dl.se_beta[0]


@needs_data
def test_beta_scale_needs_no_model_code_changes():
    """Roadmap A12-A16: the same ladder on the dispersion, one argument changed.

    If this test ever needs a change to the model layer, the design has failed.
    """
    beta = fm.load_fragility_data(_ROOT, _IM_TAG, quantity="beta",
                                  verbose=False, check=False)
    fits = fm.ModelRegistry(beta)
    ids = ("A0a", "A1a", "A2")
    for model_id in ids:
        fits.fit(model_id)
    table = fits.table()
    # One row per (model, estimator). Counted from the specs rather than
    # hard-coded, so dropping DL from a rung (A2 became REML-only) is not a
    # failure of the model layer.
    assert len(table) == sum(len(fm.MODEL_SPECS[m].estimators) for m in ids)
    assert (table["quantity"] == "beta").all()


@needs_data
def test_sampling_se_matches_the_bootstrap_for_both_estimators(real_theta):
    """sqrt(w' V w) must reproduce the bootstrap SD of M*, for DL and for REML.

    This is the like-for-like comparison: both propagate the SAME sampling
    covariance through the SAME weights, so they can only disagree if the
    bootstrap is not carrying the covariance the fit was handed.

    It must NOT be compared against ``se_beta``, which also contains tau^2 and
    is therefore a superpopulation quantity answering a different question
    (Gelman & Hill Ch. 21.2). Tolerance is loose because the two use different
    replicate counts -- V from all of them, the refits from n_boot -- so they
    agree only to Monte-Carlo error, ~1/sqrt(2 n_boot).
    """
    fits = fm.ModelRegistry(real_theta)
    fits.fit("A0a", n_boot=500)
    for estimator in ("DL", "REML"):
        fit_obj = fits["A0a", estimator]
        assert fit_obj.se_sampling == pytest.approx(fit_obj.se_boot, rel=0.10), (
            f"{estimator}: sqrt(w'Vw)={fit_obj.se_sampling:.5f} vs "
            f"bootstrap={fit_obj.se_boot:.5f}")


@needs_data
def test_full_V_gls_is_more_efficient_than_a_diagonal_fit(real_theta):
    """The full-V weights beat near-equal weights on the same replicates.

    Ignoring the shared-record covariance does not merely misreport the
    precision, it discards precision: DL weights every row almost equally,
    while GLS down-weights rows that share an arm and can take them negative.
    """
    fits = fm.ModelRegistry(real_theta)
    fits.fit("A0a", n_boot=500)
    assert fits["A0a", "REML"].se_sampling < fits["A0a", "DL"].se_sampling
    # and the GLS weights really do go negative, which is the mechanism
    assert fits["A0a", "REML"].weights.min() < 0
    assert fits["A0a", "DL"].weights.min() > 0


# --------------------------------------------------------------------------- #
# Producer side -- the helpers nb 072 now imports                             #
# --------------------------------------------------------------------------- #

def test_design_group_label_matches_the_folder_convention():
    """The label is the join key between the dataset and the by_group frames.

    It is also the folder name under wp1_design_groups/, so a change here
    silently breaks the mapping from rows to design-level bootstrap results.
    """
    assert fm.design_group_label(3, 0) == "group_3s_00"
    assert fm.design_group_label(5, 7) == "group_5s_07"
    assert fm.design_group_label(5, 25) == "group_5s_25"


@needs_data
def test_build_design_factors_reproduces_the_dataset_factors(real_theta):
    """The producer helper must agree with what is actually saved in the dataset.

    This is the join that would otherwise be able to drift: nb 072 builds
    structure_id and design_group_id, and FragilityData maps rows onto V_bb
    through them. Rebuilding them from nb 051's table and comparing to the
    saved dataset checks the producer and the consumer still agree.
    """
    designs = fm.build_design_factors(_CFG["proc_data"]["unique_structural_designs_csv"])
    got = (designs.set_index(["site", "storeys"])[["structure_id", "design_group_id"]]
           .sort_index())
    want = real_theta.rows[["structure_id", "design_group_id"]].sort_index()
    got.index.names = want.index.names
    pd.testing.assert_frame_equal(got.reindex(want.index), want)


@needs_data
def test_order_dataset_columns_is_stable_and_prefix_safe(real_theta):
    """Factors first, then arm blocks in ARMS order, with no duplicates."""
    df = real_theta.rows.reset_index()
    ordered = fm.order_dataset_columns(df)
    assert list(ordered.columns[:len(fm.DATASET_FACTORS)]) == list(fm.DATASET_FACTORS)
    assert len(set(ordered.columns)) == len(ordered.columns)
    # arm blocks appear in registry order
    seen = [p for p in fm.ARMS
            for c in ordered.columns if c.startswith(f"{p}_")]
    first_of = {p: seen.index(p) for p in dict.fromkeys(seen)}
    assert list(first_of) == [p for p in fm.ARMS if p in first_of]


@needs_data
def test_arm_column_block_column_names_match_the_schema():
    """The block must lay its columns out under DATASET_ARM_COLUMNS exactly.

    That schema is what FragilityData reads the dataset by, so producer and
    consumer are checked against the same constant here.
    """
    block = fm.arm_column_block(_ROOT, "msa_ss", fm.ARMS["msa_ss"], 3, _IM_TAG)
    assert block is not None
    cols, parts = block
    expected = {t.format(arm="msa_ss", q=q)
                for t in fm.DATASET_ARM_COLUMNS for q in ("theta", "beta")}
    expected.add("msa_ss_n_obs")
    assert set(cols) == expected
    # the intermediates the notebook asserts on are all present
    assert set(parts["quantities"]) == {"theta", "beta"}
    for q in ("theta", "beta"):
        assert {"cloud", "stat", "bias", "var_nat", "var_log"} <= set(parts["quantities"][q])


# --------------------------------------------------------------------------- #
# Heterogeneity statistics -- mra helpers and the REML wiring                 #
# --------------------------------------------------------------------------- #

def _studies(heterogeneous: bool, k: int = 12, seed: int = 3):
    """Synthetic effect sizes and variances, with or without real heterogeneity."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.002, 0.02, k)
    spread = 0.3 if heterogeneous else 0.0
    y = 0.1 + rng.normal(0.0, spread, k) + rng.normal(0.0, np.sqrt(v)) * (
        1.0 if heterogeneous else 0.2)
    return pd.Series(v), pd.Series(y)


@pytest.mark.parametrize("heterogeneous", [True, False])
def test_I_sq_from_tau_reproduces_the_Q_based_I_sq_for_DL(heterogeneous):
    """T^2/(s^2 + T^2) with DL's T^2 is exactly (Q - df)/Q.

    Because s^2 = df / C and T^2_DL = (Q - df) / C. The non-heterogeneous case
    has Q < df, so T^2 truncates to zero and both forms must give 0 - checking
    that the truncation is handled identically, not just the interior.
    """
    v, y = _studies(heterogeneous)
    w = mra.get_fe_weights(v)
    T_sq = mra.compute_Tsquared(w, y, len(y))
    s_sq = mra.compute_typical_within_variance(w)
    if not heterogeneous:
        assert T_sq == 0.0
    assert mra.compute_I_sq_from_tau(T_sq, s_sq) == pytest.approx(
        mra.compute_I_sq(w, y), abs=1e-10)


def test_typical_within_variance_matches_the_higgins_thompson_formula():
    v, _ = _studies(True)
    w = mra.get_fe_weights(v).to_numpy()
    k = len(w)
    direct = (k - 1) * w.sum() / (w.sum() ** 2 - (w ** 2).sum())
    assert mra.compute_typical_within_variance(w) == pytest.approx(direct)
    assert mra.compute_typical_within_variance(w) == pytest.approx(
        (k - 1) / mra.compute_C(w))


def test_Q_pvalue_is_the_chi_square_tail():
    from scipy.stats import chi2
    assert mra.compute_Q_pvalue(30.0, 20) == pytest.approx(chi2.sf(30.0, 20))
    assert mra.compute_Q_pvalue(0.0, 5) == pytest.approx(1.0)


def test_heterogeneity_stats_is_backward_compatible():
    """Without T_sq, every pre-existing key keeps its old value and meaning."""
    v, y = _studies(True)
    stats = mra.compute_heterogeneity_stats(v, y)
    w = mra.get_fe_weights(v)
    assert stats["Q_df"] == pytest.approx(stats["Q"] - (len(y) - 1))  # the EXCESS
    assert stats["df"] == len(y) - 1
    assert stats["T_sq"] == pytest.approx(mra.compute_Tsquared(w, y, len(y)))
    assert stats["I_sq"] == pytest.approx(mra.compute_I_sq(w, y))


def test_heterogeneity_stats_with_a_supplied_T_sq():
    """Q, df and p do not depend on T_sq; T and I_sq do."""
    v, y = _studies(True)
    dl = mra.compute_heterogeneity_stats(v, y)
    other = mra.compute_heterogeneity_stats(v, y, T_sq=2 * dl["T_sq"])
    for key in ("Q", "df", "Q_df", "p", "s_sq"):
        assert other[key] == pytest.approx(dl[key])
    assert other["T_sq"] == pytest.approx(2 * dl["T_sq"])
    assert other["I_sq"] > dl["I_sq"]


@needs_data
def test_reml_fits_now_report_heterogeneity(real_theta):
    """REML het is present, shares Q/df/p with DL, and its T_sq is the fit total."""
    fits = fm.ModelRegistry(real_theta)
    fits.fit("A0a")
    dl, reml = fits["A0a", "DL"], fits["A0a", "REML"]
    assert reml.het is not None
    for key in ("Q", "df", "p", "s_sq"):
        assert reml.het[key] == pytest.approx(dl.het[key])
    assert reml.het["T_sq"] == pytest.approx(sum(reml.tau2.values()))
    assert reml.T_sq == pytest.approx(reml.het["T_sq"])
    assert 0.0 <= reml.het["I_sq"] <= 100.0
    assert reml.het["df"] == reml.n - 1


# --------------------------------------------------------------------------- #
# Forest plot -- PI band and style keywords                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def toy_fits():
    import matplotlib
    matplotlib.use("Agg")
    fits = fm.ModelRegistry(_toy_data())
    fits.fit("A0a", n_boot=50)
    return fits


def _spans(ax):
    """The axvspan patches on an Axes, keyed by legend label."""
    return {p.get_label(): p for p in ax.patches}


def test_forest_draws_the_pi_band_behind_the_ci_band(toy_fits):
    import matplotlib.pyplot as plt
    ax = toy_fits.forest("A0a")
    spans = _spans(ax)
    pi, ci = spans["95% PI"], spans["95% CI"]
    lo, hi = toy_fits["A0a"].prediction_interval()
    x0 = pi.get_x()
    assert x0 == pytest.approx(np.exp(lo))
    assert x0 + pi.get_width() == pytest.approx(np.exp(hi))
    assert pi.get_zorder() < ci.get_zorder()
    # the PI is the wider of the two
    assert pi.get_width() > ci.get_width()
    plt.close("all")


def test_forest_bands_can_be_switched_off(toy_fits):
    import matplotlib.pyplot as plt
    ax = toy_fits.forest("A0a", show_pi=False, show_ci=False)
    assert not _spans(ax)
    plt.close("all")


def test_forest_style_keywords_reach_the_artists(toy_fits):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    ax = toy_fits.forest(
        "A0a", "DL", color="tab:purple",
        point_kws={"color": "tab:green", "markersize": 6, "marker": "s"},
        pi_kws={"color": "tab:blue", "alpha": 0.2},
        summary_kws={"ls": ":", "lw": 2.5, "label": "mean"})
    points = next(l for l in ax.lines if l.get_label() == "per study")
    assert points.get_marker() == "s"
    assert points.get_markersize() == 6
    assert to_rgba(points.get_color()) == to_rgba("tab:green")   # point_kws wins
    # color= still recoloured the intervals
    assert to_rgba(ax.collections[0].get_color()[0], alpha=1) == to_rgba("tab:purple")
    pi = _spans(ax)["95% PI"]
    assert pi.get_alpha() == pytest.approx(0.2)
    summary = next(l for l in ax.lines if l.get_label() == "mean")
    assert summary.get_linestyle() == ":"
    assert summary.get_linewidth() == 2.5
    plt.close("all")


def test_forest_accepts_long_and_short_aliases(toy_fits):
    """The defaults use `lw`; passing `linewidth` must replace, not collide."""
    import matplotlib.pyplot as plt
    ax = toy_fits.forest("A0a", ref_kws={"linewidth": 3.0, "linestyle": "-."},
                         interval_kws={"linewidth": 2.0})
    ref = next(l for l in ax.lines if l.get_label() == "no difference")
    assert ref.get_linewidth() == 3.0
    assert ref.get_linestyle() == "-."
    plt.close("all")


# --------------------------------------------------------------------------- #
# Bias-corrected fits: the `corrected` flag and the `_bc` specs               #
# --------------------------------------------------------------------------- #
#
# The toy container above sets `estimates_bc = estimates` and carries no bias,
# so on it every correction is a no-op and every test below would pass for the
# wrong reason. `_toy_data_bc` builds clouds whose position relative to the
# point estimate is KNOWN, laid out the way nb 070 lays out the real arms:
#
#     mean(cloud) = hat + s + b          hat_bc = hat - b
#
# with b a per-unit bias, s = 0 for the two MSA arms (a binomial MLE is its own
# plug-in) and s = ln sqrt(21/22) for the IDA arm (a moment fit with Bessel's
# correction, measured against the plug-in; nb 070 section 2.1).

_S_IDA = float(np.log(np.sqrt(21 / 22)))


def _toy_data_bc(k: int = 400, seed: int = 1, quantity: str = "beta",
                 correct: bool = True) -> tuple[fm.FragilityData, dict]:
    """Toy container with known biases. Returns ``(data, truth)``.

    ``truth`` holds the per-arm ``b`` and ``s`` the clouds were built with, so a
    test can check the code recovers them, rather than checking the code
    against itself. ``correct=False`` imitates theta: the bias is measured but
    corrected by zero, so ``hat_bc == hat``.
    """
    base = _toy_data(k=k, seed=seed)
    rng = np.random.default_rng(seed + 100)
    # theta's plug-in is a geometric mean, which already weights each record
    # 1/n, so no arm has an offset there (verified on the real data: s = 0 on
    # every theta arm). Only beta's moment-fit arm does.
    s = ({"a": 0.0, "b": 0.0, "c": _S_IDA} if quantity == "beta"
         else {"a": 0.0, "b": 0.0, "c": 0.0})
    reps, est, est_bc, bias, truth = {}, {}, {}, {}, {}
    for sym in ("a", "b", "c"):
        cloud = base.reps[sym]
        b = pd.Series(rng.uniform(-0.04, -0.005, len(cloud)), index=cloud.index)
        hat = cloud.mean(axis=1) - s[sym] - b
        reps[sym], est[sym], bias[sym] = cloud, hat, b
        est_bc[sym] = hat - b if correct else hat.copy()
        truth[sym] = {"b": b if correct else b * 0.0, "s": s[sym]}
    for sym in ("c_ub", "c_avg"):
        reps[sym] = est[sym] = est_bc[sym] = bias[sym] = None
    data = fm.FragilityData(quantity=quantity, im_tag="TEST", rows=base.rows,
                            groups=base.groups, reps=reps, estimates=est,
                            estimates_bc=est_bc, bias=bias)
    return data, truth


def test_spec_table_obeys_one_id_one_contrast():
    """The import-time check passes on the real table, and the eight `_bc`
    models the task names exist, each next to its partner and flagged."""
    fm._check_model_specs(fm.MODEL_SPECS)
    for mid in ("A0c", "A2", "A2_avg", "A2_ub", "A3a", "A3ac", "A3ad", "A3ae"):
        bc = fm.MODEL_SPECS[mid + "_bc"]
        assert bc.corrected and bc.model_id == mid + "_bc"
        assert not fm.MODEL_SPECS[mid].corrected
    # the copy-paste ids are fixed: key and id agree everywhere
    assert all(k == s.model_id for k, s in fm.MODEL_SPECS.items())


@pytest.mark.parametrize("bad, match", [
    # the id says corrected, the flag does not
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",)),
      "A2_bc": fm.ModelSpec("A2_bc", "y2", ("REML",))}, "must be"),
    # the flag says corrected, the id does not
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",), corrected=True)}, "must be"),
    # a _bc spec that has drifted from its partner
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",)),
      "A2_bc": fm.ModelSpec("A2_bc", "y2", ("DL", "REML"), corrected=True)},
     "estimators"),
    # a _bc spec with no partner
    ({"A9_bc": fm.ModelSpec("A9_bc", "y2", ("REML",), corrected=True)}, "partner"),
    # key and model_id disagree (the old A3b/A3c bug)
    ({"A3b": fm.ModelSpec("A3a", "y0", ("REML",))}, "model_id"),
])
def test_spec_check_rejects_a_broken_table(bad, match):
    with pytest.raises(ValueError, match=match):
        fm._check_model_specs(bad)


def test_uncorrected_path_is_untouched_on_toy_data():
    """corrected=False returns the stored cloud itself, unshifted, and the plain
    difference of the published estimates. This is the toy-level half of the
    no-regression test; the real-data half compares against a frozen baseline."""
    d, _ = _toy_data_bc()
    reps, hat, _ = d.term("c")
    assert reps is d.reps["c"]
    con = fm.make_contrast(d, "y2")
    assert not con.corrected
    pd.testing.assert_series_equal(con.y_hat, d.hat("b") - d.hat("c"),
                                   check_names=False)
    pd.testing.assert_frame_equal(con.y, d.rep("b") - d.rep("c"))


def test_plugin_offset_is_recovered_from_the_data():
    """s is read from mean(cloud) - hat - bias, not hard-coded: 0 for the MLE
    arms (snapped to exactly zero), ln sqrt(21/22) for the moment-fit arm."""
    d, truth = _toy_data_bc()
    assert d.plugin_offset("a") == 0.0
    assert d.plugin_offset("b") == 0.0
    assert d.plugin_offset("c") == pytest.approx(_S_IDA, abs=1e-12)


def test_plugin_offset_refuses_an_inconsistent_bias_column():
    """A bias column that does not belong to the cloud (e.g. a stale nb 072)
    makes s vary across units, and that must not pass silently."""
    d, _ = _toy_data_bc()
    stale = dict(d.bias)
    stale["c"] = stale["c"] * 1.5
    d_stale = fm.FragilityData(quantity=d.quantity, im_tag=d.im_tag, rows=d.rows,
                               groups=d.groups, reps=d.reps, estimates=d.estimates,
                               estimates_bc=d.estimates_bc, bias=stale)
    with pytest.raises(ValueError, match="not constant"):
        d_stale.plugin_offset("c")


@pytest.mark.parametrize("name", ["y2", "y0", "y1"])
def test_corrected_shift_is_minus_the_bias_difference(name):
    """Acceptance 2: y_hat_bc - y_hat == -(b_left - b_right), expanded to rows
    where the contrast is at row scope (y0, y1) and left at group scope (y2)."""
    d, truth = _toy_data_bc()
    left, right, scope, _ = fm._CONTRAST_SPECS[name]
    con, con_bc = fm.make_contrast(d, name), fm.make_contrast(d, name, corrected=True)

    def at_scope(sym, x):
        return d.expand(x) if scope == "row" and d.scope_of(sym) == "group" else x

    expected = -(at_scope(left, truth[left]["b"]) - at_scope(right, truth[right]["b"]))
    np.testing.assert_allclose((con_bc.y_hat - con.y_hat).to_numpy(),
                               expected.reindex(con.y_hat.index).to_numpy(),
                               atol=1e-12)
    assert con_bc.corrected and con_bc.scope == con.scope


@pytest.mark.parametrize("name", ["y2", "y0"])
def test_correction_leaves_V_and_v_unchanged(name):
    """Acceptance 3: a constant shift per row moves no spread."""
    d, _ = _toy_data_bc()
    con, con_bc = fm.make_contrast(d, name), fm.make_contrast(d, name, corrected=True)
    np.testing.assert_allclose(con_bc.V, con.V, atol=1e-14)
    np.testing.assert_allclose(con_bc.v.to_numpy(), con.v.to_numpy(), atol=1e-14)


@pytest.mark.parametrize("name", ["y2", "y0"])
def test_corrected_replicates_are_centred_on_the_corrected_estimate(name):
    """Acceptance 6, at the contrast level: after the -(2b + s) shift, every row
    of y averages to y_hat_bc EXACTLY. This is what pins where the interval
    sits. Without the shift the rows sit at y_hat + (s_L - s_R) + (b_L - b_R),
    which is shown too, so the test fails if the shift is ever dropped."""
    d, truth = _toy_data_bc()
    con, con_bc = fm.make_contrast(d, name), fm.make_contrast(d, name, corrected=True)
    np.testing.assert_allclose(con_bc.y.mean(axis=1).to_numpy(),
                               con_bc.y_hat.to_numpy(), atol=1e-12)
    # the uncorrected cloud is NOT centred on its estimate: it is off by the
    # plug-in offset plus the bias, which is the displacement the shift removes
    off = (con.y.mean(axis=1) - con.y_hat).abs().max()
    assert off > 0.005


def test_corrected_fit_ci_is_centred_on_its_estimate():
    """Acceptance 6, at the fit level: the percentile CI of the corrected fit
    sits on m_bc (to Monte-Carlo error), and is not displaced by the 2b + s a
    naive "y_hat only" correction would leave."""
    d, _ = _toy_data_bc(k=2000)
    fits = fm.ModelRegistry(d)
    fits.fit("A2_bc", n_boot=2000)
    f = fits["A2_bc"]
    lo, hi = f.ci()
    assert abs((lo + hi) / 2 - f.m) < 0.25 * f.se_boot


def test_corrected_and_uncorrected_fits_coexist():
    """Acceptance 4: both ids live in one registry, neither overwrites the
    other, the refusal to overwrite still holds for each, and the corrected fit
    really used different estimates."""
    d, _ = _toy_data_bc()
    fits = fm.ModelRegistry(d)
    fits.fit("A2", n_boot=20)
    fits.fit("A2_bc", n_boot=20)
    assert fits["A2"].contrast is not fits["A2_bc"].contrast
    assert not fits["A2"].corrected and fits["A2_bc"].corrected
    assert fits["A2"].m != pytest.approx(fits["A2_bc"].m)
    for mid in ("A2", "A2_bc"):
        with pytest.raises(ValueError, match="already been fitted"):
            fits.fit(mid)
    fits.fit("A2_bc", overwrite=True)


def test_fit_refuses_a_corrected_keyword():
    """One id, one contrast: correction is chosen by the id, never by an argument."""
    d, _ = _toy_data_bc()
    with pytest.raises(TypeError, match="A2_bc"):
        fm.ModelRegistry(d).fit("A2", corrected=True)


def test_contrast_cache_is_keyed_on_the_flag():
    """D2: asking for the uncorrected contrast after the corrected one (or the
    reverse) must not hand back the other."""
    d, _ = _toy_data_bc()
    reg = fm.ModelRegistry(d)
    bc, plain = reg.contrast("y2", True), reg.contrast("y2")
    assert bc.corrected and not plain.corrected
    assert reg.contrast("y2", True) is bc          # still memoised


def test_theta_bc_warns_and_reproduces_the_uncorrected_fit():
    """Acceptance 5, the D4 decision (2026-10-07): theta is corrected by zero,
    so a `_bc` fit on it warns and gives exactly the uncorrected numbers,
    replicates included (b = 0 and s = 0, so the shift is exactly zero)."""
    d, _ = _toy_data_bc(quantity="theta", correct=False)
    fits = fm.ModelRegistry(d)
    fits.fit("A2", n_boot=20)
    with pytest.warns(UserWarning, match="no-op"):
        fits.fit("A2_bc", n_boot=20)
    plain, bc = fits["A2"], fits["A2_bc"]
    np.testing.assert_array_equal(bc.beta, plain.beta)
    pd.testing.assert_frame_equal(bc.contrast.y, plain.contrast.y)
    pd.testing.assert_frame_equal(bc.replicates, plain.replicates)


def test_missing_bias_corrected_data_names_the_model():
    """Acceptance 7: the error says which MODEL could not be fitted, not only
    which arm was missing. Covered for both inputs a `_bc` fit needs."""
    d, _ = _toy_data_bc()
    no_bc = fm.FragilityData(quantity=d.quantity, im_tag=d.im_tag, rows=d.rows,
                             groups=d.groups, reps=d.reps, estimates=d.estimates,
                             estimates_bc={}, bias=d.bias)
    with pytest.raises(KeyError, match="A2_bc"):
        fm.ModelRegistry(no_bc).fit("A2_bc")
    no_bias = fm.FragilityData(quantity=d.quantity, im_tag=d.im_tag, rows=d.rows,
                               groups=d.groups, reps=d.reps, estimates=d.estimates,
                               estimates_bc=d.estimates_bc, bias={})
    with pytest.raises(KeyError, match="A2_bc"):
        fm.ModelRegistry(no_bias).fit("A2_bc")
    # and the uncorrected model is unaffected by either gap
    fm.ModelRegistry(no_bias).fit("A2")


def test_dl_fits_on_corrected_estimates():
    """D4: DL reads y_hat, v and (in the bootstrap) y, all from the contrast,
    so it needs nothing special. Fitted through a custom spec table because no
    shipped `_bc` model uses DL."""
    d, _ = _toy_data_bc()
    spec = fm.ModelSpec("A1a", "y1", ("DL",), diagonal_V=True)
    specs = {"A1a": spec,
             "A1a_bc": fm.ModelSpec("A1a_bc", "y1", ("DL",), diagonal_V=True,
                                    corrected=True)}
    fm._check_model_specs(specs)
    fits = fm.ModelRegistry(d, specs=specs)
    fits.fit("A1a", n_boot=50)
    fits.fit("A1a_bc", n_boot=50)
    plain, bc = fits["A1a", "DL"], fits["A1a_bc", "DL"]
    assert bc.corrected
    # The fit must be DL run on the corrected y_hat with the unchanged v, so
    # recompute that directly and compare.
    M, *_ =mra.compute_re_summary_effect(bc.contrast.v, bc.contrast.y_hat)
    assert bc.m == pytest.approx(float(M))
    assert bc.m != pytest.approx(plain.m)
    assert bc.se_boot == pytest.approx(plain.se_boot, rel=0.2)


def test_corrected_fits_say_so_in_every_report():
    """D5: summary header, table column, forest title and contrast summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, _ = _toy_data_bc()
    fits = fm.ModelRegistry(d)
    fits.fit("A2", n_boot=20)
    fits.fit("A2_bc", n_boot=20)

    assert fm.BC_TAG in fits["A2_bc"].summary().splitlines()[0]
    assert fm.BC_TAG not in fits["A2"].summary()
    assert fm.BC_TAG in fits["A2_bc"].contrast.summary()

    table = fits.table()
    assert bool(table.loc[("A2_bc", "REML"), "corrected"])
    assert not bool(table.loc[("A2", "REML"), "corrected"])

    assert fm.BC_TAG in fits.forest("A2_bc").get_title()
    assert fm.BC_TAG not in fits.forest("A2").get_title()
    plt.close("all")


# -- the same on the real pipeline output ------------------------------------

@pytest.fixture(scope="module")
def real_beta():
    return fm.load_fragility_data(_ROOT, _IM_TAG, quantity="beta",
                                  verbose=False, check=False)


@needs_data
def test_real_beta_shift_matches_the_saved_nb070_bias(real_beta):
    """Acceptance 2 on real data: the per-arm b the module infers from
    estimates - estimates_bc is the `bias` column nb 070 wrote to its CSVs, and
    y2_bc - y2 is -(b_MSA-FX - b_IDA-FX). The plug-in offset comes out at
    exactly ln sqrt(21/22) on IDA and zero on MSA."""
    d = real_beta
    saved = mra.load_saved_bias(_ROOT, ["msa_femap695", "ida_femap695"], ("beta",),
                                None, _IM_TAG, "by_group")
    b = {}
    for sym, arm in (("b", "msa_femap695"), ("c", "ida_femap695")):
        b[sym] = d.hat(sym) - d.hat(sym, corrected=True)
        csv = saved[arm]["beta"]["bias"].droplevel("n_storeys").reindex(b[sym].index)
        np.testing.assert_allclose(b[sym].to_numpy(), csv.to_numpy(), atol=1e-12)
    assert d.plugin_offset("b") == 0.0
    assert d.plugin_offset("c") == pytest.approx(_S_IDA, abs=1e-12)

    con, con_bc = fm.make_contrast(d, "y2"), fm.make_contrast(d, "y2", corrected=True)
    np.testing.assert_allclose((con_bc.y_hat - con.y_hat).to_numpy(),
                               (-(b["b"] - b["c"])).to_numpy(), atol=1e-12)
    np.testing.assert_allclose(con_bc.V, con.V, atol=1e-14)


@needs_data
def test_real_A2_bc_ci_is_centred_and_ratio_moves_as_expected(real_beta):
    """Acceptance 6 on real data, with the numbers the task predicted: the
    A2 ratio moves from ~0.984 to ~0.993, se_boot does not move, and the
    corrected CI's midpoint sits on its estimate (the uncorrected one is off by
    about one se_boot, mostly the IDA plug-in offset)."""
    fits = fm.ModelRegistry(real_beta)
    fits.fit("A2", n_boot=1000)
    fits.fit("A2_bc", n_boot=1000)
    plain, bc = fits["A2"], fits["A2_bc"]
    assert plain.ratio() == pytest.approx(0.984, abs=0.002)
    assert bc.ratio() == pytest.approx(0.993, abs=0.002)
    assert bc.se_boot == pytest.approx(plain.se_boot, rel=0.05)
    lo, hi = bc.ci()
    assert abs((lo + hi) / 2 - bc.m) < 0.25 * bc.se_boot


_BASELINE = (Path(__file__).parent / "data" / f"registry_baseline_{_IM_TAG}.pkl")


@needs_data
@pytest.mark.skipif(not _BASELINE.is_file(),
                    reason="needs tests/data/registry_baseline_*.pkl")
def test_uncorrected_fits_reproduce_the_frozen_baseline():
    """Acceptance 1: every model 073 fits, on both quantities, gives the same
    beta, tau2, se and outer-bootstrap replicates as the code did BEFORE the
    `corrected` flag existed. The baseline was written by
    tests/data/make_registry_baseline.py on the unmodified module.

    Exact equality (to 1e-12), not approximate: REML's restarts are seeded, so
    an unchanged code path must reproduce every digit. Slow (several minutes),
    because it re-runs 40 fits with their bootstrap refits."""
    with open(_BASELINE, "rb") as fh:
        base = pickle.load(fh)
    assert not base["errors"], base["errors"]
    regs = {}
    for (quantity, model_id, estimator), want in base["fits"].items():
        if quantity not in regs:
            data = fm.load_fragility_data(_ROOT, base["im_tag"], quantity=quantity,
                                          verbose=False, check=False)
            regs[quantity] = fm.ModelRegistry(data)
        reg = regs[quantity]
        if (model_id, estimator) not in reg.fitted:
            reg.fit(model_id, n_boot=base["n_boot"])
        got = reg[model_id, estimator]
        assert not got.corrected
        np.testing.assert_allclose(got.beta, want["beta"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.se_beta, want["se_beta"], rtol=0, atol=1e-12)
        assert got.tau2.keys() == want["tau2"].keys()
        for k in want["tau2"]:
            assert got.tau2[k] == pytest.approx(want["tau2"][k], abs=1e-12)
        pd.testing.assert_frame_equal(got.replicates, want["replicates"],
                                      check_exact=False, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# The joint theta / beta model (A8)                                            #
# --------------------------------------------------------------------------- #
#
# `_toy_joint` builds a PAIR of containers (theta, beta) whose replicate clouds
# share their resamples: in every arm, replicate r of theta and replicate r of
# beta are correlated draws (corr 0.6), as one MLE fit to one set of stripe
# counts would give. It is large enough (20 sites x 2 structures, 7 crossed
# designs) for the joint REML to be well posed, carries a known beta bias for
# the `_bc` specs, and carries the A3ae covariate columns A8ax / A8d need.

_A3AE = ("d_ln_saratio_mean_unif_corrected", "d_ln_ds595_mean_unif_corrected",
         "d_ln_saratio_sigma_unif_corrected", "d_ln_ds595_sigma_unif_corrected")

# A joint rung WITH covariates. It used to ship in JOINT_MODEL_SPECS as "A8ax";
# the shipped table no longer has a covariate rung, but the covariate code path
# (build_X_joint with two covariate lists, gamma, explained_covariance,
# predictive) is still there and still needs exercising, so the tests carry
# their own copy of the spec.
_A8AX = fm.JointModelSpec("A8ax", "y0_avg", components=(), correlated=("row",),
                          covariates_theta=_A3AE, covariates_beta=_A3AE)
_JOINT_SPECS_WITH_COVARIATES = {**fm.JOINT_MODEL_SPECS, "A8ax": _A8AX}


def _toy_joint(n_sites: int = 20, k: int = 400, seed: int = 0, r_samp: float = 0.6):
    """``(theta_data, beta_data)`` with replicate-aligned, correlated clouds."""
    rng = np.random.default_rng(seed)
    sites = np.repeat(np.arange(n_sites), 2)
    storeys = np.tile([3, 5], n_sites)
    design = [f"g3_{s % 4}" if ns == 3 else f"g5_{s % 3}" for s, ns in zip(sites, storeys)]
    rows = pd.DataFrame({"site": sites, "n_storeys": storeys,
                         "structure_id": 0, "tag": [f"t{i}" for i in range(len(sites))],
                         "design_group_id": design, "n_sites_in_group": 1,
                         "is_representative": False})
    for cov in _A3AE:
        rows[cov] = rng.normal(size=len(rows))
    rows = rows.set_index(["site", "n_storeys"]).sort_index()
    groups = pd.Index(sorted(rows["design_group_id"].unique()), name="design_group_id")
    cols = pd.Index(range(k), name="k")

    reps = {"theta": {}, "beta": {}}
    est = {"theta": {}, "beta": {}}
    est_bc = {"theta": {}, "beta": {}}
    bias = {"theta": {}, "beta": {}}
    for sym, index in (("a", rows.index), ("b", groups), ("c", groups), ("c_avg", groups)):
        m = len(index)
        z1 = rng.standard_normal((m, k))
        z2 = r_samp * z1 + np.sqrt(1 - r_samp ** 2) * rng.standard_normal((m, k))
        mu_t, mu_b = rng.normal(0, 0.1, (m, 1)), rng.normal(-0.9, 0.1, (m, 1))
        for q, cloud in (("theta", mu_t + 0.05 * z1), ("beta", mu_b + 0.12 * z2)):
            cloud = pd.DataFrame(cloud, index=index, columns=cols)
            b = (pd.Series(rng.uniform(-0.04, -0.005, m), index=index) if q == "beta"
                 else pd.Series(0.0, index=index))
            reps[q][sym] = cloud
            est[q][sym] = cloud.mean(axis=1) - b          # cloud sits at hat + b
            est_bc[q][sym] = est[q][sym] - b
            bias[q][sym] = b
    out = []
    for q in ("theta", "beta"):
        for d in (reps, est, est_bc, bias):
            d[q]["c_ub"] = None
        out.append(fm.FragilityData(quantity=q, im_tag="TEST", rows=rows, groups=groups,
                                    reps=reps[q], estimates=est[q],
                                    estimates_bc=est_bc[q], bias=bias[q]))
    return tuple(out)


@pytest.fixture(scope="module")
def toy_joint_fits():
    """One registry with the A8 rungs the structural tests read, fitted once."""
    theta, beta = _toy_joint()
    reg = fm.JointModelRegistry(theta, beta, _JOINT_SPECS_WITH_COVARIATES)
    for mid in ("A8a", "A8a_bc", "A8b", "A8ax"):
        reg.fit(mid, n_boot=30)
    return reg


def test_joint_V_blocks_equal_the_univariate_V_exactly():
    """Acceptance 3: np.cov on the stacked matrix does the same arithmetic per
    entry, so the diagonal blocks are the univariate V bit for bit."""
    theta, beta = _toy_joint()
    con_t, con_b = fm.make_contrast(theta, "y0_avg"), fm.make_contrast(beta, "y0_avg")
    j = fm.JointContrast.from_pair(con_t, con_b)
    assert np.array_equal(j.V_theta, con_t.V)
    assert np.array_equal(j.V_beta, con_b.V)
    assert j.V.shape == (2 * j.n, 2 * j.n)
    assert list(j.y_hat.index.names) == ["outcome", "site", "n_storeys"]
    assert list(j.y_hat.index.get_level_values("outcome").unique()) == ["theta", "beta"]
    # V_cross carries the built-in sampling correlation
    assert j.r_samp_bar == pytest.approx(0.6, abs=0.1)


def test_misaligned_beta_replicates_are_caught():
    """Acceptance 6, both ways a misalignment can arise.

    (a) the beta columns permuted WITH their labels: from_pair compares labels
        in order and refuses;
    (b) the beta VALUES permuted under unchanged labels (two files from two
        runs): only the data can show it, and check_joint_inputs' chi-square
        test on the per-row sampling correlations does.
    """
    theta, beta = _toy_joint()
    con_t, con_b = fm.make_contrast(theta, "y0_avg"), fm.make_contrast(beta, "y0_avg")
    perm = np.random.default_rng(0).permutation(con_b.y.shape[1])
    shuffled = dataclasses_replace(con_b, y=con_b.y.iloc[:, perm])
    with pytest.raises(ValueError, match="replicate columns differ"):
        fm.JointContrast.from_pair(con_t, shuffled)

    fm.check_joint_inputs(theta, beta)          # aligned: passes
    reps = {s: (None if c is None else
                pd.DataFrame(c.to_numpy()[:, perm], index=c.index, columns=c.columns))
            for s, c in beta.reps.items()}
    unaligned = fm.FragilityData(quantity="beta", im_tag="TEST", rows=beta.rows,
                                 groups=beta.groups, reps=reps, estimates=beta.estimates,
                                 estimates_bc=beta.estimates_bc, bias=beta.bias)
    with pytest.raises(ValueError, match="UNALIGNED"):
        fm.check_joint_inputs(theta, unaligned)


def dataclasses_replace(obj, **changes):
    import dataclasses
    return dataclasses.replace(obj, **changes)


def test_from_pair_refuses_a_corrected_theta_and_swapped_quantities():
    theta, beta = _toy_joint()
    con_t, con_b = fm.make_contrast(theta, "y0_avg"), fm.make_contrast(beta, "y0_avg")
    with pytest.raises(ValueError, match="never corrected"):
        fm.JointContrast.from_pair(fm.make_contrast(theta, "y0_avg", corrected=True), con_b)
    with pytest.raises(ValueError, match="quantities"):
        fm.JointContrast.from_pair(con_b, con_t)


def test_joint_spec_table_obeys_one_id_one_contrast():
    """Acceptance 7. The shipped table passes, the A8 rungs exist with their
    `_bc` / `_hb` / `_bc_hb` partners, and the table's structure is as shipped.

    The table was reduced to A8a, A8b and A8c (A8ax and A8d removed); the
    covariate rung is now a test-local spec (_A8AX)."""
    fm._check_joint_model_specs(fm.JOINT_MODEL_SPECS, fm.MODEL_SPECS)
    S = fm.JOINT_MODEL_SPECS
    for mid in ("A8a", "A8b", "A8c"):
        assert not S[mid].corrected and S[mid].bootstrap_scheme == "record"
        assert S[mid + "_bc"].corrected
        assert S[mid + "_hb"].bootstrap_scheme == "hybrid"
        assert S[mid + "_bc_hb"].corrected and S[mid + "_bc_hb"].bootstrap_scheme == "hybrid"
    assert S["A8a"].components == () and S["A8a"].correlated == ("row",)
    assert S["A8b"].components == ("design", "site") and S["A8b"].correlated == ("row",)
    assert S["A8c"].correlated == ("design", "site", "row")
    # The covariate rung, now test-local, still passes the table checks.
    fm._check_joint_model_specs(_JOINT_SPECS_WITH_COVARIATES, fm.MODEL_SPECS)
    assert _A8AX.covariates_theta == _A3AE == _A8AX.covariates_beta
    assert set(S).isdisjoint(fm.MODEL_SPECS)


@pytest.mark.parametrize("bad, match", [
    ({"A8a": fm.JointModelSpec("A8a"),
      "A8a_bc": fm.JointModelSpec("A8a_bc", correlated=(), corrected=True)}, "correlated"),
    ({"A8a_bc": fm.JointModelSpec("A8a_bc", corrected=True)}, "partner"),
    ({"A0a": fm.JointModelSpec("A0a")}, "both MODEL_SPECS"),
    ({"A8x": fm.JointModelSpec("A8x", covariates_beta=("u", "u"))}, "duplicates"),
])
def test_joint_spec_check_rejects_a_broken_table(bad, match):
    with pytest.raises(ValueError, match=match):
        fm._check_joint_model_specs(bad, fm.MODEL_SPECS)


def test_joint_spec_refuses_dl_and_unknown_levels():
    """Hard requirement 6: DL has no bivariate form."""
    with pytest.raises(ValueError, match="DerSimonian-Laird"):
        fm.JointModelSpec("A8z", estimators=("DL", "REML"))
    with pytest.raises(ValueError, match="not in the model"):
        fm.JointModelSpec("A8z", components=(), correlated=("site",))


def test_joint_registry_refusals(toy_joint_fits):
    """Overwrite refused, corrected= refused, unknown and univariate ids refused."""
    reg = toy_joint_fits
    with pytest.raises(ValueError, match="already been fitted"):
        reg.fit("A8a")
    with pytest.raises(TypeError, match="A8a_bc"):
        reg.fit("A8a", corrected=True)
    with pytest.raises(KeyError, match="ModelRegistry"):
        reg.fit("A0a")
    with pytest.raises(ValueError, match="pair"):
        fm.JointModelRegistry(reg.beta_data, reg.theta_data)


def test_joint_bc_corrects_beta_only(toy_joint_fits):
    """Hard requirement 5: the theta half of A8a_bc is the very same contrast as
    in A8a; only beta's y_hat moves, and V (a per-row constant shift) does not."""
    plain, bc = toy_joint_fits["A8a"].contrast, toy_joint_fits["A8a_bc"].contrast
    assert bc.theta is plain.theta
    assert bc.corrected == {"theta": False, "beta": True}
    assert not np.allclose(bc.beta.y_hat, plain.beta.y_hat)
    np.testing.assert_allclose(bc.V, plain.V, atol=1e-14)
    assert fm.JOINT_BC_TAG in toy_joint_fits["A8a_bc"].summary().splitlines()[0]
    assert fm.JOINT_BC_TAG not in toy_joint_fits["A8a"].summary()


def test_joint_fit_objects_are_internally_consistent(toy_joint_fits):
    """Acceptance 4 at the fit level, plus the D5/D6 accessors."""
    for mid in ("A8a", "A8b", "A8ax"):
        f = toy_joint_fits[mid]
        for T in f.T.values():
            assert np.linalg.eigvalsh(T).min() >= -1e-12
        for r in [*f.rho.values(), f.rho_tot]:
            assert np.isnan(r) or -1 <= r <= 1
        # every replicate T PSD too, and every replicate rho in range
        reps = f.replicates
        for lev in f.levels:
            tt, bb, tb = (reps[f"T_{lev}_{e}"] for e in ("tt", "bb", "tb"))
            assert (tt * bb - tb ** 2 >= -1e-12).all() and (tt >= 0).all()
        assert reps["rho_tot"].dropna().between(-1, 1).all()
        np.testing.assert_allclose(f.T_total, sum(f.T.values()))
        assert np.linalg.eigvalsh(f.Sigma_pred).min() > 0
        for method in ("percentile", "fisher"):
            ci = f.rho_ci("tot", method=method)
            assert ci["lo"] <= ci["hi"]
        assert set(f.boundary_fractions().columns) == {
            "theta_at_zero", "beta_at_zero", "rho_undefined", "rho_at_bound"}
        assert list(f.mafc_draws().columns[:5]) == [
            "m_theta", "m_beta", "T_total_tt", "T_total_bb", "T_total_tb"]
        assert f.quantity == "theta+beta"
        assert f.model_id in f.summary()
    # A8b: the cross term is estimated only at the level(s) its spec lists in
    # `correlated` (the row level, as shipped); every other level's is fixed at
    # zero. Read from the spec, so the test follows the table.
    b = toy_joint_fits["A8b"]
    assert b.levels == ["design", "site", "row"]
    for lev in b.levels:
        if lev not in b.spec.correlated:
            assert b.T[lev][0, 1] == 0, lev
    table = toy_joint_fits.table()
    assert set(table.index.get_level_values("model")) == {"A8a", "A8a_bc", "A8b", "A8ax"}


def test_explained_covariance_adds_up(toy_joint_fits):
    """With covariates: explained + residual == total, and the residual is T_total."""
    f = toy_joint_fits["A8ax"]
    ec = f.explained_covariance()
    np.testing.assert_allclose(ec["explained"] + ec["residual"], ec["total"])
    np.testing.assert_allclose(ec["residual"].to_numpy(), f.T_total)
    assert len(f.gamma("theta")) == len(f.gamma("beta")) == 4
    # a new structure at the covariate means predicts the intercepts
    mean, _ = f.predictive()
    np.testing.assert_allclose(mean, f.m)
    with pytest.raises(ValueError, match="no covariates"):
        toy_joint_fits["A8a"].explained_covariance()


def test_univariate_registry_does_not_know_the_joint_ids():
    """The two tables are disjoint; ModelRegistry is untouched by the A8 code."""
    theta, _ = _toy_joint()
    with pytest.raises(KeyError, match="unknown model"):
        fm.ModelRegistry(theta).fit("A8a")


# -- the same on the real pipeline output ------------------------------------

@needs_data
def test_real_joint_inputs_pass_step_0(real_theta, real_beta):
    """Step 0 on the real data: aligned, well conditioned, blocks exact."""
    out = fm.check_joint_inputs(real_theta, real_beta)
    assert out["conditioning"]["joint"]["n_small"] == 0
    j = out["contrast"]
    assert np.array_equal(j.V_theta, fm.make_contrast(real_theta, "y0_avg").V)
    assert np.array_equal(j.V_beta, fm.make_contrast(real_beta, "y0_avg").V)


@needs_data
def test_real_A8a_matches_the_univariate_single_level_fits(real_theta, real_beta):
    """Acceptance 8, in the two parts the plan's section 0 item 8 calls for.

    (a) With V_cross zeroed the joint fit IS the two univariate single-level
        fits of y0_avg, to 1e-6.
    (b) With the full V_known, m moves, because each outcome's GLS weights now
        borrow from the other's sampling errors. On 2026-10-08 the shifts were
        about 0.4 SE (theta) and 0.25 SE (beta); the test holds them under
        0.5 SE, and prints them so a change is visible.
    """
    reg = fm.JointModelRegistry(real_theta, real_beta)
    j = reg.contrast("y0_avg")
    spec = fm.JOINT_MODEL_SPECS["A8a"]
    X, _, _ = fm.build_X_joint(spec, j)
    Gs = fm.joint_level_gs(spec, j)
    n = j.n
    V0 = j.V.copy()
    V0[:n, n:] = 0.0
    V0[n:, :n] = 0.0
    sep = mra.fit_reml_mv(j.y_hat.to_numpy(float), X, V0, Gs, ())
    full = reg.fit("A8a")
    for q, con in enumerate((j.theta, j.beta)):
        uni = mra.fit_reml(con.y_hat.to_numpy(float), np.ones((n, 1)), con.V,
                           [np.eye(n)], names=["tau_u2"])
        assert sep["beta"][q] == pytest.approx(uni["beta"][0], abs=1e-6)
        assert sep["T"]["row"][q, q] == pytest.approx(uni["tau2"][0], rel=1e-4)
        shift = abs(full.m[q] - uni["beta"][0]) / uni["se_beta"][0]
        print(f"A8a {fm.OUTCOMES[q]}: |m_joint - m_uni| = {shift:.2f} SE")
        assert shift < 0.5


# --------------------------------------------------------------------------- #
# The hybrid bootstrap (dump/plan_hybrid_bootstrap.md)                         #
# --------------------------------------------------------------------------- #
#
# `bootstrap_scheme` on a spec chooses the outer bootstrap. "record" (the
# default) refits each record-resampling replicate as it is, which is right
# for m and inflates every replicate variance component. "hybrid" (ids ending
# `_hb`) refits X beta_hat + new effects drawn from the fitted variance
# components + the centred replicate cloud. The array-level maths is tested in
# test_data_simulation.py and test_reml_mv.py; here it is the spec rules, the
# wiring through the two registries, and what the fit objects say about
# themselves. Numbered as the task's acceptance criteria.

from dataclasses import replace as _replace          # noqa: E402

import phd_project.scripts.data_simulation as dsim   # noqa: E402


def test_bootstrap_scheme_is_off_by_default_and_keyword_only():
    """Acceptance 1 and hard requirement 2: record-only unless a spec says otherwise."""
    assert fm.ModelSpec("X", "y0").bootstrap_scheme == "record"
    assert fm.JointModelSpec("X").bootstrap_scheme == "record"
    with pytest.raises(ValueError, match="unknown bootstrap_scheme"):
        fm.ModelSpec("X", "y0", bootstrap_scheme="hybird")
    with pytest.raises(ValueError, match="unknown bootstrap_scheme"):
        fm.JointModelSpec("X", bootstrap_scheme="parametric")
    # Keyword-only: a ninth positional argument is refused, not read as the scheme.
    with pytest.raises(TypeError):
        fm.ModelSpec("X", "y0", ("REML",), (), (), False, "", False, "hybrid")

    # No existing entry changed: every id without `_hb` is record-only, every
    # `_hb` id is hybrid, and each `_hb` spec differs from its partner in the
    # scheme alone (plus id and note). Checked for EVERY `_hb` entry rather than
    # a frozen list, so adding a hybrid rung is not a test failure but adding a
    # hybrid rung that has drifted from its partner is.
    for table in (fm.MODEL_SPECS, fm.JOINT_MODEL_SPECS):
        hb = [k for k in table if k.endswith("_hb")]
        assert hb, "at least one hybrid rung is expected in each table"
        for key, spec in table.items():
            assert (spec.bootstrap_scheme == "hybrid") == key.endswith("_hb"), key
        for key in hb:
            partner = key.removesuffix("_hb")
            assert partner in table, f"{key} has no record-only partner"
            assert _replace(table[key], model_id=partner, bootstrap_scheme="record",
                            note=table[partner].note) == table[partner], key


_HB = dict(bootstrap_scheme="hybrid")


@pytest.mark.parametrize("bad, match", [
    # the id says hybrid, the flag does not
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",)),
      "A2_hb": fm.ModelSpec("A2_hb", "y2", ("REML",))}, "'hybrid'"),
    # the flag says hybrid, the id does not
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",), **_HB)}, "'hybrid'"),
    # an _hb spec with no record-only partner
    ({"A9_hb": fm.ModelSpec("A9_hb", "y2", ("REML",), **_HB)}, "record-only partner"),
    # an _hb spec that has drifted from its partner
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",)),
      "A2_hb": fm.ModelSpec("A2_hb", "y2", ("DL", "REML"), **_HB)}, "estimators"),
    # a _bc spec that disagrees with its partner on the scheme (hard req. 2):
    # the partner of A2_bc_hb is A2_hb, which here is missing its hybrid flag
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",)),
      "A2_hb": fm.ModelSpec("A2_hb", "y2", ("REML",)),
      "A2_bc_hb": fm.ModelSpec("A2_bc_hb", "y2", ("REML",), corrected=True, **_HB)},
     "bootstrap_scheme"),
    # the suffixes in the wrong order
    ({"A2": fm.ModelSpec("A2", "y2", ("REML",)),
      "A2_hb": fm.ModelSpec("A2_hb", "y2", ("REML",), **_HB),
      "A2_hb_bc": fm.ModelSpec("A2_hb_bc", "y2", ("REML",), corrected=True, **_HB)},
     r"base\[_bc\]\[_hb\]"),
])
def test_spec_check_rejects_a_broken_hybrid_table(bad, match):
    """Acceptance 2: the `_hb` rules are enforced, by the same check as `_bc`."""
    with pytest.raises(ValueError, match=match):
        fm._check_model_specs(bad)


def test_spec_check_accepts_a_full_set_of_suffixes():
    """base, base_bc, base_hb and base_bc_hb together are a consistent table."""
    fm._check_model_specs({
        "A2": fm.ModelSpec("A2", "y2", ("REML",)),
        "A2_bc": fm.ModelSpec("A2_bc", "y2", ("REML",), corrected=True),
        "A2_hb": fm.ModelSpec("A2_hb", "y2", ("REML",), **_HB),
        "A2_bc_hb": fm.ModelSpec("A2_bc_hb", "y2", ("REML",), corrected=True, **_HB)})
    # The joint check runs the same rules.
    with pytest.raises(ValueError, match="record-only partner"):
        fm._check_joint_model_specs(
            {"A9_hb": fm.JointModelSpec("A9_hb", **_HB)}, fm.MODEL_SPECS)


@pytest.mark.parametrize("name", ["y2", "y0", "y1"])
def test_hybrid_noise_is_the_same_for_a_contrast_and_its_bc_partner(name):
    """Acceptance 3 / H4: the `_bc` replicate shift cancels in e = y - ybar.

    So a bias correction enters the hybrid pseudo-data only through X beta_hat.
    Equal to rounding (1e-13), not bit for bit: (y + c) - (ybar + c) and
    y - ybar differ in the last bit. And with every replicate used, the noise
    averages zero per row to machine precision.
    """
    data, _ = _toy_data_bc()
    plain = dsim.centre_replicates(data.contrast(name).y.to_numpy(float))
    bc = dsim.centre_replicates(data.contrast(name, corrected=True).y.to_numpy(float))
    assert not np.allclose(data.contrast(name).y, data.contrast(name, corrected=True).y)
    np.testing.assert_allclose(bc, plain, rtol=0, atol=1e-13)
    assert np.abs(plain.mean(axis=1)).max() < 1e-14


@pytest.fixture(scope="module")
def toy_hybrid():
    """Record-only and hybrid fits of one crossed model, on the 40-row toy study.

    ``_toy_joint`` rather than ``_toy_data``: six rows cannot carry a design x
    site model. Fitted once for the tests below.
    """
    theta, beta = _toy_joint()
    reg = fm.ModelRegistry(beta)
    reg.fit("A0c", n_boot=30)
    reg.fit("A0c_hb", n_boot=30)
    joint = fm.JointModelRegistry(theta, beta)
    joint.fit("A8a", n_boot=30)
    joint.fit("A8a_hb", n_boot=30)
    return {"theta": theta, "beta": beta, "reg": reg, "joint": joint}


def test_hybrid_flag_leaves_the_record_only_fit_alone(toy_hybrid):
    """Acceptance 1 on toy data: the partner of an `_hb` fit is untouched by it.

    Same full-sample fit (the scheme only changes the bootstrap), and the
    record-only replicates do not depend on the seed argument at all.
    """
    reg = toy_hybrid["reg"]
    rec, hyb = reg["A0c"], reg["A0c_hb"]
    assert np.array_equal(rec.beta, hyb.beta) and rec.tau2 == hyb.tau2
    assert not rec.replicates["M_star"].equals(hyb.replicates["M_star"])

    other = fm.ModelRegistry(toy_hybrid["beta"])
    other.fit("A0c", n_boot=30, boot_seed=987654)
    pd.testing.assert_frame_equal(other["A0c"].replicates, rec.replicates,
                                  check_exact=True)
    assert rec.boot == {"scheme": "record", "seed": None, "zero_levels": ()}


def test_hybrid_replicates_are_the_array_level_path(toy_hybrid):
    """The registry's hybrid bootstrap IS ``dsim.hybrid_pseudo_data`` plus ``mra.fit_reml``.

    Rebuilt here by hand from the full-sample fit, the factor matrices of
    ``build_components`` and the seed, and compared exactly. This is what ties
    the validation study (which runs the array-level path) to the fits the
    registry actually produces.
    """
    fit = toy_hybrid["reg"]["A0c_hb"]
    con, spec = fit.contrast, fit.spec
    X, _, _ = fm.build_X(spec, con)
    comps = {name: (factor, np.sqrt(fit.tau2[name]))
             for name, (factor, _) in fm.build_components(spec, con).items()}
    Y_star = dsim.hybrid_pseudo_data(con.y.to_numpy(float), X @ fit.beta, comps,
                                     seed=fm.BOOT_SEED, n_boot=30)
    Gs, names = dsim.build_component_gs(comps)
    for j in (0, 7, 29):
        raw = mra.fit_reml(Y_star[:, j], X, con.V, Gs, names=names,
                           method="cholesky", n_starts=1)
        row = fit.replicates.iloc[j]
        assert row["M_star"] == raw["beta"][0]
        assert [row[n] for n in names] == list(raw["tau2"])


def test_hybrid_fits_are_seeded_and_do_not_depend_on_n_boot(toy_hybrid):
    """Acceptance 7: same spec, data and seed -> identical replicate frames."""
    reg = toy_hybrid["reg"]
    again = fm.ModelRegistry(toy_hybrid["beta"])
    again.fit("A0c_hb", n_boot=30)
    pd.testing.assert_frame_equal(again["A0c_hb"].replicates,
                                  reg["A0c_hb"].replicates, check_exact=True)

    reseeded = fm.ModelRegistry(toy_hybrid["beta"])
    reseeded.fit("A0c_hb", n_boot=30, boot_seed=1)
    assert not reseeded["A0c_hb"].replicates["M_star"].equals(
        reg["A0c_hb"].replicates["M_star"])
    assert reseeded["A0c_hb"].boot["seed"] == 1

    shorter = fm.ModelRegistry(toy_hybrid["beta"])
    shorter.fit("A0c_hb", n_boot=12)
    pd.testing.assert_frame_equal(shorter["A0c_hb"].replicates,
                                  reg["A0c_hb"].replicates.iloc[:12], check_exact=True)

    # The joint registry, likewise.
    joint = toy_hybrid["joint"]
    j2 = fm.JointModelRegistry(toy_hybrid["theta"], toy_hybrid["beta"])
    j2.fit("A8a_hb", n_boot=12)
    pd.testing.assert_frame_equal(j2["A8a_hb"].replicates,
                                  joint["A8a_hb"].replicates.iloc[:12], check_exact=True)
    np.testing.assert_array_equal(joint["A8a"].beta, joint["A8a_hb"].beta)


def test_hybrid_bootstrap_needs_the_full_sample_fit(toy_hybrid):
    """Calling the bootstrap functions directly without the fit explains itself."""
    reg, joint = toy_hybrid["reg"], toy_hybrid["joint"]
    with pytest.raises(ValueError, match="needs the full-sample fit"):
        fm.bootstrap_fits(reg["A0c_hb"].contrast, fm.MODEL_SPECS["A0c_hb"],
                          estimator="REML", n_boot=3)
    with pytest.raises(ValueError, match="needs the full-sample fit"):
        fm.bootstrap_joint_fits(joint["A8a_hb"].contrast, fm.JOINT_MODEL_SPECS["A8a_hb"],
                                n_boot=3)


def test_hybrid_scheme_is_visible_everywhere_univariate(toy_hybrid):
    """Acceptance 8: nobody can mistake hybrid replicates for record-only ones."""
    reg = toy_hybrid["reg"]
    rec, hyb = reg["A0c"], reg["A0c_hb"]
    assert (rec.boot_scheme, hyb.boot_scheme) == ("record", "hybrid")
    assert hyb.hybrid and not rec.hybrid
    assert fm.HB_TAG in hyb.summary() and fm.HB_TAG not in rec.summary()
    assert "POPULATION" in hyb.summary() and "POPULATION" not in rec.summary()
    assert (rec.to_row()["bootstrap"], hyb.to_row()["bootstrap"]) == ("record", "hybrid")
    assert list(reg.table().loc[["A0c", "A0c_hb"], "bootstrap"]) == ["record", "hybrid"]
    assert rec.replicates.attrs["bootstrap_scheme"] == "record"
    assert hyb.replicates.attrs == {"bootstrap_scheme": "hybrid", "seed": fm.BOOT_SEED}

    # No bootstrap at all reads as None, not as either scheme.
    bare = fm.ModelRegistry(toy_hybrid["beta"])
    bare.fit("A0c_hb")
    assert bare["A0c_hb"].boot_scheme is None
    assert bare["A0c_hb"].to_row()["bootstrap"] is None
    assert fm.HB_TAG not in bare["A0c_hb"].summary()

    # The forest title carries the tag too: its CI band changes meaning.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert fm.HB_TAG in reg.forest("A0c_hb").get_title()
    assert fm.HB_TAG not in reg.forest("A0c").get_title()
    plt.close("all")


def test_tau_ci_and_boundary_fractions(toy_hybrid):
    """An interval on a tau comes from hybrid replicates, and says how it was made."""
    reg = toy_hybrid["reg"]
    hyb = reg["A0c_hb"]
    ci = hyb.tau_ci("total")
    assert ci["scheme"] == "hybrid" and ci["lo"] <= ci["hi"]
    assert ci["estimate"] == pytest.approx(hyb.T_sq)
    bf = hyb.boundary_fractions()
    assert set(bf.index) == set(hyb.tau2) and ((bf >= 0) & (bf <= 1)).all()
    for name in hyb.tau2:
        assert hyb.tau_ci(name)["frac_at_zero"] == bf[name]
    # A component fitted at zero was drawn at zero, and the fit says which.
    assert hyb.boot["zero_levels"] == tuple(k for k, v in hyb.tau2.items() if v == 0.0)
    with pytest.raises(KeyError, match="no variance component"):
        hyb.tau_ci("tau_x2")

    # On record-only replicates the same call warns: they are inflated.
    with pytest.warns(RuntimeWarning, match="RECORD-ONLY"):
        assert reg["A0c"].tau_ci("total")["scheme"] == "record"


def test_hybrid_flag_also_drives_the_DL_bootstrap():
    """H6: a spec that runs DL and REML gets the hybrid bootstrap for both.

    The specs are built locally (no `A0a_hb` in the shipped table) and checked
    with the same rules. DL draws its effects from its own T^2, so the two
    estimators see the same standard-normal draws scaled by their own tau.
    """
    theta, _ = _toy_joint()
    base = fm.MODEL_SPECS["A0a"]
    specs = {"A0a": base,
             "A0a_hb": _replace(base, model_id="A0a_hb", bootstrap_scheme="hybrid")}
    fm._check_model_specs(specs)
    reg = fm.ModelRegistry(theta, specs=specs)
    reg.fit("A0a", n_boot=25)
    reg.fit("A0a_hb", n_boot=25)
    for est in ("DL", "REML"):
        rec, hyb = reg["A0a", est], reg["A0a_hb", est]
        assert hyb.hybrid and not rec.hybrid
        assert not rec.replicates["T_sq"].equals(hyb.replicates["T_sq"])
        # Centred on the estimate: the hybrid cloud is X beta_hat + zero-mean terms.
        assert abs(hyb.replicates["M_star"].mean() - hyb.m) < 4 * hyb.se_boot / 5
    again = fm.ModelRegistry(theta, specs=specs)
    again.fit("A0a_hb", n_boot=25)
    pd.testing.assert_frame_equal(again["A0a_hb", "DL"].replicates,
                                  reg["A0a_hb", "DL"].replicates, check_exact=True)


def test_hybrid_scheme_is_visible_everywhere_joint(toy_hybrid):
    """Acceptance 8, joint: the "PROVISIONAL" caveat is printed for record-only fits ONLY."""
    joint = toy_hybrid["joint"]
    rec, hyb = joint["A8a"], joint["A8a_hb"]
    assert "PROVISIONAL" in rec.summary()
    assert "PROVISIONAL" not in hyb.summary()
    assert fm.HB_TAG in hyb.summary() and fm.HB_TAG not in rec.summary()
    assert (rec.to_row()["bootstrap"], hyb.to_row()["bootstrap"]) == ("record", "hybrid")
    assert list(joint.table().loc[["A8a", "A8a_hb"], "bootstrap"]) == ["record", "hybrid"]
    assert (rec.rho_ci("tot")["scheme"], hyb.rho_ci("tot")["scheme"]) == ("record", "hybrid")
    # The frame that leaves the module for the MAFC step says how it was made.
    assert hyb.mafc_draws().attrs == {"bootstrap_scheme": "hybrid", "seed": fm.BOOT_SEED}
    assert rec.mafc_draws().attrs["bootstrap_scheme"] == "record"
    assert hyb.replicates.columns.equals(rec.replicates.columns)


def test_joint_hybrid_replicates_are_the_array_level_path(toy_hybrid):
    """The joint registry's hybrid bootstrap IS ``dsim.hybrid_pseudo_data_mv`` + ``fit_reml_mv``."""
    fit = toy_hybrid["joint"]["A8a_hb"]
    jcon, spec = fit.contrast, fit.spec
    X, _, _ = fm.build_X_joint(spec, jcon)
    factors = fm.joint_level_factors(spec, jcon)
    Y_star = dsim.hybrid_pseudo_data_mv(
        jcon.y.to_numpy(float), X @ fit.beta,
        {lev: (factors[lev], fit.T[lev]) for lev in fit.levels},
        seed=fm.BOOT_SEED, n_boot=30)
    Gs = fm.joint_level_gs(spec, jcon)
    for j in (0, 29):
        raw = mra.fit_reml_mv(Y_star[:, j], X, jcon.V, Gs, spec.correlated,
                              n_starts=1, T0=fit.T)
        row = fit.replicates.iloc[j]
        assert row["m_theta"] == raw["beta"][0] and row["m_beta"] == raw["beta"][1]
        assert row["T_total_bb"] == raw["T_total"][1, 1]


# -- the same on the real pipeline output -------------------------------------

def _on_estimate(mean: float, mcse: float, estimate: float) -> bool:
    """Hybrid replicate mean within 4 MCSE or 5% of the full-sample value (the looser)."""
    return abs(mean - estimate) < max(4 * mcse, 0.05 * estimate)


@needs_data
def test_real_hybrid_bootstrap_removes_the_tau2_inflation(real_theta, real_beta):
    """Acceptance 9: full-sample tau^2, record-only mean and hybrid mean, side by side.

    200 replicates per fit (the task's replicate cap is 250).

    ASSERTED on a single-level (A0b-type) fit of each quantity, whose one
    variance component is well away from zero: the hybrid replicate mean is
    within 4 MCSE or 5% of the full-sample value, and for beta -- where the
    sampling variance is five times tau^2 -- the record-only mean is more than
    double it. `A0b_hb` has no entry in the shipped table, so it is built here.

    PRINTED, not asserted, for `A0c_hb` on theta and on beta (the latter is the
    roadmap's A4c). Its beta fit has two components at zero, which a refit can
    only move up, so its hybrid total is expected a few percent high; the
    boundary fractions are printed beside it.
    """
    n_boot = 200
    specs = dict(fm.MODEL_SPECS)
    specs["A0b_hb"] = _replace(specs["A0b"], model_id="A0b_hb", bootstrap_scheme="hybrid")
    fm._check_model_specs(specs)

    print()
    for q, data in (("theta", real_theta), ("beta", real_beta)):
        reg = fm.ModelRegistry(data, specs=specs)
        for base in ("A0b", "A0c"):
            reg.fit(base, n_boot=n_boot)
            reg.fit(base + "_hb", n_boot=n_boot)
            rec, hyb = reg[base], reg[base + "_hb"]
            r, h = rec.replicates["T_sq"], hyb.replicates["T_sq"]
            mcse = h.std(ddof=1) / np.sqrt(n_boot)
            print(f"{base:4s} {q:5s}: tau2 total {hyb.T_sq:.5f} | record-only mean "
                  f"{r.mean():.5f} | hybrid mean {h.mean():.5f} (MCSE {mcse:.5f}) | "
                  f"at zero {hyb.boundary_fractions().round(2).to_dict()}")
            if base == "A0b":
                assert _on_estimate(h.mean(), mcse, hyb.T_sq), (q, h.mean(), hyb.T_sq)
                assert r.mean() > hyb.T_sq
                if q == "beta":
                    assert r.mean() > 2 * hyb.T_sq

    joint = fm.JointModelRegistry(real_theta, real_beta)
    joint.fit("A8a", n_boot=n_boot)
    joint.fit("A8a_hb", n_boot=n_boot)
    rec, hyb = joint["A8a"], joint["A8a_hb"]
    for key, idx in (("tt", (0, 0)), ("bb", (1, 1))):
        est = float(hyb.T_total[idx])
        r, h = rec.replicates[f"T_total_{key}"], hyb.replicates[f"T_total_{key}"]
        mcse = h.std(ddof=1) / np.sqrt(n_boot)
        print(f"A8a  T_{key}  : fitted {est:.5f} | record-only mean {r.mean():.5f} | "
              f"hybrid mean {h.mean():.5f} (MCSE {mcse:.5f})")
        assert _on_estimate(h.mean(), mcse, est), (key, h.mean(), est)
        assert r.mean() > est
    rc, hc = rec.rho_ci("tot"), hyb.rho_ci("tot")
    print(f"A8a  rho_tot: {hyb.rho_tot:+.3f} | record-only 95% {rc['lo']:+.3f} .. "
          f"{rc['hi']:+.3f} | hybrid 95% {hc['lo']:+.3f} .. {hc['hi']:+.3f}")
    # The honest interval is the wider one, and it contains the estimate.
    assert hc["hi"] - hc["lo"] > rc["hi"] - rc["lo"]
    assert hc["lo"] < hyb.rho_tot < hc["hi"]
