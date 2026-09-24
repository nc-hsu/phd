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
    for model_id in ("A0a", "A1a", "A2"):
        fits.fit(model_id)
    table = fits.table()
    assert len(table) == 6
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
