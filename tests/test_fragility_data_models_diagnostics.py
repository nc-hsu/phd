"""Tests for :func:`fmd.plot_whitened_residuals_vs_covariates`.

The figure is checked through its artists: which panels exist, what data each
scatter and running-mean line carries, and what the in/out note says. All of it
runs on the toy containers of ``test_fragility_data_models`` (20 sites x 2
storeys, four ``_A3AE`` covariate columns), so nothing here needs the real data.

Two test-local joint specs give the note something to distinguish:
    A8ax  theta and beta both use all four _A3AE covariates;
    A8fx  theta uses all four, beta only the last -- the toy analogue of A8f,
          whose theta and beta covariate lists differ.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                     # no display: draw to memory only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import phd_project.scripts.fragility_data_models as fm
import phd_project.scripts.fragility_data_models_diagnostics as fmd

from test_fragility_data_models import _A3AE, _JOINT_SPECS_WITH_COVARIATES, _toy_joint

_A8FX = fm.JointModelSpec("A8fx", "y0_avg", components=(), correlated=("row",),
                          covariates_theta=_A3AE, covariates_beta=_A3AE[3:])
_SPECS = {**_JOINT_SPECS_WITH_COVARIATES, "A8fx": _A8FX}


@pytest.fixture(scope="module")
def joint():
    """Joint registry with A8ax and A8fx fitted once (few replicates: speed only)."""
    theta, beta = _toy_joint()
    reg = fm.JointModelRegistry(theta, beta, _SPECS)
    for mid in ("A8ax", "A8fx"):
        reg.fit(mid, n_boot=20)
    return reg


@pytest.fixture(scope="module")
def group_fit():
    """Univariate A2: a group-scope contrast (one row per design group)."""
    theta, _ = _toy_joint()
    reg = fm.ModelRegistry(theta)
    reg.fit("A2", n_boot=20)
    return reg["A2"]


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test leaves no figure open, whatever it asserts."""
    yield
    plt.close("all")


def _drawn(fig):
    """Axes that are switched on, in grid order."""
    return [ax for ax in fig.axes if ax.axison]


def _note(ax):
    """Text of the in/out note above an axes, or None if there is none.

    The note is the only Text artist the function adds in axes coordinates
    (titles and labels are not in ``ax.texts``).
    """
    texts = [t.get_text() for t in ax.texts]
    return texts[0] if texts else None


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #

def test_one_panel_per_covariate_and_spare_axes_off(joint):
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE, ncols=3)
    assert len(fig.axes) == 6                       # 2 x 3 grid
    on = _drawn(fig)
    assert len(on) == 4
    assert [ax.get_xlabel() for ax in on] == list(_A3AE)
    assert not fig.axes[4].axison and not fig.axes[5].axison
    # ncols wider than the list: a single row of exactly four
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE, ncols=5)
    assert len(fig.axes) == 4


def test_a_single_string_is_one_covariate(joint):
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE[0])
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == _A3AE[0]


def test_y_label_on_the_first_column_only_when_shared(joint):
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE, ncols=2)
    assert [bool(ax.get_ylabel()) for ax in fig.axes] == [True, False, True, False]
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE,
                                                       ncols=2, sharey=False)
    assert all(ax.get_ylabel() for ax in fig.axes)


def test_draws_into_given_axes_without_touching_the_layout(joint):
    fig, axes = plt.subplots(2, 3)
    out, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE[:2],
                                                       axes=axes)
    assert out is fig
    assert fig._suptitle is None
    assert all(ax.axison for ax in axes.ravel())     # spare axes left alone
    assert len(axes[0, 0].collections) > 0 and len(axes[0, 2].collections) == 0


def test_too_few_axes_raises(joint):
    fig, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="need 4 axes"):
        fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE, axes=axes)


# --------------------------------------------------------------------------- #
# Data on the panels                                                          #
# --------------------------------------------------------------------------- #

def test_scatter_holds_the_covariate_and_r_white_per_storey(joint):
    model = joint["A8fx"]
    fig, res = fmd.plot_whitened_residuals_vs_covariates(model, "beta", _A3AE)
    vw = fmd.fit_view(model, "beta")
    expected = fmd.whitened_residuals(model, "beta")
    pd.testing.assert_frame_equal(res, expected)
    storeys = vw.storeys
    for ax, cov in zip(_drawn(fig), _A3AE):
        x = vw.covariate_values(cov).astype(float)
        scatters = ax.collections[:2]               # one per storey: 3 then 5
        for sc, ns in zip(scatters, (3, 5)):
            sel = storeys == ns
            np.testing.assert_array_equal(sc.get_offsets()[:, 0], x[sel])
            np.testing.assert_array_equal(sc.get_offsets()[:, 1],
                                          expected["r_white"].to_numpy()[sel])


def test_running_mean_is_the_centred_rolling_mean_in_x_order(joint):
    fig, res = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE[1],
                                                         running_window=6)
    vw = fmd.fit_view(joint["A8ax"], "theta")
    x = vw.covariate_values(_A3AE[1]).astype(float)
    r = res["r_white"].to_numpy()
    o = np.argsort(x)
    want = pd.Series(r[o]).rolling(6, center=True, min_periods=3).mean().to_numpy()
    line = next(l for l in fig.axes[0].lines if l.get_label().startswith("running mean"))
    np.testing.assert_array_equal(line.get_xdata(), x[o])
    np.testing.assert_allclose(line.get_ydata(), want, equal_nan=True)


def test_whitened_residuals_computed_once_per_model(joint, monkeypatch):
    calls = []
    real = fmd.whitened_residuals

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(fmd, "whitened_residuals", counting)
    fmd.plot_whitened_residuals_vs_covariates([joint["A8ax"], joint["A8fx"]], "theta", _A3AE)
    assert len(calls) == 2


def test_one_legend_with_one_crit_line_entry(joint):
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE)
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert len(legends) == 1 and legends[0] is fig.axes[0].get_legend()
    texts = [t.get_text() for t in legends[0].get_texts()]
    assert sum("limit" in t for t in texts) == 1


# --------------------------------------------------------------------------- #
# The in/out note                                                             #
# --------------------------------------------------------------------------- #

def test_note_differs_between_theta_and_beta_of_a_joint_fit(joint):
    fig_t, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8fx"], "theta", _A3AE)
    fig_b, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8fx"], "beta", _A3AE)
    assert [_note(ax) for ax in _drawn(fig_t)] == [
        "(A)  in model", "(B)  in model", "(C)  in model", "(D)  in model"]
    assert [_note(ax) for ax in _drawn(fig_b)] == [
        "(A)  not in model", "(B)  not in model", "(C)  not in model", "(D)  in model"]


def test_note_default_colours(joint):
    from matplotlib.colors import to_rgba
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8fx"], "beta", _A3AE)
    out_note, in_note = fig.axes[0].texts[0], fig.axes[3].texts[0]
    assert to_rgba(out_note.get_color()) == to_rgba("#ff0000")
    assert out_note.get_fontweight() == "bold"
    assert to_rgba(in_note.get_color()) == to_rgba("k")


def test_mixed_models_note_names_them(joint):
    fig, _ = fmd.plot_whitened_residuals_vs_covariates([joint["A8ax"], joint["A8fx"]], "beta",
                                                       _A3AE)
    notes = [_note(ax) for ax in _drawn(fig)]
    assert notes[0] == "(A)  in: A8ax · not in: A8fx"
    assert notes[3] == "(D)  in model"
    assert fig._suptitle.get_text() == "A8ax, A8fx: whitened residuals vs covariates (beta)"


def test_note_can_be_hidden_and_the_letter_survives(joint):
    for kw in ({"style": {"model_note": None}}, {"model_note": False}):
        fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE, **kw)
        assert all(_note(ax) is None for ax in _drawn(fig))
        assert fig.axes[0].get_title(loc="left") == "(A)"
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE,
                                                       model_note=False, panel_letters=False)
    assert fig.axes[0].get_title(loc="left") == ""


def test_note_restyled_per_status_and_per_covariate(joint):
    from matplotlib.colors import to_rgba
    style = {"model_note_status": {"out": {"s": "OMITTED"}},
             "model_note_covariates": {_A3AE[1]: {"s": "hand-written", "color": "b"},
                                       _A3AE[2]: None}}
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8fx"], "beta", _A3AE,
                                                       style=style, panel_letters=False)
    on = _drawn(fig)
    assert [_note(ax) for ax in on] == ["OMITTED", "hand-written", None, "in model"]
    # the per-covariate layer beats the status default colour
    assert to_rgba(on[1].texts[0].get_color()) == to_rgba("b")


def test_note_sits_just_above_the_top_left_corner(joint):
    fig, _ = fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE[0])
    t = fig.axes[0].texts[0]
    assert t.get_position() == (0.0, 1.01)
    assert t.get_transform() == fig.axes[0].transAxes
    assert (t.get_ha(), t.get_va()) == ("left", "bottom")
    assert fig.axes[0].get_title(loc="left") == ""          # no clash with a left title


# --------------------------------------------------------------------------- #
# Errors and warnings                                                         #
# --------------------------------------------------------------------------- #

def test_unknown_covariate_raises_before_drawing(joint):
    fig, axes = plt.subplots(1, 4)
    bad = [*_A3AE[:2], "no_such_column", _A3AE[3]]
    with pytest.raises(KeyError, match="no_such_column"):
        fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", bad, axes=axes)
    assert all(not ax.collections and not ax.lines for ax in axes)
    # without axes: no figure is created at all
    before = plt.get_fignums()
    with pytest.raises(KeyError, match="Available"):
        fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", bad)
    assert plt.get_fignums() == before


def test_non_numeric_covariate_raises(joint):
    with pytest.raises(ValueError, match="not numeric"):
        fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", ["design_group_id"])


def test_empty_covariate_list_raises(joint):
    with pytest.raises(ValueError, match="no covariates"):
        fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", [])


def test_group_scope_warns_for_a_site_level_covariate(group_fit):
    assert group_fit.contrast.scope == "group"
    with pytest.warns(UserWarning, match="group-scope"):
        fmd.plot_whitened_residuals_vs_covariates(group_fit, None, _A3AE[0])


def test_group_scope_is_silent_for_a_design_level_covariate(group_fit, recwarn):
    fmd.plot_whitened_residuals_vs_covariates(group_fit, None, "n_storeys")
    assert not [w for w in recwarn if "group-scope" in str(w.message)]


def test_save_path_writes_a_png(joint, tmp_path):
    fp = tmp_path / "grid.png"
    fmd.plot_whitened_residuals_vs_covariates(joint["A8ax"], "theta", _A3AE, save_path=fp)
    assert fp.exists() and fp.stat().st_size > 0
