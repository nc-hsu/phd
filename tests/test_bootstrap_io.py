"""Round-trip tests for the merged bootstrap CSV layout.

Every bootstrap output now holds all storey counts in one file - the replicate clouds on a
``(unit, n_storeys)`` column MultiIndex, the per-unit frames on the matching row index - so
what has to be guaranteed is that a frame written by nb 070 comes back off disk unchanged,
including the labels and dtypes a CSV cannot carry by itself.
"""

import numpy as np
import pandas as pd
import pytest

import phd_project.scripts.metaregression_analysis as mra

IM_TAG = "AvgSA_03"
K = 8
N_STOREYS = [3, 5]


def cloud(units, n, dtype=float):
    """One storey count's replicate cloud, indexed by k and columned by site or group."""
    if dtype is bool:
        values = np.ones((K, len(units)), dtype=bool)
        values[0, 0] = False
    else:
        values = np.arange(K * len(units), dtype=float).reshape(K, len(units)) + n
    df = pd.DataFrame(values, columns=pd.Index(units, name="site" if isinstance(
        units[0], int) else "group"))
    df.index.name = "k"
    return df


def unit_frame(units, n):
    """One storey count's per-unit frame, the shape of a stats or bias frame."""
    df = pd.DataFrame({"bias": np.linspace(-0.01, 0.01, len(units)) + n / 100,
                       "se_est": np.linspace(0.04, 0.05, len(units)),
                       "N_obs": float(K)},
                      index=pd.Index(units, name="site" if isinstance(
                          units[0], int) else "group"))
    return df


@pytest.fixture(params=["by_site", "by_group"])
def scope(request):
    return request.param


def units_for(scope):
    return [0, 1, 2] if scope == "by_site" else ["group_3s_00", "group_3s_01"]


def test_cloud_round_trip(tmp_path, scope):
    """A merged cloud reloads with its labels, dtypes and replicate index intact."""
    units = units_for(scope)
    merged = mra.stack_cloud_frames({n: cloud(units, n) for n in N_STOREYS})
    merged.to_csv(mra.bootstrap_csv_path(tmp_path, "ida_femap695", "theta", IM_TAG, scope))

    back = mra.load_saved_bootstrap(tmp_path, "ida_femap695", ("theta",), None,
                                    IM_TAG, scope, k_samples=K)
    pd.testing.assert_frame_equal(back["theta"], merged)
    assert back["theta"].columns.names == [scope.removeprefix("by_"), "n_storeys"]


def test_cloud_bool_dtype_survives(tmp_path):
    """The fit_ok masks are boolean and must not come back as strings or objects."""
    units = units_for("by_group")
    merged = mra.stack_cloud_frames({n: cloud(units, n, dtype=bool) for n in N_STOREYS})
    merged.to_csv(mra.bootstrap_csv_path(tmp_path, "msa_femap695", "fit_ok", IM_TAG,
                                         "by_group"))

    back = mra.load_saved_bootstrap(tmp_path, "msa_femap695", ("fit_ok",), None, IM_TAG,
                                    "by_group")
    assert (back["fit_ok"].dtypes == bool).all()
    pd.testing.assert_frame_equal(back["fit_ok"], merged)


def test_scalar_selection_drops_the_level(tmp_path, scope):
    """Asking for one storey count gives back the single-level frame 073/076 still use."""
    units = units_for(scope)
    per_storey = {n: cloud(units, n) for n in N_STOREYS}
    mra.stack_cloud_frames(per_storey).to_csv(
        mra.bootstrap_csv_path(tmp_path, "ida_femap695", "theta", IM_TAG, scope))

    back = mra.load_saved_bootstrap(tmp_path, "ida_femap695", ("theta",), 3, IM_TAG,
                                    scope, k_samples=K)["theta"]
    assert not isinstance(back.columns, pd.MultiIndex)
    pd.testing.assert_frame_equal(back, per_storey[3])


def test_sequence_selection_keeps_the_level(tmp_path):
    """A subset of storey counts keeps the level, so the frame stays self-describing."""
    units = units_for("by_site")
    merged = mra.stack_cloud_frames({n: cloud(units, n) for n in [3, 5]})
    merged.to_csv(mra.bootstrap_csv_path(tmp_path, "site_msa", "theta", IM_TAG, "by_site"))

    back = mra.load_saved_bootstrap(tmp_path, "site_msa", ("theta",), [5], IM_TAG,
                                    "by_site")["theta"]
    assert back.columns.names == ["site", "n_storeys"]
    assert set(back.columns.get_level_values("n_storeys")) == {5}


def test_missing_storey_count_reads_as_missing(tmp_path):
    """A storey count absent from the file must not half-satisfy the REUSE_SAVED path."""
    units = units_for("by_site")
    mra.stack_cloud_frames({3: cloud(units, 3)}).to_csv(
        mra.bootstrap_csv_path(tmp_path, "site_msa", "theta", IM_TAG, "by_site"))

    assert mra.load_saved_bootstrap(tmp_path, "site_msa", ("theta",), 5, IM_TAG,
                                    "by_site") is None


def test_stats_round_trip(tmp_path, scope):
    """The stats frames carry (unit, n_storeys) on the index and reload unchanged."""
    units = units_for(scope)
    merged = mra.stack_unit_frames({n: unit_frame(units, n) for n in N_STOREYS})
    merged.to_csv(mra.bootstrap_csv_path(tmp_path, "ida_femap695", "theta_stats", IM_TAG,
                                         scope))

    back = mra.load_saved_bootstrap_stats(tmp_path, "ida_femap695", ("theta_stats",),
                                          None, IM_TAG, scope)["theta_stats"]
    assert back.index.names == [scope.removeprefix("by_"), "n_storeys"]
    pd.testing.assert_frame_equal(back, merged, check_names=False)


def test_bias_round_trip(tmp_path, scope):
    """The bias frames follow the same convention, selectable by storey count."""
    units = units_for(scope)
    per_storey = {n: unit_frame(units, n) for n in N_STOREYS}
    mra.stack_unit_frames(per_storey).to_csv(
        mra.bias_csv_path(tmp_path, "ida_femap695", "beta", IM_TAG, scope))

    full = mra.load_saved_bias(tmp_path, ["ida_femap695"], ("beta",), None, IM_TAG,
                               scope)["ida_femap695"]["beta"]
    assert full.index.names == [scope.removeprefix("by_"), "n_storeys"]

    one = mra.load_saved_bias(tmp_path, ["ida_femap695"], ("beta",), 5, IM_TAG,
                              scope)["ida_femap695"]["beta"]
    pd.testing.assert_frame_equal(one, per_storey[5], check_names=False)


def test_estimates_round_trip(tmp_path, scope):
    """The estimates layout is unchanged by the merge - it was already storey-agnostic."""
    units = units_for(scope)
    frames = {n: pd.DataFrame({"theta": np.linspace(0.3, 0.5, len(units)),
                               "beta": np.linspace(0.2, 0.3, len(units)),
                               "extra": 1.0},
                              index=pd.Index(units, name=scope.removeprefix("by_")))
              for n in N_STOREYS}
    merged = mra.stack_estimates(frames)
    assert list(merged.columns) == ["theta", "beta"], "stack_estimates must not leak columns"
    merged.to_csv(mra.estimates_csv_path(tmp_path, "site_msa", IM_TAG, scope))

    back = mra.load_estimates(tmp_path, IM_TAG, scope=scope, arms=["site_msa"])["site_msa"]
    pd.testing.assert_frame_equal(back, merged)


def test_cloud_and_unit_frames_align_row_for_row(tmp_path):
    """A transposed cloud and a per-unit frame must share an index, which the models rely on."""
    units = units_for("by_site")
    merged_cloud = mra.stack_cloud_frames({n: cloud(units, n) for n in N_STOREYS})
    merged_units = mra.stack_unit_frames({n: unit_frame(units, n) for n in N_STOREYS})

    pd.testing.assert_index_equal(mra.reformat_bootstrap_df(merged_cloud).index,
                                  merged_units.index)


def test_reformat_still_accepts_a_single_level_frame():
    """073 and 076 pass a storey count and a flat frame; that path has to keep working."""
    units = units_for("by_site")
    flat = cloud(units, 3)
    out = mra.reformat_bootstrap_df(flat, 3)
    assert out.index.names == ["site", "n_storeys"]
    assert set(out.index.get_level_values("n_storeys")) == {3}


def test_select_storeys_names_what_it_has():
    """A wrong storey count should say what is actually in the frame."""
    units = units_for("by_site")
    merged = mra.stack_cloud_frames({3: cloud(units, 3)})
    with pytest.raises(KeyError, match=r"\[3\]"):
        mra.select_storeys(merged, 5, axis=1)
