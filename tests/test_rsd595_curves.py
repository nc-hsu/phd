"""Unit tests for ``phd_project.scripts.rsd595_curves`` (nb 081).

Covered: the IML grid, the per-site ceilings, the slim disaggregation frame
(that it gives pickagm exactly the same occurrence weights as the full frame),
the occurrence-sum check, the curve tables' CSV round trip, and the
interpolation helper. Nothing here needs OpenQuake or the project data.
"""

import numpy as np
import pandas as pd
import pytest

from phd_project.scripts import rsd595_curves as rc


# --------------------------------------------------------------------------- #
# grid                                                                         #
# --------------------------------------------------------------------------- #

def test_grid_sorted_unique_6sf_with_anchors():
    g = rc.build_iml_grid(0.01, 3.0, 37, anchors=(0.27, 0.5, 0.95))
    assert len(g) == 40
    assert g == sorted(g) and len(set(g)) == len(g)
    assert g[0] == 0.01 and g[-1] == 3.0
    assert all(v == float(f"{v:.6g}") for v in g)
    assert {0.27, 0.5, 0.95} <= set(g)


def test_grid_refuses_coinciding_anchor():
    with pytest.raises(ValueError, match="coincide"):
        rc.build_iml_grid(0.01, 3.0, 37, anchors=(0.01,))


def test_grid_file_round_trip_and_stable(tmp_path):
    from phd_project.scripts import oq_runner
    g = rc.build_iml_grid(0.01, 3.0, 5)
    fp = rc.write_iml_grid(g, tmp_path / "grid.csv")
    mtime = fp.stat().st_mtime_ns
    back = [float(oq_runner._iml_str(v)) for v in oq_runner.load_imls(fp)]
    assert back == g
    rc.write_iml_grid(g, fp)                      # unchanged -> not rewritten
    assert fp.stat().st_mtime_ns == mtime


# --------------------------------------------------------------------------- #
# ceilings                                                                     #
# --------------------------------------------------------------------------- #

def test_hazard_curve_ceiling():
    hc = {0: {"AvgSA": {"mean": [[0.1, 1e-2], [1.0, 1e-6], [3.0, 0.0]]}},
          1: {"AvgSA": {"mean": [[0.1, 1e-2], [1.0, 1e-6], [3.0, 1e-9]]}}}
    s = rc.hazard_curve_ceilings(hc)
    assert s.loc[0] == 1.0 and s.loc[1] == 3.0


def test_disagg_ceilings_and_hole():
    grid = [0.1, 0.2, 0.4, 0.8]
    t = rc.disagg_ceilings(grid, {1: [0.8], 2: [0.4, 0.8]}, sites=[0, 1, 2])
    assert t.loc[0, "ceiling_disagg_g"] == 0.8 and t.loc[0, "n_kept"] == 4
    assert t.loc[1, "ceiling_disagg_g"] == 0.4
    assert t.loc[2, "n_kept"] == 2
    with pytest.raises(ValueError, match="hole"):
        rc.disagg_ceilings(grid, {0: [0.2]}, sites=[0])


# --------------------------------------------------------------------------- #
# slim frames                                                                  #
# --------------------------------------------------------------------------- #

def _full_frame(seed=0, zero_bins=True):
    """A synthetic nb 021 frame: 2 TRTs x 3 Mag x 2 Dist x 6 Eps, one Z, with
    some bins that cannot exceed x (nu_m = 0)."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [["Shallow Default", "Volcanic"], [5.0, 6.0, 7.0], [10.0, 30.0],
         [-2.0, -0.67, 0.67, 2.0, 3.33, -3.33]], names=["TRT", "Mag", "Dist", "Eps"])
    df = idx.to_frame(index=False)
    df["Z"] = 0.0
    nu = rng.random(len(df)) * 1e-4
    if zero_bins:
        nu[::5] = 0.0
    occ_num = nu * rng.random(len(df))
    df["P(X>x|T,m)"] = rng.random(len(df))
    df["nu_m"] = nu
    df["P(m|X>x)"] = nu / nu.sum()
    df["P(m|X=x)"] = occ_num / occ_num.sum()
    return df


def _pickagm_weights(df):
    """What pickagm's _conditional_im_distributions does to the frame first,
    restricted to the columns it goes on to read (Eps via get_rup_ctxs)."""
    d = df.groupby(["TRT", "Mag", "Dist"]).sum().reset_index()
    d = d.loc[d["P(m|X=x)"] != 0, :]
    return d[["TRT", "Mag", "Dist", "Eps", "P(m|X=x)"]].reset_index(drop=True)


def test_slim_gives_pickagm_identical_occurrence_weights():
    full = _full_frame()
    slim = rc.slim_disagg_frame(full)
    a, b = _pickagm_weights(full), _pickagm_weights(slim)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_slim_frame_feeds_pickagm_rupture_contexts():
    """pickagm reads TRT, Mag, Dist and Eps by name to build the rupture
    contexts; a slim frame must carry all four, with the same values the full
    frame would give after pickagm's epsilon groupby."""
    from pickagm.contexts import get_rup_ctxs
    full = _full_frame()
    slim = rc.slim_disagg_frame(full)
    a = get_rup_ctxs("TRT_Mag_Dist_Eps", _pickagm_weights(full))
    b = get_rup_ctxs("TRT_Mag_Dist_Eps", _pickagm_weights(slim))
    assert a == b and len(a) > 0


def test_slim_keeps_columns_and_totals():
    full = _full_frame()
    slim = rc.slim_disagg_frame(full)
    assert list(slim.columns) == ["TRT", "Mag", "Dist", "Z", "Eps",
                                  "nu_m", "P(m|X>x)", "P(m|X=x)"]
    assert (slim["nu_m"] > 0).all()
    assert len(slim) < len(full) / 5
    for c in ["nu_m", "P(m|X>x)", "P(m|X=x)"]:
        assert np.isclose(slim[c].sum(), full[c].sum(), rtol=1e-14)


def test_slim_zero_hazard_frame_is_empty_and_excluded():
    from phd_project.scripts import oqhelpers
    full = _full_frame()
    full["nu_m"] = 0.0
    full["P(m|X>x)"] = np.nan
    full["P(m|X=x)"] = np.nan
    slim = rc.slim_disagg_frame(full)
    assert slim.empty
    assert oqhelpers._is_zero_hazard(slim, mafe=1e-9)


def test_slim_keeps_nan_occurrence_nan():
    full = _full_frame(zero_bins=False)
    full["P(m|X=x)"] = np.nan                 # occurrence could not be formed
    slim = rc.slim_disagg_frame(full)
    assert slim["P(m|X=x)"].isna().all()


def test_occurrence_sum_error():
    good = rc.slim_disagg_frame(_full_frame())
    data = {0: {"AvgSA": {0.1: good}}}
    assert rc.occurrence_sum_error(data, {}) < 1e-12
    bad = good.assign(**{"P(m|X=x)": np.nan})
    data[0]["AvgSA"][0.2] = bad
    assert rc.occurrence_sum_error(data, {}) == np.inf
    assert rc.occurrence_sum_error(data, {0: [0.2]}) < 1e-12     # excluded: skipped


# --------------------------------------------------------------------------- #
# curve tables                                                                 #
# --------------------------------------------------------------------------- #

def _gcim(site, iml, mean, sigma):
    stats = pd.DataFrame({"mean": [mean], "sigma": [sigma]}, index=["RSD595"])
    return {(site, iml): {"stats": stats}}


def test_curves_from_gcim_and_round_trip(tmp_path):
    grid = rc.build_iml_grid(0.01, 3.0, 37, anchors=(0.27, 0.5, 0.95))
    g = {}
    for s in (0, 1):
        for i, x in enumerate(grid):
            if s == 1 and x > 1.5:                  # site 1 stops at its ceiling
                continue
            g |= _gcim(s, x, 2.0 + 0.1 * np.log(x) + 1e-13 * i, 0.5 - 0.01 * i / 3)
    mu, sigma = rc.curves_from_gcim(g, grid, sites=[0, 1])
    assert mu.loc[1, 3.0] != mu.loc[1, 3.0]          # NaN above the ceiling
    fp = rc.write_curves(mu, tmp_path / "mu.csv")
    back = rc.read_curves(fp)
    assert back.index.dtype.kind == "i" and back.index.name == "site_id"
    assert list(back.columns) == grid                 # exact float keys
    pd.testing.assert_frame_equal(back, mu, check_exact=True, check_names=False)


def test_curves_from_gcim_refuses_off_grid_key():
    with pytest.raises(KeyError):
        rc.curves_from_gcim(_gcim(0, 0.123, 2.0, 0.5), [0.1, 0.2], sites=[0])


# --------------------------------------------------------------------------- #
# interpolation                                                                #
# --------------------------------------------------------------------------- #

def _table():
    cols = [0.1, 0.2, 0.4, 0.8]
    return pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, np.nan, np.nan]],
                        index=pd.Index([0, 1], name="site_id"), columns=cols)


def test_interp_exact_at_nodes():
    t = _table()
    for x, y in zip(t.columns, t.loc[0]):
        assert rc.interp_curve(t, 0, x) == y


def test_interp_linear_in_ln_x():
    t = _table()
    mid = np.sqrt(0.1 * 0.2)                          # midpoint in ln x
    assert np.isclose(rc.interp_curve(t, 0, mid), 1.5, rtol=1e-14)


def test_interp_nan_outside_range_and_above_ceiling():
    t = _table()
    assert np.isnan(rc.interp_curve(t, 0, 0.05))
    assert np.isnan(rc.interp_curve(t, 0, 0.9))
    assert np.isnan(rc.interp_curve(t, 1, 0.3))      # above site 1's last point
    assert rc.interp_curve(t, 1, 0.2) == 2.0          # its last point itself
    assert np.isnan(rc.interp_curve(t, 0, 0.0))


def test_interp_vector():
    out = rc.interp_curve(_table(), 0, [0.05, 0.1, 0.8, 1.0])
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(np.isnan(out), [True, False, False, True])


def test_interp_refuses_hole():
    t = _table()
    t.loc[0, 0.2] = np.nan
    with pytest.raises(ValueError, match="NaN inside"):
        rc.interp_curve(t, 0, 0.3)


def test_rsd595curves_wrapper(tmp_path):
    t = _table()
    fp_mu = rc.write_curves(t, tmp_path / "mu.csv")
    fp_sig = rc.write_curves(t * 0.1, tmp_path / "sig.csv")
    c = rc.RSD595Curves(fp_mu, fp_sig)
    assert c.mu(0, 0.4) == 3.0 and np.isclose(c.sigma(0, 0.4), 0.3)
    assert c.range(1) == (0.1, 0.2)
    bad = t.copy()
    bad.loc[0, 0.8] = np.nan
    fp_bad = rc.write_curves(bad, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="same cells"):
        rc.RSD595Curves(fp_mu, fp_bad)
