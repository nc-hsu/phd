"""Unit tests for ``phd_project.scripts.oq_multi_iml_disagg`` that need no OpenQuake run.

These cover the pieces whose correctness does not depend on a datastore: the
occurrence formula (Bradley 2010, Eq. 7), the level layout handed to the engine,
the CSV round trip that ties a datastore back to its targets, and the generated
config text. The datastore checks (subclass vs the old stock calcs, total
probability, independence) run on the analysis machine, from the check cells of
nb 020 / 021 / 072 or ``python -m phd_project.scripts.oq_multi_iml_disagg verify``.
"""

import numpy as np
import pandas as pd
import pytest

from phd_project.scripts import oq_multi_iml_disagg as mid
from phd_project.scripts import oqhelpers


# --------------------------------------------------------------------------- #
# occurrence                                                                   #
# --------------------------------------------------------------------------- #

def test_selftest_passes():
    # analytic agreement, sums to 1, differs from exceedance (the old bug)
    rep = mid.selftest_occurrence()
    assert (rep["status"] == "PASS").all(), rep


def _frame(nu, z=None):
    """A minimal TRT_Mag_Dist_Eps frame with given bin exceedance rates."""
    n = len(nu)
    df = pd.DataFrame({"TRT": ["A"] * n, "Mag": np.arange(n, dtype=float),
                       "Dist": np.zeros(n), "Eps": np.zeros(n),
                       "Z": np.zeros(n) if z is None else np.asarray(z, float),
                       "P(X>x|T,m)": 1.0 - np.exp(-np.asarray(nu, float))})
    return oqhelpers._get_traditional_disagg(df, 1.0)


def test_occurrence_is_rate_difference_share():
    # P(m|X=x) = (nu_m(x) - nu_m(x+dx)) / sum of the same - by hand
    nu_x, nu_xdx = np.array([4e-3, 1e-3, 0.0]), np.array([3e-3, 0.9e-3, 0.0])
    out = mid.occurrence_disagg(_frame(nu_x), _frame(nu_xdx))
    expected = (nu_x - nu_xdx) / (nu_x - nu_xdx).sum()
    np.testing.assert_allclose(out["P(m|X=x)"], expected, rtol=1e-9)
    # the exceedance columns are those of the x frame, untouched
    np.testing.assert_allclose(out["nu_m"], nu_x, rtol=1e-12)


def test_occurrence_per_realisation_index():
    # each Z sums to 1 on its own
    nu_x = np.array([4e-3, 1e-3, 2e-3, 2e-3])
    nu_xdx = np.array([3e-3, 0.5e-3, 1e-3, 1.5e-3])
    z = [0, 0, 1, 1]
    out = mid.occurrence_disagg(_frame(nu_x, z), _frame(nu_xdx, z))
    sums = out.groupby("Z")["P(m|X=x)"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=1e-12)


def test_occurrence_zero_hazard_is_nan():
    out = mid.occurrence_disagg(_frame([0.0, 0.0]), _frame([0.0, 0.0]))
    assert out["P(m|X=x)"].isna().all()


def test_occurrence_rejects_rising_rate():
    with pytest.raises(ValueError, match="rises"):
        mid.occurrence_disagg(_frame([1e-3, 1e-3]), _frame([2e-3, 0.5e-3]))


def test_occurrence_rejects_different_bins():
    a, b = _frame([1e-3, 1e-3]), _frame([0.5e-3, 0.5e-3])
    b["Mag"] = b["Mag"] + 0.1
    with pytest.raises(ValueError, match="bins"):
        mid.occurrence_disagg(a, b)


# --------------------------------------------------------------------------- #
# levels                                                                       #
# --------------------------------------------------------------------------- #

def _targets():
    # 3 sites x 2 targets, keys as nb 020 uses them (6-sig-fig IML floats)
    return pd.DataFrame([(s, k, k) for s in range(3) for k in (0.4, 0.285)],
                        columns=["site", "key", "x"])


def test_levels_layout():
    t = mid.build_levels_table(_targets(), delta=0.001)
    assert len(t) == 3 * 4
    s0 = t[t.site == 0]
    # ordered by x: 0.285 first, then 0.4; x then x+dx
    assert list(s0["p"]) == [0, 1, 2, 3]
    assert list(s0["role"]) == ["x", "x+dx", "x", "x+dx"]
    assert list(s0["key"]) == [0.285, 0.285, 0.4, 0.4]
    np.testing.assert_allclose(s0["iml_g"], [0.285, 0.285 * 1.001, 0.4, 0.4 * 1.001])
    # the x level is the key itself, bit for bit (old stock IMLs read the same text)
    assert s0.iloc[0]["iml_g"] == 0.285


def test_levels_array_shape_and_values():
    t = mid.build_levels_table(_targets())
    arr = mid.levels_array(t, n_sites=3)
    assert arr.shape == (3, 1, 4)
    assert np.all(arr > 0)


def test_levels_unequal_sites_need_pad():
    t = _targets()
    t = t[~((t.site == 2) & (t.key == 0.4))]
    with pytest.raises(ValueError, match="different numbers"):
        mid.build_levels_table(t)
    padded = mid.build_levels_table(t, pad=True)
    s2 = padded[padded.site == 2]
    assert list(s2["role"]) == ["x", "x+dx", "pad", "pad"]
    assert np.all(mid.levels_array(padded, 3) > 0)          # never 0


def test_levels_collision_raises():
    t = pd.DataFrame({"site": [0, 0], "key": ["3s", "5s"], "x": [0.4, 0.4]})
    with pytest.raises(ValueError, match="coincide"):
        mid.build_levels_table(t)


def test_levels_missing_cell_raises():
    t = mid.build_levels_table(_targets())
    with pytest.raises(ValueError, match="empty"):
        mid.levels_array(t, n_sites=4)                      # site 3 has no levels


@pytest.mark.parametrize("keys", [[0.285, 0.4], [475, 2500], ["3s", "5s"]])
def test_levels_csv_round_trip(tmp_path, keys):
    t = pd.DataFrame([(s, k, 0.3 + 0.1 * i) for s in range(2) for i, k in enumerate(keys)],
                     columns=["site", "key", "x"])
    table = mid.build_levels_table(t)
    csv_fp, npy_fp = mid.write_levels_files(table, 2, tmp_path, "unit")
    back = mid.read_levels_table(csv_fp)
    assert list(back["key"]) == list(table["key"])
    assert type(back["key"].iloc[0]) is type(table["key"].iloc[0])
    np.testing.assert_array_equal(back["iml_g"].to_numpy(), table["iml_g"].to_numpy())
    np.testing.assert_array_equal(np.load(npy_fp), mid.levels_array(table, 2))


def test_target_positions():
    t = mid.build_levels_table(_targets())
    pos = mid.target_positions(t)
    row = pos[(pos.site == 1) & (pos.key == 0.4)].iloc[0]
    assert (row.p_x, row.p_xdx) == (2, 3)


def test_split_targets_keeps_pairs_and_sites():
    t = pd.DataFrame([(s, k, k) for s in range(2) for k in (0.2, 0.3, 0.4)],
                     columns=["site", "key", "x"])
    chunks = mid.split_targets(t, max_levels=4)              # 2 targets per chunk
    assert [len(c) for c in chunks] == [4, 2]
    for c in chunks:
        assert c.groupby("site")["key"].apply(tuple).nunique() == 1


def test_merge_collations_rejects_duplicates():
    a = {0: {"AvgSA": {0.4: "a"}}}
    b = {0: {"AvgSA": {0.4: "b"}}}
    with pytest.raises(ValueError, match="more than one child"):
        mid.merge_collations([a, b])
    merged = mid.merge_collations([a, {0: {"AvgSA": {0.3: "c"}}}])
    assert list(merged[0]["AvgSA"]) == [0.3, 0.4]


# --------------------------------------------------------------------------- #
# configs                                                                      #
# --------------------------------------------------------------------------- #

def test_child_config_text(tmp_path):
    fp = mid.write_child_config(tmp_path / "c.ini", parent_calc_id=200,
                                parent_name="AvgSA_03_psha_eps4", n_levels=6,
                                description="[x]", levels_file="levels_x.npy")
    text = fp.read_text()
    assert "hazard_calculation_id = 200" in text
    assert 'intensity_measure_types_and_levels = {"AvgSA": logscale(0.0005, 3.00, 25)}' in text
    poes = [l for l in text.splitlines() if l.startswith("poes_disagg")][0].split("=")[1].split()
    assert len(poes) == 6 and len(set(poes)) == 6
    assert "num_rlzs_disagg = 0" in text and "iml_disagg" not in text.split("[disaggregation]")[1]


def test_psha_config_with_parent_block(tmp_path):
    # nb 003's PSHA config + disagg_parent_block = the old PSHA text plus exactly
    # the [disaggregation] block, so the hazard model itself is untouched
    from phd_project.scripts import oq_runner
    kw = dict(description="d", gsim_logic_tree_file=oq_runner.GMPE_LOGIC_TREES["03"],
              truncation_level=4, im_upper=3)
    plain = oq_runner.write_config(tmp_path / "a.ini", **kw).read_text()
    parent = oq_runner.write_config(tmp_path / "b.ini", extra=oq_runner.disagg_parent_block(60),
                                    **kw).read_text()
    assert parent == plain + oq_runner.disagg_parent_block(60)
    assert "[disaggregation]" in parent and "max_sites_disagg = 60" in parent


def test_parent_name_is_the_psha():
    assert mid.parent_name("03", 4) == "AvgSA_03_psha_eps4"


class _FakeDstore(dict):
    """dict-backed stand-in for a datastore: ``in``, ``[]`` and ``close``."""

    def close(self):
        pass


class _FakeOQ:
    def __init__(self, mode="classical", msd=60):
        self.calculation_mode, self.max_sites_disagg = mode, msd


def _parent_setup(tmp_path, monkeypatch, *, mode="classical", msd=60, n=60, rup=True):
    import json
    from phd_project.scripts import oq_runner
    man = tmp_path / "psha_manifest.json"
    man.write_text(json.dumps({"AvgSA_03_psha_eps4": {"calc_id": 7}}))
    monkeypatch.setattr(oq_runner, "_calc_exists", lambda cid: True)
    ds = _FakeDstore(oqparam=_FakeOQ(mode, msd), sitecol=list(range(n)))
    if rup:
        ds["rup"] = object()
    return man, (lambda cid: ds)


def test_require_parent_accepts_a_good_psha(tmp_path, monkeypatch):
    man, reader = _parent_setup(tmp_path, monkeypatch)
    assert mid.require_parent(man, "AvgSA_03_psha_eps4", 60, reader=reader) == 7


@pytest.mark.parametrize("bad", [dict(msd=10), dict(rup=False), dict(n=59),
                                 dict(mode="disaggregation")])
def test_require_parent_rejects(tmp_path, monkeypatch, bad):
    # msd=10: a PSHA from before nb 003 wrote max_sites_disagg (e.g. calc 113)
    man, reader = _parent_setup(tmp_path, monkeypatch, **bad)
    with pytest.raises(ValueError, match="Re-run nb 003"):
        mid.require_parent(man, "AvgSA_03_psha_eps4", 60, reader=reader)


def test_require_parent_missing_entry(tmp_path, monkeypatch):
    man, reader = _parent_setup(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        mid.require_parent(man, "AvgSA_06_psha_eps4", 60, reader=reader)


def test_verify_refuses_too_few_references():
    # raises before touching any datastore, so a dummy is enough
    with pytest.raises(AssertionError, match="too few IMLs"):
        mid.verify_against_reference(None, {1: (0, 0)}, min_refs=mid.MIN_STOCK_REFS)


# --------------------------------------------------------------------------- #
# hazard-curve readers by site (engine bug at disaggregation.py:205)          #
# --------------------------------------------------------------------------- #

class _FakeGetter:
    """Stands in for an engine MapGetter: only ``.sids`` is used by the wrapper."""

    def __init__(self, sids):
        self.sids = np.asarray(sids, dtype=np.uint32)


def _chunked(n_sites, n_chunks):
    # the engine's own rule (map_getters / gen_chunks): site sid -> chunk sid % n
    sids = np.arange(n_sites)
    return [_FakeGetter(sids[sids % n_chunks == c]) for c in range(n_chunks)]


def test_getters_by_sid_fewer_chunks_than_sites():
    # 60 sites in 7 chunks: the plain list fails at sid 7 (the reported error),
    # the wrapper returns the reader that holds each site
    readers = _chunked(60, 7)
    with pytest.raises(IndexError):
        readers[7]
    wrapped = mid._GettersBySid(readers)
    wrapped.check_covers(range(60))
    for sid in range(60):
        assert sid in wrapped[sid].sids
        assert wrapped[sid] is readers[sid % 7]
    # list behaviour the engine might rely on is unchanged
    assert len(wrapped) == 7 and list(wrapped) == readers


def test_getters_by_sid_one_chunk_per_site_is_unchanged():
    # the case the stock code was written for: reader k holds site k
    readers = _chunked(5, 5)
    wrapped = mid._GettersBySid(readers)
    assert all(wrapped[k] is readers[k] for k in range(5))


def test_getters_by_sid_refuses_missing_or_duplicate_sites():
    with pytest.raises(ValueError):
        mid._GettersBySid([_FakeGetter([0, 1]), _FakeGetter([1, 2])])
    wrapped = mid._GettersBySid([_FakeGetter([0, 2])])
    with pytest.raises(ValueError):
        wrapped.check_covers([0, 1, 2])
    with pytest.raises(IndexError):
        wrapped[1]
