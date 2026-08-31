"""Regression tests for the provenance-safe gm-selection refactor.

- **Fast unit tests** for ``cache_utils`` (no pipeline data) -- always run; cover
  fingerprint determinism and the load/stale/compute branches of
  ``load_or_compute``.
- **Engine orchestration** -- exercises the multi-round selection/optimisation
  wiring of ``build_final_ensembles`` with mocked selection/optimise steps (fast,
  no real data).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from phd_project.scripts import cache_utils, disagg_shards
from phd_project.scripts.WP1_ground_motion_set import gm_selection
from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    _disagg_fingerprint_input,
)
from phd_project.scripts.cache_utils import (
    StaleCacheError,
    clear_file_hash_cache,
    fingerprint,
    json_load_or_compute,
    load_or_compute,
    verify,
)


# --------------------------------------------------------------------------- #
# Fast unit tests for cache_utils                                             #
# --------------------------------------------------------------------------- #

def test_fingerprint_is_deterministic_and_content_based():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fp1 = fingerprint(gm_db=df, rng_seed=1, model="tarbali")
    fp2 = fingerprint(gm_db=df, rng_seed=1, model="tarbali")
    assert fp1["gm_db"]["hash"] == fp2["gm_db"]["hash"]
    assert "pickagm_version" in fp1
    # mutating one value changes the hash
    df2 = df.copy()
    df2.loc[0, "a"] = 99
    assert fingerprint(gm_db=df2)["gm_db"]["hash"] != fp1["gm_db"]["hash"]


def test_file_hash_matches_read_bytes(tmp_path):
    """The memoised/streamed hash must equal a plain sha256 of the whole file.

    This is what keeps every already-written manifest valid across the switch to
    streamed hashing -- if it ever fails, hundreds of cached artifacts go stale.
    """
    clear_file_hash_cache()
    p = tmp_path / "big.bin"
    p.write_bytes(os.urandom(3_000_000))  # > the 1 MB-ish chunking used internally
    assert fingerprint(f=p)["f"]["hash"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_file_hash_is_memoised(monkeypatch, tmp_path):
    clear_file_hash_cache()
    p = tmp_path / "a.bin"
    p.write_bytes(b"payload")

    calls = {"n": 0}
    real = cache_utils._hash_file

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(cache_utils, "_hash_file", counting)

    h1 = fingerprint(f=p)["f"]["hash"]
    h2 = fingerprint(f=p)["f"]["hash"]
    assert h1 == h2
    assert calls["n"] == 1  # second fingerprint served from the memo


def test_file_hash_cache_invalidates_on_rewrite(tmp_path):
    """A rewritten file must re-hash, including when the size is unchanged."""
    clear_file_hash_cache()
    p = tmp_path / "a.bin"

    p.write_bytes(b"aaaa")
    h_first = fingerprint(f=p)["f"]["hash"]

    # same size, different content -- only mtime_ns distinguishes them. Force a
    # distinct mtime so the test does not depend on the filesystem clock.
    st = p.stat()
    p.write_bytes(b"bbbb")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    h_same_size = fingerprint(f=p)["f"]["hash"]
    assert h_same_size != h_first
    assert h_same_size == hashlib.sha256(b"bbbb").hexdigest()

    # different size
    p.write_bytes(b"cccccccc")
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000))
    assert fingerprint(f=p)["f"]["hash"] == hashlib.sha256(b"cccccccc").hexdigest()


def test_load_or_compute_branches(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"value": 42}

    art = tmp_path / "artifact.pickle"
    fp = fingerprint(rng_seed=1)

    # cold compute writes artifact + manifest
    assert load_or_compute(art, fp, compute) == {"value": 42}
    assert calls["n"] == 1
    assert (tmp_path / "artifact.pickle.manifest.json").is_file()

    # warm load does not recompute
    assert load_or_compute(art, fp, compute) == {"value": 42}
    assert calls["n"] == 1

    # changed input -> auto-recompute + overwrite (no error); manifest now
    # reflects the new inputs.
    fp_bad = fingerprint(rng_seed=2)
    assert load_or_compute(art, fp_bad, compute) == {"value": 42}
    assert calls["n"] == 2
    # a subsequent load with the (now-cached) new inputs matches -> no recompute
    assert load_or_compute(art, fp_bad, compute) == {"value": 42}
    assert calls["n"] == 2

    # force_recompute overrides and overwrites even when inputs match
    assert load_or_compute(art, fp_bad, compute, force_recompute=True) == {"value": 42}
    assert calls["n"] == 3


def test_missing_manifest_recomputes(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return [1, 2, 3]

    art = tmp_path / "a.pickle"
    fp = fingerprint(x=1)
    load_or_compute(art, fp, compute)
    assert calls["n"] == 1
    # remove the manifest -> unmanaged artifact -> recompute (don't trust it)
    (tmp_path / "a.pickle.manifest.json").unlink()
    load_or_compute(art, fp, compute)
    assert calls["n"] == 2


def test_json_load_or_compute_branches(tmp_path):
    """The JSON sibling: same provenance rules, plus a cached/computed status."""
    calls = {"n": 0}
    src = tmp_path / "source.json"
    src.write_text('{"median": 0.4}')

    def compute():
        calls["n"] += 1
        return {"median": 0.4, "efc": np.array([[0.1, 0.2], [0.0, 1.0]])}

    art = tmp_path / "frag.json"

    # cold compute writes artifact + manifest, and serialises the numpy array
    res, status = json_load_or_compute(art, fingerprint(source=src), compute)
    assert (status, calls["n"]) == ("computed", 1)
    assert (tmp_path / "frag.json.manifest.json").is_file()
    assert res["efc"][0][0] == 0.1

    # warm load does not recompute; arrays come back as nested lists
    res, status = json_load_or_compute(art, fingerprint(source=src), compute)
    assert (status, calls["n"]) == ("cached", 1)
    assert res["efc"] == [[0.1, 0.2], [0.0, 1.0]]

    # changed source file -> stale -> recompute and overwrite
    src.write_text('{"median": 0.5}')
    _, status = json_load_or_compute(art, fingerprint(source=src), compute,
                                     input_paths={"source": src})
    assert (status, calls["n"]) == ("computed", 2)

    # force overrides a matching manifest
    _, status = json_load_or_compute(art, fingerprint(source=src), compute, force=True)
    assert (status, calls["n"]) == ("computed", 3)

    # a missing manifest is not trustworthy -> recompute
    (tmp_path / "frag.json.manifest.json").unlink()
    _, status = json_load_or_compute(art, fingerprint(source=src), compute)
    assert (status, calls["n"]) == ("computed", 4)


def test_verify_subset(tmp_path):
    art = tmp_path / "a.pickle"
    load_or_compute(art, fingerprint(gm_db_file="x", rng_seed=1, extra="keep"),
                    lambda: {"k": 1})
    # subset that matches -> no error
    verify(art, fingerprint(gm_db_file="x"))
    # subset that differs -> StaleCacheError
    with pytest.raises(StaleCacheError, match="gm_db_file"):
        verify(art, fingerprint(gm_db_file="y"))


# --------------------------------------------------------------------------- #
# Engine orchestration logic (mocked selection/optimise — fast, no real data)  #
# --------------------------------------------------------------------------- #

class _Stub:
    def __init__(self, s):
        self.string = s


def _basic_ctx():
    return {
        "n_ensembles": 20, "n_samples": 30, "p_value": 0.05, "usable_T": 3,
        "max_n_recs": 5, "m_bound_model": "tarbali", "d_bound_model": "tarbali",
        "vs30_bound_model": "tarbali", "occurence": True,
        "conditioning_imt": _Stub("AvgSA(0,3)"),
        "selection_imts": [_Stub("PGA"), _Stub("RSD595")],
        "imt_weights": np.array([0.5, 0.5]),
    }


# --------------------------------------------------------------------------- #
# Per-site disagg shards                                                       #
# --------------------------------------------------------------------------- #

def _fake_disagg(n_sites=3):
    """{site: {imt: {iml: DataFrame}}}, the shape nb 021 collates."""
    return {
        site: {"AvgSA": {iml: pd.DataFrame({"Mag": [5.0 + site], "P(m|X>x)": [iml]})
                         for iml in (0.27, 0.285, 1.3368)}}
        for site in range(n_sites)
    }


def _same(a, b):
    assert set(a) == set(b)
    for site in a:
        assert set(a[site]) == set(b[site])
        for imt in a[site]:
            # exact float keys matter: they are the pipeline's (site, iml) keys
            assert list(a[site][imt]) == list(b[site][imt])
            for iml in a[site][imt]:
                pd.testing.assert_frame_equal(a[site][imt][iml], b[site][imt][iml])
    return True


def test_shards_round_trip_and_subset(tmp_path):
    data = _fake_disagg()
    fp = fingerprint(rng_seed=1)
    disagg_shards.write_shards(tmp_path, data, fp)

    assert _same(disagg_shards.load_shards(tmp_path), data)

    only_1 = disagg_shards.load_shards(tmp_path, sites=[1])
    assert set(only_1) == {1}
    assert disagg_shards.load_shards(tmp_path, sites=()) == {}

    with pytest.raises(FileNotFoundError):
        disagg_shards.load_shards(tmp_path, sites=[99])


def test_shard_index_has_exact_iml_keys(tmp_path):
    """The index must reproduce the native float keys bit-for-bit.

    wanted_stripe_keys() matches against these, so an index that rounded would
    silently shift the whole (site, iml) key set.
    """
    data = _fake_disagg()
    disagg_shards.write_shards(tmp_path, data, fingerprint(rng_seed=1))
    index = disagg_shards.read_index(tmp_path)
    for site, imt_dict in data.items():
        for imt, iml_dict in imt_dict.items():
            assert index[site][imt] == sorted(iml_dict)


def test_shard_path_round_trip():
    p = disagg_shards.shard_path("d", 7)
    assert p.name == "site_007.pickle"
    assert disagg_shards.site_from_shard_name(p.name) == 7
    with pytest.raises(ValueError):
        disagg_shards.site_from_shard_name("not_a_shard.pickle")


def test_load_or_compute_shards_branches(tmp_path):
    calls = {"n": 0}
    data = _fake_disagg()

    def compute():
        calls["n"] += 1
        return data

    fp = fingerprint(rng_seed=1)

    # cold -> compute + write
    assert _same(disagg_shards.load_or_compute_shards(tmp_path, fp, compute), data)
    assert calls["n"] == 1

    # warm, matching manifests -> load, no recompute
    assert _same(disagg_shards.load_or_compute_shards(tmp_path, fp, compute), data)
    assert calls["n"] == 1

    # changed inputs -> recompute
    disagg_shards.load_or_compute_shards(tmp_path, fingerprint(rng_seed=2), compute)
    assert calls["n"] == 2

    # force -> recompute even on a match
    disagg_shards.load_or_compute_shards(
        tmp_path, fingerprint(rng_seed=2), compute, force_recompute=True)
    assert calls["n"] == 3


def test_shards_digest_tracks_content(tmp_path):
    clear_file_hash_cache()
    disagg_shards.write_shards(tmp_path, _fake_disagg(), fingerprint(rng_seed=1))
    first = disagg_shards.shards_digest(tmp_path)
    assert first == disagg_shards.shards_digest(tmp_path)

    # change one shard's bytes -> digest moves
    victim = disagg_shards.shard_path(tmp_path, 1)
    st = victim.stat()
    victim.write_bytes(victim.read_bytes() + b"extra")
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    clear_file_hash_cache()
    assert disagg_shards.shards_digest(tmp_path) != first


def test_stripe_fingerprint_is_site_and_iml_scoped(tmp_path):
    """A stripe entry must name THIS (site, iml) DataFrame, and nothing else.

    The legacy monolith shape must still work unchanged (AvgSA_06).
    """
    clear_file_hash_cache()
    disagg_shards.write_shards(tmp_path, _fake_disagg(), fingerprint(rng_seed=1))

    sharded = {"disagg_shard_dir": tmp_path}
    assert "disagg_stripe_data" in _disagg_fingerprint_input(sharded, 1, 0.27)
    assert (_disagg_fingerprint_input(sharded, 1, 0.27)
            != _disagg_fingerprint_input(sharded, 2, 0.27))
    assert (_disagg_fingerprint_input(sharded, 1, 0.27)
            != _disagg_fingerprint_input(sharded, 1, 0.285))
    # site given but no iml is a programming error, not a silently coarser hash
    with pytest.raises(ValueError):
        _disagg_fingerprint_input(sharded, 1)
    # site=None -> the whole-set digest, for batch artifacts
    assert "disagg_shards_digest" in _disagg_fingerprint_input(sharded)

    legacy = {"disagg_data_file": tmp_path / "_index.json"}
    assert list(_disagg_fingerprint_input(legacy)) == ["disagg_data_file"]
    assert _disagg_fingerprint_input(legacy, 1) == _disagg_fingerprint_input(legacy, 2)


def test_adding_an_iml_across_all_sites_leaves_other_stripes_valid(tmp_path):
    """The regression test for the all-517-stripes-stale bug.

    An OpenQuake ``iml_disagg`` run covers ALL sites at ONE iml, so adding one MSA
    stripe rewrites every shard. Under whole-shard hashes that marked every
    already-selected stripe stale. Only the NEW (site, iml) may move now.
    """
    clear_file_hash_cache()
    disagg_shards.clear_content_hash_cache()
    data = _fake_disagg()
    disagg_shards.write_shards(tmp_path, data, fingerprint(rng_seed=1))

    sharded = {"disagg_shard_dir": tmp_path}
    before = {(site, iml): _disagg_fingerprint_input(sharded, site, iml)
              for site in data for iml in data[site]["AvgSA"]}

    # a new stripe disaggregated for every site, exactly as calc 122 was
    for site in data:
        data[site]["AvgSA"][1.8039] = pd.DataFrame(
            {"Mag": [7.0 + site], "P(m|X>x)": [1.8039]})
    disagg_shards.write_shards(tmp_path, data, fingerprint(rng_seed=2))
    clear_file_hash_cache()
    disagg_shards.clear_content_hash_cache()

    for key, fp in before.items():
        assert _disagg_fingerprint_input(sharded, *key) == fp, f"{key} went stale"
    assert "disagg_stripe_data" in _disagg_fingerprint_input(sharded, 0, 1.8039)


def test_stats_row_hash_ignores_row_position(tmp_path):
    """A stripe's stats entry must not depend on where its row sits in the table.

    021 appends rows for a new iml to the single stats pickle. If the hash carried
    row position, every stripe below the insertion point would go stale.
    """
    stats = pd.DataFrame({
        "site_id": [0, 0, 1], "imt": ["AvgSA"] * 3,
        "imtl": [0.27, 0.285, 0.27], "poe": [0.1, 0.05, 0.2],
    })
    row = gm_selection.stats_row(stats, 0, "AvgSA", 0.285)
    shuffled = stats.iloc[[2, 1, 0]]
    assert (cache_utils.dataframe_content_hash(row)
            == cache_utils.dataframe_content_hash(
                gm_selection.stats_row(shuffled, 0, "AvgSA", 0.285)))
    # but a changed value in that row does move it
    changed = stats.copy()
    changed.loc[1, "poe"] = 0.06
    assert (cache_utils.dataframe_content_hash(row)
            != cache_utils.dataframe_content_hash(
                gm_selection.stats_row(changed, 0, "AvgSA", 0.285)))


def test_write_content_hashes_backfill_matches_write_shards(tmp_path):
    """The backfill must produce exactly what write_shards emits.

    The manifest migration re-stamps from the backfilled index and the pipeline
    then checks against the written one; if they differed, every migrated stripe
    would be stale.
    """
    data = _fake_disagg()
    disagg_shards.write_shards(tmp_path, data, fingerprint(rng_seed=1))
    written = disagg_shards.content_hashes_path(tmp_path).read_text()
    disagg_shards.content_hashes_path(tmp_path).unlink()
    disagg_shards.write_content_hashes(tmp_path)
    assert disagg_shards.content_hashes_path(tmp_path).read_text() == written


def test_missing_content_hashes_index_raises(tmp_path):
    """No silent fallback to a whole-shard hash -- that is the bug, invisibly."""
    disagg_shards.write_shards(tmp_path, _fake_disagg(), fingerprint(rng_seed=1))
    disagg_shards.content_hashes_path(tmp_path).unlink()
    disagg_shards.clear_content_hash_cache()
    with pytest.raises(FileNotFoundError, match="write_content_hashes"):
        _disagg_fingerprint_input({"disagg_shard_dir": tmp_path}, 1, 0.27)


def test_engine_call_pattern_and_round3_empty(tmp_path, monkeypatch):
    """Engine reproduces the golden round structure and skips round 3.

    Synthetic 5-key scenario mirroring AvgSA_03: after prelim, key 3 fails and
    key 4 has no ensemble; round 2 (force=True, d unbounded) reselects key 4 and
    optimises both into passing; round 3 has nothing to do.
    """
    import phd_project.scripts.WP1_ground_motion_set.gm_selection as g

    keys = [(0, p) for p in range(5)]
    calls = {"select": [], "optimise": []}

    def ens(passed):
        return {"ks_passed": passed, "recs": "x"}

    def fake_prelim(spd, ds, gd, db, ctx, sm, seed, *, preliminary_selection_fp,
                    only_select, input_fingerprint, desc, force_recompute,
                    quiet=False):
        calls["select"].append(list(only_select))
        reselection = ctx["d_bound_model"] is None
        out = {}
        for k in only_select:
            if reselection:
                out[k] = ens(True)               # looser bounds → reselected key passes
            else:
                idx = keys.index(k)
                out[k] = None if idx == 4 else ens(idx in (0, 1, 2))
        return out, None, None, None

    def fake_optimise(spd, ds, gd, db, ctx, sm, ensembles, *, optimised_selection_fp,
                      only_optimise, shuffle, rng_seed, input_fingerprint,
                      description, force_recompute, n_shuffles=1, rng_seeds=None,
                      quiet=False):
        calls["optimise"].append(list(only_optimise))
        d_free = ctx["d_bound_model"] is None
        out = {}
        for k in only_optimise:
            cur = ensembles.get(k)
            out[k] = None if cur is None else ens(True if d_free else cur["ks_passed"])
        return out, []

    monkeypatch.setattr(g, "preliminary_record_selection", fake_prelim)
    monkeypatch.setattr(g, "optimise_record_selection", fake_optimise)

    source_fps = {k: str(tmp_path / k) for k in
                  ["gm_db_file", "gcim_file", "disagg_data_file",
                   "disagg_stats_file", "site_model_file"]}
    stage_fps = {
        "select":   [tmp_path / f"sel_r{i}.pkl" for i in range(3)],
        "optimise": [tmp_path / f"opt_r{i}.pkl" for i in range(3)],
    }

    final = g.build_final_ensembles(
        {k: None for k in keys}, None, None, None, _basic_ctx(), None,
        source_fps=source_fps, stage_fps=stage_fps,
        output_fp=tmp_path / "final.pickle",
        round_unbounded=[[], ["d"], ["m", "d", "vs30"]],
        force_optimisation=[False, True, False],
    )

    # Round 1 selects all 5, round 2 reselects only the no-ensemble key 4,
    # round 3 selection is skipped (work-set empty).
    assert calls["select"] == [keys, [(0, 4)]]
    # Round 1 optimises the 2 failing (incl. the None key); round 2 force-optimises
    # both carried keys (incl. the reselected one); round 3 optimise skipped.
    assert calls["optimise"] == [[(0, 3), (0, 4)], [(0, 3), (0, 4)]]
    # Everything passes; the canonical artifact + manifest were written.
    assert all(final[k]["ks_passed"] for k in keys)
    assert (tmp_path / "final.pickle").is_file()
    assert (tmp_path / "final.pickle.manifest.json").is_file()
