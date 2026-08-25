"""Regression tests for the provenance-safe gm-selection refactor.

- **Fast unit tests** for ``cache_utils`` (no pipeline data) -- always run; cover
  fingerprint determinism and the load/stale/compute branches of
  ``load_or_compute``.
- **Engine orchestration** -- exercises the multi-round selection/optimisation
  wiring of ``build_final_ensembles`` with mocked selection/optimise steps (fast,
  no real data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phd_project.scripts.cache_utils import (
    StaleCacheError,
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
