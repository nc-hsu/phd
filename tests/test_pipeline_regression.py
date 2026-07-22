"""Regression tests for the provenance-safe gm-selection refactor.

Two layers:

- **Fast unit tests** for ``cache_utils`` (no pipeline data) -- always run; cover
  fingerprint determinism and the load/stale/compute branches of
  ``load_or_compute``.
- **Slice regression** -- recomputes the preliminary selection for a couple of
  ``(site, poe)`` keys through the *guarded* function and asserts the selected
  records exactly match the golden snapshot. Skipped automatically if the heavy
  input data or the golden snapshot is unavailable (e.g. on CI without the data).

The full 60-site acceptance run is intentionally NOT a pytest (it takes ~1 hr):
run the Stage-1 notebook, then ``python tests/golden_compare.py``.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phd_project.scripts.cache_utils import (
    StaleCacheError,
    fingerprint,
    load_manifest,
    load_or_compute,
    verify,
)

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "AvgSA_03"


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

    # changed input -> StaleCacheError naming the input
    fp_bad = fingerprint(rng_seed=2)
    with pytest.raises(StaleCacheError, match="rng_seed"):
        load_or_compute(art, fp_bad, compute)

    # force_recompute overrides and overwrites
    assert load_or_compute(art, fp, compute, force_recompute=True) == {"value": 42}
    assert calls["n"] == 2


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
                    only_select, input_fingerprint, desc, force_recompute):
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
                      description, force_recompute):
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


# --------------------------------------------------------------------------- #
# Slice regression against the golden snapshot                                #
# --------------------------------------------------------------------------- #

def _data_available() -> bool:
    try:
        from phd_project.config.config import load_config
    except Exception:
        return False
    if not (GOLDEN / "AvgSA_03_prelim_selection.pickle").is_file():
        return False
    cfg = load_config()
    needed = [
        cfg["proc_data"]["gm_database"],
        cfg["proc_data"]["gcim_dists"] / "gcim_dist_AvgSA_03.pickle",
        cfg["proc_data"]["AvgSA_03_disagg_data_gm_selection"],
    ]
    return all(Path(p).is_file() for p in needed)


@pytest.mark.skipif(not _data_available(),
                    reason="heavy input data or golden snapshot not available")
def test_preliminary_slice_matches_golden():
    """The guarded preliminary selection reproduces golden records for 2 keys."""
    from phd_project.config.config import load_config
    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
        setup_AvgSA03_gcim_gm_selection,
    )
    from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
        preliminary_record_selection, _selection_ctx_fingerprint_inputs,
    )

    cfg = load_config()
    golden = pickle.load(open(GOLDEN / "AvgSA_03_prelim_selection.pickle", "rb"))
    keys = [k for k, v in golden.items() if v is not None and v.get("ks_passed")][:2]

    site_poe_disaggs, disagg_stats, site_model, ctx, gm_db = setup_AvgSA03_gcim_gm_selection()
    gcim_dists = pickle.load(
        open(cfg["proc_data"]["gcim_dists"] / "gcim_dist_AvgSA_03.pickle", "rb"))

    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / "slice.pickle"
        base = {
            "gm_db_file": cfg["proc_data"]["gm_database"],
            "gcim_file": cfg["proc_data"]["gcim_dists"] / "gcim_dist_AvgSA_03.pickle",
            "disagg_data_file": cfg["proc_data"]["AvgSA_03_disagg_data_gm_selection"],
            "disagg_stats_file": cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"],
            "site_model_file": cfg["hazard_models"]["eshm20_AvgSA_site_model_all"],
            "rng_seed": 1,
        }
        fp = fingerprint(**base, stage="preliminary", only=tuple(sorted(keys)),
                         **_selection_ctx_fingerprint_inputs(ctx))
        pe, _, _, _ = preliminary_record_selection(
            site_poe_disaggs, disagg_stats, gcim_dists, gm_db, ctx, site_model, 1,
            preliminary_selection_fp=art, only_select=keys, input_fingerprint=fp)

    for k in keys:
        got = sorted(pe[k]["recs"][("metadata", "index")].tolist())
        exp = sorted(golden[k]["recs"][("metadata", "index")].tolist())
        assert got == exp, f"selection changed at {k}"
