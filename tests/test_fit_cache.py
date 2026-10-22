"""Tests for the on-disk model-fit cache (``phd_project/scripts/fit_cache.py``).

What these tests pin down:

1. A loaded fit is the SAME fit: every saved number identical to the fresh one,
   and the loaded fit hangs off the registry's own memoised contrast.
2. Every input that changes a number forces a refit, and the printed reason
   names it; inputs that change no number (a spec's note, a docstring, a
   comment, the seed of a record-only bootstrap) do NOT.
3. "5000 serves 500": a file saved with N replicates answers a request for
   n <= N by slicing, and the slice is bit-identical to a fresh n-replicate
   fit -- for the record scheme, the hybrid scheme and the joint model.
4. Per-estimator files, atomic writes, overwrite vs refit, and the errors and
   warnings ``fit()`` already gives are unchanged on the cached path.

Everything runs on the 24-row toy study of ``test_fragility_data_models`` with
k = 50, in a pytest ``tmp_path``, so the whole file takes seconds.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import pickle
import textwrap

import numpy as np
import pandas as pd
import pytest

import phd_project.scripts.fit_cache as fcache
import phd_project.scripts.fragility_data_models as fm
import phd_project.scripts.metaregression_analysis as mra

from test_fragility_data_models import _HAVE_DATA, _IM_TAG, _ROOT, _toy_joint

K = 50  # replicates in the toy cloud

# The univariate specs these tests fit. A0a has both estimators and a record
# bootstrap; A0a_hb is its hybrid twin (not in MODEL_SPECS, which has no
# intercept-only DL hybrid); A0c_hb is a crossed REML hybrid; A0a_bc is a
# theta `_bc` spec, i.e. one that triggers the "correction is a no-op" warning.
_A0A = fm.MODEL_SPECS["A0a"]
SPECS = {
    "A0a": _A0A,
    "A0a_hb": dataclasses.replace(_A0A, model_id="A0a_hb", bootstrap_scheme="hybrid"),
    "A0a_bc": dataclasses.replace(_A0A, model_id="A0a_bc", corrected=True),
    "A0c_hb": fm.MODEL_SPECS["A0c_hb"],
}


@pytest.fixture(autouse=True)
def _fresh_code_hash():
    """Each test starts with no memoised code hash, so monkeypatched source is seen."""
    fcache.clear_code_hash_cache()
    yield
    fcache.clear_code_hash_cache()


@pytest.fixture(scope="module")
def toy():
    """``(theta, beta)``: 12 sites x {3s, 5s} = 24 rows, k = 50, replicate-aligned."""
    return _toy_joint(n_sites=12, k=K)


def _reg(data, cache_dir=None, specs=SPECS, **kw):
    return fm.ModelRegistry(data, specs, cache_dir=cache_dir, **kw)


def _joint(toy, cache_dir=None, **kw):
    return fm.JointModelRegistry(*toy, cache_dir=cache_dir, **kw)


def _status(reg, model_id, estimator="REML") -> str:
    return reg.cache_log[(model_id, estimator)]


def assert_same_fit(a, b):
    """Every saved field of two fits identical (exact: the numbers are the same bits)."""
    for name in ("beta", "se_beta", "tau2", "het", "x_names", "scaling", "raw", "boot"):
        if hasattr(a, name):
            np.testing.assert_equal(getattr(a, name), getattr(b, name), err_msg=name)
    if a.replicates is None or b.replicates is None:
        assert a.replicates is None and b.replicates is None
    else:
        pd.testing.assert_frame_equal(a.replicates, b.replicates, check_exact=True)
        assert a.replicates.attrs == b.replicates.attrs


# --------------------------------------------------------------------------- #
# 1. Round trip                                                               #
# --------------------------------------------------------------------------- #

def test_no_cache_dir_reads_and_writes_nothing(toy, tmp_path, monkeypatch):
    """Without cache_dir, fit() is the old fit(): no files, no log, same numbers."""
    monkeypatch.chdir(tmp_path)
    plain = _reg(toy[0])
    fits = plain.fit("A0a", n_boot=10, refit=True)   # refit is a no-op here
    assert list(tmp_path.iterdir()) == []
    assert plain.cache_log == {}
    cached = _reg(toy[0], tmp_path / "c").fit("A0a", n_boot=10)
    for a, b in zip(fits, cached):
        assert_same_fit(a, b)


def test_round_trip_univariate(toy, tmp_path):
    first = _reg(toy[0], tmp_path)
    fresh = first.fit("A0a", n_boot=30)
    assert _status(first, "A0a", "DL").startswith("fitted")
    second = _reg(toy[0], tmp_path)
    loaded = second.fit("A0a", n_boot=30)
    for f, l in zip(fresh, loaded):
        assert _status(second, "A0a", l.estimator).startswith("loaded")
        assert_same_fit(f, l)
        # The registry's OWN memoised contrast, not an unpickled copy.
        assert l.contrast is second.contrast("y0", False)
        assert l.spec is second.specs["A0a"]
        assert f.summary() == l.summary()
    pd.testing.assert_frame_equal(first.table(), second.table())


def test_round_trip_joint_is_one_file(toy, tmp_path):
    fresh = _joint(toy, tmp_path).fit("A8a_hb", n_boot=20)
    files = sorted(p.name for p in (tmp_path).rglob("*") if p.is_file())
    assert files == ["A8a_hb.pkl", "A8a_hb.pkl.manifest.json"]
    assert (tmp_path / "TEST" / "n24_3s5s_provisional" / "joint" / "A8a_hb.pkl").is_file()
    reg = _joint(toy, tmp_path)
    loaded = reg.fit("A8a_hb", n_boot=20)
    assert _status(reg, "A8a_hb").startswith("loaded")
    assert_same_fit(fresh, loaded)
    assert loaded.contrast is reg.contrast("y0_avg", False)
    assert fresh.summary() == loaded.summary()


def test_cache_summary_counts_and_reasons(toy, tmp_path, capsys):
    _reg(toy[0], tmp_path).fit("A0a", n_boot=10)
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10, n_starts=2)        # REML refits (kw), DL loads
    text = reg.cache_summary()
    assert text.splitlines()[0] == "cache: 1 loaded, 1 fitted"
    assert "A0a [REML]" in text and "changed: kw" in text
    assert "[stale] A0a [REML]" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 2. Invalidators                                                             #
# --------------------------------------------------------------------------- #

def _refit_reason(data, tmp_path, model_id="A0a", estimator="REML", specs=SPECS,
                  first_kw=None, second_kw=None, first_data=None) -> str:
    """Fit once (saving), fit again with something changed; return the second status."""
    _reg(first_data or data, tmp_path, specs=SPECS).fit(model_id, n_boot=10,
                                                         **(first_kw or {}))
    reg = _reg(data, tmp_path, specs=specs)
    reg.fit(model_id, n_boot=10, **(second_kw or {}))
    return _status(reg, model_id, estimator)


def test_changed_data_value_refits(toy, tmp_path):
    theta = toy[0]
    est = dict(theta.estimates)
    est["a"] = est["a"].copy()
    est["a"].iloc[0] += 0.01
    changed = dataclasses.replace(theta, estimates=est)
    status = _refit_reason(changed, tmp_path, first_data=theta)
    assert status.startswith("fitted") and "changed: contrast" in status


def test_relabelled_replicates_refit_even_with_identical_values(toy, tmp_path):
    """dataframe_content_hash drops labels; the cache must not."""
    theta = toy[0]
    reps = {s: None if r is None else r.set_axis(pd.Index(range(1000, 1000 + K), name="k"),
                                                  axis=1)
            for s, r in theta.reps.items()}
    changed = dataclasses.replace(theta, reps=reps)
    status = _refit_reason(changed, tmp_path, first_data=theta)
    assert "changed: contrast" in status


def test_changed_unused_arm_does_not_refit(toy, tmp_path):
    """Q1: the contrast is fingerprinted, not the container. y0 never reads c_avg."""
    theta = toy[0]
    reps = dict(theta.reps)
    reps["c_avg"] = reps["c_avg"] + 0.5
    status = _refit_reason(dataclasses.replace(theta, reps=reps), tmp_path, first_data=theta)
    assert status.startswith("loaded")


def test_spec_field_refits_but_note_and_estimators_do_not(toy, tmp_path):
    theta = toy[0]
    _reg(theta, tmp_path).fit("A0a", n_boot=10)

    def again(spec):
        reg = _reg(theta, tmp_path, specs={**SPECS, "A0a": spec})
        reg.fit("A0a", n_boot=10, overwrite=True)
        return reg

    reg = again(dataclasses.replace(_A0A, note="rewritten"))
    assert _status(reg, "A0a", "DL").startswith("loaded")
    assert _status(reg, "A0a", "REML").startswith("loaded")
    reg = again(dataclasses.replace(_A0A, estimators=("REML",)))
    assert _status(reg, "A0a", "REML").startswith("loaded")
    reg = again(dataclasses.replace(_A0A, diagonal_V=True))
    assert "changed: spec" in _status(reg, "A0a", "REML")


def test_reml_kw_refits_reml_only(toy, tmp_path):
    _reg(toy[0], tmp_path).fit("A0a", n_boot=10)
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10, n_starts=2)
    assert _status(reg, "A0a", "DL").startswith("loaded")
    assert "changed: kw" in _status(reg, "A0a", "REML")
    # verbose changes no number and is not part of the fingerprint
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10, n_starts=2, verbose=True)
    assert _status(reg, "A0a", "REML").startswith("loaded")


def test_boot_seed_matters_only_for_hybrid(toy, tmp_path):
    status = _refit_reason(toy[0], tmp_path, model_id="A0a_hb",
                           second_kw={"boot_seed": 1})
    assert "changed: boot_seed" in status
    status = _refit_reason(toy[0], tmp_path, model_id="A0a", second_kw={"boot_seed": 1})
    assert status.startswith("loaded")


def _recompiled(module, name: str, tmp_path, *, logic: bool):
    """A copy of ``module.name`` with either a LOGIC edit or only a docs edit.

    The copy is compiled from a real file (so ``inspect.getsource`` works on it)
    into a copy of the module's own namespace (so its globals resolve to the
    same helpers and constants the walker would follow from the original).
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(module, name))))
    fn = tree.body[0]
    has_doc = (isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
               and isinstance(fn.body[0].value.value, str))
    if logic:
        # A real statement: harmless at run time, but different code.
        fn.body.insert(1 if has_doc else 0, ast.parse("_cache_test_marker = 1").body[0])
    else:
        new_doc = ast.Expr(ast.Constant("A completely rewritten docstring."))
        if has_doc:
            fn.body[0] = new_doc
        else:
            fn.body.insert(0, new_doc)
    src = "# a new comment above the function\n" + ast.unparse(tree) + "\n"
    path = tmp_path / f"_patched_{name}_{'logic' if logic else 'docs'}.py"
    path.write_text(src)
    ns = dict(module.__dict__)
    exec(compile(src, str(path), "exec"), ns)
    return ns[name]


@pytest.mark.parametrize("logic", [True, False])
def test_code_edit_refits_but_docstring_and_comment_edits_do_not(toy, tmp_path,
                                                                 monkeypatch, logic):
    """A deep helper (not itself a root) edited: logic refits, documentation does not."""
    _reg(toy[0], tmp_path).fit("A0a", n_boot=10)
    patched = _recompiled(mra, "compute_heterogeneity_stats", tmp_path, logic=logic)
    monkeypatch.setattr(mra, "compute_heterogeneity_stats", patched)
    fcache.clear_code_hash_cache()
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    status = _status(reg, "A0a", "DL")
    if logic:
        assert "code (metaregression_analysis.compute_heterogeneity_stats)" in status
    else:
        assert status.startswith("loaded")


def test_changed_constant_refits(toy, tmp_path, monkeypatch):
    """fit_reml's default zero_tol is the NAME _REML_ZERO_TOL; its value is hashed too."""
    _reg(toy[0], tmp_path).fit("A0a", n_boot=10)
    monkeypatch.setattr(mra, "_REML_ZERO_TOL", 2e-8)
    fcache.clear_code_hash_cache()
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    assert "metaregression_analysis._REML_ZERO_TOL" in _status(reg, "A0a", "REML")


def test_univariate_and_joint_code_hashes_are_separate():
    uni, joint = fm.fit_code_inventory(), fm.fit_code_inventory(joint=True)
    assert "metaregression_analysis.fit_reml_mv" in joint
    assert "metaregression_analysis.fit_reml_mv" not in uni
    assert "metaregression_analysis.compute_heterogeneity_stats" not in joint


def test_code_inventory_covers_the_fit_path_and_nothing_downstream():
    inv = fm.fit_code_inventory()
    must = ["fragility_data_models.fit_dl", "fragility_data_models.fit_reml",
            "fragility_data_models.bootstrap_fits", "fragility_data_models.build_X",
            "fragility_data_models.build_components",
            "fragility_data_models._hybrid_components",
            "fragility_data_models.ModelRegistry._fit_estimator",
            "fragility_data_models.Contrast.v", "metaregression_analysis.fit_reml",
            "metaregression_analysis.reml_nll_chol", "metaregression_analysis._build_sigma",
            "metaregression_analysis.compute_re_summary_effect",
            "metaregression_analysis.compute_heterogeneity_stats",
            "metaregression_analysis._REML_ZERO_TOL",
            "data_simulation.hybrid_pseudo_data", "data_simulation.build_component_gs",
            "data_simulation.centre_replicates", "data_simulation.draw_effects"]
    assert not [m for m in must if m not in inv]
    joint = fm.fit_code_inventory(joint=True)
    must_joint = ["fragility_data_models.bootstrap_joint_fits",
                  "fragility_data_models.joint_level_gs",
                  "fragility_data_models.joint_level_factors",
                  "fragility_data_models.build_X_joint",
                  "metaregression_analysis.fit_reml_mv",
                  "data_simulation.hybrid_pseudo_data_mv"]
    assert not [m for m in must_joint if m not in joint]
    for forbidden in ("forest", "ci", "tau_ci", "random_effects", "marginal_covariance",
                      "summary", "make_contrast", "bootstrap_cov"):
        assert not [k for k in {**inv, **joint} if k.rsplit(".", 1)[-1] == forbidden]


# --------------------------------------------------------------------------- #
# 3. "5000 serves 500"                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model_id", ["A0a", "A0a_hb", "A0c_hb"])
def test_slice_equals_a_fresh_shorter_fit_univariate(toy, tmp_path, model_id):
    _reg(toy[1], tmp_path).fit(model_id, n_boot=K)
    reg = _reg(toy[1], tmp_path)
    loaded = reg.fit(model_id, n_boot=20)
    fresh = _reg(toy[1]).fit(model_id, n_boot=20)
    for l, f in zip(loaded, fresh):
        assert _status(reg, model_id, l.estimator) == f"loaded ({K} saved, using 20)"
        assert len(l.replicates) == 20
        assert_same_fit(l, f)


@pytest.mark.parametrize("model_id", ["A8a", "A8a_hb"])
def test_slice_equals_a_fresh_shorter_fit_joint(toy, tmp_path, model_id):
    _joint(toy, tmp_path).fit(model_id, n_boot=30)
    reg = _joint(toy, tmp_path)
    loaded = reg.fit(model_id, n_boot=12)
    assert _status(reg, model_id) == "loaded (30 saved, using 12)"
    assert_same_fit(loaded, _joint(toy).fit(model_id, n_boot=12))


def test_more_replicates_than_saved_refits(toy, tmp_path):
    _reg(toy[0], tmp_path).fit("A0a", n_boot=30)
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=40)
    assert _status(reg, "A0a") == "fitted and saved (saved 30 replicates, 40 requested)"
    # The 40-replicate file now on disk serves a later 40 request.
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=40)
    assert _status(reg, "A0a").startswith("loaded (40 saved")


def test_request_beyond_the_cloud_is_capped_at_k(toy, tmp_path):
    """n_boot=80 on a 50-replicate cloud runs 50, so a 50-replicate file serves it."""
    _reg(toy[0], tmp_path).fit("A0a", n_boot=K)
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=80)
    assert _status(reg, "A0a") == f"loaded ({K} saved, using {K})"


def test_no_bootstrap_file_cannot_serve_a_bootstrap(toy, tmp_path):
    _reg(toy[0], tmp_path).fit("A0a")
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=20)
    assert "saved without a bootstrap" in _status(reg, "A0a")


def test_bootstrap_file_serves_a_no_bootstrap_request(toy, tmp_path):
    _reg(toy[0], tmp_path).fit("A0a_hb", n_boot=30)
    reg = _reg(toy[0], tmp_path)
    loaded = reg.fit("A0a_hb")
    fresh = _reg(toy[0]).fit("A0a_hb")
    for l, f in zip(loaded, fresh):
        assert l.replicates is None and l.boot == {}
        assert_same_fit(l, f)


# --------------------------------------------------------------------------- #
# 4. Files, atomicity, semantics                                              #
# --------------------------------------------------------------------------- #

def test_per_estimator_files(toy, tmp_path):
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    reg.cache_path("A0a", "REML").unlink()
    reg2 = _reg(toy[0], tmp_path)
    reg2.fit("A0a", n_boot=10)
    assert _status(reg2, "A0a", "DL").startswith("loaded")
    assert _status(reg2, "A0a", "REML") == "fitted and saved (no cached file)"


def test_failed_write_leaves_no_trusted_file(toy, tmp_path, monkeypatch):
    def broken_dump(*a, **k):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(fcache.pickle, "dump", broken_dump)
    reg = _reg(toy[0], tmp_path)
    with pytest.warns(RuntimeWarning, match="saving it to the cache failed"):
        fits = reg.fit("A0a", n_boot=10)
    assert len(fits) == 2 and ("A0a", "REML") in reg     # the fit itself is kept
    assert "NOT saved" in _status(reg, "A0a")
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftovers == []                                # no pickle, temp or manifest
    monkeypatch.undo()
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    assert _status(reg, "A0a") == "fitted and saved (no cached file)"


def test_existing_file_survives_a_failed_overwrite_untrusted(toy, tmp_path, monkeypatch):
    """A crash while REPLACING a file must leave it untrusted, never trusted-but-stale."""
    _reg(toy[0], tmp_path).fit("A0a", n_boot=10)
    monkeypatch.setattr(fcache.pickle, "dump",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("killed")))
    with pytest.warns(RuntimeWarning):
        _reg(toy[0], tmp_path).fit("A0a", n_boot=20)
    monkeypatch.undo()
    path = _reg(toy[0], tmp_path).cache_path("A0a", "REML")
    assert path.is_file() and not fcache.cu._manifest_path(path).is_file()
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    assert "no manifest" in _status(reg, "A0a")


@pytest.mark.parametrize("damage, reason", [
    ("truncate_manifest", "manifest unreadable"),
    ("corrupt_pickle", "cached file unreadable"),
    ("swap_pickle", "different runs"),
])
def test_damaged_files_are_refitted_not_trusted(toy, tmp_path, damage, reason):
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    path = reg.cache_path("A0a", "REML")
    mp = fcache.cu._manifest_path(path)
    if damage == "truncate_manifest":
        mp.write_text(mp.read_text()[:40])
    elif damage == "corrupt_pickle":
        path.write_bytes(path.read_bytes()[:100])
    else:
        # A pickle from a different run (another kernel) under this manifest.
        payload = pickle.loads(path.read_bytes())
        payload["pair_digest"] = "0" * 64
        path.write_bytes(pickle.dumps(payload))
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    assert reason in _status(reg, "A0a")


def test_manifest_records_metadata(toy, tmp_path):
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    meta = json.loads(fcache.cu._manifest_path(reg.cache_path("A0a", "DL")).read_text())
    assert meta["_meta"]["n_boot_saved"] == 10
    assert set(meta["_meta"]["versions"]) >= {"numpy", "scipy", "pandas"}
    assert "metaregression_analysis.fit_reml" in meta["_meta"]["code_inventory"]
    assert set(meta["inputs"]) == {"spec", "contrast", "code", "pickagm_version"}  # DL: no kw


def test_loading_over_an_in_memory_fit_needs_overwrite(toy, tmp_path):
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    with pytest.raises(ValueError, match="already been fitted"):
        reg.fit("A0a", n_boot=10)
    reg.fit("A0a", n_boot=10, overwrite=True)
    assert _status(reg, "A0a").startswith("loaded")


def test_refit_ignores_the_disk_and_resaves(toy, tmp_path):
    reg = _reg(toy[0], tmp_path)
    reg.fit("A0a", n_boot=10)
    mp = fcache.cu._manifest_path(reg.cache_path("A0a", "REML"))
    before = json.loads(mp.read_text())["_meta"]["written_at"]
    reg2 = _reg(toy[0], tmp_path)
    reg2.fit("A0a", n_boot=10, refit=True)
    assert _status(reg2, "A0a") == "fitted and saved (refit=True)"
    assert json.loads(mp.read_text())["_meta"]["written_at"] != before
    with pytest.raises(ValueError, match="already been fitted"):
        reg2.fit("A0a", n_boot=10, refit=True)       # refit is not overwrite


def test_errors_and_warnings_are_unchanged_on_the_cached_path(toy, tmp_path):
    with pytest.raises(KeyError, match="unknown model"):
        _reg(toy[0], tmp_path).fit("nope")
    with pytest.raises(KeyError, match="unknown joint model"):
        _joint(toy, tmp_path).fit("nope")
    missing = dataclasses.replace(toy[0], estimates_bc={})
    with pytest.raises(KeyError, match="bias-corrected spec"):
        _reg(missing, tmp_path).fit("A0a_bc")
    # theta is never corrected, so A0a_bc warns -- when fitted AND when loaded.
    with pytest.warns(UserWarning, match="no-op"):
        _reg(toy[0], tmp_path).fit("A0a_bc", n_boot=10)
    reg = _reg(toy[0], tmp_path)
    with pytest.warns(UserWarning, match="no-op"):
        reg.fit("A0a_bc", n_boot=10)
    assert _status(reg, "A0a_bc").startswith("loaded")


def test_cache_tag_and_subset_label(toy, tmp_path):
    reg = _reg(toy[0], tmp_path)
    assert reg.cache_path("A0a", "DL") == (
        tmp_path / "TEST" / "n24_3s5s_provisional" / "theta" / "A0a_DL.pkl")
    assert _reg(toy[0], tmp_path, cache_tag="").cache_path("A0a", "DL") == \
        reg.cache_path("A0a", "DL")
    tagged = _reg(toy[0], tmp_path, cache_tag="star")
    assert tagged.cache_path("A0a", "DL").parent.parent.name == "n24_3s5s_provisional__star"
    with pytest.raises(ValueError, match="cache_tag"):
        _reg(toy[0], tmp_path, cache_tag="../escape")
    # Two containers, two tags: neither overwrites the other.
    shifted = dataclasses.replace(
        toy[0], estimates={**toy[0].estimates, "a": toy[0].estimates["a"] + 0.01})
    _reg(toy[0], tmp_path).fit("A0a", n_boot=10)
    _reg(shifted, tmp_path, cache_tag="star").fit("A0a", n_boot=10)
    again = _reg(toy[0], tmp_path)
    again.fit("A0a", n_boot=10)
    assert _status(again, "A0a").startswith("loaded")


# --------------------------------------------------------------------------- #
# 5. The real data, if present                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _HAVE_DATA, reason="needs the nb-070 bootstrap output")
def test_round_trip_on_the_real_container(tmp_path):
    theta = fm.load_fragility_data(_ROOT, _IM_TAG, quantity="theta",
                                   verbose=False, check=False)
    fresh = fm.ModelRegistry(theta, cache_dir=tmp_path).fit("A0a", n_boot=20)
    reg = fm.ModelRegistry(theta, cache_dir=tmp_path)
    loaded = reg.fit("A0a", n_boot=20)
    assert reg.cache_path("A0a", "REML").parent.parent.name == "n120_3s5s"
    for f, l in zip(fresh, loaded):
        assert reg.cache_log[("A0a", l.estimator)].startswith("loaded")
        assert_same_fit(f, l)


# --------------------------------------------------------------------------- #
# FX-only models are cached in the fixed record set's folder                   #
# --------------------------------------------------------------------------- #

def test_fx_only_models_are_cached_in_fx_cache_dir(toy, tmp_path):
    """A2 (MSA-FX vs IDA-FX) uses no site-specific arm, so with ``fx_cache_dir`` set its
    fit goes to (and is loaded from) the FX folder, shared by every study; a model using
    the SS arm (A0a) stays in the study's ``cache_dir``. Without ``fx_cache_dir`` both
    go to ``cache_dir`` as before."""
    specs = {"A0a": _A0A, "A2": fm.MODEL_SPECS["A2"]}
    study, fx = tmp_path / "study", tmp_path / "fx"
    reg = _reg(toy[0], study, specs=specs, fx_cache_dir=fx)
    assert reg.cache_path("A2", "REML").is_relative_to(fx)
    assert reg.cache_path("A0a", "REML").is_relative_to(study)
    reg.fit("A2", n_boot=20)
    assert reg.cache_path("A2", "REML").is_file()
    assert not any(study.rglob("A2_*.pkl"))
    again = _reg(toy[0], tmp_path / "another_study", specs=specs, fx_cache_dir=fx)
    again.fit("A2", n_boot=20)
    assert _status(again, "A2").startswith("loaded")          # shared across studies
    plain = _reg(toy[0], study, specs=specs)
    assert plain.cache_path("A2", "REML").is_relative_to(study)
