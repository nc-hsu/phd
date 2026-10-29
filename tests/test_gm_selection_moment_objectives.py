"""Tests for the moment-based selection objectives (``__mu_sigma``, ``__ks_mu_sigma``).

Two kinds of test:

- **Synthetic tests** that need nothing on disk: the objective formulas against hand
  computations, the parameter validation, the lambda normalisation, target lookup by
  IM name, the acceptance helper.
- **Real-data tests**, skipped cleanly when the inputs are absent:
    * the FINGERPRINT GUARD -- every stripe of the original and the KS-optimised
      selection still matches its on-disk manifest under the new code (if this fails,
      the next run of nb 032 / 032-optimised would reselect and overwrite them);
    * the KS refactor is bit-identical to the pre-refactor function, and J' with
      lambdas (1, 0, 0) equals the KS objective exactly, on real stored sets.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openquake.hazardlib.imt import PGA, SA, RSD595, AvgSA

import phd_project.scripts.WP1_ground_motion_set.gm_selection as gms

try:
    from phd_project.config import config

    _CFG = config.load_config()
    # the stripe pickles of the original and the KS-optimised record sets
    _ORIG = config.ss_dir(_CFG, "results", "record_selection", "AvgSA_03__ks")
    _OPT = config.ss_dir(_CFG, "results", "record_selection", "AvgSA_03__opt_ks")
    # Both were selected on the same frozen stripe grid (the IMLs as they were then).
    _GRID = config.stripe_grid_path(_CFG, "AvgSA_03__ks")
    _GCIM = Path(_CFG["proc_data"]["gcim_dists"]) / "gcim_dist_AvgSA_03.pickle"
    _SHARDS = Path(_CFG["proc_data"]["AvgSA_03_disagg_data_shards"])
    _HAVE_SETS = _ORIG.is_dir() and _OPT.is_dir() and _GCIM.is_file()
    _HAVE_SETUP = _HAVE_SETS and _SHARDS.is_dir()
except Exception:  # pragma: no cover - bare checkout
    _HAVE_SETS = _HAVE_SETUP = False

needs_sets = pytest.mark.skipif(not _HAVE_SETS, reason="stored selections / GCIM file absent")
needs_setup = pytest.mark.skipif(not _HAVE_SETUP, reason="selection inputs absent")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

# The 22 selection IMs of the AvgSA_03 selection, built exactly as
# setup_AvgSA03_gcim_gm_selection builds them (without loading any data).
_SA_PERIODS = np.round(np.geomspace(0.025, 3, num=20), 3)
_SELECTION_IMTS = [RSD595(), PGA()] + [SA(p) for p in _SA_PERIODS]
_IM_NAMES = [im.string for im in _SELECTION_IMTS]


def _light_ctx(weight_rsd595=0.25, scheme=None):
    """A selection context with just the keys the objective / fingerprint helpers read.

    The weights come from the same helper the real setup uses (_im_weights), so they
    are bit-identical to the real context's.
    """
    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import _im_weights
    ctx = {
        "n_ensembles": 20, "n_samples": 30, "p_value": 0.05, "usable_T": 3,
        "max_n_recs": 5, "m_bound_model": "tarbali_and_bradley_2016",
        "d_bound_model": "tarbali_and_bradley_2016",
        "vs30_bound_model": "tarbali_and_bradley_2016", "occurence": True,
        "conditioning_imt": AvgSA([0, 3]),
        "selection_imts": _SELECTION_IMTS,
        "imt_weights": _im_weights(_SELECTION_IMTS, {"RSD595": weight_rsd595}),
    }
    if scheme is not None:
        ctx["selection_scheme"] = gms.normalise_selection_scheme(scheme)
    return ctx


def _frozen_ks_l2(rec_ims, target_cdfs_list, upper_ks_bounds_list, lower_ks_bounds_list,
                  im_weights, n_recs, penalty_constant):
    """Verbatim copy of weighted_ks_statistic_and_L2norm_penalty BEFORE the refactor
    into _weighted_ks_terms. The original and KS-optimised selections were scored with
    this; the refactored function must return the same float, bit for bit."""
    sorted_ims = np.sort(rec_ims, axis=0)
    steps_up = np.linspace(1.0/n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs-1.0)/n_recs, n_recs).reshape(-1, 1)
    total_weighted_ks = 0.0
    total_violation = 0.0
    for i in range(len(target_cdfs_list)):
        if im_weights[i] == 0:
            continue
        xp = target_cdfs_list[i][:, 0]
        fp = target_cdfs_list[i][:, 1]
        fp_upper = upper_ks_bounds_list[i][:, 1]
        fp_lower = lower_ks_bounds_list[i][:, 1]
        f_xi = np.interp(sorted_ims[:, i], xp, fp, left=0, right=1).reshape(-1, 1)
        f_upper_xi = np.interp(sorted_ims[:, i], xp, fp_upper).reshape(-1, 1)
        f_lower_xi = np.interp(sorted_ims[:, i], xp, fp_lower).reshape(-1, 1)
        d_plus = np.max(np.abs(steps_up - f_xi))
        d_minus = np.max(np.abs(steps_down - f_xi))
        ks_stat = max(d_plus, d_minus)
        upper_violation = np.sum(np.maximum(0, steps_up - f_upper_xi) ** 2)
        lower_violation = np.sum(np.maximum(0, f_lower_xi - steps_down) ** 2)
        total_violation += (upper_violation + lower_violation)
        total_weighted_ks += im_weights[i] * ks_stat
    return (total_violation * penalty_constant) + total_weighted_ks


def _synthetic_stats(mu, sigma, names):
    """A GCIM-stats-like frame: rows = IM names, columns mean / sigma."""
    return pd.DataFrame({"mean": mu, "sigma": sigma}, index=names)


def _fake_ensemble(ln_ims, names, ks_passed=True):
    """A minimal ensemble dict: recs with an ims_scaled block (+ the AvgSA column)."""
    ims = pd.DataFrame(np.exp(ln_ims), columns=names)
    ims.insert(2, "AvgSA([0, 3])", 1.0)          # where the real column sits
    recs = pd.concat({"ims_scaled": ims}, axis=1)
    return {"recs": recs, "ks_passed": ks_passed}


# -----------------------------------------------------------------------------
# 5. J_mu_sigma against a hand computation; zero at the target
# -----------------------------------------------------------------------------

def test_moment_objective_matches_hand_computation():
    # 4 records x 2 IMs, chosen so the moments are easy to do by hand.
    x = np.array([[0.0, 1.0],
                  [2.0, 1.0],
                  [0.0, 3.0],
                  [2.0, 3.0]])
    # column means 1, 2; sample SDs (ddof=1): sqrt(4/3) for both
    s = np.sqrt(4.0 / 3.0)
    mu_t = np.array([0.5, 2.0])
    sig_t = np.array([1.0, 2.0])
    w = np.array([0.4, 0.6])
    a_sigma = 0.3
    z = np.array([(1.0 - 0.5) / 1.0, (2.0 - 2.0) / 2.0])
    e = np.log(np.array([s / 1.0, s / 2.0]))
    expected = np.sum(w * ((1 - a_sigma) * z**2 + a_sigma * e**2))
    got = gms.weighted_moment_objective(x, mu_t, sig_t, w, a_sigma)
    assert got == pytest.approx(expected, rel=1e-14)


def test_moment_objective_is_zero_at_target_moments():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 5))
    mu_t = x.mean(axis=0)
    sig_t = x.std(axis=0, ddof=1)
    assert gms.weighted_moment_objective(x, mu_t, sig_t, np.full(5, 0.2), 0.5) == pytest.approx(0.0, abs=1e-24)


def test_sd_error_kinds():
    assert gms._sd_error(1.1, 1.0, "log") == pytest.approx(np.log(1.1))
    assert gms._sd_error(1.1, 1.0, "ratio") == pytest.approx(0.1)
    # log form is symmetric in factors
    assert gms._sd_error(1.1, 1.0)**2 == pytest.approx(gms._sd_error(1 / 1.1, 1.0)**2)
    with pytest.raises(ValueError):
        gms._sd_error(1.0, 1.0, "abs")


# -----------------------------------------------------------------------------
# 4. a_mu = 1 - a_sigma; parameter validation
# -----------------------------------------------------------------------------

def test_a_sigma_endpoints_select_one_term():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1.4, size=(30, 3))
    mu_t, sig_t, w = np.zeros(3), np.ones(3), np.array([0.5, 0.25, 0.25])
    z, e = gms._moment_errors(x, mu_t, sig_t)
    # a_sigma = 0 -> a_mu = 1: mean term only; a_sigma = 1 -> a_mu = 0: SD term only
    assert gms.weighted_moment_objective(x, mu_t, sig_t, w, 0.0) == pytest.approx(np.sum(w * z**2))
    assert gms.weighted_moment_objective(x, mu_t, sig_t, w, 1.0) == pytest.approx(np.sum(w * e**2))
    # and in between it is the convex combination
    assert gms.weighted_moment_objective(x, mu_t, sig_t, w, 0.25) == pytest.approx(
        0.75 * np.sum(w * z**2) + 0.25 * np.sum(w * e**2))


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan])
def test_a_sigma_outside_unit_interval_raises(bad):
    x = np.random.default_rng(2).normal(size=(30, 2))
    with pytest.raises(ValueError):
        gms.weighted_moment_objective(x, np.zeros(2), np.ones(2), np.full(2, 0.5), bad)
    with pytest.raises(ValueError):
        gms.normalise_selection_scheme({"objective": "mu_sigma", "a_sigma": bad})


@pytest.mark.parametrize("lams", [(-1, 1, 1), (1, -0.1, 1), (0, 0, 0), (np.nan, 1, 1)])
def test_bad_lambdas_raise(lams):
    with pytest.raises(ValueError):
        gms._normalise_lambdas(*lams)
    with pytest.raises(ValueError):
        gms.normalise_selection_scheme({"objective": "ks_mu_sigma", "lambda_ks": lams[0],
                                        "lambda_mu": lams[1], "lambda_sigma": lams[2]})


def test_scheme_validation():
    assert gms.normalise_selection_scheme(None) == gms.DEFAULT_SELECTION_SCHEME
    assert gms.normalise_selection_scheme({"rng_seed": 1}) == gms.DEFAULT_SELECTION_SCHEME
    with pytest.raises(ValueError):
        gms.normalise_selection_scheme({"objective": "sse"})
    with pytest.raises(ValueError):
        gms.normalise_selection_scheme({"acceptance": "anything"})
    with pytest.raises(ValueError):          # moment parameter next to the KS objective
        gms.normalise_selection_scheme({"a_sigma": 0.5})
    with pytest.raises(ValueError):          # the wrong objective's parameter
        gms.normalise_selection_scheme({"objective": "mu_sigma", "lambda_mu": 1})
    # normalising twice is a no-op
    sc = gms.normalise_selection_scheme({"objective": "ks_mu_sigma", "lambda_ks": 2})
    assert gms.normalise_selection_scheme(sc) == sc


# -----------------------------------------------------------------------------
# 3. lambda normalisation
# -----------------------------------------------------------------------------

def _synthetic_ks_kwargs(n_ims, n_recs=30):
    """KS kwargs on a standard-normal target with a generous band (synthetic)."""
    from scipy.stats import norm
    xp = np.linspace(-5, 5, 401)
    cdf = norm.cdf(xp)
    tgt = [np.column_stack([xp, cdf])] * n_ims
    up = [np.column_stack([xp, np.minimum(cdf + 0.24, 1)])] * n_ims
    lo = [np.column_stack([xp, np.maximum(cdf - 0.24, 0)])] * n_ims
    return {"target_cdfs_list": tgt, "upper_ks_bounds_list": up, "lower_ks_bounds_list": lo,
            "im_weights": np.full(n_ims, 1 / n_ims), "n_recs": n_recs, "penalty_constant": 10}


def test_lambda_scale_invariance():
    rng = np.random.default_rng(3)
    x = rng.normal(0.4, 1.6, size=(30, 4))      # off-target: every term non-zero
    kw = _synthetic_ks_kwargs(4)
    mt = dict(mu_target=np.zeros(4), sigma_target=np.ones(4))
    j1 = gms.weighted_ks_and_moment_objective(x, **kw, **mt, lambda_ks=1, lambda_mu=1, lambda_sigma=1)
    j5 = gms.weighted_ks_and_moment_objective(x, **kw, **mt, lambda_ks=5, lambda_mu=5, lambda_sigma=5)
    assert j1 == pytest.approx(j5, rel=1e-15)
    # and the penalty c*V is NOT scaled by the normalisation: with lambdas (0, 1, 0)
    # J' = c*V + sum w z^2 exactly
    t = gms.moment_objective_terms(x, kw, np.zeros(4), np.ones(4), kw["im_weights"])
    j010 = gms.weighted_ks_and_moment_objective(x, **kw, **mt, lambda_ks=0, lambda_mu=1, lambda_sigma=0)
    assert t["V"] > 0                            # the test would be empty otherwise
    assert j010 == pytest.approx(t["cV"] + t["sum_wz2"], rel=1e-14)


# -----------------------------------------------------------------------------
# 6. target alignment by IM name
# -----------------------------------------------------------------------------

def test_targets_looked_up_by_name_not_position():
    rng = np.random.default_rng(4)
    n = len(_IM_NAMES)
    mu = rng.normal(size=n)
    sig = rng.uniform(0.2, 0.8, size=n)
    stats = _synthetic_stats(mu, sig, _IM_NAMES)
    shuffled = stats.sample(frac=1, random_state=7)
    assert list(shuffled.index) != _IM_NAMES     # really reordered
    x = rng.normal(size=(30, n))
    ctx = _light_ctx(0.40, {"objective": "mu_sigma", "a_sigma": 0.5})
    vals = []
    for st in (stats, shuffled):
        gcim = {(1, 0.5): {"stats": st}}
        f, kw = gms.objective_and_kwargs(1, 0.5, gcim, ctx, im_columns=_IM_NAMES)
        vals.append(f(x, **kw))
    assert vals[0] == vals[1]


def test_record_columns_out_of_selection_order_raise():
    stats = _synthetic_stats(np.zeros(len(_IM_NAMES)), np.ones(len(_IM_NAMES)), _IM_NAMES)
    ctx = _light_ctx(0.40, {"objective": "mu_sigma"})
    swapped = [_IM_NAMES[1], _IM_NAMES[0]] + _IM_NAMES[2:]
    with pytest.raises(ValueError):
        gms.objective_and_kwargs(1, 0.5, {(1, 0.5): {"stats": stats}}, ctx, im_columns=swapped)


def test_record_im_columns_drop_conditioning_im():
    e = _fake_ensemble(np.zeros((3, len(_IM_NAMES))), _IM_NAMES)
    assert gms._record_im_columns(e, _light_ctx()) == _IM_NAMES


# -----------------------------------------------------------------------------
# 7. the acceptance helper
# -----------------------------------------------------------------------------

def _old_ensemble_passes(ensemble):
    """The pre-change _ensemble_passes, verbatim."""
    return ensemble is not None and ensemble["ks_passed"]


def test_acceptance_ks_reproduces_old_rule():
    full = _fake_ensemble(np.zeros((30, 3)), ["a", "b", "c"], ks_passed=True)
    fail = _fake_ensemble(np.zeros((30, 3)), ["a", "b", "c"], ks_passed=False)
    for e in (None, full, fail):
        assert gms._ensemble_passes(e) == bool(_old_ensemble_passes(e))
        assert gms.ensemble_accepted(e, "ks") == bool(_old_ensemble_passes(e))


def test_acceptance_complete_set():
    fail = _fake_ensemble(np.zeros((30, 3)), ["a", "b", "c"], ks_passed=False)
    assert gms.ensemble_accepted(fail, "complete_set", 30)          # fails KS, accepted
    assert not gms.ensemble_accepted(None, "complete_set", 30)      # no set
    short = _fake_ensemble(np.zeros((29, 3)), ["a", "b", "c"], ks_passed=True)
    assert not gms.ensemble_accepted(short, "complete_set", 30)     # incomplete
    holed = _fake_ensemble(np.zeros((30, 3)), ["a", "b", "c"])
    holed["recs"].iloc[5] = np.nan
    assert not gms.ensemble_accepted(holed, "complete_set", 30)     # empty row
    with pytest.raises(ValueError):
        gms.ensemble_accepted(fail, "nope")


# -----------------------------------------------------------------------------
# fingerprint helper: the default adds nothing; a non-default scheme adds a key
# -----------------------------------------------------------------------------

def test_default_scheme_adds_nothing_to_fingerprint():
    plain = gms._selection_ctx_fingerprint_inputs(_light_ctx())
    explicit = gms._selection_ctx_fingerprint_inputs(
        _light_ctx(scheme={"objective": "ks", "acceptance": "ks"}))
    assert "selection_scheme" not in plain
    assert plain.keys() == explicit.keys()
    assert all(np.array_equal(plain[k], explicit[k]) for k in plain)


def test_nondefault_scheme_enters_fingerprint_and_lambdas_hash_normalised():
    a = gms._selection_ctx_fingerprint_inputs(_light_ctx(0.4, {"objective": "ks_mu_sigma"}))
    b = gms._selection_ctx_fingerprint_inputs(_light_ctx(0.4, {
        "objective": "ks_mu_sigma", "lambda_ks": 5, "lambda_mu": 5, "lambda_sigma": 5}))
    c = gms._selection_ctx_fingerprint_inputs(_light_ctx(0.4, {
        "objective": "ks_mu_sigma", "lambda_ks": 2, "lambda_mu": 1, "lambda_sigma": 1}))
    assert "selection_scheme" in a
    assert a["selection_scheme"] == b["selection_scheme"]           # same normalised scheme
    assert a["selection_scheme"] != c["selection_scheme"]           # a real change shows


def test_ctx_config_guard():
    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
        SELECTION_CONFIG, SELECTION_CONFIG_OPTIMISED, SELECTION_CONFIG_MU_SIGMA,
        SELECTION_CONFIG_KS_MU_SIGMA)
    # existing configs with the plain context: pass
    gms.check_ctx_matches_config(_light_ctx(), SELECTION_CONFIG)
    gms.check_ctx_matches_config(_light_ctx(), SELECTION_CONFIG_OPTIMISED)
    # new configs with the right context: pass
    gms.check_ctx_matches_config(_light_ctx(0.40, SELECTION_CONFIG_MU_SIGMA), SELECTION_CONFIG_MU_SIGMA)
    gms.check_ctx_matches_config(_light_ctx(0.40, SELECTION_CONFIG_KS_MU_SIGMA), SELECTION_CONFIG_KS_MU_SIGMA)
    # plain context against a new config: wrong scheme -> raise
    with pytest.raises(ValueError):
        gms.check_ctx_matches_config(_light_ctx(), SELECTION_CONFIG_MU_SIGMA)
    # right scheme, wrong weight -> raise
    with pytest.raises(ValueError):
        gms.check_ctx_matches_config(_light_ctx(0.25, SELECTION_CONFIG_MU_SIGMA), SELECTION_CONFIG_MU_SIGMA)
    # other scheme -> raise
    with pytest.raises(ValueError):
        gms.check_ctx_matches_config(_light_ctx(0.40, SELECTION_CONFIG_KS_MU_SIGMA), SELECTION_CONFIG_MU_SIGMA)


# -----------------------------------------------------------------------------
# 2. real stored sets: refactor bit-identical; J'(1,0,0) == KS objective exactly
# -----------------------------------------------------------------------------

def _stored_sets(folder, n=6):
    """A spread of stored stripes: n pickles evenly spaced through the folder."""
    files = sorted(Path(folder).glob("site_*__stripe_iml_*__gm_selection.pickle"))
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    out = []
    for i in idx:
        with open(files[i], "rb") as f:
            e = pickle.load(f)
        site = int(files[i].name.split("__")[0].split("_")[1])
        out.append((site, float(e["cond_iml"]), e))
    return out


@pytest.fixture(scope="module")
def gcim_dists():
    with open(_GCIM, "rb") as f:
        return pickle.load(f)


def _gcim_key(gcim_dists, site, iml):
    return next(k for k in gcim_dists if k[0] == site and np.isclose(k[1], iml))


@needs_sets
@pytest.mark.parametrize("ss_id", ["AvgSA_03__ks", "AvgSA_03__opt_ks"])
def test_ks_refactor_and_jprime_100_bit_identical_on_real_sets(gcim_dists, ss_id):
    ctx = _light_ctx()
    jctx = _light_ctx(0.25, {"objective": "ks_mu_sigma", "lambda_ks": 1,
                             "lambda_mu": 0, "lambda_sigma": 0})
    for site, iml, e in _stored_sets(config.ss_dir(_CFG, "results", "record_selection", ss_id)):
        key = _gcim_key(gcim_dists, site, iml)
        cols = gms._record_im_columns(e, ctx)
        x = np.log(e["recs"]["ims_scaled"].drop(columns="AvgSA([0, 3])").to_numpy())
        kw = gms._optimise_obj_kwargs(*key, gcim_dists, ctx)
        new = gms.weighted_ks_statistic_and_L2norm_penalty(x, **kw)
        assert new == _frozen_ks_l2(x, **kw)                         # refactor: bit-identical
        f, jkw = gms.objective_and_kwargs(*key, gcim_dists, jctx, im_columns=cols)
        assert f is gms.weighted_ks_and_moment_objective
        assert f(x, **jkw) == new                                    # J'(1,0,0): exact
        # the dispatch under the default scheme returns the old function + kwargs
        f0, kw0 = gms.objective_and_kwargs(*key, gcim_dists, ctx, im_columns=cols)
        assert f0 is gms.weighted_ks_statistic_and_L2norm_penalty
        assert f0(x, **kw0) == new


@needs_sets
def test_moment_terms_on_real_set_match_stored_moments(gcim_dists):
    # The z / r computed by the objective's helper must agree with the stripe's own
    # stored mu_lnIM_recs / sig_lnIM_recs (written by assemble_ensemble_dict).
    ctx = _light_ctx()
    for site, iml, e in _stored_sets(_OPT, 3):
        if "sig_lnIM_recs" not in e:
            continue
        key = _gcim_key(gcim_dists, site, iml)
        cols = gms._record_im_columns(e, ctx)
        mu_t, sig_t = gms.gcim_moment_targets(gcim_dists[key]["stats"], cols)
        x = np.log(e["recs"]["ims_scaled"][cols].to_numpy())
        z, er = gms._moment_errors(x, mu_t, sig_t)
        np.testing.assert_allclose(np.exp(er) * sig_t, e["sig_lnIM_recs"][cols].to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(z * sig_t + mu_t, np.log(e["mu_lnIM_recs"][cols].to_numpy()), rtol=1e-10, atol=1e-12)


# -----------------------------------------------------------------------------
# 1. THE FINGERPRINT GUARD: existing stripes still read as valid
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def plain_setup():
    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import (
        setup_AvgSA03_gcim_gm_selection)
    _, disagg_stats, _, ctx, _ = setup_AvgSA03_gcim_gm_selection(sites=(), imls_for_selection_fp=_GRID)
    return disagg_stats, ctx


@pytest.mark.skip(reason="database rebuilt 2026-10-29 (nb 030 removes the unusable NGA-Sub "
                         "records); the older sets were selected from "
                         "ESM-NGAsub_combined_old.csv, so their stripes are stale by design")
@needs_setup
@pytest.mark.parametrize("config_name,ss_id", [
    ("SELECTION_CONFIG", "AvgSA_03__ks"),
    ("SELECTION_CONFIG_OPTIMISED", "AvgSA_03__opt_ks"),
])
def test_existing_stripes_still_valid(plain_setup, config_name, ss_id):
    """Every wanted stripe of the original / KS-optimised selection is VALID under the
    new code -- exactly as the existing notebooks compute the check (plain setup,
    their own config). If this fails, the next nb 032 run would reselect them."""
    import phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection as su
    disagg_stats, ctx = plain_setup
    sc = getattr(su, config_name)
    fp_fn, superseded = su.stripe_fingerprints(sc, su.stripe_source_fps(), ctx, disagg_stats)
    wanted = su.wanted_stripe_keys(imls_for_selection_fp=config.stripe_grid_path(_CFG, ss_id))
    to_compute, valid = gms.find_stale_stripes(wanted, config.ss_dir(_CFG, "results", "record_selection", ss_id), fp_fn,
                                               superseded=superseded)
    assert to_compute == [], f"{len(to_compute)} stripes read as stale, e.g. {to_compute[:5]}"
    assert len(valid) == len(wanted)


@needs_setup
def test_scheme_setup_for_existing_config_equals_plain(plain_setup):
    """setup_..._for_scheme(SELECTION_CONFIG_OPTIMISED) gives the plain context."""
    import phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection as su
    _, _, _, ctx2, _ = su.setup_AvgSA03_gcim_gm_selection_for_scheme(
        su.SELECTION_CONFIG_OPTIMISED, sites=())
    _, ctx = plain_setup
    assert "selection_scheme" not in ctx2
    a = gms._selection_ctx_fingerprint_inputs(ctx)
    b = gms._selection_ctx_fingerprint_inputs(ctx2)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


# -----------------------------------------------------------------------------
# Empty exclusion list (since 2026-10-29 the known bad records are removed in nb 030)
# -----------------------------------------------------------------------------

def test_empty_exclusion_list_gives_a_real_current_fingerprint(monkeypatch):
    """With no exclusion entries, the CURRENT fingerprint (the one new manifests record)
    must hash the config's own db_exclusions -- the empty tuple -- and not the
    matching-only placeholder whose hash is None. Nothing is superseded."""
    seen = {}

    def fake_fp(site, iml, source_fps, ctx, sc, disagg_stats):
        # stands in for stripe_input_fingerprint: records what it was given
        seen["sc"] = sc
        return {"db_exclusions": {"hash": repr(sc["db_exclusions"])} if "db_exclusions" in sc
                else {"hash": "missing"}}

    monkeypatch.setattr(gms, "stripe_input_fingerprint", fake_fp)
    monkeypatch.setattr(gms, "check_ctx_matches_config", lambda ctx, sc: None)
    sc = {"db_exclusions": gms._exclusion_identities([])}
    fp_fn, superseded = gms.stripe_fingerprints_by_exclusion_version(sc, [], {}, {}, None)
    assert superseded == []
    fp = fp_fn(1, 0.5)
    assert fp["db_exclusions"]["hash"] == repr(())
    assert seen["sc"]["db_exclusions"] == ()


def test_unavailable_records_list_is_empty():
    """The in-memory list is empty since nb 030 removes the records from the database."""
    import phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection as su
    assert su.UNAVAILABLE_RECORDS == []
    assert su.EXCLUSION_LIST_VERSION == 0
    for name in ("SELECTION_CONFIG", "SELECTION_CONFIG_OPTIMISED",
                 "SELECTION_CONFIG_MU_SIGMA", "SELECTION_CONFIG_KS_MU_SIGMA"):
        assert getattr(su, name)["db_exclusions"] == ()
