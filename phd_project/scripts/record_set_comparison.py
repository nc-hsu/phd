"""Original vs optimised site-specific (SS) record sets: is a full re-run justified?

This module is the reusable logic behind nb 034
(``notebooks/03_WP1_ground_motion_set/034-comparison_original_and_optimised_gm_sets.ipynb``).
The plan it implements is ``dump/plan_record_set_rerun_decision.md``; read that first; the
section numbers below (Step 1-5, D1-D6) refer to it.

The question in one paragraph
-----------------------------
Every SS record set (one per site and stripe IML, 30 records, selected by GCIM conditioned on
AvgSA[0,3] = IML) was re-selected with a stronger optimisation step. Only the SS arm ``a_i``
can change: the fixed-set arms ``b_j``, ``c_j`` use FEMA P695 and are untouched. The decision is
all-or-nothing, so it is judged at STUDY level, on the A0c fit (``y0_avg = a - c_avg``, design x
site crossed, full V_known) and the A3ab regression on theta. We cannot run the new NLTHA, so the
change in every ``a_i`` is PREDICTED from the existing per-record collapse outcomes (Step 3) and
pushed through those fits (Step 4).

Analogy. Think of each record set as a jury of 30 drawn to represent a population (the GCIM
target). The optimiser swapped most jurors for ones who represent it slightly better. We do not
get to hear the new jury's verdict; instead we learn, from the old jury, how each juror's
characteristics (duration, spectral shape) predicted their vote, and ask what the new jury
WOULD have said. Comparing predicted with predicted means the model's errors hit both juries
equally and largely cancel.

Layout
------
1. The selection IM vector and its weights (exactly as the selector built them).
2. Loading the two folders and matching sets to the MSA's structure-stripes   -> Step 1
3. Scoring each set against its GCIM target (the optimiser's own objective)   -> Step 2
4. Firth logistic regression, predicted collapse fractions, fragility refits  -> Step 3
5. Shifting the SS arm and refitting A0c (theta, beta) and A3ab (theta)       -> Step 4

Nothing here writes to ``results/04_record_selection/``; both record-set folders are read-only.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from openquake.hazardlib.imt import PGA, SA, RSD595

import phd_project.scripts.WP1_ground_motion_set.gm_selection as gms
import phd_project.scripts.data_simulation as dsim
import phd_project.scripts.gcim_saratio as gsr
import phd_project.scripts.fragility_data_models as fm
import phd_project.scripts.metaregression_analysis as mra


# =============================================================================
# 1. The selection IM vector
# =============================================================================
# The records were selected to match the GCIM on exactly these 22 IMs (see
# setup_AvgSA03_gm_selection.setup_AvgSA03_gcim_gm_selection). Every IM-level comparison in
# this module uses exactly this set -- no more (SaRatio was never selected on, so it is not
# compared, scored or used as a predictor), no less. The conditioning IM AvgSA([0, 3]) is NOT
# in it: every record of a set is scaled to the same AvgSA, so it carries no information
# within a set.
#
# The construction is repeated here (three lines) rather than calling the setup function,
# because that function loads the 120 MB ground-motion database and the disaggregation shards.
# `check_selection_ims` asserts against the pickles that the two cannot have drifted apart.

SA_PERIODS = np.round(np.geomspace(0.025, 3.0, num=20), 3)   # same rounding as the selector
SELECTION_IMTS = [RSD595(), PGA()] + [SA(float(p)) for p in SA_PERIODS]
SELECTION_IMS: list[str] = [im.string for im in SELECTION_IMTS]
CONDITIONING_IM = "AvgSA([0, 3])"

# The IM of most interest. After conditioning on AvgSA, duration has the largest effect on
# the collapse median, so it leads every table and figure.
LEAD_IM = "RSD595"
LEAD_IM_LABEL = "Ds5-95"

# The 21 spectral selection IMs (PGA + 20 SA). Within a stripe their log-average is pinned by
# the AvgSA scaling, so what varies between records is spectral SHAPE.
SPECTRAL_IMS: list[str] = [im for im in SELECTION_IMS if im != LEAD_IM]

N_RECORDS = 30           # records per set (selection n_samples)
P_VALUE = 0.05           # KS significance level used for the selection's KS bands
PENALTY_CONSTANT = 10.0  # the optimiser's default band-violation penalty (gm_selection)
IML_TOL = 6e-4           # set <-> stripe IML matching tolerance (filenames round to 3 dp)


def selection_weights() -> np.ndarray:
    """The 22 IM weights, built by the SAME helper the selector used.

    ``_im_weights(selection_imts, {"RSD595": 0.25})`` gives Ds5-95 a quarter of the weight and
    splits the rest equally over the 21 spectral IMs (0.75 / 21 each). Importing the helper,
    rather than re-typing 0.25, means a change to the selector's weighting cannot silently
    leave this module scoring with stale weights.
    """
    # imported here: the setup module loads the project config at import time, which is
    # cheap, but keeping it local makes the dependency explicit at the one place it is used.
    from phd_project.scripts.WP1_ground_motion_set.setup_AvgSA03_gm_selection import _im_weights
    return _im_weights(SELECTION_IMTS, {"RSD595": 0.25})


def check_selection_ims(ens: dict) -> None:
    """Assert that a loaded set carries exactly the selection IMs, in the selector's order.

    The optimiser builds its target-CDF list by iterating ``ks_bounds`` and the record matrix
    from ``ims_scaled`` minus the conditioning column; both must line up with
    :data:`SELECTION_IMS` position by position, or the weights would be applied to the wrong
    IMs. Cheap, so it is run on every set loaded.
    """
    rec_cols = [c for c in ens["recs"]["ims_scaled"].columns if c != CONDITIONING_IM]
    if rec_cols != SELECTION_IMS:
        raise ValueError(f"record IM columns {rec_cols} != selection IMs {SELECTION_IMS}")
    if list(ens["ks_bounds"]) != SELECTION_IMS:
        raise ValueError(f"ks_bounds keys {list(ens['ks_bounds'])} != selection IMs")


# =============================================================================
# 2. Loading and matching (Step 1)
# =============================================================================

_FNAME = re.compile(r"site_(\d+)__stripe_iml_(\d+)pt(\d+)__gm_selection\.pickle$")

# Only these entries of a stripe pickle are needed. The original pickles also carry the GCIM
# simulation arrays (sims, ctxs, eps, ...), which the optimised ones do not; keeping only the
# common subset makes the two versions structurally identical and roughly halves memory.
_KEEP = ("recs", "ks_bounds", "gcim_quantiles", "cond_iml", "ks_passed")


def load_record_sets(folder: Path | str) -> dict[tuple[int, float], dict]:
    """Load every stripe pickle of one record-set folder, keyed ``(site, cond_iml)``.

    The key uses the pickle's OWN ``cond_iml``, not the IML in the filename. The filename tag is
    rounded to 3 decimals (0.6049 is stored as ``00pt605``), while the MSA stripe IMLs are not;
    keying on the filename made 64 of the 437 run sets unmatchable (plan 0.2).

    The folder is opened read-only; nothing is ever written back to it.
    """
    out = {}
    for fp in sorted(Path(folder).glob("site_*__stripe_iml_*__gm_selection.pickle")):
        m = _FNAME.search(fp.name)
        if m is None:
            continue
        with open(fp, "rb") as f:
            raw = pickle.load(f)
        ens = {k: raw[k] for k in _KEEP}
        check_selection_ims(ens)
        if len(ens["recs"]) != N_RECORDS:
            raise ValueError(f"{fp.name}: {len(ens['recs'])} records, expected {N_RECORDS}")
        key = (int(m.group(1)), round(float(ens["cond_iml"]), 6))
        if key in out:
            raise ValueError(f"duplicate (site, cond_iml) {key} in {folder}")
        out[key] = ens
    if not out:
        raise FileNotFoundError(f"no stripe pickles found in {folder}")
    return out


def match_set_key(site: int, iml: float, keys: Sequence[tuple[int, float]],
                  tol: float = IML_TOL) -> tuple[int, float]:
    """Return the ONE set key at ``site`` whose ``cond_iml`` is within ``tol`` of ``iml``.

    Raises on zero or on several matches (D5: raise and stop, never guess). ``tol`` is a bit
    over half the 3-dp filename resolution, so it can absorb rounding but never bridge two
    genuinely different stripes (the closest stripes at a site are 0.015 g apart).
    """
    hits = [k for k in keys if k[0] == site and abs(k[1] - iml) <= tol]
    if len(hits) != 1:
        raise KeyError(f"site {site}, IML {iml}: expected exactly one record set within "
                       f"{tol}, found {hits}")
    return hits[0]


def classify_sets(orig: dict, opt: dict) -> pd.DataFrame:
    """Step 1: compare the two versions of every set on record identity and scale factor.

    A record's identity is its ``filename`` (event, station, location AND component -- the two
    horizontal components of one recording are different records to the MSA). For records kept
    in both versions the scale factor ``alpha`` must be identical, because ``alpha = IML /
    AvgSA(record)`` is deterministic; a difference would mean the two folders were built from
    different databases, and is raised rather than reported.

    Classification: ``unchanged`` (30/30 kept), ``fully changed`` (0 kept), else ``partly
    changed``.
    """
    if set(orig) != set(opt):
        raise KeyError(f"the two folders cover different (site, IML) sets: only original "
                       f"{sorted(set(orig) - set(opt))}, only optimised "
                       f"{sorted(set(opt) - set(orig))}")
    rows = []
    for key in sorted(orig):
        mo, mn = orig[key]["recs"]["metadata"], opt[key]["recs"]["metadata"]
        ao = dict(zip(mo["filename"], mo["alpha"]))
        an = dict(zip(mn["filename"], mn["alpha"]))
        common = set(ao) & set(an)
        bad = [f for f in common if not np.isclose(ao[f], an[f], rtol=1e-10)]
        if bad:
            raise ValueError(f"set {key}: kept records with different alpha: {bad}")
        n_common = len(common)
        status = ("unchanged" if n_common == N_RECORDS
                  else "fully changed" if n_common == 0 else "partly changed")
        rows.append({
            "site": key[0], "iml": key[1], "n_common": n_common,
            "n_swapped": N_RECORDS - n_common, "status": status,
            # how the record pool shifted: tectonic-region mix and database mix
            "n_esm_orig": int((mo["database"] != "NGASub").sum()),
            "n_esm_opt": int((mn["database"] != "NGASub").sum()),
            "ks_passed_orig": bool(orig[key]["ks_passed"]),
            "ks_passed_opt": bool(opt[key]["ks_passed"]),
        })
    return pd.DataFrame(rows)


def load_flag_record_names(frag_root: Path | str, site: int, n_storeys: int
                           ) -> dict[float, list[str]]:
    """The 30 record filenames behind each stripe of one structure's MSA, in slot order.

    Read from ``site_{s}/{n}s_cbf_dc2_site{s}_msa_collapseflags_AvgSA_03.json``. This is what
    lets the 0/1 flag in column ``k`` of the collapse-flag table be tied to a specific record,
    and so to that record's IMs in the pickle.
    """
    fp = (Path(frag_root) / f"site_{site}" /
          f"{n_storeys}s_cbf_dc2_site{site}_msa_collapseflags_AvgSA_03.json")
    with open(fp) as f:
        js = json.load(f)
    return {round(float(iml), 6): [v["record_names"][str(k)] for k in range(N_RECORDS)]
            for iml, v in js.items()}


def map_structure_stripes(flags: pd.DataFrame, sets: dict, frag_root: Path | str
                          ) -> pd.DataFrame:
    """Map every structure-stripe the MSA ran to its record set, and verify the records.

    ``flags`` is :func:`mra.load_site_msa_collapse_flags`'s frame, indexed ``(site, n_storeys,
    stripe_iml)``. For each structure-stripe this finds the set (:func:`match_set_key`) and
    checks that the flag JSON's 30 record names are the set's 30 filenames IN THE SAME ORDER --
    so slot ``k`` of the flags is row ``k`` of the pickle. Any mismatch raises.

    The returned frame also says whether the set is shared by the 3s and 5s structures at that
    site: wherever both ran the same stripe IML they ran the same 30 records (CLAUDE.md, the
    same-site coupling), so a change to that set moves BOTH rows.
    """
    keys = list(sets)
    names_cache: dict[tuple[int, int], dict] = {}
    rows = []
    for site, n, iml in flags.index:
        key = match_set_key(int(site), float(iml), keys)
        if (site, n) not in names_cache:
            names_cache[(site, n)] = load_flag_record_names(frag_root, int(site), int(n))
        names = names_cache[(site, n)].get(round(float(iml), 6))
        if names is None:
            raise KeyError(f"site {site} {n}s: no record names for stripe {iml} in the flag JSON")
        pickle_names = list(sets[key]["recs"]["metadata"]["filename"])
        if names != pickle_names:
            raise ValueError(f"site {site} {n}s stripe {iml}: flag record names do not match "
                             f"set {key} slot by slot")
        rows.append({"site": int(site), "n_storeys": int(n), "stripe_iml": float(iml),
                     "set_site": key[0], "set_iml": key[1]})
    out = pd.DataFrame(rows)
    n_users = out.groupby(["set_site", "set_iml"])["n_storeys"].transform("nunique")
    out["shared_3s_5s"] = n_users == 2
    return out


# =============================================================================
# 3. Scoring against the GCIM (Step 2)
# =============================================================================
# The score MUST be the optimiser's own objective (task, Step 2), so `objective_kwargs`
# rebuilds `gm_selection._optimise_obj_kwargs` from the stored `ks_bounds`, and the value comes
# from `gm_selection.weighted_ks_statistic_and_L2norm_penalty` itself. The stored ks_bounds are
# bit-identical to the ones the optimiser recomputes from nb 031's GCIM file (checked in the
# tests), so this is the same number without loading the 77 MB GCIM pickle.
#
# What the objective is (plan 0.4). For IM m with KS distance D_m (the largest vertical gap
# between the set's empirical CDF and the target CDF),
#
#     J = sum_m w_m D_m  +  c * sum_m (squared amounts by which the empirical CDF pokes out of
#                                      the 95 % KS band)
#
# The squares are on the band VIOLATIONS, not on D_m. No KS^2 appears anywhere here (user
# instruction). Every set passes the KS test, so the penalty is ~0 and J ~ sum_m w_m D_m.


def objective_kwargs(ens: dict, weights: np.ndarray,
                     penalty_constant: float = PENALTY_CONSTANT) -> dict:
    """``gm_selection._optimise_obj_kwargs`` for one set, built from its stored ``ks_bounds``.

    Line for line the same construction: each ``ks_bounds[m]`` is a (grid, 4) array with
    columns ``[IM value, lower band, target CDF, upper band]``; the optimiser drops the first
    grid row (IM value 0, whose log is -inf) and works in ln IM.
    """
    kb = ens["ks_bounds"]
    return {
        "target_cdfs_list": [np.column_stack([np.log(kb[m][1:, 0]), kb[m][1:, 2]])
                             for m in SELECTION_IMS],
        "upper_ks_bounds_list": [np.column_stack([np.log(kb[m][1:, 0]), kb[m][1:, 3]])
                                 for m in SELECTION_IMS],
        "lower_ks_bounds_list": [np.column_stack([np.log(kb[m][1:, 0]), kb[m][1:, 1]])
                                 for m in SELECTION_IMS],
        "n_recs": N_RECORDS,
        "im_weights": np.asarray(weights, dtype=float),
        "penalty_constant": penalty_constant,
    }


def set_log_ims(ens: dict) -> np.ndarray:
    """``(30, 22)`` matrix of ln IM over the selection IMs, in :data:`SELECTION_IMS` order.

    ``ims_scaled`` is used because that is what the MSA ran: spectral IMs times ``alpha``.
    Ds5-95 is unaffected by amplitude scaling, so its scaled and raw values are identical.
    """
    return np.log(ens["recs"]["ims_scaled"][SELECTION_IMS].to_numpy(dtype=float))


def ks_distances(log_ims: np.ndarray, target_cdfs_list: list[np.ndarray],
                 n_recs: int = N_RECORDS) -> np.ndarray:
    """Per-IM KS distance D_m, computed exactly as inside the optimiser's objective.

    Copied from the loop of ``gm_selection.weighted_ks_statistic_and_L2norm_penalty`` (which
    returns only the weighted total, not the per-IM values). The target CDF is linearly
    interpolated at the sorted record values; D_m is the larger of the two one-sided gaps
    against the empirical-CDF steps k/n and (k-1)/n. The test suite checks that
    ``sum(w * D)`` from here plus the penalty reproduces the optimiser's total exactly.
    """
    sorted_ims = np.sort(log_ims, axis=0)
    steps_up = np.linspace(1.0 / n_recs, 1.0, n_recs).reshape(-1, 1)
    steps_down = np.linspace(0.0, (n_recs - 1.0) / n_recs, n_recs).reshape(-1, 1)
    out = np.empty(len(target_cdfs_list))
    for i, tc in enumerate(target_cdfs_list):
        f_xi = np.interp(sorted_ims[:, i], tc[:, 0], tc[:, 1], left=0, right=1).reshape(-1, 1)
        out[i] = max(np.max(np.abs(steps_up - f_xi)), np.max(np.abs(steps_down - f_xi)))
    return out


def score_set(ens: dict, weights: np.ndarray) -> dict:
    """The optimiser's objective J for one set, split into its two parts, plus per-IM D_m.

    ``J`` comes from the optimiser's own function (the authoritative number). ``wks`` is
    ``sum_m w_m D_m`` from :func:`ks_distances`; ``penalty = J - wks`` is then the
    band-violation part. Reporting the split shows whether a change in J is a change in fit
    or a set moving into or out of the KS band.
    """
    kw = objective_kwargs(ens, weights)
    x = set_log_ims(ens)
    J = float(gms.weighted_ks_statistic_and_L2norm_penalty(x, **kw))
    D = ks_distances(x, kw["target_cdfs_list"])
    wks = float(np.sum(kw["im_weights"] * D))
    return {"J": J, "wks": wks, "penalty": J - wks, "D": D}


def set_moments(ens: dict) -> pd.DataFrame:
    """Per selection IM: the set's mean and SD of ln IM next to the GCIM target's.

    Sample SD uses ddof = 1, as the selector's ``sig_lnIM_recs`` does. The GCIM ``mean`` and
    ``sigma`` (``gcim_quantiles``) are already in log units. Two scale-free comparisons:

    * ``z_mu = (mean_set - mu_GCIM) / sigma_GCIM`` -- how many target SDs the set's centre is
      off. Comparable across sites and IMs, unlike the raw difference.
    * ``r_sigma = s_set / sigma_GCIM`` -- 1 means the set has the target's spread.
    """
    x = set_log_ims(ens)
    g = ens["gcim_quantiles"].loc[SELECTION_IMS]
    out = pd.DataFrame({
        "im": SELECTION_IMS,
        "mean_ln": x.mean(axis=0),
        "sd_ln": x.std(axis=0, ddof=1),
        "gcim_mu": g["mean"].to_numpy(dtype=float),
        "gcim_sigma": g["sigma"].to_numpy(dtype=float),
    })
    out["z_mu"] = (out["mean_ln"] - out["gcim_mu"]) / out["gcim_sigma"]
    out["r_sigma"] = out["sd_ln"] / out["gcim_sigma"]
    return out


def score_table(orig: dict, opt: dict, weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 2 tables: long per-(set, version, IM) scores, and a per-set summary.

    Long table columns: ``site, iml, version, im, D, mean_ln, sd_ln, gcim_mu, gcim_sigma, z_mu,
    r_sigma``. Summary table: one row per set with J, wks and penalty for both versions and
    their differences (optimised minus original, so negative = better fit).
    """
    long_rows, summ = [], []
    for key in sorted(orig):
        s = {"site": key[0], "iml": key[1]}
        for version, sets in (("orig", orig), ("opt", opt)):
            sc = score_set(sets[key], weights)
            mo = set_moments(sets[key])
            mo["D"] = sc["D"]
            mo.insert(0, "version", version)
            mo.insert(0, "iml", key[1])
            mo.insert(0, "site", key[0])
            long_rows.append(mo)
            s |= {f"J_{version}": sc["J"], f"wks_{version}": sc["wks"],
                  f"penalty_{version}": sc["penalty"]}
        for q in ("J", "wks", "penalty"):
            s[f"d_{q}"] = s[f"{q}_opt"] - s[f"{q}_orig"]
        summ.append(s)
    return pd.concat(long_rows, ignore_index=True), pd.DataFrame(summ)


def change_table(long: pd.DataFrame) -> pd.DataFrame:
    """Optimised minus original, per (set, IM): Delta D, Delta mean ln IM, Delta SD ln IM.

    Adds the noise-floor flag (plan Step 2). The mean of 30 independent draws from the target
    wanders by about ``sigma_GCIM / sqrt(30)`` (the standard error of a mean), so a shift in the
    set mean smaller than that cannot be told apart from ordinary selection noise; the flag
    ``within_noise`` marks those.
    """
    idx = ["site", "iml", "im"]
    o = long[long["version"] == "orig"].set_index(idx)
    n = long[long["version"] == "opt"].set_index(idx)
    out = pd.DataFrame({
        "d_D": n["D"] - o["D"],
        "d_mean_ln": n["mean_ln"] - o["mean_ln"],
        "d_sd_ln": n["sd_ln"] - o["sd_ln"],
        "z_mu_orig": o["z_mu"], "z_mu_opt": n["z_mu"],
        "r_sigma_orig": o["r_sigma"], "r_sigma_opt": n["r_sigma"],
        "gcim_sigma": o["gcim_sigma"],
    })
    out["noise_floor"] = out["gcim_sigma"] / np.sqrt(N_RECORDS)
    out["within_noise"] = out["d_mean_ln"].abs() < out["noise_floor"]
    return out.reset_index()


def direction_summary(changes: pd.DataFrame, set_weights: pd.Series | None = None
                      ) -> pd.DataFrame:
    """The sign pattern of Delta mean ln IM per selection IM -- the main warning sign.

    Why the sign pattern matters. If the optimiser shifted, say, the mean duration UP at most
    sets, every affected a_i moves the same way and the shifts ACCUMULATE in the pooled mean m
    across 120 rows. If the shifts point in random directions they mostly cancel in m and only
    inflate tau_u a little. So a one-sided pattern is what to look for.

    Reported per IM: the number of sets, the fraction with a positive shift, an exact two-sided
    sign test against 1/2 (binomial), the median shift, and the fraction within the noise floor.
    ``set_weights`` (indexed ``(site, iml)``, e.g. the number of structure-stripes that ran each
    set) adds a usage-weighted fraction positive, since a set run by both structures counts
    twice in the study.
    """
    rows = []
    for im in SELECTION_IMS:
        c = changes[changes["im"] == im]
        d = c["d_mean_ln"].to_numpy()
        nz = d[d != 0]
        n_pos = int((nz > 0).sum())
        row = {"im": im, "n_sets": len(d), "frac_positive": n_pos / max(len(nz), 1),
               "sign_test_p": float(stats.binomtest(n_pos, len(nz), 0.5).pvalue) if len(nz) else np.nan,
               "median_d_mean_ln": float(np.median(d)),
               "mean_d_mean_ln_over_sigma": float((c["d_mean_ln"] / c["gcim_sigma"]).mean()),
               "frac_within_noise": float(c["within_noise"].mean()),
               "median_d_D": float(c["d_D"].median())}
        if set_weights is not None:
            w = set_weights.reindex(pd.MultiIndex.from_frame(c[["site", "iml"]])).to_numpy(float)
            row["frac_positive_usage_weighted"] = float(np.sum(w * (d > 0)) / np.sum(w))
        rows.append(row)
    return pd.DataFrame(rows)


def tradeoff_summary(changes: pd.DataFrame) -> pd.DataFrame:
    """Did the optimiser buy a better Ds5-95 fit with a worse spectral fit, or the reverse?

    Per set: the sign of Delta D for Ds5-95, against the weighted-average Delta D over the 21
    spectral IMs (equal weights, as in the objective). Returns the 2 x 2 count table
    ``improved / worsened`` for each, so the off-diagonal cells are the trade-offs.
    """
    lead = changes[changes["im"] == LEAD_IM].set_index(["site", "iml"])["d_D"]
    spec = changes[changes["im"] != LEAD_IM].groupby(["site", "iml"])["d_D"].mean()
    df = pd.DataFrame({"ds595": np.where(lead < 0, "improved", "worsened"),
                       "spectral": np.where(spec.reindex(lead.index) < 0, "improved", "worsened")},
                      index=lead.index)
    return pd.crosstab(df["ds595"], df["spectral"])


# ---- 3b. picking and drawing example sets ------------------------------------
# The summary statistics above say how the fit changed ON AVERAGE. Two example stripes make it
# concrete: the one the optimiser improved most, and the one whose Ds5-95 spread ended up
# widest relative to the target (the heavy-tail side effect, report section 2). For each, every
# selection IM is drawn as the GCIM target CDF, its 95 % KS band and the two sets' empirical
# CDFs, i.e. exactly the objects the optimiser's objective compares.

def largest_improvement_key(set_summary: pd.DataFrame) -> tuple[int, float]:
    """``(site, iml)`` of the RUN set whose objective J fell the most (most negative d_J).

    J is the optimiser's weighted KS objective (:func:`score_set`), so this is the stripe
    where the re-selection achieved the largest improvement in KS fit. Only sets the MSA
    actually ran (``used_in_msa``) are considered: an unrun set cannot affect any fragility.
    """
    run = set_summary[set_summary["used_in_msa"]]
    best = run.loc[run["d_J"].idxmin()]
    return int(best["site"]), float(best["iml"])


def lowest_score_key(set_summary: pd.DataFrame, version: str = "opt") -> tuple[int, float]:
    """``(site, iml)`` of the RUN set with the lowest objective J in ``version``.

    J is lower-is-better, so with ``version="opt"`` (default) this is the optimised set that
    sits closest to its GCIM target over all 22 IMs -- the best the re-selection achieved.
    """
    run = set_summary[set_summary["used_in_msa"]]
    best = run.loc[run[f"J_{version}"].idxmin()]
    return int(best["site"]), float(best["iml"])


def widest_lead_dispersion_key(changes: pd.DataFrame, version: str = "opt"
                               ) -> tuple[int, float]:
    """``(site, iml)`` of the RUN set with the largest Ds5-95 ``r_sigma`` (set SD / GCIM SD).

    ``version="opt"`` (default) looks at the optimised sets, where the heavy duration tail
    appeared; ``"orig"`` looks at the original ones.
    """
    c = changes[(changes["im"] == LEAD_IM) & changes["used_in_msa"]]
    worst = c.loc[c[f"r_sigma_{version}"].idxmax()]
    return int(worst["site"]), float(worst["iml"])


def _ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step-plot coordinates of an empirical CDF: (sorted values, k/n), for ``where="post"``.

    A leading point at the smallest value with height 0 makes the first step visible.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    return np.r_[x[0], x], np.r_[0.0, np.arange(1, n + 1) / n]


def plot_ecdf_grid(ens_orig: dict, ens_opt: dict, title: str, extra: Sequence[tuple] = (),
                   labels: tuple[str, str] = ("original", "optimised")):
    """4 x 6 grid, one panel per selection IM: GCIM target CDF, 95 % KS band, every set's ECDF.

    ``ens_orig`` (grey) and ``ens_opt`` (red) are drawn with ``labels``. ``extra`` adds further
    versions of the same set as ``(ens, colour, label)`` tuples, e.g. a second optimisation
    scheme; each must share the original's GCIM target.

    What is drawn, per IM ``m``
    * the target CDF and the lower/upper KS band, straight from the set's stored
      ``ks_bounds[m]`` (columns: IM value, lower band, target CDF, upper band). The two
      versions of a set share one GCIM, so the original set's bounds are used (and asserted
      identical to the optimised set's);
    * the empirical CDF of the 30 records of each version, on the SCALED IMs the MSA ran
      (:func:`set_log_ims`, exponentiated back);
    * the KS distance D of each version (:func:`ks_distances`, the optimiser's own
      computation) in the panel corner, so the picture and the numbers can be read together.

    The x axis is logarithmic because the selection works in ln IM. The first grid row of
    ``ks_bounds`` is IM = 0, whose log is -inf; it is dropped, exactly as the optimiser does.

    Returns the matplotlib Figure (the caller saves it, with ``facecolor="w"``).
    """
    import matplotlib.pyplot as plt   # local: the rest of the module does not need matplotlib

    # every version to draw: (ensemble, colour, legend label), original first
    versions = [(ens_orig, "0.45", labels[0]), (ens_opt, "C3", labels[1]), *extra]
    kb_o = ens_orig["ks_bounds"]
    for ens, _, lbl in versions[1:]:
        for m in SELECTION_IMS:
            if not np.allclose(kb_o[m], ens["ks_bounds"][m]):
                raise ValueError(f"{m}: '{lbl}' has a different GCIM target from the original set")

    # the scaled IMs the MSA ran, back in natural units, one (30 x n_IM) array per version
    xs = [np.exp(set_log_ims(ens)) for ens, _, _ in versions]
    # per-IM KS distances, with the optimiser's own target interpolation
    targets = objective_kwargs(ens_orig, np.ones(len(SELECTION_IMS)))["target_cdfs_list"]
    ds = [ks_distances(np.log(x), targets) for x in xs]

    fig, axs = plt.subplots(4, 6, figsize=(20, 12))
    for ax in axs.flat[len(SELECTION_IMS):]:
        ax.set_visible(False)
    for k, (ax, m) in enumerate(zip(axs.flat, SELECTION_IMS)):
        g = kb_o[m][1:]                       # drop the IM = 0 row
        ax.fill_between(g[:, 0], g[:, 1], g[:, 3], color="0.88", lw=0, label="95 % KS band")
        ax.plot(g[:, 0], g[:, 2], color="k", lw=1.2, label="GCIM target")
        for x, (_, colour, lbl) in zip(xs, versions):
            ex, ey = _ecdf_xy(x[:, k])
            ax.step(ex, ey, where="post", color=colour, lw=1.3, label=lbl)
        ax.set_xscale("log")
        # narrow panels otherwise label every minor tick (2x10^-1, 3x10^-1, ...) on top of
        # each other; the major decade labels are enough to read the scale
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_ylim(0, 1)
        # x range: wide enough for the band and every record of both sets
        inside = (g[:, 3] > 0.001) & (g[:, 1] < 0.999)
        lo = min(g[inside, 0].min(), *(x[:, k].min() for x in xs))
        hi = max(g[inside, 0].max(), *(x[:, k].max() for x in xs))
        ax.set_xlim(lo / 1.1, hi * 1.1)
        # KS distance of each version, one line each in its own colour, stacked up from the
        # bottom-right corner (last version lowest)
        for j, (d, (_, colour, _)) in enumerate(zip(ds[::-1], versions[::-1])):
            ax.text(0.97, 0.05 + 0.08 * j, f"D {d[k]:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=colour)
        is_lead = m == LEAD_IM
        ax.set_title(LEAD_IM_LABEL + " [s]" if is_lead else m + " [g]", fontsize=10,
                     fontweight="bold" if is_lead else None)
    for ax in axs[:, 0]:
        ax.set_ylabel("CDF")
    axs.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


# =============================================================================
# 4. Step 3: predict each fragility's change without new NLTHA
# =============================================================================

# ---- 4a. Firth-penalised logistic regression --------------------------------
# Why Firth (plan D2). With ~150 runs per row a plain maximum-likelihood logistic fit is fine
# most of the time, but where some combination of predictors happens to split collapses from
# non-collapses perfectly ("separation"), the ML coefficient runs off to infinity. Firth (1993)
# adds the Jeffreys-prior penalty 1/2 ln|I(gamma)| to the log-likelihood. That keeps every
# estimate finite and also removes the O(1/n) small-sample bias of ML (Heinze & Schemper 2002;
# Agresti, Categorical Data Analysis, 3rd ed., Ch. 7, ~pp. 251-256, approx.). It is used for
# every row, so there is one estimator throughout and no data-dependent switch.
#
# The algorithm is Newton-Raphson on the Firth-modified score
#     U*_j = sum_i (y_i - p_i + h_i (1/2 - p_i)) x_ij,
# where h_i are the diagonal elements of the "hat" matrix W^1/2 X (X'WX)^-1 X' W^1/2, with
# step-halving on the penalised log-likelihood so that every accepted step goes uphill.


def _logistic_loglik(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    """Bernoulli log-likelihood, written with logaddexp so it never takes log(0)."""
    eta = X @ coef
    return float(-np.sum(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta)))


def firth_logit(X: np.ndarray, y: np.ndarray, *, firth: bool = True,
                max_iter: int = 100, tol: float = 1e-8) -> dict:
    """Logistic regression by Newton-Raphson, Firth-penalised unless ``firth=False``.

    Parameters
    ----------
    X : (n, q) design matrix, INCLUDING the intercept column.
    y : (n,) 0/1 outcomes.

    Returns
    -------
    dict with ``coef`` (q,), ``cov`` (q, q) = I(coef)^-1 (the usual large-sample covariance;
    for Firth this is the standard choice, Heinze & Schemper 2002), ``converged``, ``n_iter``,
    ``loglik`` (unpenalised) and ``p`` (fitted probabilities).

    ``firth=False`` is plain ML; it is used only beside the Firth fit, to DETECT separation
    (:func:`separation_flags`), never as the estimate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    coef = np.zeros(X.shape[1])

    def penalised(c):
        ll = _logistic_loglik(X, y, c)
        if not firth:
            return ll
        p = expit(X @ c)
        info = X.T @ (X * (p * (1 - p))[:, None])
        sign, logdet = np.linalg.slogdet(info)
        return ll + 0.5 * logdet if sign > 0 else -np.inf

    obj = penalised(coef)
    converged = False
    for it in range(1, max_iter + 1):
        p = expit(X @ coef)
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            break                       # singular information: give up, not converged
        resid = y - p
        if firth:
            # hat diagonal h_i = w_i x_i' I^-1 x_i
            h = w * np.einsum("ij,jk,ik->i", X, info_inv, X)
            resid = resid + h * (0.5 - p)
        step = info_inv @ (X.T @ resid)
        # step-halving: never accept a step that lowers the (penalised) log-likelihood
        t = 1.0
        for _ in range(30):
            new = coef + t * step
            new_obj = penalised(new)
            if new_obj >= obj - 1e-12:
                break
            t *= 0.5
        coef, obj = new, new_obj
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break

    p = expit(X @ coef)
    info = X.T @ (X * (p * (1 - p))[:, None])
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return {"coef": coef, "cov": cov, "converged": converged, "n_iter": it,
            "loglik": _logistic_loglik(X, y, coef), "p": p}


def separation_flags(X: np.ndarray, y: np.ndarray, firth_fit: dict) -> dict:
    """Detect (quasi-)separation by fitting plain ML beside the Firth fit (plan D2).

    A row is flagged if ANY of: ML fails to converge; an ML coefficient exceeds 10 in absolute
    value (predictors are standardised, so 10 means a factor of e^10 per SD -- implausible);
    an ML fitted probability is within 1e-8 of 0 or 1; or ML and Firth disagree by more than 2
    Firth standard errors on some coefficient. The Firth estimate is used either way; the flag
    only says where it is doing real work.
    """
    ml = firth_logit(X, y, firth=False, max_iter=50)
    se = np.sqrt(np.diag(firth_fit["cov"]))
    reasons = []
    if not ml["converged"]:
        reasons.append("ml_not_converged")
    if np.any(np.abs(ml["coef"][1:]) > 10):
        reasons.append("ml_coef_gt_10")
    if np.any((ml["p"] < 1e-8) | (ml["p"] > 1 - 1e-8)):
        reasons.append("ml_p_at_boundary")
    if np.any(np.abs(ml["coef"] - firth_fit["coef"]) > 2 * se):
        reasons.append("ml_firth_gt_2se")
    return {"separation": bool(reasons), "separation_reasons": ";".join(reasons)}


# ---- 4b. predictors (plan D3) -----------------------------------------------
# Within a stripe every record has the same AvgSA, so the 21 spectral IMs vary only through
# spectral SHAPE and are strongly collinear; putting all of them into a model with ~70
# collapses would fit noise. One model is used:
#
#   M-DS   ln IML + ln Ds5-95 + [ln SA(T*) - ln IML]
#
# i.e. duration plus ONE spectral-shape term at the structure's own period.
#
# T* is the selection period nearest the structure's T1 (in log period). The shape term
# ln SA(T*) - ln IML is used rather than ln SA(T*) itself; with ln IML in the model the two
# give identical fits and predictions (a linear reparametrisation, and the Jeffreys penalty is
# invariant under those), but the difference is far less correlated with ln IML, which keeps
# the Newton iterations well conditioned. This is NOT SaRatio: it uses a selection IM straight
# from the pickle, normalised by AvgSA(0-3), and SaRatio's 0.2T1-3T1 band never appears.

# Kept as a tuple + default so the notebook's per-model loops and tables keep their shape;
# with a single entry they simply run once.
LOGIT_MODELS = ("M-DS",)
DEFAULT_MODEL = "M-DS"


def nearest_period(T1: float) -> float:
    """The selection SA period closest to ``T1`` on a log scale."""
    return float(SA_PERIODS[np.argmin(np.abs(np.log(SA_PERIODS) - np.log(T1)))])


def raw_predictors(recs: pd.DataFrame, model: str, T_star: float) -> pd.DataFrame:
    """Unstandardised predictor columns (no intercept) for records in long form.

    ``recs`` holds one row per record with columns ``ln_iml`` and ``ln_<IM>`` for every
    selection IM.
    """
    if model not in LOGIT_MODELS:
        raise ValueError(f"unknown model {model!r}; known {LOGIT_MODELS}")
    return pd.DataFrame({
        "ln_iml": recs["ln_iml"],                                   # intensity of the stripe
        "ln_ds595": recs[f"ln_{LEAD_IM}"],                          # duration
        "shape_Tstar": recs[f"ln_SA({T_star})"] - recs["ln_iml"],   # spectral shape at T*
    }, index=recs.index)


def standardise(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Standardise with the TRAINING mean and SD, add the intercept, return design matrices.

    The same transformation is applied to every frame in ``others`` (the optimised records,
    the original records for prediction). Standardising only changes the scale of the
    coefficients, not the fit, but it makes the |gamma| > 10 separation check meaningful.
    """
    mu = train.mean()
    sd = train.std(ddof=0).replace(0.0, 1.0)

    def design(df):
        Z = ((df - mu) / sd).to_numpy(dtype=float)
        return np.column_stack([np.ones(len(Z)), Z])

    return (design(train),) + tuple(design(o) for o in others)


# ---- 4b'. one row's logistic model, opened up (the worked example in nb 034 section 3) -----
# row_shift (4e) fits each row's model and keeps only what Step 3 needs. The two functions
# below refit ONE row with exactly the same steps (raw_predictors -> standardise with the
# training mean/SD -> firth_logit) but keep everything needed to look inside it: the
# standardisation, the covariance of the coefficients, and the coefficients converted back to
# natural units. The test suite checks that the coefficients equal row_shift's.

def fit_row_logit(train: pd.DataFrame, model: str, T_star: float) -> dict:
    """Refit one row's Firth logistic model and return its parts.

    Returns a dict with
    * ``P``: the unstandardised predictors of the training records, ``y``: their 0/1 outcomes;
    * ``mu``, ``sd``: the training mean and SD used to standardise (as in :func:`standardise`);
    * ``design(df)``: maps a frame of unstandardised predictors to the design matrix (intercept +
      standardised columns), so the model can be evaluated anywhere;
    * ``fit``: :func:`firth_logit`'s output (``coef``, ``cov``, ...), ``sep``: the separation flags;
    * ``names``: coefficient names, intercept first.
    """
    P = raw_predictors(train, model, T_star)
    y = train["y"].to_numpy(dtype=float)
    mu = P.mean()
    sd = P.std(ddof=0).replace(0.0, 1.0)          # identical to standardise()

    def design(df: pd.DataFrame) -> np.ndarray:
        Z = ((df[P.columns] - mu) / sd).to_numpy(dtype=float)
        return np.column_stack([np.ones(len(Z)), Z])

    X = design(P)
    fit = firth_logit(X, y)
    return {"P": P, "y": y, "mu": mu, "sd": sd, "design": design, "fit": fit,
            "sep": separation_flags(X, y, fit), "names": ["intercept"] + list(P.columns),
            "model": model, "T_star": T_star}


def logit_coef_table(lf: dict, level: float = 0.95) -> pd.DataFrame:
    """Coefficient summary of one row's model, in standardised AND natural units.

    Standardised (what the model is fitted in): gamma_k is the change in log-odds of collapse
    per one training SD of predictor k. SE = sqrt(diag Cov), Wald z = gamma / SE, and the Wald
    interval gamma +/- z_{1-alpha/2} SE (large-sample; Agresti, Categorical Data Analysis, 3rd
    ed., Ch. 5, ~pp. 163-175, approx.).

    Natural units: the fitted linear predictor is
        eta = g0 + sum_k g_k (x_k - mu_k) / sd_k = (g0 - sum_k g_k mu_k / sd_k) + sum_k (g_k / sd_k) x_k,
    a LINEAR map b = A gamma of the coefficients, so Cov(b) = A Cov(gamma) A' exactly. b_k is the
    change in log-odds per unit of x_k (per unit of ln Ds5-95, ln IML, or the shape term). For the
    log-scale predictors, ``odds_ratio_x2`` = exp(b_k ln 2) is the factor on the odds of collapse
    when that quantity doubles -- the easiest number to read.
    """
    coef, cov = lf["fit"]["coef"], lf["fit"]["cov"]
    mu, sd = lf["mu"].to_numpy(), lf["sd"].to_numpy()
    k = len(coef)
    A = np.zeros((k, k))
    A[0, 0] = 1.0
    A[0, 1:] = -mu / sd
    A[1:, 1:] = np.diag(1.0 / sd)
    b, cov_b = A @ coef, A @ cov @ A.T
    se, se_b = np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov_b))
    zq = stats.norm.ppf(0.5 + level / 2)
    out = pd.DataFrame({
        "coef_std": coef, "se_std": se, "z": coef / se,
        "p_value": 2 * stats.norm.sf(np.abs(coef / se)),
        f"ci{int(level * 100)}_lo_std": coef - zq * se, f"ci{int(level * 100)}_hi_std": coef + zq * se,
        "coef_natural": b, "se_natural": se_b,
        "odds_ratio_x2": np.exp(b * np.log(2.0)),
    }, index=lf["names"])
    # a doubling only means something for the log-scale predictors, not the intercept
    out.loc["intercept", "odds_ratio_x2"] = np.nan
    return out


def plot_row_logit(lf: dict, train: pd.DataFrame, opt_recs: pd.DataFrame, imls: np.ndarray,
                   title: str, n_draws: int = 1000, n_curve_draws: int = 200,
                   rng: np.random.Generator | None = None,
                   extra_recs: Sequence[tuple[pd.DataFrame, str, str]] = ()):
    """Worked-example figure of one row's logistic model.

    Left: the stripe-level view, i.e. the fragility the MSA estimates. ONE logistic model serves
    every stripe of the row (ln IML is one of its predictors); at each stripe it is evaluated on
    that stripe's 30 records and averaged, p_hat = (1/30) sum_k P(C | record k) -- exactly what
    Step 3 refits. Drawn:
    * crosses: observed collapse fractions, with the MSA lognormal fit to them (black);
    * circles / squares: predicted fractions for the ORIGINAL (grey) / KS-OPTIMISED (red)
      records (the circles carry their 90 % range over coefficient draws as error bars);
      ``extra_recs`` adds further re-selected record sets as ``(records, colour, label)``,
      e.g. the mu/sigma-optimised sets, drawn as diamonds and dashed curves the same way;
    * grey / red dashed curves: the lognormal fragility fitted to those predicted fractions with
      the same expected-count MLE as Step 3 (:func:`lognorm_mle_fit_expected`). Their medians
      and dispersions are the theta_pred / beta_pred of Step 3, so the gap between the two curves
      IS Delta a_i (median) and Delta ln beta_i (slope). The shaded band is the 90 % range of
      the original-records curve over ``n_curve_draws`` coefficient draws, each refitted.

    Right, one panel per stripe: the record-level view. Each record is a point at its Ds5-95,
    at y = 1 (collapsed) or 0 (did not), jittered vertically so ties are visible. The curve is
    P(C) against Ds5-95 at that stripe's IML, with the shape term held at the stripe's median,
    and the shaded band is the 90 % range over coefficient draws gamma ~ N(gamma_hat, Cov) --
    the same draws Step 3 uses to put an uncertainty on Delta a_i.

    The band shows the uncertainty in the model's PARAMETERS only, not the scatter of the 0/1
    outcomes around it (a single record either collapses or not).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    rng = rng if rng is not None else np.random.default_rng(0)
    fit, design = lf["fit"], lf["design"]
    G = rng.multivariate_normal(fit["coef"], fit["cov"], size=n_draws)     # (n_draws, q)

    def band(X):
        """Point prediction and 5-95 % range over draws, for design matrix X."""
        P = expit(X @ G.T)                                                 # (n, n_draws)
        return expit(X @ fit["coef"]), np.quantile(P, 0.05, axis=1), np.quantile(P, 0.95, axis=1)

    s_tr = train["set_iml"].to_numpy()
    s_op = opt_recs["set_iml"].to_numpy()
    keys = [s_tr[np.argmin(np.abs(s_tr - v))] for v in imls]   # the stripes' set IMLs, in order
    n_s = len(keys)
    ncols = int(np.ceil(n_s / 2))

    fig = plt.figure(figsize=(5.5 + 3.2 * ncols, 7.5))
    gs = GridSpec(2, ncols + 2, figure=fig)
    ax_f = fig.add_subplot(gs[:, :2])

    # ---- left: stripe-level fractions
    obs = [lf["y"][s_tr == k].mean() for k in keys]
    X_tr = design(lf["P"])
    # the optimised records, through the SAME predictors and standardisation as the training
    X_op = design(raw_predictors(opt_recs, lf["model"], lf["T_star"]))

    def frac(X, mask):
        """Predicted collapse fraction of one stripe (mean P over its records): point + 90 %."""
        p_hat = expit(X[mask] @ fit["coef"]).mean()
        p_draw = expit(X[mask] @ G.T).mean(axis=0)
        return p_hat, *np.quantile(p_draw, [0.05, 0.95])

    f_tr = np.array([frac(X_tr, s_tr == k) for k in keys])
    f_op = np.array([frac(X_op, s_op == k) for k in keys])
    # any further record sets, through the same model and standardisation
    f_ex = []
    for recs, colour, lbl in extra_recs:
        X_ex = design(raw_predictors(recs, lf["model"], lf["T_star"]))
        s_ex = recs["set_iml"].to_numpy()
        f_ex.append((np.array([frac(X_ex, s_ex == k)[0] for k in keys]), colour, lbl))

    # lognormal fragilities, fitted exactly as Step 3 / the MSA do (expected-count MLE)
    th_obs, be_obs = lognorm_mle_fit_expected(imls, N_RECORDS * np.asarray(obs), N_RECORDS)
    th_or, be_or = lognorm_mle_fit_expected(imls, N_RECORDS * f_tr[:, 0], N_RECORDS)
    th_op, be_op = lognorm_mle_fit_expected(imls, N_RECORDS * f_op[:, 0], N_RECORDS)
    x = np.linspace(0.5 * imls.min(), 1.6 * imls.max(), 200)
    cdf = lambda th, be: stats.norm.cdf(np.log(x / th) / be)

    # band on the original-records curve: refit the lognormal for each of n_curve_draws draws
    curves = []
    for g in G[:n_curve_draws]:
        p_g = [expit(X_tr[s_tr == k] @ g).mean() for k in keys]
        curves.append(cdf(*lognorm_mle_fit_expected(imls, N_RECORDS * np.asarray(p_g), N_RECORDS)))
    lo_c, hi_c = np.quantile(np.array(curves), [0.05, 0.95], axis=0)

    ax_f.fill_between(x, lo_c, hi_c, color="0.45", alpha=0.18, lw=0,
                      label="90 % band, original-records curve")
    ax_f.plot(x, cdf(th_obs, be_obs), color="k", lw=1.2,
              label=f"MSA fit to observed: θ {th_obs:.3f} g, β {be_obs:.3f}")
    ax_f.plot(x, cdf(th_or, be_or), color="0.45", lw=1.8,
              label=f"fit to predicted, original: θ {th_or:.3f} g, β {be_or:.3f}")
    ax_f.plot(x, cdf(th_op, be_op), color="C3", lw=1.8, ls="--",
              label=f"fit to predicted, Opt. KS: θ {th_op:.3f} g, β {be_op:.3f}")
    ax_f.errorbar(imls, f_tr[:, 0], yerr=[f_tr[:, 0] - f_tr[:, 1], f_tr[:, 2] - f_tr[:, 0]],
                  fmt="o", color="0.45", ms=5, capsize=3, label="predicted fraction, original records")
    ax_f.plot(imls, f_op[:, 0], "s", color="C3", ms=5, mfc="none", mew=1.5,
              label="predicted fraction, Opt. KS records")
    # further record sets: their fitted fragility (dashed) and predicted fractions (diamonds);
    # the shifts against the original-records curve go into the panel title
    shifts = [f"Opt. KS: Δa = {np.log(th_op / th_or):+.4f}, Δln β = {np.log(be_op / be_or):+.4f}"]
    for f_e, colour, lbl in f_ex:
        th_e, be_e = lognorm_mle_fit_expected(imls, N_RECORDS * f_e, N_RECORDS)
        ax_f.plot(x, cdf(th_e, be_e), color=colour, lw=1.8, ls="--",
                  label=f"fit to predicted, {lbl}: θ {th_e:.3f} g, β {be_e:.3f}")
        ax_f.plot(imls, f_e, "D", color=colour, ms=5, mfc="none", mew=1.5,
                  label=f"predicted fraction, {lbl} records")
        shifts.append(f"{lbl}: Δa = {np.log(th_e / th_or):+.4f}, Δln β = {np.log(be_e / be_or):+.4f}")
    ax_f.plot(imls, obs, "kx", ms=8, mew=2, label="observed (collapses / 30)")
    ax_f.set_xlabel("stripe IML, AvgSA(0-3) [g]")
    ax_f.set_ylabel("collapse probability")
    ax_f.set_xlim(x.min(), x.max())
    ax_f.set_ylim(-0.02, 1.02)
    ax_f.legend(fontsize=7, loc="upper left")
    ax_f.set_title("stripe level: the fragility\n" + "\n".join(shifts), fontsize=9)

    # ---- right: record level, one panel per stripe
    jit = rng.uniform(-0.04, 0.04, size=len(lf["y"]))
    ds_all = np.exp(lf["P"]["ln_ds595"].to_numpy())
    for j, k in enumerate(keys):
        ax = fig.add_subplot(gs[j // ncols, 2 + j % ncols])
        m = s_tr == k
        Pm = lf["P"][m]
        # curve along Ds5-95 at this stripe's IML, shape term held at the stripe median
        grid = np.geomspace(ds_all.min() / 1.2, ds_all.max() * 1.2, 120)
        Pg = pd.DataFrame({"ln_iml": np.log(k), "ln_ds595": np.log(grid),
                           "shape_Tstar": Pm["shape_Tstar"].median()})
        p0, lo, hi = band(design(Pg))
        # neutral grey: this curve is the row's model, not any one record set
        ax.fill_between(grid, lo, hi, color="0.45", alpha=0.2, lw=0)
        ax.plot(grid, p0, color="0.3")
        yk = lf["y"][m]
        ax.scatter(ds_all[m], yk + jit[m], s=14, c=np.where(yk == 1, "C3", "0.4"), zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"IML {k:.3f} g: {int(yk.sum())}/{int(m.sum())} collapsed", fontsize=9)
        if j // ncols == 1 or j + ncols >= n_s:
            ax.set_xlabel("Ds5-95 [s]")
        if j % ncols == 0:
            ax.set_ylabel("P(collapse)")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


# ---- 4c. record tables ------------------------------------------------------

def records_long(sets: dict, keys: Sequence[tuple[int, float]], version: str) -> pd.DataFrame:
    """One row per (set, slot) with ``ln_iml`` and ``ln_<IM>`` for the selection IMs.

    ``slot`` is the record's row in the pickle, which equals column ``k`` of the collapse
    flags (verified by :func:`map_structure_stripes`).
    """
    frames = []
    for key in keys:
        x = set_log_ims(sets[key])
        df = pd.DataFrame(x, columns=[f"ln_{im}" for im in SELECTION_IMS])
        df.insert(0, "slot", np.arange(N_RECORDS))
        df.insert(0, "ln_iml", np.log(key[1]))
        df.insert(0, "set_iml", key[1])
        df.insert(0, "set_site", key[0])
        df.insert(0, "version", version)
        df["filename"] = sets[key]["recs"]["metadata"]["filename"].to_numpy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- 4d. the fragility fit on expected counts --------------------------------

def lognorm_mle_fit_expected(intensity_levels, n_collapses, n_records) -> tuple[float, float]:
    """Lognormal MSA fragility fit that accepts NON-INTEGER (expected) collapse counts.

    Why this exists (plan 0.5). ``standes.fitting.lognorm_mle_fit`` maximises
    ``sum_j binom.logpmf(z_j, n_j, p_j)``, and ``scipy.stats.binom.logpmf`` returns -inf for
    non-integer z -- so the predicted counts n * p_hat of Step 3 would break it silently.

    Here the likelihood is the binomial KERNEL
        l(theta, beta) = sum_j z_j ln p_j + (n_j - z_j) ln(1 - p_j),
        p_j = Phi((ln x_j - ln theta) / beta).
    It differs from the full binomial log-likelihood only by ln C(n_j, z_j), which does not
    depend on (theta, beta), so for integer counts the maximiser is IDENTICAL (Baker 2015,
    Earthquake Spectra 31(1), Eq. 11). Same parametrisation, start point, clipping and
    Nelder-Mead settings as standes, so the two agree to optimiser precision on integer data
    (asserted in the tests).
    """
    x = np.asarray(intensity_levels, dtype=float)
    z = np.asarray(n_collapses, dtype=float)
    n = np.broadcast_to(np.asarray(n_records, dtype=float), z.shape)
    fractions = z / n

    def neg_log_likelihood(params) -> float:
        ln_median, dispersion = params
        if dispersion <= 0:
            return np.inf
        p = stats.norm.cdf((np.log(x) - ln_median) / dispersion)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(z * np.log(p) + (n - z) * np.log1p(-p))

    ln_median_0 = np.log(x[np.argmin(np.abs(fractions - 0.5))])
    res = minimize(neg_log_likelihood, x0=[ln_median_0, 0.4], method="Nelder-Mead")
    ln_median, dispersion = res.x
    return float(np.exp(ln_median)), float(abs(dispersion))


def predicted_fraction(X: np.ndarray, coef: np.ndarray) -> float:
    """Predicted collapse fraction of one set: the mean of P(C | record) over its records.

    This is the MSA's own estimator (collapses / records) with each 0/1 outcome replaced by
    its model probability. Independent of record order by construction.
    """
    return float(np.mean(expit(np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float))))


# ---- 4e. one row, end to end ------------------------------------------------

def row_shift(train: pd.DataFrame, orig_recs: pd.DataFrame, opt_recs: pd.DataFrame,
              imls: np.ndarray, model: str, T_star: float, *,
              n_draws: int = 0, rng: np.random.Generator | None = None) -> dict:
    """Step 3 for one row (structure at a site) and one logistic model.

    Parameters
    ----------
    train : original records at the row's fitted stripes, with the 0/1 outcome ``y``.
    orig_recs, opt_recs : the original and optimised records at the same stripes (``set_iml``
        identifies the stripe). ``orig_recs`` is ``train`` without ``y``; it is passed
        separately so that prediction and training are visibly the same records.
    imls : the row's fitted stripe IMLs (the fragility's ``efc``), sorted.
    n_draws : number of coefficient draws gamma ~ N(gamma_hat, Cov) used to put an
        uncertainty on Delta a_i. 0 skips it.

    Returns a dict with the point results (``d_ln_theta``, ``d_ln_beta``, the two refits, the
    calibration refit), the per-stripe predictions, extrapolation counts, separation flags
    and, if requested, the ``d_ln_theta`` draws.
    """
    P_tr = raw_predictors(train, model, T_star)
    P_or = raw_predictors(orig_recs, model, T_star)
    P_op = raw_predictors(opt_recs, model, T_star)
    X_tr, X_or, X_op = standardise(P_tr, P_or, P_op)
    y = train["y"].to_numpy(dtype=float)

    fit = firth_logit(X_tr, y)
    sep = separation_flags(X_tr, y, fit)

    # ---- extrapolation: optimised records outside what the model was trained on
    lo, hi = P_tr.min(), P_tr.max()
    outside = ((P_op < lo) | (P_op > hi)).any(axis=1).to_numpy()
    q = P_tr.shape[1]
    cov_tr = np.cov(P_tr.to_numpy(dtype=float), rowvar=False)
    d = P_op.to_numpy(dtype=float) - P_tr.mean().to_numpy()
    maha2 = np.einsum("ij,jk,ik->i", d, np.linalg.pinv(np.atleast_2d(cov_tr)), d)
    maha_out = maha2 > stats.chi2.ppf(0.99, q)

    def fractions(coef):
        p_o = np.array([predicted_fraction(X_or[orig_recs["set_iml"].to_numpy() == s], coef)
                        for s in stripe_keys])
        p_n = np.array([predicted_fraction(X_op[opt_recs["set_iml"].to_numpy() == s], coef)
                        for s in stripe_keys])
        return p_o, p_n

    def refit(p_o, p_n):
        th_o, be_o = lognorm_mle_fit_expected(imls, N_RECORDS * p_o, N_RECORDS)
        th_n, be_n = lognorm_mle_fit_expected(imls, N_RECORDS * p_n, N_RECORDS)
        return th_o, be_o, th_n, be_n

    # the row's stripes, as the set IMLs the records carry, in the order of `imls`
    stripe_keys = [orig_recs["set_iml"].to_numpy()[np.argmin(np.abs(orig_recs["set_iml"].to_numpy() - v))]
                   for v in imls]
    p_o, p_n = fractions(fit["coef"])
    th_o, be_o, th_n, be_n = refit(p_o, p_n)

    stripes = pd.DataFrame({
        "stripe_iml": imls, "set_iml": stripe_keys,
        "p_obs": [train.loc[train["set_iml"] == s, "y"].mean() for s in stripe_keys],
        "p_hat_orig": p_o, "p_hat_opt": p_n,
        "n_opt_outside_range": [int(outside[opt_recs["set_iml"].to_numpy() == s].sum())
                                for s in stripe_keys],
        "n_opt_outside_maha99": [int(maha_out[opt_recs["set_iml"].to_numpy() == s].sum())
                                 for s in stripe_keys],
    })

    out = {
        "model": model, "T_star": T_star, "n_runs": len(y), "n_collapses": int(y.sum()),
        "n_params": X_tr.shape[1], "converged": fit["converged"],
        "theta_pred_orig": th_o, "beta_pred_orig": be_o,
        "theta_pred_opt": th_n, "beta_pred_opt": be_n,
        "d_ln_theta": np.log(th_n) - np.log(th_o),
        "d_ln_beta": np.log(be_n) - np.log(be_o),
        "n_opt_outside_range": int(outside.sum()),
        "n_opt_outside_maha99": int(maha_out.sum()),
        "extrapolated": bool(outside.sum() > 3),
        **sep,
        "coef": fit["coef"], "coef_names": ["intercept"] + list(P_tr.columns),
        "stripes": stripes,
    }

    if n_draws:
        rng = rng if rng is not None else np.random.default_rng(0)
        G = rng.multivariate_normal(fit["coef"], fit["cov"], size=n_draws)
        dd = np.empty(n_draws)
        for r, g in enumerate(G):
            a, b, c, e = refit(*fractions(g))
            dd[r] = np.log(c) - np.log(a)
        out["d_ln_theta_draws"] = dd
    return out


# =============================================================================
# 5. Step 4: shift the SS arm and refit A0c
# =============================================================================

A0C_HB = "A0c_hb"
A0C_BC_HB = "A0c_bc_hb"   # the beta check: IDA bias correction + hybrid bootstrap
A3AB = "A3ab"             # the theta regression check: A0c + the four GCIM-vs-FEMA covariates
# A3ab with each covariate replaced by its `_corrected` version: the records the MSA-SS actually
# ran minus FEMA (nb 072 section 8.10.6), not the GCIM target minus FEMA. Unlike A3ab's, these
# covariates CHANGE with the optimised sets, so the optimised fit needs them recomputed from the
# optimised records (`record_covariates`) and swapped into the container (`with_covariates`).
A3AE = "A3ae"
TAU_NAMES = {"tau_d": "tau_d2", "tau_s": "tau_s2", "tau_u": "tau_u2"}
HEADLINE = ("m", *TAU_NAMES)  # the quantities every summary carries, in table/plot order

# Plot/table labels for A3ab's four covariates. Each is "site minus FEMA set" of a GCIM
# moment, unif-weighted over the row's MSA stripes (nb 072 section 8.0). They describe the
# GCIM TARGET, not the records that were run, so they are identical for the original and the
# optimised record sets -- only the outcome y moves when the SS arm is shifted.
COVARIATE_LABELS = {
    "d_ln_saratio_mean_unif": r"$\Delta E[\ln \mathrm{SaRatio}]$",
    "d_ln_ds595_mean_unif": r"$\Delta E[\ln D_{s5-95}]$",
    "d_ln_saratio_sigma_unif": r"$\Delta SD[\ln \mathrm{SaRatio}]$",
    "d_ln_ds595_sigma_unif": r"$\Delta SD[\ln D_{s5-95}]$",
}
DECISION_THRESHOLD_SE = 0.5   # D1 (user decision 2026-10-14)
ENGINEERING_THRESHOLD = 0.05  # reported beside it: |exp(dm) - 1| > 5 %


def shift_ss_arm(data: fm.FragilityData, delta: pd.Series) -> fm.FragilityData:
    """A copy of ``data`` with every SS-arm log estimate moved by ``delta`` (per row).

    What moves: the point estimates ``a``, the bias-corrected ``a``, and EVERY replicate column
    of ``a`` by the same per-row constant. Shifting the whole replicate cloud by a constant
    leaves its covariance -- ``V_aa``, hence ``V_known`` -- exactly as it was, which is what the
    task asks for (keep V_known fixed). It also leaves the measured bias, and so
    ``plugin_offset``, consistent (cloud mean - hat - bias is unchanged).

    What does not move: ``b``, ``c`` and the other IDA arms, the rows/factor table, the design
    groups. ``FragilityData`` is frozen, so the copy is made with ``dataclasses.replace`` and
    the original container is untouched (its cached covariances stay valid).
    """
    a_hat = data.estimates["a"]
    d = delta.reindex(a_hat.index)
    if d.isna().any():
        raise KeyError(f"delta is missing rows {list(d.index[d.isna()])}")
    estimates = dict(data.estimates) | {"a": a_hat + d}
    estimates_bc = dict(data.estimates_bc)
    if estimates_bc.get("a") is not None:
        estimates_bc["a"] = estimates_bc["a"] + d
    reps = dict(data.reps) | {"a": data.reps["a"].add(d, axis=0)}
    return dataclasses.replace(data, estimates=estimates, estimates_bc=estimates_bc, reps=reps)


# ---- 5b. A3ae's record-side covariates for any record set ---------------------
# A3ae's four covariates are "records actually run minus FEMA", averaged with EQUAL weight over
# the stripes each row's MSA-SS fragility was fitted to (nb 072 section 8.10, the "_unif_corrected"
# columns). When the record sets change, so do these covariates. The functions below recompute
# them for any record-set dict (original or optimised) by nb 072's own route:
#
#   * ln Ds5-95: straight from the set's ``ims_scaled["RSD595"]`` (duration does not scale);
#   * ln SaRatio(T1): from each record's FINE-grid spectrum (acc_spectra.pickle, 0.01 s steps),
#     looked up by record_id, with the 0.01 s band 0.2 T1 : 0.01 : 3 T1 (gsr.saratio_periods).
#     The selection pickles only carry 20 log-spaced ordinates, far too coarse for the band.
#
# `record_ids` and `ln_saratio` are nb 072's functions (8.10.1 and 8.10.5), moved here unchanged
# so that nb 034 runs the SAME code. The notebook proves it: on the original sets the result
# must reproduce nb 072's four columns to 1e-10.

# column names, in A3ae's (registry) order; each maps to (property, statistic)
RECORD_COVARIATES = {
    "d_ln_saratio_mean_unif_corrected": ("saratio", "mean"),
    "d_ln_ds595_mean_unif_corrected": ("ds595", "mean"),
    "d_ln_saratio_sigma_unif_corrected": ("saratio", "sigma"),
    "d_ln_ds595_sigma_unif_corrected": ("ds595", "sigma"),
}


def record_ids(recs) -> list[str]:
    """record_id (JSON file name without .json) of every record in an ensemble's `recs`, in row order.

    nb 072 section 8.10.1, unchanged. ESM: ``{event}_{station}_{location}__{component}``;
    NGA-Sub: ``{RSN}__{component}``, with the RSN in ``metadata.index``. This is the index of
    the fine-grid spectra table.
    """
    m = recs["metadata"]
    ids = []
    for db, idx, ev, st, loc, comp in zip(m["database"], m["index"], m["event_id"],
                                          m["station_code"], m["location_code"], m["component"]):
        ids.append(f"{int(idx)}__{comp}" if "NGA" in str(db).upper() else f"{ev}_{st}_{loc}__{comp}")
    return ids


def ln_saratio(ln_sa: np.ndarray, T1: float, ln_grid: np.ndarray) -> np.ndarray:
    """ln SaRatio(T1) of every record in a stripe, shape (n_records,). nb 072 section 8.10.5, unchanged.

        SaRatio(T1) = SA(T1) / (prod_j SA(T_j))^(1/n),   T_j = 0.2 T1 : 0.01 : 3 T1

    ln_sa   : (n_records, n_grid) ln PSA of the stripe's records on the period grid `ln_grid`.
    T1      : the structure's fundamental period [s].
    ln_grid : ln of the grid periods (the fine 0.01 s grid of acc_spectra.pickle).
    """
    band = gsr.saratio_periods(T1)                    # the denominator periods T_j (one definition, in gsr)
    # np.interp would silently clamp a period outside the grid to the end ordinate - refuse instead.
    assert ln_grid[0] <= np.log(band[0]) and np.log(band[-1]) <= ln_grid[-1], "band outside the spectrum grid"

    # Numerator: SA at T1, one value per record. Interpolated linearly in (ln T, ln SA), then back to SA.
    sa_t1 = np.exp(np.array([np.interp(np.log(T1), ln_grid, r) for r in ln_sa]))          # (n_records,)
    # Band: SA at every T_j, one row per record.
    sa_band = np.exp(np.array([np.interp(np.log(band), ln_grid, r) for r in ln_sa]))      # (n_records, n)
    # Denominator: geometric mean over the band, i.e. (prod_j SA(T_j))^(1/n). gmean works through the mean
    # of the logs, so it does not underflow the way the raw product of a few hundred SA < 1 g would.
    avg_sa = stats.gmean(sa_band, axis=1)                                                  # (n_records,)
    return np.log(sa_t1 / avg_sa)


def stripe_record_stats(ens: dict, T1: float, spectra: pd.DataFrame) -> dict:
    """Mean and SD (ddof = 1) of ln Ds5-95 and of ln SaRatio(T1) over one set's 30 records.

    ``spectra`` is the fine-grid PSA table (index record_id, columns periods [s]). A record with
    no spectrum raises a KeyError naming it: the covariates are never computed from a partial set.
    """
    ln_ds = np.log(ens["recs"]["ims_scaled"][LEAD_IM].to_numpy(dtype=float))   # Ds5-95 in seconds
    ids = record_ids(ens["recs"])
    missing = [r for r in ids if r not in spectra.index]
    if missing:
        raise KeyError(f"{len(missing)} records have no fine-grid spectrum, e.g. {missing[:3]}; "
                       "run compute_record_spectra.py over their JSONs")
    ln_grid = np.log(spectra.columns.to_numpy(dtype=float))
    x = ln_saratio(np.log(spectra.loc[ids].to_numpy(dtype=float)), T1, ln_grid)
    return {"ds595_mean": ln_ds.mean(), "ds595_sigma": ln_ds.std(ddof=1),
            "saratio_mean": x.mean(), "saratio_sigma": x.std(ddof=1)}


def record_covariates(sets: dict, smap: pd.DataFrame, fragilities: dict, rows: pd.DataFrame,
                      spectra: pd.DataFrame) -> pd.DataFrame:
    """A3ae's four ``d_ln_*_unif_corrected`` covariates for every row, from the record sets ``sets``.

    For row i = (site, n_storeys):

        x_i = (1 / n_i) sum_k S(records at stripe k)  -  S_FEMA,g[i]

    over the row's n_i fitted stripes (the fragility's ``efc`` IMLs, mapped to their record set
    through ``smap`` exactly as nb 034's ``row_inputs`` does), for S = mean / SD of ln Ds5-95 and of
    ln SaRatio(T1_i). Equal ("unif") weights, as nb 072's ``_unif`` columns.

    sets        : ``load_record_sets`` dict (original or optimised); the same keys in both, so
                  the same ``smap`` serves both.
    smap        : ``map_structure_stripes`` output (site, n_storeys, stripe_iml -> set key).
    fragilities : ``mra.load_site_msa_fragilities`` output; ``[n][site]["efc"][0]`` are the IMLs.
    rows        : indexed (site, n_storeys), with ``T1`` and the FEMA statistics
                  ``fema_ln_{saratio, ds595}_{mean, sigma}`` (nb 072's covariate components).
    spectra     : the fine-grid PSA table (acc_spectra.pickle).

    Returns a frame indexed like ``rows`` with the four columns of :data:`RECORD_COVARIATES`.
    """
    sm = smap.set_index(["site", "n_storeys", "stripe_iml"])
    cache: dict[tuple, dict] = {}     # (set key, T1) -> stats: a set shared by 3s and 5s is read once per T1
    out = {}
    for (site, n), r in rows.iterrows():
        imls = np.sort([round(float(v), 6) for v in fragilities[n][site]["efc"][0]])
        per_stripe = []
        for v in imls:
            key = tuple(sm.loc[(site, n, v), ["set_site", "set_iml"]])
            ck = (key, float(r["T1"]))
            if ck not in cache:
                cache[ck] = stripe_record_stats(sets[key], float(r["T1"]), spectra)
            per_stripe.append(cache[ck])
        st = pd.DataFrame(per_stripe).mean()          # equal weight 1 / n_i over the fitted stripes
        out[(site, n)] = {col: st[f"{q}_{s}"] - r[f"fema_ln_{q}_{s}"]
                          for col, (q, s) in RECORD_COVARIATES.items()}
    res = pd.DataFrame.from_dict(out, orient="index")
    res.index = pd.MultiIndex.from_tuples(res.index, names=rows.index.names)
    return res


def with_covariates(data: fm.FragilityData, cov: pd.DataFrame) -> fm.FragilityData:
    """A copy of ``data`` whose factor table has the columns of ``cov`` replaced (per row).

    Only covariate columns of ``data.rows`` change. The estimates, the replicate clouds (hence
    V_known), the groups and every other column are carried over untouched. ``FragilityData`` is
    frozen, so the copy is made with ``dataclasses.replace``, as in :func:`shift_ss_arm`; the new
    container recomputes its cached blocks from the same replicates, so they come out identical.
    """
    missing = [c for c in cov.columns if c not in data.rows.columns]
    if missing:
        raise KeyError(f"not columns of the factor table: {missing}")
    new = cov.reindex(data.rows.index)
    if new.isna().any().any():
        raise KeyError("cov does not cover every row of the container")
    rows = data.rows.copy()
    rows[list(cov.columns)] = new.to_numpy()
    return dataclasses.replace(data, rows=rows)


class FastA0c:
    """Point REML fits of A0c for many shifted outcome vectors, sharing one V and X.

    A shift by a per-row constant changes only the outcome y (the contrast's point estimate),
    never V_known, X or the G matrices. So the expensive pieces are built once, here, from the
    ORIGINAL container, and each refit is a single ``mra.fit_reml`` call with the Cholesky
    solver and one start -- the same settings ``fm.bootstrap_fits`` uses for every bootstrap
    replicate. (The registry's full-sample fit uses the slower explicit solver with four
    starts; the notebook checks that the two agree on the shifted data.)
    """

    def __init__(self, data: fm.FragilityData, model_id: str = A0C_HB):
        self.spec = fm.MODEL_SPECS[model_id]
        self.con = fm.make_contrast(data, self.spec.contrast, corrected=self.spec.corrected)
        self.X, self.x_names, _ = fm.build_X(self.spec, self.con)
        self.V = self.con.V
        self.Gs, self.names = dsim.build_component_gs(fm.build_components(self.spec, self.con))
        self.y_hat = self.con.y_hat

    def fit(self, delta: pd.Series | None = None) -> dict:
        """``{m, tau_d, tau_s, tau_u}`` (taus as SDs) for y_hat + delta (delta per row)."""
        y = self.y_hat if delta is None else self.y_hat + delta.reindex(self.y_hat.index)
        raw = mra.fit_reml(y.to_numpy(dtype=float), self.X, self.V, self.Gs, names=self.names,
                           method="cholesky", n_starts=1)
        t2 = dict(zip(raw["names"], raw["tau2"]))
        return {"m": float(np.atleast_1d(raw["beta"])[0]),
                **{k: float(np.sqrt(t2[v])) for k, v in TAU_NAMES.items()}}


def a0c_summary(fit_hb) -> dict:
    """The four A0c headline quantities with their SEs and the qualitative markers (D1).

    Everything comes from ONE fit, ``A0c_hb`` (the hybrid outer bootstrap). The point
    estimates are the ordinary full-sample REML fit, identical to ``A0c``'s; only the
    replicates differ.

    * m and its SE: ``fit_hb.m`` and ``fit_hb.se_boot``. Because the hybrid replicates redraw
      the design, site and row effects as well as the records, this is the SUPERPOPULATION
      standard error (other sites and designs could have been drawn). It is wider than the
      record-only ("these 120 rows") SE of ``A0c``, so a given shift in m reads as fewer SEs.
    * each tau (as an SD) and its SE: the SD of sqrt(tau^2) over the hybrid replicates, the
      only honest spread for a variance component (roadmap A1c, "Inference").
    * the CI of m (hybrid percentile interval, superpopulation), the CI of tau_d - tau_s
      (formed per replicate as the roadmap prescribes) and the fraction of replicates at zero
      per component.
    """
    reps = fit_hb.replicates
    sq = {k: np.sqrt(reps[v].to_numpy(dtype=float)) for k, v in TAU_NAMES.items()}
    out = {"m": fit_hb.m, "se_m": fit_hb.se_boot, "ci_m": fit_hb.ci()}
    for k, v in TAU_NAMES.items():
        out[k] = float(np.sqrt(fit_hb.tau2[v]))
        out[f"se_{k}"] = float(np.std(sq[k], ddof=1))
        out[f"frac_zero_{k}"] = float(np.mean(reps[v].to_numpy() == 0.0))
    diff = sq["tau_d"] - sq["tau_s"]
    out["ci_tau_d_minus_tau_s"] = tuple(np.quantile(diff, [0.025, 0.975]))
    out["quantities"], out["slopes"] = list(HEADLINE), []
    return out


def regression_summary(fit) -> dict:
    """``a0c_summary`` plus every covariate slope, for a meta-regression fit such as A3ab.

    The headline part (m, the taus, their SEs and the qualitative markers) is exactly
    ``a0c_summary``'s, so the regression is read on the same terms as A0c. Note what m means
    here: covariates are centred in ``fm.build_X``, so the intercept is the fitted contrast at
    the covariate MEANS, not at zero.

    Each slope is added under its covariate's own name, with

    * the estimate ``fit.beta[k]``, "per 1 SD of the covariate" (``build_X`` scales them, so
      the four slopes are comparable in size);
    * its SE: the SD of that slope over the fit's outer-bootstrap replicates
      (``fit.se_boot_beta``), the same kind of SE as every other quantity here. Its
      population meaning follows the fit's scheme: record-only for a plain spec such as
      ``A3ab`` (sites and designs held fixed, so narrower), superpopulation for an ``_hb`` one;
    * its 95 % percentile CI over the same replicates (for the "CI includes 0" check).

    ``quantities`` lists the headline names then the slope names: the order of
    ``shift_table``'s rows and of the before/after figure's panels.
    """
    out = a0c_summary(fit)
    se = fit.se_boot_beta
    for k, name in enumerate(fit.x_names):
        if name == "intercept":          # already reported as m
            continue
        out[name] = float(fit.beta[k])
        out[f"se_{name}"] = float(se[k])
        out[f"ci_{name}"] = tuple(fit.replicates[fm.beta_column(name)].quantile([0.025, 0.975]))
        out["slopes"].append(name)
    out["quantities"] = list(HEADLINE) + out["slopes"]
    return out


def shift_table(orig: dict, shifted: dict, threshold: float = DECISION_THRESHOLD_SE
                ) -> pd.DataFrame:
    """Step 4 headline table: one row per quantity of the fit.

    The quantities are the summary's own ``quantities`` list: m and the three taus for A0c,
    plus each covariate slope for a regression (``regression_summary``). Delta / SE uses the
    ORIGINAL fit's SE (the shift compared with how well we currently know the number).
    ``exceeds_threshold`` marks |Delta / SE| above the agreed 0.5. For m the ratio change,
    exp(Delta m) - 1, is reported beside it.
    """
    rows = []
    for q in orig.get("quantities", HEADLINE):
        d = shifted[q] - orig[q]
        se = orig[f"se_{q}"]
        rows.append({"quantity": q, "original": orig[q], "shifted": shifted[q],
                     "delta": d, "se_original": se, "delta_over_se": d / se,
                     "exceeds_threshold": abs(d / se) > threshold,
                     "ratio_change": np.exp(d) - 1 if q == "m" else np.nan})
    return pd.DataFrame(rows)


def qualitative_flags(orig: dict, shifted: dict) -> pd.DataFrame:
    """The D1 qualitative checks, original vs shifted; ``changed`` marks a check that flipped.

    * sign of m;
    * whether the 95 % CI of m includes 0;
    * which of tau_d and tau_s is larger;
    * whether the CI of tau_d - tau_s includes 0;
    * for each tau, whether it sits on the zero boundary (point estimate exactly 0);
    * for each covariate slope (regression summaries only): its sign, and whether its 95 %
      CI includes 0 -- i.e. whether the covariate's direction or its distinguishability from
      zero changes.
    """
    def inc0(ci):
        return bool(ci[0] <= 0 <= ci[1])

    checks = {
        "sign of m": (np.sign(orig["m"]), np.sign(shifted["m"])),
        "CI of m includes 0": (inc0(orig["ci_m"]), inc0(shifted["ci_m"])),
        "tau_d > tau_s": (orig["tau_d"] > orig["tau_s"], shifted["tau_d"] > shifted["tau_s"]),
        "CI of tau_d - tau_s includes 0": (inc0(orig["ci_tau_d_minus_tau_s"]),
                                           inc0(shifted["ci_tau_d_minus_tau_s"])),
        **{f"{k} at zero": (orig[k] == 0.0, shifted[k] == 0.0) for k in TAU_NAMES},
    }
    for s in orig.get("slopes", []):
        checks[f"sign of {s}"] = (np.sign(orig[s]), np.sign(shifted[s]))
        checks[f"CI of {s} includes 0"] = (inc0(orig[f"ci_{s}"]), inc0(shifted[f"ci_{s}"]))
    return pd.DataFrame([{"check": c, "original": o, "shifted": s, "changed": o != s}
                         for c, (o, s) in checks.items()])
