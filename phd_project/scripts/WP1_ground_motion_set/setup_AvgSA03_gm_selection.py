import os
import json
import pickle
import numpy as np
import pandas as pd

from openquake.hazardlib.imt import PGA, SA, RSD595, AvgSA, IA, IMT

from phd_project.config.config import load_config
from phd_project.scripts.disagg_shards import load_shards, read_index

from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
    ESHM20SiteRupCtxBuilder,
    create_gmm_map,
    create_corr_model_map,
    get_poe_from_disaggstats,
    # The generic exclusion machinery lives in gm_selection.py (it cannot import
    # from this module). _exclusion_identities is re-exported here under its old
    # name, so existing imports from this module keep working.
    drop_unavailable_records as _drop_unavailable_records,
    _exclusion_identities,
    check_exclusion_versions,
    stripe_fingerprints_by_exclusion_version,
    normalise_selection_scheme,
)
import phd_project.scripts.WP1_ground_motion_set.manage_flatfiles as mf

cfg = load_config()


# Canonical selection configuration, shared by notebooks 031 (gcim percentiles),
# 032 (round config) and 033 (per-stripe manifest). It MUST be the single source
# for these values: stripe_input_fingerprint hashes them, so any divergence
# between notebooks would desynchronise the incremental stale check. Keep in sync
# with the values passed to build_final_ensembles in nb 032.
#
# The "db_exclusions" key (the current exclusion list) is added below, after
# UNAVAILABLE_RECORDS is defined -- see there.
SELECTION_CONFIG = {
    "percentiles": [0.05, 0.16, 0.33, 0.5, 0.67, 0.84, 0.95],
    "round_unbounded": [[], ["d"], ["m", "d", "vs30"], ["m", "d", "vs30"]],
    "force_optimisation": [False, False, False, False],
    "shuffle": [False, False, False, True],
    "n_shuffles": 5,
    "shuffle_rng_seeds": [1, 2, 3, 4, 5],
    "rng_seed": 1,
}


# Records that were SELECTED but cannot be used: either they cannot be obtained from
# their source database (the download crashes / the waveform is missing upstream), or
# the waveform that was obtained does not match the flatfile the selection was made
# from (so scaling it by the flatfile-based factor would give the wrong intensity).
# Each entry names one physical record by the identity its database uses in the
# combined selection DB:
#
#   NGA-Sub : ("index" == NGAsubRSN) or ("event_id", "station_code") ==
#             (NGAsubEQID, NGAsubSSN); the download unit is the whole
#             NGASub_RSN_<rsn> folder, so BOTH component rows (H1, H2) must go.
#   ESM     : ("event_id", "station_code", "location_code"); the download unit is
#             the single {event}__{station}__{loc}.h5, so all component rows
#             (U, V) must go.
#
# An identity value may be a single value (equality) or a list (membership), which
# is how a whole network is excluded: the ONA entry below predates the DB's
# "network_code" column (added 2026-09-30), so that network is expressed as its list
# of RSNs.
#
# Both selections (original and optimised) drop every record listed here from the
# database, in memory, before selecting (gm_db_for_selection below), and record the
# list in every stripe manifest (the "db_exclusions" key of SELECTION_CONFIG /
# SELECTION_CONFIG_OPTIMISED).
#
# The list is VERSIONED (see the "Database exclusions" section of gm_selection.py):
#
#   "added_in_version"  the version of this list that first contained the entry.
#                       The list "at version v" is every entry with
#                       added_in_version <= v; version 0 is no list at all. A stripe
#                       selected under an older version stays valid as long as it
#                       contains none of the records added since, so adding an
#                       entry reselects ONLY the stripes that contain it. Versions
#                       must run 1, 2, ..., N without gaps: a new entry gets
#                       N + 1 (or N, if it is added together with the newest one
#                       before anything has been selected with N). NEVER renumber
#                       or remove an existing entry -- every manifest written under
#                       an older version would then match nothing, and its stripe
#                       would be reselected.
#
#   Versions so far (and the stripes selected under each):
#     0  no list               528 original stripes
#     1  ONA                   original one-off reselection of (31, 1.15)
#     2  + RSN 6003695/6003748/6003796
#                              original one-offs (41, 0.65), (41, 0.95), (44, 1.15);
#                              all 532 optimised stripes
#     3  + RSN 7004773         everything selected from 2026-10-19 on
#   (The four original one-offs predate the versioning, and their manifests still
#   claim version 0; migrate_stripe_manifests --to recorded-exclusions re-stamps
#   them with the version their pickle records.)
#
#   "affected_stripes"  documentation only: {selection: [(site, iml), ...]} of the
#                       stripes that contained the record when it was added.
#
# "label" is a human tag for the console output; the row match is only ever on the
# identity fields. Only "database" + "identity" are hashed, so every other field can
# be edited without making any stripe stale.
UNAVAILABLE_RECORDS = [
    {
        "database": "NGASub",
        "label": "NGA-Sub network ONA (28 RSNs, 4040476-4040503)",
        # The ONA "network" is a single downhole array -- one location
        # (38.398757, 141.498107) with four sensors at 1.7 / 27.3 / 61.5 / 147.1 m
        # depth (NGAsubSSN 4002280-4002283), recorded for seven 2011 Japan events.
        # Its flatfile entries are visibly unfinished (Station_Name is the
        # placeholder "Ground Observation Point", Shortest_Usable_Period and
        # Epsilon_Phi_pt7 are -999, Vs30 is inferred with Vs30_Code "2J_3J"), and
        # the PEER portal serves no waveform for any of them: 0 of 28 have ever
        # downloaded, while K-NET / KiK-net (~40k records) download fine.
        # RSN list derived from Network == "ONA" in NGASub_SA_flatfile_rotD50.csv on
        # the HSU share. 24 of the 28 survive into the combined selection DB (48
        # rows, H1 + H2), at index labels 68473-68496 and 112550-112573.
        "identity": {"index": [
            "4040476", "4040477", "4040478", "4040479", "4040480", "4040481",
            "4040482", "4040483", "4040484", "4040485", "4040486", "4040487",
            "4040488", "4040489", "4040490", "4040491", "4040492", "4040493",
            "4040494", "4040495", "4040496", "4040497", "4040498", "4040499",
            "4040500", "4040501", "4040502", "4040503",
        ]},
        "reason": "PEER NGA-Sub portal serves no waveform for any ONA record; the "
                  "download crashes. Found via RSN 4040498, then confirmed across "
                  "the whole network.",
        "date": "2026-08-27",
        "added_in_version": 1,
        # Only one ONA record was ever selected: RSN 4040498 (Fukushimaoki 2011,
        # Mw 6.34), in this stripe of the original selection. The optimised
        # selection excluded ONA from the start.
        "affected_stripes": {"original": [(31, 1.15)]},
    },
    {
        "database": "NGASub",
        "label": "NGA-Sub RSN 6003695, 6003748, 6003796 (waveform ~3-4.5e4 x flatfile SA)",
        # Three distant records (Rjb ~1,570-1,970 km, Mw 6.4-6.7):
        #   RSN 6003695  EQID 6000158  SSN 6000279  DB rows 72851 (H1) / 116928 (H2)
        #   RSN 6003748  EQID 6000161  SSN 6000269  DB rows 72874 (H1) / 116951 (H2)
        #   RSN 6003796  EQID 6000035  SSN 6000691  DB rows 72907 (H1) / 116984 (H2)
        # The response spectra of their converted H1 JSONs are ~3-4.5e4 times the
        # flatfile SA that the GCIM selection and the scale factors alpha were computed
        # from (found 2026-09-29, record-spectra check / nb 043). The MSA applies alpha
        # to the JSON, so these records ran at an intensity ~4 orders of magnitude too
        # high. Whether the AT2 files or the flatfile are at fault is not established;
        # either way the waveform cannot be used with the flatfile-based alpha. Both
        # components are excluded, because the record (the RSN) is the unit.
        "identity": {"index": ["6003695", "6003748", "6003796"]},
        "reason": "converted H1 waveform spectra are ~3-4.5e4 x the flatfile SA the "
                  "selection and alpha were computed from, so the MSA ran them at the "
                  "wrong intensity",
        "date": "2026-09-30",
        "added_in_version": 2,
        # Selected (H1 in every case) in these stripes of the original selection,
        # slot in brackets (the optimised selection excluded them from the start):
        #   (41, 0.65): 6003695 [29]   -- run in the MSA, site 41, 3s and 5s
        #   (41, 0.95): 6003748 [26], 6003695 [29]   -- not in either structure's MSA
        #   (44, 1.15): 6003796 [29]   -- not in either structure's MSA
        "affected_stripes": {"original": [(41, 0.65), (41, 0.95), (44, 1.15)]},
    },
    {
        "database": "NGASub",
        "label": "NGA-Sub RSN 7004773 (waveform cannot be downloaded)",
        # Selected (H1, once each) in two stripes of the optimised selection; never
        # selected in the original selection. Found after the full optimised
        # selection had been run with version 2 of this list, so version 3 makes
        # only those two stripes stale. Both component rows go, because the RSN is
        # the download unit.
        "identity": {"index": ["7004773"]},
        "reason": "the waveform cannot be downloaded from the NGA-Sub database",
        "date": "2026-10-19",
        "added_in_version": 3,
        "affected_stripes": {"optimised": [(31, 0.65), (42, 0.5)]},
    },
]

# Current version of the list. Validates the versions when this module is imported
# (integers, running 1..N without gaps), so a malformed entry fails immediately
# rather than inside a stale check.
EXCLUSION_LIST_VERSION = check_exclusion_versions(UNAVAILABLE_RECORDS)

# The original selection filters with, and fingerprints against, the current list
# too. Its stripes selected before the list existed have no "db_exclusions" entry
# in their manifests (version 0); they stay valid via the versioned stale check
# (stripe_fingerprints below) as long as they contain none of the listed records.
SELECTION_CONFIG["db_exclusions"] = _exclusion_identities(UNAVAILABLE_RECORDS)


def drop_unavailable_records(gm_db: pd.DataFrame, records: list[dict] | None = None,
                             verbose: bool = True) -> pd.DataFrame:
    """Return a COPY of ``gm_db`` without every row of each record in ``records``
    (default: ``UNAVAILABLE_RECORDS``).

    Thin wrapper kept for its old signature; the implementation is
    ``gm_selection.drop_unavailable_records``. For the selection itself use
    :func:`gm_db_for_selection`, the one place the database is filtered.
    """
    records = UNAVAILABLE_RECORDS if records is None else records
    return _drop_unavailable_records(gm_db, records, verbose=verbose)


# =============================================================================
# Second, "optimised" selection (nbs 032/033/040/042 "-optimised" copies)
# =============================================================================
# A complete second AvgSA([0,3]) record selection, run alongside the original so
# the two record sets can be compared. The original (SELECTION_CONFIG above, the
# folders cfg["proc_data"]["gm_selection"] and cfg["results"]["AvgSA_03_record_
# selection"]) is NOT touched by any of this: the optimised selection has its own
# config constant and writes to its own folders (cfg["proc_data"]
# ["gm_selection_optimised"], cfg["results"]["AvgSA_03_record_selection_optimised"]).
#
# Why a separate constant rather than editing SELECTION_CONFIG: the per-stripe
# fingerprint hashes the selection config, so editing SELECTION_CONFIG's round
# settings would make all original stripes read as stale, and the next run of the
# original nb 032/033 would reselect and overwrite every one of them. (The one key
# the two configs share in meaning, "db_exclusions", is the exception: it is
# versioned -- see UNAVAILABLE_RECORDS -- so growing the list does not do that.)


# Round scheme of the optimised selection (the engine is build_final_ensembles in
# gm_selection.py; nothing in it changed for this):
#
#   round 1  select with M + Vs30 bounds, NO distance bound, then optimise EVERY
#            stripe (force_optimisation=True), including ones that already pass
#            straight after selection.
#   round 2  stripes for which round 1 could not form a record set at all are
#            reselected with NO bounds. Every stripe still failing is then
#            optimised with no bounds (force=True: the newly reselected ones too,
#            even if they already pass). A round-1 set that failed restarts from its
#            round-1 SELECTION, not from its round-1 optimised set.
#   round 3  no reselection; stripes still failing are re-optimised with no bounds
#            over 5 shuffled database orderings, keeping the better of {new, old}.
#            (force=True has no effect here, as every stripe in the round fails; it
#            is set only so all three rounds read the same.)
#
# A stripe that passes after any round is never touched again.
#
# "db_exclusions" is NOT a round setting: it is here so stripe_input_fingerprint
# hashes it (see gm_selection.stripe_input_fingerprint), which puts the exclusion
# into every optimised stripe's manifest. It is the CURRENT version of
# UNAVAILABLE_RECORDS; older versions are handled by stripe_fingerprints below.
# The filtering itself happens in gm_db_for_selection below -- the ONLY place it
# happens.
SELECTION_CONFIG_OPTIMISED = {
    "percentiles": SELECTION_CONFIG["percentiles"],   # same GCIM targets (nb 031 file shared)
    "round_unbounded": [["d"], ["m", "d", "vs30"], ["m", "d", "vs30"]],
    "force_optimisation": [True, True, True],
    "shuffle": [False, False, True],
    "n_shuffles": SELECTION_CONFIG["n_shuffles"],
    "shuffle_rng_seeds": SELECTION_CONFIG["shuffle_rng_seeds"],
    "rng_seed": SELECTION_CONFIG["rng_seed"],
    "db_exclusions": _exclusion_identities(UNAVAILABLE_RECORDS),
}


# =============================================================================
# Third and fourth selections: objectives that target the MOMENTS
# (nbs 032/033/040/042 "-optimised__mu_sigma" and "-optimised__ks_mu_sigma")
# =============================================================================
# Why: the KS-optimised selection minimised the weighted KS distance, which improved
# KS but pushed the set MOMENTS of ln IM away from the GCIM target -- most of all
# for Ds5-95, whose set SD rose to a median 1.44 x sigma_GCIM (nb 034). KS is nearly
# blind to a few extreme records (each moves the ECDF by only 1/30), but those
# records dominate the sample SD. The study uses the set mean and SD of ln Ds5-95 and
# ln SaRatio as regression covariates, so these two selections target them directly.
#
# Per stripe and selection IM m (conditioning IM dropped), with the 30 records:
#
#     z_mu,m    = (mean_m ln IM - mu_GCIM,m) / sigma_GCIM,m
#     r_sigma,m = s_m(ln IM) / sigma_GCIM,m              (sample SD, ddof=1)
#
#   __mu_sigma     J = sum_m w_m [ (1 - a_sigma) z_mu,m^2 + a_sigma (ln r_sigma,m)^2 ]
#                  moments only; no KS term, no KS-band penalty; NOT required to
#                  pass KS (acceptance "complete_set": a full 30-record set is enough;
#                  ks_passed is still stored, as a diagnostic).
#   __ks_mu_sigma  J' = c*V + ( l_KS sum w D + l_mu sum w z^2 + l_sigma sum w (ln r)^2 )
#                  with l_k = lambda_k / (lambda_KS + lambda_mu + lambda_sigma) and the
#                  KS-band penalty c*V (c = 10) outside the normalisation; still
#                  required to pass KS (acceptance "ks", as before).
#
# The objective functions and the dispatch are in gm_selection.py
# (weighted_moment_objective, weighted_ks_and_moment_objective, objective_and_kwargs).
#
# IM weights, BOTH selections: Ds5-95 0.40, each of the 21 spectral IMs 0.60/21
# ("weight_rsd595" below). It applies throughout -- the SSE-based record picking, the
# best-of-20 candidate choice, the optimiser and the round-3 keep-better -- because
# all of them read the context's imt_weights. That these selections differ from the
# KS-optimised one in the weight as well as the objective was accepted by the user.
#
# Scheme keys (read by setup_AvgSA03_gcim_gm_selection_for_scheme and the stale
# check; validated by gm_selection.normalise_selection_scheme):
#   "objective"       "ks" (default) | "mu_sigma" | "ks_mu_sigma"
#   "a_sigma"         mu_sigma only: weight of the SD error, in [0, 1]; a_mu = 1 - a_sigma
#   "lambda_ks/mu/sigma"  ks_mu_sigma only: importance weights, normalised to sum 1
#   "acceptance"      "ks" (default) | "complete_set"
#   "weight_rsd595"   the Ds5-95 IM weight the context is built with
#
# Fingerprints: the scheme enters every stripe / stage / final fingerprint through
# the selection context (gm_selection._selection_ctx_fingerprint_inputs), and ONLY
# because it is not the default -- the original and KS-optimised stripes are
# unaffected. "weight_rsd595" is fingerprinted through imt_weights, as always.
# Changing any value here therefore reselects the stripes of that selection only.
#
# Folders: cfg["proc_data"]["gm_selection_optimised__<scheme>"] and
# cfg["results"]["AvgSA_03_record_selection_optimised__<scheme>"]. The GCIM file
# (nb 031), the disaggregation inputs and the record JSON folder are shared and
# read only, as for the KS-optimised selection.

# __mu_sigma rounds (user decision D2 = (a), 2026-10-19):
#
#   round 1  select with M + Vs30 bounds, NO distance bound, then optimise EVERY
#            stripe (force_optimisation=True) against J_mu_sigma.
#   round 2  only stripes for which round 1 could not form a 30-record set at all:
#            reselect with NO bounds and optimise them.
#   (no round 3) Under acceptance "complete_set" every stripe with a set is already
#            done after round 1, so a shuffled best-of-5 round would touch nothing
#            -- round 1 already optimised every stripe against the moment objective.
SELECTION_CONFIG_MU_SIGMA = {
    "percentiles": SELECTION_CONFIG["percentiles"],   # same GCIM targets (nb 031 file shared)
    "round_unbounded": [["d"], ["m", "d", "vs30"]],
    "force_optimisation": [True, True],
    "shuffle": [False, False],
    "n_shuffles": SELECTION_CONFIG["n_shuffles"],         # unused (no shuffled round);
    "shuffle_rng_seeds": SELECTION_CONFIG["shuffle_rng_seeds"],  # kept so the keys match
    "rng_seed": SELECTION_CONFIG["rng_seed"],
    "db_exclusions": _exclusion_identities(UNAVAILABLE_RECORDS),
    "objective": "mu_sigma",
    "a_sigma": 0.5,              # equal weight on mean and SD error (a_mu = 0.5)
    "acceptance": "complete_set",
    "weight_rsd595": 0.40,
}

# __ks_mu_sigma rounds: the same three rounds as SELECTION_CONFIG_OPTIMISED (see the
# table there), all scored with J':
#
#   round 1  M + Vs30 bounds, no distance bound, force-optimise every stripe.
#   round 2  stripes with no set reselected with no bounds; every stripe still
#            failing KS optimised with no bounds.
#   round 3  stripes still failing KS: shuffled best-of-5, keep the better of
#            {new, old} (scored with J', the same objective as the optimiser).
SELECTION_CONFIG_KS_MU_SIGMA = {
    "percentiles": SELECTION_CONFIG["percentiles"],
    "round_unbounded": SELECTION_CONFIG_OPTIMISED["round_unbounded"],
    "force_optimisation": SELECTION_CONFIG_OPTIMISED["force_optimisation"],
    "shuffle": SELECTION_CONFIG_OPTIMISED["shuffle"],
    "n_shuffles": SELECTION_CONFIG["n_shuffles"],
    "shuffle_rng_seeds": SELECTION_CONFIG["shuffle_rng_seeds"],
    "rng_seed": SELECTION_CONFIG["rng_seed"],
    "db_exclusions": _exclusion_identities(UNAVAILABLE_RECORDS),
    "objective": "ks_mu_sigma",
    # equal importance (the user's choice; NOT equal influence -- see the report)
    "lambda_ks": 1.0,
    "lambda_mu": 1.0,
    "lambda_sigma": 1.0,
    "acceptance": "ks",
    "weight_rsd595": 0.40,
}

# Validate both at import, so a typo fails immediately rather than inside a run.
for _sc in (SELECTION_CONFIG_MU_SIGMA, SELECTION_CONFIG_KS_MU_SIGMA):
    normalise_selection_scheme(_sc)
del _sc


def gm_db_for_selection(gm_db: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """THE place where the selection database is filtered (both selections).

    Returns a copy of ``gm_db`` without every record listed in
    ``UNAVAILABLE_RECORDS`` -- ALL versions of the list, always, so a stripe
    (re)selected now can never pick any listed record. nb 032 and nb 032-optimised
    call it once, straight after ``setup_AvgSA03_gcim_gm_selection``, and pass ONLY
    the filtered frame to ``build_final_ensembles``.

    - The database file on disk is never edited; filtering is in memory with
      ``DataFrame.drop(index=...)``, so the surviving rows keep their original
      index labels and every selected record still points at the right row of the
      canonical CSV (nb 033's network-code lookup relies on that).
    - The file-byte hash of the database cannot see this filter, so the exclusion
      is stated in the provenance explicitly: ``db_exclusions`` in every stripe
      manifest (from the selection config), and the ``db_exclusions`` argument of
      ``build_final_ensembles`` in the stage caches and the final-ensembles
      manifest. Adding a record to ``UNAVAILABLE_RECORDS`` (as a new version)
      makes stale ONLY the stripes that contain it -- see :func:`stripe_fingerprints`.
    """
    return _drop_unavailable_records(gm_db, UNAVAILABLE_RECORDS, verbose=verbose)


# Old name, kept so existing imports keep working. Same function: the filter used
# to apply to the optimised selection only.
gm_db_for_selection_optimised = gm_db_for_selection


def stripe_fingerprints(selection_config: dict, source_fps_stripe: dict,
                        basic_selection_ctx: dict, disagg_stats):
    """``(fp_fn, superseded)`` for the stale check of one AvgSA_03 selection.

    ``selection_config`` is ``SELECTION_CONFIG`` (original selection) or
    ``SELECTION_CONFIG_OPTIMISED``. Pass both results to
    ``gm_selection.find_stale_stripes(wanted, folder, fp_fn, superseded=superseded)``.
    ``fp_fn`` is also what a newly written stripe manifest records.

    A stripe is valid if its manifest matches the current exclusion list, or an
    older version of it and the stripe contains none of the records added since.
    Every notebook that runs the stale check (031, 032, 033 and the optimised
    copies) must get its functions from here, so they all decide alike. See
    ``gm_selection.stripe_fingerprints_by_exclusion_version``.
    """
    return stripe_fingerprints_by_exclusion_version(
        selection_config, UNAVAILABLE_RECORDS, source_fps_stripe,
        basic_selection_ctx, disagg_stats)


def stripe_source_fps() -> dict:
    """Source-file paths that determine each stripe's result.

    Consumed by ``gm_selection.stripe_input_fingerprint`` (via
    ``find_stale_stripes``) in every notebook, so the incremental stale check is
    computed identically throughout. Deliberately EXCLUDES the per-site IML
    subset JSON (``AvgSA_03_imls_for_selection``): that grows the set of wanted
    stripes without changing any existing stripe's fingerprint.

    ``disagg_shard_dir`` is the *directory* of per-site shards, not a file:
    ``stripe_input_fingerprint`` resolves it to that stripe's own entry in the
    set's ``_content_hashes.json``, so only a stripe whose own disaggregation
    DataFrame changed goes stale -- re-running the disaggregation for one site, or
    adding one iml across all sites, leaves every other stripe valid. Never
    fingerprint the directory itself -- a non-file path is hashed as a string,
    which looks valid while tracking nothing.

    ``disagg_stats_file`` is still listed because the *batch* artifacts
    (``build_final_ensembles``) depend on the whole table. Per-stripe fingerprints
    do not use it: they hash that stripe's single stats row from the loaded
    DataFrame instead, so appending rows for a new iml no longer invalidates
    every stripe.
    """
    return {
        "disagg_shard_dir":  cfg["proc_data"]["AvgSA_03_disagg_data_shards"],
        "disagg_stats_file": cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"],
        "gm_db_file":        cfg["proc_data"]["gm_database"],
        "site_model_file":   cfg["hazard_models"]["eshm20_wp1_site_model"],
        "gmm_lt_file":       cfg["hazard_models"]["eshm20_AvgSA_03_median_lt"],
    }


DISAGG_IMT = "AvgSA"   # conditioning IM name used as the disagg_data key


def _select_stripe_keys(iml_keys_by_site: dict, disagg_stats, imls_for_selection: dict,
                        imt: str = DISAGG_IMT):
    """Resolve the wanted ``(site, iml)`` keys against what was disaggregated.

    ``iml_keys_by_site[site]`` is that site's list of **native** disagg iml keys.
    Returns ``(keys, invalid)``: ``keys`` in the canonical pipeline order (grouped
    by site, and within a site in the order the ``union`` list requests them),
    ``invalid`` the requested-but-unusable ``(site, iml)``.

    Single source of this logic, shared by :func:`wanted_stripe_keys` (which reads
    the shard index and opens no shard) and :func:`_set_up_selection` (which loads
    the shards). If the two ever disagreed, the stale check would target a
    different key set from the one that gets computed.
    """
    keys, invalid = [], []
    for site in sorted(iml_keys_by_site):
        site_entry = imls_for_selection.get(str(site))
        wanted = site_entry.get("union") if site_entry else None
        if wanted is None:
            continue  # site not listed in the subset JSON -> skip
        iml_keys = iml_keys_by_site[site]
        for iml in wanted:
            if iml is None:
                continue  # upper-stripe placeholder (iml above the hazard ceiling)
            # validity probe: a zero-hazard / excluded iml has no stats row (poe None)
            poe = get_poe_from_disaggstats(disagg_stats, site, imt, iml)
            key = next((k for k in iml_keys if np.isclose(k, iml)), None)
            if poe is None or key is None:
                invalid.append((site, iml))
                continue
            keys.append((site, key))
    # grouped by site; stable, so the within-site request order is preserved
    return sorted(keys, key=lambda x: x[0]), invalid


def wanted_stripe_keys(imt: str = DISAGG_IMT) -> list[tuple[int, float]]:
    """The full wanted ``(site, iml)`` key set, without opening a disagg shard.

    Reads only the shard ``_index.json``, the 0.1 MB disagg-stats pickle and the
    IML-subset JSON, so it runs in milliseconds. Notebooks 031/032/033 use it to
    run the stale check *before* deciding which shards (if any) to load.

    Guaranteed to equal ``list(setup_AvgSA03_gcim_gm_selection()[0].keys())`` --
    both go through :func:`_select_stripe_keys`.
    """
    index = read_index(cfg["proc_data"]["AvgSA_03_disagg_data_shards"])
    iml_keys_by_site = {site: d[imt] for site, d in index.items() if imt in d}

    with open(cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"], "rb") as f:
        disagg_stats = pickle.load(f)
    with open(cfg["proc_data"]["AvgSA_03_imls_for_selection"]) as f:
        imls_for_selection = json.load(f)

    keys, _ = _select_stripe_keys(iml_keys_by_site, disagg_stats, imls_for_selection, imt)
    return keys


def setup_AvgSA03_gcim_gm_selection_w_IA(weight_rsd595=0.125, weight_ia=0.125, sites=None):
    ########### set some parameters for the selection ##########################
    t_lower = 0.025     # lower SA period considered in selection
    t_upper = 3         # upper SA period considered in selection
    n_periods = 20      # number of periods to consider in selection

    conditioning_imt: IMT = AvgSA([0,3]) 
    nonSA_imts: list[IMT] = [AvgSA([0,3]), RSD595(), PGA(), IA()] 
    sa_periods = np.round(np.geomspace(t_lower, t_upper, num=n_periods), 3)
    SA_imts: list[IMT] = [SA(period) for period in sa_periods]
    selection_imts: list[IMT] = nonSA_imts[1:] + SA_imts    # not AvgSA but the others (PGA and RSD595)
    nonSA_imt_strs: list[str] = [im.string for im in nonSA_imts] # strings match the correlation matrix

    # weights of the IMs  -> assumed weights
    # remaining_weight = 1 - (weight_rsd595 + weight_ia)
    # n_other_ims = len([imt for imt in selection_imts if imt.name not in ["RSD595", "IA"]])
    # imt_weights = np.array([remaining_weight / n_other_ims if imt.name not in ["RSD595", "IA"] 
    #                         else weight_rsd595 for imt in selection_imts])
    # imt_weights /= imt_weights.sum()

    imt_weights = _im_weights(selection_imts, 
                              {"RSD595": weight_rsd595,
                               "IA": weight_ia})

    return _set_up_selection(conditioning_imt, selection_imts, imt_weights, 
                             nonSA_imt_strs, sa_periods, t_upper, sites)


def setup_AvgSA03_gcim_gm_selection_for_scheme(selection_config: dict, sites=None):
    """:func:`setup_AvgSA03_gcim_gm_selection` for a given selection scheme.

    Every notebook of a moment-objective selection (032/033 "__mu_sigma" /
    "__ks_mu_sigma" copies) builds its selection context through here, so the
    context ALWAYS carries the scheme's RSD595 weight and its selection scheme:

    - ``weight_rsd595`` from the config (default 0.25 if the config has none), so
      the IM weights -- and therefore the stripe fingerprints -- of nb 032 and nb 033
      agree. (Calling the plain setup function in one of them would give 0.25 there
      and a different fingerprint for every stripe.)
    - ``basic_selection_ctx["selection_scheme"]``: the validated, canonical scheme
      (objective, its parameters, acceptance; see
      gm_selection.normalise_selection_scheme), which the engine reads at every
      scoring and acceptance step and which enters the fingerprints.

    For ``SELECTION_CONFIG`` / ``SELECTION_CONFIG_OPTIMISED`` it returns exactly what
    the plain setup function returns (weight 0.25, no scheme entry).
    Same return value and ``sites`` semantics as the plain function.
    """
    site_iml_disaggs, disagg_stats, site_model, basic_selection_ctx, gm_db = \
        setup_AvgSA03_gcim_gm_selection(
            weight_rsd595=selection_config.get("weight_rsd595", DEFAULT_WEIGHT_RSD595),
            sites=sites)
    scheme = normalise_selection_scheme(selection_config)
    if scheme["objective"] != "ks" or scheme["acceptance"] != "ks":
        # Only a non-default scheme is stored, so the default context stays exactly
        # the one the plain setup function returns.
        basic_selection_ctx["selection_scheme"] = scheme
    return site_iml_disaggs, disagg_stats, site_model, basic_selection_ctx, gm_db


# Ds5-95 weight of the original and KS-optimised selections (the remaining weight is
# shared equally by the 21 spectral IMs).
DEFAULT_WEIGHT_RSD595 = 0.25


def setup_AvgSA03_gcim_gm_selection(weight_rsd595=DEFAULT_WEIGHT_RSD595, sites=None):
    """Build everything the AvgSA([0,3]) record selection needs.

    ``sites`` selects which per-site disagg shards to load: ``None`` loads every
    site with a wanted stripe (the old whole-monolith behaviour), an iterable loads
    only those, and ``()`` loads none -- for callers that need the selection context
    and stats but no disagg data (nb 033, and the stale check in 031/032 before it
    knows what is stale). ``site_iml_disaggs`` is restricted to the loaded sites.
    """
    ########### set some parameters for the selection ##########################
    t_lower = 0.025     # lower SA period considered in selection
    t_upper = 3         # upper SA period considered in selection
    n_periods = 20      # number of periods to consider in selection

    conditioning_imt: IMT = AvgSA([0,3]) 
    nonSA_imts: list[IMT] = [AvgSA([0,3]), RSD595(), PGA()] 
    sa_periods = np.round(np.geomspace(t_lower, t_upper, num=n_periods), 3)
    SA_imts: list[IMT] = [SA(period) for period in sa_periods]
    selection_imts: list[IMT] = nonSA_imts[1:] + SA_imts    # not AvgSA but the others (PGA and RSD595)
    nonSA_imt_strs: list[str] = [im.string for im in nonSA_imts] # strings match the correlation matrix

    imt_weights = _im_weights(selection_imts, {"RSD595": weight_rsd595})

    return _set_up_selection(conditioning_imt, selection_imts, imt_weights, 
                             nonSA_imt_strs, sa_periods, t_upper, sites)


def _im_weights(selection_imts, spec_weights: dict[str, float]):
    """ helper function to calculate the im weights based on the specified 
    weights for certain imts and equal weighting for the rest """

    selection_im_strings = [imt.string for imt in selection_imts]
    assert all(im in selection_im_strings for im in spec_weights.keys()), \
        "Specified weights must be for IMTs in the selection IMTs"

    speced_weight = sum(spec_weights.values())
    remaining_weight = 1 - speced_weight
    n_other_ims = len(selection_imts) - len(spec_weights)
    default_weight = remaining_weight / n_other_ims if n_other_ims > 0 else 0

    imt_weights = [spec_weights.get(im.string, default_weight) for im in selection_imts]
    imt_weights = np.array(imt_weights)
    imt_weights /= imt_weights.sum()
    return imt_weights
    

def _set_up_selection(
        conditioning_imt, selection_imts, imt_weights,
        nonSA_imt_strs, sa_periods, t_upper, sites=None):

    # some other things
    assumed_rake = -90  # assumed rake for RSD595 calculation
    occurence = True    # the record selection should be performed based on occurence

    ####################### Load Data ##########################################
    # The IML-based disaggregation (the sig4 / eps4 truncation set from nb 021) is
    # stored per site: <shard_dir>/site_NNN.pickle -> {imt: {iml: DataFrame}}, plus
    # an _index.json carrying the native iml keys. We resolve the wanted (site, iml)
    # from the index FIRST and then load only the shards those keys need, so a
    # caller after a couple of sites does not pay for all 60.
    # disagg_stats is a flat DataFrame, one row per (site, imt, imtl) with a poe column.
    shard_dir = cfg["proc_data"]["AvgSA_03_disagg_data_shards"]

    fp = cfg["proc_data"]["AvgSA_03_disagg_stats_gm_selection"]
    with open(fp, "rb") as f:
        disagg_stats = pickle.load(f)

    # Only build ensembles for a per-site subset of IMLs (one MSA stripe each), listed in the
    # imls_for_selection JSON: {site_id_str: {structure_tag: [iml, ...], ..., "union": [iml, ...]}}.
    # We use the per-site "union" (all IMLs any structure at the site requests).
    with open(cfg["proc_data"]["AvgSA_03_imls_for_selection"]) as f:
        imls_for_selection = json.load(f)

    # organise the disagg data, keyed by (site, iml). iml is the native key of the
    # disagg data (shard[imt][iml]); poe is derived downstream only where it is
    # needed as metadata (see get_poe_from_disaggstats). The key set comes from the
    # index via _select_stripe_keys -- the SAME helper wanted_stripe_keys() uses, so
    # the stale check can never target a different key set from what is built here.
    imt = conditioning_imt.name  # "AvgSA" (only works for AvgSA; see disagg_imt comment below)
    index = read_index(shard_dir)
    iml_keys_by_site = {site: d[imt] for site, d in index.items() if imt in d}
    all_keys, invalid = _select_stripe_keys(
        iml_keys_by_site, disagg_stats, imls_for_selection, imt)

    # Load only the shards the caller asked for (None -> every site that has a
    # wanted stripe; () -> none at all).
    if sites is None:
        load_sites = {s for s, _ in all_keys}
    else:
        load_sites = {int(s) for s in sites}
    shards = load_shards(shard_dir, load_sites)

    # Key by the exact iml float (the native key); poe is 1:1 with iml per site and
    # is derived downstream where it is needed as metadata.
    site_iml_disaggs = {(site, key): shards[site][imt][key]
                        for site, key in all_keys if site in shards}

    print(f"Built {len(site_iml_disaggs)} (site, iml) disaggregations "
          f"across {len({s for s, _ in site_iml_disaggs})} sites "
          f"({len(shards)} of {len(index)} shards loaded; "
          f"{len(all_keys)} (site, iml) wanted in total).")
    if invalid:
        print(f"WARNING: {len(invalid)} (site, iml) requested in imls_for_selection are not "
              f"present in disagg_stats/disagg_data and were skipped:")
        for site, iml in invalid:
            print(f"    site {site}: iml {iml}")

    # load the site file
    site_model = pd.read_csv(cfg["hazard_models"]["eshm20_wp1_site_model"])

    # load the flatfiles
    flatfile_folder = cfg["proc_data"]["corr_model"] / "reverse" / "flatfiles"
    flatfiles = {}
    for f in [f for f in os.listdir(flatfile_folder) if f.endswith(".csv")]:
        tag = f.split("_")[0]
        flatfiles[tag] = pd.read_csv(flatfile_folder / f, delimiter=";", index_col=0, low_memory=False)

    flatfiles["volcanic"] = pd.read_csv(cfg["raw_data"]["gm_flatfiles"] / "volcanic_lanzanoluzi_flatfile.csv", 
                                        delimiter=";", index_col=0)

    # load the preprocessed gm database
    gm_database = pd.read_csv(cfg["proc_data"]["gm_database"], sep=",", low_memory=False, header=[0, 1])

    # filter the gm_database so that only the selection and conditioning ims are present
    gm_db = gm_database.copy()
    updated_ims = mf.filter_gm_database_on_imts(
        gm_db["ims"], selection_imts + [conditioning_imt])
    updated_ims.columns = pd.MultiIndex.from_product([['ims'], updated_ims.columns])
    gm_db = pd.concat([gm_db.drop('ims', axis=1, level=0), updated_ims], axis=1)

    # Create the GMM Map for AvgSA by reading the median-branch logic tree
    AvgSA_03_lt_fp = cfg["hazard_models"]["eshm20_AvgSA_03_median_lt"]
    gmm_map = create_gmm_map(AvgSA_03_lt_fp)

    # get the correlation model map
    corr_map = create_corr_model_map(nonSA_imt_strs, sa_periods)

    ######################## Make some assumptions #############################
    # create the average depth map for each TRT
    average_depths = {
        "Craton": flatfiles["asc"]["ev_depth_km"].mean(),
        "Non-Subduction Deep": flatfiles["vran"]["ev_depth_km"].mean(),
        "Shallow Default": flatfiles["asc"]["ev_depth_km"].mean(),
        "Subduction Inslab": flatfiles["sinter"]["ev_depth_km"].mean(),
        "Subduction Interface": flatfiles["sinter"]["ev_depth_km"].mean(),
        "Volcanic": flatfiles["volcanic"]["ev_depth_km"].mean(),
    }

    # add the average depth to the top of the rupture
    average_ztor = {
        "Craton": flatfiles["asc"]["ztor"].mean(),
        "Non-Subduction Deep": flatfiles["vran"]["ztor"].mean(),
        "Shallow Default": flatfiles["asc"]["ztor"].mean(),
        "Subduction Inslab": flatfiles["sslab"]["ztor"].mean(),
        "Subduction Interface": flatfiles["sinter"]["ztor"].mean(),
        "Volcanic": flatfiles["asc"]["ztor"].mean(),
    }

    # the map of what sim trts are allowed to match with what record trts
    OK_TRT_MATCHES = {
        "Craton": ["Shallow Default"],
        "Non-Subduction Deep": ["Non-Subduction Deep", "Subduction Inslab", "Subduction"],
        "Shallow Default": ["Shallow Default"],
        "Subduction Inslab": ["Non-Subduction Deep", "Subduction Inslab", "Subduction"],
        "Subduction Interface": ["Subduction Interface"],
        "Volcanic": ["Shallow Default"],
    }

    ###################### set up the selection context ########################
    basic_selection_ctx = {
        "n_ensembles": 20,
        "n_samples": 30,
        "conditioning_imt": conditioning_imt ,
        "disagg_imt": conditioning_imt.name , # this only works for AvgSA. otherwise used .string 
        "selection_imts": selection_imts ,
        "imt_weights": imt_weights ,
        "ctx_builder": ESHM20SiteRupCtxBuilder ,
        "ctx_builder_params": ["average_depths", "assumed_rake", "average_ztor"] ,
        "average_depths": average_depths ,
        "assumed_rake": assumed_rake ,
        "average_ztor": average_ztor ,
        "gmm_map": gmm_map ,
        "corr_map": corr_map ,
        "m_bound_model": "tarbali_and_bradley_2016" ,
        "d_bound_model": "tarbali_and_bradley_2016" ,
        "vs30_bound_model": "tarbali_and_bradley_2016" ,
        "sf_bounds": None ,
        "usable_T": t_upper ,
        "max_n_recs": 5 ,
        "p_value": 0.05 ,
        "ok_trt_matches": OK_TRT_MATCHES ,
        "occurence": occurence ,
    }

    return site_iml_disaggs, disagg_stats, site_model, basic_selection_ctx, gm_db
