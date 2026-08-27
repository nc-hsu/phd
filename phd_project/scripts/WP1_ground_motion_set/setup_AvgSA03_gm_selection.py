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
)
import phd_project.scripts.WP1_ground_motion_set.manage_flatfiles as mf

cfg = load_config()


# Canonical selection configuration, shared by notebooks 031 (gcim percentiles),
# 032 (round config) and 033 (per-stripe manifest). It MUST be the single source
# for these values: stripe_input_fingerprint hashes them, so any divergence
# between notebooks would desynchronise the incremental stale check. Keep in sync
# with the values passed to build_final_ensembles in nb 032.
SELECTION_CONFIG = {
    "percentiles": [0.05, 0.16, 0.33, 0.5, 0.67, 0.84, 0.95],
    "round_unbounded": [[], ["d"], ["m", "d", "vs30"], ["m", "d", "vs30"]],
    "force_optimisation": [False, False, False, False],
    "shuffle": [False, False, False, True],
    "n_shuffles": 5,
    "shuffle_rng_seeds": [1, 2, 3, 4, 5],
    "rng_seed": 1,
}


# Records that were SELECTED but cannot be obtained from their source database
# (the download crashes / the waveform is missing upstream). Each entry names one
# physical record by the identity its database uses in the combined selection DB:
#
#   NGA-Sub : ("event_id", "station_code") == (NGAsubEQID, NGAsubSSN); the download
#             unit is the whole NGASub_RSN_<rsn> folder, so BOTH component rows
#             (H1, H2) must go.
#   ESM     : ("event_id", "station_code", "location_code"); the download unit is
#             the single {event}__{station}__{loc}.h5, so all component rows
#             (U, V) must go.
#
# This is documentation + input to the one-off reselection cell at the end of nb
# 032. Keep "rsn" (NGA-Sub) purely as a human cross-reference; the row match is on
# the identity fields.
UNAVAILABLE_RECORDS = [
    {
        "database": "NGASub",
        # (NGAsubEQID, NGAsubSSN) for RSN 4040498 -- Fukushimaoki, 2011-07-24, Mw 6.34,
        # Subduction Interface. Two rows in the combined DB: 68491 (H1), 112568 (H2).
        "identity": {"event_id": "4000044", "station_code": "4002282"},
        "rsn": 4040498,
        "reason": "PEER NGA-Sub portal crashes on this RSN; the waveform cannot be "
                  "downloaded",
        "date": "2026-08-27",
        "affected_stripes": [(31, 1.15)],
    },
]


def drop_unavailable_records(gm_db: pd.DataFrame, records: list[dict] | None = None,
                             verbose: bool = True) -> pd.DataFrame:
    """Return a COPY of ``gm_db`` with every row of each unavailable record removed.

    **Deliberately NOT called by :func:`setup_AvgSA03_gcim_gm_selection`.** The
    default selection path must keep using the full database, because
    ``gm_selection.stripe_input_fingerprint`` hashes the database *file's bytes*:
    the ~510 stripes already on disk were selected from the complete DB and their
    manifests say so. Filtering by default would make every future selection
    silently inconsistent with that provenance -- and editing the CSV itself is
    worse still, since the selection results identify records by the DataFrame's
    positional ``RangeIndex`` (see ``greedy_optimise_ensemble``), which every row
    below a deleted one would shift.

    Use this ONLY from the one-off reselection cell in nb 032, which reselects a
    single ``(site, iml)`` around a record that turned out to be undownloadable
    and records the deviation in the stripe's ``db_exclusions`` key.

    Dropping is done with ``DataFrame.drop(index=...)`` so the surviving rows keep
    their original index labels -- that equivalence with the on-disk DB is the
    whole point of filtering in memory rather than on disk.
    """
    records = UNAVAILABLE_RECORDS if records is None else records
    meta = gm_db["metadata"]

    to_drop = pd.Index([])
    for rec in records:
        mask = meta["database"] == rec["database"]
        for field, value in rec["identity"].items():
            mask &= meta[field].astype(str) == str(value)
        hits = meta.index[mask]
        if verbose:
            label = rec.get("rsn", rec["identity"])
            print(f"  excluding {rec['database']} record {label}: "
                  f"{len(hits)} row(s) {list(hits)}")
        to_drop = to_drop.union(hits)

    filtered = gm_db.drop(index=to_drop)
    if verbose:
        print(f"gm_db: {len(gm_db)} -> {len(filtered)} rows "
              f"({len(to_drop)} dropped)")
    return filtered


def stripe_source_fps() -> dict:
    """Source-file paths that determine each stripe's result.

    Consumed by ``gm_selection.stripe_input_fingerprint`` (via
    ``find_stale_stripes``) in every notebook, so the incremental stale check is
    computed identically throughout. Deliberately EXCLUDES the per-site IML
    subset JSON (``AvgSA_03_imls_for_selection``): that grows the set of wanted
    stripes without changing any existing stripe's fingerprint.

    ``disagg_shard_dir`` is the *directory* of per-site shards, not a file:
    ``stripe_input_fingerprint`` resolves it to ``site_NNN.pickle`` for the stripe's
    own site, so re-running the disaggregation for one site leaves the other 59
    sites' stripes valid. Never fingerprint the directory itself -- a non-file path
    is hashed as a string, which looks valid while tracking nothing.
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


def setup_AvgSA03_gcim_gm_selection(weight_rsd595=0.25, sites=None):
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
