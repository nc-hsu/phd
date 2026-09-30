"""Compare the spectra computed from the downloaded time histories with the database spectra.

TWO SOURCES OF THE SAME SPECTRUM
--------------------------------
Every record that was selected (and then downloaded) has two 5 %-damped PSA spectra:

  TH spectrum  computed by us from the downloaded acceleration time history
               (``phd_project/scripts/compute_record_spectra.py``), on a fine grid 0.02-6.00 s
               every 0.01 s. File: ``07_gm_records/spectra/acc_spectra.pickle``.
  DB spectrum  the spectrum stored in the ESM / NGA-Sub flatfiles, which is what the GCIM
               selection saw. File: ``04_gm_database_for_selection/ESM-NGAsub_combined.csv``.

If the downloaded record is the one that was selected, and it was processed the same way, the two
should agree to within round-off plus small differences in processing. This module measures how
far apart they are, record by record, so that the discrepancies can be investigated.

WHAT IS COMPARED
----------------
Only the DB periods, because the DB is the coarse one. The TH spectrum is interpolated to those
periods (linearly in log T - log SA, the natural space for a spectrum). Most DB periods lie exactly
on the 0.01 s TH grid, so the interpolation only really acts at 0.025 s. DB periods outside the TH
grid (SA(0.01), SA(7.0)-SA(10.0)) cannot be compared and are dropped; PGA is not a spectral
ordinate here and is ignored.

    diff = SA_TH - SA_DB                 [g]
    pct  = 100 * (SA_TH - SA_DB) / SA_DB [%]   (signed, relative to the DB)

ERROR METRIC: the default is the mean squared logarithmic error over the N_T scored periods,

    MSLE = (1 / N_T) * sum_j [ ln SA_TH(T_j) - ln SA_DB(T_j) ]^2        [-]

which is symmetric in over/under-prediction (a factor 2 either way scores the same), scale-free,
and a mean (so records scored over different numbers of periods are comparable). The earlier
metric, the sum of squared % errors ``sse_pct`` [%²], is still computed and can be selected with
``metric="sse_pct"`` wherever a function takes a ``metric``.

MAXIMUM USABLE PERIOD: each DB record carries ``max_usable_T``, the longest period its processing
(high-pass filter) leaves trustworthy. Beyond it the DB ordinates are small and filter-affected, and
on a % scale they can dominate the score (GR-1985-0007_DRA1_0__U: < 2 % error up to 1.8 s, 100-185 %
beyond 2 s, max_usable_T = 3.2 s). So by default (``use_max_usable_T=True``) every score, i.e. the
max +/- %, the MSLE and the SSE, uses only periods T <= max_usable_T. The per-period table keeps all periods
and flags the scored ones in its ``usable`` column. With the limit on, records are scored over
different numbers of periods (``n_periods``).

RECORD TAG <-> DATABASE ROW
---------------------------
The TH spectra are indexed by the record JSON file stem, the "tag":

    ESM      {event_id}_{station_code}_{location_code}__{component}  e.g. EMSC-20161101_0000060_RM33_0__V
    NGA-Sub  {RSN}__{component}                                     e.g. 1000040__H1

The DB row for a tag is found by building the same tag from the DB metadata, with the standes
naming helpers the pipeline itself used. Two details matter:
  * ESM location_code is stored as "00" in the CSV but was written as "0" into the file names
    (the selection read it as an integer), so it is converted through int here too.
  * the NGA-Sub RSN is the CSV's ``index`` column, NOT ``station_code``.

AMBIGUOUS TAGS: the CSV stores every ESM location_code as "00", so two physically different ESM
records at the same (event, station) collapse onto one tag. A handful of downloaded tags therefore
match two DB rows. Nothing in the CSV says which one was selected, so such tags are flagged
``ambiguous`` and kept out of the error histogram; every function here reports each DB row
separately for them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from standes.groundmotion import esm_record_identifier, ngasub_record_identifier, record_json_filename

from phd_project.config.config import load_config


# ---------------------------------------------------------------------------------------------
# 1. Default locations
# ---------------------------------------------------------------------------------------------

_CFG = load_config()
_PROC = _CFG["proc_data"]["root"]
# per-structure MSA collapse-flag JSONs (nb 060): which record ran in which slot of each stripe
SITE_FRAGILITY_DIR = _CFG["proc_data"]["wp1_sites_fragility_curves"]
# per-(site, stripe) GCIM selection pickles, whose "recs" table lists the 30 records in slot order
AVGSA03_SELECTION_DIR = _CFG["results"]["AvgSA_03_record_selection"]
TH_SPECTRA_FP = _PROC / "07_gm_records" / "spectra" / "acc_spectra.pickle"
DB_FP = _PROC / "04_gm_database_for_selection" / "ESM-NGAsub_combined.csv"
# The original ESM downloads (ASDF / HDF5, one file per event-station-location) on the HSU share
ESM_HDF5_DIR = _CFG["raw_data"]["esm_hdf5_folder"]

# Plot colours (fixed order, so a colour always means the same thing across plots):
# TH spectrum in slot 1 (blue); DB spectra in slots 2, 3 (orange, aqua) - a 2nd DB colour is
# only ever needed for an ambiguous tag.
_TH_COLOUR = "#2a78d6"
_DB_COLOURS = ["#eb6834", "#1baf7a"]


# ---------------------------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_th_spectra(fp: str | Path | None = None) -> pd.DataFrame:
    """TH spectra: index = tag, columns = periods [s] (float), values = PSA [g].

    Cached: the plotting and per-tag functions call this repeatedly. Do not modify the returned
    frame in place, or the cached copy changes with it.
    """
    return pd.read_pickle(Path(fp) if fp is not None else TH_SPECTRA_FP)


def db_record_tag(meta: pd.DataFrame) -> pd.Series:
    """The record tag (JSON file stem) of every DB row, built exactly as the pipeline built it.

    ``meta`` is the ``metadata`` block of the combined DB, with the id columns read as strings.
    The standes helpers are the single source of truth for record naming; the ".json" that
    ``record_json_filename`` appends is stripped to give the stem.
    """
    def tag(row) -> str:
        if row["database"] == "ESM":
            # location "00" in the CSV -> "0" in the file name (see module docstring)
            rid = esm_record_identifier(row["event_id"], row["station_code"],
                                        str(int(row["location_code"])))
        else:
            rid = ngasub_record_identifier(row["index"])        # the RSN is the "index" column
        return record_json_filename(rid, row["component"])[:-len(".json")]

    return meta.apply(tag, axis=1)


@lru_cache(maxsize=4)
def load_db_spectra(fp: str | Path | None = None) -> pd.DataFrame:
    """DB metadata + spectra as one flat frame.

    Columns: the metadata columns, a ``tag`` column, and one column per SA period as a float
    [s] (so ``df[1.0]`` is SA(1.0)). The DataFrame index is the CSV row position, which tells
    apart the two rows of an ambiguous tag. Cached: the 120 MB CSV takes a while to read.
    """
    fp = Path(fp) if fp is not None else DB_FP
    id_cols = ["index", "event_id", "station_code", "location_code", "component"]
    raw = pd.read_csv(fp, header=[0, 1], low_memory=False,
                      dtype={("metadata", c): str for c in id_cols})
    meta = raw["metadata"].copy()
    # keep only the SA columns and turn "SA(0.25)" into 0.25
    sa = raw["ims"].filter(regex=r"^SA\(")
    sa.columns = [float(c[3:-1]) for c in sa.columns]
    meta["tag"] = db_record_tag(meta)
    return pd.concat([meta, sa], axis=1)


def _db_periods(db: pd.DataFrame) -> np.ndarray:
    """The SA periods of the DB frame, i.e. its float-named columns, in ascending order."""
    return np.array(sorted(c for c in db.columns if isinstance(c, float)))


# ---------------------------------------------------------------------------------------------
# 3. Interpolation
# ---------------------------------------------------------------------------------------------

def interp_loglog(periods_src, values_src, periods_tgt) -> np.ndarray:
    """Interpolate a spectrum linearly in log(T) - log(SA).

    A spectrum is close to piecewise power-law, so straight lines in log-log space follow it far
    better than in linear space. At a target period that lies on the source grid the result is
    the source value itself (up to floating-point round-off in exp(log(x))).
    Targets outside the source range are not extrapolated; callers restrict to the overlap.
    """
    return np.exp(np.interp(np.log(periods_tgt), np.log(periods_src), np.log(values_src)))


# ---------------------------------------------------------------------------------------------
# 4. Comparison
# ---------------------------------------------------------------------------------------------

def _compared_periods(th: pd.DataFrame, db: pd.DataFrame) -> np.ndarray:
    """DB periods that lie inside the TH grid (the only ones that can be compared)."""
    t_th = th.columns.to_numpy(dtype=float)
    t_db = _db_periods(db)
    # small tolerance so that a DB period equal to a TH end point is kept despite float noise
    return t_db[(t_db >= t_th.min() - 1e-9) & (t_db <= t_th.max() + 1e-9)]


def _usable_mask(periods, max_usable_T, use_max_usable_T: bool = True) -> np.ndarray:
    """True for the periods that are scored: all of them, or only T <= max_usable_T.

    A tiny tolerance keeps a period that equals max_usable_T despite float noise.
    """
    periods = np.asarray(periods, dtype=float)
    if not use_max_usable_T:
        return np.ones(periods.shape, dtype=bool)
    return periods <= np.asarray(max_usable_T, dtype=float) + 1e-9


def _summarise(per_period: pd.DataFrame, use_max_usable_T: bool = True) -> pd.DataFrame:
    """One row per (tag, db_row): the largest positive and most negative % difference.

    For each comparison we report where the TH spectrum is furthest ABOVE the DB (max_pos_*) and
    furthest BELOW it (max_neg_*), with the period at which it happens and the real difference
    there. max_abs_pct is the larger of the two in magnitude. msle is the mean squared log
    error over the scored periods (the default metric); sse_pct / sse_g sum the squared %
    and g errors (the earlier metric, kept for reference).

    Scored periods: those with ``usable`` True (T <= max_usable_T) if ``use_max_usable_T``,
    otherwise all compared periods. n_periods says how many were scored; a comparison with none
    left keeps its row, with NaN scores.
    """
    all_keys = per_period[["tag", "db_row"]].drop_duplicates()
    if use_max_usable_T:
        per_period = per_period.loc[per_period["usable"]]
    g = per_period.groupby(["tag", "db_row"], sort=True)
    i_pos = g["pct"].idxmax()          # row labels of the largest pct per group
    i_neg = g["pct"].idxmin()          # ... and of the smallest (most negative)
    pos = per_period.loc[i_pos.values, ["tag", "db_row", "pct", "period", "diff"]]
    neg = per_period.loc[i_neg.values, ["tag", "db_row", "pct", "period", "diff"]]
    out = pos.rename(columns={"pct": "max_pos_pct", "period": "T_max_pos", "diff": "diff_at_max_pos"})
    out = out.merge(neg.rename(columns={"pct": "max_neg_pct", "period": "T_max_neg",
                                        "diff": "diff_at_max_neg"}),
                    on=["tag", "db_row"])
    out["max_abs_pct"] = np.maximum(out["max_pos_pct"].abs(), out["max_neg_pct"].abs())
    # Sum of squared errors over the compared periods - one number for the whole spectrum's
    # misfit (where max_abs_pct only sees the worst period). Two versions:
    #   sse_pct  sum of pct^2 [%^2]: every period and every record on the same relative footing.
    #            This is the one to rank records by.
    #   sse_g    sum of diff^2 [g^2]: absolute misfit. Scales with amplitude squared, so strong
    #            records and short periods dominate it; kept for reference.
    # Both groupbys use the same (tag, db_row) sort as `out`, so .values lines up row for row.
    sq = per_period.assign(pct2=per_period["pct"] ** 2, diff2=per_period["diff"] ** 2,
                           log2=per_period["log_ratio"] ** 2)
    grp = sq.groupby(["tag", "db_row"], sort=True)
    sums = grp[["pct2", "diff2"]].sum()
    out["sse_pct"] = sums["pct2"].values
    out["sse_g"] = sums["diff2"].values
    # MSLE, the DEFAULT metric: mean over the scored periods of [ln SA_TH - ln SA_DB]^2.
    # Symmetric in over/under-prediction and scale-free, and a MEAN, so records scored over
    # different numbers of periods stay comparable. sqrt(MSLE) is the typical log error:
    # exp(sqrt(MSLE)) - 1 ~ the typical relative error (0.01 -> ~10 %, 0.05 -> ~25 %).
    out["msle"] = grp["log2"].mean().values
    out["n_periods"] = g.size().values
    # bring back any comparison that had no scored period left (NaN scores, n_periods 0)
    out = all_keys.merge(out, on=["tag", "db_row"], how="left").sort_values(["tag", "db_row"])
    out["n_periods"] = out["n_periods"].fillna(0).astype(int)
    # a tag is ambiguous if it matched more than one DB row
    out["ambiguous"] = out.groupby("tag")["db_row"].transform("size") > 1
    return out.set_index(["tag", "db_row"])


def _compare(tags, th: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    """Long-form comparison for the given tags: one row per (tag, db_row, period).

    Every DB row carrying a tag is compared against that tag's TH spectrum, so an ambiguous tag
    simply yields two sets of rows (told apart by db_row).
    """
    periods = _compared_periods(th, db)
    t_th = th.columns.to_numpy(dtype=float)
    rows = db[db["tag"].isin(set(tags))]
    missing = set(tags) - set(rows["tag"])
    if missing:
        raise KeyError(f"{len(missing)} tag(s) not found in the DB, e.g. {sorted(missing)[:5]}")

    # TH values at the DB periods, one interpolation per tag (not per DB row)
    uniq = rows["tag"].unique()
    th_at = np.vstack([interp_loglog(t_th, th.loc[t].to_numpy(dtype=float), periods) for t in uniq])
    th_at = pd.DataFrame(th_at, index=uniq, columns=periods)

    sa_th = th_at.loc[rows["tag"]].to_numpy()                 # aligned row-for-row with `rows`
    sa_db = rows[list(periods)].to_numpy(dtype=float)
    n_r, n_t = sa_db.shape
    t_usable = np.repeat(rows["max_usable_T"].to_numpy(dtype=float), n_t)
    period = np.tile(periods, n_r)
    return pd.DataFrame({
        "tag": np.repeat(rows["tag"].to_numpy(), n_t),
        "db_row": np.repeat(rows.index.to_numpy(), n_t),
        "database": np.repeat(rows["database"].to_numpy(), n_t),
        "period": period,
        "SA_TH": sa_th.ravel(),
        "SA_DB": sa_db.ravel(),
        "diff": (sa_th - sa_db).ravel(),
        "pct": (100.0 * (sa_th - sa_db) / sa_db).ravel(),
        # ln(SA_TH / SA_DB): the per-period term of the MSLE
        "log_ratio": (np.log(sa_th) - np.log(sa_db)).ravel(),
        # the DB row's maximum usable period, and whether this period lies within it
        "max_usable_T": t_usable,
        "usable": _usable_mask(period, t_usable),
    })


def compare_all(th: pd.DataFrame | None = None, db: pd.DataFrame | None = None,
                use_max_usable_T: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every TH spectrum with its DB spectrum (or spectra, if the tag is ambiguous).

    Returns
    -------
    per_period : long form, one row per (tag, db_row, period) with SA_TH, SA_DB, diff, pct,
                 max_usable_T and usable. ALL compared periods, whatever ``use_max_usable_T``.
    summary    : one row per (tag, db_row), see ``_summarise``; scored only on T <= max_usable_T
                 when ``use_max_usable_T`` (default).
    """
    th = load_th_spectra() if th is None else th
    db = load_db_spectra() if db is None else db
    per_period = _compare(th.index, th, db)
    return per_period, _summarise(per_period, use_max_usable_T)


_SSE_LABELS = {"msle": "Mean squared log error, mean of [ln SA_TH − ln SA_DB]² [-]",
               "sse_pct": "Sum of squared % differences over periods [%²]",
               "sse_g": "Sum of squared differences over periods [g²]"}


def plot_error_histogram(summary: pd.DataFrame, metric: str = "msle", n_bins: int = 60,
                         log_x: bool = True, by_database: bool = True,
                         x_max: float | None = None, ax=None,
                         save_fp: str | Path | None = None):
    """Histogram of the per-record error metric, over the UNAMBIGUOUS tags only.

    Ambiguous tags are left out because we cannot tell which of their DB rows is the right
    reference (see the module docstring); they are reported separately in the notebook.

    metric       "msle" (default), "sse_pct" (relative) or "sse_g" (absolute), see
                 ``_summarise``.
    log_x        the errors span several orders of magnitude (they are SQUARES), so on a
                 linear axis almost every record piles into the first bin. With log_x the bins
                 are equally wide in log10(error) and the whole range is readable.
    by_database  stack the bars by source database (ESM / NGA-Sub), so each database's share
                 of every bin is visible; the total bar height is still the full count.
    x_max        records with an error above this are NOT drawn - a few gross outliers (orders of
                 magnitude off) would otherwise stretch the axis until the bulk is a sliver.
                 They are not silently dropped: their number is printed on the plot.

    Returns (fig, ax). If ``save_fp`` is given the figure is saved with a white background.
    """
    if metric not in _SSE_LABELS:
        raise ValueError(f"metric must be one of {list(_SSE_LABELS)}, got {metric!r}")
    s = summary.loc[~summary["ambiguous"]]
    n_amb = summary.loc[summary["ambiguous"]].index.get_level_values("tag").nunique()
    n_total = len(s)
    # records beyond x_max are counted, then removed from what is drawn
    n_over = 0 if x_max is None else int((s[metric] > x_max).sum())
    if x_max is not None:
        s = s.loc[s[metric] <= x_max]
    vals = s[metric].to_numpy(dtype=float)
    if log_x and (vals <= 0).any():
        # log bins cannot hold an exact zero; there are none in practice, but fail loudly
        raise ValueError(f"{(vals <= 0).sum()} record(s) have {metric} <= 0; use log_x=False")

    # One set of bin edges shared by every stacked group, so the bars line up.
    edges = (np.geomspace(vals.min(), vals.max(), n_bins + 1) if log_x
             else np.linspace(0.0, vals.max(), n_bins + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    if by_database:
        # "database" is not in the summary; it is fixed per tag, and the tag prefix gives it:
        # NGA-Sub tags start with the numeric RSN, ESM tags with an event id.
        is_nga = s.index.get_level_values("tag").str.match(r"^\d+__")
        groups = [("NGA-Sub", vals[is_nga]), ("ESM", vals[~is_nga])]
        # Slots 3 and 4 (aqua, yellow), NOT 1 and 2: blue/orange already mean TH/DB spectrum
        # in plot_spectra_comparison, and a colour should not change meaning between plots.
        ax.hist([v for _, v in groups], bins=edges, stacked=True,
                label=[f"{n} (n = {len(v)})" for n, v in groups],
                color=["#1baf7a", "#eda100"], edgecolor="white", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    else:
        ax.hist(vals, bins=edges, color=_TH_COLOUR, edgecolor="white", linewidth=0.6)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(_SSE_LABELS[metric] + (" (log scale)" if log_x else ""))
    ax.set_ylabel("Number of records")
    if n_over:
        ax.text(0.99, 0.80, f"{n_over} record(s) with {metric} > {x_max:g}\nnot shown",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")
    # with the max-usable-period limit, records are scored over different numbers of periods
    n_lo, n_hi = int(s["n_periods"].min()), int(s["n_periods"].max())
    per = f"{n_hi} periods each" if n_lo == n_hi else f"{n_lo}-{n_hi} periods (T <= max usable T)"
    ax.set_title(f"TH vs DB spectra: {n_total} records ({n_amb} ambiguous tags excluded), {per}",
                 fontsize=10)
    ax.grid(True, axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    if save_fp is not None:
        fig.savefig(save_fp, dpi=200, bbox_inches="tight", facecolor="w")
    return fig, ax


def plot_sse_histogram(summary: pd.DataFrame, metric: str = "sse_pct", **kwargs):
    """The earlier name: ``plot_error_histogram`` with the % SSE as the default metric."""
    return plot_error_histogram(summary, metric=metric, **kwargs)


# ---------------------------------------------------------------------------------------------
# 5. Swapped-component check
# ---------------------------------------------------------------------------------------------
#
# Hypothesis: for some records the two horizontal components got swapped somewhere between the
# database and the downloaded file (e.g. the JSON labelled __U actually holds the V trace). If so,
# the TH spectrum of tag "..._U" should match the DB spectrum of "..._V" much better than its own.

# The other horizontal component of each label. ESM uses U/V (W = vertical, not in the DB);
# NGA-Sub uses H1/H2.
OTHER_COMPONENT = {"U": "V", "V": "U", "H1": "H2", "H2": "H1"}


def other_component_tag(tag: str) -> str:
    """The tag of the same record's other horizontal component: '..._0__U' -> '..._0__V'."""
    stem, comp = tag.rsplit("__", 1)
    return f"{stem}__{OTHER_COMPONENT[comp]}"


def _sse_pct(sa_th: np.ndarray, sa_db: np.ndarray) -> float:
    """Sum of squared % differences - the same definition as ``sse_pct`` in ``_summarise``."""
    return float(np.sum((100.0 * (sa_th - sa_db) / sa_db) ** 2))


def _msle(sa_th: np.ndarray, sa_db: np.ndarray) -> float:
    """Mean squared log error - the same definition as ``msle`` in ``_summarise``."""
    return float(np.mean((np.log(sa_th) - np.log(sa_db)) ** 2))


# the per-pair scoring functions, by the metric name used in the summary columns
_SCORERS = {"msle": _msle, "sse_pct": _sse_pct}


def _scorer(metric: str):
    if metric not in _SCORERS:
        raise ValueError(f"metric must be one of {list(_SCORERS)}, got {metric!r}")
    return _SCORERS[metric]


def swapped_component_check(summary: pd.DataFrame, th: pd.DataFrame | None = None,
                            db: pd.DataFrame | None = None, threshold: float = -np.inf,
                            database: str = "ESM",
                            threshold_max: float | None = None,
                            use_max_usable_T: bool = True,
                            metric: str = "msle") -> pd.DataFrame:
    """Does each TH spectrum fit the OTHER component's DB spectrum better than its own?

    Takes every unambiguous record of ``database`` ("ESM" or "NGASub") with
    summary[metric] > threshold (default: all of them; and, if ``threshold_max`` is given,
    <= threshold_max, to test one band at a time), and compares its TH spectrum with the DB
    spectrum of the other horizontal component of the same recording (same
    event/station/location, or same RSN).

    Both errors (own label and other component) are computed HERE with ``metric`` ("msle",
    default, or "sse_pct"), on one common set of periods, so that they are directly comparable:
    with ``use_max_usable_T`` (default) the periods up to the smaller of the two DB rows'
    max_usable_T, otherwise all compared periods. The own-label error can therefore differ
    slightly from ``summary`` when the two components' limits differ.

    Only the DB side is swapped: the other component does not need to have been downloaded, as
    its DB spectrum is all that is needed.

    Returns one row per tested tag (index = tag), with {m} = the metric name:
      {m}_original         error of TH vs its own DB spectrum
      other_component      label of the other component (U, V, H1, H2)
      {m}_other_component  error of TH vs the other component's DB spectrum. NaN if that
                           component has no DB row, or more than one (ambiguous)
      other_is_lower       True if {m}_other_component < {m}_original (False when NaN)
      n_periods            number of periods both errors were computed over
    """
    score = _scorer(metric)
    th = load_th_spectra() if th is None else th
    db = load_db_spectra() if db is None else db
    periods = _compared_periods(th, db)
    t_th = th.columns.to_numpy(dtype=float)

    # records to test: unambiguous, from the chosen database, and above the threshold
    s = summary.loc[~summary["ambiguous"] & (summary[metric] > threshold)]
    if threshold_max is not None:
        s = s.loc[s[metric] <= threshold_max]
    tags = s.index.get_level_values("tag")
    is_nga = tags.str.match(r"^\d+__")          # NGA-Sub tags start with the numeric RSN
    s = s.loc[is_nga if database == "NGASub" else ~is_nga]

    # the DB rows of all the other components in one lookup, grouped by tag
    other_tags = {tag: other_component_tag(tag) for tag, _ in s.index}
    db_other = db[db["tag"].isin(set(other_tags.values()))]
    db_other_by_tag = {t: grp for t, grp in db_other.groupby("tag")}

    cols = list(periods)
    rows = {}
    for (tag, db_row), r in s.iterrows():
        other = other_tags[tag]
        cand = db_other_by_tag.get(other)
        # the same TH interpolation as in _compare
        sa_th = interp_loglog(t_th, th.loc[tag].to_numpy(dtype=float), periods)
        own = db.loc[db_row]
        sa_own = own[cols].to_numpy(dtype=float)
        sse_own, sse_new, n = np.nan, np.nan, 0
        if cand is not None and len(cand) == 1:
            other_row = cand.iloc[0]
            # one common period set for both scores: up to the stricter of the two limits
            t_lim = min(float(own["max_usable_T"]), float(other_row["max_usable_T"]))
            m = _usable_mask(periods, t_lim, use_max_usable_T)
            n = int(m.sum())
            sse_own = score(sa_th[m], sa_own[m])
            sse_new = score(sa_th[m], other_row[cols].to_numpy(dtype=float)[m])
        else:
            # other component not scorable: still report the own-label error on its own periods
            m = _usable_mask(periods, float(own["max_usable_T"]), use_max_usable_T)
            n = int(m.sum())
            sse_own = score(sa_th[m], sa_own[m])
        rows[tag] = {f"{metric}_original": sse_own,
                     "other_component": other.rsplit("__", 1)[1],
                     f"{metric}_other_component": sse_new,
                     # NaN < x is False, so an unscorable row reads as "not lower"
                     "other_is_lower": bool(sse_new < sse_own),
                     "n_periods": n}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "tag"
    return out.sort_values(f"{metric}_original", ascending=False)


# ---------------------------------------------------------------------------------------------
# 6. Back to the source: the component order inside the ESM HDF5 files
# ---------------------------------------------------------------------------------------------
#
# The JSON converter (data_handling/convert_esm_records_to_json.py) takes the datasets of the
# station group in key-sorted order and calls the 1st one U and the 2nd one V. If that
# assumption is wrong for some files, the JSON labels come out swapped relative to the DB.
# Here the spectra are recomputed straight from the HDF5 file, so the check does not depend on
# the JSON files at all: does the 1st dataset match DB U and the 2nd DB V?

def load_esm_hdf5_horizontals(tag: str, folder: str | Path | None = None) -> dict:
    """The first two datasets (key-sorted, as the converter orders them) of an ESM HDF5 file.

    Returns {"file": Path, "channels": [key1, key2], "acc": [acc1, acc2] in g, "dt": [dt1, dt2]}.
    The file, the station group, the ordering and the cm/s² -> g conversion are all done with
    the converter's own helpers, so this reads the file exactly as the pipeline did.
    """
    import h5py
    from standes.groundmotion import parse_esm_record_identifier
    from phd_project.scripts.data_handling.convert_esm_records_to_json import (
        _find_station_group, _to_g, resolve_hdf5_path)

    stem = tag.rsplit("__", 1)[0]
    event_id, station, location = parse_esm_record_identifier(stem)
    # the tag carries location "0"; the file may be named "__00" - resolve_hdf5_path falls back
    # to a glob on event + station, and returns None if that is missing or ambiguous
    fp = resolve_hdf5_path(folder if folder is not None else ESM_HDF5_DIR,
                           event_id, station, location)
    if fp is None:
        raise FileNotFoundError(f"no unique HDF5 file for {stem}")
    with h5py.File(fp, "r") as h5:
        grp = _find_station_group(h5, station)
        keys = [k for k in sorted(grp.keys()) if isinstance(grp[k], h5py.Dataset)][:2]
        acc = [_to_g(np.asarray(grp[k][()], dtype=float).ravel(), "cm/s^2") for k in keys]
        dt = [1.0 / float(grp[k].attrs["sampling_rate"]) for k in keys]
    return {"file": fp, "channels": keys, "acc": acc, "dt": dt}


def _channel_code(key: str) -> str:
    """Short channel name from an ASDF dataset key: 'HI.ATH2.00.HN2__1969...' -> 'HN2'."""
    return key.split("__", 1)[0].split(".")[-1]


def hdf5_component_check(tags, db: pd.DataFrame | None = None,
                         folder: str | Path | None = None,
                         use_max_usable_T: bool = True,
                         metric: str = "msle") -> pd.DataFrame:
    """For each ESM tag, does the HDF5 file's 1st horizontal match DB U and the 2nd DB V?

    For every tag the recording's HDF5 file is read (once per recording, even if both its U
    and V tags are listed), the PSA of the 1st and 2nd datasets is computed at the compared DB
    periods with the TH pipeline's own routine (compute_record_spectra.psa_spectrum, 5 %), and
    each is scored against the DB U and DB V spectra of the same recording with ``metric``
    ("msle", default, or "sse_pct"). With ``use_max_usable_T`` (default) all four errors use the
    same periods: those up to the smaller of the U and V rows' max_usable_T.

    Returns one row per tag (index = tag), with {m} = the metric name:
      channel_1, channel_2   channel codes of the 1st/2nd datasets (e.g. HNE/HNN, HN2/HN3)
      {m}_1_vs_U, {m}_1_vs_V, {m}_2_vs_U, {m}_2_vs_V
      first_matches_U        {m}_1_vs_U < {m}_1_vs_V
      second_matches_V       {m}_2_vs_V < {m}_2_vs_U
      order_as_expected      both of the above (1st = U and 2nd = V, as the converter assumes)
      error                  why a row could not be scored (file missing, ...), else ""
    """
    from phd_project.scripts.compute_record_spectra import psa_spectrum

    score = _scorer(metric)
    db = load_db_spectra() if db is None else db
    periods = _compared_periods(load_th_spectra(), db)
    cols = list(periods)

    per_stem: dict[str, dict] = {}           # one HDF5 read per recording
    rows = {}
    for tag in tags:
        stem = tag.rsplit("__", 1)[0]
        row = {"channel_1": "", "channel_2": "",
               **{f"{metric}_{i}_vs_{c}": np.nan for i in (1, 2) for c in ("U", "V")},
               "error": ""}
        try:
            if stem not in per_stem:
                h = load_esm_hdf5_horizontals(tag, folder)
                if len(h["acc"]) < 2:
                    raise ValueError(f"only {len(h['acc'])} dataset(s) in {h['file'].name}")
                # the same spectrum routine and damping as the TH spectra
                h["psa"] = [psa_spectrum(a, d, periods, 0.05) for a, d in zip(h["acc"], h["dt"])]
                per_stem[stem] = h
            h = per_stem[stem]
            db_uv, t_lim = {}, np.inf
            for comp in ("U", "V"):
                r = db[db["tag"] == f"{stem}__{comp}"]
                if len(r) != 1:
                    raise ValueError(f"{len(r)} DB rows for {stem}__{comp}")
                db_uv[comp] = r[cols].to_numpy(dtype=float).ravel()
                t_lim = min(t_lim, float(r["max_usable_T"].iloc[0]))
            # one common period set for all four scores (the stricter of the U / V limits)
            m = _usable_mask(periods, t_lim, use_max_usable_T)
            row["channel_1"], row["channel_2"] = map(_channel_code, h["channels"])
            for i in (1, 2):
                for comp in ("U", "V"):
                    row[f"{metric}_{i}_vs_{comp}"] = score(h["psa"][i - 1][m], db_uv[comp][m])
        except (OSError, ValueError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows[tag] = row

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "tag"
    # NaN comparisons are False, so an unscorable row never reads as "as expected"
    out["first_matches_U"] = out[f"{metric}_1_vs_U"] < out[f"{metric}_1_vs_V"]
    out["second_matches_V"] = out[f"{metric}_2_vs_V"] < out[f"{metric}_2_vs_U"]
    out["order_as_expected"] = out["first_matches_U"] & out["second_matches_V"]
    first = ["channel_1", "channel_2", "first_matches_U", "second_matches_V", "order_as_expected"]
    return out[first + [c for c in out.columns if c not in first]]


# Where the channel inventory of the whole ESM folder is cached: scanning ~22k files on the share
# takes a while, and the result only changes if files are added to the share.
ESM_CHANNEL_INVENTORY_FP = _PROC / "07_gm_records" / "esm_hdf5_channel_inventory.csv"


def _scan_one_hdf5(fp: Path) -> dict:
    """Channel layout of one ESM HDF5 file, read the way the converter reads it.

    Only the dataset KEYS are read (no waveform data), so this is quick even over the network.
    The station group is chosen with the converter's own ``_find_station_group``, and the
    datasets are listed in the same key-sorted order the converter indexes into.
    """
    import h5py
    from phd_project.scripts.data_handling.convert_esm_records_to_json import _find_station_group

    row = {"file": fp.name, "n_station_groups": np.nan, "n_datasets": np.nan,
           "channel_1": "", "channel_2": "", "channel_3": "", "all_channels": "", "error": ""}
    try:
        # file name is {event_id}__{station}__{location}.h5
        station = fp.stem.split("__")[1]
        with h5py.File(fp, "r") as h5:
            row["n_station_groups"] = len(h5["Waveforms"].keys()) if "Waveforms" in h5 else 0
            grp = _find_station_group(h5, station)
            keys = [k for k in sorted(grp.keys()) if isinstance(grp[k], h5py.Dataset)]
        codes = [_channel_code(k) for k in keys]
        row["n_datasets"] = len(codes)
        for i, c in enumerate(codes[:3], start=1):
            row[f"channel_{i}"] = c
        row["all_channels"] = "/".join(codes)
    except Exception as exc:                          # noqa: BLE001 - reported, not raised
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def scan_esm_hdf5_channels(folder: str | Path | None = None, n_workers: int = 16,
                           cache_fp: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Channel codes of every ESM HDF5 file in ``folder``, in the converter's (key-sorted) order.

    One row per file: file, n_station_groups, n_datasets, channel_1..3 (the codes the converter
    calls U, V, W), all_channels ("HNE/HNN/HNZ"), error. Files are read in parallel worker
    PROCESSES: h5py holds a global lock, so threads would open the files one at a time. (On
    Windows, call this from a notebook or under ``if __name__ == "__main__":``.) The result is
    cached as a CSV at
    ``cache_fp`` (default ESM_CHANNEL_INVENTORY_FP) and read back from there unless
    ``refresh=True`` or the folder now holds different files.
    """
    from concurrent.futures import ProcessPoolExecutor

    folder = Path(folder) if folder is not None else Path(ESM_HDF5_DIR)
    cache_fp = Path(cache_fp) if cache_fp is not None else ESM_CHANNEL_INVENTORY_FP
    files = sorted(folder.glob("*.h5"))

    if cache_fp.is_file() and not refresh:
        cached = pd.read_csv(cache_fp, dtype=str, keep_default_na=False)
        if set(cached["file"]) == {f.name for f in files}:
            for c in ("n_station_groups", "n_datasets"):
                cached[c] = pd.to_numeric(cached[c], errors="coerce")
            return cached
        print(f"cache {cache_fp.name} is out of date - rescanning")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # chunks of files per task, so the per-task process overhead stays small
        rows = list(pool.map(_scan_one_hdf5, files, chunksize=50))
    out = pd.DataFrame(rows)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_fp, index=False)
    return out


# What the converter ASSUMES the key-sorted order to be (see its docstring): E, N, Z -> U, V, W.
# A layout is "handled" only if its 1st/2nd channels end in E and N; the converter has no
# explicit handling of channel names at all - it only ever uses the sorted position.
_CONVERTER_ASSUMED = ("E", "N")


def summarise_channel_layouts(inventory: pd.DataFrame, summary: pd.DataFrame | None = None,
                              th: pd.DataFrame | None = None,
                              db: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (channel_1, channel_2) layout found in the inventory.

    n_files                 files with this layout on the share
    orientation_codes       last letters of channels 1/2, e.g. "E/N", "2/3"
    matches_converter_assumption   True if they are E then N, the order the converter assumes

    If ``summary`` (from compare_all) is given, each layout is also checked against what was
    actually downloaded: n_downloaded_tags (unambiguous ESM tags whose HDF5 file has this
    layout) and n_other_component_fits_better (of those, how many fit the OTHER component's
    DB spectrum better, i.e. look swapped, using swapped_component_check with no threshold).
    """
    inv = inventory.loc[inventory["error"] == ""].copy()
    inv["orientation_codes"] = inv["channel_1"].str[-1] + "/" + inv["channel_2"].str[-1]
    out = (inv.groupby(["channel_1", "channel_2", "orientation_codes"]).size()
              .rename("n_files").reset_index())
    out["matches_converter_assumption"] = out["orientation_codes"] == "/".join(_CONVERTER_ASSUMED)

    if summary is not None:
        # every unambiguous ESM tag, swap-tested with no threshold: other_is_lower = "looks swapped"
        sw = swapped_component_check(summary, th, db, threshold=-np.inf, database="ESM")
        # map each tag to its HDF5 file name via the stem: {event}_{sta}_{loc} -> {event}__{sta}__
        stems = sw.index.str.rsplit("__", n=1).str[0]
        ev_sta = [f"{e}__{s}__" for e, s, _ in (st.rsplit("_", 2) for st in stems)]
        inv["ev_sta"] = inv["file"].str.rsplit("__", n=1).str[0] + "__"
        # a (event, station) with several location files would be ambiguous: keep unique ones
        uniq = inv.drop_duplicates("ev_sta", keep=False).set_index("ev_sta")
        sw["channel_1"] = uniq["channel_1"].reindex(ev_sta).to_numpy()
        sw["channel_2"] = uniq["channel_2"].reindex(ev_sta).to_numpy()
        agg = (sw.dropna(subset=["channel_1"])
                 .groupby(["channel_1", "channel_2"])["other_is_lower"]
                 .agg(n_downloaded_tags="size", n_other_component_fits_better="sum")
                 .reset_index())
        out = out.merge(agg, on=["channel_1", "channel_2"], how="left")
        out[["n_downloaded_tags", "n_other_component_fits_better"]] = (
            out[["n_downloaded_tags", "n_other_component_fits_better"]].fillna(0).astype(int))
    return out.sort_values("n_files", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# 7. Where were given records used in the site-specific MSA?
# ---------------------------------------------------------------------------------------------
#
# Two sources say which record sat in which slot of which stripe:
#   * the per-structure collapse-flag JSONs written by nb 060 from the MSA stripe logs
#     ({n}s_cbf_dc2_site{i}_msa_collapseflags_{im}.json: {stripe_iml: {"record_names":
#     {slot: file}}}). These record what was actually RUN, per structure.
#   * the GCIM selection pickles (site_{i}__stripe_iml_{tag}__gm_selection.pickle), whose "recs"
#     table lists the 30 selected records in slot order. These record what was SELECTED, per
#     site and stripe (both structures at a site share them).
# The usage is read from the first and checked against the second.

_FLAGS_NAME_RE = r"^(?P<ns>\d+)s_cbf_dc2_site(?P<site>\d+)_msa_collapseflags_(?P<im>.+)\.json$"


def msa_record_usage(record_tags, im: str = "AvgSA_03", flags_dir: str | Path | None = None,
                     selection_dir: str | Path | None = None,
                     verify: bool = True) -> pd.DataFrame:
    """Every (site, structure, stripe, record slot) at which one of ``record_tags`` was run.

    record_tags    record tags (JSON stems), e.g. "GR-1999-0001_ATH2_0__V"
    im             intensity measure of the MSA (file-name suffix of the collapse-flag JSONs)
    flags_dir      root of the per-site collapse-flag folders (default: the config's
                   wp1_sites_fragility_curves)
    selection_dir  folder of the GCIM stripe pickles (default: the config's
                   AvgSA_03_record_selection; only used when ``verify``)
    verify         check every hit against the stripe pickle: the record in slot k of the
                   pickle's "recs" table must be the same file as the MSA ran in slot k.
                   Raises if any hit disagrees or its pickle is missing.

    Returns a DataFrame, one row per use, sorted:
      site          int
      structure     "3s" / "5s" (the storey folder in the analysis paths)
      stripe_iml    the stripe IML as the file-name token, e.g. "00pt270" (gm_selection's
                    iml_filename_tag, the same token as in the stripe pickle names)
      record_index  slot 0..29 of the record within that stripe
      record_tag    the record
    """
    import json
    import pickle
    import re
    from phd_project.scripts.WP1_ground_motion_set.gm_selection import (
        iml_filename_tag, stripe_pickle_path)

    flags_dir = Path(flags_dir) if flags_dir is not None else Path(SITE_FRAGILITY_DIR)
    wanted = {f"{t}.json" for t in record_tags}      # the MSA logs name records by file

    rows = []
    for fp in sorted(flags_dir.glob(f"site_*/*_msa_collapseflags_{im}.json")):
        m = re.match(_FLAGS_NAME_RE, fp.name)
        if m is None or m["im"] != im:
            continue
        site, structure = int(m["site"]), f"{m['ns']}s"
        with open(fp, "r") as f:
            stripes = json.load(f)
        for iml, s in stripes.items():
            for slot, name in s["record_names"].items():
                if name in wanted:
                    rows.append({"site": site, "structure": structure,
                                 "stripe_iml": iml_filename_tag(float(iml)),
                                 "record_index": int(slot), "record_tag": name[:-len(".json")],
                                 "_iml": float(iml)})
    out = pd.DataFrame(rows, columns=["site", "structure", "stripe_iml", "record_index",
                                      "record_tag", "_iml"])

    if verify and len(out):
        sel_dir = Path(selection_dir) if selection_dir is not None else Path(AVGSA03_SELECTION_DIR)
        bad = []
        # one pickle read per (site, stripe), shared by both structures and all hits in it
        for (site, iml), grp in out.groupby(["site", "_iml"]):
            pfp = stripe_pickle_path(sel_dir, site, iml)
            if not pfp.is_file():
                bad.append(f"missing {pfp.name}")
                continue
            with open(pfp, "rb") as f:
                files = pickle.load(f)["recs"][("metadata", "filename")].tolist()
            for _, r in grp.iterrows():
                k = r["record_index"]
                if k >= len(files) or files[k] != f"{r['record_tag']}.json":
                    bad.append(f"{pfp.name} slot {k}: pickle has "
                               f"{files[k] if k < len(files) else '-'}, MSA ran {r['record_tag']}")
        if bad:
            raise ValueError(f"{len(bad)} usage(s) disagree with the stripe pickles:\n  "
                             + "\n  ".join(bad[:20]))

    return (out.sort_values(["site", "structure", "_iml", "record_index"])
               .drop(columns="_iml").reset_index(drop=True))


# ---------------------------------------------------------------------------------------------
# 8. Single record
# ---------------------------------------------------------------------------------------------

def max_signed_differences(tag: str, th: pd.DataFrame | None = None,
                           db: pd.DataFrame | None = None,
                           use_max_usable_T: bool = True) -> pd.DataFrame:
    """The largest positive and most negative TH-vs-DB difference of one record.

    One row per DB row carrying the tag (normally one; two for an ambiguous tag), indexed by
    db_row. Columns: max_pos_pct, T_max_pos, diff_at_max_pos, max_neg_pct, T_max_neg,
    diff_at_max_neg, max_abs_pct, sse_pct, sse_g, msle, n_periods, ambiguous. Percentages are
    relative to the DB. Scored on T <= max_usable_T if ``use_max_usable_T`` (default).
    """
    th = load_th_spectra() if th is None else th
    db = load_db_spectra() if db is None else db
    if tag not in th.index:
        raise KeyError(f"{tag!r} has no TH spectrum")
    return _summarise(_compare([tag], th, db), use_max_usable_T).loc[tag]


def plot_spectra_comparison(tag: str, th: pd.DataFrame | None = None,
                            db: pd.DataFrame | None = None, ax=None,
                            save_fp: str | Path | None = None, scale: str = "loglog",
                            use_max_usable_T: bool = True):
    """Plot the TH and DB spectra of one record, with its max signed differences.

    ``use_max_usable_T`` (default True): the text-box numbers use only T <= max_usable_T, and
    the periods beyond it are shaded grey (they are still drawn, just not scored).

    ``scale`` = "loglog" (default) puts both axes on a log scale, which spreads the short
    periods out and shows relative differences at every amplitude equally; "linear" uses
    ordinary axes starting at zero, which is how a spectrum is usually drawn in design and makes
    the peak and the absolute size of differences easier to judge.

    The fine TH spectrum is a line; the DB ordinates are filled markers (one series per DB row
    if the tag is ambiguous); the TH values interpolated at the DB periods - the points that
    are actually compared - are small open markers on the TH line. The legend sits in the top
    left, and the max +/- % differences (from ``max_signed_differences``, so the numbers on the
    plot are the same ones the function returns) are printed directly under it.

    Returns (fig, ax). If ``save_fp`` is given the figure is saved with a white background.
    """
    if scale not in ("loglog", "linear"):
        raise ValueError(f"scale must be 'loglog' or 'linear', got {scale!r}")
    th = load_th_spectra() if th is None else th
    db = load_db_spectra() if db is None else db
    stats = max_signed_differences(tag, th, db, use_max_usable_T)
    periods = _compared_periods(th, db)
    t_th = th.columns.to_numpy(dtype=float)
    sa_th = th.loc[tag].to_numpy(dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(t_th, sa_th, color=_TH_COLOUR, lw=2, label="TH spectrum (from time history)", zorder=2)
    ax.plot(periods, interp_loglog(t_th, sa_th, periods), ls="none", marker="o", ms=4,
            mfc="white", mec=_TH_COLOUR, mew=1, label="TH at DB periods", zorder=3)

    ambiguous = len(stats) > 1
    for k, db_row in enumerate(stats.index):
        # DB markers get a white ring (mec) so they stay readable where they sit on the TH line.
        # A second DB row (ambiguous tag) is a smaller diamond, so that where the two DB spectra
        # coincide the first one still shows around it instead of being hidden.
        lbl = "DB spectrum" + (f" (CSV row {db_row})" if ambiguous else "")
        ax.plot(periods, db.loc[db_row, list(periods)].to_numpy(dtype=float), ls="none",
                marker="o" if k == 0 else "D", ms=7 if k == 0 else 4.5,
                color=_DB_COLOURS[k % len(_DB_COLOURS)], mec="white", mew=1 if k == 0 else 0.5,
                label=lbl, zorder=4 + k)

    lines = []
    for db_row, r in stats.iterrows():
        prefix = f"row {db_row}: " if ambiguous else ""
        lines += _stats_lines(r, prefix)
    # shade from the smallest limit among the plotted DB rows (normally there is just one)
    t_usable = float(db.loc[stats.index, "max_usable_T"].min()) if use_max_usable_T else None
    _finish_spectrum_axes(fig, ax, scale == "loglog",
                          title=tag + ("   [AMBIGUOUS: 2 DB rows]" if ambiguous else ""),
                          text_lines=lines, n_series=len(stats), save_fp=save_fp,
                          t_usable=t_usable)
    return fig, ax


def _stats_lines(r, prefix: str = "") -> list[str]:
    """The three text-box lines for one TH-vs-DB comparison: max +, max -, MSLE.

    The MSLE (3 significant figures; it spans many decades) is followed by the typical relative
    error it corresponds to, exp(sqrt(MSLE)) - 1, which is easier to read.
    """
    typical = 100.0 * (np.exp(np.sqrt(r["msle"])) - 1.0)
    return [f"{prefix}max +: {r['max_pos_pct']:+.2f} % at T = {r['T_max_pos']:g} s",
            f"{prefix}max −: {r['max_neg_pct']:+.2f} % at T = {r['T_max_neg']:g} s",
            f"{prefix}MSLE : {r['msle']:.3g} (typ. {typical:.1f} %)"]


def _finish_spectrum_axes(fig, ax, log: bool, title: str, text_lines: list[str],
                          n_series: int, save_fp=None, legend_loc: str = "upper left",
                          t_usable: float | None = None) -> None:
    """Shared styling of the spectrum plots: scales, labels, headroom, legend, text box, save.

    Kept in one place so that every spectrum plot in this module looks and reads the same.
    ``legend_loc`` is "upper left" or "upper right"; the text box always hangs directly under
    the legend, aligned to the same side. If ``t_usable`` is given and lies inside the plotted
    period range, the periods beyond it (not scored) are shaded grey, with a legend entry.
    """
    if legend_loc not in ("upper left", "upper right"):
        raise ValueError(f"legend_loc must be 'upper left' or 'upper right', got {legend_loc!r}")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        # linear axes start at zero, so ordinates are read against a true origin
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Period T [s]")
    ax.set_ylabel("PSA (5 % damping) [g]")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", color="0.9", lw=0.8)
    ax.grid(True, which="minor", color="0.95", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Headroom: the legend + text box occupy the top-left corner, and a spectrum usually peaks
    # at short periods, i.e. exactly there. Raising the top of the y-axis pushes the data into
    # the lower part of the axes so nothing is hidden. The extension is a fraction of the
    # current span - measured in decades on a log axis, in g on a linear one; with two DB
    # series there are twice the legend entries and text lines, so it gets more room.
    lo, hi = ax.get_ylim()
    frac = 0.95 if n_series > 1 else 0.6
    ax.set_ylim(lo, hi * (hi / lo) ** frac if log else hi + frac * (hi - lo))

    # Grey band over the periods beyond the DB's maximum usable period: drawn, but not scored.
    # The x-limits are read first and set again afterwards, so the band cannot widen the axis.
    x_lo, x_hi = ax.get_xlim()
    if t_usable is not None and t_usable < x_hi:
        ax.axvspan(t_usable, x_hi, color="0.92", zorder=0, lw=0,
                   label=f"T > max usable T ({t_usable:g} s): not scored")
        ax.set_xlim(x_lo, x_hi)

    leg = ax.legend(loc=legend_loc, fontsize=8, frameon=True, framealpha=1.0)

    # --- text box directly under the legend ---------------------------------------------------
    text = "TH vs DB (rel. to DB)\n" + "\n".join(text_lines)
    # The legend's position is only known after a draw; take its bottom corner (left or right,
    # matching the legend's side) in axes coordinates and hang the text box just below it.
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
    right = legend_loc == "upper right"
    ax.text(bbox.x1 if right else bbox.x0, bbox.y0 - 0.02, text, transform=ax.transAxes,
            ha="right" if right else "left", va="top",
            fontsize=8, family="monospace", zorder=6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"))

    fig.tight_layout()
    if save_fp is not None:
        fig.savefig(save_fp, dpi=200, bbox_inches="tight", facecolor="w")


# Fixed colour per component label, so e.g. "DB V" is the same colour in every plot whichever
# component the TH file is labelled as (colour follows the entity, not its position).
_COMPONENT_COLOURS = {"U": _DB_COLOURS[0], "H1": _DB_COLOURS[0],
                      "V": _DB_COLOURS[1], "H2": _DB_COLOURS[1]}


def plot_th_vs_db_components(tag: str, components="both", th: pd.DataFrame | None = None,
                             db: pd.DataFrame | None = None, ax=None,
                             save_fp: str | Path | None = None, scale: str = "loglog",
                             legend_loc: str = "upper left", use_max_usable_T: bool = True):
    """Plot one TH spectrum against the DB spectrum of either or both horizontal components.

    Made for the swapped-component question: does the TH file labelled "__U" look like the
    database's U spectrum, or its V spectrum?

    tag          TH record to plot, e.g. "GR-1999-0001_ATH2_0__V" (or an NGA-Sub "..__H1")
    components   "both" (default) = the tag's own component and the other one;
                 or a single label ("U", "V", "H1", "H2"), or a list of labels.
                 The DB spectra are those of the SAME recording (same stem) with that label.
    scale        "loglog" (default) or "linear", as in ``plot_spectra_comparison``.
    legend_loc   "upper left" (default) or "upper right"; the text box follows the legend.
    use_max_usable_T   (default True) score every plotted component on the SAME periods, up to
                 the smallest max_usable_T among them (as swapped_component_check does), so the
                 MSLEs in the text box can be compared directly; the rest is shaded grey.

    Each DB component gets a fixed colour (U/H1 orange, V/H2 aqua), and its legend entry says
    whether it is the TH file's own label or the other component. The text box gives max +/-
    % and MSLE of the TH spectrum against each plotted DB component, with the same definitions
    as everywhere else in this module. A component with two DB rows (ambiguous) is plotted
    once per row.

    Returns (fig, ax). If ``save_fp`` is given the figure is saved with a white background.
    """
    if scale not in ("loglog", "linear"):
        raise ValueError(f"scale must be 'loglog' or 'linear', got {scale!r}")
    th = load_th_spectra() if th is None else th
    db = load_db_spectra() if db is None else db
    if tag not in th.index:
        raise KeyError(f"{tag!r} has no TH spectrum")

    stem, own = tag.rsplit("__", 1)
    if isinstance(components, str):
        components = [own, OTHER_COMPONENT[own]] if components == "both" else [components]
    bad = [c for c in components if c not in OTHER_COMPONENT]
    if bad:
        raise ValueError(f"unknown component(s) {bad}; use {sorted(OTHER_COMPONENT)} or 'both'")

    periods = _compared_periods(th, db)
    t_th = th.columns.to_numpy(dtype=float)
    sa_th = th.loc[tag].to_numpy(dtype=float)
    sa_th_at = interp_loglog(t_th, sa_th, periods)     # the TH points that are compared

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(t_th, sa_th, color=_TH_COLOUR, lw=2, label=f"TH spectrum (file labelled {own})",
            zorder=2)
    ax.plot(periods, sa_th_at, ls="none", marker="o", ms=4, mfc="white", mec=_TH_COLOUR,
            mew=1, label="TH at DB periods", zorder=3)

    # all DB rows to plot, looked up first so the common scoring limit is known before scoring
    comp_rows = []
    for comp in components:
        rows = db[db["tag"] == f"{stem}__{comp}"]
        if rows.empty:
            raise KeyError(f"no DB row for component {comp} of {stem}")
        comp_rows.append((comp, rows))
    t_usable = (min(float(r["max_usable_T"].min()) for _, r in comp_rows)
                if use_max_usable_T else None)
    m = _usable_mask(periods, t_usable if t_usable is not None else np.inf, use_max_usable_T)
    p_m = periods[m]

    lines, k = [], 0
    for comp, rows in comp_rows:
        role = "own label" if comp == own else "other component"
        for db_row, row in rows.iterrows():
            sa_db = row[list(periods)].to_numpy(dtype=float)
            name = f"DB {comp}" + (f" row {db_row}" if len(rows) > 1 else "")
            # Same marker logic as plot_spectra_comparison: the 2nd series is a smaller
            # diamond, so where two DB spectra coincide the first still shows around it.
            ax.plot(periods, sa_db, ls="none", marker="o" if k == 0 else "D",
                    ms=7 if k == 0 else 4.5, color=_COMPONENT_COLOURS[comp], mec="white",
                    mew=1 if k == 0 else 0.5, label=f"{name} ({role})", zorder=4 + k)
            # the statistics of this pairing, with the module's own definitions, on the
            # scored periods only
            pct = 100.0 * (sa_th_at[m] - sa_db[m]) / sa_db[m]
            i_pos, i_neg = np.argmax(pct), np.argmin(pct)
            r = {"max_pos_pct": pct[i_pos], "T_max_pos": p_m[i_pos],
                 "max_neg_pct": pct[i_neg], "T_max_neg": p_m[i_neg],
                 "msle": _msle(sa_th_at[m], sa_db[m])}
            lines += _stats_lines(r, prefix=f"vs {name}: ")
            k += 1

    _finish_spectrum_axes(fig, ax, scale == "loglog", title=tag, text_lines=lines,
                          n_series=k, save_fp=save_fp, legend_loc=legend_loc,
                          t_usable=t_usable)
    return fig, ax
