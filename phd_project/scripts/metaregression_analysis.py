import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections.abc import Sequence
from scipy.special import psi
from scipy.stats import kurtosis, skew

from standes.fitting import lognorm_mle_fit, lognorm_moment_fit

from phd_project.scripts.femap695_records import record_tag_to_column

# A resampled MSA fit is treated as degenerate once it exceeds this multiple of the
# fit it was drawn from. The affected replicates sit many orders of magnitude away
# (theta ~ 1e21), so anything from roughly 5 to 50 selects the same ones.
_MAX_FIT_RATIO = 10.0

# One colour per arm, fixed here so every figure in the chapter reads the same way. The
# two FEMAP695 arms are warm, the site-specific arm cool, because the fixed-vs-site-
# specific record set is the distinction the comparison is actually about.
_ARM_COLORS = {"site_msa": "b", "msa_femap695": "tab:orange", "ida_femap695": "r"}
_ARM_LABELS = {"site_msa": "MSA site-specific",
               "msa_femap695": "MSA FEMA P695",
               "ida_femap695": "IDA FEMA P695"}

# Efron & Tibshirani (1993) p.128: a bias below a quarter of the estimator's own standard
# error costs less than 3.1% in RMSE and can be left uncorrected. The second threshold
# asks the different question of whether the bootstrap has resolved the bias from zero
# at all, i.e. whether it is larger than its own Monte-Carlo error.
_BIAS_SE_THRESHOLD = 0.25
_BIAS_MCSE_THRESHOLD = 2.0

# =============================================================================
# Loading
# =============================================================================

def structure_tag(site_idx: int, n_storeys: int) -> str:
    """Return the canonical identifier for a case-study structure."""
    return f"{n_storeys}s_cbf_dc2_site{site_idx}"


def load_site_fragility_curves(
    frag_root: Path,
    sites: Sequence[int],
    n_storeys: Sequence[int],
) -> tuple[dict, dict]:
    """Load the MSA and IDA collapse fragilities for every site and storey count.

    Returns ``(fragility_curves, fragility_curves_flat)``. The first is nested
    ``[site][tag][arm]``, the second is flat ``[arm][tag]``. A structure is skipped
    entirely unless *both* arms are present, so the two arms are always paired.
    """
    fragility_curves = {}
    fragility_curves_flat = {"msa": {}, "ida-femap695": {}}

    for site in sites:
        site_fcs = {}
        for n in n_storeys:
            tag = structure_tag(site, n)
            structure_fcs = {}

            for arm, suffix in [("msa", "msa"), ("ida-femap695", "ida_femap695")]:
                fc_path = (frag_root / f"site_{site}"
                           / f"{tag}_{suffix}_collapsefragility_AvgSA_03.json")
                try:
                    with open(fc_path, "r") as file:
                        fc = json.load(file)
                        structure_fcs[arm] = {k: np.array(v) if isinstance(v, list) else v
                                                for k, v in fc.items()}
                except FileNotFoundError:
                    print(f"No Fragility Curve for {arm}: site {site} and {n}s. "
                          "Skipping...")
            
            site_fcs[tag] = structure_fcs
            for arm in fragility_curves_flat:
                try:
                    fragility_curves_flat[arm][tag] = structure_fcs[arm]
                except KeyError:
                    pass

        fragility_curves[site] = site_fcs

    return fragility_curves, fragility_curves_flat


# =============================================================================
# Non-parametric bootstrap (fixed record set)
# =============================================================================
# The FEMA P695 far-field set is fixed, so both arms resample the *actual* per-record
# results - the IDA its collapse capacities (nb 053), the MSA its per-record collapse
# flags (nb 061) - rather than drawing from a fitted lognormal. One record sample is
# drawn once and reused by every structure in both arms, so replicate k means the same
# 22 records everywhere: that is what "the same record set at every site" means, and it
# is what makes the two clouds comparable replicate-for-replicate. The record tags being
# resampled are the ones defined in :mod:`phd_project.scripts.femap695_records`.

def draw_record_samples(
    record_tags: Sequence[str],
    k_samples: int,
    seed: int = 1,
) -> np.ndarray:
    """Draw ``k_samples`` bootstrap resamples of the record set, with replacement.

    Returns a ``(k_samples, n_records)`` array of record stems. Drawn once and applied to
    every structure, so each replicate represents one alternative record set seen by the
    whole study.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray(record_tags), size=(k_samples, len(record_tags)))


def load_group_collapse_imls(
    path: Path | str,
    n_storeys: Sequence[int],
    record_tags: Sequence[str],
) -> dict[int, pd.DataFrame]:
    """Load the group x record collapse capacities (in g), split by storey count.

    Columns are reindexed into record order, so column ``j`` is ``record_tags[j]`` rather
    than the string-sorted order the CSV is written in. A missing or non-positive capacity
    raises: the moment fit takes logs, so a NaN would silently poison every replicate of
    that group. This is what surfaces a storey count whose IDAs are still running.
    """
    columns = [str(i) for i in range(len(record_tags))]
    df = pd.read_csv(path, index_col=0, dtype={c: float for c in columns})

    tables = {}
    for n in n_storeys:
        table = df.loc[df.index.str.startswith(f"group_{n}s_"), columns]
        if table.empty:
            raise ValueError(f"no group_{n}s_* rows in {path}")
        bad = table.index[~(table > 0).all(axis=1)]
        if len(bad):
            raise ValueError(f"missing or non-positive collapse IMLs for {list(bad)} - "
                             f"their IDAs are incomplete")
        tables[n] = table

    return tables


def load_group_site_map(
    summary_csv: Path | str,
    n_storeys: Sequence[int],
) -> dict[int, dict[str, list[int]]]:
    """Read the group -> member-site lists out of the group fragility summary.

    The summary carries a two-row header (``IM`` / ``parameter``); the site lists live in
    ``("info", "sites")`` as space-separated site numbers.
    """
    df = pd.read_csv(summary_csv, header=[0, 1], index_col=0)
    sites = df[("info", "sites")].dropna()

    group_sites = {}
    for n in n_storeys:
        rows = sites[sites.index.str.startswith(f"group_{n}s_")]
        group_sites[n] = {g: [int(s) for s in str(v).split()] for g, v in rows.items()}

    return group_sites


def sample_collapse_imls(
    imls: pd.Series,
    samples: np.ndarray,
    tag_columns: dict[str, str],
) -> np.ndarray:
    """Apply the record resamples to one group's collapse capacities.

    ``imls`` is a row of the table from :func:`load_group_collapse_imls`, ``samples`` the
    ``(k, n_records)`` array of record stems from :func:`draw_record_samples`. Returns the
    same shape, filled with capacities.
    """
    positions = {col: i for i, col in enumerate(imls.index)}
    idx = np.vectorize(lambda tag: positions[tag_columns[tag]])(samples)
    return imls.to_numpy()[idx]


def bootstrap_ida_group_fragilities(
    collapse_imls: pd.DataFrame,
    samples: np.ndarray,
    record_tags: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit a collapse fragility to every record resample, for every design group.

    The fit is :func:`standes.fitting.lognorm_moment_fit` - exactly what
    :func:`standes.fragility_curves.fragility_from_ida` performs on these capacities, which
    are already in g. Returns ``(theta, beta)``, both ``k_samples x n_groups`` with the
    replicate number as the index and the group ids as the columns.
    """
    tag_columns = record_tag_to_column(record_tags)

    thetas, betas = {}, {}
    for group, row in collapse_imls.iterrows():
        replicates = sample_collapse_imls(row, samples, tag_columns)
        fits = [lognorm_moment_fit(r) for r in replicates]
        thetas[group], betas[group] = (np.array(f) for f in zip(*fits))

    theta = pd.DataFrame(thetas)
    beta = pd.DataFrame(betas)
    for df in (theta, beta):
        df.index.name = "k"
        df.columns.name = "group"

    return theta, beta


def load_group_collapse_flags(
    path: Path | str,
    n_storeys: Sequence[int],
    record_tags: Sequence[str],
) -> dict[int, pd.DataFrame]:
    """Load the (group, stripe) x record collapse flags, split by storey count.

    The MSA runs the same fixed record set at every stripe, so a group's results are a
    0/1 flag per (stripe, record). Each frame comes back with a ``(group, stripe_iml)``
    MultiIndex sorted by intensity and the record columns in record order, so column
    ``j`` is ``record_tags[j]`` rather than the string-sorted order the CSV is written
    in. A missing flag, or one that is not 0/1, raises - the collapse counts are sums
    over these, so a NaN would silently corrupt every replicate of that group.
    """
    columns = [str(i) for i in range(len(record_tags))]
    df = pd.read_csv(path, dtype={c: float for c in columns})

    tables = {}
    for n in n_storeys:
        table = df[df["group"].str.startswith(f"group_{n}s_")]
        if table.empty:
            raise ValueError(f"no group_{n}s_* rows in {path}")
        if not table[columns].isin([0.0, 1.0]).all(axis=None):
            raise ValueError(f"missing or non-binary collapse flags for {n}s in {path}")
        tables[n] = (table.set_index(["group", "stripe_iml"])[columns]
                          .astype(int).sort_index())

    return tables


def msa_counts_degenerate(counts: np.ndarray) -> np.ndarray:
    """Flag resampled stripe profiles whose likelihood has no interior maximum.

    ``counts`` is ``(k_samples, n_stripes)``. A profile is degenerate when it carries no
    information about *where* the fragility sits: every stripe zero, every stripe full,
    or every stripe the same fraction. The binomial likelihood is then maximised by
    pushing ``theta`` and ``beta`` off together, and the "fit" degenerates into a
    horizontal line through the stripe range - positive and finite, so
    :func:`msa_fit_ok` alone would not catch it.

    A merely **non-monotonic** profile is *not* degenerate. Collapse fractions that dip
    between stripes are ordinary sampling noise, the likelihood still has an interior
    maximum, and those replicates must be fitted rather than screened out.
    """
    counts = np.atleast_2d(np.asarray(counts))
    return (counts == counts[:, [0]]).all(axis=1)


def bootstrap_msa_group_fragilities(
    collapse_flags: pd.DataFrame,
    samples: np.ndarray,
    record_tags: Sequence[str],
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit a collapse fragility to every record resample, for every design group.

    For each replicate the group's collapse count at a stripe is the number of *sampled*
    records that collapsed there, so one record resample moves every stripe coherently.
    The fit is :func:`standes.fitting.lognorm_mle_fit` - what
    :func:`standes.fragility_curves.fragility_from_msa` performs.

    ``reference`` is the published group fit (indexed by group, with ``median`` and
    ``dispersion`` columns), used as the runaway guard in :func:`msa_fit_ok`. A replicate
    is usable when its counts are not :func:`msa_counts_degenerate` and its fit passes
    ``msa_fit_ok``; the rest come back as ``NaN`` rather than being dropped, so row ``k``
    still means the same record sample in every column and in the IDA arm.

    Returns ``(theta, beta, fit_ok)``, all ``k_samples x n_groups`` with the replicate
    number as the index and the group ids as the columns.
    """
    tag_columns = record_tag_to_column(record_tags)
    positions = {col: i for i, col in enumerate(collapse_flags.columns)}
    idx = np.vectorize(lambda tag: positions[tag_columns[tag]])(samples)

    n_records = len(record_tags)
    thetas, betas, oks = {}, {}, {}
    for group, stripes in collapse_flags.groupby(level="group"):
        imls = stripes.index.get_level_values("stripe_iml").to_numpy(float)
        # (n_stripes, n_records) -> (k_samples, n_stripes) collapse counts
        counts = stripes.to_numpy()[:, idx].sum(axis=2).T

        fits = np.array([lognorm_mle_fit(imls, c, n_records) for c in counts])
        theta, beta = fits[:, 0], fits[:, 1]

        ok = ~msa_counts_degenerate(counts) & msa_fit_ok(
            theta, beta, reference.loc[group, "median"],
            reference.loc[group, "dispersion"])

        thetas[group] = np.where(ok, theta, np.nan)
        betas[group] = np.where(ok, beta, np.nan)
        oks[group] = ok

    theta, beta, fit_ok = (pd.DataFrame(d) for d in (thetas, betas, oks))
    for df in (theta, beta, fit_ok):
        df.index.name = "k"
        df.columns.name = "group"

    return theta, beta, fit_ok


def msa_fit_ok(
    thetas: np.ndarray,
    betas: np.ndarray,
    theta_ref: float | None = None,
    beta_ref: float | None = None,
    max_ratio: float = _MAX_FIT_RATIO,
) -> np.ndarray:
    """Return a boolean mask flagging the usable binomial-MLE fits.

    Both parameters must be positive and finite, and - given the reference fit the
    replicates were drawn around - a fit that has run away from it by more than
    ``max_ratio`` is rejected too. That catches the degenerate optimisations
    :func:`msa_counts_degenerate` does not: the runaway values are astronomically large
    (``theta ~ 1e21``) yet positive and finite, so they would survive the finiteness
    check and destroy every moment taken over the cloud.
    """
    thetas = np.asarray(thetas, dtype=float)
    betas = np.asarray(betas, dtype=float)

    ok = (thetas > 0) & (betas > 0) & np.isfinite(thetas) & np.isfinite(betas)
    if theta_ref is not None:
        ok &= thetas < max_ratio * theta_ref
    if beta_ref is not None:
        ok &= betas < max_ratio * beta_ref
    return ok


def expand_groups_to_sites(
    df: pd.DataFrame,
    group_sites: dict[str, list[int]],
) -> pd.DataFrame:
    """Fan a group-indexed frame out to one column per site.

    Sites sharing a design group share a single IDA, so their columns are identical. The
    duplication is what lets the IDA arm be lined up column-by-column with the per-site MSA
    arm. Columns come out as ``int`` site numbers, sorted ascending.
    """
    columns = sorted((site, group) for group, sites in group_sites.items()
                     for site in sites)
    out = df[[group for _, group in columns]]
    out.columns = pd.Index([site for site, _ in columns], name="site")
    return out


# =============================================================================
# Non-parametric bootstrap (site-specific MSA)
# =============================================================================
# The site-specific MSA selects its records by GCIM, per site *and per stripe*, so this
# arm has no fixed record set to resample and cannot reuse the shared sample the two
# FEMAP695 arms share. Two consequences run through everything below. Each structure
# draws its own randomness, from its own seed. And because a stripe's records are its
# own, the stripes are independent samples: resampling one stripe's n binary outcomes is
# exactly a binomial draw at the observed collapse fraction, so the bootstrap needs only
# the counts - which is as well, since no per-record outcomes for this arm exist outside
# the analysis drive.

def site_msa_seed(site: int, n_storeys: int) -> int:
    """Seed for one structure's site-specific MSA bootstrap.

    Every (site, storey count) draws independently, so each needs a distinct, stable
    seed: ``1000 + site * 10 + n_storeys``. Stated once here so the notebook cannot
    drift from it.
    """
    return 1000 + site * 10 + n_storeys


def load_site_msa_fragilities(
    frag_root: Path | str,
    n_storeys: Sequence[int],
    sites: Sequence[int],
    im_tag: str,
) -> dict[int, dict[int, dict]]:
    """Load the site-specific MSA collapse fragilities, keyed ``[n_storeys][site]``.

    Reads ``site_{i}/{structure_tag(i, n)}_msa_collapsefragility_{im_tag}.json``. There is
    no summary CSV for this arm, so these files are also the only source of the published
    ``median``/``dispersion`` the bootstrap is screened and checked against. A structure
    with no file is reported and skipped rather than raising, so a storey count with gaps
    stays runnable.
    """
    frag_root = Path(frag_root)

    fragilities = {}
    for n in n_storeys:
        found, missing = {}, []
        for site in sites:
            tag = structure_tag(site, n)
            path = frag_root / f"site_{site}" / f"{tag}_msa_collapsefragility_{im_tag}.json"
            try:
                with open(path, "r") as file:
                    found[site] = json.load(file)
            except FileNotFoundError:
                missing.append(site)

        if missing:
            print(f"No site-specific MSA fragility for {n}s at sites {missing}. Skipping...")
        if not found:
            raise ValueError(f"no {n}s site MSA fragilities under {frag_root}")
        fragilities[n] = found

    return fragilities


def site_msa_stripe_counts(fc: dict, n_records: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover ``(stripe_imls, n_collapses)`` from a fragility's empirical curve.

    The MSA fragility records the collapse *fraction* at each stripe, so the counts come
    back as ``fraction * n_records``. A fraction that is not an exact multiple of
    ``1 / n_records`` means ``n_records`` is wrong for this structure, and is raised
    rather than rounded away - the counts are the entire input to the fit.
    """
    imls, fractions = (np.asarray(v, dtype=float) for v in fc["efc"])

    counts = fractions * n_records
    if not np.allclose(counts, np.round(counts)):
        raise ValueError(f"collapse fractions are not multiples of 1/{n_records}: "
                         f"{list(fractions)}")

    return imls, np.round(counts).astype(int)


def stripe_resample_p(n_collapses: np.ndarray, n_records: int) -> np.ndarray:
    """Per-stripe resampling probability for the site MSA bootstrap.

    The empirical ``z / n`` everywhere except at ``z = 0`` and ``z = n``, where it would be
    exactly 0 or 1 and the stripe would resample to the same value in every replicate -
    frozen, carrying no uncertainty while still anchoring every replicate's likelihood.
    Those stripes get the Jeffreys posterior mean ``(z + 0.5) / (n + 1)`` instead, which
    is enough to let them vary.

    The correction is deliberately confined to the extremes. Everywhere else ``z / n`` is
    already a non-degenerate empirical distribution, and shrinking it toward 0.5 would
    bias ``beta`` upward at every site rather than only where the plain bootstrap fails.
    """
    z = np.asarray(n_collapses, dtype=float)
    return np.where((z == 0) | (z == n_records),
                    (z + 0.5) / (n_records + 1),
                    z / n_records)


def bootstrap_site_msa_fragilities(
    fragilities: dict[int, dict],
    n_records: int,
    k_samples: int,
    n_storeys: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resample each stripe's collapse count and refit, for every site.

    ``fragilities`` is one storey count's ``{site: fragility_dict}``. Each site is seeded
    by :func:`site_msa_seed`, its stripes drawn independently at
    :func:`stripe_resample_p`, and every replicate refitted with
    :func:`standes.fitting.lognorm_mle_fit` - the fit
    :func:`standes.fragility_curves.fragility_from_msa` performs.

    Screening matches the group MSA arm: a replicate is usable when its counts are not
    :func:`msa_counts_degenerate` and its fit passes :func:`msa_fit_ok` against that
    site's own published fit. The rest come back as ``NaN``, with the boolean mask
    returned alongside.

    Returns ``(theta, beta, fit_ok)``, all ``k_samples x n_sites`` with the replicate
    number as the index and **integer** site numbers as the columns - the same shape
    :func:`expand_groups_to_sites` gives the other two arms, so all three line up
    column-by-column.
    """
    thetas, betas, oks = {}, {}, {}
    for site in sorted(fragilities):
        fc = fragilities[site]
        imls, z = site_msa_stripe_counts(fc, n_records)

        rng = np.random.default_rng(site_msa_seed(site, n_storeys))
        counts = rng.binomial(n_records, stripe_resample_p(z, n_records),
                              size=(k_samples, len(z)))

        fits = np.array([lognorm_mle_fit(imls, c, n_records) for c in counts])
        theta, beta = fits[:, 0], fits[:, 1]

        ok = ~msa_counts_degenerate(counts) & msa_fit_ok(
            theta, beta, fc["median"], fc["dispersion"])

        thetas[site] = np.where(ok, theta, np.nan)
        betas[site] = np.where(ok, beta, np.nan)
        oks[site] = ok

    theta, beta, fit_ok = (pd.DataFrame(d) for d in (thetas, betas, oks))
    for df in (theta, beta, fit_ok):
        df.index.name = "k"
        df.columns.name = "site"

    return theta, beta, fit_ok


# =============================================================================
# Descriptive statistics
# =============================================================================

def get_stats(values) -> dict[str, float]:
    """Return the summary statistics recorded for every bootstrapped quantity."""
    return {"N_obs": len(values),
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "variance": np.var(values),
            "std": np.std(values, ddof=1),
            "2.5pc": np.percentile(values, 2.5),
            "5pc": np.percentile(values, 5),
            "16pc": np.percentile(values, 16),
            "84pc": np.percentile(values, 84),
            "95pc": np.percentile(values, 95),
            "97.5pc": np.percentile(values, 97.5),
            "skewness": skew(values),
            "kurtosis": kurtosis(values)
            }


def print_stats(stat_dicts: list[dict[str, float]],
                headings: list[str] | None = None) -> None:
    """Print one or more :func:`get_stats` dictionaries as an aligned table."""
    title_string = f"{'Stat':12}"
    if headings:
        for heading in headings:
            title_string += f"{heading:>12}"
    title_string += "\n" + "-" * (len(stat_dicts) + 1) * 12

    keys = stat_dicts[0].keys()
    values = [d.values() for d in stat_dicts]

    value_string = ""
    for z in zip(keys, *values):
        value_string += f"{z[0]:12}"
        if any(np.array(z[1:]) < 0.001):
            for vi in z[1:]:
                value_string += f"{vi:12.3e}"
        else:
            for vi in z[1:]:
                value_string += f"{vi:12.3f}"
        value_string += "\n"

    print(title_string)
    print(value_string)


def bootstrap_csv_path(
    root: Path | str,
    arm: str,
    quantity: str,
    n_storeys: int,
    im_tag: str,
    scope: str,
) -> Path:
    """Canonical filename for a saved bootstrap frame.

    ``arm`` is e.g. ``"msa_femap695"``, ``quantity`` one of ``theta``/``beta``/
    ``theta_stats``/``beta_stats``/``fit_ok``, ``scope`` ``"by_group"`` or ``"by_site"``.
    Both the save and the reload go through here so the two cannot drift apart.
    """
    return (Path(root)
            / f"{arm}_bootstrap_{quantity}_{n_storeys}s_{im_tag}_{scope}.csv")


def read_bootstrap_frame(path: Path | str, columns_name: str) -> pd.DataFrame:
    """Read a saved replicate frame back, restoring the labels the CSV cannot carry.

    ``columns.name`` is lost on the round trip, and site numbers come back as strings;
    both are restored so a reloaded frame is indistinguishable from a freshly computed
    one. Boolean masks are parsed as ``bool`` by ``read_csv`` and need no special
    handling.
    """
    df = pd.read_csv(path, index_col=0)
    if columns_name == "site":
        df.columns = pd.Index([int(c) for c in df.columns], name="site")
    else:
        df.columns.name = columns_name
    df.index.name = "k"
    return df


def load_saved_bootstrap(
    root: Path | str,
    arm: str,
    quantities: Sequence[str],
    n_storeys: int,
    im_tag: str,
    scope: str,
    k_samples: int | None = None,
) -> dict[str, pd.DataFrame] | None:
    """Reload one arm's saved replicate frames, or ``None`` if any is missing.

    The fits are deterministic, so a previous run's output can stand in for repeating
    them. Returning ``None`` on a single missing file - rather than a partial set - is
    what keeps the caller's fallback all-or-nothing: a half-finished save can never be
    silently mixed with a fresh computation.

    ``k_samples`` guards against picking up a cloud of the wrong size, which is the one
    way a stale file could pass unnoticed after the replicate count changes.
    """
    paths = {q: bootstrap_csv_path(root, arm, q, n_storeys, im_tag, scope)
             for q in quantities}
    if not all(p.is_file() for p in paths.values()):
        return None

    columns_name = scope.removeprefix("by_")
    frames = {q: read_bootstrap_frame(p, columns_name) for q, p in paths.items()}

    if k_samples is not None:
        for q, df in frames.items():
            if len(df) != k_samples:
                raise ValueError(f"saved {arm} {q} for {n_storeys}s has {len(df)} "
                                 f"replicates, not k_samples={k_samples} - delete it or "
                                 f"set REUSE_SAVED = False to refit")
    return frames

# load the theta and beta bootstrap stats
def load_saved_bootstrap_stats(
    root: Path | str,
    arm: str,
    quantities,
    n_storeys: int,
    im_tag: str,
    scope: str,
    ) -> dict[str, pd.DataFrame] | None:
    """Reload one arm's saved replicate frames, or ``None`` if any is missing.

    The fits are deterministic, so a previous run's output can stand in for repeating
    them. Returning ``None`` on a single missing file - rather than a partial set - is
    what keeps the caller's fallback all-or-nothing: a half-finished save can never be
    silently mixed with a fresh computation.

    ``k_samples`` guards against picking up a cloud of the wrong size, which is the one
    way a stale file could pass unnoticed after the replicate count changes.
    """
    paths = {q: bootstrap_csv_path(root, arm, q, n_storeys, im_tag, scope)
             for q in quantities}
    if not all(p.is_file() for p in paths.values()):
        return None

    columns_name = "stats"
    index_name = scope.removeprefix("by_")
    frames = {q: read_bootstrap_stats_frame(p, columns_name, index_name) for q, p in paths.items()}

    
    for q, df in frames.items():
        row_index = pd.MultiIndex.from_product([df.index, [n_storeys]], names=("site", "n_storeys"))
        df = df.set_index(row_index)
        frames[q] = df
        
    return frames


def read_bootstrap_stats_frame(path: Path | str, columns_name: str, index_name: str) -> pd.DataFrame:
    """Read a saved replicate frame back, restoring the labels the CSV cannot carry.

    ``columns.name`` is lost on the round trip, and site numbers come back as strings;
    both are restored so a reloaded frame is indistinguishable from a freshly computed
    one. Boolean masks are parsed as ``bool`` by ``read_csv`` and need no special
    handling.
    """
    df = pd.read_csv(path, index_col=0)
    df.columns.name = columns_name
    df.index.name = index_name
    return df


def reformat_bootstrap_df(df: pd.DataFrame, n_storeys):
    """Reformat the bootstrap theta and beta dataframes to have multi-index rows"""
    row_index = pd.MultiIndex.from_product([df.columns, [n_storeys]], names=["site", "n_storeys"])
    df = df.T
    df = df.set_index(row_index, drop=True)
    return df


def bootstrap_stats_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise every column of a bootstrap cloud with :func:`get_stats`.

    One row per column of ``df`` (a site or a design group), one column per statistic.
    Screened-out replicates are ``NaN`` (see
    :func:`bootstrap_msa_group_fragilities`) and are dropped per column, so ``N_obs``
    reports the number of usable replicates behind each row. A column with nothing left
    comes back as ``N_obs = 0`` and ``NaN`` statistics rather than raising, so one
    unusable group does not take the whole frame down with it.
    """
    def column_stats(values):
        if values.empty:
            return dict.fromkeys(get_stats([0.0, 1.0]), np.nan) | {"N_obs": 0}
        return get_stats(values)

    stats = pd.DataFrame({col: column_stats(df[col].dropna()) for col in df}).T
    stats.index.name = df.columns.name
    return stats


# =============================================================================
# Estimator bias
# =============================================================================
# The meta-regression works on ln(theta) and ln(beta), and both are biased there. Part of
# it is the fit itself at the sample size that was actually run - the moment fit gives an
# unbiased beta**2, not an unbiased beta - and part is the log transform, which is concave
# and so pulls the mean of the transform below the transform of the mean (the Jensen
# term). Every arm already carries a cloud of replicates around the fit it resamples, so
# the bias is read straight off that cloud: E[ln x*] - ln x_hat, Efron & Tibshirani (1993)
# eq. 10.2 on the log scale. No refitting is needed.

def log_bias_stats(values, reference: float) -> dict[str, float]:
    """Summarise the log-scale bias of one estimator from its bootstrap replicates.

    ``values`` is one column of a replicate cloud and ``reference`` the point estimate it
    resamples. Screened-out replicates are ``NaN`` (see
    :func:`bootstrap_msa_group_fragilities`) and are dropped here, so ``N_obs`` reports how
    many replicates actually stand behind the row.
    """
    lboot = np.log(np.asarray(values, dtype=float))
    lboot = lboot[np.isfinite(lboot)]

    out = {
        "N_obs": len(lboot),
        "bias": lboot.mean() - np.log(reference),         # E&T eq. 10.2, log scale
        "se_est": lboot.std(ddof=1),                      # bootstrap se of the estimator
        "mcse": lboot.std(ddof=1) / np.sqrt(len(lboot)),  # MC error on the bias itself
        "raw_rel": (np.exp(lboot).mean() - reference) / reference,  # raw-scale bias
        "jensen": -0.5 * lboot.var(ddof=1),               # the log-concavity share
    }
    out["bias/se"] = abs(out["bias"] / out["se_est"])
    out["bias/mcse"] = abs(out["bias"] / out["mcse"])
    return out


def bootstrap_bias_frame(boot_df: pd.DataFrame,
                         references: pd.Series) -> pd.DataFrame:
    """Estimate the log-scale bias of every column of a bootstrap cloud.

    One row per column of ``boot_df`` (a site or a design group), one column per statistic
    of :func:`log_bias_stats`. ``references`` gives each column its own point estimate and
    must be aligned to ``boot_df.columns``. A column with no usable replicate left comes
    back as ``N_obs = 0`` and ``NaN`` statistics rather than raising - the same convention
    :func:`bootstrap_stats_frame` follows.
    """
    missing = [c for c in boot_df.columns if c not in references.index]
    if missing:
        raise KeyError(f"no reference estimate for {missing}")

    def column_bias(col):
        values = boot_df[col].dropna()
        if values.empty:
            return dict.fromkeys(log_bias_stats([1.0, 2.0], 1.0), np.nan) | {"N_obs": 0}
        return log_bias_stats(values, references[col])

    bias = pd.DataFrame({col: column_bias(col) for col in boot_df}).T
    bias.index.name = boot_df.columns.name
    return bias


def ida_beta_bias_correction(n_records: int) -> float:
    """Return the closed-form log-scale bias of the IDA dispersion estimator.

    The moment fit gives an unbiased ``beta ** 2``, but the square root and the log are
    both concave, so ``ln beta`` comes out low by a fixed amount that depends only on the
    record count. The result is negative (-0.0242 at 22 records).
    """
    return 0.5 * (psi((n_records - 1) / 2) - np.log((n_records - 1) / 2))


def add_bias_correction(bias_df: pd.DataFrame,
                        correction: float | pd.Series) -> pd.DataFrame:
    """Subtract a bias correction in place and add the residual diagnostics.

    ``correction`` may be a scalar applied to every row, or a Series giving each site its
    own. Correcting a frame by its own ``bias`` column leaves a residual that is zero by
    construction; the informative case is a correction from somewhere else - a closed form,
    or an across-site constant - where the residual is a genuine test of it.
    """
    bias_df["bias_corrected"] = bias_df["bias"] - correction
    bias_df["bias/se_corrected"] = np.abs(bias_df["bias_corrected"]) / bias_df["se_est"]
    bias_df["bias_corrected/mcse"] = np.abs(bias_df["bias_corrected"]) / bias_df["mcse"]
    return bias_df


def load_group_fit_estimates(summary_path: Path | str,
                             groups: Sequence[str],
                             im_tag: str | None = None) -> pd.DataFrame:
    """Read the published ``median``/``dispersion`` of a group fragility summary.

    The two group summaries are laid out differently: the IDA one (nb 053) carries a
    two-row header keyed by IM, the MSA one (nb 061) is flat because it only ever holds a
    single IM. Passing ``im_tag`` selects the first layout. Returns a frame indexed by
    group with columns ``theta`` and ``beta``.
    """
    if im_tag is None:
        summary = pd.read_csv(summary_path, index_col=0)
        theta, beta = summary["median"], summary["dispersion"]
    else:
        summary = pd.read_csv(summary_path, header=[0, 1], index_col=0)
        theta = summary[(im_tag, "median")].astype(float)
        beta = summary[(im_tag, "dispersion")].astype(float)

    estimates = pd.DataFrame({"theta": theta.loc[list(groups)],
                              "beta": beta.loc[list(groups)]})
    estimates.index.name = "group"
    return estimates


def site_fit_estimates(site_fcs: dict[int, dict],
                       sites: Sequence[int]) -> pd.DataFrame:
    """Return the published ``median``/``dispersion`` of the site-specific MSA fits.

    This arm has no summary CSV, so the fragility JSONs loaded by
    :func:`load_site_msa_fragilities` are also the only source of its point estimates.
    """
    return pd.DataFrame({"theta": [site_fcs[s]["median"] for s in sites],
                         "beta": [site_fcs[s]["dispersion"] for s in sites]},
                        index=pd.Index(list(sites), name="site"))


def stack_estimates(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Stack one arm's per-storey point estimates into a single ``(unit, n_storeys)`` frame.

    ``frames`` is keyed by storey count, each value a ``theta``/``beta`` frame indexed by
    site or group. The result carries every storey count in one object, indexed the way the
    meta-regression wants its rows - the same ordering :func:`reformat_bootstrap_df` uses,
    so an estimate frame and a replicate frame line up row for row.
    """
    parts = []
    for n, df in frames.items():
        part = df[["theta", "beta"]].copy()
        part.index = pd.MultiIndex.from_product(
            [df.index, [n]], names=[df.index.name or "unit", "n_storeys"])
        parts.append(part)

    return pd.concat(parts).sort_index()


def estimates_csv_path(root: Path | str, arm: str, im_tag: str, scope: str) -> Path:
    """Canonical filename for a saved point-estimate frame.

    No storey count in the name: unlike the replicate clouds a single file carries every
    storey count, on the second level of its index.
    """
    return Path(root) / f"{arm}_estimates_{im_tag}_{scope}.csv"


def read_estimates_frame(path: Path | str,
                         index_name: str = "site") -> pd.DataFrame:
    """Read a saved point-estimate frame back, restoring its two-level index.

    Site numbers and storey counts both come back as strings from a CSV; both are returned
    to ``int`` so the frame is indistinguishable from a freshly built one. Group labels are
    left as they are.
    """
    df = pd.read_csv(path, index_col=[0, 1])
    units = df.index.get_level_values(0)
    if index_name == "site":
        units = [int(u) for u in units]
    df.index = pd.MultiIndex.from_arrays(
        [units, [int(n) for n in df.index.get_level_values(1)]],
        names=[index_name, "n_storeys"])
    return df


def load_estimates(
    root: Path | str,
    im_tag: str,
    scope: str = "by_site",
    arms: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load several arms' saved point estimates at once, keyed by arm.

    ``arms`` defaults to every arm the scope has: all three by site, the two FEMAP695 arms
    by group. Unlike :func:`load_saved_bootstrap` a missing file raises rather than
    returning ``None`` - these are an input to the models downstream, not a cache standing
    in for work that can simply be redone.
    """
    if arms is None:
        arms = (["site_msa", "msa_femap695", "ida_femap695"] if scope == "by_site"
                else ["msa_femap695", "ida_femap695"])

    index_name = scope.removeprefix("by_")
    estimates = {}
    for arm in arms:
        path = estimates_csv_path(root, arm, im_tag, scope)
        if not path.is_file():
            raise FileNotFoundError(f"no saved estimates for {arm} {scope} - run "
                                    f"nb 070 section 2.1 to write {path.name}")
        estimates[arm] = read_estimates_frame(path, index_name)

    return estimates


def bias_csv_path(root: Path | str, arm: str, quantity: str, n_storeys: int,
                  im_tag: str, scope: str) -> Path:
    """Canonical filename for a saved bias frame.

    Goes through :func:`bootstrap_csv_path` with ``quantity`` suffixed ``_bias``, so the
    bias files sit alongside the clouds they were computed from under the same convention.
    """
    return bootstrap_csv_path(root, arm, f"{quantity}_bias", n_storeys, im_tag, scope)


def read_bias_frame(path: Path | str, index_name: str = "site") -> pd.DataFrame:
    """Read a saved bias frame back, restoring the index labels the CSV cannot carry.

    Unlike :func:`read_bootstrap_frame` the rows here are sites or groups, not replicates,
    so site numbers have to come back as ``int`` on the *index* rather than the columns.
    """
    df = pd.read_csv(path, index_col=0)
    if index_name == "site":
        df.index = pd.Index([int(i) for i in df.index], name="site")
    else:
        df.index.name = index_name
    return df


def load_saved_bias(root: Path | str, arms: Sequence[str],
                    quantities: Sequence[str], n_storeys: int, im_tag: str,
                    scope: str) -> dict[str, dict[str, pd.DataFrame]] | None:
    """Reload the saved bias frames of several arms, or ``None`` if any is missing.

    All-or-nothing for the same reason :func:`load_saved_bootstrap` is: a half-written set
    must never be silently mixed with a freshly computed one.
    """
    paths = {(arm, q): bias_csv_path(root, arm, q, n_storeys, im_tag, scope)
             for arm in arms for q in quantities}
    if not all(p.is_file() for p in paths.values()):
        return None

    index_name = scope.removeprefix("by_")
    frames: dict[str, dict[str, pd.DataFrame]] = {arm: {} for arm in arms}
    for (arm, q), path in paths.items():
        frames[arm][q] = read_bias_frame(path, index_name)
    return frames


def summarise_bias(bias_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Condense the per-site bias frames into one row per arm and quantity.

    The two counts are the numbers that decide what has to be corrected: how many sites
    carry a bias worth more than a quarter of the estimator's own standard error, and how
    many carry one the bootstrap has actually resolved from zero.
    """
    rows = {}
    for arm, quantities in bias_data.items():
        for quantity, df in quantities.items():
            row = {"mean_bias": df["bias"].mean(),
                   "min_bias": df["bias"].min(),
                   "max_bias": df["bias"].max(),
                   "mean_jensen": df["jensen"].mean(),
                   "mean_bias/se": df["bias/se"].mean(),
                   "mean_bias/mcse": df["bias/mcse"].mean(),
                   "n_over_se": int((df["bias/se"] > _BIAS_SE_THRESHOLD).sum()),
                   "n_over_mcse": int((df["bias/mcse"] > _BIAS_MCSE_THRESHOLD).sum()),
                   "n_sites": len(df)}
            if "bias_corrected" in df:
                row["mean_bias_corr"] = df["bias_corrected"].mean()
                row["mean_bias/se_corr"] = df["bias/se_corrected"].mean()
            rows[(arm, quantity)] = row

    summary = pd.DataFrame(rows).T
    summary.index.names = ["arm", "quantity"]
    return summary


# -----------------------------------------------------------------------------
# Bias diagnostics
# -----------------------------------------------------------------------------

def style_legend(ax, **kwargs) -> None:
    """Add a legend with the black frame used throughout the chapter."""
    leg = ax.legend(**kwargs)
    leg.get_frame().set_edgecolor("k")


def _plot_arm_series(ax, series_by_arm: dict[str, pd.Series], ylabel: str,
                     means: bool = False, se_bands: bool = False,
                     mcse_line: bool = False) -> None:
    """Scatter one diagnostic against the site number, one series per arm.

    ``means`` adds each arm's across-site mean as a dashed line of its own colour;
    ``se_bands`` and ``mcse_line`` add the two materiality thresholds.
    """
    if not (se_bands or mcse_line):
        ax.axhline(0, ls="-", color="k", lw=0.75)

    for arm, series in series_by_arm.items():
        color = _ARM_COLORS[arm]
        ax.plot(series.index, series.to_numpy(), marker=".", mfc=color, mec=color,
                ls="none", label=_ARM_LABELS[arm])
        if means:
            ax.axhline(series.mean(), ls="--", color=color,
                       label=f"Avg. {_ARM_LABELS[arm]}")

    if se_bands:
        # the band is drawn to 0.4 unless something exceeds it, since a bias that large
        # is the whole point of the panel and must not be clipped out of sight
        top = max(0.4, 1.05 * max(s.max() for s in series_by_arm.values()))
        ax.axhline(_BIAS_SE_THRESHOLD, ls="--", color="k")
        ax.axhspan(0.0, _BIAS_SE_THRESHOLD, alpha=0.2, color="g")
        ax.axhspan(_BIAS_SE_THRESHOLD, top, alpha=0.2, color="r")
        ax.set_ylim(0, top)
    if mcse_line:
        ax.axhline(_BIAS_MCSE_THRESHOLD, ls="--", color="k")

    ax.grid(ls="-.", color="0.8")
    ax.set_xlabel("Site No.")
    ax.set_ylabel(ylabel)

    style_legend(ax, fontsize="small", ncol=2 if means else 1)


def plot_bias_assessment(
    bias_data: dict[str, dict[str, pd.DataFrame]],
    references: dict[str, pd.DataFrame] | None = None,
    contrasts: Sequence[tuple[str, str]] | None = None,
) -> tuple[plt.Figure, np.ndarray, plt.Figure, plt.Axes]:
    """Chart the per-site bias of every arm, before and after correction.

    ``bias_data`` is keyed ``[arm]["theta"|"beta"]`` and every frame must already carry the
    ``bias_corrected`` columns :func:`add_bias_correction` adds. The 3x3 grid is arranged
    as columns ``ln theta`` / ``ln beta`` / corrected ``ln beta``, and rows raw bias /
    bias-to-se / bias-to-mcse.

    The second figure shows what survives on the quantity the comparison is actually about:
    the net bias on each pairwise difference of ``ln beta``, uncorrected against corrected.
    Its pooled ratios are printed, since those are the numbers that decide whether a
    correction is needed for a cross-site claim: the mean net bias against the standard
    error of the mean *observed* difference, which is what ``references`` - each arm's
    point estimates, as :func:`site_fit_estimates` returns them - is needed for. Without it
    the ratio falls back to the site-to-site scatter of the bias itself, which answers the
    different and much less useful question of how uniform the bias is.
    """
    arms = list(bias_data)
    if contrasts is None:
        contrasts = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]

    fig1, axs1 = plt.subplots(3, 3, figsize=(15, 11))

    columns = [("theta", "bias", r"Bias in $\ln{\theta}$", r"Bias [$\ln{\theta}$ units]"),
               ("beta", "bias", r"Bias in $\ln{\beta}$", r"Bias [$\ln{\beta}$ units]"),
               ("beta", "bias_corrected", r"Corrected Bias in $\ln{\beta}$",
                r"Bias [$\ln{\beta}$ units]")]

    for j, (quantity, key, title, ylabel) in enumerate(columns):
        suffix = "_corrected" if key == "bias_corrected" else ""
        _plot_arm_series(axs1[0, j],
                         {a: bias_data[a][quantity][key] for a in arms},
                         ylabel, means=True)
        _plot_arm_series(axs1[1, j],
                         {a: bias_data[a][quantity][f"bias/se{suffix}"] for a in arms},
                         "Bias / S.E. [-]", se_bands=True)
        _plot_arm_series(axs1[2, j],
                         {a: bias_data[a][quantity][f"bias{suffix}/mcse"] for a in arms},
                         "Bias / MCSE [-]", mcse_line=True)
        axs1[0, j].set_title(title)

    fig1.tight_layout()

    fig2, axs2 = plt.subplots(figsize=(9, 5))
    axs2.axhline(0, ls="-", color="k", lw=0.75)

    n_sites = len(bias_data[arms[0]]["beta"])
    print(f"Net bias on the ln(beta) contrasts, over {n_sites} sites:")
    _contrast_styles = [("o", "tab:blue"), ("s", "tab:green"), ("^", "tab:purple"),
                        ("v", "tab:brown"), ("D", "tab:pink")]
    for (a, b), (marker, color) in zip(contrasts, _contrast_styles):
        delta = bias_data[a]["beta"]["bias"] - bias_data[b]["beta"]["bias"]
        delta_corr = (bias_data[a]["beta"]["bias_corrected"]
                      - bias_data[b]["beta"]["bias_corrected"])
        label = f"{_ARM_LABELS[a]} - {_ARM_LABELS[b]}"

        axs2.plot(delta.index, delta.to_numpy(), ls="none", marker=marker, ms=5,
                  mfc="none", mec=color, alpha=0.6, label=f"{label} (uncorr.)")
        axs2.plot(delta_corr.index, delta_corr.to_numpy(), ls="none", marker=marker,
                  ms=5, mfc=color, mec=color, label=f"{label} (corr.)")

        # pooled over the sites: the mean net bias against the se of the mean observed
        # difference, i.e. is the bias material for a claim made across all of them
        if references is None:
            spread = delta
        else:
            spread = (np.log(references[a]["beta"]) - np.log(references[b]["beta"]))
        se_p = spread.std(ddof=1) / np.sqrt(len(spread))
        print(f"  {label:>37}: mean {delta.mean():+.4f} -> {delta_corr.mean():+.4f}, "
              f"|mean|/se_p {abs(delta.mean()) / se_p:6.2f} -> "
              f"{abs(delta_corr.mean()) / se_p:6.2f}")

    axs2.grid(ls="-.", color="0.8")
    axs2.set_xlabel("Site No.")
    axs2.set_ylabel(r"Net bias on $\ln{\beta}$")
    axs2.set_title(r"Total bias on the $\ln{\beta}$ contrasts")
    style_legend(axs2, fontsize="small", ncol=2)

    fig2.tight_layout()

    return fig1, axs1, fig2, axs2


# =============================================================================
# Meta-Regression
# =============================================================================

def get_fe_weights(Vis: pd.Series):
    Wis = 1/ Vis
    return Wis


def get_re_weights(Vis: pd.Series|np.ndarray, T_sq: float):
    Wis = 1 / (Vis + T_sq)
    return Wis


def compute_Q(Wis: pd.Series|np.ndarray, Yis: pd.Series|np.ndarray
              ) -> pd.Series|np.ndarray:
    """Calculates the weighted sum of the square deviations, Q
    
    follows Borenstein et al. "Introduction to Meta-Analysis" Eq. 16.3 and uses 
    the method-of-moments / DerSimonian and Laird Method
    
    """

    Q = np.sum(Wis * Yis ** 2) - (np.sum(Wis * Yis)) ** 2 / np.sum(Wis)
    return Q


def compute_C(Wis: pd.Series|np.ndarray) -> pd.Series|np.ndarray:
    """ follows Borenstein et al. "Introduction to Meta-Analysis" Eq. 12.5
    """
    C = np.sum(Wis) - np.sum(Wis ** 2) / np.sum(Wis)
    return C


def compute_Tsquared(
        Wis_fe: pd.Series|np.ndarray, 
        Yis: pd.Series|np.ndarray,
        n_studies: int,
        ) -> pd.Series|np.ndarray:
    """ The estimated between-group variance.
    
    follows Borenstein et al. "Introduction to Meta-Analysis" Eq. 12.2

    Truncated at zero. A variance cannot be negative, and Borenstein et al. (Ch. 12,
    p. 72) set T_sq to zero whenever Q falls below its degrees of freedom - the
    observed dispersion is then no more than sampling error alone would produce.
    Left untruncated a negative T_sq feeds inflated (or negative) random-effects
    weights into Eq. 12.6, with nothing downstream to flag it.
    """
    df = n_studies - 1
    Q = compute_Q(Wis_fe, Yis)
    C = compute_C(Wis_fe)

    T_sq = (Q - df) / C
    return max(float(T_sq), 0.0)


def compute_I_sq(
        Wis: pd.Series|np.ndarray, 
        Yis: pd.Series|np.ndarray):
    """Computes the I² statistic for heterogeneity
    
    follows Borenstein et al. "Introduction to Meta-Analysis" Eq. 16.9
    """

    n_studies = len(Yis)
    df = n_studies - 1
    Q = compute_Q(Wis, Yis)

    I_sq = max((Q - df) / Q * 100, 0)

    return I_sq


def summary_effect_re(
        Wis_re: pd.Series|np.ndarray, Yis: pd.Series|np.ndarray
        ) -> pd.Series|np.ndarray:
    """ The estimated between-group variance.
            
    follows Borenstein et al. "Introduction to Meta-Analysis" Eq. 12.7
    """
    M_re = np.sum(Wis_re * Yis) / np.sum(Wis_re)
    return M_re


def summary_variance_re(Wis_re: pd.Series|np.ndarray) -> pd.Series|np.ndarray:
    return 1 / np.sum(Wis_re)


def drop_incomplete_studies(
        Vis: pd.Series|np.ndarray,
        Yis: pd.Series|np.ndarray,
        ) -> tuple[pd.Series, pd.Series]:
    """Keep only the studies that carry a usable (variance, effect) pair.

    A screened-out replicate (see :func:`msa_fit_ok`) arrives here as ``NaN``, and a
    zero or negative variance would divide by zero in Eq. 11.2. Neither raises on its
    own: ``np.sum`` on a pandas Series skips ``NaN`` silently, so an unfiltered frame
    would compute Q and C over the surviving studies while the caller's ``n_studies``
    still counted the missing ones - a degrees-of-freedom mismatch that biases T_sq
    upward with no warning. Dropping the pairs here is what lets the study count be
    taken from what actually survives.
    """
    if len(Vis) != len(Yis):
        raise ValueError(f"Vis has {len(Vis)} studies but Yis has {len(Yis)}")

    # A bare array takes the other argument's labels so the two stay aligned; two
    # labelled inputs must already agree, because silently re-labelling one of them
    # is how a variance ends up against the wrong study.
    index = Vis.index if isinstance(Vis, pd.Series) else (
        Yis.index if isinstance(Yis, pd.Series) else pd.RangeIndex(len(Vis)))
    if isinstance(Vis, pd.Series) and isinstance(Yis, pd.Series) \
            and not Vis.index.equals(Yis.index):
        raise ValueError("Vis and Yis are both labelled but their indexes differ")

    Vis = pd.Series(np.asarray(Vis, dtype=float), index=index)
    Yis = pd.Series(np.asarray(Yis, dtype=float), index=index)

    keep = np.isfinite(Vis) & np.isfinite(Yis) & (Vis > 0)
    return Vis[keep], Yis[keep]


def compute_re_summary_effect(
        Vis: pd.Series|np.ndarray,
        Yis: pd.Series|np.ndarray,
        n_studies: int | None = None,
        ) -> pd.Series|np.ndarray:
    """ The estimated between-group variance.

    follows Borenstein et al. "Introduction to Meta-Analysis"
    Eq. 12.7, 12.8, 12.9

    Studies without a usable (variance, effect) pair are dropped first and the study
    count is taken from what is left, so the degrees of freedom behind T_sq always
    match the studies that were actually summed. ``n_studies`` is therefore optional
    and kept only so existing callers keep working; a value that disagrees with the
    surviving count is warned about and ignored, because the surviving count is the
    one that is right.

    returns the mean effects, the variance of the mean effect, standard error,
    the between study variance (T_sq) and the weights
    """
    Vis, Yis = drop_incomplete_studies(Vis, Yis)
    k = len(Vis)
    if k < 2:
        raise ValueError(f"a random-effects fit needs at least 2 usable studies, got {k}")
    if n_studies is not None and n_studies != k:
        warnings.warn(f"n_studies={n_studies} was passed but only {k} studies have a "
                      f"usable (variance, effect) pair; using {k} for the degrees of "
                      f"freedom behind T_sq", RuntimeWarning, stacklevel=2)

    Wis_fe = get_fe_weights(Vis)
    T_sq = compute_Tsquared(Wis_fe, Yis, k)
    Wis_re = get_re_weights(Vis, T_sq)
    M_re = summary_effect_re(Wis_re, Yis)
    V_re = summary_variance_re(Wis_re)
    SE_re = np.sqrt(V_re)

    return M_re, V_re, SE_re, T_sq, Wis_re


def compute_prediction_interval():
    """Computes the prediction interval for the random effects model
    using the estimate of the variance and hte t-distributions with n-2 dofs
    
    According to Borenstein et al. "Introduction to Meta-Analysis" eq. 17.7, 17.8"""

    # TODO::
    ...


def compute_heterogeneity_stats(
        Vis: pd.Series|np.ndarray, Yis: pd.Series|np.ndarray) -> dict[str, float]:

    # TODO::
    n_studies = len(Vis)
    Wis_fe = get_fe_weights(Vis)

    # Q
    Q = compute_Q(Wis_fe, Yis)

    # Q-df
    Q_df = Q - (n_studies - 1)

    # p-value
    # todo:: compute p-value from Q using the chi-sq distribution. 
    # todo:: s. pg. 112 of Borenstein et al.

    # T²
    T_sq = compute_Tsquared(Wis_fe, Yis, n_studies)

    # T
    T = np.sqrt(T_sq)
    
    # I²
    I_sq = compute_I_sq(Wis_fe, Yis)

    return {"Q": Q, "Q_df": Q_df, "T_sq": T_sq, "T": T, "I_sq": I_sq}


