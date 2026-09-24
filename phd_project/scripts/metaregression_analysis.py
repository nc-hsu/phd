import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections.abc import Sequence
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import chi2, kurtosis, skew

from standes.fitting import lognorm_mle_fit, lognorm_moment_fit

from phd_project.scripts.femap695_records import record_tag_to_column

# A resampled MSA fit is treated as degenerate once it exceeds this multiple of the
# fit it was drawn from. The affected replicates sit many orders of magnitude away
# (theta ~ 1e21), so anything from roughly 5 to 50 selects the same ones.
_MAX_FIT_RATIO = 10.0

# One colour per arm, fixed here so every figure in the chapter reads the same way. The
# two FEMAP695 arms are warm, the site-specific arm cool, because the fixed-vs-site-
# specific record set is the distinction the comparison is actually about.
# The three ida_femap695* arms differ only in where inside the IDA's collapse bracket
# each record's capacity is placed: the lower bound (the published fragility), the upper
# bound, or the bracket's geometric mean. See nb 053 section 10.
_ARM_COLORS = {"site_msa": "b", "msa_femap695": "tab:orange", "ida_femap695": "r",
               "ida_femap695_ub": "tab:purple", "ida_femap695_avg": "tab:green"}
_ARM_LABELS = {"site_msa": "MSA site-specific",
               "msa_femap695": "MSA FEMA P695",
               "ida_femap695": "IDA FEMA P695",
               "ida_femap695_ub": "IDA FEMA P695 (upper bound)",
               "ida_femap695_avg": "IDA FEMA P695 (geom. avg.)"}

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

    Superseded on the site-MSA path by
    :func:`bootstrap_site_msa_fragilities_from_flags`, which resamples records rather than
    counts and so has no parametric step to correct - a degenerate stripe there is
    genuinely frozen. Still used by :func:`bootstrap_site_msa_fragilities`, kept alongside
    it for comparison.

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


# -----------------------------------------------------------------------------
# Site-specific MSA: the non-parametric record resample (nb 060 section 6 -> nb 070)
# -----------------------------------------------------------------------------
# The count resample above needs only the collapse fractions, which is why it was written
# first - nb 060 exported nothing else. It is now superseded by a record resample over the
# per-record collapse flags, the same estimator the MSA-FEMAP695 arm uses.
#
# What that does and does not change. Drawing 30 record indices with replacement from one
# stripe's 30 flags and summing is *exactly* Binomial(30, z/30), so at an informative
# stripe the two are the same draw. What changes is the extremes - a 0/30 or 30/30 stripe
# is genuinely frozen, with no Jeffreys nudge to unfreeze it - and that the indices are
# written to disk, so a replicate is auditable rather than reconstructed from a seed.
#
# The coherence that makes the MSA-FEMAP695 arm correlate across stripes is NOT available
# here: GCIM re-selects 30 different records at every site and IML, so there is no shared
# record identity to propagate and each stripe is resampled on its own. Column k of the
# flag table is a record *slot*, not a record - never join it across stripes.

_IML_DECIMALS = 6


def _iml_key(iml: float) -> float:
    """Hashable, round-trip-stable key for a stripe IML.

    The IMLs reach here twice - through nb 060's CSV and through the fragility's ``efc`` -
    and have to agree to the bit for the two to be matched up. They come from the same
    stripe log, so rounding well past the 4 dp the stripe filenames encode is enough.
    """
    return round(float(iml), _IML_DECIMALS)


def site_msa_stripe_seed(site: int, stripe_number: int) -> int:
    """Seed for one ``(site, stripe)`` record resample: ``1000 + site * 10 + stripe_number``.

    The resample is per **(site, stripe IML)**, not per structure: the record selection is
    keyed ``site_{i}__stripe_iml_{...}`` with no storey count, so the 3s and 5s structures
    at one site run the same 30 recordings at a shared IML and must be handed the same
    resampled slots. ``stripe_number`` is the position of the IML in that site's sorted
    unique IML list, so ``n_storeys`` has no place in the seed.

    The formula is **not injective** - site 1 / stripe 0 and site 0 / stripe 10 both give
    1010 - so :func:`draw_site_msa_index_samples` asserts the seeds it derives are
    distinct. At the time of writing the worst site carries exactly 10 stripes, i.e.
    ``stripe_number`` reaches 9 and the guard passes with nothing to spare; one more
    stripe at that site trips it. The collision-free form is ``1000 + site * 100 + ...``.
    """
    return 1000 + site * 10 + stripe_number


def load_site_msa_collapse_flags(
    path: Path | str,
    n_storeys: Sequence[int],
    n_records: int,
) -> pd.DataFrame:
    """Load nb 060's per-record collapse flags for the site-specific MSA arm.

    The file is ``site_msa_collapse_flags_{im}.csv``: one row per ``(site, n_storeys,
    stripe_iml)``, columns ``"0"`` ... ``"{n_records - 1}"`` carrying 1 for a record that
    collapsed and 0 for one that did not. A blank means that record was not run, which
    this arm cannot resample around - the sample is an index into a full ensemble - so it
    is raised rather than tolerated.

    Returns one frame indexed by ``(site, n_storeys, stripe_iml)``, sorted, restricted to
    ``n_storeys``. The analogue of :func:`load_group_collapse_flags` for the SS arm; the
    index carries the storey count because this arm has no design groups.
    """
    columns = [str(i) for i in range(n_records)]
    df = pd.read_csv(path)

    missing = [c for c in ("site", "n_storeys", "stripe_iml", *columns)
               if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")

    table = df[df["n_storeys"].isin(list(n_storeys))].copy()
    if table.empty:
        raise ValueError(f"no rows for storey counts {list(n_storeys)} in {path}")
    if not table[columns].isin([0, 1]).all(axis=None):
        raise ValueError(f"missing or non-binary collapse flags in {path} - every record "
                         "of every stripe must have run for a record resample")

    table["stripe_iml"] = table["stripe_iml"].map(_iml_key)
    return (table.set_index(["site", "n_storeys", "stripe_iml"])[columns]
                 .astype(int).sort_index())


def site_msa_stripe_numbers(flags: pd.DataFrame) -> pd.DataFrame:
    """One row per resampled record set: ``site, stripe_iml, stripe_number``.

    ``stripe_number`` is the position of the IML in that site's ascending list of unique
    stripe IMLs, **unioned over the storey counts present**, so the two structures at a
    site agree on it. It is stable as long as the site's IML set is; adding a stripe
    renumbers everything above it, which re-seeds those resamples.
    """
    pairs = (flags.index.to_frame(index=False)[["site", "stripe_iml"]]
             .drop_duplicates().sort_values(["site", "stripe_iml"]))
    pairs["stripe_number"] = pairs.groupby("site").cumcount()
    return pairs.reset_index(drop=True)


def draw_site_msa_index_samples(
    flags: pd.DataFrame,
    k_samples: int,
    n_records: int,
) -> pd.DataFrame:
    """Draw ``k_samples`` record resamples for every ``(site, stripe_iml)`` record set.

    Each pair gets its own generator, seeded by :func:`site_msa_stripe_seed`, and draws a
    ``k_samples x n_records`` array of indices into ``0 .. n_records - 1`` with
    replacement. One array per *record set*, not per structure, so the 3s and 5s
    structures at a shared stripe are resampled identically - they ran the same records.

    Returns the long form saved to disk: ``site, stripe_iml, stripe_number, k`` and then
    ``"0"`` ... ``"{n_records - 1}"``, the drawn slots. ``k_samples`` rows per pair.
    """
    pairs = site_msa_stripe_numbers(flags)

    seeds = {}
    for r in pairs.itertuples():
        seed = site_msa_stripe_seed(int(r.site), int(r.stripe_number))
        if seed in seeds:
            other = seeds[seed]
            raise ValueError(
                f"seed collision: site {r.site} stripe {r.stripe_number} and site "
                f"{other[0]} stripe {other[1]} both seed {seed}. site_msa_stripe_seed is "
                "not injective past 10 stripes at a site - switch it to "
                "1000 + site * 100 + stripe_number and redraw.")
        seeds[seed] = (int(r.site), int(r.stripe_number))

    columns = [str(i) for i in range(n_records)]
    blocks = []
    for r in pairs.itertuples():
        rng = np.random.default_rng(site_msa_stripe_seed(int(r.site), int(r.stripe_number)))
        idx = rng.integers(0, n_records, size=(k_samples, n_records))
        block = pd.DataFrame(idx, columns=columns)
        block.insert(0, "k", np.arange(k_samples))
        block.insert(0, "stripe_number", int(r.stripe_number))
        block.insert(0, "stripe_iml", _iml_key(r.stripe_iml))
        block.insert(0, "site", int(r.site))
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def load_site_msa_index_samples(
    path: Path | str,
    k_samples: int,
    n_records: int,
) -> pd.DataFrame:
    """Read back :func:`draw_site_msa_index_samples`' table, with the checks that matter.

    Guards against the file on disk having been drawn for a different replicate count or
    a different ensemble size - the mirror of the assertions nb 070 puts around the
    FEMAP695 sample.
    """
    columns = [str(i) for i in range(n_records)]
    df = pd.read_csv(path)

    if not df[columns].isin(range(n_records)).all(axis=None):
        raise ValueError(f"{path} holds indices outside 0..{n_records - 1}")
    per_pair = df.groupby(["site", "stripe_iml"]).size()
    if not (per_pair == k_samples).all():
        bad = per_pair[per_pair != k_samples]
        raise ValueError(f"{path} was drawn for a different k_samples: expected "
                         f"{k_samples} rows per (site, stripe), got {bad.unique()} at "
                         f"{len(bad)} pair(s)")

    df["stripe_iml"] = df["stripe_iml"].map(_iml_key)
    return df


def site_msa_index_arrays(
    index_samples: pd.DataFrame,
    n_records: int,
) -> dict[tuple[int, float], np.ndarray]:
    """``{(site, stripe_iml): (k_samples, n_records) index array}``, ordered by ``k``."""
    columns = [str(i) for i in range(n_records)]
    arrays = {}
    for (site, iml), block in index_samples.groupby(["site", "stripe_iml"]):
        arrays[(int(site), _iml_key(iml))] = (block.sort_values("k")[columns]
                                              .to_numpy(dtype=int))
    return arrays


def site_msa_stripe_alignment(
    flags: pd.DataFrame,
    fragilities: dict[int, dict],
    n_storeys: int,
) -> pd.DataFrame:
    """Compare each structure's flag stripes against the stripes its fragility was fitted to.

    The fragility JSON is the only source of the published ``median`` / ``dispersion``,
    which the bootstrap is screened against and which nb 070 section 2 measures the bias
    relative to. So the resample has to be fitted to **the same stripes**, and a stripe
    present in one and not the other is a disagreement worth naming rather than silently
    reconciling.

    Returns one row per structure with ``n_fitted``, ``n_flagged``, ``extra`` (stripes
    analysed since the fragility was fitted - dropped by
    :func:`bootstrap_site_msa_fragilities_from_flags`) and ``absent`` (fitted stripes with
    no flags, which that function raises on).
    """
    rows = []
    for site in sorted(fragilities):
        fitted = {_iml_key(v) for v in np.asarray(fragilities[site]["efc"])[0, :]}
        try:
            flagged = set(flags.loc[(site, n_storeys)].index)
        except KeyError:
            flagged = set()
        rows.append({"site": site, "n_storeys": n_storeys,
                     "n_fitted": len(fitted), "n_flagged": len(flagged),
                     "extra": sorted(flagged - fitted),
                     "absent": sorted(fitted - flagged)})
    return pd.DataFrame(rows)


def bootstrap_site_msa_fragilities_from_flags(
    flags: pd.DataFrame,
    index_samples: pd.DataFrame,
    fragilities: dict[int, dict],
    n_records: int,
    n_storeys: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resample records within each stripe and refit, for every site of one storey count.

    The record analogue of :func:`bootstrap_site_msa_fragilities`, and the SS counterpart
    of :func:`bootstrap_msa_group_fragilities`. Per structure: take the stripes its
    fragility was fitted to, index each stripe's flags by that stripe's ``(k_samples,
    n_records)`` sample, sum over the record axis for ``(k_samples, n_stripes)`` collapse
    counts, and refit every replicate with :func:`standes.fitting.lognorm_mle_fit` - the
    fit :func:`standes.fragility_curves.fragility_from_msa` performs.

    Only the stripes in ``efc`` are used, so the cloud and the published point estimate
    are fitted to the same data; :func:`site_msa_stripe_alignment` reports any stripe this
    drops. A *fitted* stripe with no flags is raised - that structure cannot be bootstrapped.

    Unlike the group arm there is no single ``samples`` array: each stripe carries its own,
    because its 30 records are its own (see the note above).

    Screening matches both other MSA paths - not :func:`msa_counts_degenerate`, and
    :func:`msa_fit_ok` against that site's published fit - with rejects returned as ``NaN``
    beside the boolean mask.

    Returns ``(theta, beta, fit_ok)``, all ``k_samples x n_sites`` with the replicate
    number as the index and **integer** site numbers as the columns.
    """
    arrays = site_msa_index_arrays(index_samples, n_records)
    k_samples = len(next(iter(arrays.values())))

    thetas, betas, oks = {}, {}, {}
    for site in sorted(fragilities):
        fc = fragilities[site]
        imls, _ = site_msa_stripe_counts(fc, n_records)
        imls = np.sort(np.asarray([_iml_key(v) for v in imls]))

        try:
            stripes = flags.loc[(site, n_storeys)]
        except KeyError as err:
            raise KeyError(f"site {site} {n_storeys}s has a fragility but no collapse "
                           f"flags - re-run nb 060 section 6") from err

        counts = np.empty((k_samples, len(imls)), dtype=int)
        for j, iml in enumerate(imls):
            if iml not in stripes.index:
                raise KeyError(f"site {site} {n_storeys}s: the fragility was fitted to a "
                               f"stripe at {iml} g with no collapse flags - re-run nb 060 "
                               "section 6, or nb 060 section 3 with FORCE_RECOMPUTE")
            idx = arrays[(site, iml)]
            counts[:, j] = stripes.loc[iml].to_numpy()[idx].sum(axis=1)

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
    im_tag: str,
    scope: str,
) -> Path:
    """Canonical filename for a saved bootstrap frame.

    ``arm`` is e.g. ``"msa_femap695"``, ``quantity`` one of ``theta``/``beta``/
    ``theta_stats``/``beta_stats``/``fit_ok``, ``scope`` ``"by_group"`` or ``"by_site"``.
    Both the save and the reload go through here so the two cannot drift apart.

    No storey count in the name: one file carries every storey count, on the second level
    of its column index for the replicate clouds and of its row index for the per-unit
    frames - the convention :func:`estimates_csv_path` already followed. Keeping the storey
    counts together is what lets a consumer ask for every structure at once.
    """
    return Path(root) / f"{arm}_bootstrap_{quantity}_{im_tag}_{scope}.csv"


def select_storeys(
    df: pd.DataFrame,
    n_storeys: int | Sequence[int] | None,
    axis: int = 0,
) -> pd.DataFrame:
    """Slice a merged frame down to one or more storey counts.

    ``None`` returns the frame untouched, an ``int`` returns a single storey count with the
    ``n_storeys`` level dropped - the single-level shape every consumer used before the
    storey counts were merged into one file - and a sequence returns a subset with the level
    kept. ``axis`` picks the index the level sits on: 0 for the per-unit frames, 1 for the
    replicate clouds.

    A storey count that is not in the frame raises rather than coming back as an empty
    slice, which is the one failure that would otherwise go unnoticed downstream.
    """
    if n_storeys is None:
        return df

    index = df.columns if axis == 1 else df.index
    if "n_storeys" not in index.names:
        raise KeyError("frame carries no n_storeys level to select on")

    levels = index.get_level_values("n_storeys")
    available = sorted(set(levels))
    scalar = isinstance(n_storeys, (int, np.integer))
    wanted = [int(n_storeys)] if scalar else [int(n) for n in n_storeys]

    missing = [n for n in wanted if n not in available]
    if missing:
        raise KeyError(f"no {missing}-storey data in this frame - it holds {available}")

    if scalar:
        return df.xs(wanted[0], level="n_storeys", axis=axis)

    keep = levels.isin(wanted)
    return df.loc[:, keep] if axis == 1 else df.loc[keep]


def read_bootstrap_frame(path: Path | str, columns_name: str) -> pd.DataFrame:
    """Read a saved replicate frame back, restoring the labels the CSV cannot carry.

    The columns are a ``(unit, n_storeys)`` MultiIndex and both of its levels come back as
    strings, so site numbers and storey counts are returned to ``int`` - group labels are
    left as they are - and the level names restored. A reloaded frame is then
    indistinguishable from a freshly computed one. Boolean masks are parsed as ``bool`` by
    ``read_csv`` and need no special handling.
    """
    df = pd.read_csv(path, index_col=0, header=[0, 1])
    units = df.columns.get_level_values(0)
    if columns_name == "site":
        units = [int(u) for u in units]
    df.columns = pd.MultiIndex.from_arrays(
        [units, [int(n) for n in df.columns.get_level_values(1)]],
        names=[columns_name, "n_storeys"])
    df.index.name = "k"
    return df


def load_saved_bootstrap(
    root: Path | str,
    arm: str,
    quantities: Sequence[str],
    n_storeys: int | Sequence[int] | None,
    im_tag: str,
    scope: str,
    k_samples: int | None = None,
) -> dict[str, pd.DataFrame] | None:
    """Reload one arm's saved replicate frames, or ``None`` if any is missing.

    The fits are deterministic, so a previous run's output can stand in for repeating
    them. Returning ``None`` on a single missing file - rather than a partial set - is
    what keeps the caller's fallback all-or-nothing: a half-finished save can never be
    silently mixed with a fresh computation. A storey count absent from an otherwise
    complete file counts as missing for the same reason.

    ``n_storeys`` selects out of the merged file the way :func:`select_storeys` describes:
    ``None`` for every structure at once, an ``int`` for the single-level frame this
    returned before the storey counts shared a file.

    ``k_samples`` guards against picking up a cloud of the wrong size, which is the one
    way a stale file could pass unnoticed after the replicate count changes.
    """
    paths = {q: bootstrap_csv_path(root, arm, q, im_tag, scope) for q in quantities}
    if not all(p.is_file() for p in paths.values()):
        return None

    columns_name = scope.removeprefix("by_")
    frames = {q: read_bootstrap_frame(p, columns_name) for q, p in paths.items()}

    try:
        frames = {q: select_storeys(df, n_storeys, axis=1) for q, df in frames.items()}
    except KeyError:
        return None

    if k_samples is not None:
        for q, df in frames.items():
            if len(df) != k_samples:
                raise ValueError(f"saved {arm} {q} has {len(df)} replicates, not "
                                 f"k_samples={k_samples} - delete it or set "
                                 f"REUSE_SAVED = False to refit")
    return frames

# load the theta and beta bootstrap stats
def load_saved_bootstrap_stats(
    root: Path | str,
    arm: str,
    quantities,
    n_storeys: int | Sequence[int] | None,
    im_tag: str,
    scope: str,
    ) -> dict[str, pd.DataFrame] | None:
    """Reload one arm's saved summary-statistic frames, or ``None`` if any is missing.

    The fits are deterministic, so a previous run's output can stand in for repeating
    them. Returning ``None`` on a single missing file - rather than a partial set - is
    what keeps the caller's fallback all-or-nothing: a half-finished save can never be
    silently mixed with a fresh computation.

    The ``(unit, n_storeys)`` row index comes off disk rather than being attached here;
    ``n_storeys`` selects out of it as :func:`select_storeys` describes, so passing a
    single storey count still gives the flat frame this returned before the merge.
    """
    paths = {q: bootstrap_csv_path(root, arm, q, im_tag, scope) for q in quantities}
    if not all(p.is_file() for p in paths.values()):
        return None

    columns_name = "stats"
    index_name = scope.removeprefix("by_")
    frames = {q: read_bootstrap_stats_frame(p, columns_name, index_name)
              for q, p in paths.items()}

    try:
        return {q: select_storeys(df, n_storeys) for q, df in frames.items()}
    except KeyError:
        return None


def read_bootstrap_stats_frame(path: Path | str, columns_name: str,
                               index_name: str) -> pd.DataFrame:
    """Read a saved summary-statistic frame back, restoring the labels the CSV cannot carry.

    The rows are a ``(unit, n_storeys)`` MultiIndex, restored by
    :func:`_restore_unit_index`, and ``columns.name`` is lost on the round trip and set
    back here, so a reloaded frame is indistinguishable from a freshly computed one.
    """
    df = pd.read_csv(path, index_col=[0, 1])
    df = _restore_unit_index(df, index_name)
    df.columns.name = columns_name
    return df


def reformat_bootstrap_df(df: pd.DataFrame, n_storeys: int | None = None):
    """Reformat the bootstrap theta and beta dataframes to have multi-index rows.

    A merged cloud already carries ``(unit, n_storeys)`` on its columns, so transposing is
    the whole job. ``n_storeys`` is only needed for a frame that has had the level dropped
    - what :func:`load_saved_bootstrap` returns when asked for a single storey count - and
    attaches it back as a constant second level.
    """
    if isinstance(df.columns, pd.MultiIndex):
        return df.T

    if n_storeys is None:
        raise ValueError("single-level columns need an explicit n_storeys to reattach")

    row_index = pd.MultiIndex.from_product([df.columns, [n_storeys]],
                                           names=[df.columns.name or "site", "n_storeys"])
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


def stack_unit_frames(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-storey frames indexed by site or group onto one ``(unit, n_storeys)`` index.

    ``frames`` is keyed by storey count, each value indexed by site or group - the shape of
    a stats frame, a bias frame or a point-estimate frame. The result carries every storey
    count in one object, indexed the way the meta-regression wants its rows, and is sorted
    so that a per-unit frame and a transposed replicate cloud line up row for row.
    """
    parts = []
    for n, df in frames.items():
        part = df.copy()
        part.index = pd.MultiIndex.from_product(
            [df.index, [n]], names=[df.index.name or "unit", "n_storeys"])
        parts.append(part)

    return pd.concat(parts).sort_index()


def stack_estimates(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Stack one arm's per-storey point estimates into a single ``(unit, n_storeys)`` frame.

    A :func:`stack_unit_frames` narrowed to the two columns the meta-regression reads, so a
    summary frame carrying extra columns cannot leak them into the saved estimates.
    """
    return stack_unit_frames({n: df[["theta", "beta"]] for n, df in frames.items()})


def stack_cloud_frames(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-storey replicate clouds onto one ``(unit, n_storeys)`` column MultiIndex.

    The column-side counterpart of :func:`stack_unit_frames`: ``frames`` is keyed by storey
    count, each value a ``k`` x unit cloud, and the result holds every structure in one
    frame with the replicate index shared. Sorted for the same reason, so transposing it
    reproduces :func:`stack_unit_frames`' row order exactly.
    """
    parts = []
    for n, df in frames.items():
        part = df.copy()
        part.columns = pd.MultiIndex.from_product(
            [df.columns, [n]], names=[df.columns.name or "unit", "n_storeys"])
        parts.append(part)

    out = pd.concat(parts, axis=1).sort_index(axis=1)
    out.index.name = "k"
    return out


def estimates_csv_path(root: Path | str, arm: str, im_tag: str, scope: str) -> Path:
    """Canonical filename for a saved point-estimate frame.

    No storey count in the name: unlike the replicate clouds a single file carries every
    storey count, on the second level of its index.
    """
    return Path(root) / f"{arm}_estimates_{im_tag}_{scope}.csv"


def _restore_unit_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Return a ``(unit, n_storeys)`` row index to the types a CSV cannot carry.

    Site numbers and storey counts both come back as strings; both are returned to ``int``
    so the frame is indistinguishable from a freshly built one. Group labels are left as
    they are. Shared by every per-unit frame - estimates, stats and bias - since the merge
    gave all three the same index.
    """
    units = df.index.get_level_values(0)
    if index_name == "site":
        units = [int(u) for u in units]
    df.index = pd.MultiIndex.from_arrays(
        [units, [int(n) for n in df.index.get_level_values(1)]],
        names=[index_name, "n_storeys"])
    return df


def read_estimates_frame(path: Path | str,
                         index_name: str = "site") -> pd.DataFrame:
    """Read a saved point-estimate frame back, restoring its two-level index."""
    return _restore_unit_index(pd.read_csv(path, index_col=[0, 1]), index_name)


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


def bias_csv_path(root: Path | str, arm: str, quantity: str,
                  im_tag: str, scope: str) -> Path:
    """Canonical filename for a saved bias frame.

    Goes through :func:`bootstrap_csv_path` with ``quantity`` suffixed ``_bias``, so the
    bias files sit alongside the clouds they were computed from under the same convention.
    """
    return bootstrap_csv_path(root, arm, f"{quantity}_bias", im_tag, scope)


def read_bias_frame(path: Path | str, index_name: str = "site") -> pd.DataFrame:
    """Read a saved bias frame back, restoring the index labels the CSV cannot carry.

    Unlike :func:`read_bootstrap_frame` the rows here are sites or groups, not replicates,
    so the ``(unit, n_storeys)`` MultiIndex sits on the *index* rather than the columns.
    """
    return _restore_unit_index(pd.read_csv(path, index_col=[0, 1]), index_name)


def load_saved_bias(root: Path | str, arms: Sequence[str],
                    quantities: Sequence[str],
                    n_storeys: int | Sequence[int] | None, im_tag: str,
                    scope: str) -> dict[str, dict[str, pd.DataFrame]] | None:
    """Reload the saved bias frames of several arms, or ``None`` if any is missing.

    All-or-nothing for the same reason :func:`load_saved_bootstrap` is: a half-written set
    must never be silently mixed with a freshly computed one. ``n_storeys`` selects out of
    the merged files as :func:`select_storeys` describes.
    """
    paths = {(arm, q): bias_csv_path(root, arm, q, im_tag, scope)
             for arm in arms for q in quantities}
    if not all(p.is_file() for p in paths.values()):
        return None

    index_name = scope.removeprefix("by_")
    frames: dict[str, dict[str, pd.DataFrame]] = {arm: {} for arm in arms}
    try:
        for (arm, q), path in paths.items():
            frames[arm][q] = select_storeys(read_bias_frame(path, index_name), n_storeys)
    except KeyError:
        return None
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


def shared_figure_legend(fig, ax, y: float, ncol: int, **kwargs) -> None:
    """Attach one figure-level legend as a band across the top of the axes.

    Every panel of a multi-panel figure draws the same set of series, so a legend per
    panel is several copies of the same key competing with the data for space. This takes
    the handles off ``ax`` - whichever panel carries the full set - drops any repeated
    label (matplotlib keeps the first occurrence, so the drawing order is preserved) and
    puts the result above the axes, under the figure's suptitle.

    ``y`` is the legend's top edge in figure coordinates and must sit inside the strip the
    caller reserved with ``tight_layout(rect=...)``, between the axes and the suptitle;
    anchoring outside the canvas leaves a legend that survives only a
    ``bbox_inches="tight"`` save.
    """
    handles, labels = ax.get_legend_handles_labels()
    seen, unique = set(), []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique.append((handle, label))

    leg = fig.legend(*zip(*unique), loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=ncol, **kwargs)
    leg.get_frame().set_edgecolor("k")


def storey_suptitle(fig, text: str, n_storeys: int | None) -> None:
    """Title the whole figure, naming the storey count it was drawn for.

    Both diagnostic figures put the site number on the x axis and so show one storey
    count at a time. Drawn side by side in a notebook they are otherwise identical, so
    the suptitle is the only thing telling the reader which is which.
    """
    if n_storeys is not None:
        text = f"{text} - {n_storeys}-storey structures"
    fig.suptitle(text, fontsize="large", fontweight="bold")


def _plot_arm_series(ax, series_by_arm: dict[str, pd.Series], ylabel: str,
                     means: bool = False, se_bands: bool = False,
                     mcse_line: bool = False, legend: bool = True) -> None:
    """Scatter one diagnostic against the site number, one series per arm.

    ``means`` adds each arm's across-site mean as a dashed line of its own colour;
    ``se_bands`` and ``mcse_line`` add the two materiality thresholds. ``legend=False``
    suppresses the per-panel key, for a grid that carries one shared legend instead - the
    labels are still set on the artists, so :func:`shared_figure_legend` can collect them.
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

    if legend:
        style_legend(ax, fontsize="small", ncol=2 if means else 1)


def plot_bias_assessment(
    bias_data: dict[str, dict[str, pd.DataFrame]],
    references: dict[str, pd.DataFrame] | None = None,
    contrasts: Sequence[tuple[str, str]] | None = None,
    n_storeys: int | None = None,
) -> tuple[plt.Figure, np.ndarray, plt.Figure, plt.Axes]:
    """Chart the per-site bias of every arm, before and after correction.

    ``bias_data`` is keyed ``[arm]["theta"|"beta"]`` and every frame must already carry the
    ``bias_corrected`` columns :func:`add_bias_correction` adds. The 3x2 grid is arranged
    as columns ``ln theta`` / ``ln beta``, and rows raw bias / bias-to-se / bias-to-mcse.

    Both figures put the site number on the x axis, so they show one storey count at a
    time: pass ``n_storeys`` to pick it out of frames that still carry the level, and to
    name it in the suptitles - the two figures are otherwise indistinguishable when the
    storey counts are drawn one after the other. Plotting every storey count together
    would stack two structures on each x position, which is why a merged frame with no
    selection raises rather than drawing something misleading.

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

    merged = any("n_storeys" in df.index.names
                 for quantities in bias_data.values() for df in quantities.values())
    if merged:
        if n_storeys is None:
            available = sorted({n for quantities in bias_data.values()
                                for df in quantities.values()
                                for n in df.index.get_level_values("n_storeys")})
            raise ValueError(f"these bias frames hold storey counts {available} - pass "
                             f"n_storeys to chart one of them")
        bias_data = {arm: {q: select_storeys(df, n_storeys)
                           for q, df in quantities.items()}
                     for arm, quantities in bias_data.items()}

    fig1, axs1 = plt.subplots(3, 2, figsize=(11, 11))

    # the corrected ln(beta) column was dropped: every arm now subtracts its own bias, so
    # its residual is zero by construction and the panel carried no information
    columns = [("theta", r"Bias in $\ln{\theta}$", r"Bias [$\ln{\theta}$ units]"),
               ("beta", r"Bias in $\ln{\beta}$", r"Bias [$\ln{\beta}$ units]")]

    for j, (quantity, title, ylabel) in enumerate(columns):
        _plot_arm_series(axs1[0, j],
                         {a: bias_data[a][quantity]["bias"] for a in arms},
                         ylabel, means=True, legend=False)
        _plot_arm_series(axs1[1, j],
                         {a: bias_data[a][quantity]["bias/se"] for a in arms},
                         "Bias / S.E. [-]", se_bands=True, legend=False)
        _plot_arm_series(axs1[2, j],
                         {a: bias_data[a][quantity]["bias/mcse"] for a in arms},
                         "Bias / MCSE [-]", mcse_line=True, legend=False)
        axs1[0, j].set_title(title)

    # every panel draws the same arms, and the top-left one additionally carries the
    # dashed across-site means, so it holds the full key for the whole grid
    storey_suptitle(fig1, "Estimator bias diagnostics", n_storeys)
    fig1.tight_layout(rect=(0, 0, 1, 0.90))
    shared_figure_legend(fig1, axs1[0, 0], 0.945, ncol=3, fontsize="small")

    # wider than the panel it used to be: the contrast labels are long and the legend
    # now takes a strip of the figure to the right of the axes
    fig2, axs2 = plt.subplots(figsize=(12, 6))
    axs2.axhline(0, ls="-", color="k", lw=0.75)

    n_sites = len(bias_data[arms[0]]["beta"])
    print(f"Net bias on the ln(beta) contrasts, over {n_sites} sites:")
    _contrast_styles = [("o", "tab:blue"), ("s", "tab:green"), ("^", "tab:purple"),
                        ("v", "tab:brown"), ("D", "tab:pink")]
    if len(contrasts) > len(_contrast_styles):
        raise ValueError(f"{len(contrasts)} contrasts but only {len(_contrast_styles)} "
                         f"styles - pass an explicit `contrasts` selection")

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

    storey_suptitle(fig2, r"Total bias on the $\ln{\beta}$ contrasts", n_storeys)
    fig2.tight_layout(rect=(0, 0, 1, 0.80))
    shared_figure_legend(fig2, axs2, 0.915, ncol=2, fontsize="small")

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


def compute_Q_pvalue(Q: float, df: int) -> float:
    """p-value of the test of homogeneity: P(chi^2_df >= Q).

    follows Borenstein et al. "Introduction to Meta-Analysis" Ch. 16 (~p. 112)

    Under the null that every study shares one true effect, Q is distributed as
    chi-square on df = k - 1. That null reference is exact only when the sampling
    errors are independent - Q is built from the diagonal variances alone. When rows
    share sampling error (in this project, every row in a design group subtracting
    the same b_j or c_j) Q is inflated and this p-value is optimistic; treat it as
    nominal.
    """
    return float(chi2.sf(Q, df))


def compute_typical_within_variance(Wis: pd.Series|np.ndarray) -> float:
    """The "typical" within-study variance s^2 of Higgins & Thompson (2002).

        s^2 = (k - 1) sum(w_i) / ((sum w_i)^2 - sum(w_i^2)) = (k - 1) / C

    with fixed-effect weights w_i = 1 / v_i and C from Borenstein Eq. 12.5.

    It is the single sampling variance that stands in for the k different v_i, which
    is what lets I^2 be written for ANY estimate of the between-study variance as
    T^2 / (s^2 + T^2) - see :func:`compute_I_sq_from_tau`.

    Higgins, J.P.T. & Thompson, S.G. (2002) "Quantifying heterogeneity in a
    meta-analysis", Statistics in Medicine 21:1539-1558.
    """
    k = len(Wis)
    return float((k - 1) / compute_C(Wis))


def compute_I_sq_from_tau(T_sq: float, s_sq: float) -> float:
    """I^2 (in percent) from a between-study variance and the typical within variance.

        I^2 = T^2 / (s^2 + T^2)

    The proportion of the total variance that is between-study rather than sampling
    error. Unlike the Q-based :func:`compute_I_sq` it does not tie I^2 to the
    DerSimonian-Laird moment estimator, so it can be used with a REML T^2.

    For the DL estimate it reproduces :func:`compute_I_sq` exactly: T^2_DL =
    (Q - df) / C and s^2 = df / C, so T^2 / (s^2 + T^2) = (Q - df) / Q - and both give
    0 when T^2 is truncated at zero.

    Returned in percent, to match :func:`compute_I_sq`.
    """
    if T_sq < 0:
        raise ValueError(f"T_sq must be non-negative, got {T_sq}")
    return float(100.0 * T_sq / (s_sq + T_sq))


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


def _align_studies(
        Vis: pd.Series|np.ndarray,
        Yis: pd.Series|np.ndarray,
        ) -> tuple[pd.Series, pd.Series]:
    """Put the two inputs on one shared index as float Series, without filtering.

    A bare array takes the other argument's labels so the two stay aligned; two
    labelled inputs must already agree, because silently re-labelling one of them is
    how a variance ends up against the wrong study. With no labels on either side the
    position is the label, so a plain ndarray comes back on a ``RangeIndex`` and any
    index reported downstream is its numpy index.
    """
    if len(Vis) != len(Yis):
        raise ValueError(f"Vis has {len(Vis)} studies but Yis has {len(Yis)}")

    index = Vis.index if isinstance(Vis, pd.Series) else (
        Yis.index if isinstance(Yis, pd.Series) else pd.RangeIndex(len(Vis)))
    if (isinstance(Vis, pd.Series) and isinstance(Yis, pd.Series)
            and not Vis.index.equals(Yis.index)):
        raise ValueError("Vis and Yis are both labelled but their indexes differ")

    return (pd.Series(np.asarray(Vis, dtype=float), index=index),
            pd.Series(np.asarray(Yis, dtype=float), index=index))


def _label_list(index: pd.Index, limit: int = 10) -> str:
    """Render offending labels for an error message, truncated so it stays readable."""
    shown = ", ".join(repr(label) for label in index[:limit])
    return shown if len(index) <= limit else f"{shown}, ... (+{len(index) - limit} more)"


def validate_studies(
        Vis: pd.Series|np.ndarray,
        Yis: pd.Series|np.ndarray,
        ) -> tuple[pd.Series, pd.Series]:
    """Refuse a study set that is not fit to be summed, naming what is wrong with it.

    A screened-out replicate (see :func:`msa_fit_ok`) arrives here as ``NaN``, and a
    zero or negative variance would divide by zero in Eq. 11.2. Neither fails on its
    own: ``np.sum`` over a pandas Series skips ``NaN`` silently, so an unvalidated
    frame quietly computes Q and C over a different set of studies than the caller
    believes it passed in. Deciding which studies to exclude is a data-cleaning
    judgement that belongs to the analysis, not to the estimator, so the estimators
    raise here instead of repairing the input. Run :func:`drop_incomplete_studies`
    first, explicitly, when studies really do need removing.
    """
    Vis, Yis = _align_studies(Vis, Yis)

    bad_Y = Yis.index[~np.isfinite(Yis)]
    if len(bad_Y):
        raise ValueError(
            f"Yis is not finite for {len(bad_Y)} of {len(Yis)} studies: "
            f"{_label_list(bad_Y)}. Filter the input explicitly "
            f"(see drop_incomplete_studies) before fitting.")

    bad_V = Vis.index[~np.isfinite(Vis)]
    if len(bad_V):
        raise ValueError(
            f"Vis is not finite for {len(bad_V)} of {len(Vis)} studies: "
            f"{_label_list(bad_V)}. Filter the input explicitly "
            f"(see drop_incomplete_studies) before fitting.")

    non_positive = Vis.index[Vis <= 0]
    if len(non_positive):
        raise ValueError(
            f"Vis must be strictly positive; {len(non_positive)} of {len(Vis)} "
            f"studies are <= 0: {_label_list(non_positive)}. A non-positive sampling "
            f"variance divides by zero in the inverse-variance weights.")

    if len(Vis) < 2:
        raise ValueError(
            f"a random-effects fit needs at least 2 studies, got {len(Vis)}")

    return Vis, Yis


def drop_incomplete_studies(
        Vis: pd.Series|np.ndarray,
        Yis: pd.Series|np.ndarray,
        ) -> tuple[pd.Series, pd.Series, pd.Index]:
    """Keep only the studies that carry a usable (variance, effect) pair.

    This is the explicit pre-filter to run *before* the estimators, which validate
    rather than clean (see :func:`validate_studies`). The third return value is the
    index of everything it took out, so the exclusion is visible to the caller and can
    be reported or checked rather than happening invisibly inside a fit.

    Returns ``(Vis_kept, Yis_kept, dropped_index)``. For labelled input the dropped
    index carries the Series labels; for a bare ndarray it carries the positional
    numpy indices of the rows that were removed.
    """
    Vis, Yis = _align_studies(Vis, Yis)

    keep = np.isfinite(Vis) & np.isfinite(Yis) & (Vis > 0)
    return Vis[keep], Yis[keep], Vis.index[~keep]


def compute_re_summary_effect(
        Vis: pd.Series|np.ndarray,
        Yis: pd.Series|np.ndarray,
        ) -> pd.Series|np.ndarray:
    """ The estimated between-group variance.

    follows Borenstein et al. "Introduction to Meta-Analysis"
    Eq. 12.7, 12.8, 12.9

    Every study passed in is used. A ``NaN`` effect or variance, or a variance that is
    not strictly positive, raises: which studies to exclude is a decision for the
    analysis to make explicitly, up front, with :func:`drop_incomplete_studies`. That
    way the degrees of freedom behind T_sq always match the frame the caller thinks it
    handed over.

    returns the mean effects, the variance of the mean effect, standard error,
    the between study variance (T_sq) and the weights
    """
    Vis, Yis = validate_studies(Vis, Yis)
    k = len(Vis)

    Wis_fe = get_fe_weights(Vis)
    T_sq = compute_Tsquared(Wis_fe, Yis, k)
    Wis_re = get_re_weights(Vis, T_sq)
    M_re = summary_effect_re(Wis_re, Yis)
    V_re = summary_variance_re(Wis_re)
    SE_re = np.sqrt(V_re)

    return M_re, V_re, SE_re, T_sq, Wis_re


def compute_re_summary_effect2(df:pd.DataFrame, effect_column="effect", variance_column="variance") -> pd.DataFrame:
    """The input dataframe should have  """

    Yis = df[effect_column]
    Vis = df[variance_column]

    M_re, V_mre, SE_mre, T_sq, Wis_re = compute_re_summary_effect(Vis, Yis)

    T_sq = pd.Series({i:T_sq for i in Wis_re.index}, name="T2")
    Var_tot = Vis + T_sq
    Var_tot.name = "Var_tot"
    Wis_re.name = "W_re"
    study_values = pd.concat([Yis, Vis, T_sq, Var_tot, Wis_re], axis=1)

    return M_re, V_mre, SE_mre, study_values




def compute_prediction_interval():
    """Computes the prediction interval for the random effects model
    using the estimate of the variance and hte t-distributions with n-2 dofs
    
    According to Borenstein et al. "Introduction to Meta-Analysis" eq. 17.7, 17.8"""

    # TODO::
    ...


def compute_heterogeneity_stats(
        Vis: pd.Series|np.ndarray, Yis: pd.Series|np.ndarray,
        T_sq: float | None = None) -> dict[str, float]:
    """Heterogeneity statistics for an intercept-only random-effects fit.

    Q, its degrees of freedom and its p-value depend only on the data and the
    fixed-effect weights w_i = 1 / v_i, so they are the SAME whichever estimator
    produced the between-study variance. Only T^2 (and hence T and I^2) differ.

    Parameters
    ----------
    Vis, Yis : Series or ndarray
        Within-study variances and effect sizes.
    T_sq : float, optional
        Between-study variance to report. ``None`` (default) computes the
        DerSimonian-Laird estimate (Borenstein Eq. 12.2) and the Q-based I^2
        (Eq. 16.9) - the previous behaviour, unchanged. Pass an estimate from
        another method (e.g. the REML total) to get T and I^2 for that estimate;
        I^2 is then T^2 / (s^2 + T^2) with the Higgins & Thompson typical
        within-study variance s^2, which reduces to Eq. 16.9 when T^2 is DL's.

    Returns
    -------
    dict with keys
        Q      weighted sum of squared deviations (Eq. 16.3)
        df     k - 1
        Q_df   Q - df. NOTE: the EXCESS of Q over its null expectation, NOT the
               degrees of freedom - the name is kept for existing callers. A
               negative value means less dispersion than sampling error alone
               predicts, and DL then truncates T^2 at zero.
        p      P(chi^2_df >= Q), Borenstein Ch. 16 (~p. 112)
        s_sq   typical within-study variance (Higgins & Thompson 2002)
        T_sq   between-study variance (DL, or as passed in)
        T      sqrt(T_sq)
        I_sq   percent

    Caveats
    -------
    Q and s^2 use the DIAGONAL variances only. The chi-square null for Q holds
    for independent sampling errors; where rows share sampling error (every row
    in a design group subtracting the same b_j or c_j) Q is inflated and ``p`` is
    optimistic - read it as nominal. And these are the intercept-only forms: for a
    meta-regression the residual Q_E has k - p degrees of freedom instead.
    """
    Vis, Yis = validate_studies(Vis, Yis)
    n_studies = len(Vis)
    df = n_studies - 1
    Wis_fe = get_fe_weights(Vis)

    # Q, and its excess over the null expectation df
    Q = compute_Q(Wis_fe, Yis)
    Q_df = Q - df

    # p-value from the chi-square null
    p = compute_Q_pvalue(Q, df)

    # typical within-study variance - estimator-independent
    s_sq = compute_typical_within_variance(Wis_fe)

    # T^2 and I^2: DL's own if none was supplied, otherwise from the given T^2
    if T_sq is None:
        T_sq = compute_Tsquared(Wis_fe, Yis, n_studies)
        I_sq = compute_I_sq(Wis_fe, Yis)
    else:
        T_sq = float(T_sq)
        I_sq = compute_I_sq_from_tau(T_sq, s_sq)

    T = np.sqrt(T_sq)

    return {"Q": Q, "df": df, "Q_df": Q_df, "p": p, "s_sq": s_sq,
            "T_sq": T_sq, "T": T, "I_sq": I_sq}


# ===========================================================================
# REML fitting for the A1 family: A1a (2-level), A1b (3-level, rows nested in
# design groups) and A1c (crossed design x site), all with a KNOWN - and for
# A1b/A1c dense - sampling covariance matrix.
#
#     A1a  y_i = m1                            + u_i + eps_i
#     A1b  y_i = m1 + d_{g[i]}                 + u_i + eps_i
#     A1c  y_i = m1 + d_{g[i]} + s_{sigma[i]}  + u_i + eps_i
#
#     Sigma = V_known + tau_d^2 Z Z' + tau_s^2 S S' + tau_u^2 I
#
# All three are the same model with a different number of variance components,
# so one routine (fit_reml) serves them all: it takes a LIST of known n x n
# matrices, ``Gs``, and estimates one variance component per entry. A component
# is dropped by leaving its matrix out of the list - never by mangling Z or S
# (one site for everybody makes S S' a matrix of ones, confounded with the
# fixed intercept; one site per row makes S S' = I, aliased with tau_u^2 I).
#
#     A1a   Gs = [I]                  V_known = diag(v1)
#     A1b   Gs = [ZZt, I]             V_known = diag(v_a) + Z V_bb Z'
#     A1c   Gs = [ZZt, SSt, I]        V_known = diag(v_a) + Z V_bb Z'
#
# where
#     v1[i]      = Var_r(a_i^(r) - b_{g[i]}^(r))   combined, per ROW
#     v_a[i]     = Var_r(ln theta_SS_i^(r))        SS arm only, per ROW
#     V_bb[j,j'] = Cov_r(ln theta_FX_j^(r), ln theta_FX_j'^(r))   per DESIGN
# and exactly (the arms use disjoint record sets, so are independent)
#     v1[i] = v_a[i] + V_bb[g[i], g[i]]
#
# WHY REML AND NOT DerSimonian-Laird. DL is a single moment equation,
# E[Q] = (n-1) + C tau^2: one equation, one unknown. A1b has two variance
# components and A1c has three, so DL cannot identify them. Worse, Q itself
# presupposes a DIAGONAL sampling covariance, so once Z V_bb Z' enters Sigma
# there is no Q to write down. REML's objective is stated directly in terms of
# Sigma and is indifferent to both issues. Keep the DL functions above as the
# regression test: fit_reml with Gs = [I] and a diagonal V_known must reproduce
# them.
#
# IDENTIFIABILITY WARNING. Every row in design group j subtracts the SAME
# estimate b_j, so the sampling error contributes v^b_j to every within-group
# off-diagonal cell of Sigma - the same block pattern as tau_d^2 Z Z'. The two
# are aliased, separable only through the variation of v^b_j across j. There is
# a shared term on the site side too: the 3-storey and 5-storey structures at a
# site share records wherever they are analysed at the same stripe IML, which
# does happen, so V_aa is block-diagonal by site rather than diagonal. Do not
# assume the two grouping factors are treated asymmetrically by the sampling
# covariance. Always pass the full V_known for A1b/A1c, built as the complete
# sample covariance of the replicate matrices,
#     V_known = Cov_r(a) + Z Cov_r(b) Z'
# rather than from their diagonals; and note that np.cov can only report the
# covariance the RESAMPLING SCHEME generates -- the replicates must share record
# draws wherever the underlying analyses shared records, or the coupling reads as
# zero however the covariance is computed.
#
# Borenstein et al. (2009) Ch. 12 pp. 69-75; Ch. 14 pp. 87-95.
# Gelman & Hill (2007) Ch. 13.5 ~pp. 289-291 (crossed / non-nested);
#   Ch. 22 ~pp. 487-500 (comparing variance components).
# Gelman et al. BDA3 Ch. 5.4-5.5 ~pp. 113-124 (known-variance hierarchical
#   normal model).
# ===========================================================================

_REML_ZERO_TOL = 1e-8


# ---------------------------------------------------------------------------
# Design-matrix construction
# ---------------------------------------------------------------------------

def make_indicator(codes: np.ndarray, n_levels: int | None = None) -> np.ndarray:
    """Build an n x K 0/1 indicator (dummy) matrix from integer group codes.

    Parameters
    ----------
    codes : array of int, shape (n,)
        ``codes[i]`` is the 0-based group index of row i.
    n_levels : int, optional
        Number of columns K. Defaults to ``codes.max() + 1``. Pass it
        explicitly whenever a level might be unused in this subset of rows --
        otherwise the matrix silently loses a column and stops lining up with
        V_bb.

    Returns
    -------
    ndarray, shape (n, K), with exactly one 1.0 per row.

    Notes
    -----
    ``Z @ Z.T`` is then the n x n matrix with a 1 wherever two rows share a
    group -- which is the "known matrix" multiplying tau_d^2 in Sigma.
    """
    codes = np.asarray(codes, dtype=int)
    if codes.ndim != 1:
        raise ValueError(f"codes must be 1-D, got shape {codes.shape}")
    if codes.min() < 0:
        raise ValueError("codes must be 0-based non-negative integers")

    K = int(codes.max()) + 1 if n_levels is None else int(n_levels)
    if codes.max() >= K:
        raise ValueError(f"code {codes.max()} exceeds n_levels={K}")

    Z = np.zeros((codes.size, K))
    Z[np.arange(codes.size), codes] = 1.0

    unused = np.flatnonzero(Z.sum(axis=0) == 0)
    if unused.size:
        warnings.warn(
            f"{unused.size} level(s) of this factor have no rows: {unused.tolist()}. "
            "That is fine for Z Z' but means V_bb carries designs you are not "
            "fitting -- check this is intentional.", RuntimeWarning, stacklevel=2)
    return Z


def build_codes(row_labels: Sequence, level_labels: Sequence) -> np.ndarray:
    """Map row group LABELS onto 0-based codes in a FIXED, given level order.

    This exists to prevent the single most likely silent bug in the whole
    pipeline: aligning rows to V_bb positionally. The order in which designs
    appear in your ``by_group`` bootstrap object need not match the order rows
    appear in the ``by_site`` object. Always map by label.

    Parameters
    ----------
    row_labels : sequence of length n
        The design (or site) label of each row, e.g. from the row MultiIndex.
    level_labels : sequence of length K
        The canonical level order -- for designs this MUST be the row/column
        order of V_bb, i.e. ``list(bt_gr.index)``.

    Returns
    -------
    ndarray of int, shape (n,), values in 0..K-1.
    """
    lookup = {lab: j for j, lab in enumerate(level_labels)}
    missing = sorted({lab for lab in row_labels if lab not in lookup})
    if missing:
        raise KeyError(
            f"{len(missing)} row label(s) are absent from level_labels: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
            "The by_site and by_group bootstrap objects disagree.")
    return np.array([lookup[lab] for lab in row_labels], dtype=int)


def bootstrap_cov(replicates: np.ndarray, rowvar: bool = True) -> np.ndarray:
    """Sampling covariance matrix estimated from stored bootstrap replicates.

        V[j, j'] = 1/(k-1) * sum_r (x_j^(r) - xbar_j)(x_j'^(r) - xbar_j')

    Parameters
    ----------
    replicates : ndarray
        If ``rowvar`` is True (default): shape (J, k) -- one ROW per design,
        one COLUMN per bootstrap replicate. This matches ``np.log(theta_mgr)``
        in notebook 073, which is (groups x replicates).
        If False: shape (k, J).
    rowvar : bool
        Orientation flag, passed through to ``np.cov``.

    Returns
    -------
    ndarray, shape (J, J).

    Notes
    -----
    The result is DENSE, and that is the physically correct answer, not a
    numerical artefact: every design is fitted to the same fixed record set
    using the same bootstrap index sets, so a replicate that happens to draw
    strong records shifts every b_j in the same direction. Independent errors
    average away at 1/sqrt(n); this shared component does not average away at
    all, which is why the analytic SE from a diagonal V is too small.
    """
    V = np.cov(np.asarray(replicates, dtype=float), rowvar=rowvar)
    return np.atleast_2d(V)


def check_V_bb(V_bb: np.ndarray, n_replicates: int | None = None,
               verbose: bool = True) -> dict:
    """Sanity-check an estimated sampling covariance matrix before use.

    Returns a dict of diagnostics and (if ``verbose``) prints them.

    What to look for
    ----------------
    mean_offdiag_corr
        MUST be clearly positive. If it is near zero the replicate loop is not
        sharing record indices across designs, and the entire premise of the
        A1c contamination argument collapses. (An observed
        SE_bootstrap / SE_analytic ratio well above 1 implies it should be
        solidly positive.)
    min_eigenvalue
        Must be >= 0 up to floating-point noise. A sample covariance from k
        replicates has rank at most k-1, so k must comfortably exceed J.
    k_over_J
        You are estimating J(J+1)/2 distinct entries from k replicates. Even
        when invertible, individual off-diagonals are noisy if this is small;
        consider a one-factor approximation
        V_bb ~ lambda lambda' + diag(psi) in that case.
    """
    V_bb = np.asarray(V_bb, dtype=float)
    J = V_bb.shape[0]
    if V_bb.shape != (J, J):
        raise ValueError(f"V_bb must be square, got {V_bb.shape}")

    sym_err = float(np.abs(V_bb - V_bb.T).max())
    eig_min = float(np.linalg.eigvalsh((V_bb + V_bb.T) / 2).min())

    sd = np.sqrt(np.diag(V_bb))
    R = V_bb / np.outer(sd, sd)
    offdiag = R[~np.eye(J, dtype=bool)]

    out = {
        "J": J,
        "symmetry_error": sym_err,
        "min_eigenvalue": eig_min,
        "mean_diag": float(np.mean(np.diag(V_bb))),
        "mean_offdiag_corr": float(offdiag.mean()),
        "min_offdiag_corr": float(offdiag.min()),
        "max_offdiag_corr": float(offdiag.max()),
        "n_distinct_entries": J * (J + 1) // 2,
        "k_over_J": None if n_replicates is None else n_replicates / J,
    }

    if verbose:
        print("V_bb diagnostics")
        print(f"  shape                 : {J} x {J}")
        print(f"  symmetry error        : {sym_err:.3e}   (want ~0)")
        print(f"  min eigenvalue        : {eig_min:+.3e}   (want >= 0)")
        print(f"  mean diagonal (v^b)   : {out['mean_diag']:.5f}")
        print(f"  mean off-diag corr    : {out['mean_offdiag_corr']:+.4f}"
              f"   <-- MUST be clearly > 0")
        print(f"  off-diag corr range   : [{out['min_offdiag_corr']:+.3f},"
              f" {out['max_offdiag_corr']:+.3f}]")
        if n_replicates is not None:
            print(f"  k / J                 : {n_replicates} / {J}"
                  f" = {out['k_over_J']:.1f}   (want >> 1)")

    if sym_err > 1e-9:
        warnings.warn("V_bb is not symmetric", RuntimeWarning, stacklevel=2)
    if eig_min < -1e-10:
        warnings.warn(f"V_bb is not PSD (min eigenvalue {eig_min:.2e})",
                      RuntimeWarning, stacklevel=2)
    if out["mean_offdiag_corr"] < 0.01:
        warnings.warn(
            "mean off-diagonal correlation of V_bb is ~0. The bootstrap "
            "replicates do not appear to share record indices across designs. "
            "Check the replicate loop before fitting A1b/A1c.",
            RuntimeWarning, stacklevel=2)
    return out


# def build_V_known(v_a: np.ndarray, Z: np.ndarray | None = None,
#                   V_bb: np.ndarray | None = None) -> np.ndarray:
#     """Assemble the known (parameter-free) part of Sigma.

#     Parameters
#     ----------
#     v_a : ndarray, shape (n,)
#         For A1b/A1c: the SS-arm sampling variance per row,
#         ``Var_r(ln theta_SS_i^(r))``.
#         For A1a: pass the COMBINED variance v1 and leave Z / V_bb as None.
#     Z : ndarray, shape (n, J), optional
#         Design indicator.
#     V_bb : ndarray, shape (J, J), optional
#         MSA-FX sampling covariance over designs.

#     Returns
#     -------
#     ndarray, shape (n, n):  diag(v_a)                    if Z/V_bb omitted
#                             diag(v_a) + Z V_bb Z'        otherwise
#     """
#     v_a = np.asarray(v_a, dtype=float).ravel()
#     C = np.diag(v_a)
#     if (Z is None) != (V_bb is None):
#         raise ValueError("pass both Z and V_bb, or neither")
#     if Z is not None:
#         Z = np.asarray(Z, dtype=float)
#         V_bb = np.asarray(V_bb, dtype=float)
#         if Z.shape[1] != V_bb.shape[0]:
#             raise ValueError(
#                 f"Z has {Z.shape[1]} design columns but V_bb is "
#                 f"{V_bb.shape[0]} x {V_bb.shape[0]} -- these must match, and "
#                 "the column order of Z must be the row order of V_bb "
#                 "(use build_codes).")
#         C = C + Z @ V_bb @ Z.T
#     return C


def check_variance_split(v_a: np.ndarray, V_bb: np.ndarray,
                         design_codes: np.ndarray, v1: np.ndarray,
                         rtol: float = 1e-6) -> None:
    """Assert that the split of v1 into (v_a, V_bb) is self-consistent.

    Because the SS arm uses site-specific records and the FX arm a fixed shared
    set, the two are independent, so exactly:

        v1[i] = v_a[i] + V_bb[g[i], g[i]]

    A failure means either the row -> design mapping is wrong (positional
    instead of label-based) or the SS and FX replicates are not aligned by
    replicate index r.
    """
    lhs = np.asarray(v_a, float) + np.diag(np.asarray(V_bb, float))[design_codes]
    rhs = np.asarray(v1, float)
    if not np.allclose(lhs, rhs, rtol=rtol):
        worst = int(np.argmax(np.abs(lhs - rhs)))
        raise AssertionError(
            "v_a + diag(V_bb)[g[i]] != v1. Worst row "
            f"{worst}: {lhs[worst]:.6g} vs {rhs[worst]:.6g}. "
            "Check the label-based design mapping and the replicate alignment.")


# ---------------------------------------------------------------------------
# REML objective and fitting
# ---------------------------------------------------------------------------

def _build_sigma(V_known: np.ndarray, tau2: np.ndarray,
                 Gs: Sequence[np.ndarray]) -> np.ndarray:
    """Sigma = V_known + sum_p tau2[p] * Gs[p]."""
    Sigma = V_known.copy()
    for t, G in zip(tau2, Gs):
        Sigma = Sigma + t * G
    return Sigma


def reml_nll(logpar: np.ndarray, y: np.ndarray, X: np.ndarray,
             V_known: np.ndarray, Gs: Sequence[np.ndarray]) -> float:
    """Negative restricted log-likelihood, written in Viechtbauer's (2005) form.

    THE EQUATION
    ------------
    Writing V for the marginal covariance of y (called Sigma elsewhere in this
    module), Viechtbauer (2005), "Bias and Efficiency of Meta-Analytic Variance
    Estimators in the Random-Effects Model", JEBS 30(3), 261-293, states the
    restricted log-likelihood as

        log L_REML = -1/2 log|V|
                     -1/2 log|X' V^-1 X|
                     -1/2 y' P y                        (+ a constant)

    with the "REML projection matrix"

        P = V^-1  -  V^-1 X (X' V^-1 X)^-1 X' V^-1

    and this function returns MINUS that, dropping the constant:

        nll = 1/2 [ log|V| + log|X' V^-1 X| + y' P y ]

    In this module

        V = V_known + sum_p exp(logpar[p]) * Gs[p]

    so V_known holds the known sampling covariance and each Gs[p] is the known
    0/1 pattern (Z Z', S S', I) through which one variance component acts.

    WHAT P DOES
    -----------
    P is a GLS residual-maker. Multiplying by P simultaneously (i) whitens by
    V^-1 and (ii) projects out the column space of X. Equivalently, and this is
    worth knowing because it is the cheaper way to compute the same number,

        y' P y  ==  (y - X b)' V^-1 (y - X b),    b = (X' V^-1 X)^-1 X' V^-1 y

    i.e. the GLS-weighted sum of squared residuals about the GLS fit. The two
    expressions are algebraically identical; P just says it in one symbol
    rather than three steps. ``reml_nll_chol`` below uses the residual form.

    WHY THE MIDDLE TERM (this is the whole point of REML)
    -----------------------------------------------------
    -1/2 log|X' V^-1 X| is what distinguishes REML from plain ML. It accounts
    for the degrees of freedom spent estimating the q fixed effects, which ML
    ignores and is therefore biased low for the variance components. The
    simplest case makes it concrete: for y_i ~ N(mu, sigma^2) with X = 1,

        ML   -> sigma^2_hat = sum (y_i - ybar)^2 / n
        REML -> sigma^2_hat = sum (y_i - ybar)^2 / (n - 1)

    REML is literally "the thing that gives you n - 1 instead of n". With only
    an intercept the correction is modest but not negligible at n = 60.

    IMPLEMENTATION NOTES
    --------------------
    * ``np.linalg.slogdet`` is used, never ``log(det(V))``. det(V) underflows
      hard at these sizes: for this study's Sigma it is about 9e-86 at n = 60,
      8e-176 at n = 120 and EXACTLY 0.0 by n = 240, so log(det(V)) would
      silently return -inf.
    * The explicit inverse is safe here -- the fitted Sigma has a condition
      number around 50 (n = 60) to 100 (n = 120). It costs roughly 1.7x
      (n = 60) to 3.5x (n = 120) the Cholesky version per evaluation, which is
      sub-millisecond and irrelevant for a single fit. It is only worth
      switching to ``reml_nll_chol`` inside the outer-bootstrap loop, where
      1000 replicates turn ~3 minutes into ~11.
    * A non-PSD V (which the optimiser can propose) returns a large finite
      penalty rather than raising, so the search simply backs away from it.

    Parameters
    ----------
    logpar : ndarray, shape (p,)
        ``log(tau^2)`` for each variance component. The log scale keeps the
        components positive without constrained optimisation -- at the cost
        that exactly zero is unreachable (see _REML_ZERO_TOL).
    y : ndarray, shape (n,)
    X : ndarray, shape (n, q)
        Fixed-effects design. Intercept-only for A1: ``np.ones((n, 1))``.
    V_known : ndarray, shape (n, n)
    Gs : sequence of p arrays, each (n, n)

    Returns
    -------
    float
    """
    tau2 = np.exp(logpar)

    # V = V_known + sum_p tau2[p] * Gs[p]
    V = _build_sigma(V_known, tau2, Gs)

    # --- log|V| ------------------------------------------------------------
    # slogdet returns (sign, log|det|); sign must be +1 for a positive-definite
    # V. Anything else means the optimiser has wandered somewhere invalid.
    sign_V, logdet_V = np.linalg.slogdet(V)
    if sign_V <= 0 or not np.isfinite(logdet_V):
        return 1e10

    # --- V^-1 --------------------------------------------------------------
    try:
        Vi = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return 1e10

    # --- X' V^-1 X  and  log|X' V^-1 X| ------------------------------------
    XtViX = X.T @ Vi @ X
    sign_X, logdet_XtViX = np.linalg.slogdet(XtViX)
    if sign_X <= 0 or not np.isfinite(logdet_XtViX):
        return 1e10

    try:
        XtViX_inv = np.linalg.inv(XtViX)
    except np.linalg.LinAlgError:
        return 1e10

    # --- P = V^-1 - V^-1 X (X' V^-1 X)^-1 X' V^-1 --------------------------
    ViX = Vi @ X                                   # n x q
    P = Vi - ViX @ XtViX_inv @ ViX.T               # n x n

    # --- y' P y ------------------------------------------------------------
    quad = float(y @ P @ y)

    val = 0.5 * (logdet_V + logdet_XtViX + quad)
    return val if np.isfinite(val) else 1e10


def reml_nll_chol(logpar: np.ndarray, y: np.ndarray, X: np.ndarray,
                  V_known: np.ndarray, Gs: Sequence[np.ndarray]) -> float:
    """Same objective as ``reml_nll``, via Cholesky factorisation.

    Mathematically identical -- it evaluates y' P y in its equivalent residual
    form (y - X b)' V^-1 (y - X b) and gets both log-determinants from the
    triangular factors -- but never forms an explicit inverse and never
    materialises the n x n matrix P.

    Faster by roughly 1.7x at n = 60 and 3.5x at n = 120. Use it via
    ``fit_reml(..., method="cholesky")`` when fitting inside the outer
    bootstrap; ``reml_nll`` is the readable reference implementation and the
    default.

    The two must agree to ~1e-13. ``tests/`` should assert that.
    """
    tau2 = np.exp(logpar)
    Sigma = _build_sigma(V_known, tau2, Gs)

    try:
        cf = cho_factor(Sigma, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10
    # cho_factor returns (L, lower); diag(L) gives the log-determinant cheaply,
    # as log|Sigma| = 2 * sum(log(diag(L))).
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(cf[0])))
    if not np.isfinite(logdet_Sigma):
        return 1e10

    Si_X = cho_solve(cf, X, check_finite=False)     # Sigma^-1 X
    Si_y = cho_solve(cf, y, check_finite=False)     # Sigma^-1 y
    XtSiX = X.T @ Si_X

    try:
        cfx = cho_factor(XtSiX, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10
    logdet_XtSiX = 2.0 * np.sum(np.log(np.diag(cfx[0])))

    beta = cho_solve(cfx, X.T @ Si_y, check_finite=False)   # generalised least squares estimate
    resid = y - X @ beta
    quad = float(resid @ cho_solve(cf, resid, check_finite=False))

    val = 0.5 * (logdet_Sigma + logdet_XtSiX + quad)
    return val if np.isfinite(val) else 1e10


def fit_reml(y: np.ndarray, X: np.ndarray, V_known: np.ndarray,
             Gs: Sequence[np.ndarray], names: Sequence[str] | None = None,
             n_starts: int = 4, seed: int = 0,
             zero_tol: float = _REML_ZERO_TOL, verbose: bool = False,
             method: str = "explicit") -> dict:
    """Fit a linear mixed model with KNOWN sampling covariance by REML.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Effect sizes (log-ratios).
    X : ndarray, shape (n, q)
        Fixed-effects design; ``np.ones((n, 1))`` for an intercept-only fit.
    V_known : ndarray, shape (n, n)
        Parameter-free part of Sigma -- see ``build_V_known``.
    Gs : sequence of (n, n) arrays
        One known matrix per variance component, e.g. ``[ZZt, SSt, I]``.
    names : sequence of str, optional
        Labels for the components, e.g. ``["tau_d2", "tau_s2", "tau_u2"]``.
    n_starts : int
        Number of random restarts. REML surfaces over variance components are
        often flat or multi-modal, so a single Nelder-Mead run can stop early.
        All starts are run and the best objective wins; the spread across
        starts is reported as ``start_spread`` and should be ~0.
    seed : int
        Seed for the restart jitter, so the fit is reproducible.
    zero_tol : float
        Components below this are reported as exactly 0.0 and flagged in
        ``at_zero_boundary``.
    method : {"explicit", "cholesky"}
        Which implementation of the REML objective to minimise. Both give the
        same answer to ~1e-13.
        "explicit"  -- ``reml_nll``, written in Viechtbauer's (2005) notation
                       with the projection matrix P spelled out. The default,
                       because it is the readable reference.
        "cholesky"  -- ``reml_nll_chol``, ~1.7x faster at n = 60 and ~3.5x at
                       n = 120. Worth switching to inside the outer-bootstrap
                       loop; irrelevant for a single fit.

    Returns
    -------
    dict with keys
        beta               (q,)   GLS fixed effects
        se_beta            (q,)   standard errors, sqrt(diag((X'Sigma^-1 X)^-1))
        vcov_beta          (q,q)
        tau2               (p,)   fitted variance components (zeroed if tiny)
        tau                (p,)   sqrt of the above
        names              list of str
        at_zero_boundary   (p,) bool -- component hit the lower boundary
        converged          bool
        nll                float  minimised objective
        start_spread       float  max - min objective across restarts
        method             str    which objective implementation was used
        Sigma              (n,n)  fitted covariance, for diagnostics

    Caveats
    -------
    * ``se_beta`` conditions on the fitted variance components as if they were
      known. It therefore UNDERSTATES uncertainty in m1, mildly. Report
      bootstrap intervals as the inferential result.
    * No standard errors are produced for the variance components themselves;
      there is no honest closed form here. Get those from the outer bootstrap.
    * ``at_zero_boundary`` being True is informative, not an error: it means the
      data contain no evidence for that source of variation beyond what the
      known sampling covariance already explains.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    V_known = np.asarray(V_known, dtype=float)
    Gs = [np.asarray(G, dtype=float) for G in Gs]

    n = y.size
    p = len(Gs)
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but y has {n}")
    if V_known.shape != (n, n):
        raise ValueError(f"V_known must be {n} x {n}, got {V_known.shape}")
    for idx, G in enumerate(Gs):
        if G.shape != (n, n):
            raise ValueError(f"Gs[{idx}] must be {n} x {n}, got {G.shape}")
    if names is None:
        names = [f"tau2_{i}" for i in range(p)]
    if len(names) != p:
        raise ValueError("len(names) must match len(Gs)")

    objectives = {"explicit": reml_nll, "cholesky": reml_nll_chol}
    if method not in objectives:
        raise ValueError(f"method must be one of {sorted(objectives)}, got {method!r}")
    objective = objectives[method]

    # Starting values: split the total observed variance evenly over the
    # components, then jitter on the log scale for the restarts.
    rng = np.random.default_rng(seed)
    base = max(float(np.var(y)) / max(p, 1), 1e-8)
    starts = [np.log(np.full(p, base))]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(np.log(np.full(p, base)) + rng.normal(0.0, 1.5, p))

    results = []
    for x0 in starts:
        res = minimize(objective, x0, args=(y, X, V_known, Gs),
                       method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000,
                                "maxfev": 20000})
        results.append(res)

    objs = np.array([r.fun for r in results])
    best = results[int(np.argmin(objs))]
    start_spread = float(objs.max() - objs.min())
    if start_spread > 1e-4 and verbose:
        print(f"[fit_reml] objective spread across {len(starts)} starts: "
              f"{start_spread:.3e} -- surface may be flat or multi-modal")

    tau2 = np.exp(best.x)
    at_zero = tau2 < zero_tol
    tau2 = np.where(at_zero, 0.0, tau2)

    # Final quantities at the optimum.
    Sigma = _build_sigma(V_known, tau2, Gs)
    cf = cho_factor(Sigma, lower=True, check_finite=False)
    Si_X = cho_solve(cf, X, check_finite=False)
    XtSiX = X.T @ Si_X
    vcov = np.linalg.inv(XtSiX)
    beta = vcov @ (X.T @ cho_solve(cf, y, check_finite=False))

    return {
        "beta": beta,
        "se_beta": np.sqrt(np.diag(vcov)),
        "vcov_beta": vcov,
        "tau2": tau2,
        "tau": np.sqrt(tau2),
        "names": list(names),
        "at_zero_boundary": at_zero,
        "converged": bool(best.success),
        "nll": float(best.fun),
        "start_spread": start_spread,
        "method": method,
        "Sigma": Sigma,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers -- these just choose the right Gs and V_known
# ---------------------------------------------------------------------------

# def fit_A1a(y, v1, **kw) -> dict:
#     """Two-level random-effects meta-analysis, REML.

#         y_i = m1 + u_i + eps_i,   eps_i ~ N(0, v1_i)

#     ``v1`` is the COMBINED bootstrap variance of the difference,
#     Var_r(a_i^(r) - b_{g[i]}^(r)) -- i.e. exactly the ``v1_boot`` already used
#     for the DerSimonian-Laird fit. This is the fit to compare against DL and
#     against PyMare's ``method='REML'``.
#     """
#     y = np.asarray(y, float).ravel()
#     n = y.size
#     return fit_reml(y, np.ones((n, 1)), np.diag(np.asarray(v1, float).ravel()),
#                     [np.eye(n)], names=["tau_u2"], **kw)


# def fit_A1b(y, V_known, design_codes, n_designs=None, **kw) -> dict:
#     """Three-level model: rows nested in design groups.

#         y_i = m1 + d_{g[i]} + u_i + eps_i

#     Adds tau_d^2. Pass the FULL ``V_known`` (with Z V_bb Z') -- with a diagonal
#     V_known, tau_d^2 will absorb the shared-b sampling covariance and read too
#     high.

#     Note that A1b alone does NOT answer "does the building or the site matter
#     more": its residual tau_u^2 still bundles site effects, design x site
#     interaction and row noise together. Use it for the ICC
#     tau_d^2 / (tau_d^2 + tau_u^2) and as a stepping stone to A1c.
#     """
#     y = np.asarray(y, float).ravel()
#     n = y.size
#     Z = make_indicator(design_codes, n_designs)
#     return fit_reml(y, np.ones((n, 1)), V_known, [Z @ Z.T, np.eye(n)],
#                     names=["tau_d2", "tau_u2"], **kw)


# def fit_A1c(y, V_known, design_codes, site_codes,
#             n_designs=None, n_sites=None, X=None, **kw) -> dict:
#     """Crossed design x site model -- the reportable A1 fit.

#         y_i = m1 + d_{g[i]} + s_{sigma[i]} + u_i + eps_i

#     Design and site are CROSSED, not nested: a design spans several sites and a
#     site hosts several designs. u_i therefore absorbs the design x site
#     interaction, and must be kept -- without it the interaction is silently
#     credited to whichever of d or s the data happen to favour.

#     The comparison of interest is tau_d vs tau_s. Because they are two
#     ESTIMATED standard deviations, do not compare the two point estimates: form
#     the difference (or ratio) per outer-bootstrap replicate and report its
#     interval.

#     Pass ``X`` to add covariates (A0b / A3); default is intercept-only.
#     """
#     y = np.asarray(y, float).ravel()
#     n = y.size
#     Z = make_indicator(design_codes, n_designs)
#     S = make_indicator(site_codes, n_sites)
#     if X is None:
#         X = np.ones((n, 1))
#     return fit_reml(y, X, V_known, [Z @ Z.T, S @ S.T, np.eye(n)],
#                     names=["tau_d2", "tau_s2", "tau_u2"], **kw)


def summarise_reml_fit(fit: dict, ratio_units: bool = True) -> str:
    """Human-readable one-block summary of a fit.

    With ``ratio_units`` the effect and each tau are also shown as exp(.), the
    multiplicative scale an engineer can use: exp(tau_d) is "the typical
    multiplicative spread of the correction factor across designs". Prefer this
    over reporting tau^2, and over I^2, which is a proportion and becomes
    doubly unhelpful once there is more than one tau.
    """
    lines = []
    b, se = fit["beta"][0], fit["se_beta"][0]
    lines.append(f"m1          = {b:+.5f}   (se {se:.5f}, model-based)")
    if ratio_units:
        lines.append(f"  ratio     = {np.exp(b):.4f}"
                     f"   [{np.exp(b - 1.96 * se):.4f}, {np.exp(b + 1.96 * se):.4f}]")
    total = float(np.sum(fit["tau2"]))
    for nm, t2, zero in zip(fit["names"], fit["tau2"], fit["at_zero_boundary"]):
        share = t2 / total if total > 0 else np.nan
        flag = "  <- AT ZERO BOUNDARY" if zero else ""
        line = f"{nm:<10} = {t2:.6f}   tau = {np.sqrt(t2):.5f}   share = {share:6.1%}"
        if ratio_units:
            line += f"   exp(tau) = {np.exp(np.sqrt(t2)):.4f}"
        lines.append(line + flag)
    lines.append(f"converged   = {fit['converged']}   "
                 f"start spread = {fit['start_spread']:.2e}   "
                 f"-2logL/2 = {fit['nll']:.4f}")
    return "\n".join(lines)
