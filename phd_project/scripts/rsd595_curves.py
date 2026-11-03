"""Curves of the mean and SD of ln RSD595 given AvgSA(0-3 s), one pair per site.

What this module is for
-----------------------
The fragility-correction models (nb 072 covariates ``d_ln_ds595_{mean,sigma}_unif``,
nb 080 model ``A8f_bc_hb``) use the GCIM moments of the significant duration
:math:`D_{s5-95}` (OpenQuake IMT ``RSD595``) conditional on the intensity
:math:`x = AvgSA(0\\text{-}3\\,s)`:

.. math::

    \\mu(x) = E[\\ln RSD595 \\mid AvgSA = x], \\qquad
    \\sigma(x) = SD[\\ln RSD595 \\mid AvgSA = x]

Until nb 081 these existed only at the record-selection stripes (4-12 IMLs per
site, a different set at each site). nb 081 computes them on one common IML grid
for all 60 sites, and this module holds everything nb 081 needs that is worth
reusing or testing:

- the grid: :func:`build_iml_grid`, :func:`write_iml_grid`;
- the per-site upper limit: :func:`hazard_curve_ceilings`, :func:`disagg_ceilings`;
- the compact disaggregation frame: :func:`slim_disagg_frame`;
- a sanity check on the occurrence weights: :func:`occurrence_sum_error`;
- the curve tables: :func:`curves_from_gcim`, :func:`write_curves`, :func:`read_curves`;
- reading the curves at any IML: :func:`interp_curve` and :class:`RSD595Curves`.

The moments themselves are computed by pickagm's GCIM code (through
``gm_selection.calculate_gcim_distributions_for_sites``), exactly as in nb 031;
nothing here re-implements the mixture.

The moments are site properties (hazard only): both structures at a site share
the same pair of curves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column names of the disaggregation frames written by nb 021 / nb 081
# (oq_multi_iml_disagg.collate_targets).
OCC = "P(m|X=x)"          # occurrence share, P(m | AvgSA = x)
EXC = "P(m|X>x)"          # exceedance share, P(m | AvgSA > x)
NU = "nu_m"               # annual rate at which the bin exceeds x

# The GCIM row that holds the duration moments.
RSD595_ROW = "RSD595"

# Index name of the curve tables.
SITE_INDEX = "site_id"


# --------------------------------------------------------------------------- #
# the IML grid                                                                 #
# --------------------------------------------------------------------------- #

def _key(x: float) -> float:
    """An IML as the key the disaggregation files use: 6 significant figures.

    Same rule as ``oq_runner._iml_str``, which nb 020 / 021 use, so a grid IML
    that also appears in the record-selection grid (an anchor) gets the very
    same float there and here.
    """
    return float(f"{float(x):.6g}")


def build_iml_grid(lo: float, hi: float, n: int, anchors=()) -> list[float]:
    """Log-spaced IMLs from ``lo`` to ``hi`` (both included, in g), plus ``anchors``.

    Every value is rounded to the 6-significant-figure key (:func:`_key`). The
    anchors are extra IMLs that already exist in the record-selection grid,
    added so the new disaggregation can be compared with the old one at the
    very same levels.

    Returns the sorted keys. Raises if two of them coincide at 6 s.f., because
    two targets at one level cannot be told apart in the engine output.
    """
    if not (0 < lo < hi) or n < 2:
        raise ValueError("need 0 < lo < hi and n >= 2")
    keys = [_key(v) for v in np.geomspace(lo, hi, n)] + [_key(a) for a in anchors]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"grid IMLs coincide at 6 s.f.: {dup}")
    return sorted(keys)


def write_iml_grid(grid, fp: Path) -> Path:
    """Write the grid in nb 017's IML-file format (one column, ``np.savetxt``).

    Rewrites the file only when its text changes, so its hash (part of the
    disaggregation child's fingerprint) stays stable across unchanged re-runs.
    ``oq_runner.load_imls`` reads it back.
    """
    fp = Path(fp)
    lines = [f"{v:.18e}" for v in grid]          # np.savetxt's default format
    text = "\n".join(lines) + "\n"
    if not (fp.is_file() and fp.read_text() == text):
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(text)
    return fp


# --------------------------------------------------------------------------- #
# per-site upper limit                                                         #
# --------------------------------------------------------------------------- #

def hazard_curve_ceilings(hcurves: dict, imt: str = "AvgSA") -> pd.Series:
    """Per site, the largest IML of the hazard-curve grid with positive MAFE.

    This is nb 017's ``max_usable_iml``. It is coarse: near the top the curve's
    grid points are a factor of ~1.44 apart, and for many sites the curve is
    still positive at the last point, so the value is just the end of the grid.
    It is reported for comparison with :func:`disagg_ceilings`, which is the
    limit nb 081 actually uses.
    """
    out = {}
    for site in sorted(hcurves):
        hc = np.asarray(hcurves[site][imt]["mean"], float)
        pos = hc[hc[:, 1] > 0, 0]
        out[int(site)] = float(pos.max()) if pos.size else np.nan
    return pd.Series(out, name="ceiling_hazard_curve_g").rename_axis(SITE_INDEX)


def disagg_ceilings(grid, excluded: dict, sites) -> pd.DataFrame:
    """Per site, the largest grid IML that was **not** excluded as zero hazard.

    ``excluded`` is ``oqhelpers.get_excluded_imls``' output, ``{site: [imls]}``.
    An IML is excluded when the hazard curve gives zero rate there or when no
    disaggregation bin can exceed it: it lies above the site's epsilon-truncated
    maximum ground motion. That makes this a finer limit than
    :func:`hazard_curve_ceilings`.

    Zero hazard can only start above some level and continue from there, so an
    excluded IML **below** a kept one would mean something is wrong upstream.
    That raises rather than leaving a hole inside a site's curve.

    Returns a DataFrame indexed by ``site_id`` with ``ceiling_disagg_g`` (NaN if
    every IML is excluded) and ``n_kept``.
    """
    grid = sorted(grid)
    rows = {}
    for site in sites:
        ex = set(excluded.get(site, excluded.get(str(site), [])))
        kept = [g for g in grid if g not in ex]
        if kept and any(g < kept[-1] for g in ex):
            holes = sorted(g for g in ex if g < kept[-1])
            raise ValueError(f"site {site}: zero-hazard IMLs {holes} lie below the kept "
                             f"IML {kept[-1]} - a hole inside the curve")
        rows[int(site)] = {"ceiling_disagg_g": kept[-1] if kept else np.nan,
                           "n_kept": len(kept)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(SITE_INDEX)


# --------------------------------------------------------------------------- #
# compact disaggregation frames                                                #
# --------------------------------------------------------------------------- #

def slim_disagg_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sum a disaggregation frame over its epsilon bins and drop all-zero rows.

    Why this loses nothing for nb 081
    ---------------------------------
    pickagm's GCIM begins with ``groupby(["TRT", "Mag", "Dist"]).sum()`` and then
    drops rows with zero occurrence weight (``_conditional_im_distributions``).
    It never uses the epsilon bins: it recomputes the epsilon of ``x`` from the
    GMM at each (TRT, M, R) bin centre. The disaggregation stats
    (``oqhelpers.get_iml_disagg_stats``) use only TRT, Mag, Dist, ``nu_m`` and
    ``P(m|X>x)``, and the zero-hazard test only asks whether any ``nu_m`` is
    positive. Doing the epsilon sum here, in the same row order pandas would
    use later, gives the same numbers, and the frame is about 6x smaller
    before the zero rows are even dropped.

    What is kept
    ------------
    ``TRT, Mag, Dist, Z`` and the three additive columns ``nu_m``, ``P(m|X>x)``,
    ``P(m|X=x)``, each summed over ``Eps``. ``P(X>x|T,m)`` is dropped: it is a
    conditional probability per epsilon bin and cannot be summed.

    Rows whose ``nu_m`` is 0 are dropped. Such a bin cannot exceed ``x``, so its
    exceedance share is 0 and its occurrence share is 0 too (the occurrence
    numerator ``nu_m(x) - nu_m(x+dx)`` cannot be positive). At an IML above the
    maximum ground motion every row goes, leaving an empty frame, which the
    zero-hazard test then excludes as before.

    ``min_count=1`` keeps an all-NaN occurrence column NaN (it would otherwise
    sum to 0), so a frame whose occurrence could not be formed stays visibly
    NaN for :func:`occurrence_sum_error` to catch.
    """
    keys = ["TRT", "Mag", "Dist", "Z"]
    cols = [NU, EXC, OCC]
    out = (df.groupby(keys, sort=False)[cols].sum(min_count=1)
             .reset_index())
    out = out[out[NU].to_numpy(float) > 0].reset_index(drop=True)
    return out


def occurrence_sum_error(disagg_data: dict, excluded: dict, imt: str = "AvgSA") -> float:
    """Largest ``|sum P(m|X=x) - 1|`` over every kept (site, IML) and realisation.

    The occurrence column is built to sum to exactly 1 per realisation index Z
    (``oq_multi_iml_disagg.occurrence_disagg``). A NaN sum, meaning the
    occurrence could not be formed at an IML that was *not* excluded as zero
    hazard, is returned as ``inf``, so an assertion on the result catches it.
    """
    worst = 0.0
    for site, by_imt in disagg_data.items():
        ex = set(excluded.get(site, excluded.get(str(site), [])))
        for iml, df in by_imt[imt].items():
            if iml in ex:
                continue
            sums = df.groupby("Z")[OCC].sum(min_count=1).to_numpy(float)
            if sums.size == 0 or not np.all(np.isfinite(sums)):
                return float("inf")
            worst = max(worst, float(np.abs(sums - 1.0).max()))
    return worst


# --------------------------------------------------------------------------- #
# curve tables                                                                 #
# --------------------------------------------------------------------------- #

def curves_from_gcim(gcim_dists: dict, grid, sites) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two wide tables of the GCIM's RSD595 moments: (mu, sigma).

    ``gcim_dists[(site, iml)]["stats"]`` is pickagm's stats DataFrame, whose
    ``RSD595`` row has ``mean`` (= E[ln RSD595]) and ``sigma`` (= SD[ln RSD595]),
    both already in log units. Each table has one row per site (index
    ``site_id``) and one column per grid IML (ascending floats). A cell is NaN
    where there is no GCIM entry: above the site's ceiling, or excluded as zero
    hazard. Nothing is filled or extrapolated.
    """
    grid = sorted(grid)
    sites = [int(s) for s in sites]
    mu = pd.DataFrame(np.nan, index=pd.Index(sites, name=SITE_INDEX), columns=grid)
    sigma = mu.copy()
    for (site, iml), d in gcim_dists.items():
        if int(site) not in mu.index or iml not in mu.columns:
            raise KeyError(f"GCIM key ({site}, {iml}) is not on the grid / site list")
        row = d["stats"].loc[RSD595_ROW]
        mu.loc[int(site), iml] = float(row["mean"])
        sigma.loc[int(site), iml] = float(row["sigma"])
    return mu, sigma


def write_curves(df: pd.DataFrame, fp: Path) -> Path:
    """Write one curve table as CSV.

    pandas writes floats at full (round-trip) precision, both the values and the
    column labels, so :func:`read_curves` gets back exactly the same numbers and
    exactly the 6-significant-figure grid keys. NaN cells are written empty.
    """
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = SITE_INDEX
    out.columns = [repr(float(c)) for c in out.columns]
    out.to_csv(fp, lineterminator="\n")
    return fp


def read_curves(fp: Path) -> pd.DataFrame:
    """Read a curve table back: integer ``site_id`` index, float IML columns.

    A CSV header is text, so ``pd.read_csv`` alone always returns string
    column labels; they are converted to float here. The columns are checked
    to be strictly ascending, which :func:`interp_curve` relies on.
    """
    df = pd.read_csv(fp, index_col=SITE_INDEX, float_precision="round_trip")
    df.index = df.index.astype(int)
    df.columns = [float(c) for c in df.columns]
    if np.any(np.diff(np.asarray(df.columns, float)) <= 0):
        raise ValueError(f"{Path(fp).name}: IML columns are not strictly ascending")
    return df


# --------------------------------------------------------------------------- #
# reading the curves at any IML                                                #
# --------------------------------------------------------------------------- #

def interp_curve(curves: pd.DataFrame, site: int, x):
    """A site's moment at IML(s) ``x`` (g), linear in ln x between computed points.

    Returns NaN for any ``x`` outside the site's computed range (below its first
    or above its last non-NaN point). The curve is never extrapolated. Inside
    the range a NaN cell would be a hole, which upstream checks rule out; if one
    is found anyway this raises rather than interpolating across it.

    Why linear in ln x: the grid is log-spaced and the hazard and GMM vary
    smoothly in ln x, so straight lines between neighbouring points in ln x are
    the natural reading of a tabulated curve.

    ``x`` may be a scalar (a float is returned) or array-like (an array is
    returned).
    """
    row = curves.loc[int(site)]
    xs = np.asarray(row.index, float)
    ys = row.to_numpy(float)
    ok = np.isfinite(ys)
    scalar = np.ndim(x) == 0
    xq = np.atleast_1d(np.asarray(x, float))
    if not ok.any():
        out = np.full(xq.shape, np.nan)
        return float(out[0]) if scalar else out
    first, last = np.flatnonzero(ok)[[0, -1]]
    if not ok[first:last + 1].all():
        raise ValueError(f"site {site}: NaN inside the computed range of the curve")
    lnx = np.log(xs[first:last + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log(xq)
    out = np.interp(lq, lnx, ys[first:last + 1], left=np.nan, right=np.nan)
    out[~np.isfinite(lq)] = np.nan
    return float(out[0]) if scalar else out


class RSD595Curves:
    """Both curve tables of nb 081, read once, for use downstream.

    >>> c = RSD595Curves(cfg["proc_data"]["mu_lnRSD595_given_AvgSA_03"],
    ...                  cfg["proc_data"]["sigma_lnRSD595_given_AvgSA_03"])
    >>> c.mu(33, 0.42), c.sigma(33, [0.3, 0.6])

    Values are in log units (ln seconds for ``mu``). NaN outside a site's
    computed range, as :func:`interp_curve`.
    """

    def __init__(self, fp_mu: Path, fp_sigma: Path):
        self.mu_table = read_curves(fp_mu)
        self.sigma_table = read_curves(fp_sigma)
        # The two files come from the same GCIM run, so they must cover the same
        # sites, IMLs and cells; anything else means they were mixed up.
        if not (self.mu_table.index.equals(self.sigma_table.index)
                and list(self.mu_table.columns) == list(self.sigma_table.columns)
                and (self.mu_table.isna().to_numpy()
                     == self.sigma_table.isna().to_numpy()).all()):
            raise ValueError("the mu and sigma tables do not cover the same cells")

    def mu(self, site: int, x):
        """E[ln RSD595 | AvgSA = x] at site ``site``."""
        return interp_curve(self.mu_table, site, x)

    def sigma(self, site: int, x):
        """SD[ln RSD595 | AvgSA = x] at site ``site``."""
        return interp_curve(self.sigma_table, site, x)

    def range(self, site: int) -> tuple[float, float]:
        """(lowest, highest) IML in g where the site's curves are defined."""
        row = self.mu_table.loc[int(site)].dropna()
        return (float(row.index.min()), float(row.index.max())) if len(row) else (np.nan, np.nan)
