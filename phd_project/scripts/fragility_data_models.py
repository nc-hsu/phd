"""Data model for the A-series meta-analysis models of ``admin/analysis_models.md``.

Why this module exists
----------------------
Notebook 073 grew a naming problem rather than a data model. Every array it used
encoded FOUR independent axes into a single identifier:

    axis                          how it was spelled
    ---------------------------   ------------------------------------------
    arm                           mss / mfx / ifx / mgr / igr
    quantity (theta or beta)      a theta_/beta_ prefix, or the at/ab letters
    scope (120 rows or 51 groups) a `_gr` suffix
    kind (replicates or estimate) an `_est` suffix, or nothing

Four axes times five arms is thirty module-level names before a single model is
fitted, and every new rung of the roadmap multiplies them again. Worse, the
per-model names (`y0`, `v0_boot`, `lb`, `ub`, `T_sq`, ...) were rebound by each
successive model, so A0 results silently became A1 results and then A2 results.

This module makes the four axes into FIELDS instead of spelling:

    arm       -> a key of ``ARMS`` / a symbol in ``SYMBOLS``
    quantity  -> ``FragilityData.quantity``, fixed once at load time
    scope     -> ``Contrast.scope``, and the object own index
    kind      -> ``reps`` (replicates) vs ``estimates`` (point estimates)

and makes the roadmap id (``"A0a"``, ``"A1a"``, ``"A2"``, ...) the key of a
registry, so a fit can never overwrite another fit by accident.

The payoff is that the remaining rungs of the roadmap become DATA rather than
code. A12-A16 ("repeat the whole ladder for the dispersion beta") is
``load_fragility_data(..., quantity="beta")`` and nothing else. Notebook 073
section 2.4 (the geometric-average IDA arm, documented but never implemented) is
one entry in ``_CONTRAST_SPECS`` and one in ``MODEL_SPECS``.

Layering
--------
This module imports :mod:`phd_project.scripts.metaregression_analysis` (``mra``)
and :mod:`phd_project.scripts.data_simulation` (``dsim``); NEITHER imports it
back. That is deliberate. ``mra`` stays a library of pure functions over arrays
that can be tested with no data on disk, which is what makes the fake-data
recovery check in ``dsim`` a genuine test of the fit the notebooks actually run.
If this module ever appears in the imports of ``mra``, that property is gone.

Where the numbers come from
---------------------------
Two files, with a strict division of labour:

  * The **nb-072 dataset CSV** is the row spine AND the source of every point
    estimate and variance. It owns the crossed factor table (site, storeys,
    structure_id, design_group_id) and is the natural home for the covariates
    that A0b/A3 will need. Row order, row count and both index maps come from
    here and are never re-derived.
  * The **bootstrap replicate CSVs** (via ``mra.load_saved_bootstrap``) are the
    only possible source of the sampling COVARIANCE and of the outer bootstrap.
    A variance vector cannot be recovered into a matrix.

The individual ``*_estimates_*.csv`` files are deliberately NOT read on the
normal path: everything in them is already in the dataset CSV, and reading the
same number from two places is how two places come to disagree. Pass
``check_against_estimates=True`` for a one-off audit that they still match.

What is NOT here yet
--------------------
A1b and A1c -- the three-level and crossed-random-effects REML rungs -- are
intentionally absent from ``MODEL_SPECS``. The machinery is all present
(``FragilityData.Z``, ``.S``, ``.V_aa``, ``.V_bb``; ``ModelSpec.components``;
``dsim.build_component_gs``), so adding them is two table entries::

    "A1b": ModelSpec("A1b", "y1", ("REML",), components=("design",)),
    "A1c": ModelSpec("A1c", "y1", ("REML",), components=("design", "site")),

When you add them, pass the FULL ``Contrast.V`` (the default), never
``np.diag(v)``: the shared-b_j sampling covariance has the same block structure
as a design random effect, so a diagonal V lets tau_d^2 absorb it and read too
high. That is the identifiability trap of ``analysis_models.md`` section A1c.

On xarray
---------
``(row, replicate, arm, quantity)`` is genuinely a 4-D labelled array and xarray
is the natural conceptual fit. It is rejected here, and this note exists so the
question is not re-opened every six months: xarray is not installed in this
environment; the 120-row and 51-group scopes are different dimensions so two
Datasets would be needed anyway; the n x n covariance matrices are not naturally
xarray objects; and every function in ``mra`` and ``dsim`` takes a DataFrame or
an ndarray, so the conversion cost would be paid at every single call boundary
while the benefit (index alignment) is already delivered by pandas.

References
----------
Borenstein et al. (2009) "Introduction to Meta-Analysis":
    Ch. 12 pp. 69-75 (the random-effects model), Ch. 14 pp. 87-95 (the worked
    hand calculation), Ch. 16 (why I^2 is not an absolute measure), Ch. 17
    ~pp. 127-133 (prediction intervals).
Efron & Tibshirani (1993) "An Introduction to the Bootstrap":
    Ch. 6-7 (bootstrap standard errors), Ch. 10 ~pp. 124-133 (bias estimation,
    and the warning that correcting it inflates variance), Ch. 25 (why nuisance
    inputs are held fixed rather than re-estimated inside the loop).
Gelman & Hill (2007) "Data Analysis Using Regression...":
    Ch. 13.5 ~pp. 289-291 (crossed / non-nested random effects), Ch. 22
    ~pp. 487-500 (comparing variance components).
Viechtbauer (2005) for the REML formulation used by ``mra.fit_reml``.
Page numbers are from standard printings and should be treated as approximate.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook
from matplotlib import collections as mcoll
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
import pandas as pd
from scipy import stats

import phd_project.scripts.data_simulation as dsim
import phd_project.scripts.metaregression_analysis as mra

# =============================================================================
# The arm registry
# =============================================================================
#
# The single source of truth for what an arm is called. This dict was previously
# defined inline in nb 072 and implicitly re-derived by the loader block of
# nb 073, which is exactly how the two came to disagree about scope.
#
# The ORDER matters: nb 072 uses it to order the dataset columns, so changing it
# changes the CSV. Keep it as it is unless you mean to rewrite the dataset.
#
# The three ida_femap695* arms are ONE analysis read off the collapse bracket
# three ways -- the lower bound (the published fragility), the upper bound, and
# the geometric mean of the bracket (nb 053 section 10, nb 070 section 1.5).
# They are therefore three alternative `c` terms, not three independent results,
# and the honest way to report them is as a sensitivity band on A0/A2 rather
# than as three separate findings.

ARMS: dict[str, str] = {
    "msa_fx":  "msa_femap695",      # MSA, fixed FEMA P695 set        (by design group)
    "ida_fx":  "ida_femap695",      # IDA, bracket lower bound        (by design group)
    "ida_ub":  "ida_femap695_ub",   # IDA, bracket upper bound        (by design group)
    "ida_avg": "ida_femap695_avg",  # IDA, bracket geometric mean     (by design group)
    "msa_ss":  "site_msa",          # MSA, site-specific GCIM records (by site)
}

# The mathematical notation of the roadmap. a_i, b_j, c_j are LOG medians (or
# log dispersions, when quantity="beta"); see analysis_models.md section 0. The
# two IDA variants are alternative c terms and are named to say so.
SYMBOLS: dict[str, str] = {
    "msa_ss": "a", "msa_fx": "b",
    "ida_fx": "c", "ida_ub": "c_ub", "ida_avg": "c_avg",
}
ARM_OF_SYMBOL: dict[str, str] = {v: k for k, v in SYMBOLS.items()}

# The scope each arm NATURALLY lives at, which is the structural statement that
# retires the `_gr` suffix of nb 073. MSA-FX and IDA-FX do not have "a site
# version and a group version" -- they have ONE scope, the design group, because
# the fixed-set stripe IMLs are chosen from the span of the fragility curve
# rather than from site hazard, so they depend on the design only
# (analysis_models.md section 1(b), corrected). When a 120-row contrast needs
# them, Z expands them; that is what `FragilityData.expand` is for.
SCOPE_OF_ARM: dict[str, str] = {
    "msa_ss": "by_site",
    "msa_fx": "by_group", "ida_fx": "by_group",
    "ida_ub": "by_group", "ida_avg": "by_group",
}

# The three arms every model in the roadmap needs. Their absence is an error.
CORE_ARMS: tuple[str, ...] = ("msa_ss", "msa_fx", "ida_fx")
# The collapse-bracket variants. Loaded when present, left as None when not, so
# a storey count not yet run for a variant does not block the core three.
OPTIONAL_ARMS: tuple[str, ...] = ("ida_ub", "ida_avg")

QUANTITIES: tuple[str, ...] = ("theta", "beta")

# The seven columns nb 072 writes per arm per quantity, plus `{arm}_n_obs`.
# Written down here so the container can read the dataset BY NAME and fail with
# a useful message rather than a bare KeyError if the schema drifts.
DATASET_ARM_COLUMNS: tuple[str, ...] = (
    "{arm}_{q}", "{arm}_log_{q}", "{arm}_{q}_bc", "{arm}_log_{q}_bc",
    "{arm}_var_{q}", "{arm}_var_log_{q}", "{arm}_log_{q}_bias",
)
DATASET_FACTORS: tuple[str, ...] = (
    "site", "n_storeys", "structure_id", "tag", "design_group_id",
    "n_sites_in_group", "is_representative",
)

# nb 072 relies on this and asserts it there; it is restated because the column
# lookups of the container depend on it too. If one prefix were a prefix of
# another then a `startswith` column selection would match the longer name twice
# and produce duplicate columns.
assert not any(p != q and p.startswith(q) for p in ARMS for q in ARMS), (
    "no arm prefix may be a prefix of another - see nb 072 section 4.0")


def dataset_csv_path(root: Path | str, im_tag: str) -> Path:
    """Canonical filename for the nb-072 regression dataset.

    Mirrors ``mra.bootstrap_csv_path`` / ``mra.estimates_csv_path``: the write
    (nb 072) and every read go through here, so the name cannot drift. ``root``
    is the bootstrapping directory, i.e. ``cfg["proc_data"]["bootstrapping"]``;
    the dataset sits one level above it, beside the bootstrap folder.
    """
    return Path(root).parent / f"fragility_regression_dataset_{im_tag}.csv"


def arm_label(symbol_or_arm: str) -> str:
    """Human-readable arm name, reusing the labels ``mra`` already fixed.

    Figures across the chapter must name the arms identically, so the labels
    live in one place (``mra._ARM_LABELS``) and are looked up rather than
    retyped in each notebook.
    """
    arm = ARM_OF_SYMBOL.get(symbol_or_arm, symbol_or_arm)
    return mra._ARM_LABELS.get(ARMS.get(arm, arm), arm)


def arm_colour(symbol_or_arm: str) -> str:
    """Fixed colour for an arm, from ``mra._ARM_COLORS``. See :func:`arm_label`."""
    arm = ARM_OF_SYMBOL.get(symbol_or_arm, symbol_or_arm)
    return mra._ARM_COLORS.get(ARMS.get(arm, arm), "k")


# =============================================================================
# The container
# =============================================================================
#
# One quantity per container. That is the whole reason A12-A16 cost nothing:
# the ladder is re-run by changing an argument, not by writing a parallel set of
# variable names with a `beta_` prefix.
#
# `reps` and `estimates` are dicts keyed by SYMBOL rather than five pairs of named
# fields, so that adding an arm is an ARMS entry and nothing else. The bare
# symbols a/b/c are exposed as properties over those dicts, so a notebook cell
# can still be written as mathematics.


def _require_symbol(symbol: str) -> None:
    if symbol not in ARM_OF_SYMBOL:
        raise KeyError(
            f"unknown symbol {symbol!r} - known symbols are "
            f"{sorted(ARM_OF_SYMBOL)}")


@dataclass(frozen=True)
class FragilityData:
    """Every arm of one quantity, on the log scale, with its crossed factors.

    Frozen because a loaded container is a fact about the study, not a scratch
    variable: mutating one in place would silently invalidate every cached
    covariance derived from it. ``frozen=True`` plus ``cached_property`` also
    gives free memoisation of the expensive blocks (``V_bb`` and friends).

    Scope is structural, not a name suffix. ``a`` is per ROW (120 = 60 sites x
    {3s, 5s}); ``b``, ``c``, ``c_ub`` and ``c_avg`` are per DESIGN GROUP (51),
    because the fixed-record-set arms depend on the design only. There is no
    "b at site scope" -- there is ``Z @ b``, and Z is carried here so that the
    expansion cannot be done two different ways in two different cells.

    Attributes
    ----------
    quantity : {"theta", "beta"}
        Which fragility parameter this container holds. Fixed at load time.
    im_tag : str
        The intensity measure tag, e.g. ``"AvgSA_03"``.
    rows : DataFrame
        The nb-072 factor table -- THE spine. Indexed by ``(site, n_storeys)``.
        Carries the factors, any non-arm columns (where the A0b/A3 covariates
        will live), and the per-arm columns for THIS quantity only -- the other
        quantity is dropped at load, so ``quantity`` means one thing throughout.
    groups : Index
        ``design_group_id`` in canonical order; the row/column order of every
        group-scope object and of ``V_bb``.
    reps : dict of str -> DataFrame or None
        Symbol -> log replicate cloud, ``(n, k)`` at row scope or ``(J, k)`` at
        group scope. ``None`` for an optional arm that was not on disk.
    estimates : dict of str -> Series or None
        Symbol -> log point estimates, same index as the matching ``reps``.
    estimates_bc : dict of str -> Series or None
        The bias-corrected point estimates. See :meth:`hat`.
    """

    quantity: str
    im_tag: str
    rows: pd.DataFrame
    groups: pd.Index
    reps: dict[str, pd.DataFrame | None]
    estimates: dict[str, pd.Series | None]
    estimates_bc: dict[str, pd.Series | None] = field(default_factory=dict)

    # -- the bare mathematical symbols -------------------------------------
    # Exposed so a cell can read as the roadmap writes it. They are views onto
    # `reps`/`estimates`, not copies, so there is still exactly one array per arm.

    @property
    def a(self) -> pd.DataFrame:
        """(n, k) log replicates, MSA site-specific. Roadmap ``a_i``."""
        return self.rep("a")

    @property
    def b(self) -> pd.DataFrame:
        """(J, k) log replicates, MSA fixed set. Roadmap ``b_j``."""
        return self.rep("b")

    @property
    def c(self) -> pd.DataFrame:
        """(J, k) log replicates, IDA fixed set. Roadmap ``c_j``."""
        return self.rep("c")

    @property
    def a_hat(self) -> pd.Series:
        """(n,) log point estimates, MSA site-specific."""
        return self.hat("a")

    @property
    def b_hat(self) -> pd.Series:
        """(J,) log point estimates, MSA fixed set."""
        return self.hat("b")

    @property
    def c_hat(self) -> pd.Series:
        """(J,) log point estimates, IDA fixed set."""
        return self.hat("c")

    # -- generic accessors --------------------------------------------------

    @property
    def available(self) -> tuple[str, ...]:
        """The symbols that actually loaded, in registry order.

        A notebook branches on this rather than on a try/except, so a missing
        optional arm is a visible fact rather than a swallowed exception.
        """
        return tuple(SYMBOLS[arm] for arm in ARMS
                     if self.reps.get(SYMBOLS[arm]) is not None)

    def rep(self, symbol: str) -> pd.DataFrame:
        """Log replicate cloud for one symbol, raising a useful error if absent."""
        _require_symbol(symbol)
        out = self.reps.get(symbol)
        if out is None:
            raise KeyError(
                f"arm {ARM_OF_SYMBOL[symbol]!r} (symbol {symbol!r}) has no saved "
                f"bootstrap for quantity={self.quantity!r}; it would have come from "
                f"{mra.bootstrap_csv_path('<root>', ARMS[ARM_OF_SYMBOL[symbol]], self.quantity, self.im_tag, SCOPE_OF_ARM[ARM_OF_SYMBOL[symbol]]).name}. "
                f"Loaded symbols are {self.available}.")
        return out

    def hat(self, symbol: str, corrected: bool = False) -> pd.Series:
        """Log point estimates for one symbol.

        ``corrected=True`` returns the bias-corrected estimate that nb 070
        section 2.4 writes and nb 072 carries as ``{arm}_log_{q}_bc``.

        The default is deliberately the UNCORRECTED estimate. Efron &
        Tibshirani (Ch. 10, ~p. 128) warn that bias correction inflates
        variance and is frequently not worth it, and the roadmap (Step 1) notes
        that in a DIFFERENCE of two similarly-biased log-medians the bias
        largely cancels anyway. So: compute it, report it, check whether
        correcting moves the summary effect by more than a fraction of its
        standard error -- and only then adopt it.
        """
        _require_symbol(symbol)
        source = self.estimates_bc if corrected else self.estimates
        out = source.get(symbol)
        if out is None:
            raise KeyError(
                f"no {'bias-corrected ' if corrected else ''}point estimates for "
                f"symbol {symbol!r}; loaded symbols are {self.available}")
        return out

    def scope_of(self, symbol: str) -> str:
        """``"row"`` or ``"group"`` -- the scope this symbol naturally lives at."""
        _require_symbol(symbol)
        return "row" if SCOPE_OF_ARM[ARM_OF_SYMBOL[symbol]] == "by_site" else "group"

    def term(self, symbol: str, corrected: bool = False):
        """``(replicates, point_estimates, scope)`` for one symbol.

        The one accessor the contrast builder needs, which is what lets that
        builder be written once and serve every contrast rather than once per
        pair of arms.
        """
        return self.rep(symbol), self.hat(symbol, corrected), self.scope_of(symbol)

    def contrast(self, name: str, other: str | None = None,
                 *, corrected: bool = False) -> "Contrast":
        """``data.contrast("y1")``, or ``data.contrast("a", "b")`` as an alias.

        Two spellings because a cell is sometimes clearer in the numbering of
        the roadmap and sometimes clearer in the arms being compared. The pair
        form looks up the canonical name rather than building an anonymous
        contrast, so there is still exactly one definition of what "a - b"
        means.

        ``Contrast`` and ``make_contrast`` are defined further down this module.
        That is fine: the annotation is a string and is never evaluated, and the
        lookup happens when this method is CALLED, not when it is defined.
        """
        if other is None:
            return make_contrast(self, name, corrected=corrected)
        for cname, (left, right, _, _) in _CONTRAST_SPECS.items():
            if (left, right) == (name, other):
                return make_contrast(self, cname, corrected=corrected)
        known = [(left, right) for left, right, _, _ in _CONTRAST_SPECS.values()]
        raise KeyError(
            f"no contrast is defined as {name!r} - {other!r}; known pairs are {known}")

    # -- shape --------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of rows, i.e. (site, structure) pairs. 120 for the full study."""
        return len(self.rows)

    @property
    def J(self) -> int:
        """Number of design groups. 51 for the full study."""
        return len(self.groups)

    @property
    def n_sites(self) -> int:
        """Number of distinct sites. 60 for the full study."""
        return int(self.rows.index.get_level_values("site").nunique())

    @property
    def k(self) -> int:
        """Number of bootstrap replicates, read from the data rather than assumed."""
        return self.rep("a").shape[1]

    # -- the index maps and their indicator matrices ------------------------
    #
    # The roadmap writes these g[i] and sigma[i], in the Gelman & Hill
    # square-bracket style the project uses (CLAUDE.md section 3), not g(i).
    # `g` keeps its symbol because it is unambiguous; the site map is spelled
    # out as `site_codes` rather than `sigma`, because `sigma` collides with
    # both the marginal covariance Sigma and the ordinary use of sigma for a
    # standard deviation. The docstrings carry the notation instead.

    @cached_property
    def g(self) -> np.ndarray:
        """Row -> design-group code, ``g[i]``, in the column order of ``V_bb``.

        Built with ``mra.build_codes``, i.e. BY LABEL. Aligning rows to V_bb
        positionally is the single most likely silent bug in the pipeline: the
        order designs appear in the by_group bootstrap need not match the order
        rows appear in the by_site one.
        """
        return mra.build_codes(self.rows["design_group_id"], self.groups)

    @cached_property
    def sites(self) -> pd.Index:
        """Distinct site ids in canonical (sorted) order -- the columns of ``S``."""
        return pd.Index(sorted(self.rows.index.get_level_values("site").unique()),
                        name="site")

    @cached_property
    def site_codes(self) -> np.ndarray:
        """Row -> site code -- ``sigma[i]`` in the roadmap notation.

        Label-based like :attr:`g`, and for the same reason. Indexes into
        :attr:`sites`, which is the column order of ``S``.
        """
        return mra.build_codes(self.rows.index.get_level_values("site"), self.sites)

    @cached_property
    def Z(self) -> np.ndarray:
        """(n, J) design indicator. ``Z @ Z.T`` multiplies tau_d^2 in Sigma."""
        return mra.make_indicator(self.g, self.J)

    @cached_property
    def S(self) -> np.ndarray:
        """(n, n_sites) site indicator. ``S @ S.T`` multiplies tau_s^2 in Sigma.

        Needed from A1c onward. Note the two failure modes ``mra`` warns about:
        one site for everybody makes ``S S'`` a matrix of ones, confounded with
        the intercept; one site per row makes ``S S' = I``, aliased with
        tau_u^2. Neither applies at 60 sites x 2 structures, but both are why
        the indicator is built from the spine rather than assumed.
        """
        return mra.make_indicator(self.site_codes, len(self.sites))

    # -- sampling covariance blocks ----------------------------------------

    def cov(self, symbol: str, other: str | None = None) -> np.ndarray:
        """Sampling covariance of one arm, or the cross-covariance of two.

        Estimated from the stored replicates as the full sample covariance,
        ``V = 1/(k-1) sum_r (x_r - xbar)(x_r - xbar)'``.

        The general form exists so the collapse-bracket arms need no new
        members: ``cov("b")`` is V_bb, ``cov("b", "c")`` is V_bc, and
        ``cov("c_avg")`` costs nothing extra.

        A cross-covariance is only meaningful between two arms at the SAME
        scope, and is refused otherwise rather than broadcast into nonsense.
        """
        left = self.rep(symbol).to_numpy(dtype=float)
        if other is None:
            return mra.bootstrap_cov(left, rowvar=True)

        if self.scope_of(symbol) != self.scope_of(other):
            raise ValueError(
                f"cannot cross-covary {symbol!r} ({self.scope_of(symbol)} scope) with "
                f"{other!r} ({self.scope_of(other)} scope) - expand one to the other "
                "first with .expand()")
        right = self.rep(other).to_numpy(dtype=float)
        # np.cov on the stacked matrix, then read off the off-diagonal block.
        m = left.shape[0]
        joint = mra.bootstrap_cov(np.vstack([left, right]), rowvar=True)
        return joint[:m, m:]

    @cached_property
    def V_aa(self) -> np.ndarray:
        """(n, n) sampling covariance of the site-specific arm.

        Block-diagonal BY SITE with a 2x2 block per site, not diagonal: the
        3-storey and 5-storey structures at a site share records wherever they
        are analysed at the same stripe IML, and that does happen (CLAUDE.md
        section 2). Rows at different sites are independent because different
        sites use different records.

        Computed as the full sample covariance, so whatever coupling the
        resampling scheme generates is reported rather than assumed. Note the
        converse caveat: ``np.cov`` can only report the covariance the
        RESAMPLING generated, so if the replicates did not share record draws
        where the analyses shared records, the coupling reads as zero however
        it is computed.
        """
        return self.cov("a")

    @cached_property
    def V_bb(self) -> np.ndarray:
        """(J, J) sampling covariance of the MSA fixed-set arm. DENSE.

        Dense is the physically correct answer, not a numerical artefact: every
        design is fitted to the same fixed record set with the same bootstrap
        index sets, so a replicate that happens to draw strong records shifts
        every b_j the same way. Independent errors average away at 1/sqrt(n);
        this shared component does not average away at all.
        """
        return self.cov("b")

    @cached_property
    def V_cc(self) -> np.ndarray:
        """(J, J) sampling covariance of the IDA fixed-set arm. Dense, as V_bb."""
        return self.cov("c")

    @cached_property
    def V_bc(self) -> np.ndarray:
        """(J, J) cross-covariance of the two fixed-set arms.

        NOT zero: MSA-FX and IDA-FX share one record set, so they share
        sampling error. This is the term that makes
        ``V2 = V_bb + V_cc - 2 V_bc`` rather than ``V_bb + V_cc``, and the
        reason the roadmap says to take v2 straight from the bootstrap variance
        of the difference instead of assembling it from parts.
        """
        return self.cov("b", "c")

    # -- scope conversion ---------------------------------------------------

    def expand(self, x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Fan a group-scope object out to row scope: ``x -> x[g[i]]``.

        Label-based, via the ``design_group_id`` column of the spine, never
        positional. Equivalent to ``Z @ x`` but keeps the labels, so a mistake
        shows up as a KeyError rather than as quietly shuffled numbers.
        """
        labels = self.rows["design_group_id"]
        missing = sorted(set(labels) - set(x.index))
        if missing:
            raise KeyError(
                f"{len(missing)} design group(s) in the spine are absent from the "
                f"object being expanded: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}")
        out = x.reindex(labels)
        out.index = self.rows.index
        return out

    def to_rows(self, symbol: str, corrected: bool = False):
        """``(replicates, point_estimates)`` for one symbol AT ROW SCOPE.

        A row-scope symbol is returned unchanged; a group-scope one is expanded.
        This is the only place the expansion decision is made.
        """
        reps, estimates, scope = self.term(symbol, corrected)
        if scope == "row":
            return reps, estimates
        return self.expand(reps), self.expand(estimates)

    def summary(self) -> str:
        """One-paragraph description of what loaded, for printing in a notebook."""
        storeys = sorted(self.rows.index.get_level_values("n_storeys").unique())
        lines = [
            f"FragilityData(quantity={self.quantity!r}, im_tag={self.im_tag!r})",
            f"  rows        : {self.n}  ({self.n_sites} sites x storeys {storeys})",
            f"  designs     : {self.J}",
            f"  replicates  : {self.k}",
            f"  arms loaded : {', '.join(self.available)}",
        ]
        absent = [SYMBOLS[a] for a in ARMS
                  if self.reps.get(SYMBOLS[a]) is None]
        if absent:
            lines.append(f"  arms absent : {', '.join(absent)}")
        if self.n != 120:
            lines.append(
                "  NOTE: this is not the full 120-row study. Results computed on a "
                "subset are provisional and must be labelled as such.")
        return "\n".join(lines)


# =============================================================================
# Loading
# =============================================================================
#
# The storey filter is applied to the SPINE and only to the spine; everything
# else is joined onto it. That is the structural fix for the nb 073 failure
# where the replicate clouds were loaded with n_storeys=3 while the estimates
# were loaded unfiltered, leaving 60-row clouds against 120-row estimates --
# caught only much later, and only by accident, inside mra.validate_studies.
#
# k_samples defaults to None (read whatever is on disk) rather than to a
# literal. Hardcoding it is what broke nb 073 when the bootstrap was re-run at
# k=5000: the container should report the replicate count, not assert it.


def _load_cloud(root, arm: str, quantity: str, im_tag: str,
                k_samples: int | None) -> pd.DataFrame | None:
    """One arm log replicate cloud, indexed to match its natural scope.

    Returns ``None`` when the saved frames are missing, so the caller can decide
    whether that is fatal (core arms) or merely worth a note (variants).

    Both scopes go through ``mra.reformat_bootstrap_df``, unlike nb 073 which
    used it for the by_site frames and a bare ``.T`` for the by_group ones. That
    asymmetry is what forced a stray ``.to_numpy()`` and a separate label
    expression in the A2 cells; handling the two identically removes both.
    """
    scope = SCOPE_OF_ARM[arm]
    frames = mra.load_saved_bootstrap(
        root, ARMS[arm], (quantity,), None, im_tag, scope, k_samples=k_samples)
    if frames is None:
        return None

    cloud = mra.reformat_bootstrap_df(frames[quantity])
    # A group label ("group_3s_00") already encodes its storey count, so the
    # n_storeys level on a by_group frame is redundant. Dropping it here is what
    # makes the group index line up with `groups` without a `set_axis` idiom
    # repeated at every call site.
    if scope == "by_group" and "n_storeys" in (cloud.index.names or []):
        cloud = cloud.droplevel("n_storeys")
    return np.log(cloud)


def _collapse_to_groups(rows: pd.DataFrame, column: str, groups: pd.Index,
                        atol: float = 1e-10) -> pd.Series:
    """Collapse a design-level column from row scope (120) back to group scope (51).

    nb 072 writes every arm at by_site scope, so the four design-level arms
    appear in the dataset replicated across the rows of their design group.
    This undoes that -- and ASSERTS the value really is constant within a group
    while doing so.

    That assertion is the point. If a design-level quantity varies within its
    design group then the row-to-design mapping is wrong, and every covariance
    built on it afterwards is wrong. The check is two lines and the failure mode
    is silent and total, so it runs on every load.
    """
    grouped = rows.groupby("design_group_id", sort=False)[column]
    spread = grouped.transform(lambda s: s.max() - s.min())
    bad = spread[spread > atol]
    if len(bad):
        offenders = rows.loc[bad.index, "design_group_id"].unique()[:5]
        raise ValueError(
            f"{column!r} is not constant within its design group (max spread "
            f"{bad.max():.3g} > {atol:g}); first offenders {list(offenders)}. "
            "A design-level arm that varies within a design means the "
            "row -> design mapping is wrong.")
    return grouped.first().reindex(groups)


def load_fragility_data(
    root: Path | str,
    im_tag: str,
    quantity: str = "theta",
    *,
    k_samples: int | None = None,
    n_storeys: int | Sequence[int] | None = None,
    dataset_csv: Path | str | None = None,
    check: bool = True,
    check_against_estimates: bool = False,
    verbose: bool = True,
) -> FragilityData:
    """Load one quantity of every available arm into a :class:`FragilityData`.

    This replaces roughly fifty lines of notebook boilerplate -- five estimate
    frames, five replicate dicts, ten reshapes, fourteen log transforms and a
    four-times-repeated ``set_axis`` idiom -- with one call.

    Parameters
    ----------
    root : Path or str
        The bootstrapping directory, ``cfg["proc_data"]["bootstrapping"]``.
    im_tag : str
        e.g. ``"AvgSA_03"``.
    quantity : {"theta", "beta"}
        Which fragility parameter to load. This is the ONLY change needed to run
        the whole ladder on the dispersion (roadmap A12-A16).
    k_samples : int, optional
        Guard against picking up a cloud of the wrong size. ``None`` (default)
        accepts whatever is on disk and reports it as ``FragilityData.k``.
    n_storeys : int or sequence of int, optional
        Restrict the study. ``None`` (default) is the full 120-row study, which
        is what CLAUDE.md section 1 says to assume. Anything else is a
        provisional subset and is flagged as such.
    dataset_csv : Path or str, optional
        Override the nb-072 dataset location; defaults to
        :func:`dataset_csv_path`.
    check : bool
        Run :func:`check_fragility_data` before returning. Leave it on.
    check_against_estimates : bool
        Additionally verify the dataset point estimates against the individual
        ``*_estimates_*.csv`` files. Off by default -- the dataset is the
        preferred source and reading the same number twice is how two sources
        come to disagree -- but useful as a one-off audit of nb 072.
    verbose : bool
        Print which arms loaded, and the check report.

    Returns
    -------
    FragilityData
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")

    # -- 1. the spine. Storey filtering happens HERE and ONLY here. ---------
    path = Path(dataset_csv) if dataset_csv is not None else dataset_csv_path(root, im_tag)
    if not path.is_file():
        raise FileNotFoundError(
            f"no regression dataset at {path} - run nb 072 to write it")
    rows = pd.read_csv(path)

    missing_factors = [c for c in DATASET_FACTORS if c not in rows.columns]
    if missing_factors:
        raise KeyError(
            f"{path.name} is missing factor column(s) {missing_factors} - it "
            "predates the current nb 072 schema; re-run nb 072")

    if n_storeys is not None:
        wanted = [int(n) for n in np.atleast_1d(n_storeys)]
        absent = [n for n in wanted if n not in set(rows["n_storeys"])]
        if absent:
            raise KeyError(f"no {absent}-storey rows in {path.name}")
        rows = rows[rows["n_storeys"].isin(wanted)]

    # Drop the arm columns belonging to the OTHER quantity, so that `quantity`
    # means one thing throughout the container. Without this a theta container
    # still carries every `*_beta` column, and `data.rows.head()` shows columns
    # nothing in the object uses -- confusing, though never wrong, since the
    # off-quantity columns are simply never read.
    #
    # Only the arm x quantity blocks go. The factors stay, and so does anything
    # that is not an arm column -- which is what keeps the future A0b/A3
    # covariates safe, as those are row properties rather than per-arm results.
    # `{arm}_n_obs` also stays: the usable-replicate count is a property of the
    # fit, shared by both quantities.
    other_q = "beta" if quantity == "theta" else "theta"
    drop = [tmpl.format(arm=arm, q=other_q)
            for arm in ARMS for tmpl in DATASET_ARM_COLUMNS]
    rows = rows.drop(columns=[c for c in drop if c in rows.columns])

    rows = rows.set_index(["site", "n_storeys"]).sort_index()
    groups = pd.Index(sorted(rows["design_group_id"].unique()), name="design_group_id")

    # -- 2. the clouds, reindexed ONTO the spine ---------------------------
    reps: dict[str, pd.DataFrame | None] = {}
    estimates: dict[str, pd.Series | None] = {}
    estimates_bc: dict[str, pd.Series | None] = {}

    for arm, symbol in ((a, SYMBOLS[a]) for a in ARMS):
        cloud = _load_cloud(root, arm, quantity, im_tag, k_samples)
        if cloud is None:
            if arm in CORE_ARMS:
                raise FileNotFoundError(
                    f"no saved {quantity} bootstrap for the core arm {arm!r} "
                    f"({ARMS[arm]}) - run nb 070 first")
            if verbose:
                print(f"  {arm:8s} ({symbol:5s}): no saved bootstrap - skipped")
            reps[symbol] = estimates[symbol] = estimates_bc[symbol] = None
            continue

        est_col = f"{arm}_log_{quantity}"
        bc_col = f"{arm}_log_{quantity}_bc"
        for col in (est_col, bc_col):
            if col not in rows.columns:
                raise KeyError(
                    f"{path.name} has no column {col!r} - re-run nb 072, which "
                    "writes the log and bias-corrected columns for every arm")

        if SCOPE_OF_ARM[arm] == "by_site":
            target = rows.index
            estimates[symbol] = rows[est_col]
            estimates_bc[symbol] = rows[bc_col]
        else:
            target = groups
            estimates[symbol] = _collapse_to_groups(rows, est_col, groups)
            estimates_bc[symbol] = _collapse_to_groups(rows, bc_col, groups)

        absent = sorted(set(target) - set(cloud.index))
        if absent:
            raise KeyError(
                f"{len(absent)} spine label(s) have no {arm!r} replicates: "
                f"{absent[:5]}{'...' if len(absent) > 5 else ''}. The dataset and "
                "the bootstrap frames disagree about what was run.")
        reps[symbol] = cloud.reindex(target)
        if verbose:
            print(f"  {arm:8s} ({symbol:5s}): {reps[symbol].shape[0]:>3d} x "
                  f"{reps[symbol].shape[1]} replicates")

    data = FragilityData(quantity=quantity, im_tag=im_tag, rows=rows, groups=groups,
                         reps=reps, estimates=estimates, estimates_bc=estimates_bc)

    if check:
        report = check_fragility_data(
            data, root=root, check_against_estimates=check_against_estimates)
        if verbose:
            print(report)
    return data


# =============================================================================
# Structural checks
# =============================================================================
#
# These are the checks nb 073 had nowhere to put. Each one catches a failure
# that is otherwise invisible, or visible only much further downstream where the
# cause is no longer obvious.


def check_fragility_data(data: FragilityData, root: Path | str | None = None,
                         check_against_estimates: bool = False,
                         rtol: float = 1e-10) -> str:
    """Validate a loaded container. Returns a printable report; raises on failure.

    The checks, in order of how sharp they are:

    1. **The roadmap identity** ``y0_i == y1_i + y2_{g[i]}``. This holds
       EXACTLY -- not approximately -- because y2 is constant within a design
       group, which is why the roadmap calls it "the identity that makes the
       whole roadmap hang together". It is therefore a free and extremely sharp
       test that Z, the label-based expansion and the replicate alignment are
       all correct simultaneously. It runs on the point estimates and on every
       replicate.
    2. **The variance split** ``v1[i] == v_a[i] + V_bb[g[i], g[i]]``, exact
       because the SS and FX arms use disjoint record sets so the cross term is
       zero. Fails if the row-to-design map went positional.
    3. **V_bb is really dense** -- if the mean off-diagonal correlation is ~0
       then the replicates did not actually share record indices, and the whole
       argument for carrying a dense V_bb is vacuous.
    4. **The dataset variances match the clouds**, an internal consistency check
       needing no extra files.
    5. Row and group counts, against the full-study values of CLAUDE.md section 1.
    """
    lines: list[str] = []

    # -- 1. the identity ---------------------------------------------------
    y0 = make_contrast(data, "y0")
    y1 = make_contrast(data, "y1")
    y2 = make_contrast(data, "y2")
    y2_rows = data.expand(y2.y_hat)
    np.testing.assert_allclose(
        y0.y_hat.to_numpy(), (y1.y_hat + y2_rows).to_numpy(), rtol=rtol,
        err_msg="y0 != y1 + y2[g[i]] on the point estimates - Z, the group "
                "expansion or the row alignment is wrong")
    np.testing.assert_allclose(
        y0.y.to_numpy(), (y1.y + data.expand(y2.y)).to_numpy(), rtol=rtol,
        err_msg="y0 != y1 + y2[g[i]] on the replicates - the three arms are not "
                "aligned replicate-for-replicate")
    lines.append("  y0 == y1 + y2[g[i]]        : ok (estimates and all replicates)")

    # -- 2. the variance split, tested against Monte-Carlo error -----------
    #
    # The roadmap identity is v1[i] = v_a[i] + V_bb[g[i], g[i]], exact because
    # the SS and FX arms use disjoint record sets so Cov(a_i, b_{g[i]}) = 0.
    #
    # But that is exact for the POPULATION covariance, not for the sample
    # covariance computed from k replicates. The estimate of a zero covariance
    # is not zero, it is noise of order 1/sqrt(k), so the identity holds only up
    # to Monte-Carlo error. `mra.check_variance_split` with its default
    # rtol=1e-6 therefore fails on any real bootstrap output -- at k=5000 a 3%
    # relative discrepancy is entirely expected -- and would only pass if v1 had
    # been ASSEMBLED as v_a + V_bb rather than measured.
    #
    # So test the thing the identity is actually asserting: that the implied
    # cross-correlation between the two arms is zero to within Monte-Carlo
    # error. That is a real test of "disjoint record sets", and unlike a fixed
    # rtol it gets sharper as k grows rather than merely failing louder.
    v_a = np.diag(data.V_aa)
    v_b = np.diag(data.V_bb)[data.g]
    cross = (v_a + v_b - y1.v.to_numpy()) / 2.0          # = Cov_r(a_i, b_{g[i]})
    corr = cross / np.sqrt(v_a * v_b)
    # The sampling SD of a near-zero correlation estimated from k paired draws
    # is ~1/sqrt(k). Across n rows the largest |z| grows slowly with n, so allow
    # a generous threshold: this is looking for a systematic coupling, not for
    # the tail of 120 standard normals.
    z = np.abs(corr) * np.sqrt(data.k)
    if z.max() > 6.0:
        raise AssertionError(
            f"the SS and FX arms are correlated beyond Monte-Carlo error "
            f"(worst row {np.argmax(z)}: corr {corr[np.argmax(z)]:+.4f}, "
            f"{z.max():.1f} MC standard errors). The roadmap says these arms use "
            "disjoint record sets, so this means either the design mapping is "
            "positional rather than label-based, or the replicates are misaligned.")
    lines.append(
        f"  v1 == v_a + V_bb[g,g]      : ok to MC error "
        f"(implied corr mean {corr.mean():+.4f}, max |z| {z.max():.1f} sigma)")

    # -- 3. V_bb really is dense ------------------------------------------
    stats_bb = mra.check_V_bb(data.V_bb, n_replicates=data.k, verbose=False)
    lines.append(
        f"  V_bb mean off-diag corr    : {stats_bb['mean_offdiag_corr']:+.4f} "
        f"(min eig {stats_bb['min_eigenvalue']:+.2e}, k/J {stats_bb['k_over_J']:.1f})")

    # -- 4. dataset variances against the clouds ---------------------------
    for symbol in data.available:
        arm = ARM_OF_SYMBOL[symbol]
        col = f"{arm}_var_log_{data.quantity}"
        if col not in data.rows.columns:
            continue
        from_cloud = data.rep(symbol).var(axis=1)          # ddof=1
        if data.scope_of(symbol) == "group":
            from_csv = _collapse_to_groups(data.rows, col, data.groups)
        else:
            from_csv = data.rows[col]
        np.testing.assert_allclose(
            from_cloud.to_numpy(), from_csv.to_numpy(), rtol=1e-6,
            err_msg=f"{col} in the dataset disagrees with the replicate cloud for "
                    f"{arm!r} - the dataset and the bootstrap frames are out of step")
    lines.append("  dataset var == cloud var   : ok for every loaded arm")

    # -- 4b. optional audit against the standalone estimate files ----------
    if check_against_estimates:
        if root is None:
            raise ValueError("check_against_estimates=True needs `root`")
        _check_against_estimate_files(data, root)
        lines.append("  dataset == estimate files  : ok")

    # -- 5. shape ----------------------------------------------------------
    lines.append(f"  shape                      : n={data.n}, J={data.J}, "
                 f"sites={data.n_sites}, k={data.k}")
    if data.n != 120 or data.J != 51:
        warnings.warn(
            f"this is a {data.n}-row / {data.J}-group subset, not the full "
            "120-row / 51-group study. Results computed on it are PROVISIONAL "
            "and must be labelled as such (CLAUDE.md section 1).",
            RuntimeWarning, stacklevel=2)
        lines.append("  NOTE                       : provisional subset, not the "
                     "full study")

    return "checks:\n" + "\n".join(lines)


def _check_against_estimate_files(data: FragilityData, root: Path | str) -> None:
    """Audit the dataset point estimates against the standalone estimate CSVs.

    Off the normal path by design. The dataset is the preferred source; this
    exists so that preference can be justified once rather than assumed forever.
    """
    for symbol in data.available:
        arm = ARM_OF_SYMBOL[symbol]
        scope = SCOPE_OF_ARM[arm]
        ests = mra.load_estimates(root, data.im_tag, scope=scope,
                                  arms=[ARMS[arm]])[ARMS[arm]]
        got = np.log(ests[data.quantity])
        if scope == "by_group":
            got = got.droplevel("n_storeys").reindex(data.groups)
        else:
            got = got.reindex(data.rows.index)
        np.testing.assert_allclose(
            data.hat(symbol).to_numpy(), got.to_numpy(), rtol=1e-12,
            err_msg=f"dataset {arm}_log_{data.quantity} disagrees with "
                    f"{mra.estimates_csv_path(root, ARMS[arm], data.im_tag, scope).name}")


# =============================================================================
# Contrasts
# =============================================================================
#
# Everything in the roadmap is a contrast of a, b and c (analysis_models.md
# section 0). Making a contrast a first-class object is what stops nb 073
# rebinding `y0` to mean three different things in three different sections.
#
# The scope of a contrast is a property of the contrast, not a suffix on its
# name. y0 and y1 are 120-row quantities; y2 is a 51-row quantity and STAYS one.
# Replicating y2 to 120 rows would invent 69 rows of information that does not
# exist and shrink SE(m2) by roughly sqrt(120/51) ~ 1.53 for nothing
# (analysis_models.md section 1(b)).

# name -> (left symbol, right symbol, scope, label)
_CONTRAST_SPECS: dict[str, tuple[str, str, str, str]] = {
    "y0": ("a", "c", "row",   "MSA-SS / IDA-FX"),   # A0 -- the headline
    "y1": ("a", "b", "row",   "MSA-SS / MSA-FX"),   # A1 -- the record-set effect
    "y2": ("b", "c", "group", "MSA-FX / IDA-FX"),   # A2 -- the procedure effect
    # The collapse-bracket variants: the same contrasts against a different c.
    # nb 073 section 2.4 was written up but never implemented; it is these lines.
    "y0_ub":  ("a", "c_ub",  "row",   "MSA-SS / IDA-FX (upper bound)"),
    "y0_avg": ("a", "c_avg", "row",   "MSA-SS / IDA-FX (geom. avg.)"),
    "y2_ub":  ("b", "c_ub",  "group", "MSA-FX / IDA-FX (upper bound)"),
    "y2_avg": ("b", "c_avg", "group", "MSA-FX / IDA-FX (geom. avg.)"),
}


@dataclass(frozen=True)
class Contrast:
    """One contrast of two arms: effect sizes, replicates, and their FULL covariance.

    Carrying ``V`` (the n x n matrix) alongside ``v`` (its diagonal) is the
    whole point of this object. The DerSimonian-Laird estimator can only use
    ``v``; REML needs ``V``; and the gap between the two IS a finding -- a
    diagonal V understates SE(m) for every contrast that touches a fixed-record
    -set arm, because those arms share sampling error across rows
    (analysis_models.md section 0). Having both on one object makes that
    comparison one line instead of a reconstruction.

    Attributes
    ----------
    name : str
        ``"y0"``, ``"y1"``, ``"y2"`` or a collapse-bracket variant.
    label : str
        Human-readable, for plot titles and table rows.
    quantity : {"theta", "beta"}
    scope : {"row", "group"}
        ``"row"`` is the 120-row study; ``"group"`` is the 51 design groups.
    terms : (str, str)
        The two symbols differenced, e.g. ``("a", "c")``.
    y_hat : Series
        (n,) the effect sizes, log-ratios. ``exp(y_hat)`` is a ratio.
    y : DataFrame
        (n, k) the replicate effect sizes. Column r is arm-coherent: the same
        record resample drove every arm in it, which is what makes the outer
        bootstrap in :func:`bootstrap_fits` correct for free.
    rows : DataFrame
        The factor table for THESE rows, at this scope.
    V : ndarray
        (n, n) the full sampling covariance.
    """

    name: str
    label: str
    quantity: str
    scope: str
    terms: tuple[str, str]
    y_hat: pd.Series
    y: pd.DataFrame
    rows: pd.DataFrame
    V: np.ndarray

    @property
    def n(self) -> int:
        """Number of studies in the meta-analytic sense: 120 rows or 51 designs."""
        return len(self.y_hat)

    @property
    def k(self) -> int:
        """Number of bootstrap replicates."""
        return self.y.shape[1]

    @cached_property
    def v(self) -> pd.Series:
        """(n,) the diagonal of ``V``, labelled -- what DerSimonian-Laird takes.

        Equal to ``self.y.var(axis=1)`` with ddof=1. Taking it as the bootstrap
        variance of the DIFFERENCE, rather than assembling it as
        ``v_left + v_right``, is both algebraically identical and less
        error-prone, and it automatically carries any covariance between the two
        arms rather than assuming there is none (analysis_models.md, A1 and A2).
        """
        return pd.Series(np.diag(self.V), index=self.y_hat.index,
                         name=f"v_{self.name}")

    @cached_property
    def labels(self) -> list[str]:
        """Row labels for forest plots: "12_3s" by row, "group_3s_04" by group."""
        if self.scope == "group":
            return [str(lab) for lab in self.y_hat.index]
        return [f"{site}_{ns}s" for site, ns in self.y_hat.index]

    @cached_property
    def g(self) -> np.ndarray:
        """Design-group code per row, at THIS scope.

        At group scope this is the identity, which is the formal statement that
        one row IS one design -- so A2 needs no group structure.
        """
        groups = pd.Index(sorted(self.rows["design_group_id"].unique()))
        return mra.build_codes(self.rows["design_group_id"], groups)

    @cached_property
    def site_codes(self) -> np.ndarray:
        """Site code per row -- ``sigma[i]`` -- at THIS scope.

        Only defined at row scope: a design group spans several sites, so a
        group-scope contrast has no single site to point at.
        """
        if self.scope != "row":
            raise ValueError(
                f"contrast {self.name!r} is at group scope, where a site factor is "
                "not defined - a design group spans several sites")
        sites = pd.Index(sorted(self.rows.index.get_level_values("site").unique()))
        return mra.build_codes(self.rows.index.get_level_values("site"), sites)

    def quantiles(self, q: Sequence[float] = (0.025, 0.975)) -> pd.DataFrame:
        """Percentile intervals per study, from the replicates.

        Percentiles rather than a normal approximation because they are
        transformation-respecting: the interval on the log scale exponentiates
        to the interval on the ratio scale without further correction.
        """
        out = self.y.quantile(q=list(q), axis=1).T
        out.columns = [f"q{100 * p:g}" for p in q]
        return out

    def forest_frame(self, q: Sequence[float] = (0.025, 0.975)) -> pd.DataFrame:
        """The tidy frame a forest plot wants: effect, lower, upper, label."""
        qs = self.quantiles(q)
        return pd.DataFrame({
            "effect": self.y_hat.to_numpy(),
            "lb": qs.iloc[:, 0].to_numpy(),
            "ub": qs.iloc[:, 1].to_numpy(),
            "labels": self.labels,
        })

    def diag_V(self) -> np.ndarray:
        """``np.diag(self.v)`` -- the covariance a two-level model assumes.

        Provided explicitly so that "fit with the diagonal" is a visible choice
        at the call site rather than an accident of which array was passed.
        """
        return np.diag(self.v.to_numpy())

    def summary(self) -> str:
        """One-line description, for printing."""
        return (f"{self.name} = {self.terms[0]} - {self.terms[1]}  ({self.label}), "
                f"{self.scope} scope, n={self.n}, k={self.k}, "
                f"mean ratio {np.exp(self.y_hat.mean()):.4f}")


def make_contrast(data: FragilityData, name: str,
                  *, corrected: bool = False) -> Contrast:
    """Build one contrast from a loaded container.

    Written once and table-driven, so adding a contrast is a ``_CONTRAST_SPECS``
    entry rather than another copy of this function.

    Parameters
    ----------
    data : FragilityData
    name : str
        A key of ``_CONTRAST_SPECS``.
    corrected : bool
        Use the bias-corrected point estimates. See :meth:`FragilityData.hat`
        for why this is not the default.
    """
    if name not in _CONTRAST_SPECS:
        raise KeyError(f"unknown contrast {name!r} - known: {sorted(_CONTRAST_SPECS)}")
    left, right, scope, label = _CONTRAST_SPECS[name]

    if scope == "row":
        L, L_hat = data.to_rows(left, corrected)
        R, R_hat = data.to_rows(right, corrected)
        rows = data.rows
    else:
        # Both sides must already be at group scope; a row-scope arm cannot be
        # collapsed to a design without averaging away the thing being measured.
        for sym in (left, right):
            if data.scope_of(sym) != "group":
                raise ValueError(
                    f"contrast {name!r} is at group scope but {sym!r} is a row-scope "
                    "arm; it cannot be collapsed to a design without discarding the "
                    "site-to-site variation that is the object of the study")
        L, L_hat, _ = data.term(left, corrected)
        R, R_hat, _ = data.term(right, corrected)
        # One row per design: carry the factor columns that are constant within
        # a group, so a group-scope contrast still has a usable factor table.
        rows = (data.rows.reset_index()
                .groupby("design_group_id", sort=False)
                .first()
                .reindex(data.groups))
        rows["design_group_id"] = rows.index

    # pandas aligns on the shared index, so a mismatch becomes a NaN rather than
    # a silent shuffle -- and the check below turns that NaN into an error.
    y = L - R
    y_hat = L_hat - R_hat
    if y.isna().any().any() or y_hat.isna().any():
        raise ValueError(
            f"contrast {name!r} produced NaNs - the two arms do not share an index")

    # The single most important line in the module. Built as the full sample
    # covariance of the replicates of the contrast itself, it carries every
    # block that exists -- the 2x2 same-site coupling inside V_aa, the density
    # of V_bb, the V_bc cross-term in y2 -- without the code ever asserting
    # which pairs are coupled. Whatever structure the resampling generated is
    # what gets reported (CLAUDE.md section 2).
    #
    # Do NOT replace this with diag(v_a) + Z V_bb Z'. That decomposed form is
    # for the check_variance_split diagnostic, not for the fit: it assumes V_aa
    # is diagonal, which it is not.
    V = mra.bootstrap_cov(y.to_numpy(dtype=float), rowvar=True)

    return Contrast(name=name, label=label, quantity=data.quantity, scope=scope,
                    terms=(left, right), y_hat=y_hat, y=y, rows=rows, V=V)


# =============================================================================
# Model specifications
# =============================================================================
#
# One rung of the roadmap is DECLARED here rather than coded in a notebook.
# That is the whole design goal: the six-cell block nb 073 repeats three times
# (effect sizes -> forest plot -> heterogeneity -> replicate refits -> intervals
# -> SE comparison) becomes one row of this table plus one call to
# ModelRegistry.fit.


@dataclass(frozen=True)
class ModelSpec:
    """One rung of the ladder in ``admin/analysis_models.md``.

    Attributes
    ----------
    model_id : str
        The roadmap id, e.g. ``"A1a"``. This is the registry key, so it is also
        what stops one fit overwriting another.
    contrast : str
        Which contrast it is fitted to -- a key of ``_CONTRAST_SPECS``.
    estimators : tuple of str
        Any of ``"DL"`` (DerSimonian-Laird, Borenstein Ch. 12/14) and ``"REML"``
        (``mra.fit_reml``). DL is the hand-calculation reference; REML is the
        primary, because DL is downward-biased when the sampling variances are
        heterogeneous, which here they are.
    components : tuple of str
        Extra variance components beyond the row-level one, from
        ``{"design", "site"}``. Empty for a two-level model. ``tau_u`` (the row
        level, which absorbs the design x site interaction) is ALWAYS present
        and is not listed here.
    covariates : tuple of str
        Column names in ``Contrast.rows`` to use as fixed effects (A0b / A3).
        Centred and scaled by :func:`build_X`.
    diagonal_V : bool
        Fit with ``diag(v)`` instead of the full ``V``. Correct only where the
        two arms use disjoint record sets; see the note on A1a below.
    note : str
        Why this rung exists and what it does and does not license. Printed by
        :meth:`ModelFit.summary`, so the reasoning travels with the number.
    """

    model_id: str
    contrast: str
    estimators: tuple[str, ...] = ("DL", "REML")
    components: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    diagonal_V: bool = False
    note: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "A0a": ModelSpec(
        "A0a", "y0", ("DL", "REML"), note="""
        The headline: site-specific MSA against fixed-set IDA, intercept only.
        exp(m0) is the correction factor an engineer would apply to a
        FEMA-P695-style IDA fragility to get the site-specific answer.

        Fitted with the FULL V. Every row in a design group subtracts the SAME
        c_j, so the sampling errors are correlated across rows and a diagonal V
        would understate SE(m0). Report the prediction interval alongside the
        confidence interval (Borenstein Ch. 17, not Ch. 14): the CI says where
        the MEAN correction lies, the PI says where the NEXT building lands,
        and it is the second one an engineer needs."""),

    "A0b": ModelSpec("A0b", "y0_avg", ("REML",), note="""
            A0a against the GEOMETRIC MEAN of the IDA collapse bracket. See A0a."""),

    "A0c": ModelSpec("A0c", "y0_avg", ("REML",), components=("design", "site"), note="""
            A0a against the GEOMETRIC MEAN of the IDA collapse bracket. 
            In this model the bewteen-study variance is decomposed into random-effects,
            building group, and site effects, based on the results of A1b and A1c.
            The random-effects includes the site * structure interaction + any 
            betwen row randomness. 
            
            Using the full variance matrix (V_aa + ZVbbZ') rather 
            than the diagonal will capture crossing between the rows.
            """),

    "A0d": ModelSpec("A0d", "y0", ("REML",), components=("design", "site"), note="""
        A0a with the bewteen-study variance decomposed into random-effects,
        building group, and site effects, based on the results of A1b and A1c.
        The random-effects includes the site * structure interaction + any 
        betwen row randomness. 
        
        Using the full variance matrix (V_aa + ZVbbZ') rather 
        than the diagonal will capture crossing between the rows.
        """),

    "A1a": ModelSpec(
        "A1a", "y1", ("DL", "REML"), diagonal_V=True, note="""
        Site-specific MSA against fixed-set MSA: the pure record-set effect,
        with the analysis procedure held constant.

        This is the one rung where a DIAGONAL V_known is exactly correct, and
        it is worth being clear why. v1[i] = v_a[i] + V_bb[g[i], g[i]] holds
        exactly because the SS and FX arms use disjoint record sets, so their
        cross term is zero. That makes A1a the regression test tying REML back
        to DerSimonian-Laird: with Gs = [I] and a diagonal V_known the two must
        agree (CLAUDE.md section 5).

        Note this is about the DIAGONAL being right, not about the off-diagonal
        being absent -- rows in the same design group still share b_j. A1b and
        A1c are what take that seriously; A1a deliberately does not, which is
        why its tau^2 is the baseline everything else is measured against
        rather than a reportable number."""),

    "A1b": ModelSpec(
            "A1b", "y1", estimators=("REML",), components=("design",), note="""
            Site-specific MSA against fixed-set MSA: the pure record-set effect,
            with the analysis procedure held constant.
    
            In this model the bewteen-study variance is split into random-effects,
            and building group effects. The random-effects cannot be considered
            as the "variance due to the site" because it includes the 
            (site + site * structure interaction + random) components. 
            
            Using the full variance matrix (V_aa + ZVbbZ') rather than the diagonal 
            will capture crossing between the rows."""),

    "A1c": ModelSpec(
                "A1c", "y1", estimators=("REML",), components=("design", "site"), note="""
                Site-specific MSA against fixed-set MSA: the pure record-set effect,
                with the analysis procedure held constant.
        
                In this model the bewteen-study variance is split into random-effects,
                building group, and site effects. The random-effects includes the 
                site * structure interaction + any betwen row randomness. 
                
                Using the full variance matrix (V_aa + ZVbbZ') rather 
                than the diagonal will capture crossing between the rows."""),

    "A2": ModelSpec(
        "A2", "y2", ("DL", "REML"), note="""
        Fixed-set MSA against fixed-set IDA, at 51 rows because one row IS one
        design -- the fixed-set stripe IMLs come from the fragility curve, not
        from site hazard, so these arms depend on the design only. Replicating
        to 120 rows would invent information and shrink SE(m2) by about 1.53 for
        nothing.

        Both arms share one record set, so the record effect cancels and what is
        left is the PROCEDURE effect: stripe-count binomial MLE against
        lognormal-of-capacities. Expect a small non-zero m2 for that reason and
        say so up front, so that a non-zero m2 is not misread as a record
        effect. No group structure is needed, but the full V still is -- V_bb,
        V_cc and the V_bc cross term are all dense."""),

    # ---- the collapse-bracket variants -------------------------------------
    # nb 073 section 2.4 described these and never implemented them. They are
    # table entries, not code. Read them as a SENSITIVITY BAND on A0a/A2: the
    # three IDA arms are one analysis read three ways off the collapse bracket,
    # so the spread across them is the cost of that modelling choice, not three
    # independent findings.
    "A2_ub": ModelSpec("A2_ub", "y2_ub", ("DL", "REML"), note="""
        A2 against the upper bound of the IDA collapse bracket. See A2."""),
    "A2_avg": ModelSpec("A2_avg", "y2_avg", ("DL", "REML"), note="""
        A2 against the geometric mean of the IDA collapse bracket. See A2."""),
}


def build_components(spec: ModelSpec, con: Contrast) -> dict[str, tuple]:
    """Map component names onto factor matrices, in the ``dsim`` mapping format.

    Returns the same ``{name: (factor, tau_true)}`` mapping
    ``dsim.simulate_reml_fits`` takes. ``tau_true`` is a placeholder ``0.0`` on
    the real-data path -- it is ignored by ``dsim.build_component_gs``, which
    only needs the factor matrices, but the mapping must still validate, so it
    has to be a real number rather than ``None``. Replace the placeholders with
    plausible values to hand the same spec to the simulator.

    That shared format is the point: a spec can be handed
    straight to the simulator to check the fit can recover what it claims to
    estimate, and the Gs are then built by the SAME code in both places. If they
    were built by two different pieces of code, the simulation would not be
    testing the fit that was actually run -- which is the argument
    ``dsim.build_component_gs`` makes in its own docstring.

    ``tau_u`` is always appended LAST and never omitted. It absorbs the design x
    site interaction, and dropping it silently credits that interaction to
    whichever of design or site the data happen to favour.
    """
    out: dict[str, tuple] = {}
    for comp in spec.components:
        if comp == "design":
            out["tau_d2"] = (mra.make_indicator(con.g), 0.0)
        elif comp == "site":
            out["tau_s2"] = (mra.make_indicator(con.site_codes), 0.0)
        else:
            raise ValueError(
                f"unknown variance component {comp!r} - expected 'design' or 'site'")
    out["tau_u2"] = (np.eye(con.n), 0.0)
    return out


def build_X(spec: ModelSpec, con: Contrast) -> tuple[np.ndarray, list[str], dict]:
    """Fixed-effects design matrix: an intercept, plus centred/scaled covariates.

    Centring and scaling is not cosmetic (Gelman & Hill Ch. 4): without it the
    intercept is the fitted value at covariate zero, which is usually a building
    that does not exist, and coefficients on different scales cannot be compared
    in magnitude. The centring and scaling are returned so a coefficient can be
    put back on its natural units for reporting.

    Returns
    -------
    (X, names, scaling) : ndarray (n, q), list of str, dict
    """
    X = [np.ones(con.n)]
    names = ["intercept"]
    scaling: dict[str, dict[str, float]] = {}

    for cov in spec.covariates:
        if cov not in con.rows.columns:
            raise KeyError(
                f"covariate {cov!r} is not a column of the contrast factor table; "
                f"it needs adding to the nb-072 dataset. Available: "
                f"{[c for c in con.rows.columns if not any(c.startswith(p + '_') for p in ARMS)]}")
        raw = con.rows[cov].to_numpy(dtype=float)
        mu, sd = float(np.mean(raw)), float(np.std(raw, ddof=1))
        if sd == 0:
            raise ValueError(f"covariate {cov!r} is constant - it cannot be scaled")
        X.append((raw - mu) / sd)
        names.append(cov)
        scaling[cov] = {"mean": mu, "sd": sd}

    return np.column_stack(X), names, scaling


# =============================================================================
# Intervals
# =============================================================================


def prediction_interval(M: float, T_sq: float, V_M: float, n: int,
                        alpha: float = 0.05) -> tuple[float, float]:
    """Prediction interval for the true effect in a NEW study.

    Borenstein et al. Ch. 17, eq. 17.7-17.8::

        M +/- t_{n-2, 1-alpha/2} * sqrt(T^2 + Var(M))

    This is a different question from the confidence interval, and it is the one
    an engineer actually asks. The CI says where the MEAN correction factor
    lies; the PI says where the NEXT building lands. With a large between-study
    variance the two differ by a lot, and quoting the CI alone badly overstates
    how well the next case is pinned down.

    The t distribution has ``n - 2`` degrees of freedom, not ``n - 1``: one is
    spent on M and one on T^2.

    Parameters
    ----------
    M, T_sq : float
        The summary effect and the between-study variance.
    V_M : float
        The variance of the summary effect. Prefer the OUTER-BOOTSTRAP variance
        ``Var_r(M*)`` over the model-based ``V_re``: the latter assumes a
        diagonal sampling covariance, which is false for every contrast here
        that touches a fixed-record-set arm.
    n : int
        Number of studies.
    alpha : float

    Notes
    -----
    ``mra.compute_prediction_interval`` is an empty stub. This implementation is
    kept here rather than filling that stub, to leave ``mra`` untouched; fold
    the two together the next time ``mra`` is edited.
    """
    if n < 3:
        raise ValueError(f"a prediction interval needs at least 3 studies, got {n}")
    half = stats.t.ppf(1 - alpha / 2, n - 2) * np.sqrt(T_sq + V_M)
    return float(M - half), float(M + half)


# =============================================================================
# Fitting
# =============================================================================


@dataclass(frozen=True)
class ModelFit:
    """One fitted model, self-describing. Nothing here is ever rebound.

    In nb 073 the names ``M_star_est``, ``T_sq``, ``v0_boot``, ``lb``, ``ub``,
    ``labels``, ``df`` and ``replicate_fits`` are rebound by A1 and again by A2,
    so the A0 forest plot cannot be redrawn after A1 has run without
    re-executing A0 from the top. Every one of those is a FIELD here, and the
    registry refuses to overwrite a fit, so that failure mode is gone.

    Attributes
    ----------
    spec : ModelSpec
    estimator : {"DL", "REML"}
    contrast : Contrast
    beta : ndarray
        (q,) fixed effects. ``beta[0]`` is the summary effect m.
    se_beta : ndarray
        (q,) MODEL-BASED standard errors. These understate the uncertainty --
        see :attr:`se_boot`.
    tau2 : dict of str -> float
        Variance components by name.
    het : dict or None
        Q, df, Q_df (the excess Q - df), p, s_sq, T_sq, T, I_sq, from
        ``mra.compute_heterogeneity_stats``. Present for DL and for
        intercept-only REML fits; Q, df and p are identical between the two,
        only T_sq and I_sq differ. None for fits with covariates.
    replicates : DataFrame or None
        One row per outer-bootstrap replicate; columns ``M_star``, ``T_sq``.
    x_names : list of str
    scaling : dict
        Covariate centring and scaling, for back-transforming coefficients.
    raw : dict
        The untouched return value of the underlying estimator.
    """

    spec: ModelSpec
    estimator: str
    contrast: Contrast
    beta: np.ndarray
    se_beta: np.ndarray
    tau2: dict[str, float]
    het: dict[str, float] | None = None
    replicates: pd.DataFrame | None = None
    x_names: list[str] = field(default_factory=lambda: ["intercept"])
    scaling: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def quantity(self) -> str:
        return self.contrast.quantity

    @property
    def n(self) -> int:
        return self.contrast.n

    @property
    def m(self) -> float:
        """The summary effect, on the log scale."""
        return float(self.beta[0])

    @property
    def T_sq(self) -> float:
        """Total between-study variance: DL T^2, or the sum of the REML components.

        Summing the REML components is the right comparison to DL because DL has
        only one bucket: everything not explained by sampling error. A model
        that splits that bucket three ways should still account for the same
        total.
        """
        if self.het is not None and "T_sq" in self.het:
            return float(self.het["T_sq"])
        return float(sum(self.tau2.values()))

    def ratio(self) -> float:
        """``exp(m)`` -- the summary effect as a ratio, which is the reportable form.

        Lead with this and with ``exp(tau)`` rather than with I^2. I^2 is a
        PROPORTION of variance that is real, and because the sampling variances
        here are small (many records per stripe) it goes to 1 almost regardless,
        carrying very little information. Borenstein Ch. 16 makes exactly this
        point.
        """
        return float(np.exp(self.m))

    @cached_property
    def V_boot(self) -> float | None:
        """``Var_r(M*)`` across the outer bootstrap, or None if it was not run."""
        if self.replicates is None:
            return None
        return float(self.replicates["M_star"].var(ddof=1))

    @cached_property
    def se_boot(self) -> float | None:
        """Spread of M across the outer-bootstrap replicates, ``sqrt(Var_r(M*))``.

        This is a FINITE-POPULATION standard error. The replicates resample
        records only -- the same 60 sites and 51 designs appear in every one --
        so it answers "how well is the mean correction known FOR THESE
        structures". It carries every sampling covariance, including ones never
        written down (Efron & Tibshirani Ch. 6-7).

        That is a different question from the one :attr:`se_beta` answers; see
        :attr:`se_sampling` for the comparison that is actually like-for-like.
        """
        V = self.V_boot
        return None if V is None else float(np.sqrt(V))

    @cached_property
    def weights(self) -> np.ndarray | None:
        """(n,) the weights this estimator actually applies to y, ``w' y = m``.

        For DL, ``w_i`` proportional to ``1 / (v_i + T^2)`` -- diagonal, so every row is treated
        as independent evidence. For REML it is the GLS row
        ``(X' Sigma^-1 X)^-1 X' Sigma^-1``, which down-weights rows sharing a
        design (they carry duplicated information) and can legitimately go
        NEGATIVE, subtracting a row to cancel the error it shares with others.

        The difference between those two weight vectors is the whole reason the
        two estimators have different bootstrap standard errors on identical
        replicates.
        """
        con = self.contrast
        X, _, _ = build_X(self.spec, con)
        if self.estimator == "DL":
            w = 1.0 / (con.v.to_numpy() + self.T_sq)
            return w / w.sum()
        Sigma = self.raw.get("Sigma")
        if Sigma is None:
            return None
        Si = np.linalg.inv(Sigma)
        return (np.linalg.inv(X.T @ Si @ X) @ X.T @ Si)[0]

    @cached_property
    def se_sampling(self) -> float | None:
        """``sqrt(w' V w)`` -- sampling-error-only SE of THIS estimator.

        The like-for-like partner of :attr:`se_boot`, and the right thing to
        compare it against. It propagates the contrast's own sampling
        covariance ``V`` through the weights the estimator actually uses, which
        is precisely what the outer bootstrap measures. The two should agree to
        Monte-Carlo error; a disagreement means the bootstrap is not carrying
        the covariance the fit was handed.

        Note this is NOT ``se_beta``. ``se_beta`` is
        ``sqrt((X' Sigma^-1 X)^-1)`` with ``Sigma = V + tau^2 G``, so it also
        includes the between-study heterogeneity: it is a SUPERPOPULATION SE,
        answering "how well is the mean known for a NEW structure drawn from
        the same population". Both are legitimate; they answer different
        questions (Gelman & Hill Ch. 21.2, ~pp. 458-460).
        """
        w = self.weights
        if w is None:
            return None
        return float(np.sqrt(w @ self.contrast.V @ w))

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Percentile confidence interval for m, from the outer bootstrap.

        Percentiles rather than ``m +/- z SE`` because they are
        transformation-respecting: exponentiating the endpoints gives the
        interval on the ratio scale directly.
        """
        if self.replicates is None:
            raise ValueError(
                f"{self.model_id} was fitted without an outer bootstrap - pass "
                "n_boot to get a confidence interval")
        lo, hi = self.replicates["M_star"].quantile([alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    def prediction_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Where the NEXT building is expected to land. See :func:`prediction_interval`."""
        V_M = self.V_boot
        if V_M is None:
            V_M = float(self.se_beta[0] ** 2)
            warnings.warn(
                f"{self.model_id}: no outer bootstrap, so the prediction interval "
                "uses the model-based Var(m), which assumes a diagonal sampling "
                "covariance and is therefore too small. Pass n_boot.",
                RuntimeWarning, stacklevel=2)
        return prediction_interval(self.m, self.T_sq, V_M, self.n, alpha)

    def to_row(self) -> dict:
        """One row of a comparison table."""
        row = {
            "model": self.model_id,
            "estimator": self.estimator,
            "quantity": self.quantity,
            "contrast": self.contrast.name,
            "n": self.n,
            "ratio": self.ratio(),
            "m": self.m,
            "se_model": float(self.se_beta[0]),
            "se_sampling": self.se_sampling,
            "se_boot": self.se_boot,
            "tau_total": float(np.sqrt(self.T_sq)),
        }
        if self.replicates is not None:
            lo, hi = self.ci()
            pl, ph = self.prediction_interval()
            row |= {"ci_lo": np.exp(lo), "ci_hi": np.exp(hi),
                    "pi_lo": np.exp(pl), "pi_hi": np.exp(ph)}
        if self.het is not None:
            # `Q_df` from mra is Q - df, the excess over the null expectation.
            # Reported here under an honest name; the real df is het["df"].
            row |= {"Q": self.het["Q"], "Q_df": self.het["df"],
                    "Q_excess": self.het["Q_df"], "Q_p": self.het["p"],
                    "I_sq": self.het["I_sq"]}
        row |= {f"tau_{k.removeprefix('tau_').removesuffix('2')}": float(np.sqrt(v))
                for k, v in self.tau2.items()}
        return row

    def summary(self) -> str:
        """Multi-line report, in ratio units, with the reasoning attached."""
        out = [f"{self.model_id} [{self.estimator}]  {self.contrast.label}"
               f"  ({self.contrast.name} = {self.contrast.terms[0]} - "
               f"{self.contrast.terms[1]}, {self.quantity}, n={self.n})",
               f"  summary effect   : {self.ratio():.4f}   "
               f"(log {self.m:+.4f})"]

        if self.replicates is not None:
            lo, hi = self.ci()
            out.append(f"  95% CI           : {np.exp(lo):.4f} - {np.exp(hi):.4f}"
                       "     <- where the MEAN effect lies")
            pl, ph = self.prediction_interval()
            out.append(f"  95% PI           : {np.exp(pl):.4f} - {np.exp(ph):.4f}"
                       "     <- where the NEXT case lands")

        # Three standard errors, because they answer three different questions.
        # The diagnostic compares se_boot against se_sampling -- NOT against
        # se_beta. se_beta includes tau^2 and so is a superpopulation SE; only
        # se_sampling propagates the same sampling covariance the bootstrap
        # resamples, so only that comparison isolates whether the covariance
        # structure is being carried.
        out.append(f"  SE (superpop.)   : {self.se_beta[0]:.5f}   "
                   "<- includes tau: a NEW structure")
        if self.se_sampling is not None:
            out.append(f"  SE (sampling)    : {self.se_sampling:.5f}   "
                       "<- sqrt(w' V w), these structures")
        if self.se_boot is not None:
            out.append(f"  SE (bootstrap)   : {self.se_boot:.5f}   "
                       f"<- {len(self.replicates)} refits")

        if self.se_boot is not None and self.se_sampling:
            ratio = self.se_boot / self.se_sampling
            if not 0.8 < ratio < 1.25:
                out.append(
                    f"    -> bootstrap SE is {ratio:.2f}x sqrt(w' V w); these should "
                    "agree to Monte-Carlo error. The bootstrap is not carrying the "
                    "covariance the fit was handed - check the replicate alignment, "
                    "or raise n_boot if it is close.")
            # The estimator-quality check: a DIAGONAL V_known cannot see the
            # shared-record covariance, so its weights are near-equal and its
            # sampling SE is inflated relative to a full-V GLS fit.
            if self.spec.diagonal_V or self.estimator == "DL":
                out.append("    -> fitted with a DIAGONAL sampling covariance: the "
                           "model-based SE omits the off-diagonal entirely, and the "
                           "weights ignore that rows in a design share an arm.")

        # The components are stored as VARIANCES (tau_d2, ...) but reported as
        # standard deviations, because on the log scale a tau of 0.09 reads
        # directly as "a factor of e^0.09 = 1.09 from one case to the next",
        # which is the form an engineer can use. Lead with this rather than with
        # I_sq (Borenstein Ch. 16).
        for name, value in self.tau2.items():
            sd_name = name.removesuffix("2")
            out.append(f"  {sd_name:<16s} : {np.sqrt(value):.4f}   "
                       f"(a factor of {np.exp(np.sqrt(value)):.3f})")

        if self.het is not None:
            # NOTE the naming trap: mra returns `Q_df` = Q - df, the EXCESS of Q
            # over its null expectation, not the degrees of freedom. The df is
            # het["df"]. Both are printed under their own names.
            out.append(f"  Q                : {self.het['Q']:.2f} on "
                       f"{self.het['df']} df, p = {self.het['p']:.3g}   "
                       f"(excess over null {self.het['Q_df']:+.2f})")
            # Q is built from the diagonal v only, so its chi-square null holds
            # for independent rows. A fit handed a non-diagonal V is one where
            # rows are known to share sampling error; there Q is inflated and p
            # is optimistic.
            if self.estimator == "REML" and not self.spec.diagonal_V:
                out.append("    -> p is nominal: Q uses the diagonal v, but rows here "
                           "share sampling error (a common b_j / c_j), which inflates Q")
            source = ("T^2 / (s^2 + T^2), REML T^2" if self.estimator == "REML"
                      else "(Q - df) / Q")
            out.append(f"  I_sq             : {self.het['I_sq']:.1f}%   [{source}]   "
                       "(secondary - see Borenstein Ch. 16: I_sq is a PROPORTION "
                       "of variance that is real, and with small sampling "
                       "variances it tends to 1 almost regardless)")

        for q, name in zip(self.beta[1:], self.x_names[1:]):
            out.append(f"  beta[{name}] : {q:+.4f} per SD")

        if self.spec.note:
            out.append("  " + " ".join(self.spec.note.split()))
        return "\n".join(out)


def fit_dl(con: Contrast, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """DerSimonian-Laird two-level random-effects fit. Borenstein Ch. 12/14.

    Kept because it is the estimator the textbook hand calculation walks
    through, and because ``mra.fit_reml`` with ``Gs = [I]`` and a diagonal
    V_known must reproduce it -- which makes it the regression test for the
    REML implementation.

    It cannot go further than this. DL is one moment equation, ``E[Q] =
    (n-1) + C tau^2``: one equation, one unknown, so it cannot identify the two
    or three variance components of A1b/A1c. Worse, Q itself presupposes a
    DIAGONAL sampling covariance, so once the dense block enters there is no Q
    to write down.
    """
    if spec.components:
        raise ValueError(
            f"{spec.model_id} asks DL for components {spec.components}, but DL is a "
            "single moment equation and can only fit one variance component - use "
            "REML for anything past the two-level model")
    M, V_M, SE, T_sq, _ = mra.compute_re_summary_effect(con.v, con.y_hat)
    het = mra.compute_heterogeneity_stats(con.v, con.y_hat)
    return (np.array([M]), np.array([SE]), {"tau_u2": float(T_sq)}, het)


def fit_reml(con: Contrast, spec: ModelSpec, **kw) -> tuple:
    """REML fit via ``mra.fit_reml``, with the Gs built the ``dsim`` way.

    REML is the primary estimator: DL is known to be downward-biased when the
    sampling variances are heterogeneous, and here they very much are.
    """
    V_known = con.diag_V() if spec.diagonal_V else con.V
    Gs, names = dsim.build_component_gs(build_components(spec, con))
    X, x_names, scaling = build_X(spec, con)
    raw = mra.fit_reml(con.y_hat.to_numpy(dtype=float), X, V_known, Gs,
                       names=names, **kw)
    tau2 = {n: float(t) for n, t in zip(raw["names"], raw["tau2"])}

    # Heterogeneity. Q, its df and its p-value depend only on the data and the
    # fixed-effect weights 1/v, so they are IDENTICAL to the DL fit of the same
    # contrast. What REML changes is T^2, and hence I^2, which is formed as
    # T^2 / (s^2 + T^2) with the Higgins & Thompson typical within-study
    # variance s^2. T^2 is the TOTAL of the REML components - the same quantity
    # ModelFit.T_sq reports and DL's single bucket is compared against.
    #
    # Only for intercept-only fits. With covariates the relevant statistic is
    # the residual Q_E on k - p df, and neither Q nor s^2 above is that; left as
    # None until a meta-regression rung actually needs it.
    het = None
    if not spec.covariates:
        het = mra.compute_heterogeneity_stats(con.v, con.y_hat,
                                              T_sq=sum(tau2.values()))
    return raw["beta"], raw["se_beta"], tau2, het, x_names, scaling, raw


def bootstrap_fits(con: Contrast, spec: ModelSpec, *, estimator: str,
                   n_boot: int | None = None, method: str = "cholesky",
                   n_starts: int = 1, **kw) -> pd.DataFrame:
    """Refit the model inside every bootstrap replicate: one row per replicate.

    This is the INFERENTIAL result (Efron & Tibshirani Ch. 6-7, and
    analysis_models.md section 0). It carries every sampling covariance,
    including ones never written down, which the model-based standard error
    cannot.

    One replicate index drives all the arms for free, because column r of
    ``con.y`` was built by differencing column r of each arm -- so the arms in
    it were driven by the same record resample. That coherence is a property of
    how the contrast was constructed, not something this function has to arrange.

    ``V_known`` and the factor matrices are held FIXED at their full-sample
    values across replicates rather than re-estimated per replicate. Doing the
    latter properly needs a double bootstrap (Efron & Tibshirani Ch. 25) -- the
    same argument ``dsim._known_error_sampler`` makes about factorising once.

    ``method="cholesky"`` because it is ~3.5x faster at n=120 and agrees with
    the explicit form to ~1e-13; that only matters inside this loop.

    ``n_starts=1``, against ``fit_reml``'s default of 4, for the same reason and
    with more effect: ~3.9x faster at n=120. The random restarts guard against a
    REML surface that is flat or multi-modal, but that is a property of the
    PROBLEM GEOMETRY, which the full-sample fit has already explored with its
    own restarts. Each replicate perturbs the same surface slightly, so one
    start finds the same optimum -- measured on A0a, 200 replicates at 4 starts
    and at 1 give a bootstrap SE agreeing to every digit printed (0.020764).
    Raise it if a fit is suspected of being multi-modal in a way the full-sample
    fit did not reveal; ``start_spread`` on the full-sample fit is the thing to
    check first.
    """
    columns = list(con.y.columns)
    if n_boot is not None:
        columns = columns[:n_boot]

    if estimator == "REML":
        V_known = con.diag_V() if spec.diagonal_V else con.V
        Gs, names = dsim.build_component_gs(build_components(spec, con))
        X, _, _ = build_X(spec, con)

    out = []
    for col in columns:
        y_r = con.y[col].to_numpy(dtype=float)
        if estimator == "DL":
            M, _, _, T_sq, _ = mra.compute_re_summary_effect(
                con.v, pd.Series(y_r, index=con.y_hat.index))
            out.append({"M_star": float(M), "T_sq": float(T_sq)})
        else:
            raw = mra.fit_reml(y_r, X, V_known, Gs, names=names,
                               method=method, n_starts=n_starts, **kw)
            row = {"M_star": float(raw["beta"][0]),
                   "T_sq": float(np.sum(raw["tau2"]))}
            row |= {n: float(t) for n, t in zip(raw["names"], raw["tau2"])}
            out.append(row)

    return pd.DataFrame(out, index=pd.Index(columns, name="k"))


# =============================================================================
# The registry
# =============================================================================


def _style(defaults: dict, overrides: dict | None, artist_cls) -> dict:
    """Merge user style keywords over defaults, alias-safely.

    A naive ``{**defaults, **overrides}`` breaks as soon as the two spell the
    same property differently: defaults of ``{"lw": 1.0}`` plus a user's
    ``{"linewidth": 2}`` would hand matplotlib both, and it raises. Both dicts
    are therefore first normalised to matplotlib's canonical names for the
    artist that will receive them (``lw`` -> ``linewidth``, ``ls`` ->
    ``linestyle``, ``ms`` -> ``markersize``, ``c`` -> ``color`` ...), so the
    user's value simply replaces the default.
    """
    out = cbook.normalize_kwargs(dict(defaults), artist_cls)
    if overrides:
        out.update(cbook.normalize_kwargs(dict(overrides), artist_cls))
    return out


class ModelRegistry:
    """Every fit for one container, keyed by roadmap id. Refuses silent overwrites.

    The direct answer to the rebinding problem: in nb 073, re-running the A1
    block after A2 leaves the numbers of A2 sitting in variables named as though
    they belonged to A1, and nothing complains. Here a second ``fit("A1a")``
    raises unless ``overwrite=True`` is passed.

    Examples
    --------
    >>> theta = load_fragility_data(root, "AvgSA_03", quantity="theta")
    >>> fits = ModelRegistry(theta)
    >>> fits.fit("A0a", n_boot=500)
    >>> print(fits["A0a", "REML"].summary())
    >>> fits.table()
    """

    def __init__(self, data: FragilityData,
                 specs: dict[str, ModelSpec] | None = None):
        self.data = data
        self.specs = dict(MODEL_SPECS if specs is None else specs)
        self._fits: dict[tuple[str, str], ModelFit] = {}
        self._contrasts: dict[str, Contrast] = {}

    # -- contrasts, memoised so V is computed once per contrast -------------

    def contrast(self, name: str) -> Contrast:
        """The contrast of this name, built once and cached."""
        if name not in self._contrasts:
            self._contrasts[name] = make_contrast(self.data, name)
        return self._contrasts[name]

    # -- fitting -------------------------------------------------------------

    def fit(self, model_id: str, *, n_boot: int | None = None,
            overwrite: bool = False, verbose: bool = False,
            **kw) -> list[ModelFit]:
        """Fit one rung with every estimator its spec asks for.

        This single call replaces the six-cell block nb 073 repeats three times.

        Parameters
        ----------
        model_id : str
            A key of :data:`MODEL_SPECS`.
        n_boot : int, optional
            Number of outer-bootstrap replicates to refit in. ``None`` skips the
            outer bootstrap, which means no confidence interval and only a
            model-based prediction interval. Pass an int for anything
            reportable. Use all of them (``n_boot=data.k``) for the final run.
        overwrite : bool
            Allow replacing an existing fit of the same id.
        """
        if model_id not in self.specs:
            raise KeyError(
                f"unknown model {model_id!r} - known: {sorted(self.specs)}. "
                "A new rung is a MODEL_SPECS entry, not a new block of code.")
        spec = self.specs[model_id]
        con = self.contrast(spec.contrast)

        out: list[ModelFit] = []
        for estimator in spec.estimators:
            key = (model_id, estimator)
            if key in self._fits and not overwrite:
                raise ValueError(
                    f"{model_id} [{estimator}] has already been fitted. Pass "
                    "overwrite=True if you mean to replace it - the refusal is "
                    "deliberate, so a re-run cannot silently change what a later "
                    "cell reports.")

            if estimator == "DL":
                beta, se, tau2, het = fit_dl(con, spec)
                x_names, scaling, raw = ["intercept"], {}, {}
            elif estimator == "REML":
                beta, se, tau2, het, x_names, scaling, raw = fit_reml(con, spec, **kw)
            else:
                raise ValueError(f"unknown estimator {estimator!r}")

            reps = None
            if n_boot is not None:
                if verbose:
                    print(f"  {model_id} [{estimator}]: outer bootstrap over "
                          f"{min(n_boot, con.k)} replicates...")
                reps = bootstrap_fits(con, spec, estimator=estimator, n_boot=n_boot)

            fit_obj = ModelFit(spec=spec, estimator=estimator, contrast=con,
                               beta=np.atleast_1d(beta), se_beta=np.atleast_1d(se),
                               tau2=tau2, het=het, replicates=reps,
                               x_names=x_names, scaling=scaling, raw=raw)
            self._fits[key] = fit_obj
            out.append(fit_obj)
        return out

    # -- access --------------------------------------------------------------

    def __getitem__(self, key) -> ModelFit:
        """``fits["A1a"]`` (the primary estimator) or ``fits["A1a", "DL"]``."""
        if isinstance(key, tuple):
            model_id, estimator = key
        else:
            model_id = key
            spec = self.specs[model_id]
            # REML is the primary; fall back to whatever the spec actually ran.
            estimator = "REML" if "REML" in spec.estimators else spec.estimators[0]
        if (model_id, estimator) not in self._fits:
            raise KeyError(
                f"{model_id} [{estimator}] has not been fitted - call "
                f"fit({model_id!r}) first. Fitted: {sorted(self._fits)}")
        return self._fits[(model_id, estimator)]

    def __contains__(self, key) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    @property
    def fitted(self) -> list[tuple[str, str]]:
        """Every ``(model_id, estimator)`` pair fitted so far, in insertion order."""
        return list(self._fits)

    def table(self, ids: Sequence[str] | None = None) -> pd.DataFrame:
        """One row per fit -- the comparison nb 073 never had.

        Ratio units throughout, because that is what an engineer can use: the
        summary effect as a factor, and tau as the factor by which the next case
        varies.
        """
        rows = [f.to_row() for (mid, _), f in self._fits.items()
                if ids is None or mid in ids]
        if not rows:
            raise ValueError("no fits yet - call fit() first")
        return pd.DataFrame(rows).set_index(["model", "estimator"])

    # -- plotting ------------------------------------------------------------

    def forest(self, model_id: str, estimator: str | None = None, ax=None,
               max_rows: int | None = None, alpha: float = 0.05, *,
               color: str | None = None, show_ci: bool = True,
               show_pi: bool = True,
               point_kws: dict | None = None, interval_kws: dict | None = None,
               summary_kws: dict | None = None, ci_kws: dict | None = None,
               pi_kws: dict | None = None, ref_kws: dict | None = None,
               legend_kws: dict | None = None, label_fontsize: float = 6):
        """Forest plot of the per-study effects, with the summary effect overlaid.

        Written once here rather than copy-pasted per model. Plotted on the
        RATIO scale, with a reference line at 1.0, because a log-ratio of 0 is
        harder to read than a ratio of 1 and the ratio is the reportable form.

        Two bands sit behind the data. The darker is the CONFIDENCE interval of
        the summary effect -- where the mean correction lies. The lighter, wider
        one is the PREDICTION interval -- where the next structure is expected to
        land (Borenstein Ch. 17). The second is the one an engineer should read;
        it is drawn underneath so the CI stays visible inside it.

        Parameters
        ----------
        model_id, estimator : str
            Which fit. ``estimator=None`` picks REML when the model has one.
        ax : Axes, optional
        max_rows : int, optional
            Thin a 120-row plot to evenly spaced studies (by rank of effect)
            when the full figure would be unreadable.
        alpha : float
            1 - coverage for the per-study intervals, the CI and the PI.
        color : str, optional
            Shortcut: recolour both the per-study points and their intervals.
            Defaults to the fixed colour of the contrast's first arm.
        show_ci, show_pi : bool
            Draw the CI / PI band. The CI needs an outer bootstrap (``n_boot``);
            without one the PI falls back to the model-based Var(m), and
            ``ModelFit.prediction_interval`` warns that it is too narrow.
        point_kws, interval_kws, summary_kws, ci_kws, pi_kws, ref_kws : dict
            Style keywords for each element, merged OVER the defaults below, so
            only the keys you pass change. Any keyword the underlying matplotlib
            call accepts is allowed, aliases included (``lw``/``linewidth``,
            ``ls``/``linestyle``, ``ms``/``markersize``). A ``label`` key
            replaces that element's legend entry; ``label="_nolegend_"`` hides
            it.

            point_kws     per-study estimates, ax.plot:
                          marker="o", ls="none", ms=3, color=<arm colour>
            interval_kws  per-study intervals, ax.hlines:
                          lw=1.0, alpha=0.7, color=<arm colour>
            summary_kws   summary-effect line, ax.axvline: color="k", lw=1.4
            ci_kws        CI band, ax.axvspan: color="k", alpha=0.12, zorder=1
            pi_kws        PI band, ax.axvspan: color="k", alpha=0.05, zorder=0
            ref_kws       no-difference line at 1.0, ax.axvline:
                          color="0.4", lw=0.9, ls="--"

        legend_kws : dict, optional
            Passed to ``mra.style_legend``; defaults ``loc="best", fontsize=7``.
        label_fontsize : float
            Font size of the per-study y tick labels.

        Examples
        --------
        >>> fits.forest("A0a", "DL", color="tab:purple",
        ...             pi_kws={"color": "tab:blue", "alpha": 0.08},
        ...             summary_kws={"ls": ":", "lw": 2})
        """
        fit_obj = self[(model_id, estimator)] if estimator else self[model_id]
        con = fit_obj.contrast
        frame = con.forest_frame((alpha / 2, 1 - alpha / 2)).copy()
        frame = frame.sort_values("effect").reset_index(drop=True)

        if max_rows is not None and len(frame) > max_rows:
            keep = np.unique(np.linspace(0, len(frame) - 1, max_rows).astype(int))
            frame = frame.iloc[keep].reset_index(drop=True)

        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 0.18 * len(frame) + 2.0))

        y = np.arange(len(frame))
        eff = np.exp(frame["effect"])
        lo, hi = np.exp(frame["lb"]), np.exp(frame["ub"])
        colour = color if color is not None else arm_colour(con.terms[0])
        pct = f"{100 * (1 - alpha):.0f}%"

        # Layering: PI band (zorder 0) under the CI band (1) under the data
        # (matplotlib's default 2 for lines). The bands are the context, the
        # points are the content.
        if show_pi:
            pi_lo, pi_hi = fit_obj.prediction_interval(alpha)
            ax.axvspan(np.exp(pi_lo), np.exp(pi_hi),
                       **_style({"color": "k", "alpha": 0.05, "zorder": 0,
                                 "label": f"{pct} PI"}, pi_kws, mpatches.Patch))
        if show_ci and fit_obj.replicates is not None:
            ci_lo, ci_hi = fit_obj.ci(alpha)
            ax.axvspan(np.exp(ci_lo), np.exp(ci_hi),
                       **_style({"color": "k", "alpha": 0.12, "zorder": 1,
                                 "label": f"{pct} CI"}, ci_kws, mpatches.Patch))

        ax.hlines(y, lo, hi,
                  **_style({"color": colour, "lw": 1.0, "alpha": 0.7},
                           interval_kws, mcoll.LineCollection))
        # The marker lives in the keywords rather than a "o" format string, so
        # it can be overridden like everything else.
        ax.plot(eff, y,
                **_style({"marker": "o", "ls": "none", "ms": 3.0, "color": colour,
                          "label": "per study"}, point_kws, mlines.Line2D))
        ax.axvline(1.0, **_style({"color": "0.4", "lw": 0.9, "ls": "--",
                                  "label": "no difference"}, ref_kws, mlines.Line2D))
        ax.axvline(fit_obj.ratio(),
                   **_style({"color": "k", "lw": 1.4,
                             "label": f"summary {fit_obj.ratio():.3f}"},
                            summary_kws, mlines.Line2D))

        ax.set_yticks(y)
        ax.set_yticklabels(frame["labels"], fontsize=label_fontsize)
        # ax.set_xscale("log")
        ax.set_xlabel(f"{con.label}   ({con.quantity} ratio)")
        ax.set_title(f"{model_id} [{fit_obj.estimator}] - {con.label}")
        ax.set_ylim(-1, len(frame))
        mra.style_legend(ax, **{"loc": "best", "fontsize": 7, **(legend_kws or {})})
        return ax


# =============================================================================
# Building the dataset (notebook 072)
# =============================================================================
#
# The producer side. nb 072 assembles the dataset this module then consumes, so
# the two must agree on three things: the arm names (ARMS, above), the design
# group labelling, and the per-arm column schema. Each of those lived only in
# nb 072 until now, which meant the consumer had to re-derive them and could
# silently drift.
#
# What deliberately does NOT move here: the cross-file assertions in nb 072
# section 3.0 -- var_nat against the saved `variance`, var_log against
# `se_est**2`, `theta_bc == theta`, `log(beta/beta_bc) == bias`. Those tie
# together files written by different cells of nb 070 and are the actual
# PURPOSE of nb 072: they are a gate on the dataset. Moving them into a library
# that nb 073 imports would turn a one-off gate into a tax paid on every model
# fit, and would hide the one place where a reader can see what is being
# guaranteed. `arm_column_block` therefore returns its intermediates so the
# notebook can keep asserting on them.


def design_group_label(n_storeys: int, group_id: int) -> str:
    """The canonical design-group label, e.g. ``"group_3s_00"``.

    This string is load-bearing in three places: it is the folder name under
    ``wp1_design_groups/``, the index of the by_group bootstrap frames, and the
    join key between the dataset and those frames. Defined once here so the
    producer (nb 072) and the consumer (:class:`FragilityData`) cannot spell it
    differently.
    """
    return f"group_{n_storeys}s_{group_id:02d}"


def build_design_factors(designs_csv: Path | str,
                         expect_designs: int | None = 51,
                         expect_rows: int | None = 120,
                         expect_sites: int | None = 60) -> pd.DataFrame:
    """Read nb 051's design table and attach the two crossed factors.

    ``unique_structural_designs.csv`` lists every ``(site, storeys)`` pair and
    the design group it belongs to, but the group ids RESTART at zero for each
    storey count. This turns them into a single global factor by enumerating
    the sorted ``(storeys, group_id)`` keys, so 3s designs take ``structure_id``
    0-24 and 5s designs 25-50.

    The enumeration is built from the WHOLE table, not from whichever storey
    counts are being exported, so the ids do not shift when a storey count is
    added later. That matters because ``structure_id`` ends up in a saved
    dataset: an id that silently renumbers is worse than one that is missing.

    Parameters
    ----------
    designs_csv : Path or str
        ``cfg["proc_data"]["unique_structural_designs_csv"]``.
    expect_designs, expect_rows, expect_sites : int, optional
        Full-study dimensions to assert (CLAUDE.md section 1). Pass ``None`` to
        skip a check when deliberately working on a subset.

    Returns
    -------
    DataFrame
        The input columns plus ``structure_id`` and ``design_group_id``.
    """
    designs = pd.read_csv(designs_csv)

    keys = (designs[["storeys", "group_id"]]
            .drop_duplicates()
            .sort_values(["storeys", "group_id"])
            .reset_index(drop=True))
    keys["structure_id"] = np.arange(len(keys), dtype=int)

    if expect_designs is not None and len(keys) != expect_designs:
        raise ValueError(f"expected {expect_designs} unique structural designs, "
                         f"found {len(keys)}")
    if expect_rows is not None and len(designs) != expect_rows:
        raise ValueError(f"expected {expect_rows} (site, storeys) pairs, "
                         f"found {len(designs)}")
    if expect_sites is not None:
        sites = sorted(designs["site"].unique())
        if sites != list(range(expect_sites)):
            raise ValueError(f"expected sites 0-{expect_sites - 1}, found "
                             f"{len(sites)} distinct sites")

    designs = designs.merge(keys, on=["storeys", "group_id"], validate="many_to_one")
    designs["design_group_id"] = [design_group_label(n, g) for n, g
                                  in zip(designs["storeys"], designs["group_id"])]
    return designs


def arm_column_block(root: Path | str, prefix: str, arm: str, n_storeys: int,
                     im_tag: str, k_samples: int | None = None,
                     ) -> tuple[dict[str, pd.Series], dict] | None:
    """One arm's dataset columns for one storey count, indexed by site.

    The mechanical half of what nb 072 used to do inline: load the replicate
    cloud, the saved statistics, the bias frame and the point estimates; compute
    both variances; and lay the results out under the column names of
    :data:`DATASET_ARM_COLUMNS`.

    Returns ``None`` when the saved replicate frames for this storey count are
    missing, so the caller can skip a storey count that is not bootstrapped yet.

    Returns
    -------
    (cols, parts) : dict of str -> Series, dict
        ``cols`` is the column block, ready to be assembled into a frame.
        ``parts`` carries the intermediates -- ``ests``, and per quantity the
        cloud, the saved stats row, the bias row and both variances -- so the
        CALLER can assert the cross-file consistency of the nb 070 outputs.
        Those assertions stay in nb 072 on purpose; see the section preamble.

    Notes
    -----
    Both variances are computed from the replicate cloud with ``ddof=1``,
    matching the ``y.var(axis=1)`` that the models are fitted with. The saved
    ``*_stats_*`` frames report ``variance`` through ``np.var`` (``ddof=0``), so
    the two agree only after an ``(N-1)/N`` rescale -- which is one of the
    checks the caller makes.
    """
    boot = mra.load_saved_bootstrap(root, arm, ("theta", "beta"), n_storeys,
                                    im_tag, "by_site", k_samples=k_samples)
    if boot is None:
        return None

    stats_frames = mra.load_saved_bootstrap_stats(
        root, arm, ("theta_stats", "beta_stats"), n_storeys, im_tag, "by_site")
    bias = mra.load_saved_bias(root, [arm], ("theta", "beta"), n_storeys,
                               im_tag, "by_site")[arm]
    ests = mra.load_estimates(root, im_tag, scope="by_site",
                              arms=[arm])[arm].xs(n_storeys, level="n_storeys")

    # nb 070 section 2.4 appends these; a file without them predates that step,
    # and a bare KeyError further down would not say so.
    missing = [c for c in ("theta_bc", "beta_bc") if c not in ests.columns]
    if missing:
        raise KeyError(f"{arm} estimates are missing {missing} - re-run nb 070 "
                       "section 2.4, which appends the bias-corrected columns")

    cols: dict[str, pd.Series] = {}
    parts: dict = {"ests": ests, "quantities": {}}
    n_obs: dict[str, pd.Series] = {}

    for q in ("theta", "beta"):
        cloud = boot[q]                       # k x site
        log_cloud = np.log(cloud)
        stat = stats_frames[f"{q}_stats"]     # site x statistic
        bias_q = bias[q]                      # site x diagnostic

        var_nat = cloud.var(axis=0)           # ddof=1, skips screened replicates
        var_log = log_cloud.var(axis=0)

        cols[f"{prefix}_{q}"] = ests[q]
        cols[f"{prefix}_log_{q}"] = np.log(ests[q])
        cols[f"{prefix}_{q}_bc"] = ests[f"{q}_bc"]
        cols[f"{prefix}_log_{q}_bc"] = np.log(ests[f"{q}_bc"])
        cols[f"{prefix}_var_{q}"] = var_nat
        cols[f"{prefix}_var_log_{q}"] = var_log
        cols[f"{prefix}_log_{q}_bias"] = bias_q["bias"]

        n_obs[q] = stat["N_obs"]
        parts["quantities"][q] = {"cloud": cloud, "stat": stat, "bias": bias_q,
                                  "var_nat": var_nat, "var_log": var_log}

    parts["n_obs"] = n_obs
    cols[f"{prefix}_n_obs"] = n_obs["theta"].astype(int)

    return {k: v.rename(k).rename_axis("site") for k, v in cols.items()}, parts


def order_dataset_columns(data: pd.DataFrame,
                          factors: Sequence[str] = DATASET_FACTORS,
                          arms: Sequence[str] | None = None) -> pd.DataFrame:
    """Put the factors first, then each arm's block in registry order.

    The ``ARMS`` order sets the column order of the saved dataset, which is why
    it is worth pinning in one place. The duplicate check is not paranoia: the
    selection is by prefix, so if one arm prefix were a prefix of another the
    longer name would match twice and the frame would silently gain duplicate
    columns. :data:`ARMS` is asserted prefix-free at import for the same reason.
    """
    arms = list(ARMS if arms is None else arms)
    arm_cols = [c for p in arms for c in data.columns if c.startswith(f"{p}_")]
    if len(set(arm_cols)) != len(arm_cols):
        raise ValueError("an arm prefix shadows another - the column selection "
                         "matched the same column twice")
    return data[list(factors) + arm_cols]
