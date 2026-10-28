"""Data model for the A-series meta-analysis models (the A0-A16 ladder of contrasts
between the MSA-SS, MSA-FX and IDA-FX arms).

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
high. That is the identifiability trap of the crossed A1c model.

The joint theta/beta model (A8)
-------------------------------
The A8 rungs fit theta and beta TOGETHER, to separate the sampling correlation
of (theta_hat, beta_hat) from the correlation of the true corrections. They do
not change the one-quantity-per-container rule: a ``JointContrast`` pairs a
theta Contrast with a beta Contrast, and the A8 rungs live in their own table
(``JOINT_MODEL_SPECS``), fit object (``JointModelFit``) and registry
(``JointModelRegistry``). The bivariate REML is ``mra.fit_reml_mv``. See the
section "The joint theta / beta model (A8)" below, which also records the design
decisions.

Two bootstrap schemes
---------------------
Every spec carries ``bootstrap_scheme``, and the two schemes answer different
questions:

``"record"`` (the default; every id without an ``_hb`` suffix)
    Refit each record-resampling replicate as it is. Right for the mean m and
    its interval, conditional on these sites and designs (a finite-population
    statement). WRONG for every variance component: a replicate carries two
    rounds of sampling noise and the fit is told about one, so each replicate
    tau^2 is inflated by about the mean sampling variance.
``"hybrid"`` (ids ending ``_hb``)
    Refit pseudo-data ``X beta_hat + (new effects drawn from the fitted
    variance components) + (the record-resample deviations)``, which carries
    one round of noise. Its replicate variance components centre on the fitted
    ones, so intervals on a tau or a rho come from here. Its m interval is a
    superpopulation one: it includes the effect of drawing other sites and
    designs. The pseudo-data are built by ``dsim.hybrid_pseudo_data``.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
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
import phd_project.scripts.fit_cache as fcache
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
# The VALUES are the arm tokens of the saved files (mra.fx_arm_names), as templates:
# "{fx}" is the fixed record set's id (record_sets.fx in config.yaml), so a second fixed
# set has its own files. Fill them with :func:`arm_file_name` / :func:`arm_file_names`.
# The dataset columns are named by the KEYS, so the dataset schema is the same for every
# FX set.
#
# The three ida_{fx}* arms are ONE analysis read off the collapse bracket
# three ways -- the lower bound (the published fragility), the upper bound, and
# the geometric mean of the bracket (nb 053 section 10, nb 070 section 1.5).
# They are therefore three alternative `c` terms, not three independent results,
# and the honest way to report them is as a sensitivity band on A0/A2 rather
# than as three separate findings.

ARMS: dict[str, str] = {
    "msa_fx":  "msa_{fx}",          # MSA, fixed record set           (by design group)
    "ida_fx":  "ida_{fx}",          # IDA, bracket lower bound        (by design group)
    "ida_ub":  "ida_{fx}_ub",       # IDA, bracket upper bound        (by design group)
    "ida_avg": "ida_{fx}_avg",      # IDA, bracket geometric mean     (by design group)
    "msa_ss":  "site_msa",          # MSA, site-specific GCIM records (by site)
}


def arm_file_name(arm: str, fx: str) -> str:
    """The file token of one arm for the fixed set ``fx``: ``("ida_ub", "femap695") ->
    "ida_femap695_ub"``. Agrees with ``mra.fx_arm_names`` (asserted at import)."""
    if not fx.isalnum():
        raise ValueError(f"FX id {fx!r} must be letters and digits only")
    return ARMS[arm].format(fx=fx)


def arm_file_names(fx: str) -> dict[str, str]:
    """``{arm key: file token}`` for every arm, in ``ARMS`` (= dataset column) order."""
    return {arm: arm_file_name(arm, fx) for arm in ARMS}


def fx_of(root) -> str:
    """The FX id a bootstrapping ``root`` belongs to: ``root["fx_id"]`` of the
    ``config.bootstrap_roots`` mapping. The FX arms' files are named by it, so a loader
    given a bare folder cannot know which files to read and refuses."""
    if isinstance(root, Mapping) and "fx_id" in root:
        return root["fx_id"]
    raise TypeError("pass root=config.bootstrap_roots(cfg, study): the FX arms' file "
                    "names depend on the fixed record set, which a bare folder does not say")


# the two places that spell the tokens must agree (msa_{fx}, ida_{fx}, ida_{fx}_ub/_avg)
assert {arm_file_name(a, "x") for a in ARMS} - {mra.SS_ARM} == set(mra.fx_arm_names("x").values())

# The mathematical notation of the roadmap. a_i, b_j, c_j are LOG medians (or
# log dispersions, when quantity="beta"). The
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
# rather than from site hazard, so they depend on the design only. When a
# 120-row contrast needs them, Z expands them; that is what `FragilityData.expand` is for.
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


def dataset_csv_path(root: Path | str | Mapping, im_tag: str) -> Path:
    """Canonical filename for the nb-072 regression dataset.

    Mirrors ``mra.bootstrap_csv_path`` / ``mra.estimates_csv_path``: the write
    (nb 072) and every read go through here, so the name cannot drift.

    ``root`` is either

    * the folder mapping from ``config.bootstrap_roots`` (``{"ss", "fx", "study"}``):
      the dataset pairs one SS set with one FX set, so it lives in that STUDY's folder
      (10_.../studies/{study}/), not with either set's bootstrap files; or
    * a single bootstrapping folder (the unit tests): the dataset then sits one level
      above it, beside the bootstrap folder.
    """
    name = f"fragility_regression_dataset_{im_tag}.csv"
    if isinstance(root, Mapping):
        return Path(root["study"]) / name
    return Path(root).parent / name


def arm_label(symbol_or_arm: str, fx_label: str = "FX") -> str:
    """Human-readable arm name, reusing the labels ``mra`` already fixed.

    Figures across the chapter must name the arms identically, so the labels
    live in one place (``mra.arm_label``) and are looked up rather than
    retyped in each notebook. ``fx_label`` names the fixed record set
    (config.yaml ``record_sets.fx.{fx}.label``, e.g. "FEMA P695").
    """
    arm = ARM_OF_SYMBOL.get(symbol_or_arm, symbol_or_arm)
    if arm not in ARMS:
        return arm
    return mra.arm_label(arm_file_name(arm, "fx"), fx_label)


def arm_colour(symbol_or_arm: str) -> str:
    """Fixed colour for an arm (by kind, the same for every FX set). See :func:`arm_label`."""
    arm = ARM_OF_SYMBOL.get(symbol_or_arm, symbol_or_arm)
    if arm not in ARMS:
        return "k"
    return mra.arm_colour(arm_file_name(arm, "fx"))


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
    bias : dict of str -> Series or None
        The bootstrap bias of the log estimate exactly as nb 070 MEASURED it
        (``{arm}_log_{q}_bias`` in the nb-072 dataset), at the arm's natural
        scope. This is not necessarily the correction that was APPLIED: theta is
        measured but corrected by zero. It is kept only so
        :meth:`plugin_offset` can recover the reference the bias was measured
        against. See :meth:`replicate_shift`.
    """

    quantity: str
    im_tag: str
    rows: pd.DataFrame
    groups: pd.Index
    reps: dict[str, pd.DataFrame | None]
    estimates: dict[str, pd.Series | None]
    estimates_bc: dict[str, pd.Series | None] = field(default_factory=dict)
    bias: dict[str, pd.Series | None] = field(default_factory=dict)

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
        section 2.4 writes and nb 072 carries as ``{arm}_log_{q}_bc``. A model
        uses it only when its :class:`ModelSpec` says ``corrected=True``. Those
        are the ``*_bc`` ids, so a model id always names exactly one set of
        estimates.

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

    def plugin_offset(self, symbol: str) -> float:
        """``s``: where this arm's replicate cloud sits relative to ``hat + bias``.

        nb 070 measures each arm's bias as ``b = E*[x*] - x_ref``, where
        ``x_ref`` is the PLUG-IN value, i.e. the estimator's formula applied to
        the resampled data treated as the whole population (Efron & Tibshirani
        Ch. 4, ~pp. 35-37, approximate). For the binomial-MLE arms (both MSA
        arms) and for every theta, the plug-in IS the published estimate, so
        ``s = 0``. For the IDA moment-fit arms it is not: ``lognorm_moment_fit``
        divides by n - 1 (Bessel), the bootstrap world's truth divides by n,
        and so ``x_ref = ln(beta_hat * sqrt((n-1)/n))`` and
        ``s = ln sqrt(21/22) = -0.0233`` at 22 records (nb 070 section 2.1,
        "Plug-in reference for the moment-fit arms").

        In other words the cloud sits at ``E*[x*] = hat + s + b``. The offset is
        READ FROM THE DATA as ``mean(cloud) - hat - bias`` rather than written
        down from a record count, so this method makes no assumption about
        which arms are moment fits or how many records they used. A plug-in
        reference is a fixed rescaling of the published estimate, so ``s`` must
        be the SAME number for every unit of the arm. That is asserted: a spread
        means the bias column and the cloud do not belong together, for example
        a dataset written by nb 072 before nb 070 was re-run.

        Returns
        -------
        float
            The offset. Snapped to exactly 0.0 when it is zero to floating-point
            noise, so an arm that needs no plug-in correction (theta, MSA) gets
            no shift at all rather than one of ~1e-16.
        """
        measured = self.bias.get(symbol)
        if measured is None:
            raise KeyError(
                f"no measured bootstrap bias for symbol {symbol!r} "
                f"(arm {ARM_OF_SYMBOL.get(symbol, symbol)!r}); it comes from the "
                f"'{{arm}}_log_{self.quantity}_bias' column nb 072 writes. It is "
                "needed to centre the replicates of a bias-corrected fit - see "
                "FragilityData.replicate_shift.")
        resid = self.rep(symbol).mean(axis=1) - self.hat(symbol) - measured
        spread = float(resid.max() - resid.min())
        if spread > 1e-8:
            raise ValueError(
                f"the plug-in offset of {symbol!r} is not constant across units "
                f"(spread {spread:.2e}). mean(cloud) - hat - bias should be one "
                "number per arm (0 for an MLE arm, ln sqrt((n-1)/n) for a moment "
                "fit). The bias column and the replicate cloud disagree - re-run "
                "nb 070 then nb 072.")
        s = float(resid.mean())
        return 0.0 if abs(s) < 1e-10 else s

    def replicate_shift(self, symbol: str) -> pd.Series:
        """Per-unit constant that centres the replicates of a bias-corrected fit.

        Only ``y_hat`` changes when estimates are bias-corrected. Left alone,
        the replicate cloud stays where the UNCORRECTED estimator's cloud sits,
        and a percentile interval built from it is not an interval for the
        number being reported. On A2 (beta) its midpoint lands 0.008 off the
        corrected estimate, about half a bootstrap SE.

        The shift that fixes it, per unit (task D3, option (ii), chosen
        2026-10-07)::

            x*_shifted = x* - (2 b + s)

        where ``b = hat - hat_bc`` is the correction APPLIED and ``s`` is
        :meth:`plugin_offset`. The reasoning, in three steps:

        1. The cloud sits at ``E*[x*] = hat + s + b``.
        2. The bias-corrected estimator is ``x_bc = hat - b``. Its bootstrap
           analogue is ``x* - b`` (Efron & Tibshirani Ch. 10, eq. 10.6,
           ~p. 138, approximate), which is centred on the bootstrap world's
           truth ``hat + s``.
        3. The percentile method reads the bootstrap distribution as if the
           bootstrap world's truth were the real-world estimate (Ch. 13,
           ~pp. 168-171, approximate). That is automatic when the estimator is
           the plug-in, but here it has to be arranged: translate by
           ``x_bc - (hat + s)``. Adding the two steps gives ``-b - b - s``.

        The result is the percentile interval OF THE BIAS-CORRECTED ESTIMATOR.
        It keeps what made percentiles the choice in :meth:`ModelFit.ci` in the
        first place: they respect the log transform, and they keep the
        skewness of M*. Being one constant per unit, it changes no spread, so
        ``V``, ``v`` and every bootstrap SE are untouched.

        Dropping ``s`` (shifting by ``-2b``) would be wrong by a whole bootstrap
        SE on the IDA contrasts. Measured on A2, it pushes the interval's
        midpoint 0.026 above the estimate.

        For theta, ``b = 0`` (nb 070 corrects theta by zero) and ``s = 0``, so
        the shift is exactly zero and a "corrected" theta fit is the
        uncorrected one.
        """
        b_applied = self.hat(symbol) - self.hat(symbol, corrected=True)
        return -(2.0 * b_applied + self.plugin_offset(symbol))

    def term(self, symbol: str, corrected: bool = False):
        """``(replicates, point_estimates, scope)`` for one symbol.

        The one accessor the contrast builder needs, which is what lets that
        builder be written once and serve every contrast rather than once per
        pair of arms.

        With ``corrected=True`` BOTH halves change together: the point estimate
        becomes the bias-corrected one, and every replicate is translated by
        :meth:`replicate_shift`, so the cloud is centred where the corrected
        estimator's own bootstrap distribution belongs. This is done here, once,
        and nowhere else. ``to_rows``, ``make_contrast``, ``bootstrap_fits``,
        the per-study forest whiskers and DL all read the replicates through
        this method, so none of them can end up mixing a corrected estimate
        with an uncorrected cloud.
        """
        reps = self.rep(symbol)
        if corrected:
            # Broadcast one constant per unit (row of `reps`) across all k
            # replicates. axis=0 aligns on the unit labels, not on position.
            reps = reps.add(self.replicate_shift(symbol), axis=0)
        return reps, self.hat(symbol, corrected), self.scope_of(symbol)

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
        root, arm_file_name(arm, fx_of(root)), (quantity,), None, im_tag, scope,
        k_samples=k_samples)
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
    root : mapping, or Path / str
        The study's folders, ``config.bootstrap_roots(cfg, study)`` - SS bootstrap,
        FX bootstrap and the study folder holding the dataset (see
        :func:`dataset_csv_path` and ``mra.arm_root``). A single folder holding
        everything also works (the unit tests use one).
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
    # The bias as nb 070 MEASURED it. It is read only to centre the replicates of
    # a bias-corrected fit (FragilityData.plugin_offset). Unlike the estimate
    # columns it is optional: an older dataset without it still loads, and only
    # a `*_bc` model then refuses to fit, with a message that says why.
    bias: dict[str, pd.Series | None] = {}

    for arm, symbol in ((a, SYMBOLS[a]) for a in ARMS):
        cloud = _load_cloud(root, arm, quantity, im_tag, k_samples)
        if cloud is None:
            if arm in CORE_ARMS:
                raise FileNotFoundError(
                    f"no saved {quantity} bootstrap for the core arm {arm!r} "
                    f"({arm_file_name(arm, fx_of(root))}) - run nb 070 first")
            if verbose:
                print(f"  {arm:8s} ({symbol:5s}): no saved bootstrap - skipped")
            reps[symbol] = estimates[symbol] = estimates_bc[symbol] = None
            bias[symbol] = None
            continue

        est_col = f"{arm}_log_{quantity}"
        bc_col = f"{arm}_log_{quantity}_bc"
        for col in (est_col, bc_col):
            if col not in rows.columns:
                raise KeyError(
                    f"{path.name} has no column {col!r} - re-run nb 072, which "
                    "writes the log and bias-corrected columns for every arm")

        bias_col = f"{arm}_log_{quantity}_bias"
        has_bias = bias_col in rows.columns

        if SCOPE_OF_ARM[arm] == "by_site":
            target = rows.index
            estimates[symbol] = rows[est_col]
            estimates_bc[symbol] = rows[bc_col]
            bias[symbol] = rows[bias_col] if has_bias else None
        else:
            target = groups
            estimates[symbol] = _collapse_to_groups(rows, est_col, groups)
            estimates_bc[symbol] = _collapse_to_groups(rows, bc_col, groups)
            # Collapsed with the same within-group constancy check as the
            # estimates: a design-level bias that varies inside its design would
            # be the same row -> design mapping error.
            bias[symbol] = (_collapse_to_groups(rows, bias_col, groups)
                            if has_bias else None)

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
                         reps=reps, estimates=estimates, estimates_bc=estimates_bc,
                         bias=bias)

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
        token = arm_file_name(arm, fx_of(root))
        ests = mra.load_estimates(root, data.im_tag, scope=scope, arms=[token])[token]
        got = np.log(ests[data.quantity])
        if scope == "by_group":
            got = got.droplevel("n_storeys").reindex(data.groups)
        else:
            got = got.reindex(data.rows.index)
        np.testing.assert_allclose(
            data.hat(symbol).to_numpy(), got.to_numpy(), rtol=1e-12,
            err_msg=f"dataset {arm}_log_{data.quantity} disagrees with "
                    f"{mra.estimates_csv_path(root, token, data.im_tag, scope).name}")


# =============================================================================
# Contrasts
# =============================================================================
#
# Everything in the roadmap is a contrast of a, b and c. Making a contrast a first-class object is what stops nb 073
# rebinding `y0` to mean three different things in three different sections.
#
# The scope of a contrast is a property of the contrast, not a suffix on its
# name. y0 and y1 are 120-row quantities; y2 is a 51-row quantity and STAYS one.
# Replicating y2 to 120 rows would invent 69 rows of information that does not
# exist and shrink SE(m2) by roughly sqrt(120/51) ~ 1.53 for nothing.

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
    -set arm, because those arms share sampling error across rows. Having both on one object makes that
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
    corrected : bool
        Whether ``y_hat`` holds the bias-corrected estimates, with ``y`` centred
        to match (:meth:`FragilityData.replicate_shift`). Carried on the
        contrast itself so every printout and plot built from it can say which
        estimates it shows, without needing the spec.
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
    # Last and defaulted, so every existing keyword construction of a Contrast
    # still means what it did.
    corrected: bool = False

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
        arms rather than assuming there is none.
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
                f"mean ratio {np.exp(self.y_hat.mean()):.4f}"
                f"{'  ' + BC_TAG if self.corrected else ''}")


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
        for why this is not the default. This changes ``y_hat`` AND translates
        each row of the replicates ``y`` by a constant, so the cloud is centred
        where the corrected estimator's bootstrap distribution belongs
        (:meth:`FragilityData.replicate_shift`). ``V`` is unchanged, because a
        constant shift per row does not change any covariance.
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
                    terms=(left, right), y_hat=y_hat, y=y, rows=rows, V=V,
                    corrected=corrected)


# =============================================================================
# Model specifications
# =============================================================================
#
# One rung of the roadmap is DECLARED here rather than coded in a notebook.
# That is the whole design goal: the six-cell block nb 073 repeats three times
# (effect sizes -> forest plot -> heterogeneity -> replicate refits -> intervals
# -> SE comparison) becomes one row of this table plus one call to
# ModelRegistry.fit.

# The outer-bootstrap schemes a spec can ask for; see "Two bootstrap schemes" in
# the module docstring. Taken from dsim so the validation study and the specs
# cannot disagree on the spelling.
BOOTSTRAP_SCHEMES: tuple[str, ...] = dsim.BOOTSTRAP_SCHEMES

# The tag every printout and plot of a hybrid-bootstrap fit carries, so its
# intervals are never read as record-only ones (or the reverse).
HB_TAG = "[hybrid bootstrap]"

# Default seed of the hybrid bootstrap's random-effect draws. One constant, so
# two notebooks that fit the same `_hb` id get the same replicates without
# having to agree on a number. Override per call with fit(..., boot_seed=).
BOOT_SEED = 20261013


def _check_bootstrap_scheme(model_id: str, scheme: str) -> None:
    """Raise if ``scheme`` is not one of :data:`BOOTSTRAP_SCHEMES`.

    Shared by ModelSpec and JointModelSpec, so both refuse a misspelt scheme
    when the spec is CONSTRUCTED rather than letting it fall through to the
    record-only path and report the wrong intervals under a hybrid label.
    """
    if scheme not in BOOTSTRAP_SCHEMES:
        raise ValueError(f"{model_id}: unknown bootstrap_scheme {scheme!r}; "
                         f"expected one of {BOOTSTRAP_SCHEMES}")


@dataclass(frozen=True)
class ModelSpec:
    """One rung of the A-series model ladder (A0 ... A16).

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
    corrected : bool
        Fit on the bias-corrected point estimates (nb 070 section 2.4), with the
        replicates centred to match (:meth:`FragilityData.replicate_shift`).
        False by default, which reproduces every earlier result exactly.

        ONE ID, ONE CONTRAST. Whether a fit is corrected is a property of the
        SPEC, never of an argument passed at fit time or a setting on the
        registry. So any id a notebook or the thesis cites, such as "A2",
        always points at the same numbers. The corrected version of a model is
        its own hand-written entry with a ``_bc`` suffix (``"A2_bc"``), placed
        directly below its partner so every analysed model is visible in the
        source. :func:`_check_model_specs` enforces, at import, that the suffix
        and the flag agree, and that a ``_bc`` spec differs from its partner
        only in ``model_id``, ``corrected`` and ``note``.
    bootstrap_scheme : {"record", "hybrid"}
        Which outer bootstrap :meth:`ModelRegistry.fit` runs; see "Two
        bootstrap schemes" in the module docstring and :func:`bootstrap_fits`.
        ``"record"`` by default, which reproduces every earlier result exactly.
        Keyword-only.

        Like ``corrected``, this is a property of the SPEC, for the same
        reason: an id must always mean one set of numbers. The hybrid version
        of a model is its own hand-written entry with an ``_hb`` suffix
        (``"A0c_hb"``; ``"A0c_bc_hb"`` if it is also bias-corrected, in that
        order), and :func:`_check_model_specs` enforces that the suffix and the
        flag agree and that an ``_hb`` spec differs from its partner only in
        ``model_id``, ``bootstrap_scheme`` and ``note``.
    """

    model_id: str
    contrast: str
    estimators: tuple[str, ...] = ("DL", "REML")
    components: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    diagonal_V: bool = False
    note: str = ""
    # LAST, after `note`, so that no existing spec that passes arguments by
    # position changes meaning. Always pass it by keyword.
    corrected: bool = False
    # After `corrected`, and keyword-ONLY (enforced, unlike `corrected`), so it
    # can never be set by a stray positional argument.
    bootstrap_scheme: str = field(default="record", kw_only=True)

    def __post_init__(self):
        _check_bootstrap_scheme(self.model_id, self.bootstrap_scheme)


MODEL_SPECS: dict[str, ModelSpec] = {
    "A0a": ModelSpec(
        "A0a", "y0", ("REML",), note="""
        The basic result: site-specific MSA against fixed-set IDA, intercept only.
        exp(m0) is the correction factor an engineer would apply to a
        FEMA-P695-style IDA fragility to get the site-specific answer.

        Fitted with the FULL V. Every row in a design group subtracts the SAME
        c_j, so the sampling errors are correlated across rows and a diagonal V
        would understate SE(m0). Report the prediction interval alongside the
        confidence interval (Borenstein Ch. 17, not Ch. 14): the CI says where
        the MEAN correction lies, the PI says where the NEXT building lands,
        and it is the second one an engineer needs."""),

    "A0a_hb": ModelSpec(
        "A0a_hb", "y0", ("REML",), bootstrap_scheme="hybrid", 
        note="""
        The basic result: As for A0a - site-specific MSA against fixed-set IDA, 
        intercept only. Uses the hybrid bootstrap scheme so teh bootstrap SEs 
        represent the superpopulation.
        """),

    "A0b": ModelSpec(
        "A0b", "y0_avg", ("REML",), note="""
        A0a against the GEOMETRIC MEAN of the IDA collapse bracket. See A0a."""),

    "A0b_bc": ModelSpec(
        "A0b_bc", "y0_avg", ("REML",),
        corrected=True, note="""
        A0b on bias-corrected estimates. See A0c_bc."""),
    
    "A0b_hb": ModelSpec(
        "A0b_hb", "y0_avg", ("REML",),
        bootstrap_scheme="hybrid", note="""
        A0b with the hybrid outer bootstrap. See A0c_hb"""),

    "A0b_bc_hb": ModelSpec(
        "A0b_bc_hb", "y0_avg", ("REML",),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A0b_bc with the HYBRID outer bootstrap: see A0b_bc and A0b_hb."""),

    "A0c": ModelSpec(
        "A0c", "y0_avg", ("REML",), components=("design", "site"), note="""
        A0a against the GEOMETRIC MEAN of the IDA collapse bracket. 
        In this model the bewteen-study variance is decomposed into random-effects,
        building group, and site effects, based on the results of A1b and A1c.
        The random-effects includes the site * structure interaction + any 
        betwen row randomness. 
        
        Using the full variance matrix (V_aa + ZVbbZ') rather 
        than the diagonal will capture crossing between the rows.
        """),

    "A0c_bc": ModelSpec(
        "A0c_bc", "y0_avg", ("REML",), components=("design", "site"),
        corrected=True, note="""
        A0c on bias-corrected estimates. See A0c.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),
    
    "A0c_hb": ModelSpec(
        "A0c_hb", "y0_avg", ("REML",), components=("design", "site"),
        bootstrap_scheme="hybrid", note="""
        A0c with the HYBRID outer bootstrap: each replicate is X beta_hat plus
        new design, site and row effects drawn from the fitted tau's, plus the
        record-resample deviations y^(r) - ybar. Use this fit, not A0c, for any
        interval on tau_d, tau_s or tau_u (ModelFit.tau_ci), and read the
        boundary fractions beside them: a component fitted at zero is drawn at
        zero. Its CI on the summary effect is a SUPERPOPULATION interval (other
        sites and designs could have been drawn); the interval for these 120
        rows is A0c's."""),

    "A0c_bc_hb": ModelSpec(
        "A0c_bc_hb", "y0_avg", ("REML",), components=("design", "site"),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A0c_bc with the HYBRID outer bootstrap: see A0c_bc and A0c_hb."""),

    "A0d": ModelSpec(
        "A0d", "y0", ("REML",), components=("design", "site"), note="""
        A0a with the bewteen-study variance decomposed into random-effects,
        building group, and site effects, based on the results of A1b and A1c.
        The random-effects includes the site * structure interaction + any 
        betwen row randomness."""),

    "A0d_hb": ModelSpec(
        "A0d_hb", "y0", ("REML",), components=("design", "site"), 
        bootstrap_scheme="hybrid", note="""
        A0a with the bewteen-study variance decomposed into random-effects,
        building group, and site effects, based on the results of A1b and A1c.
        The random-effects includes the site * structure interaction + any 
        betwen row randomness."""),

    "A1a": ModelSpec(
        "A1a", "y1", ("REML",), diagonal_V=True, note="""
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
        "A2", "y2", ("REML",), note="""
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

    "A2_bc": ModelSpec(
        "A2_bc", "y2", ("REML",), corrected=True, note="""
        A2 on bias-corrected estimates. See A2. This is the rung the correction
        matters most for: before correction |bias|/se on y2 is about 0.64-0.87
        (nb 070), well above the 0.25 threshold.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A2_hb": ModelSpec(
        "A2_hb", "y2", ("REML",), bootstrap_scheme="hybrid", 
        note="""
        A2 with the hybrid bootstrap scheme. 
        See A2 and A0a_hb."""),    
    
    "A2_bc_hb": ModelSpec(
        "A2_bc_hb", "y2", ("REML",), corrected=True, bootstrap_scheme="hybrid", 
        note="""
        A2 on bias-corrected estimates with the hybrid bootstrap scheme. 
        See A2_bc and A2_hb."""),

    # ---- the collapse-bracket variants -------------------------------------
    # nb 073 section 2.4 described these and never implemented them. They are
    # table entries, not code. Read them as a SENSITIVITY BAND on A0a/A2: the
    # three IDA arms are one analysis read three ways off the collapse bracket,
    # so the spread across them is the cost of that modelling choice, not three
    # independent findings.
    "A2_ub": ModelSpec(
        "A2_ub", "y2_ub", ("REML",), note="""
        A2 against the upper bound of the IDA collapse bracket. See A2."""),
    
    "A2_ub_bc": ModelSpec(
        "A2_ub_bc", "y2_ub", ("REML",), corrected=True, note="""
        A2_ub on bias-corrected estimates. See A2_ub and A2_bc.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A2_ub_hb": ModelSpec(
        "A2_ub_hb", "y2_ub", ("REML",), bootstrap_scheme="hybrid", note="""
        A2 with the hybrid bootstrap scheme. See A2 and A2_hb."""),

    "A2_ub_bc_hb": ModelSpec(
        "A2_ub_bc_hb", "y2_ub", ("REML",), corrected=True, bootstrap_scheme="hybrid",
        note="""
        A2_ub on bias-corrected estimates with the hybrid bootstrap scheme. 
        See A2_ub and A2_bc_hb."""),
    
    "A2_avg": ModelSpec("A2_avg", "y2_avg", ("REML",), note="""
        A2 against the geometric mean of the IDA collapse bracket. See A2."""),
    
    "A2_avg_bc": ModelSpec("A2_avg_bc", "y2_avg", ("REML",), corrected=True, note="""
        A2_avg on bias-corrected estimates. See A2_avg and A2_bc.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A2_avg_hb": ModelSpec(
        "A2_avg_hb", "y2_avg", ("REML",), bootstrap_scheme="hybrid", note="""
        A2 with the hybrid bootstrap scheme. See A2 and A2_hb."""),

    "A2_avg_bc_hb": ModelSpec(
        "A2_avg_bc_hb", "y2_avg", ("REML",), corrected=True, bootstrap_scheme="hybrid",
        note="""
        A2_avg on bias-corrected estimates with the hybrid bootstrap scheme. 
        See A2_avg and A2_bc_hb."""),

    # ---- Regressions -------------------------------------------------------
    "A3a": ModelSpec(
        "A3a", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_rtp5000", 
            "d_ln_ds595_mean_rtp5000", 
            "d_ln_saratio_sigma_rtp5000",
            "d_ln_ds595_sigma_rtp5000"), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3a_bc": ModelSpec(
        "A3a_bc", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_rtp5000",
            "d_ln_ds595_mean_rtp5000",
            "d_ln_saratio_sigma_rtp5000",
            "d_ln_ds595_sigma_rtp5000"),
        corrected=True, note="""
        A3a on bias-corrected estimates. See A3a. The slopes move too, because
        the bias differs from row to row; their bootstrap CIs are centred by the
        same per-row shift.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A3a_hb": ModelSpec(
        "A3a_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_rtp5000",
            "d_ln_ds595_mean_rtp5000",
            "d_ln_saratio_sigma_rtp5000",
            "d_ln_ds595_sigma_rtp5000"),
        bootstrap_scheme="hybrid", note="""
        A3a with hybrid bootstrap scheme."""),

    "A3a_bc_hb": ModelSpec(
        "A3a_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_rtp5000",
            "d_ln_ds595_mean_rtp5000",
            "d_ln_saratio_sigma_rtp5000",
            "d_ln_ds595_sigma_rtp5000"),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A3a with bias correction and the hybrid bootstrap scheme."""),

    "A3aa": ModelSpec(
        "A3aa", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_rtp5000", 
            "d_ln_ds595_mean_rtp5000", 
            "d_ln_saratio_sigma_rtp5000",
            "d_ln_ds595_sigma_rtp5000",
            "ks_unif"), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation.

    The variance is divided out into the "design" and "site" components
    """
    ),
    
    "A3ab": ModelSpec(
        "A3ab", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
        ), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3ab_bc": ModelSpec(
        "A3ab_bc", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
        ), corrected=True, note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation. Bias corrected

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3ab_hb": ModelSpec(
        "A3ab_hb", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
        ), bootstrap_scheme="hybrid", note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered. Hybrid Bootstrap.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3ab_bc_hb": ModelSpec(
        "A3ab_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
        ), corrected=True, bootstrap_scheme="hybrid", note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered. Bias corrected, Hybrid Bootstrap.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3ac": ModelSpec(
        "A3ac", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "drec_ln_saratio_mean_unif",
            "drec_ln_ds595_mean_unif",
            "drec_ln_saratio_sigma_unif",
            "drec_ln_ds595_sigma_unif",
        ), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3ac_bc": ModelSpec(
        "A3ac_bc", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif",
            "d_ln_ds595_mean_unif",
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "drec_ln_saratio_mean_unif",
            "drec_ln_ds595_mean_unif",
            "drec_ln_saratio_sigma_unif",
            "drec_ln_ds595_sigma_unif",
        ),
        corrected=True, note="""
        A3ac on bias-corrected estimates. See A3ac and A3a_bc.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A3ac_hb": ModelSpec(
        "A3ac_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif",
            "d_ln_ds595_mean_unif",
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "drec_ln_saratio_mean_unif",
            "drec_ln_ds595_mean_unif",
            "drec_ln_saratio_sigma_unif",
            "drec_ln_ds595_sigma_unif",
        ),
        bootstrap_scheme="hybrid", note="""
        A3ac with the hybrid bootstrap scheme"""),

    "A3ac_bc_hb": ModelSpec(
        "A3ac_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif",
            "d_ln_ds595_mean_unif",
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "drec_ln_saratio_mean_unif",
            "drec_ln_ds595_mean_unif",
            "drec_ln_saratio_sigma_unif",
            "drec_ln_ds595_sigma_unif",
        ),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A3ac with the hybrid bootstrap scheme"""),

    "A3ad": ModelSpec(
            "A3ad", "y0_avg", estimators=("REML",), components=("design", "site"), 
            covariates=(
                "d_ln_saratio_mean_fisher", 
                "d_ln_ds595_mean_fisher", 
                "d_ln_saratio_sigma_fisher",
                "d_ln_ds595_sigma_fisher",
                "drec_ln_saratio_mean_fisher",
                "drec_ln_ds595_mean_fisher",
                "drec_ln_saratio_sigma_fisher",
                "drec_ln_ds595_sigma_fisher",
            ), note="""
        Regression analysis of the difference between site-specific MSA and IDA FX
        using the average collapse IML. In this regression only site-specific
        covariates are considered: SARatio + its standard deviation and DS_595 + its
        standard deviation.
    
        The variance is divided out into the "design" and "site" components
        """
        ),

    "A3ad_bc": ModelSpec(
        "A3ad_bc", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_fisher",
            "d_ln_ds595_mean_fisher",
            "d_ln_saratio_sigma_fisher",
            "d_ln_ds595_sigma_fisher",
            "drec_ln_saratio_mean_fisher",
            "drec_ln_ds595_mean_fisher",
            "drec_ln_saratio_sigma_fisher",
            "drec_ln_ds595_sigma_fisher",
        ),
        corrected=True, note="""
        A3ad on bias-corrected estimates. See A3ad and A3a_bc.
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A3ad_hb": ModelSpec(
        "A3ad_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_fisher",
            "d_ln_ds595_mean_fisher",
            "d_ln_saratio_sigma_fisher",
            "d_ln_ds595_sigma_fisher",
            "drec_ln_saratio_mean_fisher",
            "drec_ln_ds595_mean_fisher",
            "drec_ln_saratio_sigma_fisher",
            "drec_ln_ds595_sigma_fisher",
        ),
        bootstrap_scheme="hybrid", note="""
        A3ad with the hybrid bootstrap scheme."""),

    "A3ad_bc_hb": ModelSpec(
        "A3ad_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_fisher",
            "d_ln_ds595_mean_fisher",
            "d_ln_saratio_sigma_fisher",
            "d_ln_ds595_sigma_fisher",
            "drec_ln_saratio_mean_fisher",
            "drec_ln_ds595_mean_fisher",
            "drec_ln_saratio_sigma_fisher",
            "drec_ln_ds595_sigma_fisher",
        ),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A3ad on bias-corrected estimates with the hybrid bootstrap scheme."""),

    "A3ae": ModelSpec(
        "A3ae", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif_corrected", 
            "d_ln_ds595_mean_unif_corrected", 
            "d_ln_saratio_sigma_unif_corrected",
            "d_ln_ds595_sigma_unif_corrected",
        ), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3ae_bc": ModelSpec(
        "A3ae_bc", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif_corrected",
            "d_ln_ds595_mean_unif_corrected",
            "d_ln_saratio_sigma_unif_corrected",
            "d_ln_ds595_sigma_unif_corrected",
        ),
        corrected=True, note="""
        A3ae on bias-corrected estimates. See A3ae and A3a_bc. (The
        "_corrected" in the covariate names is unrelated: it refers to the
        covariates, not to the bias correction of the estimates.)
        Fitted on the BIAS-CORRECTED point estimates (nb 070 section 2.4): each
        arm's ln(beta) has its own bootstrap bias removed, and the replicates
        are shifted by -(2b + s) so the CI is the percentile interval of the
        corrected estimator. Corrected because the MSA binomial MLE and the IDA
        moment fit are biased differently, so the bias does not cancel in the
        contrast (Efron & Tibshirani Ch. 10, ~p. 128: |bias|/se above 0.25).
        Meant for beta. theta is never corrected, so on a theta container this
        is the uncorrected fit, and fitting it warns."""),

    "A3ae_hb": ModelSpec(
        "A3ae_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif_corrected",
            "d_ln_ds595_mean_unif_corrected",
            "d_ln_saratio_sigma_unif_corrected",
            "d_ln_ds595_sigma_unif_corrected",
        ),
        bootstrap_scheme="hybrid", note="""
        A3ae with the hybrid bootstrap scheme"""),

    "A3ae_bc_hb": ModelSpec(
        "A3ae_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif_corrected",
            "d_ln_ds595_mean_unif_corrected",
            "d_ln_saratio_sigma_unif_corrected",
            "d_ln_ds595_sigma_unif_corrected",
        ),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A3ae on bias-corrected estimates with the hybrid bootstrap scheme."""),

    "A3af": ModelSpec(
        "A3af", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif_recs", 
            "d_ln_ds595_mean_unif_recs", 
            "d_ln_saratio_sigma_unif_recs",
            "d_ln_ds595_sigma_unif_recs",
        ), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only site-specific
    covariates are considered: SARatio + its standard deviation and DS_595 + its
    standard deviation. The values are based on the as-used record sets and not 
    the target GCIM distributions

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3af_bc": ModelSpec(
        "A3af_bc", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif_recs",
            "d_ln_ds595_mean_unif_recs",
            "d_ln_saratio_sigma_unif_recs",
            "d_ln_ds595_sigma_unif_recs",
        ),
        corrected=True, note="""
        A3ae on bias-corrected estimates. See A3af and A3a_bc."""),

    "A3af_hb": ModelSpec(
        "A3af_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif_recs",
            "d_ln_ds595_mean_unif_recs",
            "d_ln_saratio_sigma_unif_recs",
            "d_ln_ds595_sigma_unif_recs",
        ),
        bootstrap_scheme="hybrid", note="""
        A3af with the hybrid bootstrap scheme"""),

    "A3af_bc_hb": ModelSpec(
        "A3af_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"),
        covariates=(
            "d_ln_saratio_mean_unif_recs",
            "d_ln_ds595_mean_unif_recs",
            "d_ln_saratio_sigma_unif_recs",
            "d_ln_ds595_sigma_unif_recs",
        ),
        corrected=True, bootstrap_scheme="hybrid", note="""
        A3af with bias correction and the hybrid bootstrap scheme"""),

    "A3b": ModelSpec(
        "A3b", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "ln_T1", "ln_overstrength", "ln_ductility",), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only structure-specific
    covariates are considered.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3b_bc": ModelSpec(
        "A3b_bc", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=("ln_T1", "ln_overstrength", "ln_ductility",), corrected=True,
        note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only structure-specific
    covariates are considered. Bias corrected.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3b_hb": ModelSpec(
        "A3b_hb", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "ln_T1", "ln_overstrength", "ln_ductility",), 
        bootstrap_scheme="hybrid", note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only structure-specific
    covariates are considered. The hybrid bootstrap scheme is used.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3b_bc_hb": ModelSpec(
        "A3b_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "ln_T1", "ln_overstrength", "ln_ductility",), 
        bootstrap_scheme="hybrid", corrected=True, note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression only structure-specific
    covariates are considered. The hybrid bootstrap scheme is used. Bias corrected

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3c": ModelSpec(
        "A3c", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "ln_T1", "ln_overstrength", "ln_ductility",), note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression all of the A3a and A3b 
    covariates are considered

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3c_bc": ModelSpec(
        "A3c_bc", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "ln_T1", "ln_overstrength", "ln_ductility",), 
        corrected=True, note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression all of the A3a and A3b 
    covariates are considered. Bias corrected.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3c_hb": ModelSpec(
        "A3c_hb", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "ln_T1", "ln_overstrength", "ln_ductility",), 
        bootstrap_scheme="hybrid", note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression all of the A3a and A3b 
    covariates are considered. Hybrid bootstrap.

    The variance is divided out into the "design" and "site" components
    """
    ),

    "A3c_bc_hb": ModelSpec(
        "A3c_bc_hb", "y0_avg", estimators=("REML",), components=("design", "site"), 
        covariates=(
            "d_ln_saratio_mean_unif", 
            "d_ln_ds595_mean_unif", 
            "d_ln_saratio_sigma_unif",
            "d_ln_ds595_sigma_unif",
            "ln_T1", "ln_overstrength", "ln_ductility",), 
        corrected=True, bootstrap_scheme="hybrid", note="""
    Regression analysis of the difference between site-specific MSA and IDA FX
    using the average collapse IML. In this regression all of the A3a and A3b 
    covariates are considered. Bias corrected. Hybrid bootstrap.

    The variance is divided out into the "design" and "site" components
    """
    ),
}


# The tag every printout and plot of a corrected fit carries. One string, so
# the summary header, the forest title and the contrast summary cannot drift.
BC_TAG = "[bias-corrected estimates]"

# The only fields in which a `_bc` spec may differ from its uncorrected partner.
_BC_FREE_FIELDS = frozenset({"model_id", "corrected", "note"})

# Likewise for an `_hb` spec and its record-only partner.
_HB_FREE_FIELDS = frozenset({"model_id", "bootstrap_scheme", "note"})


def _check_model_specs(specs: dict[str, ModelSpec]) -> None:
    """Enforce "one id, one contrast" on a spec table. Raises ValueError.

    The `_bc` specs are written out by hand rather than generated, which is
    better for seeing what has been analysed, but it means nothing stops
    ``"A2_bc"`` drifting from ``"A2"`` (a covariate added to one and not the
    other, say). Then "A2 vs A2_bc" would no longer isolate the bias correction.
    This check is what prevents that. It runs at import (below), so a broken
    table fails the moment the module loads, not three cells into a notebook.

    Rules:

    1. The dict key equals ``spec.model_id``. The key is what ``fit`` and the
       registry use; the id is what every printout uses. If they differ, the
       numbers get printed under another model's name, which is what A3b and
       A3c did until 2026-10-08.
    2. ``spec.corrected`` is True exactly when the id ends in ``_bc``, so the id
       alone always says which estimates were used.
    3. Every ``_bc`` spec has an uncorrected partner (the id without the suffix)
       and equals it in every field except ``model_id``, ``corrected`` and
       ``note``. That includes ``bootstrap_scheme``: the partner of
       ``"A0c_bc_hb"`` is ``"A0c_hb"``, so the pair always agree on the scheme.
    4. ``spec.bootstrap_scheme`` is ``"hybrid"`` exactly when the id ends in
       ``_hb``. An id is ``base[_bc][_hb]``, in that order, so ``_hb`` is always
       the LAST suffix.
    5. Every ``_hb`` spec has a record-only partner (the id without ``_hb``)
       and equals it in every field except ``model_id``, ``bootstrap_scheme``
       and ``note``.
    """
    problems: list[str] = []

    def drift_from(spec, partner, free: frozenset) -> list[str]:
        # fields(type(spec)), not fields(ModelSpec), so the same rules serve the
        # joint table (JointModelSpec, A8). For a ModelSpec the two are identical.
        return [f.name for f in fields(type(spec))
                if f.name not in free
                and getattr(spec, f.name) != getattr(partner, f.name)]

    for key, spec in specs.items():
        if key != spec.model_id:
            problems.append(f"key {key!r} holds a spec whose model_id is {spec.model_id!r}")

        # Read the suffixes off the id: `_hb` is stripped first because it comes
        # last, then `_bc`. So "A0c_bc_hb" is both, and a wrong-order id such as
        # "A0c_hb_bc" reads as "_bc only" and fails rule 4 below.
        is_hb = key.endswith("_hb")
        rest = key.removesuffix("_hb")
        is_bc = rest.endswith("_bc")
        base = rest.removesuffix("_bc")
        hybrid = spec.bootstrap_scheme == "hybrid"

        if spec.corrected != is_bc:
            problems.append(
                f"{key!r}: corrected={spec.corrected}, but only ids ending in '_bc' "
                "may be (and must be) corrected")
        if hybrid != is_hb:
            problems.append(
                f"{key!r}: bootstrap_scheme={spec.bootstrap_scheme!r}, but only ids "
                "ending in '_hb' may be (and must be) 'hybrid'; the suffix order is "
                "base[_bc][_hb]")

        if is_bc and spec.corrected:
            # The uncorrected partner keeps the `_hb` suffix, if any.
            partner_id = base + ("_hb" if is_hb else "")
            partner = specs.get(partner_id)
            if partner is None:
                problems.append(f"{key!r} has no uncorrected partner {partner_id!r}")
            else:
                drift = drift_from(spec, partner, _BC_FREE_FIELDS)
                if drift:
                    problems.append(
                        f"{key!r} differs from {partner_id!r} in {drift}; a _bc spec "
                        f"may differ only in {sorted(_BC_FREE_FIELDS)}")

        if is_hb and hybrid:
            partner_id = rest
            partner = specs.get(partner_id)
            if partner is None:
                problems.append(f"{key!r} has no record-only partner {partner_id!r}")
            else:
                drift = drift_from(spec, partner, _HB_FREE_FIELDS)
                if drift:
                    problems.append(
                        f"{key!r} differs from {partner_id!r} in {drift}; an _hb spec "
                        f"may differ only in {sorted(_HB_FREE_FIELDS)}")
    if problems:
        raise ValueError("MODEL_SPECS is inconsistent:\n  " + "\n  ".join(problems))


_check_model_specs(MODEL_SPECS)


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
        that touches a fixed-record-set arm. (That is so for DL and for a
        diagonal-V fit. A REML fit given the full V has a sound model-based
        variance.) Which bootstrap matters too: the formula wants the
        SUPERPOPULATION variance, which the hybrid replicates give and the
        record-only ones slightly understate -- see
        :meth:`ModelFit.prediction_interval`.
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
        One row per outer-bootstrap replicate; columns ``M_star``, one
        ``beta_<name>`` per coefficient (:func:`beta_column`), ``T_sq`` and,
        for REML, one per variance component.
    x_names : list of str
    scaling : dict
        Covariate centring and scaling, for back-transforming coefficients.
    raw : dict
        The untouched return value of the underlying estimator.
    boot : dict
        How the outer bootstrap was run: ``{"scheme", "seed", "zero_levels"}``.
        Written by the registry from what was ACTUALLY run. Empty for a fit
        built by hand, which :attr:`boot_scheme` reads as record-only (the only
        scheme that existed before this field did). ``zero_levels`` names the
        components that were at zero in the full-sample fit, where the hybrid
        scheme therefore drew no effects.
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
    boot: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def quantity(self) -> str:
        return self.contrast.quantity

    @property
    def boot_scheme(self) -> str | None:
        """``"record"`` or ``"hybrid"``; None if no outer bootstrap was run.

        Read from :attr:`boot`, i.e. from what was run, not from the spec --
        the same principle as :attr:`corrected`. Every reader of the replicates
        that means something different under the two schemes (the CI, the SEs,
        the tau intervals) branches on this.
        """
        if self.replicates is None:
            return None
        return self.boot.get("scheme", "record")

    @property
    def hybrid(self) -> bool:
        """True when the replicates come from the hybrid bootstrap."""
        return self.boot_scheme == "hybrid"

    @property
    def corrected(self) -> bool:
        """Whether this fit used the bias-corrected estimates.

        Read from the CONTRAST, i.e. from what was actually fitted, rather than
        from the spec. The registry guarantees the two agree. Reading the
        contrast means a fit built by hand outside the registry still reports
        honestly.
        """
        return self.contrast.corrected

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

        ALL OF THE ABOVE IS THE RECORD-ONLY SCHEME. On a hybrid fit
        (:attr:`hybrid`) each replicate also redraws the site, design and row
        effects, so this number is a SUPERPOPULATION standard error and its
        like-for-like partner is :attr:`se_beta`, not :attr:`se_sampling`.
        Expect it slightly above ``se_beta``: the replicates re-estimate the
        tau's, and so the weights, which the closed form treats as known.
        """
        V = self.V_boot
        return None if V is None else float(np.sqrt(V))

    @cached_property
    def weight_matrix(self) -> np.ndarray | None:
        """(q, n) the weights for EVERY coefficient: ``beta_hat = W y``.

        Row k is the weight vector that turns the data into coefficient k, so
        ``W[0]`` is :attr:`weights` (the intercept) and ``W[1:]`` are the
        slopes. For GLS this is ``(X' Sigma^-1 X)^-1 X' Sigma^-1`` (BDA3
        section 14.7, ~pp. 369-371, approximate; Chatterjee & Hadi Ch. 7,
        ~pp. 179-186, approximate).

        A slope's weights sum to ZERO, not one: a slope is a contrast between
        rows with high and low covariate values, so it adds some rows and
        subtracts others. The intercept's weights sum to one because it is an
        (adjusted) average.

        For DL there is only the intercept, so this is a 1 x n matrix built from
        the diagonal DL weights. None if a REML fit did not keep its Sigma.
        """
        con = self.contrast
        X, _, _ = build_X(self.spec, con)
        if self.estimator == "DL":
            w = 1.0 / (con.v.to_numpy() + self.T_sq)
            return (w / w.sum())[None, :]
        Sigma = self.raw.get("Sigma")
        if Sigma is None:
            return None
        # Solve rather than invert Sigma explicitly: numerically kinder, same
        # answer. Si_X = Sigma^-1 X, and (X' Sigma^-1 X)^-1 X' Sigma^-1 is then
        # (X' Si_X)^-1 Si_X' because Sigma is symmetric.
        Si_X = np.linalg.solve(Sigma, X)
        return np.linalg.solve(X.T @ Si_X, Si_X.T)

    @cached_property
    def se_sampling_beta(self) -> np.ndarray | None:
        """(q,) sampling-error-only SE of every coefficient, ``sqrt(diag(W V W'))``.

        The per-coefficient version of :attr:`se_sampling`: the contrast's full
        sampling covariance V pushed through each coefficient's own weight row.
        Each entry is the like-for-like partner of the matching entry of
        :attr:`se_boot_beta` -- they should agree to Monte-Carlo error.
        """
        W = self.weight_matrix
        if W is None:
            return None
        # Only the diagonal of W V W' is needed, so compute it row by row
        # rather than forming the full q x q product: diag(A B) = sum(A * B', 1).
        return np.sqrt(np.einsum("kn,nm,km->k", W, self.contrast.V, W))

    @cached_property
    def se_boot_beta(self) -> np.ndarray | None:
        """(q,) bootstrap SE of every coefficient: SD of its refits across replicates.

        Same finite-population meaning as :attr:`se_boot` (records resampled,
        sites and designs held fixed), now for the slopes too -- and, like it,
        superpopulation on a hybrid fit. Entries are NaN
        for a coefficient the replicate frame does not hold -- for example
        replicates produced before :func:`bootstrap_fits` kept the slopes.
        """
        if self.replicates is None:
            return None
        out = np.full(len(self.x_names), np.nan)
        for k, name in enumerate(self.x_names):
            col = beta_column(name)
            if col in self.replicates:
                out[k] = self.replicates[col].std(ddof=1)
            elif k == 0:
                # Older frames only carry the intercept, as M_star.
                out[k] = self.replicates["M_star"].std(ddof=1)
        return out

    def coef_table(self, alpha: float = 0.05, natural_units: bool = False) -> pd.DataFrame:
        """One row per coefficient: estimate, its three SEs, z and a bootstrap CI.

        Columns
        -------
        beta        the estimate
        se_model    sqrt(diag((X' Sigma^-1 X)^-1)) -- includes tau, so a
                    SUPERPOPULATION SE (see :attr:`se_sampling`)
        se_sampling sqrt(diag(W V W')) -- sampling error only, these structures
        se_boot     SD across the outer-bootstrap refits
        z           beta / se_model, the Wald statistic (normal reference;
                    it treats the tau^2 as known -- see Knapp-Hartung /
                    parametric bootstrap if n is small)
        p           two-sided p-value for z
        ci_lo/ci_hi percentile bootstrap interval (NaN without replicates).
                    Like se_boot this is finite-population: it does NOT include
                    between-study heterogeneity. (Record-only scheme. On a
                    hybrid fit se_boot and the CI DO include it, and compare
                    with se_model rather than se_sampling.)

        Parameters
        ----------
        natural_units : bool
            Covariates are centred and scaled in :func:`build_X`, so a slope is
            "per SD of the covariate". With True, slopes and their SEs/CIs are
            divided by that SD to read "per unit of the covariate". The
            intercept is unchanged either way: centring makes it the value at
            the covariate MEAN, not at zero. z and p are scale-free.
        """
        q = len(self.beta)
        nan = np.full(q, np.nan)
        se_s = self.se_sampling_beta if self.se_sampling_beta is not None else nan
        se_b = self.se_boot_beta if self.se_boot_beta is not None else nan
        tbl = pd.DataFrame({"beta": self.beta, "se_model": self.se_beta,
                            "se_sampling": se_s, "se_boot": se_b},
                           index=pd.Index(self.x_names, name="coefficient"))

        tbl["z"] = tbl["beta"] / tbl["se_model"]
        tbl["p"] = 2 * stats.norm.sf(np.abs(tbl["z"]))

        tbl["ci_lo"], tbl["ci_hi"] = np.nan, np.nan
        if self.replicates is not None:
            for name in self.x_names:
                col = beta_column(name)
                if col not in self.replicates and name == "intercept":
                    col = "M_star"
                if col in self.replicates:
                    lo, hi = self.replicates[col].quantile([alpha / 2, 1 - alpha / 2])
                    tbl.loc[name, ["ci_lo", "ci_hi"]] = [lo, hi]

        if natural_units:
            # Dividing by the SD rescales every column measured in the units of
            # beta; z and p are ratios and are left alone.
            for name, sc in self.scaling.items():
                cols = ["beta", "se_model", "se_sampling", "se_boot", "ci_lo", "ci_hi"]
                tbl.loc[name, cols] = tbl.loc[name, cols] / sc["sd"]
        return tbl

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

        This is row 0 of :attr:`weight_matrix`; it is taken from there so the
        intercept and slope weights are guaranteed to come from one calculation.
        """
        W = self.weight_matrix
        return None if W is None else W[0]

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

        WHICH QUESTION IT ANSWERS depends on :attr:`boot_scheme`. Record-only:
        where the mean lies for THESE sites and designs (records resampled,
        everything else held fixed). Hybrid: where the mean lies for the
        POPULATION they were drawn from -- the replicates also redraw the
        effects, so the interval is wider. The record-only interval of an
        ``_hb`` fit is its partner's (``"A0c"`` for ``"A0c_hb"``).
        """
        if self.replicates is None:
            raise ValueError(
                f"{self.model_id} was fitted without an outer bootstrap - pass "
                "n_boot to get a confidence interval")
        lo, hi = self.replicates["M_star"].quantile([alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    def boundary_fractions(self) -> pd.Series:
        """Per variance component, the fraction of replicates in which it is exactly 0.

        The number to read BEFORE reading an interval on a tau. A component at
        zero in most replicates is not being estimated, it is being detected as
        absent, and a percentile interval there is one-sided: [0, upper].
        """
        if self.replicates is None:
            raise ValueError(f"{self.model_id}: no outer bootstrap - pass n_boot")
        return pd.Series({name: float((self.replicates[name] == 0.0).mean())
                          for name in self.tau2 if name in self.replicates},
                         name="frac_at_zero", dtype=float).rename_axis("component")

    def tau_ci(self, name: str = "total", alpha: float = 0.05) -> dict:
        """Percentile bootstrap interval for one variance component, as a tau^2.

        Parameters
        ----------
        name : str
            A key of :attr:`tau2` (``"tau_d2"``, ``"tau_s2"``, ``"tau_u2"``),
            or ``"total"`` for their sum (the ``T_sq`` column).

        Returns
        -------
        dict
            ``{"estimate", "lo", "hi", "replicate_mean", "frac_at_zero",
            "scheme"}``, all on the VARIANCE scale. Take square roots for a
            tau: percentiles respect monotone transforms, so
            ``sqrt(lo), sqrt(hi)`` is the percentile interval for tau itself
            (Efron & Tibshirani Ch. 13, ~pp. 170-174, approximate).

        ONLY MEANINGFUL ON A HYBRID FIT. Record-only replicates carry two
        rounds of sampling noise, so their variance components are inflated by
        about the mean sampling variance and this "interval" does not even
        contain the estimate in the worst cases; calling it there warns.

        Reading it on a hybrid fit:

        * ``replicate_mean`` should sit on ``estimate``. If it does not, either
          a component is on its boundary (below) or the fit uses a diagonal V
          (see :func:`bootstrap_fits`, point 2).
        * A component fitted at zero is drawn at zero, so its interval is
          ``[0, upper]``: "consistent with zero, and values above ``upper`` are
          unlikely if it is zero". It is not a 95% interval in the usual sense
          -- a bootstrap does not behave regularly for a parameter on its
          boundary -- so quote ``frac_at_zero`` with it.
        * A percentile interval for a variance is only first-order accurate;
          with about 120 rows it covers at roughly 0.945 rather than 0.95.
        """
        if self.replicates is None:
            raise ValueError(f"{self.model_id}: no outer bootstrap - pass n_boot")
        col = "T_sq" if name == "total" else name
        if name != "total" and name not in self.tau2:
            raise KeyError(f"no variance component {name!r} in {self.model_id}; it has "
                           f"{sorted(self.tau2)} (or 'total')")
        if col not in self.replicates:
            raise KeyError(f"the replicates of {self.model_id} [{self.estimator}] hold "
                           f"no {col!r} column")
        if not self.hybrid:
            warnings.warn(
                f"{self.model_id}: these are RECORD-ONLY replicates, whose variance "
                "components are inflated by about the mean sampling variance. This is "
                "not an interval for tau - fit the '_hb' version of the model.",
                RuntimeWarning, stacklevel=2)
        vals = self.replicates[col].to_numpy(dtype=float)
        lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
        est = self.T_sq if name == "total" else self.tau2[name]
        return {"estimate": float(est), "lo": float(lo), "hi": float(hi),
                "replicate_mean": float(vals.mean()),
                "frac_at_zero": float(np.mean(vals == 0.0)),
                "scheme": self.boot_scheme}

    def prediction_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Where the NEXT building is expected to land. See :func:`prediction_interval`.

        ``m +/- t * sqrt(T^2 + Var(m))``. The new building has its own site,
        design and row effect, so the Var(m) that belongs here is the
        SUPERPOPULATION one, which counts the fact that these rows are
        themselves a sample (Borenstein Ch. 17, ~pp. 127-133, approximate).

        * Hybrid fit: :attr:`V_boot` is exactly that, with the full sampling
          covariance and the tau-estimation noise included.
        * Record-only fit: :attr:`V_boot` is ``w' V w``, which leaves out the
          ``w' (sum tau^2 G) w`` part, so the interval is slightly too NARROW.
          The effect is small because T^2 dominates (0.8% of the half-width
          for A0b theta). Left as it is by decision (2026-10-13), so that no
          record-only number moves; the ``_hb`` fit has the right one.
        """
        V_M = self.V_boot
        if V_M is None:
            V_M = float(self.se_beta[0] ** 2)
            warnings.warn(
                f"{self.model_id}: no outer bootstrap, so the prediction interval "
                "uses the model-based Var(m), which assumes a diagonal sampling "
                "covariance and is therefore too small. Pass n_boot.",
                RuntimeWarning, stacklevel=2)
        return prediction_interval(self.m, self.T_sq, V_M, self.n, alpha)

    def covariance(self) -> CovarianceDecomposition:
        """The fitted marginal covariance Sigma, split into its sources.

        ``Sigma = V_known + sum_c tau_c^2 Z_c Z_c'``. A thin wrapper so the call
        reads ``fits["A0c"].covariance()``; see :func:`marginal_covariance`
        for the mathematics and :class:`CovarianceDecomposition` for what comes
        back.
        """
        return marginal_covariance(self)

    def random_effects(self, alpha: float = 0.05) -> RandomEffects:
        """The estimated effect of every site, design and row (BLUPs).

        ``b_hat_c = tau_c^2 Z_c' Sigma^-1 (y - X beta_hat)``. A thin wrapper so
        the call reads ``fits["A0c"].random_effects().site``; see
        :func:`random_effects` for the mathematics and, before plotting them,
        its caveats on shrinkage.
        """
        return random_effects(self, alpha=alpha)

    def to_row(self) -> dict:
        """One row of a comparison table."""
        row = {
            "model": self.model_id,
            "estimator": self.estimator,
            "quantity": self.quantity,
            "contrast": self.contrast.name,
            # Which estimates: the id already says so (the _bc suffix), but a
            # column makes it filterable, so a table cannot mix corrected and
            # uncorrected rows without that showing.
            "corrected": self.corrected,
            # Which outer bootstrap the se_boot / ci / pi columns come from
            # (None if none was run). The id says so too (the _hb suffix); the
            # column makes it filterable, because under "record" the CI is for
            # these rows and under "hybrid" it is for the population.
            "bootstrap": self.boot_scheme,
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

        # Slopes, one group of columns per covariate (intercept-only fits add
        # nothing). Per-SD units, matching beta as fitted; coef_table(
        # natural_units=True) gives them per unit of the covariate.
        if len(self.x_names) > 1:
            ct = self.coef_table()
            for name in self.x_names[1:]:
                row |= {f"b[{name}]": ct.at[name, "beta"],
                        f"se_model[{name}]": ct.at[name, "se_model"],
                        f"se_sampling[{name}]": ct.at[name, "se_sampling"],
                        f"se_boot[{name}]": ct.at[name, "se_boot"]}
        return row

    def summary(self) -> str:
        """Multi-line report, in ratio units, with the reasoning attached."""
        tag = f"  {BC_TAG}" if self.corrected else ""
        if self.hybrid:
            tag += f"  {HB_TAG}"
        out = [f"{self.model_id} [{self.estimator}]{tag}  {self.contrast.label}"
               f"  ({self.contrast.name} = {self.contrast.terms[0]} - "
               f"{self.contrast.terms[1]}, {self.quantity}, n={self.n})",
               f"  summary effect   : {self.ratio():.4f}   "
               f"(log {self.m:+.4f})"]

        if self.replicates is not None:
            lo, hi = self.ci()
            # The two schemes answer different questions; say which, on the
            # line itself, so the interval cannot be quoted without it.
            what = ("where the MEAN effect lies, for the POPULATION of sites and "
                    "designs (effects redrawn)" if self.hybrid
                    else "where the MEAN effect lies")
            out.append(f"  95% CI           : {np.exp(lo):.4f} - {np.exp(hi):.4f}"
                       f"     <- {what}")
            if self.corrected:
                # Say how the interval was built. A reader comparing A2 with
                # A2_bc should know the replicates were moved, and by what.
                out.append("    -> percentile interval of the bias-corrected "
                           "estimator: replicates shifted by -(2b + s), see "
                           "FragilityData.replicate_shift")
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
                       f"<- {len(self.replicates)} refits"
                       + (", hybrid: compare with SE (superpop.)" if self.hybrid else ""))

        if self.hybrid:
            # Under the hybrid scheme the replicates redraw the effects, so the
            # like-for-like partner of se_boot is the SUPERPOPULATION SE, not
            # sqrt(w' V w). It should be equal or a little larger (the refits
            # also re-estimate the tau's). A large gap means the pseudo-data do
            # not have the covariance of the fitted model. Only checked for a
            # full-V REML fit: with a diagonal V the model-based SE is itself
            # wrong, so there is nothing sound to compare with.
            if self.estimator == "REML" and not self.spec.diagonal_V:
                ratio = self.se_boot / float(self.se_beta[0])
                if not 0.8 < ratio < 1.35:
                    out.append(
                        f"    -> hybrid bootstrap SE is {ratio:.2f}x the superpopulation "
                        "SE; these should agree to Monte-Carlo error (a little above "
                        "1 is expected). Check the fitted tau's and the replicate "
                        "alignment, or raise n_boot if it is close.")
            else:
                out.append("    -> fitted with a DIAGONAL sampling covariance, but the "
                           "hybrid noise keeps its full dependence (as the real data "
                           "do): replicate tau's sit ABOVE the fitted ones by the "
                           "shared covariance the diagonal fit absorbs.")
        elif self.se_boot is not None and self.se_sampling:
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
            line = (f"  {sd_name:<16s} : {np.sqrt(value):.4f}   "
                    f"(a factor of {np.exp(np.sqrt(value)):.3f})")
            # An interval on a tau only from HYBRID replicates: record-only
            # ones are inflated (see tau_ci), so printing theirs would put a
            # wrong interval beside a right estimate.
            if self.hybrid and name in self.replicates:
                ci = self.tau_ci(name)
                line += (f"   95% CI {np.sqrt(ci['lo']):.4f} - {np.sqrt(ci['hi']):.4f}"
                         f", {ci['frac_at_zero']:.0%} of replicates at 0")
            out.append(line)
        if self.hybrid:
            zero = self.boot.get("zero_levels", ())
            if zero:
                out.append(f"    -> {', '.join(zero)} fitted at 0, so drawn at 0: "
                           "replicates there can only move UP. Read those intervals "
                           "as [0, upper], and the replicate total as biased up.")

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

        # Covariates: every slope with its standard errors. The headline pair is
        # coefficient +/- se_model, the superpopulation SE that matches the
        # intercept's "SE (superpop.)" line above, with its Wald z and p. The
        # sampling and bootstrap SEs follow, for the same like-for-like check as
        # the intercept (they should agree with each other, and both be smaller
        # than se_model, which also carries tau). Slopes are per SD of the
        # centred/scaled covariate as fitted; the natural-units line divides by
        # that SD so the number can be read in the covariate's own units.
        if len(self.x_names) > 1:
            ct = self.coef_table()
            nat = self.coef_table(natural_units=True)
            out.append("  covariates (log scale; slope per SD of covariate):")
            for name in self.x_names[1:]:
                r = ct.loc[name]
                out.append(f"    {name}")
                out.append(f"      coefficient  : {r['beta']:+.5f}  "
                           f"SE {r['se_model']:.5f}   "
                           f"(z = {r['z']:+.2f}, p = {r['p']:.3g})   "
                           "<- SE includes tau")
                if np.isfinite(r["se_sampling"]):
                    out.append(f"      SE (sampling): {r['se_sampling']:.5f}")
                if np.isfinite(r["se_boot"]):
                    out.append(f"      SE (bootstr.): {r['se_boot']:.5f}   "
                               f"95% CI {r['ci_lo']:+.5f} to {r['ci_hi']:+.5f}")
                if name in self.scaling:
                    n_r = nat.loc[name]
                    out.append(f"      per unit     : {n_r['beta']:+.5f}  "
                               f"SE {n_r['se_model']:.5f}   "
                               f"(covariate SD = {self.scaling[name]['sd']:.4g})")

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


def _hybrid_components(spec: ModelSpec, con: Contrast, estimator: str,
                       tau2: dict[str, float]) -> dict[str, tuple]:
    """``{name: (factor, tau_hat)}`` for the hybrid bootstrap's effect draws.

    The factor matrices come from :func:`build_components`, the very call that
    builds the Gs of the fit, with its placeholder taus replaced by the FITTED
    standard deviations -- the use its docstring describes. So new effects are
    drawn on exactly the grouping the model was fitted with.

    DL has one bucket of between-row variance and no group structure, so its
    single component is the row level whatever the spec lists (a spec asking DL
    for more has already been refused by :func:`fit_dl`).
    """
    if estimator == "DL":
        return {"tau_u2": (np.eye(con.n), float(np.sqrt(tau2["tau_u2"])))}
    comps = build_components(spec, con)
    missing = sorted(set(comps) - set(tau2))
    if missing:
        raise ValueError(
            f"{spec.model_id}: the full-sample fit has no variance component(s) "
            f"{missing}, so the hybrid bootstrap cannot draw effects for them; it "
            f"holds {sorted(tau2)}")
    return {name: (factor, float(np.sqrt(tau2[name])))
            for name, (factor, _) in comps.items()}


def bootstrap_fits(con: Contrast, spec: ModelSpec, *, estimator: str,
                   n_boot: int | None = None, method: str = "cholesky",
                   n_starts: int = 1, full_fit: dict | None = None,
                   boot_seed: int = BOOT_SEED, **kw) -> pd.DataFrame:
    """Refit the model inside every bootstrap replicate: one row per replicate.

    This is the INFERENTIAL result (Efron & Tibshirani Ch. 6-7). It carries every sampling covariance,
    including ones never written down, which the model-based standard error
    cannot.

    Schemes -- ``spec.bootstrap_scheme`` chooses what is refitted
    ------------------------------------------------------------
    ``"record"`` (the default)
        Column r of ``con.y`` itself: the record resample. In Borenstein's
        notation (Fig. 12.3) a replicate is::

            y_i^(r) = (mu + zeta_i) + eps_i + delta_i^(r)
                       true, fixed    real     record
                                      analysis resample

        The spread of ``M_star`` is then ``w' V w``: how well the mean is known
        FOR THESE ROWS, a finite-population statement (Gelman & Hill Ch. 21.2,
        ~pp. 458-460, approximate). That part is right. The variance-component
        columns are NOT draws of the variance components: the replicate
        carries two rounds of sampling noise (eps and delta) while V_known
        describes one, so each replicate tau^2 is too large by about the mean
        sampling variance. Do not build an interval on a tau from them.
    ``"hybrid"``
        Pseudo-data with one round of noise (``dsim.hybrid_pseudo_data``)::

            y*^(r) = X beta_hat + zeta*^(r) + (y^(r) - ybar)

        with ``zeta*`` NEW effects drawn from the fitted variance components.
        ``Var(y*)`` is then the fitted model, so the replicate tau^2 centre on
        the fitted ones and :meth:`ModelFit.tau_ci` is meaningful. Because the
        effects are redrawn, the spread of ``M_star`` is about
        ``(X' Sigma^-1 X)^-1``: a SUPERPOPULATION statement, how well the mean
        is known for the population these sites and designs came from. It is a
        little wider still, because each replicate re-estimates the tau's and
        hence the weights. Semi-parametric: parametric in the effects,
        non-parametric in the noise (Efron & Tibshirani Ch. 6, ~pp. 53-56,
        approximate).

        Three things to know. (1) A component fitted at ZERO is drawn at zero;
        replicates can still estimate it above zero, never below, so its
        replicate mean is biased up and its interval is one-sided -- read
        :meth:`ModelFit.boundary_fractions` with it. (2) The noise keeps its
        full dependence even when the spec fits with a diagonal V (or DL):
        that is the position the real data put the estimator in, so there the
        replicate tau^2 sit ABOVE the fitted one by whatever shared covariance
        the diagonal fit absorbs. (3) ``full_fit`` is required.

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

    Columns of the returned frame
    -----------------------------
    ``M_star``
        The intercept, ``beta[0]``. Kept under this name because
        :meth:`ModelFit.ci`, :attr:`ModelFit.V_boot` and the forest plot read it.
    ``beta_<name>``
        EVERY fixed-effect coefficient, one column per entry of ``x_names``
        (so ``beta_intercept`` duplicates ``M_star``). Previously only the
        intercept was kept and the slopes were discarded, even though every
        refit computes them -- so a covariate had no bootstrap SE or CI at all.
        Labels come from :func:`beta_column`.
    ``T_sq``, plus one column per variance component (REML only).

    The frame's ``attrs`` record ``bootstrap_scheme`` and ``seed``, so a frame
    that has been separated from its fit still says how it was made.

    Parameters added for the hybrid scheme
    --------------------------------------
    full_fit : dict, optional
        ``{"beta": (q,), "tau2": {name: value}}`` from the full-sample fit of
        the SAME estimator. Required when the scheme is ``"hybrid"`` (the draws
        need beta_hat and the fitted tau's); ignored otherwise.
    boot_seed : int
        Seeds the effect draws. Named apart from ``mra.fit_reml``'s own
        ``seed`` (the restart jitter), which ``**kw`` can still reach. Same
        spec, data and seed give identical replicates, and replicate r does not
        depend on ``n_boot``. Ignored by the record-only scheme, which has
        nothing random in it.
    """
    columns = list(con.y.columns)
    if n_boot is not None:
        columns = columns[:n_boot]

    # The replicate matrix the loop below reads, one column per replicate.
    # Record-only: the cloud itself, untouched, so that path is exactly what it
    # was before the scheme existed. Hybrid: the pseudo-data, built in one call.
    hybrid = spec.bootstrap_scheme == "hybrid"
    if hybrid:
        if full_fit is None:
            raise ValueError(
                f"{spec.model_id}: the hybrid bootstrap needs the full-sample fit "
                "(full_fit={'beta': ..., 'tau2': {...}}), because its pseudo-data "
                "are X beta_hat plus effects drawn from the fitted tau's. "
                "ModelRegistry.fit passes it.")
        # DL is intercept-only whatever X holds, so its fitted values are M.
        X_fitted = np.ones((con.n, 1)) if estimator == "DL" else build_X(spec, con)[0]
        fitted = X_fitted @ np.atleast_1d(np.asarray(full_fit["beta"], dtype=float))
        # The WHOLE cloud goes in, and n_boot is applied inside: the noise is
        # centred on the mean of all k replicates, which is what V_known is the
        # covariance about (dsim.centre_replicates).
        Y_star = dsim.hybrid_pseudo_data(
            con.y.to_numpy(dtype=float), fitted,
            _hybrid_components(spec, con, estimator, full_fit["tau2"]),
            seed=boot_seed, n_boot=len(columns))

    # The coefficient names label the beta_ columns. build_X is cheap and
    # deterministic, and it is the SAME function the full-sample fit used, so
    # the labels are guaranteed to be in the same order as the fitted beta
    # vector. (DL ignores X -- it is intercept-only -- so for DL only the
    # intercept name is used below.)
    X, x_names, _ = build_X(spec, con)

    if estimator == "REML":
        V_known = con.diag_V() if spec.diagonal_V else con.V
        Gs, names = dsim.build_component_gs(build_components(spec, con))

    out = []
    for j, col in enumerate(columns):
        y_r = Y_star[:, j] if hybrid else con.y[col].to_numpy(dtype=float)
        if estimator == "DL":
            M, _, _, T_sq, _ = mra.compute_re_summary_effect(
                con.v, pd.Series(y_r, index=con.y_hat.index))
            out.append({"M_star": float(M), beta_column("intercept"): float(M),
                        "T_sq": float(T_sq)})
        else:
            raw = mra.fit_reml(y_r, X, V_known, Gs, names=names,
                               method=method, n_starts=n_starts, **kw)
            beta_r = np.atleast_1d(raw["beta"])
            row = {"M_star": float(beta_r[0])}
            # One column per coefficient, labelled by NAME, so nothing
            # downstream has to know the column order of X.
            row |= {beta_column(nm): float(b) for nm, b in zip(x_names, beta_r)}
            row["T_sq"] = float(np.sum(raw["tau2"]))
            row |= {n: float(t) for n, t in zip(raw["names"], raw["tau2"])}
            out.append(row)

    frame = pd.DataFrame(out, index=pd.Index(columns, name="k"))
    # attrs are metadata only: they change no value and are not compared by
    # assert_frame_equal, so a record-only frame is still identical to one made
    # before the scheme existed.
    frame.attrs["bootstrap_scheme"] = spec.bootstrap_scheme
    frame.attrs["seed"] = boot_seed if hybrid else None
    return frame


def beta_column(name: str) -> str:
    """Column label in :attr:`ModelFit.replicates` for the coefficient ``name``.

    A single function so the writer (:func:`bootstrap_fits`) and the readers
    (:attr:`ModelFit.se_boot_beta`, :meth:`ModelFit.coef_table`) cannot drift
    apart on the spelling.
    """
    return f"beta_{name}"


# =============================================================================
# Covariance structure and the estimated random effects
# =============================================================================
#
# THE MODEL EVERY REML RUNG FITS
# ------------------------------
# Written once for all of them, in matrix form (Gelman & Hill Ch. 12.5 and
# Ch. 13, ~pp. 262-270 and 279-282, approximate; BDA3 Ch. 5.4-5.6,
# ~pp. 113-128, approximate):
#
#     y = X beta + sum_c Z_c b_c + eps
#
#     b_c ~ N(0, tau_c^2 I_{L_c})     one vector per variance component c,
#                                     independent of each other
#     eps ~ N(0, V_known)             the sampling error, from the bootstrap
#
# where, for the crossed rungs (A0c, A0d, A1c, A3a):
#
#   component  | Z_c                           | b_c                 | L_c
#   -----------+-------------------------------+---------------------+-----
#   design     | Z_d, 120 x 51, Z_d[i,j] = 1   | d_j, one per design | 51
#              |   when g[i] = j               |                     |
#   site       | Z_s, 120 x 60, Z_s[i,k] = 1   | s_k, one per site   | 60
#              |   when sigma[i] = k           |                     |
#   row        | I_120                         | u_i, one per row    | 120
#
# so that, row by row,
#
#     y_i = x_i' beta + d_{g[i]} + s_{sigma[i]} + u_i + eps_i .
#
# Two things follow, and each gets a function below:
#
#   1. The MARGINAL covariance of y (what REML actually fits), obtained by
#      integrating the b_c out:
#
#          Sigma = Cov(y) = V_known + sum_c tau_c^2 Z_c Z_c'
#
#      -> marginal_covariance()
#
#   2. The best estimate of each b_c GIVEN the data, i.e. how far THIS site,
#      THIS design, THIS row sits from the average once sampling noise has
#      been allowed for (the BLUPs, or "partial pooling" estimates)
#
#          b_c_hat = tau_c^2 Z_c' Sigma^-1 (y - X beta_hat)
#
#      -> random_effects()
#
# Both CONDITION on the fitted variance components tau_c^2, i.e. treat them as
# known. The consequences of that are spelled out in random_effects().


# Plain-English names for the component keys build_components() produces,
# used as the keys of the returned objects. "row" is tau_u: the design x site
# interaction plus anything else specific to one (site, structure) pair.
_COMPONENT_NAMES = {"tau_d2": "design", "tau_s2": "site", "tau_u2": "row"}


def _component_levels(name: str, con: Contrast) -> pd.Index:
    """The labels of the columns of Z_c, in the SAME order the codes use.

    Getting this order wrong would attach site 7's effect to site 8's label and
    nothing would complain, so the labels are built exactly the way
    :attr:`Contrast.g` and :attr:`Contrast.site_codes` build their codes: from
    the SORTED unique labels, mapped with ``mra.build_codes``.
    """
    if name == "tau_d2":
        return pd.Index(sorted(con.rows["design_group_id"].unique()),
                        name="design_group_id")
    if name == "tau_s2":
        return pd.Index(sorted(con.rows.index.get_level_values("site").unique()),
                        name="site")
    if name == "tau_u2":
        # One level per row: the identity matrix's columns are the rows.
        return con.y_hat.index
    raise ValueError(f"unknown component {name!r}")


def _fit_V_known(fit: ModelFit) -> np.ndarray:
    """The sampling covariance the fit was ACTUALLY handed.

    Must match :func:`fit_reml` / :func:`fit_dl` exactly, otherwise the Sigma
    rebuilt here is not the Sigma that was fitted:

    * REML on a ``diagonal_V`` spec (A1a) -> ``diag(v)``;
    * DerSimonian-Laird -> ``diag(v)`` always, because DL only ever sees the
      variance vector (Borenstein Ch. 12, ~pp. 72-75, approximate);
    * every other REML fit -> the full ``V``.
    """
    con = fit.contrast
    if fit.spec.diagonal_V or fit.estimator == "DL":
        return con.diag_V()
    return con.V


@dataclass(frozen=True)
class CovarianceDecomposition:
    """The marginal covariance of the effect sizes, and every piece of it.

    WHAT IT IS
    ----------
    ::

        Sigma = V_known + tau_d^2 Z_d Z_d' + tau_s^2 Z_s Z_s' + tau_u^2 I

    Element by element, for two rows i and k::

        Sigma[i, k] = V_known[i, k]                    sampling error they share
                    + tau_d^2 * [g[i] == g[k]]         same design?
                    + tau_s^2 * [sigma[i] == sigma[k]] same site?
                    + tau_u^2 * [i == k]               same row?

    where ``[.]`` is 1 when the condition holds and 0 otherwise. So:

    * the DIAGONAL is the total variance of one effect size -- sampling
      variance plus every tau^2 that applies to it;
    * the 3-storey and 5-storey rows at one site are coupled by ``tau_s^2``
      (plus whatever record sharing puts into ``V_known``);
    * rows sharing a design are coupled by ``tau_d^2`` and, through the shared
      fixed-set arm, by the dense ``Z V_bb Z'`` part of ``V_known``;
    * rows sharing neither are coupled only through ``V_known``.

    An analogy: think of each effect size as a measurement made with several
    instruments at once. The sampling error is the noise of the instrument;
    each tau^2 is a "shared cause" that nudges every measurement made at the
    same site (or with the same design) up or down together. Sigma records,
    for every pair of measurements, how many shared causes they have.

    Attributes
    ----------
    Sigma : ndarray (n, n)
        The fitted marginal covariance. For REML fits this is checked against
        ``raw["Sigma"]`` from ``mra.fit_reml`` when it is built.
    V_known : ndarray (n, n)
        The sampling part, exactly as the fit used it (full or diagonal).
    parts : dict of str -> ndarray (n, n)
        ``tau_c^2 Z_c Z_c'`` per component, keyed ``"design"``, ``"site"``,
        ``"row"``. ``V_known + sum(parts.values()) == Sigma``.
    tau2 : dict of str -> float
        The fitted variance components, keyed as ``parts``.
    Z : dict of str -> ndarray (n, L_c)
        The factor (indicator) matrices, keyed as ``parts``.
    levels : dict of str -> Index
        The label of each column of ``Z[c]``: design ids, site ids, row labels.
    row_index : Index
        The label of each row of ``Sigma``.
    """

    Sigma: np.ndarray
    V_known: np.ndarray
    parts: dict[str, np.ndarray]
    tau2: dict[str, float]
    Z: dict[str, np.ndarray]
    levels: dict[str, pd.Index]
    row_index: pd.Index

    def correlation(self) -> pd.DataFrame:
        """``Sigma`` rescaled to a correlation matrix, labelled by row.

        ``R[i, k] = Sigma[i, k] / sqrt(Sigma[i, i] * Sigma[k, k])``. Easier to
        read than Sigma itself: it says directly how strongly two effect sizes
        move together, on a 0-1 scale.
        """
        sd = np.sqrt(np.diag(self.Sigma))
        return pd.DataFrame(self.Sigma / np.outer(sd, sd),
                            index=self.row_index, columns=self.row_index)

    def variance_table(self) -> pd.DataFrame:
        """The DIAGONAL of Sigma, split into its sources, one row per effect size.

        Columns: ``sampling`` (``V_known[i, i]``), one column per component
        (``tau_c^2``, the same for every row), ``total`` (``Sigma[i, i]``), and
        ``share_<source>`` = that source / total.

        The shares answer "how much of the uncertainty about THIS effect size is
        sampling noise, and how much is real between-study variation, by
        source". Their mean over rows is a descriptive summary, not an estimate
        of anything -- and, as nb 073 section 2.2.3 warns, the between-source
        shares depend on how widely the sites and designs in THIS study were
        sampled (Gelman & Hill Ch. 21, ~pp. 459-463, approximate).
        """
        out = pd.DataFrame({"sampling": np.diag(self.V_known)}, index=self.row_index)
        for name, part in self.parts.items():
            out[name] = np.diag(part)
        out["total"] = np.diag(self.Sigma)
        for col in ["sampling", *self.parts]:
            out[f"share_{col}"] = out[col] / out["total"]
        return out


def marginal_covariance(fit: ModelFit) -> CovarianceDecomposition:
    """Rebuild the fitted marginal covariance ``Sigma`` of a fit, piece by piece.

    ``Sigma = V_known + sum_c tau_c^2 Z_c Z_c'`` -- see
    :class:`CovarianceDecomposition` for what each piece means.

    The Z matrices come from :func:`build_components`, i.e. from the SAME code
    the fit used, and ``V_known`` from :func:`_fit_V_known`, which mirrors the
    fit's choice of full or diagonal. For a REML fit the result is then checked
    against the ``Sigma`` that ``mra.fit_reml`` returned: if the two disagree,
    this function is not describing the model that was fitted, and it says so
    rather than returning a plausible-looking wrong answer.

    Works for DL fits too (one component, ``tau_u``, and a diagonal V_known),
    which makes it the same object for every rung.

    Parameters
    ----------
    fit : ModelFit
        Any fitted rung, e.g. ``fits["A0c"]``.

    Returns
    -------
    CovarianceDecomposition
    """
    con = fit.contrast
    V_known = np.asarray(_fit_V_known(fit), dtype=float)

    # build_components returns {"tau_d2": (Z_d, 0.0), "tau_s2": (Z_s, 0.0),
    # "tau_u2": (I, 0.0)} -- the placeholder 0.0s are the simulator's tau_true
    # and are ignored here. Only the components THIS spec asked for are present.
    comps = build_components(fit.spec, con)

    Z, parts, tau2, levels = {}, {}, {}, {}
    Sigma = V_known.copy()
    for key, (factor, _) in comps.items():
        name = _COMPONENT_NAMES[key]
        factor = np.asarray(factor, dtype=float)
        # The fitted tau^2 for this component. fit.tau2 is keyed by the same
        # "tau_d2"/"tau_s2"/"tau_u2" names build_components produces; a missing
        # key would mean the spec and the fit disagree, so fail loudly.
        if key not in fit.tau2:
            raise KeyError(f"{fit.model_id}: component {key!r} is in the spec but not "
                           f"in the fitted tau2 {sorted(fit.tau2)}")
        t2 = float(fit.tau2[key])
        Z[name] = factor
        tau2[name] = t2
        # G_c = Z_c Z_c' has a 1 wherever two rows share a level of c.
        parts[name] = t2 * (factor @ factor.T)
        levels[name] = _component_levels(key, con)
        Sigma = Sigma + parts[name]

    # For REML the fitter kept its own Sigma: the two must be the same matrix.
    # (DL keeps no Sigma, and its tau2 is the DL moment estimate, so there is
    # nothing to compare against.)
    fitted_Sigma = fit.raw.get("Sigma")
    if fitted_Sigma is not None and not np.allclose(Sigma, fitted_Sigma,
                                                    rtol=1e-10, atol=1e-14):
        raise RuntimeError(
            f"{fit.model_id} [{fit.estimator}]: the rebuilt Sigma does not match the "
            f"one mra.fit_reml returned (max |diff| "
            f"{np.max(np.abs(Sigma - fitted_Sigma)):.2e}). The spec, V_known or the "
            "component order has drifted from the fit.")

    return CovarianceDecomposition(Sigma=Sigma, V_known=V_known, parts=parts,
                                   tau2=tau2, Z=Z, levels=levels,
                                   row_index=con.y_hat.index)


@dataclass(frozen=True)
class RandomEffects:
    """The estimated effect of every site, design and row, and the leftover noise.

    Attributes
    ----------
    effects : dict of str -> DataFrame
        One frame per variance component, keyed ``"design"``, ``"site"``,
        ``"row"`` (only the components the spec has). One row per LEVEL of that
        component (51 designs, 60 sites, 120 rows), with columns

        ``effect``       b_hat, on the log scale
        ``se``           sqrt of the prediction-error variance (see below)
        ``lo``, ``hi``   effect -/+ z * se
        ``ratio``        exp(effect): "this site multiplies the ratio by ..."
        ``ratio_lo``, ``ratio_hi``
        ``reliability``  1 - PEV / tau^2, in [0, 1]: how much the data pin
                         this level down (0 = not at all, estimate is 0)
        ``n_rows``       rows that belong to this level
    rows : DataFrame
        One row per effect size: ``y``, the fixed part ``fixed`` = x_i' beta_hat,
        one column per component giving that component's contribution to this
        row (``design`` = d_hat_{g[i]}, ``site`` = s_hat_{sigma[i]}, ``row`` =
        u_hat_i), ``sampling`` = eps_hat_i, and the factor columns. By
        construction ``y == fixed + design + site + row + sampling`` exactly.
    covariance : CovarianceDecomposition
        The Sigma these were computed from.
    alpha : float
        The level the ``lo``/``hi`` bands were built at.
    """

    effects: dict[str, pd.DataFrame]
    rows: pd.DataFrame
    covariance: CovarianceDecomposition
    alpha: float

    def __getitem__(self, name: str) -> pd.DataFrame:
        """``re["site"]`` -- the per-level table of one component."""
        if name not in self.effects:
            raise KeyError(f"no {name!r} component in this fit - it has "
                           f"{sorted(self.effects)}")
        return self.effects[name]

    @property
    def site(self) -> pd.DataFrame:
        """The 60 site effects s_hat_k (only for fits with a site component)."""
        return self["site"]

    @property
    def design(self) -> pd.DataFrame:
        """The 51 design effects d_hat_j (only for fits with a design component)."""
        return self["design"]

    @property
    def row(self) -> pd.DataFrame:
        """The row-level effects u_hat_i (the design x site interaction)."""
        return self["row"]


def random_effects(fit: ModelFit, alpha: float = 0.05) -> RandomEffects:
    """Estimate every random effect of a fitted model: sites, designs, rows.

    THE QUESTION
    ------------
    tau_s = 0.086 says the SITES differ from one another by about 9 % on
    average. It does not say WHICH sites sit high and which sit low. This
    function answers that: one number per site (and per design, and per row),
    with an uncertainty.

    THE ESTIMATE (BLUP / partial pooling)
    -------------------------------------
    Because every b_c and y are jointly normal, the best estimate of b_c given
    the data is its conditional mean (BDA3 Appendix A, the conditional-normal
    formula, ~p. 580, approximate). With ``Cov(b_c, y) = tau_c^2 Z_c'``::

        E[b_c | y] = Cov(b_c, y) Sigma^-1 (y - X beta)
                   = tau_c^2 Z_c' Sigma^-1 r,        r = y - X beta_hat

    once the unknown beta is replaced by its GLS estimate. This is Henderson's
    best linear unbiased predictor (BLUP). The sampling-noise share of each
    residual comes out the same way, with ``Cov(eps, y) = V_known``::

        eps_hat = V_known Sigma^-1 r

    Adding them up gives back the residual EXACTLY, because
    ``V_known + sum_c tau_c^2 Z_c Z_c' = Sigma``::

        sum_c Z_c b_hat_c + eps_hat = Sigma Sigma^-1 r = r .

    So every effect size splits into ``fixed + design + site + row + sampling``
    with nothing left over; :attr:`RandomEffects.rows` shows that split and it
    is asserted here. And when X contains an intercept, the GLS normal
    equations give ``1' Sigma^-1 r = 0``, so every component's effects sum to
    exactly zero over its levels (``Z_c`` has one 1 per row, so
    ``1' Z_c' = 1'``) -- also asserted.

    WHY THE ESTIMATES ARE "SHRUNK"
    ------------------------------
    In the simplest case -- one site, its own rows independent, nothing else --
    the formula reduces to (Gelman & Hill Ch. 12.2 eq. 12.1, ~p. 253; BDA3
    eq. 5.17, ~p. 116; both approximate)::

        s_hat_k = [tau_s^2 / (tau_s^2 + v_k / n_k)] * (mean residual at site k)

    The bracket is between 0 and 1. A site measured precisely (small v_k)
    keeps nearly all of its raw deviation; a noisy one is pulled towards zero,
    because much of its deviation is probably noise. An analogy: a teacher
    judging a new pupil from two test scores does not take an extreme score at
    face value, but pulls it part-way back towards the class average -- more so
    the fewer and noisier the tests. In the crossed model the same logic runs
    through Sigma^-1, which also allows for the sampling errors and effects
    that rows share.

    THE UNCERTAINTY (prediction-error variance)
    -------------------------------------------
    With ``P = Sigma^-1 - Sigma^-1 X (X' Sigma^-1 X)^-1 X' Sigma^-1`` (the
    REML projection matrix, which also removes the part of y used to estimate
    beta)::

        Var(b_hat_c)       = tau_c^4 Z_c' P Z_c
        Var(b_hat_c - b_c) = tau_c^2 I - tau_c^4 Z_c' P Z_c     (PEV)

    ``se`` is the square root of the diagonal of the PEV: how far the TRUE
    effect of this site is likely to be from the estimate. It includes the
    uncertainty from estimating beta, but NOT that from estimating tau_c^2.

    ``reliability = 1 - PEV / tau_c^2 = Var(b_hat) / tau_c^2`` reads as "what
    fraction of this level's true variation the data have captured". Near 1:
    the data determine this site well. Near 0: the estimate is close to zero
    simply because the data say little about this site.

    CAVEATS -- READ BEFORE PLOTTING
    -------------------------------
    * The tau_c^2 are treated as known. The ``se`` are therefore somewhat too
      small, most so for components estimated from few levels. The Bayesian
      route (A4/A5) removes this; the outer bootstrap does not, since it holds
      the tau^2 structure fixed too.
    * A component fitted at tau_c^2 = 0 (tau_d in A0c/A1c) has every effect
      EXACTLY zero with se zero. That is the model's statement that the data
      show no design-to-design variation, not a numerical failure.
    * Shrinkage pulls the estimates towards zero, and it pulls noisy levels
      further. So a plot of s_hat against a covariate shows the PATTERN
      faithfully but understates the SLOPE, and regressing the s_hat on a
      covariate is not a substitute for putting the covariate in X (Gelman &
      Hill Ch. 12.6, ~pp. 265-269, approximate). Use the plot to look, and the
      meta-regression to measure.
    * For that look, take the effects from the model WITHOUT the covariate
      (e.g. A0c or A1c), so nothing has been absorbed yet. From a model WITH
      covariates (A3a) the effects are what the covariates leave unexplained --
      useful for checking whether a relationship is gone.
    * The spread of the estimates is SMALLER than tau_c (shrinkage again):
      roughly ``var(b_hat) + mean(PEV) ~ tau_c^2``. Do not compare the SD of
      the s_hat with tau_s and conclude the fit is wrong.

    Parameters
    ----------
    fit : ModelFit
        Any fitted rung. REML fits are the intended use; a DL fit gives the
        two-level version (one ``row`` component, diagonal V_known).
    alpha : float
        For the ``lo``/``hi`` bands: effect -/+ z_{1-alpha/2} * se. These are
        normal-theory bands conditional on tau^2, not confidence intervals in
        the bootstrap sense used for m.

    Returns
    -------
    RandomEffects

    Examples
    --------
    >>> re = fits["A0c"].random_effects()
    >>> re.site.sort_values("effect").head()          # the lowest sites
    >>> re.rows[["y", "fixed", "site", "row", "sampling"]].head()
    >>> cov = fits["A0c"].covariance()
    >>> cov.variance_table().filter(like="share_").mean()
    """
    con = fit.contrast
    cov = marginal_covariance(fit)

    # The fixed part, exactly as the fit built it (intercept + centred/scaled
    # covariates), and the marginal residual r = y - X beta_hat.
    X, _, _ = build_X(fit.spec, con)
    y = con.y_hat.to_numpy(dtype=float)
    fixed = X @ fit.beta
    r = y - fixed

    # Every product below needs Sigma^-1 applied to something. Solve rather
    # than invert where only a product is needed: it is cheaper and more
    # accurate. Sigma is symmetric positive definite, so np.linalg.solve is
    # safe; n = 120 makes speed irrelevant.
    Si_r = np.linalg.solve(cov.Sigma, r)          # Sigma^-1 r
    Si_X = np.linalg.solve(cov.Sigma, X)          # Sigma^-1 X
    XtSiX_inv = np.linalg.inv(X.T @ Si_X)         # (X' Sigma^-1 X)^-1 = Var(beta_hat)

    # The sampling share of each residual, eps_hat = V_known Sigma^-1 r.
    eps_hat = cov.V_known @ Si_r

    z = stats.norm.ppf(1 - alpha / 2)
    effects: dict[str, pd.DataFrame] = {}
    row_parts: dict[str, np.ndarray] = {}

    for name, Zc in cov.Z.items():
        t2 = cov.tau2[name]

        # --- the BLUP: b_hat_c = tau_c^2 Z_c' Sigma^-1 r ------------------------
        b_hat = t2 * (Zc.T @ Si_r)

        # --- its prediction-error variance ------------------------------------
        # Z_c' P Z_c without forming the n x n matrix P:
        #   Z' P Z = Z' Sigma^-1 Z  -  (Z' Sigma^-1 X)(X' Sigma^-1 X)^-1 (X' Sigma^-1 Z)
        Si_Z = np.linalg.solve(cov.Sigma, Zc)                 # Sigma^-1 Z_c
        ZtPZ = Zc.T @ Si_Z - (Zc.T @ Si_X) @ XtSiX_inv @ (Si_X.T @ Zc)
        pev = t2 * np.eye(Zc.shape[1]) - t2 ** 2 * ZtPZ
        # PEV is a variance, so any negative diagonal is round-off (it happens
        # when tau^2 is at or near zero); clip it rather than take sqrt(-1e-18).
        pev_diag = np.clip(np.diag(pev), 0.0, None)
        se = np.sqrt(pev_diag)

        # reliability = 1 - PEV / tau^2; undefined when tau^2 = 0, where the
        # data have nothing to say (and the effect is 0 by construction).
        with np.errstate(invalid="ignore", divide="ignore"):
            reliability = np.where(t2 > 0, 1.0 - pev_diag / t2, np.nan)

        effects[name] = pd.DataFrame({
            "effect": b_hat,
            "se": se,
            "lo": b_hat - z * se,
            "hi": b_hat + z * se,
            "ratio": np.exp(b_hat),
            "ratio_lo": np.exp(b_hat - z * se),
            "ratio_hi": np.exp(b_hat + z * se),
            "reliability": reliability,
            "n_rows": Zc.sum(axis=0).astype(int),
        }, index=cov.levels[name])

        # This component's contribution to each row: Z_c b_hat_c, i.e.
        # d_hat_{g[i]}, s_hat_{sigma[i]} or u_hat_i.
        row_parts[name] = Zc @ b_hat

    # --- the per-row split: y = fixed + sum of components + sampling -----------
    rows = pd.DataFrame({"y": y, "fixed": fixed}, index=con.y_hat.index)
    for name, part in row_parts.items():
        rows[name] = part
    rows["sampling"] = eps_hat
    # Carry the factors so the table can be grouped, coloured or joined to
    # covariates without another lookup. Only the columns that exist at this
    # scope (a group-scope contrast has no site column).
    for col in ("design_group_id", "structure_id", "n_storeys"):
        if col in con.rows.columns:
            rows[col] = con.rows[col].to_numpy()

    # --- the two exact identities (see the docstring) ---------------------------
    rebuilt = rows["fixed"] + sum(rows[name] for name in row_parts) + rows["sampling"]
    if not np.allclose(rebuilt.to_numpy(), y, rtol=0, atol=1e-10):
        raise RuntimeError(
            f"{fit.model_id}: fixed + random + sampling does not reproduce y "
            f"(max |diff| {np.max(np.abs(rebuilt.to_numpy() - y)):.2e}); "
            "Sigma and the components disagree.")
    has_intercept = np.allclose(X[:, 0], 1.0)
    if has_intercept:
        for name, frame in effects.items():
            total = frame["effect"].sum()
            if abs(total) > 1e-8:
                raise RuntimeError(
                    f"{fit.model_id}: the {name} effects sum to {total:.2e}, not 0. "
                    "With an intercept they must (1' Sigma^-1 r = 0); beta_hat does "
                    "not appear to be the GLS estimate for this Sigma.")

    return RandomEffects(effects=effects, rows=rows, covariance=cov, alpha=alpha)


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


class ModelRegistry(fcache.FitCacheMixin):
    """Every fit for one container, keyed by roadmap id. Refuses silent overwrites.

    The direct answer to the rebinding problem: in nb 073, re-running the A1
    block after A2 leaves the numbers of A2 sitting in variables named as though
    they belonged to A1, and nothing complains. Here a second ``fit("A1a")``
    raises unless ``overwrite=True`` is passed.

    On-disk cache
    -------------
    Pass ``cache_dir`` and :meth:`fit` looks on disk first: a saved fit whose
    inputs are unchanged is LOADED (minutes of bootstrap saved), anything else
    is fitted and SAVED. Without ``cache_dir`` nothing is read or written and
    :meth:`fit` is exactly what it always was. The rules -- what makes a file
    valid, why a 5000-replicate file serves a 200-replicate request, the folder
    layout -- are in the docstring of :mod:`phd_project.scripts.fit_cache`.
    ``cache_tag`` separates two containers that share ``im_tag``, quantity and
    storeys but hold different data (nb 034's ``theta`` and ``theta_star``);
    without it they would overwrite each other's files on every run. ``None``
    and ``""`` both mean no tag.

    ``fx_cache_dir`` (optional) is where the fits of models whose contrast uses
    ONLY fixed-record-set arms (the A2 family, y2 = b - c) are cached instead of
    ``cache_dir``. Those fits do not depend on the site-specific record set, so
    they belong to the FX set rather than to one study: every study that pairs an
    SS set with this FX set then shares them. Sharing is safe because a fit's
    fingerprint holds only its own contrast, spec and code - the whole-container
    hash is recorded in the metadata but never compared.

    Examples
    --------
    >>> theta = load_fragility_data(root, "AvgSA_03", quantity="theta")
    >>> fits = ModelRegistry(theta)
    >>> fits.fit("A0a", n_boot=500)
    >>> print(fits["A0a", "REML"].summary())
    >>> fits.table()

    >>> cached = ModelRegistry(theta, cache_dir=FIT_CACHE)
    >>> for model_id in ["A0a", "A1c"]:
    ...     cached.fit(model_id, n_boot=theta.k)
    >>> cached.cache_summary()
    """

    def __init__(self, data: FragilityData,
                 specs: dict[str, ModelSpec] | None = None, *,
                 cache_dir: Path | str | None = None,
                 cache_tag: str | None = None,
                 fx_cache_dir: Path | str | None = None):
        self.data = data
        self.specs = dict(MODEL_SPECS if specs is None else specs)
        self.fx_cache_dir = None if fx_cache_dir is None else Path(fx_cache_dir)
        self._fits: dict[tuple[str, str], ModelFit] = {}
        # Keyed on (name, corrected). Keyed on the name alone, the corrected
        # and uncorrected versions of a contrast would share one slot, and a
        # `_bc` fit could silently reuse the uncorrected contrast (or the
        # reverse), depending only on which was fitted first.
        self._contrasts: dict[tuple[str, bool], Contrast] = {}
        # Cache state: cache_dir, cache_tag, cache_log (fit_cache.FitCacheMixin).
        self._init_fit_cache(cache_dir, cache_tag)
        # Contrast fingerprint entries, memoised on the same key as _contrasts:
        # hashing the 120 x 5000 replicate matrix once per contrast, not once
        # per model and estimator.
        self._contrast_fp: dict[tuple[str, bool], dict] = {}
        self._container_hash: str | None = None

    # -- contrasts, memoised so V is computed once per contrast -------------

    def contrast(self, name: str, corrected: bool = False) -> Contrast:
        """The contrast of this name, built once per ``(name, corrected)`` and cached.

        ``corrected`` exists here only so :meth:`fit` can pass the SPEC's flag
        through. Notebooks choose corrected or uncorrected by choosing a model
        id (``"A2"`` or ``"A2_bc"``), not by calling this.
        """
        key = (name, bool(corrected))
        if key not in self._contrasts:
            self._contrasts[key] = make_contrast(self.data, name, corrected=corrected)
        return self._contrasts[key]

    # -- fitting -------------------------------------------------------------

    def fit(self, model_id: str, *, n_boot: int | None = None,
            overwrite: bool = False, verbose: bool = False,
            boot_seed: int = BOOT_SEED, refit: bool = False,
            **kw) -> list[ModelFit]:
        """Fit one rung with every estimator its spec asks for.

        This single call replaces the six-cell block nb 073 repeats three times.

        With a ``cache_dir`` on the registry, each estimator is first looked
        up on disk and loaded if its saved inputs match (one line printed per
        estimator, saying "loaded" or why it was refitted); otherwise it is
        fitted and saved. DL and REML are separate files, so a valid DL file
        is loaded even when REML has to be refitted. A loaded fit is the same
        object a fresh one would be -- same spec, the registry's own memoised
        contrast, replicates sliced to ``n_boot`` -- so everything downstream
        works identically on either.

        Parameters
        ----------
        model_id : str
            A key of :data:`MODEL_SPECS`.
        n_boot : int, optional
            Number of outer-bootstrap replicates to refit in. ``None`` skips the
            outer bootstrap, which means no confidence interval and only a
            model-based prediction interval. Pass an int for anything
            reportable. Use all of them (``n_boot=data.k``) for the final run.
            With the cache, a file saved with at least this many replicates is
            loaded and sliced; one saved with fewer is refitted.
        overwrite : bool
            Allow replacing an existing fit of the same id IN MEMORY. Applies
            to a loaded fit exactly as to a fitted one: loading never silently
            replaces what a later cell reports.
        boot_seed : int
            Seeds the random-effect draws of a HYBRID-bootstrap spec (an
            ``_hb`` id). The default is the module constant, so the same id
            gives the same replicates in every notebook. Has no effect on a
            record-only spec, whose bootstrap has nothing random in it.
        refit : bool
            Ignore what is on disk: fit, and overwrite the saved file. No effect
            without a ``cache_dir``. Different from ``overwrite``, which is about
            the in-memory fit.

        There is deliberately no ``bootstrap_scheme`` argument either, for the
        reason given next for ``corrected``: the scheme is part of the spec.

        There is deliberately no ``corrected`` argument. Whether a fit uses the
        bias-corrected estimates is a property of its spec (``"A2"`` vs
        ``"A2_bc"``), so the id alone always identifies the numbers. Passing
        ``corrected=`` raises TypeError rather than disappearing into ``**kw``.
        """
        if model_id not in self.specs:
            raise KeyError(
                f"unknown model {model_id!r} - known: {sorted(self.specs)}. "
                "A new rung is a MODEL_SPECS entry, not a new block of code.")
        if "corrected" in kw:
            # Without this guard, `corrected=` would slip through **kw into
            # mra.fit_reml and fail there with an error that does not say what
            # went wrong. Worse, if fit_reml ever accepted the keyword, it would
            # silently make "A2" mean two different things.
            raise TypeError(
                f"fit({model_id!r}, corrected=...) is not supported: whether a fit "
                "uses bias-corrected estimates is part of the model's SPEC, so that "
                f"an id always means one set of numbers. Fit {model_id + '_bc'!r} "
                "(its corrected version) instead, adding it to MODEL_SPECS if it "
                "does not exist yet.")
        spec = self.specs[model_id]

        try:
            con = self.contrast(spec.contrast, spec.corrected)
        except KeyError as exc:
            if not spec.corrected:
                raise
            # FragilityData.hat / plugin_offset name the ARM that is missing; this
            # adds the MODEL, which is what the person running the notebook
            # actually asked for.
            raise KeyError(
                f"{model_id} is a bias-corrected spec, but the data it needs is "
                f"missing: {exc.args[0] if exc.args else exc}. The bias-corrected "
                "estimates come from nb 070 section 2.4 via nb 072.") from exc

        if spec.corrected:
            # theta is never corrected (nb 070 section 2.2 corrects it by zero),
            # so on a theta container a `_bc` spec reproduces its partner exactly.
            # Fitted anyway (user decision 2026-10-07), but said out loud, so a
            # table row labelled "bias-corrected" is not read as a correction
            # that happened.
            left, right = con.terms
            if all((self.data.hat(s) == self.data.hat(s, corrected=True)).all()
                   for s in (left, right)):
                warnings.warn(
                    f"{model_id}: the bias-corrected and uncorrected point estimates "
                    f"are identical for both arms of {spec.contrast} "
                    f"(quantity={self.data.quantity!r}), so the correction is a "
                    f"no-op on the estimate and {model_id} gives the same summary "
                    f"effect as {model_id.removesuffix('_bc')}. theta is never "
                    "bias-corrected (nb 070 section 2.2).",
                    UserWarning, stacklevel=2)

        out: list[ModelFit] = []
        for estimator in spec.estimators:
            key = (model_id, estimator)
            if key in self._fits and not overwrite:
                raise ValueError(
                    f"{model_id} [{estimator}] has already been fitted. Pass "
                    "overwrite=True if you mean to replace it - the refusal is "
                    "deliberate, so a re-run cannot silently change what a later "
                    "cell reports.")

            # Without a cache this is the original loop body, unchanged; with
            # one, the same body runs only when no valid file exists.
            if self._model_cache_dir(model_id) is None:
                fit_obj = self._fit_estimator(spec, con, estimator, n_boot=n_boot,
                                              verbose=verbose, boot_seed=boot_seed, **kw)
            else:
                fit_obj = self._cached_fit_estimator(
                    spec, con, estimator, n_boot=n_boot, verbose=verbose,
                    boot_seed=boot_seed, refit=refit, **kw)
            self._fits[key] = fit_obj
            out.append(fit_obj)
        return out

    def _fit_estimator(self, spec: ModelSpec, con: Contrast, estimator: str, *,
                       n_boot: int | None, verbose: bool, boot_seed: int,
                       **kw) -> ModelFit:
        """One estimator of one spec: full-sample fit, then the outer bootstrap.

        The body of the estimator loop of :meth:`fit`, moved here unchanged so
        the cache can run it only when needed. It stores nothing; :meth:`fit`
        does. This method is a root of the code hash
        (:data:`UNIVARIATE_FIT_CODE_ROOTS`): it builds the ``boot`` record that
        is saved with the numbers.
        """
        model_id = spec.model_id
        if estimator == "DL":
            beta, se, tau2, het = fit_dl(con, spec)
            x_names, scaling, raw = ["intercept"], {}, {}
        elif estimator == "REML":
            beta, se, tau2, het, x_names, scaling, raw = fit_reml(con, spec, **kw)
        else:
            raise ValueError(f"unknown estimator {estimator!r}")

        reps, boot = None, {}
        if n_boot is not None:
            hybrid = spec.bootstrap_scheme == "hybrid"
            if verbose:
                print(f"  {model_id} [{estimator}]: outer bootstrap over "
                      f"{min(n_boot, con.k)} replicates"
                      f"{' (hybrid)' if hybrid else ''}...")
            # The full-sample fit goes in because the hybrid scheme draws
            # its pseudo-data from it (beta_hat and the fitted tau's). The
            # record-only scheme ignores both extra arguments.
            reps = bootstrap_fits(con, spec, estimator=estimator, n_boot=n_boot,
                                  full_fit={"beta": beta, "tau2": tau2},
                                  boot_seed=boot_seed)
            # What was actually run, carried on the fit so that no reader of
            # the replicates has to guess the scheme.
            boot = {"scheme": spec.bootstrap_scheme,
                    "seed": boot_seed if hybrid else None,
                    "zero_levels": tuple(k for k, v in tau2.items() if v == 0.0)
                    if hybrid else ()}

        return ModelFit(spec=spec, estimator=estimator, contrast=con,
                        beta=np.atleast_1d(beta), se_beta=np.atleast_1d(se),
                        tau2=tau2, het=het, replicates=reps,
                        x_names=x_names, scaling=scaling, raw=raw, boot=boot)

    # -- the on-disk cache (active only with cache_dir) -----------------------

    def _model_cache_dir(self, model_id: str) -> Path | None:
        """``fx_cache_dir`` for a model whose contrast uses only fixed-set arms, else
        ``cache_dir`` (see the class docstring)."""
        spec = self.specs.get(model_id)
        if self.fx_cache_dir is not None and spec is not None:
            arms = _CONTRAST_SPECS.get(spec.contrast, ("a",))[:2]
            if "a" not in arms:
                return self.fx_cache_dir
        return self.cache_dir

    def cache_path(self, model_id: str, estimator: str) -> Path:
        """Where ``(model_id, estimator)`` is saved: ``.../<quantity>/<id>_<est>.pkl``."""
        cache_dir = self._model_cache_dir(model_id)
        if cache_dir is None:
            raise ValueError("this registry has no cache_dir")
        return (fcache.cache_folder(cache_dir, self.data, self.cache_tag)
                / self.data.quantity / f"{model_id}_{estimator}.pkl")

    def _cache_fingerprint(self, spec: ModelSpec, con: Contrast, estimator: str,
                           kw: dict, boot_seed: int) -> tuple[dict, dict, dict]:
        """``(fingerprint, meta, code inventory)`` of one ``(spec, estimator)``.

        The five entries of the fit_cache module docstring. Per estimator:
        DL ignores ``**kw`` (it never reaches fit_dl or the DL bootstrap), so
        its fingerprint has no ``kw`` entry and a change of ``n_starts`` does
        not refit it.
        """
        ckey = (con.name, bool(con.corrected))
        if ckey not in self._contrast_fp:
            self._contrast_fp[ckey] = fcache.contrast_entry(con)
        digest, inventory = fcache.code_hash(UNIVARIATE_FIT_CODE_ROOTS)
        fp = fcache.build_fingerprint(
            contrast=self._contrast_fp[ckey],
            spec_tuple=fcache.spec_items(spec, frozenset({"note", "estimators"})),
            code_digest=digest, n_code=len(inventory),
            kw=dict(kw) if estimator == "REML" else None,
            boot_seed=boot_seed if spec.bootstrap_scheme == "hybrid" else None)
        if self._container_hash is None:
            self._container_hash = fcache.container_hash(self.data)
        meta = {"model_id": spec.model_id, "estimator": estimator,
                "im_tag": self.data.im_tag, "quantity": self.data.quantity,
                "k": con.k, "container_hash": self._container_hash,
                "versions": fcache.library_versions()}
        return fp, meta, inventory

    def _cached_fit_estimator(self, spec: ModelSpec, con: Contrast, estimator: str, *,
                              n_boot: int | None, verbose: bool, boot_seed: int,
                              refit: bool, **kw) -> ModelFit:
        """Load ``(spec, estimator)`` from the cache if valid, else fit it and save it."""
        fp, meta, inventory = self._cache_fingerprint(spec, con, estimator, kw, boot_seed)
        # The EFFECTIVE count: bootstrap_fits never runs more than the cloud has.
        want = None if n_boot is None else min(int(n_boot), con.k)

        def from_payload(p: dict) -> ModelFit:
            # Rebuilt through the constructor with the registry's OWN spec and
            # memoised contrast: nothing project-specific was pickled, a loaded
            # fit shares its contrast with every fresh fit, and every
            # cached_property starts empty.
            reps, boot = fcache.slice_replicates(p, want)
            return ModelFit(spec=spec, estimator=estimator, contrast=con,
                            beta=p["beta"], se_beta=p["se_beta"], tau2=p["tau2"],
                            het=p["het"], replicates=reps, x_names=p["x_names"],
                            scaling=p["scaling"], raw=p["raw"], boot=boot)

        return self._load_or_fit(
            (spec.model_id, estimator), self.cache_path(spec.model_id, estimator),
            fp, meta, inventory, want, refit,
            fit_fn=lambda: self._fit_estimator(spec, con, estimator, n_boot=n_boot,
                                               verbose=verbose, boot_seed=boot_seed, **kw),
            to_payload=_model_fit_payload, from_payload=from_payload)

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
        # The hybrid tag goes in the title because the CI band means something
        # different there (population of sites and designs, not these rows).
        ax.set_title(f"{model_id} [{fit_obj.estimator}] - {con.label}"
                     f"{'  ' + BC_TAG if fit_obj.corrected else ''}"
                     f"{'  ' + HB_TAG if fit_obj.hybrid else ''}")
        ax.set_ylim(-1, len(frame))
        mra.style_legend(ax, **{"loc": "best", "fontsize": 7, **(legend_kws or {})})
        return ax


# =============================================================================
# The joint theta / beta model (A8)
# =============================================================================
#
# WHY A JOINT MODEL
# -----------------
# At every row, theta_hat and beta_hat come out of ONE fit to ONE set of stripe
# counts (or capacities), so their sampling errors are correlated. Fitting
# A0c and A4c separately is fine for each quantity alone, but wrong for any
# statement about the PAIR -- and the MAFC propagation needs the pair: the
# correlation between the TRUE theta and beta corrections, with the sampling
# correlation taken out (model A8).
#
# HOW IT IS BUILT, AND WHY THIS WAY
# ---------------------------------
# FragilityData stays ONE quantity per container; that rule is what keeps the
# arm accessors free of a quantity axis (module docstring). So nothing here
# generalises the container. Instead two univariate Contrasts -- one from a
# theta container, one from a beta container -- are PAIRED into a
# JointContrast, and that pairing is the only place the two quantities meet.
#
#     JointContrast.from_pair      stacks (theta; beta), checks the two are
#                                  replicate-aligned, takes V from np.cov of
#                                  the stacked 240 x k replicate matrix
#     JointModelSpec / JOINT_MODEL_SPECS
#                                  one A8 rung per entry, same "one id, one
#                                  contrast" rules as MODEL_SPECS
#     mra.fit_reml_mv              the bivariate REML (pure arrays)
#     bootstrap_joint_fits         the outer bootstrap, column r of BOTH outcomes
#     JointModelFit                the result, with T_l, rho_l, rho_tot,
#                                  Sigma_pred and the MAFC draws
#     JointModelRegistry           fits keyed by (model_id, "REML"), refusing
#                                  overwrites, holding the two containers
#     check_joint_inputs           step 0 of the A8 plan
#
# None of the univariate classes is modified, so no univariate result can move.

OUTCOMES: tuple[str, ...] = ("theta", "beta")

# The tag a joint fit carries in place of BC_TAG: only beta is ever corrected
# (theta is corrected by zero in nb 070 section 2.2), so the tag says so.
JOINT_BC_TAG = "[beta bias-corrected; theta never]"


@dataclass(frozen=True)
class JointContrast:
    """One contrast for BOTH quantities, stacked theta-first, with the full joint covariance.

    Built only through :meth:`from_pair`, which is where the alignment checks
    live.

    Attributes
    ----------
    theta, beta : Contrast
        The two univariate contrasts, kept whole. Everything univariate (row
        labels, ``g``, ``site_codes``, the factor table, the covariates) is read
        from them rather than copied, so it cannot drift.
    y_hat : Series (2n,)
        Indexed ``(outcome, site, n_storeys)`` -- ``(outcome, design_group_id)``
        at group scope -- theta block first.
    y : DataFrame (2n, k)
        Same index; columns are the replicate labels SHARED by the two outcomes.
    V : ndarray (2n, 2n)
        The full sample covariance of ``y``. Its diagonal blocks are exactly
        ``theta.V`` and ``beta.V``; the off-diagonal block ``V_cross`` is the new
        information the joint model exists to use.
    """

    theta: Contrast
    beta: Contrast
    y_hat: pd.Series
    y: pd.DataFrame
    V: np.ndarray

    @classmethod
    def from_pair(cls, con_theta: Contrast, con_beta: Contrast) -> "JointContrast":
        """Stack two univariate contrasts into one joint contrast. Raises on misalignment.

        THE CHECK THAT MATTERS. Column r of ``con_theta.y`` and column r of
        ``con_beta.y`` must come from the SAME resample, or the cross-covariance
        computed below is the covariance of two unrelated draws -- i.e. zero up
        to noise -- and the joint model then silently reports no sampling
        correlation at all (task hard requirement 3). Labels are compared with
        ``Index.equals``, which checks values AND order. Nothing is reindexed to
        "fix" a mismatch: a silent fix is exactly the failure being guarded.

        Labels can agree while the values come from different runs; only the
        data can show that, and :func:`check_joint_inputs` tests it.

        theta is never bias-corrected, so ``con_theta.corrected`` must be False;
        a joint ``_bc`` fit corrects beta only (the registry arranges this).
        """
        problems = []
        if con_theta.quantity != "theta" or con_beta.quantity != "beta":
            problems.append(f"quantities are ({con_theta.quantity!r}, "
                            f"{con_beta.quantity!r}), expected ('theta', 'beta')")
        for attr in ("name", "scope", "terms"):
            if getattr(con_theta, attr) != getattr(con_beta, attr):
                problems.append(f"{attr} differs: {getattr(con_theta, attr)!r} vs "
                                f"{getattr(con_beta, attr)!r}")
        if not con_theta.y_hat.index.equals(con_beta.y_hat.index):
            problems.append("the row indexes differ (labels or order)")
        if not con_theta.y.columns.equals(con_beta.y.columns):
            problems.append(
                "the replicate columns differ in labels or order, so column r of "
                "theta and column r of beta are not the same resample and V_cross "
                "would be meaningless")
        if con_theta.corrected:
            problems.append("the theta contrast is bias-corrected, but theta is never "
                            "corrected (nb 070 section 2.2); build it with "
                            "corrected=False")
        if problems:
            raise ValueError("cannot pair these contrasts:\n  " + "\n  ".join(problems))

        # pd.concat with a dict prepends an "outcome" level and keeps the dict
        # order, so theta is the first block -- the order fit_reml_mv assumes.
        y_hat = pd.concat({"theta": con_theta.y_hat, "beta": con_beta.y_hat},
                          names=["outcome"])
        y = pd.concat({"theta": con_theta.y, "beta": con_beta.y}, names=["outcome"])
        # The full sample covariance of the STACKED replicates -- never assembled
        # from blocks, never given an assumed structure (CLAUDE.md section 2).
        V = mra.bootstrap_cov(y.to_numpy(dtype=float), rowvar=True)
        return cls(theta=con_theta, beta=con_beta, y_hat=y_hat, y=y, V=V)

    # -- shape and identity ---------------------------------------------------

    @property
    def n(self) -> int:
        """Rows PER OUTCOME (120 for the full study); the stacked length is 2n."""
        return self.theta.n

    @property
    def k(self) -> int:
        return self.y.shape[1]

    @property
    def name(self) -> str:
        return self.theta.name

    @property
    def label(self) -> str:
        return self.theta.label

    @property
    def scope(self) -> str:
        return self.theta.scope

    @property
    def terms(self) -> tuple[str, str]:
        return self.theta.terms

    @property
    def corrected(self) -> dict[str, bool]:
        """Per outcome, read from the two contrasts -- never stored twice."""
        return {"theta": self.theta.corrected, "beta": self.beta.corrected}

    # -- the blocks of V --------------------------------------------------------
    # Named V_theta / V_beta / V_cross, NOT V_tt / V_bb: FragilityData.V_bb
    # already means the MSA fixed-set covariance, and reusing the name for "the
    # beta block" would invite a silent mix-up.

    @property
    def V_theta(self) -> np.ndarray:
        """(n, n) theta-theta block. Equal to ``self.theta.V`` exactly."""
        return self.V[:self.n, :self.n]

    @property
    def V_beta(self) -> np.ndarray:
        """(n, n) beta-beta block. Equal to ``self.beta.V`` exactly."""
        return self.V[self.n:, self.n:]

    @property
    def V_cross(self) -> np.ndarray:
        """(n, n) theta-beta block: ``Cov_r(y_theta_i, y_beta_k)``.

        DENSE on the real data, not diagonal: rows sharing a design share the
        IDA arm's theta-beta covariance, and the two structures at a site share
        records in the SS arm. Its off-diagonal mass is comparable to its
        diagonal, so per-row correlations alone would badly misdescribe it.
        """
        return self.V[:self.n, self.n:]

    # -- sampling correlations ---------------------------------------------------

    @cached_property
    def r_samp(self) -> pd.Series:
        """(n,) per-row sampling correlation ``V_cross[i,i] / sqrt(V_theta[i,i] V_beta[i,i])``."""
        r = np.diag(self.V_cross) / np.sqrt(np.diag(self.V_theta) * np.diag(self.V_beta))
        return pd.Series(r, index=self.theta.y_hat.index, name="r_samp")

    @property
    def r_samp_bar(self) -> float:
        """Mean of :attr:`r_samp`. Report it with the median and range, which can differ a lot."""
        return float(self.r_samp.mean())

    @property
    def r_naive(self) -> float:
        """Pearson correlation of ``y_hat_theta`` and ``y_hat_beta`` across rows.

        The "naive" correlation: true correlation plus sampling correlation plus
        any shared-arm structure, all mixed. The joint model's rho is what is
        left when the sampling part is taken out.
        """
        return float(np.corrcoef(self.theta.y_hat.to_numpy(float),
                                 self.beta.y_hat.to_numpy(float))[0, 1])

    def summary(self) -> str:
        r = self.r_samp
        tag = f"  {JOINT_BC_TAG}" if self.corrected["beta"] else ""
        return (f"{self.name} = {self.terms[0]} - {self.terms[1]}  ({self.label}), "
                f"theta+beta, {self.scope} scope, n={self.n} per outcome, k={self.k}{tag}\n"
                f"  r_samp mean {r.mean():+.3f}, median {r.median():+.3f}, range "
                f"{r.min():+.3f} .. {r.max():+.3f};  r_naive {self.r_naive:+.3f}")


# -- the joint spec -------------------------------------------------------------

_JOINT_LEVELS: tuple[str, ...] = ("design", "site", "row")


@dataclass(frozen=True)
class JointModelSpec:
    """One A8 rung. The joint counterpart of :class:`ModelSpec`.

    A separate class rather than extra fields on ModelSpec, because every
    reader of ModelSpec (build_X, build_components, ModelFit, bootstrap_fits,
    the forest plot) has scalar semantics; defaulted joint fields there would
    have to be branched on or silently ignored everywhere.

    Attributes
    ----------
    model_id : str
        The roadmap id, e.g. ``"A8a"``. Disjoint from every MODEL_SPECS id
        (checked at import).
    contrast : str
        A key of ``_CONTRAST_SPECS``. The A8 rungs use ``"y0_avg"``.
    components : tuple of str
        Levels beyond the row level, from {"design", "site"}, as in ModelSpec.
        The row level is always present.
    correlated : tuple of str
        The levels whose theta-beta cross term is ESTIMATED, from {"design",
        "site", "row"}. At the other levels it is fixed at zero.
    covariates_theta, covariates_beta : tuple of str
        Fixed-effect covariates, set SEPARATELY per outcome (columns of the
        contrast factor table, centred and scaled exactly as build_X does).
    estimators : tuple of str
        REML only. DerSimonian-Laird has no bivariate form (it is one moment
        equation, and its Q presupposes a diagonal V), so asking for it raises.
    note : str
    corrected : bool
        Fit on bias-corrected BETA estimates. theta is never corrected. The
        same rules as ModelSpec apply: an id ends in ``_bc`` exactly when this
        is True, and a ``_bc`` spec matches its partner in every other field.
    bootstrap_scheme : {"record", "hybrid"}
        Which outer bootstrap :meth:`JointModelRegistry.fit` runs; keyword-only,
        ``"record"`` by default. The same rules as ModelSpec: an id ends in
        ``_hb`` exactly when this is ``"hybrid"``, and an ``_hb`` spec matches
        its partner in every field but ``model_id``, this one and ``note``.
        Every interval on a rho or a T should come from a hybrid fit; see
        :func:`bootstrap_joint_fits`.
    """

    model_id: str
    contrast: str = "y0_avg"
    components: tuple[str, ...] = ()
    correlated: tuple[str, ...] = ("row",)
    covariates_theta: tuple[str, ...] = ()
    covariates_beta: tuple[str, ...] = ()
    estimators: tuple[str, ...] = ("REML",)
    note: str = ""
    # LAST, as in ModelSpec. Always pass it by keyword.
    corrected: bool = False
    # After `corrected` and keyword-only, exactly as in ModelSpec.
    bootstrap_scheme: str = field(default="record", kw_only=True)

    def __post_init__(self):
        _check_bootstrap_scheme(self.model_id, self.bootstrap_scheme)
        if "DL" in self.estimators:
            raise ValueError(
                f"{self.model_id}: DerSimonian-Laird is undefined for a bivariate "
                "model -- it is a single moment equation for a single variance, "
                "and its Q statistic presupposes a diagonal sampling covariance. "
                "Joint specs are REML only.")
        if tuple(self.estimators) != ("REML",):
            raise ValueError(f"{self.model_id}: unknown estimator(s) {self.estimators}; "
                             "joint specs take ('REML',)")
        bad = sorted(set(self.components) - {"design", "site"})
        if bad:
            raise ValueError(f"{self.model_id}: unknown components {bad}")
        missing = sorted(set(self.correlated) - set(self.levels))
        if missing:
            raise ValueError(f"{self.model_id}: correlated level(s) {missing} are not "
                             f"in the model, whose levels are {self.levels}")

    @property
    def levels(self) -> tuple[str, ...]:
        """The variance levels, in the order the Gs and the T's are reported."""
        return (*self.components, "row")


JOINT_MODEL_SPECS: dict[str, JointModelSpec] = {
    "A8a": JointModelSpec(
        "A8a", "y0_avg", components=(), correlated=("row",), note="""
        Joint theta/beta version of A0b: one row-level 2x2 T_u, intercepts only.
        Gives rho_tot, the correlation between the TRUE theta and beta
        corrections with the sampling correlation (V_cross) taken out. Its
        marginal tau^2's should be close to the A0c totals for the univariate models; 
        with V_cross set to zero they equal the univariate single-level fits exactly, and
        with it they move a little, because each outcome's GLS weights now
        borrow from the other's sampling errors."""),
    
    "A8a_bc": JointModelSpec(
        "A8a_bc", "y0_avg", components=(), correlated=("row",), corrected=True,
        note="""A8a on bias-corrected beta estimates (theta is never corrected).
        See A8a and A0c_bc."""),
    
    "A8a_hb": JointModelSpec(
        "A8a_hb", "y0_avg", components=(), correlated=("row",),
        bootstrap_scheme="hybrid", note="""
        A8a with the HYBRID outer bootstrap: each replicate is X beta_hat plus
        new (theta, beta) row-effect pairs drawn from the fitted T_u, plus the
        record-resample deviations y^(r) - ybar (which carry V_cross). Use this
        fit, not A8a, for the interval on rho_tot and for mafc_draws(): the
        record-only replicates inflate every T by the sampling covariance,
        which pulls their rho towards the sampling correlation and makes the
        interval too narrow. Its CIs on the two summary effects are
        SUPERPOPULATION intervals; the intervals for these 120 rows are
        A8a's."""),

    "A8a_bc_hb": JointModelSpec(
        "A8a_bc_hb", "y0_avg", components=(), correlated=("row",), corrected=True,
        bootstrap_scheme="hybrid",note="""A8a on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8a_bc and A8a_hb."""),

    "A8b": JointModelSpec(
        "A8b", "y0_avg", components=("design", "site"),
        correlated=("row",), note="""
        The full crossed bivariate model: a 2x2 T at design, site and row level
        (9 parameters). Judged against the pre-set failure criteria. rho_d and
        rho_u are estimated from levels holding a small share of the variance
        and are expected to sit on the |rho| = 1 boundary; read their boundary
        fractions before reading their values."""),
    
    "A8b_bc": JointModelSpec(
        "A8b_bc", "y0_avg", components=("design", "site"),
        correlated=("row",), corrected=True,
        note="""A8b on bias-corrected beta estimates. See A8b."""),

    "A8b_hb": JointModelSpec(
        "A8b_hb", "y0_avg", components=("design", "site"),
        correlated=("row",), bootstrap_scheme="hybrid",
        note="""A8b on hybrid bootstrap estimates. See A8b."""),

    "A8b_bc_hb": JointModelSpec(
        "A8b_bc_hb", "y0_avg", components=("design", "site"),
        correlated=("row",), corrected=True,
        bootstrap_scheme="hybrid", note="""A8b on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8b_bc and A8b_hb."""),

    "A8c": JointModelSpec(
        "A8c", "y0_avg", components=("design", "site"),
        correlated=("design", "site", "row"), note="""
        The full crossed bivariate model: a 2x2 T at design, site and row level
        (9 parameters). Judged against the pre-set failure criteria. rho_d and
        rho_u are estimated from levels holding a small share of the variance
        and are expected to sit on the |rho| = 1 boundary; read their boundary
        fractions before reading their values."""),
    
    "A8c_bc": JointModelSpec(
        "A8c_bc", "y0_avg", components=("design", "site"),
        correlated=("design", "site", "row"), corrected=True,
        note="""A8c on bias-corrected beta estimates. See A8c."""),

    "A8c_hb": JointModelSpec(
        "A8c_hb", "y0_avg", components=("design", "site"),
        correlated=("design", "site", "row"), bootstrap_scheme="hybrid",
        note="""A8c on hybrid bootstrap estimates. See A8c."""),

    "A8c_bc_hb": JointModelSpec(
        "A8c_bc_hb", "y0_avg", components=("design", "site"),
        correlated=("design", "site", "row"), corrected=True,
        bootstrap_scheme="hybrid", note="""A8c on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8c_bc and A8c_hb."""),

    "A8d": JointModelSpec(
        "A8d", "y0_avg", components=("design", "site"),
        correlated=("site",), note="""
        The full crossed bivariate model: a 2x2 T at design, site and row level
        (9 parameters). Judged against the pre-set failure criteria. rho_d are
        rho_u are set to zero because their variance fraction is small and the rho
        values are likely to be unreliable anyway."""),
    
    "A8d_bc": JointModelSpec(
        "A8d_bc", "y0_avg", components=("design", "site"),
        correlated=("site",), corrected=True,
        note="""A8d on bias-corrected beta estimates. See A8d."""),

    "A8d_hb": JointModelSpec(
        "A8d_hb", "y0_avg", components=("design", "site"),
        correlated=("site",), bootstrap_scheme="hybrid",
        note="""A8d on hybrid bootstrap estimates. See A8d."""),

    "A8d_bc_hb": JointModelSpec(
        "A8d_bc_hb", "y0_avg", components=("design", "site"),
        correlated=("site",), corrected=True,
        bootstrap_scheme="hybrid", note="""A8d on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8d_bc and A8d_hb."""),

    "A8e": JointModelSpec(
        "A8e", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        correlated=("site",), note="""
        The full crossed bivariate model: a 2x2 T at design, site and row level
        (9 parameters). Judged against the pre-set failure criteria. rho_d are
        rho_u are set to zero because their variance fraction is small and the rho
        values are likely to be unreliable anyway."""),
    
    "A8e_bc": JointModelSpec(
        "A8e_bc", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        correlated=("site",), corrected=True,
        note="""A8d on bias-corrected beta estimates. See A8d."""),

    "A8e_hb": JointModelSpec(
        "A8e_hb", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        correlated=("site",), bootstrap_scheme="hybrid",
        note="""A8d on hybrid bootstrap estimates. See A8d."""),

    "A8e_bc_hb": JointModelSpec(
        "A8e_bc_hb", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_saratio_mean_unif", 
        "d_ln_ds595_mean_unif", 
        "d_ln_saratio_sigma_unif",
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        correlated=("site",), corrected=True,
        bootstrap_scheme="hybrid", note="""A8d on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8d_bc and A8d_hb."""),

####################################################################################
##### Joint Models with reduced covariates

    "A8f": JointModelSpec(
        "A8f", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), note="""
        The full crossed bivariate model: a 2x2 T at design, site and row level
        (9 parameters). Judged against the pre-set failure criteria. rho_d are
        rho_u are set to zero because their variance fraction is small and the rho
        values are likely to be unreliable anyway. All structural parameters + 
        the significant site parameters."""),
    
    "A8f_bc": JointModelSpec(
        "A8f_bc", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), corrected=True,
        note="""A8f on bias-corrected beta estimates. See A8f."""),

    "A8f_hb": JointModelSpec(
        "A8f_hb", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), bootstrap_scheme="hybrid",
        note="""A8f on hybrid bootstrap estimates. See A8f."""),

    "A8f_bc_hb": JointModelSpec(
        "A8f_bc_hb", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_T1", "ln_overstrength", "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), corrected=True,
        bootstrap_scheme="hybrid", note="""A8d on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8f_bc and A8f_hb."""),

    "A8g": JointModelSpec(
        "A8g", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), note="""
        The full crossed bivariate model: a 2x2 T at design, site and row level
        (9 parameters). Judged against the pre-set failure criteria. rho_d are
        rho_u are set to zero because their variance fraction is small and the rho
        values are likely to be unreliable anyway. All structural parameters + 
        the significant site parameters."""),
    
    "A8g_bc": JointModelSpec(
        "A8g_bc", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), corrected=True,
        note="""A8f on bias-corrected beta estimates. See A8f."""),

    "A8g_hb": JointModelSpec(
        "A8g_hb", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), bootstrap_scheme="hybrid",
        note="""A8f on hybrid bootstrap estimates. See A8f."""),

    "A8g_bc_hb": JointModelSpec(
        "A8g_bc_hb", "y0_avg", components=("design", "site"),
        covariates_theta=(
        "d_ln_ds595_mean_unif", 
        "d_ln_ds595_sigma_unif",
        "ln_ductility",),
        covariates_beta=(
        "d_ln_ds595_sigma_unif",),
        correlated=("site",), corrected=True,
        bootstrap_scheme="hybrid", note="""A8d on bias-corrected beta estimates 
        (theta is never corrected). Hybrid bootstrap. See A8f_bc and A8f_hb."""),
}


def _check_joint_model_specs(specs: dict[str, JointModelSpec],
                             univariate: dict[str, ModelSpec]) -> None:
    """Enforce "one id, one contrast" on the joint table. Raises ValueError.

    1. Every rule of :func:`_check_model_specs` (key == id, ``_bc`` <=>
       corrected, a ``_bc`` spec equal to its partner except in model_id,
       corrected and note).
    2. No id appears in both tables, so an id always names one fit whichever
       registry it is looked up in.
    3. Each covariate list is free of duplicates (a repeated column makes X
       singular, and the error would otherwise surface deep inside the fit).
    """
    _check_model_specs(specs)
    problems = []
    clash = sorted(set(specs) & set(univariate))
    if clash:
        problems.append(f"ids {clash} are in both MODEL_SPECS and JOINT_MODEL_SPECS")
    for key, spec in specs.items():
        for q in OUTCOMES:
            covs = getattr(spec, f"covariates_{q}")
            if len(set(covs)) != len(covs):
                problems.append(f"{key!r}: covariates_{q} has duplicates")
    if problems:
        raise ValueError("JOINT_MODEL_SPECS is inconsistent:\n  " + "\n  ".join(problems))


_check_joint_model_specs(JOINT_MODEL_SPECS, MODEL_SPECS)


# -- building the inputs to mra.fit_reml_mv -------------------------------------

def build_X_joint(spec: JointModelSpec, jcon: JointContrast
                  ) -> tuple[np.ndarray, list[tuple[str, str]], dict]:
    """Block-diagonal fixed-effects design ``X = blockdiag(X_theta, X_beta)``.

    Each block is built by :func:`build_X` itself (through a throw-away
    ModelSpec carrying that outcome's covariates), so the centring and scaling
    are IDENTICAL to the univariate fits -- which is what lets a joint slope be
    compared with an A3 slope at all.

    Returns
    -------
    X : (2n, p_theta + p_beta)
    x_names : list of (outcome, name), in the column order of X
    scaling : {outcome: {covariate: {"mean", "sd"}}}
    """
    blocks, x_names, scaling = [], [], {}
    for q in OUTCOMES:
        con = getattr(jcon, q)
        Xq, names, sc = build_X(
            ModelSpec(f"{spec.model_id}[{q}]", spec.contrast,
                      covariates=tuple(getattr(spec, f"covariates_{q}"))), con)
        blocks.append(Xq)
        x_names += [(q, nm) for nm in names]
        scaling[q] = sc
    n = jcon.n
    X = np.zeros((2 * n, sum(b.shape[1] for b in blocks)))
    X[:n, :blocks[0].shape[1]] = blocks[0]
    X[n:, blocks[0].shape[1]:] = blocks[1]
    return X, x_names, scaling


def joint_level_gs(spec: JointModelSpec, jcon: JointContrast) -> dict[str, np.ndarray]:
    """``{level: G}``, the UNIVARIATE n x n patterns, built the dsim way.

    Goes through :func:`build_components` and ``dsim.build_component_gs``, the
    same code every univariate REML rung uses, so a joint fit and a univariate
    fit of the same levels use literally the same Z Z', S S' and I. The
    Kronecker blocks are built inside ``mra.fit_reml_mv``.
    """
    comps = build_components(ModelSpec("_", spec.contrast, components=spec.components),
                             jcon.theta)
    Gs, names = dsim.build_component_gs(comps)
    return {_COMPONENT_NAMES[nm]: G for nm, G in zip(names, Gs)}


def fit_reml_joint(jcon: JointContrast, spec: JointModelSpec, **kw) -> tuple:
    """Full-sample bivariate REML fit. Returns ``(raw, x_names, scaling)``.

    Always the FULL joint V -- there is no diagonal option, because dropping
    V_cross is precisely the error the joint model exists to avoid.
    """
    X, x_names, scaling = build_X_joint(spec, jcon)
    raw = mra.fit_reml_mv(jcon.y_hat.to_numpy(dtype=float), X, jcon.V,
                          joint_level_gs(spec, jcon), spec.correlated, **kw)
    return raw, x_names, scaling


def _joint_column(outcome: str, name: str) -> str:
    """Replicate-frame label of one coefficient, e.g. ``beta_theta_intercept``."""
    return f"beta_{outcome}_{name}"


def joint_level_factors(spec: JointModelSpec, jcon: JointContrast) -> dict[str, np.ndarray]:
    """``{level: factor (n, K)}``: the indicator matrices behind :func:`joint_level_gs`.

    Built by the same :func:`build_components` call, so the hybrid bootstrap
    draws its effect pairs on exactly the grouping whose ``factor factor'`` the
    fit was given as G.
    """
    comps = build_components(ModelSpec("_", spec.contrast, components=spec.components),
                             jcon.theta)
    return {_COMPONENT_NAMES[nm]: np.asarray(factor, dtype=float)
            for nm, (factor, _) in comps.items()}


def bootstrap_joint_fits(jcon: JointContrast, spec: JointModelSpec, *,
                         n_boot: int | None = None,
                         T0: dict[str, np.ndarray] | None = None,
                         n_starts: int = 1, verbose: bool = False,
                         full_fit: dict | None = None,
                         boot_seed: int = BOOT_SEED,
                         **kw) -> pd.DataFrame:
    """Refit the joint model in every outer-bootstrap replicate. One row per replicate.

    Column r of ``jcon.y`` is the stacked 2n-vector of replicate r. Its theta
    and beta halves came from the SAME resample -- JointContrast.from_pair
    checked their labels -- so the sampling correlation between them is carried
    into the refits for free, just as the arm coherence is in bootstrap_fits.

    ``V_known``, the Gs and X are held FIXED at their full-sample values, as in
    :func:`bootstrap_fits` (re-estimating V per replicate needs a double
    bootstrap, Efron & Tibshirani Ch. 25).

    ``T0`` is the warm start, normally the full-sample optimum, with a single
    start per replicate: the full-sample fit has already explored the surface
    with its restarts, and each replicate perturbs it only slightly (the
    argument of bootstrap_fits). fit_reml_mv floors a warm start away from
    every boundary, so a component at zero in the full sample is still free to
    move in each replicate.

    Schemes -- ``spec.bootstrap_scheme`` chooses what is refitted
    ------------------------------------------------------------
    The argument is the one in :func:`bootstrap_fits`, and it bites harder here.

    ``"record"`` (the default)
        Column r of ``jcon.y``. Right for ``m_theta`` / ``m_beta`` and their
        intervals, conditional on these rows. Every replicate T is inflated by
        the sampling covariance (two rounds of noise, one accounted for), so
        the replicate rho's are the correlation of ``T + (sampling part)``:
        pulled towards the sampling correlation, and far too tightly spread.
        On A8a the interval for rho_tot came out about 2.4x too narrow.
    ``"hybrid"``
        ``dsim.hybrid_pseudo_data_mv``: ``X beta_hat`` plus NEW (theta, beta)
        effect pairs drawn from the fitted T of each level, plus the stacked
        record-resample deviations ``y^(r) - ybar``. The deviations carry the
        whole joint sampling covariance, V_cross included, without modelling
        it. The replicate T's then centre on the fitted ones, and
        :meth:`JointModelFit.rho_ci`, ``boundary_fractions`` and
        ``mafc_draws`` mean what they say. The m intervals become
        superpopulation ones. Needs ``full_fit`` (the full-sample ``raw``
        dict). A singular fitted T (|rho| = 1, or a variance at zero) is drawn
        through ``dsim._psd_root``: the pairs lie on a line, or one of them is
        zero, and replicates then pile up on that boundary -- read the
        boundary fractions first.

    ``boot_seed`` seeds the effect draws; same spec, data and seed give
    identical replicates, and replicate r does not depend on ``n_boot``. The
    frame's ``attrs`` record ``bootstrap_scheme`` and ``seed``.

    Columns
    -------
    ``m_theta``, ``m_beta``           the two intercepts
    ``beta_<outcome>_<name>``         every coefficient
    ``T_<level>_tt/_bb/_tb``          every T entry, and ``T_total_tt/_bb/_tb``
    ``rho_<level>``, ``rho_tot``      NaN where undefined (a variance at 0)
    ``at_zero_<level>_theta/_beta``, ``rho_at_bound_<level>``
    ``converged``, ``nll``
    """
    columns = list(jcon.y.columns)
    if n_boot is not None:
        columns = columns[:n_boot]
    X, x_names, _ = build_X_joint(spec, jcon)
    Gs = joint_level_gs(spec, jcon)
    Y = jcon.y.to_numpy(dtype=float)

    # Hybrid: swap the replicate matrix for the pseudo-data. Everything after
    # this block is shared, so the two schemes store identical columns. The
    # record-only path does not enter the block and is exactly what it was.
    hybrid = spec.bootstrap_scheme == "hybrid"
    if hybrid:
        if full_fit is None:
            raise ValueError(
                f"{spec.model_id}: the hybrid bootstrap needs the full-sample fit "
                "(full_fit=raw from mra.fit_reml_mv), because its pseudo-data are "
                "X beta_hat plus effect pairs drawn from the fitted T's. "
                "JointModelRegistry.fit passes it.")
        factors = joint_level_factors(spec, jcon)
        # The SNAPPED T's of the full-sample fit: the ones its reported Sigma
        # and beta were computed with.
        Y = dsim.hybrid_pseudo_data_mv(
            Y, X @ np.asarray(full_fit["beta"], dtype=float),
            {lev: (factors[lev], full_fit["T"][lev]) for lev in full_fit["levels"]},
            seed=boot_seed, n_boot=len(columns))

    out = []
    report_every = max(len(columns) // 10, 1)
    for j, col in enumerate(columns):
        raw = mra.fit_reml_mv(Y[:, j], X, jcon.V, Gs, spec.correlated,
                              n_starts=n_starts, T0=T0, **kw)
        row = {_joint_column(q, nm): float(b) for (q, nm), b in zip(x_names, raw["beta"])}
        row = {"m_theta": row[_joint_column("theta", "intercept")],
               "m_beta": row[_joint_column("beta", "intercept")], **row}
        for lev, T in [*raw["T"].items(), ("total", raw["T_total"])]:
            row |= {f"T_{lev}_tt": float(T[0, 0]), f"T_{lev}_bb": float(T[1, 1]),
                    f"T_{lev}_tb": float(T[0, 1])}
        for lev in raw["levels"]:
            row[f"rho_{lev}"] = raw["rho"][lev]
            row[f"at_zero_{lev}_theta"], row[f"at_zero_{lev}_beta"] = raw["at_zero"][lev]
            row[f"rho_at_bound_{lev}"] = raw["rho_at_bound"][lev]
        row |= {"rho_tot": raw["rho_tot"], "converged": raw["converged"],
                "nll": raw["nll"]}
        out.append(row)
        if verbose and (j + 1) % report_every == 0:
            print(f"  {spec.model_id}: {j + 1}/{len(columns)} replicates", flush=True)
    frame = pd.DataFrame(out, index=pd.Index(columns, name="k"))
    # Metadata only (see bootstrap_fits): no value changes.
    frame.attrs["bootstrap_scheme"] = spec.bootstrap_scheme
    frame.attrs["seed"] = boot_seed if hybrid else None
    return frame


# -- the fit object ---------------------------------------------------------------

@dataclass(frozen=True)
class JointModelFit:
    """One fitted A8 rung. The joint counterpart of :class:`ModelFit`.

    A separate class because ModelFit's accessors are scalar throughout
    (``m`` is ``beta[0]``, ``T_sq`` a float, the PI a pair of numbers).

    Attributes
    ----------
    spec : JointModelSpec
    contrast : JointContrast
    raw : dict
        The untouched return value of ``mra.fit_reml_mv``.
    x_names : list of (outcome, name)
    scaling : {outcome: {covariate: {"mean", "sd"}}}
    replicates : DataFrame or None
        From :func:`bootstrap_joint_fits`.
    estimator : str
        Always "REML"; kept so the registry key has the same shape as
        ModelRegistry's.
    boot : dict
        How the outer bootstrap was run: ``{"scheme", "seed", "zero_levels"}``,
        as on :class:`ModelFit`. ``zero_levels`` lists ``(level, outcome)``
        pairs whose variance was zero in the full-sample fit, where the hybrid
        scheme drew no effects.
    """

    spec: JointModelSpec
    contrast: JointContrast
    raw: dict
    x_names: list
    scaling: dict
    replicates: pd.DataFrame | None = None
    estimator: str = "REML"
    boot: dict = field(default_factory=dict)

    @property
    def boot_scheme(self) -> str | None:
        """``"record"`` or ``"hybrid"``; None without an outer bootstrap.

        Read from what was run (:attr:`boot`), not from the spec. A fit built
        by hand with no ``boot`` reads as record-only, the only scheme that
        existed before the field did.
        """
        if self.replicates is None:
            return None
        return self.boot.get("scheme", "record")

    @property
    def hybrid(self) -> bool:
        """True when the replicates come from the hybrid bootstrap."""
        return self.boot_scheme == "hybrid"

    # -- identity -------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def quantity(self) -> str:
        """``"theta+beta"``, so a joint row can never pass for a univariate one."""
        return "theta+beta"

    @property
    def corrected(self) -> dict[str, bool]:
        return self.contrast.corrected

    @property
    def n(self) -> int:
        return self.contrast.n

    # -- fixed effects --------------------------------------------------------

    @property
    def beta(self) -> np.ndarray:
        return np.asarray(self.raw["beta"])

    @property
    def vcov_beta(self) -> np.ndarray:
        """Model-based covariance of every coefficient (superpopulation: includes the T's)."""
        return np.asarray(self.raw["vcov_beta"])

    def _pos(self, outcome: str, name: str = "intercept") -> int:
        return self.x_names.index((outcome, name))

    @property
    def m(self) -> np.ndarray:
        """(2,) the two intercepts ``(m_theta, m_beta)``, log scale.

        Covariates are centred, so with covariates these are the corrections at
        the covariate MEANS.
        """
        return self.beta[[self._pos("theta"), self._pos("beta")]]

    def ratio(self) -> np.ndarray:
        """``exp(m)``: the bivariate correction factor (theta, beta)."""
        return np.exp(self.m)

    @property
    def vcov_m(self) -> np.ndarray:
        """(2, 2) MODEL-BASED covariance of ``m``. See :attr:`vcov_m_boot`."""
        idx = [self._pos("theta"), self._pos("beta")]
        return self.vcov_beta[np.ix_(idx, idx)]

    @cached_property
    def vcov_beta_boot(self) -> np.ndarray | None:
        """Bootstrap covariance of every coefficient, or None without replicates.

        Finite-population, like ModelFit.se_boot: records resampled, these 60
        sites and 51 designs held fixed. That is the record-only scheme; on a
        hybrid fit the effects are redrawn too, so this is the SUPERPOPULATION
        covariance, comparable with :attr:`vcov_beta`.
        """
        if self.replicates is None:
            return None
        cols = [_joint_column(q, nm) for q, nm in self.x_names]
        return np.atleast_2d(np.cov(self.replicates[cols].to_numpy(float), rowvar=False))

    @property
    def vcov_m_boot(self) -> np.ndarray | None:
        """(2, 2) bootstrap covariance of ``(m_theta, m_beta)``."""
        V = self.vcov_beta_boot
        if V is None:
            return None
        idx = [self._pos("theta"), self._pos("beta")]
        return V[np.ix_(idx, idx)]

    def gamma(self, outcome: str) -> pd.Series:
        """The slopes of one outcome, per SD of each (centred, scaled) covariate.

        Named gamma, not beta, because beta is the dispersion.
        """
        names = [nm for q, nm in self.x_names if q == outcome and nm != "intercept"]
        return pd.Series([self.beta[self._pos(outcome, nm)] for nm in names],
                         index=pd.Index(names, name="covariate"), name=f"gamma_{outcome}",
                         dtype=float)

    def ci(self, outcome: str, alpha: float = 0.05) -> tuple[float, float]:
        """Percentile bootstrap CI for ``m_<outcome>``, log scale.

        Record-only scheme: where the mean lies for THESE sites and designs.
        Hybrid scheme: where it lies for the population they were drawn from
        (wider). See :meth:`ModelFit.ci`.
        """
        if self.replicates is None:
            raise ValueError(f"{self.model_id} was fitted without an outer bootstrap - "
                             "pass n_boot to get a confidence interval")
        lo, hi = self.replicates[f"m_{outcome}"].quantile([alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    def coef_table(self, alpha: float = 0.05) -> pd.DataFrame:
        """One row per coefficient: estimate, model SE, bootstrap SE and percentile CI."""
        idx = pd.MultiIndex.from_tuples(self.x_names, names=["outcome", "coefficient"])
        tbl = pd.DataFrame({"estimate": self.beta,
                            "se_model": np.sqrt(np.diag(self.vcov_beta))}, index=idx)
        tbl["se_boot"], tbl["ci_lo"], tbl["ci_hi"] = np.nan, np.nan, np.nan
        if self.replicates is not None:
            for q, nm in self.x_names:
                s = self.replicates[_joint_column(q, nm)]
                lo, hi = s.quantile([alpha / 2, 1 - alpha / 2])
                tbl.loc[(q, nm), ["se_boot", "ci_lo", "ci_hi"]] = [s.std(ddof=1), lo, hi]
        return tbl

    # -- variance components and correlations -------------------------------

    @property
    def levels(self) -> list[str]:
        return list(self.raw["levels"])

    @property
    def T(self) -> dict[str, np.ndarray]:
        """``{level: 2x2}``, each PSD by construction (Cholesky parameterisation)."""
        return self.raw["T"]

    @property
    def T_total(self) -> np.ndarray:
        """``sum_l T_l``: the total between-row covariance of the true corrections."""
        return self.raw["T_total"]

    @property
    def rho(self) -> dict[str, float]:
        """Per-level correlation. NaN where undefined (a variance at zero).

        A level NOT in ``spec.correlated`` reports 0.0 because its cross term is
        FIXED at zero by the model; the summary says "(fixed)" beside it.
        """
        return self.raw["rho"]

    @property
    def rho_tot(self) -> float:
        """Correlation of the summed T's -- the overall true-correction correlation."""
        return self.raw["rho_tot"]

    # -- prediction -------------------------------------------------------------

    def predictive(self, x0_theta: dict | None = None, x0_beta: dict | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and 2x2 covariance of the (theta, beta) correction for a NEW structure.

        ::

            mean  = A beta_hat
            Sigma = T_total + A Var(beta_hat) A'

        where A (2 x p) holds the new structure's covariate rows -- block-
        diagonal, the theta row in the theta columns, the beta row in the beta
        columns -- centred and scaled with the fit's own ``scaling``.
        ``x0_theta`` / ``x0_beta`` map covariate name -> value in NATURAL
        units; a covariate left out sits at its mean. With no covariates (or
        none given) A picks out the intercepts and Sigma = T_total + Var(m).

        Var(beta_hat) is the BOOTSTRAP covariance when replicates exist and the
        model-based one otherwise (with a warning), the same preference as
        :meth:`ModelFit.prediction_interval`.

        A new structure has its own effects, so the Var(beta_hat) that belongs
        here is the SUPERPOPULATION one. The hybrid bootstrap gives exactly
        that. The record-only bootstrap gives ``W V W'``, which omits the
        ``W (sum_l T_l kron G_l) W'`` part, so on a record-only fit this Sigma
        is slightly too small (T_total dominates, so the effect is of the
        order of 1%). Left as it is by decision (2026-10-13); use the ``_hb``
        fit for a number that has to be right.
        """
        p = len(self.beta)
        A = np.zeros((2, p))
        for row, (q, x0) in enumerate((("theta", x0_theta), ("beta", x0_beta))):
            A[row, self._pos(q)] = 1.0
            for cov, val in (x0 or {}).items():
                if (q, cov) not in self.x_names:
                    raise KeyError(f"{cov!r} is not a {q} covariate of {self.model_id}")
                sc = self.scaling[q][cov]
                A[row, self._pos(q, cov)] = (float(val) - sc["mean"]) / sc["sd"]
        V = self.vcov_beta_boot
        if V is None:
            V = self.vcov_beta
            warnings.warn(f"{self.model_id}: no outer bootstrap, so Sigma_pred uses the "
                          "model-based Var(beta). Pass n_boot.", RuntimeWarning,
                          stacklevel=2)
        return A @ self.beta, self.T_total + A @ V @ A.T

    @property
    def Sigma_pred(self) -> np.ndarray:
        """2x2 predictive covariance at the covariate means: ``T_total + Var(m)``.

        For sampling ONE correction pair for a new building from
        ``N(m, Sigma_pred)``. For a hierarchical draw over bootstrap replicates
        use :meth:`mafc_draws` instead -- see its docstring for why Var(m) must
        not be added there.
        """
        return self.predictive()[1]

    # -- diagnostics passed through from the contrast -----------------------

    @property
    def r_samp(self) -> pd.Series:
        return self.contrast.r_samp

    @property
    def r_samp_bar(self) -> float:
        return self.contrast.r_samp_bar

    @property
    def r_naive(self) -> float:
        return self.contrast.r_naive

    # -- covariates: explained vs residual covariance ---------------------------

    def explained_covariance(self) -> dict[str, pd.DataFrame | float]:
        """Split the total between-row covariance into covariate-explained and residual.

        With covariates, the true correction of row i is
        ``x_i' gamma + (random effects)``, so across rows::

            Cov(true theta, true beta) = gamma_theta' S_x gamma_beta + T_total
                                         \\__ explained ________/   \\_ residual

        where S_x is the sample covariance of the covariates. The two outcomes
        may use different sets, so S_x is taken over the UNION of the two sets
        and each slope vector is zero-padded to it (a covariate absent from an
        outcome has slope 0 there). Slopes are per SD and S_x is computed on the
        same standardised columns, so the units match. rho_res is the residual
        correlation; that is the number the MAFC propagation uses.

        Returns ``{"explained", "residual", "total"}`` as labelled 2x2 frames,
        plus ``"share_cov_explained"`` and the three correlations
        ``rho_explained``, ``rho_res`` and ``rho_total``.
        """
        g_t, g_b = self.gamma("theta"), self.gamma("beta")
        union = list(dict.fromkeys([*g_t.index, *g_b.index]))
        if not union:
            raise ValueError(f"{self.model_id} has no covariates - nothing is explained")
        rows = self.contrast.theta.rows
        Zx = np.column_stack([(rows[c].to_numpy(float) - rows[c].mean())
                              / rows[c].std(ddof=1) for c in union])
        S_x = np.atleast_2d(np.cov(Zx, rowvar=False))
        G = np.column_stack([g_t.reindex(union).fillna(0.0).to_numpy(),
                             g_b.reindex(union).fillna(0.0).to_numpy()])
        explained = G.T @ S_x @ G
        residual = self.T_total
        total = explained + residual

        def corr(M):
            d = np.diag(M)
            return float(M[0, 1] / np.sqrt(d[0] * d[1])) if np.all(d > 0) else np.nan

        lab = list(OUTCOMES)
        return {"explained": pd.DataFrame(explained, index=lab, columns=lab),
                "residual": pd.DataFrame(residual, index=lab, columns=lab),
                "total": pd.DataFrame(total, index=lab, columns=lab),
                "share_cov_explained": (float(explained[0, 1] / total[0, 1])
                                        if total[0, 1] != 0 else np.nan),
                "rho_explained": corr(explained), "rho_res": corr(residual),
                "rho_total": corr(total)}

    # -- bootstrap inference on rho ------------------------------------------------

    def rho_ci(self, which: str = "tot", alpha: float = 0.05,
               method: str = "percentile") -> dict:
        """Bootstrap interval for one correlation: a level name or ``"tot"``.

        method="percentile"
            The (alpha/2, 1-alpha/2) quantiles of the defined replicates.
            Note: a percentile interval computed on Fisher's z = atanh(rho) and
            mapped back with tanh is IDENTICAL to this one, because percentiles
            respect monotone transforms (Efron & Tibshirani Ch. 13, ~pp. 170-174,
            approximate). So that is not offered as a separate "Fisher" method.
        method="fisher"
            The normal-theory interval on z::

                tanh( atanh(rho_hat) +/- z_{alpha/2} * sd_r(atanh rho*) )

            Replicates at |rho| = 1 have z = +/- infinity; they are clipped to
            +/-(1 - 1e-6) first and the clipped fraction is reported. When that
            fraction is large the interval is not meaningful -- read the
            percentile interval and the boundary fraction instead.

        Replicates where rho is undefined (a variance at 0) are left out of both,
        and that fraction is reported too.

        WHICH SCHEME (the returned ``"scheme"`` says). The interval is only
        sound on a HYBRID fit (an ``_hb`` id), whose replicate T's centre on
        the fitted ones. On a RECORD-ONLY fit it is provisional: those
        replicates inflate every T by the sampling covariance, so the
        replicate rho's are the rho of ``T + (sampling part)``. On A8a that
        centres the interval near 0 instead of on the estimate and makes it
        about 2.4x too narrow. See :func:`bootstrap_joint_fits`.

        Even on a hybrid fit, a level whose fitted T is singular (|rho| = 1 or
        a variance at zero) is drawn singular, and its replicates pile up on
        that boundary; read :meth:`boundary_fractions` first.

        Returns ``{"lo", "hi", "estimate", "n_used", "frac_undefined",
        "frac_clipped", "scheme"}``.
        """
        if self.replicates is None:
            raise ValueError(f"{self.model_id}: no outer bootstrap - pass n_boot")
        col = "rho_tot" if which == "tot" else f"rho_{which}"
        if col not in self.replicates:
            raise KeyError(f"no {which!r} correlation in {self.model_id}; levels are "
                           f"{self.levels}")
        est = self.rho_tot if which == "tot" else self.rho[which]
        vals = self.replicates[col].to_numpy(float)
        defined = vals[np.isfinite(vals)]
        out = {"estimate": est, "n_used": int(defined.size),
               "frac_undefined": float(1 - defined.size / vals.size),
               "frac_clipped": np.nan, "lo": np.nan, "hi": np.nan,
               "scheme": self.boot_scheme}
        if defined.size < 2 or not np.isfinite(est):
            return out
        if method == "percentile":
            out["lo"], out["hi"] = (float(v) for v in
                                    np.quantile(defined, [alpha / 2, 1 - alpha / 2]))
            out["frac_clipped"] = 0.0
        elif method == "fisher":
            lim = 1 - 1e-6
            out["frac_clipped"] = float(np.mean(np.abs(defined) > lim))
            z = np.arctanh(np.clip(defined, -lim, lim))
            centre = np.arctanh(np.clip(est, -lim, lim))
            half = stats.norm.ppf(1 - alpha / 2) * np.std(z, ddof=1)
            out["lo"], out["hi"] = float(np.tanh(centre - half)), float(np.tanh(centre + half))
        else:
            raise ValueError(f"method must be 'percentile' or 'fisher', got {method!r}")
        return out

    def boundary_fractions(self) -> pd.DataFrame:
        """Per level, the fraction of replicates on each kind of boundary.

        Columns: ``theta_at_zero``, ``beta_at_zero``, ``rho_undefined`` (either
        variance at zero) and ``rho_at_bound`` (|rho| within rho_bound_tol of 1).
        """
        if self.replicates is None:
            raise ValueError(f"{self.model_id}: no outer bootstrap - pass n_boot")
        r = self.replicates
        return pd.DataFrame({
            lev: {"theta_at_zero": r[f"at_zero_{lev}_theta"].mean(),
                  "beta_at_zero": r[f"at_zero_{lev}_beta"].mean(),
                  "rho_undefined": r[f"rho_{lev}"].isna().mean(),
                  "rho_at_bound": r[f"rho_at_bound_{lev}"].mean()}
            for lev in self.levels}).T.rename_axis("level")

    def mafc_draws(self) -> pd.DataFrame:
        """Per-replicate ``(m^(r), T_total^(r))``, plus slopes -- the input to the MAFC step.

        One row per outer-bootstrap replicate: ``m_theta``, ``m_beta``,
        ``T_total_tt``, ``T_total_bb``, ``T_total_tb`` and every slope.

        HOW TO USE IT, and why it carries T_total and not Sigma_pred. A
        hierarchical draw -- pick a replicate r, then draw a correction pair
        from ``N(m^(r), T_total^(r))`` -- already contains Var(m): it IS the
        spread of m^(r) across r. Drawing from ``N(m^(r), Sigma_pred^(r))``
        instead would count Var(m) twice. For a single-distribution draw use
        ``N(fit.m, fit.Sigma_pred)``.

        WHICH SCHEME -- check ``draws.attrs["bootstrap_scheme"]``.

        HYBRID fit (an ``_hb`` id): use the pairs as they are. Each row is one
        refit of pseudo-data that redraws the effects, so ``m^(r)`` carries the
        superpopulation uncertainty in the mean, ``T_total^(r)`` is a fair draw
        of the between-row covariance, and the two are JOINT draws (same
        refit). This is the frame the MAFC propagation should read.

        RECORD-ONLY fit: the caveat found 2026-10-08 applies. The replicates
        resample RECORDS only, so each one's data are y_hat plus a second dose
        of sampling noise while the fit is told about one (V_known), and every
        replicate T is inflated by roughly the mean sampling variance. On
        y0_avg the beta variance comes out about 4x too large (replicate mean
        0.031 against 0.0093). There, pair m^(r) with the full-sample
        ``fit.T_total`` -- and note that m^(r) then carries only the
        uncertainty for these rows, which is too little for a new building.
        """
        if self.replicates is None:
            raise ValueError(f"{self.model_id}: no outer bootstrap - pass n_boot")
        slopes = [_joint_column(q, nm) for q, nm in self.x_names if nm != "intercept"]
        draws = self.replicates[["m_theta", "m_beta", "T_total_tt", "T_total_bb",
                                 "T_total_tb", *slopes]].copy()
        # Set from the FIT, not inherited from the frame: pandas does not
        # promise to carry attrs through every operation, and this frame is
        # the one that leaves the module for the MAFC step.
        draws.attrs["bootstrap_scheme"] = self.boot_scheme
        draws.attrs["seed"] = self.boot.get("seed")
        return draws

    # -- reporting -----------------------------------------------------------------

    def to_row(self) -> dict:
        """One row of :meth:`JointModelRegistry.table`."""
        row = {"model": self.model_id, "estimator": self.estimator,
               "quantity": self.quantity, "contrast": self.contrast.name,
               "corrected": self.corrected["beta"],
               # Which outer bootstrap the se_boot / ci / rho_tot_ci columns
               # come from; None if none was run. See ModelFit.to_row.
               "bootstrap": self.boot_scheme, "n": self.n}
        se_model = np.sqrt(np.diag(self.vcov_m))
        se_boot = (np.sqrt(np.diag(self.vcov_m_boot)) if self.vcov_m_boot is not None
                   else [np.nan, np.nan])
        for k, q in enumerate(OUTCOMES):
            row |= {f"ratio_{q}": float(np.exp(self.m[k])), f"m_{q}": float(self.m[k]),
                    f"se_model_{q}": float(se_model[k]), f"se_boot_{q}": float(se_boot[k])}
            if self.replicates is not None:
                lo, hi = self.ci(q)
                row |= {f"ci_lo_{q}": np.exp(lo), f"ci_hi_{q}": np.exp(hi)}
        for lev in self.levels:
            row |= {f"tau_{lev}_theta": float(np.sqrt(self.T[lev][0, 0])),
                    f"tau_{lev}_beta": float(np.sqrt(self.T[lev][1, 1]))}
            if lev in self.spec.correlated:
                row[f"rho_{lev}"] = self.rho[lev]
        row["rho_tot"] = self.rho_tot
        if self.replicates is not None:
            ci = self.rho_ci("tot")
            row |= {"rho_tot_ci_lo": ci["lo"], "rho_tot_ci_hi": ci["hi"]}
            bf = self.boundary_fractions()
            for lev in self.spec.correlated:
                row[f"frac_bound_rho_{lev}"] = float(bf.loc[lev, "rho_at_bound"]
                                                     + bf.loc[lev, "rho_undefined"])
        row |= {"r_samp_bar": self.r_samp_bar, "r_naive": self.r_naive,
                "converged": self.raw["converged"]}
        return row

    def summary(self) -> str:
        """Multi-line report: per outcome, then the correlation block, then the note."""
        tag = f"  {JOINT_BC_TAG}" if self.corrected["beta"] else ""
        if self.hybrid:
            tag += f"  {HB_TAG}"
        con = self.contrast
        out = [f"{self.model_id} [{self.estimator}]{tag}  {con.label}  ({con.name} = "
               f"{con.terms[0]} - {con.terms[1]}, theta+beta, n={self.n} per outcome)"]
        se_model = np.sqrt(np.diag(self.vcov_m))
        for k, q in enumerate(OUTCOMES):
            line = (f"  {q:<5s} summary effect : {np.exp(self.m[k]):.4f}   "
                    f"(log {self.m[k]:+.4f}, SE superpop. {se_model[k]:.5f}")
            if self.vcov_m_boot is not None:
                line += f", SE boot {np.sqrt(self.vcov_m_boot[k, k]):.5f}"
            out.append(line + ")")
            if self.replicates is not None:
                lo, hi = self.ci(q)
                out.append(f"        95% CI         : {np.exp(lo):.4f} - {np.exp(hi):.4f}"
                           + ("   <- for the POPULATION of sites and designs "
                              "(effects redrawn)" if self.hybrid else ""))
        V_m = self.vcov_m_boot if self.vcov_m_boot is not None else self.vcov_m
        out.append(f"  corr(m_theta, m_beta) : {V_m[0, 1] / np.sqrt(V_m[0, 0] * V_m[1, 1]):+.3f}"
                   f"   [{'bootstrap' if self.vcov_m_boot is not None else 'model-based'}]")

        bf = self.boundary_fractions() if self.replicates is not None else None
        out.append("  between-row covariance, per level (tau as SD; rho = true-correction corr.):")
        for lev in self.levels:
            T, (sh_t, sh_b) = self.T[lev], self.raw["share"][lev]
            z_t, z_b = self.raw["at_zero"][lev]
            line = (f"    {lev:<6s}  tau_theta {np.sqrt(T[0, 0]):.4f}{' (0)' if z_t else ''}"
                    f" [{sh_t:5.1%}]  tau_beta {np.sqrt(T[1, 1]):.4f}{' (0)' if z_b else ''}"
                    f" [{sh_b:5.1%}]  ")
            if lev not in self.spec.correlated:
                line += "rho fixed at 0"
            elif not np.isfinite(self.rho[lev]):
                line += "rho UNDEFINED (a variance is 0)"
            else:
                line += f"rho {self.rho[lev]:+.3f}"
                if self.raw["rho_at_bound"][lev]:
                    line += "  <- ON THE |rho| = 1 BOUNDARY"
                elif min(sh_t, sh_b) < 0.05:
                    line += "  <- weakly identified (level holds < 5% of a variance)"
                if bf is not None:
                    line += (f"   boot: {bf.loc[lev, 'rho_at_bound']:.0%} at bound, "
                             f"{bf.loc[lev, 'rho_undefined']:.0%} undefined")
            out.append(line)
        line = f"  rho_tot          : {self.rho_tot:+.3f}"
        if self.replicates is not None:
            p, f = self.rho_ci("tot"), self.rho_ci("tot", method="fisher")
            line += (f"   95% CI {p['lo']:+.3f} .. {p['hi']:+.3f} (percentile), "
                     f"{f['lo']:+.3f} .. {f['hi']:+.3f} (Fisher z)")
        out.append(line)
        if self.replicates is not None and not self.hybrid:
            # Only under the record-only scheme: the hybrid replicates are
            # built so that their T's centre on the fitted ones.
            out.append("    -> PROVISIONAL: record-only replicates inflate T by the sampling "
                       "covariance, which biases these intervals (see JointModelFit.rho_ci)")
        if self.hybrid:
            r = self.replicates
            out.append("    -> hybrid replicates: mean T_total "
                       f"tt {r['T_total_tt'].mean():.5f}, bb {r['T_total_bb'].mean():.5f}, "
                       f"tb {r['T_total_tb'].mean():+.5f}   against the fitted "
                       f"{self.T_total[0, 0]:.5f}, {self.T_total[1, 1]:.5f}, "
                       f"{self.T_total[0, 1]:+.5f} (they should agree, unless a "
                       "level sits on a boundary)")
            zero = self.boot.get("zero_levels", ())
            if zero:
                out.append("    -> fitted at 0, so drawn at 0: "
                           + ", ".join(f"{lev} {q}" for lev, q in zero)
                           + ". Replicates there can only move UP; expect the replicate "
                           "T_total above the fitted one by that much.")
        r = self.r_samp
        out.append(f"  r_samp (per row) : mean {r.mean():+.3f}, median {r.median():+.3f}, "
                   f"range {r.min():+.3f} .. {r.max():+.3f}   <- SAMPLING correlation")
        out.append(f"  r_naive          : {self.r_naive:+.3f}   <- corr of the observed "
                   "y_theta, y_beta: true + sampling, mixed")
        if any(nm != "intercept" for _, nm in self.x_names):
            ec = self.explained_covariance()
            # Covariances rather than "share explained": when the total
            # cross-covariance is near zero the share is a ratio of two small
            # numbers and can read +350 %, which says nothing useful.
            out.append(f"  covariates       : rho_res {ec['rho_res']:+.3f}, rho_explained "
                       f"{ec['rho_explained']:+.3f}, rho_total {ec['rho_total']:+.3f}; "
                       f"theta-beta cov explained {ec['explained'].iloc[0, 1]:+.2e}, "
                       f"residual {ec['residual'].iloc[0, 1]:+.2e}, total "
                       f"{ec['total'].iloc[0, 1]:+.2e}")
            out.append(self.coef_table().to_string(float_format=lambda v: f"{v:+.5f}"))
        out.append(f"  converged {self.raw['converged']}, start spread "
                   f"{self.raw['start_spread']:.1e}, nll {self.raw['nll']:.4f}")
        if self.spec.note:
            out.append("  " + " ".join(self.spec.note.split()))
        return "\n".join(out)


# -- the registry -----------------------------------------------------------------

class JointModelRegistry(fcache.FitCacheMixin):
    """Every joint (A8) fit for one pair of containers, keyed ``(model_id, "REML")``.

    Separate from :class:`ModelRegistry` so that nothing on the univariate path
    branches on spec type. It keeps ModelRegistry's contract: the id alone
    identifies the numbers, a second ``fit`` of the same id raises unless
    ``overwrite=True``, and ``corrected=`` is refused at fit time (the corrected
    version of a model is its own ``_bc`` spec).

    ``cache_dir`` / ``cache_tag`` work as on :class:`ModelRegistry`. A joint
    fit is ONE file per id, ``<...>/joint/<model_id>.pkl``, beside the
    ``theta/`` and ``beta/`` folders of the same container pair: it covers both
    quantities, so it belongs to neither. The folder is named from the theta
    container (the constructor has already checked the two agree on im_tag
    and rows).

    Examples
    --------
    >>> theta = load_fragility_data(root, "AvgSA_03", quantity="theta")
    >>> beta = load_fragility_data(root, "AvgSA_03", quantity="beta")
    >>> print(check_joint_inputs(theta, beta)["report"])
    >>> joint = JointModelRegistry(theta, beta)
    >>> joint.fit("A8a", n_boot=5000)
    >>> print(joint["A8a"].summary())
    """

    def __init__(self, theta_data: FragilityData, beta_data: FragilityData,
                 specs: dict[str, JointModelSpec] | None = None, *,
                 cache_dir: Path | str | None = None,
                 cache_tag: str | None = None):
        problems = []
        if (theta_data.quantity, beta_data.quantity) != OUTCOMES:
            problems.append(f"quantities are ({theta_data.quantity!r}, "
                            f"{beta_data.quantity!r}), expected ('theta', 'beta')")
        if theta_data.im_tag != beta_data.im_tag:
            problems.append(f"im_tags differ: {theta_data.im_tag!r} vs {beta_data.im_tag!r}")
        if not theta_data.rows.index.equals(beta_data.rows.index):
            problems.append("the row spines differ (labels or order)")
        if not theta_data.groups.equals(beta_data.groups):
            problems.append("the design groups differ")
        if theta_data.k != beta_data.k:
            problems.append(f"replicate counts differ: {theta_data.k} vs {beta_data.k}")
        if problems:
            raise ValueError("cannot pair these containers:\n  " + "\n  ".join(problems))
        self.theta_data, self.beta_data = theta_data, beta_data
        # One ModelRegistry per quantity, used ONLY for its contrast cache, so a
        # univariate and a joint fit of the same contrast in one notebook share
        # the very same Contrast objects.
        self._uni = {"theta": ModelRegistry(theta_data), "beta": ModelRegistry(beta_data)}
        self.specs = dict(JOINT_MODEL_SPECS if specs is None else specs)
        self._fits: dict[tuple[str, str], JointModelFit] = {}
        self._contrasts: dict[tuple[str, bool], JointContrast] = {}
        # The inner univariate registries above stay cache-less: they serve
        # contrasts only. The cache belongs to this registry.
        self._init_fit_cache(cache_dir, cache_tag)
        self._contrast_fp: dict[tuple[str, bool], dict] = {}
        self._container_hash: str | None = None

    def contrast(self, name: str, corrected: bool = False) -> JointContrast:
        """The joint contrast, built once per ``(name, corrected)``.

        ``corrected`` applies to BETA only. The theta half is always the
        uncorrected contrast -- the same object as in the uncorrected joint
        contrast -- because theta is never corrected.
        """
        key = (name, bool(corrected))
        if key not in self._contrasts:
            self._contrasts[key] = JointContrast.from_pair(
                self._uni["theta"].contrast(name, False),
                self._uni["beta"].contrast(name, bool(corrected)))
        return self._contrasts[key]

    def fit(self, model_id: str, *, n_boot: int | None = None, overwrite: bool = False,
            verbose: bool = False, boot_seed: int = BOOT_SEED, refit: bool = False,
            **kw) -> JointModelFit:
        """Fit one A8 rung, with an outer bootstrap if ``n_boot`` is given.

        ``**kw`` is passed to ``mra.fit_reml_mv`` for the FULL-SAMPLE fit only
        (e.g. ``n_starts``, ``method``); the replicates are warm-started from
        its optimum, one start each.

        Which outer bootstrap runs is the spec's ``bootstrap_scheme`` (an
        ``_hb`` id is the hybrid one), never an argument here, so an id always
        means one set of numbers. ``boot_seed`` seeds the hybrid scheme's
        effect draws and has no effect on a record-only spec.

        With a ``cache_dir`` on the registry the fit is loaded from disk when
        its saved inputs match, and fitted and saved otherwise, exactly as in
        :meth:`ModelRegistry.fit`; ``refit=True`` ignores the disk. Without a
        ``cache_dir`` nothing is read or written.
        """
        if model_id not in self.specs:
            hint = (" It is a univariate model: fit it with ModelRegistry."
                    if model_id in MODEL_SPECS else "")
            raise KeyError(f"unknown joint model {model_id!r} - known: "
                           f"{sorted(self.specs)}.{hint}")
        if "corrected" in kw:
            raise TypeError(
                f"fit({model_id!r}, corrected=...) is not supported: whether beta is "
                "bias-corrected is part of the model's SPEC, so that an id always means "
                f"one set of numbers. Fit {model_id + '_bc'!r} instead.")
        spec = self.specs[model_id]
        key = (model_id, "REML")
        if key in self._fits and not overwrite:
            raise ValueError(
                f"{model_id} [REML] has already been fitted. Pass overwrite=True if you "
                "mean to replace it - the refusal is deliberate, so a re-run cannot "
                "silently change what a later cell reports.")
        try:
            jcon = self.contrast(spec.contrast, spec.corrected)
        except KeyError as exc:
            if not spec.corrected:
                raise
            raise KeyError(
                f"{model_id} is a bias-corrected spec, but the beta data it needs is "
                f"missing: {exc.args[0] if exc.args else exc}. The bias-corrected "
                "estimates come from nb 070 section 2.4 via nb 072.") from exc

        if spec.corrected:
            left, right = jcon.terms
            if all((self.beta_data.hat(s) == self.beta_data.hat(s, corrected=True)).all()
                   for s in (left, right)):
                warnings.warn(f"{model_id}: the bias-corrected and uncorrected beta "
                              f"estimates are identical for both arms of {spec.contrast}, "
                              f"so {model_id} reproduces {model_id.removesuffix('_bc')}.",
                              UserWarning, stacklevel=2)

        # Without a cache: the original body, unchanged (now in _fit_joint).
        if self.cache_dir is None:
            fit_obj = self._fit_joint(spec, jcon, n_boot=n_boot, verbose=verbose,
                                      boot_seed=boot_seed, **kw)
        else:
            fit_obj = self._cached_fit_joint(spec, jcon, n_boot=n_boot, verbose=verbose,
                                             boot_seed=boot_seed, refit=refit, **kw)
        self._fits[key] = fit_obj
        return fit_obj

    def _fit_joint(self, spec: JointModelSpec, jcon: JointContrast, *,
                   n_boot: int | None, verbose: bool, boot_seed: int,
                   **kw) -> JointModelFit:
        """Full-sample joint fit, then the outer bootstrap. Stores nothing.

        The computing part of :meth:`fit`, moved here unchanged so the cache can
        run it only when needed. A root of the joint code hash
        (:data:`JOINT_FIT_CODE_ROOTS`): it builds the saved ``boot`` record.
        """
        model_id = spec.model_id
        raw, x_names, scaling = fit_reml_joint(jcon, spec, **kw)
        reps, boot = None, {}
        if n_boot is not None:
            hybrid = spec.bootstrap_scheme == "hybrid"
            if verbose:
                print(f"  {model_id} [REML]: outer bootstrap over "
                      f"{min(n_boot, jcon.k)} replicates"
                      f"{' (hybrid)' if hybrid else ''}...")
            # `raw` goes in because the hybrid scheme draws its pseudo-data
            # from the full-sample fit; the record-only scheme ignores it.
            reps = bootstrap_joint_fits(jcon, spec, n_boot=n_boot, T0=raw["T"],
                                        verbose=verbose, full_fit=raw,
                                        boot_seed=boot_seed)
            # What was actually run. zero_levels: every (level, outcome) whose
            # variance is zero in the full-sample fit, where no effect is drawn.
            boot = {"scheme": spec.bootstrap_scheme,
                    "seed": boot_seed if hybrid else None,
                    "zero_levels": tuple(
                        (lev, q) for lev in raw["levels"]
                        for q, zero in zip(OUTCOMES, raw["at_zero"][lev]) if zero)
                    if hybrid else ()}
        return JointModelFit(spec=spec, contrast=jcon, raw=raw, x_names=x_names,
                             scaling=scaling, replicates=reps, boot=boot)

    # -- the on-disk cache (active only with cache_dir) -----------------------

    def cache_path(self, model_id: str) -> Path:
        """Where a joint fit is saved: ``.../joint/<model_id>.pkl``."""
        if self.cache_dir is None:
            raise ValueError("this registry has no cache_dir")
        return (fcache.cache_folder(self.cache_dir, self.theta_data, self.cache_tag)
                / "joint" / f"{model_id}.pkl")

    def _cache_fingerprint(self, spec: JointModelSpec, jcon: JointContrast,
                           kw: dict, boot_seed: int) -> tuple[dict, dict, dict]:
        """``(fingerprint, meta, code inventory)`` of one joint spec.

        As :meth:`ModelRegistry._cache_fingerprint`, except that only ``note``
        is left out of the spec (``estimators`` is always ``("REML",)``) and the
        code hash is the JOINT one, so editing a univariate-only function does
        not refit the joint models and vice versa.
        """
        ckey = (jcon.name, bool(jcon.beta.corrected))
        if ckey not in self._contrast_fp:
            self._contrast_fp[ckey] = fcache.joint_contrast_entry(jcon)
        digest, inventory = fcache.code_hash(JOINT_FIT_CODE_ROOTS)
        fp = fcache.build_fingerprint(
            contrast=self._contrast_fp[ckey],
            spec_tuple=fcache.spec_items(spec, frozenset({"note"})),
            code_digest=digest, n_code=len(inventory), kw=dict(kw),
            boot_seed=boot_seed if spec.bootstrap_scheme == "hybrid" else None)
        if self._container_hash is None:
            self._container_hash = fcache._sha(
                fcache.container_hash(self.theta_data) + "|"
                + fcache.container_hash(self.beta_data))
        meta = {"model_id": spec.model_id, "estimator": "REML",
                "im_tag": self.theta_data.im_tag, "quantity": "theta+beta",
                "k": jcon.k, "container_hash": self._container_hash,
                "versions": fcache.library_versions()}
        return fp, meta, inventory

    def _cached_fit_joint(self, spec: JointModelSpec, jcon: JointContrast, *,
                          n_boot: int | None, verbose: bool, boot_seed: int,
                          refit: bool, **kw) -> JointModelFit:
        """Load a joint fit from the cache if valid, else fit it and save it."""
        fp, meta, inventory = self._cache_fingerprint(spec, jcon, kw, boot_seed)
        want = None if n_boot is None else min(int(n_boot), jcon.k)

        def from_payload(p: dict) -> JointModelFit:
            # Same rebuild rule as the univariate registry: the registry's own
            # spec and memoised joint contrast, through the constructor.
            reps, boot = fcache.slice_replicates(p, want)
            return JointModelFit(spec=spec, contrast=jcon, raw=p["raw"],
                                 x_names=p["x_names"], scaling=p["scaling"],
                                 replicates=reps, boot=boot)

        return self._load_or_fit(
            (spec.model_id, "REML"), self.cache_path(spec.model_id),
            fp, meta, inventory, want, refit,
            fit_fn=lambda: self._fit_joint(spec, jcon, n_boot=n_boot, verbose=verbose,
                                           boot_seed=boot_seed, **kw),
            to_payload=_joint_fit_payload, from_payload=from_payload)

    def __getitem__(self, key) -> JointModelFit:
        """``joint["A8a"]`` or ``joint["A8a", "REML"]``."""
        model_id, estimator = key if isinstance(key, tuple) else (key, "REML")
        if (model_id, estimator) not in self._fits:
            raise KeyError(f"{model_id} [{estimator}] has not been fitted - call "
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
        return list(self._fits)

    def table(self, ids: Sequence[str] | None = None) -> pd.DataFrame:
        """One row per joint fit. Kept apart from ModelRegistry.table: the columns differ."""
        rows = [f.to_row() for (mid, _), f in self._fits.items()
                if ids is None or mid in ids]
        if not rows:
            raise ValueError("no joint fits yet - call fit() first")
        return pd.DataFrame(rows).set_index(["model", "estimator"])


# -- the on-disk fit cache: what is saved, and which code it depends on ---------------
#
# The mechanics live in fit_cache; these are the parts that must know the
# project's classes.

def _model_fit_payload(fit: ModelFit) -> dict:
    """The RESULTS of a ModelFit as a plain dict, for the cache file.

    Builtin, numpy and pandas objects only -- no project class -- so unpickling
    never depends on the dataclass definitions. ``spec`` and ``contrast`` are
    deliberately NOT saved: the registry supplies its own on load (the contrast
    alone holds the 120 x 5000 replicate matrix, which would otherwise be
    duplicated in every file). The replicate frame's ``attrs`` are stored
    separately as well, so they survive even a pandas that dropped them on pickle.
    """
    reps = fit.replicates
    return {"estimator": fit.estimator, "beta": fit.beta, "se_beta": fit.se_beta,
            "tau2": fit.tau2, "het": fit.het, "x_names": fit.x_names,
            "scaling": fit.scaling, "raw": fit.raw, "boot": fit.boot,
            "replicates": reps,
            "replicates_attrs": None if reps is None else dict(reps.attrs)}


def _joint_fit_payload(fit: JointModelFit) -> dict:
    """The results of a JointModelFit as a plain dict (see :func:`_model_fit_payload`)."""
    reps = fit.replicates
    return {"raw": fit.raw, "x_names": fit.x_names, "scaling": fit.scaling,
            "boot": fit.boot, "replicates": reps,
            "replicates_attrs": None if reps is None else dict(reps.attrs)}


# The entry points of the CODE HASH (fit_cache.code_hash; user decision Q3,
# 2026-10-19). Starting from these, the hash follows every function and module
# constant they reach in fragility_data_models, metaregression_analysis and
# data_simulation, so a helper deep in the call tree is covered without being
# listed. What must be listed by hand is what a static walk cannot resolve:
# methods reached through an instance, ``con.v`` / ``con.g`` / ``jcon.n``.
# Hashing ignores comments and docstrings, so documenting any of these never
# triggers a refit; changing logic or a constant they use always does.
#
# Not here, on purpose: make_contrast, _CONTRAST_SPECS, mra.bootstrap_cov and
# JointContrast.from_pair. The contrast is fingerprinted by CONTENT, so how it
# was built cannot change a fit without changing that hash. Nor anything run
# after loading (ci, tau_ci, random_effects, marginal_covariance, forest,
# summary): editing those must never refit.
#
# Two lists, one per registry, so editing the bivariate REML does not refit the
# univariate ladder and vice versa.
UNIVARIATE_FIT_CODE_ROOTS: tuple = (
    fit_dl, fit_reml, bootstrap_fits, ModelRegistry._fit_estimator,
    Contrast.n, Contrast.k, Contrast.v, Contrast.diag_V, Contrast.g,
    Contrast.site_codes,
)
JOINT_FIT_CODE_ROOTS: tuple = (
    fit_reml_joint, bootstrap_joint_fits, JointModelRegistry._fit_joint,
    JointContrast.n, JointContrast.k, Contrast.n, Contrast.g, Contrast.site_codes,
)


def fit_code_inventory(joint: bool = False) -> dict[str, str]:
    """``{function or constant: sha256}`` the code hash covers. For inspection.

    Shows exactly which functions a refit can be triggered by; the same mapping
    is written into every cache manifest (``_meta.code_inventory``).
    """
    return fcache.code_hash(JOINT_FIT_CODE_ROOTS if joint
                            else UNIVARIATE_FIT_CODE_ROOTS)[1]


# -- step 0: input checks -------------------------------------------------------------

def check_joint_inputs(theta_data: FragilityData, beta_data: FragilityData,
                       contrast: str = "y0_avg", n_skew_rows: int = 10,
                       alpha: float = 0.01) -> dict:
    """Step 0 of the A8 plan: is the pair of containers fit for a joint model?

    Raises on the two failures that would make the joint fit silently wrong
    (items 1-3); reports everything else. Returns a dict of the results with a
    printable ``"report"`` string.

    1. **Labels.** For every loaded arm, theta and beta replicate columns are
       equal in labels and order, and so are the unit indexes.
    2. **Same usable replicates.** NaN / inf patterns identical between theta
       and beta, ``{arm}_n_obs`` identical across the containers. A screened
       replicate is rejected per FIT and one fit gives both quantities, so these
       should agree by construction; a re-run that broke it would put theta and
       beta on different replicate sets. Units with ``n_obs < k`` are listed,
       because np.cov on a NaN column poisons V.
    3. **Statistical alignment.** Labels can agree while the values come from
       different resamples (two files from two runs). Under that failure the
       per-row sampling correlations r_i are pure noise, ``r_i sqrt(k) ~ N(0,1)``,
       so ``k sum r_i^2 ~ chi2_n``. The check raises if the data are consistent
       with that (p > alpha). On the real data the statistic is far in the
       tail, so a genuine near-zero sampling correlation is not a practical
       worry; if it ever were, it would show here first.
    4. **Conditioning** of the 2n x 2n V_known and of its blocks, and k against
       2n. Reported, not acted on: the default is always the full np.cov (the
       alternatives - shrinkage or a block-diagonal V - would each trade a known
       bias for a smaller variance, so they are not used by default).
    5. **Skewness** of the theta and ln beta replicates of the contrast, per row,
       with the ``n_skew_rows`` most skewed beta rows listed for a Q-Q plot.
    6. **Context**: r_samp summary, r_naive and the per-arm within-unit
       theta-beta correlation of the replicates.
    """
    lines, problems, out = [], [], {}
    k = theta_data.k

    # -- 1 and 2: labels, NaN patterns, n_obs, per arm ---------------------
    arm_corr = {}
    for sym in theta_data.available:
        rt, rb = theta_data.rep(sym), beta_data.rep(sym)
        if not rt.columns.equals(rb.columns):
            problems.append(f"{sym}: replicate columns differ in labels or order")
        if not rt.index.equals(rb.index):
            problems.append(f"{sym}: unit indexes differ")
        vt, vb = rt.to_numpy(float), rb.to_numpy(float)
        if vt.shape == vb.shape:
            if not np.array_equal(~np.isfinite(vt), ~np.isfinite(vb)):
                problems.append(f"{sym}: theta and beta have different NaN/inf patterns, "
                                "i.e. different usable replicates")
            if np.isfinite(vt).all() and np.isfinite(vb).all():
                # per-unit corr(theta*, beta*), vectorised over units
                zt = (vt - vt.mean(1, keepdims=True)) / vt.std(1, keepdims=True)
                zb = (vb - vb.mean(1, keepdims=True)) / vb.std(1, keepdims=True)
                arm_corr[sym] = (zt * zb).mean(1)
        col = f"{ARM_OF_SYMBOL[sym]}_n_obs"
        if col in theta_data.rows and col in beta_data.rows:
            if not (theta_data.rows[col] == beta_data.rows[col]).all():
                problems.append(f"{col} differs between the theta and beta containers")
            short = theta_data.rows.index[theta_data.rows[col] < k]
            if len(short):
                problems.append(f"{col} < k={k} for {len(short)} unit(s), e.g. "
                                f"{list(short[:3])}: screened-out replicates are NaN")
    if problems:
        raise ValueError("joint inputs are not aligned:\n  " + "\n  ".join(problems))
    lines.append(f"  labels / NaN patterns / n_obs : ok for every arm "
                 f"({', '.join(theta_data.available)}); k = {k}")

    # -- the joint contrast (from_pair re-checks the contrast's labels) ----
    jcon = JointContrast.from_pair(make_contrast(theta_data, contrast),
                                   make_contrast(beta_data, contrast))
    n = jcon.n

    # -- 3: statistical alignment --------------------------------------------
    r = jcon.r_samp.to_numpy()
    stat = float(k * np.sum(r ** 2))
    p = float(stats.chi2.sf(stat, n))
    out["alignment_chi2"], out["alignment_p"] = stat, p
    if p > alpha:
        raise ValueError(
            f"the theta and beta replicates of {contrast} look UNALIGNED: the per-row "
            f"theta-beta sampling correlations are consistent with zero (k sum r^2 = "
            f"{stat:.1f} on {n} df, p = {p:.3f}). Labels match, so the likely cause is "
            "theta and beta bootstrap files written by different runs of nb 070.")
    lines.append(f"  statistical alignment         : ok (k sum r_samp^2 = {stat:.0f} on "
                 f"{n} df; ~{n} if unaligned)")

    # -- 4: conditioning -------------------------------------------------------
    def cond(M):
        ev = np.linalg.eigvalsh(M)
        return {"min_eig": float(ev.min()), "max_eig": float(ev.max()),
                "cond": float(ev.max() / ev.min()) if ev.min() > 0 else np.inf,
                "n_small": int(np.sum(ev < 1e-10 * ev.max()))}
    out["conditioning"] = {"joint": cond(jcon.V), "theta": cond(jcon.V_theta),
                           "beta": cond(jcon.V_beta)}
    cj = out["conditioning"]["joint"]
    lines.append(f"  V_known {2 * n}x{2 * n}               : min eig {cj['min_eig']:.2e}, "
                 f"cond {cj['cond']:.2e} (theta block "
                 f"{out['conditioning']['theta']['cond']:.1e}, beta block "
                 f"{out['conditioning']['beta']['cond']:.1e}); k/2n = {k / (2 * n):.1f}")
    if cj["n_small"] or k < 4 * n:
        lines.append("    -> WARNING: V_known is near-singular or k is small against 2n; "
                     "shrinkage or a block-diagonal V are the alternatives (default stays the full np.cov)")

    # -- 5: skewness -----------------------------------------------------------
    sk_t = pd.Series(stats.skew(jcon.theta.y.to_numpy(float), axis=1),
                     index=jcon.theta.y.index, name="skew_theta")
    sk_b = pd.Series(stats.skew(jcon.beta.y.to_numpy(float), axis=1),
                     index=jcon.beta.y.index, name="skew_beta")
    out["skew"] = pd.concat([sk_t, sk_b], axis=1)
    out["most_skewed_beta_rows"] = sk_b.abs().sort_values(ascending=False).index[:n_skew_rows]
    lines.append(f"  skewness of replicates        : ln beta median {sk_b.median():+.2f}, "
                 f"range {sk_b.min():+.2f}..{sk_b.max():+.2f}; theta median "
                 f"{sk_t.median():+.2f}, range {sk_t.min():+.2f}..{sk_t.max():+.2f}")

    # -- 6: context --------------------------------------------------------------
    out["r_samp"], out["r_naive"] = jcon.r_samp, jcon.r_naive
    out["arm_corr"] = {s: pd.Series(v) for s, v in arm_corr.items()}
    lines.append(f"  r_samp per row                : mean {r.mean():+.3f}, median "
                 f"{np.median(r):+.3f}, range {r.min():+.3f}..{r.max():+.3f}")
    lines.append(f"  r_naive                       : {jcon.r_naive:+.3f}")
    for s, v in arm_corr.items():
        lines.append(f"  arm {s:<6s} corr(theta*, beta*) : mean {v.mean():+.3f}, range "
                     f"{v.min():+.3f}..{v.max():+.3f}")
    out["contrast"] = jcon
    out["report"] = f"joint input checks ({contrast}):\n" + "\n".join(lines)
    return out


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
