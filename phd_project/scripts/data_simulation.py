"""Fake-data simulation for verifying the REML meta-regression fits.

Why this module exists
----------------------
There is no Python package that fits the crossed design x site model of A1c
(``metafor::rma.mv`` in R does, PyMare does not -- it takes a variance *vector*,
not a matrix). So ``mra.fit_reml`` is hand-rolled, and the only honest way to
believe a hand-rolled estimator is to feed it data whose truth you already know
and check that it hands the truth back.

That is what this module does. It generates ``y`` from the generative story the
roadmap writes down for A1c, refits it with
``mra.fit_reml``, and reports how well the truth was recovered: bias against
Monte-Carlo error, whether the model-based standard errors match the actual
spread of the estimates, and whether nominal confidence intervals cover at their
nominal rate.

The same three calls serve every rung of the roadmap's ladder -- A1a, A1b, A1c
and a meta-regression on top of A1c. Only the inputs change:

    analysis   X                      components
    --------   --------------------   -------------------------------------
    A1a        ones (n, 1)            {tau_u: I}
    A1b        ones (n, 1)            {tau_d: Z, tau_u: I}
    A1c        ones (n, 1)            {tau_d: Z, tau_s: S, tau_u: I}
    A3 on A1c  [1, x1, ..., xq]       {tau_d: Z, tau_s: S, tau_u: I}

Generation is from FACTOR matrices, not from the marginal covariance
----------------------------------------------------------------------
You hand over ``Z`` (n x J), ``S`` (n x n_s) and ``I`` (n x n) -- the matrices
that map a random effect onto rows -- and this module draws the random effects
themselves::

    d ~ N(0, tau_d^2 I_J)      one draw per design
    s ~ N(0, tau_s^2 I_ns)     one draw per site
    u ~ N(0, tau_u^2 I_n)      one draw per row (the design x site interaction)
    eps ~ N(0, V_known)        the sampling error, from the inner bootstrap

    y = X beta + Z d + S s + u + eps

It then forms ``Gs = [Z Z', S S', I]`` itself and passes those to ``fit_reml``.

The alternative -- accepting the caller's ``Gs`` and drawing
``y = X beta + chol(Sigma) z`` -- is algebraically identical and strictly worse
as a test. If the simulator consumed the same ``Gs`` the fitter consumes, then a
mistake in how ``Z Z'`` was built would appear on both sides of the comparison
and cancel: the test would pass on broken code. Deriving the marginal from the
factor makes the whole "indicator -> Gs -> fit" path an object of the test, which
is where the realistic bugs live (positional instead of label-based row mapping,
a dropped level, a site indicator with 120 columns instead of 60).

What ``V_known`` must be, and why it changes between A1a and A1b
-----------------------------------------------------------------
``V_known`` is the parameter-free part of Sigma: the sampling covariance carried
over from the inner bootstrap, held FIXED across simulations because it is a
feature of the study, not something the simulation is testing.

  * A1a: ``V_known = diag(v1)`` where ``v1[i] = Var_r(a_i - b_g[i])``. The
    combined variance is exact on the diagonal here because the SS and FX arms
    use disjoint record sets, so the cross term is zero.
  * A1b / A1c / regression: ``V_known = diag(v_a) + Z V_bb Z'``. Every row in
    design group j subtracts the SAME b_j, so the FX contribution is a dense
    off-diagonal block, not a diagonal one. Passing ``diag(v1)`` here both
    double-counts the diagonal and throws away exactly the covariance that
    ``tau_d^2`` would otherwise absorb -- the identifiability trap of A1c.

See ``mra.check_variance_split`` for the assertion that ties the two together.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm

import phd_project.scripts.metaregression_analysis as mra

# A V_known assembled as diag(v_a) + Z V_bb Z' from only k ~ 10 replicates can come
# back very slightly indefinite through rounding alone. Eigenvalues above -this are
# treated as numerical zero and clipped; anything more negative is a real problem
# with V_bb and is raised rather than papered over.
_PSD_TOL = 1e-10

# Below this, a factor matrix is considered diagonal and the Cholesky is replaced by an
# elementwise sqrt. This is not a micro-optimisation: A1a's V_known is diagonal and a
# 5000-simulation run would otherwise pay for a 120 x 120 triangular solve it does not
# need.
_DIAG_TOL = 1e-14


# =============================================================================
# Component specification
# =============================================================================

def _validate_components(components: Mapping[str, tuple], n: int
                         ) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Unpack and check the ``{name: (factor, tau_true)}`` mapping.

    Returns the names, the factor matrices and the TRUE standard deviations, all in
    the mapping's insertion order -- which becomes the order of ``Gs`` and therefore
    the order of everything ``fit_reml`` reports back. Python dicts preserve
    insertion order, so "the order you wrote them in" is a reliable contract.

    Raises rather than warns on a shape mismatch. The bug this module was written
    after was a shape *coincidence* that let a wrong array through silently; the
    lesson is that anything shape-related here should be loud.
    """
    if not components:
        raise ValueError("components is empty -- every model needs at least one "
                         "variance component (for A1a that is {'tau_u': (I, tau)})")

    names, factors, taus = [], [], []
    for name, spec in components.items():
        try:
            factor, tau = spec
        except (TypeError, ValueError):
            raise ValueError(
                f"components[{name!r}] must be a (factor, tau_true) pair, got "
                f"{spec!r}") from None

        factor = np.asarray(factor, dtype=float)
        if factor.ndim != 2:
            raise ValueError(f"components[{name!r}] factor must be 2-D (n, K), got "
                             f"shape {factor.shape}")
        if factor.shape[0] != n:
            raise ValueError(
                f"components[{name!r}] factor has {factor.shape[0]} rows but the "
                f"model has n = {n}. The factor maps rows onto levels, so it is "
                f"(n, K) -- for the site component that is (120, 60), not (120, 120).")

        tau = float(tau)
        if tau < 0:
            raise ValueError(f"components[{name!r}] tau_true = {tau} is negative. "
                             "tau is a standard deviation; pass sqrt(tau2).")

        names.append(name)
        factors.append(factor)
        taus.append(tau)

    return names, factors, np.asarray(taus, dtype=float)


def build_component_gs(components: Mapping[str, tuple]
                       ) -> tuple[list[np.ndarray], list[str]]:
    """Turn factor matrices into the ``Gs`` list ``fit_reml`` expects.

    ``G_p = M_p M_p'`` is the n x n matrix with a 1 wherever two rows share a level
    of component p -- for the design factor that is ``Z Z'``, which is the matrix
    multiplying ``tau_d^2`` in Sigma.

    Exposed publicly so the real-data fit in the notebook can build its ``Gs`` the
    same way the simulation does. If they are built by two different pieces of code,
    the simulation is not testing the fit you actually ran.

    Parameters
    ----------
    components : mapping of str -> (factor (n, K), tau_true)
        ``tau_true`` is ignored here; the signature is shared with
        ``simulate_reml_fits`` so the same dict can be passed to both.

    Returns
    -------
    (Gs, names) : list of (n, n) arrays, list of str -- in insertion order.
    """
    first = next(iter(components.values()))
    n = np.asarray(first[0]).shape[0]
    names, factors, _ = _validate_components(components, n)
    return [M @ M.T for M in factors], names


# =============================================================================
# Drawing one dataset
# =============================================================================

def _known_error_sampler(V_known: np.ndarray):
    """Return a closure drawing ``eps ~ N(0, V_known)``, factorised once.

    Factorising V_known on every simulation would dominate the run time for no
    reason: V_known is fixed across the whole study by construction (it comes from
    the inner bootstrap, and re-estimating it per replicate would need a double
    bootstrap -- Efron & Tibshirani Ch. 25). So the factorisation happens here, once,
    and the returned closure is called n_sims times.
    """
    V_known = np.asarray(V_known, dtype=float)
    n = V_known.shape[0]

    # A1a's V_known is diagonal, and the elementwise sqrt is both faster and exactly
    # what the two-level model means: independent sampling errors, one per row.
    off_diagonal = V_known - np.diag(np.diag(V_known))
    if np.max(np.abs(off_diagonal)) < _DIAG_TOL:
        sd = np.sqrt(np.diag(V_known))
        if np.any(np.diag(V_known) < 0):
            raise ValueError("V_known has a negative diagonal entry -- a sampling "
                             "variance cannot be negative.")

        def draw(rng):
            return sd * rng.standard_normal(n)
        return draw

    # Dense case: diag(v_a) + Z V_bb Z'. Try the cheap Cholesky first; fall back to an
    # eigendecomposition only to distinguish "indefinite by rounding" (fine, clip)
    # from "V_bb is genuinely broken" (raise, and say where to look).
    try:
        L = cholesky(V_known, lower=True)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(V_known)
        if eigenvalues.min() < -_PSD_TOL:
            raise ValueError(
                f"V_known is not positive semi-definite (min eigenvalue "
                f"{eigenvalues.min():.3e}). With only a handful of bootstrap "
                f"replicates, Z V_bb Z' can be rank-deficient -- run "
                f"mra.check_V_bb(V_bb, n_replicates=k) before simulating. Not "
                f"jittering it silently, because a broken V_bb invalidates the "
                f"whole A1b/A1c fit, not just the simulation.") from None
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        L = eigenvectors * np.sqrt(eigenvalues)

    def draw(rng):
        return L @ rng.standard_normal(n)
    return draw


def draw_effects(rng: np.random.Generator, factors: Sequence[np.ndarray],
                 taus: Sequence[float], base: np.ndarray | None = None,
                 return_parts: bool = False):
    """Draw one set of random effects and add them, level by level, onto ``base``.

    ``base + sum_p M_p z_p`` with ``z_p ~ N(0, tau_p^2 I_K)``: one independent
    draw per group at every level (per design, per site, per row).

    This is the ONE place a univariate random effect is drawn. The fake-data
    simulators (:func:`simulate_y`, :func:`simulate_reml_fits`) and the hybrid
    bootstrap (:func:`hybrid_pseudo_data`) all call it, so the bootstrap draws
    its "new true effects" with literally the code the recovery tests have
    already checked. If the two drew effects with different code, a passing
    simulation would say nothing about the bootstrap.

    Parameters
    ----------
    rng : np.random.Generator
    factors : sequence of (n, K_p) arrays
        The indicator matrices mapping groups onto rows, in component order.
    taus : sequence of float
        Standard DEVIATIONS, one per factor. A zero draws nothing at that level
        (the standard normals are still consumed, so the random stream does not
        depend on which components happen to be zero).
    base : (n,) array, optional
        What the effects are added to -- ``X beta`` in every caller. Defaults
        to zeros. The contributions are added ONE AT A TIME, in order, onto
        this vector, rather than summed first and added once. Floating-point
        addition is not associative, and this order is exactly what the
        simulators did before this function existed, so every seeded
        simulation reproduces bit for bit.
    return_parts : bool
        Also return ``[{"z": z_p, "contribution": M_p z_p}, ...]``.

    Returns
    -------
    (n,) array -- or ``(array, parts)`` if ``return_parts``.
    """
    n = np.asarray(factors[0]).shape[0]
    y = np.zeros(n) if base is None else base
    parts = []
    for M, tau in zip(factors, taus):
        z = tau * rng.standard_normal(M.shape[1])
        contribution = M @ z
        # `y = y + ...`, never `y += ...`: the caller's `base` must not be
        # modified in place (the bootstrap passes the same X beta every time).
        y = y + contribution
        if return_parts:
            parts.append({"z": z, "contribution": contribution})
    return (y, parts) if return_parts else y


def simulate_y(rng: np.random.Generator, beta_true, X: np.ndarray,
               V_known: np.ndarray, components: Mapping[str, tuple],
               return_parts: bool = False):
    """Draw one fake dataset from the generative model.

    ``y = X beta + sum_p M_p z_p + eps``, with ``z_p ~ N(0, tau_p^2 I_K)`` and
    ``eps ~ N(0, V_known)``.

    Exposed so a single fake dataset can be inspected or plotted next to the real
    one -- if the fake data do not look like the real data, the model is wrong in a
    way no amount of coverage checking will reveal.

    Parameters
    ----------
    rng : np.random.Generator
    beta_true : scalar or (q,)
        True fixed effects, on the log-ratio scale.
    X : (n, q)
    V_known : (n, n)
    components : mapping of str -> (factor (n, K), tau_true)
    return_parts : bool
        If True, also return the realised ``{name: z_p}`` draws, the per-row random
        effect contributions and ``eps``, so the variance decomposition can be
        checked against its own realised values.

    Returns
    -------
    y : (n,)   -- or ``(y, parts)`` if ``return_parts``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    beta_true = np.atleast_1d(np.asarray(beta_true, dtype=float))
    names, factors, taus = _validate_components(components, n)

    # The effects come from draw_effects, the same function the hybrid
    # bootstrap uses; see its docstring for why that sharing matters.
    y, effect_parts = draw_effects(rng, factors, taus, base=X @ beta_true,
                                   return_parts=True)
    parts = dict(zip(names, effect_parts)) if return_parts else {}

    eps = _known_error_sampler(V_known)(rng)
    y = y + eps

    if return_parts:
        parts["eps"] = eps
        return y, parts
    return y


# =============================================================================
# The simulation loop
# =============================================================================

@dataclass
class SimulationResult:
    """Everything one simulation run produced, plus what it was run against.

    ``draws`` holds one row per simulation with columns ``beta_0 ...``,
    ``se_beta_0 ...``, ``tau2_<name>``, ``at_zero_<name>``, ``converged`` and
    ``nll``. The inputs are carried alongside so a result is self-describing --
    a saved ``SimulationResult`` can be summarised months later without needing
    the notebook cell that produced it.
    """
    draws: pd.DataFrame
    beta_true: np.ndarray
    tau_true: np.ndarray
    component_names: list[str]
    X: np.ndarray
    V_known: np.ndarray
    n_sims: int
    seed: int
    fit_V_known: np.ndarray | None = None
    fit_kwargs: dict = field(default_factory=dict)

    @property
    def misspecified(self) -> bool:
        """True if the fit was given a different V_known from the generator."""
        return (self.fit_V_known is not None
                and not np.array_equal(self.fit_V_known, self.V_known))

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def q(self) -> int:
        return self.X.shape[1]

    @property
    def tau2_true(self) -> np.ndarray:
        return self.tau_true ** 2

    @property
    def frac_converged(self) -> float:
        return float(self.draws["converged"].mean())


def simulate_reml_fits(beta_true, X: np.ndarray, V_known: np.ndarray,
                       components: Mapping[str, tuple], n_sims: int = 2000,
                       seed: int = 20260915, fit_V_known: np.ndarray | None = None,
                       fit_kwargs: dict | None = None,
                       progress: bool = True) -> SimulationResult:
    """Generate ``n_sims`` fake datasets, refit each with REML, collect the estimates.

    Parameters
    ----------
    beta_true : scalar or (q,)
        True fixed effects in log-ratio units. For an intercept-only fit this is the
        true ``m1``; for the regression it is ``[b0, b1, ..., bq-1]``.
    X : (n, q)
        Fixed-effects design. ``np.ones((n, 1))`` for A1a/A1b/A1c. For the
        regression, centre and scale the covariates first (roadmap A3; Gelman &
        Hill Ch. 4) -- otherwise the intercept is uninterpretable and the
        coefficients are not comparable in magnitude.
    V_known : (n, n)
        The known sampling covariance, held fixed across simulations. See the module
        docstring for which one to pass at which rung of the ladder.
    components : mapping of str -> (factor (n, K), tau_true)
        ``tau_true`` is a standard DEVIATION, not a variance -- the roadmap reports
        in ratio units ``exp(tau)``, so SDs are the scale you have intuition for.
    n_sims : int
        More simulations shrink the Monte-Carlo error on every diagnostic as
        1/sqrt(n_sims). 2000 resolves a coverage of 0.95 to about +/-0.005, which is
        enough to see a real miscoverage; 5000 for a headline number.
    seed : int
        One generator drives the whole run, so ``(seed, n_sims)`` reproduces it
        exactly.
    fit_V_known : (n, n), optional
        Fit with a DIFFERENT known covariance from the one the data were generated
        with. Defaults to ``V_known``, i.e. a correctly specified fit.

        This is how a misspecification is measured rather than argued about, and it
        is the roadmap's central worry made testable: generate with the true
        ``diag(v_a) + Z V_bb Z'`` and fit with ``diag(v1)``, and the shared-b_j
        covariance the fit was not told about has nowhere to go except ``tau_d^2``.
        Since the two have the same block structure, the model cannot tell a real
        design effect from the record-set noise, and the bias runs one way -- in
        favour of "the building matters more", the conclusion A1c is meant to test.
    fit_kwargs : dict, optional
        Forwarded to ``mra.fit_reml``. ``{"method": "cholesky", "n_starts": 1}``
        is roughly an order of magnitude faster at n = 120 and is usually fine
        inside a simulation, where a single bad fit is diluted by the other 1999 --
        but check ``frac_converged`` and the spread of the taus before trusting it,
        because a flat REML surface is exactly what the restarts exist for.
    progress : bool
        Print a progress line every 10%. Simulation runs are minutes long and
        silence is indistinguishable from a hang.

    Returns
    -------
    SimulationResult
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, q = X.shape

    V_known = np.asarray(V_known, dtype=float)
    if V_known.shape != (n, n):
        raise ValueError(f"V_known must be {n} x {n} to match X, got {V_known.shape}")

    fit_V_known = (V_known if fit_V_known is None
                   else np.asarray(fit_V_known, dtype=float))
    if fit_V_known.shape != (n, n):
        raise ValueError(f"fit_V_known must be {n} x {n} to match X, got "
                         f"{fit_V_known.shape}")

    beta_true = np.atleast_1d(np.asarray(beta_true, dtype=float))
    if beta_true.size != q:
        raise ValueError(f"beta_true has {beta_true.size} entries but X has {q} "
                         f"columns -- one true coefficient per column, including "
                         f"the intercept.")

    names, factors, taus = _validate_components(components, n)
    Gs = [M @ M.T for M in factors]
    fit_kwargs = dict(fit_kwargs or {})

    # Both the error sampler and Gs are built once, outside the loop, for the same
    # reason v_i is held fixed in the outer bootstrap: they are study features, not
    # things being resimulated.
    draw_eps = _known_error_sampler(V_known)
    rng = np.random.default_rng(seed)

    records = []
    report_every = max(n_sims // 10, 1)
    for sim in range(n_sims):
        # Inlined rather than calling simulate_y, so the factorisation of V_known and
        # the validation of components are not repeated n_sims times.
        y = draw_effects(rng, factors, taus, base=X @ beta_true)
        y = y + draw_eps(rng)

        fit = mra.fit_reml(y, X, fit_V_known, Gs, names=names, **fit_kwargs)

        record = {"converged": bool(fit["converged"]), "nll": float(fit["nll"]),
                  "start_spread": float(fit["start_spread"])}
        for k in range(q):
            record[f"beta_{k}"] = float(fit["beta"][k])
            record[f"se_beta_{k}"] = float(fit["se_beta"][k])
        for p, name in enumerate(names):
            record[f"tau2_{name}"] = float(fit["tau2"][p])
            record[f"at_zero_{name}"] = bool(fit["at_zero_boundary"][p])
        records.append(record)

        if progress and (sim + 1) % report_every == 0:
            print(f"[simulate_reml_fits] {sim + 1}/{n_sims}", flush=True)

    return SimulationResult(draws=pd.DataFrame(records), beta_true=beta_true,
                            tau_true=taus, component_names=names, X=X,
                            V_known=V_known, n_sims=n_sims, seed=seed,
                            fit_V_known=fit_V_known, fit_kwargs=fit_kwargs)


# =============================================================================
# Diagnostics
# =============================================================================

def summarise_simulation(result: SimulationResult,
                         levels: Sequence[float] = (0.68, 0.95)) -> pd.DataFrame:
    """Recovery diagnostics, one row per estimated parameter.

    Three questions, in order of how badly a "no" would hurt:

    1. **Is the estimator unbiased?** ``z = bias / mcse``, where ``mcse`` is the
       Monte-Carlo standard error of the mean estimate. ``|z| > 2`` says the bias is
       resolved from zero by this many simulations -- it does NOT say the bias
       matters. Read ``pct_error`` for that; a bias of 0.1% of the truth is
       statistically detectable at 5000 simulations and practically irrelevant.
    2. **Are the standard errors honest?** ``se_ratio = mean_analytic_se /
       empirical_sd``. The analytic SE comes from ``(X' Sigma^-1 X)^-1`` conditional
       on the fitted variance components; the empirical SD is the actual spread of
       the estimates. A ratio below 1 means the reported intervals are too narrow.
       Expect slightly below 1 by construction, because ``fit_reml`` treats the
       fitted taus as known (see its Caveats).
    3. **Do the intervals cover?** The realised fraction of simulations where
       ``|estimate - truth| <= z_level * se``. This is the check that actually
       matters for a reported confidence interval, and it folds 1 and 2 together.

    Variance components get columns 1 only: ``fit_reml`` deliberately produces no
    standard errors for them ("there is no honest closed form here"), so the SE and
    coverage columns come back NaN. Their uncertainty comes from the outer bootstrap
    instead. ``frac_at_zero`` is reported for them, which is the diagnostic that
    replaces coverage -- a component pinned at the boundary in most simulations is
    not being estimated, it is being detected as absent.

    Parameters
    ----------
    result : SimulationResult
    levels : sequence of float
        Nominal coverage levels, e.g. ``(0.68, 0.95)``. Each adds a ``cov_<pct>``
        column. Two-sided normal quantiles are used, matching how the fit's
        ``se_beta`` would be turned into an interval in practice.

    Returns
    -------
    DataFrame indexed by parameter name (``beta_0 ...``, then ``tau2_<name>``).
    """
    draws = result.draws
    n_sims = len(draws)
    rows = {}

    def moments(estimates: np.ndarray, truth: float) -> dict:
        """Bias, its Monte-Carlo error, and the relative error. Shared by both kinds."""
        mean = float(np.mean(estimates))
        bias = mean - truth
        # MCSE is the SD of the estimates divided by sqrt(n_sims): how precisely this
        # run pins down the MEAN estimate, which is what a bias claim rests on.
        mcse = float(np.std(estimates, ddof=1) / np.sqrt(n_sims))
        return {"truth": truth, "mean": mean, "bias": bias, "mcse": mcse,
                "z": bias / mcse if mcse > 0 else np.nan,
                "pct_error": bias / truth * 100 if truth != 0 else np.nan,
                "empirical_sd": float(np.std(estimates, ddof=1))}

    # --- fixed effects -----------------------------------------------------
    for k in range(result.q):
        estimates = draws[f"beta_{k}"].to_numpy()
        ses = draws[f"se_beta_{k}"].to_numpy()
        truth = float(result.beta_true[k])

        row = moments(estimates, truth)
        row["mean_analytic_se"] = float(np.mean(ses))
        row["se_ratio"] = row["mean_analytic_se"] / row["empirical_sd"]
        for level in levels:
            z_crit = norm.ppf(0.5 + level / 2)
            row[f"cov_{int(round(level * 100))}"] = float(
                np.mean(np.abs(estimates - truth) <= z_crit * ses))
        row["frac_at_zero"] = np.nan
        rows[f"beta_{k}"] = row

    # --- variance components -----------------------------------------------
    # Reported on the tau2 scale because that is what is estimated and what the
    # variance-share decomposition is built from. exp(tau) for reporting is
    # summarise_reml_fit's job, not this function's.
    for p, name in enumerate(result.component_names):
        estimates = draws[f"tau2_{name}"].to_numpy()
        truth = float(result.tau2_true[p])

        row = moments(estimates, truth)
        row["mean_analytic_se"] = np.nan
        row["se_ratio"] = np.nan
        for level in levels:
            row[f"cov_{int(round(level * 100))}"] = np.nan
        row["frac_at_zero"] = float(draws[f"at_zero_{name}"].mean())
        rows[f"tau2_{name}"] = row

    columns = (["truth", "mean", "bias", "mcse", "z", "pct_error",
                "mean_analytic_se", "empirical_sd", "se_ratio"]
               + [f"cov_{int(round(level * 100))}" for level in levels]
               + ["frac_at_zero"])
    out = pd.DataFrame.from_dict(rows, orient="index")[columns]
    out.index.name = "parameter"
    return out


def print_simulation_report(result: SimulationResult,
                            levels: Sequence[float] = (0.68, 0.95)) -> str:
    """Header line plus the summary table, as one printable block.

    The header carries the things that invalidate the table if they are wrong --
    how many fits converged, and whether the REML surface was flat enough that the
    restarts disagreed. A summary table computed from non-converged fits looks
    exactly like one computed from converged fits.
    """
    lines = [f"{result.n_sims} simulations, n = {result.n}, q = {result.q}, "
             f"seed = {result.seed}",
             f"components  : {', '.join(result.component_names)}",
             f"converged   : {result.frac_converged:.1%}",
             *(["MISSPECIFIED: fitted with a different V_known from the generator "
                "-- bias here is the cost of that, not an estimator fault"]
               if result.misspecified else []),
             f"start spread: max {result.draws['start_spread'].max():.2e} "
             f"(should be ~0; large means a flat or multi-modal surface)",
             "",
             summarise_simulation(result, levels).to_string(float_format="%.5f")]
    report = "\n".join(lines)
    print(report)
    return report


# =============================================================================
# Bivariate (joint theta / beta) simulation -- the A8 models
# =============================================================================
#
# The same fake-data recovery check as above, for ``mra.fit_reml_mv``. The
# generative story is the A8 model:
#
#     (d_theta_j, d_beta_j) ~ N2(0, T_d)    one PAIR per design
#     (s_theta_k, s_beta_k) ~ N2(0, T_s)    one pair per site
#     (u_theta_i, u_beta_i) ~ N2(0, T_u)    one pair per row
#     eps ~ N(0, V_known)                   (2n,), the stacked sampling error
#
#     y = X b + [M d_theta ; M d_beta] + ... + eps
#
# and, as for the univariate simulator, the effects are drawn from the FACTOR
# matrices and the Kronecker Gs are built by the fitter, never by the
# simulator. A mistake in how the fitter lays out T kron G therefore shows up
# as a recovery failure rather than cancelling on both sides of the test.
#
# V_known is the full 2n x 2n matrix. Its off-diagonal blocks V^theta-beta are
# the point of the joint model: leaving them out of the FIT (``fit_V_known``)
# while they are present in the DATA is how the "sampling correlation read as
# true correlation" bias of model A8 is measured.


def _validate_components_mv(components: Mapping[str, tuple], n: int
                            ) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Unpack and check ``{level: (factor (n, K), T_true (2, 2))}``.

    ``T_true`` must be symmetric PSD. A SINGULAR T_true is allowed -- that is
    how |rho| = 1 or a zero variance is simulated, which the boundary tests need.
    """
    if not components:
        raise ValueError("components is empty -- the joint model needs at least "
                         "the row level")
    names, factors, Ts = [], [], []
    for name, spec in components.items():
        try:
            factor, T_true = spec
        except (TypeError, ValueError):
            raise ValueError(f"components[{name!r}] must be a (factor, T_true) "
                             f"pair, got {spec!r}") from None
        factor = np.asarray(factor, dtype=float)
        if factor.ndim != 2 or factor.shape[0] != n:
            raise ValueError(f"components[{name!r}] factor must be ({n}, K), got "
                             f"{factor.shape}")
        T_true = np.asarray(T_true, dtype=float)
        if T_true.shape != (2, 2) or not np.allclose(T_true, T_true.T):
            raise ValueError(f"components[{name!r}] T_true must be a symmetric 2 x 2")
        if np.linalg.eigvalsh(T_true).min() < -_PSD_TOL:
            raise ValueError(f"components[{name!r}] T_true is not PSD -- a covariance "
                             "needs |tau_tb| <= tau_t * tau_b")
        names.append(name)
        factors.append(factor)
        Ts.append(T_true)
    return names, factors, Ts


def _psd_root(T: np.ndarray) -> np.ndarray:
    """A matrix R with ``R R' = T`` that also works for a SINGULAR T.

    A Cholesky factor fails at rank one (|rho| = 1), which is exactly the case
    the boundary tests want to simulate, so an eigen-root is used instead.
    """
    w, U = np.linalg.eigh(T)
    return U * np.sqrt(np.clip(w, 0.0, None))


def draw_effects_mv(rng: np.random.Generator, factors: Sequence[np.ndarray],
                    roots: Sequence[np.ndarray], base: np.ndarray | None = None,
                    return_parts: bool = False):
    """Draw one set of (theta, beta) effect PAIRS and add them onto ``base`` (2n,).

    The bivariate counterpart of :func:`draw_effects`, and likewise the one
    place such a draw is made: the simulators and the joint hybrid bootstrap
    (:func:`hybrid_pseudo_data_mv`) share it.

    At each level, K independent pairs are drawn, each ``~ N2(0, T)``: the rows
    of ``Z R'`` with Z a (K, 2) block of standard normals and ``R R' = T``. The
    theta column is mapped onto rows by the factor and stacked on top of the
    beta column, matching the theta-first layout of ``mra.fit_reml_mv``.

    Parameters
    ----------
    factors : sequence of (n, K) arrays
    roots : sequence of (2, 2) arrays
        ``_psd_root(T)`` per level, computed ONCE by the caller. An eigen-root,
        so a singular T (|rho| = 1, or a variance at zero) is drawn without a
        special case: the pairs then lie exactly on a line, or one of the two
        is identically zero.
    base : (2n,) array, optional
        ``X beta``; contributions are added in order, one level at a time, for
        the bit-for-bit reason given in :func:`draw_effects`.
    return_parts : bool
        Also return ``[{"effects": D (K, 2), "contribution": (2n,)}, ...]``.
    """
    n = np.asarray(factors[0]).shape[0]
    y = np.zeros(2 * n) if base is None else base
    parts = []
    for M, R in zip(factors, roots):
        D = rng.standard_normal((M.shape[1], 2)) @ R.T
        contribution = np.concatenate([M @ D[:, 0], M @ D[:, 1]])
        y = y + contribution
        if return_parts:
            parts.append({"effects": D, "contribution": contribution})
    return (y, parts) if return_parts else y


def simulate_y_mv(rng: np.random.Generator, beta_true, X: np.ndarray,
                  V_known: np.ndarray, components: Mapping[str, tuple],
                  return_parts: bool = False):
    """Draw one fake bivariate dataset ``y`` (2n,), theta stacked on beta.

    Parameters
    ----------
    rng : np.random.Generator
    beta_true : (p_theta + p_beta,)
        True fixed effects in the column order of X.
    X : (2n, p_theta + p_beta)
        Block-diagonal design, as ``mra.fit_reml_mv`` wants it.
    V_known : (2n, 2n)
        The stacked sampling covariance, INCLUDING its theta-beta blocks.
    components : mapping of str -> (factor (n, K), T_true (2, 2))
        ``T_true`` is the true COVARIANCE of the level's (theta, beta) effect
        pair (unlike the univariate simulator, which takes an SD -- a 2 x 2
        has no single "SD").
    return_parts : bool
        Also return the realised effect pairs and eps.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0] // 2
    beta_true = np.atleast_1d(np.asarray(beta_true, dtype=float))
    names, factors, Ts = _validate_components_mv(components, n)

    # K independent PAIRS per level, each ~ N2(0, T), via the function the
    # joint hybrid bootstrap also uses.
    y, effect_parts = draw_effects_mv(rng, factors, [_psd_root(T) for T in Ts],
                                      base=X @ beta_true, return_parts=True)
    parts = dict(zip(names, effect_parts)) if return_parts else {}
    eps = _known_error_sampler(V_known)(rng)
    y = y + eps
    if return_parts:
        parts["eps"] = eps
        return y, parts
    return y


@dataclass
class MVSimulationResult:
    """One bivariate simulation run and what it was run against.

    ``draws`` holds one row per simulation: ``beta_<k>``, ``se_beta_<k>``, and
    per level ``T_<level>_tt``, ``T_<level>_bb``, ``T_<level>_tb``,
    ``rho_<level>``, ``at_zero_<level>`` (either variance at 0) and
    ``rho_at_bound_<level>``; plus ``rho_tot``, ``converged``, ``nll``.
    """
    draws: pd.DataFrame
    beta_true: np.ndarray
    T_true: dict
    correlated: tuple
    X: np.ndarray
    V_known: np.ndarray
    n_sims: int
    seed: int
    fit_V_known: np.ndarray | None = None
    fit_kwargs: dict = field(default_factory=dict)

    @property
    def misspecified(self) -> bool:
        """True if the fit was given a different V_known from the generator."""
        return (self.fit_V_known is not None
                and not np.array_equal(self.fit_V_known, self.V_known))

    @property
    def rho_tot_true(self) -> float:
        tot = sum(self.T_true.values())
        return float(tot[0, 1] / np.sqrt(tot[0, 0] * tot[1, 1]))


def simulate_reml_mv_fits(beta_true, X: np.ndarray, V_known: np.ndarray,
                          components: Mapping[str, tuple],
                          correlated: Sequence[str] = ("row",), n_sims: int = 500,
                          seed: int = 20261008, fit_V_known: np.ndarray | None = None,
                          fit_kwargs: dict | None = None,
                          progress: bool = True) -> MVSimulationResult:
    """Generate ``n_sims`` bivariate datasets, refit each with ``mra.fit_reml_mv``.

    The bivariate analogue of :func:`simulate_reml_fits`; see it for the
    meaning of every shared argument. ``correlated`` is passed to the fitter:
    it says which levels' cross terms are ESTIMATED. A level whose ``T_true``
    has a non-zero cross term but which is not in ``correlated`` is a
    misspecified fit, which is legitimate to simulate but is the caller's
    choice.

    ``fit_V_known`` is the misspecification switch for the joint model's main
    worry: pass ``V_known`` with its off-diagonal blocks set to zero, and the
    sampling correlation the fit was not told about has nowhere to go except
    the estimated rho, which is then pulled towards it.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    n = N // 2
    V_known = np.asarray(V_known, dtype=float)
    if V_known.shape != (N, N):
        raise ValueError(f"V_known must be {N} x {N}, got {V_known.shape}")
    fit_V_known = V_known if fit_V_known is None else np.asarray(fit_V_known, float)
    beta_true = np.atleast_1d(np.asarray(beta_true, dtype=float))
    if beta_true.size != X.shape[1]:
        raise ValueError(f"beta_true has {beta_true.size} entries but X has "
                         f"{X.shape[1]} columns")
    names, factors, Ts = _validate_components_mv(components, n)
    # Univariate n x n patterns; the fitter builds the Kronecker blocks.
    level_Gs = {name: M @ M.T for name, M in zip(names, factors)}
    roots = [_psd_root(T) for T in Ts]
    fit_kwargs = dict(fit_kwargs or {})

    # Factorised once, as in simulate_reml_fits.
    draw_eps = _known_error_sampler(V_known)
    rng = np.random.default_rng(seed)
    records = []
    report_every = max(n_sims // 10, 1)
    for sim in range(n_sims):
        y = draw_effects_mv(rng, factors, roots, base=X @ beta_true)
        y = y + draw_eps(rng)

        fit = mra.fit_reml_mv(y, X, fit_V_known, level_Gs, correlated, **fit_kwargs)
        rec = {"converged": fit["converged"], "nll": fit["nll"],
               "rho_tot": fit["rho_tot"]}
        for k in range(X.shape[1]):
            rec[f"beta_{k}"] = float(fit["beta"][k])
            rec[f"se_beta_{k}"] = float(fit["se_beta"][k])
        for name in names:
            T = fit["T"][name]
            rec[f"T_{name}_tt"], rec[f"T_{name}_bb"], rec[f"T_{name}_tb"] = (
                float(T[0, 0]), float(T[1, 1]), float(T[0, 1]))
            rec[f"rho_{name}"] = fit["rho"][name]
            rec[f"at_zero_{name}"] = any(fit["at_zero"][name])
            rec[f"rho_at_bound_{name}"] = fit["rho_at_bound"][name]
        records.append(rec)
        if progress and (sim + 1) % report_every == 0:
            print(f"[simulate_reml_mv_fits] {sim + 1}/{n_sims}", flush=True)

    return MVSimulationResult(
        draws=pd.DataFrame(records), beta_true=beta_true,
        T_true=dict(zip(names, Ts)), correlated=tuple(correlated), X=X,
        V_known=V_known, n_sims=n_sims, seed=seed, fit_V_known=fit_V_known,
        fit_kwargs=fit_kwargs)


def summarise_simulation_mv(result: MVSimulationResult) -> pd.DataFrame:
    """Recovery diagnostics for a bivariate run, one row per parameter.

    Columns: ``truth``, ``mean``, ``bias``, ``mcse``, ``z`` (= bias / mcse),
    ``n_used`` and ``frac_boundary``.

    rho is summarised over the simulations where it is DEFINED (both variances
    non-zero); ``n_used`` says how many that was. ``frac_boundary`` is, for a
    T entry, the fraction of fits with either variance of that level at zero,
    and for rho the fraction with |rho| at the bound. These replace the
    coverage columns of the univariate summary: fit_reml_mv, like fit_reml,
    produces no closed-form SE for a variance parameter, so there is nothing
    honest to cover with.
    """
    d = result.draws
    rows = {}

    def moments(est: np.ndarray, truth: float) -> dict:
        est = est[np.isfinite(est)]
        m = float(np.mean(est)) if est.size else np.nan
        mcse = float(np.std(est, ddof=1) / np.sqrt(est.size)) if est.size > 1 else np.nan
        bias = m - truth
        return {"truth": truth, "mean": m, "bias": bias, "mcse": mcse,
                "z": bias / mcse if mcse and mcse > 0 else np.nan,
                "n_used": int(est.size)}

    for k in range(result.X.shape[1]):
        rows[f"beta_{k}"] = moments(d[f"beta_{k}"].to_numpy(float),
                                    float(result.beta_true[k])) | {"frac_boundary": np.nan}
    for name, T in result.T_true.items():
        zero = float(d[f"at_zero_{name}"].mean())
        for key, val in (("tt", T[0, 0]), ("bb", T[1, 1]), ("tb", T[0, 1])):
            rows[f"T_{name}_{key}"] = (moments(d[f"T_{name}_{key}"].to_numpy(float),
                                               float(val)) | {"frac_boundary": zero})
        if name in result.correlated and T[0, 0] > 0 and T[1, 1] > 0:
            r_true = float(T[0, 1] / np.sqrt(T[0, 0] * T[1, 1]))
            rows[f"rho_{name}"] = (moments(d[f"rho_{name}"].to_numpy(float), r_true)
                                   | {"frac_boundary":
                                      float(d[f"rho_at_bound_{name}"].mean())})
    rows["rho_tot"] = (moments(d["rho_tot"].to_numpy(float), result.rho_tot_true)
                       | {"frac_boundary": np.nan})
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "parameter"
    return out[["truth", "mean", "bias", "mcse", "z", "n_used", "frac_boundary"]]


# =============================================================================
# The hybrid bootstrap -- pseudo-data carrying ONE round of sampling noise
# =============================================================================
#
# THE PROBLEM IT SOLVES
# ---------------------
# The record-only outer bootstrap refits each record-resampling replicate
# y^(r) with V_known held fixed. Write a replicate as
#
#     y_i^(r) = (mu + zeta_i)  +  eps_i  +  delta_i^(r)
#               true, fixed       the real    the record
#                                 analysis    resample
#
# V_known = Cov_r(y^(r)) measures the size of delta, which is the plug-in for
# the size of eps. The full-sample fit sees ONE round of noise (eps) and is
# told about one, so it is right. A replicate carries TWO (eps + delta) and is
# still told about one, so REML has nowhere to put the other except tau^2:
#
#     E[tau2^(r)] ~ tau2 + (a weighted mean sampling variance)
#
# The replicate means m^(r) are fine -- the frozen eps only moves their centre
# -- but every replicate variance component, and every correlation built from
# them, is inflated.
#
# THE FIX
# -------
# Build pseudo-data that follows the fitted model with ONE round of noise:
#
#     y*^(r) = X beta_hat  +  zeta*^(r)  +  e^(r)
#
#     zeta*^(r)  NEW true effects, drawn from the fitted variance components
#                with draw_effects / draw_effects_mv -- the parametric part
#     e^(r)      y^(r) - ybar, the record-resample DEVIATIONS -- the
#                non-parametric part. Taken from the real replicate cloud, so
#                every dependence in it (the dense fixed-record-set block, the
#                3-/5-storey coupling at a site, the theta-beta cross block) is
#                carried without being modelled.
#
# Then Var(y*) = sum_l tau2_l G_l + V_known, exactly the fitted model, so the
# replicate variance components centre on the fitted ones and their spread is
# an honest picture of how well they are known. This is a semi-parametric
# bootstrap (Efron & Tibshirani Ch. 6, ~pp. 53-56, approximate).
#
# These are pure-array functions: fragility_data_models builds their inputs,
# and the scheme-comparison study below uses the same functions, so what is
# validated is what is run.


def centre_replicates(Y: np.ndarray) -> np.ndarray:
    """``e^(r) = y^(r) - ybar``: the replicate cloud as deviations from its own row means.

    The ONE definition of the hybrid bootstrap's sampling noise.

    Centred on the cloud MEAN ``ybar``, not on the point estimate ``y_hat``.
    The cloud sits at ``y_hat + bias`` (the bootstrap bias, plus the plug-in
    offset for the moment-fit arms), so subtracting ``y_hat`` would leave that
    bias in every e^(r) as a fixed offset per row. It would shift every m*, and
    because it differs from row to row it would be read as extra between-row
    variance. Subtracting ``ybar`` gives noise that averages exactly zero.

    The same choice makes a bias-corrected (``_bc``) contrast cost nothing: it
    adds one constant per row to every replicate, the row mean moves by the
    same constant, and the difference is unchanged.

    The mean is over ALL the columns of ``Y``. Pass the whole cloud and slice
    afterwards (as :func:`hybrid_pseudo_data` does), so that replicate r does
    not depend on how many replicates a run happens to use, and so the centring
    matches ``V_known``, which is the covariance about this same full mean.

    One small-print point. ``V_known`` divides by k - 1 and the mean square of
    these deviations divides by k, so the noise has variance ``(1 - 1/k) V_known``
    -- short by one part in k. At the study's k = 5000 that is 0.02% and is
    left alone, as the task defines ``e = y - ybar``. It is only visible when a
    cloud is small, which is why the scheme-comparison study below simulates a
    cloud much larger than the number of replicates it refits.

    Parameters
    ----------
    Y : (n, k) array -- one row per study, one column per replicate.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be (n, k): rows are studies, columns are "
                         f"replicates; got shape {Y.shape}")
    return Y - Y.mean(axis=1, keepdims=True)


def _hybrid_noise(Y: np.ndarray, n_boot: int | None) -> np.ndarray:
    """The first ``n_boot`` columns of the centred cloud (all of them if None)."""
    E = centre_replicates(Y)
    k = E.shape[1]
    n_boot = k if n_boot is None else int(n_boot)
    if not 1 <= n_boot <= k:
        raise ValueError(f"n_boot must be between 1 and the number of replicates "
                         f"({k}), got {n_boot}")
    return E[:, :n_boot]


def hybrid_pseudo_data(Y: np.ndarray, fitted: np.ndarray,
                       components: Mapping[str, tuple], *, seed: int,
                       n_boot: int | None = None) -> np.ndarray:
    """Hybrid-bootstrap pseudo-data for a univariate model: an (n, n_boot) matrix.

    Column r is ``fitted + zeta*^(r) + e^(r)`` -- see the block comment above.

    Parameters
    ----------
    Y : (n, k)
        The record-resampling replicate cloud of the contrast.
    fitted : (n,)
        ``X beta_hat`` from the full-sample fit.
    components : mapping of str -> (factor (n, K), tau_hat)
        The SAME mapping format the simulator takes, and the one
        ``fragility_data_models.build_components`` returns, with the FITTED
        standard deviations in place of the true ones. A ``tau_hat`` of zero
        (a component on its boundary) simply draws nothing at that level.
    seed : int
        Seeds the effect draws. One generator, used replicate by replicate and
        level by level, so the same inputs and seed give the same matrix, and
        the first m columns do not depend on ``n_boot``.
    n_boot : int, optional
        Use the first ``n_boot`` replicates. Default: all of them.

    Notes
    -----
    The noise is NEVER a Gaussian draw from V_known: it is the actual cloud,
    whose shape (skewness, heavy tails) and dependence are then carried as they
    are. And V_known is never re-estimated here; that would be a double
    bootstrap (Efron & Tibshirani Ch. 25).
    """
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[0]
    fitted = np.asarray(fitted, dtype=float).ravel()
    if fitted.size != n:
        raise ValueError(f"fitted has {fitted.size} entries but Y has {n} rows")
    _, factors, taus = _validate_components(components, n)
    E = _hybrid_noise(Y, n_boot)

    rng = np.random.default_rng(seed)
    out = np.empty_like(E)
    for j in range(E.shape[1]):
        out[:, j] = draw_effects(rng, factors, taus, base=fitted) + E[:, j]
    return out


def hybrid_pseudo_data_mv(Y: np.ndarray, fitted: np.ndarray,
                          components: Mapping[str, tuple], *, seed: int,
                          n_boot: int | None = None) -> np.ndarray:
    """Hybrid-bootstrap pseudo-data for the joint model: a (2n, n_boot) matrix.

    The bivariate counterpart of :func:`hybrid_pseudo_data`; see it for the
    shared arguments.

    Parameters
    ----------
    Y : (2n, k)
        The STACKED replicate cloud, theta block on top of the beta block.
        Column r holds the same resample in both halves, so e^(r) carries the
        theta-beta sampling correlation.
    fitted : (2n,)
    components : mapping of str -> (factor (n, K), T_hat (2, 2))
        The fitted 2x2 covariance per level. A singular ``T_hat`` (|rho| = 1,
        or a variance snapped to zero) is fine: it is drawn through
        :func:`_psd_root`, exactly as the simulator draws a singular truth.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.shape[0] % 2:
        raise ValueError(f"Y must stack two outcomes of equal length, got "
                         f"{Y.shape[0]} rows")
    n = Y.shape[0] // 2
    fitted = np.asarray(fitted, dtype=float).ravel()
    if fitted.size != 2 * n:
        raise ValueError(f"fitted has {fitted.size} entries but Y has {2 * n} rows")
    _, factors, Ts = _validate_components_mv(components, n)
    # Factorised once: the T's are fixed at their full-sample values for the
    # whole bootstrap.
    roots = [_psd_root(T) for T in Ts]
    E = _hybrid_noise(Y, n_boot)

    rng = np.random.default_rng(seed)
    out = np.empty_like(E)
    for j in range(E.shape[1]):
        out[:, j] = draw_effects_mv(rng, factors, roots, base=fitted) + E[:, j]
    return out


# =============================================================================
# Comparing the bootstrap schemes on fake data
# =============================================================================
#
# The recovery study for the BOOTSTRAP rather than for the estimator. Each
# simulated "study" gets what the real one has:
#
#     y_hat      one draw of the data from known variance components
#     a cloud    n_boot columns y_hat + delta^(r), delta ~ N(0, V_known),
#                standing in for the record resamples
#
# It is fitted once, then bootstrapped under each scheme, and for every
# variance parameter two things are recorded: whether the replicate MEAN sits
# on the full-sample estimate (the inflation check), and whether the
# percentile interval covers the TRUTH (the coverage check).

BOOTSTRAP_SCHEMES = ("record", "hybrid")


@dataclass
class BootstrapSchemeResult:
    """One scheme-comparison run.

    ``draws`` has one row per simulated study. For every parameter ``p`` it
    holds ``p|hat`` (the full-sample estimate) and, per scheme ``s``,
    ``p|s|mean``, ``p|s|lo``, ``p|s|hi`` (the replicate mean and percentile
    interval over the DEFINED replicates), ``p|s|frac_zero`` and
    ``p|s|frac_undefined``.
    """
    draws: pd.DataFrame
    truth: dict
    schemes: tuple
    n_sims: int
    n_boot: int
    seed: int
    alpha: float
    estimate_V: bool
    mean_v: float                 # mean diagonal of V_known: the expected inflation
    n_cloud: int = 0              # columns in each simulated replicate cloud


def _scheme_columns(prefix: str, reps: np.ndarray, alpha: float) -> dict:
    """Replicate mean, percentile interval and boundary fractions for one parameter."""
    reps = np.asarray(reps, dtype=float)
    ok = reps[np.isfinite(reps)]
    lo, hi = (np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if ok.size > 1
              else (np.nan, np.nan))
    return {f"{prefix}|mean": float(ok.mean()) if ok.size else np.nan,
            f"{prefix}|lo": float(lo), f"{prefix}|hi": float(hi),
            f"{prefix}|frac_zero": float(np.mean(ok == 0.0)) if ok.size else np.nan,
            f"{prefix}|frac_undefined": float(1 - ok.size / reps.size)}


def _check_schemes(schemes) -> tuple:
    schemes = tuple(schemes)
    unknown = sorted(set(schemes) - set(BOOTSTRAP_SCHEMES))
    if unknown:
        raise ValueError(f"unknown bootstrap scheme(s) {unknown}; expected a "
                         f"subset of {BOOTSTRAP_SCHEMES}")
    return schemes


def _check_cloud_size(n_boot: int, n_cloud: int) -> None:
    if not 1 <= n_boot <= n_cloud:
        raise ValueError(f"n_boot ({n_boot}) must be between 1 and n_cloud ({n_cloud}): "
                         "the replicates refitted are the first n_boot columns of "
                         "the simulated cloud")


def simulate_bootstrap_schemes(beta_true, X: np.ndarray, V_known: np.ndarray,
                               components: Mapping[str, tuple], *, n_sims: int,
                               n_boot: int, n_cloud: int = 1000,
                               seed: int = 20261013,
                               estimate_V: bool = False, alpha: float = 0.05,
                               schemes: Sequence[str] = BOOTSTRAP_SCHEMES,
                               fit_kwargs: dict | None = None,
                               boot_kwargs: dict | None = None,
                               progress: bool = True) -> BootstrapSchemeResult:
    """Simulate studies, bootstrap each under every scheme, record bias and coverage.

    The univariate study; :func:`simulate_bootstrap_schemes_mv` is the joint
    one. Arguments shared with :func:`simulate_reml_fits` mean the same here.

    Parameters
    ----------
    n_sims : int
        Simulated studies. Coverage is a binomial proportion, so its Monte
        Carlo SE near 0.95 is about ``0.22 / sqrt(n_sims)``.
    n_boot : int
        Replicates REFITTED per study and per scheme. Required, with no
        default, so the cost of a run is always a visible choice: the run makes
        ``n_sims * len(schemes) * n_boot`` fits.
    n_cloud : int
        Columns in each study's simulated replicate cloud; the refits use the
        first ``n_boot`` of them. Kept much larger than ``n_boot``, as on the
        real data (k = 5000, of which a run may refit a few hundred), and for
        a reason that matters: deviations from the mean of k columns have
        variance ``(1 - 1/k) V``, not ``V``. With a cloud of only ``n_boot``
        columns the hybrid noise would be short by ``v / n_boot``, and that
        artefact of the simulation would show up as a small negative bias in
        tau^2. At k = 1000 it is 0.1% of v; on the real data, 0.02%. Drawing
        the extra columns costs no fits.
    estimate_V : bool
        False (default): the fits are given the TRUE ``V_known``, which
        isolates the bootstrap scheme from the error in estimating V. True:
        they are given the sample covariance of that study's cloud, as on the
        real data (where k / n is far larger than a simulation can afford).
    alpha : float
        Percentile intervals are at ``(alpha/2, 1 - alpha/2)``.
    fit_kwargs, boot_kwargs : dict, optional
        Forwarded to ``mra.fit_reml`` for the full-sample fit (default
        cholesky, 2 starts) and for each replicate (default cholesky, 1 start,
        as ``fragility_data_models.bootstrap_fits`` uses).

    Notes
    -----
    Parameters recorded: ``m`` (the intercept), one per variance component, and
    ``T_sq`` (their total). The cover-the-truth check for ``m`` is against the
    SUPERPOPULATION truth, so the record-only interval, which is conditional on
    the effects drawn, is expected to under-cover it; that is the two schemes
    answering different questions, not a fault in either.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    V_known = np.asarray(V_known, dtype=float)
    beta_true = np.atleast_1d(np.asarray(beta_true, dtype=float))
    names, factors, taus = _validate_components(components, n)
    Gs = [M @ M.T for M in factors]
    schemes = _check_schemes(schemes)
    _check_cloud_size(n_boot, n_cloud)
    fit_kwargs = {"method": "cholesky", "n_starts": 2} | dict(fit_kwargs or {})
    boot_kwargs = {"method": "cholesky", "n_starts": 1} | dict(boot_kwargs or {})

    truth = {"m": float(beta_true[0]),
             **{nm: float(t ** 2) for nm, t in zip(names, taus)},
             "T_sq": float(np.sum(taus ** 2))}
    draw_eps = _known_error_sampler(V_known)
    rng = np.random.default_rng(seed)
    records = []
    report_every = max(n_sims // 10, 1)
    for sim in range(n_sims):
        # One study: the data, then its stand-in for the record resamples.
        y_hat = draw_effects(rng, factors, taus, base=X @ beta_true) + draw_eps(rng)
        cloud = y_hat[:, None] + np.column_stack([draw_eps(rng) for _ in range(n_cloud)])
        V_fit = mra.bootstrap_cov(cloud, rowvar=True) if estimate_V else V_known
        full = mra.fit_reml(y_hat, X, V_fit, Gs, names=names, **fit_kwargs)
        rec = {"m|hat": float(full["beta"][0]), "T_sq|hat": float(np.sum(full["tau2"])),
               **{f"{nm}|hat": float(t) for nm, t in zip(names, full["tau2"])}}

        # One seed per study for the hybrid draws, taken from the main stream
        # so the whole run is reproduced by (seed, n_sims, n_boot).
        boot_seed = int(rng.integers(2 ** 31 - 1))
        for scheme in schemes:
            if scheme == "record":
                Y_r = cloud[:, :n_boot]
            else:
                fitted_comps = {nm: (M, float(np.sqrt(t)))
                                for nm, M, t in zip(names, factors, full["tau2"])}
                Y_r = hybrid_pseudo_data(cloud, X @ full["beta"], fitted_comps,
                                         seed=boot_seed, n_boot=n_boot)
            reps = {p: np.empty(n_boot) for p in truth}
            for j in range(n_boot):
                f = mra.fit_reml(Y_r[:, j], X, V_fit, Gs, names=names, **boot_kwargs)
                reps["m"][j] = f["beta"][0]
                reps["T_sq"][j] = np.sum(f["tau2"])
                for nm, t in zip(names, f["tau2"]):
                    reps[nm][j] = t
            for p, values in reps.items():
                rec |= _scheme_columns(f"{p}|{scheme}", values, alpha)
        records.append(rec)
        if progress and (sim + 1) % report_every == 0:
            print(f"[simulate_bootstrap_schemes] {sim + 1}/{n_sims}", flush=True)

    return BootstrapSchemeResult(
        draws=pd.DataFrame(records), truth=truth, schemes=schemes, n_sims=n_sims,
        n_boot=n_boot, seed=seed, alpha=alpha, estimate_V=estimate_V,
        mean_v=float(np.mean(np.diag(V_known))), n_cloud=n_cloud)


def simulate_bootstrap_schemes_mv(beta_true, X: np.ndarray, V_known: np.ndarray,
                                  components: Mapping[str, tuple],
                                  correlated: Sequence[str] = ("row",), *,
                                  n_sims: int, n_boot: int, n_cloud: int = 1000,
                                  seed: int = 20261013, estimate_V: bool = False, alpha: float = 0.05,
                                  schemes: Sequence[str] = BOOTSTRAP_SCHEMES,
                                  fit_kwargs: dict | None = None,
                                  progress: bool = True) -> BootstrapSchemeResult:
    """The joint (theta, beta) version of :func:`simulate_bootstrap_schemes`.

    ``components`` is ``{level: (factor (n, K), T_true (2, 2))}`` and
    ``correlated`` is passed to ``mra.fit_reml_mv``, as in
    :func:`simulate_reml_mv_fits`. Each replicate is warm-started from the
    full-sample T with one start, as ``bootstrap_joint_fits`` does.

    Parameters recorded: ``m_theta``, ``m_beta``, ``T_<level>_tt/_bb/_tb`` and
    ``T_total_*``, ``rho_<level>`` for each correlated level, and ``rho_tot``.
    A rho is NaN in a replicate where a variance is zero; its mean and interval
    are over the defined replicates and ``frac_undefined`` says how many were
    not.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    n = N // 2
    V_known = np.asarray(V_known, dtype=float)
    beta_true = np.atleast_1d(np.asarray(beta_true, dtype=float))
    names, factors, Ts = _validate_components_mv(components, n)
    level_Gs = {nm: M @ M.T for nm, M in zip(names, factors)}
    roots = [_psd_root(T) for T in Ts]
    correlated = tuple(correlated)
    schemes = _check_schemes(schemes)
    _check_cloud_size(n_boot, n_cloud)
    fit_kwargs = dict(fit_kwargs or {})
    theta_cols, beta_cols = mra._mv_outcome_columns(X, n)
    i_t, i_b = int(theta_cols[0]), int(beta_cols[0])

    def corr(T):
        d = np.diag(T)
        return float(T[0, 1] / np.sqrt(d[0] * d[1])) if np.all(d > 0) else np.nan

    def flatten(fit) -> dict:
        """Every recorded parameter of one fit, by name."""
        out = {"m_theta": float(fit["beta"][i_t]), "m_beta": float(fit["beta"][i_b]),
               "rho_tot": fit["rho_tot"]}
        for lev, T in [*fit["T"].items(), ("total", fit["T_total"])]:
            out |= {f"T_{lev}_tt": float(T[0, 0]), f"T_{lev}_bb": float(T[1, 1]),
                    f"T_{lev}_tb": float(T[0, 1])}
        out |= {f"rho_{lev}": fit["rho"][lev] for lev in correlated}
        return out

    T_tot = sum(Ts)
    truth = {"m_theta": float(beta_true[i_t]), "m_beta": float(beta_true[i_b]),
             "rho_tot": corr(T_tot)}
    for lev, T in [*zip(names, Ts), ("total", T_tot)]:
        truth |= {f"T_{lev}_tt": float(T[0, 0]), f"T_{lev}_bb": float(T[1, 1]),
                  f"T_{lev}_tb": float(T[0, 1])}
    truth |= {f"rho_{lev}": corr(T) for lev, T in zip(names, Ts) if lev in correlated}

    draw_eps = _known_error_sampler(V_known)
    rng = np.random.default_rng(seed)
    records = []
    report_every = max(n_sims // 10, 1)
    for sim in range(n_sims):
        y_hat = draw_effects_mv(rng, factors, roots, base=X @ beta_true) + draw_eps(rng)
        cloud = y_hat[:, None] + np.column_stack([draw_eps(rng) for _ in range(n_cloud)])
        V_fit = mra.bootstrap_cov(cloud, rowvar=True) if estimate_V else V_known
        full = mra.fit_reml_mv(y_hat, X, V_fit, level_Gs, correlated, **fit_kwargs)
        rec = {f"{p}|hat": v for p, v in flatten(full).items()}

        boot_seed = int(rng.integers(2 ** 31 - 1))
        for scheme in schemes:
            if scheme == "record":
                Y_r = cloud[:, :n_boot]
            else:
                fitted_comps = {nm: (M, full["T"][nm]) for nm, M in zip(names, factors)}
                Y_r = hybrid_pseudo_data_mv(cloud, X @ full["beta"], fitted_comps,
                                            seed=boot_seed, n_boot=n_boot)
            reps = {p: np.empty(n_boot) for p in truth}
            for j in range(n_boot):
                f = mra.fit_reml_mv(Y_r[:, j], X, V_fit, level_Gs, correlated,
                                    n_starts=1, T0=full["T"])
                for p, v in flatten(f).items():
                    reps[p][j] = v
            for p, values in reps.items():
                rec |= _scheme_columns(f"{p}|{scheme}", values, alpha)
        records.append(rec)
        if progress and (sim + 1) % report_every == 0:
            print(f"[simulate_bootstrap_schemes_mv] {sim + 1}/{n_sims}", flush=True)

    return BootstrapSchemeResult(
        draws=pd.DataFrame(records), truth=truth, schemes=schemes, n_sims=n_sims,
        n_boot=n_boot, seed=seed, alpha=alpha, estimate_V=estimate_V,
        mean_v=float(np.mean(np.diag(V_known))), n_cloud=n_cloud)


def summarise_bootstrap_schemes(result: BootstrapSchemeResult) -> pd.DataFrame:
    """One row per (parameter, scheme): is the replicate mean inflated, and does the interval cover?

    Columns
    -------
    truth        the value the data were simulated from
    mean_hat     mean full-sample estimate across the simulated studies
    mean_rep     mean, across studies, of the replicate mean
    excess       ``mean_rep - mean_hat`` computed study by study and averaged:
                 how far the replicates sit from the estimate they are supposed
                 to describe. About the mean sampling variance for a variance
                 component under the record-only scheme; about zero under the
                 hybrid scheme.
    mcse, z      Monte Carlo SE of ``excess`` and ``excess / mcse``
    coverage     fraction of studies whose percentile interval contains the
                 truth, with its binomial SE in ``cov_se``
    mean_width   mean interval width
    frac_zero, frac_undefined
                 mean fraction of replicates exactly at zero / undefined (NaN)
    n_used       studies with a defined interval
    """
    d = result.draws
    rows = {}
    for p, truth in result.truth.items():
        hat = d[f"{p}|hat"].to_numpy(float)
        for s in result.schemes:
            mean = d[f"{p}|{s}|mean"].to_numpy(float)
            lo, hi = d[f"{p}|{s}|lo"].to_numpy(float), d[f"{p}|{s}|hi"].to_numpy(float)
            ok = np.isfinite(mean) & np.isfinite(hat) & np.isfinite(lo)
            diff = (mean - hat)[ok]
            n_used = int(ok.sum())
            mcse = (float(np.std(diff, ddof=1) / np.sqrt(n_used)) if n_used > 1
                    else np.nan)
            excess = float(np.mean(diff)) if n_used else np.nan
            cov = (float(np.mean((lo[ok] <= truth) & (truth <= hi[ok])))
                   if n_used and np.isfinite(truth) else np.nan)
            rows[(p, s)] = {
                "truth": truth,
                "mean_hat": float(np.mean(hat[ok])) if n_used else np.nan,
                "mean_rep": float(np.mean(mean[ok])) if n_used else np.nan,
                "excess": excess, "mcse": mcse,
                "z": excess / mcse if mcse and mcse > 0 else np.nan,
                "coverage": cov,
                "cov_se": (float(np.sqrt(cov * (1 - cov) / n_used))
                           if n_used and np.isfinite(cov) else np.nan),
                "mean_width": float(np.mean((hi - lo)[ok])) if n_used else np.nan,
                "frac_zero": float(d[f"{p}|{s}|frac_zero"].mean()),
                "frac_undefined": float(d[f"{p}|{s}|frac_undefined"].mean()),
                "n_used": n_used}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["parameter", "scheme"])
    return out
