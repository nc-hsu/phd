"""Fake-data simulation for verifying the REML meta-regression fits.

Why this module exists
----------------------
There is no Python package that fits the crossed design x site model of A1c
(``metafor::rma.mv`` in R does, PyMare does not -- it takes a variance *vector*,
not a matrix). So ``mra.fit_reml`` is hand-rolled, and the only honest way to
believe a hand-rolled estimator is to feed it data whose truth you already know
and check that it hands the truth back.

That is what this module does. It generates ``y`` from the generative story the
roadmap writes down (``admin/analysis_models.md``, A1c), refits it with
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

    y = X @ beta_true
    parts = {}
    for name, M, tau in zip(names, factors, taus):
        z = tau * rng.standard_normal(M.shape[1])
        contribution = M @ z
        y = y + contribution
        if return_parts:
            parts[name] = {"z": z, "contribution": contribution}

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
        y = X @ beta_true
        for M, tau in zip(factors, taus):
            y = y + M @ (tau * rng.standard_normal(M.shape[1]))
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
