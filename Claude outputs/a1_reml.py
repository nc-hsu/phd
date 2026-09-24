"""
REML fitting for the A1 family of meta-analysis models (SS vs MSA-FX).
=======================================================================

Supersedes ``a1c_reml.py``, which only handled A1c.

WHAT THIS MODULE IS FOR
-----------------------
All three A1 fits are the *same* model with a different number of variance
components, so they are all served by one fitting routine here:

    A1a  y_i = m1                            + u_i + eps_i      (2-level)
    A1b  y_i = m1 + d_{g[i]}                 + u_i + eps_i      (3-level)
    A1c  y_i = m1 + d_{g[i]} + s_{sigma[i]}  + u_i + eps_i      (crossed)

with

    d_j      ~ N(0, tau_d^2)     design-group effect  (J of them)
    s_k      ~ N(0, tau_s^2)     site effect          (K of them)
    u_i      ~ N(0, tau_u^2)     row-level / interaction
    eps_i    ~ N(0, ...)         sampling error, variance KNOWN from bootstrap

Writing Z (n x J) and S (n x K) for the design and site indicator matrices, the
marginal covariance of y is

    Sigma = V_known + tau_d^2 * Z Z' + tau_s^2 * S S' + tau_u^2 * I

Every term after V_known is "a variance component times a known n x n matrix".
That is the structure this module exploits: you hand it a LIST of those known
matrices (``Gs``) and it estimates one variance component per entry. Dropping a
component is done by leaving its matrix out of the list -- never by mangling Z
or S.

    A1a   Gs = [I]                  -> tau_u^2
    A1b   Gs = [ZZt, I]             -> tau_d^2, tau_u^2
    A1c   Gs = [ZZt, SSt, I]        -> tau_d^2, tau_s^2, tau_u^2

WHY NOT JUST PASS A DEGENERATE Z OR S?
--------------------------------------
Two tempting hacks both fail:

  * one site for everybody  -> S S' = matrix of ones = a global random
    intercept, which is confounded with the fixed intercept m1. tau_s^2 is then
    unidentified and the optimiser wanders.
  * every row its own site  -> S S' = I, perfectly aliased with tau_u^2 * I.
    Only the sum of the two is identified.

Hence the list-of-components design.

THE TWO "KNOWN" INPUTS
----------------------
``V_known`` is the part of Sigma that does NOT depend on any estimated
parameter. What goes in it differs between A1a and A1b/A1c:

  A1a:      V_known = diag(v1)            where v1_i = Var(a_i - b_{g[i]})
            i.e. the COMBINED bootstrap variance of the difference. A1a does
            not model group structure, so the shared-b variance rightly stays
            on the diagonal. This is what makes A1a-REML directly comparable
            to the DerSimonian-Laird fit and to PyMare.

  A1b/A1c:  V_known = diag(v_a) + Z V_bb Z'
            The same total variance, but SPLIT: the SS-arm part v_a stays
            strictly on the diagonal, while the MSA-FX part is promoted to a
            full J x J covariance matrix V_bb and smeared by Z across every row
            that shares a design. That off-diagonal smear is the whole point --
            see the identifiability note below.

    v_a[i]      = Var_r( ln theta_SS_i^(r) )         SS arm only, per ROW
    V_bb[j,j']  = Cov_r( ln theta_FX_j^(r),
                         ln theta_FX_j'^(r) )        MSA-FX arm, per DESIGN

    Consistency (exact, since the SS arm uses different records so the arms are
    independent):   v_a[i] + V_bb[g[i], g[i]] == v1[i]

IDENTIFIABILITY WARNING (read before trusting tau_d)
----------------------------------------------------
Every row in design group j subtracts the SAME estimate b_j, so the sampling
error contributes v^b_j to every within-group off-diagonal cell of Sigma. That
is exactly the same block pattern as tau_d^2 * Z Z'. The two are aliased, and
separable only through the fact that v^b_j varies across j while tau_d^2 does
not -- weak leverage.

The site side carries its own shared term. The 3-storey and 5-storey
structures at a site share records wherever they are analysed at the same stripe
IML, which does happen, so Cov(a_3s,k, a_5s,k) != 0 and V_aa is block-diagonal
by site rather than diagonal. Do not assume the two grouping factors are treated
asymmetrically by the sampling covariance.

Therefore:

  * always pass the FULL V_known for A1b/A1c, built as the complete sample
    covariance of the replicate matrices,
        V_known = Cov_r(a) + Z Cov_r(b) Z'
    rather than from their diagonals. Built that way the code carries no
    assumption about which pairs are coupled.
  * the replicates must share record draws wherever the underlying analyses
    shared records. np.cov can only report the covariance the RESAMPLING SCHEME
    generates -- an independently seeded scheme returns zeros whatever the truth.

ESTIMATION
----------
REML (restricted maximum likelihood) is used rather than DerSimonian-Laird
because DL is a single moment equation and cannot produce more than one
variance component, and because DL's Q statistic presupposes a diagonal
sampling covariance. The REML objective, up to an additive constant,

    -2 l_REML = log|Sigma| + log|X' Sigma^-1 X| + (y - X b)' Sigma^-1 (y - X b)

with b the GLS estimate  b = (X' Sigma^-1 X)^-1 X' Sigma^-1 y,  is minimised
over the variance components. It is indifferent to how many components there
are and to whether Sigma is dense.

References
----------
Borenstein et al. (2009) Ch. 12 (pp. 69-75) two-level random-effects model;
Ch. 14 (pp. 87-95) worked hand calculation.
Gelman & Hill (2007) Ch. 13.5 (~pp. 289-291) crossed / non-nested models;
Ch. 22 (~pp. 487-500) comparing variance components.
Gelman et al. BDA3 Ch. 5.4-5.5 (~pp. 113-124) hierarchical normal model with
known variances.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

# A fitted variance component below this is reported as exactly zero. The
# optimiser works on log(tau^2), so tau^2 = 0 is unreachable -- a component that
# is truly zero shows up as something like 1e-12 with the optimiser crawling off
# to -inf. "Effectively zero" is a legitimate FINDING (no design effect beyond
# noise), so it must be detectable rather than hidden.
ZERO_TOL = 1e-8


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


def build_V_known(v_a: np.ndarray, Z: np.ndarray | None = None,
                  V_bb: np.ndarray | None = None) -> np.ndarray:
    """Assemble the known (parameter-free) part of Sigma.

    Parameters
    ----------
    v_a : ndarray, shape (n,)
        For A1b/A1c: the SS-arm sampling variance per row,
        ``Var_r(ln theta_SS_i^(r))``.
        For A1a: pass the COMBINED variance v1 and leave Z / V_bb as None.
    Z : ndarray, shape (n, J), optional
        Design indicator.
    V_bb : ndarray, shape (J, J), optional
        MSA-FX sampling covariance over designs.

    Returns
    -------
    ndarray, shape (n, n):  diag(v_a)                    if Z/V_bb omitted
                            diag(v_a) + Z V_bb Z'        otherwise
    """
    v_a = np.asarray(v_a, dtype=float).ravel()
    C = np.diag(v_a)
    if (Z is None) != (V_bb is None):
        raise ValueError("pass both Z and V_bb, or neither")
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        V_bb = np.asarray(V_bb, dtype=float)
        if Z.shape[1] != V_bb.shape[0]:
            raise ValueError(
                f"Z has {Z.shape[1]} design columns but V_bb is "
                f"{V_bb.shape[0]} x {V_bb.shape[0]} -- these must match, and "
                "the column order of Z must be the row order of V_bb "
                "(use build_codes).")
        C = C + Z @ V_bb @ Z.T
    return C


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
    """Negative restricted log-likelihood (up to an additive constant).

    Parameters
    ----------
    logpar : ndarray, shape (p,)
        ``log(tau^2)`` for each variance component. Working on the log scale
        keeps the components positive without constrained optimisation -- at
        the cost that exactly zero is unreachable (see ZERO_TOL).
    y : ndarray, shape (n,)
    X : ndarray, shape (n, q)
        Fixed-effects design. Intercept-only for A1: ``np.ones((n, 1))``.
    V_known : ndarray, shape (n, n)
    Gs : sequence of p arrays, each (n, n)

    Returns
    -------
    float :  0.5 * [ log|Sigma| + log|X' Sigma^-1 X| + r' Sigma^-1 r ]

    Notes
    -----
    The middle term is what makes this REML rather than ML. It corrects the
    downward bias in variance components that ML incurs by ignoring the degrees
    of freedom spent estimating the fixed effects. With only one fixed effect
    (the intercept) the correction is modest but not negligible at n = 60.

    Uses a Cholesky factorisation for a numerically stable log-determinant and
    for all solves. A non-PSD Sigma (which the optimiser can propose) returns a
    large finite penalty rather than raising, so the search just backs away.
    """
    tau2 = np.exp(logpar)
    Sigma = _build_sigma(V_known, tau2, Gs)

    try:
        cf = cho_factor(Sigma, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10
    # cho_factor returns (L, lower); diag(L) gives the log-determinant cheaply.
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

    beta = cho_solve(cfx, X.T @ Si_y, check_finite=False)   # GLS estimate
    resid = y - X @ beta
    quad = float(resid @ cho_solve(cf, resid, check_finite=False))

    val = 0.5 * (logdet_Sigma + logdet_XtSiX + quad)
    return val if np.isfinite(val) else 1e10


def fit_reml(y: np.ndarray, X: np.ndarray, V_known: np.ndarray,
             Gs: Sequence[np.ndarray], names: Sequence[str] | None = None,
             n_starts: int = 4, seed: int = 0,
             zero_tol: float = ZERO_TOL, verbose: bool = False) -> dict:
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

    # Starting values: split the total observed variance evenly over the
    # components, then jitter on the log scale for the restarts.
    rng = np.random.default_rng(seed)
    base = max(float(np.var(y)) / max(p, 1), 1e-8)
    starts = [np.log(np.full(p, base))]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(np.log(np.full(p, base)) + rng.normal(0.0, 1.5, p))

    results = []
    for x0 in starts:
        res = minimize(reml_nll, x0, args=(y, X, V_known, Gs),
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
        "Sigma": Sigma,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers -- these just choose the right Gs and V_known
# ---------------------------------------------------------------------------

def fit_A1a(y, v1, **kw) -> dict:
    """Two-level random-effects meta-analysis, REML.

        y_i = m1 + u_i + eps_i,   eps_i ~ N(0, v1_i)

    ``v1`` is the COMBINED bootstrap variance of the difference,
    Var_r(a_i^(r) - b_{g[i]}^(r)) -- i.e. exactly the ``v1_boot`` already used
    for the DerSimonian-Laird fit. This is the fit to compare against DL and
    against PyMare's ``method='REML'``.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    return fit_reml(y, np.ones((n, 1)), np.diag(np.asarray(v1, float).ravel()),
                    [np.eye(n)], names=["tau_u2"], **kw)


def fit_A1b(y, V_known, design_codes, n_designs=None, **kw) -> dict:
    """Three-level model: rows nested in design groups.

        y_i = m1 + d_{g[i]} + u_i + eps_i

    Adds tau_d^2. Pass the FULL ``V_known`` (with Z V_bb Z') -- with a diagonal
    V_known, tau_d^2 will absorb the shared-b sampling covariance and read too
    high.

    Note that A1b alone does NOT answer "does the building or the site matter
    more": its residual tau_u^2 still bundles site effects, design x site
    interaction and row noise together. Use it for the ICC
    tau_d^2 / (tau_d^2 + tau_u^2) and as a stepping stone to A1c.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    Z = make_indicator(design_codes, n_designs)
    return fit_reml(y, np.ones((n, 1)), V_known, [Z @ Z.T, np.eye(n)],
                    names=["tau_d2", "tau_u2"], **kw)


def fit_A1c(y, V_known, design_codes, site_codes,
            n_designs=None, n_sites=None, X=None, **kw) -> dict:
    """Crossed design x site model -- the reportable A1 fit.

        y_i = m1 + d_{g[i]} + s_{sigma[i]} + u_i + eps_i

    Design and site are CROSSED, not nested: a design spans several sites and a
    site hosts several designs. u_i therefore absorbs the design x site
    interaction, and must be kept -- without it the interaction is silently
    credited to whichever of d or s the data happen to favour.

    The comparison of interest is tau_d vs tau_s. Because they are two
    ESTIMATED standard deviations, do not compare the two point estimates: form
    the difference (or ratio) per outer-bootstrap replicate and report its
    interval.

    Pass ``X`` to add covariates (A0b / A3); default is intercept-only.
    """
    y = np.asarray(y, float).ravel()
    n = y.size
    Z = make_indicator(design_codes, n_designs)
    S = make_indicator(site_codes, n_sites)
    if X is None:
        X = np.ones((n, 1))
    return fit_reml(y, X, V_known, [Z @ Z.T, S @ S.T, np.eye(n)],
                    names=["tau_d2", "tau_s2", "tau_u2"], **kw)


def summarise(fit: dict, ratio_units: bool = True) -> str:
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


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n, n_design, n_site = 120, 51, 12

    # --- a partially crossed layout, like the real study -------------------
    design = rng.integers(0, n_design, n)
    site = rng.integers(0, n_site, n)
    design[:n_design] = np.arange(n_design)    # make sure every level is used
    site[:n_site] = np.arange(n_site)

    Z = make_indicator(design, n_design)
    S = make_indicator(site, n_site)

    # --- truth --------------------------------------------------------------
    TD2, TS2, TU2, M1 = 0.04, 0.09, 0.02, -0.25
    d = rng.normal(0, np.sqrt(TD2), n_design)
    s = rng.normal(0, np.sqrt(TS2), n_site)
    u = rng.normal(0, np.sqrt(TU2), n)

    # --- known sampling covariance -----------------------------------------
    v_a = rng.uniform(0.005, 0.02, n)                     # SS arm, per row
    # V_bb built as a one-factor structure  lam lam' + diag(psi), which is how
    # the real thing arises: one common "this replicate drew strong records"
    # factor loading on every design, plus design-specific noise. Loadings are
    # all the same sign, so the off-diagonal correlations are strongly positive
    # -- the signature check_V_bb looks for.
    lam = rng.uniform(0.05, 0.11, n_design)               # common-factor loading
    psi = rng.uniform(0.002, 0.008, n_design)             # design-specific
    V_bb = np.outer(lam, lam) + np.diag(psi)
    V_known = build_V_known(v_a, Z, V_bb)

    check_V_bb(V_bb, n_replicates=1000)

    eps = rng.multivariate_normal(np.zeros(n), V_known)
    y = M1 + Z @ d + S @ s + u + eps

    # the combined per-row variance the A1a fit would be given
    v1 = v_a + np.diag(V_bb)[design]
    check_variance_split(v_a, V_bb, design, v1)

    print("\n=== A1c, full V_known (correct) ===")
    fit_c = fit_A1c(y, V_known, design, site, n_designs=n_design, n_sites=n_site)
    print(summarise(fit_c))
    print(f"  truth: m1 {M1:+.3f}  tau_d {np.sqrt(TD2):.3f}  "
          f"tau_s {np.sqrt(TS2):.3f}  tau_u {np.sqrt(TU2):.3f}")

    print("\n=== A1c, DIAGONAL V_known (contaminated) ===")
    fit_diag = fit_A1c(y, np.diag(np.diag(V_known)), design, site,
                       n_designs=n_design, n_sites=n_site)
    print(summarise(fit_diag))
    # The robust signature of ignoring V_bb is the SE of m1, not tau_d: the
    # shared-b error does not average away over rows, so a diagonal V tells the
    # model it will, and the SE comes out too small. (tau_d is contaminated too,
    # but on any SINGLE draw that is swamped by ordinary sampling scatter -- it
    # only shows up as a bias averaged over many simulations.)
    print(f"  SE(m1):  correct {fit_c['se_beta'][0]:.4f}"
          f"   vs diagonal-V {fit_diag['se_beta'][0]:.4f}"
          f"   -> understated by {fit_c['se_beta'][0] / fit_diag['se_beta'][0]:.2f}x")
    print(f"  tau_d :  correct {fit_c['tau'][0]:.3f}"
          f"   vs diagonal-V {fit_diag['tau'][0]:.3f}   (true {np.sqrt(TD2):.3f})")

    print("\n=== A1b, design only ===")
    print(summarise(fit_A1b(y, V_known, design, n_designs=n_design)))

    print("\n=== A1a, two-level ===")
    print(summarise(fit_A1a(y, v1)))

    # --- ladder step 3 check ------------------------------------------------
    # Feeding A1a's own V_known through the general routine with Gs = [I] must
    # reproduce fit_A1a exactly. This is the regression test that guards the
    # refactor: same model, two code paths, identical answer.
    alt = fit_reml(y, np.ones((n, 1)), np.diag(v1), [np.eye(n)], names=["tau_u2"])
    assert np.allclose(alt["tau2"], fit_A1a(y, v1)["tau2"], rtol=1e-6)
    print("\nladder step-3 identity check: PASSED")
