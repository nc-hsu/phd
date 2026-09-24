"""
A1c: crossed (design x site) random-effects meta-regression with a KNOWN,
possibly dense, sampling covariance matrix.

    y = X beta + Z d + S s + u + eps
    Sigma = C_known + td2 * Z Z' + ts2 * S S' + tu2 * I

C_known = diag(v_a) + Z V_bb Z'   (known from the stored bootstrap replicates)

Fitted by REML over the three positive variance components with scipy.
"""
import numpy as np
from scipy.optimize import minimize


def make_indicator(codes):
    """codes: integer array of length n with values 0..K-1 -> n x K indicator."""
    codes = np.asarray(codes)
    K = codes.max() + 1
    Z = np.zeros((len(codes), K))
    Z[np.arange(len(codes)), codes] = 1.0
    return Z


def reml_nll(logpar, y, X, C_known, ZZt, SSt):
    """Negative REML log-likelihood. logpar = log of (td2, ts2, tu2)."""
    td2, ts2, tu2 = np.exp(logpar)
    Sigma = C_known + td2 * ZZt + ts2 * SSt + tu2 * np.eye(len(y))

    # Cholesky -> stable logdet and solves
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return 1e10
    logdet_Sigma = 2.0 * np.sum(np.log(np.diag(L)))

    Si_X = np.linalg.solve(Sigma, X)
    Si_y = np.linalg.solve(Sigma, y)
    XtSiX = X.T @ Si_X
    Lx = np.linalg.cholesky(XtSiX)
    logdet_XtSiX = 2.0 * np.sum(np.log(np.diag(Lx)))

    beta = np.linalg.solve(XtSiX, X.T @ Si_y)
    resid = y - X @ beta
    quad = resid @ np.linalg.solve(Sigma, resid)

    return 0.5 * (logdet_Sigma + logdet_XtSiX + quad)


def fit_a1c(y, X, C_known, design, site, x0=None):
    Z = make_indicator(design)
    S = make_indicator(site)
    ZZt, SSt = Z @ Z.T, S @ S.T
    if x0 is None:
        x0 = np.log(np.full(3, np.var(y) / 3 + 1e-6))

    res = minimize(reml_nll, x0, args=(y, X, C_known, ZZt, SSt), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    td2, ts2, tu2 = np.exp(res.x)

    Sigma = C_known + td2 * ZZt + ts2 * SSt + tu2 * np.eye(len(y))
    Si_X = np.linalg.solve(Sigma, X)
    XtSiX = X.T @ Si_X
    beta = np.linalg.solve(XtSiX, X.T @ np.linalg.solve(Sigma, y))
    Vbeta = np.linalg.inv(XtSiX)
    return dict(beta=beta, se_beta=np.sqrt(np.diag(Vbeta)),
                tau_d2=td2, tau_s2=ts2, tau_u2=tu2, converged=res.success, nll=res.fun)


# ----------------------------------------------------------------- demo -----
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n, n_design, n_site = 120, 51, 12

    # partially crossed: each row gets a design and a site
    design = rng.integers(0, n_design, n)
    site = rng.integers(0, n_site, n)
    design[:n_design] = np.arange(n_design)          # every design used
    site[:n_site] = np.arange(n_site)                # every site used

    Z, S = make_indicator(design), make_indicator(site)

    TD2, TS2, TU2, M1 = 0.04, 0.09, 0.02, -0.25
    d = rng.normal(0, np.sqrt(TD2), n_design)
    s = rng.normal(0, np.sqrt(TS2), n_site)
    u = rng.normal(0, np.sqrt(TU2), n)

    v_a = rng.uniform(0.005, 0.02, n)                 # SS sampling variance
    A = rng.normal(0, 0.05, (n_design, n_design))
    V_bb = A @ A.T / n_design + np.diag(rng.uniform(0.002, 0.008, n_design))
    C_known = np.diag(v_a) + Z @ V_bb @ Z.T

    eps = rng.multivariate_normal(np.zeros(n), C_known)
    y = M1 + Z @ d + S @ s + u + eps
    X = np.ones((n, 1))                               # intercept-only (add covariates here)

    fit = fit_a1c(y, X, C_known, design, site)
    print(f"converged      : {fit['converged']}")
    print(f"m1   true {M1:+.3f}   est {fit['beta'][0]:+.3f}  (se {fit['se_beta'][0]:.3f})")
    print(f"tau_d  true {np.sqrt(TD2):.3f}   est {np.sqrt(fit['tau_d2']):.3f}")
    print(f"tau_s  true {np.sqrt(TS2):.3f}   est {np.sqrt(fit['tau_s2']):.3f}")
    print(f"tau_u  true {np.sqrt(TU2):.3f}   est {np.sqrt(fit['tau_u2']):.3f}")

    # what happens if you ignore the shared-b contamination (diagonal V)
    fit_diag = fit_a1c(y, X, np.diag(np.diag(C_known)), design, site)
    print(f"\ntau_d with DIAGONAL V (contaminated): {np.sqrt(fit_diag['tau_d2']):.3f}"
          f"   vs true {np.sqrt(TD2):.3f}")
