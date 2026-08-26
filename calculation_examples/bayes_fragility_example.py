"""
A minimal Bayesian analysis, worked end-to-end on a problem from your own field:
fitting a lognormal fragility curve to multiple-stripe analysis (MSA) results.

No PyMC / Stan needed - we do it on a grid so you can literally see the
prior, the likelihood and the posterior as arrays of numbers.

Model
-----
At intensity level  im , the probability of exceeding the damage state is

    P(DS | im) = Phi( ln(im / theta) / beta )

    theta = median capacity (g)   <- unknown parameter
    beta  = lognormal dispersion  <- unknown parameter
    Phi   = standard normal CDF

Data: at each stripe we ran n_j nonlinear time-history analyses and z_j of
them collapsed.  Given theta, beta the counts are Binomial, so the
likelihood is a product of binomial terms.
"""

import numpy as np
from scipy.stats import norm, binom

# ----------------------------------------------------------------------
# 1. DATA: multiple-stripe analysis results
#    (intensity level, number of records run, number that collapsed)
# ----------------------------------------------------------------------
im = np.array([0.20, 0.40, 0.60, 0.80, 1.00, 1.20])   # Sa(T1) in g
n = np.array([20, 20, 20, 20, 20, 20])                # records per stripe
z = np.array([0, 2, 6, 12, 16, 19])                   # collapses observed

# ----------------------------------------------------------------------
# 2. PRIOR: what you believed before running a single analysis
#    theta ~ LogNormal(median 0.9 g, sigma 0.4)  - from similar buildings
#    beta  ~ LogNormal(median 0.45, sigma 0.25)  - typical record-to-record
# ----------------------------------------------------------------------
theta_grid = np.linspace(0.30, 2.00, 300)
beta_grid = np.linspace(0.10, 1.20, 300)
TH, BE = np.meshgrid(theta_grid, beta_grid, indexing="ij")

log_prior = (
    norm.logpdf(np.log(TH), loc=np.log(0.90), scale=0.40)
    + norm.logpdf(np.log(BE), loc=np.log(0.45), scale=0.25)
)

# ----------------------------------------------------------------------
# 3. LIKELIHOOD: how well each (theta, beta) pair explains the MSA counts
# ----------------------------------------------------------------------
log_lik = np.zeros_like(TH)
for im_j, n_j, z_j in zip(im, n, z):
    # for each beta/theta pair the P[C] is calculated
    p_j = norm.cdf(np.log(im_j / TH) / BE)          # model prediction
    p_j = np.clip(p_j, 1e-12, 1 - 1e-12)            # keep logs finite
    # This line scores how well each theta/beta matches the stripe data (i.e. evidence for that pair)
    log_lik += binom.logpmf(z_j, n_j, p_j)          # add this stripe's evidence

# ----------------------------------------------------------------------
# 4. POSTERIOR = prior x likelihood, then normalise
#
#    Two distinct steps, easily confused:
#
#    (a) subtracting log_post.max() is PURE NUMERICAL SAFETY.  Raw values
#        run from about -25 at the peak to -900 in the bad corners, and
#        np.exp(-900) underflows to exactly 0.0 (float64 bottoms out near
#        1e-308).  Shifting the peak to exp(0) = 1.0 keeps everything in
#        range.  Subtracting a constant in log space = dividing by a
#        constant in normal space, so every cell is scaled identically and
#        the SHAPE of the posterior is untouched.  This is the standard
#        log-sum-exp trick (see scipy.special.logsumexp).
#
#    (b) dividing by post.sum() is the ACTUAL division by the evidence
#        p(y) = integral of p(y|theta) p(theta) d(theta).  This is what
#        turns relative weights into a probability distribution, and it
#        also cancels out whatever constant step (a) introduced.
# ----------------------------------------------------------------------
log_post = log_prior + log_lik
post = np.exp(log_post - log_post.max())             # (a) rescale safely
post /= post.sum()                                   # (b) divide by evidence

# The log-evidence itself, for MODEL COMPARISON (e.g. lognormal vs Weibull
# fragility).  Note the cell-area factor: post.sum() normalises the weights
# correctly for means and credible intervals, where the constant cancels,
# but the evidence as a genuine integral needs the grid spacing.
from scipy.special import logsumexp

d_theta = theta_grid[1] - theta_grid[0]
d_beta = beta_grid[1] - beta_grid[0]
log_evidence = logsumexp(log_post) + np.log(d_theta * d_beta)
print(f"log evidence log p(y) = {log_evidence:.2f}")
print("  (only meaningful when compared against another model's value,")
print("   fitted to the SAME data - the difference is the log Bayes factor)\n")

# ----------------------------------------------------------------------
# 5. READ THE ANSWER OFF THE POSTERIOR
# ----------------------------------------------------------------------
def summarise(grid, marginal, name):
    mean = np.sum(grid * marginal)  # theta value * margianl pmf of evidence
    cdf = np.cumsum(marginal)
    lo = grid[np.searchsorted(cdf, 0.025)]
    hi = grid[np.searchsorted(cdf, 0.975)]
    print(f"{name}: posterior mean {mean:.3f}, 95% credible interval [{lo:.3f}, {hi:.3f}]")

summarise(theta_grid, post.sum(axis=1), "theta (median capacity, g)")
summarise(beta_grid, post.sum(axis=0), "beta  (dispersion)       ")

# ----------------------------------------------------------------------
# 6. THE BAYESIAN PAYOFF: push the *whole* posterior through to the
#    quantity you actually care about, so the risk number carries the
#    epistemic uncertainty with it.
# ----------------------------------------------------------------------
im_test = 0.75                                        # design-level Sa(T1)
p_grid = norm.cdf(np.log(im_test / TH) / BE)          # P(DS) for every parameter pair
w = post.ravel()
p_flat = p_grid.ravel()
order = np.argsort(p_flat)
cdf = np.cumsum(w[order])

print(f"\nP(collapse | Sa = {im_test} g)")
print(f"  posterior mean : {np.sum(w * p_flat):.3f}")
print(f"  90% cred. int. : [{p_flat[order][np.searchsorted(cdf, 0.05)]:.3f}, "
      f"{p_flat[order][np.searchsorted(cdf, 0.95)]:.3f}]")

# ----------------------------------------------------------------------
# 7. Same model in PyMC, once you outgrow the grid (>2-3 parameters)
# ----------------------------------------------------------------------
PYMC_VERSION = """
import pymc as pm

with pm.Model() as fragility:
    theta = pm.LogNormal("theta", mu=np.log(0.90), sigma=0.40)
    beta  = pm.LogNormal("beta",  mu=np.log(0.45), sigma=0.25)

    p = pm.math.invprobit(pm.math.log(im / theta) / beta)
    pm.Binomial("obs", n=n, p=p, observed=z)

    idata = pm.sample(2000, tune=1000, target_accept=0.9)

# az.summary(idata)  /  az.plot_trace(idata)  /  az.plot_posterior(idata)
"""
