"""
Bayesian random-effects META-REGRESSION, worked on a grid so you can see
every moving part - the same prior x likelihood -> posterior machinery as
the fragility example, but the DATA are now other people's published
results rather than your own analyses.

The setup
---------
You have trawled the literature for reported median collapse capacities of
RC frames.  Each study k gives you:

    y_k  = ln(theta_k)   the reported log median capacity   <- the "effect"
    s_k  = its standard error                               <- KNOWN, from the paper
    x_k  = number of storeys                                <- a moderator

Two-level model
---------------
Level 1 (sampling / within-study error - the study measured its own truth
         imperfectly, and it told you how imperfectly via s_k):

    y_k ~ Normal( mu_k , s_k^2 )

Level 2 (between-study heterogeneity - each study's OWN truth differs from
         the regression line, because they used different codes, records,
         collapse definitions, modelling assumptions):

    mu_k ~ Normal( a + b * x_k , tau^2 )

Marginalising out mu_k (Normal inside Normal collapses analytically) gives
the likelihood you actually compute:

    y_k ~ Normal( a + b*x_k , s_k^2 + tau^2 )

THIS IS THE WHOLE IDEA.  Each study's variance is its own reported error
PLUS a shared extra variance tau^2 that you also estimate.  tau is the
parameter that says "how much do studies genuinely disagree, over and above
their stated precision".

    tau = 0    -> studies agree perfectly; this reduces to weighted least
                 squares (the "fixed-effect" model)
    tau large  -> the literature is a mess; a precise-looking study gets
                 heavily down-weighted, because its stated precision is not
                 credible in light of how much studies disagree

Three unknowns: a (intercept), b (slope per storey), tau (heterogeneity).
"""

import numpy as np
from scipy.stats import norm, halfnorm
from scipy.special import logsumexp

# ----------------------------------------------------------------------
# 1. DATA: one row per published study
# ----------------------------------------------------------------------
study = ["Aslani 09", "Haselton 11", "Liel 11", "Baker 15", "Zhang 16",
         "Kircher 17", "Ramirez 18", "Silva 19", "Martins 20", "Chen 21",
         "Nowak 22", "Ferreira 23"]
x = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 10], float)          # storeys
y = np.array([-0.124, -0.381, -0.638, -0.219, -0.464, -0.370,
              -0.671, -0.644, -0.805, -0.621, -0.568, -0.638])       # ln(theta)
s = np.array([0.144, 0.185, 0.166, 0.084, 0.095, 0.181,
              0.051, 0.173, 0.170, 0.120, 0.095, 0.092])             # reported SE

xc = x - x.mean()   # CENTRE the moderator: makes the intercept mean
                    # "log capacity of an average-height frame" instead of
                    # "log capacity of a 0-storey frame", and stops a and b
                    # from being strongly correlated in the posterior.

# ----------------------------------------------------------------------
# 2. PRIORS - one per unknown, on the same footing as before
# ----------------------------------------------------------------------
# a: typical ln(theta) for an average-height frame. exp(-0.35) ~ 0.70 g,
#    sd 0.50 in log space is deliberately loose.
# b: change in ln(theta) per storey. Centred on ZERO - the sceptical
#    position, "height may not matter at all" - with sd 0.10, which allows
#    up to about +-20% capacity change per storey. This is a weakly
#    informative prior that regularises without dictating.
# tau: HalfNormal(0.25). Must be positive. Says "some heterogeneity is
#    expected, a lot would be surprising". NEVER put a flat prior on tau
#    with few studies - it will wander off to absurd values.
a_grid = np.linspace(-1.20, 0.40, 120)
b_grid = np.linspace(-0.30, 0.20, 120)
tau_grid = np.linspace(1e-4, 0.60, 100)

A, B, T = np.meshgrid(a_grid, b_grid, tau_grid, indexing="ij")

log_prior = (
    norm.logpdf(A, loc=-0.35, scale=0.50)
    + norm.logpdf(B, loc=0.0, scale=0.10)
    + halfnorm.logpdf(T, scale=0.25)
)

# ----------------------------------------------------------------------
# 3. LIKELIHOOD - loop over STUDIES exactly as you looped over STRIPES
# ----------------------------------------------------------------------
log_lik = np.zeros_like(A)
for y_k, s_k, x_k in zip(y, s, xc):
    mu_k = A + B * x_k                       # what the line predicts
    sd_k = np.sqrt(s_k**2 + T**2)            # within + between variance
    log_lik += norm.logpdf(y_k, loc=mu_k, scale=sd_k)

# ----------------------------------------------------------------------
# 4. POSTERIOR
# ----------------------------------------------------------------------
log_post = log_prior + log_lik
post = np.exp(log_post - log_post.max())     # numerical safety
post /= post.sum()                           # divide by the evidence

d = (a_grid[1]-a_grid[0]) * (b_grid[1]-b_grid[0]) * (tau_grid[1]-tau_grid[0])
print(f"log evidence = {logsumexp(log_post) + np.log(d):.2f}\n")


def summarise(grid, marginal, name, transform=lambda v: v):
    mean = np.sum(grid * marginal)
    cdf = np.cumsum(marginal)
    lo, hi = grid[np.searchsorted(cdf, .025)], grid[np.searchsorted(cdf, .975)]
    print(f"{name:34s} {transform(mean):7.3f}   95% CrI [{transform(lo):.3f}, {transform(hi):.3f}]")


summarise(a_grid, post.sum(axis=(1, 2)), "a  intercept, ln(theta) @ mean ht")
summarise(b_grid, post.sum(axis=(0, 2)), "b  slope, dln(theta) per storey")
summarise(tau_grid, post.sum(axis=(0, 1)), "tau between-study sd")

# ----------------------------------------------------------------------
# 5. THE QUESTIONS A META-REGRESSION IS ACTUALLY FOR
# ----------------------------------------------------------------------
w = post.ravel()

# (a) Direct probability statements - no p-values, no null hypothesis.
p_neg = w[(B.ravel() < 0)].sum()
print(f"\nP(capacity decreases with height, i.e. b < 0) = {p_neg:.3f}")

# (b) Is there real heterogeneity, or do the studies just have noise?
p_tau = w[(T.ravel() > 0.10)].sum()
print(f"P(tau > 0.10, i.e. substantial disagreement)  = {p_tau:.3f}")

# (c) Effect size on the scale people understand
b_mean = np.sum(b_grid * post.sum(axis=(0, 2)))
print(f"Each extra storey changes median capacity by  {100*(np.exp(b_mean)-1):+.1f}% ")

# ----------------------------------------------------------------------
# 6. TWO DIFFERENT PREDICTIONS - the distinction that matters most
# ----------------------------------------------------------------------
x_new = 6.0
xn = x_new - x.mean()
mu_new = (A + B * xn).ravel()          # the LINE at 6 storeys

# (i) Where is the mean of the literature at 6 storeys?
#     Uncertainty in the regression line only.
order = np.argsort(mu_new); c = np.cumsum(w[order])
lo, hi = mu_new[order][np.searchsorted(c, .025)], mu_new[order][np.searchsorted(c, .975)]
print(f"\nMean ln(theta) for 6-storey frames : {np.sum(w*mu_new):+.3f} "
      f"[{lo:+.3f}, {hi:+.3f}]  -> theta = {np.exp(np.sum(w*mu_new)):.3f} g")

# (ii) What should you expect for THE NEXT building - yours?
#      Must add tau, because your frame is a new draw from the population,
#      not the population average. This is the number that belongs in a
#      risk assessment, and it is always wider.
rng = np.random.default_rng(1)
idx = rng.choice(w.size, size=200_000, p=w)
pred = (A.ravel()[idx] + B.ravel()[idx] * xn) + rng.normal(0, T.ravel()[idx])
print(f"NEXT 6-storey frame (adds tau)     : {pred.mean():+.3f} "
      f"[{np.quantile(pred,.025):+.3f}, {np.quantile(pred,.975):+.3f}]"
      f"  -> theta = {np.exp(pred.mean()):.3f} g")

# ----------------------------------------------------------------------
# 7. SHRINKAGE - what pooling does to each individual study
#    Each study's own truth is pulled toward the line. How hard depends on
#    its precision relative to tau. This is "borrowing strength" and it is
#    the single most useful property of the hierarchical model.
# ----------------------------------------------------------------------
a_m = np.sum(a_grid * post.sum(axis=(1, 2)))
t_m = np.sum(tau_grid * post.sum(axis=(0, 1)))
print("\nstudy         reported   line     shrunk   pulled")
for k in range(len(y)):
    line = a_m + b_mean * xc[k]
    wt = t_m**2 / (t_m**2 + s[k]**2)          # weight on the study's own value
    shrunk = wt * y[k] + (1 - wt) * line
    print(f"{study[k]:13s} {y[k]:+.3f}   {line:+.3f}   {shrunk:+.3f}   {100*(1-wt):4.0f}%")

# ----------------------------------------------------------------------
# 8. The same model in PyMC - what you will actually use, because real
#    meta-regressions have several moderators and the grid dies at ~4
#    parameters.
# ----------------------------------------------------------------------
PYMC = """
import pymc as pm

with pm.Model() as meta:
    a   = pm.Normal("a", mu=-0.35, sigma=0.50)
    b   = pm.Normal("b", mu=0.0,   sigma=0.10)
    tau = pm.HalfNormal("tau", sigma=0.25)

    # non-centred parameterisation - fit this way, it samples far better
    # when tau is small (avoids the classic "funnel" pathology)
    z  = pm.Normal("z", 0, 1, shape=len(y))
    mu = a + b * xc + z * tau

    pm.Normal("obs", mu=mu, sigma=s, observed=y)   # s is KNOWN, not estimated

    idata = pm.sample(2000, tune=2000, target_accept=0.95)

# az.summary(idata, var_names=["a","b","tau"])
# az.plot_forest(idata, var_names=["mu"])   # the classic forest plot
"""
