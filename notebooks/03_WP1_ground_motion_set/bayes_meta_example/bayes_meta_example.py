"""
Bayesian random-effects meta-analysis -- a worked toy example
=============================================================

The same shape as your fragility-curve fitting, one level up.

For the fragility curve you had:   records -> collapse outcomes -> (theta, beta)
Here you have:                     sites   -> log ratios        -> (m, tau)

The trick that makes this work is that we already KNOW how noisy each site's
observation is (from the bootstrap). We hand that to the model as data, so the
model can subtract it and recover the real between-site variation underneath.

Run:  python bayes_meta_example.py
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----------------------------------------------------------------------------
# plotting setup (validated palette, light surface)
# ----------------------------------------------------------------------------
C_DATA, C_TRUTH, C_ALT = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e0dfda"
rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})

RNG = np.random.default_rng(42)

# ============================================================================
# STAGE 0 -- Simulate a world where we know the truth
# ============================================================================
# We invent 57 sites. We DECIDE the true answer, then hide it and see whether
# the model can find it again. This is the only way to check that a fitting
# procedure works -- with real data you never get to look at the answer.

N_SITES = 57
TRUE_M = 0.12       # true systematic shift in ln(theta): fixed set ~13% high
TRUE_TAU = 0.15     # true site-to-site variation around that shift

# Each site's TRUE discrepancy: drawn from the population distribution.
# In your study these are real physical effects -- what the fixed record set
# genuinely does at that site -- not errors.
true_site = RNG.normal(TRUE_M, TRUE_TAU, N_SITES)

# Each site's measurement noise. In your study this comes from the bootstrap,
# and it VARIES: sites whose stripes sit at awkward collapse probabilities are
# measured less precisely. We mimic that with a spread of SE values.
se = np.exp(RNG.normal(np.log(0.13), 0.35, N_SITES))

# What you actually observe = truth + noise.
obs = true_site + RNG.normal(0, se)

print("=" * 68)
print("STAGE 0  the simulated world")
print("=" * 68)
print(f"  true m                    {TRUE_M:.3f}")
print(f"  true tau                  {TRUE_TAU:.3f}")
print(f"  typical measurement SE    {np.median(se):.3f}")

# ============================================================================
# STAGE 1 -- The naive look, and why it misleads
# ============================================================================
# If you just took the 57 observed ratios and computed their spread, you would
# be measuring real variation AND measurement noise added together.

naive_sd = obs.std(ddof=1)
print()
print("=" * 68)
print("STAGE 1  what a naive summary would tell you")
print("=" * 68)
print(f"  mean of observed ratios   {obs.mean():+.3f}   (close to true m -- means are easy)")
print(f"  SD of observed ratios     {naive_sd:.3f}    <-- looks like tau but is NOT")
print(f"  true tau                  {TRUE_TAU:.3f}")
print(f"  inflation                 {naive_sd / TRUE_TAU:.2f}x")
print()
print("  The naive SD overstates the real site-to-site variation, because it")
print("  contains the estimation noise as well. Fit a correction factor to")
print("  that scatter and you are partly fitting noise.")

# ============================================================================
# STAGE 2 -- Write the model down
# ============================================================================
# Two layers, exactly as discussed:
#
#   LAYER 2 (population):  true_i  ~  Normal(m, tau)
#   LAYER 1 (measurement): obs_i   ~  Normal(true_i, se_i)      se_i KNOWN
#
# Plus priors -- the only genuinely new ingredient versus the frequentist fit.
# These say "m is somewhere within a factor of a few; tau is positive and
# probably below ~0.5". They rule out absurdity, nothing more.

with pm.Model() as model:

    # --- priors -------------------------------------------------------------
    m = pm.Normal("m", mu=0.0, sigma=0.5)          # weakly informative
    tau = pm.HalfNormal("tau", sigma=0.3)          # positive, mass near small

    # --- layer 2: each site's true effect comes from the population ---------
    # (non-centred parameterisation: samples much better for hierarchical models)
    z = pm.Normal("z", 0.0, 1.0, shape=N_SITES)
    true_i = pm.Deterministic("true_i", m + tau * z)

    # --- layer 1: what we observed, with KNOWN noise ------------------------
    pm.Normal("obs", mu=true_i, sigma=se, observed=obs)

    # --- what a brand-new site would look like ------------------------------
    # This is the deliverable: not a parameter, a prediction.
    pm.Deterministic("new_site", m + tau * pm.Normal.dist(0, 1))

    idata = pm.sample(2000, tune=2000, chains=4, target_accept=0.9,
                      random_seed=42, progressbar=False)

# ============================================================================
# STAGE 3 -- Check the sampler actually worked
# ============================================================================
# Unlike a frequentist fit, a Bayesian one can fail quietly. Always look.
#   r_hat  ~ 1.00  means the chains agree with each other
#   ess    > ~400  means you have enough effective samples
#   divergences > 0 means the geometry defeated the sampler -- investigate

summ = az.summary(idata, var_names=["m", "tau"], round_to=3)
n_div = int(idata.sample_stats["diverging"].sum())

print()
print("=" * 68)
print("STAGE 3  sampler diagnostics")
print("=" * 68)
print(summ[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string())
print(f"  divergences: {n_div}   (want 0)")

# ============================================================================
# STAGE 4 -- Read the answer
# ============================================================================
post = idata.posterior
m_s = post["m"].values.ravel()
tau_s = post["tau"].values.ravel()
new_s = post["new_site"].values.ravel()

print()
print("=" * 68)
print("STAGE 4  the posterior")
print("=" * 68)
print(f"  m    median {np.median(m_s):+.3f}   90% [{np.quantile(m_s,.05):+.3f},"
      f" {np.quantile(m_s,.95):+.3f}]   truth {TRUE_M:+.3f}")
print(f"  tau  median {np.median(tau_s):.3f}   90% [{np.quantile(tau_s,.05):.3f},"
      f" {np.quantile(tau_s,.95):.3f}]   truth {TRUE_TAU:.3f}")
print(f"  naive SD would have said tau = {naive_sd:.3f}")
print()
print("  Note what the posterior gives you that a point estimate cannot:")
print(f"    P(tau < 0.10)  = {np.mean(tau_s < 0.10):.2f}")
print(f"    P(tau > 0.20)  = {np.mean(tau_s > 0.20):.2f}")
print("  i.e. an honest statement of how well tau is pinned down.")

# --- the deliverable --------------------------------------------------------
lo, hi = np.quantile(new_s, [0.05, 0.95])
print()
print("  PREDICTION for a site never analysed (this is the thesis deliverable):")
print(f"    after correcting by m, ln-ratio lands in [{lo:+.3f}, {hi:+.3f}] 90% of the time")
print(f"    i.e. theta within a factor of {np.exp(max(abs(lo), abs(hi))):.2f}")
print("    -- and this already includes uncertainty in m AND in tau.")

# ============================================================================
# STAGE 5 -- Shrinkage: what the model does to individual sites
# ============================================================================
# A noisy site's observation gets pulled towards the population mean, because
# the model knows an extreme reading from an imprecise site is more likely to
# be noise than a real extreme. Precise sites barely move. This is "borrowing
# strength" and it is the mechanism behind everything above.

site_post = post["true_i"].values.reshape(-1, N_SITES).mean(axis=0)

print()
print("=" * 68)
print("STAGE 5  shrinkage")
print("=" * 68)
noisiest, cleanest = int(np.argmax(se)), int(np.argmin(se))
for lbl, i in [("noisiest site", noisiest), ("most precise site", cleanest)]:
    print(f"  {lbl:18s} se={se[i]:.3f}  observed {obs[i]:+.3f}"
          f"  ->  model {site_post[i]:+.3f}   (truth {true_site[i]:+.3f})")
err_obs = np.sqrt(np.mean((obs - true_site) ** 2))
err_mod = np.sqrt(np.mean((site_post - true_site) ** 2))
print(f"  RMS error vs truth:  raw observations {err_obs:.3f}"
      f"   model estimates {err_mod:.3f}  ({100*(1-err_mod/err_obs):.0f}% better)")

# ============================================================================
# FIGURES
# ============================================================================
def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

# --- Figure 1: the data -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
o = np.argsort(obs)
x = np.arange(N_SITES)
ax.errorbar(x, obs[o], yerr=1.645 * se[o], fmt="o", ms=4.5, lw=1.4,
            color=C_DATA, ecolor=C_DATA, alpha=0.85, capsize=0,
            label="observed log ratio (90% CI)", zorder=3)
ax.scatter(x, true_site[o], s=14, color=C_TRUTH, zorder=4,
           label="true site effect (hidden in real life)")
ax.axhline(TRUE_M, color=INK2, lw=1.2, ls="--", zorder=2)
ax.annotate("true m", xy=(1, TRUE_M), xytext=(0, 6),
            textcoords="offset points", ha="left", color=INK2, fontsize=8)
ax.set_xlabel("site (sorted by observed value)")
ax.set_ylabel("ln(theta ratio)")
ax.set_title("Stage 1  The observed scatter is wider than the real scatter",
             loc="left", fontsize=10.5, color=INK, pad=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
_grid(ax)
fig.tight_layout(); fig.savefig("/tmp/mm/fig1_data.png", dpi=170)

# --- Figure 2: posteriors ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for ax, s, truth, name in [(axes[0], m_s, TRUE_M, "m  (systematic shift)"),
                           (axes[1], tau_s, TRUE_TAU, "tau  (real site-to-site variation)")]:
    ax.hist(s, bins=60, color=C_DATA, alpha=0.85, zorder=3)
    ax.axvline(truth, color=C_TRUTH, lw=2, zorder=4)
    ax.annotate("truth", xy=(truth, ax.get_ylim()[1] * 0.92), xytext=(5, 0),
                textcoords="offset points", color=C_TRUTH, fontsize=8)
    ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
    ax.set_yticks([]); _grid(ax, axis="x")
axes[1].axvline(naive_sd, color=INK2, lw=1.6, ls="--", zorder=4)
axes[1].annotate("naive SD\nof the raw ratios", xy=(naive_sd, axes[1].get_ylim()[1] * 0.55),
                 xytext=(6, 0), textcoords="offset points", color=INK2, fontsize=8)
fig.suptitle("Stage 4  m is pinned down tightly; tau is not -- and the posterior says so",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("/tmp/mm/fig2_posterior.png", dpi=170)

# --- Figure 3: shrinkage ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.6))
lim = [min(obs.min(), site_post.min()) - 0.06, max(obs.max(), site_post.max()) + 0.06]
ax.plot(lim, lim, color=GRID, lw=1.4, zorder=1)
ax.axhline(np.median(m_s), color=INK2, lw=1.0, ls="--", zorder=2)
sc = ax.scatter(obs, site_post, c=se, cmap="viridis_r", s=34, zorder=3,
                edgecolor=SURFACE, linewidth=0.8)
cb = fig.colorbar(sc, ax=ax); cb.set_label("measurement SE at that site", fontsize=8)
cb.outline.set_visible(False)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.annotate("grey line = no shrinkage;\nevery point falls short of it",
            xy=(lim[1] - 0.03, lim[1] - 0.03), xytext=(-6, -34),
            textcoords="offset points", ha="right", fontsize=8, color=INK2)
ax.annotate("dashed line = population mean;\npoints are pulled towards it",
            xy=(lim[0] + 0.03, np.median(m_s)), xytext=(4, 10),
            textcoords="offset points", ha="left", fontsize=8, color=INK2)
ax.set_xlabel("raw observation at the site")
ax.set_ylabel("model's estimate for that site")
ax.set_title("Stage 5  Noisy sites are pulled towards the population mean",
             loc="left", fontsize=10.5, color=INK, pad=10)
_grid(ax, axis="both")
fig.tight_layout(); fig.savefig("/tmp/mm/fig3_shrinkage.png", dpi=170)

# --- Figure 4: the deliverable ---------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.hist(new_s, bins=70, color=C_ALT, alpha=0.9, zorder=3, label="a new, unseen site")
ax.hist(m_s, bins=70, color=C_DATA, alpha=0.9, zorder=4, label="the average shift m")
ax.axvspan(lo, hi, color=C_ALT, alpha=0.12, zorder=1)
for v in (lo, hi):
    ax.axvline(v, color=C_ALT, lw=1.4, ls="--", zorder=5)
ax.annotate(f"90% prediction interval\n[{lo:+.2f}, {hi:+.2f}]",
            xy=(hi, ax.get_ylim()[1] * 0.75), xytext=(8, 0),
            textcoords="offset points", fontsize=8, color=INK2)
ax.set_xlabel("ln(theta ratio)")
ax.set_yticks([])
ax.set_title("Stage 4  Knowing the average shift precisely is not the same as\n"
             "knowing what happens at the next site",
             loc="left", fontsize=10.5, color=INK, pad=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
_grid(ax, axis="x")
fig.tight_layout(); fig.savefig("/tmp/mm/fig4_prediction.png", dpi=170)

# ============================================================================
# STAGE 6 -- The same true tau, seen through noisier measurements
# ============================================================================
# This is the theta-vs-beta situation in miniature. Keep the real between-site
# variation FIXED and only change how precisely each site is measured. The
# naive SD balloons; the model does not.

def fit_once(tau_true, se_median, seed):
    rng = np.random.default_rng(seed)
    t = rng.normal(TRUE_M, tau_true, N_SITES)
    s = np.exp(rng.normal(np.log(se_median), 0.35, N_SITES))
    y = t + rng.normal(0, s)
    with pm.Model():
        mm_ = pm.Normal("m", 0.0, 0.5)
        tt_ = pm.HalfNormal("tau", 0.3)
        zz_ = pm.Normal("z", 0.0, 1.0, shape=N_SITES)
        pm.Normal("obs", mu=mm_ + tt_ * zz_, sigma=s, observed=y)
        idt = pm.sample(2000, tune=2000, chains=4, target_accept=0.9,
                        random_seed=seed, progressbar=False)
    tp = idt.posterior["tau"].values.ravel()
    return y.std(ddof=1), np.median(tp), np.quantile(tp, [0.05, 0.95])

print()
print("=" * 68)
print("STAGE 6  same real variation, different measurement precision")
print("=" * 68)
print(f"  (true tau held fixed at {TRUE_TAU:.2f} in both rows)")
print(f"  {'regime':<22}{'typical SE':>11}{'naive SD':>10}{'posterior tau':>16}{'90% interval':>20}")
for label, se_med, seed in [("theta-like (precise)", 0.06, 7),
                            ("beta-like (noisy)", 0.24, 7)]:
    nsd, tmed, tci = fit_once(TRUE_TAU, se_med, seed)
    print(f"  {label:<22}{se_med:>11.2f}{nsd:>10.3f}{tmed:>16.3f}"
          f"{f'[{tci[0]:.3f}, {tci[1]:.3f}]':>20}")
print()
print("  The naive SD is badly inflated in the noisy row -- it would have you")
print("  believe sites vary far more than they do. The model recovers roughly")
print("  the right tau in both, but honestly reports much wider uncertainty")
print("  when the measurements are poor. That widening is the point: it is")
print("  what stops you building a correction factor on top of noise.")

print()
print("=" * 68)
print("figures written: fig1_data.png fig2_posterior.png"
      " fig3_shrinkage.png fig4_prediction.png")
print("=" * 68)
