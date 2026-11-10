"""
Collapse fragility comparison: IDA-FX vs MSA-SS "true", with the 95% CI band,
in the style of the original figure but without the MSA-SS estimate curve.

IMPORTANT: the curves are lognormal CDFs whose medians and dispersions were
read off the original image by eye. They reproduce its appearance, not its
data. To plot the real results, replace the (theta, beta) pairs or the CDF
arrays with the values from the analysis.

Fragility model (Baker, Bradley & Stafford 2020, Ch. 9):
    P[C | IM = x] = Phi( ln(x / theta) / beta )
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Intensity grid, matching the original x-range (0 to ~1.1 g).
iml = np.linspace(1e-4, 1.11, 600)

def lognormal_cdf(x, theta, beta):
    """Lognormal fragility: median theta (g), log-standard deviation beta."""
    return norm.cdf(np.log(x / theta) / beta)

# By-eye parameters (theta in g, beta dimensionless).
p_ida_fx  = lognormal_cdf(iml, 0.390, 0.205)
p_ss_true = lognormal_cdf(iml, 0.330, 0.185)
p_ci_lo   = lognormal_cdf(iml, 0.322, 0.170)   # left/upper CI bound
p_ci_hi   = lognormal_cdf(iml, 0.405, 0.255)   # right/lower CI bound

fig, ax = plt.subplots(figsize=(6, 4))

# Keep the original colour assignment: orange = IDA-FX, green = MSA-SS true,
# so the figure reads consistently next to the version with the blue curve.
ax.plot(iml, p_ida_fx,  color="C1", lw=1.5, label="IDA-FX")
ax.plot(iml, p_ss_true, color="C2", lw=1.5, label="MSA-SS true")
ax.plot(iml, p_ci_lo,   color="grey", lw=1.5, ls="--", label="95% CI")
ax.plot(iml, p_ci_hi,   color="grey", lw=1.5, ls="--")

ax.set_ylim(0, 1)
ax.set_xlabel("IML (g)")
ax.set_ylabel("P[Collapse]")
ax.grid(True, ls="-.", color="0.8")
ax.legend(loc="upper left", edgecolor="k", fancybox=False, framealpha=1)

fig.tight_layout()
fig.savefig("fragility_no_msa_est.png", dpi=200, facecolor="w")
