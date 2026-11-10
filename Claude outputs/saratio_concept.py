"""
Concept figure: Spectral Acceleration ratio, SaRatio(T1), for T1 = 1.0 s.

SaRatio(T1) = SA(T1) / ( prod_j SA(T_j) )^(1/n),   T_j = 0.2*T1 : 0.01 : 3*T1

The numerator is the spectral ordinate at the fundamental period; the
denominator is the geometric mean of the spectrum over the band 0.2*T1-3*T1
(i.e. AvgSA over that band). A value > 1 means the spectrum has a local peak
at T1 relative to its neighbourhood; < 1 means a local trough.

The spectrum here is synthetic (illustrative only): a smooth "design-like"
shape with a local bump placed at T1 so the peak is easy to see.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

T1 = 1.0

# ---- Synthetic 5%-damped spectrum -------------------------------------------
# Base shape: plateau at short periods, decaying ~1/T at long periods.
T = np.logspace(np.log10(0.05), np.log10(5.0), 600)
base = 0.9 / (1.0 + (T / 0.45) ** 1.3) + 0.08
# Small ripples so it looks like a recorded spectrum rather than a code curve.
ripple = 1.0 + 0.10 * np.sin(9.0 * np.log(T)) + 0.06 * np.sin(17.0 * np.log(T) + 1.0)
# Gaussian bump (in log-period) centred on T1 -> the "peak" at T1.
bump = 1.0 + 0.6 * np.exp(-0.5 * (np.log(T / T1) / 0.09) ** 2)
SA = base * ripple * bump

def sa_at(tq):
    """Interpolate SA in log-log space (spectra are smooth in log-log)."""
    return np.exp(np.interp(np.log(tq), np.log(T), np.log(SA)))

# ---- SaRatio ingredients ------------------------------------------------------
# Period band for the denominator, exactly as defined: 0.2*T1 : 0.01 : 3*T1.
Tj = np.arange(0.2 * T1, 3.0 * T1 + 1e-9, 0.01)
SAj = sa_at(Tj)
# Geometric mean via scipy.stats.gmean (exp of mean log) - never a raw product,
# which would underflow over ~280 ordinates below 1 g.
avg_sa = gmean(SAj)
sa_T1 = sa_at(T1)
sa_ratio = sa_T1 / avg_sa

# ---- Plot ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.2))
ink, muted, accent, avg_col = "#1f2328", "#6b7280", "#2a6fdb", "#d1495b"

# Shade the averaging band so the reader sees where the denominator comes from.
ax.axvspan(Tj[0], Tj[-1], color=accent, alpha=0.07, lw=0)
ax.text(Tj[0] * 1.08, 0.035, r"averaging band $0.2T_1$ – $3T_1$",
        ha="left", va="bottom", color=muted, fontsize=9)

# Full spectrum, and the portion used in the average drawn heavier.
ax.plot(T, SA, color=accent, lw=1.2, alpha=0.45)
ax.plot(Tj, SAj, color=accent, lw=2.2, label="Response spectrum SA(T)")

# Geometric-mean line across the band.
ax.hlines(avg_sa, Tj[0], Tj[-1], color=avg_col, lw=2, ls="--",
          label=r"Geometric mean over band, $\overline{SA}$")

# Peak point at T1 plus a vertical guide down to the period axis.
ax.vlines(T1, 0.03, sa_T1, color=muted, lw=0.8, ls=":")
ax.plot(T1, sa_T1, "o", ms=9, color=ink, mec="w", mew=2, zorder=5,
        label=r"$SA(T_1)$ at $T_1 = 1.0$ s")

# Labels next to the two numbers that form the ratio.
ax.annotate(f"SA(T$_1$) = {sa_T1:.3f} g", (T1, sa_T1), xytext=(14, 6),
            textcoords="offset points", fontsize=10, color=ink)
ax.annotate(rf"$\overline{{SA}}$ = {avg_sa:.3f} g", (Tj[-1], avg_sa),
            xytext=(-4, 5), textcoords="offset points", fontsize=10,
            color=avg_col, ha="right", va="bottom")

# Formula with the worked numbers, placed in the empty upper-right space.
formula = (
    rf"$SaRatio(1.0\,\mathrm{{s}}) = \dfrac{{{sa_T1:.3f}}}{{{avg_sa:.3f}}} = {sa_ratio:.2f}$"
)
ax.text(0.97, 0.95, formula, transform=ax.transAxes, ha="right", va="top",
        fontsize=12, color=ink,
        bbox=dict(boxstyle="round,pad=0.6", fc="w", ec="#d0d7de"))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.05, 5)
ax.set_ylim(0.03, 8)
ax.set_xlabel("Period, T (s)")
ax.set_ylabel("Spectral acceleration, SA (g)")
from matplotlib.ticker import FuncFormatter
plain = FuncFormatter(lambda v, _: f"{v:g}")
ax.xaxis.set_major_formatter(plain)
ax.yaxis.set_major_formatter(plain)
ax.grid(True, which="major", color="#e5e7eb", lw=0.6)
ax.grid(True, which="minor", color="#f3f4f6", lw=0.4)
# Full black frame around the axes.
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_color("k")
    sp.set_linewidth(1.0)
ax.legend(loc="lower left", frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("saratio_concept.png", dpi=200, facecolor="w")
print(f"SA(T1)={sa_T1:.3f}  AvgSA={avg_sa:.3f}  SaRatio={sa_ratio:.3f}  n={len(Tj)}")
