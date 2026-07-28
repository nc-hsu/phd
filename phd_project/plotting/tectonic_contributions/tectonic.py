import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


SOURCE_COLS = {
    "Shallow Default [%]":        "#3266AD",
    "Subduction Interface [%]":   "#1D9E75",
    "Non-Subduction Deep [%]":    "#E24B4A",
    "Subduction Inslab [%]":      "#BA7517",
    "Craton [%]":                 "#7F77DD",
    "Volcanic [%]":               "#D4537E",
}
SOURCE_LABELS = {
    "Shallow Default [%]":        "Shallow Default",
    "Subduction Interface [%]":   "Subduction Interface",
    "Non-Subduction Deep [%]":    "Non-Subduction Deep",
    "Subduction Inslab [%]":      "Subduction Inslab",
    "Craton [%]":                 "Craton",
    "Volcanic [%]":               "Volcanic",
}
HATCHES = ["", "///", "...", "xxx", "---", "+++"]

# Return-period labels for the 5 standard PoE levels
POE_RP = {
    0.02:       "50 yr",
    0.002103:   "475 yr",
    0.000667:   "1500 yr",
    0.000404:   "2475 yr",
    0.000201:   "4975 yr",
}

SOURCES = list(SOURCE_COLS.keys())

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.color":        "#cccccc",
    "axes.axisbelow":    True,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df["poe"] = df["poe"].astype(float)
    df["region"] = df["region"].astype(str)
    # Fill missing source columns with 0
    for s in SOURCES:
        if s not in df.columns:
            df[s] = 0.0
    return df


def closest_poe(df: pd.DataFrame, target: float) -> float:
    """Return the PoE value in the data closest to *target*."""
    available = df["poe"].unique()
    return available[np.argmin(np.abs(available - target))]


def site_label(row: pd.Series) -> str:
    return f"S{int(row['site_id'])}  ({row['lat']:.1f}°N, {row['lon']:.1f}°E)"


def make_legend(ax, title=None, ncol=3):
    handles = [
        mpatches.Patch(
            facecolor=SOURCE_COLS[s],
            edgecolor="white",
            hatch=HATCHES[i],
            label=SOURCE_LABELS[s],
        )
        for i, s in enumerate(SOURCES)
    ]
    ax.legend(
        handles=handles,
        title=title,
        ncol=ncol,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#cccccc",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Source contributions per site at a fixed PoE
# ─────────────────────────────────────────────────────────────────────────────

def fig_by_site(df: pd.DataFrame, poe_target: float = 0.002103,
                fmt: str = "png", dpi: int = 200, out_dir: Path = Path(".")):

    poe = closest_poe(df, poe_target)
    sub = df[np.isclose(df["poe"], poe)].copy()
    sub = sub.sort_values(["seismicity", "region", "site_id"])

    seismicity_classes = sub["seismicity"].unique()
    n_panels = len(seismicity_classes)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7 * n_panels, max(4, len(sub) // n_panels * 0.38 + 1.5)),
        sharey=False,
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"Seismic source contributions by site  —  PoE {poe} ({POE_RP.get(poe, '')} return period)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for ax, seis in zip(axes, seismicity_classes):
        rows = sub[sub["seismicity"] == seis].reset_index(drop=True)
        labels = [site_label(r) for _, r in rows.iterrows()]
        y = np.arange(len(rows))
        lefts = np.zeros(len(rows))

        for i, s in enumerate(SOURCES):
            vals = rows[s].values
            ax.barh(
                y, vals, left=lefts,
                color=SOURCE_COLS[s],
                hatch=HATCHES[i],
                edgecolor="white", linewidth=0.4,
                label=SOURCE_LABELS[s],
            )
            lefts += vals

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Contribution (%)")
        ax.set_xlim(0, 105)
        ax.set_title(f"Seismicity: {seis}", fontweight="bold")
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g%%"))

        # Annotate region boundaries
        prev_reg = None
        for yi, (_, r) in enumerate(rows.iterrows()):
            if r["region"] != prev_reg:
                if yi > 0:
                    ax.axhline(yi - 0.5, color="#888888", lw=0.6, ls="--")
                ax.text(
                    102, yi, f"R{r['region']}",
                    va="center", ha="left", fontsize=7,
                    color="#555555",
                )
                prev_reg = r["region"]

    # Shared legend on the last axis
    make_legend(axes[-1], title="Source type", ncol=2)

    fig.tight_layout()
    out = out_dir / f"fig1_by_site.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – PoE sensitivity: one sub-plot per site
# ─────────────────────────────────────────────────────────────────────────────

def fig_poe_sensitivity(df: pd.DataFrame, seismicity: str = None,
                        fmt: str = "png", dpi: int = 200, out_dir: Path = Path(".")):

    sub = df.copy()
    if seismicity:
        sub = sub[sub["seismicity"] == seismicity]

    sites = sub["site_id"].unique()
    sites = sorted(sites)
    n = len(sites)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.8, nrows * 2.4),
        sharey=True,
    )
    axes_flat = np.array(axes).flatten()

    seis_title = f" ({seismicity})" if seismicity else ""
    fig.suptitle(
        f"Source contributions across return periods{seis_title}",
        fontsize=11, fontweight="bold", y=1.01,
    )

    poes_sorted = sorted(df["poe"].unique(), reverse=True)
    rp_labels = [POE_RP.get(p, f"{p:.4f}") for p in poes_sorted]

    x = np.arange(len(poes_sorted))
    bar_w = 0.72

    for idx, sid in enumerate(sites):
        ax = axes_flat[idx]
        site_rows = sub[sub["site_id"] == sid].copy()
        # one row per PoE
        site_rows = site_rows.set_index("poe").reindex(poes_sorted)

        lefts = np.zeros(len(poes_sorted))
        for i, s in enumerate(SOURCES):
            vals = site_rows[s].fillna(0).values
            ax.bar(
                x, vals, bottom=lefts,
                width=bar_w,
                color=SOURCE_COLS[s],
                hatch=HATCHES[i],
                edgecolor="white", linewidth=0.4,
            )
            lefts += vals

        # Site metadata
        meta = sub[sub["site_id"] == sid].iloc[0]
        ax.set_title(
            f"S{int(sid)} | R{meta['region']} | {meta['seismicity']}\n"
            f"{meta['lat']:.1f}°N {meta['lon']:.1f}°E",
            fontsize=7.5, pad=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(rp_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g%%"))
        ax.set_xlabel("")

    
    for ax in axes_flat[n:]:
        ax.set_visible(False)

  
    handles = [
        mpatches.Patch(
            facecolor=SOURCE_COLS[s], edgecolor="white",
            hatch=HATCHES[i], label=SOURCE_LABELS[s],
        )
        for i, s in enumerate(SOURCES)
    ]
    fig.legend(
        handles=handles, ncol=6,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        framealpha=0.9, edgecolor="#cccccc", fontsize=8,
        title="Source type",
    )

    fig.tight_layout()
    suffix = f"_{seismicity}" if seismicity else ""
    out = out_dir / f"fig2_poe_sensitivity{suffix}.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Regional and seismicity-class aggregates across PoEs
# ─────────────────────────────────────────────────────────────────────────────

def fig_by_region(df: pd.DataFrame,
                  fmt: str = "png", dpi: int = 200, out_dir: Path = Path(".")):

    poes_sorted = sorted(df["poe"].unique(), reverse=True)
    rp_labels   = [POE_RP.get(p, f"{p:.4f}") for p in poes_sorted]

    regions    = sorted(df["region"].unique())
    seis_types = sorted(df["seismicity"].unique())

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(10, len(regions) * len(poes_sorted) * 0.55 + 2), 9),
    )
    fig.suptitle(
        "Average source contributions — regional and seismicity-class aggregates",
        fontsize=11, fontweight="bold",
    )

    def _draw_grouped(ax, groups, group_label_fn, title):
        """Draw a grouped stacked bar chart."""
        n_groups = len(groups)
        n_poes   = len(poes_sorted)
        group_w  = n_poes + 1            # width reserved per group (bars + gap)
        bar_w    = 0.8

        tick_positions = []
        tick_labels    = []

        for gi, grp in enumerate(groups):
            grp_sub = df[df[group_label_fn] == grp]
            offset  = gi * group_w
            xs      = np.arange(n_poes) + offset

            tick_positions.append(offset + (n_poes - 1) / 2)
            tick_labels.append(str(grp))

            for pi, poe in enumerate(poes_sorted):
                rows  = grp_sub[np.isclose(grp_sub["poe"], poe)]
                avgs  = {s: rows[s].mean() if len(rows) else 0.0 for s in SOURCES}
                left  = 0.0
                for i, s in enumerate(SOURCES):
                    v = avgs[s]
                    ax.bar(
                        xs[pi], v, bottom=left,
                        width=bar_w,
                        color=SOURCE_COLS[s],
                        hatch=HATCHES[i],
                        edgecolor="white", linewidth=0.4,
                    )
                    left += v

          
            for pi, rp in enumerate(rp_labels):
                ax.text(
                    xs[pi], -4, rp,
                    ha="center", va="top", fontsize=6.5, rotation=45,
                    color="#555555",
                )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylim(0, 112)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g%%"))
        ax.set_ylabel("Average contribution (%)")
        ax.set_title(title, fontweight="bold")

        # Vertical group separators
        for gi in range(1, n_groups):
            ax.axvline(gi * group_w - 0.7, color="#aaaaaa", lw=0.8, ls=":")

    _draw_grouped(axes[0], regions,    "region",      "By region (average across sites & seismicity)")
    _draw_grouped(axes[1], seis_types, "seismicity",  "By seismicity class (average across all sites & regions)")

    # Shared legend
    handles = [
        mpatches.Patch(
            facecolor=SOURCE_COLS[s], edgecolor="white",
            hatch=HATCHES[i], label=SOURCE_LABELS[s],
        )
        for i, s in enumerate(SOURCES)
    ]
    fig.legend(
        handles=handles, ncol=6,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        framealpha=0.9, edgecolor="#cccccc", fontsize=8,
        title="Source type",
    )

    fig.tight_layout()
    out = out_dir / f"fig3_by_region.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot seismic source contributions.")
    p.add_argument("--csv",  default="example_df1_stats.csv",
                   help="Path to input CSV file")
    p.add_argument("--poe",  type=float, default=0.002103,
                   help="PoE level to fix for Figure 1 (default: 0.002103 = 475 yr)")
    p.add_argument("--fmt",  default="png", choices=["png", "pdf", "svg"],
                   help="Output file format")
    p.add_argument("--dpi",  type=int, default=200,
                   help="Resolution for raster formats")
    p.add_argument("--outdir", default="./figures",
                   help="Directory to save figures")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.csv}")
    df = load_data(args.csv)
    print(f"  {len(df)} rows | {df['site_id'].nunique()} sites | "
          f"{df['poe'].nunique()} PoEs | seismicity: {sorted(df['seismicity'].unique())}")

    print("\nFigure 1 — source contributions by site")
    fig_by_site(df, poe_target=args.poe, fmt=args.fmt, dpi=args.dpi, out_dir=out_dir)

    print("\nFigure 2 — PoE sensitivity (all seismicity classes)")
    fig_poe_sensitivity(df, fmt=args.fmt, dpi=args.dpi, out_dir=out_dir)
    # Also produce one panel per seismicity class for cleaner layout
    for seis in df["seismicity"].unique():
        fig_poe_sensitivity(df, seismicity=seis, fmt=args.fmt, dpi=args.dpi, out_dir=out_dir)

    print("\nFigure 3 — regional and seismicity-class aggregates")
    fig_by_region(df, fmt=args.fmt, dpi=args.dpi, out_dir=out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
