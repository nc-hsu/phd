# Correlation-model results of notebook 002

This folder holds the intensity-measure (IM) correlation matrices produced by
`notebooks/03_WP1_ground_motion_set/002-im_correlations.ipynb`. Each sub-folder holds
one set of matrices per fault-type assumption, and each file is one positive-definite (PD)
correlation matrix for one tectonic region and one residual type.

```
results/02_im_correlation_model/
    reverse/      <region>_<residual type>_PD.csv   (20 files)
    normal/       <region>_<residual type>_PD.csv   (20 files)
    strikeslip/   <region>_<residual type>_PD.csv   (20 files)
```

The **reverse-faulting** matrices are the correlation model of the paper (CG26). The
normal and strike-slip matrices are used only in the rake-sensitivity check of nb 002.

## File names: `<region>_<residual type>_PD.csv`

### Sub-folder: fault-type assumption

The style of faulting (rake) assigned to every record before the ground-motion model
predictions, and so the residuals, were computed.

| Folder | Fault type |
|---|---|
| `reverse` | reverse faulting (the published model) |
| `normal` | normal faulting |
| `strikeslip` | strike-slip faulting |

### `<region>`: tectonic region

| Tag | Region |
|---|---|
| `asc` | active shallow crust |
| `sslab` | subduction in-slab |
| `sinter` | subduction interface |
| `vran` | Vrancea (intermediate-depth) |

The subduction and Vrancea datasets are defined by the ESM record ID lists in
`data_raw/eshm20_trt_record_ids/`. The active-shallow-crust dataset is the rest of the
ESM flatfile, after the filters applied in nb 002.

### `<residual type>`: which residuals the correlations are computed from

The total residual $\Delta$ of each record is split by a mixed-effects regression (R,
`lme4`) into a between-event, a site-to-site and a within-site residual:

$$
\Delta_{es} = \delta B_e + \delta S2S_s + \delta WS_{es}
$$

| Tag | Correlation of |
|---|---|
| `rho_dBe` | the between-event residuals $\delta B_e$ |
| `rho_dS2Ss` | the site-to-site residuals $\delta S2S_s$ |
| `rho_dWSes` | the within-site (event- and site-corrected) residuals $\delta WS_{es}$ |
| `rho_total` | the total residuals, built from the three components: the sum of the three component covariances, divided by the product of the total standard deviations, with $\sigma^2 = \sigma_{B}^2 + \sigma_{S2S}^2 + \sigma_{WS}^2$. **This is the correlation model.** |
| `rho_total_B` | the total residuals $\Delta$, correlated directly (pairwise, ignoring missing values), as a check on `rho_total` |

### `_PD`: positive definite

Every matrix has been replaced by the nearest positive-definite correlation matrix
(`statsmodels.stats.correlation_tools.corr_nearest`, threshold `1e-8`, 200 iterations),
so it can be used directly, for example in a Cholesky decomposition or to simulate
correlated ground motions.

## Inside each file

Each file is a square correlation matrix with the same 38 IMs on both axes, in the same
order. Rows and columns carry three label levels: **component**, **IM** and
**period**.

| Component | IM | Period | Meaning |
|---|---|---|---|
| `GM` | `IA` | `None` | Arias intensity, geometric mean of the two horizontal components |
| `GM` | `RSD595` | `None` | significant duration (5–95 % of Arias intensity), geometric mean |
| `rotD50` | `AvgSA([0, 3])` | `3.0` | average spectral acceleration over the period range in brackets (seconds); the period label repeats the upper end of the range |
| `rotD50` | `AvgSA([0, 6])` | `6.0` | as above, over 0–6 s |
| `rotD50` | `PGA` | `None` | peak ground acceleration |
| `rotD50` | `SA` | `0.025` … `8.0` | 5 %-damped spectral acceleration at 33 periods, in seconds |

The first three lines of a file hold the column labels (component, IM, period), and the
first three columns of every following line hold the row labels. The period level is
mixed: a number where the IM has a period, and the text `None` where it has none.

## Reading a file

A plain `pandas.read_csv` turns the `None` labels into missing values or the periods into
text. Use the reader in this repo, which rebuilds the labels with the right types:

```python
from pathlib import Path
from phd_project.scripts.WP1_ground_motion_set.correlation import (
    read_correlation_csv,
    load_positive_definite_correlations,
)

# one matrix
rho = read_correlation_csv(Path("results/02_im_correlation_model/reverse/asc_rho_total_PD.csv"))
rho.loc[("rotD50", "SA", 1.0), ("GM", "RSD595", "None")]

# all regions and residual types of one fault type, as {region: {residual type: DataFrame}}
cg26 = load_positive_definite_correlations(Path("results/02_im_correlation_model"), "reverse")
cg26["asc"]["rho_total"]
```

## Source and licence

Produced by nb 002 of this repo from the ESM flatfile (INGV,
doi:10.13127/esm/flatfile.1.0, CC BY 4.0, which asks that it be cited).

Licence of these files: GNU General Public License v3.0 or later (SPDX
`GPL-3.0-or-later`); the full text is in `LICENSE` at the root of the repository.
