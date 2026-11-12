"""
Export the ASO24 correlation coefficients used in nb 002 to a CSV file.

ASO24 is the ANN-based correlation model of Aristeidou, Shahnazaryan and O'Reilly (2024),
see ``data_raw/correlation_models/ASO24/reference.txt``. Its coefficients are computed with
``asocorrelationmodel``, a locally modified fork of the authors' code that is not published.
nb 002 therefore does not compute the model itself: it reads the matrix this script writes,

    data_raw/correlation_models/ASO24/aso24_correlations.csv

and running this script is the only way that file is produced.

How to run
----------
From the repository root, in the project environment with ``asocorrelationmodel`` installed
(an editable install of the fork) and after nb 002 has written the positive-definite
correlation CSVs (the SA periods are taken from them)::

    python -m phd_project.scripts.WP1_ground_motion_set.correlationmodels.export_aso24_correlations

The matrix
----------
Rows and columns are the same (component, IM, period) labels nb 002 uses for its own models:

- ``("rotD50", "SA", T)`` for every SA period T of the CG26 reverse-faulting ASC model,
- ``("rotD50", "RSD595", "None")``,
- ``("rotD50", "AvgSA_3", 1.0)`` and ``("rotD50", "AvgSA_3", 2.0)``: ASO24's Sa_avg3 at
  T = 1 s and 2 s.

The diagonal is 1. The file is written with ``to_csv(..., index=True)`` and is read back with
``correlation.read_correlation_csv``, which restores the float and "None" period labels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from asocorrelationmodel.correlation_models import aso2024_correlation

from phd_project.config.config import load_config
from phd_project.scripts.WP1_ground_motion_set.correlation import (
    load_positive_definite_correlations,
)

# Periods of the two Sa_avg3 IMs included in the matrix
AVG_SA_T1 = 1
AVG_SA_T2 = 2


def output_path(cfg: dict) -> Path:
    """Where the ASO24 matrix is written (and where nb 002 reads it)."""
    return cfg["raw_data"]["rho_models"] / "ASO24" / "aso24_correlations.csv"


def sa_periods_of_cg26(cfg: dict) -> list[float]:
    """The SA periods of the CG26 model, in the order nb 002 uses them.

    Taken from the columns of the reverse-faulting ASC total-residual matrix, as nb 002 does
    for every literature model it compares against.
    """
    cg26 = load_positive_definite_correlations(cfg["results"]["corr_model"], "reverse")
    ims = cg26["asc"]["rho_total"].columns
    return [float(c[2]) for c in ims if c[1] == "SA"]


def build_aso24_matrix(im_periods: list[float]) -> pd.DataFrame:
    """The ASO24 correlation matrix for the given SA periods, RSD595 and Sa_avg3(1 s, 2 s).

    Index layout of the square array: the SA periods first (positions 0 .. n-1), then RSD595
    (n), then Sa_avg3 at T1 (n+1) and at T2 (n+2). Every pair is filled symmetrically.
    """
    n_sa = len(im_periods)
    aso24 = np.eye(n_sa + 3, n_sa + 3)
    i_rsd = n_sa          # position of RSD595
    i_avg = n_sa + 1      # position of Sa_avg3(T1); Sa_avg3(T2) follows it

    # SA-SA correlations, upper triangle mirrored to the lower one
    for ii, Ti in enumerate(im_periods):
        for jj, Tj in enumerate(im_periods[ii + 1:]):
            aso24[ii, ii + jj + 1] = aso2024_correlation("SA-SA", Ti, Tj)
            aso24[ii + jj + 1, ii] = aso2024_correlation("SA-SA", Ti, Tj)

    # SA-RSD595 correlations
    for ii, Ti in enumerate(im_periods):
        aso24[ii, i_rsd] = aso2024_correlation("SA-Ds595", Ti)
        aso24[i_rsd, ii] = aso2024_correlation("SA-Ds595", Ti)

    # SA-Sa_avg3 correlations
    for ii, Ti in enumerate(im_periods):
        for jj, Tj in enumerate([AVG_SA_T1, AVG_SA_T2]):
            aso24[ii, i_avg + jj] = aso2024_correlation("SA-Sa_avg3", Ti, Tj)
            aso24[i_avg + jj, ii] = aso2024_correlation("SA-Sa_avg3", Ti, Tj)

    # Sa_avg3-RSD595 correlations
    for jj, Tj in enumerate([AVG_SA_T1, AVG_SA_T2]):
        aso24[i_rsd, i_avg + jj] = aso2024_correlation("Sa_avg3-Ds595", Tj)
        aso24[i_avg + jj, i_rsd] = aso2024_correlation("Sa_avg3-Ds595", Tj)

    # Sa_avg3(T1)-Sa_avg3(T2) correlation
    aso24[i_avg, i_avg + 1] = aso2024_correlation("Sa_avg3-Sa_avg3", AVG_SA_T1, AVG_SA_T2)
    aso24[i_avg + 1, i_avg] = aso2024_correlation("Sa_avg3-Sa_avg3", AVG_SA_T1, AVG_SA_T2)

    # Labels in nb 002's (component, IM, period) convention; "None" marks an IM without a
    # period. The Sa_avg3 periods are floats so they compare equal to 1.0 and 2.0.
    aso24_labels = [("rotD50", "SA", t) for t in im_periods] + \
                   [("rotD50", "RSD595", "None")] + \
                   [("rotD50", "AvgSA_3", float(AVG_SA_T1))] + \
                   [("rotD50", "AvgSA_3", float(AVG_SA_T2))]

    return pd.DataFrame(aso24,
                        index=pd.MultiIndex.from_tuples(aso24_labels),
                        columns=pd.MultiIndex.from_tuples(aso24_labels))


def main() -> Path:
    """Build the ASO24 matrix and write it to ``output_path``. Returns the file path."""
    cfg = load_config()
    aso24 = build_aso24_matrix(sa_periods_of_cg26(cfg))
    fp = output_path(cfg)
    fp.parent.mkdir(parents=True, exist_ok=True)
    aso24.to_csv(fp, index=True)
    print(f"ASO24 correlations ({aso24.shape[0]} x {aso24.shape[1]}) written to {fp}")
    return fp


if __name__ == "__main__":
    main()
