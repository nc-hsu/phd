"""Reusable loading-protocol generators.

Extracted (verbatim) from the inline definitions previously duplicated in
``wp1pt4pt1a_create_mdof_analysis_files.ipynb`` and
``wp1pt4pt1b_create_sdof_analysis_files.ipynb`` so the FEMA 461 cyclic-pushover
displacement history has a single source of truth.
"""

import json
from pathlib import Path

import numpy as np


def FEMA_461_loading_protocol(U_max, nlevels):
    # fixed at 10 steps
    # a_i+1 = 1.4*a_i
    # a_1 = 0.048*delta_M
    C = 1 / (1.4 ** (nlevels - 1))
    a_s = np.array([1.4 ** (ii) * C * U_max for ii in range(nlevels)])
    a_s = [sign * a for a in a_s for sign in (1, -1, 1, -1)]
    return a_s


def get_FEMA461_displacements_for_building(design_file_path: Path, n_levels, U_max: float | None = None):

    # maximum displacements are taken from analysis of the monotonic pushover curves
    # and 20% is added
    with open(design_file_path, "r") as f:
        design_data = json.load(f)

    height = design_data["structure"]["level_coordinates"][-1] / 1000   # im m
    Vbc = design_data["seismic_design_outputs"]["design_baseshear"] / \
          (design_data["seismic_design_outputs"]["seismic_mass"] * 9.81)

    def _dy(h, Vbc):
        # approximation of yield displacement
        return 0.00005 * h + 0.000188 * h * Vbc + 0.001858

    def _dr(h, Vbc):
        return 1.02 * 6.258 * _dy(h, Vbc)

    if U_max is None:
        U_max = 1.2 * _dr(height, Vbc) * height * 1000         # * height * 100 to get it in displacements in mm

    displacements = np.append(FEMA_461_loading_protocol(U_max, n_levels), [0])
    return displacements
