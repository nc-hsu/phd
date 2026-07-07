"""Modal -> equivalent-SDOF conversion (participation factor and effective mass).

Single source of truth for the participation-factor maths originally developed
inline in cell 6 of
``notebooks/03_WP1_ground_motion_set/wp1pt4pt1e_parameterised_sdof_model_for_ida.ipynb``.

Given the modal-analysis output of an MDOF model (mass matrix, mode shapes,
influence matrix and node info, as written by a ``standes`` modal analysis into a
``modal/`` results folder) these functions compute the first-mode participation
factor ``Gamma`` and equivalent-SDOF mass ``m_star`` (Fajfar 2000;
Baltzopoulos et al. 2018), and convert a full-system pushover curve into
equivalent-SDOF coordinates.
"""

import json
from pathlib import Path

import numpy as np


def _calculate_participation_factor(
    mass_matrix, modeshape_matrix, influence_matrix, node_info
):
    """only works for 2D frames with excitation in the x-direction"""
    unconstrained_node_info = sorted(
        [n for n in node_info if n[2] >= 0], key=lambda x: x[2]
    )

    r1 = influence_matrix[:, 0]  # influence vector for the first mode
    idx_influence = np.where(r1 == 1)[0]  # dofs with mass in the x direction

    # required parameters
    phi_1 = modeshape_matrix[:, 0]  # first mode shape

    aa = sorted(
        [unconstrained_node_info[ii] for ii in idx_influence], key=lambda x: x[0]
    )  # sort by node tag
    # node tags / dof indexes for the x-direction dofs, reshaped to one column
    # per column line (4 columns for a single-bay-by-3-bay frame)
    node_tags_used = np.reshape([a[0] for a in aa], (-1, 4), order="F")
    idxs = np.reshape([a[2] for a in aa], (-1, 4), order="F")

    phi = phi_1[idxs]
    norm_idx = np.abs(phi[-1, :]).argmax()
    normalisation_factor = phi[-1, norm_idx]
    phi /= normalisation_factor
    phi = phi.reshape(-1, 1)  # column vector

    m_vals = np.diagonal(mass_matrix)[idxs]
    m = np.zeros((len(phi), len(phi)))
    np.fill_diagonal(m, m_vals)  # diagonal mass matrix

    r = np.ones(phi.shape)

    generalised_mass = (phi.T @ m @ phi)[0, 0]  # modal mass
    m_star = (phi.T @ m @ r)[0, 0]  # Fajfar (2000); Baltzopoulos et al. (2018)
    effective_mass = (
        (r.T @ m @ phi @ phi.T @ m @ r) / (phi.T @ m @ phi)
    )[0, 0]  # participating mass
    Gamma = m_star / generalised_mass
    return Gamma, m, phi, node_tags_used, m_star, generalised_mass, effective_mass


def convert_poc_to_eq_sdof_poc(modal_results_folder, poc=None):
    """Convert a full-system pushover curve to equivalent-SDOF coordinates.

    Loads the modal-analysis products from ``modal_results_folder`` and returns
    ``(eq_sdof_poc, Gamma, m_star, generalised_mass, effective_mass)``. When
    ``poc`` is ``None`` the first element is ``None`` (only the participation
    quantities are needed). Otherwise ``eq_sdof_poc = poc / Gamma``.
    """
    modal_results_folder = Path(modal_results_folder)

    M = np.loadtxt(modal_results_folder / "mass_matrix.csv", delimiter=",")
    Phi = np.loadtxt(modal_results_folder / "mode_shapes.csv", delimiter=",")
    influence_matrix = np.loadtxt(
        modal_results_folder / "influence_matrix.csv", delimiter=","
    )
    node_info = json.load(open(modal_results_folder / "node_info.json", "r"))

    (
        Gamma,
        m,
        phi,
        node_tags_used,
        m_star,
        generalised_mass,
        effective_mass,
    ) = _calculate_participation_factor(
        M, Phi, influence_matrix, node_info
    )

    eq_sdof_poc = None if poc is None else poc / Gamma

    return eq_sdof_poc, Gamma, m_star, generalised_mass, effective_mass
