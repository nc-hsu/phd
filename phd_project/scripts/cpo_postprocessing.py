"""Reduce a cyclic-pushover (CPO) and modal analysis to structural properties.

Notebook ``055`` writes one FEMA 461 cyclic pushover and one modal analysis per
design group; they run on a separate machine. Notebook ``056`` reduces their raw
output to the overstrength and period-based ductility that the nb ``072``
regression covariates need. This module holds the parts of that reduction worth
keeping out of the notebook: the ones with fiddly index bookkeeping (mapping a
node/dof to a row of the mode-shape matrix), an equation from a standard worth
citing exactly (FEMA P695), or a rule that deserves a unit test (when a
truncated analysis makes a quantity unavailable).

Layering, as for ``metaregression_analysis``: these are pure functions over
arrays and paths. They do not import the notebook's config, they do not plot,
and they do not know what a "design group" is -- the notebook owns the loop over
the 51 groups and the assembly of the results table.

Units throughout, fixed by what the analyses write:

===================  ========  ====================================================
quantity             unit      source
===================  ========  ====================================================
displacement         mm        ``po_curve.csv`` col 0 (roof control node)
force / base shear   N         ``po_curve.csv`` col 1 (load factor; the EC8
                               triangular pattern is normalised to 1, so the load
                               factor *is* the base shear)
mass                 kg        design file ``seismic_design_outputs.storey_masses``
weight               N         mass * 9.81
period               s         ``modal_properties.json`` ``eigenPeriod``
===================  ========  ====================================================

``G_MM`` below is gravity in mm/s^2 so that Eq. 6-7 returns a displacement in mm
without any further conversion. Mixing this up with 9.81 m/s^2 is the single
easiest way to get a ductility wrong by a factor of 1000, so the constant is
defined once, here, and nothing in this module uses a bare 9.81.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from fitpo import fit_tetralinear_backbone
from scipy.optimize import minimize

from standes.utils import generate_type_1_tag

FloatArr = npt.NDArray[np.float64]

# Gravity in mm/s^2. See the module docstring: everything geometric is in mm.
G_MM = 9810.0

# Gravity in m/s^2, used only to turn a mass [kg] into a weight [N].
G_SI = 9.81

# Fraction of the peak base shear that defines the ultimate displacement
# delta_u: FEMA P695 takes delta_u at 20 % strength loss on the descending
# branch of the pushover (Sec. 6.3, p. 6-8, and Figure 6-5).
ULTIMATE_STRENGTH_FRACTION = 0.80

# Row meanings of the (5, 2) breakpoint array fitpo.fit_cbf_backbone returns:
#   0 origin, 1 yield, 2 start of the brace-fracture drop, 3 bottom of that
#   drop / start of the residual branch, 4 end of the residual branch.
# Named so the notebook never indexes the array with a bare integer.
BB_ORIGIN, BB_YIELD, BB_FRACTURE, BB_RESIDUAL_START, BB_END = range(5)


# ---------------------------------------------------------------------------
# Loading raw analysis output
# ---------------------------------------------------------------------------


def load_cpo_curve(mdof_dir: str | Path, results_folder: str = "cyclic_pushover") -> FloatArr:
    """Load one cyclic-pushover curve.

    ``po_curve.csv`` is written by the analysis template with
    ``np.savetxt(np.column_stack([disps, lf]))``: no header, comma-delimited,
    two columns of roof displacement [mm] and base shear [N], in time order
    over the whole cyclic history (so the displacement column is *not*
    monotone -- that is the point of it).

    Returns an ``(N, 2)`` array, exactly the shape ``fitpo.fit_envelope``
    expects.
    """
    path = Path(mdof_dir) / results_folder / "po_curve.csv"
    return np.loadtxt(path, delimiter=",")


def load_modal_properties(mdof_dir: str | Path, results_folder: str = "modal") -> dict:
    """Load ``modal_properties.json`` (OpenSees ``modalProperties -return``).

    Every value is a list over modes. The one this module needs is
    ``eigenPeriod``; ``eigenPeriod[0]`` is the fundamental period T1 [s].
    """
    path = Path(mdof_dir) / results_folder / "modal_properties.json"
    with open(path) as f:
        return json.load(f)


def floor_node_tags(n_levels: int) -> list[int]:
    """Node tags of the floors, base excluded, in order from level 2 to the roof.

    The model tags its grid nodes ``generate_type_1_tag(prefix, x, y, z, q, a)``
    = ``f"{prefix}{xx}{yy}{zz}{q}{a}"`` with two-digit indices, and the design
    helpers take the top-*left* corner (x = y = 1, q = a = 0) as the control
    node. Level index ``z`` runs 1 .. n_levels with ``z = 1`` the (restrained)
    base, so the floors carrying mass are ``z = 2 .. n_levels``:

        3-storey (n_levels = 4): 101010200, 101010300, 101010400
        5-storey (n_levels = 6): 101010200, ...,       101010600

    Derived rather than hardcoded so that a change to the tagging scheme in
    ``standes`` surfaces here instead of silently selecting the wrong nodes.
    """
    return [generate_type_1_tag(1, 1, 1, z, 0, 0) for z in range(2, n_levels + 1)]


def first_mode_translational(
    mdof_dir: str | Path, n_levels: int, results_folder: str = "modal"
) -> FloatArr:
    """First-mode horizontal ordinates at each floor, base excluded.

    ``mode_shapes.csv`` is an ``(systemSize, n_modes)`` matrix whose *rows* are
    equation numbers, not nodes. ``node_info.json`` is the map: a list of
    ``[node_tag, dof, eq_number]`` triples, with ``eq_number = -1`` marking a
    restrained dof. So reading phi at a floor means going
    node tag -> (tag, dof = 1) -> equation number -> row of the matrix.

    Returns a 1-D array of length ``n_levels - 1``, ordered from the first floor
    up to the roof, so that ``phi[-1]`` is the roof ordinate ``phi_1r`` of
    FEMA P695 Eq. 6-8.

    Raises
    ------
    KeyError
        If a floor node has no free horizontal dof -- which would mean the
        model or the tagging convention is not what this function assumes,
        and silently dropping the level would corrupt C0.
    """
    modal_dir = Path(mdof_dir) / results_folder
    with open(modal_dir / "node_info.json") as f:
        node_info = json.load(f)

    # (node_tag, dof) -> equation number, keeping only the free dofs.
    eq_of = {(int(tag), int(dof)): int(eq) for tag, dof, eq in node_info if int(eq) >= 0}

    phi = np.loadtxt(modal_dir / "mode_shapes.csv", delimiter=",")
    phi = np.atleast_2d(phi)

    ordinates = []
    for tag in floor_node_tags(n_levels):
        # dof 1 is the horizontal (x) translation -- the pushover direction.
        key = (tag, 1)
        if key not in eq_of:
            raise KeyError(
                f"node {tag} has no free dof-1 in {modal_dir / 'node_info.json'}; "
                "the model's node tagging is not what floor_node_tags() assumes"
            )
        ordinates.append(phi[eq_of[key], 0])

    return np.asarray(ordinates, dtype=np.float64)


def load_design_quantities(design_file: str | Path) -> dict:
    """Pull the design-side quantities the reduction needs out of ``{tag}_out.json``.

    The base-shear and mass bases are already consistent with the pushover:
    ``design_baseshear / (seismic_mass * g)`` reproduces
    ``design_spectral_acceleration``, and equals the ``Vb_coeff`` (C_s) that nb
    ``072`` uses, while the analysis model carries exactly ``seismic_mass``
    (checked against the modal ``totalMass``). So no per-frame rescaling is
    applied anywhere in this module.

    Returns a dict with
    ``Vb_design_N``, ``W_N``, ``seismic_mass_kg``, ``storey_masses_kg``,
    ``design_period_s``, ``roof_height_mm`` and ``n_levels``.
    """
    with open(design_file) as f:
        d = json.load(f)
    sdo = d["seismic_design_outputs"]
    level_coords = d["structure"]["level_coordinates"]

    return {
        "Vb_design_N": float(sdo["design_baseshear"]),
        "W_N": float(sdo["seismic_mass"]) * G_SI,
        "seismic_mass_kg": float(sdo["seismic_mass"]),
        "storey_masses_kg": np.asarray(sdo["storey_masses"], dtype=np.float64),
        "design_period_s": float(sdo["design_period"]),
        "roof_height_mm": float(level_coords[-1]),
        "n_levels": len(level_coords),
    }


# ---------------------------------------------------------------------------
# FEMA P695 quantities
# ---------------------------------------------------------------------------


def fema_C0(storey_masses: npt.ArrayLike, phi_floors: npt.ArrayLike) -> float:
    """FEMA P695 Eq. 6-8: the modal coefficient C0.

    Relates fundamental-mode (SDOF) displacement to roof displacement:

        C0 = phi_1r * SUM_x (m_x * phi_1x) / SUM_x (m_x * phi_1x^2)

    where ``m_x`` is the mass at level x and ``phi_1x`` (``phi_1r``) the
    fundamental-mode ordinate at level x (at the roof). The sum runs over the N
    levels above the base. FEMA P695 takes this from Eq. C3-4 of ASCE/SEI
    41-06.

    The expression is invariant to the scaling of the mode shape -- replacing
    phi by a*phi multiplies numerator and denominator by a^2 -- so whatever
    normalisation OpenSees used for the eigenvector does not matter, and
    ``phi_floors`` is passed through unnormalised. ``test_cpo_postprocessing``
    pins that invariance.

    Parameters
    ----------
    storey_masses
        Masses at the N levels above the base [kg], ordered from the first
        floor up to the roof.
    phi_floors
        Fundamental-mode ordinates at the same N levels, in the same order, so
        that the last entry is the roof.
    """
    m = np.asarray(storey_masses, dtype=np.float64)
    phi = np.asarray(phi_floors, dtype=np.float64)
    if m.shape != phi.shape:
        raise ValueError(
            f"storey_masses {m.shape} and phi_floors {phi.shape} must have the "
            "same length (one entry per level above the base)"
        )
    denominator = float(np.sum(m * phi**2))
    if denominator == 0.0:
        raise ValueError("SUM(m * phi^2) is zero; the mode shape is all zeros at the floors")
    return float(phi[-1] * np.sum(m * phi) / denominator)


def delta_y_eff_p695(C0: float, V_max: float, W: float, period: float) -> float:
    """FEMA P695 Eq. 6-7: the effective yield roof drift displacement [mm].

        delta_y,eff = C0 * (V_max / W) * (g / (4 pi^2)) * (max(T, T1))^2

    Verified verbatim against FEMA P695 Ch. 6 Sec. 6.3, p. 6-8, where "T is the
    fundamental period (CuTa, defined by Equation 5-5), and T1 is the
    fundamental period of the archetype model computed using eigenvalue
    analysis".

    **This function takes whichever period the caller decides to use and does
    not apply the max() itself.** Notebook 056 deliberately passes the modal T1
    for the reported value -- a departure from the letter of P695, which would
    take the larger of the code period and T1, and for these EC8 designs the
    code period is often the larger. The notebook also computes the strict
    ``max(design_period, T1)`` variant alongside as a diagnostic, so the size of
    that choice is visible rather than buried. Keeping the max() out of here is
    what makes both callable.

    Parameters
    ----------
    C0
        Modal coefficient from :func:`fema_C0`.
    V_max
        Peak base shear of the pushover [N].
    W
        Seismic weight [N], on the same basis as ``V_max``.
    period
        Period [s] -- see the note above on which one.
    """
    return float(C0 * (V_max / W) * (G_MM / (4.0 * np.pi**2)) * period**2)


# ---------------------------------------------------------------------------
# Reading quantities off an idealised backbone
# ---------------------------------------------------------------------------


def envelope_peak_force(envelope: npt.ArrayLike) -> tuple[float, float]:
    """Peak base shear of the fitted envelope, and the displacement it occurs at.

    This is ``V_max``: the definition nb 055 and nb 072 both use, and the one
    the overstrength ``Omega = V_max / V_design`` is built on. It is read from
    the envelope rather than from the idealised backbone because the backbone's
    yield force is a fitted line intersection that can sit well above any force
    the structure actually carried -- see :func:`backbone_metrics`.

    Returns ``(V_max [N], d_at_V_max [mm])``.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if env.ndim != 2 or env.shape[1] != 2:
        raise ValueError(f"envelope must be shape (M, 2); got {env.shape}")
    i = int(np.argmax(env[:, 1]))
    return float(env[i, 1]), float(env[i, 0])


def backbone_overshoots_envelope(
    f_y_backbone: float, v_max_envelope: float, tolerance: float = 0.02
) -> bool:
    """Has the idealised backbone's yield force run above the envelope peak?

    ``fitpo.fit_cbf_backbone`` places the yield point where the idealised
    elastic line meets the post-buckling branch. On a curve with no clear
    fracture, or a long soft plateau before one, that intersection can land far
    above the largest force in the data -- a sign the idealisation has not
    described this curve well, and that its ``d_y`` (and so every ductility
    built on it) should not be trusted without looking at the plot.

    ``tolerance`` allows the small overshoot that is normal and harmless.
    """
    return bool(f_y_backbone > (1.0 + tolerance) * v_max_envelope)


def backbone_metrics(backbone: npt.ArrayLike) -> dict:
    """Read the strength and displacement quantities off a CBF backbone.

    ``backbone`` is the ``(5, 2)`` breakpoint array from
    ``fitpo.fit_cbf_backbone``:

        [[0, 0], [d_y, f_y], [d_joint, f_joint], [d_r2, f_r2], [d_max, f_end]]

    with the segments elastic -> post-buckling -> fracture drop -> residual.

    ``d_f`` is taken as ``d_joint`` (row 2): the point where the post-buckling
    branch ends and the steep drop caused by brace fracture begins, which is
    the displacement the brief asks for.

    ``f_max_bb_N`` is the largest force over *all* breakpoints, taken over the
    whole array rather than from a fixed row because the peak sits at
    ``f_joint`` when the post-buckling branch hardens but back at ``f_y`` when
    it has already softened.

    **Do not use ``f_max_bb_N`` as the overstrength numerator.** The backbone's
    yield point is the intersection of the idealised elastic line with the
    post-buckling branch, so ``f_y`` routinely overshoots the force the
    structure actually reached -- on the ``group_3s_00`` test case it lands at
    734 kN against an envelope peak of 442 kN, which would report Omega = 8.4
    instead of 5.05. V_max must come from the envelope
    (:func:`envelope_peak_force`), which is also how nb 055 and nb 072 define
    it. ``f_y_N`` is still worth carrying for the secondary Fy/Vb ratio, and
    :func:`backbone_overshoots_envelope` flags the designs where the
    idealisation has run away from the data.

    Returns a dict of ``d_y_mm, f_y_N, d_f_mm, f_f_N, f_max_bb_N, d_max_mm,
    k_e_N_per_mm, k_pb_N_per_mm, k_r_N_per_mm``; stiffnesses in N/mm.
    """
    bb = np.asarray(backbone, dtype=np.float64)
    if bb.shape != (5, 2):
        raise ValueError(f"backbone must be shape (5, 2); got {bb.shape}")

    d_y, f_y = bb[BB_YIELD]
    d_f, f_f = bb[BB_FRACTURE]
    d_r, f_r = bb[BB_RESIDUAL_START]
    d_end, f_end = bb[BB_END]

    def _slope(p0: FloatArr, p1: FloatArr) -> float:
        # Guarded against a zero-length segment, which fit_cbf_backbone can
        # produce when the fracture drop is vertical.
        return float((p1[1] - p0[1]) / max(p1[0] - p0[0], 1e-12))

    return {
        "d_y_mm": float(d_y),
        "f_y_N": float(f_y),
        "d_f_mm": float(d_f),
        "f_f_N": float(f_f),
        "f_max_bb_N": float(np.max(bb[:, 1])),
        "d_max_mm": float(d_end),
        "k_e_N_per_mm": _slope(bb[BB_ORIGIN], bb[BB_YIELD]),
        "k_pb_N_per_mm": _slope(bb[BB_YIELD], bb[BB_FRACTURE]),
        "k_r_N_per_mm": _slope(bb[BB_RESIDUAL_START], bb[BB_END]),
    }


def displacement_at_strength_loss(
    curve: npt.ArrayLike,
    v_max: float | None = None,
    strength_fraction: float = ULTIMATE_STRENGTH_FRACTION,
) -> float:
    """Displacement at which ``curve`` has fallen to ``strength_fraction * v_max`` [mm].

    FEMA P695's ultimate roof displacement delta_u: the displacement, *after*
    the peak, at which the pushover has lost 20 % of its strength (Sec. 6.3,
    p. 6-8). Works on any piecewise-linear ``(M, 2)`` curve -- a fitted envelope
    or an idealised backbone -- by walking segments from the peak onwards and
    interpolating within the first one that brackets the target force. The curve
    is piecewise linear, so that interpolation is exact.

    ``v_max`` is the reference force the 80 % is taken of. **Pass the envelope's
    peak explicitly when measuring on a backbone.** Left to default, each curve
    uses its own peak, and the idealised backbone's peak can be an inflated
    elastic-line intersection rather than a force the structure ever carried:
    on the ``group_3s_01`` test case that pulls delta_u from 117.2 mm down to
    53.7 mm, a spurious halving of the ductility. The envelope peak is also what
    nb 055 and nb 072 mean by V_max.

    Returns ``nan`` when the curve never falls that far -- typically a cyclic
    pushover truncated before the structure degraded. It must stay NaN:
    substituting the last point of a trace that merely stopped early would
    invent a ductility fixed by where the analysis happened to halt. The
    notebook flags these for manual review instead.
    """
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"curve must be shape (M, 2); got {arr.shape}")

    d, f = arr[:, 0], arr[:, 1]
    i_peak = int(np.argmax(f))
    reference = float(f[i_peak]) if v_max is None else float(v_max)
    target = strength_fraction * reference

    for i in range(i_peak, len(d) - 1):
        f0, f1 = f[i], f[i + 1]
        # The first descending segment that brackets the target force.
        if f0 >= target >= f1 and f0 != f1:
            return float(d[i] + (d[i + 1] - d[i]) * (f0 - target) / (f0 - f1))

    return float("nan")


# ---------------------------------------------------------------------------
# Did the analysis actually finish?
# ---------------------------------------------------------------------------


def cpo_completeness(
    displacements: npt.ArrayLike,
    target_displacements: npt.ArrayLike,
    du: float,
    amplitude_tolerance: float = 0.99,
    return_to_zero_steps: float = 5.0,
) -> dict:
    """Decide whether a cyclic pushover ran its whole displacement trace.

    The FEMA 461 protocol nb ``055`` writes is a ladder of 12 amplitude levels,
    each 1.4x the previous, every level run as (+, -, +, -), then a final return
    to 0 -- 49 target displacements in all. A run that converged all the way
    through therefore both reaches the last amplitude and comes back to zero.
    A run that stopped early usually fails both, but a run that died *during*
    the final return fails only the second, so both are checked and reported
    separately rather than collapsed into one boolean.

    Parameters
    ----------
    displacements
        Column 0 of ``po_curve.csv`` [mm].
    target_displacements
        The protocol's target displacements [mm], from
        ``loading_protocols.get_FEMA461_displacements_for_building``. Only their
        magnitudes are used, so the trailing 0 is harmless.
    du
        The ramp displacement step [mm] the analysis actually used
        (``cpo_du.resolve_cpo_du``). Sets how close to zero the trace must end.
    amplitude_tolerance
        Fraction of U_max that counts as having reached the last level.
    return_to_zero_steps
        The final point must be within this many ``du`` of zero.

    Returns
    -------
    dict with
        ``cpo_complete`` (both conditions), ``reached_max_amplitude``,
        ``ends_at_zero``, ``max_level_reached`` (1-based index into the
        amplitude ladder, 0 if not even the first was reached), ``n_levels``,
        ``max_abs_d_mm``, ``U_max_mm`` and ``frac_of_Umax``.
    """
    d = np.asarray(displacements, dtype=np.float64)
    targets = np.abs(np.asarray(target_displacements, dtype=np.float64))

    # The distinct amplitude levels, smallest first: the protocol repeats each
    # one four times and ends with a 0, so unique() recovers the ladder.
    levels = np.unique(targets[targets > 0.0])
    u_max = float(levels[-1])

    max_abs_d = float(np.max(np.abs(d)))

    # How many rungs of the ladder the trace actually got to.
    max_level_reached = int(np.sum(levels <= max_abs_d * (1.0 / amplitude_tolerance)))
    max_level_reached = min(max_level_reached, len(levels))

    reached_max_amplitude = max_abs_d >= amplitude_tolerance * u_max
    ends_at_zero = abs(float(d[-1])) < return_to_zero_steps * du

    return {
        "cpo_complete": bool(reached_max_amplitude and ends_at_zero),
        "reached_max_amplitude": bool(reached_max_amplitude),
        "ends_at_zero": bool(ends_at_zero),
        "max_level_reached": max_level_reached,
        "n_levels": int(len(levels)),
        "max_abs_d_mm": max_abs_d,
        "U_max_mm": u_max,
        "frac_of_Umax": max_abs_d / u_max,
    }


# ---------------------------------------------------------------------------
# Preparing the envelope for fitting
# ---------------------------------------------------------------------------


def truncate_envelope_at_zero(envelope: npt.ArrayLike) -> FloatArr:
    """Cut the envelope's post-peak tail at the first non-positive force.

    Every usable sample envelope runs on past the point where the braces have
    gone and into negative force -- down to -16 % of Fmax on the 3-storeys and
    **-56 %** on ``group_5s_06``. That tail is not part of a backbone, and
    leaving it in gives the area-minimising fit a long stretch of meaningless
    curve to chase, which drags the residual branch down with it.

    The cut point is interpolated, so the returned curve ends exactly at
    ``(d, 0)`` rather than at whichever sample happened to be first below zero.
    An envelope that never goes non-positive is returned unchanged.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if env.ndim != 2 or env.shape[1] != 2:
        raise ValueError(f"envelope must be shape (M, 2); got {env.shape}")

    d, f = env[:, 0], env[:, 1]
    i_peak = int(np.argmax(f))
    below = np.flatnonzero(f[i_peak:] <= 0.0)
    if below.size == 0:
        return env.copy()

    j = int(below[0]) + i_peak
    if j == 0:
        raise ValueError("envelope is non-positive at its peak")
    f0, f1 = f[j - 1], f[j]
    d_zero = d[j - 1] + (d[j] - d[j - 1]) * f0 / (f0 - f1)
    return np.vstack([env[:j], [[d_zero, 0.0]]])


def resample_envelope(
    envelope: npt.ArrayLike, df_tol: float, dd_tol: float
) -> FloatArr:
    """Subdivide the envelope's segments without changing its shape.

    Points are inserted *along* each segment, so the piecewise-linear curve is
    mathematically identical -- the area under it is unchanged to machine
    precision (measured 4.5e-16) and every original vertex survives in the
    output. What changes is only the resolution available to whatever is
    choosing knots on it.

    Each segment is split into enough pieces that it spans no more than
    ``df_tol`` in force and no more than ``dd_tol`` in displacement.

    Parameters
    ----------
    envelope
        ``(M, 2)`` piecewise-linear curve.
    df_tol
        Maximum force span of a segment, in force units (the caller normally
        passes a fraction of Fmax).
    dd_tol
        Maximum displacement span of a segment [mm].
    """
    env = np.asarray(envelope, dtype=np.float64)
    if env.ndim != 2 or env.shape[1] != 2:
        raise ValueError(f"envelope must be shape (M, 2); got {env.shape}")
    if df_tol <= 0.0 or dd_tol <= 0.0:
        raise ValueError(f"tolerances must be positive; got {df_tol}, {dd_tol}")

    out = [env[0]]
    for i in range(len(env) - 1):
        d0, f0 = env[i]
        d1, f1 = env[i + 1]
        n = int(max(abs(f1 - f0) / df_tol, abs(d1 - d0) / dd_tol, 1))
        for k in range(1, n + 1):
            t = k / n
            out.append([d0 + t * (d1 - d0), f0 + t * (f1 - f0)])
    return np.asarray(out, dtype=np.float64)


def fit_window(envelope: npt.ArrayLike, fraction: float = 0.5) -> tuple[float, str]:
    """Upper limit of the window the backbone fit is scored over.

    The area error is measured from 0 up to the post-peak displacement at
    ``fraction * Fmax``. Where a truncated analysis never gets that far, the
    end of the envelope is used instead and the rule is reported, so a window
    that does not mean what it usually means is visible in the results table
    rather than silently different.

    Returns ``(d_hi, rule)`` with ``rule`` either ``"<fraction>*Fmax"`` or
    ``"end-of-envelope"``.
    """
    env = np.asarray(envelope, dtype=np.float64)
    v_max, _ = envelope_peak_force(env)
    d_hi = displacement_at_strength_loss(env, v_max, fraction)
    if np.isfinite(d_hi):
        return float(d_hi), f"{fraction:g}*Fmax"
    return float(env[-1, 0]), "end-of-envelope"


def backbone_area_error(
    envelope: npt.ArrayLike, backbone: npt.ArrayLike, d_hi: float
) -> float:
    """Absolute area between envelope and backbone over ``[0, d_hi]``.

    Both curves are piecewise linear, so integrating on the union of their
    abscissae (plus the window ends) is exact rather than an approximation:
    between consecutive breakpoints of either curve the difference is linear,
    which the trapezoid rule integrates without error. The one caveat is a
    crossing *inside* a panel, where the absolute difference has a kink the
    rule rounds off; with both curves sampled this finely the effect is
    negligible.
    """
    env = np.asarray(envelope, dtype=np.float64)
    bb = np.asarray(backbone, dtype=np.float64)
    if d_hi <= 0.0:
        raise ValueError(f"d_hi must be positive; got {d_hi}")

    d_e, f_e = env[:, 0], env[:, 1]
    d_b, f_b = bb[:, 0], bb[:, 1]
    grid = np.unique(np.concatenate([
        d_e[(d_e >= 0.0) & (d_e <= d_hi)],
        d_b[(d_b >= 0.0) & (d_b <= d_hi)],
        [0.0, d_hi],
    ]))
    diff = np.abs(np.interp(grid, d_e, f_e) - np.interp(grid, d_b, f_b))
    return float(np.trapezoid(diff, grid))


# ---------------------------------------------------------------------------
# Choosing the backbone's two interior knots
# ---------------------------------------------------------------------------


def fit_tetralinear_optimised(
    envelope: npt.ArrayLike,
    *,
    d_hi: float,
    grid_n: int = 20,
    refine: bool = True,
    knots: tuple[float, float] | None = None,
    collapse_residual: bool = False,
) -> dict:
    """Fit a tetralinear backbone, choosing its two knots to minimise area error.

    ``fitpo.fit_tetralinear_backbone`` fixes the two interior knots and then
    fits the three forces. This chooses those knots: it searches over the knot
    *displacements* ``(d_f, d_r)`` for the pair whose fitted backbone departs
    least, by absolute area over ``[0, d_hi]``, from the envelope.

    Knots are addressed by displacement rather than by force fraction
    deliberately. The fraction interface resolves a fraction through
    "first data point at or below ``fraction * Fmax``", which is not monotone
    in the sampling of the curve, so a finer search could return a *worse* fit.
    By displacement the objective is continuous and a finer search can only
    help -- which is what makes the coarse-grid-then-refine strategy below
    sound.

    Parameters
    ----------
    envelope
        ``(M, 2)`` fit-ready envelope: already truncated at zero force and, if
        wanted, resampled. The backbone is fitted over the whole of it, not
        only over ``[0, d_hi]`` -- see the note below.
    d_hi
        Upper limit of the scoring window, from :func:`fit_window`.
    grid_n
        Number of candidate displacements per axis in the coarse stage; that
        stage costs about ``grid_n * (grid_n - 1) / 2`` fits.
    refine
        Run a Nelder-Mead polish from the coarse winner.
    knots
        Pin ``(d_f, d_r)`` instead of searching, for a curve whose automatic
        fit has been rejected on review.
    collapse_residual
        Passed through to ``fit_tetralinear_backbone``.

    Returns
    -------
    dict with the backbone, the chosen knots, the equivalent force fractions
    (derived, for the record), the absolute and normalised area error, the
    number of candidates evaluated, and whether the optimum sits on a search
    bound.

    Notes
    -----
    The backbone is fitted over the full envelope while the score covers only
    ``[0, d_hi]``. That is deliberate: on every sample curve the optimal
    ``d_r`` lies *beyond* ``d(0.5*Fmax)``, so truncating the curve at the
    scoring window would delete the knot. The window still identifies ``d_r``
    well, because the cliff segment leading to it crosses the window. One
    consequence to carry into any interpretation: ``f_r`` acts as a control on
    the cliff's slope rather than as a physical residual strength, and sits
    well below the envelope at ``d_r``.

    Raises
    ------
    ValueError
        If no candidate knot pair yields a valid fit -- typically a curve with
        no descending branch, which must be flagged rather than fitted.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if env.ndim != 2 or env.shape[1] != 2:
        raise ValueError(f"envelope must be shape (M, 2); got {env.shape}")

    d, f = env[:, 0], env[:, 1]
    v_max, d_at_v_max = envelope_peak_force(env)
    d_max = float(d[-1])
    n_eval = 0

    def attempt(d_f: float, d_r: float):
        # One candidate: fit at these knots and score it. Invalid geometry and
        # fitpo's own rejections both come back as None so the search can move
        # on rather than abort.
        nonlocal n_eval
        if not 0.0 < d_f < d_r <= d_max:
            return None
        try:
            bb = fit_tetralinear_backbone(
                env, d_f=float(d_f), d_r=float(d_r),
                collapse_residual=collapse_residual)
        except ValueError:
            return None
        n_eval += 1
        return backbone_area_error(env, bb, d_hi), float(d_f), float(d_r), bb

    if knots is not None:
        best = attempt(*knots)
        if best is None:
            raise ValueError(f"pinned knots {knots} give no valid fit")
        stage = "pinned"
    else:
        # --- coarse stage: a grid over the descending branch -------------------
        # The joint cannot precede the peak, and both knots must fit between it
        # and the end of the curve.
        lo = d_at_v_max + (d_max - d_at_v_max) / (grid_n + 1)
        candidates = np.linspace(lo, d_max, grid_n)
        best = None
        for i, d_f in enumerate(candidates):
            for d_r in candidates[i + 1:]:
                got = attempt(d_f, d_r)
                if got is not None and (best is None or got[0] < best[0]):
                    best = got
        if best is None:
            raise ValueError(
                "no valid tetralinear fit for any candidate knot pair "
                "(no usable descending branch?)")
        stage = "grid"

        # --- refinement: the objective is continuous in (d_f, d_r) -------------
        if refine:
            span = (d_max - d_at_v_max) / grid_n

            def objective(x):
                got = attempt(x[0], x[1])
                # A large finite penalty keeps Nelder-Mead inside the feasible
                # region without it having to handle NaN or inf.
                return got[0] if got is not None else 1e18

            res = minimize(objective, x0=[best[1], best[2]], method="Nelder-Mead",
                           options={"xatol": span * 1e-3, "fatol": best[0] * 1e-6,
                                    "maxiter": 200, "disp": False})
            polished = attempt(res.x[0], res.x[1])
            if polished is not None and polished[0] < best[0]:
                best, stage = polished, "grid+refine"

    area, d_f, d_r, bb = best
    return {
        "backbone": bb,
        "d_f_mm": d_f,
        "d_r_mm": d_r,
        # Derived, so the fit can still be described the way fitpo's other
        # entry point would, and cross-checked against a hand-picked pair.
        "joint_force_fraction": float(np.interp(d_f, d, f)) / v_max,
        "residual_force_fraction": float(np.interp(d_r, d, f)) / v_max,
        "fit_area_error": area,
        # Normalised by the box the window spans, so designs of very different
        # strength are comparable and one threshold can flag bad fits.
        "fit_area_error_norm": area / (v_max * d_hi),
        "fit_window_d_hi_mm": float(d_hi),
        "n_knot_candidates": n_eval,
        "fit_stage": stage,
        # The grid's outer edge is the end of the curve, so a winner sitting on
        # it means the search wanted to go further than the curve allows.
        "knot_on_bound": bool(abs(d_r - d_max) < 1e-9),
    }
