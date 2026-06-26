"""Parameterised equivalent-SDOF model for 3-storey CBFs.

Single source of truth for the SDOF parameterisation functions that were
originally developed inline in
``notebooks/03_WP1_ground_motion_set/wp1pt4pt1e_parameterised_sdof_model_for_ida.ipynb``.

The functions predict the backbone and hysteretic parameters of an equivalent
SDOF oscillator as a function of the design base-shear coefficient
``Vb_coeff`` (= design base shear / total seismic weight). The SDOF is modelled
in OpenSees as two parallel zero-length springs:

* a ``Steel02`` spring representing the soft-storey / column-frame contribution
  (``steel02_bb_params`` + ``steel02_hyst_params``), and
* a ``Hysteretic`` spring representing the braced-frame contribution
  (``hysteretic_bb_params`` + ``hysteretic_hyst_params``),

matching ``phd_project/scripts/templates/template_model_cbf_sdof.py``.

The fitted coefficients embedded below were obtained from the cyclic-pushover
fitting / optimisation of the case-study CBF dataset (3-storey buildings).
"""

import numpy as np
import pandas as pd


# Mean optimised Steel02 hysteresis parameters [a1, a2, R0, cR1, cR2] across the
# 3-storey case-study buildings (mean of *_optimised_parameters.json in
# data_processed/09_sdof_fitting/optimisation_results). Used by
# get_approximate_steel02_hyst_params(); previously the notebook global
# `hyst_param_stats.loc["mean", ...]`.
STEEL02_HYST_PARAMS_MEAN = [
    0.10928171323806772,
    3.458661900789691,
    20.18733608731711,
    0.7838453700836108,
    0.33423025227370395,
]


# ---------------------------------------------------------------------------
# Fitting primitives
# ---------------------------------------------------------------------------
def linear_model(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * (x**2) + b * x + c


def exponential_shifted(x, a, b, c):
    return a * np.exp(b * x) + c


def linear_constant_model(x, x_knot, m1, c1):
    """Initial linear drop that becomes perfectly constant after x_knot."""
    y_knot = m1 * x_knot + c1
    if isinstance(x, float) or isinstance(x, int):
        return float(np.where(x < x_knot, m1 * x + c1, y_knot))
    else:
        return np.where(x < x_knot, m1 * x + c1, y_knot)


def bilinear_piecewise_model(x, x_knot, m1, c1, m2):
    """Initial linear drop that becomes a different linear slope after x_knot."""
    y_knot = m1 * x_knot + c1
    if isinstance(x, float) or isinstance(x, int):
        return float(np.where(x < x_knot, m1 * x + c1, m2 * (x - x_knot) + y_knot))
    else:
        return np.where(x < x_knot, m1 * x + c1, m2 * (x - x_knot) + y_knot)


def calculate_fit_metrics(y_data, y_pred, num_params):
    n = len(y_data)

    # Residual Sum of Squares
    rss = np.sum((y_data - y_pred) ** 2)

    # Total Sum of Squares
    tss = np.sum((y_data - np.mean(y_data)) ** 2)

    # R-squared
    r2 = 1 - (rss / tss)

    # Adjusted R-squared
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - num_params - 1))

    # AIC (Akaike Information Criterion)
    # Constant terms are omitted as we only care about relative differences
    aic = 2 * num_params + n * np.log(rss / n)

    metric_tags = ["RSS", "R²", "R²_adj", "AIC"]
    metrics = pd.Series(dict(zip(metric_tags, (rss, r2, r2_adj, aic))))
    return metrics


# ---------------------------------------------------------------------------
# Soft-storey (column-frame) backbone
# ---------------------------------------------------------------------------
def _Fy_col(Vb_coeff, Wt):
    x_knot = 0.0853
    m = -12.4245
    c = 1.3411
    return linear_constant_model(Vb_coeff, x_knot, m, c) * Vb_coeff * Wt


def _E0_col(Vb_coeff, Wt, hi):
    m = -0.0161
    c = 0.0300
    theta_y = linear_model(Vb_coeff, m, c)
    fy = _Fy_col(Vb_coeff, Wt)
    return fy / theta_y / hi


def _b_col(Vb_coeff):
    a = -1.0444
    b = -5.5025
    c = -0.0023
    return exponential_shifted(Vb_coeff, a, b, c)


def get_approximate_ss_backbone_params(Vb_coeff, Wt, hi):
    Fy = _Fy_col(Vb_coeff, Wt)
    E0 = _E0_col(Vb_coeff, Wt, hi)
    b = _b_col(Vb_coeff)
    return Fy, E0, b


def get_approximate_ss_backbone(Vb_coeff, Wt, hi):
    """returns the points of backbone curve for the soft storey system calculated
    using the fitted predictions equations for Fy, E0, b and mu_ult."""

    Fy, E0, b = get_approximate_ss_backbone_params(Vb_coeff, Wt, hi)

    dy = Fy / E0
    du = dy + Fy / (abs(b) * E0)

    p1 = [0, 0]
    p2 = [dy, Fy]
    p3 = [du, 0]

    return np.column_stack([p1, p2, p3]).T


# ---------------------------------------------------------------------------
# Braced-frame backbone
# ---------------------------------------------------------------------------
def _Fy_brace(Vb_coeff, Wt):
    x_knot = 0.0674
    m1 = -41.2913
    c1 = 5.2864
    m2 = -1.5976
    return bilinear_piecewise_model(Vb_coeff, x_knot, m1, c1, m2) * Vb_coeff * Wt


def _E0_brace(Vb_coeff, Wt, hi, brace_angle):
    a = 1810.2138
    b = -12.5761
    c = 508.4157
    E0_normalised = exponential_shifted(Vb_coeff, a, b, c)
    normalisation_factor = Vb_coeff * Wt * np.sin(brace_angle) * np.cos(brace_angle) ** 2 / hi
    return E0_normalised * normalisation_factor


def _b_brace(Vb_coeff):
    x_knot = 0.1148
    m1 = -1.0342
    c1 = 0.0256
    m2 = -0.0277
    return bilinear_piecewise_model(Vb_coeff, x_knot, m1, c1, m2)


def _mu_fr_brace(Vb_coeff):
    x_knot = 0.0852
    m1 = -69.2558
    c1 = 7.9101
    return linear_constant_model(Vb_coeff, x_knot, m1, c1)


def _mu_r_brace(Vb_coeff):
    x_knot = 0.0788
    m1 = -100.6322
    c1 = 11.4245
    return linear_constant_model(Vb_coeff, x_knot, m1, c1)


def br_backbone_params(Vb_coeff, Wt, hi, brace_angle):
    Fy = float(np.round(_Fy_brace(Vb_coeff, Wt), 3))
    E0 = float(np.round(_E0_brace(Vb_coeff, Wt, hi, brace_angle), 3))
    b = np.round(_b_brace(Vb_coeff), 3)
    mu_fr = np.round(_mu_fr_brace(Vb_coeff), 3)
    mu_r = np.round(_mu_r_brace(Vb_coeff), 3)
    return Fy, E0, b, mu_fr, mu_r


def get_approximate_br_backbone(Vb_coeff, Wt, hi, aspect):
    """returns the points of backbone curve for the soft storey system calculated
    using the fitted predictions equations for Fy, E0, b, mu_fr and mu_r.
    aspect = aspect ratio of the braced_bay (height / width)"""

    brace_angle = np.arctan(aspect)

    Fy, E0, b, mu_fr, mu_r = br_backbone_params(Vb_coeff, Wt, hi, brace_angle)

    dy = Fy / E0
    dfr = mu_fr * dy
    dr = mu_r * dy

    p1 = [0, 0]
    p2 = [dy, Fy]
    p3 = [dfr, Fy + b * E0 * (dfr - dy)]
    p4 = [dr, 0]

    return np.column_stack([p1, p2, p3, p4]).T


# ---------------------------------------------------------------------------
# Hysteretic parameters
# ---------------------------------------------------------------------------
def _pinchX(Vb_coeff):
    x_knot = 0.1698
    m1 = -5.0467
    c1 = 1.0325
    m2 = 0.3478
    return bilinear_piecewise_model(Vb_coeff, x_knot, m1, c1, m2)


def _pinchY(Vb_coeff):
    m = 1.3196
    c = 0.1735
    return linear_model(Vb_coeff, m, c)


def _damage1(Vb_coeff):
    m = 0.3332
    c = 0.0354
    return linear_model(Vb_coeff, m, c)


def get_approximate_hysteretic_hyst_params(Vb_coeff):
    """returns pinchX, pinchY, damage1, damage2, beta"""
    pinchX = _pinchX(Vb_coeff)
    pinchY = _pinchY(Vb_coeff)
    d1 = _damage1(Vb_coeff)
    d2 = 0
    beta = 0.337

    return [pinchX, pinchY, d1, d2, beta]


def get_approximate_steel02_hyst_params():
    """returns a1, a2, R0, cR1, cR2 (mean optimised values across 3-storey CBFs)"""
    return list(STEEL02_HYST_PARAMS_MEAN)


# ---------------------------------------------------------------------------
# Combined backbone
# ---------------------------------------------------------------------------
def total_backbone(Vb_coeff, Wt, hi, aspect):
    br_bb = get_approximate_br_backbone(Vb_coeff, Wt, hi, aspect)
    ss_bb = get_approximate_ss_backbone(Vb_coeff, Wt, hi)

    all_x = np.unique(np.concatenate([br_bb[:, 0], ss_bb[:, 0]]))
    all_x.sort()
    br_y = np.interp(all_x, br_bb[:, 0], br_bb[:, 1])
    ss_y = np.interp(all_x, ss_bb[:, 0], ss_bb[:, 1])
    total = br_y + ss_y

    return np.column_stack([all_x, total])


# ---------------------------------------------------------------------------
# Assembly into OpenSees material-parameter lists
# ---------------------------------------------------------------------------
def assemble_sdof_material_params(Vb_coeff, Wt, hi, aspect, mass, gamma=1.0):
    """Assemble the OpenSees material parameter lists for the parameterised SDOF.

    Mirrors the construction in wp1pt4pt1e (cells 36, 56, 83). ``gamma`` is the
    modal participation factor used to convert from MDOF "full system"
    coordinates to equivalent-SDOF coordinates; pass ``gamma=1.0`` (default)
    when designing an SDOF directly (no MDOF conversion).

    Returns a dict with keys matching ``template_model_cbf_sdof.py``:
    ``steel02_bb_params`` ([Fy, E0, b, du]),
    ``hysteretic_bb_params`` (12 backbone coordinates: positive s1,e1..s3,e3
    then their negatives),
    ``steel02_hyst_params`` ([a1, a2, R0, cR1, cR2]),
    ``hysteretic_hyst_params`` ([pinchX, pinchY, damage1, damage2, beta]),
    ``mass``.
    """
    # soft-storey (Steel02 spring) backbone -> [Fy, E0, b, du] in SDOF coords
    ss_params = get_approximate_ss_backbone_params(Vb_coeff, Wt, hi)
    ss_bb = get_approximate_ss_backbone(Vb_coeff, Wt, hi)
    du = (ss_bb[2, 0] - ss_bb[1, 0]) / gamma
    steel02_bb_params = [
        ss_params[0] / gamma,   # Fy
        ss_params[1],           # E0
        ss_params[2],           # b
        du,
    ]

    # braced-frame (Hysteretic spring) backbone -> 12 (force, disp) coords
    br_bb = get_approximate_br_backbone(Vb_coeff, Wt, hi, aspect)
    hysteretic_bb_params = (
        np.concatenate([br_bb[1:, [1, 0]], -br_bb[1:, [1, 0]]]) / gamma
    ).flatten().tolist()

    return {
        "steel02_bb_params": steel02_bb_params,
        "hysteretic_bb_params": hysteretic_bb_params,
        "steel02_hyst_params": get_approximate_steel02_hyst_params(),
        "hysteretic_hyst_params": get_approximate_hysteretic_hyst_params(Vb_coeff),
        "mass": mass,
    }
