"""Tests for the cyclic-pushover reduction used by nb 056.

The quantities under test all end up as regression covariates in nb 072, where
a silent factor-of-two would be invisible. So what is pinned here is the
arithmetic of the FEMA P695 equations, the NaN convention that keeps a truncated
analysis from inventing a ductility, and the level-counting of the completeness
check.
"""

import numpy as np
import pytest

from phd_project.scripts import cpo_postprocessing as cpp


# ---------------------------------------------------------------------------
# FEMA P695 Eq. 6-8: the modal coefficient C0
# ---------------------------------------------------------------------------


def test_fema_C0_hand_computed_three_level_case():
    # Equal masses and a straight-line mode shape, so C0 can be done by hand:
    #   SUM(m phi)  = 100 * (1 + 2 + 3)       = 600
    #   SUM(m phi^2)= 100 * (1 + 4 + 9)       = 1400
    #   C0 = phi_r * 600 / 1400 = 3 * 3/7     = 9/7
    m = np.array([100.0, 100.0, 100.0])
    phi = np.array([1.0, 2.0, 3.0])
    assert cpp.fema_C0(m, phi) == pytest.approx(9.0 / 7.0)


def test_fema_C0_is_invariant_to_mode_shape_scaling():
    # C0 is a ratio of terms quadratic in phi, so whatever normalisation
    # OpenSees happened to use for the eigenvector must not matter. This is why
    # first_mode_translational() passes the ordinates through unnormalised.
    m = np.array([107889.9, 107889.9, 96330.3])
    phi = np.array([-0.03219762, -0.05774588, -0.0767581])
    base = cpp.fema_C0(m, phi)
    for scale in (-3.7, 0.001, 1.0, 250.0):
        assert cpp.fema_C0(m, scale * phi) == pytest.approx(base)


def test_fema_C0_of_a_rigid_translation_is_one():
    # A mode shape that is flat over the height is pure rigid-body translation:
    # the SDOF and the roof move together, so C0 must be exactly 1.
    m = np.array([50.0, 80.0, 30.0])
    assert cpp.fema_C0(m, np.ones(3)) == pytest.approx(1.0)


def test_fema_C0_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cpp.fema_C0(np.ones(3), np.ones(4))


# ---------------------------------------------------------------------------
# FEMA P695 Eq. 6-7: the effective yield displacement
# ---------------------------------------------------------------------------


def test_delta_y_eff_matches_a_hand_evaluation():
    # group_3s_00: C0 = 1.26294, V_max = 441503 N, W = 3.0618e6 N, T1 = 0.649888 s
    #   delta_y_eff = 1.26294 * 0.144194 * (9810 / (4 pi^2)) * 0.649888^2
    got = cpp.delta_y_eff_p695(1.2629430, 441503.0, 3061800.0, 0.6498881)
    assert got == pytest.approx(19.11, abs=0.01)


def test_delta_y_eff_scales_with_the_square_of_the_period():
    # Eq. 6-7 is quadratic in T. This is the guard on the max(T, T1) choice:
    # a 10 % longer period must move delta_y_eff by 21 %, not by 10 %.
    a = cpp.delta_y_eff_p695(1.3, 4.0e5, 3.0e6, 1.0)
    b = cpp.delta_y_eff_p695(1.3, 4.0e5, 3.0e6, 1.1)
    assert b / a == pytest.approx(1.21)


def test_delta_y_eff_is_in_millimetres():
    # A unit slip between m and mm is the failure mode this whole module is
    # arranged to prevent, so pin the order of magnitude explicitly: a realistic
    # frame must yield at tens of mm, not tens of m and not tens of microns.
    got = cpp.delta_y_eff_p695(1.3, 4.4e5, 3.1e6, 0.65)
    assert 1.0 < got < 100.0


# ---------------------------------------------------------------------------
# delta_u and the NaN convention
# ---------------------------------------------------------------------------


def _backbone(d_y, f_y, d_f, f_f, d_r, f_r, d_end, f_end):
    return np.array([[0.0, 0.0], [d_y, f_y], [d_f, f_f], [d_r, f_r], [d_end, f_end]])


def test_delta_u_interpolates_within_the_crossing_segment():
    # Peak 1000 at d = 20, falling linearly to 0 at d = 30. The 0.8 * 1000 = 800
    # crossing is a fifth of the way down that segment, i.e. at d = 22.
    bb = _backbone(10.0, 900.0, 20.0, 1000.0, 30.0, 0.0, 40.0, 0.0)
    assert cpp.displacement_at_strength_loss(bb) == pytest.approx(22.0)


def test_delta_u_is_nan_when_the_curve_never_drops_that_far():
    # A truncated analysis: still rising when it stopped. Returning the last
    # point here would invent a ductility set by where the run happened to halt.
    bb = _backbone(10.0, 500.0, 20.0, 800.0, 30.0, 950.0, 40.0, 1000.0)
    assert np.isnan(cpp.displacement_at_strength_loss(bb))


def test_delta_u_honours_an_explicit_reference_force():
    # The reason the parameter exists: measured against its own inflated peak
    # (1000) the target is 800 and the crossing is early, but against the
    # envelope's true peak (500) the target is 400 and the crossing is later.
    bb = _backbone(10.0, 1000.0, 20.0, 700.0, 30.0, 300.0, 40.0, 300.0)
    own = cpp.displacement_at_strength_loss(bb)
    ref = cpp.displacement_at_strength_loss(bb, v_max=500.0)
    assert own == pytest.approx(10.0 + 10.0 * 200.0 / 300.0)   # 16.67
    assert ref == pytest.approx(27.5)
    assert ref > own


def test_delta_u_ignores_a_pre_peak_crossing():
    # The elastic branch passes through 0.8 * f_max on its way up. delta_u is
    # defined on the descending branch only, so the search starts at the peak.
    bb = _backbone(10.0, 1000.0, 20.0, 1000.0, 30.0, 0.0, 40.0, 0.0)
    assert cpp.displacement_at_strength_loss(bb) == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# Backbone metrics
# ---------------------------------------------------------------------------


def test_backbone_metrics_reads_the_documented_rows():
    bb = _backbone(10.0, 900.0, 25.0, 1000.0, 26.0, 200.0, 40.0, 150.0)
    m = cpp.backbone_metrics(bb)
    assert m["d_y_mm"] == pytest.approx(10.0)
    assert m["f_y_N"] == pytest.approx(900.0)
    assert m["d_f_mm"] == pytest.approx(25.0)      # start of the fracture drop
    assert m["f_max_bb_N"] == pytest.approx(1000.0)
    assert m["k_e_N_per_mm"] == pytest.approx(90.0)
    assert m["k_r_N_per_mm"] == pytest.approx((150.0 - 200.0) / (40.0 - 26.0))


def test_backbone_f_max_falls_back_to_the_yield_point_when_post_buckling_softens():
    # With a softening post-buckling branch the largest force on the backbone is
    # f_y, not f_joint. Indexing a fixed row would understate it.
    bb = _backbone(10.0, 1000.0, 25.0, 700.0, 26.0, 200.0, 40.0, 150.0)
    assert cpp.backbone_metrics(bb)["f_max_bb_N"] == pytest.approx(1000.0)


def test_backbone_overshoot_flag():
    # The idealised elastic line can intersect above any force in the data.
    assert cpp.backbone_overshoots_envelope(734091.0, 441503.0) is True
    assert cpp.backbone_overshoots_envelope(402914.0, 448191.0) is False
    # A small overshoot is normal and within tolerance.
    assert cpp.backbone_overshoots_envelope(101.0, 100.0) is False


def test_backbone_metrics_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\(5, 2\)"):
        cpp.backbone_metrics(np.zeros((4, 2)))


# ---------------------------------------------------------------------------
# Completeness of the FEMA 461 trace
# ---------------------------------------------------------------------------


def _protocol(u_max=630.0, n_levels=12):
    # The ladder nb 055 writes: 12 amplitudes, each 1.4x the previous, every one
    # run as (+, -, +, -), then a final return to zero.
    c = 1.0 / (1.4 ** (n_levels - 1))
    amps = [1.4**i * c * u_max for i in range(n_levels)]
    return [s * a for a in amps for s in (1, -1, 1, -1)] + [0.0]


def test_completeness_accepts_a_finished_run():
    targets = _protocol()
    # Reached +/- U_max and came back to zero.
    d = np.array([0.0, 630.0, -630.0, 630.0, -630.0, 0.05])
    got = cpp.cpo_completeness(d, targets, du=0.2)
    assert got["cpo_complete"] is True
    assert got["max_level_reached"] == 12
    assert got["frac_of_Umax"] == pytest.approx(1.0)


def test_completeness_flags_a_trace_that_stopped_early():
    # group_3s_00: stopped at +/-164 mm of a 630 mm protocol, mid-cycle.
    targets = _protocol()
    d = np.array([0.0, 163.994, -163.994, 156.806])
    got = cpp.cpo_completeness(d, targets, du=0.2)
    assert got["cpo_complete"] is False
    assert got["reached_max_amplitude"] is False
    assert got["ends_at_zero"] is False
    assert got["max_level_reached"] == 8          # 164 mm is the 8th rung
    assert got["frac_of_Umax"] == pytest.approx(163.994 / 630.0)


def test_completeness_separates_the_two_failure_modes():
    # Died during the final return to zero: full amplitude reached, but the
    # trace does not end at zero. Both conditions are reported separately so
    # this case is distinguishable from a run that stopped early.
    targets = _protocol()
    d = np.array([0.0, 630.0, -630.0, 400.0])
    got = cpp.cpo_completeness(d, targets, du=0.2)
    assert got["reached_max_amplitude"] is True
    assert got["ends_at_zero"] is False
    assert got["cpo_complete"] is False


def test_completeness_counts_levels_from_the_amplitude_ladder():
    targets = _protocol()
    # Just past the 5th rung (59.76 mm) but short of the 6th (83.67 mm).
    d = np.array([0.0, 60.0, -60.0, 0.0])
    assert cpp.cpo_completeness(d, targets, du=0.2)["max_level_reached"] == 5


# ---------------------------------------------------------------------------
# Node tagging
# ---------------------------------------------------------------------------


def test_floor_node_tags_match_the_model_convention():
    # Verified against node_info.json of the 3-storey test models; the base
    # (101010100) is restrained and so excluded.
    assert cpp.floor_node_tags(4) == [101010200, 101010300, 101010400]
    assert cpp.floor_node_tags(6) == [
        101010200, 101010300, 101010400, 101010500, 101010600,
    ]
