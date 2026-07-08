"""Automatic soft-storey mechanism identification for MDOF steel CBFs.

Given the recorder output of a (cyclic) pushover analysis, identify which
storey(s) form the collapse mechanism and which braces should be removed to
build the "soft-storey" (``_ss``) variant of the frame.

Reproduces the manual reasoning previously done by eye (notebook ``wp1pt4pt1d``):
  * **drift concentration** locates the mechanism -- the soft storey's drift runs
    away while the other storeys plateau (this is the *primary* signal), and
  * **brace fracture** confirms it -- braces in the soft storey lose their axial
    capacity (low force at high displacement).

The output ``braces_to_remove`` (list of ``[bay, level]``) can be fed directly to
the ``_remove_braces_and_gussets`` helper used in ``wp1pt4pt1a``.

Data source: ``<folder>/cyclic_pushover/recorders.pickle`` (+ ``{tag}_designfile.json``).
Reuses ``standes.analysis.recorders.get_recorders`` and the tag decoders
``standes.utils.get_x_idx`` / ``get_z_idx``.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from standes.analysis.recorders import get_recorders
from standes.utils import get_x_idx, get_z_idx

BRACE_PREFIX = "4"  # component id for brace elements (config.BR_ID)


def load_recorders(folder, analysis: str = "cyclic_pushover"):
    """Load the pickled recorder list from ``<folder>/<analysis>/recorders.pickle``."""
    folder = Path(folder)
    with open(folder / analysis / "recorders.pickle", "rb") as f:
        return pickle.load(f)


def load_design(folder, design_file=None) -> dict:
    """Load the design json (``{tag}_designfile.json``, fallback ``{tag}_out.json``)."""
    folder = Path(folder)
    if design_file is None:
        cands = sorted(folder.glob("*_designfile.json")) or sorted(folder.glob("*_out.json"))
        if not cands:
            raise FileNotFoundError(f"no design json (*_designfile.json / *_out.json) in {folder}")
        design_file = cands[0]
    with open(design_file) as f:
        return json.load(f)


def _analysis_length(recorders) -> int:
    """Number of analysis steps, taken from the (never-removed) drift recorders."""
    drifts = get_recorders(recorders, "drift")
    return max((len(r.record[1]) for r in drifts.values()), default=0)


def storey_peak_drifts(recorders, dof: int = 1) -> dict:
    """Peak absolute storey drift per storey, from the grid-line-1 drift recorders.

    Returns ``{storey:int -> peak_abs_drift:float}`` sorted by storey (1-based, base
    = storey 1). The soft storey is the one whose drift runs away relative to the rest.
    """
    out = {}
    for tag, r in get_recorders(recorders, "drift").items():
        if get_x_idx(tag) != 1:  # keep only the reference grid line (drift_line_1)
            continue
        storey = get_z_idx(tag) - 1  # drift recorder tag = top node (level z) of storey z-1
        out[storey] = max(abs(r.min[dof]), abs(r.max[dof]))
    return dict(sorted(out.items()))


def brace_force_envelopes(recorders, braces) -> dict:
    """Per-brace axial-force magnitude history.

    Returns ``{(bay, level) -> |P| history (np.ndarray)}``. For each design brace,
    all of its ``section_force`` sub-element recorders are combined element-wise as
    ``max(|P|)`` (the representative axial force carried each step). An empty array
    means the brace has no surviving force record (e.g. removed very early).
    """
    by_bl: dict = {}
    for tag, r in get_recorders(recorders, "section_force").items():
        ele = r.ele_tag
        if str(ele)[0] != BRACE_PREFIX:  # braces only (skip columns/beams)
            continue
        bl = (get_x_idx(ele), get_z_idx(ele))
        by_bl.setdefault(bl, []).append(np.abs(np.asarray(r.record["P"], dtype=float)))

    out = {}
    for b in braces:
        bl = (b["bay"], b["level"])
        arrs = by_bl.get(bl)
        if not arrs:
            out[bl] = np.array([])
            continue
        m = min(len(a) for a in arrs)  # sub-elements share length; guard anyway
        out[bl] = np.vstack([a[:m] for a in arrs]).max(axis=0)
    return out


def brace_fracture_ratio(p_hist, full_len: int, tail_frac: float = 0.25,
                         removed_tol: float = 0.9) -> float:
    """Residual axial capacity of a brace at high displacement (0 = fully lost).

    ``residual = max|P| over the final ``tail_frac`` of the record  /  max|P| overall``.

    Taking the *max* over the late window (rather than the last value) makes this
    robust to the cyclic protocol returning to zero displacement at the very end. A
    brace physically removed mid-analysis (record shorter than ``removed_tol*full_len``)
    has lost its capacity by definition and returns 0.
    """
    p = np.asarray(p_hist, dtype=float)
    if p.size == 0:
        return 0.0
    if full_len and len(p) < removed_tol * full_len:
        return 0.0  # removed mid-analysis -> fractured
    peak = float(p.max())
    if peak <= 0.0:
        return 0.0
    n_tail = max(1, int(round(tail_frac * len(p))))
    return float(p[-n_tail:].max() / peak)


def identify_soft_storey_mechanism(
    folder,
    *,
    analysis: str = "cyclic_pushover",
    design_file=None,
    recorders=None,
    concentration_frac: float = 0.5,
    fracture_threshold: float = 0.30,
    brace_confirm_frac: float = 0.5,
    tail_frac: float = 0.25,
) -> dict:
    """Identify the soft-storey mechanism and the braces to remove for one frame.

    Decision rule (drift primary, brace confirms):
      1. Candidate soft storeys = storeys whose peak drift concentrates relative to the
         worst storey, i.e. ``peak_drift[s] >= concentration_frac * max(peak_drift)``.
         (Measured against the peak, not the median: when one storey dominates, the many
         small upper-storey drifts would drag a median-based threshold down and spuriously
         flag a barely-elevated neighbour.)
      2. Each storey is brace-*confirmed* if a fraction >= ``brace_confirm_frac`` of its
         braces have residual capacity < ``fracture_threshold``.
      3. ``soft_storeys`` = the drift candidates (drift primary); brace confirmation is
         computed and reported (`storey_brace_confirmed`) to validate the finding.
      4. ``braces_to_remove`` = every design brace at a soft storey.

    Returns a JSON-serialisable dict (see keys below). ``braces_to_remove`` is a list
    of ``[bay, level]`` pairs (feed to ``_remove_braces_and_gussets`` as tuples).
    """
    folder = Path(folder)
    design = load_design(folder, design_file)
    braces = design["structure"]["braces"]
    n_storeys = len(design["structure"]["level_coordinates"]) - 1

    if recorders is None:
        recorders = load_recorders(folder, analysis)

    peak_drift = storey_peak_drifts(recorders)
    full_len = _analysis_length(recorders)
    envelopes = brace_force_envelopes(recorders, braces)
    frac = {bl: brace_fracture_ratio(p, full_len, tail_frac) for bl, p in envelopes.items()}

    # 1. drift-primary candidate soft storeys (concentration relative to the worst storey)
    storeys = sorted(peak_drift)
    max_drift = max((peak_drift[s] for s in storeys), default=0.0)
    candidates = [s for s in storeys if max_drift > 0 and peak_drift[s] >= concentration_frac * max_drift]

    # 2. brace confirmation per storey
    storey_conf = {}
    for s in storeys:
        s_braces = [bl for bl in frac if bl[1] == s]
        if s_braces:
            n_frac = sum(frac[bl] < fracture_threshold for bl in s_braces)
            storey_conf[s] = (n_frac / len(s_braces)) >= brace_confirm_frac
        else:
            storey_conf[s] = False

    soft = sorted(candidates)
    soft_set = set(soft)
    braces_to_remove = sorted((b["bay"], b["level"]) for b in braces if b["level"] in soft_set)

    return {
        "tag": folder.name,
        "n_storeys": n_storeys,
        "storey_peak_drift": {int(s): float(peak_drift[s]) for s in storeys},
        "soft_storeys": [int(s) for s in soft],
        "storey_brace_confirmed": {int(s): bool(storey_conf[s]) for s in storeys},
        "brace_fracture_ratio": {f"{b}_{l}": float(v) for (b, l), v in sorted(frac.items())},
        "braces_to_remove": [[int(b), int(l)] for b, l in braces_to_remove],
        "diagnostics": {
            "concentration_frac": concentration_frac,
            "max_drift": float(max_drift),
            "fracture_threshold": fracture_threshold,
            "brace_confirm_frac": brace_confirm_frac,
            "tail_frac": tail_frac,
            "drift_candidates": [int(s) for s in candidates],
        },
    }
