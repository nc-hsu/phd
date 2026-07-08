"""Helpers for reading and editing case-study design JSON files.

Single source of truth for the small design-file utilities that were previously
duplicated across the ``wp1pt4pt1a/b/f/h`` notebooks: deriving the roof control node,
the number of primary/damping modes, and removing the soft-storey braces to build a
mechanism (``_ss``) variant of a frame.
"""

import json
from pathlib import Path

from standes.utils import generate_type_1_tag


def load_design_file(path: str | Path) -> dict:
    """Load a design JSON file into a dict."""
    with open(path) as f:
        return json.load(f)


def get_control_node_from_design_file(path: str | Path) -> int:
    """Roof control node = top-left corner node tag (over ``n_levels``)."""
    d = load_design_file(path)
    n_levels = len(d["structure"]["level_coordinates"])
    return generate_type_1_tag(1, 1, 1, n_levels, 0, 0)


def get_n_primary_modes_from_design_file(path: str | Path) -> int:
    """Number of primary horizontal modes = number of storeys (levels - 1)."""
    d = load_design_file(path)
    return len(d["structure"]["level_coordinates"]) - 1


def get_n_damping_modes_from_design_file(path: str | Path) -> int:
    """Damp all horizontal modes (n_primary * primary grid nodes per level)."""
    d = load_design_file(path)
    return get_n_primary_modes_from_design_file(path) * len(d["structure"]["grid_coordinates"])


def remove_soft_storey_braces(src_json: str | Path, dst_json: str | Path,
                              bays_levels_to_remove) -> None:
    """Write a copy of the design JSON with the given ``(bay, level)`` braces removed.

    Accepts an iterable of ``(bay, level)`` list/tuple pairs; each is compared as a tuple
    so ``[bay, level]`` and ``(bay, level)`` both work. (Was ``_remove_braces_and_gussets``.)
    """
    d = load_design_file(src_json)
    to_go = {tuple(bl) for bl in bays_levels_to_remove}
    d["structure"]["braces"] = [
        b for b in d["structure"]["braces"] if (b["bay"], b["level"]) not in to_go
    ]
    with open(dst_json, "w") as f:
        json.dump(d, f, indent=4)
