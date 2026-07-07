"""Reusable, parallelisable per-site MDOF case-study design.

This is a refactor of the loop body in ``01_design_case_study_structures.py``
into a top-level function (``design_site_mdof``) so it can be pickled by a
``ProcessPoolExecutor`` worker (required on Windows ``spawn``) and reused by the
``wp1pt4pt1f`` notebook.

The design *algorithm* is unchanged: this is a thin wrapper around the existing
``standes`` flow (``Model`` -> ``set_*`` -> ``design_ULS_GQWIE`` -> ``to_json`` ->
pickle -> beam-splice/hinge post-processing). The only intentional input change
vs. the original script is that the seismic site category is set explicitly
(default ``"A"``) instead of relying on the template default (``"C"``).
"""

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from standes.config import GRAVITY
from standes.model import Model, to_json

from phd_project.scripts.case_study_design_scripts.cbf_design_ULS_GQWEI import (
    design_ULS_GQWIE,
)

# Default in-repo design templates (live next to this module).
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_FILEPATH = _THIS_DIR / "template_design_parameters.json"
DEFAULT_SPLICE_TEMPLATE_FILEPATH = _THIS_DIR / "template_beam_splices.json"


def _gusset_end_i(beam: dict) -> bool:
    if max(beam["gusset_offset_i"], beam["column_offset_i"]) == beam["gusset_offset_i"]:
        return True
    return False


def _gusset_end_j(beam: dict) -> bool:
    if max(beam["gusset_offset_j"], beam["column_offset_j"]) == beam["gusset_offset_j"]:
        return True
    return False


def design_site_mdof(
    S_alpha_RP: float,
    building_tag: str,
    out_root,
    *,
    site_category: str = "A",
    ductility_class: int = 2,
    storey_height: int = 3500,
    n_storeys: int = 3,
    g_floor: float = 0.0075,
    g_roof: float = 0.0075,
    q_floor: float = 0.003,
    q_roof: float = 0,
    q_floor_type: str = "QB",
    q_roof_type: str = "QH",
    template_filepath=DEFAULT_TEMPLATE_FILEPATH,
    splice_template_filepath=DEFAULT_SPLICE_TEMPLATE_FILEPATH,
    max_iters: int = 15,
) -> dict:
    """Design a single site-specific MDOF CBF and persist its design files.

    Mirrors the per-structure body of ``01_design_case_study_structures.py``.
    Writes ``{building_tag}/{building_tag}_in.json``, ``..._out.json`` and
    ``..._out.pickle`` under ``out_root`` and returns a summary dict with the
    design base-shear coefficient.

    Returns a dict with keys: ``tag``, ``success``, ``S_alpha_RP``,
    ``design_in_json``, ``design_out_json``, ``pickle``, and (on success)
    ``Vb``, ``Wt``, ``Vb_coeff``, ``T``, ``q_design``, ``S_r``. On failure
    (no suitable section) ``success`` is ``False`` and an ``error`` key is set.
    """
    out_root = Path(out_root)
    template_filepath = Path(template_filepath)
    splice_template_filepath = Path(splice_template_filepath)

    # per-storey load vectors (no live load on the roof)
    g_floors = list(g_floor * np.ones(n_storeys))
    g_floors[-1] = g_roof
    q_floors = list(q_floor * np.ones(n_storeys))
    q_floors[-1] = q_roof
    q_floor_types = [q_floor_type for _ in range(n_storeys)]
    q_floor_types[-1] = q_roof_type

    building_folder = out_root / building_tag
    building_folder.mkdir(parents=True, exist_ok=True)

    in_name_root = f"{building_tag}_in"
    out_name_root = f"{building_tag}_out"

    # build the site-specific design input json
    with open(template_filepath, "r") as file:
        design_in_dict = json.load(file)

    design_inputs = design_in_dict["design_inputs"]
    design_inputs["geometry"]["heights"] = int(n_storeys) * [storey_height]
    eq = design_inputs["load_parameters"]["eq_parameters"]
    eq["S_alpha_RP"] = S_alpha_RP
    eq["ductility_class"] = ductility_class
    eq["site_category"] = site_category  # Soil Class A (template default is "C")
    design_inputs["load_parameters"]["g_floors"] = g_floors
    design_inputs["load_parameters"]["q_floors"] = q_floors
    design_inputs["load_parameters"]["q_floor_types"] = q_floor_types

    design_in_filepath = building_folder / f"{in_name_root}.json"
    with open(design_in_filepath, "w") as fid:
        json.dump(design_in_dict, fid, indent=4)

    result = {
        "tag": building_tag,
        "S_alpha_RP": S_alpha_RP,
        "design_in_json": str(design_in_filepath),
        "design_out_json": None,
        "pickle": None,
        "success": False,
    }

    # design the structure (algorithm unchanged from the original script)
    model = Model(design_in_filepath)
    model.set_ductility_class(ductility_class)
    model.set_behaviour_factors()
    model.set_allowable_column_sections(["HEB", "HEM"])
    model.set_allowable_beam_sections(["IPE", "HEA"])
    model.set_allowable_brace_sections(["CHS"])

    try:
        model, checks, scs = design_ULS_GQWIE(model, max_iters=max_iters)
    except ValueError as e:
        if "No suitable section available" in str(e):
            result["error"] = "no suitable sections"
            return result
        raise

    # output json + pickle
    design_out_filepath = building_folder / f"{out_name_root}.json"
    to_json(model, design_out_filepath)

    model_save_filepath = building_folder / f"{out_name_root}.pickle"
    with open(model_save_filepath, "wb") as file:
        pickle.dump(model, file)

    # add beam-splice / beam-column-connection data to the output json
    with open(design_out_filepath, "r") as file:
        json_data = json.load(file)

    beams = json_data["structure"]["beams"]
    with open(splice_template_filepath) as file:
        splice_data = json.load(file)

    key = "splice"
    for beam in beams:
        beam.pop("hinge_i", None)
        beam.pop("hinge_j", None)

        if _gusset_end_i(beam):  # beam splice
            beam["column_connection_i"] = None
            beam["beam_splice_i"] = splice_data[key]
        else:  # beam-column connection
            beam["column_connection_i"] = splice_data[key]
            beam["beam_splice_i"] = None

        if _gusset_end_j(beam):  # beam splice
            beam["column_connection_j"] = None
            beam["beam_splice_j"] = splice_data[key]
        else:  # beam-column connection
            beam["column_connection_j"] = splice_data[key]
            beam["beam_splice_j"] = None

    with open(design_out_filepath, "w") as file:
        json.dump(json_data, file, indent=4)

    # design base-shear coefficient from the seismic design outputs
    sdo = model.seismic_design_outputs
    Vb = sdo["design_baseshear"]
    seismic_mass = sdo["seismic_mass"]
    Wt = seismic_mass * GRAVITY
    result.update(
        {
            "success": bool(scs),
            "design_out_json": str(design_out_filepath),
            "pickle": str(model_save_filepath),
            "Vb": Vb,
            "Wt": Wt,
            "Vb_coeff": Vb / Wt,
            "T": sdo["design_period"],
            "q_design": sdo.get("q_design"),
            "S_r": sdo.get("design_spectral_acceleration"),
        }
    )
    return result


def default_n_workers() -> int:
    """``max(cpu_count - 3, 1)`` — same convention as the rest of the repo."""
    ideal = max((os.cpu_count() or 1) - 3, 1)
    max_allowed = 61 - 3 # (windows spawn limit)
    return min(ideal, max_allowed)


def design_sites_parallel(jobs, *, n_workers=None, **design_kwargs):
    """Design many sites in parallel.

    ``jobs`` is an iterable of ``(S_alpha_RP, building_tag, out_root)`` tuples.
    ``design_kwargs`` are forwarded to every ``design_site_mdof`` call. A
    ``ProcessPoolExecutor`` is used (not threads) because OpenSeesPy keeps
    interpreter-global state. Progress is reported with ``tqdm``. Returns a list
    of result dicts (order not guaranteed).
    """
    from tqdm.auto import tqdm

    if n_workers is None:
        n_workers = default_n_workers()

    jobs = list(jobs)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                design_site_mdof, S_alpha_RP, tag, out_root, **design_kwargs
            ): tag
            for (S_alpha_RP, tag, out_root) in jobs
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Designing MDOFs"
        ):
            results.append(future.result())
    return results
