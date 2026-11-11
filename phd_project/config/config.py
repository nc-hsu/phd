"""Project configuration: load config.yaml, turn its paths into absolute Paths, and
resolve record sets (SS / FX / study) into folders and run-folder names.

Two kinds of section live in config.yaml:

* PATH sections (raw_data, proc_data, results, ...): every string value is a path relative
  to the project root (or already absolute) and is returned as an absolute ``Path``.
* ``record_sets``: plain names, NOT paths. It lists the site-specific (SS) and fixed (FX)
  record sets and the studies that pair them, plus ``active_study``. It is returned as
  loaded, and should be read through the helpers below rather than by hand.
* ``study_outputs``: plain names, NOT paths. Named sub-folders of a study folder (e.g.
  ``model_plots``), resolved for one study by :func:`study_output_dir`.

Record-set folder layout (see the comment block in config.yaml): at each stage that
depends on record sets, the stage folder has an ``ss/``, ``fx/`` and/or ``studies/`` level:

    ss_dir(cfg, "proc_data", "gm_selection", "AvgSA_03__ks")
        -> <root>/data_processed/06_gm_selection/ss/AvgSA_03__ks
    fx_dir(cfg, "proc_data", "structure_fragility_curves", "femap695")
        -> <root>/data_processed/09_structure_fragility_curves/fx/femap695
    study_dir(cfg, "proc_data", "fragility_statistical_analysis", "AvgSA_03__ks__vs__femap695")
        -> <root>/data_processed/10_fragility_curve_statistical_analysis/studies/AvgSA_03__ks__vs__femap695

The helpers check the id against the registry, so a typo fails loudly instead of quietly
creating a new, empty folder.

Machine-specific settings (the Rscript path, data on other drives or network shares) do not
live in config.yaml, which only describes the repo itself. They go in an optional,
unpublished ``config.local.yaml`` next to it, which :func:`load_config` merges over
config.yaml key by key. ``config.local.example.yaml`` is the template for it.
"""

import yaml
from pathlib import Path

# Sections whose values are names, not paths: left exactly as loaded.
NON_PATH_SECTIONS = frozenset({"record_sets", "study_outputs"})

# Name of the optional per-machine override file, looked for in the same folder as the main
# config file. It is git-ignored, so it never reaches the repo or the public export.
LOCAL_CONFIG_NAME = "config.local.yaml"


def merge_config(base: dict, override: dict) -> dict:
    """Return ``base`` with ``override`` merged over it, recursively. Neither input is changed.

    The rule, applied at every level:

    * a key whose value is a dict in BOTH inputs is merged again, key by key, so an override
      holding only ``{"executables": {"rscript": ...}}`` changes that one key and keeps every
      other key of ``executables``;
    * any other value in ``override`` replaces the one in ``base`` (including a dict
      replacing a non-dict and vice versa);
    * keys only in ``override`` are added; keys only in ``base`` are kept.

    So an override can change or add settings but never delete one.
    """
    merged = dict(base)  # shallow copy: base's own top-level dict is not modified
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # Both sides are sections: recurse, which also copies the nested dict, so the
            # nested dicts of base are never modified either.
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_local_config(local_path: Path) -> dict:
    """Read the local override file; an empty file means "no overrides".

    ``yaml.safe_load`` returns None for an empty (or comments-only) file, which is a valid
    way to have a local file that overrides nothing. Anything other than a mapping at the
    top level cannot be merged section by section, so it is rejected with the file named.
    """
    with open(local_path, "r") as f:
        local = yaml.safe_load(f)
    if local is None:
        return {}
    if not isinstance(local, dict):
        raise ValueError(f"{local_path} must contain a mapping of sections at the top level, "
                         f"got {type(local).__name__}")
    return local


def load_config(path=None, cfg_file: str | None = None, use_local: bool = True):
    """Load the project config and return it with every path made absolute.

    Steps:

    1. Read the main config file: ``path`` if given, else ``cfg_file`` in this folder, else
       ``config.yaml`` in this folder.
    2. If ``use_local`` is True and a ``config.local.yaml`` exists in the SAME folder as the
       main file, merge it over the main config with :func:`merge_config` (nested sections
       key by key; other values replaced; new keys added). Without that file the result is
       exactly the main file's. Pass ``use_local=False`` to ignore it, so tests and the
       public-repo export do not depend on whatever local file sits on the machine.
    3. Resolve ``project_root`` (the local file may set it, and then wins) relative to the
       main file's folder.
    4. In every section except ``NON_PATH_SECTIONS``, turn each string value into an
       absolute ``Path``: relative ones against the project root, absolute ones kept. The
       merge happens before this step, so local values are treated exactly like main ones.
    """
    if path is None and cfg_file is None:
        path = Path(__file__).parent / "config.yaml"
    elif path is None and not cfg_file is None:
        path = Path(__file__).parent / cfg_file
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Merge the per-machine overrides, if any, BEFORE the path conversion below, so a path
    # in the local file is resolved by exactly the same rules as one in the main file.
    local_path = path.parent / LOCAL_CONFIG_NAME
    if use_local and local_path.is_file():
        cfg = merge_config(cfg, _read_local_config(local_path))

    # Convert all paths to absolute Path objects
    root = (path.parent / cfg["project_root"]).resolve()

    def fix(p):
        path_obj = Path(p)
        # if the path is already absolut (i.e. like C:/...) don't prefix with root
        if path_obj.is_absolute():
            return path_obj.resolve()
        return (root / p).expanduser().resolve()

    for section_name, section in cfg.items():
        # record_sets holds ids and descriptions; turning them into Paths would break them.
        if section_name in NON_PATH_SECTIONS:
            continue
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, str):
                    section[k] = fix(v)

    return cfg


# ---------------------------------------------------------------------------------------
# Record sets
# ---------------------------------------------------------------------------------------
def _registry(cfg: dict) -> dict:
    """The record_sets section, with a clear error if config.yaml predates it."""
    try:
        return cfg["record_sets"]
    except KeyError:
        raise KeyError("config.yaml has no 'record_sets' section") from None


def _check(kind: str, set_id: str, known: dict) -> str:
    """Return ``set_id`` if it is registered under ``kind``, else raise listing the options.

    Every folder helper goes through here: an unregistered id would otherwise build a
    perfectly valid path to a folder that does not exist (or worse, get created by mkdir).
    """
    if set_id not in known:
        raise KeyError(f"unknown {kind} id {set_id!r}; registered: {sorted(known)}")
    return set_id


def ss_info(cfg: dict, ss_id: str) -> dict:
    """Registry entry of an SS set: {"im", "scheme", "description"}."""
    reg = _registry(cfg)["ss"]
    return reg[_check("SS set", ss_id, reg)]


def fx_info(cfg: dict, fx_id: str) -> dict:
    """Registry entry of an FX set:
    {"records", "records_analysis_machine", "record_glob", "label", "description"}.

    The id must be letters and digits only. It is spliced into the arm tokens of the
    processed file names (``msa_{fx}``, ``ida_{fx}_ub``, see mra.fx_arm_names) and into
    the ``{ss}__vs__{fx}`` study names, and both are parsed back by splitting on "_".
    """
    reg = _registry(cfg)["fx"]
    _check("FX set", fx_id, reg)
    if not fx_id.isalnum():
        raise ValueError(f"FX id {fx_id!r} must be letters and digits only (no '_')")
    return reg[fx_id]


def study(cfg: dict, name: str | None = None) -> dict:
    """Resolve a study (default: ``active_study``) to {"name", "ss", "fx", "im"}.

    ``im`` is the SS set's IM: a study compares the arms at one IM, and the SS records were
    selected for it, so it is the study's IM too.
    """
    reg = _registry(cfg)
    name = reg["active_study"] if name is None else name
    entry = reg["studies"][_check("study", name, reg["studies"])]
    ss_id, fx_id = entry["ss"], entry["fx"]
    fx_info(cfg, fx_id)   # both halves must be registered too
    return {"name": name, "ss": ss_id, "fx": fx_id, "im": ss_info(cfg, ss_id)["im"]}


def ss_dir(cfg: dict, section: str, key: str, ss_id: str) -> Path:
    """``cfg[section][key] / "ss" / ss_id`` for a registered SS set."""
    ss_info(cfg, ss_id)
    return Path(cfg[section][key]) / "ss" / ss_id


def fx_dir(cfg: dict, section: str, key: str, fx_id: str) -> Path:
    """``cfg[section][key] / "fx" / fx_id`` for a registered FX set."""
    fx_info(cfg, fx_id)
    return Path(cfg[section][key]) / "fx" / fx_id


def study_dir(cfg: dict, section: str, key: str, study_name: str) -> Path:
    """``cfg[section][key] / "studies" / study_name`` for a registered study."""
    study(cfg, study_name)
    return Path(cfg[section][key]) / "studies" / study_name


def study_output_dir(cfg: dict, name: str, study_name: str | None = None) -> Path:
    """Folder of the per-study output ``name`` registered under ``study_outputs``.

    Resolves to ``study_dir(cfg, section, key, study) / subdir`` for ``study_name``
    (default: ``active_study``), e.g. "model_plots" ->
        <root>/data_processed/10_fragility_curve_statistical_analysis/studies/
        AvgSA_03__ks__vs__femap695/model_plots
    Both the output name and the study are checked against their registries, so a typo
    raises instead of building a path to a folder nobody reads. The folder is not created
    here: the caller decides when to mkdir.
    """
    try:
        outputs = cfg["study_outputs"]
    except KeyError:
        raise KeyError("config.yaml has no 'study_outputs' section") from None
    entry = outputs[_check("study output", name, outputs)]
    st = study(cfg, study_name)
    return study_dir(cfg, entry["section"], entry["key"], st["name"]) / entry["subdir"]


def bootstrap_roots(cfg: dict, study_name: str | None = None) -> dict:
    """The three folders the bootstrap / dataset functions need for one study (default:
    ``active_study``), as {"ss": ..., "fx": ..., "study": ...}:

        ss     10_.../ss/{ss_id}/bootstrapping     the SS arm (site_msa_*) replicates
        fx     10_.../fx/{fx_id}/bootstrapping     the FX arms (msa_{fx}*, ida_{fx}*) replicates
        study  10_.../studies/{study}              the nb 072 dataset and everything built on it
        ss_id, fx_id                               the two record-set ids (strings)

    Pass the whole dict wherever a function asks for the bootstrapping ``root``:
    ``mra.arm_root`` picks the SS or FX folder per arm, and ``fm.dataset_csv_path`` the
    study folder. The SS and FX bootstraps are kept apart because they are resampled from
    different record sets, so one SS set's replicates can be paired with any FX set's.
    """
    st = study(cfg, study_name)
    key = "fragility_statistical_analysis"
    return {"ss": ss_dir(cfg, "proc_data", key, st["ss"]) / "bootstrapping",
            "fx": fx_dir(cfg, "proc_data", key, st["fx"]) / "bootstrapping",
            "study": study_dir(cfg, "proc_data", key, st["name"]),
            # The ids travel with the folders: the FX id names the FX arms' files
            # (msa_{fx}_bootstrap_..., see mra.fx_arm_names), so fm reads it from here.
            "ss_id": st["ss"], "fx_id": st["fx"]}


def stripe_grid_path(cfg: dict, ss_id: str) -> Path:
    """The stripe-grid JSON (which (site, IML) stripes are wanted) an SS set is selected on.

    ``record_sets.ss.{ss_id}.stripe_grid`` is either "live" - the main-line grid
    proc_data.AvgSA_03_imls_for_selection, which nb 017 keeps extending - or the id of the
    set whose 06_gm_selection folder holds the frozen copy (imls_for_selection_{im}.json).
    Pass the result as ``imls_for_selection_fp`` to the setup_AvgSA03_gm_selection functions.
    """
    info = ss_info(cfg, ss_id)
    grid = info.get("stripe_grid", "live")
    if grid == "live":
        return Path(cfg["proc_data"][f"{info['im']}_imls_for_selection"])
    return ss_dir(cfg, "proc_data", "gm_selection", grid) / f"imls_for_selection_{info['im']}.json"


def stripe_lineage(cfg: dict, ss_id: str) -> list[str]:
    """``ss_id`` and every set it builds its stripe grid on, base first.

    Follows ``record_sets.ss.*.stripe_grid_base``. nb 017 merges the remediation IMLs of
    every set in the lineage into the live grid, and nb 050 runs the reserve stripes nb 060
    flagged for any of them, so a live set keeps the stripes it was started from.
    """
    chain, cur = [], ss_id
    while cur is not None:
        if cur in chain:
            raise ValueError(f"stripe_grid_base forms a loop: {chain + [cur]}")
        chain.append(cur)
        cur = ss_info(cfg, cur).get("stripe_grid_base")
    return chain[::-1]


def fx_records_dir(cfg: dict, fx_id: str) -> Path:
    """In-repo folder of an FX set's record JSONs (data_raw/fixed_record_sets/{records})."""
    return Path(cfg["raw_data"]["fixed_record_sets"]) / fx_info(cfg, fx_id)["records"]


def fx_record_files(cfg: dict, fx_id: str, records_dir: Path | str | None = None) -> list[Path]:
    """An FX set's record JSONs in TAG order: the sorted ``record_glob`` listing.

    ``records_dir`` defaults to the in-repo copy (fx_records_dir); pass the analysis-machine
    folder to list the files the runs read. Position i in this list is record tag ``str(i)``
    - the convention every IDA / MSA run and the bootstrap use (scripts/fixed_record_sets.py).
    """
    folder = fx_records_dir(cfg, fx_id) if records_dir is None else Path(records_dir)
    return sorted(folder.glob(fx_info(cfg, fx_id)["record_glob"]))


# ---------------------------------------------------------------------------------------
# Run-folder names on the analysis drive (inside each structure's mdof/ folder).
# One place builds them, so notebooks 050 / 052 / 054 / 060 / 061 and the rename script
# cannot disagree about a name.
# ---------------------------------------------------------------------------------------
def ss_msa_names(cfg: dict, ss_id: str) -> dict:
    """MSA-SS names for one SS set, e.g. for "AvgSA_03__ks":
    {"results": "msa_AvgSA_03__ks", "stripes": "AvgSA_03__ks_stripes",
     "config": "config_msa_AvgSA_03__ks.py"}
    The ss_id already carries the IM, so the IM is not repeated."""
    ss_info(cfg, ss_id)
    return {"results": f"msa_{ss_id}",
            "stripes": f"{ss_id}_stripes",
            "config": f"config_msa_{ss_id}.py"}


def fx_msa_names(cfg: dict, fx_id: str, im: str) -> dict:
    """MSA-FX names for one FX set at one IM, e.g. ("femap695", "AvgSA_03"):
    {"results": "msa_AvgSA_03__femap695", "stripes": "AvgSA_03__femap695_stripes",
     "config": "config_msa_AvgSA_03__femap695.py"}"""
    fx_info(cfg, fx_id)
    return {"results": f"msa_{im}__{fx_id}",
            "stripes": f"{im}__{fx_id}_stripes",
            "config": f"config_msa_{im}__{fx_id}.py"}


def fx_ida_names(cfg: dict, fx_id: str) -> dict:
    """IDA-FX names for one FX set, e.g. "femap695":
    {"results": "ida__femap695", "config": "config_ida_htf__femap695.py"}
    IDA is not tied to an IM (fragilities are converted afterwards), so no IM in the name."""
    fx_info(cfg, fx_id)
    return {"results": f"ida__{fx_id}",
            "config": f"config_ida_htf__{fx_id}.py"}


def setup_project(cfg: dict):
    """
    Explicitly creates the directory structure.
    """
    print("Initializing project directories...")
    for section_name, section in cfg.items():
        if section_name in NON_PATH_SECTIONS:
            continue
        if isinstance(section, dict):
            for path in section.values():
                if isinstance(path, Path):
                    # If it's a file path, create the parent directory
                    # If it's a directory path, create it directly
                    if path.suffix:  # Simple check if it looks like a file
                        path.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        path.mkdir(parents=True, exist_ok=True)
