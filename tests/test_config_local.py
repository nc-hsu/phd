"""Tests for the optional per-machine config file (config.local.yaml) read by load_config.

Every test builds its own small config.yaml (and, where needed, config.local.yaml) in a
pytest temporary folder, so the real project config files are never read or changed. The
temporary folder stands in for phd_project/config/; ``project_root: "."`` makes that same
folder the project root, so expected paths are easy to write down.
"""

import copy
from pathlib import Path

import pytest
import yaml

from phd_project.config.config import load_config, merge_config


# A minimal main config: one path section with a nested-looking pair of keys, one non-path
# section (record_sets) whose strings must stay strings, and a top-level project_root.
MAIN_CFG = {
    "project_root": ".",
    "raw_data": {"root": "data_raw", "esm_flatfile": "data_raw/esm.csv"},
    "record_sets": {"fx": {"femap695": {"records": "femap695", "label": "FEMA P695"}}},
}


def _write(folder: Path, name: str, content) -> Path:
    """Write ``content`` (a dict, dumped as YAML, or a raw string) to folder/name."""
    fp = folder / name
    fp.write_text(content if isinstance(content, str) else yaml.safe_dump(content))
    return fp


@pytest.fixture
def main_cfg_path(tmp_path):
    """A temporary folder holding only the main config.yaml; returns its path."""
    return _write(tmp_path, "config.yaml", MAIN_CFG)


# ---------------------------------------------------------------------------------------
# merge_config: the recursive merge on its own
# ---------------------------------------------------------------------------------------
def test_merge_nested_override_keeps_siblings():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = merge_config(base, {"a": {"x": 10}})
    assert out == {"a": {"x": 10, "y": 2}, "b": 3}


def test_merge_adds_new_keys_at_any_level():
    out = merge_config({"a": {"x": 1}}, {"a": {"z": 5}, "c": {"k": "v"}})
    assert out == {"a": {"x": 1, "z": 5}, "c": {"k": "v"}}


def test_merge_non_dict_replaces_dict_and_back():
    # A non-dict override replaces a dict outright, and a dict override replaces a scalar:
    # only dict-over-dict recurses.
    assert merge_config({"a": {"x": 1}}, {"a": "flat"}) == {"a": "flat"}
    assert merge_config({"a": "flat"}, {"a": {"x": 1}}) == {"a": {"x": 1}}


def test_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "deep": {"p": 1}}, "b": 2}
    override = {"a": {"deep": {"p": 9}}, "c": 3}
    base_before, override_before = copy.deepcopy(base), copy.deepcopy(override)
    merge_config(base, override)
    assert base == base_before
    assert override == override_before


# ---------------------------------------------------------------------------------------
# load_config with and without a local file
# ---------------------------------------------------------------------------------------
def test_no_local_file_same_as_main_alone(main_cfg_path):
    assert load_config(path=main_cfg_path) == load_config(path=main_cfg_path, use_local=False)


def test_empty_local_file_same_as_none(main_cfg_path):
    # yaml.safe_load gives None for an empty or comments-only file: no overrides.
    _write(main_cfg_path.parent, "config.local.yaml", "# nothing here\n")
    assert load_config(path=main_cfg_path) == load_config(path=main_cfg_path, use_local=False)


def test_local_file_not_a_mapping_raises(main_cfg_path):
    _write(main_cfg_path.parent, "config.local.yaml", "- just\n- a list\n")
    with pytest.raises(ValueError, match="config.local.yaml"):
        load_config(path=main_cfg_path)


def test_local_overrides_nested_value_and_resolves_relative_path(main_cfg_path):
    root = main_cfg_path.parent.resolve()
    _write(main_cfg_path.parent, "config.local.yaml",
           {"raw_data": {"esm_flatfile": "elsewhere/esm.csv"},
            "record_sets": {"fx": {"femap695": {"records_analysis_machine": "D:/p695"}}}})
    cfg = load_config(path=main_cfg_path)
    # The overridden key is the local value, resolved against project_root like any other.
    assert cfg["raw_data"]["esm_flatfile"] == root / "elsewhere" / "esm.csv"
    # Its sibling still comes from config.yaml.
    assert cfg["raw_data"]["root"] == root / "data_raw"
    # A key added inside a non-path section stays a plain string, and its siblings stay.
    fx = cfg["record_sets"]["fx"]["femap695"]
    assert fx["records_analysis_machine"] == "D:/p695"
    assert fx["records"] == "femap695" and fx["label"] == "FEMA P695"


def test_local_absolute_path_stays_absolute(main_cfg_path, tmp_path_factory):
    # An absolute folder outside the project root, written the way a user would in YAML.
    outside = tmp_path_factory.mktemp("other_drive").resolve()
    _write(main_cfg_path.parent, "config.local.yaml",
           {"executables": {"rscript": (outside / "Rscript.exe").as_posix()}})
    cfg = load_config(path=main_cfg_path)
    assert cfg["executables"]["rscript"] == outside / "Rscript.exe"


def test_local_project_root_wins(main_cfg_path):
    sub = main_cfg_path.parent / "sub"
    sub.mkdir()
    _write(main_cfg_path.parent, "config.local.yaml", {"project_root": "sub"})
    cfg = load_config(path=main_cfg_path)
    assert cfg["raw_data"]["root"] == sub.resolve() / "data_raw"


def test_use_local_false_ignores_local_file(main_cfg_path):
    _write(main_cfg_path.parent, "config.local.yaml",
           {"raw_data": {"root": "somewhere_else"}, "executables": {"rscript": "R"}})
    cfg = load_config(path=main_cfg_path, use_local=False)
    assert cfg["raw_data"]["root"] == main_cfg_path.parent.resolve() / "data_raw"
    assert "executables" not in cfg
