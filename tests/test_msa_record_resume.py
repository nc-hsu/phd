"""Tests for per-record MSA resume and per-record result deletion.

Covers
  * standes.analysis.msa.record_is_complete / record_log_mismatch -- a record log only
    counts as complete when it was written for the record (name) and scale factor (to
    3 dp) the stripe selection currently names for that slot;
  * the launcher template's job list -- a stale slot is re-dispatched, a current one is
    not, a missing one is;
  * phd_project.scripts.data_handling.delete_msa_record_results -- dry run deletes
    nothing, a real run deletes exactly the listed slots, a slot whose log names a
    different record is skipped.

Everything runs on a fake results tree under pytest's tmp_path; no analysis is run.
"""
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from standes.analysis.msa import record_is_complete, record_log_mismatch

from phd_project.scripts.data_handling import delete_msa_record_results as dmr


TEMPLATE = (Path(__file__).resolve().parents[1] / "phd_project" / "scripts" / "templates"
            / "template_run_batch_msa_per_record.py")

# the five per-record artefacts msa_one_record writes, minus the log
PICKLE_SUFFIXES = ["recorders.pickle", "collapse_recorders.pickle", "timearray.pickle",
                   "injection_function_returns.pickle"]


def write_record(stripe_dir: Path, k: int, record_name: str, scale_factor: float) -> None:
    """Fake the five files msa_one_record leaves for slot k."""
    stripe_dir.mkdir(parents=True, exist_ok=True)
    for suffix in PICKLE_SUFFIXES:
        (stripe_dir / f"record_{k}_{suffix}").write_bytes(b"x")
    with open(stripe_dir / f"record_{k}_log.json", "w") as f:
        json.dump({"record_name": record_name, "collapsed": False, "ls_collapse": False,
                   "edps": [], "scale_factor": scale_factor}, f)


def write_stripe_pickle(folder: Path, iml_tag: str, records: list[tuple[str, float]]) -> Path:
    """Fake a GCIM stripe selection pickle with the fields read_stripe_selection reads."""
    folder.mkdir(parents=True, exist_ok=True)
    recs = pd.DataFrame({("metadata", "filename"): [r[0] for r in records],
                         ("metadata", "alpha"): [r[1] for r in records]})
    fp = folder / f"site_0__stripe_iml_{iml_tag}__gm_selection.pickle"
    with open(fp, "wb") as f:
        pickle.dump({"recs": recs, "cond_imt": "AvgSA([0, 3])", "cond_iml": 0.31,
                     "rtp": 475.0, "poe": 0.1}, f)
    return fp


# ---------------------------------------------------------------------------
# record_is_complete
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, scale, expected", [
    ("A.json", 1.2345, True),           # exact match
    ("B.json", 1.2345, False),          # different record in the slot
    ("A.json", 1.2345 + 4e-4, True),    # agrees to 3 dp
    ("A.json", 1.2345 + 6e-4, False),   # does not
])
def test_record_is_complete_identity(tmp_path, name, scale, expected):
    write_record(tmp_path, 3, "A.json", 1.2345)
    assert record_is_complete(tmp_path, 3, name, scale) is expected


def test_record_is_complete_backwards_compatible(tmp_path):
    # without expectations only existence is checked, as before
    write_record(tmp_path, 3, "A.json", 1.0)
    assert record_is_complete(tmp_path, 3)
    assert not record_is_complete(tmp_path, 4)


def test_mismatch_reason_names_both_values():
    reason = record_log_mismatch({"record_name": "A.json", "scale_factor": 1.0},
                                 "B.json", 1.0)
    assert "A.json" in reason and "B.json" in reason
    # a log without the key cannot be shown to match
    assert record_log_mismatch({}, "A.json") is not None


# ---------------------------------------------------------------------------
# launcher template: which (stripe, record) pairs get dispatched
# ---------------------------------------------------------------------------

def test_launcher_redispatches_only_stale_and_missing(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("launcher", TEMPLATE)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    sel_dir, out_dir = tmp_path / "sel", tmp_path / "msa"
    write_stripe_pickle(sel_dir, "00pt310", [("A.json", 1.0), ("NEW.json", 2.0), ("C.json", 3.0)])
    stripe_dir = out_dir / "stripe_00pt310"
    write_record(stripe_dir, 0, "A.json", 1.0)      # current      -> skipped
    write_record(stripe_dir, 1, "OLD.json", 2.0)    # swapped out  -> stale, re-run
    #                                                 slot 2 absent -> run

    config = {"result_dst": out_dir, "gm_selection_src": sel_dir,
              "stripe_order_ascending": True, "max_n_records": None, "resume": True,
              "post_process": False}
    (tmp_path / "config.py").write_text("")          # run() only checks the path exists
    monkeypatch.setattr(launcher, "import_from_path", lambda _: SimpleNamespace(config=config))
    dispatched = []
    monkeypatch.setattr(launcher, "run_records_in_subprocesses",
                        lambda **kw: dispatched.extend(kw["record_tags"]))

    launcher.run(tmp_path / "config.py", use_semaphore=False)
    assert dispatched == ["00pt310:1", "00pt310:2"]


# ---------------------------------------------------------------------------
# delete_msa_record_results
# ---------------------------------------------------------------------------

@pytest.fixture
def results_tree(tmp_path):
    """site_0/3s stripe 00pt310 with slots 2, 4 and 29 run; a CSV listing 2, 4, 7."""
    root = tmp_path / "sites"
    stripe_dir = root / "site_0" / "3s" / "mdof" / "msa_AvgSA_03" / "stripe_00pt310"
    write_record(stripe_dir, 2, "R2.json", 1.0)
    write_record(stripe_dir, 4, "OTHER.json", 1.0)   # slot re-selected since the CSV
    write_record(stripe_dir, 29, "R29.json", 1.0)    # not listed: must survive
    csv = tmp_path / "slots.csv"
    pd.DataFrame({"site": [0, 0, 0], "structure": ["3s"] * 3,
                  "stripe_iml": ["00pt310"] * 3, "record_index": [2, 4, 7],
                  "record_tag": ["R2", "R4", "R7"]}).to_csv(csv, index=False)
    return root, stripe_dir, csv


def test_delete_dry_run_touches_nothing(results_tree):
    root, stripe_dir, csv = results_tree
    before = sorted(p.name for p in stripe_dir.iterdir())
    plans = dmr.run(csv, root, dry_run=True)
    assert [p.status for p in plans] == [dmr.DELETE, dmr.MISMATCH, dmr.NOTHING]
    assert len(plans[0].files) == 5
    assert sorted(p.name for p in stripe_dir.iterdir()) == before


def test_delete_removes_only_listed_matching_slots(results_tree):
    root, stripe_dir, csv = results_tree
    dmr.run(csv, root, dry_run=False)
    left = {p.name.split("_")[1] for p in stripe_dir.iterdir()}
    assert left == {"4", "29"}                       # slot 2 gone; 4 (mismatch) and 29 kept
    assert len(list(stripe_dir.glob("record_29_*"))) == 5


def test_delete_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dmr.run(tmp_path / "x.csv", tmp_path / "not_mounted", dry_run=True)
