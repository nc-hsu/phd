"""Tests for find_rscript: how run_r_code decides which Rscript to call.

No real R is needed: the configured path is a dummy file in a temporary folder, and the PATH
lookup (shutil.which) is replaced by monkeypatching, so the results do not depend on whether
R is installed on the machine running the tests.
"""

from pathlib import Path

import pytest

import phd_project.scripts.run_r_code as rrc


def _which_returning(value):
    """A stand-in for shutil.which that always returns ``value`` (and records its calls)."""
    calls = []

    def fake_which(name):
        calls.append(name)
        return value
    fake_which.calls = calls
    return fake_which


def test_configured_path_used_and_path_not_consulted(tmp_path, monkeypatch):
    exe = tmp_path / "Rscript.exe"
    exe.write_text("")  # any existing file counts; it is never run here
    fake = _which_returning("/should/not/be/used")
    monkeypatch.setattr(rrc.shutil, "which", fake)
    assert rrc.find_rscript({"executables": {"rscript": exe}}) == exe
    assert fake.calls == []


def test_missing_configured_path_raises_even_if_on_path(tmp_path, monkeypatch):
    missing = tmp_path / "no_such" / "Rscript.exe"
    monkeypatch.setattr(rrc.shutil, "which", _which_returning("/usr/bin/Rscript"))
    with pytest.raises(FileNotFoundError, match="no_such"):
        rrc.find_rscript({"executables": {"rscript": missing}})


def test_no_key_uses_path(monkeypatch):
    monkeypatch.setattr(rrc.shutil, "which", _which_returning("/usr/bin/Rscript"))
    assert rrc.find_rscript({}) == Path("/usr/bin/Rscript")


def test_no_key_and_not_on_path_raises(monkeypatch):
    monkeypatch.setattr(rrc.shutil, "which", _which_returning(None))
    with pytest.raises(FileNotFoundError, match="config.local.yaml"):
        rrc.find_rscript({})


# ---------------------------------------------------------------------------------------
# run_r_script: failures raise, warnings are shown. subprocess.run and find_rscript are
# replaced, so these tests check the Python side only and need no R installation.
# ---------------------------------------------------------------------------------------
class _FakeCompleted:
    """The fields of subprocess.CompletedProcess that run_r_script reads."""

    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def r_script(tmp_path, monkeypatch):
    """An existing dummy .R file, with find_rscript stubbed to a dummy executable."""
    fp = tmp_path / "script.R"
    fp.write_text("# dummy")
    monkeypatch.setattr(rrc, "find_rscript", lambda cfg: Path("Rscript"))
    return fp


def test_failure_raises_with_r_error_output(r_script, monkeypatch):
    monkeypatch.setattr(rrc.subprocess, "run",
                        lambda *a, **k: _FakeCompleted(1, stderr="Error in lmer: boom"))
    with pytest.raises(rrc.RScriptError, match="boom") as info:
        rrc.run_r_script(r_script)
    assert info.value.returncode == 1


def test_missing_script_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no_such.R"):
        rrc.run_r_script(tmp_path / "no_such.R")


def test_success_prints_stderr_warnings(r_script, monkeypatch, capsys):
    monkeypatch.setattr(rrc.subprocess, "run",
                        lambda *a, **k: _FakeCompleted(0, stdout="done", stderr="boundary (singular) fit"))
    rrc.run_r_script(r_script)
    out = capsys.readouterr().out
    assert "boundary (singular) fit" in out and "completed successfully" in out


def test_str_path_and_params_passed_through(r_script, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return _FakeCompleted(0)
    monkeypatch.setattr(rrc.subprocess, "run", fake_run)
    rrc.run_r_script(str(r_script), ["a", 3])
    assert seen["command"] == ["Rscript", str(r_script.resolve()), "a", "3"]
