"""Run R scripts from Python through Rscript.

Rscript is looked up in this order (see :func:`find_rscript`):

1. ``executables.rscript`` in the merged config, i.e. a full path set in the per-machine
   ``phd_project/config/config.local.yaml``. If it is set it must exist.
2. Otherwise ``Rscript`` on the PATH (the normal case on Linux and macOS).
3. Otherwise an error explaining both ways to make R findable.

Rscript is always started with the project root as its working folder (see
:func:`project_root`). R reads an ``.Rprofile`` from the folder it starts in, so in a repo
that is an renv project (the public repo ships ``.Rprofile`` and ``renv/activate.R``) this
activates the project library that ``renv::restore()`` filled with the versions pinned in
``renv.lock``. In a repo without an ``.Rprofile`` R starts exactly as it would anywhere
else and uses the user's normal library.
"""

import os
import shutil
import subprocess
from pathlib import Path

import phd_project.config.config as _config_module
from phd_project.config.config import load_config

# The merged project config (config.yaml plus config.local.yaml, if present), loaded once
# when the module is imported.
cfg = load_config()

# The folder that holds config.yaml. load_config resolves project_root against this folder,
# and project_root() below must use the same rule to arrive at the same folder.
_CONFIG_DIR = Path(_config_module.__file__).parent


def project_root(cfg: dict) -> Path:
    """Return the project root as an absolute path: the folder Rscript is started in.

    ``load_config`` turns every path in the config's sections into an absolute Path, but it
    leaves the top-level ``project_root`` value as written in the file (e.g. ``"../../"``,
    relative to the config folder). It is resolved here by the same rule load_config uses
    internally, so both agree on where the root is. An absolute value is used as it is.
    """
    root = Path(cfg["project_root"])
    if not root.is_absolute():
        root = _CONFIG_DIR / root
    return root.resolve()

# Shown in every "Rscript not found" error: the two ways a user can make R findable.
_HOW_TO_FIX = (
    "R must be installed. On Windows, R's installer does not put R's bin folder on the PATH, "
    "so either add that folder (e.g. C:/Program Files/R/R-4.5.2/bin) to the PATH, or set "
    "executables.rscript to the FULL path of Rscript.exe in "
    "phd_project/config/config.local.yaml (template: config.local.example.yaml). "
    "A bare name such as 'Rscript' does not work there: it is read as a path inside the repo. "
    "To use the PATH, leave executables.rscript out."
)


def find_rscript(cfg: dict) -> Path:
    """Return the Rscript executable to use, following the lookup order in the module docstring.

    ``cfg`` is passed in, rather than read from the module-level config, so the lookup can be
    tested with a plain dict.

    A configured path that does not exist raises instead of falling back to the PATH: the
    path was set on purpose, so a typo in it should be fixed, not silently bypassed.
    """
    configured = cfg.get("executables", {}).get("rscript")
    if configured is not None:
        # load_config has already turned the value into an absolute Path.
        configured = Path(configured)
        if not configured.is_file():
            raise FileNotFoundError(
                f"executables.rscript is set to {configured}, which does not exist. {_HOW_TO_FIX}")
        return configured

    # No configured path: use whatever "Rscript" the PATH finds (Rscript.exe on Windows).
    on_path = shutil.which("Rscript")
    if on_path is None:
        raise FileNotFoundError(f"Rscript was not found on the PATH. {_HOW_TO_FIX}")
    return Path(on_path)


class RScriptError(RuntimeError):
    """An R script exited with a non-zero status. Carries R's exit code and error output."""

    def __init__(self, script: Path, returncode: int, stderr: str):
        self.script, self.returncode, self.stderr = script, returncode, stderr
        super().__init__(f"R script '{script.name}' failed with exit code {returncode}.\n"
                         f"--- R error output ---\n{stderr}")


def run_r_script(script_path, additional_params: list | None = None) -> None:
    """Run an R script through Rscript, printing its output, and raise if it fails.

    Example (in a notebook)::

        run_r_script(cfg["scripts"]["residual_partitioning"] / "residual_partitioning_lmer.R",
                     [str(folder), "asc,sslab", ...])
        df = pd.read_csv(folder / "asc_..._partitioned_residuals_lmer.csv")

    Args:
        script_path (Path | str): the .R file to run.
        additional_params (list, optional): command-line arguments passed to the
            script, in order (R reads them with ``commandArgs(trailingOnly = TRUE)``).
            Path objects are made absolute; strings are passed unchanged, so a path given
            as a string must be absolute (R runs in the project root; see the module docstring).

    Raises:
        FileNotFoundError: the script does not exist, or no Rscript can be found
            (see :func:`find_rscript`).
        RScriptError: R exited with a non-zero status. Raising, rather than printing and
            returning, stops a notebook at the failing cell, so later cells cannot go on to
            read output files left over from an earlier run as if they were new.

    R's stderr is printed even when the script succeeds: R writes its warnings and messages
    there (e.g. lme4's singular-fit and convergence warnings), and they must be seen.
    """
    # A fresh list per call (a list default would be one object shared by every call).
    # Rscript runs in the project root, not in the caller's working folder, so a relative
    # path would point somewhere else inside R. Every path object (Path or any other
    # os.PathLike) is therefore resolved to an absolute path against the caller's working
    # folder, i.e. where the caller meant it, before it is turned into a string. Plain
    # strings are passed through unchanged: they can be non-path values (model tags,
    # file-name tails, numbers), so a path given as a string must already be absolute; the
    # notebook builds its residuals folder from the config, whose paths are absolute.
    params = [] if additional_params is None else [
        str(Path(p).resolve()) if isinstance(p, os.PathLike) else str(p)
        for p in additional_params]

    # Accept a str as well as a Path, and resolve it to an absolute path, so R finds the
    # script although Rscript starts in the project root rather than the caller's folder.
    script = Path(script_path).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"R script not found: {script}")

    # The configured Rscript if config.local.yaml sets one, else Rscript on the PATH; raises
    # FileNotFoundError with instructions if neither is available.
    r_exe = find_rscript(cfg)

    command = [str(r_exe), str(script)] + params
    print(f"Executing R script: {script.name}...")

    # cwd=project root: R looks for .Rprofile in the folder it starts in. In an renv
    # project that file activates the project library, so the packages pinned in renv.lock
    # are the ones the script loads; without an .Rprofile nothing changes.
    # check=False: the exit code is inspected below, so R's stderr can be put into the
    # exception. R on Windows (UCRT builds) writes UTF-8; undecodable bytes are replaced
    # rather than allowed to crash the call after R has already finished.
    result = subprocess.run(command, cwd=project_root(cfg), check=False, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")

    if result.stdout:
        print("--- R Standard Output ---")
        print(result.stdout)

    if result.returncode != 0:
        raise RScriptError(script, result.returncode, result.stderr)

    if result.stderr.strip():
        print("--- R messages and warnings ---")
        print(result.stderr)

    print(f"R script '{script.name}' completed successfully.")
