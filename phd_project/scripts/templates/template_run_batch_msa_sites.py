import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


## TOP-LEVEL multi-site MSA launcher.
##
## Launches one MSA coordinator per site, each in its own (minimised) console
## window that acts as that site's live dashboard. The number of coordinators
## running at once is capped by `max_coordinators` -- this is a SEPARATE, local
## limit, deliberately NOT the worker semaphore: the coordinators are sleep-bound
## bookkeeping processes and must not consume NLTHA worker slots, or they would
## starve (and deadlock) the very workers they are waiting on.
##
## Each coordinator (run_msa_site.py -> run_batch_msa_per_stripe_record.py) draws
## its NLTHA workers from the shared, machine-global process_semaphore, so across
## ALL active sites at most `physical_cores - 3` NLTHA runs are live at once. As a
## site finishes its 6 x ~30 analyses, its coordinator retires and the launcher
## starts the next site in its place.

# --- configure here -------------------------------------------------------
# parent folder whose immediate subfolders are the per-site analysis folders
# (each containing config_msa.py and run_msa_site.py)
sites_root = Path(r"E:/msa_sites")
# which sites to run. Leave empty to run EVERY site found under sites_root;
# otherwise list the sites to run -- each entry is either a full path to a site
# folder or a bare folder name resolved under sites_root. Sites run in the order
# listed.
site_names: list[str] = []
max_coordinators = 10          # how many site coordinators / windows at once
window_name_index = 0         # the number of folder to go up in directory to get to unique folder name. e.g. 0 = the site folder itself, 1 = its parent
site_script_name = "run_msa_site.py"
site_config_name = "config_msa.py"
# --------------------------------------------------------------------------


def _is_site_folder(folder: Path) -> bool:
    return (folder.is_dir()
            and (folder / site_config_name).exists()
            and (folder / site_script_name).exists())


def discover_sites(root: Path) -> list[Path]:
    """Immediate subfolders of ``root`` that hold a site config + run script."""
    return sorted(folder for folder in root.iterdir() if _is_site_folder(folder))


def resolve_sites(root: Path, names: list[str]) -> list[Path]:
    """Resolve the sites to run.

    If ``names`` is empty, discover every site under ``root``. Otherwise resolve
    each entry, in order, as either a full path (used as-is when it exists) or a
    bare folder name under ``root``, and validate it is a proper site folder.
    """
    if not names:
        return discover_sites(root)

    sites: list[Path] = []
    invalid: list[str] = []
    for name in names:
        candidate = Path(name)
        folder = candidate if candidate.is_dir() else root / name
        if _is_site_folder(folder):
            sites.append(folder)
        else:
            invalid.append(name)

    if invalid:
        raise FileNotFoundError(
            "The following site_names did not resolve to a folder containing "
            f"{site_config_name} + {site_script_name}: {invalid}")

    return sites


def launch_site(site_folder: Path, window_name_index: int):
    """Launch a site's MSA coordinator in its own minimised console window."""
    SW_SHOWMINNOACTIVE = 7
    venv_python = Path(sys.executable)

    script_path = site_folder / site_script_name
    config_path = site_folder / site_config_name
    window_title = config_path.parents[window_name_index].name

    py_code = (
        f"import importlib.util; "
        f"path = r'''{script_path}'''; "
        f"spec = importlib.util.spec_from_file_location('mod', path); "
        f"mod = importlib.util.module_from_spec(spec); "
        f"spec.loader.exec_module(mod); "
        f"mod.run(r'''{config_path}''')"
    )

    ps_command = (
        f"$host.UI.RawUI.WindowTitle = '{window_title}'; "
        f"& '{venv_python}' -c \"{py_code}\"; "
        f"if ($LASTEXITCODE -ne 0) {{ pause }}"
    )

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_SHOWMINNOACTIVE

    proc = subprocess.Popen(
        ["pwsh", "-NoLogo", "-NoProfile", "-Command", ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startupinfo,
    )
    return proc, window_title


def main(max_coordinators: int = max_coordinators,
         window_name_index: int = window_name_index):
    sites = resolve_sites(sites_root, site_names)
    if not sites:
        raise FileNotFoundError(
            f"No site folders (with {site_config_name} + {site_script_name}) "
            f"found under {sites_root}")

    print(f"Launch Time: {datetime.now()} | {len(sites)} site(s), "
          f"up to {max_coordinators} coordinator(s) at once")

    running = []  # {proc, title, start_time}
    pbar = tqdm(total=len(sites), desc="Sites complete", unit="site")

    def reap():
        still_running = []
        for entry in running:
            if entry["proc"].poll() is None:
                still_running.append(entry)
            else:
                elapsed = datetime.now() - entry["start_time"]
                pbar.write(f"finished site: {entry['title']} "
                           f"in {elapsed.total_seconds():.0f}s")
                pbar.update(1)
        running[:] = still_running

    for site_folder in sites:
        # local coordinator cap (NOT the worker semaphore)
        while len(running) >= max_coordinators:
            reap()
            time.sleep(0.5)

        proc, title = launch_site(site_folder, window_name_index)
        running.append({"proc": proc, "title": title,
                        "start_time": datetime.now()})
        pbar.write(f"launched site: {title} PID={proc.pid}")

    while running:
        reap()
        time.sleep(0.5)

    pbar.close()
    print(f"All sites complete: {datetime.now()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MSA for several sites at once, sharing one machine-global core cap.")
    parser.add_argument("--max-coordinators", type=int, default=max_coordinators,
                        help="how many site coordinators (windows) to run at once")
    parser.add_argument("--window-name-index", type=int, default=window_name_index,
                            help="the index of the folder in config.parents to be used as coordinator window title. 0 corresponds to the site folder itself")
    args = parser.parse_args()

    main(max_coordinators=args.max_coordinators,
         window_name_index=args.window_name_index)
