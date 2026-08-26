import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from phd_project.process_semaphore.process_semaphore import (
    set_max_concurrent, describe_max_concurrent, DEFAULT_RESERVED_CORES)


## TOP-LEVEL multi-building MSA launcher (the MSA analogue of
## run_batch_ida_buildings.py).
##
## Launches one MSA coordinator per building, each in its own (minimised) console
## window that acts as that building's live dashboard. The number of coordinators
## running at once is capped by `max_coordinators` -- this is a SEPARATE, local
## limit, deliberately NOT the worker semaphore: the coordinators are sleep-bound
## bookkeeping processes and must not consume NLTHA worker slots, or they would
## starve (and deadlock) the very workers they are waiting on.
##
## Each coordinator (run_batch_msa_per_record.py) dispatches every (stripe, record)
## pair at once, drawing its NLTHA workers from the shared, machine-global
## process_semaphore by default, so across ALL active buildings at most `--n-cores`
## NLTHA runs are live at once. As a building finishes, its coordinator retires and
## the launcher starts the next building in its place.
##
## --n-cores is REQUIRED on the semaphore path: it sets the machine-global cap for
## every analysis process on this box (not just this launcher's) and is sticky until
## the next --n-cores changes it, so each campaign has to state the budget it wants.
## Large models need a much lower cap than small ones -- one worker per core thrashes
## the L3 cache and saturates the memory bus, making the workstation unusable for
## everyone else.
##
## The semaphore can be swapped for a fixed per-building worker count (--max-workers
## / --no-semaphore), in which case --n-cores does not apply. Each building's
## per-worker console windows show by default -- pass --quiet to hide them and stream
## worker output to worker_logs/ instead.

# --- configure here -------------------------------------------------------
# The buildings to run. Each entry is a folder containing run_batch_msa_per_record.py
# plus the named config. `config` may be a bare filename (resolved inside the
# folder) or an absolute path.
buildings: list[dict[str, str]] = [
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_00\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_00\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_01\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_01\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_02\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_02\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_03\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_03\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_04\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_04\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_05\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_05\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_06\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_06\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_07\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_07\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_08\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_08\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_09\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_09\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_10\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_10\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_11\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_11\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_12\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_12\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_13\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_13\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_14\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_14\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_15\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_15\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_16\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_16\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_17\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_17\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_18\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_18\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_19\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_19\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_20\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_20\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_21\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_21\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_22\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_22\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_23\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_23\3s\mdof\config_msa_AvgSA_03_femap695.py"},
    {"folder": r"D:\08_wp1_fixed_record_sets\group_3s_24\3s\mdof",
     "config": r"D:\08_wp1_fixed_record_sets\group_3s_24\3s\mdof\config_msa_AvgSA_03_femap695.py"},
]
max_coordinators = 10          # how many building coordinators / windows at once
window_name_index = 2         # folders up from the config for the window title. For site_{ii}/{n}s/mdof/config -> 2 = the site_{ii} folder
runner_script_name = "run_batch_msa_per_record.py"
# Show a console window per record worker (default). Set to False (or pass
# --quiet) to hide them and stream each worker's output to worker_logs/ instead --
# useful when many workers run at once and the windows would be noise.
show_worker_windows = True
# Draw workers from the shared machine-global semaphore (default). Set to False
# (or pass --no-semaphore / --max-workers) to run each building with a fixed worker
# count instead.
use_semaphore = True
max_workers = None            # fixed worker count per building when use_semaphore is False
# Machine-global cap on simultaneous NLTHA workers (the process_semaphore cap, NOT
# the per-building max_workers). None -> must be supplied via --n-cores.
n_cores = None
# --------------------------------------------------------------------------


def resolve_buildings(specs: list[dict[str, str]]) -> list[tuple[Path, Path]]:
    """Resolve each building spec to (runner_script, config_path), validating both exist."""
    resolved: list[tuple[Path, Path]] = []
    invalid: list[str] = []
    for spec in specs:
        folder = Path(spec["folder"])
        runner = folder / runner_script_name
        config = Path(spec["config"])
        if not config.is_absolute():
            config = folder / config
        if runner.is_file() and config.is_file():
            resolved.append((runner, config))
        else:
            missing = [str(p) for p in (runner, config) if not p.is_file()]
            invalid.append(f"{folder.name}: missing {missing}")
    if invalid:
        raise FileNotFoundError("Could not resolve these buildings:\n  " + "\n  ".join(invalid))
    return resolved


def launch_building(runner: Path, config: Path, window_name_index: int,
                    show_worker_windows: bool = show_worker_windows,
                    use_semaphore: bool = use_semaphore,
                    max_workers: int | None = max_workers):
    """Launch a building's MSA coordinator in its own minimised console window.

    The coordinator's workers draw from the shared semaphore unless use_semaphore
    is False, in which case each building runs with its own max_workers cap; per-worker
    windows show unless show_worker_windows is False (then they stream to worker_logs/).
    """
    SW_SHOWMINNOACTIVE = 7
    venv_python = Path(sys.executable)
    window_title = config.parents[window_name_index].name

    py_code = (
        f"import importlib.util; "
        f"path = r'''{runner}'''; "
        f"spec = importlib.util.spec_from_file_location('mod', path); "
        f"mod = importlib.util.module_from_spec(spec); "
        f"spec.loader.exec_module(mod); "
        f"mod.run(r'''{config}''', use_semaphore={use_semaphore}, "
        f"max_workers={max_workers}, show_worker_windows={show_worker_windows})"
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


def run(building_specs: list[dict[str, str]] | None = None,
        max_coordinators: int = max_coordinators,
        window_name_index: int = window_name_index,
        show_worker_windows: bool = show_worker_windows,
        use_semaphore: bool = use_semaphore,
        max_workers: int | None = max_workers,
        n_cores: int | None = n_cores):

    specs = building_specs if building_specs is not None else buildings
    resolved = resolve_buildings(specs)

    # Set the machine-global worker cap before anything is launched. The spawned
    # coordinators read it from the semaphore's config file rather than being
    # passed it, so deployed run_batch_msa_per_record.py copies need no update.
    if use_semaphore and n_cores is not None:
        set_max_concurrent(n_cores, set_by=Path(__file__).name)

    workers_desc = (f"shared semaphore, {describe_max_concurrent()}" if use_semaphore
                    else f"{max_workers if max_workers is not None else f'cores - {DEFAULT_RESERVED_CORES}'} workers/building")
    print(f"Launch Time: {datetime.now()} | {len(resolved)} building(s), "
          f"up to {max_coordinators} coordinator(s) at once, "
          f"workers: {workers_desc}, "
          f"worker windows: {'on' if show_worker_windows else 'off (logging to worker_logs/)'}")

    running = []  # {proc, title, start_time}
    pbar = tqdm(total=len(resolved), desc="Buildings complete", unit="building")

    def reap():
        still_running = []
        for entry in running:
            if entry["proc"].poll() is None:
                still_running.append(entry)
            else:
                elapsed = datetime.now() - entry["start_time"]
                pbar.write(f"finished building: {entry['title']} "
                           f"in {elapsed.total_seconds():.0f}s")
                pbar.update(1)
        running[:] = still_running

    for runner, config in resolved:
        # local coordinator cap (NOT the worker semaphore)
        while len(running) >= max_coordinators:
            reap()
            time.sleep(0.5)

        proc, title = launch_building(runner, config, window_name_index,
                                      show_worker_windows=show_worker_windows,
                                      use_semaphore=use_semaphore,
                                      max_workers=max_workers)
        running.append({"proc": proc, "title": title, "start_time": datetime.now()})
        pbar.write(f"launched building: {title} PID={proc.pid}")

    while running:
        reap()
        time.sleep(0.5)

    pbar.close()
    print(f"All buildings complete: {datetime.now()}")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_batch_msa_buildings.py --n-cores 20
    #   python run_batch_msa_buildings.py --n-cores 60 --quiet --max-coordinators 5
    #   python run_batch_msa_buildings.py --max-workers 4        (no semaphore, no --n-cores)
    # By default the buildings list configured at the top of this file is run.
    parser = argparse.ArgumentParser(
        description="Run MSAs for several buildings at once, sharing one machine-global core cap.")
    parser.add_argument("--n-cores", type=int, default=n_cores,
                        help="machine-global cap on simultaneous NLTHA workers across ALL "
                             f"launchers on this machine (default policy: physical cores - "
                             f"{DEFAULT_RESERVED_CORES}). Sticky until the next --n-cores. "
                             "Required unless --max-workers/--no-semaphore is used. Use a low "
                             "value (~20) for large models, which otherwise thrash the cache "
                             "and make the workstation unusable for others.")
    parser.add_argument("--max-coordinators", type=int, default=max_coordinators,
                        help="how many building coordinators (windows) to run at once")
    parser.add_argument("--window-name-index", type=int, default=window_name_index,
                        help="the index of the folder in config.parents to be used as the coordinator window title")
    parser.add_argument("--quiet", action="store_true",
                        help="hide per-record worker windows; stream them to worker_logs/ instead")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="run each building with this fixed worker count instead of the "
                             "shared semaphore (implies --no-semaphore)")
    parser.add_argument("--no-semaphore", action="store_true",
                        help="do not use the machine-global process_semaphore (each building "
                             "runs with a fixed worker count from --max-workers, or cores - 4)")
    args = parser.parse_args()

    use_semaphore = not args.no_semaphore and args.max_workers is None
    if use_semaphore and args.n_cores is None:
        parser.error("--n-cores is required when using the shared semaphore; pass "
                     "--max-workers N to run each building with a fixed worker count instead")

    run(max_coordinators=args.max_coordinators,
        window_name_index=args.window_name_index,
        show_worker_windows=not args.quiet,
        use_semaphore=use_semaphore,
        max_workers=args.max_workers,
        n_cores=args.n_cores)
