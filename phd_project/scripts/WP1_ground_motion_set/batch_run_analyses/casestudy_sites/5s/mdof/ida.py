import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


## TOP-LEVEL multi-building IDA launcher.
##
## Launches one IDA coordinator per building, each in its own (minimised) console
## window that acts as that building's live dashboard. The number of coordinators
## running at once is capped by `max_coordinators` -- this is a SEPARATE, local
## limit, deliberately NOT the worker semaphore: the coordinators are sleep-bound
## bookkeeping processes and must not consume IDA worker slots, or they would
## starve (and deadlock) the very workers they are waiting on.
##
## Each coordinator (run_batch_ida_per_record.py) draws its per-record IDA workers
## from the shared, machine-global process_semaphore, so across ALL active
## buildings at most `physical_cores - 3` IDA runs are live at once. As a building
## finishes its records, its coordinator retires and the launcher starts the next
## building in its place. This is how a handful of buildings fill a many-core box
## from a single launch script.

# --- configure here -------------------------------------------------------
# The buildings to run. Each entry is a folder containing run_batch_ida_per_record.py
# plus the named config. `config` may be a bare filename (resolved inside the
# folder) or an absolute path.
buildings: list[dict[str, str]] = [
    {"folder": r"D:\07_wp1_casestudy_sites\site_0\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_0\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_1\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_1\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_2\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_2\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_3\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_3\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_4\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_4\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_5\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_5\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_6\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_6\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_7\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_7\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_8\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_8\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_9\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_9\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_10\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_10\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_11\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_11\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_12\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_12\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_13\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_13\5s\mdof\config_ida_htf_femap695.py"},
    {"folder": r"D:\07_wp1_casestudy_sites\site_14\5s\mdof",
     "config": r"D:\07_wp1_casestudy_sites\site_14\5s\mdof\config_ida_htf_femap695.py"},

    # {"folder": r"D:\07_wp1_casestudy_sites\site_15\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_15\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_16\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_16\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_17\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_17\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_18\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_18\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_19\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_19\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_20\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_20\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_21\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_21\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_22\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_22\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_23\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_23\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_24\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_24\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_25\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_25\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_26\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_26\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_27\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_27\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_28\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_28\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_29\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_29\5s\mdof\config_ida_htf_femap695.py"},
    
    # {"folder": r"D:\07_wp1_casestudy_sites\site_30\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_30\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_31\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_31\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_32\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_32\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_33\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_33\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_34\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_34\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_35\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_35\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_36\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_36\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_37\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_37\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_38\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_38\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_39\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_39\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_40\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_40\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_41\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_41\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_42\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_42\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_43\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_43\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_44\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_44\5s\mdof\config_ida_htf_femap695.py"},
    
    # {"folder": r"D:\07_wp1_casestudy_sites\site_45\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_45\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_46\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_46\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_47\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_47\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_48\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_48\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_49\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_49\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_50\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_50\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_51\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_51\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_52\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_52\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_53\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_53\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_54\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_54\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_55\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_55\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_56\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_56\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_57\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_57\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_58\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_58\5s\mdof\config_ida_htf_femap695.py"},
    # {"folder": r"D:\07_wp1_casestudy_sites\site_59\5s\mdof",
    #  "config": r"D:\07_wp1_casestudy_sites\site_59\5s\mdof\config_ida_htf_femap695.py"},
]
max_coordinators = 10          # how many building coordinators / windows at once
window_name_index = 1         # the number of folder to go up in directory to get to unique folder name. e.g. 1 = parent folder of config
runner_script_name = "run_batch_ida_per_record.py"
# Show a console window per record worker (default). Set to False (or pass
# --quiet) to hide them and stream each worker's output to worker_logs/ instead --
# useful when many workers run at once and the windows would be noise.
show_worker_windows = True
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


def launch_building(runner: Path, config: Path, show_windows: bool, window_name_index: int):
    """Launch a building's IDA coordinator in its own minimised console window.

    The coordinator uses the shared semaphore so all buildings' workers share one
    machine-global cap; its per-record workers show windows unless show_windows is
    False, in which case they stream to worker_logs/.
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
        f"mod.run(r'''{config}''', use_semaphore=True, show_worker_windows={show_windows})"
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
        show_worker_windows: bool = show_worker_windows,
        window_name_index: int = window_name_index):

    specs = building_specs if building_specs is not None else buildings
    resolved = resolve_buildings(specs)

    print(f"Launch Time: {datetime.now()} | {len(resolved)} building(s), "
          f"up to {max_coordinators} coordinator(s) at once, "
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

        proc, title = launch_building(runner, config, show_worker_windows, window_name_index)
        running.append({"proc": proc, "title": title, "start_time": datetime.now()})
        pbar.write(f"launched building: {title} PID={proc.pid}")

    while running:
        reap()
        time.sleep(0.5)

    pbar.close()
    print(f"All buildings complete: {datetime.now()}")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_batch_ida_buildings.py
    #   python run_batch_ida_buildings.py --quiet --max-coordinators 5
    # By default the buildings list configured at the top of this file is run.
    parser = argparse.ArgumentParser(
        description="Run IDAs for several buildings at once, sharing one machine-global core cap.")
    parser.add_argument("--max-coordinators", type=int, default=max_coordinators,
                        help="how many building coordinators (windows) to run at once")
    parser.add_argument("--quiet", action="store_true",
                        help="hide per-record worker windows; stream them to worker_logs/ instead")
    parser.add_argument("--window-name-index", type=int, default=window_name_index,
                            help="the index of the folder in config.parents to be used as coordinator window title. 1 corresponds to parent folder")
    args = parser.parse_args()

    run(max_coordinators=args.max_coordinators,
        show_worker_windows=not args.quiet,
        window_name_index=args.window_name_index)
