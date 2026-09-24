import argparse
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from phd_project.process_semaphore.process_semaphore import (
    acquire_slot, release_slot, get_current_running, get_max_concurrent,
    set_max_concurrent, describe_max_concurrent, DEFAULT_RESERVED_CORES)

## Generic batch launcher: runs every (script, config) pair below, with the number
## of simultaneous jobs capped by the shared, machine-global process_semaphore.
##
## --n-cores is REQUIRED: it sets that cap for every analysis process on this box
## (not just this launcher's) and is sticky until the next --n-cores changes it, so
## each campaign has to state the budget it wants. Large models need a much lower
## cap than small ones -- one job per core thrashes the L3 cache and saturates the
## memory bus, making the workstation unusable for everyone else.

running_processes = []  # List of dicts: {proc, script, config, start_time}

# Machine-global cap on simultaneous jobs. None -> must be supplied via --n-cores.
n_cores = None

# Paths to your scripts and the configurations
scripts_and_configs = [
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_00/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_00/3s/mdof/config_modal.py')],
        "name": ['group_3s_00_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_01/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_01/3s/mdof/config_modal.py')],
        "name": ['group_3s_01_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_02/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_02/3s/mdof/config_modal.py')],
        "name": ['group_3s_02_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_03/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_03/3s/mdof/config_modal.py')],
        "name": ['group_3s_03_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_04/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_04/3s/mdof/config_modal.py')],
        "name": ['group_3s_04_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_05/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_05/3s/mdof/config_modal.py')],
        "name": ['group_3s_05_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_06/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_06/3s/mdof/config_modal.py')],
        "name": ['group_3s_06_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_07/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_07/3s/mdof/config_modal.py')],
        "name": ['group_3s_07_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_08/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_08/3s/mdof/config_modal.py')],
        "name": ['group_3s_08_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_09/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_09/3s/mdof/config_modal.py')],
        "name": ['group_3s_09_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_10/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_10/3s/mdof/config_modal.py')],
        "name": ['group_3s_10_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_11/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_11/3s/mdof/config_modal.py')],
        "name": ['group_3s_11_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_12/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_12/3s/mdof/config_modal.py')],
        "name": ['group_3s_12_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_13/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_13/3s/mdof/config_modal.py')],
        "name": ['group_3s_13_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_14/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_14/3s/mdof/config_modal.py')],
        "name": ['group_3s_14_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_15/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_15/3s/mdof/config_modal.py')],
        "name": ['group_3s_15_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_16/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_16/3s/mdof/config_modal.py')],
        "name": ['group_3s_16_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_17/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_17/3s/mdof/config_modal.py')],
        "name": ['group_3s_17_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_18/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_18/3s/mdof/config_modal.py')],
        "name": ['group_3s_18_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_19/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_19/3s/mdof/config_modal.py')],
        "name": ['group_3s_19_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_20/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_20/3s/mdof/config_modal.py')],
        "name": ['group_3s_20_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_21/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_21/3s/mdof/config_modal.py')],
        "name": ['group_3s_21_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_22/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_22/3s/mdof/config_modal.py')],
        "name": ['group_3s_22_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_23/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_23/3s/mdof/config_modal.py')],
        "name": ['group_3s_23_modal']
    },
    {
        "script": Path("D:/08_wp1_fixed_record_sets/group_3s_24/3s/mdof/run_modal.py"),
        "config": [Path('D:/08_wp1_fixed_record_sets/group_3s_24/3s/mdof/config_modal.py')],
        "name": ['group_3s_24_modal']
    }
]


def launch_script(script_path: Path, config_path: Path, window_title: str | None = None):
    SW_SHOWMINNOACTIVE = 7
    venv_python = Path(sys.executable)

    # default title = the config file's full path (unique per job); an explicit
    # window_title (e.g. from a job's "name" list) overrides it
    if window_title is None:
        window_title = str(config_path)

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
        startupinfo=startupinfo
    )

    return proc, window_title


def check_for_finished(pbar):
    """Checks for finished processes, releases slots, and updates tqdm."""
    global running_processes
    still_running = []
    for entry in running_processes:
        if entry["proc"].poll() is not None:
            elapsed = datetime.now() - entry["start_time"]
            pbar.write(f"✅ Finished: {entry['display_name']} in {elapsed.total_seconds():.1f}s")
            release_slot(entry["proc"].pid)
            pbar.update(1)
        else:
            still_running.append(entry)
    running_processes = still_running

def main(n_cores: int | None = n_cores):
    if n_cores is not None:
        set_max_concurrent(n_cores, set_by=Path(__file__).name)
    print(f"Launch Time: {datetime.now()} | {describe_max_concurrent()}")

    total_scripts = sum(len(entry["config"]) for entry in scripts_and_configs)
    pbar = tqdm(total=total_scripts, desc="Batch Progress", unit="script")

    for entry in scripts_and_configs:
        script, config_paths = entry["script"], entry["config"]
        names = entry.get("name")  # optional list of window titles aligned to config_paths

        for i, cfg_path in enumerate(config_paths):
            # ACTIVE WAITING: Check for finished jobs while waiting for a slot
            while True:
                check_for_finished(pbar) # This ensures the bar moves during the launch phase
                if len(get_current_running()) < get_max_concurrent():
                    break
                time.sleep(0.1)

            title = names[i] if names and i < len(names) else None
            proc, display_name = launch_script(script, cfg_path, window_title=title)
            acquire_slot(proc.pid)
            
            pbar.write(f"▶️ Launched: {display_name}")
            running_processes.append({
                "proc": proc,
                "display_name": display_name,
                "start_time": datetime.now()
            })

    # FINAL WAIT: For the last remaining batch
    while running_processes:
        check_for_finished(pbar)
        time.sleep(0.1)
    
    pbar.close()


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_batch_scripts.py --n-cores 20
    parser = argparse.ArgumentParser(
        description="Run every configured (script, config) pair, sharing one "
                    "machine-global core cap.")
    parser.add_argument("--n-cores", type=int, default=n_cores, required=(n_cores is None),
                        help="machine-global cap on simultaneous jobs across ALL launchers "
                             f"on this machine (default policy: physical cores - "
                             f"{DEFAULT_RESERVED_CORES}). Sticky until the next --n-cores. "
                             "Use a low value (~20) for large models, which otherwise thrash "
                             "the cache and make the workstation unusable for others.")
    args = parser.parse_args()

    main(n_cores=args.n_cores)