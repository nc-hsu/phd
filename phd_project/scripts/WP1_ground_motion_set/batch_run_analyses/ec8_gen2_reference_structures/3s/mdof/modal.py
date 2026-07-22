import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from phd_project.process_semaphore.process_semaphore import (
    acquire_slot, release_slot, get_current_running, get_max_concurrent)

running_processes = []  # List of dicts: {proc, script, config, start_time}

# Paths to your scripts and the configurations
scripts_and_configs = [
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_13/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_13/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_13_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_20/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_20/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_20_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_30/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_30/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_30_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_40/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_40/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_40_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_50/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_50/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_50_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_60/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_60/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_60_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_70/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_70/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_70_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_100/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_100/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_100_modal']
    },
    {
        "script": Path("D:/06_reference_structures_dc2_scA/3s_dc2_scA_120/mdof/run_modal.py"),
        "config": [Path('D:/06_reference_structures_dc2_scA/3s_dc2_scA_120/mdof/config_modal.py')],
        "name": ['3s_dc2_scA_120_modal']
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

def main():
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
    main()