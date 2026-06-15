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
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_10_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_10_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_102_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_102_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_122_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_122_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_20_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_20_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_31_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_31_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_41_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_41_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_51_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_51_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_61_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_61_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_71_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/3s_cbf_dc2_71_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_10_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_10_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_102_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_102_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_122_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_122_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_20_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_20_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_31_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_31_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_41_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_41_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_51_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_51_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_61_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_61_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_71_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/5s_cbf_dc2_71_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_10_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_10_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_20_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_20_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_31_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_31_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_41_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_41_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_51_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_51_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_61_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_61_ss/config_pushover.py')]
    },
    {
        "script": Path("E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_71_ss/run_pushover.py"),
        "config": [Path('E:/03_wp1pt4pt1_dc2_sdof_fitting/7s_cbf_dc2_71_ss/config_pushover.py')]
    }
]


def launch_script(script_path: Path, config_path: Path):
    SW_SHOWMINNOACTIVE = 7
    venv_python = Path(sys.executable)

    window_title = f"{script_path.parts[-2]}/{script_path.name} [{config_path.name}]"

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
        
        for cfg_path in config_paths:
            # ACTIVE WAITING: Check for finished jobs while waiting for a slot
            while True:
                check_for_finished(pbar) # This ensures the bar moves during the launch phase
                if len(get_current_running()) < get_max_concurrent():
                    break
                time.sleep(0.1)

            proc, display_name = launch_script(script, cfg_path)
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