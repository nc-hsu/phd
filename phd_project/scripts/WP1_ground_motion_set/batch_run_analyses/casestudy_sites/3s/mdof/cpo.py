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
        "script": Path("D:/07_wp1_casestudy_sites/site_0/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_0/mdof/config_cyclic_pushover.py')],
        "name": ['site_0_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_1/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_1/mdof/config_cyclic_pushover.py')],
        "name": ['site_1_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_2/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_2/mdof/config_cyclic_pushover.py')],
        "name": ['site_2_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_3/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_3/mdof/config_cyclic_pushover.py')],
        "name": ['site_3_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_4/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_4/mdof/config_cyclic_pushover.py')],
        "name": ['site_4_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_5/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_5/mdof/config_cyclic_pushover.py')],
        "name": ['site_5_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_6/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_6/mdof/config_cyclic_pushover.py')],
        "name": ['site_6_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_7/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_7/mdof/config_cyclic_pushover.py')],
        "name": ['site_7_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_8/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_8/mdof/config_cyclic_pushover.py')],
        "name": ['site_8_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_9/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_9/mdof/config_cyclic_pushover.py')],
        "name": ['site_9_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_10/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_10/mdof/config_cyclic_pushover.py')],
        "name": ['site_10_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_11/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_11/mdof/config_cyclic_pushover.py')],
        "name": ['site_11_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_12/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_12/mdof/config_cyclic_pushover.py')],
        "name": ['site_12_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_13/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_13/mdof/config_cyclic_pushover.py')],
        "name": ['site_13_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_14/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_14/mdof/config_cyclic_pushover.py')],
        "name": ['site_14_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_15/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_15/mdof/config_cyclic_pushover.py')],
        "name": ['site_15_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_16/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_16/mdof/config_cyclic_pushover.py')],
        "name": ['site_16_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_17/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_17/mdof/config_cyclic_pushover.py')],
        "name": ['site_17_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_18/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_18/mdof/config_cyclic_pushover.py')],
        "name": ['site_18_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_19/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_19/mdof/config_cyclic_pushover.py')],
        "name": ['site_19_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_20/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_20/mdof/config_cyclic_pushover.py')],
        "name": ['site_20_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_21/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_21/mdof/config_cyclic_pushover.py')],
        "name": ['site_21_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_22/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_22/mdof/config_cyclic_pushover.py')],
        "name": ['site_22_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_23/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_23/mdof/config_cyclic_pushover.py')],
        "name": ['site_23_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_24/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_24/mdof/config_cyclic_pushover.py')],
        "name": ['site_24_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_25/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_25/mdof/config_cyclic_pushover.py')],
        "name": ['site_25_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_26/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_26/mdof/config_cyclic_pushover.py')],
        "name": ['site_26_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_27/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_27/mdof/config_cyclic_pushover.py')],
        "name": ['site_27_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_28/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_28/mdof/config_cyclic_pushover.py')],
        "name": ['site_28_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_29/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_29/mdof/config_cyclic_pushover.py')],
        "name": ['site_29_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_30/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_30/mdof/config_cyclic_pushover.py')],
        "name": ['site_30_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_31/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_31/mdof/config_cyclic_pushover.py')],
        "name": ['site_31_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_32/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_32/mdof/config_cyclic_pushover.py')],
        "name": ['site_32_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_33/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_33/mdof/config_cyclic_pushover.py')],
        "name": ['site_33_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_34/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_34/mdof/config_cyclic_pushover.py')],
        "name": ['site_34_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_35/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_35/mdof/config_cyclic_pushover.py')],
        "name": ['site_35_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_36/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_36/mdof/config_cyclic_pushover.py')],
        "name": ['site_36_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_37/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_37/mdof/config_cyclic_pushover.py')],
        "name": ['site_37_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_38/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_38/mdof/config_cyclic_pushover.py')],
        "name": ['site_38_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_39/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_39/mdof/config_cyclic_pushover.py')],
        "name": ['site_39_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_40/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_40/mdof/config_cyclic_pushover.py')],
        "name": ['site_40_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_41/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_41/mdof/config_cyclic_pushover.py')],
        "name": ['site_41_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_42/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_42/mdof/config_cyclic_pushover.py')],
        "name": ['site_42_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_43/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_43/mdof/config_cyclic_pushover.py')],
        "name": ['site_43_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_44/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_44/mdof/config_cyclic_pushover.py')],
        "name": ['site_44_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_45/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_45/mdof/config_cyclic_pushover.py')],
        "name": ['site_45_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_46/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_46/mdof/config_cyclic_pushover.py')],
        "name": ['site_46_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_47/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_47/mdof/config_cyclic_pushover.py')],
        "name": ['site_47_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_48/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_48/mdof/config_cyclic_pushover.py')],
        "name": ['site_48_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_49/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_49/mdof/config_cyclic_pushover.py')],
        "name": ['site_49_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_50/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_50/mdof/config_cyclic_pushover.py')],
        "name": ['site_50_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_51/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_51/mdof/config_cyclic_pushover.py')],
        "name": ['site_51_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_52/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_52/mdof/config_cyclic_pushover.py')],
        "name": ['site_52_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_53/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_53/mdof/config_cyclic_pushover.py')],
        "name": ['site_53_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_54/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_54/mdof/config_cyclic_pushover.py')],
        "name": ['site_54_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_55/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_55/mdof/config_cyclic_pushover.py')],
        "name": ['site_55_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_56/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_56/mdof/config_cyclic_pushover.py')],
        "name": ['site_56_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_57/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_57/mdof/config_cyclic_pushover.py')],
        "name": ['site_57_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_58/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_58/mdof/config_cyclic_pushover.py')],
        "name": ['site_58_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_59/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_59/mdof/config_cyclic_pushover.py')],
        "name": ['site_59_cpo']
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