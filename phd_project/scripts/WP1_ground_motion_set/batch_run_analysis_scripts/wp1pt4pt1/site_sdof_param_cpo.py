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
        "script": Path("D:/04_site_influence_investigation/site_0/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_0/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_0_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_1/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_1/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_1_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_2/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_2/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_2_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_3/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_3/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_3_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_4/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_4/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_4_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_5/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_5/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_5_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_6/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_6/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_6_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_7/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_7/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_7_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_8/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_8/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_8_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_9/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_9/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_9_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_10/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_10/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_10_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_11/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_11/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_11_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_12/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_12/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_12_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_13/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_13/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_13_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_14/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_14/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_14_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_15/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_15/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_15_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_16/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_16/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_16_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_17/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_17/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_17_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_18/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_18/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_18_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_19/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_19/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_19_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_20/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_20/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_20_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_21/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_21/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_21_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_22/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_22/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_22_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_23/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_23/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_23_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_24/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_24/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_24_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_25/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_25/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_25_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_26/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_26/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_26_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_27/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_27/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_27_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_28/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_28/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_28_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_29/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_29/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_29_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_30/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_30/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_30_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_31/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_31/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_31_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_32/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_32/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_32_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_33/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_33/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_33_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_34/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_34/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_34_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_35/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_35/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_35_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_36/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_36/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_36_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_37/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_37/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_37_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_38/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_38/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_38_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_39/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_39/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_39_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_40/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_40/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_40_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_41/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_41/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_41_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_42/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_42/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_42_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_43/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_43/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_43_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_44/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_44/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_44_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_45/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_45/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_45_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_46/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_46/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_46_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_47/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_47/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_47_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_48/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_48/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_48_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_49/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_49/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_49_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_50/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_50/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_50_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_51/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_51/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_51_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_52/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_52/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_52_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_53/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_53/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_53_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_54/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_54/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_54_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_55/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_55/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_55_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_56/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_56/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_56_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_57/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_57/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_57_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_58/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_58/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_58_sdof_param_cpo']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_59/sdof_param/run_cyclic_pushover.py"),
        "config": [Path('D:/04_site_influence_investigation/site_59/sdof_param/config_cyclic_pushover.py')],
        "name": ['site_59_sdof_param_cpo']
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