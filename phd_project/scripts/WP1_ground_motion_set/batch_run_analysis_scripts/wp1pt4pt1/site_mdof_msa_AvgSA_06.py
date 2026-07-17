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
        "script": Path("D:/04_site_influence_investigation/site_1/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_1/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_1_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_2/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_2/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_2_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_3/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_3/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_3_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_4/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_4/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_4_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_5/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_5/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_5_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_6/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_6/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_6_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_8/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_8/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_8_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_9/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_9/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_9_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_10/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_10/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_10_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_11/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_11/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_11_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_13/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_13/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_13_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_14/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_14/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_14_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_15/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_15/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_15_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_16/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_16/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_16_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_19/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_19/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_19_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_21/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_21/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_21_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_23/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_23/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_23_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_25/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_25/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_25_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_26/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_26/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_26_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_29/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_29/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_29_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_30/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_30/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_30_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_32/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_32/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_32_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_33/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_33/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_33_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_36/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_36/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_36_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_37/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_37/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_37_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_39/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_39/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_39_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_40/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_40/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_40_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_43/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_43/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_43_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_45/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_45/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_45_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_46/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_46/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_46_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_51/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_51/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_51_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_53/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_53/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_53_mdof_msa_AvgSA_06']
    },
    {
        "script": Path("D:/04_site_influence_investigation/site_54/mdof/run_msa_site.py"),
        "config": [Path('D:/04_site_influence_investigation/site_54/mdof/config_msa_AvgSA_06.py')],
        "name": ['site_54_mdof_msa_AvgSA_06']
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