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
        "script": Path("D:/07_wp1_casestudy_sites/site_2/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_2/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_2_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_3/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_3/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_3_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_4/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_4/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_4_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_5/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_5/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_5_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_6/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_6/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_6_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_7/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_7/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_7_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_8/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_8/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_8_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_9/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_9/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_9_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_10/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_10/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_10_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_11/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_11/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_11_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_13/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_13/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_13_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_14/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_14/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_14_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_15/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_15/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_15_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_16/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_16/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_16_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_19/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_19/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_19_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_20/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_20/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_20_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_21/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_21/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_21_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_23/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_23/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_23_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_25/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_25/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_25_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_26/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_26/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_26_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_29/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_29/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_29_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_30/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_30/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_30_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_32/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_32/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_32_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_33/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_33/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_33_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_35/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_35/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_35_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_36/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_36/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_36_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_37/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_37/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_37_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_39/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_39/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_39_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_40/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_40/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_40_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_43/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_43/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_43_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_45/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_45/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_45_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_46/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_46/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_46_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_47/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_47/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_47_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_49/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_49/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_49_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_51/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_51/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_51_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_53/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_53/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_53_3s_mdof_msa_AvgSA_03']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_54/3s/mdof/run_msa_site.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_54/3s/mdof/config_msa_AvgSA_03.py')],
        "name": ['site_54_3s_mdof_msa_AvgSA_03']
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