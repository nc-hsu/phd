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
        "script": Path("D:/07_wp1_casestudy_sites/site_0/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_0/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site0_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_1/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_1/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site1_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_2/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_2/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site2_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_3/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_3/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site3_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_4/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_4/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site4_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_5/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_5/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site5_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_6/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_6/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site6_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_7/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_7/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site7_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_8/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_8/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site8_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_9/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_9/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site9_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_10/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_10/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site10_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_11/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_11/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site11_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_12/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_12/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site12_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_13/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_13/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site13_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_14/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_14/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site14_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_15/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_15/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site15_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_16/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_16/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site16_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_17/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_17/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site17_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_18/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_18/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site18_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_19/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_19/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site19_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_20/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_20/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site20_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_21/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_21/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site21_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_22/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_22/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site22_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_23/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_23/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site23_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_24/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_24/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site24_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_25/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_25/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site25_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_26/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_26/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site26_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_27/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_27/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site27_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_28/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_28/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site28_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_29/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_29/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site29_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_30/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_30/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site30_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_31/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_31/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site31_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_32/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_32/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site32_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_33/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_33/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site33_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_34/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_34/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site34_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_35/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_35/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site35_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_36/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_36/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site36_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_37/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_37/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site37_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_38/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_38/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site38_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_39/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_39/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site39_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_40/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_40/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site40_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_41/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_41/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site41_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_42/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_42/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site42_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_43/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_43/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site43_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_44/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_44/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site44_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_45/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_45/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site45_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_46/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_46/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site46_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_47/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_47/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site47_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_48/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_48/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site48_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_49/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_49/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site49_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_50/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_50/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site50_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_51/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_51/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site51_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_52/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_52/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site52_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_53/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_53/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site53_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_54/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_54/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site54_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_55/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_55/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site55_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_56/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_56/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site56_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_57/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_57/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site57_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_58/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_58/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site58_modal']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_59/5s/mdof/run_modal.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_59/5s/mdof/config_modal.py')],
        "name": ['5s_cbf_dc2_site59_modal']
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