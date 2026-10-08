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
        "script": Path("D:/07_wp1_casestudy_sites/site_0/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_0/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site0_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_1/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_1/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site1_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_2/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_2/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site2_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_3/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_3/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site3_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_4/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_4/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site4_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_5/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_5/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site5_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_6/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_6/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site6_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_7/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_7/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site7_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_8/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_8/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site8_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_9/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_9/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site9_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_10/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_10/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site10_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_11/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_11/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site11_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_12/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_12/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site12_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_13/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_13/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site13_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_14/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_14/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site14_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_15/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_15/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site15_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_16/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_16/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site16_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_17/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_17/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site17_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_18/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_18/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site18_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_19/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_19/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site19_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_20/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_20/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site20_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_21/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_21/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site21_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_22/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_22/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site22_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_23/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_23/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site23_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_24/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_24/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site24_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_25/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_25/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site25_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_26/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_26/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site26_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_27/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_27/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site27_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_28/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_28/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site28_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_29/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_29/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site29_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_30/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_30/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site30_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_31/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_31/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site31_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_32/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_32/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site32_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_33/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_33/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site33_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_34/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_34/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site34_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_35/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_35/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site35_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_36/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_36/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site36_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_37/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_37/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site37_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_38/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_38/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site38_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_39/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_39/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site39_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_40/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_40/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site40_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_41/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_41/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site41_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_42/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_42/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site42_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_43/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_43/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site43_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_44/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_44/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site44_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_45/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_45/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site45_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_46/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_46/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site46_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_47/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_47/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site47_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_48/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_48/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site48_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_49/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_49/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site49_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_50/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_50/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site50_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_51/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_51/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site51_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_52/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_52/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site52_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_53/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_53/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site53_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_54/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_54/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site54_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_55/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_55/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site55_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_56/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_56/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site56_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_57/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_57/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site57_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_58/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_58/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site58_cpo']
    },
    {
        "script": Path("D:/07_wp1_casestudy_sites/site_59/7s/mdof/run_cyclic_pushover.py"),
        "config": [Path('D:/07_wp1_casestudy_sites/site_59/7s/mdof/config_cyclic_pushover.py')],
        "name": ['7s_cbf_dc2_site59_cpo']
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