import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from process_semaphore.process_semaphore import (
    acquire_slot, release_slot, get_current_running, get_max_concurrent)

running_processes = []  # List of dicts: {proc, script, config, start_time}

# Paths to your scripts and the configurations
scripts_and_configs = [
    {
        "script": Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\run_multiple_ida_htf.py"),
        "config": [
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\config_multiple_ida_htf_SA_1_to_6.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\config_multiple_ida_htf_SA_7_to_12.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\config_multiple_ida_htf_SA_13_to_17.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\config_multiple_ida_htf_SA_18_to_22.py"),
        ]
    },
    {
        "script": Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12\run_multiple_ida_htf.py"),
        "config": [
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12\config_multiple_ida_htf_SA_1_to_6.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12\config_multiple_ida_htf_SA_7_to_12.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12\config_multiple_ida_htf_SA_13_to_17.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12\config_multiple_ida_htf_SA_18_to_22.py"),
        ]
    },
    {
        "script": Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12_updating\run_multiple_ida_htf.py"),
        "config": [
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12_updating\config_multiple_ida_htf_SA_1_to_6.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12_updating\config_multiple_ida_htf_SA_7_to_12.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12_updating\config_multiple_ida_htf_SA_13_to_17.py"),
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_modal_12_updating\config_multiple_ida_htf_SA_18_to_22.py"),
        ]
    },
]

def launch_script(script_path: Path, config_path: Path):
    SW_SHOWMINNOACTIVE = 7
    venv_python = Path(sys.executable)

    # Unique window title based on script and config
    window_title = f"{script_path.parts[-2]}/{script_path.name} [{config_path.name}]"

    # Inline Python code to run: import the script and call run(config_path)
    py_code = (
        f"import importlib.util; "
        f"path = r'''{script_path}'''; "
        f"spec = importlib.util.spec_from_file_location('mod', path); "
        f"mod = importlib.util.module_from_spec(spec); "
        f"spec.loader.exec_module(mod); "
        f"mod.run(r'''{config_path}''')"
    )

    # PowerShell command with title, run, and pause on error
    ps_command = (
        f"$host.UI.RawUI.WindowTitle = '{window_title}'; "
        f"& '{venv_python}' -c \"{py_code}\"; "
        f"if ($LASTEXITCODE -ne 0) {{ pause }}"
    )

    # Set up startupinfo to launch minimized
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_SHOWMINNOACTIVE  # Minimized, no focus


    # check if we are allowed to launch a new subprocess

    proc = subprocess.Popen(
        ["pwsh", "-NoLogo", "-NoProfile", "-Command", ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startupinfo
    )

    return proc, window_title


def check_running_processes():
    global running_processes
    still_running = []

    for entry in running_processes:
        proc = entry["proc"]
        if proc.poll() is None:
            still_running.append(entry)
        else:
            end_time = datetime.now()
            elapsed = end_time - entry["start_time"]
            print(f"✅ Finished: {entry["display_name"]}] "
                  f"in {elapsed.total_seconds():.1f}s")
    
    running_processes = still_running
    

def main():
    for entry in scripts_and_configs:
        script = entry["script"]
        config_paths = entry["config"]

        if not script.exists():
            print(f"❌ Script not found: {script}")
            continue

        for cfg_path in config_paths:
            if not cfg_path.exists():
                print(f"⚠️ Config not found: {cfg_path}")
                continue

            # Wait for a slot before launching
            # We pass the PID right after launching, so here we just loop until slot available:
            # To avoid race, you can acquire slot just after launching too.
            while True:
                running = get_current_running()
                max_slots = get_max_concurrent()
                if len(running) < max_slots:
                    break
                time.sleep(0.5)

            proc, display_name = launch_script(script, cfg_path)

            # Now register the PID to semaphore file
            acquire_slot(proc.pid)

            start_time = datetime.now()
            print(f"▶️ Launched: {display_name} PID={proc.pid} at {start_time.strftime('%H:%M:%S')}")
            running_processes.append({
                "proc": proc,
                "display_name": display_name,
                "start_time": start_time
            })

    # Wait for all to finish
    while running_processes:
        still_running = []
        for entry in running_processes:
            proc = entry["proc"]
            if proc.poll() is None:
                still_running.append(entry)
            else:
                end_time = datetime.now()
                elapsed = end_time - entry["start_time"]
                print(f"✅ Finished: {entry['display_name']} PID={proc.pid} in {elapsed.total_seconds():.1f}s")
                release_slot(proc.pid)
        running_processes[:] = still_running
        time.sleep(0.5)


if __name__ == "__main__":
    main()