import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

MAX_CONCURRENT = 40
running_processes = []  # List of dicts: {proc, script, config, start_time}

# Paths to your scripts and the configurations
scripts_and_configs = [
    {
        "script": Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\run_snapback.py"),
        "config": [
            Path(r"E:\01_model_verification_analyses\damping_model\3s_cbf_dc2_41_rayleigh_initial\config_snapback.py")
        ]
    }
]

def launch_script(script_path: Path, config_path: Path):
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

    proc = subprocess.Popen(
        ["powershell.exe", "-Command", ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE
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

            # Wait if at concurrency limit
            while len(running_processes) >= MAX_CONCURRENT:
                check_running_processes()
                time.sleep(0.5)

            # Launch script
            proc, display_name = launch_script(script, cfg_path)
            start_time = datetime.now()
            print(f"▶️ Launched: {display_name} at {start_time.strftime('%H:%M:%S')}")
            running_processes.append({
                "proc": proc,
                "display_name": display_name,
                "start_time": start_time
            })

    # Wait for all to finish
    while running_processes:
        check_running_processes()
        time.sleep(0.5)

if __name__ == "__main__":
    main()