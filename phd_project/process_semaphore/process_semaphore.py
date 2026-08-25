# import json
# import time
# import psutil
# from pathlib import Path
# from filelock import FileLock
# from contextlib import contextmanager

# SEMAPHORE_PATH = Path(__file__).parents[0] / "global_process_counter.json"
# LOCK = FileLock(str(SEMAPHORE_PATH) + ".lock")

# def calculate_max_concurrent_dynamic():
#     usage = psutil.cpu_percent(interval=1)
#     if usage < 40:
#         return psutil.cpu_count(logical=False)
#     elif usage < 60:
#         return max(1, psutil.cpu_count(logical=False) - 1)
#     else:
#         return max(1, psutil.cpu_count(logical=False) - 2)
    

# def increment_semaphore():
#     with LOCK:
#         count = 0
#         if SEMAPHORE_PATH.exists():
#             count = json.loads(SEMAPHORE_PATH.read_text())["count"]
#         if count >= MAX_GLOBAL_PROCS:
#             return False
#         SEMAPHORE_PATH.write_text(json.dumps({"count": count + 1}))
#         return True


# def decrement_semaphore():
#     with LOCK:
#         if SEMAPHORE_PATH.exists():
#             data = json.loads(SEMAPHORE_PATH.read_text())
#             data["count"] = max(0, data["count"] - 1)
#             SEMAPHORE_PATH.write_text(json.dumps(data))


# def wait_for_slot():
#     while not increment_semaphore():
#         time.sleep(0.5)


# def get_current_count():
#     with LOCK:
#         if SEMAPHORE_PATH.exists():
#             return json.loads(SEMAPHORE_PATH.read_text())["count"]
#         return 0
    

# @contextmanager
# def acquire_process_slot():
#     wait_for_slot()
#     try:
#         yield
#     finally:
#         decrement_semaphore()

import json
import time
from pathlib import Path
from filelock import FileLock
import psutil
from datetime import datetime

SEMAPHORE_PATH = Path(__file__).parent / "process_slots.json"
LOCK = FileLock(str(SEMAPHORE_PATH) + ".lock")

def _read_data():
    if SEMAPHORE_PATH.exists():
        return json.loads(SEMAPHORE_PATH.read_text())
    return {"running": []}

def _write_data(data):
    SEMAPHORE_PATH.write_text(json.dumps(data, indent=2))

def _cleanup_stale_processes(data):
    # Remove processes that no longer exist
    running = []
    for proc_info in data["running"]:
        pid = proc_info.get("pid")
        if pid is not None and psutil.pid_exists(pid):
            running.append(proc_info)
    data["running"] = running

def get_max_concurrent():
    # Example dynamic max concurrent, adjust as you like
    cpu_count = psutil.cpu_count(logical=False)
    free_cores = 40
    return max(1, cpu_count - free_cores)

def acquire_slot(pid: int, display_name: str = ""):
    """Try to acquire a slot for a new process with given pid."""
    while True:
        with LOCK:
            data = _read_data()
            _cleanup_stale_processes(data)
            max_slots = get_max_concurrent()
            if len(data["running"]) < max_slots:
                data["running"].append({
                    "pid": pid,
                    "display_name": display_name,
                    "start_time": datetime.now().isoformat()
                })
                _write_data(data)
                return True
        time.sleep(0.5)

def release_slot(pid: int):
    """Release the slot for the given pid."""
    with LOCK:
        data = _read_data()
        data["running"] = [p for p in data["running"] if p.get("pid") != pid]
        _write_data(data)

def get_current_running():
    with LOCK:
        data = _read_data()
        _cleanup_stale_processes(data)
        _write_data(data)
        return data["running"]

