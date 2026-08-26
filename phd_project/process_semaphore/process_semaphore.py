"""
Machine-global process semaphore for the parallel analysis launchers.

A single, file-locked JSON file (``process_slots.json``) records the PIDs of every
NLTHA worker currently running on this machine, so that several independently
launched batches still share ONE core budget instead of each claiming the whole
box.

The cap itself lives in ``semaphore_config.json`` (git-ignored, per-machine) and is
set from the launchers' ``--n-cores`` flag via :func:`set_max_concurrent`. It is
sticky: it stays until the next ``--n-cores`` changes it. When nothing has been
set, the cap falls back to ``physical cores - DEFAULT_RESERVED_CORES``.

Why the cap is a parameter: large (e.g. 5-storey) models do not fit in the L3
cache slice of a core, so running one worker per core saturates the memory bus and
makes the shared workstation unusable for everyone else. Those campaigns need a
much lower cap than small models do, and that is a per-campaign decision rather
than a property of the machine.

Public API (duck-typed by ``standes.parallel.run_records_in_subprocesses``):
``acquire_slot``, ``release_slot``, ``get_current_running``, ``get_max_concurrent``.
"""
import json
import time
from pathlib import Path
from filelock import FileLock
import psutil
from datetime import datetime

SEMAPHORE_PATH = Path(__file__).parent / "process_slots.json"
LOCK = FileLock(str(SEMAPHORE_PATH) + ".lock")

# cores left free for the OS and interactive work when no cap has been set
DEFAULT_RESERVED_CORES = 4

# the sticky, machine-global cap written by the launchers' --n-cores flag. A
# SEPARATE file (and lock) from process_slots.json: the slots file is hot runtime
# state, and acquire_slot() reads the cap while already holding LOCK.
CONFIG_PATH = Path(__file__).parent / "semaphore_config.json"
CONFIG_LOCK = FileLock(str(CONFIG_PATH) + ".lock")


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


def _read_config() -> dict:
    """The stored cap settings, or {} if nothing is set or the file is unusable.

    Never raises: a missing, corrupt or half-written config must fall back to the
    default cap rather than kill a batch that is already running.
    """
    try:
        with CONFIG_LOCK:
            if not CONFIG_PATH.exists():
                return {}
            data = json.loads(CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def physical_cores() -> int:
    """Physical core count, falling back to logical cores then 1.

    ``psutil.cpu_count(logical=False)`` returns None on some machines/containers.
    """
    return psutil.cpu_count(logical=False) or psutil.cpu_count() or 1


def default_max_concurrent() -> int:
    """The cap used when --n-cores has never been set: physical cores - 4."""
    return max(1, physical_cores() - DEFAULT_RESERVED_CORES)


def get_max_concurrent() -> int:
    """The machine-global cap on simultaneous workers, clamped to [1, cores]."""
    n = _read_config().get("max_concurrent")
    if n is None:
        return default_max_concurrent()
    try:
        n = int(n)
    except (TypeError, ValueError):
        return default_max_concurrent()
    return max(1, min(n, physical_cores()))


def set_max_concurrent(n: int | None, set_by: str = "") -> int:
    """Set (or, with n=None, clear) the sticky machine-global cap.

    Takes effect immediately for every process on this machine: coordinators and
    launchers re-read the cap on each slot check, so lowering it mid-run drains
    the running workers down to the new value instead of replacing them.

    Returns the cap now in force.
    """
    with CONFIG_LOCK:
        CONFIG_PATH.write_text(json.dumps({
            "max_concurrent": None if n is None else int(n),
            "set_at": datetime.now().isoformat(timespec="seconds"),
            "set_by": set_by,
        }, indent=2))
    return get_max_concurrent()


def describe_max_concurrent() -> str:
    """One-line summary of the active cap, for a launcher's startup banner."""
    config = _read_config()
    cap = get_max_concurrent()
    cores = physical_cores()
    if config.get("max_concurrent") is None:
        return f"cap: {cap} of {cores} physical cores (default, cores - {DEFAULT_RESERVED_CORES})"
    origin = " by " + config["set_by"] if config.get("set_by") else ""
    return (f"cap: {cap} of {cores} physical cores "
            f"(set {config.get('set_at', '?')}{origin})")


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
