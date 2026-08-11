import psutil

def audit_openseespy_cores(script_identifier=""):
    """
    Audits running Python processes to map logical CPU affinities
    to physical cores.
    
    `script_identifier`: Optional string to filter your specific 
                         script name (e.g., 'stripe_analysis.py').
    """
    logical_count = psutil.cpu_count(logical=True)
    physical_count = psutil.cpu_count(logical=False)
    threads_per_core = logical_count // physical_count  # Usually 2 with SMT

    print(f"System: {physical_count} Physical Cores | {logical_count} Logical Threads\n")
    print(f"{'PID':<8} {'Logical Thread IDs':<25} {'Physical Core Index':<20} {'Status'}")
    print("-" * 65)

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Look for python processes
            if 'python' in proc.info['name'].lower():
                cmdline = " ".join(proc.info['cmdline'] or [])
                
                # Filter by script name if provided
                if script_identifier.lower() in cmdline.lower():
                    p = psutil.Process(proc.info['pid'])
                    affinity = p.cpu_affinity()
                    
                    # Map logical CPU IDs to physical core indices
                    # e.g., Logical 0 & 1 -> Physical 0
                    phys_cores = sorted(list({c // threads_per_core for c in affinity}))
                    
                    # Check for thread-stacking risk
                    is_pinned = len(affinity) == 1
                    status = "Pinned (Safe)" if is_pinned else "Unpinned (OS scheduled)"

                    print(
                        f"{p.pid:<8} {str(affinity):<25} "
                        f"{str(phys_cores):<20} {status}"
                    )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

if __name__ == "__main__":
    # Pass part of your script filename to filter out other Python tasks if needed
    audit_openseespy_cores()