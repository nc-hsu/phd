from pathlib import Path

from standes.utils import import_from_path


## thin per-site entry point for the multi-site batch (run_batch_msa_sites.py).
## The site batch launcher only calls run(config), so this wrapper exists to pin
## the per-site coordinator options: draw NLTHA workers from the shared, machine
## global semaphore (so every site competes for the same core budget) and run the
## workers quietly (no per-worker console window -- output goes to worker_logs/),
## leaving just this site's own coordinator window as its dashboard.
def run(config_data: str | Path):
    batch = import_from_path(
        Path(__file__).parent / "run_batch_msa_per_stripe_record.py")
    batch.run(config_data, use_semaphore=True, show_worker_windows=False)


if __name__ == "__main__":
    run(Path(__file__).parent / "config_msa.py")
