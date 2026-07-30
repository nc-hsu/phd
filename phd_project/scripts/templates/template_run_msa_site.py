import argparse
from pathlib import Path

from standes.utils import import_from_path


## thin per-site entry point for the multi-site batch (run_batch_msa_sites.py).
## Defaults draw NLTHA workers from the shared, machine-global semaphore (so every
## site competes for the same core budget). The multi-site launcher passes these
## options through, so they can be overridden per launch: swap the semaphore for a
## fixed --max-workers count, or hide the per-worker console windows with --quiet
## (output then goes to worker_logs/), leaving just this site's own coordinator
## window as its dashboard.
def run(config_data: str | Path,
        use_semaphore: bool = True,
        max_workers: int | None = None,
        show_worker_windows: bool = True):
    batch = import_from_path(
        Path(__file__).parent / "run_batch_msa_per_stripe_record.py")
    batch.run(config_data, use_semaphore=use_semaphore,
              max_workers=max_workers, show_worker_windows=show_worker_windows)


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_msa_site.py config_msa.py
    #   python run_msa_site.py config_msa.py --max-workers 4 --quiet
    parser = argparse.ArgumentParser(description="Run an MSA for a single site.")
    parser.add_argument("config", nargs="?", default="config_msa.py",
                        help="config file (relative to this folder, or an absolute path)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="run with this fixed worker count instead of the shared "
                             "semaphore (implies --no-semaphore)")
    parser.add_argument("--no-semaphore", action="store_true",
                        help="do not use the machine-global process_semaphore "
                             "(fixed worker count from --max-workers, or cores - 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="no worker windows; stream worker output to worker_logs/ instead")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    use_semaphore = not args.no_semaphore and args.max_workers is None
    run(config_path, use_semaphore=use_semaphore, max_workers=args.max_workers,
        show_worker_windows=not args.quiet)
