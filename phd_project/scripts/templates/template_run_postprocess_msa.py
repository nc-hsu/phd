import argparse
from pathlib import Path

from standes.analysis.msa import (
    collate_msa_results, collate_stripe, find_stripe_pickles, process_msa_results,
    stripe_id)
from standes.utils import import_from_path


## Re-run ONLY the MSA post-processing on an already-completed MSA folder.
##
## This reads the existing per-record logs under each stripe_<id>/ folder and rebuilds
## the stripe logs, the collated msa_log.json and the summary / collapse-fragility
## outputs -- it never touches OpenSees and runs no analyses. Use it to (re)generate
## msa_summary.json and collapse_fragility.json after the analyses are done, e.g. after
## changing edp_idxs / edp_tags or dropping in a stripe that was run separately.
def run(config_data: str | Path):
    config_path = Path(config_data)
    if not (config_path.exists() and config_path.is_file()):
        raise ValueError(f"Invalid path for config_data: {config_path}")

    config: dict = import_from_path(config_path).config

    output_folder: Path = config["result_dst"]
    if not output_folder.exists():
        raise FileNotFoundError(
            f"{output_folder} not found -- run the MSA first so there are results to "
            "post-process.")

    # re-assemble the stripe logs from the per-record logs, so the collated log
    # reflects whatever is currently on disk
    stripe_pickles = find_stripe_pickles(
        config["gm_selection_src"], ascending=config["stripe_order_ascending"])
    if not stripe_pickles:
        raise FileNotFoundError(
            f"No stripe selection pickles found in {config['gm_selection_src']}")

    for stripe_pickle in stripe_pickles:
        collate_stripe(output_folder, stripe_id(stripe_pickle), stripe_pickle)
    collate_msa_results(output_folder)

    process_msa_results(
        output_folder,
        edp_idxs=config["edp_idxs"],
        edp_tags=config["edp_tags"],
        collapse_fragility=config["calculate_collapse_fragility"])

    print(f"Post-processing complete: {output_folder}")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_postprocess_msa.py config_msa.py
    parser = argparse.ArgumentParser(
        description="Re-run only the post-processing of a completed MSA (no analyses).")
    parser.add_argument("config", nargs="?", default="config_msa.py",
                        help="MSA config file (relative to this folder, or an absolute path)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path)
