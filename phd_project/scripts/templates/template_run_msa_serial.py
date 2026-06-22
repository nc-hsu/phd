from datetime import datetime
from pathlib import Path

from standes.analysis.msa import msa_multiple_stripes, find_stripe_pickles
from standes.utils import import_from_path


## serial MSA driver: run every stripe in turn, every record of a stripe in turn,
## in a single process. Simplest of the three run modes -- good for debugging or
## small problems. The parallel launchers (run_batch_msa_*.py) produce identical
## output.
def run(config_data: str | Path | dict):
    # load the configuration data from the provided file path
    if isinstance(config_data, str):
        config_data = Path(config_data)
        if not (config_data.exists() and config_data.is_file()):
            raise ValueError(f"Invalid path for config_data: {config_data}")

    if isinstance(config_data, Path):
        config: dict = import_from_path(config_data).config
    elif not isinstance(config_data, dict):
        raise ValueError
    else:
        config = config_data

    print(f"Launch Time: {datetime.now()}")

    output_folder: Path = config["result_dst"]
    output_folder.mkdir(parents=True, exist_ok=True)

    # discover and order the stripe selection pickles
    stripe_pickles = find_stripe_pickles(
        config["gm_selection_src"], ascending=config["stripe_order_ascending"])
    if not stripe_pickles:
        raise FileNotFoundError(
            f"No stripe selection pickles found in {config['gm_selection_src']}")

    start_time = datetime.now()

    msa_multiple_stripes(
        result_dst=output_folder,
        stripe_pickles=stripe_pickles,
        record_src=config["record_src"],
        msa_parameters=config["msa_parameters"],
        injection_functions=config["injection_functions"],
        max_n_records=config["max_n_records"])

    elapsed_time = datetime.now() - start_time
    print(f"All stripes complete. Time elapsed: {elapsed_time}")

    ## POST PROCESSING
    if config["post_process"]:
        from standes.analysis.msa import process_msa_results
        process_msa_results(
            output_folder,
            edp_idxs=config["edp_idxs"],
            edp_tags=config["edp_tags"],
            collapse_fragility=config["calculate_collapse_fragility"])


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config_msa.py"
    run(config_path)
