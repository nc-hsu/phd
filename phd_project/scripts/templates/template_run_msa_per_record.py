from datetime import datetime
from pathlib import Path

from standes.analysis.msa import (
    msa_one_record, find_stripe_pickles, read_stripe_selection)
from standes.utils import import_from_path


## worker: run the NLTHA for a SINGLE (stripe, record) of the config. Dispatched
## one-per-process by the batch launchers so records can be analysed concurrently
## (one CPU core per record). The unit of work is selected by `tag`, formatted as
## "{stripe}:{record}" (e.g. "3:0"), so no per-record config file is needed.
##
## The per-stripe and collated logs are assembled by the launcher (via
## standes.analysis.msa.collate_stripe / collate_msa_results) once every worker
## for a stripe has finished.
def run(config_data: str | Path | dict, tag: str):
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

    stripe_tag, record_tag = (int(part) for part in str(tag).split(":"))
    print(f"Launch Time: {datetime.now()} | stripe {stripe_tag} record {record_tag}")

    output_folder: Path = config["result_dst"]
    stripe_folder = output_folder / f"stripe_{stripe_tag}"
    stripe_folder.mkdir(parents=True, exist_ok=True)

    # reproduce the launcher's stripe ordering to resolve this stripe's pickle
    stripe_pickles = find_stripe_pickles(
        config["gm_selection_src"], ascending=config["stripe_order_ascending"])
    stripe_pickle = stripe_pickles[stripe_tag - 1]

    # resolve this record's filename and (alpha) scale factor from the selection
    record = read_stripe_selection(stripe_pickle).records[record_tag]
    record_path = Path(config["record_src"]) / record.record_name

    start_time = datetime.now()
    msa_one_record(
        stripe_folder=stripe_folder,
        record_tag=record_tag,
        record_path=record_path,
        scale_factor=record.scale_factor,
        msa_parameters=config["msa_parameters"],
        injection_functions=config["injection_functions"])

    elapsed_time = datetime.now() - start_time
    print(f"Time elapsed (stripe {stripe_tag} record {record_tag}): {elapsed_time}")


if __name__ == "__main__":
    # example usage: run a single (stripe, record) for debugging
    config_path = Path(__file__).parent / "config_msa.py"
    run(config_path, "1:0")
