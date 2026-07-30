import argparse
from datetime import datetime
from pathlib import Path

from standes.analysis.msa import (
    msa_one_record, stripe_pickles_by_id, read_stripe_selection)
from standes.utils import import_from_path


## worker: run the NLTHA for a SINGLE (stripe, record) of the config. Dispatched
## one-per-process by the batch launchers so records can be analysed concurrently
## (one CPU core per record). The unit of work is selected by `tag`, formatted as
## "{stripe_id}:{record}" (e.g. "00pt360:1"), so no per-record config file is needed.
## The stripe id is the stripe's intensity tag (stable regardless of which other
## stripes exist); records are 0-indexed into the stripe selection.

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

    # tag = "{stripe_id}:{record}"; the record index is the last field
    stripe_tag, record_str = str(tag).rsplit(":", 1)
    record_tag = int(record_str)
    print(f"Launch Time: {datetime.now()} | stripe {stripe_tag} record {record_tag}")

    output_folder: Path = config["result_dst"]
    stripe_folder = output_folder / f"stripe_{stripe_tag}"
    stripe_folder.mkdir(parents=True, exist_ok=True)

    # resolve this stripe's pickle from its (position-independent) stripe id
    stripe_pickle = stripe_pickles_by_id(
        config["gm_selection_src"],
        ascending=config["stripe_order_ascending"])[stripe_tag]

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
        injection_functions=config["injection_functions"],
        skip_if_complete=config.get("resume", True))

    elapsed_time = datetime.now() - start_time
    print(f"Time elapsed (stripe {stripe_tag} record {record_tag}): {elapsed_time}")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_msa_per_record.py config_msa.py 00pt360:0
    parser = argparse.ArgumentParser(description="Run the MSA for a SINGLE (stripe, record) of the config.")
    parser.add_argument("config", nargs="?", default="config_msa.py",
                        help="config file (relative to this folder, or an absolute path)")
    parser.add_argument("tag", nargs="?", default="00pt360:0",
                        help="the single stripe_id:record tag to run")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path, args.tag)
