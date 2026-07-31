import argparse
import pickle
from pathlib import Path

from standes.analysis.ida import process_ida_results
from standes.groundmotion import load_ground_motion_from_json
from standes.utils import import_from_path


## Re-run ONLY the IDA post-processing on an already-completed IDA folder.
##
## This reads the existing ida_results.pickle and rebuilds the fragility / IDA-curve
## outputs -- it never touches OpenSees and runs no analyses. Use it to (re)generate
## the collapse fragility with its intensity_measure tag and the per-IM converted
## collapse fragilities (config["convert_collapse_to_ims"]) after the analyses are done.
def run(config_data: str | Path):
    config_path = Path(config_data)
    if not (config_path.exists() and config_path.is_file()):
        raise ValueError(f"Invalid path for config_data: {config_path}")

    config: dict = import_from_path(config_path).config

    output_folder: Path = config["result_dst"]

    results_pickle = output_folder / "ida_results.pickle"
    if not results_pickle.exists():
        raise FileNotFoundError(
            f"{results_pickle} not found -- run the IDA first so there are results to "
            "post-process.")

    with open(results_pickle, "rb") as file:
        ida_results = pickle.load(file)

    # only load records when collapse-IM conversions are requested (else skip I/O)
    convert_collapse_to_ims = config.get("convert_collapse_to_ims")
    gm_records = None
    if convert_collapse_to_ims:
        gm_record_files = {k: Path(config["gm_json_src"]) / v
                           for k, v in config["gm_json_files"].items()}
        gm_records = {tag: load_ground_motion_from_json(gm_record_files[tag])
                      for tag in ida_results.keys()}

    process_ida_results(
        ida_results.values(),
        output_folder,
        edp_idxs=config["edp_idxs"],
        edp_tags=config["edp_tags"],
        ida_fractiles=config["ida_fractiles"],
        collapse_fragility=config["calculate_collapse_fragility"],
        record_ids=ida_results.keys(),
        intensity_measure=config["ida_parameters"]["intensity_measure"],
        gm_records=gm_records,
        convert_collapse_to_ims=convert_collapse_to_ims,
        gravity_factor=config["ida_parameters"]["gravity_factor"])

    print(f"Post-processing complete: {output_folder}")


if __name__ == "__main__":
    # launch from the terminal, e.g.
    #   python run_postprocess_ida.py config_ida_htf.py
    parser = argparse.ArgumentParser(
        description="Re-run only the post-processing of a completed IDA (no analyses).")
    parser.add_argument("config", nargs="?", default="config_ida_htf.py",
                        help="IDA config file (relative to this folder, or an absolute path)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    run(config_path)
