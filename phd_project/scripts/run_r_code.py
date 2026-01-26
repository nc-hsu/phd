import subprocess
from pathlib import Path
from phd_project.config.config import load_config

# Load your centralized configuration
cfg = load_config()

def run_r_script(script_path, additional_params: list=[]):
    """
    Launches an R script via the command line.

    Example Usage in your notebook:
    execute_r_script("seismic_analysis_v2.R")
    df = pd.read_csv("r_output_data.csv")
    
    Args:
        script_path (Path): A pathlib.Path object pointing to the .R file.
        additional_params(list): List of strings that are passed to the R script
            as parameters
    """

    # 1. Validation and existence check
    if not isinstance(script_path, Path):
        print("Error: script_path must be a pathlib.Path object.")
        return

    if not script_path.exists():
        print(f"Error: The file at {script_path} was not found.")
        return
    
    r_exe = cfg['executables']['rscript'] # Access the Rscript executable via path in config the config file
    
    if not r_exe.exists():
        raise FileNotFoundError(f"Rscript not found at {r_exe}. Check phd_project/config/config.yaml")

    # 2. Resolve to absolute path for robustness
    # This ensures R finds the script even if the working directory changes
    abs_path = script_path.resolve()
    
    # 3. Set up the command
    command = [str(r_exe), str(abs_path)] + additional_params

    print(f"Executing R script: {abs_path.name}...")

    try:
        # 4. Execute
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print("--- R Standard Output ---")
            print(result.stdout)

        print(f"R script '{abs_path.name}' completed successfully.")
            
    except subprocess.CalledProcessError as e:
        print(f"Execution failed with exit code {e.returncode}")
        print("--- R Error Output ---")
        print(e.stderr)



if __name__ == "__main__":
    r_script_fp = cfg["scripts"]["residual_partitioning"] / "residual_partitioning_lmer.R"

    folder = str(cfg['data']['residuals']) 
    models = ",".join(["asc", "sslab", "sinter", "vran"])
    tails = [
        "_total_residuals.csv",
        "_periods.csv",
        "_partitioned_residuals_lmer.csv",
        "_partitioned_residuals_summary_lmer.json"
    ]
    additional_parameters = [folder, models] +  tails + [str(3)]
    run_r_script(r_script_fp, additional_parameters)
