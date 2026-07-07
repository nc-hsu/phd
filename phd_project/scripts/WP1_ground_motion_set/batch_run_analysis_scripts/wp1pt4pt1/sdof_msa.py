import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from phd_project.process_semaphore.process_semaphore import (
    acquire_slot, release_slot, get_current_running, get_max_concurrent)

running_processes = []  # List of dicts: {proc, script, config, start_time}

# Paths to your scripts and the configurations
scripts_and_configs = [
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_0/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_0/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_0/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_1/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_1/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_1/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_2/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_2/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_2/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_3/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_3/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_3/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_4/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_4/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_4/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_5/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_5/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_5/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_6/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_6/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_6/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_7/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_7/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_7/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_8/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_8/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_8/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_9/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_9/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_9/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_10/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_10/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_10/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_11/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_11/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_11/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_12/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_12/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_12/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_13/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_13/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_13/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_14/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_14/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_14/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_15/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_15/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_15/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_16/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_16/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_16/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_17/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_17/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_17/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_18/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_18/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_18/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_19/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_19/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_19/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_20/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_20/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_20/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_21/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_21/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_21/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_22/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_22/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_22/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_23/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_23/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_23/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_24/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_24/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_24/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_25/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_25/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_25/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_26/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_26/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_26/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_27/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_27/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_27/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_28/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_28/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_28/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_29/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_29/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_29/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_30/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_30/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_30/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_31/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_31/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_31/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_32/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_32/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_32/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_33/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_33/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_33/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_34/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_34/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_34/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_35/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_35/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_35/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_36/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_36/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_36/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_37/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_37/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_37/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_38/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_38/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_38/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_39/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_39/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_39/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_40/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_40/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_40/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_41/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_41/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_41/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_42/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_42/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_42/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_43/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_43/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_43/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_44/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_44/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_44/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_45/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_45/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_45/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_46/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_46/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_46/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_47/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_47/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_47/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_48/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_48/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_48/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_49/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_49/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_49/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_50/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_50/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_50/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_51/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_51/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_51/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_52/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_52/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_52/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_53/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_53/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_53/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_54/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_54/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_54/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_55/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_55/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_55/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_56/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_56/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_56/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_57/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_57/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_57/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_58/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_58/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_58/SDOF_param/config_msa_AvgSA_06.py')]
    },
    {
        "script": Path("C:/Users/clemettn/Desktop/test_folder/site_59/SDOF_param/run_msa_site.py"),
        "config": [Path('C:/Users/clemettn/Desktop/test_folder/site_59/SDOF_param/config_msa_AvgSA_03.py'), Path('C:/Users/clemettn/Desktop/test_folder/site_59/SDOF_param/config_msa_AvgSA_06.py')]
    }
]


def launch_script(script_path: Path, config_path: Path):
    SW_SHOWMINNOACTIVE = 7
    venv_python = Path(sys.executable)

    window_title = f"{script_path.parts[-2]}/{script_path.name} [{config_path.name}]"

    py_code = (
        f"import importlib.util; "
        f"path = r'''{script_path}'''; "
        f"spec = importlib.util.spec_from_file_location('mod', path); "
        f"mod = importlib.util.module_from_spec(spec); "
        f"spec.loader.exec_module(mod); "
        f"mod.run(r'''{config_path}''')"
    )

    ps_command = (
        f"$host.UI.RawUI.WindowTitle = '{window_title}'; "
        f"& '{venv_python}' -c \"{py_code}\"; "
        f"if ($LASTEXITCODE -ne 0) {{ pause }}"
    )

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_SHOWMINNOACTIVE 

    proc = subprocess.Popen(
        ["pwsh", "-NoLogo", "-NoProfile", "-Command", ps_command],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startupinfo
    )

    return proc, window_title


def check_for_finished(pbar):
    """Checks for finished processes, releases slots, and updates tqdm."""
    global running_processes
    still_running = []
    for entry in running_processes:
        if entry["proc"].poll() is not None:
            elapsed = datetime.now() - entry["start_time"]
            pbar.write(f"✅ Finished: {entry['display_name']} in {elapsed.total_seconds():.1f}s")
            release_slot(entry["proc"].pid)
            pbar.update(1)
        else:
            still_running.append(entry)
    running_processes = still_running

def main():
    total_scripts = sum(len(entry["config"]) for entry in scripts_and_configs)
    pbar = tqdm(total=total_scripts, desc="Batch Progress", unit="script")

    for entry in scripts_and_configs:
        script, config_paths = entry["script"], entry["config"]
        
        for cfg_path in config_paths:
            # ACTIVE WAITING: Check for finished jobs while waiting for a slot
            while True:
                check_for_finished(pbar) # This ensures the bar moves during the launch phase
                if len(get_current_running()) < get_max_concurrent():
                    break
                time.sleep(0.1)

            proc, display_name = launch_script(script, cfg_path)
            acquire_slot(proc.pid)
            
            pbar.write(f"▶️ Launched: {display_name}")
            running_processes.append({
                "proc": proc,
                "display_name": display_name,
                "start_time": datetime.now()
            })

    # FINAL WAIT: For the last remaining batch
    while running_processes:
        check_for_finished(pbar)
        time.sleep(0.1)
    
    pbar.close()


if __name__ == "__main__":
    main()