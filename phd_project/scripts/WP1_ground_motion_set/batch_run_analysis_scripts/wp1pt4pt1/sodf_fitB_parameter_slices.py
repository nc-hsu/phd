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
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_0_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_50_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_10_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_15_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_20_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_25_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_30_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_35_d2_19/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_0/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_0/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_1/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_1/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_2/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_2/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_3/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_3/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_4/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_4/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_5/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_5/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_6/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_6/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_7/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_7/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_8/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_8/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_9/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_9/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_10/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_10/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_11/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_11/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_12/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_12/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_13/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_13/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_14/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_14/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_15/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_15/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_16/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_16/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_17/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_17/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_18/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_18/config_cyclic_pushover_1.py')]
    },
    {
        "script": Path("E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_19/run_cyclic_pushover.py"),
        "config": [Path('E:/02_wp1pt4pt1_po_analyses_for_sdof_fitting/fitB_optimisation/beta_40_d2_19/config_cyclic_pushover_1.py')]
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
                time.sleep(0.5)

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
        time.sleep(0.5)
    
    pbar.close()


if __name__ == "__main__":
    main()