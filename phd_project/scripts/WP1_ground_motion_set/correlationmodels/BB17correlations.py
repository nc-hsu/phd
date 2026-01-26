"""
Correlation Model from:
Baker and Bradley (2017).
"""
import re
import pandas as pd
from phd_project.config.config import load_config
cfg = load_config()

def extract_floats_from_brackets(text):
    # Regex breakdown:
    # \(      : Matches literal opening bracket
    # (       : Starts the capturing group
    # [-+]?   : Optional plus or minus sign
    # \d* : Zero or more digits
    # \.?     : Optional decimal point
    # \d+     : One or more digits
    # )       : Ends the capturing group
    # \)      : Matches literal closing bracket
    pattern = r"\(([-+]?\d*\.?\d+)\)"
    
    # findall returns a list of all strings that match the capturing group
    match = re.search(pattern, text)
    
    # Convert the resulting strings to actual floats
    return float(match.group(1)) if match else "None"


def extract_text_before_brackets(text):
    # Regex breakdown:
    # \w      : Matches alpha numeric characters
    # \.?     : Optional decimal point
    # \w      : Matches alpha numeric characters
    # \(      : Matches literal opening bracket
    pattern = r"^[^(\n]+"
    # findall returns a list of all strings that match the capturing group
    match = re.search(pattern, text)
    
    # Convert the resulting strings to actual floats
    return match.group(0).strip() if match else text


def bb17_empirical_correlation_model():
    # load the baker and bradley 2017 correlation model
    bb17 = pd.read_csv(cfg["data"]["rho_models"] / "BB17/rhoDataPD_corrected.csv", sep=",",
                    index_col=0)
    labels = [("rotD50",
            extract_text_before_brackets(c),
            extract_floats_from_brackets(c)) for c in bb17.columns]
    bb17.index = pd.MultiIndex.from_tuples(labels)
    bb17.columns = pd.MultiIndex.from_tuples(labels)
    return bb17