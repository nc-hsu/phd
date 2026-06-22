import pickle
from pathlib import Path

fp = Path("C:/Users/clemettn/Desktop/test_msa/site_0__stripe_2__gm_selection.pickle")
with open(fp, "rb") as file:
    gms = pickle.load(file)

pass