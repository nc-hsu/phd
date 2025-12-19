import pickle
from pathlib import Path

filename = "3s_cbf_dc2_41_out.pickle"
folder = Path("D:/case_studies_set1_dc2/3s_cbf_dc2_41")

with open(folder / filename, "rb") as file:
    model = pickle.load(file)

pass