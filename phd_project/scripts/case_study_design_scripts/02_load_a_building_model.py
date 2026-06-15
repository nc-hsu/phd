import pickle
from pathlib import Path

filename = "3s_cbf_dc2_10_out.pickle"
folder = Path(r"C:\Users\clemettn\Documents\phd\casestudy_structures\concentrically_braced_frames\ec8_gen2\3s_cbf_dc2_10")

with open(folder / filename, "rb") as file:
    model = pickle.load(file)

pass