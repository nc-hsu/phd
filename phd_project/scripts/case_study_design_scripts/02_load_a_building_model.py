import pickle
from pathlib import Path

filename = "3s_cbf_dc2_site15_out.pickle"
folder = Path(r"C:\Users\clemettn\Documents\phd\casestudy_structures\concentrically_braced_frames\ec8_gen2_site_specific_designs\3s_cbf_dc2_site15")

with open(folder / filename, "rb") as file:
    model = pickle.load(file)

pass