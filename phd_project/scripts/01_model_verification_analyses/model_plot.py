from pathlib import Path
import opsvis as opsv
import matplotlib.pyplot as plt
from standes.opsmodels.nlcbf_3d_01 import build_model_nlcbf_3d_01

file = Path(r"E:\01_model_verification_analyses\diaphragm_models\model_02\3s_cbf_dc2_41_out.json")
build_model_nlcbf_3d_01(file)
opsv.plot_model(node_labels=False, element_labels=False, az_el=(-90,0))
plt.show()