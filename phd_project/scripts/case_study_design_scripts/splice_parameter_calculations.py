import context
from standes.geometries.connections.flushendplatesplice import FlushEndplateSplice
import eurocodedesign.geometry.steelsections as ss


strength_class = "10.9"
thread_in_shear = True
n_bolt_rows = 3
a_pf = 0 
a_pw = 0
f_y_ep = f_y_wb = f_y_fb = 235

section_name: str = "IPE400"
section: ss.ISection = ss.get(section_name)
t_ep: float = 16 # mm               # must conform to Eq. E.9 in ec8-1-1:2023 -> tp <= 0,3 * d_bolt * sqrt(fub/fyp)
bolt_size: str = "M27"
e_ep: float = 40
p: float = (section.h - 2 * section.t_f - 30) / (n_bolt_rows - 1)
splice = FlushEndplateSplice(section, t_ep, bolt_size, strength_class, thread_in_shear,
                             n_bolt_rows, p, e_ep, a_pf, a_pw, f_y_ep, f_y_wb, f_y_fb)

N_j_t_Rx = splice.N_j_t_Rd()
M_j_Rx = splice.M_j_Rx()

print(f"N_jtRd = {N_j_t_Rx / 1000:.2f} kN")
print(f"M_jRx = {M_j_Rx / 1e6:.2f} kNm")
print(f"Governing Moment Mechanism: {splice.governing_mechanism()}")