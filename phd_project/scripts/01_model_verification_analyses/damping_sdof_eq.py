import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
from standes.groundmotion import load_ground_motion_from_json
 
# Period and stiffness
m = 1 # Mass
Tn = 2
wn = 2 * np.pi / Tn
k = m*(2*3.14159/Tn)**2

# damping
ksi = 0.05
c = 2 * m * wn * ksi
 
ops.wipe()
ops.model('basic','-ndm',1)
 
ops.node(1,0); 
ops.fix(1,1)

ops.node(2,0); 
ops.mass(2,m)
 
ops.uniaxialMaterial('Elastic', 1, k)
ops.element('zeroLength', 1, 1, 2,'-mat', 1,'-dir', 1)
 
# modal analysis for damping
ops.eigen("-fullGenLapack", 1)
modal_props = ops.modalProperties("-return")
print(modal_props["eigenPeriod"])
ops.modalDamping(ksi)

# load record and create time series/pattern
gm = load_ground_motion_from_json("C:/Users/clemettn/OneDrive - Helmut-Schmidt-Universität/01_arbeit/14_PhD/data/fema_P695_records/fema_p695_120621.json")
ops.timeSeries("Path", 1, 
               "-values", *gm[1], 
               "-time", *gm[0], 
               "-factor", 9.81) # scale for putting record in correct units of gravity

ops.pattern("UniformExcitation", 1, 1, "-accel", 1, "-factor", 2)    # scale for im level

# Initial state
time = np.array(0.0)
disp = np.array(ops.nodeDisp(2,1))
vel  = np.array(ops.nodeVel(2,1))
acc  = np.array(ops.nodeAccel(2,1))
abs_acc = np.array(ops.nodeAccel(2,1))
fs1 = np.array(-ops.eleForce(1, 1)) 

Tf = 10
dt = 0.01
Nsteps = int(Tf/dt)

ops.analysis('Transient')
for ii in range(Nsteps):
    ops.analyze(1,dt)
    time = np.append(time, ops.getTime())
    disp = np.append(disp, ops.nodeDisp(2,1))
    vel  = np.append(vel, ops.nodeVel(2,1))
    acc  = np.append(acc, ops.nodeAccel(2,1))
    abs_acc  = np.append(abs_acc, ops.nodeAccel(2,1) + ops.getLoadFactor(1))
    fs1 = np.append(fs1, -ops.eleForce(1, 1))
print("Analysis Finished")

Fs = disp * k
Fd = vel * c
Fi = abs_acc * m
Fd_calc = -Fs - Fi      # from equilibrium

plt.plot(time[1:], Fs[1:], label="Fs")
plt.plot(time[1:], Fd[1:], label= "Fd")
plt.plot(time[1:], Fi[1:], label="Fi")
plt.plot(time[1:], fs1[1:], label="F_spring")
plt.plot(time[1:], Fd_calc[1:], label="Fd_eq")
plt.legend()
plt.show()
