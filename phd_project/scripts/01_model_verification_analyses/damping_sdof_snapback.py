import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
 
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
 
# Initial displacement and velocity
u0 = 1
ops.setNodeDisp(2, 1, u0, '-commit')

wn2 = ops.eigen("-fullGenLapack", 1)
wn_calc = np.sqrt(wn2)
print(wn, wn_calc)
ops.modalDamping(ksi)

ops.analysis('Transient')

# Initial state
time = np.array(0.0)
disp = np.array(ops.nodeDisp(2,1))
vel  = np.array(ops.nodeVel(2,1))
acc  = np.array(ops.nodeAccel(2,1))
fs1 = np.array(-ops.eleForce(1, 1)) 

Tf = 10
dt = 0.01
Nsteps = int(Tf/dt)

for ii in range(Nsteps):
    ops.analyze(1,dt)
    time = np.append(time, ops.getTime())
    disp = np.append(disp, ops.nodeDisp(2,1))
    vel  = np.append(vel, ops.nodeVel(2,1))
    acc  = np.append(acc, ops.nodeAccel(2,1))
    fs1 = np.append(fs1, -ops.eleForce(1, 1))
print("Analysis Finished")

Fs = disp * k
Fd = vel * c
Fi = acc * m
Fd_calc = -Fs - Fi      # from equilibrium

plt.plot(time[1:], Fs[1:], label="Fs")
plt.plot(time[1:], Fd[1:], label= "Fd")
plt.plot(time[1:], Fi[1:], label="Fi")
plt.plot(time[1:], fs1[1:], label="F_spring")
plt.plot(time[1:], Fd_calc[1:], label="Fd_eq")
plt.legend()
plt.show()
