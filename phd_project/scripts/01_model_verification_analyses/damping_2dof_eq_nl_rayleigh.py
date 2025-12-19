import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

from standes.groundmotion import load_ground_motion_from_json
from standes.analysis.damping import set_rayleigh_damping, rayleigh_alphaM, rayleigh_betaK
from standes.analysis.eigen import eigenvector_matrix, nodal_massmatrix
 
# Period and stiffness
m = 1    # Mass
k = 25   # Stiffness
print(k)
# damping
ksi = 0.05
 
ops.wipe()
ops.model('basic','-ndm',1)
 
ops.node(1,0) 
ops.fix(1,1)

ops.node(2,0) 
ops.mass(2,m)

ops.node(3,0)
ops.mass(3,m)
 
ops.uniaxialMaterial('ElasticPP', 1, 1.5*k, 0.2)
ops.uniaxialMaterial('ElasticPP', 2, k, 0.267)
ops.element('zeroLength', 1, 1, 2,'-mat', 1,'-dir', 1, "-doRayleigh", 1)
ops.element('zeroLength', 2, 2, 3,'-mat', 2,'-dir', 1, "-doRayleigh", 1)

coeffs = set_rayleigh_damping(1, 2, 0.05, solver="-fullGenLapack")
# print(coeffs)
# # extract the rayleigh damping matrix and eigenvectors
# ops.numberer('Plain')
# ops.system('FullGeneral')
# ops.analysis('Transient')
# ops.integrator("GimmeMCK", 0, 1, 0)
# ops.analyze(1, 0.0)
# N = ops.systemSize()
# C = np.array(ops.printA('-ret')).reshape((N,N))

# print("\nRayleigh Damping Matrix - From GimmeMCK")
# print(C)

# ops.integrator('GimmeMCK',0,0,1)
# ops.analyze(1,0.0)
# K = np.array(ops.printA('-ret')).reshape((N,N))
# print("\nStiffness Matrix - From GimmeMCK")
# print(K)

# ops.integrator('GimmeMCK',1,0,0)
# ops.analyze(1,0.0)
# M = np.array(ops.printA('-ret')).reshape((N,N))
# print("\nMass Matrix - From GimmeMCK")
# print(M)
 
# w2 = np.array(ops.eigen("-fullGenLapack", 2))
# wn = w2 ** 0.5
# PHI = eigenvector_matrix(2, N)
# print("\nEigenvectors")
# print(PHI)

# alphaM = rayleigh_alphaM(wn[0], wn[1], ksi)
# betaK = rayleigh_betaK(wn[0], wn[1], ksi)
# print("\nRayleigh Coefficients:")
# print(f"alpha_M = {alphaM:.4f}")
# print(f"beta_K = {betaK:.6f}")

# my_M = nodal_massmatrix(N)
# print("\nMass Matrix - my Calcs")
# print(my_M)

# my_C = alphaM * my_M + betaK * K
# print("\nRayleigh Damping Matrix - my Calcs")
# print(my_C)


# load record and create time series/pattern
gm = load_ground_motion_from_json("C:/Users/clemettn/OneDrive - Helmut-Schmidt-Universität/01_arbeit/14_PhD/data/fema_P695_records/fema_p695_120621.json")
ops.timeSeries("Path", 1, 
               "-values", *gm[1], 
               "-time", *gm[0], 
               "-factor", 9.81) # scale for putting record in correct units of gravity

ops.pattern("UniformExcitation", 1, 1, "-accel", 1, "-factor", 6)    # scale for im level

# Initial state
time = np.array(0.0)
u2 = np.array(ops.nodeDisp(2,1))
v2  = np.array(ops.nodeVel(2,1))
a2  = np.array(ops.nodeAccel(2,1))
fs1 = np.array(-ops.eleForce(1, 1))

u3 = np.array(ops.nodeDisp(3,1))
v3  = np.array(ops.nodeVel(3,1))
a3= np.array(ops.nodeAccel(3,1))
fs2 = np.array(-ops.eleForce(2, 1))

Tf = 60
dt = 0.01
Nsteps = int(Tf/dt)


# get the initial damping matrix and modeshape
ops.wipeAnalysis()
ops.numberer('RCM')
ops.system('FullGeneral')
ops.analysis('Transient')
ops.integrator("GimmeMCK", 0, 1, 0)
ops.analyze(1, 0.0)
N = ops.systemSize()
C_0 =np.array(ops.printA('-ret')).reshape((N,N))
PHI_0 = eigenvector_matrix(2, N)

Cs = [C_0]
PHIs = [PHI_0]

ops.wipeAnalysis()
ops.constraints("Plain")
ops.numberer('RCM')
ops.system("BandGen")
ops.test("NormDispIncr", 1e-6, 50)
ops.algorithm("Newton")
ops.integrator("Newmark", 0.5, 0.25)
ops.analysis('Transient')

for ii in range(Nsteps):
    ops.analyze(1,dt)

    # after each analysis step extract the damping matrix and the eigenvector matrix
    ## do the modal analysis
    ops.eigen("-fullGenLapack", 2)

    ## extract matrices
    ops.system('FullGeneral')
    ops.algorithm("Linear")
    ops.integrator("GimmeMCK", 0, 1, 0)
    ops.analyze(1, 0.0)
    N = ops.systemSize()
    Cs.append(np.array(ops.printA('-ret')).reshape((N,N)))
    PHIs.append(eigenvector_matrix(2, N))

    # revert to original system and integrator
    ops.system("BandGen")
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)

    time = np.append(time, ops.getTime())
    u2 = np.append(u2, ops.nodeDisp(2,1))
    v2  = np.append(v2, ops.nodeVel(2,1))
    a2  = np.append(a2, ops.nodeAccel(2,1) + ops.getLoadFactor(1))
    fs1 = np.append(fs1, -ops.eleForce(1, 1))
    
    u3 = np.append(u3, ops.nodeDisp(3,1))
    v3  = np.append(v3, ops.nodeVel(3,1))
    a3  = np.append(a3, ops.nodeAccel(3,1) + ops.getLoadFactor(1))
    fs2 = np.append(fs2, -ops.eleForce(2, 1))

print("Analysis Finished")

Fi_1 = (a2 * m)
Fi_2 = (a3 * m)
sum_Fi = Fi_1 + Fi_2
Fd_1 = -fs1 - sum_Fi      # from equilibrium
Fd_2 = -fs2 - Fi_2

plt.plot(time[1:], fs1[1:], label="Fs")
plt.plot(time[1:], Fd_1[1:], label= "Fd")
plt.plot(time[1:], sum_Fi[1:], label="Fi - total")
plt.legend()
plt.show()

print(f"Max Vs1 [kN]: {max(fs1)}")
print(f"Max Vd1 [kN]: {max(Fd_1)}")
print(f"Vd1 / Vs1   : {max(abs(Fd_1)) / max(abs(fs1))}")

dst = Path(__file__).parent
with open(dst / "2dof_rayleigh_damping_matrices.pickle", "wb") as file:
    pickle.dump(Cs, file)
with open(dst / "2dof_rayleigh_modal_matrices.pickle", "wb") as file:
    pickle.dump(PHIs, file)

pass