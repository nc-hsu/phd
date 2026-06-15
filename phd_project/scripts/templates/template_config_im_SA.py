import openseespy.opensees as ops
from structural_model import model_init # type: ignore
from standes.intensitymeasures import SpectralAcceleration

model_init()
ops.wipeAnalysis()

nodes = ops.getNodeTags()
nodes_w_mass = [n for n in nodes if sum(ops.nodeMass(n)) > 0]

if len(nodes_w_mass) == 1:
    solver = "-fullGenLapack"
else:
    solver = "-genBandArpack"

ops.eigen(solver, 1)
T1 = ops.modalProperties("-return")["eigenPeriod"][0]
im = SpectralAcceleration(T1)
ops.wipe()