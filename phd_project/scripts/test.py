import openseespy.opensees as ops
import numpy as np

backbone = np.array(
    [[0, 0], 
     [81.05, 1.546e+05],        # mm and N
     [100.1, 1.615e+05], 
     [304.7, 0]])

mass = 151.8        # tonne

# Model parameters
support_node: int = 1
free_node: int = 2
ele_tag: int = 1

stiffness = backbone[1, 1] / backbone[1, 0]     # Ke = Fy / dy
period = 2 * np.pi * np.sqrt(mass / stiffness) 
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 3)

ops.node(support_node, 0, 0)
ops.node(free_node, 0, 0)

ops.fix(support_node, 1, 1, 1)
ops.fix(free_node, 0, 1, 1)

ops.mass(free_node, mass, 0, 0)


ops.uniaxialMaterial(
    "Hysteretic", 1,
    *[154600.0, 81.05],
    *[161500.0, 100.1],
    *[0.0, 307],
    *[-154600.0, -81.05],
    *[-161500.0, -100.1],
    *[0, -307],
    1.0, 1.0, 0.0, 0.0, 0.0
)

ops.element("zeroLength", ele_tag, support_node, free_node, 
            "-mat", 1, "-dir", 1)


pass