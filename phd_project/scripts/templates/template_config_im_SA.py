import openseespy.opensees as ops
from standes.intensitymeasures import SpectralAcceleration


def make_im(model_init):
    """Build the intensity measure from the structural model.

    The analysis config imports the structural model (by `model_file_name` /
    `init_function`) and passes its `model_init` in here, so this file does not
    import `structural_model` itself and is model-name agnostic.
    """
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
    return im
