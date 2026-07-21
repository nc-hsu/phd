from standes.intensitymeasures import avgsa_06

damping_ratio = 0.05


def make_im(model_init):
    """Return the average-spectral-acceleration (0-6 s) intensity measure.

    `model_init` is accepted so every `config_im_*.py` presents the same
    `make_im(model_init)` contract (the IDA config selects the IM file by name);
    this measure has a fixed period range and does not use it.
    """
    return avgsa_06(damping_ratio=damping_ratio)
