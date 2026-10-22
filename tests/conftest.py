"""Shared pytest configuration: the ``slow`` marker and the ``--runslow`` switch.

Some tests are Monte Carlo studies that take tens of minutes (the bootstrap
coverage studies of the hybrid bootstrap, H9). They are marked
``@pytest.mark.slow`` and SKIPPED by default, so the ordinary ``pytest`` run
stays the same length it was. Run them on purpose with::

    pytest --runslow -m slow -s

(``-s`` so the coverage tables they print are shown.)

Nothing here changes how any unmarked test is collected or run.
"""

import pytest


def pytest_addoption(parser):
    """Add ``--runslow``. Off by default, so slow tests never run by accident."""
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests marked slow (tens of minutes)")


def pytest_configure(config):
    """Register the marker, so ``@pytest.mark.slow`` is not an 'unknown mark' warning."""
    config.addinivalue_line(
        "markers", "slow: a long Monte Carlo study; skipped unless --runslow is given")


def pytest_collection_modifyitems(config, items):
    """Skip every ``slow`` test unless ``--runslow`` was passed.

    Skipped rather than deselected, so the run's summary still shows that they
    exist and were not run.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow Monte Carlo study: pass --runslow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
