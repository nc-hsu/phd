"""Cyclic-pushover displacement-step (``dU``) resolution.

Some OpenSees cyclic-pushover models need a finer displacement increment than the
default to converge. Those tuned values live inside each folder's
``config_cyclic_pushover.py`` and are easy to destroy by re-running the notebook cell
that builds the analysis folders. The helpers here let a folder be rebuilt without
clobbering a tuned ``dU``:

    resolve_cpo_du precedence:  explicit override  >  tuned value on disk  >  default

Used by ``010_cbfs_reference_designs_ec8_gen2_and_analyses`` and
``011-create_site_casestudies_and_analyses``.
"""

import re
from pathlib import Path

# Only the indented entry inside `config = {...}` counts -- the config template also
# mentions "dU" at column 0 inside its explanatory docstring, which must be ignored.
# The character class includes a literal tab so tab-indented configs match too.
_DU_RE = re.compile(r'^[ \t]+"dU"\s*:\s*([0-9.eE+-]+)', re.MULTILINE)


def existing_cpo_du(config_path):
    """dU currently written in a config_cyclic_pushover.py, or None if absent."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    m = _DU_RE.search(config_path.read_text())
    return float(m.group(1)) if m else None


def resolve_cpo_du(key, config_path, overrides, default, preserve_tuned=True):
    """dU to write for this config: explicit override > tuned value on disk > default.

    key        : identifier looked up in ``overrides`` (e.g. "site_12/mdof").
    config_path: the config_cyclic_pushover.py that would be (over)written.
    overrides  : {key: dU} of explicit choices.
    default    : dU to use when there is no override and nothing tuned on disk.
    preserve_tuned: if True, keep a dU already present on disk over the default.
    """
    if key in overrides:
        return overrides[key]
    if preserve_tuned:
        current = existing_cpo_du(config_path)
        if current is not None:
            return current
    return default


def audit_cpo_du(items, default, overrides):
    """Report configs whose on-disk dU differs from what would be written.

    items: iterable of (key, config_path).
    Returns {key: current_dU} for every existing config whose dU differs from the
    value ``overrides`` / ``default`` would otherwise write.
    """
    divergent = {}
    for key, config_path in items:
        current = existing_cpo_du(config_path)
        intended = overrides.get(key, default)
        if current is not None and current != intended:
            divergent[key] = current
    return divergent
