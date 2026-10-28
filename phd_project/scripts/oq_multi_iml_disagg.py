"""Multi-IML OpenQuake disaggregation: one classical parent, cheap children.

Why this module exists
----------------------
OpenQuake's stock disaggregation can take **one** intensity level per IMT for
every site (``iml_disagg``), or levels read off each site's hazard curve at fixed
PoEs (``poes_disagg``). The WP1 work needs neither: it needs *many* levels per
site, chosen by us (nb 020: 38 target IMLs at every site; nb 065: each row's
θ_IDA; nb 072: each site's return-period levels) - and every level twice, at
``x`` and ``x + δx``, because the occurrence disaggregation is a difference of
two exceedance disaggregations (Bradley 2010, EESD 39:1321-1342, Eq. 7, p. 1326):

    P(m | X = x) ≈ [ν_m(x) - ν_m(x+δx)] / [ν(x) - ν(x+δx)],   ν(x) = Σ_m ν_m(x)

where ``m`` is one TRT x Mag x Dist x Eps bin and ``ν_m(x)`` is that bin's annual
rate of exceeding ``x`` (the ``nu_m`` column).

How it works (plain English)
----------------------------
The expensive part of a disaggregation is finding which ruptures can reach each
site and how strongly (distances, GMM mean and sigma). None of that depends on
the intensity level we ask about. The level only enters at the very end, in
``P(IM > x | rup) = 1 - Φ((ln x - μ) / σ)``. So:

1. A **parent** classical calculation runs once, with ``max_sites_disagg = 60``
   so the engine stores those rupture "contexts" (``base.py:548-553``).
2. A **child** disaggregation (``hazard_calculation_id = parent``) reads them
   back and evaluates any number of levels per site. The engine already loops
   over a per-site level array ``hmap3`` of shape (N sites, M IMTs, P levels)
   (``disaggregation.py:81-87``); the child ini sets ``poes_disagg`` only as a
   *placeholder* that fixes P, and :class:`MultiIMLDisaggCalculator` replaces
   ``hmap3`` with our own levels before any task reads it.

The child keeps the parent's 25-level IMTL grid, so the engine's realisation
ranking (``best_rlzs``, ``disaggregation.py:198-216``) compares like with like -
the mismatch that stopped nb 020 from ever chaining off the PSHA.

The overwrite touches only the **child's own new datastore**, before the
disaggregation tasks start, so the finished file's ``hmap3`` is exactly the set
of levels that was disaggregated. Parent and reference datastores are never
written.

Contents
--------
- calculator: :func:`calculator_class` / :func:`run_child_calculation`
- configs:    :func:`write_parent_config`, :func:`write_child_config`
- levels:     :func:`build_levels_table`, :func:`levels_array`,
              :func:`levels_at_poes`, :func:`split_targets`,
              :func:`write_levels_files`, :func:`read_levels_table`
- running:    :func:`run_or_reuse_child` (+ the ``run`` CLI subcommand)
- reading:    :func:`read_levels`, :func:`occurrence_disagg`,
              :func:`collate_targets`, :func:`merge_collations`
- checks:     :func:`selftest_occurrence`, :func:`check_site_order`,
              :func:`verify_against_reference`, :func:`check_total_probability`,
              :func:`check_independence`, :func:`compare_processed_shards`
              (+ the ``verify`` CLI subcommand)

Engine version
--------------
Written against the local fork ``pypackages/oq-engine-nc`` (imports as 3.27.0;
its dist-info says 3.25.0). The subclass relies on two private behaviours of
``DisaggregationCalculator``: ``full_disaggregation`` writes ``hmap3`` and then
calls ``compute()``, and ``compute_disagg`` reads ``hmap3`` per site. Both the
engine version and the fork commit are pinned in every manifest entry this
module writes, and the regression checks fail loudly if either behaviour
changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from . import oq_runner, oqhelpers
from .cache_utils import _diff_inputs, _git_commit, fingerprint

# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #

#: The disaggregation output every WP1 product uses.
DISAGG_TYPE = "TRT_Mag_Dist_Eps"

#: Relative step for the occurrence pair: x and x·(1 + δ). Bradley (2010) Eq. 7
#: uses a forward difference; 0.001 is the value the old (buggy) code used, kept
#: so that nothing but the formula changes (plan (1), D1).
DELTA_DEFAULT = 0.001

#: Roles a level can play inside one (site, key) target.
ROLE_X, ROLE_XDX, ROLE_PAD = "x", "x+dx", "pad"

#: The bin columns of a TRT_Mag_Dist_Eps frame, plus the realisation index Z.
BIN_COLS = ["TRT", "Mag", "Dist", "Eps", "Z"]

#: Printed by the ``run`` subcommand so the parent process can read the calc id
#: without scraping engine log lines.
_CALC_ID_TAG = "MULTI_IML_DISAGG_CALC_ID="


# --------------------------------------------------------------------------- #
# engine provenance                                                            #
# --------------------------------------------------------------------------- #

def _read_git_head(repo_root: Path) -> str:
    """Commit hash of a git checkout, read from ``.git`` *files* (no git command).

    Handles a plain ``.git`` directory, a detached HEAD, a branch ref stored as
    a loose file, and one stored only in ``packed-refs``. Returns ``"unknown"``
    if anything is missing - provenance is best effort, never a reason to fail.
    """
    git = Path(repo_root) / ".git"
    try:
        if git.is_file():  # a worktree: ".git" holds "gitdir: <path>"
            git = Path(git.read_text().split(":", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD: the hash itself
        ref = head.split(":", 1)[1].strip()
        loose = git / ref
        if loose.is_file():
            return loose.read_text().strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
    except Exception:
        pass
    return "unknown"


def engine_info() -> dict:
    """Version, dist-info version and fork commit of the imported engine.

    The fork reports 3.27.0 from ``openquake.baselib.__version__`` while its
    dist-info still says 3.25.0, so both are recorded: they answer different
    questions (what code ran vs what pip thinks is installed).
    """
    import openquake.baselib as oq_baselib

    try:
        from importlib.metadata import version
        dist = version("openquake.engine")
    except Exception:
        dist = "unknown"
    oq_version = oq_baselib.__version__
    # `openquake` itself is a namespace package (no __file__), so locate the
    # checkout from openquake/baselib/__init__.py: three levels up is the repo root.
    root = Path(oq_baselib.__file__).resolve().parent.parent.parent
    return {"engine_version": oq_version, "dist_version": dist,
            "engine_commit": _read_git_head(root), "engine_root": str(root)}


# --------------------------------------------------------------------------- #
# the calculator                                                               #
# --------------------------------------------------------------------------- #

_CALCULATOR_CLASS = None


class _GettersBySid(list):
    """The engine's list of hazard-curve readers, made indexable by site id.

    Why this exists (engine bug, oq-engine-nc b8f3077961):
    a classical job stores its hazard rates in ``n`` site *chunks*; site ``sid``
    goes to chunk ``sid % n`` (``map_array.gen_chunks``). ``getters.map_getters``
    returns one reader (``MapGetter``) per chunk, holding the sites of that chunk
    in ``.sids``. ``n`` comes from ``getters.get_num_chunks``, read from the
    *parent's* oqparam:

    - if the job is itself a disaggregation, or N < concurrent_tasks // 2,
      ``n = N`` -> one chunk per site, and reader k holds exactly site k;
    - otherwise (our 60-site *classical* parent on the analysis machine)
      ``n = concurrent_tasks // 2`` (or the memory-based tile count) < N.

    ``DisaggregationCalculator.full_disaggregation`` then picks the reader for
    each site as ``self.mgetters[sid]`` (``disaggregation.py:205``), which is only
    right in the first case. In the second it reads the wrong chunk for sites
    ``0..n-1`` and raises ``IndexError`` at ``sid = n``.

    This list keeps the readers in their original order (so iteration, ``len``
    and slicing behave as the engine expects) but answers an integer index with
    the reader whose ``.sids`` actually contains that site. It is built from the
    readers' own site lists rather than from ``sid % n``, so it does not depend
    on the chunking rule, and it refuses a site that is missing or in two chunks.
    """

    def __init__(self, getters):
        super().__init__(getters)
        self._by_sid = {}
        for getter in getters:
            for sid in getter.sids:
                sid = int(sid)
                if sid in self._by_sid:
                    raise ValueError(f"site {sid} is in more than one hazard chunk")
                self._by_sid[sid] = getter

    def __getitem__(self, key):
        # integer index = site id (the only way disaggregation.py indexes it);
        # slices fall through to ordinary list behaviour
        if isinstance(key, (int, np.integer)):
            try:
                return self._by_sid[int(key)]
            except KeyError:
                raise IndexError(f"no hazard chunk holds site {int(key)}") from None
        return super().__getitem__(key)

    def check_covers(self, sids):
        """Raise unless every site of the calculation has exactly one reader."""
        missing = sorted(set(int(s) for s in sids) - set(self._by_sid))
        if missing:
            raise ValueError(f"sites {missing[:10]} are in no hazard chunk")


def calculator_class():
    """Return :class:`MultiIMLDisaggCalculator`, defining it on first use.

    Defined lazily so that importing this module (e.g. for the occurrence
    function, the level tables or the notebook helpers on the laptop) does not
    import the OpenQuake calculators. The class is **not** registered in
    ``openquake.calculators.base.calculators``: registering it would replace the
    stock ``disaggregation`` calculator for the whole process. It is only ever
    instantiated directly, by :func:`run_child_calculation`.
    """
    global _CALCULATOR_CLASS
    if _CALCULATOR_CLASS is not None:
        return _CALCULATOR_CLASS

    from openquake.calculators.disaggregation import DisaggregationCalculator

    class MultiIMLDisaggCalculator(DisaggregationCalculator):
        """Stock disaggregation with user-supplied levels per site.

        Set :attr:`iml_levels` (shape (N, M, P), in **g**, not ln) and, for the
        provenance attributes, :attr:`levels_sha256` before calling ``run()``.

        Everything else is the stock calculator: the bins, the realisation
        ranking, the per-site tasks and the result saving are untouched.
        """

        iml_levels: np.ndarray | None = None
        levels_sha256: str = "unknown"

        def full_disaggregation(self):
            """Stock ``full_disaggregation``, with the per-site reader lookup fixed.

            The stock method builds ``self.mgetters`` from
            ``getters.map_getters(...)`` and indexes it by site id, which fails
            when the classical parent stored fewer chunks than sites (see
            :class:`_GettersBySid`). The engine source is not edited. Instead
            ``getters.map_getters`` is swapped, in this process, for a wrapper
            that returns the same readers in a :class:`_GettersBySid`. The
            wrapper is one-shot: on its first call it puts the original
            function back, so only the single call at ``disaggregation.py:196``
            is affected. The ``finally`` restores it even if the stock method
            raises before reaching that call.

            The readers, the hazard curves they return and the realisation
            ranking are all stock; only which reader serves which site changes.
            """
            from openquake.calculators import getters as oq_getters

            original = oq_getters.map_getters
            sids = self.sitecol.sids

            def map_getters_by_sid(*args, **kwargs):
                oq_getters.map_getters = original      # one-shot: undo at once
                wrapped = _GettersBySid(original(*args, **kwargs))
                wrapped.check_covers(sids)
                return wrapped

            oq_getters.map_getters = map_getters_by_sid
            try:
                return super().full_disaggregation()
            finally:
                oq_getters.map_getters = original

        def compute(self):
            """Replace the placeholder ``hmap3`` with :attr:`iml_levels`, then
            run the stock task submission.

            ``full_disaggregation`` has just written ``hmap3`` from the
            placeholder ``poes_disagg``; no task has started, so nothing has
            read it yet (``disaggregation.py:241-245``).
            """
            oq = self.oqparam
            if self.iml_levels is None:
                raise RuntimeError("iml_levels was not set on the calculator")
            levels = np.asarray(self.iml_levels, dtype=float)

            # --- shape: P is fixed by the number of placeholder poes_disagg,
            # N by the parent's site collection, M by the IMTs. A mismatch here
            # means the ini and the levels array were built from different
            # tables.
            placeholder = self.datastore["hmap3"][()]
            if placeholder.shape != levels.shape:
                raise ValueError(
                    f"levels shape {levels.shape} != engine hmap3 shape "
                    f"{placeholder.shape} (N sites, M IMTs, P poes_disagg)")

            # --- values: a 0 makes compute_disagg skip the whole site silently
            # (`if iml2.sum() == 0: continue`, disaggregation.py:83), a NaN or
            # negative value gives nonsense. Refuse both.
            if not np.all(np.isfinite(levels)) or np.any(levels <= 0):
                raise ValueError("every level must be finite and > 0 (in g)")

            # --- parent/child consistency. The engine skips check_imtls for
            # mode 'disaggregation' (base.py:785 tests the string 'disagg'), so
            # nothing else confirms that the contexts and cmakers read from the
            # parent were built for the same IMTs and truncation level.
            parent = self.datastore.parent
            if not parent:
                raise RuntimeError(
                    "MultiIMLDisaggCalculator must run as a child "
                    "(hazard_calculation_id = <classical parent>)")
            poq = parent["oqparam"]
            if list(poq.imtls) != list(oq.imtls):
                raise ValueError(f"child IMTs {list(oq.imtls)} != parent "
                                 f"IMTs {list(poq.imtls)}")
            for imt in oq.imtls:
                if not np.array_equal(np.asarray(poq.imtls[imt]),
                                      np.asarray(oq.imtls[imt])):
                    raise ValueError(
                        f"child IMTL grid for {imt} differs from the parent's; "
                        f"best_rlzs would rank realisations on different curves")
            if poq.truncation_level != oq.truncation_level:
                raise ValueError(
                    f"child truncation_level {oq.truncation_level} != parent "
                    f"{poq.truncation_level}")

            # --- the overwrite: in the child's own datastore only. The
            # datastore's __setitem__ deletes and recreates the dataset.
            self.datastore["hmap3"] = levels
            attrs = self.datastore.hdf5["hmap3"].attrs
            attrs["levels_source"] = "MultiIMLDisaggCalculator"
            attrs["levels_sha256"] = self.levels_sha256
            # the import name, not __name__ (which is "__main__" under the CLI)
            attrs["levels_module"] = "phd_project.scripts.oq_multi_iml_disagg"
            attrs["note"] = ("oqparam.poes_disagg holds placeholders that only "
                             "set P; the disaggregated levels are this array")
            self.datastore.flush()
            return super().compute()

    _CALCULATOR_CLASS = MultiIMLDisaggCalculator
    return _CALCULATOR_CLASS


def run_child_calculation(ini_fp: Path, levels_npy: Path,
                          parent_calc_id: int) -> int:
    """Run one child disaggregation in this process and return its calc id.

    Mirrors ``openquake.calculators.base.run_calc`` (``base.py:1910-1924``),
    except that the calculator is :class:`MultiIMLDisaggCalculator` instead of
    the registered ``disaggregation`` class. ``logs.init`` creates the job
    record in the engine database (so the calc shows in
    ``oq engine --list-hazard-calculations``) and sets up logging.

    Must be called with the working directory set to the ini's folder: every
    path inside the ini is relative to it.
    """
    from openquake.commonlib import logs

    levels = np.load(levels_npy)
    levels_sha = hashlib.sha256(Path(levels_npy).read_bytes()).hexdigest()
    with logs.init(str(ini_fp), hc_id=int(parent_calc_id)) as log:
        oq = log.get_oqparam()
        calc = calculator_class()(oq, log.calc_id)
        calc.iml_levels = levels
        calc.levels_sha256 = levels_sha
        calc.run()
        calc_id = int(log.calc_id)
    return calc_id


# --------------------------------------------------------------------------- #
# configs                                                                      #
# --------------------------------------------------------------------------- #

def parent_description(im: str = "03", eps: int = 4) -> str:
    """Engine description of the classical parent. Front-loaded and unique, so
    ``oq_runner._calc_id_from_listing`` can tell it apart from the PSHA runs."""
    return f"[disagg-parent AvgSA{im} eps{eps} - ESHM20 all sites]"


def write_parent_config(out_fp: Path, *, im: str = "03",
                        truncation_level: int = 4,
                        max_sites_disagg: int = 60) -> Path:
    """Write the classical parent ini.

    It is nb 003's ``config_AvgSA_{im}_psha_eps{eps}.ini`` (same template, same
    model, same 25-level IMTL grid) with two changes:

    - a new ``description`` (so the calc id can be recovered from the listing);
    - ``max_sites_disagg = 60``, which makes the classical step store the
      rupture contexts the children read back (``base.py:548-553``: contexts are
      stored only when the number of sites is <= max_sites_disagg).
    """
    extra = (
        "\n[disaggregation]\n"
        "# Store the rupture contexts (rup/) so that disaggregation children\n"
        "# (phd_project/scripts/oq_multi_iml_disagg.py) can reuse them.\n"
        f"max_sites_disagg = {int(max_sites_disagg)}\n"
    )
    return oq_runner.write_config(
        out_fp,
        description=parent_description(im, truncation_level),
        gsim_logic_tree_file=oq_runner.GMPE_LOGIC_TREES[im],
        truncation_level=truncation_level,
        im_upper=oq_runner.im_upper(im),
        generated_by="phd_project/scripts/oq_multi_iml_disagg.py (write_parent_config)",
        extra=extra,
    )


# The ERF, site and logic-tree lines are copied verbatim from the parent
# template (oq_runner._CONFIG_TEMPLATE) rather than left to be inherited, so the
# ini reads standalone and oq_runner.fingerprint_calc finds the three model
# files it hashes. The child's IMTL grid MUST equal the parent's (asserted by the
# calculator). The hazard statistics are written explicitly too: the engine
# raises if the child's stats differ from the parent's (base.py:803-807), and
# writing them removes any doubt about what is inherited.
_CHILD_CONFIG_TEMPLATE = Template("""\
# Disaggregation CHILD of the classical parent calc $parent_calc_id ($parent_name).
#
# The poes_disagg values below are PLACEHOLDERS. They only set the number of
# levels per site (P = $n_levels). MultiIMLDisaggCalculator replaces the
# engine's hmap3 with the levels in $levels_file before any task runs.
#
# * Do NOT run this file with `oq engine --run`: the stock calculator would
#   disaggregate at the placeholder PoEs.
# * Do NOT read the result with `extract`'s iml-based paths, which show the
#   placeholders. Read hmap3 and disagg-* from the datastore.
#
# GENERATED by phd_project/scripts/oq_multi_iml_disagg.py (write_child_config)
# Do not edit by hand.

[general]
description = $description
calculation_mode = disaggregation
random_seed = 23
hazard_calculation_id = $parent_calc_id

[geometry]

[logic_tree]

[erf]
rupture_mesh_spacing = 5
complex_fault_mesh_spacing = 50
width_of_mfd_bin = 0.1
area_source_discretization = 10.0

pointsource_distance = 75
ps_grid_spacing = 50

[site_params]
reference_vs30_type = measured
reference_vs30_value = 800.0
site_model_file = $site_model_file

reference_depth_to_1pt0km_per_sec = 30.0

[calculation]
source_model_logic_tree_file = $source_model_logic_tree_file
gsim_logic_tree_file = $gsim_logic_tree_file

investigation_time = 1.0

intensity_measure_types_and_levels = {"AvgSA": logscale(0.0005, 3.00, 25)}

truncation_level = $truncation_level
maximum_distance = {'Craton': 400.0,
                    'Subduction Interface': 250.0,
                    'Subduction Inslab': 300.0,
                    'Non-Subduction Deep':500.0,
                    'Volcanic':50.0,
                    'default': 200.0}

[output]
mean_hazard_curves = true
quantile_hazard_curves = 0.16 0.50 0.84

[disaggregation]
# placeholders - see the header
poes_disagg = $poes_disagg
disagg_outputs = $disagg_type
max_sites_disagg = 60
num_rlzs_disagg = $num_rlzs_disagg
mag_bin_width = 0.2
distance_bin_width = 10.0
coordinate_bin_width = 5
num_epsilon_bins = 6
""")


def placeholder_poes(n_levels: int) -> list[float]:
    """``n_levels`` distinct annual PoEs between 1e-2 and 1e-4.

    Their values are never used for the disaggregation (the levels are
    replaced). They only have to be valid probabilities, distinct after the
    6-significant-figure formatting of the ini, and give a non-zero hazard map
    somewhere, because ``full_disaggregation`` exits when the whole placeholder
    map is zero (``disaggregation.py:236``).
    """
    poes = [float(f"{p:.6g}") for p in np.geomspace(1e-2, 1e-4, int(n_levels))]
    if len(set(poes)) != len(poes):
        raise ValueError(f"cannot make {n_levels} distinct placeholder poes")
    return poes


def write_child_config(out_fp: Path, *, parent_calc_id: int, parent_name: str,
                       n_levels: int, description: str, levels_file: str,
                       im: str = "03", truncation_level: int = 4,
                       num_rlzs_disagg: int = 0,
                       disagg_type: str = DISAGG_TYPE) -> Path:
    """Write a child disaggregation ini (see the template header).

    ``num_rlzs_disagg = 0`` means all realisations and is required, not a
    tuning knob: the mean (``disagg-stats``) is only computed when more than one
    realisation is kept. The bins are nb 020's, unchanged. Rewrites the file
    only if its text changes, so the fingerprint does not churn.
    """
    out_fp = Path(out_fp)
    text = _CHILD_CONFIG_TEMPLATE.substitute(
        parent_calc_id=int(parent_calc_id),
        parent_name=parent_name,
        n_levels=int(n_levels),
        levels_file=levels_file,
        description=description,
        site_model_file=oq_runner.SITE_MODEL_FILE,
        source_model_logic_tree_file=oq_runner.SOURCE_MODEL_LOGIC_TREE_FILE,
        gsim_logic_tree_file=oq_runner.GMPE_LOGIC_TREES[im],
        truncation_level=int(truncation_level),
        poes_disagg=" ".join(f"{p:.6g}" for p in placeholder_poes(n_levels)),
        disagg_type=disagg_type,
        num_rlzs_disagg=int(num_rlzs_disagg),
    )
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    if not (out_fp.is_file() and out_fp.read_text() == text):
        out_fp.write_text(text)
    return out_fp


def child_description(name: str) -> str:
    """Engine description of a child: front-loaded with its unique name."""
    return f"[{name} - multi-IML disagg child - ESHM20 all sites]"


# --------------------------------------------------------------------------- #
# levels                                                                       #
# --------------------------------------------------------------------------- #

def _key_kind(keys) -> str:
    """Type tag for a key column, so a CSV round-trip restores the dict keys
    exactly: nb 020 keys are IML floats, nb 072 keys are return-period ints,
    nb 065 keys are storey labels."""
    vals = [k for k in keys if k is not None and not (isinstance(k, float) and np.isnan(k))]
    if all(isinstance(k, (bool, np.bool_)) for k in vals):
        raise TypeError("boolean keys are not supported")
    if all(isinstance(k, (int, np.integer)) for k in vals):
        return "int"
    if all(isinstance(k, (int, float, np.integer, np.floating)) for k in vals):
        return "float"
    return "str"


def build_levels_table(targets: pd.DataFrame, delta: float = DELTA_DEFAULT,
                       *, pad: bool = False) -> pd.DataFrame:
    """Lay out the levels to disaggregate, two per target.

    Parameters
    ----------
    targets : DataFrame with columns ``site`` (0..N-1, the datastore site id),
        ``key`` (the label the result is filed under: an IML key, a return
        period, a storey label ...) and ``x`` (the level in **g**). Optional
        ``imt`` column (default ``"AvgSA"``). One row per (site, imt, key).
    delta : relative step of the occurrence pair, x+dx = x·(1 + delta).
    pad : if the sites carry different numbers of targets, repeat each short
        site's last level (role ``"pad"``, never read) instead of raising.
        Never pad with 0: it makes the engine skip the whole site.

    Returns
    -------
    DataFrame with one row per (site, imt, p): ``site, imt, p, key, role,
    iml_g, key_kind``. Within a (site, imt) the targets are ordered by ``x``
    (ties by key), and target k occupies ``p = 2k`` (role ``"x"``) and
    ``p = 2k + 1`` (role ``"x+dx"``). Every site has the same P, as the engine's
    (N, M, P) array requires.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    t = targets.copy()
    if "imt" not in t.columns:
        t["imt"] = "AvgSA"
    missing = {"site", "key", "x"} - set(t.columns)
    if missing:
        raise KeyError(f"targets is missing columns {sorted(missing)}")
    if t.duplicated(["site", "imt", "key"]).any():
        dup = t[t.duplicated(["site", "imt", "key"], keep=False)]
        raise ValueError(f"duplicate (site, imt, key) targets:\n{dup}")
    x = t["x"].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("every target level x must be finite and > 0 (in g)")

    kind = _key_kind(list(t["key"]))
    per_site = {}   # (site, imt) -> list of rows
    for (site, imt), grp in t.groupby(["site", "imt"], sort=True):
        grp = grp.assign(_k=grp["key"].astype(str)).sort_values(["x", "_k"])
        rows = per_site[(int(site), imt)] = []
        for k, (key, xv) in enumerate(zip(grp["key"], grp["x"].astype(float))):
            rows.append((int(site), imt, 2 * k, key, ROLE_X, xv))
            rows.append((int(site), imt, 2 * k + 1, key, ROLE_XDX, xv * (1.0 + delta)))

    # Same P everywhere (the engine's array is rectangular).
    sizes = {len(r) for r in per_site.values()}
    if len(sizes) != 1:
        if not pad:
            raise ValueError(
                "sites carry different numbers of targets "
                f"({sorted(sizes)} levels); pass pad=True to repeat the "
                "last level of the short sites")
        p_max = max(sizes)
        for (site, imt), rows in per_site.items():
            last_level = rows[-1][5]
            rows.extend((site, imt, p, None, ROLE_PAD, last_level)
                        for p in range(len(rows), p_max))
    table = pd.DataFrame([r for rows in per_site.values() for r in rows],
                         columns=["site", "imt", "p", "key", "role", "iml_g"])

    # Two coinciding levels at one site would make x and x+dx of different
    # targets indistinguishable in the output; refuse them.
    real = table[table.role != ROLE_PAD]
    for (site, imt), grp in real.groupby(["site", "imt"]):
        lv = np.sort(grp["iml_g"].to_numpy())
        if np.any(np.diff(lv) <= 0):
            raise ValueError(f"site {site}, {imt}: two levels coincide")

    table["key_kind"] = kind
    return table.sort_values(["site", "imt", "p"]).reset_index(drop=True)


def levels_array(table: pd.DataFrame, n_sites: int,
                 imts: list[str] | tuple[str, ...] = ("AvgSA",)) -> np.ndarray:
    """(N, M, P) array of levels in g, in the engine's site and IMT order.

    Asserts every cell is filled with a finite, positive level: no
    placeholders reach the engine.
    """
    imts = list(imts)
    p_max = int(table["p"].max()) + 1
    arr = np.full((int(n_sites), len(imts), p_max), np.nan)
    m_of = {imt: m for m, imt in enumerate(imts)}
    for site, imt, p, iml in table[["site", "imt", "p", "iml_g"]].itertuples(index=False):
        arr[int(site), m_of[imt], int(p)] = float(iml)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        bad = np.argwhere(~np.isfinite(arr) | (arr <= 0))[:5]
        raise ValueError(f"levels array has empty or non-positive cells, e.g. "
                         f"(site, m, p) = {bad.tolist()}")
    return arr


def levels_at_poes(parent_dstore, poes) -> np.ndarray:
    """Each site's level at the given annual PoEs on the parent's MEAN curve.

    Uses the engine's own routine (``map_array.compute_hmaps``) on the same
    input the stock ``poes_disagg`` path uses (``disaggregation.py:226-233``),
    so the levels equal what a stock ``poes_disagg`` job would have used - up
    to the summation noise between two runs of the same model. Returns
    (N, M, P) in g. A zero (curve never reaches the PoE) is left in place for
    the caller to see; :func:`levels_array` refuses it.
    """
    from openquake.hazardlib import map_array

    oq = parent_dstore["oqparam"]
    mean = parent_dstore.sel("hcurves-stats", stat="mean")[:, 0]  # (N, M, L1)
    return map_array.compute_hmaps(mean, oq.imtls, list(poes))


def split_targets(targets: pd.DataFrame, max_levels: int | None) -> list[pd.DataFrame]:
    """Split a target table into chunks of at most ``max_levels`` levels per site.

    Keys are kept whole (x and x+dx always travel together) and every chunk
    holds the same keys at every site, so each chunk is a valid child on its
    own. ``None`` returns the table unchanged, as a single chunk.
    """
    if max_levels is None:
        return [targets]
    per = max(int(max_levels) // 2, 1)
    keys = sorted(targets["key"].unique(), key=lambda k: (str(type(k)), k))
    return [targets[targets["key"].isin(keys[i:i + per])]
            for i in range(0, len(keys), per)]


def levels_file_stem(name: str) -> str:
    """``levels_<name>``: the stem of the CSV / NPY written next to the ini."""
    return f"levels_{name}"


def write_levels_files(table: pd.DataFrame, n_sites: int, wp1_dir: Path,
                       name: str, imts=("AvgSA",)) -> tuple[Path, Path]:
    """Write ``levels_<name>.csv`` (human-readable, fingerprinted) and
    ``levels_<name>.npy`` (what the calculator loads) into ``wp1_dir``.

    The CSV is written with ``repr``-exact floats so the round trip is
    lossless. Each file is rewritten only when its content changes.
    """
    wp1_dir = Path(wp1_dir)
    csv_fp = wp1_dir / f"{levels_file_stem(name)}.csv"
    npy_fp = wp1_dir / f"{levels_file_stem(name)}.npy"
    text = table.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    if not (csv_fp.is_file() and csv_fp.read_text() == text):
        csv_fp.write_text(text)
    arr = levels_array(table, n_sites, imts)
    if not (npy_fp.is_file() and np.array_equal(np.load(npy_fp), arr)):
        np.save(npy_fp, arr)
    return csv_fp, npy_fp


def read_levels_table(csv_fp: Path) -> pd.DataFrame:
    """Read a levels CSV back, restoring the key type from ``key_kind``."""
    # float_precision="round_trip": pandas' default parser can be 1 ulp off,
    # and collate_targets compares these levels with hmap3 for exact equality.
    table = pd.read_csv(csv_fp, dtype={"key": str}, float_precision="round_trip")
    kind = table["key_kind"].iloc[0]
    conv = {"int": int, "float": float, "str": str}[kind]
    table["key"] = [None if (isinstance(k, float) and np.isnan(k)) or k is None
                    or (isinstance(k, str) and k == "") else conv(k)
                    for k in table["key"]]
    return table


def target_positions(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (site, imt, key): the p of its ``x`` and ``x+dx`` levels."""
    real = table[table.role != ROLE_PAD]
    wide = real.pivot_table(index=["site", "imt", "key"], columns="role",
                            values=["p", "iml_g"], aggfunc="first")
    out = pd.DataFrame({
        "p_x": wide[("p", ROLE_X)].astype(int),
        "p_xdx": wide[("p", ROLE_XDX)].astype(int),
        "x": wide[("iml_g", ROLE_X)],
        "x_dx": wide[("iml_g", ROLE_XDX)],
    }).reset_index()
    return out


# --------------------------------------------------------------------------- #
# running                                                                      #
# --------------------------------------------------------------------------- #

def child_fingerprint(child_ini: Path, levels_csv: Path, wp1_dir: Path, *,
                      parent_entry: dict | None, include_engine: bool = True) -> dict:
    """Content fingerprint of a child calculation.

    - the child ini and the model files it names (``oq_runner.fingerprint_calc``;
      the ini contains the parent calc id, so a new parent changes it);
    - the levels CSV (the levels are the thing being disaggregated);
    - the parent's own manifest fingerprint (so a parent re-run on changed inputs
      makes the child stale even if it got the same calc id elsewhere);
    - with ``include_engine``, the engine version, dist-info version and fork
      commit. Leave it out for a *subset* check on a machine whose engine
      differs from the analysis machine (e.g. the laptop reading a product).
    """
    fp = oq_runner.fingerprint_calc(child_ini, wp1_dir)
    fp.update(fingerprint(levels=Path(levels_csv)))
    if parent_entry is not None:
        parent_inputs = json.dumps(parent_entry.get("inputs", {}), sort_keys=True)
        fp.update(fingerprint(parent_inputs=parent_inputs,
                              parent_calc_id=int(parent_entry["calc_id"])))
    if include_engine:
        info = engine_info()
        fp.update(fingerprint(engine_version=info["engine_version"],
                              engine_commit=info["engine_commit"]))
    return fp


def _run_cmd_in_new_window(cmd: list[str], cwd: Path, log_fp: Path) -> tuple[int, str]:
    """Run ``cmd`` in its own console window, teeing output to ``log_fp``.

    Same approach as ``oq_runner._run_in_new_window`` (PowerShell Tee-Object,
    ``exit $LASTEXITCODE`` so a failure is not reported as success), but for an
    arbitrary command rather than ``oq engine --run``.
    """
    log_fp.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(f"'{c}'" for c in cmd)
    ps = (f"& {quoted} 2>&1 | Tee-Object -FilePath '{log_fp}'; "
          f"exit $LASTEXITCODE")
    shell = shutil.which("pwsh") or "powershell"
    proc = subprocess.Popen([shell, "-NoProfile", "-Command", ps], cwd=cwd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE)
    proc.wait()
    return proc.returncode, oq_runner._read_log(log_fp)


def _calc_id_from_child_output(text: str) -> int | None:
    """The calc id printed by the ``run`` subcommand, else the engine's own
    log-line patterns (``oq_runner._calc_id_from_output``)."""
    m = re.findall(re.escape(_CALC_ID_TAG) + r"(\d+)", text)
    if m:
        return int(m[-1])
    return oq_runner._calc_id_from_output(text)


def run_or_reuse_child(name: str, child_ini: Path, levels_table: pd.DataFrame,
                       n_sites: int, wp1_dir: Path, manifest_fp: Path, *,
                       parent_name: str, force_rerun: bool = False,
                       dry_run: bool = False, new_window: bool = False,
                       imts=("AvgSA",)) -> int | None:
    """Run a child disaggregation, or reuse the recorded one if nothing changed.

    Mirrors ``oq_runner.run_or_reuse``: changed inputs re-run (printing what
    changed), a vanished datastore re-runs, otherwise the recorded calc id is
    returned. The parent must already be in the manifest under ``parent_name``.

    The calculation runs in a **subprocess** (``python -m
    phd_project.scripts.oq_multi_iml_disagg run ...``, cwd = ``wp1_dir``), like
    the stock jobs: the engine's process pool, logging and database handles
    then live and die with that process, not with the notebook kernel. Output
    is teed to ``wp1_dir/logs/<name>.log``.

    Returns the calc id (``None`` under ``dry_run``).
    """
    child_ini, wp1_dir, manifest_fp = Path(child_ini), Path(wp1_dir), Path(manifest_fp)
    manifest = oq_runner.load_manifest(manifest_fp)
    if parent_name not in manifest:
        raise KeyError(f"parent '{parent_name}' is not in {manifest_fp.name}; "
                       f"run it first")
    parent_entry = manifest[parent_name]
    parent_calc_id = int(parent_entry["calc_id"])

    csv_fp, npy_fp = write_levels_files(levels_table, n_sites, wp1_dir, name, imts)
    fp_dict = child_fingerprint(child_ini, csv_fp, wp1_dir, parent_entry=parent_entry)
    entry = manifest.get(name)

    if entry and not force_rerun:
        changes = _diff_inputs(entry.get("inputs", {}), fp_dict)
        if changes and not dry_run:
            print(f"[oq] '{name}' inputs changed since calc_id={entry['calc_id']}:")
            print("\n".join(changes))
        elif changes:
            print(f"[oq] '{name}' inputs changed - would re-run (dry_run).")
            return None
        elif not oq_runner._calc_exists(entry["calc_id"]):
            print(f"[oq] '{name}' calc_id={entry['calc_id']} no longer in the "
                  f"datastore; re-running.")
        else:
            print(f"[oq] '{name}' reusing calc_id={entry['calc_id']} (inputs match).")
            return int(entry["calc_id"])

    if dry_run:
        print(f"[oq] '{name}' would run {child_ini.name} with "
              f"{npy_fp.name} on parent {parent_calc_id} (dry_run).")
        return None

    cmd = [sys.executable, "-m", "phd_project.scripts.oq_multi_iml_disagg", "run",
           "--ini", child_ini.name, "--levels", npy_fp.name,
           "--parent", str(parent_calc_id)]
    log_fp = wp1_dir / "logs" / f"{name}.log"
    if new_window and sys.platform == "win32":
        print(f"[oq] '{name}' running {child_ini.name} in a new window "
              f"(log: logs/{log_fp.name}) ...")
        returncode, text = _run_cmd_in_new_window(cmd, wp1_dir, log_fp)
    else:
        print(f"[oq] '{name}' running {child_ini.name} ...")
        proc = subprocess.run(cmd, cwd=wp1_dir, capture_output=True, text=True)
        returncode, text = proc.returncode, proc.stdout + proc.stderr
        log_fp.parent.mkdir(parents=True, exist_ok=True)
        log_fp.write_text(text, encoding="utf-8")

    if returncode != 0:
        raise RuntimeError(f"child '{name}' failed (exit {returncode}):\n{text[-4000:]}")
    calc_id = _calc_id_from_child_output(text)
    if calc_id is None:
        raise RuntimeError(f"child '{name}' ran but no calc id was found in its "
                           f"output:\n{text[-2000:]}")

    info = engine_info()
    oq_runner._write_manifest_entry(manifest_fp, name, {
        "calc_id": calc_id,
        "kind": "multi_iml_child",
        "config": child_ini.name,
        "description": oq_runner._read_key(child_ini, "description"),
        "parent": parent_name,
        "parent_calc_id": parent_calc_id,
        "levels_csv": csv_fp.name,
        "levels_npy": npy_fp.name,
        "n_levels": int(levels_table["p"].max()) + 1,
        "inputs": fp_dict,
        "_meta": {"git_commit": _git_commit(),
                  "written_at": datetime.now(timezone.utc).isoformat(),
                  **info},
    })
    print(f"[oq] '{name}' completed: calc_id={calc_id}")
    return calc_id


def child_entries(manifest: dict, name_regex: str) -> dict[str, dict]:
    """Manifest entries of the children whose names match ``name_regex``
    (full match), sorted by name. Used by nb 021 / nb 072 to pick up every
    child under one prefix, including ones added later."""
    rx = re.compile(name_regex)
    return {k: v for k, v in sorted(manifest.items())
            if rx.fullmatch(k) and v.get("kind") == "multi_iml_child"}


# --------------------------------------------------------------------------- #
# reading and occurrence                                                       #
# --------------------------------------------------------------------------- #

def _dset_slice(n_dims: int, site: int, m: int, p: int) -> tuple:
    """Index of one (site, imt, level) in a disagg-* dataset whose shape is
    (site, <bins...>, imt, poe, Z): every bin axis and Z kept."""
    return (site,) + (slice(None),) * (n_dims - 4) + (m, p, slice(None))


def read_levels(dstore, disagg_type: str = DISAGG_TYPE, *, sites=None,
                levels: dict | None = None) -> dict:
    """Exceedance disaggregation frames of the MEAN, per (site, imt, p).

    Returns ``out[site][imt][p] -> DataFrame`` with columns ``TRT, Mag, Dist,
    Eps, Z, P(X>x|T,m), nu_m, P(m|X>x)`` (the nb 021 columns minus the
    occurrence), built from the same ``oqhelpers`` blocks the old readers used.
    ``levels`` optionally restricts the p read per site:
    ``{site: [p, ...]}``.
    """
    t = dstore["oqparam"].investigation_time
    dset = dstore["disagg-stats"][disagg_type]
    bins = oqhelpers.get_bins(dstore, disagg_type)
    n_sites, n_dims = dset.shape[0], len(dset.shape)
    n_p = dset.shape[-2]
    out = {}
    for site in (range(n_sites) if sites is None else sites):
        out[site] = {}
        ps = range(n_p) if levels is None else levels[site]
        for m, imt in enumerate(bins["imt"]):
            out[site][imt] = {}
            for p in ps:
                arr = dset[_dset_slice(n_dims, site, m, p)]
                df = oqhelpers._disagg_array_to_df(arr, disagg_type, bins)
                out[site][imt][p] = oqhelpers._get_traditional_disagg(df, t)
    return out


def occurrence_disagg(df_x: pd.DataFrame, df_xdx: pd.DataFrame, *,
                      neg_tol: float = 1e-12) -> pd.DataFrame:
    """Add ``P(m|X=x)`` to ``df_x`` by Bradley (2010) Eq. 7 (forward difference).

    .. math::

        P(m \\mid X=x) \\approx
        \\frac{\\nu_m(x) - \\nu_m(x+\\delta x)}{\\nu(x) - \\nu(x+\\delta x)},
        \\qquad \\nu(x) = \\sum_m \\nu_m(x)

    ``ν_m`` is the ``nu_m`` column of each frame (annual exceedance rate of the
    bin). Taking ``ν`` as the sum of the bins, rather than from a separate
    hazard curve, makes the column sum to exactly 1 per realisation index Z; in
    exact arithmetic it equals Bradley's λ(x) - λ(x+δx).

    Each bin's exceedance rate can only fall as x rises, so the numerator is
    ≥ 0 up to rounding. Values down to ``-neg_tol * max|numerator|`` are
    rounding and are clipped to 0; anything more negative means the two frames
    do not belong together and raises. Where ``ν(x) - ν(x+δx) <= 0`` (x beyond
    the ε-truncated maximum ground motion: zero hazard) the column is NaN; such
    IMLs are listed by ``oqhelpers.get_excluded_imls``.

    The bins of the two frames must be identical (same row order); asserted.
    """
    for c in BIN_COLS:
        if not np.array_equal(df_x[c].to_numpy(), df_xdx[c].to_numpy()):
            raise ValueError(f"x and x+dx frames have different '{c}' bins")
    out = df_x.copy()
    num = df_x["nu_m"].to_numpy(float) - df_xdx["nu_m"].to_numpy(float)
    occ = np.full(len(out), np.nan)
    for z in np.unique(out["Z"].to_numpy()):
        sel = out["Z"].to_numpy() == z
        nz = num[sel]
        scale = np.nanmax(np.abs(nz)) if nz.size else 0.0
        if np.any(nz < -neg_tol * scale):
            raise ValueError(
                f"Z={z}: a bin's exceedance rate rises with x "
                f"(min numerator {nz.min():.3e}, scale {scale:.3e})")
        nz = np.clip(nz, 0.0, None)
        den = nz.sum()
        if den > 0 and np.isfinite(den):
            occ[sel] = nz / den
    out["P(m|X=x)"] = occ
    return out


def collate_targets(dstore, levels_table: pd.DataFrame,
                    disagg_type: str = DISAGG_TYPE) -> dict:
    """Occurrence-and-exceedance frames per (site, imt, key) from one child.

    Returns ``out[site][imt][key] -> DataFrame`` with the nb 021 columns
    ``TRT, Mag, Dist, Eps, Z, P(X>x|T,m), nu_m, P(m|X>x), P(m|X=x)``, where
    the exceedance columns are those at ``x`` and the occurrence column pairs
    ``x`` with ``x+dx``. This is the format nb 021 and nb 072 consume.

    The datastore's own ``hmap3`` is the source of truth for the levels: it is
    asserted equal (exactly) to ``levels_table`` before anything is read, so a
    levels CSV that does not belong to this datastore fails here rather than
    mislabelling results.
    """
    hmap3 = np.asarray(dstore["hmap3"][()])
    imts = list(dstore["oqparam"].imtls)
    expected = levels_array(levels_table, hmap3.shape[0], imts)
    if expected.shape != hmap3.shape or not np.array_equal(expected, hmap3):
        raise ValueError("the levels table does not match this datastore's hmap3")

    pos = target_positions(levels_table)
    out = {}
    for site, grp in pos.groupby("site", sort=True):
        need = sorted(set(grp["p_x"]) | set(grp["p_xdx"]))
        frames = read_levels(dstore, disagg_type, sites=[int(site)],
                             levels={int(site): need})[int(site)]
        out[int(site)] = {}
        for row in grp.itertuples(index=False):
            df = occurrence_disagg(frames[row.imt][row.p_x], frames[row.imt][row.p_xdx])
            out[int(site)].setdefault(row.imt, {})[row.key] = df
    return out


def merge_collations(collations: list[dict]) -> dict:
    """Merge several ``collate_targets`` results (one per child) into one.

    Raises if two children supply the same (site, imt, key): a level must have
    exactly one source.
    """
    out = {}
    for col in collations:
        for site, imt_dict in col.items():
            for imt, key_dict in imt_dict.items():
                dst = out.setdefault(site, {}).setdefault(imt, {})
                clash = set(dst) & set(key_dict)
                if clash:
                    raise ValueError(f"site {site}, {imt}: keys {sorted(clash)} "
                                     f"supplied by more than one child")
                dst.update(key_dict)
    # deterministic key order inside every (site, imt)
    return {s: {imt: dict(sorted(kd.items())) for imt, kd in d.items()}
            for s, d in sorted(out.items())}


# --------------------------------------------------------------------------- #
# checks                                                                       #
# --------------------------------------------------------------------------- #

def _report(rows: list[dict], report_fp: Path | None, what: str) -> pd.DataFrame:
    """Turn check rows into a DataFrame, optionally write it, raise on FAIL."""
    rep = pd.DataFrame(rows)
    if report_fp is not None:
        Path(report_fp).parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(report_fp, index=False)
    if len(rep) and (rep["status"] == "FAIL").any():
        bad = rep[rep["status"] == "FAIL"]
        raise AssertionError(f"{what}: {len(bad)} check(s) failed\n"
                             f"{bad.head(20).to_string()}")
    return rep


def selftest_occurrence(delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """Check :func:`occurrence_disagg` on a synthetic case with a known answer.

    Two bins with power-law exceedance rates ν_i(x) = a_i x^(-k_i). The exact
    occurrence share of bin i is a_i k_i x^(-k_i-1) / Σ_j a_j k_j x^(-k_j-1),
    which the forward difference approximates to O(δ). The test asserts:

    1. agreement with the analytic answer to within 2δ (relative);
    2. the column sums to 1;
    3. it differs from the exceedance share P(m|X>x) - the old code returned
       exactly the exceedance share, so this is the bug guard.

    Needs no OpenQuake run, so it can also run on the laptop.
    """
    a = np.array([1e-3, 4e-3])
    k = np.array([1.5, 3.5])
    x = 0.4

    def frame(xv):
        nu = a * xv ** (-k)
        df = pd.DataFrame({"TRT": ["A", "B"], "Mag": [5.0, 7.0], "Dist": [10.0, 50.0],
                           "Eps": [0.0, 0.0], "Z": [0.0, 0.0],
                           "P(X>x|T,m)": 1.0 - np.exp(-nu)})
        return oqhelpers._get_traditional_disagg(df, 1.0)

    occ = occurrence_disagg(frame(x), frame(x * (1 + delta)))["P(m|X=x)"].to_numpy()
    w = a * k * x ** (-k - 1)
    exact = w / w.sum()
    exceed = frame(x)["P(m|X>x)"].to_numpy()
    rows = [
        {"check": "analytic", "value": float(np.max(np.abs(occ / exact - 1))),
         "tol": 2 * delta},
        {"check": "sums_to_1", "value": float(abs(occ.sum() - 1)), "tol": 1e-12},
        {"check": "differs_from_exceedance",
         "value": float(np.max(np.abs(occ - exceed))), "tol": 1e-2},
    ]
    for r in rows:
        ok = r["value"] >= r["tol"] if r["check"] == "differs_from_exceedance" \
            else r["value"] <= r["tol"]
        r["status"] = "PASS" if ok else "FAIL"
    return _report(rows, None, "selftest_occurrence")


def check_site_order(dstore, site_model_csv: Path, sites_csv: Path | None = None,
                     *, parent_dstore=None, decimals: int = 5,
                     report_fp: Path | None = None) -> pd.DataFrame:
    """sid i = row i of the site model CSV (and of ``sites.csv``, and of the
    parent's site collection).

    The engine keeps the site-model file order (``readinput.py:583-603``), and
    the nb 021 shards / nb 065 row map key results by that sid, so a reorder
    would silently attach every disaggregation to the wrong site.
    """
    sc = dstore["sitecol"]
    lon, lat = np.asarray(sc.lons), np.asarray(sc.lats)
    rows = []

    def cmp(name, ref_lon, ref_lat):
        ok = (len(ref_lon) == len(lon)
              and np.allclose(np.round(ref_lon, decimals), np.round(lon, decimals), atol=0)
              and np.allclose(np.round(ref_lat, decimals), np.round(lat, decimals), atol=0))
        d = (float(np.max(np.abs(np.r_[ref_lon - lon, ref_lat - lat])))
             if len(ref_lon) == len(lon) else np.inf)
        rows.append({"check": f"sitecol == {name}", "n": len(ref_lon),
                     "max_abs_deg": d, "status": "PASS" if ok else "FAIL"})

    sm = pd.read_csv(site_model_csv)
    cmp(Path(site_model_csv).name, sm["lon"].to_numpy(float), sm["lat"].to_numpy(float))
    if sites_csv is not None:
        st = pd.read_csv(sites_csv)
        cmp(Path(sites_csv).name, st["lon"].to_numpy(float), st["lat"].to_numpy(float))
    if parent_dstore is not None:
        psc = parent_dstore["sitecol"]
        cmp("parent sitecol", np.asarray(psc.lons), np.asarray(psc.lats))
    return _report(rows, report_fp, "check_site_order")


def _engine_version(dstore) -> str:
    return str(dstore.hdf5.attrs.get("engine_version", "unknown"))


# Absolute noise floor for comparing two independent runs of the same model.
# The engine forms each bin's probability through 1 - prod(1 - p) / 1 - exp(-rate)
# steps, which leave an ABSOLUTE error of a few machine epsilons (2.2e-16) on the
# scale of 1, whatever the size of the result. Observed: |diff| <= 4e-15 between
# the 56-level child and the 1-level stock nb 020 jobs (same engine commit);
# relative differences of ~2e-8 then appear only in bins of size ~1e-7. 1e-13 is
# ~450 eps: far above that noise, and orders of magnitude below any error a real
# fault would cause (a wrong site, chunk or level moves bins by 1e-3 or more).
NOISE_ATOL = 1e-13


def _rel_err(a: np.ndarray, b: np.ndarray, atol: float) -> tuple[float, float]:
    """(max |a-b|, max |a-b| / max(|b|, atol))."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    if d.size == 0:
        return 0.0, 0.0
    return float(d.max()), float((d / np.maximum(np.abs(b), atol)).max())


def verify_against_reference(new_dstore, refs: dict, *, reader=None,
                             disagg_type: str = DISAGG_TYPE, rtol: float = 1e-9,
                             atol: float = NOISE_ATOL, level_rtol: float = 0.0,
                             report_fp: Path | None = None) -> pd.DataFrame:
    """Compare a child with stock reference datastores, level by level.

    Parameters
    ----------
    new_dstore : the child datastore.
    refs : ``{ref_calc_id: (p_ref, p_new)}`` - level ``p_ref`` of the reference
        is compared with level ``p_new`` of the child (e.g. an old nb 020
        ``iml_disagg`` job has one level, ``p_ref = 0``).
    level_rtol : allowed relative difference of the levels themselves. 0 (exact)
        for nb 020, whose reference IMLs come from the same 6-significant-figure
        text; small but non-zero for nb 072, whose reference levels came from a
        separate run of the same hazard model.
    rtol, atol : for the disaggregation arrays (all probabilities), passed as
        ``|new - ref| <= atol + rtol * |ref|``. The child and the references are
        independent runs of the same model, so they agree to summation-order
        noise, not bit for bit. That noise is ABSOLUTE (a few machine epsilons,
        observed <= 4e-15), so tiny bins show large relative differences; the
        default ``atol = NOISE_ATOL`` (1e-13) sits above it. Do not set ``atol``
        below ~1e-14 or the smallest bins fail on rounding alone.

    Compared per reference: the site collection, the bins
    (``disagg-bins/{Mag,Dist,Eps,TRT,Lon,Lat}``), the level (``hmap3``), the
    mean (``disagg-stats``), every realisation (``disagg-rlzs``) and ``poe4``.
    Realisations are matched **by realisation id** through each datastore's
    ``best_rlzs``: the z order depends on the engine's ranking against the
    mean curve, which differs between a 1-level stock job and a 25-level child.

    If the reference was written by a different engine version, a mismatch is
    reported as ``REVIEW`` rather than ``FAIL``: a version change can move the
    numbers legitimately, and that is for the analyst to judge.
    """
    if reader is None:
        from openquake.commonlib.datastore import read as reader
    new_ver = _engine_version(new_dstore)
    new_sc = new_dstore["sitecol"]
    new_best = np.asarray(new_dstore["best_rlzs"][()])
    new_stats = new_dstore["disagg-stats"][disagg_type]
    new_rlzs = new_dstore["disagg-rlzs"][disagg_type]
    new_poe4 = np.asarray(new_dstore["poe4"][()])
    new_hmap = np.asarray(new_dstore["hmap3"][()])
    n_dims = len(new_stats.shape)
    rows = []
    for ref_id, (p_ref, p_new) in refs.items():
        ref = reader(int(ref_id))
        try:
            ver = _engine_version(ref)
            fail = "FAIL" if ver == new_ver else "REVIEW"

            def add(check, ok, max_abs=np.nan, max_rel=np.nan):
                rows.append({"ref_calc_id": int(ref_id), "p_ref": int(p_ref),
                             "p_new": int(p_new), "check": check,
                             "max_abs": max_abs, "max_rel": max_rel,
                             "status": "PASS" if ok else fail,
                             "ref_engine": ver, "new_engine": new_ver})

            rsc = ref["sitecol"]
            add("sitecol", np.array_equal(np.asarray(rsc.lons), np.asarray(new_sc.lons))
                and np.array_equal(np.asarray(rsc.lats), np.asarray(new_sc.lats)))
            for b in ("Mag", "Dist", "Eps", "Lon", "Lat"):
                ra, na = np.asarray(ref[f"disagg-bins/{b}"]), np.asarray(new_dstore[f"disagg-bins/{b}"])
                ok = ra.shape == na.shape and np.allclose(ra, na, rtol=1e-12, atol=0)
                add(f"bins/{b}", ok)
            add("bins/TRT", list(ref["disagg-bins/TRT"]) == list(new_dstore["disagg-bins/TRT"]))

            r_lv = np.asarray(ref["hmap3"][()])[:, :, p_ref]
            n_lv = new_hmap[:, :, p_new]
            a, r = _rel_err(n_lv, r_lv, 1e-300)
            add("hmap3 level", r <= level_rtol, a, r)

            # mean
            rs = ref["disagg-stats"][disagg_type]
            ra = np.stack([rs[_dset_slice(n_dims, s, 0, p_ref)] for s in range(rs.shape[0])])
            na = np.stack([new_stats[_dset_slice(n_dims, s, 0, p_new)]
                           for s in range(new_stats.shape[0])])
            a, r = _rel_err(na, ra, atol)
            add("disagg-stats (mean)", np.allclose(na, ra, rtol=rtol, atol=atol), a, r)

            # realisations, matched by rlz id
            rr = ref["disagg-rlzs"][disagg_type]
            r_best = np.asarray(ref["best_rlzs"][()])
            r_poe4 = np.asarray(ref["poe4"][()])
            worst_a, worst_r, ok_all, ok_poe4, n_cmp = 0.0, 0.0, True, True, 0
            p4a, p4r = 0.0, 0.0
            for s in range(new_best.shape[0]):
                r_arr = rr[_dset_slice(n_dims, s, 0, p_ref)]   # (bins..., Z_ref)
                n_arr = new_rlzs[_dset_slice(n_dims, s, 0, p_new)]
                z_new = {int(rl): z for z, rl in enumerate(new_best[s])}
                for z_ref, rl in enumerate(r_best[s]):
                    zn = z_new.get(int(rl))
                    if zn is None:
                        ok_all = False
                        continue
                    a1, r1 = _rel_err(n_arr[..., zn], r_arr[..., z_ref], atol)
                    worst_a, worst_r = max(worst_a, a1), max(worst_r, r1)
                    ok_all &= bool(np.allclose(n_arr[..., zn], r_arr[..., z_ref],
                                               rtol=rtol, atol=atol))
                    a2, r2 = _rel_err(new_poe4[s, 0, p_new, zn], r_poe4[s, 0, p_ref, z_ref], atol)
                    p4a, p4r = max(p4a, a2), max(p4r, r2)
                    ok_poe4 &= bool(np.isclose(new_poe4[s, 0, p_new, zn],
                                               r_poe4[s, 0, p_ref, z_ref], rtol=rtol, atol=atol))
                    n_cmp += 1
            add(f"disagg-rlzs by rlz id ({n_cmp} site-rlz pairs)", ok_all, worst_a, worst_r)
            add("poe4 by rlz id", ok_poe4, p4a, p4r)
        finally:
            ref.close()
    return _report(rows, report_fp, "verify_against_reference")


def check_total_probability(dstore, parent_dstore, *, hcurves_004: dict | None = None,
                            rtol_grid: float = 1e-5, rtol_004: float = 1e-5,
                            disagg_type: str = DISAGG_TYPE,
                            report_fp: Path | None = None) -> pd.DataFrame:
    """The disaggregation adds up to the parent's hazard curve.

    For one realisation, the bins partition the ruptures, so the probability of
    exceeding a level is ``1 - Π_bins (1 - p_bin)``. The engine's mean hazard
    curve is the weighted mean of the realisations' **probabilities**
    (``getters.build_stat_curve``, ``use_rates = False``), while the mean
    disaggregation is the weighted mean of **rates**; the two differ at second
    order (Jensen), so the like-for-like comparison is

        Σ_r w_r · [1 - Π_bins (1 - p_bin,r)]   vs   mean curve at the level

    using ``disagg-rlzs`` and the realisation weights. At levels on the
    parent's 25-point grid this is exact up to summation noise (``rtol_grid``).
    Off the grid the curve has to be interpolated (linear in ln x, ln poe);
    those rows are reported with a curvature-based tolerance
    ``1.5 · max|Δ² ln poe| / 8`` (the bound on linear-interpolation error) and
    are informational - the grid rows are the test.

    With ``hcurves_004`` (the nb 004 pickle, ``[IML, MAFE]`` per site), the
    parent's mean curve is also compared with nb 004's: same model, so it must
    agree to summation noise.

    Tolerances are set by float32, not float64:

    - The engine stores its hazard rates (``map_array.rates_dt``) and its
      curves (``hcurves-stats``) as float32 (~7 significant figures), while the
      disaggregation is recomputed in float64. Differences of ~1e-6 relative
      (about 10 float32 steps) are expected; ``rtol_grid = 1e-5`` (~80 steps)
      sits above that and far below a real fault (a missing source or a wrong
      site shows at the percent level).
    - The nb 004 pickle holds MAFE computed by ``oqhelpers.get_hcurves_from_dstore``
      as ``-log(1 - y) / T`` with ``y`` the float32 poe, so ``1 - y`` is rounded
      to float32 steps of 1.2e-7: small MAFE values are quantised and those with
      poe below ~6e-8 are exactly 0. A rate comparison would divide by those
      zeros, so the comparison is made in **poe**: the pickle is converted back
      with ``1 - exp(-T * MAFE)`` and allowed an absolute error of two float32
      steps near 1 (``2 * spacing(float32(1))``, the most the rounding of
      ``1 - y`` can cause) plus ``rtol_004`` relative.
    """
    oq = parent_dstore["oqparam"]
    imts = list(oq.imtls)
    grid = {imt: np.asarray(oq.imtls[imt], float) for imt in imts}
    mean = parent_dstore.sel("hcurves-stats", stat="mean")[:, 0]  # (N, M, L)
    weights = np.asarray(dstore["weights"][()], float)
    if weights.ndim > 1:   # IMT-dependent weights: the last column is the total
        weights = weights[:, -1]
    # With use_rates the engine averages rates, not probabilities.
    use_rates = bool(getattr(oq, "use_rates", False))
    best = np.asarray(dstore["best_rlzs"][()])
    hmap3 = np.asarray(dstore["hmap3"][()])
    dset = dstore["disagg-rlzs"][disagg_type]
    n_dims = len(dset.shape)
    rows = []
    for s in range(hmap3.shape[0]):
        for m, imt in enumerate(imts):
            lx = np.log(grid[imt])
            with np.errstate(divide="ignore"):
                ly = np.log(mean[s, m])
            for p in range(hmap3.shape[2]):
                level = float(hmap3[s, m, p])
                arr = dset[_dset_slice(n_dims, s, m, p)]           # (bins..., Z)
                poe_z = 1.0 - np.prod(1.0 - arr.reshape(-1, arr.shape[-1]), axis=0)
                w = weights[best[s]]
                if use_rates:
                    rate = np.sum(w * -np.log1p(-poe_z)) / np.sum(w)
                    poe_d = float(-np.expm1(-rate))
                else:
                    poe_d = float(np.sum(w * poe_z) / np.sum(w))
                on_grid = np.isclose(grid[imt], level, rtol=1e-12, atol=0)
                if on_grid.any():
                    poe_c = float(mean[s, m][np.argmax(on_grid)])
                    tol, info = rtol_grid, False
                    rel = abs(poe_d / poe_c - 1) if poe_c > 0 else np.inf
                else:
                    j = int(np.clip(np.searchsorted(lx, np.log(level)) - 1, 0, len(lx) - 2))
                    poe_c = float(np.exp(np.interp(np.log(level), lx, ly)))
                    with np.errstate(invalid="ignore"):   # -inf where the curve is 0
                        d2 = np.abs(np.diff(ly, 2))
                    near = d2[max(j - 1, 0): j + 1]
                    tol = 1.5 * float(np.nanmax(near)) / 8 + 1e-6 if near.size else np.inf
                    rel = abs(np.log(poe_d) - np.log(poe_c)) if poe_c > 0 and poe_d > 0 else np.inf
                    info = True
                ok = rel <= tol
                rows.append({"site": s, "imt": imt, "p": p, "level_g": level,
                             "on_grid": not info, "poe_disagg": poe_d, "poe_curve": poe_c,
                             "rel_or_dln": rel, "tol": tol,
                             "status": ("PASS" if ok else ("INFO" if info else "FAIL"))})
        if hcurves_004 is not None:
            for m, imt in enumerate(imts):
                hc = np.asarray(hcurves_004[s][imt]["mean"])
                # back from the pickle's MAFE to poe (exact inverse of the
                # -log(1 - y) / T it was made with, up to that step's rounding)
                poe_004 = -np.expm1(-oq.investigation_time * hc[:, 1].astype(float))
                poe_parent = mean[s, m].astype(float)
                # absolute floor: the float32 rounding of 1 - y in nb 004
                atol_004 = 2.0 * float(np.spacing(np.float32(1.0)))
                ok = (np.allclose(hc[:, 0], grid[imt], rtol=1e-12, atol=0)
                      and np.allclose(poe_parent, poe_004, rtol=rtol_004, atol=atol_004))
                a, r = _rel_err(poe_parent, poe_004, atol_004)
                rows.append({"site": s, "imt": imt, "p": -1, "level_g": np.nan,
                             "on_grid": True, "poe_disagg": np.nan, "poe_curve": np.nan,
                             "rel_or_dln": r, "tol": rtol_004,
                             "status": "PASS" if ok else "FAIL",
                             "check": "parent mean curve == nb 004 pickle (poe)"})
    return _report(rows, report_fp, "check_total_probability")


def check_independence(base_dstore, perturbed_dstore, changed: list[tuple[int, int]], *,
                       disagg_type: str = DISAGG_TYPE, rtol: float = 1e-12,
                       report_fp: Path | None = None) -> pd.DataFrame:
    """Changing some levels changes only those levels' results.

    ``changed`` lists the (site, p) cells whose level differs between the two
    children. Those must differ; every other cell, including the other levels
    of the same site, must be unchanged. "Unchanged" is ``rtol = 1e-12``, not
    bit-identity: the engine adds task results in completion order, so two
    runs of the same job can differ in the last bits.
    """
    hb = np.asarray(base_dstore["hmap3"][()])
    hp = np.asarray(perturbed_dstore["hmap3"][()])
    db = base_dstore["disagg-stats"][disagg_type]
    dp = perturbed_dstore["disagg-stats"][disagg_type]
    n_dims = len(db.shape)
    changed = {(int(s), int(p)) for s, p in changed}
    rows = []
    for s in range(hb.shape[0]):
        for p in range(hb.shape[2]):
            is_changed = (s, p) in changed
            lv_same = hb[s, 0, p] == hp[s, 0, p]
            a, b = db[_dset_slice(n_dims, s, 0, p)], dp[_dset_slice(n_dims, s, 0, p)]
            same = bool(np.allclose(a, b, rtol=rtol, atol=1e-300))
            ok = (not lv_same and not same) if is_changed else (lv_same and same)
            mx, rl = _rel_err(b, a, 1e-300)
            rows.append({"site": s, "p": p, "changed": is_changed, "level_same": lv_same,
                         "result_same": same, "max_abs": mx, "max_rel": rl,
                         "status": "PASS" if ok else "FAIL"})
    return _report(rows, report_fp, "check_independence")


def compare_processed_shards(old_dir: Path, new_dir: Path, *, excluded: dict | None = None,
                             rtol: float = 1e-9, sum_tol: float = 1e-9,
                             atol: float = NOISE_ATOL,
                             report_fp: Path | None = None) -> pd.DataFrame:
    """nb 021 check 4: the rebuilt shards against the pre-fix backup.

    Per (site, imt, iml): the exceedance columns ``P(X>x|T,m)``, ``nu_m`` and
    ``P(m|X>x)`` must be unchanged (the levels and the model are the same) up
    to rounding noise: ``atol`` on the bin probabilities, carried through to
    ``nu_m`` and ``P(m|X>x)`` as explained in the loop below;
    ``P(m|X=x)`` must now sum to 1, be >= 0, and **differ** from
    ``P(m|X>x)`` - the old code wrote the exceedance share into the occurrence
    column, so equality there means the fix did not take effect. Zero-hazard
    IMLs (``excluded``: ``{site: [imls]}``) are exempt from the occurrence tests.
    """
    from .disagg_shards import load_shards

    old, new = load_shards(old_dir), load_shards(new_dir)
    excl = {int(s): set(map(float, v)) for s, v in (excluded or {}).items()}
    rows = []
    if set(old) != set(new):
        rows.append({"site": -1, "imt": "", "iml": np.nan, "check": "same sites",
                     "value": np.nan, "status": "FAIL"})
    for s in sorted(set(old) & set(new)):
        for imt in old[s]:
            ok_keys = set(old[s][imt]) == set(new[s].get(imt, {}))
            if not ok_keys:
                rows.append({"site": s, "imt": imt, "iml": np.nan, "check": "same imls",
                             "value": np.nan, "status": "FAIL"})
            for iml in sorted(set(old[s][imt]) & set(new[s].get(imt, {}))):
                o, n = old[s][imt][iml], new[s][imt][iml]
                for c in BIN_COLS:
                    if not np.array_equal(o[c].to_numpy(), n[c].to_numpy()):
                        rows.append({"site": s, "imt": imt, "iml": iml, "check": f"bins {c}",
                                     "value": np.nan, "status": "FAIL"})
                # The old shards come from the 1-level stock jobs and the new
                # ones from the child: independent runs, so they agree to the
                # absolute noise floor `atol` on the bin probabilities, not bit
                # for bit. The floor is carried through to the derived columns
                # (oqhelpers._get_traditional_disagg):
                #   nu_m      = -ln(1 - p) / T      -> floor atol / T
                #   P(m|X>x)  = nu_m / nu           -> floor (atol / T) / nu,
                # and the share also inherits the relative noise of nu itself,
                # (atol / T) / nu, which matters at low-hazard IMLs where nu is
                # tiny. T (investigation time) and nu are recovered from the old
                # frame itself, so no extra inputs are needed.
                p_o = o["P(X>x|T,m)"].to_numpy(float)
                nu_o = o["nu_m"].to_numpy(float)
                pos = (p_o > 0) & (nu_o > 0)
                T = float(np.median(-np.log1p(-p_o[pos]) / nu_o[pos])) if pos.any() else 1.0
                share_sum = float(o["P(m|X>x)"].sum())
                nu_tot = float(nu_o.sum()) / share_sum if share_sum > 0 else np.inf
                floor_nu = atol / T
                floor_share = floor_nu / nu_tot
                tols = {"P(X>x|T,m)": (rtol, atol),
                        "nu_m": (rtol, floor_nu),
                        "P(m|X>x)": (rtol + floor_share, floor_share)}
                for c, (rt, at) in tols.items():
                    _, r = _rel_err(n[c].to_numpy(), o[c].to_numpy(), at)
                    ok = np.allclose(n[c], o[c], rtol=rt, atol=at)
                    rows.append({"site": s, "imt": imt, "iml": iml, "check": c,
                                 "value": r, "status": "PASS" if ok else "FAIL"})
                if float(iml) in excl.get(s, set()):
                    continue
                occ = n["P(m|X=x)"].to_numpy(float)
                rows.append({"site": s, "imt": imt, "iml": iml, "check": "occ sums to 1",
                             "value": abs(np.nansum(occ) - 1),
                             "status": "PASS" if abs(np.nansum(occ) - 1) <= sum_tol else "FAIL"})
                rows.append({"site": s, "imt": imt, "iml": iml, "check": "occ >= 0",
                             "value": float(np.nanmin(occ)),
                             "status": "PASS" if np.nanmin(occ) >= 0 else "FAIL"})
                diff = float(np.nanmax(np.abs(occ - n["P(m|X>x)"].to_numpy(float))))
                rows.append({"site": s, "imt": imt, "iml": iml,
                             "check": "occ differs from exceedance", "value": diff,
                             "status": "PASS" if diff > 1e-6 else "FAIL"})
    return _report(rows, report_fp, "compare_processed_shards")


# --------------------------------------------------------------------------- #
# command line                                                                 #
# --------------------------------------------------------------------------- #

def _cli_run(args) -> int:
    """``run``: one child calculation in this process (cwd = the ini's folder)."""
    calc_id = run_child_calculation(Path(args.ini), Path(args.levels), int(args.parent))
    # A plain tagged line, so run_or_reuse_child does not depend on the engine's
    # log format to learn the calc id.
    print(f"{_CALC_ID_TAG}{calc_id}", flush=True)
    return 0


def _cli_verify(args) -> int:
    """``verify``: the datastore checks of plan (1), from the shell.

    ``--which imlgrid``: site order, every old nb 020 calc in the manifest vs
    the child(ren) matching ``AvgSA_03_disagg_eps4_imlgrid*``, and the total
    probability. ``--which rtps``: calc 126 (``AvgSA_03_disagg_eps4_rtps``)
    vs ``AvgSA_03_disagg_eps4_rtps_ml``. ``--which selftest``: the synthetic
    occurrence test only. Reports go to ``<wp1>/logs/verify_<name>_*.csv``.
    """
    from openquake.commonlib.datastore import read

    if args.which == "selftest":
        print(selftest_occurrence().to_string())
        return 0

    from phd_project.config import config
    cfg = config.load_config()
    wp1 = Path(cfg["hazard_models"]["eshm20_wp1"])
    manifest = oq_runner.load_manifest(cfg["hazard_models"]["eshm20_wp1_disagg_manifest"])
    logs_dir = wp1 / "logs"
    parent = read(int(manifest["AvgSA_03_disagg_eps4_parent"]["calc_id"]))

    if args.which == "imlgrid":
        old_re = re.compile(r"^AvgSA_03_disagg_eps4_(\d+pt\d+)$")
        old = {name: e for name, e in manifest.items() if old_re.match(name)}
        for cname, centry in child_entries(manifest, r"AvgSA_03_disagg_eps4_imlgrid(_c\d+)?").items():
            child = read(int(centry["calc_id"]))
            table = read_levels_table(wp1 / centry["levels_csv"])
            pos = target_positions(table)
            check_site_order(child, wp1 / oq_runner.SITE_MODEL_FILE,
                             cfg["results"]["selected_sites_csv"], parent_dstore=parent,
                             report_fp=logs_dir / f"verify_{cname}_site_order.csv")
            # Old names carry the IML as a 3-dp token (oq_runner._iml_fname_token);
            # the child's keys are the 6-significant-figure IMLs. Match on the
            # token; every level sits at the same p at every site (same targets).
            refs = {}
            site0 = pos[pos.site == 0]
            p_of_tok = {oq_runner._iml_fname_token(k): int(px)
                        for k, px in zip(site0["key"], site0["p_x"])}
            for name, e in old.items():
                tok = old_re.match(name)[1]
                if tok in p_of_tok:
                    refs[int(e["calc_id"])] = (0, p_of_tok[tok])
            print(verify_against_reference(
                child, refs, report_fp=logs_dir / f"verify_{cname}_vs_stock.csv"
            ).groupby(["check", "status"]).size().to_string())
            print(check_total_probability(
                child, parent, report_fp=logs_dir / f"verify_{cname}_total_probability.csv"
            )["status"].value_counts().to_string())
            child.close()
    elif args.which == "rtps":
        old_id = int(manifest["AvgSA_03_disagg_eps4_rtps"]["calc_id"])
        centry = manifest["AvgSA_03_disagg_eps4_rtps_ml"]
        child = read(int(centry["calc_id"]))
        table = read_levels_table(wp1 / centry["levels_csv"])
        pos = target_positions(table)
        old = read(old_id)
        poes = list(old["oqparam"].poes_disagg)
        old.close()
        rtps = [-1.0 / np.log1p(-p) for p in poes]
        p_new = {}
        for p_ref, r in enumerate(rtps):
            k = [kk for kk in pos[pos.site == 0]["key"] if np.isclose(kk, r, rtol=1e-4)]
            p_new[p_ref] = int(pos[(pos.site == 0) & (pos.key == k[0])]["p_x"].iloc[0])
        for p_ref, pn in p_new.items():
            print(verify_against_reference(
                child, {old_id: (p_ref, pn)}, level_rtol=1e-9, rtol=1e-6,
                report_fp=logs_dir / f"verify_rtps_ml_vs_126_p{p_ref}.csv"
            ).groupby(["check", "status"]).size().to_string())
        child.close()
    parent.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phd_project.scripts.oq_multi_iml_disagg",
                                 description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one child disaggregation (cwd = the ini folder)")
    r.add_argument("--ini", required=True)
    r.add_argument("--levels", required=True, help="levels .npy, shape (N, M, P), in g")
    r.add_argument("--parent", required=True, help="calc id of the classical parent")
    r.set_defaults(func=_cli_run)
    v = sub.add_parser("verify", help="run the plan (1) datastore checks")
    v.add_argument("--which", choices=["imlgrid", "rtps", "selftest"], required=True)
    v.set_defaults(func=_cli_verify)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
