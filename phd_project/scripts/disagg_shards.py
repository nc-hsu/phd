"""Per-site storage for the IML-based disaggregation data.

Notebook 021 used to write one flat ``disagg_data[site][imt][iml] -> DataFrame``
pickle. For AvgSA([0, 3]) that file is 2.3 GB, and it caused two problems for the
record-selection pipeline downstream:

- **Provenance was all-or-nothing.** Every ``(site, iml)`` stripe's fingerprint
  hashed the whole monolith, so re-running the disaggregation for a *single* site
  marked all ~450 stripes stale and triggered hours of needless GCIM + selection.
- **Every consumer paid for every site.** Notebooks 031/032/033 unpickled the full
  2.3 GB even when only a handful of sites needed work -- and 033 never touches the
  disagg data at all.

So the data is stored as one pickle per site instead:

.. code-block:: text

    AvgSA_03_disagg_data_wp1sites_eps4/
        site_000.pickle               {imt: {iml: DataFrame}} for site 0
        site_000.pickle.manifest.json standard cache_utils sidecar
        ...
        _index.json                   {"sites": {"0": {"AvgSA": [0.27, ...]}}}
        _split_provenance.json        only when split from a legacy monolith

``_index.json`` is what makes the stale check cheap: it carries the **native float
iml keys** of every shard, so the set of wanted ``(site, iml)`` can be rebuilt
without opening a single shard.

Provenance is unchanged in kind -- every shard carries an ordinary
``cache_utils.write_manifest`` sidecar, and all shards of one build share the same
fingerprint dict (they come from the same disaggregation run).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from phd_project.scripts.cache_utils import (
    file_hash,
    manifest_matches,
    write_manifest,
)

INDEX_NAME = "_index.json"
SPLIT_PROVENANCE_NAME = "_split_provenance.json"
SHARD_GLOB = "site_*.pickle"

_SHARD_RE = re.compile(r"^site_(\d+)\.pickle$")


def shard_path(shard_dir, site: int) -> Path:
    """Canonical path of one site's shard. Zero-padded so the directory sorts."""
    return Path(shard_dir) / f"site_{int(site):03d}.pickle"


def site_from_shard_name(name: str) -> int:
    """Inverse of :func:`shard_path` for a bare filename."""
    m = _SHARD_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"not a shard filename: {name!r}")
    return int(m.group(1))


def shard_paths(shard_dir) -> list[Path]:
    """Every shard in the directory, ordered by site index."""
    return sorted(Path(shard_dir).glob(SHARD_GLOB), key=lambda p: site_from_shard_name(p.name))


# --------------------------------------------------------------------------- #
# writing                                                                      #
# --------------------------------------------------------------------------- #

def build_index(disagg_data: dict) -> dict:
    """``{"sites": {site_str: {imt: [iml, ...]}}}`` from a site-major disagg dict.

    The iml values are the **native float keys** of ``disagg_data[site][imt]``.
    JSON round-trips Python floats through ``repr``, so they come back bit-identical
    -- which matters, because those exact floats are the keys the selection pipeline
    uses for every ``(site, iml)``.
    """
    return {"sites": {str(site): {imt: sorted(iml_dict)
                                  for imt, iml_dict in imt_dict.items()}
                      for site, imt_dict in sorted(disagg_data.items())}}


def write_shards(shard_dir, disagg_data: dict, fp_dict: dict,
                 *, source_sha: str | None = None) -> list[Path]:
    """Write one pickle (+ manifest) per site, plus ``_index.json``.

    All shards share ``fp_dict``: they are produced by one disaggregation run, so
    they have one provenance. ``source_sha`` records the sha256 of the legacy
    monolith when the shards were split from one (see ``shard_disagg_data.py``).
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for site, imt_dict in sorted(disagg_data.items()):
        fp = shard_path(shard_dir, site)
        with open(fp, "wb") as f:
            pickle.dump(imt_dict, f)
        write_manifest(fp, fp_dict)
        written.append(fp)

    with open(shard_dir / INDEX_NAME, "w") as f:
        json.dump(build_index(disagg_data), f, indent=1)

    if source_sha is not None:
        with open(shard_dir / SPLIT_PROVENANCE_NAME, "w") as f:
            json.dump({"source_sha256": source_sha,
                       "written_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    return written


# --------------------------------------------------------------------------- #
# reading                                                                      #
# --------------------------------------------------------------------------- #

def read_index(shard_dir) -> dict[int, dict[str, list[float]]]:
    """``{site: {imt: [iml, ...]}}`` -- the shard contents, without opening a shard."""
    with open(Path(shard_dir) / INDEX_NAME) as f:
        raw = json.load(f)["sites"]
    return {int(site): {imt: [float(v) for v in imls] for imt, imls in imt_dict.items()}
            for site, imt_dict in raw.items()}


def read_split_provenance(shard_dir) -> dict | None:
    """The recorded monolith sha256, or None when the shards were built natively."""
    fp = Path(shard_dir) / SPLIT_PROVENANCE_NAME
    if not fp.is_file():
        return None
    with open(fp) as f:
        return json.load(f)


def load_shards(shard_dir, sites: Iterable[int] | None = None) -> dict:
    """Load the site-major disagg dict, restricted to ``sites``.

    ``sites=None`` loads every shard (the old monolith behaviour); an explicit
    iterable loads only those; ``sites=()`` loads nothing and returns ``{}`` -- used
    by callers that need the rest of the selection context but no disagg data.
    """
    shard_dir = Path(shard_dir)
    if sites is None:
        wanted = [site_from_shard_name(p.name) for p in shard_paths(shard_dir)]
    else:
        wanted = sorted(set(int(s) for s in sites))

    out = {}
    for site in wanted:
        fp = shard_path(shard_dir, site)
        if not fp.is_file():
            raise FileNotFoundError(f"No disagg shard for site {site}: {fp}")
        with open(fp, "rb") as f:
            out[site] = pickle.load(f)
    return out


def shards_digest(shard_dir) -> str:
    """One digest over the whole shard set, for artifacts that depend on all of it.

    The batch-level caches (``rd*_selection``, ``final_ensembles``) genuinely depend
    on every site in their batch, so they fingerprint this rather than a single
    shard. Built from the per-shard content hashes, which are memoised by
    ``cache_utils``, so repeated calls in one process are free.
    """
    h = hashlib.sha256()
    for p in shard_paths(shard_dir):
        h.update(p.name.encode())
        h.update(file_hash(p).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# provenance-guarded build                                                     #
# --------------------------------------------------------------------------- #

def shards_are_valid(shard_dir, fp_dict: dict) -> bool:
    """True iff the index exists, shards exist, and every manifest matches."""
    shard_dir = Path(shard_dir)
    if not (shard_dir / INDEX_NAME).is_file():
        return False
    paths = shard_paths(shard_dir)
    if not paths:
        return False
    return all(manifest_matches(p, fp_dict) for p in paths)


def load_or_compute_shards(shard_dir, fp_dict: dict,
                           compute_fn: Callable[[], dict],
                           *, force_recompute: bool = False) -> dict:
    """Shard-set sibling of :func:`cache_utils.load_or_compute`.

    Loads every shard when the set is present and its manifests match ``fp_dict``;
    otherwise runs ``compute_fn`` (which returns the full site-major dict), writes
    the shards and returns it. Same branches as ``load_or_compute``: force, absent,
    unmanaged/mismatched -> recompute; match -> load.

    ``compute_fn`` still builds every site in memory -- the collation reads one
    OpenQuake datastore per IML, each carrying all sites, so it cannot stream by
    site. Sharding pays off on the *read* side, which is what runs repeatedly.
    """
    shard_dir = Path(shard_dir)

    def _compute_and_save():
        result = compute_fn()
        write_shards(shard_dir, result, fp_dict)
        return result

    if force_recompute:
        return _compute_and_save()

    if not shards_are_valid(shard_dir, fp_dict):
        if shard_paths(shard_dir):
            print(f"[cache] disagg shards in '{shard_dir.name}' are stale or "
                  f"unmanaged; recomputing to establish provenance.")
        return _compute_and_save()

    print(f"[cache] disagg shards in '{shard_dir.name}' loaded (inputs match).")
    return load_shards(shard_dir)
