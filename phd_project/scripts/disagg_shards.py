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
        _content_hashes.json          {"sites": {"0": {"AvgSA": {"0.27": "<sha256>"}}}}
        _split_provenance.json        only when split from a legacy monolith

``_index.json`` is what makes the stale check cheap: it carries the **native float
iml keys** of every shard, so the set of wanted ``(site, iml)`` can be rebuilt
without opening a single shard.

``_content_hashes.json`` is what makes it *stripe*-precise. Sharding by site was not
enough on its own: an OpenQuake ``iml_disagg`` run covers all 60 sites at one iml, so
adding one MSA stripe rewrites every shard and a whole-shard hash marks all ~500
already-selected stripes stale. The per-stripe hashes let a fingerprint name the one
DataFrame it actually depends on, and being a small JSON they cost no shard reads --
see :func:`stripe_content_hash`.

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

import numpy as np

from phd_project.scripts.cache_utils import (
    dataframe_content_hash,
    file_hash,
    manifest_matches,
    write_manifest,
)

INDEX_NAME = "_index.json"
CONTENT_HASHES_NAME = "_content_hashes.json"
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


def build_content_hashes(disagg_data: dict) -> dict:
    """``{"sites": {site_str: {imt: {iml_repr: sha256}}}}`` -- one hash per stripe.

    Same shape as :func:`build_index`, but the leaf is a content hash of that
    ``(site, imt, iml)`` DataFrame instead of a bare iml list. This is what makes
    the downstream stale check *stripe*-precise rather than shard-precise: an
    OpenQuake ``iml_disagg`` run covers ALL sites at ONE iml, so adding a stripe
    rewrites every shard, and a whole-shard hash then marks every already-selected
    stripe stale. Hashing per stripe means only the stripes whose own
    disaggregation moved go stale.

    Written next to the shards so the check costs one small JSON read rather than
    opening 60 x 68 MB pickles -- see :func:`stripe_content_hash`.

    Keys are ``repr(float)`` of the native iml, exactly as :func:`build_index`
    stores them, so ``float(key)`` round-trips bit-identically.
    """
    return {"sites": {str(site): {imt: {repr(iml): dataframe_content_hash(df)
                                        for iml, df in sorted(iml_dict.items())}
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

    with open(shard_dir / CONTENT_HASHES_NAME, "w") as f:
        json.dump(build_content_hashes(disagg_data), f, indent=1)

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


# Parsed _content_hashes.json, keyed by (path, size, mtime_ns) -- same memo scheme
# and same rationale as cache_utils._FILE_HASH_CACHE. The stale check looks up 516
# stripes against the SAME index, so without this it would re-parse the JSON 516
# times per notebook. In-process only; nothing is persisted.
_CONTENT_HASH_CACHE: dict[tuple[str, int, int], dict] = {}


def content_hashes_path(shard_dir) -> Path:
    """Path of the per-stripe content-hash index for a shard set."""
    return Path(shard_dir) / CONTENT_HASHES_NAME


def read_content_hashes(shard_dir) -> dict[int, dict[str, dict[float, str]]]:
    """``{site: {imt: {iml: sha256}}}`` -- every stripe's content hash, no shard opened.

    Raises :class:`FileNotFoundError` when the index is absent rather than falling
    back to a whole-shard hash: a silent fallback would quietly restore the
    all-stripes-stale behaviour this index exists to prevent (cf. the missing-shard
    guard in ``gm_selection._disagg_fingerprint_input``).
    """
    fp = content_hashes_path(shard_dir)
    if not fp.is_file():
        raise FileNotFoundError(
            f"No {CONTENT_HASHES_NAME} in {shard_dir}.\n"
            f"Stripe fingerprints are computed from it, so the stale check cannot "
            f"run without it. For a shard set written before this index existed, "
            f"backfill it once with disagg_shards.write_content_hashes(shard_dir); "
            f"shards written by write_shards() get it automatically.")
    st = fp.stat()
    key = (str(fp), st.st_size, st.st_mtime_ns)
    parsed = _CONTENT_HASH_CACHE.get(key)
    if parsed is None:
        with open(fp) as f:
            raw = json.load(f)["sites"]
        parsed = _CONTENT_HASH_CACHE[key] = {
            int(site): {imt: {float(iml): h for iml, h in iml_dict.items()}
                        for imt, iml_dict in imt_dict.items()}
            for site, imt_dict in raw.items()}
    return parsed


def clear_content_hash_cache() -> None:
    """Drop the in-process content-hash index memo."""
    _CONTENT_HASH_CACHE.clear()


def stripe_content_hash(shard_dir, site: int, imt: str, iml: float) -> str:
    """Content hash of one ``(site, imt, iml)`` disaggregation DataFrame.

    The native float key is matched exactly first; ``np.isclose`` is the fallback
    for an iml that reached the caller through a JSON list with slightly different
    rounding (same tolerance rule as ``gm_selection.get_poe_from_disaggstats``).
    """
    by_imt = read_content_hashes(shard_dir).get(int(site))
    if by_imt is None:
        raise KeyError(f"site {site} is not in {CONTENT_HASHES_NAME} ({shard_dir})")
    hashes = by_imt.get(imt)
    if hashes is None:
        raise KeyError(f"site {site} has no imt {imt!r} in {CONTENT_HASHES_NAME}")
    h = hashes.get(float(iml))
    if h is None:
        match = next((k for k in hashes if np.isclose(k, iml)), None)
        if match is None:
            raise KeyError(f"site {site} has no disagg at iml {iml} for imt {imt!r}")
        h = hashes[match]
    return h


def write_content_hashes(shard_dir) -> Path:
    """Backfill :data:`CONTENT_HASHES_NAME` for a shard set written without it.

    Opens every shard once (GB-scale, minutes) and writes the index that
    :func:`write_shards` now emits automatically. Needed for shard sets built
    before the per-stripe index existed; running it does not change a single shard
    byte, so it never affects a stripe's provenance.
    """
    shard_dir = Path(shard_dir)
    out = {"sites": {}}
    for fp in shard_paths(shard_dir):
        site = site_from_shard_name(fp.name)
        with open(fp, "rb") as f:
            imt_dict = pickle.load(f)
        out["sites"][str(site)] = {
            imt: {repr(iml): dataframe_content_hash(df)
                  for iml, df in sorted(iml_dict.items())}
            for imt, iml_dict in imt_dict.items()}
    target = content_hashes_path(shard_dir)
    with open(target, "w") as f:
        json.dump(out, f, indent=1)
    clear_content_hash_cache()
    return target


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
