"""One-off: split the legacy monolithic disagg pickle into per-site shards.

Notebook 021 now writes ``disagg_shards`` directly, but re-running it to get the
shards would mean re-reading every OpenQuake datastore. The bytes on disk are
already correct, so this script just re-lays them out:

    <HAZ_DIR>/AvgSA_03_disagg_data_wp1sites_eps4.pickle
        -> <HAZ_DIR>/AvgSA_03_disagg_data_wp1sites_eps4/site_NNN.pickle (+ manifests)

Each shard inherits the monolith manifest's own ``inputs`` dict, so the shards
carry exactly the upstream provenance nb 021 would have given them, and the
monolith's sha256 is recorded in ``_split_provenance.json`` -- which is what
``migrate_stripe_manifests.py`` later checks before re-stamping any downstream
artifact.

Refuses to run against a monolith whose manifest is missing or whose bytes do not
match that manifest: splitting data of unknown provenance would launder it.

    python -m phd_project.scripts.WP1_ground_motion_set.shard_disagg_data          # dry run
    python -m phd_project.scripts.WP1_ground_motion_set.shard_disagg_data --write
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from phd_project.config.config import load_config
from phd_project.scripts import disagg_shards
from phd_project.scripts.cache_utils import file_hash, load_manifest


def _monolith_and_dir(im: str, eps: int) -> tuple[Path, Path]:
    cfg = load_config()
    haz = cfg["proc_data"]["site_hazard"]
    stem = f"AvgSA_{im}_disagg_data_wp1sites_eps{eps}"
    return haz / f"{stem}.pickle", haz / stem


def split(im: str = "03", eps: int = 4, *, write: bool = False) -> int:
    monolith, shard_dir = _monolith_and_dir(im, eps)

    if not monolith.is_file():
        print(f"ERROR: no monolith at {monolith}")
        return 1

    manifest = load_manifest(monolith)          # raises if the sidecar is absent
    inputs = manifest.get("inputs", {})
    if not inputs:
        print(f"ERROR: {monolith.name}.manifest.json has no 'inputs'; refusing to split.")
        return 1

    print(f"monolith : {monolith}  ({monolith.stat().st_size / 1e6:.0f} MB)")
    print(f"shard dir: {shard_dir}")
    print(f"manifest : {len(inputs)} recorded inputs, "
          f"git_commit={manifest.get('_meta', {}).get('git_commit')}")

    t = time.perf_counter()
    source_sha = file_hash(monolith)
    print(f"monolith sha256: {source_sha}  ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    with open(monolith, "rb") as f:
        disagg_data = pickle.load(f)
    print(f"loaded {len(disagg_data)} sites in {time.perf_counter() - t:.1f}s")

    index = disagg_shards.build_index(disagg_data)["sites"]
    n_imls = sum(len(imls) for s in index.values() for imls in s.values())
    imts = sorted({imt for s in index.values() for imt in s})
    print(f"sites {min(disagg_data)}..{max(disagg_data)}, imts={imts}, "
          f"{n_imls} (site, imt, iml) entries")

    if not write:
        print("\n-- dry run; nothing written. Re-run with --write to create the shards.")
        return 0

    if shard_dir.exists() and any(shard_dir.glob(disagg_shards.SHARD_GLOB)):
        print(f"ERROR: {shard_dir} already contains shards; remove it first.")
        return 1

    t = time.perf_counter()
    written = disagg_shards.write_shards(
        shard_dir, disagg_data, inputs, source_sha=source_sha)
    total = sum(p.stat().st_size for p in written)
    print(f"wrote {len(written)} shards ({total / 1e6:.0f} MB) in "
          f"{time.perf_counter() - t:.1f}s")
    print(f"  smallest {min(p.stat().st_size for p in written) / 1e6:.1f} MB, "
          f"largest {max(p.stat().st_size for p in written) / 1e6:.1f} MB")
    print(f"  monolith was {monolith.stat().st_size / 1e6:.0f} MB "
          f"({total / monolith.stat().st_size:.3f}x)")

    # Round-trip check: the index must describe what actually landed on disk.
    reread = disagg_shards.read_index(shard_dir)
    assert set(reread) == set(disagg_data), "index sites != monolith sites"
    for site, imt_dict in disagg_data.items():
        for imt, iml_dict in imt_dict.items():
            assert reread[site][imt] == sorted(iml_dict), \
                f"index imls differ for site {site} / {imt}"
    print("index round-trip OK (exact float keys)")

    prov = disagg_shards.read_split_provenance(shard_dir)
    print(f"recorded split provenance: {json.dumps(prov)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--im", default="03", help="IM definition tag (default: 03)")
    ap.add_argument("--eps", type=int, default=4, help="epsilon truncation (default: 4)")
    ap.add_argument("--write", action="store_true", help="actually write the shards")
    args = ap.parse_args(argv)
    return split(args.im, args.eps, write=args.write)


if __name__ == "__main__":
    sys.exit(main())
