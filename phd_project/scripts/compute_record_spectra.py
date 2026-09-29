"""Compute 5 %-damped pseudo-spectral acceleration for every ground-motion record JSON in a folder.

WHAT IT DOES
------------
Reads every ``*.json`` record directly inside ``input_dir`` (subfolders are ignored), computes its
pseudo-spectral acceleration (PSA) on a fine period grid (default 0.02-4.00 s every 0.01 s, 399
periods), and saves one pandas DataFrame as a pickle in ``output_dir``:

    index    record_id  = the JSON file name without ".json", e.g. "1000040__H1" or
                          "GR-1995-0047_PATA_0__U". This is the same name the MSA collapse-flag
                          files list under ``record_names`` (there with ".json" appended).
    columns  the periods [s] as floats (0.02, 0.03, ..., 4.0)
    values   PSA in the units of the record (the project's records are in g, so PSA is in g)

``df.attrs`` records how the spectra were made (damping, period grid, source folder, start time,
and any records that failed), so the file documents itself.

WHY NOT standes.groundmotion.response_spectrum DIRECTLY
-------------------------------------------------------
``standes`` solves the oscillator exactly: it steps the state [displacement, velocity] forward
with the matrix exponential of the equation of motion, once per time step, assuming the ground
acceleration is constant over each step (Wang 1996; the Kalkan Matlab code). The maths is right,
but it runs as a pure-Python loop over periods x time steps. At 399 periods and 20-50 k steps per
record that is about a minute per record, far too slow for ~2,300 records.

This script uses the SAME recursion, rewritten as a second-order digital filter and run by
``scipy.signal.lfilter`` (compiled C). For one period, standes computes

    y_k = Ae y_{k-1} + b a_k,      y_0 = 0,      k = 1 .. N-1

with Ae = expm(A dt), b = (Ae - I) A^-1 [0, 1]^T and A = [[0, 1], [-w^2, -2 xi w]]. Taking the
z-transform, the displacement (first component of y) is the input filtered by

            b1 + (a12 b2 - a22 b1) z^-1
    H(z) = -------------------------------------------------
            1 - (a11 + a22) z^-1 + (a11 a22 - a12 a21) z^-2

(the adjugate of I - Ae z^-1, divided by its determinant). The arithmetic is identical, so the
results agree with standes to round-off (``--verify`` checks this) at a small fraction of the cost:
well under a second per record. Note that standes starts its loop at k = 1, so the first sample
of the record is never used. The same is done here (u[0] = 0) so that the two match exactly.

PSA = w^2 max|displacement|, exactly as standes defines it.

PARALLELISM AND RESTARTS
------------------------
Records are independent, so they are shared out over ``--n-cores`` worker processes. Every
``--checkpoint-every`` records the results so far are written to ``<output>.partial.pickle``. If
the run is interrupted, re-running with ``--resume`` skips the records already in that file. The
partial file is removed once the final file has been written.

USAGE
-----
    python -m phd_project.scripts.compute_record_spectra INPUT_DIR OUTPUT_DIR --n-cores 8
    python phd_project/scripts/compute_record_spectra.py INPUT_DIR OUTPUT_DIR --n-cores 8 --verify

Run ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.signal import lfilter


# ---------------------------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------------------------

T_MIN, T_MAX, T_STEP = 0.02, 4.00, 0.01     # period grid [s], both ends included
DAMPING = 0.05                              # ratio of critical damping
OUTPUT_NAME = "record_spectra_psa_T0.02-4.00_dT0.01.pickle"
CHECKPOINT_EVERY = 100                      # records between partial saves


def period_grid(t_min: float, t_max: float, t_step: float) -> np.ndarray:
    """The periods t_min, t_min + t_step, ..., t_max, rounded so that they print cleanly.

    Built from integer step counts rather than np.arange on floats, because floating-point
    arange can drop or duplicate the last point (e.g. 4.0 may come out as 3.9999999).
    """
    n = int(round((t_max - t_min) / t_step)) + 1
    decimals = max(0, -int(np.floor(np.log10(t_step)))) + 2
    return np.round(t_min + t_step * np.arange(n), decimals)


# ---------------------------------------------------------------------------------------------
# 2. The oscillator as a digital filter
# ---------------------------------------------------------------------------------------------

def filter_coefficients(period: float, dt: float, damping: float) -> tuple[np.ndarray, np.ndarray]:
    """(numerator, denominator) of the filter that maps ground acceleration to displacement.

    Built from the same matrices as standes.groundmotion.response_spectrum (see module doc):
    Ae = expm(A dt) and b = (Ae - I) A^-1 [0, 1]^T.
    """
    omega = 2.0 * np.pi / period
    A = np.array([[0.0, 1.0],
                  [-omega ** 2, -2.0 * damping * omega]])
    Ae = expm(A * dt)
    # standes uses pinv(A); A is invertible for any period > 0, so inv gives the same b.
    b = (Ae - np.eye(2)) @ np.linalg.solve(A, np.array([0.0, 1.0]))

    (a11, a12), (a21, a22) = Ae
    b1, b2 = b
    numerator = np.array([b1, a12 * b2 - a22 * b1])
    denominator = np.array([1.0, -(a11 + a22), a11 * a22 - a12 * a21])
    return numerator, denominator


def psa_spectrum(acc: np.ndarray, dt: float, periods: np.ndarray, damping: float) -> np.ndarray:
    """Pseudo-spectral acceleration at every period: w^2 * max |relative displacement|."""
    u = np.asarray(acc, dtype=float).copy()
    u[0] = 0.0                     # standes' loop starts at k = 1: the first sample is unused
    psa = np.empty(len(periods))
    for i, T in enumerate(periods):
        num, den = filter_coefficients(T, dt, damping)
        displacement = lfilter(num, den, u)
        psa[i] = (2.0 * np.pi / T) ** 2 * np.max(np.abs(displacement))
    return psa


# ---------------------------------------------------------------------------------------------
# 3. One record (this is what each worker process runs)
# ---------------------------------------------------------------------------------------------

def load_record(fp: Path) -> tuple[np.ndarray, float, dict]:
    """(acceleration, dt, metadata) of one record JSON.

    The JSON layout is the project's (standes) one: "record" holds the samples and "dt" the
    time step; "units" and "record_type" are checked, since a velocity or displacement
    history would silently give a meaningless spectrum.
    """
    with open(fp, "r") as f:
        rec = json.load(f)
    if rec.get("record_type", "acc") != "acc":
        raise ValueError(f"record_type is {rec.get('record_type')!r}, expected 'acc'")
    acc = np.asarray(rec["record"], dtype=float)
    dt = float(rec["dt"])
    if acc.ndim != 1 or len(acc) < 2 or not np.isfinite(acc).all():
        raise ValueError("record is not a finite 1-D series of at least 2 samples")
    if not dt > 0:
        raise ValueError(f"dt = {dt} is not positive")
    return acc, dt, {"units": rec.get("units"), "dt": dt, "n_steps": len(acc)}


def process_record(fp: str, periods: np.ndarray, damping: float):
    """Worker entry point. Returns (record_id, psa, meta, error); psa is None on failure.

    Errors are caught and returned rather than raised, so that one bad file cannot stop an
    overnight run. They are listed at the end and stored in df.attrs["failed"].
    """
    fp = Path(fp)
    try:
        acc, dt, meta = load_record(fp)
        return fp.stem, psa_spectrum(acc, dt, periods, damping), meta, None
    except Exception as exc:                                  # noqa: BLE001 - reported below
        return fp.stem, None, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------------------------
# 4. Saving
# ---------------------------------------------------------------------------------------------

def to_frame(results: dict, periods: np.ndarray, attrs: dict) -> pd.DataFrame:
    """One row per record (index record_id, sorted), one column per period."""
    ids = sorted(results)
    df = pd.DataFrame(np.vstack([results[i] for i in ids]) if ids else np.empty((0, len(periods))),
                      index=pd.Index(ids, name="record_id"),
                      columns=pd.Index(periods, name="period_s"))
    df.attrs.update(attrs)
    return df


def write_pickle(df: pd.DataFrame, fp: Path) -> None:
    """Write via a temporary file and rename, so an interruption never leaves a half-written file."""
    tmp = fp.with_name(fp.name + ".tmp")
    df.to_pickle(tmp)
    os.replace(tmp, fp)


# ---------------------------------------------------------------------------------------------
# 5. Self-check against standes
# ---------------------------------------------------------------------------------------------

def verify_against_standes(fp: Path, damping: float) -> None:
    """Compare this script's PSA with standes.groundmotion.response_spectrum on one record.

    standes is slow, so only a handful of periods spread over the grid are compared.
    """
    from standes.groundmotion import response_spectrum

    check_periods = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    acc, dt, _ = load_record(fp)
    t0 = time.perf_counter()
    ours = psa_spectrum(acc, dt, check_periods, damping)
    t_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    theirs, _, _ = response_spectrum(acc, dt, check_periods, xi=damping)
    t_theirs = time.perf_counter() - t0

    rel = np.max(np.abs(ours / theirs - 1.0))
    print(f"verify on {fp.name}: max relative difference to standes = {rel:.1e} "
          f"(this script {t_ours:.3f} s, standes {t_theirs:.1f} s, for {len(check_periods)} periods)")
    if rel > 1e-8:
        raise SystemExit("verification FAILED: spectra differ from standes by more than 1e-8")


# ---------------------------------------------------------------------------------------------
# 6. Command line
# ---------------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute PSA spectra for every record JSON in a folder and save them as one "
                    "pickled DataFrame (index record_id, columns period).")
    p.add_argument("input_dir", type=Path, help="folder containing the record *.json files")
    p.add_argument("output_dir", type=Path, help="folder the DataFrame pickle is written to")
    p.add_argument("--n-cores", type=int, default=1,
                   help="number of worker processes (default 1)")
    p.add_argument("--damping", type=float, default=DAMPING,
                   help=f"damping ratio (default {DAMPING})")
    p.add_argument("--t-min", type=float, default=T_MIN, help=f"first period [s] (default {T_MIN})")
    p.add_argument("--t-max", type=float, default=T_MAX, help=f"last period [s] (default {T_MAX})")
    p.add_argument("--t-step", type=float, default=T_STEP, help=f"period step [s] (default {T_STEP})")
    p.add_argument("--output-name", default=OUTPUT_NAME,
                   help=f"file name of the pickle (default {OUTPUT_NAME})")
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                   help=f"save partial results every N records (default {CHECKPOINT_EVERY})")
    p.add_argument("--resume", action="store_true",
                   help="skip records already in the partial file of an interrupted run")
    p.add_argument("--verify", action="store_true",
                   help="before starting, check the spectra against standes on the first record")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N records (for a quick trial)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input_dir.is_dir():
        raise SystemExit(f"input folder not found: {args.input_dir}")
    if args.n_cores < 1:
        raise SystemExit("--n-cores must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_fp = args.output_dir / args.output_name
    partial_fp = args.output_dir / (Path(args.output_name).stem + ".partial.pickle")
    periods = period_grid(args.t_min, args.t_max, args.t_step)

    files = sorted(args.input_dir.glob("*.json"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no *.json files in {args.input_dir}")
    # record_id is the file stem, so two files may not share one
    stems = [fp.stem for fp in files]
    if len(set(stems)) != len(stems):
        raise SystemExit("two record files share a name - record_id would not be unique")

    if args.verify:
        verify_against_standes(files[0], args.damping)

    # ---- resume: reuse whatever an interrupted run already computed ------------------------
    results, meta, failed = {}, {}, {}
    if args.resume and partial_fp.is_file():
        prev = pd.read_pickle(partial_fp)
        same_grid = np.array_equal(prev.columns.to_numpy(dtype=float), periods)
        same_damping = prev.attrs.get("damping") == args.damping
        if not (same_grid and same_damping):
            raise SystemExit(f"{partial_fp.name} was made with a different period grid or damping; "
                             "delete it or run without --resume")
        results = {rid: row.to_numpy() for rid, row in prev.iterrows()}
        meta = dict(prev.attrs.get("record_meta", {}))
        print(f"resuming: {len(results)} records already done in {partial_fp.name}")
    todo = [fp for fp in files if fp.stem not in results]

    attrs = {"quantity": "PSA (pseudo-spectral acceleration), units of the input record",
             "damping": args.damping, "periods_s": periods.tolist(),
             "source_dir": str(args.input_dir.resolve()),
             "method": "exact piecewise-constant-excitation recursion (as standes."
                       "groundmotion.response_spectrum), run as a scipy.signal.lfilter IIR filter",
             "started": datetime.now().isoformat(timespec="seconds")}

    print(f"{len(files)} records in {args.input_dir}, {len(todo)} to compute, "
          f"{len(periods)} periods ({periods[0]}-{periods[-1]} s), damping {args.damping}, "
          f"{args.n_cores} core(s)")
    print(f"output: {out_fp}", flush=True)

    def checkpoint():
        write_pickle(to_frame(results, periods, {**attrs, "record_meta": meta,
                                                 "failed": failed}), partial_fp)

    # ---- the work: independent records shared over the worker processes -------------------
    t_start = time.perf_counter()
    done_since_ckpt = 0
    with ProcessPoolExecutor(max_workers=args.n_cores) as pool:
        futures = [pool.submit(process_record, str(fp), periods, args.damping) for fp in todo]
        for n_done, fut in enumerate(as_completed(futures), start=1):
            rid, psa, rec_meta, err = fut.result()
            if err is None:
                results[rid] = psa
                meta[rid] = rec_meta
            else:
                failed[rid] = err
                print(f"  FAILED {rid}: {err}", flush=True)

            done_since_ckpt += 1
            if done_since_ckpt >= args.checkpoint_every:
                checkpoint()
                done_since_ckpt = 0
            if n_done % 25 == 0 or n_done == len(todo):
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n_done * (len(todo) - n_done)
                print(f"  {n_done}/{len(todo)} records, {elapsed / 60:.1f} min elapsed, "
                      f"~{eta / 60:.1f} min to go", flush=True)

    # ---- save ------------------------------------------------------------------------------
    units = {m["units"] for m in meta.values()}
    if len(units) > 1:
        print(f"WARNING: records are in mixed units {sorted(map(str, units))}; "
              "see df.attrs['record_meta'] per record")
    attrs["finished"] = datetime.now().isoformat(timespec="seconds")
    df = to_frame(results, periods, {**attrs, "record_meta": meta, "failed": failed})
    write_pickle(df, out_fp)
    if partial_fp.is_file():
        partial_fp.unlink()

    print(f"wrote {out_fp}: {df.shape[0]} records x {df.shape[1]} periods "
          f"in {(time.perf_counter() - t_start) / 60:.1f} min")
    if failed:
        print(f"{len(failed)} record(s) failed (listed in df.attrs['failed']):")
        for rid, err in failed.items():
            print(f"  {rid}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":          # required on Windows: worker processes re-import this module
    sys.exit(main())
