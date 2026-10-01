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

TWO METHODS: "fft" (DEFAULT) AND "zoh"
--------------------------------------
The recursion above is exact for an input that is held CONSTANT over each time step (a "zero-order
hold", hence "zoh"). A real record is a sampled version of a smooth signal, not a staircase, so the
error depends on how coarse the sampling is compared with the oscillator period:

* ``method="zoh"`` (the original). The recursion is run directly on the record at its own dt.
  Accurate when dt is small compared with T (the 200 Hz ESM records, dt = 0.005 s: typically ~1 %
  or less). For coarse records it is biased: the staircase smears the high frequencies and adds a
  half-step delay, so PSA is LOW at short and moderate periods, and erratic once T approaches dt.
  At dt = 0.05 s (some NGA-Sub records) PSA was ~13 % low at T = 0.2 s and ~60 % low at
  T = 0.05 s, which made 47 NGA-Sub records look like poor matches to the database (nb 043 §9).
  This is also what standes.groundmotion.response_spectrum does.
* ``method="fft"`` (the default). Before the same recursion, the record is first resampled to a finer
  step with band-limited (FFT, i.e. sinc) interpolation. This is the interpolation implied by the
  sampling theorem, so no content is added or removed below the original Nyquist frequency. The
  record is upsampled by the smallest integer factor k with dt / k <= ``fft_max_dt`` (default
  0.002 s = T_min / 10 for the 0.02 s shortest period), so records already that fine are unchanged.
  It is zero-padded before the FFT, so that the FFT's implied wrap-around does not join the end of
  the record to its start. Checked 2026-10-01 against the NGA-Sub flatfile on 254 records at dt =
  0.005-0.05 s: median MSLE ~1e-6 (agreement to ~0.1 %) at every dt, against 0.04 for "zoh" at dt =
  0.05 s. So the flatfile spectra were evidently computed this way. For the 200 Hz ESM records the
  two methods differ by about 1 % (at most ~9 % at one period).

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
from scipy.fft import next_fast_len
from scipy.linalg import expm
from scipy.signal import lfilter, resample


# ---------------------------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------------------------

T_MIN, T_MAX, T_STEP = 0.02, 6.00, 0.01     # period grid [s], both ends included
DAMPING = 0.05                              # ratio of critical damping
OUTPUT_NAME = f"record_spectra_psa_T{T_MIN}-{T_MAX}_dT{T_STEP}.pickle"
CHECKPOINT_EVERY = 100                      # records between partial saves
METHODS = ("fft", "zoh")                    # see "TWO METHODS" in the module docstring
DEFAULT_METHOD = "fft"
FFT_MAX_DT = 0.002                          # [s] "fft" upsamples until dt <= this (T_MIN / 10)


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


def fft_upsample(acc: np.ndarray, dt: float, max_dt: float = FFT_MAX_DT) -> tuple[np.ndarray, float]:
    """Band-limited (FFT) resampling of a record to a step of at most ``max_dt``.

    The record is upsampled by the smallest integer factor k with dt / k <= max_dt. An integer
    factor keeps every original sample where it was (the new series passes through them) and
    keeps the same start time. k = 1 returns the record unchanged.

    Before the FFT the record is padded with zeros: by at least 10 % of its length (and at least
    64 samples), up to a length the FFT handles fast. Without the padding the FFT treats the record
    as periodic and joins its last sample to its first, which rings at both ends if they differ.
    The padding is cut off again afterwards, so the output has (n - 1) k + 1 samples and spans the
    same duration as the input.
    """
    acc = np.asarray(acc, dtype=float)
    k = int(np.ceil(dt / max_dt - 1e-9))        # 1e-9: dt = 0.004, max_dt = 0.002 -> k = 2, not 3
    if k <= 1:
        return acc, dt
    n = len(acc)
    n_pad = next_fast_len(n + max(n // 10, 64))
    padded = np.zeros(n_pad)
    padded[:n] = acc
    fine = resample(padded, n_pad * k)          # sinc interpolation, done in the frequency domain
    return fine[:(n - 1) * k + 1], dt / k


def psa_spectrum(acc: np.ndarray, dt: float, periods: np.ndarray, damping: float,
                 method: str = DEFAULT_METHOD, fft_max_dt: float = FFT_MAX_DT) -> np.ndarray:
    """Pseudo-spectral acceleration at every period: w^2 * max |relative displacement|.

    method
        "fft" (default): resample the record with ``fft_upsample`` to a step of at most
              ``fft_max_dt`` (band-limited / sinc interpolation), then run the exact recursion.
              Accurate at any record dt; matches the NGA-Sub flatfile spectra to ~0.1 %.
        "zoh": the original. Run the exact recursion on the record at its own dt, which treats
              the input as constant over each step. Identical to standes.groundmotion.
              response_spectrum. Accurate only when dt << T: biased low at short and moderate
              periods for coarse records (e.g. ~13 % at T = 0.2 s and ~60 % at T = 0.05 s when
              dt = 0.05 s).
    See "TWO METHODS" in the module docstring for the details and the check against the flatfile.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    u = np.asarray(acc, dtype=float).copy()
    if method == "fft":
        u, dt = fft_upsample(u, dt, fft_max_dt)
    else:
        u[0] = 0.0                 # standes' loop starts at k = 1: the first sample is unused
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


def process_record(fp: str, periods: np.ndarray, damping: float,
                   method: str = DEFAULT_METHOD, fft_max_dt: float = FFT_MAX_DT):
    """Worker entry point. Returns (record_id, psa, meta, error); psa is None on failure.

    Errors are caught and returned rather than raised, so that one bad file cannot stop an
    overnight run. They are listed at the end and stored in df.attrs["failed"].

    meta also holds the JSON's modification time (``json_mtime_ns``), so that ``update_spectra``
    can tell later whether the JSON was rewritten (e.g. by a reconversion) after its spectrum was
    computed.
    """
    fp = Path(fp)
    try:
        acc, dt, meta = load_record(fp)
        meta["json_mtime_ns"] = fp.stat().st_mtime_ns
        return fp.stem, psa_spectrum(acc, dt, periods, damping, method, fft_max_dt), meta, None
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

    standes is slow, so only a handful of periods spread over the grid are compared. standes uses
    the "zoh" scheme, so that is the method checked here: it confirms the filter reproduces the
    recursion. "fft" runs the same filter on the resampled record, so it is covered as well.
    """
    from standes.groundmotion import response_spectrum

    check_periods = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    acc, dt, _ = load_record(fp)
    t0 = time.perf_counter()
    ours = psa_spectrum(acc, dt, check_periods, damping, method="zoh")
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
# 6. Incremental update of an existing spectra file (used by nb 042)
# ---------------------------------------------------------------------------------------------

def _method_label(method: str, fft_max_dt: float) -> str:
    """Human-readable description of the method, stored in df.attrs["method"]."""
    if method == "fft":
        return (f"band-limited (FFT) resampling to dt <= {fft_max_dt} s, then the exact "
                "piecewise-constant-excitation recursion, run as a scipy.signal.lfilter IIR filter")
    return ("exact piecewise-constant-excitation recursion (as standes.groundmotion."
            "response_spectrum), run as a scipy.signal.lfilter IIR filter")


def _stored_method(attrs: dict) -> str:
    """The method a spectra file was made with. Files written before the method option existed
    have no "psa_method" attribute; they were all made with the original "zoh" scheme."""
    if "psa_method" in attrs:
        return attrs["psa_method"]
    return "zoh" if "piecewise-constant" in str(attrs.get("method", "")) else "unknown"


def update_spectra(records_dir: str | Path, spectra_fp: str | Path,
                   method: str = DEFAULT_METHOD, n_cores: int = 1, damping: float = DAMPING,
                   t_min: float = T_MIN, t_max: float = T_MAX, t_step: float = T_STEP,
                   fft_max_dt: float = FFT_MAX_DT) -> pd.DataFrame:
    """Bring the spectra file in line with the record JSONs in ``records_dir``, and return it.

    The record JSONs are the ``*.json`` files directly in ``records_dir`` (subfolders, such as a
    backup folder of superseded JSONs, are ignored). Spectra and JSONs are matched by name:
    record_id = JSON file stem.

    1. If ``spectra_fp`` exists it is read. Its spectra are reused only if they were made with the
       SAME settings (method, fft_max_dt for "fft", damping and period grid). Otherwise they are
       not comparable to new ones: the reason is printed and every spectrum is recomputed.
    2. Spectra without a JSON are removed, and their record_ids are printed.
    3. Spectra whose JSON has changed since the spectrum was computed (its modification time
       differs from the one stored with the spectrum, or none was stored) are recomputed. This is
       what makes a reconversion of existing JSONs (e.g. the ESM U/V fix) reach the spectra, even
       though the names are unchanged. They are reported as "stale".
    4. Every JSON without a spectrum is computed and added.

    The file is then written back (atomically). df.attrs["record_meta"] holds dt, n_steps, units
    and json_mtime_ns per record; df.attrs["failed"] lists records whose JSON could not be used.
    Records that fail are not added; they are reported and retried on the next call.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    records_dir, spectra_fp = Path(records_dir), Path(spectra_fp)
    if not records_dir.is_dir():
        raise FileNotFoundError(f"record folder not found: {records_dir}")
    periods = period_grid(t_min, t_max, t_step)
    files = {fp.stem: fp for fp in sorted(records_dir.glob("*.json"))}

    # ---- 1. the existing file, if its spectra are comparable with new ones -------------------
    results, meta = {}, {}
    if spectra_fp.is_file():
        prev = pd.read_pickle(spectra_fp)
        prev_method = _stored_method(prev.attrs)
        mismatch = []
        if prev_method != method:
            mismatch.append(f"method {prev_method!r} (requested {method!r})")
        if method == "fft" and prev.attrs.get("fft_max_dt") != fft_max_dt:
            mismatch.append(f"fft_max_dt {prev.attrs.get('fft_max_dt')} (requested {fft_max_dt})")
        if prev.attrs.get("damping") != damping:
            mismatch.append(f"damping {prev.attrs.get('damping')} (requested {damping})")
        if not np.array_equal(prev.columns.to_numpy(dtype=float), periods):
            mismatch.append("a different period grid")
        if mismatch:
            print(f"{spectra_fp.name} was made with {', '.join(mismatch)}: "
                  f"all {len(prev)} spectra are discarded and recomputed")
        else:
            results = {rid: row.to_numpy() for rid, row in prev.iterrows()}
            meta = dict(prev.attrs.get("record_meta", {}))
            print(f"read {spectra_fp.name}: {len(results)} spectra ({method})")
    else:
        print(f"{spectra_fp} does not exist yet: all spectra are computed")

    # ---- 2. spectra without a JSON are removed ----------------------------------------------
    removed = sorted(set(results) - set(files))
    for rid in removed:
        results.pop(rid)
        meta.pop(rid, None)
    print(f"\nspectra removed (no JSON): {len(removed)}")
    for rid in removed:
        print(f"  {rid}")

    # ---- 3. + 4. stale spectra and JSONs without a spectrum ---------------------------------
    stale = sorted(rid for rid in results
                   if meta.get(rid, {}).get("json_mtime_ns") != files[rid].stat().st_mtime_ns)
    for rid in stale:
        results.pop(rid)
    missing = sorted(set(files) - set(results))          # includes the stale ones
    print(f"spectra recomputed because the JSON changed (stale): {len(stale)}")
    print(f"spectra to compute (new + stale): {len(missing)}", flush=True)

    failed = {}
    if missing:
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = [pool.submit(process_record, str(files[rid]), periods, damping, method,
                                   fft_max_dt) for rid in missing]
            for n_done, fut in enumerate(as_completed(futures), start=1):
                rid, psa, rec_meta, err = fut.result()
                if err is None:
                    results[rid], meta[rid] = psa, rec_meta
                else:
                    failed[rid] = err
                    print(f"  FAILED {rid}: {err}", flush=True)
                if n_done % 250 == 0 or n_done == len(missing):
                    print(f"  {n_done}/{len(missing)} computed, "
                          f"{(time.perf_counter() - t_start) / 60:.1f} min", flush=True)

    # ---- save --------------------------------------------------------------------------------
    attrs = {"quantity": "PSA (pseudo-spectral acceleration), units of the input record",
             "damping": damping, "periods_s": periods.tolist(),
             "psa_method": method, "method": _method_label(method, fft_max_dt),
             "source_dir": str(records_dir.resolve()),
             "updated": datetime.now().isoformat(timespec="seconds"),
             "record_meta": meta, "failed": failed}
    if method == "fft":
        attrs["fft_max_dt"] = fft_max_dt
    df = to_frame(results, periods, attrs)
    spectra_fp.parent.mkdir(parents=True, exist_ok=True)
    write_pickle(df, spectra_fp)
    print(f"\nwrote {spectra_fp}: {df.shape[0]} records x {df.shape[1]} periods "
          f"({len(files)} JSONs, {len(failed)} failed)")
    return df


# ---------------------------------------------------------------------------------------------
# 7. Command line
# ---------------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute PSA spectra for every record JSON in a folder and save them as one "
                    "pickled DataFrame (index record_id, columns period).")
    p.add_argument("input_dir", type=Path, help="folder containing the record *.json files")
    p.add_argument("output_dir", type=Path, help="folder the DataFrame pickle is written to")
    p.add_argument("--n-cores", type=int, default=1,
                   help="number of worker processes (default 1)")
    p.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD,
                   help=f"'fft' (default): FFT-resample to dt <= --fft-max-dt first; 'zoh': the "
                        f"original, on the record's own dt (see the module docstring)")
    p.add_argument("--fft-max-dt", type=float, default=FFT_MAX_DT,
                   help=f"largest time step [s] after FFT resampling (default {FFT_MAX_DT})")
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
        same_method = (_stored_method(prev.attrs) == args.method
                       and (args.method != "fft" or prev.attrs.get("fft_max_dt") == args.fft_max_dt))
        if not (same_grid and same_damping and same_method):
            raise SystemExit(f"{partial_fp.name} was made with a different period grid, damping or "
                             "method; delete it or run without --resume")
        results = {rid: row.to_numpy() for rid, row in prev.iterrows()}
        meta = dict(prev.attrs.get("record_meta", {}))
        print(f"resuming: {len(results)} records already done in {partial_fp.name}")
    todo = [fp for fp in files if fp.stem not in results]

    attrs = {"quantity": "PSA (pseudo-spectral acceleration), units of the input record",
             "damping": args.damping, "periods_s": periods.tolist(),
             "source_dir": str(args.input_dir.resolve()),
             "psa_method": args.method, "method": _method_label(args.method, args.fft_max_dt),
             "started": datetime.now().isoformat(timespec="seconds")}
    if args.method == "fft":
        attrs["fft_max_dt"] = args.fft_max_dt

    print(f"{len(files)} records in {args.input_dir}, {len(todo)} to compute, "
          f"{len(periods)} periods ({periods[0]}-{periods[-1]} s), damping {args.damping}, "
          f"method {args.method}, {args.n_cores} core(s)")
    print(f"output: {out_fp}", flush=True)

    def checkpoint():
        write_pickle(to_frame(results, periods, {**attrs, "record_meta": meta,
                                                 "failed": failed}), partial_fp)

    # ---- the work: independent records shared over the worker processes -------------------
    t_start = time.perf_counter()
    done_since_ckpt = 0
    with ProcessPoolExecutor(max_workers=args.n_cores) as pool:
        futures = [pool.submit(process_record, str(fp), periods, args.damping, args.method,
                               args.fft_max_dt) for fp in todo]
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
