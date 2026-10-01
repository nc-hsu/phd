"""Convert selected ESM records (ASDF/HDF5) to the project's JSON format.

ESM records are stored one HDF5 file per recording on the share, named
``{event_id}__{station_code}__{location_code}.h5`` (ASDF format). Each file's
``/Waveforms/{NET}.{STA}/`` group(s) hold the acceleration datasets, one per
channel, keyed ``{NET}.{STA}.{LOC}.{CHA}__{start}__{end}__{tag}``. The waveform
data is in cm/s^2 and is converted to g on output.

WHICH DATASET IS U, V OR W
--------------------------
The component is taken from the full SEED channel code (``CHA``, e.g. "HNE",
"HN3") through the explicit map ``CHANNEL_TO_COMPONENT``, never from the
dataset's position. (Until 2026-09-30 this script took the key-sorted 1st/2nd/3rd
dataset as U/V/W. That is right for E/N/Z channels, but for HN2/HN3 files it
labels HN2 as U, while the ESM database's U is HN3: 56 downloaded records came out
with U and V swapped. See nb 043, sections 5 and 6.)

Some files hold more than one instrument, told apart by the SEED location code
``LOC`` (e.g. SGPA: "" = HNE/HNN/HNZ and "01" = HGE/HGN/HGZ). The instrument used
is the one whose location matches the location in the FILE NAME, with "", "0" and
"00" all meaning "00" (checked against the DB spectra of all 29 multi-instrument
files on the share, nb 043 / 2026-09-30). Anything that rule or the channel map
cannot resolve uniquely is NOT guessed: the record is skipped with the reason,
and listed at the end of the run.

WHICH NETWORK (added 2026-09-30)
--------------------------------
The download request names only the event and the station code, so when two
networks run a station with the same code (278 GE/HL pairs in ESM, plus 2 FR/RA
and 1 AC/IT) ESM puts BOTH into one file: ``/Waveforms/GE.KTHA`` and
``/Waveforms/HL.KTHA``. They are different recordings, and they are two different
rows of the selection database. The record identifier
``{event}_{station}_{location}`` cannot tell them apart. So the selection CSV now
carries an optional ``network_code`` column (from the selection database, via
nb 033 / 040). When it is given, only the ``{network_code}.{station_code}`` group
is read. Without a network code the old behaviour is kept, and a file holding two
networks fails with "2 instruments".

NETWORK CODE ALIASES (added 2026-10-01): sometimes the flatfile names one network
(e.g. IV) but the file stores the same recording under another code (e.g.
``8P.T1201``, a temporary deployment). If the requested group is not in the file,
the codes listed for it in ``NETWORK_CODE_ALIASES`` are tried. Exactly one must be
present; if none or several are, the record fails. There is no fallback to a
network that is not listed. The JSON's ``network_code`` is the code actually read,
and ``requested_network_code`` is the one the CSV asked for.

Given a selection CSV of ``(record_identifier, component)`` rows -- where
``record_identifier`` is ``{event_id}_{station_code}_{location_code}`` -- this
script writes one JSON record per row, using the same schema and filename
convention as the NGA-Sub converter (``record`` field last;
``{record_identifier}__{component}.json``), plus five traceability fields:
``network_code``, ``requested_network_code``, ``channel``, ``instrument_location``
and ``source_file``.

Usage:
    python convert_esm_records_to_json.py <src> <selection_csv> <dst>
        [--overwrite] [--backup-dir DIR] [--dry-run] [--max-workers N]

    src           Folder containing the ESM ``*.h5`` files.
    selection_csv CSV with ``record_identifier, component`` columns and, optionally,
                  ``network_code``.
    dst           Output folder for JSON files.
    --overwrite   Reconvert records whose JSON already exists.
    --backup-dir  With --overwrite: copy an existing JSON here first if its data
                  (record or dt) will change.
    --dry-run     Resolve every row's dataset (old positional rule vs the channel
                  map) and report, writing nothing.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from standes.groundmotion import record_json_filename, parse_esm_record_identifier

# Console messages use emoji; make sure they survive a non-UTF-8 stdout (e.g.
# the default Windows cp1252 console) instead of crashing the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

G0 = 9.80665  # m/s^2 per g

# ---------------------------------------------------------------------------------------------
# Channel code -> component. FULL channel codes only; a code that is not listed makes the record
# fail rather than be guessed. Evidence for each group (all against the ESM DB spectra, MSLE up to
# max_usable_T, nb 043 and the one-off check of 2026-09-30):
#   H?E/H?N/H?Z  E -> U, N -> V: every downloaded HNE/HNN, HGE/HGN, HLE/HLN record matches its
#                own DB label (2,148 records), and so do all 66 DB rows of the multi-instrument files.
#   HN3/HN2      HN3 -> U, HN2 -> V: all 56 downloaded HN2/HN3 records match the DB this way round.
#   HG3/HG2      HG3 -> U, HG2 -> V: both files on the share (SDIF, OTER1), by a wide margin.
#   HN1          HN1 -> U (with HN2 -> V): the one file with this layout (OGSI).
# Z channels are the vertical, W.
# ---------------------------------------------------------------------------------------------
CHANNEL_TO_COMPONENT = {
    "HNE": "U", "HNN": "V", "HNZ": "W",
    "HGE": "U", "HGN": "V", "HGZ": "W",
    "HLE": "U", "HLN": "V", "HLZ": "W",
    "HN3": "U", "HN2": "V",
    "HG3": "U", "HG2": "V",
    "HN1": "U",
}
COMPONENTS = ("U", "V", "W")

# ---------------------------------------------------------------------------------------------
# Network code aliases: flatfile network code -> the codes the SAME recording may be stored under in
# the HDF5 file. The flatfile (and so the selection CSV) sometimes names a permanent network (e.g.
# IV) where the HDF5 file's station group carries the code of the temporary deployment or the
# historical/virtual network the data was archived under (e.g. 8P.T1201, EU.GMN). It is one
# recording relabelled, not a second instrument.
#
# Used only when the requested "{network_code}.{station}" group is NOT in the file; the aliases are
# then tried and exactly one must be present (two or more is ambiguous and fails). Source: every
# failure in dump/network_code_errors.txt (2026-10-01, 75 rows), each of which had exactly one
# station group in the file:
#   IV -> 8P (T12xx, 2016 Central Italy), 8H (RMxx, 2009 L'Aquila), 5G (T08xx, 2012 Emilia),
#         9N (T0721), EU (GMN, 1976 Friuli)
#   RO -> EU, CR -> EU, IP -> EU (historical records), SI -> SL (Slovenia, 1998)
# GE and HL deliberately have NO entries: GE.KTHA and HL.KTHA are different recordings (see "WHICH
# NETWORK" at the top), so neither may ever stand in for the other.
# ---------------------------------------------------------------------------------------------
NETWORK_CODE_ALIASES = {
    "IV": ["8P", "8H", "5G", "9N", "EU"],
    "RO": ["EU"],
    "CR": ["EU"],
    "IP": ["EU"],
    "SI": ["SL"],
}

# The OLD positional rule (key-sorted 1st/2nd/3rd dataset = U/V/W). Kept ONLY so that the dry run
# can report which records the channel map changes; it is not used to convert anything.
_OLD_POSITIONAL_INDEX = {"U": 0, "V": 1, "W": 2}


def _normalise(text):
    """Lowercase and strip whitespace (for fuzzy header matching)."""
    return re.sub(r"\s+", "", str(text)).lower()


def _find_column(fieldnames, *aliases):
    """Return the actual header matching one of ``aliases`` (case-insensitive)."""
    wanted = {_normalise(a) for a in aliases}
    for name in fieldnames or []:
        if _normalise(name) in wanted:
            return name
    return None


def _to_g(values, units):
    """Convert an acceleration array to units of g."""
    u = (units or "").strip().lower()
    if u in ("g", "grav", "gravity", "gravities"):
        return values.astype(float)
    if "cm/s" in u or "gal" in u:
        return values.astype(float) / (100.0 * G0)
    if "m/s" in u:
        return values.astype(float) / G0
    return values.astype(float)


def read_selection(path):
    """Yield ``(record_identifier, component, network_code)`` tuples from the selection CSV.

    Detects the header row containing ``record_identifier`` so that both a
    plain single-header CSV and the 2-row MultiIndex header produced by
    ``DataFrame.to_csv`` (columns like ``("metadata", "record_identifier")``)
    are handled.

    ``network_code`` is read from an optional ``network_code`` column (see "WHICH
    NETWORK" at the top). It is ``None`` if the column is absent or the cell is
    empty, and then the record is resolved as before.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        rows = list(csv.reader(f))

    header_idx = None
    for i, row in enumerate(rows):
        norm = {_normalise(c) for c in row}
        if "record_identifier" in norm and ("component" in norm or "comp" in norm):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"{path.name}: could not find a header row with "
            f"'record_identifier' and 'component'."
        )

    header = rows[header_idx]
    id_col = next(i for i, c in enumerate(header)
                  if _normalise(c) == "record_identifier")
    comp_col = next(i for i, c in enumerate(header)
                    if _normalise(c) in ("component", "comp"))
    # optional: absent in selection CSVs written before 2026-09-30
    net_col = next((i for i, c in enumerate(header)
                    if _normalise(c) == "network_code"), None)

    for row in rows[header_idx + 1:]:
        if len(row) <= max(id_col, comp_col):
            continue
        record_identifier = str(row[id_col]).strip()
        component = str(row[comp_col]).strip().upper()
        if not record_identifier:
            continue
        network_code = None
        if net_col is not None and net_col < len(row):
            network_code = str(row[net_col]).strip() or None
        yield record_identifier, component, network_code


def resolve_hdf5_path(src, event_id, station_code, location_code):
    """Locate the HDF5 file for a record, tolerating a zero-padded location.

    Returns the resolved ``Path`` or ``None`` if not found / ambiguous.
    """
    src = Path(src)
    exact = src / f"{event_id}__{station_code}__{location_code}.h5"
    if exact.is_file():
        return exact

    matches = sorted(src.glob(f"{event_id}__{station_code}__*.h5"))
    if len(matches) == 1:
        return matches[0]
    return None


def _find_station_group(h5, station_code):
    """Return the /Waveforms station group (single group, or matched by code).

    Used by the OLD positional rule (dry-run comparison) and by nb 043's verification module,
    which reproduces how earlier conversions read the files. The conversion itself uses
    ``_station_datasets``, which looks at every matching station group.
    """
    if "Waveforms" not in h5:
        raise KeyError("HDF5 missing /Waveforms")
    wf = h5["Waveforms"]
    keys = list(wf.keys())
    if len(keys) == 1:
        return wf[keys[0]]
    matches = [k for k in keys
               if f".{station_code}" in str(k) or str(k).endswith(station_code)]
    if matches:
        matches.sort(key=lambda x: len(str(x)))
        return wf[matches[0]]
    raise KeyError(
        f"station group not found for station_code={station_code}; "
        f"Waveforms keys={keys[:10]}"
    )


# ---------------------------------------------------------------------------------------------
# Choosing the dataset by channel name and instrument location
# ---------------------------------------------------------------------------------------------

def _norm_location(loc):
    """SEED location code for comparison: "", "0" and "00" all mean "00"; others unchanged."""
    loc = str(loc).strip()
    return "00" if loc in ("", "0", "00") else loc


def _parse_dataset_key(key):
    """``"IT.SGPA..HNE__2016...__..._acc_mp"`` -> ``("IT", "SGPA", "", "HNE")``.

    The part before the first "__" is the SEED id NET.STA.LOC.CHA; LOC may be empty.
    """
    seed_id = str(key).split("__", 1)[0]
    parts = seed_id.split(".")
    if len(parts) != 4:
        raise ValueError(f"unexpected dataset key {key!r} (expected NET.STA.LOC.CHA__...)")
    return tuple(parts)


def _location_from_filename(h5_path):
    """The location code in the HDF5 file name ``{event}__{station}__{loc}.h5``."""
    return Path(h5_path).stem.rsplit("__", 1)[-1]


def _station_datasets(h5, station_code, network_code=None):
    """Every dataset of this station, over ALL its station groups: [(dataset, net, loc, cha)].

    A station group is named "NET.STA"; groups whose STA equals ``station_code`` are used. If
    none matches by name but there is exactly one group, that group is used (as before).

    With ``network_code`` only the group "{network_code}.{station_code}" is used. This is what
    separates two networks that share a station code in one file (e.g. GE.KTHA / HL.KTHA). If
    that group is missing, the aliases of ``network_code`` in ``NETWORK_CODE_ALIASES`` are
    tried (same recording stored under another code, e.g. IV -> 8P). Exactly one alias group
    must be present; none, or more than one (ambiguous), fails with ``KeyError``. A network that
    is not a listed alias is never used in its place, since it is a different recording.
    """
    if "Waveforms" not in h5:
        raise KeyError("HDF5 missing /Waveforms")
    wf = h5["Waveforms"]
    if network_code is not None:
        wanted = f"{network_code}.{station_code}"
        if wanted in wf:
            # the requested code always wins; aliases are not even looked at
            groups = [wanted]
        else:
            # check EVERY alias (not just the first hit) so that an ambiguous file is caught
            aliases = NETWORK_CODE_ALIASES.get(network_code, [])
            found = [f"{a}.{station_code}" for a in aliases if f"{a}.{station_code}" in wf]
            if len(found) > 1:
                raise KeyError(f"no station group {wanted} and {len(found)} alias groups match "
                               f"({', '.join(found)}); ambiguous")
            if not found:
                tried = f" (aliases tried: {', '.join(aliases)})" if aliases else ""
                raise KeyError(f"no station group {wanted}{tried}; groups: {list(wf)[:10]}")
            groups = found
    else:
        groups = [g for g in wf if str(g).split(".", 1)[-1] == station_code]
    if not groups:
        if len(wf) != 1:
            raise KeyError(f"no station group for {station_code}; groups: {list(wf)[:10]}")
        groups = list(wf)
    out = []
    for g in groups:
        for k in sorted(wf[g].keys()):
            obj = wf[g][k]
            if isinstance(obj, h5py.Dataset):
                net, sta, loc, cha = _parse_dataset_key(k)
                out.append((obj, net, loc, cha))
    return out


def select_component_dataset(h5, station_code, component, file_location, network_code=None):
    """The one dataset that holds ``component`` (U/V/W): returns ``(dataset, net, loc, cha)``.

    0. With ``network_code``, look only at that network's station group (``_station_datasets``).
    1. Keep the datasets whose location matches the file-name location (``_norm_location``).
    2. They must belong to ONE instrument, i.e. one (network, location, band+instrument code).
    3. Its channels are mapped with ``CHANNEL_TO_COMPONENT`` (full codes); exactly one must be
       ``component``.
    Any other outcome raises ``ValueError`` with the reason; nothing is guessed.
    """
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}")
    all_ds = _station_datasets(h5, station_code, network_code)
    available = sorted({f"{net}.{loc!r}.{cha}" for _, net, loc, cha in all_ds})

    keep = [d for d in all_ds if _norm_location(d[2]) == _norm_location(file_location)]
    if not keep:
        raise ValueError(f"no dataset at the file-name location {file_location!r}; "
                         f"available: {available}")

    instruments = sorted({(net, loc, cha[:2]) for _, net, loc, cha in keep})
    if len(instruments) > 1:
        raise ValueError(f"{len(instruments)} instruments at location {file_location!r}: "
                         f"{instruments}; cannot tell which one the DB record is")

    unknown = sorted({cha for _, _, _, cha in keep if cha not in CHANNEL_TO_COMPONENT})
    if unknown:
        raise ValueError(f"channel code(s) {unknown} not in CHANNEL_TO_COMPONENT")

    hits = [d for d in keep if CHANNEL_TO_COMPONENT[d[3]] == component]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} channels map to {component} "
                         f"({[d[3] for d in keep]}); expected exactly one")
    return hits[0]


def _positional_network(h5, station_code):
    """The network whose group the OLD rule read (dry-run comparison only).

    The old rule took the shortest matching "NET.STA" group name, which for two networks of
    equal-length codes is the first in key order (GE before HL).
    """
    grp = _find_station_group(h5, station_code)
    return str(grp.name).rsplit("/", 1)[-1].split(".", 1)[0]


def _positional_channel(h5, station_code, component):
    """The channel the OLD positional rule would have used (dry-run comparison only)."""
    grp = _find_station_group(h5, station_code)
    keys = [k for k in sorted(grp.keys()) if isinstance(grp[k], h5py.Dataset)]
    idx = _OLD_POSITIONAL_INDEX.get(component)
    if idx is None or idx >= len(keys):
        return ""
    return _parse_dataset_key(keys[idx])[3]


# ---------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------

def _same_data(old_fp, new_record_dict):
    """True if the JSON at ``old_fp`` holds the same ``dt`` and ``record`` as the new one."""
    try:
        with open(old_fp, "r") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return False
    return (old.get("dt") == new_record_dict["dt"]
            and np.array_equal(np.asarray(old.get("record", []), dtype=float),
                               np.asarray(new_record_dict["record"], dtype=float)))


def convert_record(record_identifier, component, src, dst, backup_dir=None, dry_run=False,
                   network_code=None):
    """Convert one (record_identifier, component) selection into a JSON file.

    ``network_code`` (optional) picks the station group when the file holds the same station
    code for more than one network (see "WHICH NETWORK" at the top).

    Returns a dict:
      reason               None if converted (or resolved, in a dry run), else why it was skipped
      source_file          the HDF5 file name
      network_code         network of the dataset used (e.g. "HL")
      channel              channel code used (e.g. "HN3")
      instrument_location  that dataset's SEED location code ("" if blank)
      old_network          the network whose group the old rule read (always the first, e.g. "GE")
      old_channel          the channel the old positional rule would have used
      backed_up            True if an existing JSON was copied to ``backup_dir`` first
    In a dry run only the file KEYS are read and nothing is written.
    """
    info = {"reason": None, "source_file": "", "network_code": "", "channel": "",
            "instrument_location": "", "old_network": "", "old_channel": "",
            "backed_up": False}
    try:
        event_id, station_code, location_code = parse_esm_record_identifier(record_identifier)
    except ValueError:
        info["reason"] = "bad record_identifier"
        return info
    if component not in COMPONENTS:
        info["reason"] = "unknown component"
        return info

    h5_path = resolve_hdf5_path(src, event_id, station_code, location_code)
    if h5_path is None:
        info["reason"] = "HDF5 file not found"
        return info
    info["source_file"] = h5_path.name

    try:
        with h5py.File(h5_path, "r") as h5:
            try:
                info["old_network"] = _positional_network(h5, station_code)
                info["old_channel"] = _positional_channel(h5, station_code, component)
            except (KeyError, ValueError):
                info["old_network"], info["old_channel"] = "", ""
            ds, net, loc, cha = select_component_dataset(
                h5, station_code, component, _location_from_filename(h5_path), network_code)
            info["network_code"], info["channel"], info["instrument_location"] = net, cha, loc
            if dry_run:
                return info
            sr = ds.attrs.get("sampling_rate", None)
            if sr is None:
                info["reason"] = "sampling_rate missing"
                return info
            dt = 1.0 / float(sr)
            record = _to_g(np.array(ds[()], dtype=float).ravel(), "cm/s^2").tolist()
    except (KeyError, OSError, ValueError) as e:
        info["reason"] = str(e)
        return info

    record_dict = {
        "eq_index": record_identifier,
        "database": "esm",
        "dt": dt,
        "duration": dt * len(record),
        "units": "g",
        "record_type": "acc",
        "component": component,
        "event_id": event_id,
        "station_code": station_code,
        "location_code": location_code,
        # traceability: exactly which dataset of which file this is
        "network_code": net,
        # the network the selection CSV asked for (None if not given). Differs from network_code
        # only when an alias from NETWORK_CODE_ALIASES was used (e.g. requested IV, read 8P).
        "requested_network_code": network_code,
        "channel": cha,
        "instrument_location": loc,
        "source_file": h5_path.name,
        "record": record,
    }

    out_path = Path(dst) / record_json_filename(record_identifier, component)
    # Keep the old version if its DATA is about to change (e.g. a previously swapped U/V);
    # a file whose data is unchanged only gains the new metadata fields and is not backed up.
    if out_path.exists() and backup_dir is not None and not _same_data(out_path, record_dict):
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, Path(backup_dir) / out_path.name)
        info["backed_up"] = True
    with open(out_path, "w") as f:
        json.dump(record_dict, f, indent=4)
    return info


def convert_esm_records(src, selection_csv, dst, max_workers=None, overwrite=False,
                        backup_dir=None, dry_run=False):
    """Convert all ``(record_identifier, component[, network_code])`` rows of the selection CSV.

    ``src``, ``selection_csv`` and ``dst`` may be paths or strings. Records are
    read from the (often network-mounted) source in parallel using a thread
    pool, since the work is I/O-bound. ``max_workers`` defaults to
    ``max(cpu_count - 3, 1)``. Rows whose target JSON already exists in ``dst``
    are skipped without reading the source, unless ``overwrite`` is True; with
    ``backup_dir`` an existing JSON whose data changes is copied there first.

    Normal run: returns ``(converted_count, skipped_records)`` where ``skipped_records`` is a
    list of ``{"record_identifier", "component", "reason"}`` dicts for every row that could not
    be converted.

    ``dry_run=True``: writes nothing and returns a list of dicts, one per row (existing JSON or
    not), with ``record_identifier, component, json_exists`` and the fields of
    ``convert_record`` (``network_code``, ``channel``, ``old_network``, ``old_channel``,
    ``reason`` ...) plus ``requested_network`` (the CSV's network code) and
    ``changes = (old_network, old_channel) != (network_code, channel)``. A change of network
    counts: GE.KTHA..HNN and HL.KTHA..HNN are different recordings with the same channel code.
    """
    if max_workers is None:
        max_workers = max((os.cpu_count() or 1) - 3, 1)

    dst = Path(dst)
    rows = list(read_selection(selection_csv))

    if dry_run:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [(rid, comp, net, executor.submit(convert_record, rid, comp, src, dst,
                                                        None, True, net))
                       for rid, comp, net in rows]
            report = []
            for rid, comp, net, fut in tqdm(futures, total=len(futures), desc="Dry run (ESM)"):
                info = fut.result()
                # requested_network: from the CSV (None if not given); network_code: the one used
                info.update(record_identifier=rid, component=comp, requested_network=net,
                            json_exists=(dst / record_json_filename(rid, comp)).exists())
                info["changes"] = bool(
                    info["reason"] is None
                    and (info["old_network"], info["old_channel"])
                    != (info["network_code"], info["channel"]))
                report.append(info)
        n_fail = sum(r["reason"] is not None for r in report)
        n_change = sum(r["changes"] for r in report)
        print(f"\nDry run: {len(report)} rows, {n_change} would change network or channel, "
              f"{n_fail} cannot be resolved.")
        return report

    dst.mkdir(parents=True, exist_ok=True)
    converted = 0
    backed_up = 0
    skipped_existing = 0
    skipped_records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for record_identifier, component, network_code in rows:
            out_path = dst / record_json_filename(record_identifier, component)
            if out_path.exists() and not overwrite:
                skipped_existing += 1
                continue
            futures.append((record_identifier, component,
                            executor.submit(convert_record, record_identifier,
                                            component, src, dst, backup_dir, False,
                                            network_code)))
        for record_identifier, component, future in tqdm(
                futures, total=len(futures), desc="Converting ESM records"):
            info = future.result()
            if info["reason"] is None:
                converted += 1
                backed_up += info["backed_up"]
            else:
                skipped_records.append({
                    "record_identifier": record_identifier,
                    "component": component,
                    "reason": info["reason"],
                })

    print(f"\nDone: {converted} converted ({backed_up} changed and backed up"
          f"{f' to {backup_dir}' if backup_dir is not None else ''}), "
          f"{skipped_existing} already present (skipped), {len(skipped_records)} failed "
          f"(max_workers={max_workers}).")
    if skipped_records:
        print("Skipped records:")
        for s in skipped_records:
            print(f"  {s['record_identifier']} {s['component']}: {s['reason']}")
    return converted, skipped_records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert selected ESM records (HDF5) to project JSON format."
    )
    parser.add_argument("src", help="Folder containing the ESM *.h5 files.")
    parser.add_argument("selection_csv", help="CSV with record_identifier, component columns "
                                              "(and optionally network_code).")
    parser.add_argument("dst", help="Output folder for JSON files.")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers (default: max(cpu_count - 3, 1)).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert records even if their JSON already exists in dst.")
    parser.add_argument("--backup-dir", default=None,
                        help="With --overwrite: copy an existing JSON here first if its data "
                             "will change.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve every row's dataset and report; write nothing.")
    args = parser.parse_args(argv)

    if args.dry_run:
        report = convert_esm_records(args.src, args.selection_csv, args.dst,
                                     max_workers=args.max_workers, dry_run=True)
        for r in report:
            if r["changes"] or r["reason"] is not None:
                print(f"  {r['record_identifier']} {r['component']}: "
                      f"{r['old_network'] or '-'}.{r['old_channel'] or '-'} -> "
                      f"{r['network_code'] or '-'}.{r['channel'] or '-'}"
                      f"{'' if r['reason'] is None else '  FAILS: ' + r['reason']}")
        return 0

    convert_esm_records(args.src, args.selection_csv, args.dst,
                        max_workers=args.max_workers, overwrite=args.overwrite,
                        backup_dir=args.backup_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
