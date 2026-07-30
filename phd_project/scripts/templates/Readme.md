# Analysis template user guide

This folder holds **template** scripts for running structural analyses on OpenSees
models. You do not run the templates in place. Instead you **copy** the ones you
need into a per-model folder (dropping the `template_` prefix), edit a few values,
and run them there. A helper module, `copy_templates_to_folders.py`, automates the
copying and editing so you can generate many analysis folders programmatically.

---

## 1. The core idea

Every analysis is a **run script + config file** pair:

| File | Contains | Role |
|------|----------|------|
| `run_<analysis>.py` | a `run(config_data)` function | *how* to run the analysis |
| `config_<analysis>.py` | a `config: dict` variable | *what* to run it on |

You start an analysis by calling `run(path_to_config)`. The run script imports the
config, reads its `config` dict, and drives OpenSees. The same run script works for
any number of configs, so to analyse the same model in different ways you just add
more config files.

Each **model lives in its own folder**, which contains:

- `structural_model.py` — builds the OpenSees model (see §2),
- the design `.json` file it reads,
- one or more `run_*.py` + `config_*.py` pairs,
- any supporting files the configs import (process recorders, injection functions,
  intensity-measure definitions).

Config files are imported **by path** (via `standes.utils.import_from_path`). Each config
names its structural-model file through two top-level variables — `model_file_name` (default
`"structural_model.py"`) and `init_function` (default `"model_init"`) — and imports it by
path:

```python
_model = import_from_path(Path(__file__).parent / model_file_name)
model_init = getattr(_model, init_function)
```

The config is the **single place** that imports the model. This is why the model file can
have any name (the notebooks write it as `nlcbf_structural_model.py`): only the config's
`model_file_name` needs to point at it. The config then passes the model pieces to its
sibling helpers — `config_im` and `injection_functions` — which are model-agnostic factories
(§2) and no longer import `structural_model` themselves.

### Convention: writing config files that the copy tools can edit

The copy helpers in §8 rewrite config files with simple text/regex rules, so configs
follow two rules (stated in each template's own docstring):

- **`results_folder_name` must be the first variable** in the file. The copy tools
  anchor on it to rename the output folder and to inject new variables.
- **No comments inside the `config = { ... }` dict.** Put explanatory comments in the
  module docstring above the variables instead. A stray comment or trailing content
  in the dict can break the regex edits.

---

## 2. Building-block templates

These are not analyses on their own — they are the pieces the run/config pairs import.

### The full CBF model → `config_structural_model.py` + `initialise_model.py`
A full (multi-storey) nonlinear concentrically-braced-frame model is described by **two**
files in the model folder (split for single-source-of-truth; the old single-file
`template_nlcbf_structural_model.py` is deprecated — see below):

- **`template_config_structural_model.py` → `config_structural_model.py`** — *what the model
  is*. Exposes `design_json` (name of the design file in the folder), `structure_data`, the
  OpenSees model config(s) `ops_model_config` (+ `ops_model_config_no_G`), `damping_config`,
  `damping_model`, and a `build_model(structure_data, **model_config)` wrapper. No recorder
  logic, no `model_init`.
- **`template_initialise_model_nlcbf_nltha.py`** (full recorders) or
  **`template_initialise_model_nlcbf_nlthareduced.py`** (reduced recorders) **→
  `initialise_model.py`** — *how one analysis is initialised*. Holds all recorder handling
  (`recorder_config`, `generate_recorders`, `add_limit_state_recorders`) and exposes a single
  zero-argument `model_init` returning `(recorders, collapse_recorders)`.

`initialise_model.py` imports the model config **by path** via
`config_structural_model_file_name` (so that file can be renamed/swapped), and pins **which**
named model config it uses via `model_config_name` (e.g. set it to `"ops_model_config_no_G"`
for a no-gravity run — this replaces the removed `model_init_no_g`). It re-exports
`damping_config` / `damping_model` so the injection-function factory still finds them.

The file the config imports (`model_file_name`) is `initialise_model.py`; the **contract** it
must satisfy is: expose `model_init`, `damping_config`, `damping_model`.

**Full vs reduced recorders** is chosen by *which `initialise_model` template you copy*. The
reduced variant subtracts every recorder not needed for collapse detection or the drift EDP
(keeping storey drifts, storey-line displacements + accelerations, gusset forces and all
limit-state / element-removal recorders), cutting the recorder pickles by ~97% for IDA/MSA.
Its `add_limit_state_recorders` body is shared verbatim with the full template — **edit both
if you change the recorder-construction logic**.

> One model per Python process: `initialise_model.py` and its sibling config are imported by
> file path (keyed by file stem), so do not import two different model folders in a single
> process.

### `template_model_cbf_sdof.py` → `structural_model.py`
The single-degree-of-freedom equivalent model — same `model_init` contract, but the
model is a fitted SDOF oscillator (parameters embedded in the file). Use this instead
of the full model when running SDOF studies.

### `template_ida_process_recorder_roof_drift.py` / `_x_displacement.py` → `ida_process_recorders.py`
Defines how each time-history run is reduced to the engineering demand parameter (EDP)
that the IDA/MSA curves are plotted against. Exposes `process_recorder_func`,
`edp_tags`, and `edp_idxs`. Pick the roof-drift or roof-displacement variant (or write
your own). MSA configs import the same style of file as `msa_process_recorders.py`.

### `template_nltha_injection_func_update_damping.py` → `injection_functions.py`
Optional hooks injected into a time-history run at defined points (`pre_nltha`,
`pre_analyse`, `post_analyse`, `post_nltha`). Exposes a factory
`make_injection_functions(model, **injection_function_params)` that the analysis config calls
with the model module (plus any extras from its `injection_function_params` dict, default
`{}`); the supplied example reads `damping_config` / `damping_model` off the model and
re-applies the modal damping model after each step. Because the config supplies the model,
this file does not import `structural_model` itself.

### `template_config_im_SA.py` / `_AvgSA_03.py` / `_AvgSA_06.py` → `config_im_*.py`
Defines the **intensity measure** used to scale ground motions in an IDA. Exposes a factory
`make_im(model_init)`, which the analysis config calls with its `model_init`:

- `config_im_SA` — spectral acceleration at the model's first-mode period (calls `model_init()`
  and runs a modal analysis to find it).
- `config_im_AvgSA_03` / `_06` — geometric-mean spectral acceleration over 0–3 s / 0–6 s
  (a fixed period range; they accept `model_init` for a uniform contract but don't use it).

The IDA config selects its IM **by filename** through its `config_im_file` variable, so
several `config_im_*.py` files can sit in one folder and each IDA config can point at a
different one without editing import statements. Because the config passes `model_init` in,
these files do not import `structural_model` themselves.

### `deprecated/`
Holds the superseded monolithic model templates — `template_nlcbf_structural_model.py` (full
model + recorders in one file) and `template_nlcbf_structural_model_reduced_recorders.py` (its
reduced-recorder copy). They are kept only for older notebooks that still call
`copy_structural_model`; **for the full CBF model use the two-file
`config_structural_model.py` + `initialise_model.py` set above** (via `copy_nlcbf_model`), not
these.

---

## 3. Single-run analyses

Each of these is a straightforward run/config pair. Copy both into the model folder and
run it. Every one exposes a terminal CLI — `python run_<analysis>.py [config]`, where
`config` is optional and defaults to the matching `config_<analysis>.py` in the same
folder — or you can call `run(config)` from Python.

| Analysis | Run template | Config template | What it does |
|----------|--------------|-----------------|--------------|
| Modal | `template_run_modal.py` | `template_config_modal.py` | Eigenvalue analysis; periods and mode shapes. |
| Pushover | `template_run_pushover.py` | `template_config_pushover.py` | Monotonic nonlinear static pushover. |
| Cyclic pushover | `template_run_cyclic_pushover.py` | `template_config_cyclic_pushover.py` | Pushover following a prescribed displacement history. |
| Snapback | `template_run_snapback.py` | `template_config_snapback.py` | Release-from-displacement test to check damping. |
| NLTHA | `template_run_nltha.py` | `template_config_nltha.py` | One nonlinear time-history run for one record. |
| NLTHA (updating damping) | `template_run_nltha_w_updating_damping.py` | `template_config_nltha.py` | As above, re-applying modal damping each step (needs the injection-functions file). |

Example — from the terminal (in the model folder):

```bash
python run_pushover.py                    # uses config_pushover.py
python run_nltha.py config_nltha_120111.py
```

or from Python:

```python
from pathlib import Path
import run_pushover
run_pushover.run(Path("config_pushover.py"))
```

---

## 4. IDA (Incremental Dynamic Analysis)

An IDA runs a model against a set of records, scaling each one up (hunt-trace-fill)
until collapse, to build fragility curves. All IDA modes share the **same config
format** (`template_config_ida_htf.py`), which lists the records, the IM file, and the
hunt-trace-fill parameters. They differ only in *how the records are executed*.

| Mode | Templates | Records run… | Use when |
|------|-----------|--------------|----------|
| Single record | `run_ida_htf_single_record` | the one record in the config | debugging one record |
| Serial, all records | `run_ida_htf_multiple_records` | one after another, one process | small jobs / debugging |
| Parallel, all records | `run_batch_ida_per_record` **+** `run_ida_htf_per_record` | concurrently, one core per record | a single building, fast |
| Many buildings | `run_batch_ida_buildings` | several buildings at once, sharing one core budget | filling a many-core machine |

All modes produce the **same outputs** (`ida_results.pickle`, per-record folders,
fragility curves) — so you can develop with the serial mode and scale up later without
changing the config.

### Parallel, single building — `run_batch_ida_per_record`

This is a **coordinator + worker** pair. Copy **both** `run_batch_ida_per_record.py`
(the coordinator) and `run_ida_htf_per_record.py` (the worker) into the folder — the
coordinator looks for the worker as a sibling of exactly that name. Every record is
driven by the *same* config; only the record tag differs, so there is no per-record
config file.

```bash
python run_batch_ida_per_record.py config_ida_htf_femap695_set.py
python run_batch_ida_per_record.py config_ida_htf_femap695_set.py --max-workers 4
python run_batch_ida_per_record.py config_ida_htf_femap695_set.py --quiet
```

- `--max-workers N` — cap how many records run at once (default: physical cores − 3).
- `--use-semaphore` — use the shared machine-global cap instead (see §6).
- `--quiet` — don't open a window per worker; stream each worker's output to a log file
  instead (see §7).

### Many buildings — `run_batch_ida_buildings`

One launch script that runs the IDAs of several buildings at once and lets them share a
single machine-wide core budget — the way to saturate a many-core box. Edit the
`buildings` list at the top (each entry is a folder + a config filename), then:

```bash
python run_batch_ida_buildings.py
python run_batch_ida_buildings.py --quiet --max-coordinators 5
```

It launches one `run_batch_ida_per_record` coordinator per building (each with the shared
semaphore on), so across all buildings at most `physical cores − 3` record analyses run
at once. Each building needs its coordinator + worker + config already in place. See §6
for why this uses the semaphore and a separate coordinator cap.

### Folder layout per IDA mode

Every IDA folder holds the same input files (structural model + design JSON, injection
functions, IDA process recorders, an intensity-measure file, and the config). The modes
differ only in which **run scripts** are present. Files marked `← created` appear after a
run.

**Single record** (`run_ida_htf_single_record`) and **serial, all records**
(`run_ida_htf_multiple_records`) — one run script, no worker:

```
3s_cbf_dc2_10_sdof/
├─ 3s_cbf_dc2_10_out.json            ← design file
├─ structural_model.py
├─ injection_functions.py
├─ ida_process_recorders.py
├─ config_im_SA.py                   ← intensity measure (selected by config_im_file)
├─ config_ida_htf_femap695_set.py    ← the IDA config
├─ run_ida_htf_multiple_records.py   ← swap for run_ida_htf_single_record.py for one record
└─ ida_femap695/                     ← created by the run (named by results_folder_name)
   ├─ record_0/  …  record_21/
   ├─ ida_results.pickle
   ├─ record_logs.json
   └─ collapse_fragility.json
```

**Parallel, single building** (`run_batch_ida_per_record`) — adds the worker script
alongside the coordinator; `worker_logs/` appears inside the results folder only under
`--quiet`:

```
3s_cbf_dc2_10_sdof/
├─ 3s_cbf_dc2_10_out.json
├─ structural_model.py
├─ injection_functions.py
├─ ida_process_recorders.py
├─ config_im_SA.py
├─ config_ida_htf_femap695_set.py
├─ run_batch_ida_per_record.py       ← coordinator
├─ run_ida_htf_per_record.py         ← worker (sibling of exactly this name — required)
└─ ida_femap695/                     ← created by the run
   ├─ record_0/  …  record_21/
   ├─ ida_results.pickle
   ├─ collapse_fragility.json
   └─ worker_logs/                   ← only with --quiet
      ├─ worker_record_0.log
      └─ …
```

**Many buildings** (`run_batch_ida_buildings`) — one launch script in a parent folder;
each building subfolder is a complete "parallel, single building" layout:

```
03_wp1pt4pt1_dc2_sdof_fitting/
├─ run_batch_ida_buildings.py        ← the single launch script (edit its `buildings` list)
├─ 3s_cbf_dc2_10_sdof/
│  ├─ … structural_model.py, config_im_SA.py, config_ida_htf_femap695_set.py, …
│  ├─ run_batch_ida_per_record.py    ← coordinator
│  ├─ run_ida_htf_per_record.py      ← worker
│  └─ ida_femap695/                  ← created
├─ 3s_cbf_dc2_20_sdof/
│  └─ … same layout
└─ 3s_cbf_dc2_31_sdof/
   └─ … same layout
```

---

## 5. MSA (Multiple-Stripe Analysis)

An MSA runs a model against **stripes** — sets of records selected for a given hazard
level. Every mode shares the same config format (`template_config_msa.py`), which points
at the stripe selection pickles and the record source. As with IDA, the modes differ only
in execution strategy, and **all produce identical output**. The layout mirrors IDA:
a single-building coordinator, plus a top-level multi-building launcher.

| Template | Strategy | Notes |
|----------|----------|-------|
| `run_msa_serial` | **Serial.** Every stripe in turn, every record in turn, one process. | Simplest; debugging / small jobs. |
| `run_batch_msa_per_record` **+** `run_msa_per_record` | **Parallel across (stripe, record) pairs.** All stripes and records are flattened and run concurrently for one building. | The MSA analogue of `run_batch_ida_per_record`; fills cores fastest for one building. |
| `run_batch_msa_buildings` | **Many buildings at once.** One coordinator per building, sharing one core budget. | The MSA analogue of `run_batch_ida_buildings`. |

The coordinator (`run_batch_msa_per_record`) and its worker (`run_msa_per_record`, tag =
`"stripe_id:record"`) throw every (stripe, record) pair into one pool and run them all
concurrently, so the machine fills as fast as possible. Stripes finish in no particular
order; each stripe's log and the collated `msa_log.json` are assembled once all workers
have finished.

### Resuming / adding a stripe later

Runs are **resumable**. Each (stripe, record) pair writes a `record_<t>_log.json` last, so
re-running the batch skips any pair whose log already exists. Because stripe result folders
are keyed by the stripe's **intensity tag** (`stripe_<iml>`, e.g. `stripe_00pt360`) rather
than a positional index, dropping an extra stripe pickle into `gm_selection_src` and
re-running analyses *only the new stripe* — existing stripes keep their identity no matter
where the new one falls in intensity order. Controlled by `resume` in `config_msa`
(default `True`; set `False` to force a full re-run).

### Multi-building — `run_batch_msa_buildings`

Populate the `buildings` list at the top of the file (each entry a `{folder, config}` pair,
where `folder` contains `run_batch_msa_per_record.py` + the named config), by hand or with
`copy_batch_ida_buildings`. The launcher runs up to `max_coordinators` building coordinators
at once; each coordinator draws its NLTHA workers from the shared semaphore, so total live
workers stay capped machine-wide. Pass `--max-workers` / `--no-semaphore` to run each
building with a fixed worker count instead of the shared semaphore.

### Folder layout per MSA mode

Every MSA folder holds the same input files: structural model + design JSON, injection
functions, `msa_process_recorders.py`, and `config_msa.py`. There is **no** intensity-measure
file — stripes are pre-selected, and the config points at the stripe selection pickles
(`gm_selection_src`) and record source (`record_src`), which live outside the model folder.
The modes differ only in which **run scripts** are present. Files marked `← created` appear
after a run.

**Serial** (`run_msa_serial`) — one run script, no worker:

```
3s_cbf_dc2_41_ss/
├─ 3s_cbf_dc2_41_out.json            ← design file
├─ structural_model.py
├─ injection_functions.py
├─ msa_process_recorders.py
├─ config_msa.py                     ← the MSA config (points at the stripe pickles)
├─ run_msa_serial.py
└─ msa/                              ← created by the run (named by results_folder_name)
   ├─ stripe_00pt310/  …  stripe_00pt500/  ← per-stripe folders keyed by intensity tag;
   │                                          per-record pickles + record_<t>_log.json + msa_log_stripe_<iml>.json
   ├─ msa_log.json
   ├─ msa_summary.json
   └─ collapse_fragility.json
```

**Parallel** (`run_batch_msa_per_record` **+** `run_msa_per_record`) — the coordinator plus
its worker. `worker_logs/` appears inside the results folder only under `--quiet`:

```
3s_cbf_dc2_41_ss/
├─ 3s_cbf_dc2_41_out.json
├─ structural_model.py
├─ injection_functions.py
├─ msa_process_recorders.py
├─ config_msa.py
├─ run_batch_msa_per_record.py          ← coordinator
├─ run_msa_per_record.py                ← worker (sibling of exactly this name — required)
└─ msa/                                 ← created by the run
   ├─ stripe_00pt310/  …  stripe_00pt500/
   ├─ msa_log.json
   ├─ msa_summary.json
   ├─ collapse_fragility.json
   └─ worker_logs/                      ← only with --quiet
      ├─ worker_record_00pt360_0.log    (tag "stripe_id:record", with ":" → "_")
      └─ …
```

**Many buildings** (`run_batch_msa_buildings`) — one launch script in a parent folder,
listing the building folders; each building folder is a normal parallel MSA folder:

```
casestudy_sites/3s/mdof/
├─ msa_AvgSA03.py                    the single launch script (its `buildings` list names the folders below)
├─ …/site_2/3s/mdof/
│  ├─ … structural_model.py, msa_process_recorders.py, config_msa_AvgSA_03.py, …
│  ├─ run_batch_msa_per_record.py    coordinator (called directly — no wrapper)
│  ├─ run_msa_per_record.py          worker
│  └─ msa_AvgSA_03/                  ← created
├─ …/site_3/3s/mdof/
│  └─ … same layout
└─ …/site_4/3s/mdof/
   └─ … same layout
```

---

## 6. Parallelisation: how the core budget is managed

Two independent ideas control concurrency. Understanding them tells you which mode to use.

### Local cap vs. shared semaphore

- **Local `max_workers`** (default) — each coordinator limits *its own* workers to
  `physical cores − 3`. Fine for **one** coordinator. Run several coordinators this way
  and they don't know about each other, so you oversubscribe the machine.
- **Shared semaphore** (`--use-semaphore` / `use_semaphore=True`) — a single
  machine-global file-locked counter caps **all** workers across **all** coordinators at
  `physical cores − 3` combined. This is what lets several buildings/sites run together
  without fighting over cores.

> The cap is `physical cores − 3`. On a machine where you want a different reserve, adjust
> it in `phd_project/process_semaphore/process_semaphore.py`.

### Coordinators vs. workers

The multi-building / multi-site launchers make a deliberate distinction:

- A **worker** is one record/stripe analysis — CPU-bound, consumes a core (a semaphore
  slot).
- A **coordinator** is a bookkeeping loop that launches workers and waits — sleep-bound,
  and it **does not** take a worker slot.

That is why the top-level launchers (`run_batch_ida_buildings`, `run_batch_msa_buildings`)
cap coordinators with a *separate local* limit (`--max-coordinators`) and let only the
workers use the semaphore. If coordinators consumed worker slots they could starve the
very workers they are waiting on.

**Rule of thumb:** running one building → local `max_workers` is fine. Running several at
once → use the top-level launcher, which turns the semaphore on for you.

---

## 7. Worker windows and log files

When workers run in parallel you choose how to see their output:

- **Default (windows on):** each worker gets its own minimised console window, titled by
  record/stripe — handy for a few workers.
- **Quiet (`--quiet`):** no windows; each worker streams its stdout **and** stderr to a
  log file. This is what you want when dozens of workers run at once.

Quiet log files are written **inside the results folder** for that config:

```
<model folder>/<results_folder_name>/worker_logs/worker_record_<tag>.log
```

Because each config writes to its own results folder, two different configs in the same
model folder never clobber each other's logs. Tail one live:

```powershell
Get-Content "E:\...\ida_femap695_avgsa03\worker_logs\worker_record_0.log" -Wait
```

Re-running the *same* config overwrites its own previous logs (expected). The
coordinator's own progress always prints to its own window regardless of `--quiet`.

---

## 8. Copying and editing templates programmatically

`copy_templates_to_folders.py` turns the templates into ready-to-run analysis folders. A
typical setup script reads template paths from `phd_project/config/config.yaml`, then
calls these helpers.

### `copy_file(src, dst)`
Plain copy, no edits. Use for files that don't need per-model changes: run scripts,
worker scripts, process recorders, injection functions, intensity-measure files.

```python
copy_file(templates["run_ida_htf_per_record"], folder / "run_ida_htf_per_record.py")
copy_file(templates["config_im_SA"],           folder / "config_im_SA.py")
```

### `copy_nlcbf_model(templates, dst_folder, design_json, ops_updates=None, recorder_updates=None, damping_updates=None, reduced=True, model_config_name=None)`
The current helper for the full CBF model. Writes **both** files into `dst_folder`:
`config_structural_model.py` (edited with `design_json` + `ops_updates` + `damping_updates`)
and `initialise_model.py` (from the full or reduced initialise_model template per `reduced`,
edited with `recorder_updates`, e.g. `{"drift_limit": ...}`). Pass `model_config_name` to pin
which model config initialise_model uses (e.g. `"ops_model_config_no_G"`). `templates` is the
resolved `cfg["templates"]` dict (needs keys `config_structural_model`,
`initialise_model_nltha`, `initialise_model_nltha_reduced`).

```python
copy_nlcbf_model(
    cfg["templates"], folder,
    design_json="3s_cbf_dc2_10_out.json",
    damping_updates={"n_modes": 1, "damping_ratio": 0.05},
    recorder_updates={"drift_limit": 0.2},
    reduced=True,
)
```

### `copy_structural_model(src, dst, design_json=None, ops_updates=None, recorder_updates=None, damping_updates=None)`
Retained only for the **deprecated** single-file `structural_model.py` (see `deprecated/`).
Copies one file and rewrites the `design_json` filename and any values in its
`ops_model_config` / `recorder_config` / `damping_config` dicts. `copy_nlcbf_model` uses it
internally to edit each of the two split files.

### `copy_analysis_config(src, dst, results_folder_name=None, update_config=None, **kwargs)`
The workhorse for config files. It:

1. **Rewrites top-level variables** given as keyword arguments (`record_filenames=[...]`,
   `config_im_file="config_im_AvgSA03.py"`, `model_file_name="nlcbf_structural_model.py"`, …).
   If a variable already exists it is replaced; if not, it is **injected** right after
   `results_folder_name`.
2. **Renames the results folder** via `results_folder_name=...`.
3. **Edits values inside the `config` dict** via `update_config={...}` (matched by key).

This is why configs must keep `results_folder_name` first and free of in-dict comments
(§1). Example — make an AvgSA IDA config from an existing SA one, keeping the record list:

```python
copy_analysis_config(
    folder / "config_ida_htf_femap695_set.py",
    folder / "config_ida_htf_femap695_set_avgsa03.py",
    results_folder_name="ida_femap695_avgsa03",
    config_im_file="config_im_AvgSA03.py",
)
```

### `configure_batch_run_file(src, dst, scripts_configs_list)`
Fills in the `scripts_and_configs` list of a `run_batch_scripts.py` (see §9) from a list
of `{"script": ..., "config": [...], "name": [...]}` entries.

### `configure_optimisation_batch_run(src, dst, configs_list)`
Fills in the `configs` list of a `run_group_sdof_optimisation.py` (see §10).

---

## 9. Running heterogeneous batches — `run_batch_scripts`

`template_run_batch_scripts.py` launches a mixed list of **different** analyses (e.g. a
modal, a pushover and an NLTHA across several buildings) as separate windowed processes,
throttled by the shared semaphore. Populate its `scripts_and_configs` list by hand or with
`configure_batch_run_file`.

Use it for "run this heterogeneous set of one-shot analyses". It is **not** the right tool
for running several parallel IDA/MSA coordinators together — those are sleep-bound
coordinators, and this launcher would make them consume worker semaphore slots. For that
case use `run_batch_ida_buildings` / `run_batch_msa_buildings` (§4, §5), which handle
coordinators correctly.

---

## 10. SDOF parameter optimisation — `run_group_sdof_optimisation`

A specialised batch launcher that fits SDOF model parameters to target data with
differential evolution, for a list of buildings. Configure its `configs` list (each entry
names the target data, parameter bounds, DE settings, cores, …) directly or with
`configure_optimisation_batch_run`, then run it. Unlike the analysis launchers this drives
an optimisation routine rather than OpenSees analyses, but it follows the same
list-of-jobs pattern.

---

## Quick reference: which template do I use?

- **One model, one analysis** → the matching `run_*` + `config_*` pair (§3).
- **One building IDA, all records fast** → `run_batch_ida_per_record` + `run_ida_htf_per_record` (§4).
- **Several buildings' IDAs on a big machine** → `run_batch_ida_buildings` (§4, §6).
- **One building MSA** → `run_batch_msa_per_record` + `run_msa_per_record` (§5).
- **Several buildings' MSAs** → `run_batch_msa_buildings` (§5, §6).
- **A mixed bag of one-shot analyses** → `run_batch_scripts` (§9).
- **Generate the folders for any of the above** → the helpers in `copy_templates_to_folders.py` (§8).
