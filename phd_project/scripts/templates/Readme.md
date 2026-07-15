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

Config files are imported **by path** (via `standes.utils.import_from_path`), which
also puts the model folder on `sys.path` for the duration of the import. That is why
a config can write `from structural_model import model_init` and have it resolve to
the `structural_model.py` sitting next to it.

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

### `template_nlcbf_structural_model.py` → `structural_model.py`
Builds a full (multi-storey) nonlinear concentrically-braced-frame model from a design
`.json`. **Must** expose:

- `design_json: str` — the name of the design file (in the same folder), and
- `model_init` — a no-argument callable that builds the model in OpenSees and returns
  its recorders.

`functools.partial` is the usual way to build `model_init`. It may also expose
`damping_config` / `damping_model` for analyses that need them (NLTHA with updating
damping).

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
`pre_analyse`, `post_analyse`, `post_nltha`). The supplied example re-applies the modal
damping model after each step. Configs that don't need hooks can import an empty set.

### `template_config_im_SA.py` / `_AvgSA_03.py` / `_AvgSA_06.py` → `config_im_*.py`
Defines the **intensity measure** used to scale ground motions in an IDA. Exposes a
module-level `im`:

- `config_im_SA` — spectral acceleration at the model's first-mode period (runs a modal
  analysis to find it).
- `config_im_AvgSA_03` / `_06` — geometric-mean spectral acceleration over 0–3 s / 0–6 s.

The IDA config selects its IM **by filename** through its `config_im_file` variable, so
several `config_im_*.py` files can sit in one folder and each IDA config can point at a
different one without editing import statements.

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

---

## 5. MSA (Multiple-Stripe Analysis)

An MSA runs a model against **stripes** — sets of records selected for a given hazard
level. Every mode shares the same config format (`template_config_msa.py`), which points
at the stripe selection pickles and the record source. As with IDA, the modes differ only
in execution strategy, and **all produce identical output**.

There are three execution strategies plus two site-level wrappers:

| Template | Strategy | Notes |
|----------|----------|-------|
| `run_msa_serial` | **Serial.** Every stripe in turn, every record in turn, one process. | Simplest; debugging / small jobs. |
| `run_batch_msa_per_stripe_record` **+** `run_msa_per_record` | **Parallel across (stripe, record) pairs.** All stripes and records are flattened and run concurrently. | Maximum parallelism; fills cores fastest for one building. |
| `run_batch_msa_per_record` **+** `run_msa_per_record` | **Stripe by stripe, parallel within a stripe.** Finishes and collates one stripe before starting the next. | Parallel, but stripes complete in order — useful when you want early stripes done first. |
| `run_msa_site` | Thin wrapper → `run_batch_msa_per_stripe_record` with the shared semaphore on and workers quiet. | The per-site entry point used by the multi-site launcher. |
| `run_batch_msa_sites` | **Many sites at once.** One coordinator per site, sharing one core budget. | The MSA analogue of `run_batch_ida_buildings`. |

### Key difference between the two parallel MSA modes

Both use the same worker (`run_msa_per_record`, tag = `"stripe:record"`); they differ in
*how work is grouped*:

- **`per_stripe_record`** throws every (stripe, record) into one pool and runs them all
  concurrently. Stripes finish in no particular order; best raw throughput.
- **`per_record`** processes stripes sequentially, parallelising only the records within
  the current stripe, and collates each stripe before moving on. Slightly less peak
  parallelism, but stripes complete in order and results appear stripe-by-stripe.

Choose `per_stripe_record` (or the site wrappers) to fill the machine as fast as
possible; choose `per_record` when ordered, incremental stripe results matter.

### Multi-site — `run_batch_msa_sites`

Set `sites_root` (and optionally an explicit `site_names` list) at the top of the file.
Each immediate subfolder that contains `config_msa.py` + `run_msa_site.py` is treated as a
site. The launcher runs up to `max_coordinators` site coordinators at once; each
coordinator draws its NLTHA workers from the shared semaphore, so total live workers stay
capped machine-wide.

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

That is why the top-level launchers (`run_batch_ida_buildings`, `run_batch_msa_sites`)
cap coordinators with a *separate local* limit (`--max-coordinators`) and let only the
workers use the semaphore. If coordinators consumed worker slots they could starve the
very workers they are waiting on.

**Rule of thumb:** running one building/site → local `max_workers` is fine. Running
several at once → use the top-level launcher, which turns the semaphore on for you.

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

### `copy_structural_model(src, dst, design_json=None, ops_updates=None, recorder_updates=None, damping_updates=None)`
Copies a `structural_model.py` and rewrites the `design_json` filename and any values in
its `ops_model_config` / `recorder_config` / `damping_config` dicts. Supports in-place
editing (src == dst).

```python
copy_structural_model(
    templates["structural_model"], folder / "structural_model.py",
    design_json="3s_cbf_dc2_10_out.json",
    ops_updates={"mass": 205.3},
)
```

### `copy_analysis_config(src, dst, results_folder_name=None, update_config=None, **kwargs)`
The workhorse for config files. It:

1. **Rewrites top-level variables** given as keyword arguments (`record_filenames=[...]`,
   `config_im_file="config_im_AvgSA03.py"`, …). If a variable already exists it is
   replaced; if not, it is **injected** right after `results_folder_name`.
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
case use `run_batch_ida_buildings` / `run_batch_msa_sites` (§4, §5), which handle
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
- **One building MSA** → `run_batch_msa_per_stripe_record` + `run_msa_per_record` (§5).
- **Many MSA sites** → `run_batch_msa_sites` (§5, §6).
- **A mixed bag of one-shot analyses** → `run_batch_scripts` (§9).
- **Generate the folders for any of the above** → the helpers in `copy_templates_to_folders.py` (§8).
