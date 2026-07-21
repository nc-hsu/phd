# WP1 — Ground motion set

Notebooks for WP1: site selection, PSHA, case-study structure design, equivalent-SDOF
parameterisation, disaggregation, ground-motion record selection and MSA setup.

## Numbering convention

Files are `0NN-<name>.ipynb`, grouped by decade:

| Decade | Topic |
|---|---|
| `00x` | Site selection, IM correlations, PSHA |
| `01x` | Case-study designs, analysis-file creation, SDOF parameterisation |
| `02x` | Disaggregation |
| `03x` | Ground-motion record selection (GCIM) |
| `04x` | Record download and conversion |
| `05x` | MSA setup |

Prose in the older notebooks cross-references a legacy work-package code scheme
(`wp1pt4pt1f`, `wp1pt4pt1g`, …). These map onto the numbers above; the numbers are
authoritative.

Two subfolders hold work that is not part of the main chain: `archived/`,
`partly_completed_ideas/` and `checks_and_tests/`.

## Pipelines

`010` and `011` are the active case-study pipelines. The EC8-gen2 SDOF-parameterisation chain
`012` → `013` → `014` has been moved to
`partly_completed_ideas/reference_sdofs_and_sdof_parameterisation/` and is no longer part of
the main workflow.

| | Reference structures | Site-specific case studies |
|---|---|---|
| Notebook | `010` | `011` |
| Analysis root | `cfg["analysis_data"]["reference_structures_dc2_scA"]` | `D:/04_site_influence_investigation` |
| Scope | 9 reference 3-storey CBFs across `S_alpha,RP` | 60 selected sites, one 3-storey CBF each |
| Systems built | MDOF | MDOF |
| Analyses | modal, pushover, cyclic pushover, IDA | modal, cyclic pushover, FEMA P695 IDA |
| Batch launchers | `batch_run_analyses/ec8_gen2_reference_structures/3s/mdof/` | `batch_run_analyses/casestudy_sites/3s/mdof/` |

Both build **MDOF-only** folders — the equivalent-SDOF approximation that `011` used to
produce has been removed. Batch launchers now live under
`cfg["scripts"]["batch_run_analyses"]` in a `<structure-set>/<storeys>/<system>/` tree, so the
filenames are just the analysis type (`po.py`, `cpo.py`, `modal.py`, `ida.py`).

## Dependency order

| Notebook | Reads | Writes | Run after |
|---|---|---|---|
| `001-site_selection` | ESHM20 / site data | `results/01_site_selection/sites.csv` | — |
| `002-im_correlations` | record flatfiles | IM correlation models | — |
| `003-psha_setup_and_analyses` | `sites.csv` | OpenQuake hazard calcs | `001` |
| `004-psha_results` | OQ calcs | hazard curves per site | `003` |
| `010_cbfs_reference_designs…` | — | `casestudy_designs_dc2_scA_reference` designs, reference MDOF folders, design-parameter dataset `…/08_casestudy_structure_datasets/ec8_gen2_dc2_scA_reference_cbfs.pickle`, `…/ec8_gen2_reference_structures/3s/mdof/{po,cpo,modal,ida}.py` | — |
| `011-create_site_casestudies…` | `sites.csv` | site MDOF folders (modal + cpo + FEMA P695 IDA), `site_designs_summary.csv`, design-parameter dataset `…/08_casestudy_structure_datasets/ec8_gen2_site_specific_cbfs.pickle`, `…/casestudy_sites/3s/mdof/{cpo,ida}.py` | `001` |
| `012`–`014` *(archived)* | — | moved to `partly_completed_ideas/reference_sdofs_and_sdof_parameterisation/` | — |
| `017-disagg_imls_for_msa_stripes` | fragility pickles (from the archived `014`), hazard curves (`004`), `site_designs_summary.csv` (`011`) | stripe IMLs for MSA | `004`, `011` |
| `020-disaggregation` | OQ calcs | disaggregation results | `017` |
| `030`–`037` | flatfile, disagg results | GCIM distributions, selected record sets | `020` |
| `040`–`042` | selection results | downloaded + converted records | `037` |
| `050_setup_msa_runs…` | stripe IMLs, converted records, model folders | MSA files + `site_*_msa_*` launchers | `042`, `011`, `017` |

> **Note.** `050` still adds MSA files to a per-site `sdof_param/` folder and writes
> `site_sdof_param_msa_*` launchers. Since `011` no longer builds SDOF folders, that half of
> `050` now has no inputs — update `050` (or drop its `sdof_param` system) before running it.

## The 012 → 013 → 014 chain *(archived)*

> These notebooks now live in
> `partly_completed_ideas/reference_sdofs_and_sdof_parameterisation/`. The description below
> is retained for reference when opening them.

This is the SDOF-parameterisation pipeline. It **cannot be run straight through**: it
alternates between generating OpenSees input files and consuming the results of running
them. Each notebook marks its stopping points with a ⛔ **RUN BARRIER** heading.

```
012  build MDOF models, design dataset, launchers
      ⛔ run mdof_modal / mdof_po / mdof_cpo  (+ the three mdof_ss_* equivalents)
013  fit frame + brace backbones, write SDOF params, optimisation launchers
      ⛔ run optimisation_3s_steel02_and_hysteretic (+ _mechanism_steel02)
     regress hysteresis params, build approximate-SDOF systems
      ⛔ run approx_sdof_3s_cyclic_pushover
     verification plots
014  build IDA folders + launchers
      ⛔ run mdof/sdof/approx_sdof_3s_ida_femaP695  (MDOF IDA ~18-20 h)
     fragility curves, median IDA curves, combined backbones
```

Read each notebook's header cell before running it — it lists the exact launchers, in order,
with expected runtimes.

### Where the fitted equations live

The prediction equations derived in `013` are stored in
`phd_project/scripts/sdof_parameterisation.py`, **not** in the notebook. That module is the
single source of truth and is also used by
`phd_project/scripts/templates/template_model_cbf_sdof.py`. The notebook shows the
derivation and the fitted coefficients; if you refit, update the module.

## Batch launchers

Launchers are generated **Python** scripts (not `.bat`/`.sh`). Each spawns one console per job
and executes a run-script/config pair inside a per-model folder.

The active notebooks (`010`, `011`) write launchers into `cfg["scripts"]["batch_run_analyses"]`
in a `<structure-set>/<storeys>/<system>/` tree — e.g.
`batch_run_analyses/ec8_gen2_reference_structures/3s/mdof/` and
`batch_run_analyses/casestudy_sites/3s/mdof/`. Because the folder path already encodes the
structure set, storeys and system, the filenames are just the analysis type: `po.py`, `cpo.py`,
`modal.py`, `ida.py`. (The archived `012`–`014` and `050` still use the older flat folder
`cfg["scripts"]["wp1pt4pt1_batch_run"]` with `<system>_<analysis>.py` names.)

Generation goes through `phd_project/scripts/templates/copy_templates_to_folders.py`:
`configure_batch_run_file` for the po/cpo/modal launchers, and `copy_batch_ida_buildings` for
the multi-building IDA coordinator (`ida.py`). Total concurrency is capped globally by
`phd_project/process_semaphore/process_semaphore.py` (physical cores − 3), so launchers can be
started back to back without oversubscribing the machine.

> **Read `phd_project/scripts/templates/Readme.md`** before writing a new setup notebook — it
> documents the run-script/config contract, the IDA and MSA modes, the semaphore, and the
> coordinator-vs-worker rules.

### Cyclic-pushover `dU` overrides

`010` and `011` resolve each folder's cyclic-pushover displacement step through
`phd_project/scripts/cpo_du.py` (`resolve_cpo_du`, `audit_cpo_du`, `existing_cpo_du`):
explicit override → `dU` already tuned on disk → the `CPO_DU` default. Each notebook has a
`CPO_DU_OVERRIDES` dict and a `PRESERVE_TUNED_DU` flag, plus an audit cell that lists any
folder whose on-disk `dU` diverges and prints a ready-to-paste override snippet. This stops a
rebuild from silently clobbering a `dU` that was hand-tuned for convergence.

## PSHA notebooks

`003` and `020` do not generate batch launchers. They use
`phd_project/scripts/WP1_ground_motion_set/oq_runner.py` to write OpenQuake `.ini` job
configs, fingerprint the inputs, and launch `oq engine --run`, recording `calc_id` in
`psha_manifest.json`. They expose `DRY_RUN` / `FORCE_RERUN` / `NEW_WINDOW` flags.
