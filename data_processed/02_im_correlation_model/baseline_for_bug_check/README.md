# Baseline for the ASC flatfile filter bug check

## Why this folder exists

`manage_flatfiles.get_active_shallow_crust_flatfile` is meant to build the active shallow
crust (ASC) flatfile from every ESM record that is **not** in the ESHM20 subduction in-slab
(`sslab`), subduction interface (`sinter`) or Vrancea (`vran`) datasets. Its first filter,
`database_df.index.isin(deep_events)`, removes nothing: `deep_events` is a DataFrame, `isin`
iterates over its column labels (text), and those never match the integer row labels of the ESM
frame. Only the 40 km depth cut and the "at least 3 records per event" rule keep regional
records out, so some subduction interface and shallow in-slab records end up in the ASC set.

This folder freezes the nb 002 outputs as they were **before** that filter is fixed, so the
effect of the fix on the residuals and correlations can be measured against them. The task is
described in `dump/tasks_and_plans/task_20261111_asc_flatfile_filter_bug_check.md`.

## What it contains

| folder | copied from | written by nb 002 cell |
|---|---|---|
| `data_processed/{ft}/flatfiles/` | `data_processed/02_im_correlation_model/{ft}/flatfiles/` | 17 |
| `data_processed/{ft}/residual_partitioning/` | `data_processed/02_im_correlation_model/{ft}/residual_partitioning/` | 19 (total residuals), 20-21 (lmer partitioning and summaries) |
| `data_processed/{ft}/correlations/` | `data_processed/02_im_correlation_model/{ft}/correlations/` | 30 (raw correlations) |
| `results/{ft}/` | `results/02_im_correlation_model/{ft}/*_PD.csv` | 35 (positive-definite correlations) |

`{ft}` is each of `normal`, `reverse`, `strikeslip`. The hdf5 and `.pkl` files,
`_to_delete/` and `test_tables_for_openquake/` were not copied, because nb 002 does not write
them.

`MANIFEST.json` lists every file with its path, the path it was copied from, its size, the
modification time of the original and its SHA-256. It also records the state of the code when
the copy was made: the git commit and the list of uncommitted changes (`git status
--porcelain`), the Python, pandas, numpy, OpenQuake, R and lme4 versions, and the SHA-256 of
`manage_flatfiles.py` and of nb 002. Every copy was re-hashed against its source after
copying, and all matched. The files were copied with their original modification times and then
set read-only.

`checkpoint_report.md` is the report written after nb 002 was rerun on the unchanged code and
compared with this baseline (task step A6).

## The files are not all from one run

The flatfiles are dated **2026-10-02**, but the residuals and correlations are dated
**2026-03-27**. The baseline is therefore not the output of a single run of nb 002: the
residuals and correlations were computed from flatfiles made in March, and the flatfiles were
written again in October. The rerun of the unchanged code (task step A4) therefore tests two
things at once: that today's code reproduces the October flatfiles, and that it reproduces the
March residuals and correlations from the flatfiles it rebuilds itself.

## Rules

- Do not edit anything here and do not use it as a live input to any notebook or script. It is
  a frozen reference for one comparison.
- The folder can be deleted once the bug check is closed.
