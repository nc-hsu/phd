# Baseline for the covmar migration — instructions for agents

This folder freezes what the current phd fitting code produces on the real study
data, so that the covmar package (a copy of that code, later refactored) can be
checked against it. It is the reference covmar is tested against: whatever is
written here must come from the **unmodified phd code**.

The design of the baseline (cases, tolerances, result format, calibration) is in
the covmar roadmap, `C:\Users\clemettn\Documents\pypackages\covmar\dev\roadmap.md`,
sections "Step 3 — baseline test matrix", "Case data", "Tolerances" and
"Adapter and harness". Read them before doing anything here. If this file and
the roadmap disagree, stop and ask.

---

## 1. Rules

- **New files only, and only in this folder.** Never edit, move or delete any
  existing file in the phd repository outside this folder — not the scripts, not
  the notebooks, not the config. The baseline is only valid if the phd code is
  exactly the code the thesis results came from.
- **Never write to the covmar repository.** Read its roadmap only.
- **`cases.json` and `tolerances.json` are the specification.** Do not change
  them to make something run; if a case cannot be built as specified, stop and
  report which case and why.
- **Git:** the phd rule applies — never add, commit, branch, tag or push. The
  user commits the baseline. The one exception is *read-only* provenance:
  `git rev-parse HEAD` and `git status --porcelain` may be run to record which
  commit produced the baseline and whether the working tree was clean.
- **The real data stays here.** Nothing in this folder is ever copied into the
  covmar repository. covmar's baseline tests read it in place via the
  `COVMAR_BASELINE_DIR` environment variable.
- All the phd working preferences apply (extensive comments, no change-history
  notes, type hints, `facecolor="w"` on any saved figure).

## 2. Layout

```
_baseline_metaanalysis_migration/
  CLAUDE.md               this file
  cases.json              the 34 cases (specification — do not edit)
  tolerances.json         comparison tolerances (specification — do not edit)
  phd_adapter.py          builds inputs and expected results from the phd code
  inputs/<case_id>.npz    what each fit consumes
  expected/<case_id>.json canonical result per case
  expected/<case_id>_replicates.npz   bootstrap cases only; for diagnosis, not compared
  calibration/report.json row-permutation noise floors per case and output
  provenance.json         phd commit, clean/dirty tree, library versions, date
```

## 3. Building each case

`phd_adapter.py` reads `cases.json` and, for each case, uses only the existing
phd functions:

- Load with `load_fragility_data(root, "AvgSA_03", quantity=...)`; build the
  contrast named in the case (`y0_avg`).
- **Engine level** (`"engine"` in `levels_run`): extract `y`, `X`, the full
  `V` and the factor codes, transform `V` as the case says, and call
  `mra.fit_reml` (univariate, with the case's `objective`) or
  `mra.fit_reml_mv` (multivariate, with the case's `optimiser` and `correlated`
  levels) directly.
  - `"diag"`: `np.diag(np.diag(V))`.
  - `"blockdiag"`: keep each outcome's own block, zero the between-outcome
    blocks.
  - `"dense"`: `V` unchanged.
- **Model level** (`"model"` in `levels_run`): build the equivalent
  `ModelSpec` / `JointModelSpec` and fit through the model layer, including the
  bootstrap cases (`bootstrap_scheme`, `corrected`, the case's `n_boot` and
  seed).
- When a case runs at both levels, both results are saved, and they must agree
  with each other within tolerance. If they do not, stop and report it: it means
  the engine and the model layer disagree in the original code.

## 4. Canonical names

The expected JSON uses covmar's general names, not phd names:

| phd | canonical |
|---|---|
| component `design` | `factor1` |
| component `site` | `factor2` |
| component `row` | `row` |
| fixed-effect columns | `intercept`, then the covariate column names |
| multivariate outcomes theta, beta | `0`, `1` |

`inputs/<case_id>.npz` stores `y`, `X`, `V` (as used), integer codes per row for
each factor, the level labels, the fixed-effect column names, and — for
multivariate cases — the stacking order of the outcomes in `y` (outcome 0 rows
first, then outcome 1). It also stores `mean_diag_V`, used for the scaled
absolute tolerance.

## 5. Result format

Exactly the canonical JSON in the roadmap ("Adapter and harness"): keyed by name
never by position; `tau2` for univariate fits, `T` (`var`, `cov`, `corr` per
level) for multivariate; `bootstrap` summaries (mean, SD, 2.5 / 50 / 97.5 %
quantiles per parameter) for bootstrap cases; NaN written as `null`; floats
written with full precision.

## 6. Calibration and checks

After the expected results are written:

1. Refit every case once with its rows randomly permuted (fixed seed, recorded),
   permuting `y`, `X`, `V` and the factor codes consistently.
2. Record the largest relative difference per output in
   `calibration/report.json`.
3. Report every case where a tolerance in `tolerances.json` is less than 10x the
   measured noise floor. Do not change any tolerance — report, and let the user
   decide.
4. Report every frozen tau^2 or T diagonal entry inside the near-threshold band
   (`[1e-10, 1e-6]`).
5. Resolve the `zero_boundary` case: the first grid case in file order with any
   zero-boundary flag set. Record which case it is, or that none exists.

## 7. Provenance

`provenance.json` records the phd commit hash, whether the working tree was
clean (a baseline from uncommitted code cannot be reproduced — if the tree is
dirty, say so prominently), Python / numpy / scipy / pandas versions, and the
date. Every expected JSON carries the same information under
`meta.produced_by`.

## 8. Finishing

Report: cases written (should be 34), any case that could not be built, the
engine-vs-model agreement, the calibration findings, the near-threshold
warnings, and the zero-boundary resolution. Then stop: the user reviews and
commits.
