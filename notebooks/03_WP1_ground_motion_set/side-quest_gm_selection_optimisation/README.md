# Side quest — optimising the site-specific record selection

The original site-specific (SS) record sets were selected with a weak optimisation step.
This folder holds the investigation of whether stronger optimisation changes the study's
result enough to justify re-running the MSA-SS analyses. Its outcome is the main line: the
KS + moments scheme (`AvgSA_03__opt_ks_mu_sigma`) is now selected by the plain-numbered
notebooks `032`, `033`, `040` and `042` one folder up.

## Notebooks

| Record set (`config.yaml`, `record_sets.ss`) | Scheme | Notebooks |
|---|---|---|
| `AvgSA_03__ks` — the original sets, the only ones analysed so far (NLTHA) | `SELECTION_CONFIG` | `032-…compute`, `033-…postprocess`, `040-…download`, `042-…jsons` |
| `AvgSA_03__opt_ks` — KS objective | `SELECTION_CONFIG_OPTIMISED` | the same four, suffix `-optimised` |
| `AvgSA_03__opt_mu_sigma` — mean / SD of each ln IM | `SELECTION_CONFIG_MU_SIGMA` | the same four, suffix `-optimised__mu_sigma` |

`034-comparison_original_and_optimised_gm_sets` compares all four sets (the three above and
the main-line `AvgSA_03__opt_ks_mu_sigma`) and decides whether a re-run is justified.

## Reproducibility

- Each notebook here is **pinned to its own record set** (`SS = "..."` near the top). None of
  them reads `record_sets.active_study`, so changing the main line never changes what they
  do or which folders they write.
- They select on a **frozen stripe grid**: the copy of the stripe IMLs as they were when the
  side quest ran, `data_processed/06_gm_selection/ss/AvgSA_03__ks/imls_for_selection_AvgSA_03.json`
  (`config.stripe_grid_path`). Stripes added to the main line later are ignored here.
- `034` uses only the original set's `(site, cond_iml)` stripes. It asserts every other set
  still covers them and drops any extra stripes, so new main-line stripes do not change it.
- `034` reads the fragilities, bootstraps and regression dataset of the study
  `AvgSA_03__ks__vs__femap695` (the original sets against FEMA P695), and caches its model
  fits in `data_processed/14_record_set_comparison/models/`, one folder per record set
  (`…__ks`, `…__opt_ks`, `…__opt_mu_sigma`, `…__opt_ks_mu_sigma`).

## Where the outputs are

| What | Folder |
|---|---|
| selection files, download / conversion lists | `data_processed/06_gm_selection/ss/{ss}/` |
| stripe pickles | `results/04_record_selection/ss/{ss}/` |
| converted record JSONs | `data_processed/07_gm_records/` (shared by every set: one file per physical record) |
| `034` tables / figures | `data_processed/14_record_set_comparison/`, `results/09_record_set_comparison/` |
