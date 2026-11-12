# Checkpoint report: rerun of nb 002 on the unchanged code (task step A6)

Task: `dump/tasks_and_plans/task_20261111_asc_flatfile_filter_bug_check.md`.
Written 2026-11-11, after nb 002 was rerun top to bottom with `manage_flatfiles.py` unchanged
and its outputs compared with this baseline folder.

## Result

**All 168 output files are byte-identical to the baseline.** Nothing is "within noise" and
nothing differs. Today's code reproduces the October flatfiles exactly and, from them, the March
residuals, lmer summaries, raw correlations and positive-definite correlations exactly. The
mixed file dates (see `README.md`) therefore do not matter: the baseline behaves as the output
of one run of the current code. Phase B can go ahead.

| stage | files | identical | within noise | different | max abs diff |
|---|---|---|---|---|---|
| 1 flatfiles | 12 | 12 | 0 | 0 | 0 |
| 2 total residuals | 12 | 12 | 0 | 0 | 0 |
| 3 partitioned residuals | 12 | 12 | 0 | 0 | 0 |
| 3 lmer summary JSONs | 12 | 12 | 0 | 0 | 0 |
| 4 raw correlations | 60 | 60 | 0 | 0 | 0 |
| 4 PD correlations | 60 | 60 | 0 | 0 | 0 |
| **total** | **168** | **168** | 0 | 0 | 0 |

Each stage covers 3 fault types x 4 regions (x 5 residual types for the correlations). The
table with one row per file is in the appendix. It was produced by
`dump/bug_check/compare_002_outputs.py <this folder> live` (the comparison treats differences up
to 1e-10 as floating-point noise; none occurred). The full output is in
`dump/bug_check/outputs/A5_compare.md` and `.json`.

## How the rerun was done

```
jupyter nbconvert --to notebook --execute --inplace --allow-errors \
    --ExecutePreprocessor.timeout=-1 notebooks/03_WP1_ground_motion_set/002-im_correlations.ipynb
```

`--allow-errors` was added to the command in the task. nb 002 has a bug in cells 70, 76 and 81
(conditional spectra section): they use `conditioning_period`, which is never defined (the
variable is `conditioning_periods`). Without `--allow-errors`, nbconvert stops at cell 70 and
does not save the notebook, which would have lost the cell outputs needed for this review. The
error comes after every output file is written (cell 35), so it does not affect the comparison.
Cells 70-72, 76-77 and 81-82 now show `NameError` outputs in the saved notebook.
Log: `dump/bug_check/logs/A4_nb002_rerun.log`.

## Leaked records (step B0, from the baseline ASC flatfiles)

The leakage is the same for all three fault types (the ASC flatfiles are identical, see the
rake section). Shares are of the baseline ASC totals. Stations are counted by `station_code`,
as in the lmer summaries.

| regional list | records | % of ASC records | events | % of ASC events | stations | % of ASC stations |
|---|---|---|---|---|---|---|
| sslab | 24 | 0.13 | 4 | 0.38 | 21 | 1.12 |
| sinter | 382 | 2.07 | 63 | 6.06 | 123 | 6.58 |
| vran | 0 | 0 | 0 | 0 | 0 | 0 |
| in more than one list | 0 | 0 | 0 | 0 | 0 | 0 |
| **any regional list** | **406** | **2.20** | **67** | **6.45** | **126** | **6.74** |
| ASC total | 18,481 | 100 | 1,039 | 100 | 1,869 | 100 |

No record is in more than one list.

Records surviving each stage of the ASC filter:

| stage | all records | regional (any) | sslab | sinter | vran | regional events |
|---|---|---|---|---|---|---|
| ESM flatfile | 23,014 | 2,323 | 704 | 681 | 938 | 349 |
| (i) regional-record filter (no-op) | 23,014 | 2,323 | 704 | 681 | 938 | 349 |
| (ii) hypocentral depth <= 40 km | 20,947 | 576 | 29 | 547 | 0 | 127 |
| (iii) >= 3 records per event | 19,941 | 503 | 26 | 477 | 0 | 74 |
| missing-data drops (cell 15) = saved ASC flatfile | 18,481 | 406 | 24 | 382 | 0 | 67 |

The depth cut does nearly all the work: it removes every Vrancea record and 96 % of the in-slab
records, but only 20 % of the interface records. The interface events are mostly shallow, so
382 of the 681 interface records (56 %) end up in the ASC set. Script:
`dump/bug_check/count_leakage.py`; tables in `dump/bug_check/outputs/leakage_*.csv`.

## Rake issue: confirmed (not changed)

Checked on the baseline with `dump/bug_check/check_rake_issue.py`.

1. **The ASC flatfile ignores the fault-type rake.** For ASC, every output (flatfile, total and
   partitioned residuals, all raw and PD correlations) is byte-identical for normal, reverse and
   strikeslip. In the ASC flatfiles the `rake` column is the same in all 18,481 rows for all
   three fault types. That matches the code: cell 13 passes `esm_df` (rake = 90 for unknown
   mechanisms, set in cell 10) to `get_active_shallow_crust_flatfile`, not `df`.
2. **`set_new_rake` does not update the derived geometry.** In the regional flatfiles `rake`
   differs between reverse and the other fault types in 67 (sslab), 252 (sinter) and 114
   (vran) rows, but `fault_dip`, `rup_width` and `ztor` differ in **0** rows. They keep the
   values derived from rake = 90.

Consequence seen in the outputs: the sinter flatfiles differ between fault types, but the
sinter residuals and correlations are identical across fault types. The sinter GMMs give the
same predictions whatever the rake. Only sslab and vran outputs change with the fault type.

## Run log

- **lme4:** no warnings. `run_r_script` discards R's stderr when the script succeeds, so the
  notebook cannot show lme4 warnings. To check, the R script was rerun on scratch copies of the
  rerun's total residuals (all three fault types), with stderr captured. stderr held only
  package-loading messages (tidyverse attach, masked objects); there were no singular-fit,
  convergence or boundary warnings.
- **Positive definiteness:** a PD matrix was found for all 60 (fault type, region, residual
  type) combinations. Iterations are not printed (`print_out=False`).
- **Other:** a `DtypeWarning` (mixed column types) in cell 68 when the flatfile is read without
  `dtype` for the conditional-spectra section, which is harmless here. The `NameError`s in cells
  70-82 are described above. nbconvert also printed a zmq/Proactor event-loop
  `RuntimeWarning` (Windows, harmless).

## Appendix: one row per output file

| stage | file | status | max abs diff |
|---|---|---|---|
| 1_flatfile | `normal/flatfiles/asc_normal_flatfile.csv` | identical | 0 |
| 1_flatfile | `normal/flatfiles/sslab_normal_flatfile.csv` | identical | 0 |
| 1_flatfile | `normal/flatfiles/sinter_normal_flatfile.csv` | identical | 0 |
| 1_flatfile | `normal/flatfiles/vran_normal_flatfile.csv` | identical | 0 |
| 2_total_residuals | `normal/residual_partitioning/asc_normal_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `normal/residual_partitioning/sslab_normal_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `normal/residual_partitioning/sinter_normal_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `normal/residual_partitioning/vran_normal_total_residuals.csv` | identical | 0 |
| 3_partitioned_residuals | `normal/residual_partitioning/asc_normal_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `normal/residual_partitioning/asc_normal_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `normal/residual_partitioning/sslab_normal_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `normal/residual_partitioning/sslab_normal_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `normal/residual_partitioning/sinter_normal_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `normal/residual_partitioning/sinter_normal_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `normal/residual_partitioning/vran_normal_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `normal/residual_partitioning/vran_normal_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 4_raw_correlations | `normal/correlations/asc_normal_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/asc_normal_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/asc_normal_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/asc_normal_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/asc_normal_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sslab_normal_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sslab_normal_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sslab_normal_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sslab_normal_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sslab_normal_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sinter_normal_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sinter_normal_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sinter_normal_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sinter_normal_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/sinter_normal_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/vran_normal_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/vran_normal_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/vran_normal_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/vran_normal_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `normal/correlations/vran_normal_rho_dWSes.csv` | identical | 0 |
| 4_pd | `results/normal/asc_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/normal/asc_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/normal/asc_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/normal/asc_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/normal/asc_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sslab_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sslab_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sslab_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sslab_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sslab_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sinter_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sinter_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sinter_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sinter_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/normal/sinter_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/normal/vran_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/normal/vran_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/normal/vran_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/normal/vran_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/normal/vran_rho_dWSes_PD.csv` | identical | 0 |
| 1_flatfile | `reverse/flatfiles/asc_reverse_flatfile.csv` | identical | 0 |
| 1_flatfile | `reverse/flatfiles/sslab_reverse_flatfile.csv` | identical | 0 |
| 1_flatfile | `reverse/flatfiles/sinter_reverse_flatfile.csv` | identical | 0 |
| 1_flatfile | `reverse/flatfiles/vran_reverse_flatfile.csv` | identical | 0 |
| 2_total_residuals | `reverse/residual_partitioning/asc_reverse_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `reverse/residual_partitioning/sslab_reverse_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `reverse/residual_partitioning/sinter_reverse_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `reverse/residual_partitioning/vran_reverse_total_residuals.csv` | identical | 0 |
| 3_partitioned_residuals | `reverse/residual_partitioning/asc_reverse_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `reverse/residual_partitioning/asc_reverse_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `reverse/residual_partitioning/sslab_reverse_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `reverse/residual_partitioning/sslab_reverse_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `reverse/residual_partitioning/sinter_reverse_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `reverse/residual_partitioning/sinter_reverse_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `reverse/residual_partitioning/vran_reverse_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `reverse/residual_partitioning/vran_reverse_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/asc_reverse_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/asc_reverse_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/asc_reverse_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/asc_reverse_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/asc_reverse_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sslab_reverse_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sslab_reverse_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sslab_reverse_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sslab_reverse_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sslab_reverse_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sinter_reverse_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sinter_reverse_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sinter_reverse_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sinter_reverse_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/sinter_reverse_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/vran_reverse_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/vran_reverse_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/vran_reverse_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/vran_reverse_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `reverse/correlations/vran_reverse_rho_dWSes.csv` | identical | 0 |
| 4_pd | `results/reverse/asc_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/asc_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/asc_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/asc_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/asc_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sslab_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sslab_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sslab_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sslab_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sslab_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sinter_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sinter_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sinter_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sinter_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/sinter_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/vran_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/vran_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/vran_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/vran_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/reverse/vran_rho_dWSes_PD.csv` | identical | 0 |
| 1_flatfile | `strikeslip/flatfiles/asc_strikeslip_flatfile.csv` | identical | 0 |
| 1_flatfile | `strikeslip/flatfiles/sslab_strikeslip_flatfile.csv` | identical | 0 |
| 1_flatfile | `strikeslip/flatfiles/sinter_strikeslip_flatfile.csv` | identical | 0 |
| 1_flatfile | `strikeslip/flatfiles/vran_strikeslip_flatfile.csv` | identical | 0 |
| 2_total_residuals | `strikeslip/residual_partitioning/asc_strikeslip_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `strikeslip/residual_partitioning/sslab_strikeslip_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `strikeslip/residual_partitioning/sinter_strikeslip_total_residuals.csv` | identical | 0 |
| 2_total_residuals | `strikeslip/residual_partitioning/vran_strikeslip_total_residuals.csv` | identical | 0 |
| 3_partitioned_residuals | `strikeslip/residual_partitioning/asc_strikeslip_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `strikeslip/residual_partitioning/asc_strikeslip_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `strikeslip/residual_partitioning/sslab_strikeslip_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `strikeslip/residual_partitioning/sslab_strikeslip_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `strikeslip/residual_partitioning/sinter_strikeslip_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `strikeslip/residual_partitioning/sinter_strikeslip_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 3_partitioned_residuals | `strikeslip/residual_partitioning/vran_strikeslip_partitioned_residuals_lmer.csv` | identical | 0 |
| 3_lmer_summary | `strikeslip/residual_partitioning/vran_strikeslip_partitioned_residuals_summary_lmer.json` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/asc_strikeslip_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/asc_strikeslip_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/asc_strikeslip_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/asc_strikeslip_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/asc_strikeslip_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sslab_strikeslip_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sslab_strikeslip_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sslab_strikeslip_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sslab_strikeslip_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sslab_strikeslip_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sinter_strikeslip_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sinter_strikeslip_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sinter_strikeslip_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sinter_strikeslip_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/sinter_strikeslip_rho_dWSes.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/vran_strikeslip_rho_total.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/vran_strikeslip_rho_total_B.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/vran_strikeslip_rho_dBe.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/vran_strikeslip_rho_dS2Ss.csv` | identical | 0 |
| 4_raw_correlations | `strikeslip/correlations/vran_strikeslip_rho_dWSes.csv` | identical | 0 |
| 4_pd | `results/strikeslip/asc_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/asc_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/asc_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/asc_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/asc_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sslab_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sslab_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sslab_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sslab_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sslab_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sinter_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sinter_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sinter_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sinter_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/sinter_rho_dWSes_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/vran_rho_total_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/vran_rho_total_B_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/vran_rho_dBe_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/vran_rho_dS2Ss_PD.csv` | identical | 0 |
| 4_pd | `results/strikeslip/vran_rho_dWSes_PD.csv` | identical | 0 |