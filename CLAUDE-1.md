# Project facts — PhD (site-specific vs fixed record set fragility comparison)

Standing assumptions for this project. Read before answering anything about the
analysis roadmap, the bootstrap, or the A-series models.

---

## 1. Study dimensions — ALWAYS ASSUME 120 ROWS

**Unless explicitly told otherwise, assume the full study:**

- **120 rows.** A row = one (site, structure) pair.
- **60 sites.** Each site hosts **two structures: a 3-storey and a 5-storey.**
  So every site contributes exactly 2 rows.
- **51 design groups.** A design group is a set of sites sharing one structural
  design. `g[i]` maps row -> design group; `sigma[i]` maps row -> site.
- Design and site are **crossed, not nested**: a design spans several sites, a
  site hosts several designs.

Do not silently reason about the 60-row / 25-group 3-storey subset. That subset
exists only because the 5-storey runs were incomplete at one point
(`N_STOREYS = [3]` in nb 073); results computed on it are provisional and must
be labelled as such. If a question is about that subset, the user will say so.

---

## 2. The three arms and their record sets

This is the distinction that governs the whole covariance structure. Get it
wrong and every conclusion about variance components is wrong.

| arm | symbol | varies with | record set | resampling |
|---|---|---|---|---|
| MSA site-specific | `a_i` | site & structure | **selected per stripe, per site**, by GCIM conditioned on that stripe's IM level | binomial resample of each stripe's collapse count, **independently per stripe** |
| MSA fixed set | `b_j` | design only | **one fixed FEMA P695 set**, shared by every stripe and every design | resample record *indices*; one resample applied coherently across all stripes and all designs |
| IDA fixed set | `c_j` | design only | same fixed FEMA P695 set as MSA-FX | same shared record resample |

### Why MSA-SS stripes are independent — do not "fix" this

In a true site-specific MSA the records are **re-selected at every stripe**
(GCIM conditions on the IM level, so each stripe gets its own selection matched
to that stripe's conditional spectrum). No record identity carries across
stripes, so there is nothing to preserve, and the independent per-stripe
binomial resample in `bootstrap_site_msa_fragilities` is the **correct** model,
not a degraded approximation to a coherent one.

MSA-FX is the odd one out: it deliberately holds ONE record set fixed across
every stripe and every structure. That is the point of it in this study — the
shared set is what generates the dense `V_bb`, which is what lets the record-set
contribution be isolated. **Do not treat the FX arm's structure as the norm and
read the SS arm as a broken version of it.** That mistake has been made once.

### Consequences for V_known

- `V_bb` (`Cov_r(b)`) is **dense**: all designs share one record set.
  `Z V_bb Z'` therefore couples rows within AND across design groups.
- `V_aa` (`Cov_r(a)`) is **block-diagonal by site**: different sites use
  different records so rows at different sites are independent, but the two
  structures AT a site are not — see below.

### Same-site coupling at 120 rows — RESOLVED: it exists

**The 3-storey and 5-storey structures at a site share records whenever they are
analysed at the same stripe IML, and this does happen.** Sharing is always
stripe-by-stripe (never across stripes, per above), so it is partial: the two
structures' stripe sets overlap rather than coincide.

Therefore `Cov(a_3s,k, a_5s,k) != 0` in general, and `V_aa` is **block-diagonal
by site with a 2x2 block per site**, not diagonal. Treat the two rows at a site
as dependent unless shown otherwise for a particular site.

The consequence for the fit: `V_known` must be built as the FULL sample
covariance of the replicate matrices,

    V_known = Cov_r(a) + Z Cov_r(b) Z'

not from the diagonals alone. Built that way the code carries no assumption
about which pairs are coupled — whatever structure exists is reported. The one
thing this does not fix is that `np.cov` can only report the covariance the
*resampling scheme* generates: the replicates must share record draws between
the two structures wherever the underlying analyses shared records, or the
coupling will read as zero however the covariance is computed.

---

## 3. Notation conventions

- Index maps are written **Gelman & Hill style with square brackets**:
  `b_{g[i]}`, `d_{g[i]}`, `s_{sigma[i]}`. Not `g(i)`.
- `a_i`, `b_j`, `c_j` are **log**-medians: `a_i = ln(theta_SS_i)` etc.
- Contrasts: `y^(0) = a - c` (A0), `y^(1) = a - b` (A1), `y^(2) = b - c` (A2).
- Variance components: `tau_d^2` design, `tau_s^2` site, `tau_u^2` row-level
  (the design x site interaction). `tau_u` is named after the symbol `u_i`, not
  after a word — the user has decided to keep that inconsistency.
- `v_a[i]` = SS-arm sampling variance for row i (diagonal only).
  `v1[i]` = combined `Var_r(a_i - b_{g[i]})` = `v_a[i] + V_bb[g[i],g[i]]`
  (exact — the arms use disjoint record sets, so the cross term is zero).

---

## 4. Working preferences

- **Explain in plain English**, minimum jargon, with an analogy where it helps.
  Include the mathematics where it clarifies rather than decorates.
- **Cite textbooks with chapter AND page number.** The accessible list is in the
  project instructions. Page numbers are from standard printings and should be
  flagged as approximate — the PDFs are not in `literature/` yet.
- **All code gets extensive documentation comments** explaining what each part
  does and why, not just what. This is a standing instruction.
- Do not write code unless asked.
- **NEVER write to files on the user's machine unless explicitly told to.**
  Do not stage a file for commit, and do not commit one, on your own initiative
  — not even a file you just wrote, not even an obvious edit to a file already
  under discussion. Produce the result in the chat, say what it is, and wait.
  "Write it", "save it", "commit it to <path>" is the green light; anything
  short of that is not.
- Before editing a file on the user's machine that was staged earlier in the
  session, RE-STAGE it first. A staged copy is a point-in-time snapshot and the
  user may have edited the file since. Working from a stale snapshot has already
  clobbered an edit once.
- Key files: `admin/analysis_models.md` (the roadmap),
  `phd_project/scripts/metaregression_analysis.py` (the library),
  `notebooks/03_WP1_ground_motion_set/073-statistical_models.ipynb` (A0/A1/A2).

---

## 5. Model ladder for A1 (for orientation)

| fit | model | estimator |
|---|---|---|
| A1a | `y_i = m1 + u_i + eps_i` | DL (Borenstein Ch. 12/14) **and** REML |
| A1b | `+ d_{g[i]}` | REML only |
| A1c | `+ d_{g[i]} + s_{sigma[i]}` — crossed, the reportable fit | REML only, full `V_known` |

DL cannot extend past A1a: it is one moment equation with one unknown, and its
`Q` statistic presupposes a diagonal sampling covariance. Keep the DL code as
the regression test for the REML implementation.

A1c is **not** "four-level" — site and design are crossed, so level-counting
stops being the right language.
