# A1 — modelling steps (SS vs MSA-FX)

Working summary of the A1 build. Supersedes nothing in `analysis_models.md` until
merged; keep as a working note.

---

## 0. Notation

- $i$ — row = one (site, structure). Full study: $n=120$. Currently running: $n=60$
  (3-storey only, `N_STOREYS=[3]`).
- $j=g[i]$ — design group. Full: $J=51$. Currently: $J=25$.
- $k=\sigma[i]$ — site. $K=n_s$.
- $a_i=\ln\hat\theta^{\mathrm{SS}}_i$, $b_j=\ln\hat\theta^{\mathrm{MSA\text{-}FX}}_j$
  (design-level, broadcast to rows).
- Outcome: $y^{(1)}_i = a_i - b_{g[i]}$.
- $v^{(1)}_i$ — taken **directly** as the bootstrap variance of the difference
  $a^{(r)}_i-b^{(r)}_{g[i]}$, not assembled as $v^a+v^b$.

---

## 1. The three fits

| fit | model | purpose | estimator |
|---|---|---|---|
| **A1a** | $y_i = m_1 + u_i + \varepsilon_i$ | two-level baseline, hand calcs, pedagogy | DL **and** REML |
| **A1b** | $+\,d_{g[i]}$ | three-level, rows nested in designs; gives the ICC | REML |
| **A1c** | $+\,d_{g[i]} + s_{\sigma[i]}$ | **crossed** design × site — the reportable model | REML, full $\boldsymbol\Sigma$ |

### The levels

| level | name | unit | random term | variance |
|---|---|---|---|---|
| 1 | within-study / sampling | one record resample | $\varepsilon_i$ | $v_i$, **known** |
| 2 | between-study | one row = (site, structure) | $u_i$ | $\tau^2$ (A1a) / $\tau_u^2$ |
| 3 | between-group | one design group | $d_j$ | $\tau_d^2$ |

One observation per "study" — this only works because $v_i$ is supplied from the
bootstrap rather than estimated from replication. That is the defining feature of
a meta-analysis versus an ordinary regression.

**A1c is not "four-level".** Sites and designs are *crossed*, not nested — a design
spans several sites. Describe it as "a mixed model with two crossed grouping
factors plus known sampling variance" (G&H Ch. 13.5, ~pp. 289–291).

### Interpreting the terms

- $d_j$ — this design runs high/low **everywhere**. $\tau_d$ = spread across designs.
- $s_k$ — this site runs high/low for **everything**. $\tau_s$ = spread across sites.
- $u_i$ — this design **at this site** departs from $m_1+d+s$. The **interaction**.
  Physically real and expected here (basin long-period content matters for a tall
  frame, not a stiff wall). If $\tau_u\approx0$ the effects are separable and a user
  could apply a site factor × a building factor; if not, they can't.
- $\varepsilon_i$ — finite-record estimation error. Vanishes with infinite records.
  **Not** the regression residual: an OLS residual is $u_i+\varepsilon_i$ bundled.

---

## 2. A1c in matrix form

$$
\mathbf y^{(1)} = \mathbf 1 m_1 + \mathbf Z\mathbf d + \mathbf S\mathbf s + \mathbf u
+ \boldsymbol\varepsilon^a - \mathbf Z\mathbf e^b
$$

$\mathbf Z$ ($n\times J$) design indicator, $\mathbf S$ ($n\times K$) site indicator,
$\mathbf e^b\sim N(\mathbf 0,\mathbf V^{bb})$ the MSA-FX sampling error.

**$\mathbf d$ and $\mathbf e^b$ are premultiplied by the same $\mathbf Z$.** That one
line is the whole identifiability problem.

$$
\boldsymbol\Sigma = \tau_d^2\mathbf Z\mathbf Z' + \tau_s^2\mathbf S\mathbf S'
+ \tau_u^2\mathbf I + \operatorname{diag}(v^a_i) + \mathbf Z\mathbf V^{bb}\mathbf Z'
$$

Last two terms **known**; only three scalars estimated.

### Correlation structure

| rows $i\neq i'$ | $\Sigma_{ii'}$ |
|---|---|
| same design, diff. site | $\tau_d^2 + v^b_j$ |
| same site, diff. design | $\tau_s^2 + V^{bb}_{jj'}$ |
| neither shared | $V^{bb}_{jj'}$ |
| diagonal | $\tau_d^2+\tau_s^2+\tau_u^2+v^a_i+v^b_j$ |

Three readings:

1. $\tau_d^2$ and $v^b_j$ occupy **identical cells** → aliased, separable only through
   the variation of $v^b_j$ across $j$. Weak.
2. **No cell is zero** — $\mathbf V^{bb}$ is dense (shared record set), so even rows
   sharing nothing are correlated.
3. $\tau_u^2$ appears only on the diagonal; separable from $v^a_i$ only because
   $v^a_i$ is known.

### $\mathbf V^{bb}$

$J\times J$ sampling covariance of the MSA-FX log-medians,
$\hat{\mathbf V}^{bb}=\frac{1}{k-1}\sum_r(\mathbf b^{(r)}-\bar{\mathbf b})(\mathbf b^{(r)}-\bar{\mathbf b})'$
= `np.cov(B.T)`.

Dense because **all designs share one record set and one bootstrap index set**. SS
errors are independent (site-specific records) → diagonal. The shared component does
**not** average away over rows, which is why the analytic SE is too small.

- Needs $k>J$ for non-singularity. $k=1000$, $J=51$: fine, but 1326 entries from
  1000 replicates is noisy.
- Consider a one-factor approximation $\mathbf V^{bb}\approx\boldsymbol\lambda\boldsymbol\lambda'+\operatorname{diag}(\psi_j)$
  — better conditioned, loadings interpretable (similar $T_1$ → similar loading).
- **Smell test:** mean off-diagonal correlation must be clearly positive. Near zero
  ⇒ the replicate loop isn't sharing record indices across designs.

---

## 3. Estimator choice

**DL is fine for A1a and cannot be used for A1b/A1c** — not a bias argument, an
existence argument:

- DL is *one* moment equation ($\mathbb E[Q]=(n-1)+C\tau^2$), one unknown. A1b has two
  variance components, A1c has three.
- $Q=\sum w_i(y_i-\hat m_{FE})^2$ presupposes a **diagonal** $\mathbf V$. Once
  $\mathbf Z\mathbf V^{bb}\mathbf Z'$ is in $\boldsymbol\Sigma$ there is no $Q$ to write.

REML's objective is written in terms of $\boldsymbol\Sigma$ directly and is indifferent
to both issues:

$$
\ell_{\text{REML}}=-\tfrac12\big[\ln|\boldsymbol\Sigma|+\ln|\mathbf 1'\boldsymbol\Sigma^{-1}\mathbf 1|+\mathbf y'\mathbf P\mathbf y\big],
\qquad \hat m_1=\frac{\mathbf 1'\boldsymbol\Sigma^{-1}\mathbf y}{\mathbf 1'\boldsymbol\Sigma^{-1}\mathbf 1}
$$

Borenstein's "differences are small" is correct **within the two-level,
one-component setting** — it applies to A1a only.

**Empirical check:** `v1_boot.max()/v1_boot.min()`. $\lesssim5$ ⇒ DL≈REML;
$\gtrsim20$ ⇒ visible gap, DL lower, report REML.

**Subtle reason REML matters for A1c:** the Route-2 bias correction bundles two
effects of *opposite sign* — shared-$b_j$ contamination (up) and moment-estimator bias
(down). With DL they partially cancel and any "clean" $\hat\tau_d^2$ is clean by
coincidence. REML's own bias is small enough that the correction isolates the
contamination.

Keep DL permanently as a regression test: any REML refactor must reproduce the DL
number given a diagonal $\mathbf V$ and one component.

---

## 4. Two routes to an uncontaminated $\hat\tau_d^2$

**Route 1 — full-$\boldsymbol\Sigma$ REML (preferred, exact).** Supply
$\mathbf Z\mathbf V^{bb}\mathbf Z'$ as a known offset; maximise over the three
components. Clean by construction. PyMare cannot do this (variance *vector* only);
`metafor::rma.mv(y, V=Vfull, random=list(~1|design, ~1|site, ~1|row))` in R; one
`MvNormal` in PyMC (reused for A4/A5).

**Route 2 — bootstrap bias correction (cross-check).** Fit with diagonal $\mathbf V$
inside every outer-bootstrap replicate:

$$
\hat\tau^2_{d,\text{BC}}=2\hat\tau_d^2-\tfrac1k\textstyle\sum_r\hat\tau_d^{2(r)}
$$

(E&T Ch. 10, ~pp. 124–133). Bias correction inflates variance, and $\hat\tau_d^2\ge0$
so the result can go negative — truncate at 0 and **report that it did**, because
"negative" *is* the finding (no design effect beyond noise).

**Cross-check:** Routes 1 and 2 should agree; the gap
$\hat\tau_d^2-\hat\tau^2_{d,\text{BC}}$ should land near $\overline{v^b}$.

**Pre-fit smell test:** if $\hat\tau_d^2\approx\overline{v^b}$, it's noise.

---

## 5. Build-and-verify ladder

| step | build | check against |
|---|---|---|
| 1 | two-level, diagonal $\mathbf V$, DL by hand | Borenstein Ch. 14 worked table (pp. 87–95) **first**, then PyMare |
| 2 | same model, REML by numerical optimisation | reproduces DL approximately; PyMare `method='REML'` exactly |
| 3 | swap diagonal for full $\boldsymbol\Sigma$ | setting $\mathbf V^{bb}=\operatorname{diag}(v^b_j)$ **must** return step 2's answer |
| 4 | add $\mathbf Z$ (design) = **A1b** | $\hat\tau^2_{\text{step2}}\approx\hat\tau_d^2+\hat\tau_u^2$ |
| 5 | add $\mathbf S$ (site) = **A1c** | fake-data simulation |

Verify **every** step by fake-data simulation (G&H Ch. 8.1–8.2, ~pp. 155–160):
simulate from known $(m_1,\tau_d,\tau_s,\tau_u)$ using the *real* $\mathbf Z$,
$\mathbf S$, $\mathbf V^{bb}$; refit; check recovery and ~95% interval coverage over a
few hundred runs. This also answers, before any real fitting, whether $\tau_s$ is
recoverable at all given the actual $n_s$ and the ragged design × site table.

### PyMare check (step 1)

Not currently installed — `pip install pymare`.

Compare **four** quantities, not one: $\hat\tau^2$, $\hat m$, $\operatorname{SE}(\hat m)$,
$Q$. DL is closed form ⇒ agreement to ~machine precision; any gap is a bug.

Gotchas:

1. Zero-truncation — confirm PyMare truncates too.
2. SE must be the unadjusted $1/\sqrt{\sum w^*}$, **not** Knapp–Hartung.
3. Row alignment — assert identical indices before `.to_numpy()`; `v1_boot` and
   `y1_est` come from different objects.
4. Dropped rows — `compute_re_summary_effect` drops incomplete studies and recounts
   $k$; PyMare won't. Feed both pre-cleaned arrays.

Put it in `tests/` with the Borenstein table hard-coded, not in the notebook — it then
protects the A1b/A1c refactor of `metaregression_analysis.py`.

---

## 6. Inference

Point estimates from Route 1. **All** intervals ($m_1,\tau_d,\tau_s,\tau_u$) from the
outer bootstrap — the closed-form SE assumes diagonal $\mathbf V$ and there is no honest
closed form for a variance component here.

Form the site-vs-design comparison **per replicate**, not as two point estimates:

$$
\Delta^{(r)}=\hat\tau_d^{(r)}-\hat\tau_s^{(r)}
\quad\text{or}\quad R^{(r)}=\hat\tau_d^{(r)}/\hat\tau_s^{(r)}
$$

Relative uncertainty on $\hat\tau_s$ is roughly $1/\sqrt{2(n_s-1)}$ — with a dozen
sites, ~20% before anything else. **"We cannot distinguish them" is a legitimate
reportable outcome.**

---

## 7. Reporting

Ratio units, not variances. $I^2$ is secondary (two $\tau$'s make it doubly unhelpful).

| quantity | report as |
|---|---|
| $\hat m_1$ | $e^{\hat m_1}$ — mean SS/MSA-FX correction |
| $\hat\tau_d$ | $e^{\hat\tau_d}$ — multiplicative spread across designs |
| $\hat\tau_s$ | $e^{\hat\tau_s}$ — across sites |
| $\hat\tau_u$ | $e^{\hat\tau_u}$ — residual / interaction |
| shares | $\tau_d^2/(\tau_d^2+\tau_s^2+\tau_u^2)$, etc. |

**Main figure:** Gelman-style ANOVA display (G&H Ch. 22, ~pp. 487–500) — one row per
variance component, estimate + interval, common axis. That plot *is* the answer.

**Secondary table:** $\hat\tau^2_{\text{A1a}}$ vs
$\hat\tau_d^2+\hat\tau_s^2+\hat\tau_u^2$ — where the two-level heterogeneity went.

### Two caveats that belong in the text

1. **Variance share = importance × sampled range.** $\tau_d$ describes the designs you
   built, $\tau_s$ the sites you picked. Report the span of key covariates ($T_1$,
   ductility; $V_{s30}$, $k_1$) for each so the reader can calibrate. Finite-population
   vs superpopulation: G&H Ch. 21.2, ~pp. 458–460 — only the superpopulation reading
   supports a general claim.
2. **Effective $n$ for the site component is $n_s$, not 120.**

### What A1c licenses

- $\tau_s\gg\tau_d$ → a global factor is inadequate; **site information must be
  supplied downstream** → motivates the site meta-regression A0b/A3.
- $\tau_d\gg\tau_s$ → the correction is a property of the structure, usable without a
  hazard deaggregation → regress on structural covariates instead.
- Comparable, or $\Delta$ straddles 0 → say so; deliverable is $m_1$ + full PI.

---

## 8. Current status (notebook 073, §2.2)

§2.2 **is A1a**, correctly implemented: contrast, broadcast $b_{g[i]}$, $v_i$ as the
bootstrap variance of the difference, DL per Borenstein Eq. 12.2/12.7–12.9, $v_i$ held
fixed while $\hat\tau^2$ is re-estimated per replicate, bootstrap SE, $t_{n-2}$ PI.

Open items:

- [ ] $n=60$, $J=25$ (3-storey only) — plan's 120/51 arithmetic does not yet apply.
- [ ] **SE_boot 0.0283 vs SE_model 0.0112 — a factor of 2.5.** This is the empirical
      justification for the whole full-$\mathbf V$ apparatus. Promote from a `print` to
      a documented result.
- [ ] Rename `y0`/`v0_boot`/`lb`/`ub` → `y1`/`v1_boot`/`*_A1`; they collide with A0's
      names in the same kernel namespace.
- [ ] `compute_heterogeneity_stats` does not call `drop_incomplete_studies` — can give a
      different $\hat\tau^2$ from `compute_re_summary_effect` on the same data. Silent.
- [ ] `Q_df` is $Q-(n-1)$, not the df. Rename `Q_minus_df`.
- [ ] `compute_prediction_interval()` is a stub; PI computed inline. Fold in.
- [ ] Add REML alongside DL at A1a.
- [ ] Report $\hat\tau=0.0694 \Rightarrow e^{\hat\tau}=1.072$ (±7% site-to-site) with
      mean effect $0.9955$, in ratio units.
- [ ] $I^2=64.6\%$ — moderate, not ~100% as §4 of the plan predicted. Soften that caveat.

---

## 9. References

| topic | source |
|---|---|
| OLS → WLS → GLS, correlated errors | Chatterjee & Hadi Ch. 7 (~pp. 171–195), Ch. 8 (~pp. 197–216) |
| Two-level RE model; worked hand calc | Borenstein Ch. 12 (pp. 69–75); Ch. 14 (pp. 87–95) |
| $\tau$ over $I^2$ | Borenstein Ch. 16 (~pp. 117–125) |
| Complex/dependent data structures | Borenstein Ch. 26 (~pp. 245–248) — names the problem, doesn't solve it |
| Known-variance hierarchical normal model | **BDA3 Ch. 5.4–5.5 (~pp. 113–124)** — best single walkthrough |
| Meta-analysis example | BDA3 Ch. 5.6 (~pp. 124–128) |
| Priors on variance components | BDA3 Ch. 5.7 (~pp. 128–132) |
| Bayesian ANOVA / variance components | BDA3 Ch. 15.6 (~pp. 395–398) |
| Partial pooling; many groups, few members | G&H Ch. 12.2 (~pp. 252–254), Ch. 12.9 (~pp. 275–276) |
| **Crossed / non-nested models** | **G&H Ch. 13.5 (~pp. 289–291)** |
| Variance-component comparison, ANOVA display | G&H Ch. 22 (~pp. 487–500) |
| Finite- vs superpopulation SD | G&H Ch. 21.2 (~pp. 458–460) |
| Fake-data simulation | G&H Ch. 8.1–8.2 (~pp. 155–160) |
| Coding multilevel models | G&H Ch. 17 (~pp. 356–380) |
| Bootstrap SE / bias / replicate count | E&T Ch. 6.4 (~pp. 50–53), Ch. 10 (~pp. 124–133), Ch. 25 |

*Page numbers are from the standard printings and may be off by a page or two.*
