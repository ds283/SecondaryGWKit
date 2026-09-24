# `QuadSourceIntegral` — what the quadrature tolerance costs, and what it buys

**Campaign:** [`prompts/handover`](../../prompts/handover/README.md) ·
**Prompt:** *none — opened by the A3 baseline pipeline run, not by a prompt* ·
**Issue:** `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`
**Reports to:** `prompts/levin-refactor` and `prompts/qsi-phase-groups`, which own
`DEFAULT_QUADRATURE_ATOL` and `DEFAULT_QUADRATURE_RTOL`
([`tolerance-convergence` README §0.4](../../prompts/tolerance-convergence/README.md)) — this
document **measures and hands over**; it does not change a constant
**Measured:** 2026-09-23 and 2026-09-24, on the tree at `050a7e3`, branch `handover-remedial`,
**clean** (`git status --porcelain` empty at the time of writing)
**Script:** [`quadsource_atol_sweep.py`](quadsource_atol_sweep.py), fixed at `51cf304`
**Instrument:** `main.py`'s own pipeline, reached through five exact textual substitutions; no
stage, tag or integrator is reimplemented
**Datastore:** `var/datastores/handover-atol-sweep*`, a copy of the A3 baseline store —
**deleted after this document was written**, which is why every number below is transcribed here
rather than left as rows (`prompts/run-registry` README §0 item 4)
**Written by:** Claude Opus 5

---

## 0. The answers

**1. `rtol` is the lever; `atol` is not, but `atol` is not inert either.** Relaxing `atol` collapses
the cost and destroys the result — at `1e-22` and `1e-25` several cases return with the **wrong
sign**, and `1e-28` is still 32 % out. `5255ac0`'s tightening from `1e-25` to `1e-32` was therefore
right in a *stronger* sense than the finding it cited.

**2. What that tightening did, unmeasured, was remove the only reachable acceptance branch for
production's squeezed geometry.** `_adaptive_levin` accepts a region on
`abserr < local_atol or relerr < rtol`, and `local_atol` is `atol` divided by the region's share of
the interval. On squeezed triples the bisection reaches up to 1,530,190 regions, so each region's
share of `1e-32` is around `1e-38` and the first branch is dead; `rtol = 1e-8` must then be met
against a phase good to about six digits, which it is not. Every such row runs to
`DEFAULT_LEVIN_MAX_DEPTH = 20` and is stored unconverged, at **220×** the mean cost of a row that
stops short of the cap.

**3. `rtol = 1e-7` is the loosest rung converged in all three response-redshift bands**, and
`rtol = 1e-6` is ruled out by the mid band alone, where it is **1.1e-02** wrong.

**4. Low $z_{\rm response}$ is representation-limited, not quadrature-limited.** Four rungs across
three decades agree to ~`1e-05`; on one case `1e-7` and `1e-8` are bit-identical. No tolerance
reaches beneath that floor, so the lever there is **D1**/**D2**.

**5. For science outputs someone should re-run at `rtol = 1e-9`** (user decision, 2026-09-24), with
the qualification of answer 4: at low $z$ it will buy cost and no accuracy; the band where it could
still bite is the mid one.

**Why `tolerance-convergence` prompt 06 did not see any of this, and is not superseded.**
[`QUADSOURCE-READONLY.md`](../tolerance-convergence/QUADSOURCE-READONLY.md) §0 finding 1 reports
`atol` **inert over twenty-eight decades**. That measurement stands. It scored the *residual*, on
the eighteen offline cases of `test_quadsource_integral.py`, and on those cases `atol` is inert. It
could not see this for two structural reasons: its instrument has **no $(k,q,r)$ triangle**, so the
squeezed geometry does not occur in it, and the statistic that moves here is the **cost**, which a
residual sweep does not record. Its own finding 4 recorded `DEFAULT_LEVIN_MAX_DEPTH = 20` and
`limit = 100` as *never chosen*; this document is what that costs.

---

## 1. The cost cliff, on the A3 baseline's 7600 stored rows

| `WKB_Levin_max_depth` | rows | mean `compute_time` |
|---|---|---|
| NULL (no Levin call) | 2489 | 0.062 s |
| 1–19 | 3054 | 0.776 s |
| **20** | **2057** | **170.6 s** |

A cliff, not a scaling: `max_depth` on the expensive rows is **20 exactly, with zero variance**.
Depth-20 rows carry **97.5 of the store's 98.2 CPU-hours** and **43.2 %** of them (889) are stored
`total_converged = 0`. Of the 195 rows costing over 60 s, **74.4 %** are unconverged, their median
`|total|` is 3.31e-24 and their median `total_abserr/|total|` is **2.15e-06** — six digits, against
the eight `rtol` demands.

The cost is **entirely Levin**: `WKB_Levin_elapsed` sums to **351,296 s** of the **353,489 s** of
`compute_time`, against **9 s** of `numeric_quad_compute_time` across all 7600 rows.

### 1.1 The geometry, and all three conditions are necessary

1. **`r == k ≫ q`** — the hard leg on the response, a soft `q`. The mirror `k ≪ q ≈ r` costs 0.3 s.
2. **Mid-range `k`** — peaks at `k = 9.7e6`/Mpc and is **absent at both ends**: `k = 1e5` has zero
   rows over 60 s, and so does `k = 3e8`.
3. **Low $z_{\rm response}$**, i.e. large $\eta_R$. Within one triple the cost grows roughly as
   $\eta_R^3$ and then flattens as the cap binds. For `(9.7e6, 9.85e5, 9.7e6)`: **15.5 s** at
   $z = 8343$, **737 s** at $z = 1589$, **6245 s** at $z = 174$, **12,714 s** at $z = 11$.

Binned by $k\eta_R$ the **median** cost is flat at ~1.3 s across **twelve decades** — Levin is doing
what Levin is for — and only the mean moves. Any statistic that does not separate the tail misses
this entirely.

---

## 2. Phase 1 — the mid band, $43.7 \le z_{\rm response} \le 1589$

Nine production work items (four severe, three moderate, two controls), eight tolerance pairs. The
zero point is the baseline's own rows read back as lookups: 7237.2 s over the nine, 7 at depth 20,
5 unconverged. Each cell is the worst relative difference of `total` against that zero point.

| `atol` | `rtol` | worst $\lvert\Delta\rvert/\lvert total\rvert$ | time, 9 cases | speed-up |
|---|---|---|---|---|
| 1e-22 | 1e-8 | 1.8e+00 | 3.0 s | — |
| 1e-25 | 1e-8 | 1.8e+00 | 2.5 s | — |
| 1e-28 | 1e-8 | 3.2e-01 | 5.6 s | — |
| 1e-30 | 1e-8 | 1.1e-02 | 100.5 s | — |
| 1e-32 | 1e-5 | 1.1e-02 | 4.9 s | ×1477 |
| 1e-32 | 1e-6 | 1.1e-02 | 31.1 s | ×233 |
| **1e-32** | **1e-7** | **1.9e-06** | **1060.5 s** | ×6.8 |
| 1e-32 | 1e-8 | — (reference) | 7237.2 s | 1 |

### 2.1 c5, the case that decides the pair

$(k, q, r) = (3.09\text{e}6,\ 1\text{e}5,\ 3.09\text{e}6)$ at $z_{\rm response} = 1589$:

| `rtol` | `total` | regions |
|---|---|---|
| 1e-5 | 2.30906966e-22 | 73 |
| 1e-6 | 2.30824301e-22 | 116 |
| **1e-7** | **2.33450622e-22** | **245** |
| 1e-8 | 2.33450857e-22 | 24,015 |

`1e-5` and `1e-6` agree with **each other** to 3.6e-04 and are both **1.1e-02** from the converged
value, which `1e-7` reaches in 245 regions and `1e-8` confirms to 1.0e-06 after 24,015.

**Two loose rungs agreeing is not convergence.** This is the counterexample inside this sweep, and
it is why phase 3b was run rather than inferred. It is also the whole pathology in two rows: 24,015
regions to change the answer by $10^{-6}$.

---

## 3. Phase 2 — the low band, $z_{\rm response} \le 6.32$

Seven items that no run had ever attempted, re-using triples phase 1 showed to be severe. No `1e-8`
rung: at these $\eta_R$ a production-pair cell is a $10^4$ s item.

| case | rtol 1e-5 | rtol 1e-6 | rtol 1e-7 | 1e-5 vs 1e-6 | 1e-6 vs 1e-7 |
|---|---|---|---|---|---|
| $k$=9.7e6 $q$=9.85e5 $z$=0.1 | 5.674108e-27 | 5.674034e-27 | 5.674095e-27 | 1.3e-05 | 1.1e-05 |
| $k$=9.7e6 $q$=3.09e6 $z$=0.91 | 1.105019e-26 | 1.105016e-26 | 1.105020e-26 | 2.6e-06 | 3.0e-06 |
| $k$=3.05e7 $q$=3.14e5 $z$=2.09 | -3.353665e-28 | -3.353881e-28 | -3.353763e-28 | 6.4e-05 | 3.5e-05 |
| $k$=3.09e6 $q$=1e5 $z$=6.32 | 9.373868e-25 | 9.373924e-25 | 9.373818e-25 | 6.0e-06 | 1.1e-05 |
| $k$=9.7e6 $q$=9.7e6 $z$=0.30 | 4.671954e-27 | 4.671963e-27 | 4.671952e-27 | 1.9e-06 | 2.4e-06 |
| $k$=3.05e7 $q$=3e8 $z$=0.1 | 1.193076e-29 | 1.193076e-29 | 1.193076e-29 | 0 | 0 |
| $k$=3.05e7 $q$=3.05e7 $z$=0.1 | 3.030463e-28 | 3.030447e-28 | 3.030451e-28 | 5.6e-06 | 1.6e-06 |

Totals: **11.6 s / 1628.9 s / 20,366.7 s** (×140, then ×12.5). Six of seven hit depth 20 even at
`rtol = 1e-5`, and all seven are stored unconverged at every rung. **The two difference columns are
the same size** — a decade of `rtol` does not shrink the change, which is a representation floor at
~`1e-05` and not a quadrature error.

---

## 4. Phase 3a — the seam band, $z_{\rm response} > z_{\rm min}$

Five `mixed`-policy and four `numeric`-policy items, severe geometry. This is the only band in
which a numeric branch of $G_k$ exists at all (see §6).

Worst over the nine: **1e-6 vs 1e-7 → 9.97e-07**, **1e-7 vs 1e-8 → 8.26e-07**.

| `rtol` | total time, 9 cases | max regions | at depth 20 |
|---|---|---|---|
| 1e-6 | 7.8 s | 112 | 0/9 |
| 1e-7 | 7.7 s | 153 | 1/9 |
| 1e-8 | 23.0 s | 205 | 5/9 |

**No c5 here**, and the pathology is absent: 205 regions at the tightest rung against 1.53e6 in the
low band. Tolerance in this band is free — take `1e-8`.

---

## 5. Phase 3b — is `rtol = 1e-7` converged at low $z$?

Four rungs on the three cheapest phase-2 cases plus the free control. 9682 s.

| case | 1e-5 | 1e-6 | 1e-7 | 1e-8 | regions at 1e-8 |
|---|---|---|---|---|---|
| $k$=3.05e7 $q$=3e8 $z$=0.1 | 1.1930758691e-29 | *(identical)* | *(identical)* | *(identical)* | 6 |
| $k$=3.05e7 $q$=3.05e7 $z$=0.1 | 3.0304634170e-28 | 3.0304465343e-28 | **3.0304512700e-28** | **3.0304512700e-28** | 50,043 |
| $k$=9.70e6 $q$=9.70e6 $z$=0.30 | 4.6719537771e-27 | 4.6719627085e-27 | 4.6719515930e-27 | 4.6719383359e-27 | 249,010 |
| $k$=3.09e6 $q$=1e5 $z$=6.32 | 9.3738675972e-25 | 9.3739236570e-25 | 9.3738178384e-25 | 9.3738219940e-25 | 1,238,448 |

Worst relative difference against `1e-8`: **1e-5 → 4.86e-06, 1e-6 → 1.08e-05, 1e-7 → 2.84e-06**.
Times: **5.7 s / 330.3 s / 2597.9 s / 11,463.1 s**.

**The plateau is real and c5 does not recur.** On case 2 the `1e-7` and `1e-8` answers are
*bit-identical* — same value, same 50,043 regions — and on the others the `1e-7`→`1e-8` step is
**smaller** than the `1e-6`→`1e-8` step, the opposite of the c5 signature. Case 4 spent 1,238,448
regions and 9648 s at `1e-8` to move the answer by $10^{-5}$.

**There is no oracle that could settle this differently.** `ComputeTargets/tests/domenech.py` and
`kohri_terada.py` are both constant-$w$; production is LambdaCDM with the Saikawa–Shirai QCD
equation of state. Self-convergence is the only instrument available, and one rung tighter is the
only way to use it.

---

## 6. Where the hand-over lives — a structural correction

Recorded because this document's first reading of it, in conversation on 2026-09-23, was **wrong**
in a way a later reader could repeat.

There are two distinct seams, and they have opposite $z_{\rm response}$ distributions.

**(a) The producer hand-over**, keyed on $(k, z_{\rm source})$. `GkNumericIntegration` solves from
the source initial condition; `GkWKBIntegration` picks up at `z_init` with `G_init`/`Gprime_init`
handed over from it, at `init_efolds_subh` of **3.0 to 28.0 e-folds inside** the horizon (median
10.8). Every `GkNumericIntegration` object for a given $k$ terminates at the **same** $z_{\rm min}$,
6.2–6.4 e-folds inside entry:

| k | $z_{\rm exit}$ | numeric `z_min` | $z_{\rm exit}/z_{\rm min}$ | source samples with a hand-over | pure LG |
|---|---|---|---|---|---|
| 1e5 | 4.635e10 | 7.627e7 | 607.7 | 747 | 993 |
| 3.09e6 | 1.433e12 | 2.772e9 | 516.9 | 594 | 1146 |
| 9.70e6 | 4.497e12 | 8.374e9 | 537.1 | 540 | 1200 |
| 3e8 | 1.390e14 | 2.531e11 | 549.3 | 391 | 1349 |

**(b) The consumer crossover**, keyed on $(k, z_{\rm response})$ — the `crossover_z` that
`GkSourcePolicyData` picks when assembling $G_k$ along the $z_{\rm source}$ axis. It exists only in
`mixed`-type rows: **72 of 1160**, all above $z_{\rm response} = 10^5$. `type = "numeric"` means
*no usable WKB region*, so those rows carry no seam either.

**The correction.** Below $z_{\rm response} = z_{\rm min}$ there is no numeric branch **because the
numeric integration was stopped there** — that absence *is* the hand-over, not evidence against one.
So those rows are not hand-over-free; every $G_k$ value in them that came from a source above
$z_{\rm min}$ is LG continuation carrying `G_init`/`Gprime_init` across the seam. What is confined
to the band above $z_{\rm min}$ is (b) alone, which is one defect — D1's numeric spline, taken in
$\log(1+z_{\rm source})$ at [`GkSourcePolicyData.py:668`](../../ComputeTargets/GkSourcePolicyData.py) —
and not the seam as a whole.

The accurate statement, due to the user: within any row roughly **30 %** of the 1740 source samples
carry a hand-over and **70 %** are pure LG with boundary conditions applied directly in that
representation. Since $z_{\rm source,max} = 1.64\times10^{16}$ exceeds every $z_{\rm exit}$, that
30 % is inside the integration range of every row, so there are always hand-over-derived
contributions at $z_{\rm response} \lesssim z_{\rm exit}$ and none above it.

---

## 7. What this costs to act on

Full 9280-item rebuild at `(1e-32, 1e-7)`, from two independent routes that agree: scaling the
store's own 1119 CPU-hr at `1e-8` by the measured ×4.4 low-band ratio gives ~250 CPU-hr; counting
~18 severe triples × 16 low-$z$ redshifts × the 2910 s phase-2 mean gives ~233 CPU-hr for the low
band plus ~17 elsewhere. **~250 CPU-hr ≈ 36 wall-hours** at ~7 effective CPUs.

It stages itself, because the work list runs high $z$ first (observed: the A3 baseline's only gaps
after 10 hours were at $z \le 174$) and the pipeline computes only what is missing:

| stage | items | est. CPU-hr | est. wall |
|---|---|---|---|
| seam band, $z > 10^5$ | 6080 | ~1 | minutes |
| mid band, $9 - 10^5$ | 2112 | ~16 | ~2.5 h |
| low band, $z < 9$ | 1088 | ~233 | ~33 h |

One short session therefore yields 8192 of 9280 items, including the whole seam band and the band
c5 sits in. The 1088-item tail is both the expensive part and the representation-limited part, so a
later D1/D2 fix would force it to be recomputed anyway.
