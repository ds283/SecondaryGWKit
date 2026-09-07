# Implementation state — source remediation campaign

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/spec-code-audit-2026-09.md`](../../docs/spec-code-audit-2026-09.md)
**Baseline commit:** `e9a43a2` (`main`, clean)
**Last updated:** 2026-09-07 — plan written, nothing started.

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA, model and log link, update the item-level table,
> and add or clear entries in §3 (Active issues). Do not edit rows other than your own except to
> close an issue you resolved.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — independent correctness and hygiene fixes

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [GenericEOS sound speed](01-genericeos-sound-speed.md) | A1 | Opus | ⬜ | | |
| 02 | [WKB value hygiene](02-wkb-value-hygiene.md) | B1, B2, B3, B4, A6, B9, B10 | Sonnet | ⬜ | | |
| 03 | [Background derivative ends](03-background-derivative-ends.md) | A7 | Opus | ⬜ | | |

### Workstream D — scheduling

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 04 | [Triangle filter](04-triangle-filter.md) | A5 | Sonnet | ⬜ | | |

### Workstream B — transfer-function LG representation and the source grid

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 05 | [`TkSourceFunctions`](05-tk-source-functions.md) | A2 (1/3) | Opus | ⬜ | | |
| 06 | [`QuadSource` regions](06-quadsource-regions.md) | A3, A2 (2/3) | Opus | ⬜ | | |

### Workstream C — the source time integral

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 07 | [Phase-group algebra](07-phase-group-algebra.md) | A4 (1/3) | Fable | ⬜ | | |
| 08 | [`QuadSourceIntegral` phase-group integration](08-qsi-phase-group-integration.md) | A4 (2/3), A2 (3/3) | Fable | ⬜ | | |
| 09 | [Errors, schema, tolerances](09-qsi-errors-schema-tolerances.md) | B5, B6, B7, B8, B11 | Opus | ⬜ | | |
| 10 | [`main.py` plumbing](10-qsi-main-plumbing.md) | A4 (3/3) | Opus | ⬜ | | |

### Workstream E — close-out

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Spec annotations](11-spec-annotations.md) | audit §6 | Sonnet | ⬜ | | |
| 12 | [Verification](12-verification.md) | audit §4; campaign | Opus | ⬜ | | |

**Progress:** 0 / 12 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| A1 | **DEFECT, physics** | `LambdaCDM_GenericEOS.wPerturbations` divides by the total density incl. $\rho_\Lambda$ | 01 | ⬜ |
| A2 | **DEFECT, representation** | `QuadSource` splines the oscillating source; unusable beyond ~95 cycles | 05, 06, 08 | ⬜ |
| A3 | **DEFECT, regression** | `compute_quad_source` walks the full grid against a both-ends-truncated $T_k$ grid → `IndexError` | 06 | ⬜ |
| A4 | **DEFECT, known** | Levin call receives only $\theta_G$; no $T_q,T_r$ input to the Levin decision | 07, 08, 10 | ⬜ |
| A5 | **DEFECT, known** | 92 % of scheduled $(k,q,r)$ triples are not triangles | 04 | ⬜ |
| A6 | **DEFECT, policy** | `"WKB_minimal"` tests `numeric_clearance` | 02 | ⬜ |
| A7 | **DEFECT, accuracy** | `_build_derivative` end bias (ε″ 30 % at the $z=0.1$ end for GenericEOS models) | 03 | ⬜ |
| B1 | diagnostic | `TkWKBValue.analytic_*_w` return `_rad` | 02 | ⬜ |
| B2 | diagnostic | `GkWKBValue.analytic_*_w` return `_rad` | 02 | ⬜ |
| B3 | dead code | pre-flight WKB warnings omit `fabs` | 02 | ⬜ |
| B4 | wrong exception | `_init_efolds_suph` typo (Tk and Gk WKB) | 02 | ⬜ |
| B5 | tolerance | `Y3` Levin call uses module constants, not passed tolerances | 09 | ⬜ |
| B6 | tolerance | `analytic_integral` ignores its `atol`/`rtol` | 09 | ⬜ |
| B7 | provenance | no `b` column on `QuadSourceIntegral` | 09 | ⬜ |
| B8 | error bound | `total` has no error bound | 09 | ⬜ |
| B9 | consistency | `Levin_z` θ-spline chunking differs from the evaluated spline | 02 (evaluate) | ⬜ |
| B10 | cosmetic | `QuadSource` spline wrapper labelled `"T_k"` | 02 | ⬜ |
| B11 | robustness | region-nonempty guards use a ratio in $z$ not $1+z$ | 09 | ⬜ |
| §4.1 | UNVERIFIED | continuity of $G$ at `crossover_z` | 12 | ⬜ |
| §4.2 | UNVERIFIED | reachability of A6 | 12 | ⬜ |
| §4.3 | UNVERIFIED | whether `has_WKB_violation` modes should be rejected | 12 (measure only) | ⬜ |
| §4.4 | UNVERIFIED | A7 end bias on a real GenericEOS run | 12 | ⬜ |
| §6 | spec edits | close spec 02 Q9, record spec 04 Q7 convention, annotate spec 01 R16/Q4 | 11 | ⬜ |

**Out of scope (do not schedule):** Tier 2 LG output from `QuadSourceIntegral`; `OneLoopIntegral`;
`csSquared(z)`; `z_response` averaging. See `README.md` §1.1.

---

## 3. Active and unresolved issues

*(none yet)*

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation that a later prompt has
> to work around, a measured cost that changes a later prompt's decision. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

*(none yet)*

---

## 5. Standing notes for implementers

1. **Datastores built before this campaign are stale after prompt 06** (`QuadSource` rows have
   fewer redshifts per pair) **and unreadable after prompt 09** (`QuadSourceIntegral` schema
   change). The verification prompt rebuilds from scratch; do not try to migrate.
2. **`b_value = 0.0` is hardwired in `main.py:419`.** Nothing in this campaign changes that, but
   prompt 09 makes `b` persisted so that a future non-zero run is distinguishable.
3. **The phase sign convention is $d\theta/dz = +\omega_{\rm eff}$ integrated towards smaller
   $z$, so every stored phase is negative and decreasing** (audit §3.2). Composed phases
   $\theta_G\pm\theta_q\pm\theta_r$ inherit this. Nothing depends on the sign, but tests that
   assert monotonicity must assert the right direction.
