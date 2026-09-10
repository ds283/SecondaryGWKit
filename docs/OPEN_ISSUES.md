# Open issues — project-wide index

**Last updated:** 2026-09-10 · **27 open** across five campaigns.

This file exists so that an issue opened by one campaign is not lost when that campaign closes.
It is an **index, not a record**: one line per issue, pointing at the campaign status board that
holds the measurements, the impact statement and the next step. Never put issue content here — if
the two disagree, the board is right.

> **Maintenance rule.** Whenever you add, narrow or close an entry in a campaign board's
> §3 (Active and unresolved issues) or §4 (Resolved issues), update this file **in the same
> commit**. Add the line, move it between sections here, or delete it, and correct the count and
> the date above. See `CLAUDE.md`.

**Boards.** [`source-remediation`](../prompts/source-remediation/IMPLEMENTATION_STATE.md) ·
[`levin-refactor`](../prompts/levin-refactor/IMPLEMENTATION_STATE.md) ·
[`backport-modules`](../prompts/backport-modules/IMPLEMENTATION_STATE.md) ·
[`transfer-remedial`](../prompts/transfer-remedial/IMPLEMENTATION_STATE.md) ·
[`GkTk-remedial`](../prompts/GkTk-remedial/IMPLEMENTATION_STATE.md)

---

## 1. Assigned to a future campaign

Work is identified and owned; the issue is parked deliberately, not forgotten.

### 1.1 The hand-over campaign

The numeric→Liouville–Green seam of $T_k$ and $G_k$. These six are **one place** and must be
attacked together — prompt 12 of `source-remediation` could not separate their contributions by
measurement alone at production $x$. Background reading:
[`docs/lg-phase-and-handover-followup-2026-09.md`](lg-phase-and-handover-followup-2026-09.md).

| Issue | Board | Hook |
|---|---|---|
| `[08-handover-clamp-error]` | source-remediation | The WKB grid starts below `crossover_z`, so the LG accessors are clamped across a gap; a one-step gap moves `total` by 5.1e-03. |
| `[12-handover-clamp-error-in-production]` | source-remediation | Supersedes the above for magnitude: on real rows the gap is universal and costs 6.6e-02–4.6e-01 against 6.2e-05 unclamped. **The campaign's main outstanding accuracy decision.** |
| `[05-numeric-region-is-now-the-accuracy-floor]` | source-remediation | A cubic spline loses 1–2 orders in its last two intervals, and the numeric grid *ends* at the hand-over. |
| `[06-source-spline-residual-vs-handover]` | source-remediation | $f$ oscillates at twice the transfer-function phase, so its spline is the worst of the three; 1.4e-04 of envelope on real rows, but $O(1)$ if the hand-over is allowed to fall to the bottom of `main.py`'s search window. |
| `[07-lg-derivative-truncation-at-handover]` | source-remediation | The irreducible one: `omega`/`dlnM_dz` are LG quantities, off by $O(x^{-4})$. Grid-independent — only a deeper hand-over helps. |
| `[00-tk-lg-truncation-floor]` | GkTk-remedial | The $T_k$ LG representation is not exact even in radiation: ~$1.4\times10^{-4}$ of the envelope at the production hand-over $x_T\approx15.5$, scaling as $x_i^{-3}$. A later hand-over ($x_T=50$ gives $4\times10^{-6}$), a higher-order LG frequency, or the Bessel exact form are the levers. |

### 1.2 The `AdaptiveLevin` Clenshaw–Curtis fallback campaign

| Issue | Board | Hook |
|---|---|---|
| `[10-levin-wholesale-cc-fallback]` | source-remediation | 2.65–3.56× wall clock for 1.00–1.44× integrand evaluations, i.e. per-region overhead, not wasted work. Route wholesale to Clenshaw–Curtis at the outset instead of bisecting into it. **Do not** re-add a threshold in `QuadSourcePolicy` — that is defect A4 in weaker form. |
| `[03-fallback-cost-on-difference-groups]` | levin-refactor | The total-variation gate now tests the Levin branch's eagerly computed comparison children too; structurally required, but raises evaluation counts 2.2–3.4× on three-Bessel groups. |
| `[04-roundoff-floor-can-be-infinite]` | levin-refactor | `phase_err`/`abserr_roundoff`/`total_err` can be `+inf` at an interior stationary point inside a strongly oscillatory Levin region. |
| `[04-theta-abserr-cc-branch-proxy]` | levin-refactor | The endpoint term is exact on a Levin region but a 50/50 split of a lumped proxy on a Clenshaw–Curtis one, which has no Levin antiderivative to weight by. |
| `[05-zero-width-span-raises]` | levin-refactor | `adaptive_levin_sincos((5.0, 5.0), …)` raises `ValueError` from `build_Levin_data()`; the early return the README describes does not exist. |

### 1.3 The $T_k$ / $G_k$ numerical-precision campaign

Now planned as [`prompts/GkTk-remedial/`](../prompts/GkTk-remedial/README.md) (2026-09-10; 13
prompts, none executed). `[07-phase-spline-chunking-precision]` was closed WONTFIX on the
expectation that this campaign **removes** the chunked splines — its prompt 08 does; if that plan
changes, reopen it.

| Issue | Board | Hook |
|---|---|---|
| `[12-phase-spline-error-grows-with-x]` | source-remediation | Stored-phase re-spline error $\simeq h^4x/384$, growing **linearly in $x$**; ~1 % of envelope extrapolated to production. **Reassigned here from §1.1** (2026-09-10): the campaign's prompts 09–10 evaluate the leading term from a table and spline only the residual. |
| `[00-tau-storage-decision]` | GkTk-remedial | Persist the low-order limb of the τ/τ_s node tables as new `BackgroundModelValue` columns — **confirmed by the user 2026-09-10**; closes when prompt 03 lands. Regeneration attached. |
| `[00-unresolved-osc-print-policy]` | GkTk-remedial | With its units fixed and evaluated on the caller's actual grid, `has_unresolved_osc` will fire on essentially every $G_k$ numeric object. Prompt 11 measures the rate; the user chooses the print policy. |
| `[00-consumer-anchoring-floor]` | GkTk-remedial | `PrimitivePhase` reduces $k\Delta\tau$ against a global anchor, so `theta_mod_2pi` carries the $\varepsilon k\tau$ floor ($9\times10^{-4}$ rad at $k=3\times10^8$). Per-region anchoring is the follow-up. |
| `[00-transfer-remedial-test-file-overlap]` | GkTk-remedial | Prompt 10 edits stand-in fixtures in `test_tk_source_functions.py`/`test_phase_groups.py`; `transfer-remedial` 08 edits tolerances in the same files. **Decided 2026-09-10:** Workstreams A–C, E run in parallel; D waits for the `transfer-remedial` merge. |

---

## 2. Error-bound completeness

A family: several layers compute an error estimate that is correct for what it claims and does not
bound the true error. Closing any of them properly needs the representation error of §1.1 first.

| Issue | Board | Hook |
|---|---|---|
| `[09-abserr-is-a-quadrature-bound]` | source-remediation | `total_abserr` is the linear sum of quadrature estimates and nothing else; the true residual is up to 4.4e4× larger. `total_converged = False` is not a failure. |
| `[12-atol-too-loose-for-the-source-integral]` | source-remediation | 58 % of work items have a raw integral below `DEFAULT_QUADRATURE_ATOL = 1e-25`, so their tolerance is met before any work is done. Needs a production decision: scale `atol` with the integrand, go `rtol`-only, or lower the default for this stage. |
| `[09-abserr-does-not-bound-phase-spline-floor]` | levin-refactor | `quad_JJJ`/`quad_YJJ`'s `abserr` misses the true error against the analytic oracle by up to 11.5× on 5 of 7 three-Bessel closed forms. |
| `[09-quadsource-total-error-incomplete]` | levin-refactor | Partly superseded: `source-remediation` prompt 09 added `total_abserr`. Re-read against the current tree before acting. |

---

## 3. Inert — recorded so a later reader does not misread a residual

No action defined. These are floors on what a test may *assert*, not on what the code computes.

| Issue | Board | Hook |
|---|---|---|
| `[01-genericeos-tz-spline-floor]` | source-remediation | Whether the `T(z)` spline grid is adequately defined. A hot-fix's fixed 500 points give 1.3e-9 at `max_z=1e4`, 6.4e-7 at the default 1e20; it is why prompt 01's test asserts 1e-8, not 1e-10. Two sign bugs in the grid's *range* were fixed 2026-09-10 and are not part of this. |
| `[03-derivative-pad-clamp-on-coarse-grids]` | source-remediation | The background derivative-fit padding is clamped near $z=0$; harmless at the shipped 100 samples/decade, binds at 50. A trap only if `source_samples_log10z` is lowered. |
| `[00-tk-superhorizon-ic-series]` | GkTk-remedial | Once the $T_k$ numeric `atol` is fixed (prompt 12), the floor is the super-horizon initial condition $T=1,T'=0$ at $2.5\times10^{-6}$; removable with the series $T\approx1-x^2/10$, a spec-level decision. |

---

## 4. Verification debt

Something was asserted statically or on a stand-in, and a live exercise is still owed.

| Issue | Board | Hook |
|---|---|---|
| `[04-read-table-service]` | backport-modules | Audit §8 item 6; item 5 was closed live by prompt 10. |
| `[05-persist-handler-split]` | backport-modules | Audit §8 item 7: a real driver run exercising the `store_handler`/`persist_handler` split end to end. |
| `[00-plan-vs-tree-corrections]` | transfer-remedial | Four `DRAFT-PLAN.md` claims that do not survive reconciliation against the tree; the prompts are built on the corrected versions. |
| `[00-qsi-three-bessel-levin-excluded]` | transfer-remedial | `_three_bessel_Levin`'s eight `adaptive_levin_sincos` calls supply no `theta_deriv`, so Levin differentiates the raw phase spectrally there. |

---

## 5. Standing caveat that is not an issue

**The verification runs never reached production $x$.** `source-remediation`'s run A used
`zend = 1e7` to stay inside radiation domination where `analytic_rad` is a valid oracle, so its
largest accumulated phase was $x = 4.63\times10^5$ against $x\sim1.4\times10^7$ at the production
`zend = 0.1` for the largest $k$. Every "verified live" claim in that campaign carries this
ceiling, and `[12-phase-spline-error-grows-with-x]` is the term it matters most for.
