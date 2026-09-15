# Reconciliation — the tolerance campaign's plan against the tree it will run on

**Planned:** 2026-09-12 at `622b84b`, on `gktk-remedial` `891d84a`.
**Rebased:** 2026-09-16 at `acd5b8e` (`tolerance-convergence`, cut from `main`, clean).
**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)

This document exists for the same reason
[`prompts/GkTk-remedial/RECONCILIATION.md`](../GkTk-remedial/RECONCILIATION.md) does: the plan was
written against a tree that has since moved, and a plan whose factual claims are not separately
audited will be executed as though they were still true. **Every claim of the 2026-09-12 README is
scored here against the code at `acd5b8e`**, and the README has been rewritten from this document
rather than the other way round.

The verdicts are `HOLDS`, `NARROWED`, `FALSE` and `SUPERSEDED`. A `FALSE` row is a statement that
would have sent a prompt in the wrong direction; a `SUPERSEDED` one was true of a tree that no
longer exists.

---

## 1. What landed between the plan and the rebase

Thirty-six commits, `622b84b..acd5b8e`. Three campaigns:

| Campaign | Prompts | What it moved that this campaign cares about |
|---|---|---|
| `GkTk-remedial` (closed 20 / 20, merged `e8f746d`) | 18, 19, 20, 13 | The numeric ODE splits at the cosmology's declared non-smoothness; *which* kind is a per-sector choice; the choice is a datastore key column |
| `phase-representation` (closed 1 / 2, `6177340`) | 01 | `WKB_mod_2pi`'s cycle count; no tolerance moved |
| `qcd-background-audit` (closed 16 / 16, merged `acd5b8e`) | 04–07, 08, 11, 13, 15 | The QCD background's `T(z)` representation, its declared break-point set, the background's own derivative splines, and **the production source grid**, twice |

The last of these is the one the plan did not anticipate at all. The plan's measurements are all
scored on "the production grid"; the production grid has been rebuilt twice since, and the test
tree's reproduction of it has not followed.

---

## 2. Claims that are now false

### 2.1 "Four of the five targets share two tolerance constants" — **FALSE**

README §0.1 and §0.2 count five compute targets and say the shared pair keys four object types.
Measured, by reading every `pool.object_get` in `main.py` that passes a tolerance:

| Object type | `atol` | `rtol` | Does the tolerance reach a solver? |
|---|---|---|---|
| `wavenumber_exit_time` (`main.py:846`) | `DEFAULT_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | **Yes** — `root_scalar(q, bracket=…, xtol=atol, rtol=rtol)` in $\log(1+z)$, `CosmologyConcepts/wavenumber.py:979-984` |
| `BackgroundModel` (`main.py:1024`) | `DEFAULT_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | **No** — "accepted for signature compatibility … the table has no tolerances" (`ComputeTargets/BackgroundModel.py:409`). The knobs are `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` |
| `TkNumericIntegration` (`main.py:1181`) | `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | Yes — DOP853 |
| `TkWKBIntegration` (`main.py:1396`) | `DEFAULT_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | **No** — vestigial key columns (`TkWKBIntegration.py:38-39`) |
| `GkNumericIntegration` (`main.py:1791`) | `DEFAULT_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | Yes — DOP853 |
| `GkWKBIntegration` (`main.py:2052`, `:2102`) | `DEFAULT_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | **No** — vestigial (`GkWKBIntegration.py:334-336`) |
| `GkSource` (`main.py:2365`) | `DEFAULT_ABS_TOLERANCE` | `DEFAULT_REL_TOLERANCE` | **No** — stored on the object (`GkSource.py:418-437`) and never read by the assembly |
| `QuadSourceIntegral` (`main.py:3290`) | `DEFAULT_QUADRATURE_ATOL` | `DEFAULT_QUADRATURE_RTOL` | Yes — decoupled already, and not this campaign's |

So the shared pair keys **six** object types and `DEFAULT_REL_TOLERANCE` keys a seventh, of which
**three are live** (`wavenumber_exit_time`, `GkNumericIntegration`, `TkNumericIntegration`) and
**four are vestigial** (`BackgroundModel`, both WKB sectors, `GkSource`). The plan's "five compute
targets" omits `wavenumber_exit_time` and `BackgroundModel` entirely — the first a live root solve
that fixes where every grid begins, the second the largest of the order-governed targets.

**Consequence for the plan:** the inventory is not a thing a prompt can assume; prompt 02 of the
rebased plan measures it before anything else is swept, and prompt 05's `ast` guard must widen its
predicate, which today matches only class names ending in `"Integration"`
(`ComputeTargets/tests/test_main_plumbing.py:544`) and so sees four of the eight.

### 2.2 "Two of the five targets have no tolerance to converge" — **NARROWED to four of eight**

README §2 (a) is right about the two WKB sectors and right about why. It misses `BackgroundModel`,
whose `atol`/`rtol` became vestigial at `GkTk-remedial` prompts 03/04 when $\tau$, $c_s\tau$ and $F$
stopped being ODEs and became Gauss–Legendre cumulative tables, and `GkSource`, which stores the
pair and never reads it. **This is the campaign's stated target, not a detail:** the user's framing
is that a Liouville–Green-type representation is calibrated by an integer order, and
`BackgroundModel` is the target that carries three such orders.

### 2.3 "`[17-qcd-reference-not-converged]` is a hard precondition; this campaign cannot start" — **SUPERSEDED**

README §0.3 and the board's header block the campaign on `GkTk-remedial` prompt 18. That issue is
**closed** (`docs/OPEN_ISSUES.md` §1.5, 2026-09-13): prompts 18 and 19 took the worst QCD
reference-convergence drift to 8.72e-09 ($T_k$) and 8.41e-09 ($G_k$) against the 3.4e-08 criterion,
zero offenders at all 50 production wavenumbers on all three models. `qcd-background-audit` prompt
08 then re-took the same measurement on the corrected background and improved it again — worst
7.08e-09 ($T_k$, `BREAK_POINT_ALL`) and 3.67e-09 ($G_k$)
(`docs/qcd-background-audit/PER-SECTOR-POLICY.md` §2, §4).

**The campaign is unblocked, and the branch is already cut.**

### 2.4 The cost figure the `rtol` decision rests on — **FALSE by a factor of 3.5**

`docs/OPEN_ISSUES.md` §1.5 records, as the cost this campaign inherits, "a QCD $T_k$ numeric object
is ~31.5k right-hand-side evaluations rather than ~9.8k (+220 %, ~49 s for the 50-object sector per
model)". That was `GkTk-remedial` prompt 19's measurement, taken when `BREAK_POINT_ALL` meant 404
spline knots plus three crossings. `qcd-background-audit` prompt 07 took the knots out of the
declaration, and prompt 08 re-measured (`PER-SECTOR-POLICY.md` §5):

| sector | model | prompt 19, `all` | now, `all` | now, `discontinuity` |
|---|---|---|---|---|
| $T_k$ | QCDModel | 31,521 | **8,986** | 8,897 |
| $G_k$ | QCDModel | 13,320 | **13,419** | 13,343 |

The wider policy now costs **+0.99 %** in the $T_k$ sector and +0.57 % in $G_k$, against the +220 %
and +155 % the index still quotes.

### 2.5 "A recurring compute cost on a sector with ~65,000 objects per model" — **FALSE, wrong sector**

README §7 D1 attaches that object count to the $T_k$ `rtol` finding. Measured from `main.py`'s own
loop structure: `TkNumericIntegration` is built once per wavenumber (`main.py:1180`, inside the
$k$ loop alone) — **50 objects per model** — while `GkNumericIntegration` is built once per
$(k, z_{\rm source})$ (`main.py:1770-1791`, two nested loops), the same shape as
`GkWKBIntegration`, whose production count `GkTk-remedial` README §6 records as ~65,000 per model.

So one decade of `rtol` at +23–25 % evaluations is **negligible in the sector where it was
measured** and **the whole compute decision in the sector where it was not**. D1's weight inverts:
the number that matters is $G_k$'s, and $G_k$ has never been swept.

### 2.6 "`rtol` is the lever in both sectors — measured twice, independently" — **NARROWED to once and a half**

README §2 (c) cites review §10.1 on $G_k$ and prompt 17 on $T_k$. Prompt 17's measurement does hold
`atol` fixed at 1e-13 and move `rtol` alone, so it separates the axes. **Review §10.1 does not**:
its table (`docs/gk-wkb-review-fable-2026-09-09.md:469-474`) moves along a diagonal, `(1e-10, 1e-8)`
against `(1e-13, 1e-11)`, so "the error is set by `rtol`" there is an interpretation of a
two-variable step and not a decoupled measurement. The prior stands as one clean measurement in the
sector with 50 objects and one diagonal in the sector with 65,000.

The same paragraph carries the argument against acting on it, which the plan does not quote: review
§10.1 puts the consumer's cubic spline of the numeric $G$ at "1e-5 to 1e-4 near the hand-over … the
larger error by two orders". That is the floor $G_k$'s `rtol` is competing against, and prompt 03
of the rebased plan has to measure it before recommending anything.

### 2.7 "The production grid" — **SUPERSEDED, and the test tree does not know**

Every figure in the plan's acceptance table is scored on a source grid built by
`ComputeTargets/tests/wkb_reference.production_source_grid`, which is a bare `np.logspace`
reproducing `populate_z_sample` and citing `main.py:410-419`. Production has moved twice since:

| generation | what builds it | QCD samples | LambdaCDM samples |
|---|---|---|---|
| version 0 | `np.logspace` — what `wkb_reference.py:129-160` still reproduces | 1,732 | 1,732 |
| version 1 | `qcd-background-audit` prompt 11: straddling pairs at the declared crossings, ±5 intervals refined by 2 | 1,773 | 1,732 |
| **version 2** (production) | prompt 15: every base interval the criterion $h^4\lvert\varphi''''\rvert/384\le\varepsilon$ finds too wide subdivided, under `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` | **1,996** | **1,778** |

`SOURCE_GRID_CONSTRUCTION_VERSION = 2` (`CosmologyConcepts/wavenumber.py:61`). There are now
**three disagreeing reproductions of "the production grid" in the tree**: `wkb_reference.py`'s
(version 0), `ComputeTargets/tests/test_background_segmentation.py:90`'s (version 1 — it passes
`break_z` and `feature_z` but no `spacing` profile), and `main.py:911-930`'s (version 2).

`docs/gktk-remedial/tk_numeric_atol_sweep.py:215` imports the first of them, so **the script the
plan folds into its harness measures a grid production stopped using two campaigns ago.**

**Consequence for the plan:** prompt 01's stated acceptance test — "it must reproduce GkTk-remedial
prompt 17's published drift figures for $T_k$ exactly" — pins the new harness to a superseded grid.
It becomes a two-part test in the rebased plan: reproduce prompt 17 *on prompt 17's grid*, as a
construction check, and report the same statistic on the production grid beside it.

### 2.8 "The QCD $H(z)$ discontinuity floor" — **SUPERSEDED; that floor is gone**

README §2 (e) lists, among the floors, "on `QCDModel`, whatever prompt 18 leaves of the $H(z)$
discontinuity floor", and prompt 17's hand-off puts it at ~6e-06 of the envelope at four
wavenumbers. Prompt 18 split the ODE at the jumps and `qcd-background-audit` prompts 04–06 removed
the representation error underneath it: the QCD background's $\int\mathrm{d}z/H$ is now
**bit-identical** to an independently root-solved exact background where it carried 3.461e-08, and
the equivalent phase error is 0.000e+00 rad at $k = 10^5$, $10^7$ and $3\times10^8$/Mpc
(`docs/qcd-background-verification.md` §2). There is no discontinuity floor left to measure
against.

### 2.9 Two citations that do not resolve

- README §2 (a) cites "`RECONCILIATION.md` §10" for the vestigial WKB columns. It is
  [`prompts/GkTk-remedial/RECONCILIATION.md`](../GkTk-remedial/RECONCILIATION.md) **§2 item 10**,
  which is what the code itself cites (`GkWKBIntegration.py:336`).
- README §2 (b) and `config/defaults.py`'s own comment disagree with
  `ComputeTargets/QuadSourceIntegral.py:1550`, which still says "the pipeline supplies
  `DEFAULT_QUADRATURE_ATOL = 1e-25`". The constant has been 1e-32 since `source-remediation`
  prompt 12. Recorded, not fixed — it is a `QuadSourceIntegral.py` comment and §0.4 puts that file
  out of bounds.

---

## 3. Claims that hold

| README claim | Verdict | Evidence at `acd5b8e` |
|---|---|---|
| §2 (a) The two WKB sectors' `atol`/`rtol` have no referent and are kept only as key columns | **HOLDS** | `GkWKBIntegration.py:334-336`, `TkWKBIntegration.py:38-39` |
| §2 (b) `QuadSourceIntegral` is decoupled and its `atol` distributed per sub-interval by log-width | **HOLDS** | `QuadSourceIntegral.py:815-819`; `DEFAULT_QUADRATURE_ATOL = 1e-32`, `_RTOL = 1e-8` |
| §2 (d) `atol` is not uniform across sectors because the magnitudes are not | **HOLDS** | `config/defaults.py:7-33`, unchanged |
| §2 (e) The $T=1,T'=0$ initial-condition floor, 2.52e-06 of the envelope, $k$-independent | **HOLDS** | The grid still starts five e-folds outside the horizon at every $k$ (`main.py:924`, `outside_horizon_efolds=5`) |
| §2 (e) $T_k$ WKB LG truncation, 3.8e-05 of envelope from $x_i = 24$ | **HOLDS** | `GkTk-remedial` prompt 07; no LG order changed since |
| §2 (f) Decoupling is a datastore change and the plumbing is the risky half | **HOLDS, and widened** | `GkTk-remedial` prompt 20 added a second key column (`break_point_kind`) to both numeric factories, so a numeric row's key is now `(…, atol_serial, rtol_serial, break_point_kind)` |
| §2 (g) The analytic anchors take a general constant $w$ | **HOLDS** | `ComputeTargets/analytic_Gk.py`, `analytic_Tk.py` unchanged |
| §2 (h) Counts, not wall time | **HOLDS** | Restated by `PER-SECTOR-POLICY.md` §5 |
| §7 D1 `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` is settled and not reopened | **HOLDS** | The user, 2026-09-12; `config/defaults.py:33` |
| §7 D2 The datastore grows and that is the intended outcome | **HOLDS** | And `qcd-background-audit` `21d80b2` restates it campaign-independently: "regeneration cost is not a constraint" |
| §0.4 `QuadSourceIntegral`, `QuadSource`, `phase_groups`, `AdaptiveLevin/` are out of bounds | **HOLDS** | `docs/OPEN_ISSUES.md` §1.2, §1.3 |

---

## 4. What the rebase changes in the plan

1. **Six prompts, not five.** A new prompt 02 measures the tolerance inventory §2.1 shows the plan
   got wrong, and the remaining prompts shift by one.
2. **Prompt 01 owns the production grid.** It builds one reproduction, at version 2, and retires
   the two stale ones rather than adding a fourth.
3. **The order-governed targets get their own prompt** (04), covering `BackgroundModel`'s three
   orders as well as the two WKB sectors' — the user's "integer order rather than an
   `atol`/`rtol`" — and it inherits the stale evidence §5 below describes.
4. **The acceptance table's "now" column is re-scored or marked unmeasured.** Three of its five
   rows were taken on a superseded grid or a superseded background.
5. **D1's decision weight moves from $T_k$ to $G_k$** (§2.5), and D3 widens from "what to do with
   two vestigial column pairs" to "what to do with four, three of which have an integer order that
   should be in the key instead" — which closes `[20-wkb-gauss-orders-not-in-lookup-key]`.

---

## 5. The stale evidence prompt 04 inherits

`ComputeTargets/tests/wkb_reference_data.json`'s top-level `convergence` block is where
$N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$ are recorded, and it is the evidence every
`*_GAUSS_ORDER` comment in the tree cites. Read at `acd5b8e`:

```
convergence.generated  = 2026-09-10
convergence.campaign   = prompts/GkTk-remedial (prompt 02)
convergence.generator  = docs/gktk-remedial/residual_convergence.py
decision.recommended_scheme = "branch+knots"
```

It therefore predates, and was measured against:

- the `T(z)` representation `qcd-background-audit` prompts 04, 05 and 06 replaced (node solve
  `rtol` 1e-4 → 1e-14; $T$-against-$u$ → a segmented entropy factor; max error
  7.236e-04 → 6.807e-11);
- an `integration_break_points` that returned the interpolant's ~404 knots, which prompt 07
  removed — **so the winning scheme, `"branch+knots"`, names a knot set that no longer exists**;
- the background derivative splines prompt 13 segmented.

`[01-convergence-block-has-a-separate-generator]` on the `qcd-background-audit` board records this
and is **unassigned**: prompts 08 and 09 of that campaign each named the next prompt as the place
to close it and each found the JSON and `ComputeTargets/tests/test_background_tau.py` outside their
"files you may touch". Its visible residue is `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` in that
test module, loosened from 1.4e-05 for no reason but the block's age.

**The rebase assigns it to prompt 04 of this campaign**, which is the first prompt anywhere whose
charter is the Gauss orders themselves and which therefore cannot avoid re-running that script.
Note what that costs: prompt 04 stops being a read-only prompt, because closing it means writing
the JSON and editing a test module. That is README §7 **D5**, and it is the user's.

---

## 6. Issues this reconciliation opens or moves

| Issue | Action | Where |
|---|---|---|
| `[00-three-production-grid-reproductions]` | **Opened** by this rebase, 2026-09-16 (§2.7) | board §3, assigned to prompt 01 |
| `[00-gk-numeric-never-swept-and-carries-the-cost]` | **Opened** by this rebase, 2026-09-16 (§2.5, §2.6) | board §3, assigned to prompt 03 |
| `[01-convergence-block-has-a-separate-generator]` | **Assigned** to prompt 04 of this campaign (§5) | stays on the `qcd-background-audit` board, which holds its measurements |
| `[20-wkb-gauss-orders-not-in-lookup-key]` | **Assigned** to prompt 05 of this campaign | stays on the `GkTk-remedial` board |
| `[12-tk-numeric-atol-largest-k-excursion]` | Already assigned here 2026-09-12; its cost figures corrected per §2.4 | stays on the `GkTk-remedial` board |
