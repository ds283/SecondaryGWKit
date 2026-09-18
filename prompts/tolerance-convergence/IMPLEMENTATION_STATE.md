# Implementation state — tolerance and convergence campaign

**Campaign:** [`README.md`](README.md) · **Rebase record:** [`RECONCILIATION.md`](RECONCILIATION.md)
· **Logs:** [`logs/`](logs/) · **Orchestrator:** [`orchestrator/`](orchestrator/)
**Planned:** 2026-09-12 at `622b84b` · **Rebased:** 2026-09-16 at `acd5b8e` ·
**Re-anchored:** 2026-09-16 at `bc6dc97` (`RECONCILIATION.md` §7)
**Baseline commit:** `bc6dc97` (`tolerance-convergence`, 25 commits ahead of `main`, clean; suites
re-run for the re-anchor and green — `ComputeTargets` **452**, `CosmologyModels` **39**)
**Superseded baseline:** `acd5b8e`, `ComputeTargets` 447, `CosmologyModels` 30 — the anchor of any
figure in this campaign's documents dated before 2026-09-16 19:32
**Last updated:** 2026-09-18 · **Status: in progress — 6 / 6 measured, plus the insertions 02a, 04b, 05b and 05a; **04b has landed the block prompt 04 stopped short of writing**, **05 has landed the schema change** and **05b has made the recorded order the order that was used**; 05a–06 held.**
**Every user decision needed to start is settled** — **D1 closed 2026-09-17** (all four pairs
accepted), **D3 settled 2026-09-18 and implemented by prompt 05**, D2 settled 2026-09-12, D4 settled by
README §0.4, **D5 settled yes 2026-09-16**.
**Prompts 01, 02, 02a, 03 and 03a have all landed; 04–06 are held** (§7 **D7** split §3.3's charter on 2026-09-17: **03** takes `GkNumericIntegration` and the consumer-spline floor, **T4**; **03a** takes `TkNumericIntegration` and `wavenumber_exit_time`, **T5** and **T6**, and is written after 03 lands) (§1 below; the 2026-09-17
decision — 02's table says which targets the audits own, and 02a settles the anchor that **T6**,
prompt 03a's row since §7 **D7**, turns on). **Prompt 02a has landed** (README §3.2a, authorised by §7 **D6**,
2026-09-17): `main.source_grid_spacing_profile` now guards the stencil evaluation where the
Liouville–Green expansion does not exist, so **the version-2 source grid builds at every production
anchor on every production cosmology**. QCD at its own anchor is **2034 samples / `21ffc126`, 53
guarded nodes**; both published grids are **bit-identical** (1996 / `4849552b`, 1778 / `60a3205a`,
zero guarded), so no figure in the record moves. **The two production anchors are now named
separately** — `wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM` and `PRODUCTION_Z_INIT_QCD` — and prompts
03, 04 and 06 must say which one a figure was taken at, exactly as §2 (b) makes them say which grid
generation. **Prompt 01 has landed**: the convergence facility is
`ComputeTargets/tests/convergence_reference.py` and the source-grid generations are named in
`ComputeTargets/tests/wkb_reference.py`. Every later prompt measures through them.
**Prompt 03 has landed**: `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` is the $G_k$ numeric
matrix and `gk_numeric_sweep.py` regenerates it. **Its answer is `unchanged`** — `atol` is inert
over four decades and `rtol` is the whole lever, but the consumer's own spline of the numeric $G$
carries **1.6e-04 to 9.4e-03** of the envelope near and below the hand-over against **2.6e-07** for
the solver, so tightening buys nothing (**T4**; README §6.1 rule 4). **D1's compute question in
this sector is therefore "spend nothing", which is the reverse of what §7 D1 expected.**
**The user accepted that recommendation on 2026-09-17**, so `GkNumericIntegration`'s pair is
settled at `(1e-10, 1e-8)` and prompt 05 decouples to it. **D1 as a whole closed later the same
day**, when the user accepted prompt 03a's two recommendations as well. The consumer-spline floor the answer rests on is **assigned out of this
campaign** to the hand-over campaign on the same decision (§3, `docs/OPEN_ISSUES.md` §1.1). Two
things in prompt 03's own charter did not hold and are §3 issues: the outermost $z_{\rm source}$ is
**not** the least favourable (so the sweep characterises the sector rather than bounding it), and
the consumer splines the **source** grid, not the response grid the prompt named. The sector's
object count on the version-2 grid is **29,290 / 38,105 / 58,350** per model, not the ~65,000
README §2 (c) quotes from version 0.
**Prompt 03a has landed**: `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` is the $T_k$
numeric and exit-time measurement and `tk_numeric_exit_sweep.py` regenerates it. **Its two answers
are changes, not `unchanged`.** `TkNumericIntegration` should take **`rtol = 3e-11`** — the loosest
of nine settings whose maximum over all fifty wavenumbers on all three models (**3.88e-08**) clears
the freshly re-measured initial-condition floor of **2.39e-06**, at **+39.4 %** of the sector's
right-hand-side evaluations across 50 objects per model (**T5**; README §6.1 rules 2 and 3). The
production `rtol = 1e-8` leaves **14 of 150** runs above README §6's 3e-6, worst **3.36e-04**.
`wavenumber_exit_time` should take **`rtol = 1e-9`** with **`xtol` unchanged at `1e-10` and inert**:
the `xtol` term of Brent's `xtol + rtol*|u|` binds at **0 of 150** (k, offset) pairs on each model,
and `1e-9` is the loosest `rtol` whose *guarantee* at $|u| = 38.04$ (**3.81e-08**) clears the only
floor in the tree, `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, for **+0.7 %** of the target's
Hubble calls (**T6**). **Both were accepted by the user on 2026-09-17, which closes D1 and
unblocks prompt 05** — and the exit-time half was accepted **on the guarantee reading** of §6.1's
rule, the one log 03a's deviation 4 flags as reversible; on the achieved-displacement reading the
same measurement gives `unchanged`. The $T_k$ caveat was accepted with the number rather than
waived: `[03a-tk-numeric-excursion-is-sporadic-in-rtol]` stays open, so `3e-11` ships as the
loosest setting that clears the floor *in that sweep* and not as a bound. **Prompt 17's configuration reproduces exactly** where it can:
version-0 per-$k$ under `BREAK_POINT_DISCONTINUITY` gives 3 / 13 wavenumbers above target and
**8.64e-04** worst on the two smooth models. **The grid change is the larger of the two moves**,
not the policy: `[12-tk-numeric-atol-largest-k-excursion]`'s **3 / 13 / 8** becomes **1 / 9 / 4**,
and the policy is a no-op on the two cosmologies that declare no break points. Three §3 issues are
opened, and the sharpest is that **the QCD version-2 grid's sample count is not reproducible**
(2034 → 2013–2032 under a 1e-14 relative anchor shift), which removes one of the two options
`[02a-grid-digest-not-reproducible]` offers prompt 05.
**Prompt 04 has run and stopped.** `docs/tolerance-convergence/ORDER-AUDIT.md` is the order and
margin audit and `order_audit.py` regenerates it. **T7's answer is `unchanged` on all five knobs.**
$N_\tau$, $N_{c_s\tau}$, $N_F$ and $N_\rho$ are all **4**, measured for the first time on the
corrected background and the 3-point break set, over the version-2 grid at each cosmology's own
anchor and — for $N_\rho$ — at **all fifty** production wavenumbers in both sectors against
`GkTk-remedial` prompt 02's three, which it finds were right and lucky by only **×1.5 to ×1.7**.
The competing floor is double-precision accumulation over the grid, **2.16e-16 / 3.30e-16 /
8.01e-14 / 6.51e-17 rad**, dominating the loosest order the ladder holds by **×3.03e5 /
×1.99e5 / ×48.4 / ×2.35e6** (README §6.1 rule 4's factor).
`RESIDUAL_WKB_REGION_MARGIN = 0.5` is **`unchanged` under README §6.1 rule 6**: it has no accuracy
floor, because the residual a producer reads is a `delta` between two fixed redshifts whose panels
are the grid's own; what it does have is a measured *reachability* bound, cleared at every margin
from 0.05 to **0.9** on every model and both sectors, with the residual **bit-identical** across
that range. **D3's recommendation is `replace with the orders`** — three integer columns on
`BackgroundModel`, one on each WKB target, `GkSource` a `drop`.
**What stopped it is prompt 04 §11's first condition, predicted in terms by its own §2.2 (i):** the
regenerated reference floor is **58×** ($\tau$) and **43×** ($c_s\tau$) tighter than the
2026-09-10 one, and the two tests that read it as a *threshold* — one of them in
`test_background_cs_tau_friction.py`, **outside** the D5 carve-out — then fail by **6.99×** and
**5.08×** with their numerators unmoved to every digit.
**So the `convergence` block was not written**, `test_background_tau.py` was not edited, and
`QCD_BREAK_POINT_ALIGNMENT_TOL` stays at `1.5e-04` — although §9.4 answers prompt §3.3's question
anyway: against the regenerated block the worst offset is **1.421085e-14**, so the constant goes
back *ten orders further* than the 1.4e-05 that prompt hoped for, in whichever commit writes the
block. `[01-convergence-block-has-a-separate-generator]` is **not closed**; it has moved from being
blocked on *scope* to being blocked on a *decision*, and that decision is
`[04-convergence-floor-used-as-a-test-threshold]`.
**Prompt 05 has landed the schema half of D3 (2026-09-18, README §7 D9).** The four object types
whose `(atol, rtol)` pair reached no solver have lost it: `BackgroundModel` keys
`tau_gauss_order`, `cs_tau_gauss_order` and `friction_F_gauss_order`, both WKB sectors key
`rho_gauss_order`, and `GkSource` keys neither, integrating nothing. **The acceptance was the key
and not the column** — each order is an equality criterion of its factory's `build()`, and
repointing its single declaration moves the query, the stored value and the compute class's
accessor together, because every path resolves one module constant at call time with no keyword,
payload key or default anywhere on it (`ComputeTargets/tests/test_gauss_order_key.py`, 16 tests).
**No number moved**: `config/defaults.py` is byte-identical and all four orders are still 4.
`RESIDUAL_WKB_REGION_MARGIN` does **not** become a column, and that half closes on `ORDER-AUDIT.md`
§7.2's bit-identical residual from margin 0.05 to 0.9 rather than being dropped. Reader breakage was
six `extract_*.py` scripts and `main.py` needed **nine** payloads, not the four targets the prompt
counts. `ComputeTargets` **508**, `CosmologyModels` **39**, both OK;
`test_numeric_break_point_key.py` passes unedited; the three published grid digests are unmoved. So
**`[20-wkb-gauss-orders-not-in-lookup-key]` is closed** (§4) and prompt 05a decouples **four**
targets rather than eight (**T9**). One issue is opened,
`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]`, and two are
re-pointed at 05a.
**Prompt 05b has made the recorded order the order that was used (2026-09-18, README §7 D10).** An
object reports the Gauss order it was **built at** on both paths: the compute path carries the
residual table's own order out of `WKB_phase_function`'s payload and the three `compute_background`
echoes into `store()`, and the rehydration path selects each row's order column(s), hands them to
the constructor and reassembles `BackgroundModel`'s three cumulative tables at them. The `order=`
parameter keeps a default, now a **call-time sentinel**, so the def-time snapshot is gone while the
five doc-script call sites of three campaigns still work. **`build()` still filters on the current
module constants** — a row computed at another order is a different row. **Nothing moved**: no
schema, no number (`config/defaults.py` byte-identical, all four orders 4), no `main.py` line, and
no computed value — a SHA-256 over 2,296 doubles of the production path is identical at `90d0114`
and after. Both §6 tests were confirmed to fail at `90d0114`. `ComputeTargets` **516**,
`CosmologyModels` **39**, both OK. So
**`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]` is closed** (§4),
with the rehydration half it does not name, and **T15** is done. Nothing is opened, and log 05's
`main.py` site table stands untouched for 05a.
**Prompt 04b landed the convergence block (2026-09-18, README §7 D8).** The `convergence` block is
**current** — `generated` 2026-09-18, `recommended_scheme` **`branch`**, orders **4 / 4 / 4 / 4**,
`branch+knots` kept as a populated control, and only that key changed. The two threshold
assertions are **rebuilt rather than loosened**: `QCD_NODE_REL_TOL` and `QCD_CS_TAU_REL_TOL`, both
**1.0e-14**, bound the production quantities (2.254e-15, 2.212e-15) against the measured
accumulation floors (2.16e-16, 3.30e-16) instead of against the fixture's own agreement with a
converged reference, and `QCD_FLOOR_FACTOR` is gone from both modules.
**`QCD_BREAK_POINT_ALIGNMENT_TOL` is 3.0e-14**, ten orders back, against offsets that are 1, 2 and
4 ulp of $u$. So **`[01-convergence-block-has-a-separate-generator]` and
`[04-convergence-floor-used-as-a-test-threshold]` are both closed** (§4), and prompt 05 inherits
neither the stale block nor the stop (**T14**).
**One issue got its measurement**: `[01-density-criterion-imposed-outside-the-wkb-region]` has had
only prompt 01's 69-%-of-added-samples figure since it was opened, and prompt 02a's census was
withdrawn as evidence for it. §8 measures it directly — **4 %–49 %** of the `Gk` band is
super-horizon and the band top reaches **13.0 e-folds** outside, while the `Tk` band is
super-horizon **nowhere** — and rebuilding the grid over a horizon-limited band costs **QCD 114 of
2,034 samples and the other two cosmologies none**.
**Prompt 02 has landed**: `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` is the inventory and
`inventory.py` regenerates its tables. **Prompts 03 and 04 take their target lists from log 02's
"State handed to the next prompt", not from README §2 (a).** README §2 (a)'s eight rows are each
confirmed against the tree; what is *not* confirmed is the count — there is a **ninth** keyed
object type, `OneLoopIntegral`, which prompt 02 §8 makes a stop and which is left for the user
(§3 below). README **§0.1's summary sentence** — "of those six only one actually uses it" — is
also wrong: **two** of the six live `(atol, rtol)` sharers use the value, `GkNumericIntegration`
and `wavenumber_exit_time`. §2 (a)'s table says so on both rows; prompt 06 reconciles the README.

> **The campaign is unblocked.** The 2026-09-12 plan was blocked on `prompts/GkTk-remedial` prompt
> 18, because until it landed the reference on `QCDModel` did not converge at four wavenumbers.
> That campaign closed at 20 / 20 and merged (`e8f746d`); `prompts/qcd-background-audit` then ran
> to 16 / 16 and merged (`acd5b8e`). `[17-qcd-reference-not-converged]` is **closed**, and the
> worst QCD reference-convergence drift is now 7.08e-09 ($T_k$) and 3.67e-09 ($G_k$) against the
> 3.4e-08 criterion, zero offenders at all 50 production wavenumbers.
>
> **It is also rebased, and the rebase changed the subject.** The plan counted five compute targets
> sharing two constants across four object types. Measured at `acd5b8e`, **eight** object types are
> keyed on an accuracy parameter; the shared pair keys **six** of them and `rtol` alone keys a
> seventh; and of the six, **only one actually uses the value it is given**. Two targets the plan
> never mentioned — `wavenumber_exit_time`, a live `root_scalar` that fixes where every grid
> begins, and `BackgroundModel`, whose three Gauss orders are the largest instance of the
> integer-order case — are now in scope. `RECONCILIATION.md` scores every claim of the old plan and
> is the document to read before trusting any figure inherited from it.
>
> **Three further things the plan assumed are no longer true.** (i) The production source grid has
> been rebuilt twice and the test tree holds **three disagreeing reproductions** of it, none of
> which is the one `main.py` builds — so every published figure in README §6 was taken on a
> superseded grid. (ii) The cost that D1 turns on was measured in the wrong sector: one decade of
> `rtol` is free across 50 $T_k$ objects per model and is the entire compute decision across
> ~65,000 $G_k$ objects per model, which has never been swept. (iii) The Gauss orders' evidence
> predates two replacements of the background and the removal of the break-point set it was scored
> against; `decision.recommended_scheme` in the fixture still reads `"branch+knots"` and the knots
> do not exist.
>
> **Re-anchored 2026-09-16 onto `bc6dc97`.** A third campaign,
> `prompts/background-solver-robustness`, was planned and run to 9 / 9 **on this branch** after the
> rebase above was written, and merged at `bc6dc97`. It is not a precondition — it is a campaign
> that landed on one of this campaign's own subjects, the `LambdaCDM_GenericEOS.py` root solves.
> `RECONCILIATION.md` **§7** scores it. It moved no tolerance this campaign sets, and
> `config/defaults.py` remains byte-identical to the file the 2026-09-12 plan was written against.
> Three things it changed for the prompts: the three root solves of README §3.2 arrive **already
> settled**, with provenance prompt 02 lifts rather than derives; there are **four**
> production-grid reproductions in the test tree and one is already version 2, so prompt 01
> **hoists rather than builds** (§3 below); and `cosmology_feature_redshifts`, which prompt 01
> lifts, now asks the cosmology for its equality redshifts and raises with no fallback.
>
> **D5 is settled yes** (README §7), so prompt 04 may re-run `residual_convergence.py`, write
> `ComputeTargets/tests/wkb_reference_data.json` and edit `test_background_tau.py` —
> `[01-convergence-block-has-a-separate-generator]` has, for the first time, a prompt allowed to
> close it. **Prompt 04 was written on 2026-09-17 and found that the carve-out is one file short of
> the work**: `test_background_cs_tau_friction.py` and `test_phase_residual.py` also read the
> `convergence` block, the first of them using a recorded floor as a **threshold** — *production
> error ≤ 3 × floor* — so a regenerated, tighter floor can fail a test D5 does not reach. Prompt 04
> §2 makes that a stop rather than a licence, and tells its agent to determine the four orders
> **before** writing the fixture, so that a tree it may not repair is never left red.

---

## 1. The board

| # | Prompt | Covers | Model | Written? | Status | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | The convergence harness and one production grid | README §2 (b), (h); `[00-three-production-grid-reproductions]` | Opus | ✍️ [`01-…`](01-convergence-harness-and-grid.md) | ✅ | *"Build the convergence harness and name the source-grid generations"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/01-…`](logs/01-convergence-harness-and-grid.md) |
| 02 | The accuracy-parameter inventory | README §2 (a), (c), (g); `RECONCILIATION.md` §2.1 | Opus | ✍️ [`02-…`](02-accuracy-parameter-inventory.md) | ⚠️ | *"Inventory every accuracy parameter in the pipeline"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/02-…`](logs/02-accuracy-parameter-inventory.md) |
| 02a | Make the grid buildable at every production anchor | README §3.2a, §7 **D6**; `[01-v2-density-raises-at-the-qcd-production-anchor]` | Opus | ✍️ [`02a-…`](02a-source-grid-density-guard.md) | ⚠️ | *"Guard the source grid density criterion off-node"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/02a-…`](logs/02a-source-grid-density-guard.md) |
| 03 | `GkNumericIntegration`, and the floor that decides whether it matters | README §2 (d), (e), (f); §7 **D7**; review §10.1, §12.5; `[00-gk-numeric-never-swept-and-carries-the-cost]` | Opus | ✍️ [`03-…`](03-gk-numeric-and-its-floor.md) | ⚠️ | *"Sweep the Gk numeric tolerance matrix against its floor"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/03-…`](logs/03-gk-numeric-and-its-floor.md) |
| 03a | `TkNumericIntegration` and `wavenumber_exit_time` | README §3.3a, §7 **D7**; `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`, `[12-tk-numeric-atol-largest-k-excursion]`, `[02a-grid-digest-not-reproducible]` | Opus | ✍️ [`03a-…`](03a-tk-numeric-and-exit-time.md) | ⚠️ | *"Sweep the Tk numeric and exit-time tolerances against their floors"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/03a-…`](logs/03a-tk-numeric-and-exit-time.md) |
| 04 | Audit the order-governed targets | README §2 (a), §3.4; §7 **D5** (settled yes) and **D3**; `[01-convergence-block-has-a-separate-generator]`, `[20-wkb-gauss-orders-not-in-lookup-key]`, `[01-density-criterion-imposed-outside-the-wkb-region]` | Opus | ✍️ [`04-…`](04-order-governed-targets.md) | ⚠️ **stopped** | *"Audit the four Gauss orders and the WKB region margin"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/04-…`](logs/04-order-governed-targets.md) |
| 04b | Regenerate the convergence block, and repair the two tests that read it | README §3.4b; §7 **D5** as widened by **D8** (settled yes 2026-09-18); `[01-convergence-block-has-a-separate-generator]`, `[04-convergence-floor-used-as-a-test-threshold]` | Opus | ✍️ [`04b-…`](04b-regenerate-the-convergence-block.md) | ✅ | *"Land the regenerated convergence block"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/04b-…`](logs/04b-regenerate-the-convergence-block.md) |
| 05 | Replace the vestigial key columns with the orders | README §3.5; §7 **D3** (settled 2026-09-18), **D9** (split 2026-09-18); `[20-wkb-gauss-orders-not-in-lookup-key]` | Opus | ✍️ [`05-…`](05-replace-the-vestigial-key-columns.md) | ✅ | *"Key the order-governed targets on their Gauss orders"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/05-…`](logs/05-replace-the-vestigial-key-columns.md) |
| 05b | Make the recorded order the order that was used | README §3.5b; §7 **D10** (settled 2026-09-18); `[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]` | Opus | ✍️ [`05b-…`](05b-make-the-recorded-order-the-order-used.md) | ✅ | *"Make an object report the Gauss order it was built at"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/05b-…`](logs/05b-make-the-recorded-order-the-order-used.md) |
| 05a | Decouple the tolerances that are real | README §3.5a; §2 (a), (e), (g); §7 **D1** (closed 2026-09-17), **D9** | Opus | ✍️ [`05a-…`](05a-decouple-the-tolerances-that-are-real.md) — **written 2026-09-18, not yet dispatched** (was: held to be written against what 05 left, which log 05's "State handed to the next prompt" now enumerates site by site** | ⬜ | | |
| 06 | `QuadSourceIntegral`, close-out, the provenance note | README §0.4, §1.2 | Opus | ⏸️ **held** | ⬜ | | |

Status key: ⬜ not started · 🔄 in flight · ✅ complete · ⚠️ complete with a recorded caveat ·
❌ blocked. Written key: ✍️ prompt file exists · ⏸️ deliberately held.

> **Prompts 03–06 are held back by decision, 2026-09-16 — this is not an unfinished plan.**
> README §3 fixes each prompt's charter and §6 fixes its acceptance, so what is held is the
> **method**, not the commitment; the campaign is fully specified and partly written. The reason is
> §4's own dependency graph: *"02 must precede both audits: it is what says which targets 03 and 04
> each own, and the old plan's allocation was wrong."* Writing 03 and 04 now would allocate their
> targets from the same unverified inventory that prompt 02 exists to replace — the error the
> 2026-09-12 plan made, and the reason new 02 has no predecessor. 05's content **is** D1 and D3,
> which do not exist until 03 and 04 report; 06 assembles from the earlier logs.
>
> **The staging, as revised 2026-09-17:** 01, 02 and **02a** landed; **03 written after 02 and 02a
> both landed**, against 02's table and 02a's hand-off rather than against a guess at either; **03a
> and 04 after 03 lands**; 05 after the user settles D1 and D3; 06 last. 02a joined the precondition
> because **T6 turns on the anchor 02a settles** — a prompt drafted from 02 alone would allocate
> `wavenumber_exit_time` without the measurement that changes its charter. **Since §7 D7, T6 is
> prompt 03a's**, and 03a is written after 03 so the $T_k$ re-take knows what the $G_k$ sweep found
> about the axes. This costs nothing in elapsed time — §4.1 already declares a stop after each of
> 02, 03, 03a and 04, and 02a ends in a hand-back for the same reason, so the user is in the loop
> anyway. Orchestrator prompts
> are staged with them ([`orchestrator/README.md`](orchestrator/README.md)).

> **02a is an insertion, not a renumbering.** It carries a letter so that the charters of README
> §3.3–§3.6, the acceptance rows of §6.2 and the T-numbers of §2 keep the numbers every other
> document in the tree cites. It is the first production change this campaign makes; §7 D6 records
> the widened §0.5 boundary that permits it, and the carve-out is
> `main.source_grid_spacing_profile` alone. Its acceptance is **bit-identity** with the two
> published grid digests, not improvement — a moved digest is a stop under README §4.3.

**The 2026-09-12 board carried five prompts.** The mapping, so that a reader of the old plan is not
lost: old 01 → new 01 (widened by the grid); old 02 → new 03 (widened by `wavenumber_exit_time` and
by $G_k$ becoming the centre); old 03 → new 04 (widened by `BackgroundModel` and by the stale
fixture); old 04 → new 05; old 05 → new 06. **New 02 has no predecessor** — it is the inventory the
old plan assumed and got wrong.

---

## 2. Item-level state

One row per thing the campaign claims to establish. Filled in as prompts land; a row whose evidence
is a single wavenumber or a single model is **not** ✅, which is the specific failure this campaign
was created by. A row whose evidence does not say which source-grid generation it was taken on is
not ✅ either, which is the specific failure the rebase found.

| Item | Kind | Statement | Prompt | Status |
|---|---|---|---|---|
| T1 | **MACHINERY** | One reusable convergence facility, in the test tree, covering every target and calibrated against the constant-$w$ anchors at every use, with "one step tighter" meaning a decade for a tolerance and one order for a Gauss order | 01 | ✅ `convergence_reference.py`: `TolerancePair` / `GaussOrder`, `reference_drift` → `DriftVerdict` (the criterion evaluated, never just reported), `radiation_anchors` over all eleven closed forms, the two numeric sectors folded in from `tk_numeric_atol_sweep.py` |
| T2 | **MACHINERY** | **One** reproduction of the production source grid at `SOURCE_GRID_CONSTRUCTION_VERSION = 2`, with the version-0 and version-1 constructions retained and named rather than silently re-scored | 01 | ⚠️ `wkb_reference.source_grid(SOURCE_GRID_V0/_V1/_V2, …)`, no default generation, bit-identical to all three constructions it replaces. **Caveat:** `production_source_grid` keeps its historic behaviour because 28 call sites in 20 files outside prompt 01's scope import it; it is documented as version 0 rather than made to fail loudly |
| T3 | **MEASUREMENT** | The accuracy-parameter inventory: every parameter, what it keys, whether it reaches a solver, what the real knob is, and the object count of the sector it keys | 02 | ⚠️ `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` + `inventory.py`: **12 keyed tables** derived from `Datastore/SQL/Datastore.py`'s `_factories` by `ast`, **39 parameter rows** in nine columns, **10 hard-coded literals** in production. README §2 (a)'s eight rows are each **confirmed**. **Caveat:** there is a **ninth** keyed object type, `OneLoopIntegral` — prompt 02 §8's stop condition, left unassigned for the user |
| T4 | **MEASUREMENT** | `GkNumericIntegration` characterised over the production response grid on three models in both `atol` and `rtol`, against the consumer-spline floor — the sector with ~65,000 objects per model, never swept, and where the campaign's compute decision actually lives | 03 | ⚠️ `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` + `gk_numeric_sweep.py`: 50 $k$ × 3 models × 15 `(atol, rtol)` cells, **version-2 grid at each cosmology's own anchor**, `BREAK_POINT_DISCONTINUITY`, reference `(1e-18, 1e-12)` converged **50/50 on all three**. **§2 (d)'s prior is confirmed**: four decades of `atol` move the maximum by ≤1.2 %, four decades of `rtol` by ×13,300, and the corners agree to three significant figures — `atol` cannot bind, $\lvert G\rvert$ being 2.1e+12–2.9e+18 in `Mpc_units`. **The floor is freshly measured and dominates**: the consumer's `numeric_Gk` spline carries 1.6e-04/1.7e-04/1.9e-04 of the envelope three e-folds inside the horizon and up to **9.4e-03** at four, against 2.6e-07 for the solver — ×631 to ×37,700, confirming review §10.1. **Target: `unchanged`** (§6.1 rule 4). Objects per model on the v2 grid are **29,290 / 38,105 / 58,350**, not the ~65,000 README §2 (c) quotes from version 0. **Caveat:** §2.2's premise fails — the outermost $z_{\rm source}$ is **not** the least favourable at any of nine probes (≤×1.45), so the sweep characterises the sector rather than bounding it; and the prompt's §5 names the wrong lattice for the consumer's spline (it is the **source** grid, 12× finer). Both are §3 issues |
| T5 | **MEASUREMENT** | `TkNumericIntegration` likewise, re-taken on the version-2 grid and under its own `BREAK_POINT_ALL` policy | 03a | ⚠️ `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` + `tk_numeric_exit_sweep.py`: 50 $k$ × 3 models × 13 `(atol, rtol)` cells, **version-2 grid at each cosmology's own anchor**, **`BREAK_POINT_ALL`**, reference `(1e-18, 1e-12)` converged **50/50 on all three**. **Prompt 17's configuration reproduces**: version-0 per-$k$ under `discontinuity` gives 3 / 13 above 3e-6 and worst **8.64e-04** on Radiation / LambdaCDM (QCD gives 9, not 8, its background having been replaced since). **Re-taken as production runs it, 3 / 13 / 8 becomes 1 / 9 / 4 and the worst 8.64e-04 becomes 3.36e-04** — and the **grid** change, not the policy, is most of that: `all` and `discontinuity` are the same code path on the two cosmologies that declare nothing, and on QCD the policy moves 9→6 (v0) and 7→4 (v2) for +0.82 % / +0.61 % of the evaluations. **`rtol` sets the level and `atol` selects which wavenumber excurses** — two decades of `atol` move the median of per-$k$ maxima by ≤2.1× and the maximum by up to **205×** — so README §2 (e)'s magnitude argument does *not* transfer from the $G_k$ sector. **Floor re-confirmed**: the $T=1,T'=0$ truncation is **2.39e-06 to 2.64e-06** against the inherited 2.52e-06, and the radiation oracle and the series measure agree to three figures. **Target: `rtol = 3e-11`** (§6.1 rules 2 and 3), 3.88e-08 worst, 62× under the floor, **+39.4 %** of 1,272,891 → 1,774,977 RHS evaluations for 50 objects × 3 models. **Caveat:** the maximum is **not monotone** in `rtol` — exactly one of the 150 runs sits above target at each of `1e-9`, `3e-10` and `1e-10`, a different one each time — so `3e-11` is the loosest that clears *in this sweep*, not a bound (§3 issue) |
| T6 | **MEASUREMENT** | `wavenumber_exit_time`'s root solve measured at all — nothing in the record says what `xtol = 1e-10`, `rtol = 1e-8` in $\log(1+z)$ buys or costs. **Scored against the exact $z_{\rm exit}$ on `RadiationModel` first** (README §3.1): $1 + z = k/(H_0 e^{N})$, confirmed at the rebase to 2.3e-16 relative or better | 03a | ⚠️ Measured through `_solve_horizon_exit` and **never the datastore**: 50 $k$ × 3 models × 3 offsets (0, −5, +4) × 63 `(xtol, rtol)` cells, reference `(1e-300, 1e-14)`, drift ≤7.82e-14 in $u$ and **3.55e-15** from the exact inversion on the control, where the production setting's displacement is **exactly 0**. **`xtol = 1e-10` binds at 0 of 150 pairs on every model**: `DEFAULT_ABS_TOLERANCE` reaches this target and does nothing, and what fixes the anchor is `rtol*|u|` at $|u|$ up to 38.04. It takes over only below `rtol ≈ 2.6e-12`, and **floors the pair at 1e-10 relative** however far `rtol` goes alone. The **$u\to z$ recovery never competes**: one ulp of $u$ is 7.11e-15 relative in $1+z$ and the production criterion stands 5.3e7 above it. **Two consumers, two answers.** Against the row match `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7` a floor exists and the rule gives **`rtol = 1e-9`** (guarantee 3.81e-08; production's guarantee 3.8e-07 misses by 3.8× while its *achieved* 7.86e-08 clears by 1.3×), for **+0.7 %** of 6,963 Hubble calls. Against the **grid digest** — bit-identity — **no floor could be established, therefore no target** (§6.1 rule 6): the digest turns over at a **1e-14** relative anchor shift against a tightest available pin of 7.11e-15. **Caveat:** the recommendation applies the rule to Brent's *guarantee*, not to the achieved displacement; on the achieved reading the answer is `unchanged` (log 03a, deviation 4) |
| T7 | **MEASUREMENT** | $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$ and `RESIDUAL_WKB_REGION_MARGIN` audited at every production $k$ on the corrected background and the 3-point break set, replacing evidence generated 2026-09-10. **Every one of the four orders has a closed-form anchor on `RadiationModel`** (README §3.1), including $\rho_G \equiv 0$, which makes the $N_\rho$ measurement pure quadrature error with no reference to build | 04 | ⚠️ `docs/tolerance-convergence/ORDER-AUDIT.md` + `order_audit.py`: three models on the **version-2 grid at each cosmology's own anchor**, orders 2/4/6/8/12/16, the three primitives $k$-independent and $N_\rho$ at **all fifty** production wavenumbers in both sectors (**300 cases**), everything through `convergence_reference.GaussOrder` and `reference_drift`. **All four orders are `unchanged` at 4**, first order clearing floors of **2.16e-16 / 3.30e-16 / 8.01e-14 / 6.51e-17 rad**, which dominate the loosest order swept by **×3.03e5 / ×1.99e5 / ×48.4 / ×2.35e6** (§6.1 rule 4's factor). **The anchors did the work**: $\rho_G$ is **bit-exactly zero at every order** on the control and $\rho_T$ is scored against its closed form, so the radiation column has no reference and no drift. `GkTk-remedial` prompt 02's three wavenumbers were right and lucky by only **×1.5–×1.7**. `RESIDUAL_WKB_REGION_MARGIN` is **`unchanged` under §6.1 rule 6** — no accuracy floor; a measured reachability bound, cleared from 0.05 to **0.9**, residual **bit-identical** throughout, and `residual_node_range` refuses only *at* margin 1, where the test degenerates to "is the correction positive?". **Caveats, three.** (i) The prompt **stopped** before writing the fixture (§3 below, `[04-convergence-floor-used-as-a-test-threshold]`), so the evidence is taken back but not yet *recorded* in the block every `*_GAUSS_ORDER` comment cites — **caveat (i) is discharged: prompt 04b wrote the block on 2026-09-18 (T14), unchanged from the run measured here**. (ii) On the two spline models **every order from 4 up is at or below the reference's own drift** and is marked unresolved; what is resolved is the step from order 2, and the exact-radiation oracle. (iii) The prompt's own §7 warning about non-monotone Gauss rules did not bite — every ladder is monotone at and above the chosen order |
| T8 | **DECISION** | The decoupled tolerance pairs settled by the user (§7 D1) and shipped with the measurement that chose each, in `config/defaults.py` **Reassigned to prompt 05a by §7 D9, 2026-09-18**, with T10: the schema half of the old §3.5 is prompt 05 and changes no number, so every constant in this row moves in 05a. | 05a | ⬜ — **the `GkNumericIntegration` half is decided**: the user accepted prompt 03's `unchanged` on 2026-09-17, so `DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10` (inert and unchosen) and `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8` (chosen, `GK-NUMERIC-SWEEP.md` §5.2, §7). **`TkNumericIntegration`'s `rtol` and `wavenumber_exit_time`'s pair were accepted on 2026-09-17, which closes D1**: `DEFAULT_TK_NUMERIC_REL_TOLERANCE = 3e-11` (changed; `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` stays 1e-13, a step-selection knob rather than an accuracy one), `DEFAULT_HEXIT_REL_TOLERANCE = 1e-9` (changed) and `DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10` (unchanged, **inert and unchosen**, and coupled — it floors the pair at 1e-10 relative below `rtol ≈ 2.6e-12`). The five provenance fields for all four are in log 03a's "State handed to the next prompt". **All four pairs are now settled and prompt 05 is unblocked on D1**; it still waits on D3, which is prompt 04's. Two things ship with the numbers rather than being dropped: the exit-time pair was accepted on the **guarantee** reading (log 03a, deviation 4), and `3e-11` carries `[03a-tk-numeric-excursion-is-sporadic-in-rtol]`, which the acceptance did not close |
| T9 | **DECISION** | What replaces the vestigial `atol`/`rtol` key columns on the three order-governed targets (§7 D3) — the user's stated target for the campaign. **Settled 2026-09-18: `replace with the orders`**, on prompt 04's recommendation. `BackgroundModel` loses `atol_serial`/`rtol_serial` and gains `tau_gauss_order`, `cs_tau_gauss_order`, `friction_F_gauss_order`; `GkWKBIntegration` and `TkWKBIntegration` each gain `rho_gauss_order`; `GkSource` loses the pair and gains nothing, integrating nothing. The user's standing instruction came with it and binds every later schema question here: **there is nothing to rebuild and nothing to backfill, and a needed schema change is always the right answer** — only *reader breakage* may be costed (three `extract_*.py` scripts). Prompt 05 implements **This row is the whole of prompt 05 under §7 D9, 2026-09-18.** The acceptance is not that the columns exist but that the order is *filtered on, not merely selected* — `TOLERANCE-INVENTORY.md` §2.4 measured that the solver table is joined so the order can be read back and then never compared, and a new column keyed the same way would reproduce that exactly. | 05 | ✅ **Landed 2026-09-18.** `BackgroundModel` loses `atol_serial`/`rtol_serial` and gains `tau_gauss_order`, `cs_tau_gauss_order`, `friction_F_gauss_order`; `GkWKBIntegration` and `TkWKBIntegration` each gain `rho_gauss_order`; `GkSource` loses the pair and gains nothing. **The acceptance is the key, not the column**, and it is met: each order is an equality criterion of its factory's `build()` — read back off the compiled `whereclause` with no database behind it, the technique `test_numeric_break_point_key.py` established — and repointing its **single** declaration moves the query, the value `store()` writes and the compute class's accessor **together**. The mechanism is one name resolved at call time on every path: `compute_background` and `WKB_phase_function` read the module constant, the compute object's property returns it, `store()` writes that property, `build()` filters on the same module attribute, and no keyword, payload key or default exists by which a caller could supply another. `ComputeTargets/tests/test_gauss_order_key.py`, **16 tests**, is the proof, including that no site inlined the number. **No number moves**: `config/defaults.py` byte-identical, all four orders still 4. `RESIDUAL_WKB_REGION_MARGIN` does **not** become a column, verified against `ORDER-AUDIT.md` §7.2 rather than taken from the prompt — bit-identical residual at every margin from 0.05 to 0.9 on three models and both sectors, with the qualification that at `margin >= 0.99` the `Tk` anchor is *clamped* (a reachability failure, not a differing row) and at 0.9999 two QCD `Gk` cells move by one ulp. **15 production files**; `ComputeTargets` **508**, `CosmologyModels` **39**, both OK; `test_numeric_break_point_key.py` passes **unedited**; the three published grid digests unmoved. Reader breakage was **six** `extract_*.py` scripts as §5 said, and `main.py` needed **nine** payloads rather than the four targets the prompt counts (log 05, deviation 1) |
| T10 | **PLUMBING** | Every `object_get` of a retuned target carries its own parameter, with an `ast` guard whose predicate reaches all eight targets and fails on an unclassified site **Reassigned to prompt 05a by §7 D9, 2026-09-18**, with T8. After 05 the guard enumerates four targets carrying a tolerance and four carrying none, and a tolerance reappearing on one of the latter is itself a regression it should catch. | 05a | ⬜ |
| T11 | **HAND-OFF** | `QuadSourceIntegral` measured read-only and reported to `levin-refactor` / `qsi-phase-groups` | 06 | ⬜ |
| T12 | **PROVENANCE** | `docs/TOLERANCE-PROVENANCE.md` covers **every** accuracy parameter in the pipeline — including the ones this campaign inherits and does not set, and the ones nobody has ever chosen — with value, choosing measurement and its grid generation, competing floor, cost times object count, and citation (README §1.2) | 06 | ⬜ |
| T13 | **MACHINERY** | The version-2 source grid **builds at every production anchor on every production cosmology**, with a node at which the Liouville–Green expansion does not exist marked unusable and *counted* rather than raising, a refusal above a measured fraction of the band, and QCD's own anchor named in the test tree. Acceptance is bit-identity with the two published digests (1996 / `4849552b`, 1778 / `60a3205a`), not accuracy | 02a | ⚠️ QCD at its own anchor builds: **2034 / `21ffc126`, 53 guarded `(k, sector)` evaluations of ONE base-lattice node** (Gk 34, Tk 19, 53 of 100 cases, all hitting $z = 8.63614\times10^{11}$). Both published grids **bit-identical with zero guarded**, so the `except` branch is never entered on either. **Corrected 2026-09-17 (log 02a §7):** the guard is inert more strongly than first claimed — that node lies 1.0e-03 in $u$ from a **declared** break, inside the 6.0e-03 crossing mask, so it is discarded anyway and the guard can change no profile — but the two published zeros are **lattice alignment, not construction**, so a change to `z_init` or `samples_per_log10z` could start guarding. Refusal ceiling `SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05`, **64.8x** the worst measured band (7.716e-04). **Caveat:** the constant's only proper home is `CosmologyConcepts/wavenumber.py`, a file the prompt's own file list forbade; the agent stopped rather than editing it, and the **user amended the scope on 2026-09-17** to permit that one append-only definition (log 02a, deviation 1) |
| T14 | **EVIDENCE** | The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` is **current**: regenerated by its own generator against the corrected $T(z)$ representation and the 3-point break set, recommending a scheme production can execute, with `branch+knots` retained as a measured control — and the two threshold tests that read it bound the production quantity against a measured floor rather than against the fixture's own agreement with a converged reference. `QCD_BREAK_POINT_ALIGNMENT_TOL` back from 1.5e-04 to ~1.42e-14 | 04b | ✅ **Landed 2026-09-18.** The block is `generated` **2026-09-18** by `prompts/tolerance-convergence (prompt 04, board item T7)`, `schema_version` 2, **`recommended_scheme` = `branch`**, orders back **4 / 4 / 4 / 4**, `branch+knots` populated as `control_scheme`, and **only the `convergence` key differs from `HEAD~1`** (checked key by key on the parsed JSON). Written by `residual_convergence.py` with no flags in 538.4 s and **never by hand**; every figure reproduces prompt 04's dry run. The two assertions are **rebuilt, not loosened**: `QCD_NODE_REL_TOL = 1.0e-14` and `QCD_CS_TAU_REL_TOL = 1.0e-14`, absolute bounds on the production quantities (2.254e-15 / 2.212e-15) against the accumulation floors `ORDER-AUDIT.md` §§3.1 and 5 measure (2.16e-16 / 3.30e-16) — ×4.4 / ×4.5 headroom, ×46 / ×30 the floor — and `QCD_FLOOR_FACTOR` is **removed** from both modules. **`QCD_BREAK_POINT_ALIGNMENT_TOL` = 3.0e-14**, against offsets of 1, 2 and 4 ulp of $u$ (3.552714e-15 / 7.105427e-15 / **1.421085e-14**). `ComputeTargets` **491**, `CosmologyModels` **39**, `test_convergence_reference` **32**, the three readers **41**, all OK, `test_phase_residual.py` **unedited**; the three grid digests unmoved. `ORDER-AUDIT.md` §12 records it additively |
| T15 | **CORRECTNESS** | An object reports the Gauss order it was **built at**, on every path by which it can come into existence — computed fresh, the order its tables were actually constructed with; rehydrated from a row, the row's stored order, with its tables rebuilt at that order. Prompt 05's mechanism (a property that re-reads the module constant) is right about the key and wrong about the object, and fails in two places: `phase_residual`'s `order=` keyword can persist a table at an order it was not built at, and `BackgroundModel._build_tau_primitive` / `_build_cs_tau_primitive` / `_build_friction_F_primitive` reassemble a stored model at the **current** constant while the row's three order columns never reach the constructor at all — and, as prompt 05b found, were not in `build()`'s select list to begin with (log 05b, deviation 2), so the column had to be selected as well as passed. Both are masked because `build()` filters on the constant, so a served row always happens to carry it — correctness resting on an argument about the filter rather than on construction. **`build()` keeps filtering on the current constant**: a row computed at another order is a different row, not a miss to repair. Settled by the user 2026-09-18 as a correctness issue (§7 **D10**) | 05b | ✅ **Done, 2026-09-18.** The order travels as data on both paths: `WKB_phase_function` carries the residual table's own `CumulativeTable.order` out in its payload and `store()` records it; `BackgroundModel.store()` records `compute_background`'s three echoes; each `build()` **selects** its order column(s) and hands them to the constructor, and the three `_build_*_primitive` methods rebuild at the row's orders. The `order=` default is kept as a **call-time sentinel** (`Optional[int] = None`) rather than made required, so the five doc-script call sites of three campaigns still work and the def-time snapshot is gone. `build()` still filters on the current module constants, asserted. **Nothing moved**: `config/defaults.py` byte-identical, all four orders 4, `main.py` zero lines, no `register()` touched, and a SHA-256 over 2,296 doubles of the production path is identical at `90d0114` and after. `ComputeTargets` **516**, `CosmologyModels` **39**. Log 05b |

---

## 3. Active and unresolved issues

Issues opened here must be added to [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) **in the
same commit** (`CLAUDE.md`), with the count corrected.

Opened by the **2026-09-16 rebase**:

- **[00-three-production-grid-reproductions]** *(rebase, 2026-09-16; **narrowed at the re-anchor,
  2026-09-16**; assigned to prompt 01)* — the test tree holds three constructions each called "the
  production source grid" and they are three different grids. `ComputeTargets/tests/wkb_reference.py:152` is a bare `np.logspace` reproducing
  `populate_z_sample` and citing `main.py:410-419` — **version 0**, what production built before
  `qcd-background-audit` prompt 11. `ComputeTargets/tests/test_background_segmentation.py:90`
  passes `break_z` and `feature_z` but no `spacing` profile — **version 1**, prompt 11's grid.
  `main.py:911-930` passes the curvature spacing profile — **version 2**,
  `SOURCE_GRID_CONSTRUCTION_VERSION = 2`, 1,996 samples on QCD and 1,778 on LambdaCDM against
  version 0's 1,732. **Impact:** every figure in README §6 and every figure in
  `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` was scored on version 0, because
  `tk_numeric_atol_sweep.py:215` imports the first of the three; a tolerance chosen from those
  figures would be chosen for a grid production does not use. Not a wrong *number* — each was
  correct for the tree it was taken on — but an unmarked one, which is why README §5 rule 6 now
  requires the generation beside the figure. **Next step:** prompt 01 builds one reproduction at
  version 2, lifting `source_grid_spacing_profile` and `cosmology_feature_redshifts` from `main.py`
  with `load_main_py_functions`, and has the other sites import it. Keep versions 0 and 1
  constructible and named: prompt 17's figures and
  `test_background_segmentation`'s assertions are scored on them and must not be silently
  re-based.

  > **Narrowed at the re-anchor, 2026-09-16** (additively; `RECONCILIATION.md` §7.3). It is
  > **four** constructions, not three, and the rebase missed the fourth — which was already there
  > at `acd5b8e`. `ComputeTargets/tests/test_source_grid.py:127` `_production_grid` **is the
  > version-2 construction**, complete, under the suite, and lifting both `main.py` functions the
  > way the next step above prescribes; and `:151` `_production_base_grid` is a v0 base that is
  > deliberate and named, mirroring `main.py:944`'s own base-grid step. **The corrected next step:
  > prompt 01 *hoists* `_production_grid` out of that test module into something the other sites
  > can import, and repoints `wkb_reference.py:152` and `tk_numeric_atol_sweep.py:215` at it.** It
  > is private and it is in a test module; that is now the whole defect. The **impact is
  > unchanged** — every figure in README §6 and in `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` is
  > still scored on version 0, because `tk_numeric_atol_sweep.py:215` still imports the v0 one. One
  > further constraint the re-anchor adds: `cosmology_feature_redshifts` now asks the cosmology for
  > `z_matter_radiation_equality` / `z_matter_lambda_equality` and **raises with no fallback**, so
  > the stand-in prompt 01 lifts it against must answer both (`RECONCILIATION.md` §7.4);
  > `test_source_grid.py:189` already carries one that does.

  > **Narrowed again by prompt 01, 2026-09-16** (additively). The hoist is done and is
  > bit-identical: `ComputeTargets/tests/wkb_reference.py` now carries
  > `source_grid(generation, …)` over three **named** generations — `SOURCE_GRID_V0`,
  > `SOURCE_GRID_V1`, `SOURCE_GRID_V2` — with **no default**, tagged
  > `SOURCE_GRID_V2_REPRODUCES_VERSION = 2` and cross-checked against production's own
  > `SOURCE_GRID_CONSTRUCTION_VERSION` rather than asserting it. `test_source_grid.py:127`,
  > `test_background_segmentation.py:90` and `tk_numeric_atol_sweep.py`'s two geometries are
  > repointed, the last of them naming version 0 explicitly; `test_source_grid.py:152`
  > `_production_base_grid` is left alone, as it should be. Verified: 1,732 / `0960e169` (v0),
  > 1,773 / `81c6e682` (v1 QCD), 1,996 / `4849552b` (v2 QCD), 1,778 / `60a3205a` (v2 LambdaCDM),
  > every one bit-identical to the construction it replaces, and
  > `tk_numeric_atol_sweep.py`'s entire 294-line output unchanged but for the wall clock it prints
  > about itself.
  >
  > **What is left, and it is the whole reason this is a narrowing and not a closure:**
  > `wkb_reference.production_source_grid` still exists with its historic behaviour, because
  > **28 call sites in 20 files** import it — 11 test modules under `ComputeTargets/tests/` and 9
  > scripts under `docs/gktk-remedial/` and `docs/qcd-background-audit/` — and none of those files
  > was in prompt 01's scope. They are all version 0 and all correct; what they do not do is *say
  > so*. The prompt asked that the bare name "fail loudly rather than default", which cannot be
  > done without editing all twenty (log 01, deviation 1). Its docstring now opens "**The
  > version-0 source grid**". **Next step:** a prompt given those twenty files repoints them at
  > `source_grid(SOURCE_GRID_V0, …)` and deletes the alias. It is mechanical and every call is
  > bit-identical; it is scope, not difficulty.

- **[00-gk-numeric-never-swept-and-carries-the-cost]** *(rebase, 2026-09-16; assigned to prompt
  03)* — the campaign's central prior, "the error is set by `rtol`", is one clean measurement and
  one diagonal. `GkTk-remedial` prompt 17 holds `atol = 1e-13` and moves `rtol` alone, so it
  separates the axes — in the sector that is **one object per $k$, 50 per model**
  (`main.py:1180`, `:1215` at the re-anchor). Review §10.1 on $G_k$ moves `(1e-10, 1e-8)` → `(1e-13, 1e-11)`, a diagonal, at
  four source redshifts on two models
  (`docs/gk-wkb-review-fable-2026-09-09.md:469-474`) — in the sector that is one object per
  $(k, z_{\rm source})$, ~65,000 per model. **Impact:** the +23–25 % evaluations one decade of
  `rtol` costs is free where it was measured and is the campaign's entire compute decision where it
  was not; and README §7 D1 attaches the ~65,000 object count to the wrong sector. Worse, the same
  review paragraph says the consumer's cubic spline of the numeric $G$ carries 1e-5 to 1e-4 near
  the hand-over, "the larger error by two orders" — so the honest outcome may be that $G_k$'s
  `rtol` should not move at all. **Next step:** prompt 03 sweeps $G_k$ in both axes on three models
  over the production response grid and measures the consumer-spline floor beside it, before any
  `rtol` is recommended for either sector.

Opened by **prompt 01**, 2026-09-16:

- **[01-density-criterion-imposed-outside-the-wkb-region]** *(orchestrator review of prompt 01,
  2026-09-16; **narrowed 2026-09-17, no longer blocking; measured by prompt 02a, 2026-09-17**; unassigned — candidate for **T7**)* — `main.source_grid_spacing_profile` imposes the
  fourth-derivative equidistribution criterion over the band `residual_node_range` returns, and
  that band reaches **1.5 to 2.1 e-folds outside the horizon**, where the Liouville-Green phase
  spline the criterion exists to protect is never evaluated. The horizon condition is
  `k/aH = omega_0 (1+z) = 1`, with `leading = omega_0^2 = (k/H)^2` and `H` the ordinary Hubble
  rate: **`omega_0 = 1` is not horizon crossing** and misreads the band by five orders of
  magnitude in z. Measured on `QCD_Cosmology` at its own anchor, `Gk` sector:

  | $k$ [1/Mpc] | $z$ at $k = aH$ | band top | $k/aH$ there | e-folds outside |
  |---|---|---|---|---|
  | 1e+05 | 5.08e+10 | 2.67e+11 | 0.196 | 1.63 |
  | 1.39e+05 | 7.04e+10 | 3.21e+11 | 0.228 | 1.48 |
  | 1.92e+05 | 9.77e+10 | 8.25e+11 | 0.128 | 2.06 |
  | 2.26e+05 | 1.15e+11 | 8.44e+11 | 0.147 | 1.92 |

  `RESIDUAL_WKB_REGION_MARGIN`'s own comment states that production anchors sit **three e-folds
  inside** the horizon, so the criterion is imposed roughly **five e-folds beyond the last place
  its consumer exists**. At the raise point of
  `[01-v2-density-raises-at-the-qcd-production-anchor]` the mode is **1.78 e-folds outside** the
  horizon and `|C| / omega_0^2 = 5.9e+04` — the "correction" is 59,000x the leading term, so the
  expansion being differentiated has stopped meaning anything, and the criterion is in effect
  chasing its own breakdown. The cause is a **reuse**, not a coding error: that margin was designed
  as a *permissive* bound so the band never excludes a producer's anchor ("the range covers every
  anchor the producer accepts, with margin"), and the spacing profile reuses it as an upper bound
  on where the spline needs resolving. **Impact:** of the samples version 2 adds over version 1, at
  the LambdaCDM anchor — **QCD 154 of 223 (69%)** lie above horizon crossing for the smallest
  production $k$ and **36 (16%)** above crossing for *every* production $k$; **LambdaCDM 14 of 46
  (30%)** and **0 (0%)**. In that region the consumer is `GkNumericIntegration` /
  `TkNumericIntegration`, whose accuracy is governed by `atol` / `rtol` and the solver's own step
  control, not by a spline-interpolation bound. Because `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` the
  criterion may only ever *refine*, so these samples corrupt no result — they are **unjustified
  rather than wrong**, and 223 in 1,996 is not a cost problem; the sharp consequence is the sibling
  issue's hard failure. **Next step:** decide whether the spacing profile should run over a
  horizon-based band of its own rather than over `residual_node_range`'s anchor-coverage band. That
  is a production change to the source grid, which README §0.5 holds fixed here; item **T7**
  already audits `RESIDUAL_WKB_REGION_MARGIN` at every production $k$, so prompt 04's charter is
  the natural home for it when that prompt is written.

  > **Narrowed 2026-09-17, and no longer blocking.** Prompt **02a** (README §3.2a, §7 D6) makes the
  > grid buildable without touching the band, so this issue stops being a hard failure and becomes
  > what it always was underneath: unjustified refinement with a measurable cost. The mechanism is
  > now stated — the band is established **node-wise** and the criterion is evaluated **off-node**,
  > so the expansion's non-existence is reached at `u ± delta` between two nodes that both pass the
  > margin test. 02a's **guarded-node count is the evidence this issue has been waiting for**: 53
  > nodes at QCD's own anchor, zero at either published anchor, which is a direct measure of how far
  > the band overreaches and where. **Still unassigned as a decision, still a candidate for T7**:
  > whether the spacing profile should run over a horizon-based band of its own is prompt 04's to
  > recommend, and 02a is explicitly forbidden from pre-empting it.

  > **Measured by prompt 02a, 2026-09-17** (additively). The guarded-node census is the direct
  > measure of the overreach this issue asserts, and it is the evidence the issue has been
  > waiting for. At production geometry, **source-grid generation version 2**, fifty wavenumbers,
  > both sectors:
  >
  > | cosmology / anchor | guarded | band-node evaluations | fraction | Gk | Tk | cases | worst single band |
  > |---|---|---|---|---|---|---|---|
  > | QCD at LambdaCDM's, 2.0636395964161516e+16 | 0 | 136492 | 0 | 0 | 0 | 0 / 100 | 0 |
  > | LambdaCDM at its own | 0 | 150932 | 0 | 0 | 0 | 0 / 100 | 0 |
  > | **QCD at its own, 3.30033444460513e+16** | **53** | **136453** | **3.884e-04** | **34** | **19** | **53 / 100** | **7.716e-04** |
  >
  > **The shape matters as much as the total: every affected case guards exactly one node.** So
  > the band reaches *just* past the edge of the region at a single lattice node, rather than
  > running for a stretch through a region of breakdown — which is a different claim from the
  > 69%-of-added-samples figure above, and a narrower one. Both are true and they measure
  > different things: that one counts samples the criterion *added* above horizon crossing, this
  > one counts nodes at which the expansion it differentiates **does not exist at all**.
  >
  > **Still unassigned as a decision, still a candidate for T7.** Prompt 02a was forbidden to
  > touch the band and did not: `residual_node_range` returns exactly what it returned before.
  > `SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05` now caps what the guard may absorb, at 64.8x the
  > worst measured band, so a *materially* worse band refuses rather than being filled silently —
  > but the ceiling is an acceptance bound, not an answer to where the criterion should apply.

  > **CORRECTED 2026-09-17 — the block above misreads its own census, and the correction matters
  > because prompt 04's charter is written from this text.** Re-derived at `effb70b` by replicating
  > `source_grid_spacing_profile`'s `(k, sector)` loop (53 reproduced exactly, so the replication is
  > faithful). **The counts stand; the claim that they measure this issue does not, and is
  > withdrawn.**
  >
  > | claimed above | re-derived |
  > |---|---|
  > | 53 guarded **nodes** | 53 `(k, sector)` evaluations of **one** node, $z = 863613639675.2578$. Distinct guarded redshifts: **1** |
  > | the band reaches **just past the edge** of the region | the node is **interior** — rank **1293** from the band's bottom against band sizes **1296–1567**, i.e. **2 to 273 nodes below the top edge** |
  > | a **direct measure of the overreach** past the horizon | **42 of the 53 cases are sub-horizon there.** $k(1+z)/H$ runs **0.1695 to 71.58**; only **11** are outside the horizon at all |
  >
  > **This issue is about the band reaching 1.5–2.1 e-folds *outside the horizon*, and the census
  > does not measure that.** It measures something else entirely: the node sits **1.000e-03** in $u$
  > from the cosmology's **third declared break** ($z = 864478111114.07$, the redshift the original
  > `ValueError` quoted), which is inside the stencil's 2.0e-03 reach and well inside the 6.0e-03
  > crossing mask that discards it after the loop. That is
  > `[02a-stencil-reaches-across-a-declared-break-before-the-mask]`, opened below, and it is where
  > the census belongs.
  >
  > **So this issue is still without a measurement**, and prompt 04 must not be written believing
  > otherwise. The 69%-of-added-samples figure in the body above remains the only evidence for it,
  > and it is prompt 01's, not 02a's. **Still unassigned, still a candidate for T7.**

  > **MEASURED by prompt 04, 2026-09-17** (additively). This issue has had no measurement of its
  > own since it was opened; it has one now, it is **larger in e-folds** than the body above
  > claims and **much smaller in consequence**, and half of its surface does not exist.
  > `docs/tolerance-convergence/ORDER-AUDIT.md` §8, at all fifty production wavenumbers on all
  > three models in both sectors, version-2 grid at each cosmology's own anchor, counting a band
  > node as outside the region when the sector's **own** leading frequency has
  > $\omega_0(1+z) < 1$:
  >
  > | model / sector | band nodes | super-horizon nodes | fraction, min / median / max | band top, e-folds outside (max) |
  > |---|---|---|---|---|
  > | `RadiationModel` `Gk` | 2,306 | 217–1,137 | 0.094 / 0.213 / 0.493 | **13.01** |
  > | `LambdaCDMModel` `Gk` | 1,778 | 218–579 | 0.123 / 0.220 / 0.326 | **13.01** |
  > | `QCDModel` `Gk` | 1,391–1,884 | 62–197 | 0.040 / 0.062 / 0.108 | **3.66** |
  > | every model, `Tk` | 828–2,034 | **0** | 0 / 0 / 0 | — (always sub-horizon) |
  >
  > **Three things follow.** (i) **The `Tk` sector is not affected at all**, at any wavenumber on
  > any model: $\omega_T^2 > 0$ already requires the mode to be inside the *sound* horizon, so
  > the overreach is entirely a `Gk` phenomenon and the issue's surface is half what it looks.
  > (ii) **The reach is 13 e-folds, not 1.5–2.1**, on the two models whose radiation era is clean:
  > the Green's-function correction $C$ vanishes identically in exact radiation, so the margin
  > test passes at *every* node and the band is the whole grid. The body's 1.5–2.1 was measured on
  > QCD at the four smallest $k$; QCD's maximum over all fifty is 3.66. (iii) **The consequence is
  > 5.6 % of one grid.** Rebuilding the version-2 grid with `main.source_grid_spacing_profile`
  > executed **unmodified** against a band handed to it with the super-horizon nodes dropped:
  > `RadiationModel` 2,306 → **2,306** (`3bef2c06`, unchanged), `LambdaCDMModel` 1,778 → **1,778**
  > (`60a3205a`, unchanged), `QCDModel` 2,034 / `21ffc126` → **1,920 / `85aef41a`, −114 samples**.
  > So on the two cosmologies that declare no break points the overreach costs **nothing at all**,
  > and on QCD it costs 114 samples.
  >
  > **Still unassigned as a decision.** Prompt 04 recommends and does not change the band: README
  > §0.5 holds the source grid fixed and D6's carve-out was prompt 02a's and is spent. What the
  > decision now has that it did not have is a cost — 114 samples on one cosmology — to set
  > against the argument that the criterion should not be equidistributing the fourth derivative
  > of an expansion five e-folds beyond where its consumer exists.

Opened by **prompt 02a's correction**, 2026-09-17:

- **[02a-stencil-reaches-across-a-declared-break-before-the-mask]** *(prompt 02a correction,
  2026-09-17; unassigned — the source grid's owner, `prompts/qcd-background-audit`, or prompt 04
  if it opens the spacing profile anyway)* — **this is what prompt 02a's guarded-node census
  actually measures**, and it is not the band overreach the census was first attributed to.

  `source_grid_spacing_profile` evaluates its five-point stencil at $u \pm \delta$ and
  $u \pm 2\delta$ with `SOURCE_GRID_CURVATURE_STEP_U = 1.0e-3`, a reach of **2.0e-03** in $u$,
  and applies the declared-crossing mask
  `usable &= |u_profile - u_break| > SOURCE_GRID_CROSSING_MASK_U` (**6.0e-03**) **only after that
  loop has run**. So a node inside the mask's gap — one the criterion has already decided to
  discard — is still evaluated, and its arms can land across the declared break, where $H$ steps
  and $\omega^2$ goes non-positive. That is the whole of what was crashing before prompt 02a.

  **Measured at `effb70b`**, QCD at its own anchor, production geometry, version-2 grid: all **53**
  guarded `(k, sector)` evaluations are the **same** base-lattice node, $z = 863613639675.2578$,
  which lies **1.000493e-03** in $u$ from the third declared break at $z = 864478111114.07$ —
  **twice inside** the stencil's reach and **six times inside** the mask. It is interior to the band
  (rank 1293 of 1296–1567; 2 to 273 nodes below the top edge) and **sub-horizon in 42 of the 53
  cases** ($k(1+z)/H$ from 0.1695 to 71.58). The physics is the QCD crossover the break declares:
  `wBackground` falls 0.306035 → 0.288406 → 0.264299 across $z = 6\times10^{11}$, the node, and
  $z = 1.2\times10^{12}$.

  **Impact, and it is mostly favourable.** Because every guarded node is inside the mask, the guard
  changes **no `usable` outcome** and `d4[i]` is only read through `d4[usable]` — so prompt 02a's
  guard cannot alter any profile, only allow one to be computed. That is a *stronger* inertness
  claim than bit-identity on two grids. What is left is two things. (i) The criterion does work it
  has already decided to throw away, which is wasted evaluation and was a hard crash until
  `effb70b`. (ii) **The two published grids guard nothing by lattice alignment, not by
  construction**: QCD's anchor puts a node 1.000e-03 from the break, LambdaCDM's straddles it at
  7.582e-03, outside both the reach and the mask. **A change to `z_init` or `samples_per_log10z`
  could start guarding without anything else changing** — and `z_init` is a root-solve output
  (`[02a-grid-digest-not-reproducible]`), so this is not hypothetical.

  **Next step:** move the mask ahead of the stencil loop, which on this evidence would have
  prevented every one of the 53. **Prompt 02a was right not to do it, but not for the reason its
  log first gave**: reordering changes which nodes are evaluated, hence `usable`, hence the
  log-interpolation fit, hence the profile — so it could move a published digest and **nothing has
  measured whether it does**. Whoever takes it must measure both published grids across the change.
  Note that the *undeclared* steep-step case in `RESIDUAL_WKB_REGION_MARGIN`'s comment
  ($z = 3.61\times10^{15}$) is real, would survive any reordering, and **does not arise at either
  production anchor** — so it is not an argument against reordering, only against calling it a
  complete fix. Evidence: `logs/02a-source-grid-density-guard.md` §7.

Opened by the **orchestrator's review of prompt 02a's charter**, 2026-09-17:

- **[02a-grid-digest-not-reproducible]** *(2026-09-17; assigned — **T6** / prompt 03 for the
  prerequisite, prompt **05** for the fix)* — the source and response grid tags digest the **exact
  bits** of the grid's values (`CosmologyConcepts.redshift.redshift_grid_digest`, `main.py:854`),
  but `z_init` is a root-solve output, so the tag is not reproducible across machines, library
  versions, or any change that forces the anchor to be re-derived. **The binding number is not the
  one the record quotes.** `_solve_horizon_exit` calls `root_scalar(..., xtol=atol, rtol=rtol)` in
  `u = log(1+z)` (`CosmologyConcepts/wavenumber.py:979`) and Brent stops at `xtol + rtol*|u|`; with
  `rtol = DEFAULT_REL_TOLERANCE = 1e-8` at `u ≈ 37.6` that is **3.8e-7 relative**, not the
  `xtol = 1e-10` README §6.2's `wavenumber_exit_time` row and prompt 02's inventory both name. The
  `xtol` term never binds. Measured against a converged re-solve (`xtol=1e-300, rtol=1e-14`, the
  tolerance `_solve_T_z` already uses): the shipped anchor is **3.6e-13** off on LambdaCDM and
  **2.5e-9** off on QCD, and neither is bounded by better than 3.8e-7.
  **Impact.** 3.8e-7 is *coarser* than `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, the tolerance
  at which `Datastore/SQL/ObjectFactories/redshift.py:40` matches an existing redshift row. So the
  two invalidation criteria in the pipeline disagree by six orders of magnitude, and the failure
  mode is silent total cache invalidation rather than corruption: measured across two anchors 3.6e-13
  apart, **all 1778 samples differ bitwise and zero of them exceed 1e-7**, so every redshift row is
  re-matched and reused with its old `store_id` while the grid tag turns over completely and every
  lookup filtered on `SourceZGridSizeTag` misses — in the `Gk` sector, ~65,000 objects per model
  recomputed against rows that were already correct. This is the *inverse* of the collision the
  digest was introduced to prevent (`main.py:837`), so the digest is not wrong; it is quantised
  finer than anything upstream of it is determined to.
  **Not a datastore defect, and not reached by a plain re-run.** `wavenumber_exit_time.build()`
  reads `z_exit_suph_e5` back from a `Float(64)` column when the row exists
  (`Datastore/SQL/ObjectFactories/wavenumber.py:197`, `:292`), so within one datastore lineage the
  anchor is bit-stable and the grid is reproducible — which is the regime every figure in this
  campaign has been taken in, and why nothing has tripped over it. It bites when the anchor is
  *re-derived*: a fresh or dropped store, another machine or libm, or a change of the cosmology row,
  which is what `qcd-background-audit` did twice.
  **Next step, in two parts.** (i) **T6 / prompt 03a** (prompt 03's, until §7 **D7** split the
  charter on 2026-09-17) measures `wavenumber_exit_time`'s pair and recommends the tightening; at `rtol = 1e-14` the anchor is pinned to 3.8e-13, measured. (ii)
  **Prompt 05** defines one design tolerance and applies it to *both* the redshift row match and the
  digest quantisation, so that the tag can never distinguish two grids the datastore cannot. With
  the anchor tightened the wobble budget is 3.8e-13 (anchor), ~1e-14 (break redshifts, already at
  Brent's floor), 3.6e-15 (the `u → z` recovery) and ~1e-15 (libm `pow`/`log10`/`log1p`), so a
  design tolerance around **1e-11** sits two orders above the worst contributor. Carry the coupling
  with it: `SOURCE_GRID_MIN_SEPARATION = 10.0 * DEFAULT_REDSHIFT_RELATIVE_PRECISION`
  (`CosmologyConcepts/wavenumber.py:135`) only *relaxes*, but the four-constraint argument above
  `SOURCE_GRID_BREAK_STANDOFF` (`:79`) loses its first bullet and needs rewriting, not just
  retuning. **Rounding buys a margin, not a proof** — straddle probability ~ `N * wobble / quantum`,
  ~2e-4 per grid at these numbers; the exact alternative is to digest the *determining data*
  (snapped `z_init`, `z_end`, `samples_per_log10z`, construction version, snapped break and feature
  lists, and the integer subdivision vector, whose ties are a measured 6e-3 clear), which prompt 05
  should record as considered even if it ships the quantised-values version.
  **Explicitly not prompt 02a's** (README §4, §7 D6): 02a's acceptance is bit-identity with the
  published digests, and tightening the anchor would move both.

  > **Re-pointed 2026-09-18 by prompt 05** (additively). Part (ii) above, and README §4's sentence
  > that names prompt 05, were written before §7 **D9** split that prompt. Defining one design
  > tolerance and applying it to the redshift row match and the digest quantisation is a
  > *parameter* decision — it moves `DEFAULT_REDSHIFT_RELATIVE_PRECISION` or a new constant beside
  > it — and §5 rule 8 as D9 amended it puts every parameter move in **05a**, prompt 05 being the
  > schema half and `config/defaults.py` byte-identical at its end. **So part (ii) is prompt
  > 05a's.** Nothing else in this entry changes, and prompt 05 measured nothing against it. The
  > option it was asked to "record as considered" — digesting the determining data — is already
  > narrowed by `[03a-qcd-v2-grid-sample-count-is-not-reproducible]`, which showed the integer
  > subdivision vector itself moves on QCD.

  > **Part (i) delivered by prompt 03a, 2026-09-17** (additively). Measured through
  > `_solve_horizon_exit` at 50 $k$ × 3 models × 3 offsets × 63 `(xtol, rtol)` cells
  > (`docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §7). **Three things this entry did not
  > have.**
  >
  > 1. **Its two anchor figures are confirmed independently, as a swept target rather than a review
  >    probe.** At $k = 3\times10^8$ and offset −5 — the anchor itself — the production setting's
  >    displacement from a converged re-solve is **3.62e-13** on LambdaCDM and **2.50e-09** on QCD,
  >    which are this entry's 3.6e-13 and 2.5e-9. The `xtol` term binds at **0 of 150** (k, offset)
  >    pairs on every model, so "the `xtol` term never binds" is now measured rather than inferred.
  > 2. **The guarantee and the achieved displacement disagree about the 1e-7 row match, and both
  >    matter.** 3.8e-7 is the *bound*; the worst *achieved* displacement anywhere in the production
  >    range is **7.86e-08**, which clears 1e-7 by 1.3×. Prompt 03a recommends `rtol = 1e-9` — the
  >    loosest whose bound clears — on the ground that a location fixing a datastore key should be
  >    bounded rather than lucky, and records that the other reading gives `unchanged`.
  >    **A caution for part (ii):** with `xtol = 1e-10` the pair cannot pin the anchor better than
  >    **1e-10** relative however far `rtol` is tightened — `xtol` takes over below
  >    `rtol ≈ 2.6e-12` — so a design tolerance of 1e-11 requires `xtol` to move too, and the
  >    "`rtol = 1e-14` pins the anchor to 3.8e-13" line above holds only at `xtol ≤ 1e-12`.
  > 3. **The exact alternative this entry offers prompt 05 does not survive on QCD.** Digesting the
  >    determining data relies on "the integer subdivision vector, whose ties are a measured 6e-3
  >    clear". Rebuilt at perturbed anchors, the version-2 grid's **sample count** on `QCDModel`
  >    moves from 2034 to between 2013 and 2032 at relative shifts from **1e-14** upwards, while
  >    LambdaCDM's stays at 1778 and Radiation's at 2306 throughout.
  >    `[03a-qcd-v2-grid-sample-count-is-not-reproducible]` holds that measurement.

Opened by **prompt 03**, 2026-09-17:

- **[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]** *(prompt 03,
  2026-09-17; **assigned 2026-09-17 — the hand-over campaign**, `docs/OPEN_ISSUES.md` §1.1)* —
  the consumer of `GkNumericIntegration` is the
  `numeric_Gk` spline of `ComputeTargets/GkSourcePolicyData.py:654-680`: a cubic
  `make_interp_spline` in $\log(1+z_{\rm source})$ over the **source-grid** nodes that carry
  numeric data, at fixed $z_{\rm response}$. Measured at all fifty production wavenumbers on all
  three models, **source-grid generation version 2 at each cosmology's own anchor**, against a
  direct solve at the midpoint of each interval:

  | rung ($z_{\rm source}$, e-folds inside the horizon) | +4 | +3 | +2 | +1 | 0 | −2 |
  |---|---|---|---|---|---|---|
  | RadiationModel, max over $k$ | 4.57e-04 | 1.64e-04 | 3.39e-06 | 3.08e-08 | 4.93e-08 | 2.24e-08 |
  | LambdaCDMModel, max over $k$ | **9.35e-03** | 1.73e-04 | 3.63e-06 | 3.26e-08 | 1.01e-07 | 3.01e-08 |
  | QCDModel, max over $k$ | **7.79e-03** | 1.86e-04 | 3.61e-06 | 2.20e-06 | 1.96e-07 | 3.90e-08 |

  Floor uncertainty **0.00 %** under a further decade of solver tolerance; on the radiation control
  the probe that supplies the spline's nodes is within **1.0e-12 to 5.6e-12** of the exact
  `compute_analytic_G`, of the same envelope. **Impact.** Near the hand-over the numeric Green's
  function is delivered to `QuadSourceIntegral` with **1e-04 to 9e-03** of its envelope of
  *interpolation* error — between ×631 and ×37,700 the solver error at the production
  `(1e-10, 1e-8)`, and larger than every floor in README §2 (f) except the WKB LG truncation. The
  cause is spacing, not tolerance: the source lattice carries up to **1.24 rad** of $G$'s
  oscillation in $z_{\rm source}$ per interval at the bottom of the numeric region, and a cubic
  interpolant's error over such an interval is $(h\,\mathrm{d}\theta/\mathrm{d}u)^4/384$. **It
  confirms review §10.1's inherited "1e-5 to 1e-4 of the value near the hand-over" and shows it
  understates the dominance.** The density that sets it is the version-2 criterion's, which is
  sized for the phase-residual spline (`SOURCE_GRID_CONSUMER_TARGET_RAD`) and not for $G$.
  **Assigned (2026-09-17): the hand-over campaign** (`docs/OPEN_ISSUES.md` §1.1), by the user's
  decision on accepting **D1** — the source grid's spacing at the hand-over is to be looked at
  **after this campaign**, not inside it. It goes there rather than to the source grid's owner
  because every lever below is a seam decision, and §1.1's entries are the ones that "must be
  attacked together": it sits directly beside `[05-numeric-region-is-now-the-accuracy-floor]`,
  which is the same spline's *end* effect rather than its interior spacing.
  **Next step:** a design decision rather than a tuning one — either the
  density criterion gains $G$'s own oscillation as a second consumer, or the hand-over moves
  deeper, or `GkSourcePolicyData` stops splining $G$ itself and splines its Liouville–Green
  amplitude and phase as the WKB limb already does. All three are outside this campaign
  (README §0.5 holds the source grid fixed, and §0.4 puts the consumer's own sector elsewhere); it
  sits beside `[05-numeric-region-is-now-the-accuracy-floor]`, which is the same spline's *end*
  effect rather than its interior spacing. Evidence:
  `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` §7, §8.

- **[03-outermost-z-source-is-not-the-least-favourable]** *(prompt 03, 2026-09-17; **unassigned —
  prompt 03 §2.2's stop condition, the user decides**)* — `gk_geometry`
  (`ComputeTargets/tests/convergence_reference.py`) takes one source redshift per wavenumber, the
  outermost, "which is the longest and therefore the least favourable run", and prompt 03 §2.2
  makes that claim the premise on which fifty runs per model *bound* a sector of ~29,000–58,000
  objects. Measured at three wavenumbers per model at seven source redshifts from five e-folds
  outside the horizon to three inside, at the production setting, each against its own converged
  reference (drift 8.3e-12 to 5.1e-10): **the outermost is not the worst at any of the nine
  probes**, the excess running from ×1.01 to ×1.45. The maximum envelope-relative error is **flat**
  in $z_{\rm source}$ — 1.09e-07 to 2.39e-07 across the whole table, with no trend — while the
  *median* rises monotonically with e-folds inside the horizon and the evaluation count falls.
  **Impact:** every figure in `GK-NUMERIC-SWEEP.md` characterises the sector to about a factor of
  1.5 and none of them bounds it; README §6.2's `GkNumericIntegration` row must say "over fifty
  wavenumbers at the outermost source redshift" rather than "over the sector". Nothing this prompt
  concludes turns on a factor of 1.5 — the floor above dominates the solver by two to four and a
  half orders — so the failure is of the word "bound", not of the sweep. **Next step: the user
  decides.** Either a bound over the $(k, z_{\rm source})$ plane is wanted, which is a much larger
  measurement and a different prompt, or the characterisation is enough and §2.2's wording and
  `gk_geometry`'s docstring should say "representative". Prompt 03 was forbidden to widen the sweep
  to compensate (§2.2) and did not. Evidence: `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` §6;
  log 03, deviation 8.

Opened by **prompt 03a**, 2026-09-17:

- **[03a-tk-numeric-excursion-is-sporadic-in-rtol]** *(prompt 03a, 2026-09-17; **open — the user
  accepted `rtol = 3e-11` on 2026-09-17 with this caveat standing, not waived**, so what ships is
  the loosest setting that clears the floor in that sweep and prompt 05's provenance note must say
  so; unassigned)* — the maximum
  envelope-relative error of `TkNumericIntegration` over the production grid is **not a decreasing
  function of `rtol`**. Measured at 50 $k$ × 3 models × nine `rtol` settings, **version-2 grid at
  each cosmology's own anchor**, `BREAK_POINT_ALL`, `atol = 1e-13`, each figure against its own
  converged reference (drift ≤4.26e-11 / 5.23e-11 / 4.65e-09):

  | rtol | 1e-8 | 3e-9 | 1e-9 | 3e-10 | 1e-10 | 3e-11 | 1e-11 |
  |---|---|---|---|---|---|---|---|
  | worst max, three models | 3.36e-04 | 6.60e-04 | 9.08e-04 | 5.76e-04 | 1.31e-04 | **3.88e-08** | 5.25e-08 |
  | $k$ above 3e-6, of 150 | 14 | 4 | 1 | 1 | 1 | 0 | 0 |

  From `1e-9` down to `1e-10` **exactly one** of the 150 runs sits above target at each setting, and
  it is a **different** run each time — $k = 1.894\times10^6$ on QCD at `1e-9`,
  $4.972\times10^7$ on QCD at `3e-10`, $5.943\times10^6$ on LambdaCDM at `1e-10`. It is the
  phenomenon `GkTk-remedial` prompt 17 §7 (b) demonstrated by perturbing $k$ by one part in $10^6$:
  an occasional step sequence mis-resolves the first oscillations as the mode enters the horizon and
  carries the error to the end of the run. The same measurement shows `atol` selecting *which*
  wavenumber draws it — two decades of `atol` move the median of the per-$k$ maxima by ≤2.1× and the
  maximum by up to **205×**. **Impact:** README §6.1's rule selects `rtol = 3e-11` as the loosest
  setting that clears the 2.39e-06 floor, and that selection is an observation over 150 runs rather
  than a bound over the sector: nothing measured here shows the excursion cannot appear at `3e-11`
  at a wavenumber, a model or a background not in the sweep. Fifty wavenumbers per model **are** the
  whole sector — unlike the $G_k$ sector there is no second axis — so the gap is not coverage but
  the nature of the phenomenon. **Next step:** the user takes it with D1, knowing that the
  recommendation buys a factor of 62 of headroom rather than a proof. If a bound is wanted the work
  is a different measurement — the excursion's *rate* against `rtol`, over perturbed $k$ at fixed
  setting, which is prompt 17 §7 (b)'s diagnostic run as a statistic rather than as an illustration.
  Evidence: `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §4, §4.1; log 03a, deviation 1.

- **[03a-qcd-v2-grid-sample-count-is-not-reproducible]** *(prompt 03a, 2026-09-17; **unassigned —
  candidate for prompt 05, which is the prompt `[02a-grid-digest-not-reproducible]` part (ii) is
  assigned to**)* — rebuilding the **version-2** source grid at a perturbed anchor changes the
  **number of samples** on `QCD_Cosmology`, at perturbations far below anything the anchor solve can
  resolve. Measured by rebuilding at eight relative shifts of $z_{\rm init}$, production geometry,
  fifty wavenumbers:

  | relative anchor shift | 1e-16 | 1e-14 | 1e-12 | 1e-10 | 1e-8 | 1e-7 | 3.8e-7 | 1e-6 |
  |---|---|---|---|---|---|---|---|---|
  | QCDModel samples | 2034 | **2032** | **2016** | **2022** | **2013** | **2029** | **2018** | **2025** |
  | LambdaCDMModel | 1778 | 1778 | 1778 | 1778 | 1778 | 1778 | 1778 | 1778 |
  | RadiationModel | 2306 | 2306 | 2306 | 2306 | 2306 | 2306 | 2306 | 2306 |

  (`1e-16` is the control: `1 + 1e-16` rounds to 1, so that column is the unperturbed build and
  confirms the rebuild is deterministic.) The base lattice cannot be the cause — its count is
  `round(100 (log10 z_init − log10 z_end) + 0.5)`, which a 1e-14 shift cannot move — so what moves is
  the **integer subdivision vector** the density criterion returns, and it moves on the one
  production cosmology that declares break points. On the two that do not, the count is fixed and
  only the bits move (5 of 1778 rows still bitwise identical at 1e-12, 1 at 1e-10). **Impact, and it
  is prompt 05's.** `[02a-grid-digest-not-reproducible]` offers two ways to make the grid tag
  reproducible: quantise the values, or digest the *determining data* including "the integer
  subdivision vector, whose ties are a measured 6e-3 clear". **The second is not available on QCD**
  — the vector is itself undetermined at the anchor's precision, and 1e-14 is six orders below the
  displacement the production solve achieves (7.86e-08). It also means a QCD grid rebuilt on another
  machine may hold a *different number of redshift rows*, not merely different bits, so the
  row-reuse argument that keeps the datastore usable across a tag change does not carry over
  unexamined. **Next step:** prompt 05, when it defines the design tolerance, measures whether
  quantising the base lattice before the criterion runs stabilises the vector; or the source grid's
  owner (`prompts/qcd-background-audit`) takes the criterion's tie-breaking. This prompt measured it
  and stopped — the source grid is held fixed by README §0.5 and `main.py` is outside prompt 03a's
  file list. Evidence: `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §7.5.

- **[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve]** *(prompt 03a, 2026-09-17;
  **unassigned — small, and the facility's owner is prompt 01's charter, which is closed**)* —
  `ComputeTargets/tests/convergence_reference.SCIPY_RTOL_FLOOR = 2.220446049250313e-14` is
  `100 * eps`, the clamp `scipy.integrate`'s Runge–Kutta solvers apply
  (`scipy/integrate/_ivp/common.py`), and `TolerancePair.rtol_step_is_effective` and every
  `DriftVerdict`'s notes apply it to **any** `TolerancePair`. Prompt 03a drives
  `_solve_horizon_exit` through the same facility, as board standing note 14 requires, and that is
  a `scipy.optimize.brentq` whose floor is **`4 * eps = 8.88e-16`** — it raises below that and
  honours everything above it. So a root-solve pair at `rtol = 1e-15` is reported as "silently
  ignored" when it is not. **Impact:** cosmetic today and misleading tomorrow. Prompt 03a's
  exit-time reference tightening is `(1e-300, 1e-15)` and draws the spurious note, which the
  document records rather than suppresses; the measured displacement at that setting is 7.11e-15,
  one ulp of $u$, which is the proof the step *was* applied. The hazard is a later prompt taking
  the note at face value and stopping its root-solve axis a decade early. **Next step:** the floor
  belongs to the method, not to the pair — either `TolerancePair` gains the consumer's floor as a
  field, or `rtol_step_is_effective` takes it as an argument. Either is additive; neither is prompt
  03a's, whose licence over that file was additive-only and which needed nothing. Evidence:
  `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §7.2; log 03a, observation 1.

Opened by **prompt 04b**, 2026-09-18:

- **[04b-break-point-print-string-describes-a-superseded-block]** *(prompt 04b, 2026-09-18;
  unassigned — one f-string, for whoever next edits that module)* —
  `ComputeTargets/tests/test_background_tau.test_qcd_break_points` prints *"the convergence block
  still records `{n}` knots, from before prompt 06"*. With the block regenerated, `n` is the
  **current** interior-knot count of the $T(z)$ tabulation in range (2,411, measured in the same
  test and agreeing exactly), so the number is right and the characterisation "from before prompt
  06" is wrong. **Impact: none on any verdict** — nothing asserts on it, and the count the test
  *asserts* (zero declared knots) is unaffected. It is recorded because this commit is what made
  the string stale, and because a reader debugging that test would be told the block is older than
  it is. **Next step:** delete the clause, in any prompt given that module for any reason. Prompt
  04b's grant there was `QCD_NODE_REL_TOL` and `QCD_BREAK_POINT_ALIGNMENT_TOL` and it did not
  extend to a `print`. Evidence: log 04b, observation 2.

Opened by **prompt 04**, 2026-09-17, and **closed by prompt 04b on 2026-09-18** —
`[04-convergence-floor-used-as-a-test-threshold]`'s entry is in **§4**, which is the record,
with the arithmetic that chose the repair and the reasoning that rejected raising the factor.

Opened by **prompt 02**, 2026-09-16:

- **[02-oneloopintegral-is-a-ninth-keyed-object-type]** *(prompt 02, 2026-09-16; **unassigned —
  prompt 02 §8's stop condition, the user decides where it goes**)* — README §2 (a) counts eight
  object types keyed on an accuracy parameter. There are **nine**.
  `Datastore/SQL/ObjectFactories/OneLoopIntegral.py:101-116` declares `atol_serial` and
  `rtol_serial` as indexed, non-nullable foreign keys into `tolerance`, exactly as the other eight
  do, and `build()` filters on both at `:153-154`. The table is registered
  (`Datastore/SQL/Datastore.py:120`), sharded on `k` (`config/sharding.py:34`), given a
  client-pool budget (`ClientPool.py:43`) and given a drop action (`Datastore.py:130`).
  **`main.py` never builds one** — its only `OneLoop` strings are two `store_tag` labels at
  `:1021-1022` — and `ComputeTargets/OneLoopIntegral.py`'s `compute()` (`:94-112`) does nothing
  but replace a label, so the object count of the sector is **0**. Two defects inside that stub,
  folded in here rather than opened separately: `:105-107` raises "value haa already been
  computed" when `self._value is **None**` (inverted condition, and the typo), and `store()`'s
  first `raise` at `:116` is followed by an unreachable comment. **Impact:** small today and
  structural tomorrow. Nothing is stored, so nothing is wrong in the datastore; but the campaign's
  count of its own subject is wrong by one for the second time, prompt 05's `ast` guard must
  enumerate **nine** targets rather than eight if this one is in scope, and a target whose schema
  is already keyed on the shared pair will inherit whatever prompt 05 decides **unless someone
  decides otherwise on purpose**. **Next step: the user decides.** It is not obviously prompt 03's
  (no solver to sweep) nor prompt 04's (no order); it may be the cheap moment for prompt 05,
  before any row exists to invalidate, or it may be premature to decouple a target that computes
  nothing. Prompt 02 measured it and stopped there, as §8 requires. Evidence:
  `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.1.

- **[02-wavenumber-exit-time-tolerance-is-an-inequality-key]** *(prompt 02, 2026-09-16; candidate
  for prompts 03 and 05)* — README §2 (g) says "every accuracy parameter is part of its object's
  lookup key, so a new one makes every existing row of that type unreachable". True of eight of the
  nine; **false of `wavenumber_exit_time`**. Its query
  (`Datastore/SQL/ObjectFactories/wavenumber.py:242-254`) joins the `tolerance` table twice and
  filters `stored.log10_tol - requested.log10_tol <= DEFAULT_FLOAT_PRECISION` on each — accept any
  row **at least as tight as** the request — then orders by `log10_tol` **descending** and takes
  `one_or_none()`, i.e. the **loosest** qualifying row. It does the same with
  `stepping >= target_stepping`. **Impact:** tightening misses and recomputes, as an equality key
  would; **loosening hits a tighter stored row and returns its `z_exit`**, and the object's own
  `atol`/`rtol` properties (`CosmologyConcepts/wavenumber.py:729-734`) then report the *stored*
  pair, not the requested one. A tolerance sweep run through the datastore would therefore be
  served the same answer at several of its points and would measure nothing.
  `MultipleResultsFound` is caught at `:260`, so two qualifying rows raise rather than choosing.
  This is deliberate "best available" behaviour and is **not proposed for change here**.
  **Next step:** prompt 03 sweeps this target by calling
  `CosmologyConcepts.wavenumber._solve_horizon_exit` directly and never through `object_get`;
  prompt 05's `ast` guard must not assume equality semantics at this site. Evidence:
  `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.3.

- **[02-shared-atol-doubles-as-a-float-comparison-epsilon]** *(prompt 02, 2026-09-16; candidate for
  prompt 05)* — `DEFAULT_ABS_TOLERANCE` is not only the shared solver tolerance. It is also a bare
  float-comparison epsilon at **seven** sites with no connection to the Green's-function ODE:
  `ComputeTargets/GkSource.py:96`, `:104`, `:275`;
  `Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, `:790`;
  `LiouvilleGreen/WKBtools.py:83`. All seven are `fabs(a - b) < DEFAULT_ABS_TOLERANCE`-shaped
  guards — a redshift-equality check, a residual check, a phase-modulo check. **Impact:** prompt 05
  is chartered to retune or split this constant, and the moment it does, all seven comparison
  thresholds move with it, silently and in modules that are not in that prompt's file list. At
  `1e-10` none of them is near its margin, so nothing is wrong today; the hazard is entirely in the
  change. **Next step:** prompt 05 gives these seven sites a constant of their own — or states, in
  its log, that it has checked each one against the new value. Evidence:
  `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §4, closing paragraph.

- **[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]** *(prompt 02, 2026-09-16;
  unassigned; prompt 05 touches the same six files)* — `extract_TkWKB_data.py:433-445` builds one
  `query_payload` with `"atol": atol` where `atol = DEFAULT_ABS_TOLERANCE` (`:364`) and uses it for
  **both** `TkNumericIntegration` and `TkWKBIntegration`. `main.py` writes every
  `TkNumericIntegration` under `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` — its own comment at
  `main.py:3475-3479` says "every `TkNumericIntegration` `object_get` — the work items and **every
  lookup** — must use it" — and that target's lookup filters `atol_serial ==`. **So this query
  cannot match a production row.** None of the six `extract_*.py` readers imports
  `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` at all. **Impact:** an extraction script that has been unable
  to plot the numeric limb of $T_k$ since `GkTk-remedial` prompt 12 shipped the split constant.
  Not verified by running it — that needs a datastore, which prompt 02 may not stand up — so what
  is established is that the key cannot match, not what the script does next. **Next step:** the
  one-line fix is to import the constant and pass it for the numeric target only; prompt 05 already
  has to revisit all six readers when it splits the constants further, and this is the same edit.
  Evidence: `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §5.1 (the derived
  `atol_serial ==` predicate) and log 02, observation 1.

  > **Re-pointed 2026-09-18 by prompt 05** (additively). Prompt 05 did revisit all six readers and
  > **left this**. It split `extract_TkWKB_data.py`'s shared `query_payload`, because
  > `TkWKBIntegration` no longer takes a tolerance and `TkNumericIntegration` still does — so the
  > numeric lookup now passes `atol=`/`rtol=` explicitly and the defect is in one place instead of
  > being hidden in a dict serving two targets. What it did not do is change *which* constant that
  > lookup names: that is a decision about `TkNumericIntegration`'s tolerance, a target prompt 05
  > §8 forbids, and `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` is one of the constants **05a** ships.
  > **Assigned to prompt 05a**, which touches `config/defaults.py` and the readers' constants
  > anyway. Evidence: log 05, observation 1.

Assigned to this campaign from other boards (each stays on the board that holds its measurements;
the closure is recorded there):

| Issue | Owning board | Assigned to | Why here |
|---|---|---|---|
| `[12-tk-numeric-atol-largest-k-excursion]` | GkTk-remedial | prompts 03, 05 | Assigned 2026-09-12. Its `atol` half is settled — the user kept `1e-13` — and what remains is the `rtol` retuning, which is D1. It closes when prompt 05 ships a settled `rtol`. **Re-taken by prompt 03a, 2026-09-17**, on the version-2 grid under `BREAK_POINT_ALL`: its 3 / 13 / 8 wavenumbers above 3e-6 become **1 / 9 / 4** and its worst 8.64e-04 becomes **3.36e-04**, most of that from the grid rather than the policy; `rtol = 3e-11` takes the count to 0 / 150 and the worst to 3.88e-08. **Its cost figures in `docs/OPEN_ISSUES.md` §1.5 were corrected at the rebase** (`RECONCILIATION.md` §2.4): the QCD $T_k$ object is 8,986 right-hand-side evaluations, not ~31.5k |
| `[01-convergence-block-has-a-separate-generator]` | qcd-background-audit | prompts 04 and **04b** — **CLOSED 2026-09-18** (§4) | **Assigned 2026-09-16.** The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` records $N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$, was generated 2026-09-10, and its `decision.recommended_scheme` is `"branch+knots"` — a knot set `qcd-background-audit` prompt 07 removed. Prompts 08 and 09 of that campaign each declined it on scope. Prompt 04 here is the first prompt anywhere whose charter is the orders themselves, so it cannot avoid re-running the generator; that it must then write a fixture and edit `test_background_tau.py` is README §7 **D5**, **settled yes at the 2026-09-16 re-anchor**. **Prompt 04 ran the generator on 2026-09-17 and did not write the block**: the regenerated reference floor is 58x ($\tau$) and 43x ($c_s\tau$) tighter, and the two tests that read it as a threshold then fail by 6.99x and 5.08x with their numerators unmoved — one of them in a module outside the D5 carve-out. So this issue has moved from *blocked on scope* to *blocked on a decision*, and the decision is `[04-convergence-floor-used-as-a-test-threshold]`. Everything else it asks for is measured: all four orders are `unchanged` at 4, the recommended scheme becomes `branch`, and `QCD_BREAK_POINT_ALIGNMENT_TOL` would go to **1.421085e-14** |
| `[20-wkb-gauss-orders-not-in-lookup-key]` | GkTk-remedial | prompts 04, 05 | **Assigned 2026-09-16.** `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER`, `RHO_GAUSS_ORDER` and `RESIDUAL_WKB_REGION_MARGIN` are configuration axes in no lookup key, while the `atol`/`rtol` columns that *are* in the key describe nothing. That is README §7 **D3**, and D3 is the user's stated target for the campaign: for a Liouville–Green-type representation the key should carry an order. **Prompt 04 reported on 2026-09-17**: the audit is complete (**T7**) and the D3 recommendation is **`replace with the orders`** — three integer columns on `BackgroundModel` ($N_\tau$, $N_{c_s\tau}$, $N_F$), one on each WKB target ($N_\rho$), and `drop` for `GkSource`, which integrates nothing. `RESIDUAL_WKB_REGION_MARGIN` is measured for the first time and is `unchanged`. **D3 settled 2026-09-18 and prompt 05 was written the same day**, carrying this row alone: §7 **D9** split the old §3.5 into a schema half (05, board **T9**) and a tolerance half (05a, **T8** and **T10**). The margin does **not** become a column and the reason is prompt 04's own measurement — the stored residual is bit-identical from margin 0.05 to 0.9, so it cannot separate two rows, which is the only thing a key column is for. Reader breakage is **six** `extract_*.py` scripts, not D3's three. **Closed by prompt 05 on 2026-09-18** — both halves; the entry with the evidence is in §4 above, and `GkTk-remedial`'s own board carries the closure note |

Recorded by the rebase, **not owned here** and not scheduled (README §0.5):

- `[11-stop-point-root-tolerance]` (hand-over campaign, `docs/OPEN_ISSUES.md` §1.1) —
  `find_phase_extremum`'s `root_scalar(xtol=1e-6, rtol=1e-4)`,
  `LiouvilleGreen/integration_tools.py:95`. It appears in prompt 02's inventory and in
  `docs/TOLERANCE-PROVENANCE.md`, and it is not retuned here.
- `LambdaCDM_GenericEOS.py:1008` — **settled 2026-09-16 by `prompts/background-solver-robustness`,
  which is "whoever owns that file". It is no longer an unbracketed secant and no longer at
  `:1008`.** It is `_find_rho_equality`'s `root_scalar` at **`:1137`** (`def` at `:1001`), Brent on
  a $\sqrt2$ bracket expanded in $1+z$ about the caller's guess and clamped to the $T(z)$
  representation's own bounds, at **`xtol=1e-300, rtol=8.9e-16`** — Brent's own $4\varepsilon$
  floor, decided by the user on that campaign's README §7 **D1** as amended, because `rtol=1e-14`
  was measured to stop 7 ulp from an independent reference. **Its provenance is
  [`prompts/background-solver-robustness/PROVENANCE.md`](../background-solver-robustness/PROVENANCE.md)
  §3, which is written in the shape `docs/TOLERANCE-PROVENANCE.md` will want: prompt 02 here should
  lift that entry rather than re-derive it.** Two further things prompt 02's inventory needs and
  the old bullet could not have known: the *quantity* is **not** a diagnostic — since that
  campaign's prompt 09 the model answers for its own equality redshifts and they are production
  source-grid sample locations inside a `BackgroundModel` lookup key (PROVENANCE.md §3.1) — and the
  same document's §1 and §2 settle the file's **other two** solves, so all three of `:579`, `:864`
  and `:1008` in README §3.2's list arrive here already established.
- `ComputeTargets/QuadSourceIntegral.py:1550` still says "the pipeline supplies
  `DEFAULT_QUADRATURE_ATOL = 1e-25`"; the constant has been 1e-32 since `source-remediation`
  prompt 12. A stale comment in a file README §0.4 puts out of bounds.

---

## 4. Resolved issues

Closed by **prompt 05b**, 2026-09-18:

- **[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]** (opened by prompt
  05, 2026-09-18; widened the same day) — **CLOSED, both halves**, including the rehydration half
  the title does not name.

  **An object now reports the order it was built at, on every path by which it can come into
  existence.** On the compute path the order travels as data: `WKB_phase_function` reads
  `CumulativeTable.order` off the residual table it actually used, into `metadata["N_rho"]` and
  into a payload key of its own, and both WKB `store()` methods record it;
  `BackgroundModel.store()` records the three orders `compute_background` already echoed from its
  three table constructions. On the rehydration path each factory's `build()` **selects** its
  order column(s) and passes them to the constructor — `BackgroundModel`'s query did not select
  its three at all, which is why the row's orders could not have reached it — and
  `_build_tau_primitive`, `_build_cs_tau_primitive` and `_build_friction_F_primitive` reassemble
  their tables at the row's orders rather than the module's. An object that has computed nothing
  and came from no row refuses rather than answering with the constant.

  **The keyword half closes as a call-time sentinel rather than as a required argument.**
  `build_phase_residual`, `cached_phase_residual` and `phase_residual_cache_key` take
  `order: Optional[int] = None` and resolve `RHO_GAUSS_ORDER` in the body. That removes the
  def-time snapshot the issue names — under which re-pointing the declaration moved the key column
  while the table went on being built at the old order — without editing the five doc-script call
  sites across three campaigns that prompt §4 protects, and without deleting the parameter
  `order_audit.py` sweeps. The "make it required" next step this entry proposed is therefore
  **not** what was done, and the log says why: required still leaves the divergence expressible,
  while reading the record off the table does not.

  **`build()` still filters on the current module constants**, unweakened and asserted: a row
  computed at another order is a different row rather than a miss to repair (§7 **D10**).

  **Nothing moved.** No schema (`register()` untouched in all six factories, `GkSource` included,
  `break_point_kind` included), no number (`config/defaults.py` byte-identical, all four orders
  still 4), no `main.py` line, and **no computed value**: a SHA-256 over 2,296 doubles produced by
  `compute_background` and by both WKB sectors' phase and residual tables is identical at
  `90d0114` and after. Both of the prompt's §6 tests were confirmed to fail at `90d0114` with the
  messages log 05b §4 quotes. `ComputeTargets` **516**, `CosmologyModels` **39**, both OK.
  Evidence: log 05b; board item **T15**.

Closed by **prompt 05**, 2026-09-18:

- **[20-wkb-gauss-orders-not-in-lookup-key]** (opened by `GkTk-remedial` prompt 20, 2026-09-13;
  assigned here 2026-09-16) — **CLOSED, both halves.**

  **The four orders are in a key.** `BackgroundModel` carries `tau_gauss_order`,
  `cs_tau_gauss_order` and `friction_F_gauss_order`; `GkWKBIntegration` and `TkWKBIntegration`
  carry `rho_gauss_order`; `GkSource` carries neither those nor the tolerance pair, integrating
  nothing. The pair is gone from all four. That is README §7 **D3** exactly as the user settled it
  on 2026-09-18, implemented and not extended.

  **And the key is a key.** The defect this issue names is not that the order was missing from the
  schema but that it was missing from the *comparison*: `TOLERANCE-INVENTORY.md` §2.4 measured that
  the solver table is joined so the label can be read back and then never filtered on, and a new
  column keyed the same way would have reproduced it one identifier later. Each order is an
  equality criterion of its factory's `build()`, read back off the compiled `whereclause` with no
  database behind it (`ComputeTargets/tests/test_gauss_order_key.py`, 16 tests, on
  `test_numeric_break_point_key.py`'s pattern). Repointing a declaration moves the query, the
  stored value and the compute class's accessor together, and `test_no_order_is_written_as_a_literal`
  fails any site that inlined the number.

  **`RESIDUAL_WKB_REGION_MARGIN` is the fifth axis this issue names, and it closes as `not a key
  column` on measured evidence rather than being dropped.** Verified by prompt 05 against
  `ORDER-AUDIT.md` §7.2 rather than taken from its own prompt: the `delta` a producer reads between
  two fixed redshifts is **bit-identical** to production's at every margin from 0.05 to 0.9, on
  three models and both sectors at five wavenumbers, while the band's node count moves by hundreds.
  A key column exists to separate rows that differ and this one separates none. Two qualifications
  the prompt's own sentence did not carry, recorded because they are what a re-reader will find in
  §7.2: at `margin >= 0.99` every `Tk` row reads `anchor clamped` — the band stops reaching the
  production anchor, which is a reachability failure and not a differing row — and at
  `margin = 0.9999` two QCD `Gk` cells depart by 2e-16, one ulp, from the rounding of a cumulative
  whose top node has moved. Both are four decades outside anything the campaign contemplates and
  neither changes the verdict. Where the *density criterion* should apply remains
  `[01-density-criterion-imposed-outside-the-wkb-region]`, which is a different question and stays
  open.

  **No number moved.** `config/defaults.py` is byte-identical and all four orders are still 4;
  prompt 04 measured every one `unchanged` (§6.1 rule 4) and prompt 04b landed that evidence. The
  bill is the one D2 accepted: every row of the four targets in an existing store is unreachable by
  its old key. Evidence: log 05; board item **T9**.

Closed by **prompt 04b**, 2026-09-18:

- **[04-convergence-floor-used-as-a-test-threshold]** — **CLOSED.** Neither assertion divides the
  model's accuracy by the reference's any more. `test_background_tau.py` bounds the production
  order-4 $\tau$ table at **`QCD_NODE_REL_TOL = 1.0e-14`** and
  `test_background_cs_tau_friction.py` bounds $c_s\tau$ at **`QCD_CS_TAU_REL_TOL = 1.0e-14`**, each
  an absolute bound on the quantity it names, with the measured floor under it and the arithmetic
  in the module. `QCD_FLOOR_FACTOR` is **removed** from both; no reference to the name survives.

  **The arithmetic.** The user chose this over the cheap repair — raising both `QCD_FLOOR_FACTOR`s
  past 7 — because that keeps a comparison between two independent quantities alive with a fresher
  number in it and needs raising again the next time the reference improves (README §7 **D8**).
  The quantities
  are **2.254220593813745e-15** ($\tau$) and **2.2119104448437696e-15** ($c_s\tau$), both at
  $z = 1.005\times10^7$ — prompt 04's figures and `qcd-background-audit` prompt 06's, to every
  digit. The floor under each is double-precision accumulation of the cumulative over the grid's
  1,731 intervals, **2.16e-16** and **3.30e-16** (`ORDER-AUDIT.md` §§3.1, 5), so production sits
  ×10.4 and ×6.7 above its floor. The bound is ×4.4 / ×4.5 the production figure and ×46 / ×30 the
  floor. **It keeps the discrimination the old form had**: every QCD representation before
  `qcd-background-audit` prompt 06 fails it (2.194e-14 and 5.835e-14 on $\tau$; 2.186e-14 and
  1.5501e-13 on $c_s\tau$, the figures those modules' own comment blocks record), and unlike the
  old form it cannot be moved by anything that happens to the *reference*.

  Both tests still **read and print** the reference's own agreement beside the bound, labelled as
  not asserted against, so the two numbers stay visible side by side. Evidence:
  [`logs/04b-regenerate-the-convergence-block.md`](logs/04b-regenerate-the-convergence-block.md),
  `docs/tolerance-convergence/ORDER-AUDIT.md` §12.2.

- **[01-convergence-block-has-a-separate-generator]** — **CLOSED**, on the
  `qcd-background-audit` board that holds its measurements, and here because this campaign closed
  it. The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` is **regenerated**:
  `generated` **2026-09-18**, `campaign` `prompts/tolerance-convergence (prompt 04, board item
  T7)`, `schema_version` 2, `recommended_scheme` **`branch`** — a scheme production can execute —
  and $N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$ measured against the corrected $T(z)$
  representation and the 3-point break set. `branch+knots` survives as a populated **control**
  (`decision.control_scheme`), because two test modules index it by name and because prompt 04
  measured what it now buys: nothing — `branch` reaches 1.658e-16 on QCD's $\tau$ floor against
  the control's 3.224e-16, which confirms `qcd-background-audit` prompt 07 at the quadrature level.

  **Written by `docs/gktk-remedial/residual_convergence.py` with no flags and never by hand**, in
  538.4 s; `generated` and `campaign` both moved, which is the check against hand-editing, and
  **only the `convergence` key differs from `HEAD~1`**, verified key by key on the parsed JSON.
  The tolerance owed on this issue's account is paid: **`QCD_BREAK_POINT_ALIGNMENT_TOL` goes
  1.5e-04 → 3.0e-14**, against measured offsets of 3.552714e-15, 7.105427e-15 and **1.421085e-14**
  — 1, 2 and 4 ulp of $u$ — so the whole of the former 1.418851e-04 really was the block's age, as
  four comment blocks of another campaign had predicted since its prompt 06.

  Declined on scope by `qcd-background-audit` prompts 08 and 09 and blocked on a decision at prompt
  04 here; README §7 **D8** removed the boundary and this prompt landed it. Evidence: log 04b,
  `ORDER-AUDIT.md` §12.

Closed by **prompt 02a**, 2026-09-17:

- **[01-v2-density-raises-at-the-qcd-production-anchor]** — **CLOSED.** The version-2 source
  grid now builds at every production anchor on every production cosmology. The stencil
  evaluation in `main.source_grid_spacing_profile` is guarded where the Liouville-Green
  expansion does not exist: such a node is marked `usable = False` and filled by the
  log-interpolation the criterion already applies to a declared crossing's neighbourhood, and
  the guarded nodes are **counted** rather than absorbed silently.

  **What closed it, measured.** `_solve_horizon_exit(QCD, k = 3e8/Mpc, -5)` =
  **3.30033444460513e+16**, reproducing the board's figure to 15 digits. The construction
  raised there at every relative `z_init` perturbation from **1e-16 to 1e-8** and first built
  clean at **1e-6**, with a *stable* 53 guarded evaluations across the whole trip band — so this
  was never one float's accident, and `main.py` genuinely could not build a QCD source grid. QCD
  at its own anchor now builds at **2034 samples / `21ffc126`, 53 guarded `(k, sector)`
  evaluations of one base-lattice node** (corrected 2026-09-17; they are not 53 distinct nodes —
  log 02a §7, and `[02a-stencil-reaches-across-a-declared-break-before-the-mask]`).

  **The acceptance was bit-identity and it held.** QCD at LambdaCDM's anchor is **1996 /
  `4849552b`** and LambdaCDM at its own is **1778 / `60a3205a`**, each with **zero** guarded
  nodes — so the `except` branch is never entered on either and the guard is provably inert on
  every figure in the record. Neither digest moved.

  **What it did not do**, and whose each piece is: the band is exactly as `residual_node_range`
  returns it (**T7**, prompt 04); the crossing mask is not reordered, because that changes which
  nodes are evaluated and could move a published grid; the anchor solve is not tightened (**T6**,
  prompt 03); the digest is untouched (prompt 05). Evidence:
  [`logs/02a-source-grid-density-guard.md`](logs/02a-source-grid-density-guard.md).

---

## 5. Standing notes

1. **A number without its reference's drift beside it is not a measurement** (README §5 rule 5).
   Every figure quoted against a converged reference carries that reference's drift, and no
   conclusion is drawn from a signal that does not exceed it.

2. **A number without its grid generation beside it is not comparable** (README §5 rule 6, new at
   the rebase). Version 0, 1 or 2 — say which. The two campaigns that closed before this one are
   full of figures from all three, and nothing in the record distinguishes them.

3. **The floors are not targets, and the target rule says how one becomes the other**
   (README §2 (f) and **§6.1**). An agent reporting an accuracy below a declared floor has made an
   error, and it is a campaign-wide stop — not a caveat, not a footnote. **The QCD $H(z)$
   discontinuity floor is no longer one of them**: `qcd-background-audit` prompts 04–06 removed it
   and the equivalent phase error is 0.000e+00 rad. **§6.1 rule 5 is the one qualification**: a
   prompt may *re-measure* a floor and supersede an inherited figure, and for $G_k$'s consumer
   spline it must; the stop is a claim below the floor the prompt has itself just measured.

4. **Cost is a per-object count times an object count** (README §2 (c)). $T_k$ numeric is 50
   objects per model; $G_k$ numeric and both WKB sectors are ~65,000. A percentage without the
   multiplier is not a cost.

5. **Counts, not wall time** (README §2 (i)): this machine's elapsed times overstate by up to 53 %.

6. **Four of the eight keyed targets have no tolerance to converge** (README §2 (a)). An agent
   proposing to tighten a WKB or `BackgroundModel` tolerance has misread the tree; the knob there
   is an integer order.

7. **`QuadSourceIntegral` is read-only here** (README §0.4). Touching it, `QuadSource.py`,
   `phase_groups.py` or `AdaptiveLevin/` is a stop.

8. **More datastore objects is the intended outcome, not a cost** (README §4.2, D2 settled
   2026-09-12 and restated campaign-independently by `qcd-background-audit` `21d80b2`). No prompt
   may argue for keeping a shared constant on the grounds that decoupling multiplies rows.

9. **No parameter without its provenance** (README §1.2, §5 rule 9). One recommended or shipped
   without its five provenance fields in the log is an unfinished prompt.

10. **`DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` is settled** (the user, 2026-09-12, on
    `GkTk-remedial` prompt 17's recommendation). It is not reopened here; its provenance entry is
    written from that campaign's record.

11. **`BREAK_POINT_KIND`, the source grid and `RESIDUAL_WKB_REGION_MARGIN`'s value are held fixed**
    (README §0.5). Prompt 04 measures what the margin is worth, because nobody has; changing any of
    the three is another campaign's decision and touching one is a stop.

12. **The target is the loosest setting that clears the measured floor** (README §6.1), swept
    loose-to-tight, scored at the **maximum** over the whole production grid on all three models,
    with the cost at that setting and one step either side. Where the error is already below the
    floor the target is the word `unchanged` and the row records the dominating factor — that is a
    result. Recommending a setting two decades tighter than the one that first clears is the same
    kind of error as recommending one that misses.

13. **The constant-$w$ anchors cover more than $G_k$ and $T_k$** (README §3.1). `RadiationModel`
    carries exact closed forms for $\tau$, $c_s\tau$, $F$, $\theta_G$, $\rho_G \equiv 0$ and
    $\rho_T$, and $z_{\rm exit}$ is the elementary inversion $1 + z = k/(H_0 e^{N})$ — so **T6 and
    T7 have oracles, not only self-convergence**, and a drift quoted for any of them without the
    oracle error beside it is uncalibrated.

14. **Every measurement in this campaign goes through the facility** (prompt 01, board item T1).
    `ComputeTargets/tests/convergence_reference.py` is the one implementation of the convergence
    test, and it is shaped so that notes 1 and 2 are the easy path: `reference_drift` has no
    default for the smallest difference the caller will report and returns the verdict with the
    numbers, and a geometry cannot be built without naming its source-grid generation. A prompt
    that writes its own drift statistic has stepped around both rules, and the reviewer should ask
    why.

15. **The baseline is `bc6dc97`, not `acd5b8e`** (`RECONCILIATION.md` §7, 2026-09-16).
    `ComputeTargets` **452**, `CosmologyModels` **39**. A prompt quoting a suite count, a file
    line number or a "what the tree does" claim from a document dated before the re-anchor must
    re-resolve it: `main.py` alone gained 42 lines above its citation sites, and §7.6 tabulates
    the ones this campaign's documents used. The five results of README §0.3 are unaffected.

16. **The suite counts moved when prompt 01 landed, and one `ComputeTargets` test fails on this
    machine** (prompt 02, 2026-09-16). `ComputeTargets` is **484**, not the 452 of the `bc6dc97`
    re-anchor: prompt 01 added a test module. `CosmologyModels` is still **39, OK**. Of the 484,
    **483 pass**; `test_tk_wkb_phase.TestCost.test_wall_time_per_object` fails at
    `cold = 0.0605 s` against its `0.06 s` limit, reproducibly, **with no `.py` file changed**.
    It is already attributed to `[07-tk-per-object-cost-is-all-setup]` on the `GkTk-remedial`
    board by `prompts/phase-representation` log 02 observation 5, which saw 0.0638–0.0979 s on
    this machine four runs of four. **A prompt here that sees this failure has not caused it**;
    note 5 (counts, not wall time) is why. No issue is opened for it here.

17. **There are nine keyed object types, not eight, and `wavenumber_exit_time`'s key is an
    inequality** (prompt 02, 2026-09-16; note 6 is unchanged and still correct about the four).
    The ninth is `OneLoopIntegral`, which production never builds. And README §2 (g)'s "a new
    accuracy parameter makes every existing row of that type unreachable" holds for eight of the
    nine: `wavenumber_exit_time` accepts any stored row at least as tight as the request and
    returns the loosest such, so **loosening silently reuses a tighter row**. A sweep of that
    target must bypass the datastore entirely. Both are §3 issues above, with the evidence in
    `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.1 and §2.3.

18. **A number without its *anchor* beside it is not comparable either** (prompt 02a, 2026-09-17;
    the same species as note 2, one level down). The two production cosmologies do **not** share a
    `z_init`: `wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM = 2.0636395964161516e16` and
    `PRODUCTION_Z_INIT_QCD = 3.30033444460513e16`, and `PRODUCTION_Z_INIT` is an alias of the
    first. The constant's comment used to claim one value served both, and that claim is why
    **every version-2 QCD figure previously in the record was taken at LambdaCDM's anchor** —
    including `test_source_grid.py`'s 1996 / `4849552b`, `RECONCILIATION.md` §2.7's "1,996 samples
    on QCD" and `docs/qcd-background-verification.md` §10's density measurements. None of those is
    wrong; each is at the other anchor. **The QCD production grid at QCD's own anchor is 2034
    samples / `21ffc126`**, and it is anchor-sensitive to far below the anchor solve's own 3.8e-7
    convergence (`[02a-grid-digest-not-reproducible]`), so quote the anchor with the digest.

19. **`SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05` bounds what the density guard may absorb**
    (prompt 02a, 2026-09-17). It is in no lookup key and cannot change a grid that builds — it can
    only convert a build into a refusal — so it invalidates nothing. It exists because a guard with
    no ceiling would let an arbitrarily misplaced band be filled by log-interpolation in silence,
    which is worse than the raise it replaced. The worst production band reaches **7.716e-04**;
    the ceiling is **64.8x** that.

20. **`atol` is not inert everywhere, and the $G_k$ sector's answer does not transfer** (prompt
    03a, 2026-09-17). In the $T_k$ numeric sector `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` does
    not set the error level — two decades of it move the median of the per-$k$ maxima by ≤2.1× —
    but it does choose **which** wavenumber draws a bad step sequence, moving the maximum by up to
    205×. README §2 (e)'s magnitude argument is why `atol` cannot bind on $G_k$, where
    $|G|\sim10^{12}$–$10^{18}$; $T$ starts at 1 and decays, so the argument does not carry. A
    prompt quoting "`atol` is inert" outside the $G_k$ sector has crossed a sector boundary.

21. **A root solve's `rtol` floor is Brent's `4 eps`, not `solve_ivp`'s `100 eps`** (prompt 03a,
    2026-09-17). The facility reports the latter for every `TolerancePair`
    (`[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve]`), so a `DriftVerdict` note about
    a clamped `rtol` on `_solve_horizon_exit` or `_solve_T_z` is about the wrong solver and should
    be checked, not believed.

22. **The four orders' floor is accumulation, not quadrature, and the ladder is the only way to
    see it** (prompt 04, 2026-09-17). README §6.2 gives the floor for these targets as
    "double-precision accumulation over the grid" and states no number; `ORDER-AUDIT.md` measures
    it as the best any order in the ladder reaches — **2.16e-16 / 3.30e-16 / 8.01e-14 /
    6.51e-17 rad** — and `RadiationModel`'s $N_F$ proves the reading, because there the friction
    integrand is a *constant*, every Gauss order is exact, and the 8.0e-14 that remains is the
    accumulation of 2,305 panels. It follows that "at or below the floor" cannot be applied
    literally to an integer knob: it would select whichever order happened to reach the minimum.
    `order_audit.FLOOR_FACTOR = 3.0`, which is `residual_convergence.smallest_within_factor`'s
    value and meaning, is what "clears" means here and is why the two measurements are comparable.

23. **On the two spline models nothing at or above order 4 is resolved, and that is a statement
    about the reference rather than a gap** (prompt 04, 2026-09-17). An order-32 table scored
    against order 33 drifts by 1.7e-18 to 2.8e-15 depending on the quantity, which is the same
    level as the candidates it is scoring. What survives the drift is the step *from* order 2,
    four to six orders larger, and the exact-radiation control, which has a closed form and
    therefore no drift at all. A prompt quoting "order 8 is better than order 4" from
    `ORDER-AUDIT.md` has read a daggered cell.

24. **The `Tk` sector's Liouville–Green band is never super-horizon** (prompt 04, 2026-09-17), at
    any of the fifty production wavenumbers on any of the three models, because $\omega_T^2 > 0$
    already requires the mode to be inside the *sound* horizon.
    `[01-density-criterion-imposed-outside-the-wkb-region]` is therefore entirely a `Gk`
    phenomenon, and its cost is **114 of QCD's 2,034 source-grid samples and nothing on the other
    two cosmologies**.

25. **The `convergence` block is current, and `QCD_FLOOR_FACTOR` no longer exists** (prompt 04b,
    2026-09-18). `ComputeTargets/tests/wkb_reference_data.json`'s `convergence` key was generated
    **2026-09-18** by `prompts/tolerance-convergence (prompt 04, board item T7)` against the
    corrected $T(z)$ representation and the 3-point break set, recommends **`branch`**, and records
    the four orders at 4; `branch+knots` survives as a measured **control**, not a candidate. Only
    `docs/gktk-remedial/residual_convergence.py` may write that key, and hand-editing it is caught
    by `generated` and `campaign` failing to move. The two tests that read the block now bound
    their **own** production quantities — `QCD_NODE_REL_TOL` and `QCD_CS_TAU_REL_TOL`, both
    1.0e-14, against measured accumulation floors of 2.16e-16 and 3.30e-16 — and
    `QCD_BREAK_POINT_ALIGNMENT_TOL` is **3.0e-14**, ×2.11 above a worst offset of four ulp of $u$.
    A prompt quoting 1.5e-04, 1.878541e-14 or a `QCD_FLOOR_FACTOR` is reading a document written
    before this commit.
