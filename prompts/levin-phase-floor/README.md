# Campaign — levin-phase-floor

**Written:** 2026-10-25 on branch `handover-remedial` at `effba46`, by Claude Opus 5.5, on the
user's decision the same day to open it. It owns the **depth-20 cost cliff** of the
`QuadSourceIntegral` Levin integration: on A3 v2, 2265 of 9280 rows run to
`DEFAULT_LEVIN_MAX_DEPTH = 20`, carry 285.8 of the store's 286.3 CPU-hours, and 1179 of them are
stored with `total_converged = 0`.

**Its audit is [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2**,
written the same day at `71962f6`. That section measured that the declared error on depth-20 rows
tracks $\varepsilon k\eta(z_{\rm response})$, the double-precision floor of a phase of size
$k\eta$. It proposed a mechanism, a three-part test (§2.6) and a three-layer remedy (§2.7). This
campaign runs the test, and then the remedy that the test supports.

## 0. What this campaign is

**Correctness, not cost.** The cliff is a cost, but the cost signals something wrong. The driver
is asked for a tolerance its input cannot deliver, nothing tells it so, and it stores a flag that
says little about the answer (findings §2.5). The campaign makes the integrator's answer honest:
the phase is good to what it is declared to be good to, and the stored error and convergence say
what is true of `total`. Lower cost should follow, but it is not the goal, and no argument below
rests on it.

### 0.1 What this campaign is not

- **Not the hand-over seam.** The $T_k$ and $G_k$ seam is `handover`'s. Its error, a few percent
  of `total` (findings §3.2), is larger than this floor, and the two are independent: the seam
  sits at $x_T \approx 15$, the floor at late response times. Nothing here waits for `handover`,
  and nothing there waits for this.
- **Not the Clenshaw–Curtis fallback.** OPEN_ISSUES §1.3 holds it, `[10-levin-wholesale-cc-fallback]`
  among it. This campaign may touch `AdaptiveLevin/`, but only to tell the driver about its input
  (§2.7 layer 2).
- **Not a rebuild.** Prompts 01–04 read A3 v2 or run offline. The next production build is whoever
  runs it after both this campaign's remedy and `handover`'s have landed.
- **Not `handover`'s D2 as a whole.** D2 is the phase carried from producer to consumer: the
  $h^4x/384$ term, the anchoring floor and the storage granularity, and `handover` README §7 D4
  asks whether it belongs there. This campaign takes only the anchoring floor
  (`[00-consumer-anchoring-floor]`), and only at the consumer, where the Levin driver meets it. If
  §2.7 layer 1 has to change what `PrimitivePhase` stores, that is a decision for the user at gate
  **G1** (§6.3).

### 0.2 Reconciliation against `effba46`, and what it changes

Measured by the orchestrator on 2026-10-25 while writing this plan.

- **The tree.** `handover-remedial` at `effba46`, clean. Since A3 v2 was built at `b90752d`, the
  code that computes a `QuadSourceIntegral` row has changed in one file: `ComputeTargets/analytic_Gk.py`
  (handover 09: `None` outside $1+3w>0$). It does not touch the Levin path. Neither does
  `config/ray_local.py` or `main.py`'s `ray.init`. `AdaptiveLevin/`, `phase_groups.py` and
  `QuadSourceIntegral.py` are byte-identical to `b90752d`.
- **The Mac Studio.** Its checkout is at `c9026af`, clean; py313-venv 02 left it there. The
  registry shows nothing `running` or stale, and `~/.raypoolkit/` is empty, so the lock is free.
  The store is in `var/datastores/handover-A3-v2-lambdacdm{,-shard000[0-3]}.sqlite`.
- **What a row stores about its groups.** `phase_group_Levin_integral` returns one record per
  phase group per Levin sub-interval, with `label`, `value`, `abserr`, `converged`,
  `phase_limited`, `regions`, `simple_regions`, `evaluations` and `elapsed`
  (`QuadSourceIntegral.py:1076-1096`). Every row stores them in
  `metadata["WKB_Levin"]["subintervals"][i]["groups"]`. The census of findings §2.1 did not read
  them, and §2.4 says *"which group carries the depth-20 regions is not recorded"*. **The
  per-group cost is recorded. The per-group depth is not**, since a row stores only the maximum
  over its groups.

**The reconnaissance.** One read of those records over the 629 depth-20 rows that took more than
100 s. It ran read-only and Ray-free on the Mac Studio:
[`measure/recon_groups.py`](measure/recon_groups.py), output at
[`measure/out/recon_groups.txt`](measure/out/recon_groups.txt).

| (regime, group) | Levin CPU-hr | share | regions |
|---|---|---|---|
| `Gr`, `G+r` (slow on `r == k`) | 149.02 | 52.7 % | 71,528,183 |
| `Gr`, `G-r` (fast) | 94.68 | 33.5 % | 45,184,170 |
| `G`, `G` | 17.70 | 6.3 % | 20,069,867 |
| `Gqr`, four groups | 21.47 | 7.6 % | 6,661,356 |

The most expensive group of each row:
- `G+r` in 367 rows, all `r == k`;
- `G-r` in 90;
- the lone `G` in 62 (35 of them `r == k`);
- a `Gqr` group in the remaining 110.

Most of the cost is in the low band: 229.7 of the 282.9 CPU-hours. *(Corrected 2026-10-25: 247.9, by prompt 01's re-derivation, its I3. The 229.7 was an arithmetic slip in writing this plan; the recon output was right.)*

**Which label is the slow group.** In the code's conventions $\theta_G$ decreases with $z'$ while
$\theta_q$ and $\theta_r$ increase (`source-remediation` log 07, note 4). So the slowly varying
group is the one where $G$ and a transfer factor carry the **same** sign. On `r == k` that is
`G+r`, where the $\tau_s$ terms of $G$ and $T_r$ cancel. `G-r` is the fast group, at about $2k$.
*(Corrected 2026-10-25 by prompt 01, its I1: they cancel only in part. $T$'s phase runs on the sound
horizon, at $c_s k/H$, so `G+r` oscillates at about $(1-c_s)k/H$ and `G-r` at $(1+c_s)k/H$.
`[01-slow-group-leading-terms-do-not-cancel]`.)*
That note also warns to check `theta_deriv` rather than assume this. 01 checks it in the code, and
03 checks it numerically.

**What it changes.**

- **Findings §2.2 is supported where it points.** The slow, cancelling group `G+r` carries 52.7 %
  of the cost, and it is the most expensive group in 367 of the 629 rows.
- **It is not the whole story.** At least 40 % of the cost is in groups with no cancelling pair:
  - the fast `G-r` on the same `r == k` rows, 33.5 %;
  - the lone `G`, 6.3 %;
  - part of the `Gqr` share; the split by geometry is 01's.
- **§2.6's refutation clause does not decide a split like this.** It says the hypothesis is wrong
  *"if item 1 finds the depth-20 regions in a group with no cancelling pair"*. Here they are in
  both.
- **A wider form would account for the rest.** `PrimitivePhase` reduces $G$'s phase $k\Delta\tau$
  against its anchor at $z_{\rm response}$ (`[00-consumer-anchoring-floor]`). For a source time
  well before the response, $|\Delta\tau| \approx \eta_R$, so **$G$'s own remainder** carries
  jitter of about $\varepsilon k\eta_R$. That jitter enters **every group that contains an
  oscillatory $G$**: sum, difference or alone, cancelling or not. This is the floor findings §2.1
  measured, $\varepsilon k\eta(z_{\rm response})$. It needs no cancellation.
- **This is not established.** The reconnaissance is descriptive: no depth per group, no
  parent–child discrepancy, no check of the error per group against the floor. Prompt 01 measures
  it properly, and prompt 03 tests the mechanism directly. Until then the campaign carries **two
  named hypotheses**, which are not exclusive:
  - **H-cancel**: findings §2.2 as written, the floor confined to failed cancellations;
  - **H-anchor**: the floor carried by $G$'s remainder into every group containing $G$.

  Both can hold: a floor everywhere, made worse where a cancellation leaves a slow group. Their
  predictions are written into 01 and 02 before either runs (§5 rule 3).
- **Which hypotheses stand decides the remedy's shape.** §2.7 layer 1 has two parts:
  1. take the constant response phase $k\tau_R$ out of the integrand;
  2. compose a group's leading terms before reducing.

  Under H-anchor, part 1 is needed in every group. Under H-cancel, part 2 does the work where a
  cancelling pair exists. The remedy prompts are not written until **G1** (§2), so the plan
  commits to neither.

## 1. Scope

**In scope:**
- the §2.6 test, as prompts 01–04;
- after gate **G1**, the remedy §2.7 calls layers 1 and 2, re-measured on A3 v2's rows offline;
- after **G2**, layer 3: the depth cap, the per-region acceptance test with its dead `local_atol`
  branch, and what `total_converged` means.

**Four issues are assigned here** on 2026-10-25 (§6.2 C5). Each stays on its own board:
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`);
- `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` (`handover`);
- `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` (`handover`);
- `[00-consumer-anchoring-floor]` (`GkTk-remedial`).

**Out of scope:** §0.1, and:
- `[04-roundoff-floor-can-be-infinite]`. It stays in OPEN_ISSUES §1.3. 01 confirms whether it
  reaches A3 v2 (findings §2.5 item 2) and narrows it there.
- **Every `extract_*` reader.**
- **Any store write.** A3 v2 is read, never written: byte-identical before and after, fingerprint
  `0e32f24a…`.

## 2. Prompts and gates

| # | Prompt | Covers | Where | Model | Status |
|---|---|---|---|---|---|
| 01 | [`01-phase-group-census.md`](01-phase-group-census.md) | every group record in A3 v2: cost, regions and declared error per group against $\varepsilon k\eta_R$; H-cancel against H-anchor on the store; findings §2.5 confirmed on the store; the candidate rows for 03 | Mac Studio, read-only, no Ray | Opus | written |
| 02 | [`02-phase-noise-injection.md`](02-phase-noise-injection.md) | findings §2.6 item 2: deterministic phase noise of amplitude $\delta$ added to the composed `theta_mod_2pi` on A2's fixtures, by group; depth and regions against $\delta$ | laptop, offline, no Ray, no store | Opus | ⛔ stopped at step 3 and superseded by 03 (user, 2026-10-25); N4 moves to 04 |
| 03 | [`03-instrument-depth-20-rows.md`](03-instrument-depth-20-rows.md) | findings §2.6 item 1: seven A3 v2 rows from 01's list, re-run in-process and bit-identical to the store; depth per group, the jitter, and the discrepancy along a region's ancestry against the floor, with a jitter-removed control | Mac Studio, reads A3 v2 | Opus | written 2026-10-25, after 01 |
| 04 | [`04-remove-the-floor.md`](04-remove-the-floor.md) | findings §2.6 item 3: in a harness, not in production, take $k\tau_R$ out of $G$'s phase and/or compose the group's leading terms before reduction, as 01–03 indicate; re-run 03's rows; and, from 02's N4, whether each row's stored `total_abserr` covers $|{\rm total} - {\rm total}_{\rm jitter\ removed}|$ | Mac Studio, reads A3 v2, at the pre-B1 tree `98e68d3` | Opus | landed 2026-10-25: R1, A1 and R2 hold, N4 fails on 3 of 7; §8 amended two gates after a stop at step 5 |
| **G1** | gate | the user reads 01–04 and decides which hypothesis stands and which remedy layers follow (§6.3 Q1–Q3) | — | — | answered 2026-10-26 (the board's Decisions) |
| 05 | [`05-reanchor-g-phase.md`](05-reanchor-g-phase.md) | layer 1(a): $G$'s remainder exact per table node, against $2\pi$; adds `tools/store_value_diff.py`; R9/R10 compared with it; a wider re-measurement on A3 v2 | laptop; Mac Studio for §5–§6 | Opus | written 2026-10-26 |
| 05a | [`05a-reanchor-t-phase.md`](05a-reanchor-t-phase.md) | layer 1(a) and the $2\pi$ modulus for $T$, as 05 did them for $G$: `TkSourceFunctions` turns on the node-anchored remainder; R11/R12 compared with `tools/store_value_diff.py`; 05's 39 rows re-run, $c$ against $t$, with a probe of $T$'s jitter | laptop; Mac Studio for §5–§6 | Opus | written 2026-10-26, on the user's decision the same day |
| 06 | [`06-producers-store-the-residual.md`](06-producers-store-the-residual.md) | the producers' stored phase (G1 Q3), for $G$ and $T$: each value stores its small residual $\Delta\theta - \rho.\delta$ exactly, its remainder is reduced exactly against $2\pi$, and the consumers build $\varphi$ from the residual; R13/R14 compared with `tools/store_value_diff.py`; 05's 39 rows re-run with the producers re-run in-process, with 04's $\varphi$ metric on R1–R7 | laptop; Mac Studio for §5–§6 | Opus | landed 2026-10-27: S1 and S3 hold, S2, S4 and S5 fail; opened `[06-levin-accepts-a-region-across-a-stationary-point]` for G2 |
| **G2** | gate | the user decides whether layer 2 is needed (§6.3 L2), layer 3 (Q4), now with the stationary-point guard, and the `rtol` question (Q5); written up on the board, §2.3 | — | — | answered 2026-10-27: L2 not needed at this code after 06a's census, to be revisited after 07 or later; Q4 (i)–(ii) is 07, the rest at G3 |
| 06a | [`06a-depth-census.md`](06a-depth-census.md) | L2 (b): all 9,280 A3 v2 rows re-run in-process at `HEAD`, Ray-free, at `1e-7` and `1e-8`, for the distribution of Levin depth; the user then decides layer 2 | Mac Studio | Opus | landed 2026-10-27: C1, C2 and C4 hold, C3 fails on stationary-point rows only; no row reaches depth 18 at either tolerance |
| 06b | [`06b-roundoff-exit-premise.md`](06b-roundoff-exit-premise.md) | `[07-roundoff-exit-assumes-bisection-cannot-help]`, found by 07's Part A: the round-off-limited exit fires only if bisection would not lower the region's floor by half; R15/R16; 06a's census re-run as 07's "before" | laptop; Mac Studio for §4–§5 | Opus | landed 2026-10-28: §4, W0, W1 and W3 hold, W2 and W4 fail; the header issue stays open until W2 is fixed |
| 06c | [`06c-global-error-control.md`](06c-global-error-control.md) | `[07-per-region-acceptance-misses-a-cancelling-total]`, found by 07's §2: one error budget for the call, the refinable leaf with the largest error bisected until $\sum e_i \le \max(\texttt{atol}, \texttt{rtol}|\sum v_i|)$, in place of per-region acceptance; `_local_atol` removed; every computed value moves; R17/R18; 06b's census re-run as 07's "before" | laptop; Mac Studio for §4–§5 | Opus | written 2026-10-28, on the user's decision the same day, after 06b and before 07's §2 |
| 06d | [`06d-cc-roundoff-floor.md`](06d-cc-roundoff-floor.md) | `[07-cc-solves-use-the-levin-roundoff-floor]`, found by 07's third stop: a round-off floor for every Clenshaw–Curtis solve derived from the rule itself, with no $1/\theta'$, in place of eq. (151); and a guard in the depth-18 log | laptop | Opus | written 2026-10-29, before 06c's Part B, which measures both |
| 06e | [`06e-exact-end-nodes.md`](06e-exact-end-nodes.md) | `[06c-chebyshev-end-nodes-miss-the-span]` and `[06d-cc-node-term-is-pessimistic]`, found by 07's fourth stop and a check of production's depth-20 calls: the Chebyshev grid's end nodes set exactly to the span's ends, so leaves tile; 06d's node term replaced by the signed error of the exact node displacements, still a bound; R19/R20; 06c's census re-run as 07's "before" | laptop; Mac Studio for §4–§5 | Opus | written 2026-10-30, before 07 resumes |
| 06f | [`06f-cc-floor-terms-across-leaves.md`](06f-cc-floor-terms-across-leaves.md) | `[06e-node-errors-add-in-absolute-value]` and `[06e-cc-declared-phase-term-is-a-crude-proxy]`, from 06e's census (Z2 and D4): how the Clenshaw–Curtis node term and declared-phase term add up across a call's leaves, each a per-leaf absolute value summed though the error it stands for is one signed quantity; measured on captured census calls against the signed node error and the call's first-order response to a phase error of the declared size | Mac Studio, reads A3 v2's scratch copy, no Ray | Opus | written 2026-10-31, on the user's decisions after 06e's Part B; a measurement, no code |
| 06g | the trust check's value | `[06e-trusted-leaf-returns-its-coarsest-value]` (06e's D3) and the slowly converging leaf of 06e's Z3 sample 3 (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`): return the finest value computed, with a declared error that covers it; R21/R22; a census re-run as 07's "before" | laptop; Mac Studio for its rehearsals and census | Opus | ⬜ not written; held until 06f's results (user, 2026-10-31) |
| 07 | [`07-stationary-point-guard.md`](07-stationary-point-guard.md) | Q4 (i)–(ii): `[06-levin-accepts-a-region-across-a-stationary-point]`, generically, in the Levin integrator only (the late resonances are a separate physics prompt, `[06-time-integrals-must-reach-late-resonances]`); stationary points located in advance from the basis's closed-form phase rates in the generic code, a window of phase half-width $\Phi_w$ round each, and a nodal guard as the safety net; $\Phi_w$ and $\Lambda_c$ measured at a planned stop | laptop; Mac Studio for §5–§6 | Opus | written 2026-10-27; amended 2026-10-28 (§9) after Part A stopped before §2: 06b first; and again (§9 point 6) after §2 stopped on its rule: 06c first; and (§9 point 7) after a third stop: 06d first; and (§9 point 8) after a fourth: 06e first, and QUADPACK's rule for window leaves |
| 08 | fact-finding for layer 3 and `rtol` | Q4 (iii)–(v) and Q5, measured with 07's guard in place | Mac Studio | Opus | ⬜ not written |
| 08+ | remedy | layer 2 if 06a's census calls for it; layer 3 and `rtol` as G3 decides | — | — | ⬜ not written |
| **G3** | gate | the user answers Q4 (iii)–(v) and Q5 on 08's measurements | — | — | ⬜ |

**Why 01 and 02 are written now and 03 and 04 are not.** Neither 01's task nor 02's depends on
anything still to be measured: 01 reads what is stored, and 02 runs on fixtures. 03's rows and the
groups it instruments come from 01. 04's construction depends on which hypothesis 01–03 leave
standing. Each held prompt's charter is fixed in the table above. Only its method waits.

**01 and 02 can run at the same time.** 01 runs on the Mac Studio and 02 on the laptop. Neither
uses Ray (§5 rule 6), so neither takes a lock, and ChamPBH or StochasticInstantons holding the
laptop's lock does not block 02.

**02 was stopped and superseded (user, 2026-10-25).** Its first noised probe cell ran ~69 minutes
without finishing. Phase noise whose size does not shrink with region width defeats the driver's
per-region relative test at every depth once it exceeds `rtol`, so every Levin call bisects the
full depth-20 tree, and the sweep would have taken weeks. 03 then measured the mechanism on A3 v2's
own rows, removing the jitter and nothing else. That answers N1 and N3 more directly than injection
on fixtures could. N4, whether the declared error covers the true one, moves to 04, where the
jitter-removed re-run gives each real row a reference `total`. N2's original form is not tested;
the user judged that it does not matter. The records are 02's log and `VERIFICATION.md` §2.

## 3. Datastores

- **01, 03 and 04 read A3 v2** (`var/datastores/handover-A3-v2-lambdacdm.sqlite`, `0e32f24a…`)
  on the Mac Studio. Each records the SHA-256 of every store file before and after, and they must
  be equal.
- **02 uses no store.**
- **No prompt creates a store.** If 03 or 04 needs Ray to deserialise a row's inputs, it takes the
  Mac Studio's lock and creates nothing.

## 4. The interfaces

### 4.1 Where a Mac Studio prompt writes, and who commits

As `handover` 10: a Claude session on the Mac Studio, started from a laptop session over the saved
SSH connection, commits nothing. It writes under `var/levin-phase-floor-NN/{out,scripts,state}`,
with `log.md`, `verification.md` and `issues.md` beside them. The laptop orchestrator copies all of
it except bulk output to `docs/levin-phase-floor/NN/`, and makes the prompt's one commit:
- the copied records;
- the log in `logs/`;
- this campaign's board;
- `docs/OPEN_ISSUES.md`.

The verification document is [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md).
Prompt NN writes its §NN, and nothing else in it. **01 creates the file.**

### 4.2 Rollback

Every prompt is one commit, undone by `git revert`. 01–04 add only `docs/` scripts and records, so
reverting one leaves no state behind. A3 v2 is never written.

### 4.3 What 01 hands to 03

A table of candidate rows, at least two per class:
- `G+r` (slow, cancelling) dominant on `r == k`;
- `G-r` (fast) dominant on `r == k`;
- lone `G` dominant;
- a `Gqr` group dominant.

Each row is given by `(shard, serial)` and its $(k, q, r, z_{\rm response})$, with its per-group
records. Rows of 100–800 s are preferred, in the mid band where one exists. The orchestrator picks
03's rows from that table.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, and:

1. **01–04 change no production code.** Nothing under `AdaptiveLevin/`, `ComputeTargets/` (outside
   `tests/`), `LiouvilleGreen/`, `CosmologyModels/`, `ObjectFactories/`, `config/` or `main.py`
   changes. Instrumentation is done in the harness's own process:
   - by wrapping a function at the module attribute that production code calls it through, e.g.
     `ComputeTargets.QuadSourceIntegral.build_phase_groups`;
   - or by calling `adaptive_levin_sincos` directly with production's arguments.

   No test file is edited.
2. **A3 v2 is read-only.** Every shard is opened `mode=ro` where SQLite is used directly. Every
   store file is hashed before and after, and the hashes must match.
3. **Predictions come before measurements.** Each prompt states what H-cancel and H-anchor predict,
   with a decision rule, before it runs. The log records each verdict as **holds**, **fails** or
   **undecided**, against the rule as written. A rule that turns out badly posed is recorded as
   such. It is not re-posed after the numbers are in.
4. **Correctness is the only objective** (the user's standing rule). A finding about cost is
   evidence about what the driver is being asked to do. It is never an argument by itself for a
   remedy, an order or a tolerance.
5. **Provenance.** Every script writes into its output header:
   - the SHA-256 of itself and of each input it reads;
   - the `git HEAD` it ran at;
   - the store's fingerprint where it reads one.

   A record from a script whose hash has changed is recomputed, never re-stamped (`CLAUDE.md`
   rule 7).
6. **Ray.** 01 and 02 do not use it. 03 and 04 use it only if deserialising a row's inputs needs
   it, and then only under the Mac Studio's lock, with `--ray-address local` and `/tmp/ray-sgk`.
7. **Long runs.** Anything that may exceed about ten minutes is registered with `runregistrykit`
   and launched detached, and the turn ends (`CLAUDE.md`).

### 5.1 The log template

- the subject, commit (or "records committed by the orchestrator") and result;
- **What shipped**;
- **Predictions and verdicts**: each rule as the prompt wrote it, the number it was scored on, and
  holds, fails or undecided;
- **Deviations from the prompt**, each classified `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE`
  or `UNINTENDED DRIFT`;
- **Verification performed**;
- **Observations not acted on**;
- **State handed on**.

## 6. Decisions

### 6.1 Made by the user (2026-10-25)

| | Decided |
|---|---|
| **U1** | Open this campaign, from findings §2, on the plan proposed the same day: the §2.6 test first, then the remedy |
| **U2** | A3 v2 analysis runs on the Mac Studio, where the store is; the laptop's Ray is left to ChamPBH and StochasticInstantons |

### 6.2 Choices of method made in writing this plan

These are the orchestrator's choices, and the user can reverse any of them.
- **C1 — the name** `levin-phase-floor`.
- **C2 — a census first.** The per-group records were already stored, and the reconnaissance
  (§0.2) shows they can move the hypothesis. So the first prompt reads all of them, before any row
  is re-run. Findings §2.6 item 1 becomes 03, and is aimed by 01.
- **C3 — two hypotheses, not one.** H-anchor is added beside findings §2.2's mechanism and
  pre-registered with it. §2.6's clause cannot decide a split in which both kinds of group carry
  depth-20 cost, and the reconnaissance shows that split.
- **C4 — 02 injects in the exact flavour too.** The exact flavour's composed phases are closed
  forms, and A2 found its cost flat in $x$. If noise alone makes it grow, then
  `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` is answered by sufficiency: phase noise
  is enough to produce the realistic flavour's growth.
- **C5 — the four issues assigned here** (§1). Findings §2.7 says nothing owns them:
  `levin-refactor` and `GkTk-remedial` are closed, and `handover` README §0.3 excludes
  `AdaptiveLevin/`. `[04-roundoff-floor-can-be-infinite]` already sits under §1.3's
  campaign-to-be, so it is not moved.
- **C6 — no remedy prompt before G1.** The batching rule: a prompt is written once its task is
  determined, and the remedy's task is what G1 decides.
- **C7 — 04 runs layer 1(a) and an exact limit, not layer 1(b)** (added 2026-10-25, writing 04).
  03 found the floor in every scored group, cancelling or not, and
  `[01-slow-group-leading-terms-do-not-cancel]` shows that a composition of leading terms cannot
  reach a zero coefficient. So 04's arms are:
  - **A**, $k\tau_R$ out of $G$'s phase, which production could build without touching
    `AdaptiveLevin/`;
  - **X**, every factor's remainder reduced exactly. That is the limit any phase layer reaches,
    (b) and per-region anchoring included, and the reference for N4.

  Both are reduced against the double `TWO_PI`, as production's `fmod` is. So X differs from
  production by the jitter alone, and the modulus's own smooth error (about
  $0.18\,\varepsilon|\theta|$) is reported, not removed.
- **C8 — 06 stores the small part** (added 2026-10-26, writing 06). G1 Q3 asked for a method of
  its own for the producers' stored phase. The noise is the loss of the small terms $\rho$ and
  $\Delta\theta$ when the producer forms one double of size $k\tau$; the consumer cannot recover
  what is not stored. So the producers store the residual $r = \Delta\theta - \rho.\delta$ from
  its small terms, reduce the stored remainder exactly against $2\pi$ by 05's limb construction,
  and the consumers build $\varphi$ from $r$. The alternative, storing $\theta$ in extended
  precision and keeping the consumer's subtraction, keeps a cancellation the residual avoids.

### 6.3 Open, for the user at the gates

*Q1–Q3 were answered at G1 on 2026-10-26; the answers are on the board, under Decisions. L2, Q4 and
Q5 remain for G2, written up with their evidence on the board, §2.3 (2026-10-27).*

- **Q1 (G1): which hypothesis stands.** H-cancel, H-anchor, both or neither.
- **Q2 (G1): which layer 1 is wanted.**
  - (a) $k\tau_R$ taken out of $G$'s phase at the consumer;
  - (b) a group composed from its leading terms before reduction;
  - (c) both;
  - (d) neither, with layer 2 alone.
- **Q3 (G1): whether layer 1 changes `PrimitivePhase`.** If it does, layer 1 changes what
  `PrimitivePhase` exposes, and possibly what `TkWKBIntegration`/`GkSourcePolicyData` store. Then
  the user decides whether it lands here or as `handover`'s D2 (README §7 D4).
- **Q4 (G2): layer 3.**
  - `DEFAULT_LEVIN_MAX_DEPTH`, which was never chosen;
  - the per-region acceptance test and its dead length-share `local_atol` branch;
  - what `total_converged` means, and whether a consumer gate reads `total_abserr/|total|`
    (findings §2.5);
  - **the stationary-point guard** (added 2026-10-27, on the user's decision after 06):
    `[06-levin-accepts-a-region-across-a-stationary-point]`. The acceptance test is blind to an
    interior stationary point of a region's phase. It is fixed here, generically: the points are
    located in advance from the closed-form $\theta'$ and split out, and the driver refuses a
    region whose $\theta'$ changes sign or nearly vanishes.
- **L2 (G2): whether layer 2 is needed at all,** now that layer 1 and the producers' stored phase
  have landed. G1 deferred it here. The evidence after 06 is on the board, §2.3.
- **Q5 (G2): whether `rtol = 1e-7` is right for accuracy.** Added 2026-10-25 at the user's request.
  - `a3-v2-readiness` prompt 01 (D1, D2) moved `DEFAULT_QUADRATURE_RTOL` from `1e-8` to `1e-7` as
    the loosest rung the 2026-09 tolerance sweep found self-converged, partly on cost.
  - A3 v2 removed the cost case: at `1e-7`, 2265 rows still reach depth 20 and carry 285.8 of the
    store's 286.3 CPU-hours (findings §1).
  - What stays open is the accuracy lost against `1e-8`. The sweep measured it at 1.9e-6 (mid band),
    2.84e-6 (low) and 8.3e-7 (seam), on 25 work items only. On depth-20 rows, where the floor exceeds
    both rungs, it is unmeasured.
  - 04 re-runs real A3 v2 rows anyway; running them at both rungs would measure it there. *(04
    does, with arm X at `1e-7` and `1e-8`: its R2 and Q5's number.)*
  - Cost is not an argument either way (§5 rule 4).

## 7. Baselines

01–04 change no code, so they re-run no test root. Each records `git status --short` before and
after, and the only additions are under `var/` (on the Mac Studio) or `docs/levin-phase-floor/`
(on the laptop). The remedy prompts' baselines are set when they are written.
