# Prompt 06l — the trust yardstick with the halves' node errors signed, measured on A3 v2 (a measurement)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-05, on the user's decisions after 06k (the board's Decisions, **06l**):
- on 06k's results: *"Remedy: looks like modified route 1 that doesn't carry the CC floors into the
  parent test. If the C2 issue is just
  `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]` then it does not seem fair to
  hold this against route 1. So as far as I can see, this modified route 1 does the intended job."*;
- told that the modification leaves half of route 1's high-ratio trust in place, on a yardstick
  inflated by the halves' cancelling node errors (§0.1), and offered a signed yardstick (B) measured
  first: *"OK, measure B first."*

It runs **before the remedy prompt** for
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` (06m), and before 07 resumes.
Its results decide whether 06m implements B, or route 1 with the user's modification (A).

**Where it runs:** on the Mac Studio, Ray-free, with no lock, on 06's certified scratch copy of
A3 v2, read-only. **It changes no code in the tree** and builds no store. B is a scratch variant of
`AdaptiveLevin/levin_quadrature.py` kept outside the tree, loaded by 06k's harness as 06k loaded its
routes. One agent, dispatched and reviewed by an orchestrator session on the Mac Studio; the records
are committed on the laptop.

**Closes:** nothing. It measures one remedy for
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`, beside 06k's route 1.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entry for `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`,
   with its narrowings of 2026-11-03 (06j), 2026-11-04 and 2026-11-05 (06k and after), and the
   Decisions rows **06k results** and **06l**.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_LEVIN_TRUST_MAX_RATIO` and the comment above it;
   - `_adaptive_levin`'s `_check_leaf` (the trust check one level down), `_solve_halves`, and the
     halves' data dicts as `_make_leaf` carries them;
   - `_adaptive_levin_subregion_cc`'s return dict: `phase_err`, `node_error_signed` ($E$),
     `node_error_bound` and `phase_err_less_node`; and `_cc_roundoff_floor`'s docstring, for the
     node term's place in the floor.
3. 06k's records, `docs/levin-phase-floor/06k/`: `out/results.md`, `log.md` (its deviations and
   observations), `review.md`, `out/partL_leaves.csv`, `out/partC/`, `KEPT_ON_MAC_STUDIO.md`, and
   `scripts/` (with `variants/`, `variant_loader.py`, `make_variants.py`, `run_census.py`,
   `run_k0.py`, `k0_summary.py`, `x1_capture.py`, `check_captures.py`, `probe_l.py`, `score_l.py`,
   `analyse_c.py`, `movers_c.py`, `score_h1.py`, `extract_census.py`).
4. 06j's `out/results.md`, for the two populations of slow leaves.

**Gate:**
- 06k's records commit (`34437951`) is on `origin/handover-remedial`;
- the Mac Studio's main checkout is clean, and `runregistrykit list` shows nothing `running` or
  stale;
- the scratch copy matches its certification by SHA-256, and its fingerprint is `0e32f24a…`;
- 06g's census checkpoint is `03adcbba…` with its 16 region pickles, and `census/out/v8_calls.json`
  is `1e9a62b0…`;
- route 1's census checkpoint
  (`var/runs/levin-phase-floor-06k-census-route1-20261104T103009/checkpoint.jsonl`) is `36e3dd4c…`,
  with its 16 region pickles, and the extracts `var/levin-phase-floor-06k/state/extract/g.pkl` and
  `route1.pkl` are present;
- the Mac Studio has at least 20 GB free (one census run directory, about 4.5 GB).

---

## 0. What this prompt is

### 0.1 What 06k left, and what the orchestrator found after it

06k's route 1 widened rule (b)'s noise clause from $\phi_P$ to $\nu_1 = \phi_P + \phi_L + \phi_R$. It
recovered 98.6–98.9 % of V8's lost calls, moved no value past both errors and left no reference
worse. It failed C2 (7 calls lost to the node-limited stop, which the user does not hold against it)
and K1: 104,662 of its trust firings sat at $\Delta_1/\phi_P \ge 5.8$.

**The firings, by the parent's kind** (the orchestrator's count from route 1's census checkpoint,
2026-11-05):

| Route 1 trust firings | all | at $\Delta_1/\phi_P \ge 5.8$ | largest $\Delta_1/\phi_P$ | $(\phi_L + \phi_R)/\phi_P$, median |
|---|---|---|---|---|
| Levin parent | 60,609 | 48,131 | $1.1\times10^4$ | 140 |
| Clenshaw–Curtis parent | 206,969 | 56,531 | 862 | 2.2 |

The user's modification, a Clenshaw–Curtis half's floor not carried into a Levin parent's test,
removes the first row. The second stays.

**The second row's yardstick is inflated by node errors that cancel.** All 142 of 06j's sampled slow
leaves have a Clenshaw–Curtis parent and Clenshaw–Curtis halves; all 17 of 06h's carrying leaves have
a Levin parent and Levin halves (06k's `partL_leaves.csv`). On the 30 slow leaves that route 1
trusts at $\Delta_1/\phi_P \ge 5.8$:
- each half's $|E|$ is 200 to $1.6\times10^6$ times $|E_P|$;
- but $|E_L + E_R|$ is at most 3.7 % of $|E_L| + |E_R|$ (median 0.7 %): the halves' node errors are
  nearly equal and opposite, and cancel in $V_L + V_R$;
- $\Delta_1/\nu_1$ is 0.005–0.037, so the leaves pass by a factor of about 30;
- $\Delta_1$ is 0.8–500× $|E_P - (E_L + E_R)|$ (median 1.6): these are mostly 06j's second
  population, where the node errors do not explain $\Delta_1$.

The cancellation is consistent with the end-node term at the halves' shared midpoint, which enters
the two halves with opposite signs. That reading is not checked here; Part L records the numbers.

So route 1 trusts much of 06j's second population because $\nu_1$ adds $|E_L| + |E_R|$, a term that
is not in $\Delta_1$. Its honesty on A3 v2 is measured, but the trust is not bounded by anything the
floors declare.

### 0.2 Route B — the halves' node errors combined signed

One change in `_check_leaf`'s noise clause, and nothing else. A checked leaf is trusted if
$r < 1/4$, or if $r < 1$ and $\Delta_1 \le \nu_B$, where:

$$
\nu_B = \phi_P + \sum_{h \in \mathcal{L}} \phi_h + \sum_{h \in \mathcal{C}} \phi^{-}_h + \Big|\sum_{h \in \mathcal{C}} E_h\Big| .
$$

- $\mathcal{L}$ is the halves that are Levin solves, and $\mathcal{C}$ the halves that are
  Clenshaw–Curtis solves (`is_direct`).
- $\phi^{-}_h$ is the half's `phase_err_less_node`, its floor with $|E_h|$ taken out of the node
  term; $E_h$ is its `node_error_signed`.
- **On a Levin parent, $\mathcal{C}$ is empty by rule:** a Clenshaw–Curtis half contributes nothing
  to a Levin parent's yardstick. This is the user's modification.
- $\phi_P$ is the parent's whole floor, $|E_P|$ included, as now.

Each case:

| Parent | Halves | $\nu_B$ |
|---|---|---|
| Levin | Levin, Levin | $\phi_P + \phi_L + \phi_R$, as route 1 |
| Levin | a Clenshaw–Curtis half | $\phi_P$ plus the Levin half's floor, if any |
| Clenshaw–Curtis | Clenshaw–Curtis, Clenshaw–Curtis | $\phi_P + \phi^{-}_L + \phi^{-}_R + \lvert E_L + E_R\rvert$ |
| Clenshaw–Curtis | a Levin half | the Levin half's $\phi$, the other half's $\phi^{-} + \lvert E\rvert$ |

**Unchanged:** the value returned, the declarations, $r_{\rm trust}$, the leaf rules, the floors, the
call's signed node sum, the round-off exit, `depth_max`, and everything outside `_adaptive_levin`.

**The basis.**
- **The guarantee for the first population.** There $\Delta_1 = |E_P - (E_L + E_R)|$ to 0.6 % (06j).
  By the triangle inequality, $|E_P - (E_L + E_R)| \le |E_P| + |E_L + E_R| \le \nu_B$, since
  $\phi_P \ge |E_P|$. So B trusts these leaves because the floors bound their $\Delta_1$, not by
  a margin that depends on the halves' $|E|$.
- **The signed combination is 06g's,** applied one level down. 06g's §0.2 (c) carries the call's
  node errors as $|\sum E|$, because their signs vary and bisection does not lower $\sum|E|$. The
  same argument applies to the two halves whose values $\Delta_1$ adds.
- **A Levin parent's test is against Levin floors.** On a Levin parent with Clenshaw–Curtis halves,
  $(\phi_L + \phi_R)/\phi_P$ has median 140 in route 1's census. The Clenshaw–Curtis rule's floor is
  a different and much larger quantity than eq. (151)'s. Such a leaf is refined, as under 06g, and
  its Clenshaw–Curtis children are tested as Clenshaw–Curtis parents.
- **B trusts no leaf route 1 would not, at leaf level:** $\phi^{-}_L + \phi^{-}_R + |E_L + E_R| \le
  \phi_L + \phi_R$, and the Levin case only drops terms. Part L checks this on every leaf. A call's
  refinement under B can still differ from route 1's after its first differing decision.

**What it costs.** By the orchestrator's estimate from 06k's leaf table (with $\phi_h - |E_h|$ in
place of $\phi^{-}_h$), B trusts about 85 of the 142 sampled slow leaves (route 1: 114). That is all
75 of the first population and about 10 of the second. The rest of the second population stays
refined to `depth_max`, as under 06g. The census measures what that costs in convergence.

### 0.3 What it does not do

- **No change to the tree.** Nothing under `AdaptiveLevin/`, `ComputeTargets/`, `LiouvilleGreen/`,
  `CosmologyModels/`, `ObjectFactories/`, `config/`, `tools/` or `main.py` changes, and no test file
  is edited. The variant lives under `var/levin-phase-floor-06l/scripts/variants/`.
- **No census of A.** Route 1's census stands in for it. A differs from route 1 only on Levin
  parents with a Clenshaw–Curtis half (60,609 of route 1's 267,578 firings), and §3 reports which of
  route 1's recovered calls depend on them.
- **No floor for the second population.** Its $\Delta_1$ is the integrand's sensitivity to its
  inputs at the nodes, which no floor declares (06j). A floor for it is later work.
- **No remedy.** Which route 06m implements is for the user.
- **Not the stationary-point rows.** They are reported apart, and not scored (07).
- **No new references.**

## 1. The harness

As 06k §1, with records under `var/levin-phase-floor-06l/{out,scripts,state}` and `log.md`,
`verification.md` and `issues.md` beside them (README §4.1).
- **The tree** is a worktree at this prompt's dispatch SHA. Its production code is 06g's (`3ae6f3a`):
  `git diff --stat 1c5336e <dispatch SHA> -- . ':!docs' ':!prompts'` shows only
  `tools/filon_reference.py`, its test and `CLAUDE.md`.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after.
- **The scripts** are copied from `docs/levin-phase-floor/06k/scripts/` into
  `var/levin-phase-floor-06l/scripts/`. Save the diff against their sources, and record each change
  as a deviation.
- **The variants.** Two: the identity variant (byte-identical to the tree's file), and route B,
  built by 06k's `make_variants.py` extended for it.
  - The diff against the tree's file is saved. An AST comparison shows that only `_adaptive_levin`
    differs, and within it only `_check_leaf`'s noise clause and B's firing record.
  - **B records every firing:** a leaf it trusts that 06g's rule (b) would not. Each record holds
    the leaf's depth, span, $r$, $\Delta_1$, $\phi_P$, the parent's kind and each half's kind,
    $\phi_L$, $\phi_R$, $\phi^{-}_L$, $\phi^{-}_R$, $E_P$, $E_L$, $E_R$, $\nu_1$ and $\nu_B$, under
    the result key `route_06l`, which the comparison ignores.
- **Long runs.** Anything that may exceed about ten minutes is registered with `runregistrykit`
  (`--campaign levin-phase-floor --prompt 06l`) and launched detached (README §5 rule 7). At most
  six workers.
- **The census run is checked, not cut off.** A run past its projection is reported with its
  progress and projected finish. It is stopped only if the projection is a day or more, counting
  the units in flight.

**Gate K0, before anything is scored.** On 06k's 100 K0 units (`docs/levin-phase-floor/06k/out/k0/units.json`):
- **The identity variant is transparent:** it reproduces 06g's census records bit for bit (value,
  abserr, regions, region arrays, `converged`, checked leaves and the harness's counters).
- **B fires only where it is meant to:** every call on which B's firing count is 0 is bit-identical to
  06g's record.

A K0 failure is a stop (§7).

## 2. Part L — B at leaf level

On 06j's 46 calls, re-captured as 06k captured them, each checked against 06g's census bit for bit.
Evaluate B's trust decision in isolation, as 06k's probes did, on the same three populations:
- **L-a:** 06j's 142 sampled slow leaves, split into 06j's two populations ($\Delta_1/\phi_P < 2$:
  75; $\ge 2$: 67);
- **L-b:** the 17 distinct 06h carrying leaves (22 evaluations);
- **L-c:** the 1,882 returned leaves of the 24 S calls that 06g trusted.

For each leaf report, as CSV, beside 06k's columns for it: $\nu_B$ computed exactly from the halves'
solves ($\phi^{-}$ read from `phase_err_less_node`), each half's kind, $|E_L + E_R|/(|E_L| + |E_R|)$,
$\Delta_1/\nu_B$, and B's decision beside route 1's.

**Gate L3, before the census:** on every Part L leaf, $\nu_B \le \nu_1$ to within rounding (a
relative $10^{-12}$: $\phi^{-}_h + |E_h|$ need not reproduce $\phi_h$ to the last bit), and B trusts
a leaf only if route 1 trusts it, unless $\Delta_1$ lies within that rounding of $\nu_1$, which is
reported. This is §0.2's leaf-level inclusion. A failure means the construction is not what §0.2
says, and is a stop (§7).

**The LiouvilleGreen draws.** Run 06g's 20 seeded draws under B, against 06g's code, as 06k did.
Report, beside route 1's from 06k: the Levin calls whose convergence changes, and the cases that move
between honest and dishonest, the near-singular `test_YJJ_log_singularity` cases
($\epsilon \le 10^{-4}$) apart from the rest.

## 3. Part C — the census under B

One registered census run, 06g's census (all 9,280 rows at `1e-7` and `1e-8`, 18,560 units) re-run
at the tree with B loaded. **The projection** is made before the run from 06g's (20,361 s) and route
1's (16,758 s): B trusts no leaf route 1 would not, so it should lie between them.

**Against 06g's census**, with 06k's comparisons (`analyse_c.py`, `movers_c.py`) re-run for B:
- **Convergence:** of V8's lost calls outside the 720, how many converge under B, and what holds back
  those that do not: an untrusted leaf, `node_limited`, or something else. Calls converged under 06g
  and not under B, each with its cause and its abserr against 06g's.
- **Depth:** calls at `depth_max`.
- **Moves:** every call outside the 720 whose value moves by more than $e_B + e_g$, with its row,
  group and the leaves that carry the move; rows whose `total` moves past both errors. Rows are
  shard-serial, and units are counted apart (06k's deviation 29).
- **Firings:** by depth, by the parent's and the halves' kinds, by $\Delta_1/\phi_P$, by
  $\Delta_1/\nu_B$ and by $r$.
- **Cost:** census wall, unit wall per rtol and group-call regions, against 06g's, 06h's and route
  1's.
- **The 720 stationary-point rows,** with the same counts, reported apart.

**Against route 1's census** (checkpoint `36e3dd4c…`, read-only):
- **The calls route 1 converges and B does not,** outside the 720, at each rtol. For each, the leaf or
  leaves that hold it: depth, $r$, $\Delta_1/\phi_P$, $\Delta_1/\nu_B$, $\Delta_1/\nu_1$, the
  parent's and halves' kinds, and $|E_L + E_R|/(|E_L| + |E_R|)$. How many of them route 1 recovered
  only through a Levin parent with a Clenshaw–Curtis half, which is A's difference from route 1.
- **The calls B converges and route 1 does not,** with their cause.
- **Values:** every call outside the 720 whose value moves by more than $e_B + e_{\rm route\,1}$.

**Against the references.** Score every referenced call outside the 720, and apart within it, as
$|v_B - {\rm ref}|/e_B$ beside 06g's and route 1's ratios. The references are 06k's 124 matched:
06g's 102 (`docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json`, `rows`) and 06j's
22 (`docs/levin-phase-floor/06j/out/w_calls.csv`). V1's 12 are calls of R24, not of the census, and
are not used (06k's deviation 24, on the user's ruling).

## 4. Predictions, fixed before the run

README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
**undecided** against the rule as written. None is a stop. "Outside the 720" means outside the 720
stationary-point rows.

- **L1 — B does not trust 06h's kind of leaf.** **Rule:** of the 17 06h carrying leaves (L-b), B
  trusts none.
- **L2 — B trusts every first-population slow leaf.** **Rule:** B trusts all 75 of 06j's sampled slow
  leaves with $\Delta_1/\phi_P < 2$.
- **C1 — convergence comes back.** **Rule:** at least 90 % of V8's lost calls outside the 720, at
  each rtol, converge under B.
- **C2 — nothing converged is lost but to the node-limited stop.** **Rule:** every call outside the
  720 that 06g converges and B does not ends `node_limited`.
- **H1 — no referenced call gets worse.** **Rule:** on every one of the 115 matched references
  outside the 720, $|v_B - {\rm ref}|/e_B \le \max(1,\ |v_g - {\rm ref}|/e_g)$.
- **S1 — no move past both errors.** **Rule:** no call outside the 720 moves by more than $e_B + e_g$
  against 06g's census.
- **K1 — B trusts Levin parents only at noise level.** **Rule:** every trust firing on a Levin
  parent, outside the 720, has $\Delta_1/\phi_P < 5.8$.

**Descriptive, not scored:** the same counts for Clenshaw–Curtis parents, by $\Delta_1/\phi_P$ and by
$\Delta_1/(|E_P| + |E_L + E_R|)$; how many of 06j's second population B trusts, and why.

**The basis.**
- **L1, H1, S1 and K1** are 06k's safety constraint.
  - L1: 06h's leaves are Levin over Levin, where $\nu_B = \nu_1$ and route 1 trusted none
    ($\Delta_1/\nu_1 \ge 5.7$).
  - K1: on a Levin parent, $\nu_B$ holds only Levin floors. In 06k's L-c sample a Levin half's
    floor is at most 0.96 of its parent's, so $\nu_B \le 3\phi_P$ there.
  - K1 is scored on Levin parents only. On a Clenshaw–Curtis parent $\phi_P$ understates the
    rounding in $\Delta_1$ (§0.1), so $\Delta_1/\phi_P$ against 5.8, a threshold taken from Levin
    leaves, is not a test of noise there.
- **L2** is §0.2's guarantee, checked. 06j measured $r$ at 0.489–0.506 on these leaves, so the
  $r < 1$ condition holds.
- **C1** keeps 06k's rule for comparison. The leaf-level estimate (about 60 % of sampled slow leaves
  trusted, against route 1's 80 %) makes a shortfall likely, and the recovered fraction itself is the
  measure.
- **C2** follows the user's ruling that losses to the node-limited stop are
  `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`'s, not the route's.

B passes as a candidate remedy if L1, L2, C2, H1, S1 and K1 hold. C1 and the comparison with route 1
say how much it recovers, and what A's extra recovery rests on.

## 5. Records

- **`var/levin-phase-floor-06l/log.md`** in README §5.1's form, with every deviation classified;
  `verification.md`; `issues.md` with any observation proposed as an issue. The laptop orchestrator
  copies the log to `prompts/levin-phase-floor/logs/06l-signed-trust-yardstick.md`.
- **`out/`:** the variant's diff and AST check; K0's checks; Part L's table and the draws' table;
  the census's comparison tables against 06g and route 1 (CSV or JSON, each with a provenance
  header); its scores against the references; and a `results.md`.
- **`results.md`** states each verdict with its numbers, and sets B beside route 1 and, as far as
  route 1's census shows it, A. It proposes no code.
- **Provenance** in every output header (README §5 rule 5).
- **Kept on the Mac Studio:** the capture pickles, the census run directory, and anything over 2 MB,
  listed with SHA-256.

## 6. Acceptance

1. Gates K0 and L3 hold.
2. L1, L2, C1, C2, H1, S1 and K1 scored against the rule as written, with their numbers.
3. Part L complete for every leaf; the census complete, 18,560 units; §3's tables complete.
4. The store, 06g's census checkpoint and route 1's census checkpoint unchanged by SHA-256; the tree
   clean, with `--ignored`; nothing created under `var/datastores/`; no Ray started.
5. **Records:** the log, `verification.md`, `issues.md` and `out/results.md`;
   `docs/levin-phase-floor/VERIFICATION.md` §6l, written by the laptop orchestrator; the records in
   `docs/levin-phase-floor/06l/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **Gate K0** fails.
- **B cannot be confined:** it cannot be written as a change inside `_check_leaf`, or a half's
  `phase_err_less_node` or `node_error_signed` is not available where the check is made. Report what
  is missing.
- **Gate L3** fails on any leaf.
- **A measurement cannot be made as written:** a leaf's decision cannot be evaluated in isolation,
  or a reference cannot be matched to its call.
- **The census run's projection is a day or more,** counting the units in flight.
- **Any store file** changes, a change to the tree looks necessary, or Ray is needed.
