# Prompt 07b — the Levin leaf beside a window, and how honest the census is

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-13, on the user's decisions after 07a's records landed (the board's Decisions,
**07a Part C** and **07b**):
- on what follows 07a: *"I think A next"*: the H3 prompt, with an extended reference sample, before 08;
- on the reference sample: *"also take some calls at random from the whole census, to give us a sense
  of what happens"*;
- on the calls targeted for other issues: *"yes, include these extras"*;
- on 06b's under-declaration: *"yes, include this"*: test whether it is the same mechanism;
- on the name: *"07b is fine"*.

Earlier, at 07a's Part A stop: *"H3 defect: yes, this is an issue to fix in a separate prompt"*, and
*"option 3 sounds good for the H3 prompt"* (a random reference sample of about 30 of the 396 class-(c)
rows as its measurement).

**Where it runs:** in three parts, as 07a did.
- **Part A, §2, the measurement:** on the Mac Studio, Ray-free, with no lock, in a worktree at the
  dispatch SHA, whose code is 07a's. It finds why the leaf's declaration understates, scores candidate
  fixes, and takes the reference sample. **It ends at a planned stop:** the user chooses the fix with
  its tables in front of them, and the orchestrator writes it into §1 and §3 as an amendment (§9),
  committed and pushed before Part B.
- **Part B, §1, §3 and §4, the code:** on the laptop, by a fresh-context subagent, in the main checkout
  at `HEAD`. One commit.
- **Part C, §5 and §6:** on the Mac Studio, after the code commit is pushed: **§5, rehearsal R31,**
  under the lock, against 07a's R30; **§6, the census,** Ray-free, on 06's certified scratch copy of
  A3 v2, against 07a's census as the "before". **This census becomes 08's "before"**, in place of
  07a's: the fix moves stationary-point values, so 07a's no longer meets the user's condition
  (*"provided the code doesn't move significantly between"*).

The records of Parts A and C are committed on the laptop afterwards: Part A's with §9's amendment,
Part C's in 07b's records commit.

**Closes, in its records commit, on the user's word:**
- `[07a-levin-leaf-beside-a-window-understates-its-error]`, if §6's U4 and §5 hold;
- `[06b-moved-calls-declare-less-than-their-error]`, if §6's U6 holds; otherwise it is narrowed by what
  §2.2 finds;
- `[07-iroff1-window-stop-can-leave-a-call-dishonest]`, if §2.4's references for W7's calls show them
  honest at 07a's code and §6's U4 holds on them.

**Narrows, by its references:** `[07a-trust-check-auxiliary-route-moves-the-declared-error]` (closed
on the user's word if every call referenced for it is honest under both codes), and
`[07-c2-gate-threshold-was-never-measured]` (the own-class references, evidence only).

**Recommended model:** **Opus**, for every agent.

**Read first:**
1. **The board's §3 entries,** with every narrowing:
   `[07a-levin-leaf-beside-a-window-understates-its-error]`,
   `[06b-moved-calls-declare-less-than-their-error]`, `[07-iroff1-window-stop-can-leave-a-call-dishonest]`,
   `[07a-trust-check-auxiliary-route-moves-the-declared-error]`,
   `[07-c2-gate-threshold-was-never-measured]` and `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`;
   and the Decisions rows **07a Part A**, **07a Part C** and **07b**.
2. **07a's prompt,** [`07a-window-stop-and-phase-variation.md`](07a-window-stop-and-phase-variation.md),
   §0–§2 and §9; and 07's §1 and §2 (the windows, and how $\Phi_w = 2\pi$ and $\Lambda_c = 4$ were set).
3. **`AdaptiveLevin/levin_quadrature.py`:**
   - the module docstring's items on the C2 gate and on stationary points, which record that near a
     stationary point *"a Levin region's estimate and its halves' all lack the stationary-phase term,
     so the step-(4) residual, and the trust check one level down, are blind to it"*;
   - `_LEVIN_TRUST_MAX_RATIO` and the comment above it; `_trust_noise_yardstick`;
   - `_LEVIN_WINDOW_HALF_PHASE`, `_LEVIN_COALESCENCE_LAMBDA`, `_LEVIN_WINDOW_PIECE_PHASE` and their
     comments; `_CC_THETA_PRIME_REL_ERROR_SUPPLIED`; `_declared_endpoint_phase_err`;
   - in `_adaptive_levin`: its docstring on the trust check one level down; `_solve`, `_solve_halves`,
     `_check_leaf` (the declaration $\max({\rm residual}, \delta_1)/(1 - r)$ and the value $Q$ returned);
     the windows' edges; and the seed regions either side of a window;
   - `_Basis_SinCos.build_Levin_data` and `_adaptive_levin_subregion_impl`, the Levin solve itself.
4. **07a's Part A records,** `docs/levin-phase-floor/07a/measurement/`: `out/stop_2_report.md`,
   `out/attr/attr_tables_final.md` (every call's H1–H3 split and closure), `out/o9/o9_tables.md` (each
   H3 leaf with its parent and sibling), and `scripts/attr_07a.py` (how a leaf was referenced on its
   own span).
5. **07a's census records,** `docs/levin-phase-floor/07a/census/`: `log.md`, `out/scores/summary.json`
   (`aux_only`, `route_changes`), `out/scores/moves_beyond.json`, `out/scores/w7_detail.json`,
   `out/refs/report.md`, and `scripts/` (`make_capture.py`, `check_captures.py`, `variant_loader.py`,
   `worker.py`, and the references: `routes_refs.py`, `run_refs.py`, `refs_projection_07a.py`).
6. **06b's W2,** `docs/levin-phase-floor/06b/census/out/w2_reference.json` and
   `out/cmd/37_analyse06b_report_rerun.txt`: 24 calls at `1e-7` with references $S_{10}$.
7. **07's V4 method,** `docs/levin-phase-floor/07/census/out/v4/report.md` and the board's Decisions
   **07 Part B stops 2–4**: FCC at two resolutions, QAWO where projected at 30 minutes or less, and the
   rule by which readings agree.

**Gate (Part A):**
- 07a's records (`a74e7e10`) and this prompt's commit are on `origin/handover-remedial`;
- the laptop's main checkout is clean, and `runregistrykit list` shows nothing `running` or stale on
  either machine;
- **the code is 07a's:** `git diff --stat 1ec734d7 <dispatch SHA> -- AdaptiveLevin ComputeTargets
  LiouvilleGreen tools` lists only `89c368f2`'s four files, and each is AST-equal to `1ec734d7`'s with
  docstrings removed;
- on the Mac Studio: 07's V4 captures `C00`–`C24` and class-(c) captures `X00`–`X09` match the SHA-256
  in 07's `census/KEPT_ON_MAC_STUDIO.txt`; 07a's census checkpoint is `461df2ec…`; A3 v2's scratch copy
  matches its certification.

---

## 0. What this prompt is

### 0.1 The leaf

07's V4 referenced 24 new stationary-point calls; five are converged and 2.21–3.16× their declared
error from agreeing readings. 07a's Part A attributed every call's error leaf by leaf, against
references on each leaf's own span (closure within 0.0033 of the tolerance on all 24 attributed calls):
- **In all five the whole excess is in the Levin leaves outside the window (H3)**; the window's own
  leaves are $10^{-9}$ to $5\times10^{-7}$ of abserr.
- **Almost all of it (share 0.999–1.008) is in one leaf that touches the window,** at its upper edge.
  Its true error is 2.66–7.44× its declared $e'$.
- **In v08, v12 and v17 that leaf is the seed region** from the window's edge to the end of the span
  (about $[9.23, 27\text{–}28]$, some $10^8$ rad of phase), solved once as a Levin leaf, checked, trusted,
  and never bisected. In v09 it is at depth 1; in v22 at depth 5.
- **It is not the window's stop,** which none of 07a's seven stop variants changed, and **not the
  gate's estimate** (O9): on the 120 regions round those leaves no route changes, and $\theta'$ changes
  strict sign on none.
- **The 20 honest calls** have leaves touching their windows too. Whether those declare honestly, or
  are merely small, is not known.

07a's census reproduced the five exactly (W4), and the cross-check's two pairs at $r_g > 1$,
`3-14501` and `3-35006`, both converged `G+q+r` calls (W6). Whether the pairs are this defect is not
measured.

### 0.2 How such a leaf is accepted

A leaf is checked one level down: with $V$ each solve's own value, $\delta_1 = |V_P - (V_L + V_R)|$,
$\delta_2 = |(V_L + V_R) - Q|$, $Q$ the four quarters' sum, and $r = \delta_2/\delta_1$. It is trusted
if $r < 1/4$, or if $r < 1$ and $\delta_1 \le \nu_B$, the noise yardstick. A trusted leaf declares
$\max({\rm residual}, \delta_1)/(1 - r)$ and returns $Q$. That bounds $Q$'s error **only if the error
falls geometrically from this level on**, at the rate observed, and **only if it is an error the
subdivisions differ in**. So the leaf can understate in three ways, which §2.2 tells apart:
- **(M1) A common error.** An error that $V_P$, $V_L + V_R$ and $Q$ share is invisible to $\delta_1$
  and $\delta_2$. The module docstring records one kind: near a stationary point every Levin solve
  lacks the stationary-phase term alike. A leaf touching a window begins $\Phi_w = 2\pi$ of phase from
  its point, and its left half and left quarter begin at the same edge. If the excess is a remnant of
  the stationary contribution beyond the window, it is common to every level, and it falls as $\Phi_w$
  grows.
- **(M2) A pre-asymptotic ratio.** $r < 1/4$ by coincidence, while the error is not yet falling
  geometrically: `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`'s mechanism, one level lower. It
  shows as true errors at the leaf, halves and quarters that do not fall as $r$ predicts, and as
  eighths that break the pattern.
- **(M3) An error in the inputs.** An error in the phase, its rate or the integrand at the window's
  edge enters every level's solve in the same way, so it is also common, but it does not fall with
  $\Phi_w$. It shows as sensitivity of the leaf's value to its inputs perturbed within their declared
  errors (`_CC_THETA_PRIME_REL_ERROR_SUPPLIED` declares $\theta'$ supplied to $2^{-10}$ relative), or
  to an input recomputed more precisely, where that is possible.

These are hypotheses, not a list to choose from. §2.2's records are fixed so that the measurement can
say "none of these", and the agent reports what it finds.

### 0.3 06b's under-declaration

`[06b-moved-calls-declare-less-than-their-error]`: on 24 group calls at `1e-7` that 06b's premise check
newly bisected, 17 declared less than their error against references $S_{10}$ (median 1.79×, worst
19.7×, `b-1e-7-3-34455` call 1). On two of them the driver at `rtol` $10^{-11}$ returned the same value
with the same regions, each resolved: **an error common to a region and its halves**, which the
residual cannot see. What it is was never diagnosed. 06c's code no longer showed the pattern on the
calls it checked, through its trust check. Two questions, which §2.1 item 4 answers:
- **Does it persist at 07a's code?** The 24 are re-scored from 07a's census against their committed
  references.
- **Is it H3's mechanism?** Any of them still dishonest goes through §2.2's checks with the five.

### 0.4 How honest is the census?

Every dishonest call found in 07 and 07a was found by a reference, never by the census's internal
checks. References have been taken only on chosen calls: V4's stationary-point calls, two rows at the
user's request. **No honesty rate has been measured on calls drawn at random.** §2.4 takes three sets
of references, once, at Part A: each reading depends on the integrand alone, so the same references
score 07a's code now and 07b's code in §6.
- **(R1) The class-(c) sample** the user ruled for this prompt: about 30 of the 396 rows where 07 and
  06m disagree beyond both errors, stratified. It decides there whether 07 or 06m is right, and scores
  07a's honesty on calls near stationary points.
- **(R2) A random draw from the whole census:** calls with nothing special about them, *"to give us a
  sense of what happens"*.
- **(R3) Targeted calls,** each for an open issue: W7's moved calls, the auxiliary-route calls, own
  route changes, and the two pairs.

### 0.5 What it does not do

- **Whether $6\pi$ is right** (`[07-c2-gate-threshold-was-never-measured]`): R3's own-class references
  are evidence for it; this prompt does not change the gate.
- **The calls at the depth cap** (G2's L2): 08's experiment.
- **A fix outside the leaf's mechanism.** If §2.2 finds the mechanism general (not tied to windows),
  §2.3 scores general variants, and the user chooses at the stop whether this prompt takes it or a later
  one does.

## 1. The code (Part B)

**Written at §2's stop, as §9's amendment,** from the user's choice. Whatever is chosen, §9 states:
1. **What changes,** in `AdaptiveLevin/levin_quadrature.py` only unless §9 says otherwise, with its
   reference implementation (Part A's variant) and its diff against the tree.
2. **Which leaves it touches,** and so which calls are predicted bit-identical to 07a's (§6 U1).
3. **Its basis:** §2's measurement, with every constant named and commented.
4. **What it does to `converged`.** A call whose referenced leaf declares less than it carries is not
   reported converged on that declaration.
5. **Docstrings and comments** that describe the trust check, the windows or the mechanism, rewritten
   to say what is now done and why, each listed by the agent with its line.

## 2. The measurement — Part A, on the Mac Studio, a planned stop

**Run at the dispatch SHA, whose code is 07a's.** Nothing in the tree changes. Variants are copies of
`levin_quadrature.py` in the agent's `scripts/variants/`, loaded by 07's `variant_loader.py` pattern,
each with its diff against the tree saved. Records go in `var/levin-phase-floor-07b/measurement/`, in
`scripts/`, `state/` and `out/`, every output carrying the SHA-256 of its script and of the module or
variant it ran.

### 2.1 The calls

1. **V4's 25 captures,** `var/levin-phase-floor-07/census/state/v4/capture/C00`–`C24` (v00–v23 and
   `cal_z12`), with their readings from 07's `census/out/v4/` (z12's is 06g's). Each is replayed at
   07a's code and checked bit for bit (value, abserr, `converged`, regions, depth) against **07a's**
   census record. The five failures are v08, v09, v12, v17 and v22.
2. **The two pairs,** `3-14501` and `3-35006`, index 1, at both rtols: 07a's Part A captures
   (`var/levin-phase-floor-07a/measurement/state/pairs_capture/`), checked against 07a's census.
3. **`1-34599`,** group call 2 at `1e-7`, at 07's route: 07a's census captured it with the parent's
   module (`var/levin-phase-floor-07a/census/state/refs/capture_parent/`). There a region with $V$ just
   under $6\pi$ went to the Levin solve, and the call was 4.03× its declared error: a second case of a
   trusted Levin leaf understating, away from any window.
4. **06b's 24 W2 calls,** first **re-scored from 07a's census checkpoint** against their committed
   references (`docs/levin-phase-floor/06b/census/out/w2_reference.json`; unit `b-1e-7-X-NNNNN` is 07a's
   `h-1e-7-X-NNNNN`, same call index), by 06b's rule, $|v - S_{10}|/{\rm abserr}$. Each one dishonest at
   07a's code is captured afresh from A3 v2's scratch copy by 07a's `make_capture.py`, checked against
   07a's census, and joins item 1's five in §2.2.

**A replay of 07a's code that does not reproduce its census record bit for bit is a stop.**

### 2.2 The mechanism

**The leaves:**
- in each of the five failures, every Levin leaf that touches a window (an end at a window's edge),
  the leaf carrying the excess first;
- in each of the 20 honest V4 calls and z12, every Levin leaf that touches a window (the controls);
- in item 3's call, the leaf carrying the excess;
- in each call item 4 adds, the leaf or leaves carrying the excess, found as 07a's Part A found them.

**For each, recorded at 07a's code:**
- $V_P$, $V_L + V_R$, $Q$, $\delta_1$, $\delta_2$, $r$; which clause trusted it ($r < 1/4$, or the noise
  clause with $\nu_B$); its residual, floor and declared $e'$; its depth, span, phase variation, and
  $|\theta'|$ at each end;
- **a reference on the leaf's span, on each half's and on each quarter's,** by a method independent of
  the code under test, with a stated accuracy at least 100× finer than the leaf's declared error where
  that is reachable, and the accuracy reached where it is not. The method is the agent's choice with
  its basis, as in 07a's Part A (`tools/filon_reference.py`'s routines on a sub-span);
- **from these, the true errors** of $V_P$, of $V_L + V_R$ and of $Q$, signed, against $\delta_1$,
  $\delta_2$ and $e'$.

**The tests, each a table:**
- **(M1)** whether the three true errors share sign and size (a common error), and where along the
  span the error lies: the true errors of the left and right halves and quarters separately. Then the
  leaf **replayed with the window's half-width at $4\pi$ and $8\pi$** (a variant changing only
  $\Phi_w$): whether the leaf's true error falls, and how, as the leaf starts further from the point.
- **(M2)** whether the true errors fall from leaf to halves to quarters at the rate $r$ predicts; and
  the eighths, solved and referenced, on the five's carrying leaves.
- **(M3)** the leaf's value with its inputs perturbed within their declared errors at the window's
  edge: $\theta'$ by $\pm 2^{-10}$ relative where supplied, and the phase at the end by its
  `theta_abserr` where declared; and an input recomputed more precisely where the agent can, with its
  method. The change in value against the leaf's true error.

**The cost is projected first,** from a pilot on v08's carrying leaf. With §2.4's references, a
projection of Part A beyond 36 hours is reported to the orchestrator before the rest runs, with what
it would drop to fit.

### 2.3 The variants

Each is a candidate fix, scored on every call of §2.1 and on every call of §2.4 that has a capture.
Each states, from its diff, which leaves it changes; a variant tied to windows shows on five calls with
no window that they run the tree's code.
- **(a) 07a as is** (the replays).
- **(b) A wider window:** $\Phi_w = 4\pi$, and $8\pi$ (two variants). 07 §2 set $2\pi$ by fixed rules at
  rtol $10^{-9}$; the window's cost is reported beside its effect.
- **(c) A deeper trust check for a leaf that touches a window:** trusted only on a check two levels down
  (the eighths), the rule the agent's, with its basis.
- **(d) An independent check for a leaf that touches a window:** its sub-span next to the window, of
  phase the agent chooses with its basis, also solved by the nested Clenshaw–Curtis pair; the leaf is
  untrusted if the two disagree beyond their declared errors.
- **(e), (f)** at most two more, from what §2.2 finds, each with its basis. A general variant (one not
  tied to windows), if §2.2 finds the mechanism general, is one of these.

### 2.4 The references

**Read by 07's V4 method,** as 07a's census ran it for `1-34599` and `3-678`: FCC at resolutions r1
and r2, with QAWO added where projected at 30 minutes or less; the readings agree under 07's rule; the
reference is the finest reading and its stated error the readings' spread. **One reference per
(row, call index) serves both rtols** where the two calls' integrands are shown bit-identical. Each
call is captured from A3 v2's scratch copy by `make_capture.py` and checked bit for bit against 07a's
census before it is read. Every draw below uses a seed fixed in `state/` before the draw, and the
selection is saved before any reading.

- **(R1) Class (c):** 30 of the 396 rows of 07's cross-check, stratified by band and $k$ in proportion,
  each its stationary-point call at `1e-8` (07's `xcheck_capture_selection.json`; the captures
  `X00`–`X09` are 07's code's, so each is re-checked against 07a's census, or captured afresh).
- **(R2) The whole census:** 30 calls drawn uniformly at random from every Levin call of 07a's census
  at both rtols, group and analytic. The draw is not stratified; the strata it lands in (kind, band,
  $k$, rtol, class) are reported. A projection beyond §2.2's budget may cut R2, to no fewer than 20.
- **(R3) Targeted:**
  - **W7** (`[07-iroff1-window-stop-can-leave-a-call-dishonest]`): `h-1e-8-0-33950` (0.145) and
    `h-1e-8-0-2006` (0.137), the six-minimum calls at $k = 3.14\times10^5$ that moved most; and the
    nineteen-minimum call at $k = 10^5$ that moved most (`out/scores/w7_detail.json`). Each scored at
    07a's and at 07's code.
  - **Auxiliary routes** (`[07a-trust-check-auxiliary-route-moves-the-declared-error]`): `3-843`'s
    analytic J2 at `1e-7` and `3-13508`'s `G+q-r` at `1e-7`, which moved by more than 07's abserr;
    and the four of the 47 shrinkers whose abserr shrank most, other than `3-678`
    (`summary.json`, `aux_only`). Each scored at both codes.
  - **Own route changes** (`[07-c2-gate-threshold-was-never-measured]`): three calls drawn at random
    from the own-class calls whose value moved, other than `1-34599`. Each scored at both codes, with
    the region whose route changed, its $V$ and 07's estimate.
  - **The two pairs,** `3-14501` and `3-35006` index 1: one reference each, scoring both rtols.

**The cost is projected** with 07a's refitted model (`refs_projection_07a.py`) before any reading,
within §2.2's budget.

### 2.5 The tables, fixed before the run

1. **Honesty at 07a's code** of every referenced call, $|v - {\rm ref}|/{\rm abserr}$ with `converged`,
   in four groups: V4's 25 (07a's Part A readings), R1, R2 and R3; and 06b's 24 against $S_{10}$,
   beside their ratios at 06b's code. For R1, which of 07 and 06m is nearer.
2. **The mechanism,** per leaf: §2.2's records and its three tests, the five, item 3, item 4's calls and
   the controls in separate blocks.
3. **The variants:** for every call and variant, value, abserr, `converged`, regions, maximum depth and
   wall time; honesty where referenced; and for the pairs, $r_g$.
4. **A variant qualifies** if no referenced call is converged and dishonest under it, and each pair has
   $r_g \le 1$ or an unconverged member. Among those that qualify, the cost against (a): calls
   unconverged, regions and wall time.

### 2.6 Stop and report

The agent reports §2.5's tables, which variants qualify, what §2.2 finds the mechanism to be (or that
it is none of §0.2's), and anything else it found, and stops. **It recommends nothing.** Its log, in
README §5.1's form, is written before it reports, with every deviation numbered and classified. The
user chooses; the orchestrator writes §1, §3 and the predictions' additions from that choice as §9's
amendment.

## 3. Tests (Part B)

Written with §1 at §2's stop (§9). They include, whatever the fix:
1. **A synthetic problem that reproduces the mechanism** §2.2 found, with a closed-form or
   independently computed answer: dishonest and converged at the parent, honest or unconverged at the
   commit.
2. **A problem the fix must not touch,** bit-identical at the parent and the commit.
3. **Deliberate breakages,** each caught by the test named for it, each re-applied and reverted.
4. **Every changed existing assertion listed,** with why.
5. **The LiouvilleGreen draws** (07a's `docs/levin-phase-floor/07a/code/lg_draws_07a.py`, run at the
   parent and the commit): no test newly failing, and every call the fix should not touch bit-identical.

## 4. Records of the code (Part B)

The log `logs/07b-levin-leaf-beside-a-window.md` in README §5.1's form, with §5's and §6's predictions
(and §9's additions) copied in, marked not yet scored; the board's 07b row; the LiouvilleGreen draws'
harness and outputs in `docs/levin-phase-floor/07b/code/`. **No issue moves in the code commit:** they
move in 07b's records commit after Part C.

## 5. Rehearsal R31 — on the Mac Studio, after the code commit is pushed

As 07a §5, with records under `var/levin-phase-floor-07b/rehearsal/`, slug `r31-07b`, `--prompt 07b`,
`--job-name levin-phase-floor-07b`, and the fingerprint `1b332ffa…`. The parent rehearsal is **R30**
(`levin-phase-floor-07a-r30-07a-20261112T113806`, `var/datastores/levin-phase-floor-07a-r30*`, kept),
built at `4bf50d34`, whose code is `1ec734d7`'s.
1. **S0′ — R30 is the parent.** `git diff --stat 1ec734d7 <code commit's parent> -- . ':!docs'
   ':!prompts'` lists only `CLAUDE.md` and `89c368f2`'s four files, each AST-equal with docstrings
   removed. **Holds** if so; if not, it stops and asks.
2. **The lock:** `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command
   of its own.
3. **R31 against R30,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py` (job
   names normalised in a copy): outside `QuadSourceIntegral` every value table identical, every other
   difference of the four volatile kinds, no row in one store only; inside it, every moved row
   tabulated, with the rows whose `total_converged` changes in each direction.
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{31} - {\rm total}_{30}| \le {\rm total\_abserr}_{30} + {\rm total\_abserr}_{31}$, or is
   one §9 predicts to move further, listed.
5. **S1, S2 and S4** re-scored on R31 as 07a §5 items 5 and 6: V1's twelve against 06g's references,
   P1's two rows against 07's `p1_references`, and R29's four blocks against their block references.
6. **The readers** run on R31, exit 0, and the store is byte-identical before and after. **The lock** is
   released and `ray-processes` lists nothing. **R30 and R31 are kept.**

## 6. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

07a's census harness (`docs/levin-phase-floor/07a/census/scripts/`), copied, changed where marked
"prompt 07b", re-run at the code commit against 07a's census (`levin-phase-floor-07a-census-20261112T132744`,
checkpoint `461df2ec…`) as the "before". Diagnostics as §9 sets them, recording on every call whether
the fix acted and where.

**Gate U0, before the run:**
- **The parent's code is 07a's:** as S0′.
- **The harness reproduces 07a's census at the parent,** bit for bit, on 06k's 100 K0 units. One
  registered run, slug `k0`.
- **The instrument changes nothing:** a unit where the fix acts and one where it does not, at the new
  code, with and without it, bit-identical.
- **The new code is deterministic:** two runs of one unit where the fix acts, bit-identical.

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md` unchanged,
with §9's additions:
- **U1 — nothing else moves.** Every call in which the fix does not act equals 07a's call bit for bit,
  in the call's outputs. Every row all of whose calls are such equals 07a's in `total` and
  `total_abserr`. **Holds** if true at both rtols.
- **U2 — no Levin leaf is returned across a stationary point** (07a's W2).
- **U3 — every phase with something to locate is located** (07a's W3).
- **U4 — honest where it is referenced.** Every call referenced in Part A (V4's 25 and 07a's census's
  other eight V4 calls, R1, R2, R3, and `1-34599` and `3-678`), scored from the census against its
  reference by 07's rule. **Holds** if every one is honest, or unconverged under §9's rule.
- **U5 — the gate is consistent, and the guard is a backstop** (07a's W5, on gated solves).
- **U6 — 06b's 24 are honest,** against $S_{10}$ by 06b's rule, or unconverged.
- **U7 — the pairs:** $r_g$ over all 2,484 pairs; every pair has $r_g \le 1$, or a member unconverged.

**Descriptive, not scored:** the calls where the fix acts, by class, band, $k$ and rtol; `converged`
changes in each direction; the abserr ratio against 07a's; regions and cost against 07a's.

## 7. Acceptance

- **Part A:** §2.5's tables complete and its stop reached, with every replay reproducing 07a's census.
- **Part B:** §1 as amended; every test of §3 passing and every breakage caught; the seven roots
  rising only by the tests added; the LiouvilleGreen draws with no new failure.
- **Part C:** §5's S0′–S4 and §6's U0 hold; U1–U7 scored, each a verdict against its rule.

## 8. Stop conditions — stop and ask

- A replay at 07a's code that does not reproduce its census record.
- A capture that does not check against 07a's census.
- Part A's projection beyond 36 hours, before the rest runs.
- §2.2 finding that the leaves' references do not close against the call's reference (as 07a's Part A
  checked closure).
- In Part B: an acceptance test that cannot pass without a change outside §1 as amended, or a
  LiouvilleGreen test failing at the commit that passes at the parent.
- In Part C: S0′ failing; a call U1 says is untouched differing from 07a's, before anything else is
  scored; a lock held.

## 9. Amendments at §2's stop

*Written by the orchestrator from the user's choice at Part A's stop.*
