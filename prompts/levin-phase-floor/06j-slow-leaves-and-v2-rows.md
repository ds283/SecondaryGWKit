# Prompt 06j — where 06g's slow leaves come from, and which code is right on V2's rows (a measurement)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-03, on the user's decisions after 06g's Part B (the board's Decisions,
**06j**):
- on the slow leaves at $r \approx 1/2$: *"So yes, we want to the measurement for the
  `r \approx 1/2` leaves."*;
- on V2: *"V2 - OK, I agree with the plant to compute these with QAWO and FCC and check which (if
  any) agrees. We need the measurement prompt 06j, but don't dispatch it"*;
- on further references for r24, r26, r47 and r53: *"I don't see a good rationale to continue"*.
  The three stopped QAWO readings are not rerun.

It runs **before any remedy prompt** for the two issues it measures, and before 07 resumes. Its
results decide whether a remedy prompt (06k) is written, and what it does.

**Where it runs:** on the Mac Studio, Ray-free, with no lock, on 06's certified scratch copy of
A3 v2, read-only. **It changes no code** and builds no store. One agent, dispatched and reviewed by
an orchestrator session on the Mac Studio; the records are committed on the laptop.

**Closes:** nothing. It measures `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`
(Part S) and `[06g-g-plus-r-rows-move-past-both-errors]` (Part W), and states for each what a
remedy would have to do.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries for the two issues above, and
   `[06e-node-errors-add-in-absolute-value]` and `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_adaptive_levin`: the trust check one level down ($\Delta_1$, $\Delta_2$, $r$), rule (b)
     (`_LEVIN_TRUST_MAX_RATIO`), and how an untrusted leaf is refined and returned;
   - the Levin leaf's floor `phase_err` ($\phi_P$), and the Clenshaw–Curtis fallback's floor and
     node term.
3. `tools/filon_reference.py` at this prompt's tree: Route B, `production_edges` (the integrand's
   representation seams), the direct segments, and the count rule.
4. 06g's records, `docs/levin-phase-floor/06g/`:
   - `census/issues.md` C1–C5 and C7, `census/out/v8.md`, `census/out/v8_slow_leaves.md`, and
     `census/out/cmd/51_values_beyond.txt`;
   - `census/scripts/` (`harness.py`, `worker.py`, `x1_capture.py`, `v8_slow_leaves.py`) and
     `references/scripts/` (`make_capture.py`, `check_captures.py`, `one_reading.py`, `run_refs.py`);
   - `references/log.md` and `review.md`, for how the references were made and what went wrong
     with their backstops;
   - `KEPT_ON_MAC_STUDIO.md`, for the census checkpoint this prompt reads.

**Gate:**
- 06g's records commit (`fdc3a2b`) is on `origin/handover-remedial`;
- the Mac Studio's main checkout is clean, and `runregistrykit list` shows nothing `running` or
  stale;
- the scratch copy matches its certification by SHA-256, and its fingerprint is `0e32f24a…`;
- 06g's census checkpoint is `03adcbba…`, and 06h's is `3104c18e…`;
- 06g's census outputs that Part S reads, `census/out/v8_calls.json` (`1e9a62b0…`) and
  `census/out/v8_slow_leaves.json` (`b4643033…`), match their SHA-256.

---

## 0. What this prompt is

### 0.1 Part S: the slow leaves

Under 06g's rule (b), a checked leaf at $1/4 \le r < 1$ with $\Delta_1 > \phi_P$ is not trusted:
it is refined, and at `depth_max` returned untrusted, the call unconverged. Outside the 720
stationary-point rows, 06g's census lost convergence on 801 calls at `1e-7` and 827 at `1e-8`
this way, every one a `G±q±r` group call that meets its request on abserr. 966 and 996 group calls
reached `depth_max`, and the census took 2.2× as long as 06h's. On 12 of these calls,
referenced, 06g's value is within its error on 11.

The scoped slow leaves (`census/out/v8_slow_leaves.md`) have three properties that suggest a cause
other than slow spectral convergence:
- **$r$ clusters at 1/2:** of 4,342 at `1e-7`, 1,763 lie within 0.01 of it, and along recorded
  chains $r$ settles near 1/2 over depths 16–19;
- **they sit at fixed points:** a median of 2 clusters per call, none at the call's ends, the
  cluster centres weighted to the later part of the interval;
- **they are near the floor:** $\Delta_1/\phi_P$ has median 2.27, and 1.5 for $r < 1/2$.

**Two explanations, which this part distinguishes:**
- **(J) a non-smooth point.** A leaf holding a finite jump in the integrand, or in a derivative,
  converges algebraically. With a jump in the value its error is proportional to its width, so it
  halves with each halving and $r \to 1/2$ exactly. The integrand is built from representations
  that change at known points (`production_edges`: the phase splines' knots, the Green's function
  amplitude spline's knots, the cumulative tables' nodes, switch midpoints and breaks). A leaf
  holding one of these, if the representation jumps there, would behave like this.
- **(N) rounding noise that scales with the leaf's width.** A rounding error committed per node,
  such as the node displacement 06e's node term models, enters a leaf's value through weights of
  order $h$, so it too can halve with the leaf. If $\phi_P$ understates it by a small factor, the
  leaf sits just above its floor ($\Delta_1/\phi_P \sim 2$) at $r \approx 1/2$.

The two call for different remedies. (J) would be met by giving Levin the seams as breakpoints, so
that no leaf straddles one; (N) by a floor that covers the noise. Both may be present.

### 0.2 Part W: V2's rows

06g's V2 compared each census row's `total` under 06g and under 06h. On 5 rows at each rtol they
differ by more than the sum of their declared errors (worst 2.164, `0-12959`), so at least one code
is dishonest on each. In every such row the difference is in the `G+r` call, index 1, which 06h
integrates with 2 regions at depth 1 and 06g with 12–16 at depth 5. At call level, 62 calls in 59
rows (`1e-7`) and 80 in 75 rows (`1e-8`) differ by more than both errors, all group calls, all on a
different partition (`census/out/cmd/51_values_beyond.txt`). The `G+r` calls among them move by
nearly the same amount on every row, $7.0$–$7.2\times10^{-33}$.

06g's references found the parent the one in error wherever both codes were compared against a
reference: on V1's 12 calls, and on 3 of the 12 sampled V8 calls. Part W asks the same question of
V2's calls, with the same tool.

### 0.3 What it does not do

- **No code change.** Nothing under `AdaptiveLevin/`, `ComputeTargets/` (outside `tests/`),
  `LiouvilleGreen/`, `CosmologyModels/`, `ObjectFactories/`, `config/`, `tools/` or `main.py`
  changes, and no test file is edited (README §5 rule 1). Instrumentation is by wrapping a module
  function in the harness's own process, as 06e's and 06f's replays did.
- **No counterfactual driver.** It does not re-run the driver with a different rule. A leaf may be
  probed in isolation (its rule evaluated on its span, on halves, on sub-spans, with inputs
  perturbed); each such probe says what it is.
- **Not the stationary-point rows** (07), and not a remedy.

## 1. The harness

Records go under `var/levin-phase-floor-06j/{out,scripts,state}`, with `log.md`, `verification.md`
and `issues.md` beside them (README §4.1).
- **The tree** is a worktree at this prompt's dispatch SHA. Its production code is `1c5336e`'s
  (06g's, `3ae6f3a`), and its `tools/filon_reference.py` is `b5626ecb`'s: `git diff --stat 1c5336e
  <dispatch SHA> -- . ':!docs' ':!prompts'` shows only `tools/filon_reference.py`, its test and
  `CLAUDE.md`.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after.
- **The scripts** are copied from `docs/levin-phase-floor/06g/census/scripts/` and
  `docs/levin-phase-floor/06g/references/scripts/` into `var/levin-phase-floor-06j/scripts/`. Save
  the diff against 06g's copies, and record each change as a deviation.
- **Captures.** Each sampled call is captured at the tree as 06g's references captured theirs, and
  checked against 06g's census record bit for bit: value, abserr, regions, maximum depth and
  converged. 06h's values come from 06h's census checkpoint.
- **Long runs.** Anything that may exceed about ten minutes is registered with `runregistrykit`
  (`--campaign levin-phase-floor --prompt 06j`) and launched detached (README §5 rule 7).
- **Reference readings** follow 06g's practice as the user ruled it there:
  - QAWO and FCC at resolutions 1 and 2. A Route B reading is usable when `tmap.all_monotone` is
    True, or when every non-monotone segment is integrated directly (`direct_segments`).
  - **Cost** is projected from each call's phase span (QAWO about 0.06–0.10 µs per radian at
    resolution 1 on six workers, about 4× that at resolution 2), not from its segment count.
  - **A long reading is checked, not cut off.** A reading past 30 minutes, or past its projection,
    is reported with its progress and projected finish. It is stopped only if the projection is a
    day or more, and the projection counts the chunks in flight, not only those finished.
  - **Every reference is scored** with the readings it has, its readings and their spread stated.
    Most weight goes to references with two implementations.

**Gate J0, before scoring:**
- **The captures reproduce 06g's census.** Each sampled call's replay equals its census record bit
  for bit.
- **The tool reproduces.** One reading from 06g's references, a Route B QAWO r1 of a monotone call,
  re-read at the tree, is bit-identical.
- **The replay is deterministic.** One Part S call twice in fresh processes, bit-identical down to
  each returned leaf's value and $r$.

A call that fails J0 is reported and replaced from the same stratum. If more than three fail, stop.

## 2. Part S — the slow leaves

**The sample,** drawn by a seeded script whose seed, rule and output are written before any probe
is run:
- 24 of V8's lost calls outside the 720, 12 at each rtol, spread over the `G±q±r` groups that have
  them, and over the number of slow-leaf clusters per call;
- in each, every scoped slow leaf (`v8_slow_leaves.py`'s scope: returned, untrusted,
  $1/4 \le r < 1$, $\Delta_1 > \phi_P$), up to 10 per call, chosen across its clusters.

**S-a. Where the leaves sit.** For each sampled call, compute `production_edges` on its capture,
with each seam's kind: phase spline knot, amplitude spline knot, table node, switch midpoint or
break. For each sampled leaf, record:
- whether its span $[a, b]$ holds a seam, and of which kind;
- otherwise, its distance to the nearest seam in units of its width.

**The null:** the fraction of depth-20 positions in the same calls that would hold a seam if leaves
were placed uniformly at random, from the seams' count and the leaves' width. Also report the same
counts for the call's **trusted** returned leaves at depth 15 and above, as a comparison.

**S-b. Whether $r$ stays at 1/2.** For each sampled leaf, probe it in isolation: evaluate the
driver's leaf rule on the leaf's span and on its descendants, for at least four further halvings
(depths 21–24), recording $\Delta_1$, $\Delta_2$, $r$ and $\phi_P$ at each. Do this along the
branch that keeps the seam, if there is one, and along a branch that does not. Under (J), $r$ stays
near 1/2 on the branch holding the seam, and falls on the other.

**S-c. Whether the leaf is noise.** For each sampled leaf, measure the scatter of its value under
rounding-level perturbation:
- re-evaluate the leaf's rule with its nodes moved by a few ulp, by a seeded rule; and
- separately, with the integrand's inputs at the nodes moved by a few ulp.

Each set has at least 16 draws. Report the scatter's rms against $\Delta_1$ and against $\phi_P$.
Under (N), $\Delta_1$ is of the scatter's size; under (J), $\Delta_1$ exceeds it by far.

**S-d. Splitting at the seam.** For each sampled leaf whose span holds a seam, evaluate the leaf
rule on the two sub-spans either side of the seam, with the trust check one level down on each.
Record $r$, $\Delta_1/\phi_P$ and whether each side would be trusted under rule (b). Under (J), both
sides converge spectrally.

**S-e. What the seam does.** At each seam held by a sampled leaf, measure the integrand's
one-sided limits, its value and its first two derivatives by one-sided finite differences at
several step sizes, stated, and report any jump against its size. Name the representation that
changes there.

## 3. Part W — V2's rows

**The sample,** written before any reading:
- **W-5:** the 5 rows at each rtol that fail V2 on `total`, their `G+r` call, index 1. If a row's
  integrand at `1e-7` equals its integrand at `1e-8` (as 06g's z00 and z12 were checked: route,
  span, signs, each factor's phase spline and leading table by bytes, and the integrand at 65
  points), one reference serves both, and the equality is recorded.
- **W-12:** 12 further calls from the call-level set (62 at `1e-7`, 80 at `1e-8`), 6 at each rtol,
  chosen at evenly spaced quantiles of their ratio $|v_g - v_{06h}|/(e_g + e_{06h})$ by a rule
  written before any reading, spread over the groups that have them.

**For each call:**
- **The reference:** four readings, their spread against the count tolerance (1/100 of the smaller
  declared error), and whether they agree within it.
- **The codes against it:** $|v_g - {\rm ref}|/e_g$ and $|v_{06h} - {\rm ref}|/e_{06h}$, with each
  code's regions, depth and converged flag.
- **What the codes returned:** under 06g, the call's returned leaves, with depth, $r$, trusted
  flag and whether it holds a seam (S-a's rule). Under 06h, the same where its census records them.
- **The constant move.** On the `G+r` calls, whether the near-constant $|v_g - v_{06h}| \approx
  7.1\times10^{-33}$ is localised. That is, whether one returned leaf or one sub-span of the call
  carries it, located by comparing the two codes' partial sums over a common refinement of their
  partitions.

## 4. Predictions, fixed before the run

README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
**undecided** against the rule as written. None is a stop.

**Part S:**
- **S1 — the slow leaves sit on seams.** **Rule:** at least 80 % of the sampled slow leaves hold a
  seam in their span, and the fraction is at least five times the null's.
- **S2 — $r$ stays at 1/2 where the seam is kept.** **Rule:** on at least 80 % of the sampled leaves
  that hold a seam, $r$ stays within $[0.4, 0.6]$ at every one of depths 21–24 along the branch
  holding the seam.
- **S3 — the slow leaves are not noise.** **Rule:** on at least 80 % of the sampled leaves,
  $\Delta_1$ exceeds ten times the larger of S-c's two scatters.
- **S4 — splitting at the seam restores trust.** **Rule:** on at least 80 % of the sampled leaves
  that hold a seam, both sides of the split would be trusted under rule (b).

**Part W:**
- **W1 — the references count.** **Rule:** on every W call, the four readings agree within the
  count tolerance.
- **W2 — 06g is honest on V2's rows.** **Rule:** on every W-5 call, $|v_g - {\rm ref}| \le e_g$.
- **W3 — the parent is the one in error.** **Rule:** on every W-5 call, $|v_{06h} - {\rm ref}| > e_{06h}$.
- **W4 — the pattern is general.** **Rule:** over the W-12 calls, 06g is within its error on at
  least 11, and 06h is outside its error on at least 6.

**The basis of the 80 % and the factors.** S1–S4 are each read as a test of (J) against (N), and
80 % leaves room for a minority of leaves of the other kind. The factor of five over the null and
of ten over the scatter are chosen generously, so that a pass is not a near-miss: README §5's
practice for a cut-off that decides between remedies. Every number behind each verdict is reported,
so that a different reading can be made from the record.

## 5. Records

- **`var/levin-phase-floor-06j/log.md`** in README §5.1's form, with every deviation classified;
  `verification.md`; `issues.md` with any observation proposed as an issue. The laptop orchestrator
  copies the log to `prompts/levin-phase-floor/logs/06j-slow-leaves-and-v2-rows.md`.
- **`out/`:** the samples and their rules, J0's checks, per-leaf and per-call tables (CSV or JSON),
  every reading, and a `results.md`. The results state each verdict with its numbers and, for each
  part, what a remedy would have to do on this evidence. They propose no code.
- **Provenance** in every output header (README §5 rule 5).
- **Kept on the Mac Studio:** the captured pickles and anything over 2 MB, listed with SHA-256.

## 6. Acceptance

1. J0 holds, with every replacement listed.
2. S1–S4 and W1–W4 each scored against the rule as written, with their numbers.
3. S-a to S-e complete for every sampled leaf, and §3's tables complete for every W call.
4. The store and both census checkpoints unchanged by SHA-256; nothing created under
   `var/datastores/`; no Ray started.
5. **Records:** the log, `verification.md`, `issues.md` and `out/results.md`;
   `docs/levin-phase-floor/VERIFICATION.md` §6j, written by the laptop orchestrator; the records in
   `docs/levin-phase-floor/06j/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **J0** fails on more than three calls, or the tool's re-read is not bit-identical.
- **A measurement cannot be made as written:** a seam cannot be located on a call's capture, a
  leaf's rule cannot be evaluated in isolation without a code change, or S-c's perturbation cannot
  be applied by wrapping. Report what was tried.
- **A W call's readings cannot be made usable**: a Route B reading is neither all monotone nor
  covered by direct segments, or a reading's projection is a day or more.
- **A code change** looks necessary to measure something. Report it; do not make it.
- **Any store file** changes, or Ray is needed.
