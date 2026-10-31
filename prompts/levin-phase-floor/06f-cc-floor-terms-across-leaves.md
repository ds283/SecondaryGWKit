# Prompt 06f — how two Clenshaw–Curtis floor terms add up across a call's leaves (a measurement)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-31, on the user's decisions after 06e's Part B (the board's Decisions, **06e
follow-ups**):
- on the census's Z2 call: *"Z2: OK, measure it."*;
- on its D4: *"D4: OK, measure it."*;
- on their shape: *"I prefer a separate measurement prompt. Can we do Z2 and D4 together?"*

It runs **before 06g**, which fixes the trust check's value (`[06e-trusted-leaf-returns-its-coarsest-value]`,
with sample 3 of 06e's Z3) and is held until this prompt's results are in, and before 07 resumes.
If either measurement here calls for a code change, the user decides whether it joins 06g before
06g's census.

**Where it runs:** on the Mac Studio, Ray-free, with no lock, on 06's certified scratch copy of
A3 v2, read-only. **It changes no code** and builds no store. One agent, dispatched and reviewed by
an orchestrator session on the Mac Studio; the records are committed on the laptop.

**Closes:** nothing. It narrows `[06e-node-errors-add-in-absolute-value]` and
`[06e-cc-declared-phase-term-is-a-crude-proxy]` with measurements, and states for each what a
remedy would have to do.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries `[06e-node-errors-add-in-absolute-value]`,
   `[06e-cc-declared-phase-term-is-a-crude-proxy]` and
   `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_cc_node_term_parts` (returns $E$ signed, $R_g$, $R_2$, $R_c$), `_cc_roundoff_floor` and how
     `_adaptive_levin_subregion_cc` forms both;
   - `_phase_error`, `_declared_endpoint_phase_err`, the comment above `endpoint_proxy` in
     `_adaptive_levin_subregion_cc`, and the Levin branch's use of $|p|$ at each endpoint;
   - `_adaptive_levin`: how a leaf's error and the call's error are summed, and the floor-limited
     test.
3. `LiouvilleGreen/three_bessel_integrals.py`, `BesselPhaseGroup.theta_abserr` and `levin_theta`,
   and `LiouvilleGreen/bessel_phase.py`, `BesselPhaseFunction.theta_abserr_at`: what the analytic
   blocks declare, and why the constituents are summed linearly.
4. 06e's census records, `docs/levin-phase-floor/06e/`:
   - `census/issues.md` D1 and D4, and `census/out/cmd/55_report.txt` and `58_z2_call_detail.txt`;
   - `census/scripts/worker.py`'s `_cc_floor_terms`, which reproduces each Clenshaw–Curtis leaf's
     floor terms with $E$ signed, and `x1_capture.py`, which captures chosen calls and checks each
     against the census bit for bit;
   - `census/references/replay_g.py` and `investigations/d3-leaf-values/d3_probe.py`, which replay a
     captured call with a module function wrapped;
   - `KEPT_ON_MAC_STUDIO.md`, for the census checkpoint (`a0dc4d7f…`) this prompt samples from.

**Gate:**
- 06e's records commit has landed (the board's 06e row is ✅);
- the Mac Studio's main checkout is clean, and `runregistrykit list` shows nothing `running` or
  stale;
- the scratch copy matches its certification by SHA-256, and its fingerprint is `0e32f24a…`;
- 06e's census checkpoint is `a0dc4d7f…`.

---

## 0. What this prompt is

### 0.1 The two terms

Both are parts of a Clenshaw–Curtis leaf's floor $\phi_{\rm CC}$. A call's declared error is the sum
of its leaves' errors, and a floor-limited leaf's error is its floor. So each term below enters the
call **as a sum over leaves of a per-leaf absolute value**.

**(Z2) The node term.** 06e's $\phi_{\rm node} = |E| + R_g + R_2 + R_c$, where
$E = \sum_k w_k\,g'(\hat x_k)\,\Delta_k$ is the signed first-order error that the rounded nodes
commit. On 06e's census one call outside the stationary-point rows reached depth 20:
`g-1e-8-0-33995` call 5 (`G−q+r`, $k = r = 3.14\times10^5$, $q = 10^5$, $z_{\rm response} =
4.809\times10^6$). It ended unconverged with 8,706 regions, 8,600 of them Clenshaw–Curtis. Its floor
was $1.333\times10^{-33}$ against an atol of $6.69\times10^{-34}$, and $1.318\times10^{-33}$ of that
was the node term, $1.312\times10^{-33}$ of it $|E|$. Its declared-phase term was 0.

**(D4) The declared-phase term.** Where the caller supplies `theta_abserr` (the analytic blocks
do, through `BesselPhaseGroup.levin_theta`), each leaf adds
$\texttt{theta\_abserr}(a)\,P_a + \texttt{theta\_abserr}(b)\,P_b$. On a Levin leaf $P$ is $\sum_i|p_i|$
at that endpoint. On a Clenshaw–Curtis leaf it is the proxy $\tfrac12 f_{\rm scale}\,h$ at each end,
so the pair is $\texttt{theta\_abserr}\cdot f_{\rm scale}\,h$, a sup-norm bound on
$\int|f|\,\delta\theta$. On 06e's census this term is 99.4 % of the summed floor over the
floor-attributed Clenshaw–Curtis leaves (5,628 calls at `1e-7`, 7,326 at `1e-8`). The sum is
weighted to the largest calls: per call it is the largest floor term on 2,493 and 2,785 calls,
against 2,644 and 3,216 for $(n+4)uS$ and 491 and 1,323 for the node term.

### 0.2 Why one prompt: the error each stands for is one signed quantity

**The node errors.** The call's value is the sum of its leaves' values. So to first order its node
error is $\sum_{\rm leaves} E$, signed, and the remainders $R_g$, $R_2$, $R_c$ bound what is left.
The floor carries $\sum|E|$ instead. The $\Delta_k$ are set by the spacing of doubles near $x$, not
by a leaf's width, so $\sum|E|$ does not fall when a leaf is split. If the signs of $E$ are effectively
random from leaf to leaf, $|\sum E|$ is of order $\sqrt N$ of a typical term and $\sum |E|$ of
order $N$. On Z2's call, $N = 8{,}600$.

**The declared phase error.** If the phase carries an error $\delta\theta(x)$, the call's value
moves by $\delta I \approx \int f\,\delta\theta\,b'(\theta)$, with $b = (\sin, \cos)$. The
constituents' errors are summed linearly because they share a construction and drift together
(`BesselPhaseGroup.theta_abserr`'s docstring), so $\delta\theta$ is smooth. A smooth $\delta\theta$
under an oscillating integrand largely cancels, and on the Levin branch the endpoint terms at a
breakpoint shared by two leaves cancel between them (`_phase_error`'s docstring says the step-(4)
residual cannot see them for the same reason). The floor adds each leaf's worst case instead, and
on a Clenshaw–Curtis leaf it takes $\int|f|$.

So each question has two parts: **is the per-leaf term the right size, and is adding the per-leaf
terms in absolute value the right combination?** This prompt measures both, for both terms. It
does not design a remedy.

### 0.3 What it does not do

- **No code change.** Nothing under `AdaptiveLevin/`, `ComputeTargets/` (outside `tests/`),
  `LiouvilleGreen/`, `CosmologyModels/`, `ObjectFactories/`, `config/` or `main.py` changes, and no
  test file is edited (README §5 rule 1). Instrumentation is by wrapping a module function in the
  harness's own process.
- **No counterfactual driver.** It does not re-run the driver with a different floor. A statement
  of what a call would have done under another combination is made on the call's returned
  partition, and says so.
- **Not the trust check** (06g), the stationary-point rows (07), the nested Clenshaw–Curtis estimate
  (`[06d-cc-nested-estimate-measures-the-coarse-rule]`), or how `theta_abserr` is constructed.

## 1. The harness

Records go under `var/levin-phase-floor-06f/{out,scripts,state}`, with `log.md`, `verification.md`
and `issues.md` beside them (README §4.1).
- **The tree** $g$ is a worktree at this prompt's dispatch SHA, whose code is `f4d8e17`'s (06e's).
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after. Nothing else in that
  directory is touched.
- **The scripts** are copied from `docs/levin-phase-floor/06e/census/scripts/` (and
  `census/references/` where used) into `var/levin-phase-floor-06f/scripts/`. Save the diff against
  06e's copies, and record each change as a deviation.
- **The population** is read from 06e's census checkpoint, `a0dc4d7f…`, and its per-call
  `cc_floor` sums. A call is **node-dominated** if the node term is its largest floor term among
  $(n+4)uS$, argument, node and declared, as 06e's report counts it, and **declared-dominated** if
  the declared term is.
- **The samples,** drawn by a seeded script whose seed and output are recorded:
  - **M1:** Z2's call, and 20 node-dominated calls outside 06a's 720 stationary-point rows, 10 at
    each rtol. Of each 10, at least 5 have 100 or more Clenshaw–Curtis leaves, if that many exist.
  - **M2:** 20 declared-dominated calls outside the 720, 10 at each rtol, spread over the analytic
    blocks that have them.
- **Capture and replay.** Each sampled call is captured as `x1_capture.py` captures, at $g$, and
  replayed in one process with `_cc_node_term_parts`, `_cc_roundoff_floor` and
  `_declared_endpoint_phase_err` wrapped, as `_cc_floor_terms` does. Per **returned** leaf (not the
  trial halves and quarters, which `_adaptive_levin` also solves), record its branch, span, depth,
  value, floor-limited flag, $E$ signed, $R_g$, $R_2$, $R_c$ and the declared term, split by
  endpoint.
- **Long runs.** Anything that may exceed about ten minutes is registered with `runregistrykit`
  (`--campaign levin-phase-floor --prompt 06f`) and launched detached (README §5 rule 7).

**Gate F0, before scoring:**
- **The captures reproduce 06e.** Each sampled call's replay equals 06e's census record bit for
  bit: value, abserr, regions, maximum depth and converged.
- **The terms reproduce.** Over each call's returned Clenshaw–Curtis leaves, the summed terms equal
  the checkpoint's `cc_floor` sums.
- **The replay is deterministic.** Z2's call twice in fresh processes, bit-identical.

A call that fails F0 is reported and replaced from the same stratum. If more than three fail, stop.

## 2. M1 — the node errors

On each M1 call, over its returned Clenshaw–Curtis leaves, tabulate:
- $N_{\rm CC}$, $\sum|E|$, $|\sum E|$, their ratio, and $\sqrt{N_{\rm CC}}$;
- $\sum(R_g + R_2 + R_c)$;
- the call's request $\max(\texttt{atol}, \texttt{rtol}\,|v|)$ and its declared error;
- **the signed-combination error on the returned partition:** the declared error with
  $\sum|E|$ replaced by $|\sum E|$, every other term unchanged. Also the same with $|E|$ replaced
  per leaf by $|\sum_{\rm sibling\ pair} E|$, to show how much of the cancellation is local.

For Z2's call, also by depth: $\sum|E|$ and $|\sum E|$ over the leaves at each depth, and how
$\sum|E|$ over a subtree compares with its root leaf's $|E|$ where both exist (from the trial solves
`_adaptive_levin` made), to show whether splitting leaves $\sum|E|$ flat.

**Descriptive:** the distribution of $E$'s sign by position along the call (runs of one sign would
mean a systematic component), and $|\sum E|$ against $\sqrt{N_{\rm CC}}\,{\rm rms}(E)$.

## 3. M2 — the declared phase error

On each M2 call:
1. **The declaration's tightness.** At the fine nodes of the returned leaves (all of them, or a
   seeded subsample of at least 1,000 per call, stated), the group's actual phase error
   $\delta\theta(x)$: production's `theta_mod_2pi` against the group phase computed from `mpmath`'s
   $J_\nu$ and $Y_\nu$ at 30 digits for each constituent, composed with the group's signs and
   coefficients and reduced against $2\pi$. Tabulate $\max|\delta\theta|$, its median, and their
   ratios to `theta_abserr` at the same nodes. **Method check:** the comparison reproduces, on one
   constituent at one argument, the error `LiouvilleGreen/bessel_phase.py` documents there.
2. **The term as the driver adds it:** $D = \sum_{\rm leaves}$ (declared term), split by branch.
3. **The call's response to a phase error of the declared size.** For
   $\delta\theta(x) = s\,\texttt{theta\_abserr}(x)$ with $s = +1$ and $s = -1$, the first-order
   change $\delta I = \int f\,\delta\theta\,b'(\theta)$ over the call's span, and per returned leaf
   $\delta I_\ell$. Computed independently of the floor being measured, to two significant figures,
   with convergence shown by doubling the resolution. The method is the agent's; one route is the
   driver itself on the integrand $f\,\delta\theta$ with the basis components exchanged, since
   $b'$ is $b$ rotated.
4. **The response to the measured error:** $\delta I$ for $\delta\theta$ from item 1, at the same
   accuracy, where item 1 sampled every node; otherwise reported as not computed.

Tabulate per call: $D$, $|\delta I|$ (both signs), $\sum_\ell|\delta I_\ell|$, the ratios
$D/|\delta I|$ and $\sum_\ell|\delta I_\ell|/|\delta I|$, the call's request and declared error, and
the call's total phase $|\theta(b) - \theta(a)|$. **Descriptive:** the same with the Clenshaw–Curtis
proxy replaced by $\sum_k |w_k|\,|f_k|$ per leaf (the honest per-leaf $\int|f|$), to separate the
proxy's own slack from the combination's.

## 4. Predictions, fixed before the run

README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
**undecided** against the rule as written.

**M1:**
- **F1 — the node errors cancel on Z2's call.** **Rule:** $\sum|E| \ge 10\,|\sum E|$ over its
  returned Clenshaw–Curtis leaves.
- **F2 — Z2's call meets its request with signed node errors.** **Rule:** on its returned partition,
  the signed-combination error is at most $\max(\texttt{atol}, \texttt{rtol}\,|v|)$.
- **F3 — the cancellation is general.** **Rule:** over the sampled node-dominated calls with
  $N_{\rm CC} \ge 100$, the median of $\sum|E|/|\sum E|$ is at least 10. Undecided if fewer than five
  such calls exist.

**M2:**
- **F4 — the declaration is honest.** **Rule:** on every M2 call, $|\delta\theta(x)| \le
  \texttt{theta\_abserr}(x)$ at every node sampled. A violation is a finding for the user, not a
  stop.
- **F5 — the declared term overstates the response.** **Rule:** the median over M2 calls of
  $D / \max_s|\delta I|$ is at least 10.
- **F6 — combining per leaf is part of it.** **Rule:** the median over M2 calls of
  $\sum_\ell|\delta I_\ell| / \max_s|\delta I|$ is at least 10.

## 5. Records

- **`var/levin-phase-floor-06f/log.md`** in README §5.1's form, with every deviation classified;
  `verification.md`; `issues.md` with any observation proposed as an issue. The laptop
  orchestrator copies the log to `prompts/levin-phase-floor/logs/06f-cc-floor-terms-across-leaves.md`.
- **`out/`:** per-call and per-leaf tables (CSV or JSON), the sample, the F0 checks, and a
  `results.md` that states each verdict with its numbers and, for each term, what a remedy would
  have to do on this evidence. It proposes no code.
- **Provenance** in every output header (README §5 rule 5).
- **Kept on the Mac Studio:** the captured pickles and anything over 2 MB, listed with SHA-256.

## 6. Acceptance

1. F0 holds, with every replacement listed.
2. F1–F6 each scored against the rule as written, with their numbers.
3. M1's and M2's tables complete for every sampled call; M2 item 1's method check passed.
4. The store unchanged by SHA-256; nothing created under `var/datastores/`; no Ray started.
5. **Records:** the log, `verification.md`, `issues.md` and `out/results.md`;
   `docs/levin-phase-floor/VERIFICATION.md` §6f, written by the laptop orchestrator; the records in
   `docs/levin-phase-floor/06f/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **F0** fails on more than three calls, or Z2's call does not reproduce.
- **A measurement cannot be made as written:** M2 item 1's method check fails, or item 3's
  $\delta I$ does not converge to two figures on a call (report the numbers).
- **A code change** looks necessary to measure something. Report it; do not make it.
- **Any store file** changes, or Ray is needed.
