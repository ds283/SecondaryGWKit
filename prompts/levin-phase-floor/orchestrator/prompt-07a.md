# Orchestrator — prompt 07a, the window's early stop and the phase variation the gate routes on

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md`](../07a-window-stop-and-phase-variation.md).
**Board item:** 07a · **Closes, in its records commit, on the user's word:**
`[07-iroff1-window-stop-can-leave-a-call-dishonest]`; **narrows**
`[07-c2-gate-threshold-was-never-measured]` (O9 removed) · **Model:** Opus, for every agent. You
review on whatever model you were started with.

The prompt runs in three parts:
- **Part A (prompt §2), the measurement,** on the Mac Studio: one Claude Code session there, which
  is the measuring agent, in a worktree at the dispatch SHA. It ends at a planned stop. This note's
  §A.
- **Part B (prompt §1, §3, §4), the code,** on the laptop: a fresh-context subagent in the main
  checkout, one commit. This note's §B.
- **Part C (prompt §5, §6),** on the Mac Studio, through an orchestrator session there, as 07's
  Part B ran: R30 under the lock, then the census. This note's §C, completed by an addendum at its
  dispatch.

## 0. The order, and why

1. **Part A first, because §1.2 waits on it.** The user chose to see the replays before choosing the
   stop's fix: *"Yes, replay the calls on the Mac Studio and then stop so that we can assess what to
   do."* Part A changes nothing in the tree.
2. **At Part A's stop** you relay the agent's tables **as it wrote them**, with your own re-scoring
   of §2.4 item 4 beside them (§A.3). The user chooses. You write §1.2 and §9 from the choice, with
   the tests and breakages it needs, and show the text to the user. It is committed and pushed on
   their word; that commit is Part B's dispatch SHA.
3. **Part B lands before Part C starts.** R30 and the census tree are the code commit.
4. **Every tree on the Mac Studio is a worktree.** The main checkout there stays at `28ebaf07`; it
   supplies the venv, the stores, `var/` and the registry. Git there is read with
   `--no-optional-locks`, and only the laptop writes git on the Mac Studio (`fetch`, `worktree add`
   and `remove`).
5. **What must not be touched:** A3 v2 and its scratch copy
   `var/levin-phase-floor-06/wider/state/a3v2_copy/` (read-only, kept for 08); R29's store; 07's
   census and references run directories; `var/levin-phase-floor-07/`.

### 0.1 What 07 taught, which this note carries

- **Re-check every premise at dispatch.** V5's premise was stale by the time 07's Part B ran, and
  nobody re-checked it (the board's **G2 L2, reopened**). Before each dispatch, re-derive every
  number this note and the prompt quote from the records they cite, and say what changed.
- **A rule you write for an agent can be wrong.** §B.7's rule for R29 stopped the agent on a case
  that was a finding, not a fault. When an agent stops on a rule of yours, check the rule first.
- **Records copied back are exactly what the note lists.** 07's copy-back took R29's readers'
  extract by mistake. Make the list first, from the note, and copy that.
- **Launch, one check, end the turn.** Wait on a detached run with one background `kill -0` loop on
  its pid, never by polling.
- **Lock commands** are each a shell command of their own, output by `>>`, never piped.
- **`HEAD` is not the agent's.** Other sessions may commit while it works.

---

## A. Part A — the measurement, on the Mac Studio

### A.0 Facts checked by the orchestrator at `b5405eb0`, 2026-11-10

Re-check each at dispatch.
- **The code is 07's.** `git diff --stat adffa21b b5405eb0 -- AdaptiveLevin ComputeTargets
  LiouvilleGreen tools` is empty. `AdaptiveLevin/levin_quadrature.py` is 5,719 lines:
  - `SIX_PI` `:237`, `_LEVIN_DIRECT_SOLVE_PHASE_SPAN` `:472`, the `iroff1` constants `:524–526`;
  - `_Basis_SinCos` `:2092`, its `build_Levin_data` `:2174`, `phase_span` computed at `:2268`;
  - the gate at `:2574`, the LU choice at `:2666`;
  - `_strictly_one_sign` `:3143`, `_changes_sign` `:3148`, `_nodal_guard` `:3153`,
    `_oriented_cc` `:3190`;
  - `_locate_stationary_points` `:3281`, Rule A at `:3395–3411`;
  - `_adaptive_levin` `:3667`; `_floor_limit_stopped_window_leaf` `:4500`, `_stop_window`
    `:4529`, the count at about `:4825–4858`; `location_skipped_phases` at `:5195`;
  - `adaptive_levin_sincos` `:5475`.
- **`phase_span` is read only** by the gate, the LU choice and the result dicts; nothing else in
  `AdaptiveLevin/`, `ComputeTargets/` or `LiouvilleGreen/` reads it (a comment in
  `LiouvilleGreen/bessel_phase.py` names it). `build_Levin_data` has three implementations:
  `_Basis_SinCos`, and the test bases in `test_levin_quadrature.py:62` and
  `test_levin_stationary.py:453`.
- **On the Mac Studio** (main checkout `28ebaf07`), `var/levin-phase-floor-07/` is 1.8 GB and holds:
  - `census/state/v4/capture/C00`–`C24`, 44 MB: v00–v23 and `cal_z12` (unit `g-1e-8-1-2174`), each
    with `inputs_*.pkl`, `regions_*.pkl`, `spec.json`, `emit.jsonl` and `worker.log`; the mapping
    is `census/state/v4_capture_selection.json`;
  - `census/state/xcheck/capture/X00`–`X09`, 679 MB: the 396 class-(c) calls at `1e-8`
    (`xcheck_capture_selection.json`);
  - the pickles and large files listed in `docs/levin-phase-floor/07/census/KEPT_ON_MAC_STUDIO.txt`
    (50 lines under `v4/capture/`, 28 under `xcheck/capture/`).
- **The two pairs** (`docs/levin-phase-floor/07/census/out/xcheck/report.md`): `3-14501`, $r_g$
  1.39, and `3-35006`, 1.08, index 1, `G+q+r`, band low, $k = 10^5$, class (e). 06m's values agree
  across rtols on both ($r_{06m}$ 0.00122 and $5.5\times10^{-5}$), and 07's are 444–656× their
  abserr from 06m's. **Which is nearer the truth is not known.** They are not captured.
- **R29**, `var/datastores/levin-phase-floor-07-r29*.sqlite`, five files, primary `28a83aa0…`, built
  at `e182bc7a`.
- **07's census:** `levin-phase-floor-07-census-20261108T171137`, checkpoint `03dfe786…`, kept whole.

### A.1 Before you dispatch

1. The prompt and this note are committed and pushed. That commit is the **dispatch SHA**.
2. **On the Mac Studio, read-only:** the main checkout clean at `28ebaf07`; `runregistrykit list`
   with nothing `running` or stale; the lock free; no `var/levin-phase-floor-07a*`; load and free
   disk recorded; the captures of §A.0 hashed against the `KEPT_ON_MAC_STUDIO.txt` lines; the scratch
   copy against `copy_files_certified.txt`; 07's census checkpoint and R29's primary hashed.
3. **The worktree:** `git fetch origin` in the Mac Studio main checkout, then
   `git worktree add --detach ../SecondaryGWKit-lpf07a-<dispatch short SHA> <dispatch SHA>`. No
   venv in it.
4. **The marker:** `touch` a file in your scratchpad, and record the time on the Mac Studio.

### A.2 Dispatch

Start one Claude Code session on the Mac Studio over the saved `MacStudio` SSH connection, in the
worktree, on Opus, with `mcp__ccd_session__start_session` (the folder must be trusted in the app
first). Its brief: prompt §2 is its whole scope; §A.0; the dispatch SHA; and these rules:
- **Every command** calls the main checkout's `venv/bin/python` by its absolute path, with
  `PYTHONPATH` set to the worktree and `-B`. Nothing is written in the worktree.
- **It writes only** `var/levin-phase-floor-07a/measurement/` and its own registered run
  directories. No Ray, no lock, no store but A3 v2's scratch copy, read-only, for §2.1 item 2's four
  captures.
- **`--cpus 6`**, and at most six worker processes.
- **Anything over ten minutes** runs as a registered detached run (`CLAUDE.md`, the run registry),
  launched, checked once, and waited on by one background `kill -0` loop.
- **Its records:** `log.md` in README §5.1's form, every deviation numbered and classified;
  `verification.md`, its part of `VERIFICATION.md` §7a; `issues.md`; `out/`, `scripts/`,
  `state/`; and `KEPT_ON_MAC_STUDIO.txt`, listing every pickle and file over 2 MB with SHA-256 and
  size.
- **It stops at prompt §2.5,** and at any of §8's conditions, and reports to the laptop session.
  It recommends nothing.

### A.3 At the stop, before you relay

Read over `ssh macstudio-tunnel`, read-only, with scratch readers of your own:
1. **Scope:** `find var -newer <marker>` lists only the agent's paths; the worktree's
   `git status --short --ignored` is empty; the captures, the scratch copy, 07's census checkpoint
   and R29 are unchanged.
2. **The replays:** for three calls of §2.1 item 1, two of them failures, re-run 07's code on the
   capture yourself and compare value, abserr and `converged` with the census record in `float.hex`.
3. **The four new captures** checked against 07's census by its `check_captures.py`, in the agent's
   outputs.
4. **The variants:** each diff against the tree touches only the window's stop. Re-run one failure
   under two variants yourself.
5. **§2.4 item 4 re-scored** by your own reader from the agent's per-call outputs: which variants
   qualify.
6. **The attribution:** the closure check in each attributed call.

Then relay the report, with your re-scoring beside it. If the report and the outputs disagree, say
so.

### A.4 After the user's choice

Write prompt §1.2 and §9 from the user's words, quoted: the rule, its constants with their basis,
what it does to `converged`, the tests that exercise it and the breakages that must be caught, and
any addition to §6's predictions. Show it to the user. On their word commit and push it, with Part
A's records copied back as §C.4 says for Part C (no pickles, nothing over 2 MB, by a list made
first), each file checked by SHA-256. That commit is Part B's dispatch SHA.

---

## B. Part B — the code, on the laptop

### B.1 Before you dispatch

1. The amendment commit is pushed. Nothing in `AdaptiveLevin/`, `ComputeTargets/` or
   `LiouvilleGreen/` has changed since `adffa21b`.
2. **Baselines:** the seven roots, serially, at the dispatch SHA, into the scratchpad, with `uptime`
   beside each.
3. **The marker:** `touch` a file in the scratchpad.

### B.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA,
the baselines and §A.0. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Before staging it runs `git log --oneline <dispatch SHA>..HEAD`, reads
  anything that landed, and merges a shared file by reading it, never by overwriting it.
- **Its scope is prompt §1, §3 and §4.** It copies §5's S0′–S4 and §6's W0–W6, with §9's additions,
  into its log's Predictions and verdicts, marked not yet scored.
- **One commit, staged by explicit path,** not pushed, covering only:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_phase_variation.py`, new, and
    `AdaptiveLevin/tests/test_levin_stationary.py`;
  - another module in `AdaptiveLevin/tests/` only for an assertion prompt §3 allows, listed;
  - `docs/levin-phase-floor/07a/` (the LiouvilleGreen draws' harness and outputs);
  - `prompts/levin-phase-floor/logs/07a-window-stop-and-phase-variation.md` and the board;
  - `CLAUDE.md`, only to add record-script directories to its `black` paragraph.

  **A path outside that list is a stop.**
- **No Ray and no store.** It writes nothing under `var/`.
- **Every deviation is numbered and classified** before it reports.

### B.3 The review — on the laptop

1. **Scope.** `git show --stat` lists only §B.2's paths; `git status --short` is clean;
   `find var -newer <marker>` lists nothing of the agent's.
2. **Nothing else moved,** by AST with docstrings removed, against the dispatch SHA:
   `_roundoff_floor`, `_levin_G0_G1`, `_cc_roundoff_floor`, `_trust_noise_yardstick`, `_check_leaf`,
   `_nodal_guard`, the window edges and pieces, `SIX_PI`, `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`,
   $\Phi_w$, $\Lambda_c$ and `DEFAULT_LEVIN_MAX_DEPTH` unchanged; `adaptive_levin_sincos`'s body
   unchanged. `git diff <dispatch SHA> <code commit> -- ComputeTargets LiouvilleGreen` is empty.
3. **The construction, read** against prompt §1.1 and §1.2 as amended:
   - the location's first step skips only an identically zero rate; minima and roots of every other
     phase run as before;
   - $V$ in each `build_Levin_data`: $|\theta(b) - \theta(a)|$ when the rate has no strict sign
     change on the nodes, the node-partition sum otherwise; the raw phase sampled at the ends only
     in the first case;
   - every use of `phase_span` receives $V$, as the agent's list says;
   - §1.2 as the user chose, and nothing more.
4. **The roots, re-run here.** Five equal their baselines; `AdaptiveLevin` rises by exactly the
   tests added, counted from the diff. The known flake (`test_tk_wkb_phase …
   test_wall_time_per_object`) is re-run alone.
5. **The tests, read.** Each of prompt §3's exists. Run the new module alone, with `-v`. Run test 2's
   and test 5's problems at the dispatch SHA yourself, in a scratch worktree, and compare with their
   comments.
6. **The breakages.** Re-apply (i), (ii) and (iii) yourself, and one of §9's: run the one module,
   revert with `git checkout -- <file>`, `git status --short` clean after.
7. **Replays.** Replay z12 and v08 at the code commit from copies of their captures (copied to the
   scratchpad over `ssh`, hashed), and score each against its reading.
8. **The records:** the log's headings and deviations; its Predictions and verdicts word for word
   (by substring); the board's 07a row; `black --check`; the commit message.

### B.4 Landing

If the review passes, ask the user to land it, with the review's findings. On their word push it,
then write §C.5's addendum, commit and push it. That commit is Part C's dispatch SHA. Part C starts
only on the user's word.

---

## C. Part C — R30 and the census, on the Mac Studio

### C.0 When it starts

As 07's note §B.0, with: no `var/levin-phase-floor-07a/rehearsal` or `census` yet, and no
`var/datastores/levin-phase-floor-07a-*`; R29's primary `28a83aa0…` and 07's census checkpoint
`03dfe786…` intact; the scratch copy certified. The laptop fetches, makes one worktree
`SecondaryGWKit-lpf07a-<Part C dispatch short SHA>`, removes Part A's, and starts the Mac Studio
orchestrator session in it with this note's §C as its brief.

### C.1 What the Mac Studio orchestrator is told

As 07's note §B.2, with these differences:
- **No parent worktree:** R29 is the parent rehearsal (prompt §5 S0′). If S0′ fails, it stops and
  asks; R31 is run only on the user's word.
- **The rehearsal agent:** prompt §5. It writes `var/levin-phase-floor-07a/rehearsal/`, the store
  `var/datastores/levin-phase-floor-07a-r30*`, its run directory and launch log, the lock and
  `/tmp/ray-sgk`. **Planned stop 1** after R30 and its comparison with R29, before S1–S4 and the
  readers.
- **The census agent:** prompt §6, Ray-free with no lock, writing only
  `var/levin-phase-floor-07a/census/` and its run directories. **W0 before the launch.** The
  predictions go in `state/predictions.md` after W0 and before the launch. **Planned stop 2** after
  W0; **planned stop 3** after the census, `done`, 18,560 of 18,560, before any verdict is scored.
  **A call W1 says should be untouched that differs from 07's is a stop,** before anything else is
  scored.
- **No new references** are taken. W4 is scored on 07's readings.

### C.2 The review — on the Mac Studio

As 07's note §B.3, checks 1–3 and 8–9, with R30 in place of R27 and R28. Then:
- **The rehearsal values:** `tools.store_value_diff` on R29 and R30 re-run, reproducing the saved
  JSON; outside `QuadSourceIntegral` every value table identical; S3 by your own reader; S4 by your
  own reading of R30's four blocks.
- **W0–W6, re-scored** from the two checkpoints by your own readers, each holds, fails or undecided
  against the rule as written. W1's untouched calls bit for bit at both rtols; W4 on all 33 calls
  from the readings; W5's three counts from the instrument's records; W6 over all 2,484 pairs.

### C.3 What it reports

Pass or stop on the first line; then R30 and S0′–S4; W0–W6 with their numbers; the route changes;
the newly located calls; the windows and the stop; convergence against 07's; cost.

### C.4 After the review — on the laptop

Make the copy-back list first, from this section: `var/levin-phase-floor-07a/` without pickles,
files over 2 MB or the readers' extract, and the small run directories (R30, `k0`). Copy exactly
that list, check every file by SHA-256 against a list made on the Mac Studio, and re-check from the
copies: W4 on z12 and on two of the five; S1 on three w calls; one untouched rehearsal row and one
moved row, in `float.hex`.

Then ask the user about the issues and anything the agents' `issues.md` propose, and make **one
records commit**, pushed: the records in `docs/levin-phase-floor/07a/`, scripts byte-identical with
their directories added to `CLAUDE.md`'s `black` paragraph; the logs appended to
`logs/07a-window-stop-and-phase-variation.md` with the verdicts; `VERIFICATION.md` §7a; the board,
with 07a's row and a Decisions row for the user's rulings; the two issues on the user's word;
`docs/OPEN_ISSUES.md` with its count and date corrected by counting; this directory's README row and
the campaign README's.

**Kept until the user says otherwise:** R29 and R30, 07a's census run directory (08's "before"), 07's
census run directory, and A3 v2's scratch copy (for 08).

### C.5 Addendum: Part C's dispatch

*Written at Part B's landing.*

## 5. Stop and ask the user

- At Part A's planned stop, always.
- At any stop condition of prompt §8 that an agent reports, with the agent's words.
- When a premise this note or the prompt states is found false at a dispatch.
- Before any write outside the paths named here, any deletion, and any push.
