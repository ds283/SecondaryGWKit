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

*Written 2026-11-12, at Part B's landing.*

**Part B** is the code commit **`1ec734d7`**, whose parent is **`dadd63a0`**. It was reviewed in
§B.3's eight checks, all passing, and pushed on the user's word: *"Yes, please land it – then write,
commit, and push the Part C addendum."* Two orchestrator commits came before it, both pushed:
- `a9f6abc3` amended the prompt at Part A's stop (its §9: §1.2 is variant (d), the H3 issue opened,
  W6 on $r_g$, W4's and W6's failures predicted, W7 added);
- `dadd63a0` amended it at Part B's stop (its §10). §1.1's $V$ needs the raw phase, which
  LiouvilleGreen's `test_the_raw_phase_is_never_evaluated` forbade. The user: *"Option 1, and record
  the 6π finding on that issue"*. The test is rewritten to its purpose as
  `test_the_raw_phase_enters_no_value_or_error`, and two LiouvilleGreen docstrings are corrected.

**The tree.** Part C runs at **this addendum's commit**, in the worktree
`SecondaryGWKit-lpf07a-<this commit's short SHA>`. Its code is `1ec734d7`'s.
- **S0′ holds by inspection:** `git diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'` lists
  only `CLAUDE.md`. R29 is the parent rehearsal, and no R31 is needed. The agent still runs the
  check as prompt §5 writes it.
- `git diff --stat dadd63a0 1ec734d7 -- . ':!docs' ':!prompts'` touches only:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/` (`test_levin_phase_variation.py`, new; `test_levin_stationary.py`;
    `test_levin_quadrature.py`'s test bases);
  - `LiouvilleGreen/bessel_phase.py` and `three_bessel_integrals.py`, docstrings only (their AST
    without docstrings is unchanged);
  - `LiouvilleGreen/tests/test_bessel_compatibility.py`.

  Nothing under `ComputeTargets/` or `ObjectFactories/` moved. **So outside `QuadSourceIntegral`
  R30 is predicted to equal R29**, as §5 says.

**What Part B's review found that Part C must carry:**
- **Every number in the review was re-run on the laptop:**
  - the seven roots: 116, 781, 39, 156 (1 skipped), 280, 127 and 64, all OK;
  - test 2's, test 5's and 14 (d)'s problems, at the parent and at the commit, each as its comment
    says;
  - breakages (i), (ii), (iii), (ix) and (x) re-applied, each caught;
  - **z12 and v08 replayed at the commit, bit-identical to 07's census:** z12 honest at 0.718 of its
    error, v08 at 3.16. So W4's failure on v08 is reproduced on the code Part C runs, as §9.5
    predicts.
- **Route changes go both ways, and some cost convergence** (prompt §10.4). In the LiouvilleGreen
  draws, 434 regions change route, every one towards Clenshaw–Curtis ($V$ 0.31–0.999 of 07's
  estimate, as §0.2's end-weighting predicts), and 771 change their LU choice. Three calls at rtol
  $10^{-10}$ go from converged to unconverged when a region with $V$ just below $6\pi$ goes to
  Clenshaw–Curtis. **The census lists every such call** (§10.4's descriptive item), with $V$, 07's
  estimate and abserr under each code. They go to `[07-c2-gate-threshold-was-never-measured]`. They
  are not a stop. W1 sets them aside, as calls whose route changed, and W7 does not cover them.
- **No region took the node-partition sum in the draws** (493,516 regions gated): with `theta_deriv`
  supplied, every phase with a sign change is located and windowed. W5's second count is predicted
  zero in the census for the same reason. A non-zero count is a finding, with its calls listed.
- **The raw phase now routes** (the agent's Observations 2 and 3). The gate reads
  $|\theta(b) - \theta(a)|$ from the caller's raw phase. Any error in that phase, not just its
  rounding, moves $V$. The draws reach $|\theta| = 1.02\times10^{13}$ (ulp $1.95\times10^{-3}$ rad).
  **The census records, as descriptive, the largest $|\theta|$ sampled at a gated region's end, by
  kind and rtol,** from the diagnostics it already computes, and the smallest margin $|V - 6\pi|$ and
  $|V - 20\pi|$ among regions whose route did not change. Neither is scored.
- **Cost.** The production path evaluates the raw phase at two points per gated region. In the draws
  the summed wall was 762 s against the parent's 692 s, on a shared machine. The census reports
  cost as §6 says; cost is not an argument.
- **Not Part C's:** the agent's Observation 1, four comments made stale by §1.1:
  - `LiouvilleGreen/bessel_phase.py`, the `bessel_phase()` docstring, still says `theta` is never
    evaluated, and that `phase_span` comes from `theta_deriv`;
  - `three_bessel_integrals.py`'s `residual_log_deriv` docstring and
    `ComputeTargets/tests/test_quadsource_integral.py:2121–2122` also say `phase_span` comes from
    $\theta'$.

  Their fix is put to the user on the laptop. No Part C agent touches them.

**§C.0, re-checked read-only on the Mac Studio, 11:22 UTC, 2026-11-12:**
- **The main checkout** is `handover-remedial` at `28ebaf07`, with `git status --short` empty.
  `git worktree list` shows it and Part A's `SecondaryGWKit-lpf07a-f93aafd` (at `f93aafd0`), which
  is removed at dispatch (§C.0).
- **The registry:** 73 runs, 65 done, 4 failed and 4 killed, all from earlier prompts. Nothing is
  `running` or stale.
- **The lock and Ray:** `rehearsal_lock status` reports the lock free (`~/.raypoolkit/rehearsal-lock`
  does not exist). `ray-processes` lists no Ray process.
- **Load and disk:** the load is 4.61 / 4.45 / 3.69, and 149 GiB is free (84 % used).
- **The 07a paths:**
  - `var/levin-phase-floor-07a/` holds only Part A's `measurement/` (61 files kept there, listed in
    `docs/levin-phase-floor/07a/measurement/KEPT_ON_MAC_STUDIO.txt`);
  - there is no `rehearsal/` or `census/` yet, and no `var/datastores/levin-phase-floor-07a-*`;
  - `var/runs/` holds Part A's four run directories, `levin-phase-floor-07a-attr-1-…`, `-attr-2-…`,
    `-bg2-items12-…` and `-x396-fast-…`. Their slugs do not collide with `r30-07a`, `k0` or
    `census`.
- **The inputs, by SHA-256 against their recorded lists,** 160 files, all OK:
  - R29's five store files against `docs/levin-phase-floor/07/rehearsal/r29/KEPT_ON_MAC_STUDIO.txt`,
    primary `28a83aa0…`;
  - 07's census run directory `levin-phase-floor-07-census-20261108T171137`, all 150 files against
    `docs/levin-phase-floor/07/census/KEPT_ON_MAC_STUDIO.txt`, checkpoint `03dfe786…` among them;
  - the scratch copy's five files against `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`,
    with no `-wal` or `-shm`.
- **The venv** is Python 3.13.16, with `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
  0.2.0, `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy` 2.2.4, `scipy` 1.16.3 and `mpmath` 1.3.0. The
  laptop's are the same.

**The dispatch** waits for the user's word. Then the laptop does three things:
1. It fetches into the Mac Studio's main checkout, which moves no working tree.
2. It repeats §C.0's checks, removes Part A's worktree and makes the one worktree at this commit,
   detached and with no venv.
3. It starts the Mac Studio orchestrator session in that worktree, through the saved `MacStudio`
   connection, with this note's §C as its brief. The folder must be trusted in the app first.

## 5. Stop and ask the user

- At Part A's planned stop, always.
- At any stop condition of prompt §8 that an agent reports, with the agent's words.
- When a premise this note or the prompt states is found false at a dispatch.
- Before any write outside the paths named here, any deletion, and any push.
