# Orchestrator — prompt 07, the stationary-point guard

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/07-stationary-point-guard.md`](../07-stationary-point-guard.md),
as rewritten on 2026-11-07.
**Board item:** 07 · **Closes, in its records commit, on the user's word:**
`[06-levin-accepts-a-region-across-a-stationary-point]` · **Model:** Opus, as the prompt
recommends, for every agent. You, the orchestrator, review on whatever model you were started with.

**This note was rewritten on 2026-11-07 with the prompt.** The note before it is `da5c24de`'s, with
its addendum of 2026-10-28. It dispatched the first text four times, and its facts are 06a's. The
orchestrator notes of 06b, 06c, 06d and 06e record each set-aside. This note replaces it.

The prompt runs in two parts, as 06g's and 06m's did:
- **Part A (prompt §1–§4), the code,** on the laptop. A fresh-context subagent works in the **main
  checkout at `HEAD`** and makes one commit. **Prompt §2 is a planned stop** inside it. This note's
  §A.
- **Part B (prompt §5 and §6),** on the Mac Studio, through an orchestrator session started there by
  the laptop, as 06l's and 06m's were:
  - R27 and R28 under the lock;
  - 06m's census re-run at the code;
  - V4's references.

  This note's §B. Its facts are re-checked, and its trees named, in an addendum written at its
  dispatch (§B.5).

## 0. The order, and why

1. **Part A first.** R28 and §6's tree $g$ are the code commit. Nothing on the Mac Studio starts
   until Part A has passed its review here and been pushed.
2. **Part A stops once by design**, at prompt §2.6.
   - The agent reports §2's tables, the 14 real calls, the LiouvilleGreen draws and §2.5's account.
     You relay them to the user **as the agent wrote them**. You add nothing of your own but the
     state of the files (§A.2).
   - The user settles $\Phi_w$ and $\Lambda_c$, and rules on §1.3, the stopping rule for window
     leaves.
   - **If the user rules a rule in,** you write it as an amendment to the prompt, from §2.5's cases,
     and show it to the user before the agent resumes. Its basis is QUADPACK's counters, as prompt
     §2.5 states them. Its thresholds start at the generous end, with a stated basis, as the first
     text's §9 point 8 required. The amendment is committed and pushed before the agent resumes.
   - The agent resumes only on the user's word, with the user's words passed to it verbatim.
3. **The set-aside work is not touched,** by you or by any agent, beyond prompt §0.5's reads. The
   four stashes and `var/levin-phase-floor-07/wip/` stay until the user decides their fate, after
   07's records commit.
4. **Every tree on the Mac Studio is a worktree.** The main checkout there does not move. It
   supplies the venv, the stores, `var/` and the registry.
5. **Part B's agents run one at a time:** the rehearsal agent under the lock, then the census
   agent, Ray-free with no lock, which also makes V4's references. Each is held at planned stops
   (§B.2).

### 0.1 What 05–06m taught, which this note carries

- **Every deviation is numbered and classified before an agent reports,** failed attempts
  included.
- **A question relayed to the user carries the state of the numbers.** Before relaying, check what
  has already been written, and say so.
- **No fact in this note is to be trusted over the code or the store.** Each agent checks what it
  relies on.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`,
  `rho_reuse` and work counters (`[05-store-value-diff-compares-volatile-columns]`). For R27 against
  R28 it also excludes the four new counts, which only R28 has.
- **05's `classify_non_qsi.py` needs both runs to share a job name** (06m's rehearsal, deviation 1).
  S0 compares R26 (job `levin-phase-floor-06m`) with R27 (`levin-phase-floor-07`). The labels are
  normalised in a copy of the JSON, as 06m did, and the classifier is not changed.
- **`HEAD` is not the agent's.** Other sessions may commit while it works.
- **Launch, one check, end the turn.** Wait on a detached run with one background `kill -0` loop on
  its pid, never by polling.
- **Lock commands** are each a shell command of their own, output by `>>`, never piped.
- **A high load on the Mac Studio** has come from system services (WindowServer, ChronoSync,
  fileproviderd, Spotlight), not from jobs. Report it; it is not by itself a stop. It matters most to
  the census's 300 s per-unit budget.

---

## A. Part A — the code, on the laptop

### A.0 Facts checked by the orchestrator at `a8c7854c`, 2026-11-07

Pass them on. Re-check each at dispatch.
- **The checkout.** `handover-remedial` at `a8c7854c`, `git status --short` empty, one worktree
  (itself). The laptop's registry shows nothing `running` or stale: 52 `done`, 5 `failed`, 21
  `killed`, 2 `lost`, 2 `unknown` from before the registry. Part A launches no run, starts no Ray and
  opens no production store.
- **The libraries:** `mpmath` 1.3.0, `scipy` 1.16.3, `numpy` 2.2.4.
- **The set-aside work** (prompt §0.5):
  - `git stash list` shows four, each named for 07:
    - `stash@{0}`, rebased on 06d at the fourth stop (`19eb784e…`);
    - `stash@{1}`, on 06c (`4e6db6…`);
    - `stash@{2}`, on 06b with §2 run (`1278f4…`);
    - `stash@{3}`, §1 before 06b (`a89ac9…`).
  - `var/levin-phase-floor-07/wip/` holds four patches and four tarballs, listed in
    `SHA256SUMS.txt`.
  - **The latest tarball,** `07-part-a-rebased-on-06d-untracked.tar.gz` (`f87d8231…`), holds:
    - `docs/levin-phase-floor/07/calibration/`: stops 1–3 under `stop1/`–`stop3/`, and the fourth
      stop's `scripts/` and `out/`;
    - `prompts/levin-phase-floor/logs/07-stationary-point-guard.md` (50,877 bytes).
- **The 14 captures** for prompt §2.3 are in `var/levin-phase-floor-06g-fcc-laptop/inputs/`: z00,
  z12, w00–w11 and r55. They are copies of the Mac Studio's, made for 06g's laptop FCC batch and
  matched by SHA-256 then (`docs/levin-phase-floor/06g/KEPT_ON_MAC_STUDIO.md`). The harness that
  made and checked them is in `var/levin-phase-floor-06g-fcc-laptop/harness/`
  (`make_capture.py`, `x1_capture.py`, `check_captures.py`). `tools/filon_reference.py` reads them
  with `--unpickler ray.cloudpickle`.
- **The references** are in `docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json`:
  - `V1` holds w00–w11, with `ref_hex`, `tol` and R24's value and abserr;
  - `rows` holds z00 and z12, with `ref_hex` and 06g's values under `g`. z12's reference is
    `0x1.0c5f18f86f3efp-86`.
- **The module,** `AdaptiveLevin/levin_quadrature.py` (4,557 lines):
  - `_roundoff_floor` `:551`, `_cc_roundoff_floor` `:755`, `_trust_noise_yardstick` `:933`;
  - `used_interval` `:1054`, `_sample_vectorized` `:1859`, `_Basis_SinCos` `:1907`;
  - `_adaptive_levin_subregion_impl` `:2237`, whose C2 gate goes to the Clenshaw–Curtis rule at
    about `:2314`;
  - `_adaptive_levin_subregion_cc` `:2602`;
  - `_adaptive_levin` `:2899`. In it:
    - the docstring's "no stall rule" at about `:2950`;
    - `_check_leaf` `:3439`;
    - the root leaf at about `:3596`;
    - the stop test at about `:3664–3688`;
    - the floor-limited exit at about `:3720–3738`;
  - `adaptive_levin_sincos` `:4335`.
  - **No path forces a Clenshaw–Curtis solve** on a span of $6\pi$ or more. Prompt §1.2 item 4 and
    §1.4 add one.
- **`AdaptiveLevin/tests/`** holds nine modules. `_TwoIndependentSinCosBasis`
  (`test_levin_quadrature.py:22`) has no Clenshaw–Curtis methods, so any Clenshaw–Curtis leaf would
  raise on it. Prompt §3 test 9 subclasses it. It is also imported by `test_levin_roundoff_exit.py`.
- **The callers:**
  - `ComputeTargets/QuadSourceIntegral.py`:
    - `phase_group_Levin_integral` `:888`, with `theta_deriv` supplied (`LEVIN_USE_THETA_DERIV = True`,
      `:102`);
    - the per-group record, the dict at about `:960–971`;
    - `_three_bessel_Levin` `:1109`;
  - `LiouvilleGreen/three_bessel_integrals.py:775`, whose `levin_theta` supplies `theta_deriv`.
  - **No test in `ComputeTargets/tests/` pins a group record's key set.** Prompt §3 test 11 adds
    one.
- **The known flake** in the `ComputeTargets` root is `test_tk_wkb_phase …
  test_wall_time_per_object`, a wall-clock assertion. Re-run it alone and record it; it is not a
  stop. Record `uptime` beside each roots run.
- **A worktree on the laptop** has no venv. Symlink the main checkout's `venv` into it, and add
  `venv` to `.git/info/exclude`. Remove it with `git worktree remove` when done.

### A.1 Before you dispatch

1. The rewritten prompt and this note are committed and pushed (the commit that wrote them). Check
   that `HEAD` is on `origin/handover-remedial`. That commit, or a later one with nothing in
   `AdaptiveLevin/`, `ComputeTargets/` or `LiouvilleGreen/`, is the **dispatch SHA**.
2. **Baselines:** run the seven roots serially in the main checkout, at the dispatch SHA, into the
   session scratchpad.
3. **The set-aside work, hashed:** `git stash list` and `git rev-parse stash@{0..3}` into the
   scratchpad, and `shasum -a 256` of every file in `var/levin-phase-floor-07/wip/`. §A.3 check 1
   compares against them.
4. **The marker:** `touch` a file in the scratchpad immediately before the dispatch.

### A.2 Dispatch

Dispatch one fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the
dispatch SHA, the baselines and §A.0. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Before staging, it runs `git log --oneline <dispatch SHA>..HEAD`. If
  anything has landed, it reads it, and merges any shared file by reading it, never by overwriting
  it.
- **Its scope is prompt §1–§4.** §5 and §6 run later on the Mac Studio. It copies §5's S0–S2 and
  §6's V0–V5 into its log's **Predictions and verdicts** unchanged, marked not yet scored, and
  scores nothing of them.
- **The set-aside work is read only** (prompt §0.5): `git stash list` and `git stash show -p`
  only, and tarballs extracted into the scratchpad only. Every piece it takes is named in the log
  with its source.
- **§2 is a planned stop.** It builds §1 without §1.3, runs §2, and reports as §2.6 says. It writes
  no test of §3 before the user's word.
  - **Its outputs** each record the SHA-256 of their script and of `levin_quadrature.py` as
    measured, the `HEAD`, and that the tree was dirty.
  - **The replays of §2.3** run each call in its own process. The scratch worktree at `3ae6f3a4`
    is made by the agent, used read-only, and removed afterwards.
- **After the user's word,** it sets the two constants, builds §1.3 as the user ruled (with the
  amendment, if one was written), and goes on to §3 and §4. If `levin_quadrature.py` changes after
  §2's recorded run beyond prompt §4's allowance, §2 is re-run on the final code, and a rung that
  then moves is a stop.
- **One commit, at the end, staged by explicit path**, not pushed. It may cover:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_stationary.py`, new;
  - in `AdaptiveLevin/tests/`, an existing module only for an assertion prompt §3 test 12 allows,
    listed;
  - `ComputeTargets/QuadSourceIntegral.py`: the four counts in its Levin records (prompt §1.6), and
    nothing else;
  - in `ComputeTargets/tests/`: prompt §3 test 11, and nothing else;
  - `docs/levin-phase-floor/07/calibration/` and `docs/levin-phase-floor/07/earlier-attempts/`;
  - `prompts/levin-phase-floor/logs/07-stationary-point-guard.md` and this campaign's board;
  - `CLAUDE.md`, only to add `calibration/scripts/` and `earlier-attempts/` to its `black` list;
  - `docs/OPEN_ISSUES.md`, if the commit opens an issue.

  **A path outside that list is a stop.** In particular, these are not touched:
  `ComputeTargets/phase_groups.py`, every other file of `ComputeTargets/`, `LiouvilleGreen/`,
  `ObjectFactories/`, `config/`, `tools/`, `StoreFixtures/`, `AdaptiveLevin/__init__.py` and
  `main.py`.
- **No Ray and no store** in its tests and its checks. It writes nothing under `var/`.
- **The header's issue does not move in this commit** (prompt §4).
- **Every deviation is numbered and classified** in the log before it reports, at §2's stop too.
- **Stop and ask** at any of prompt §8's conditions that apply to §1–§4.

**At §2's stop, before you relay:**
- list the files the agent has written;
- check that its outputs' recorded SHA-256 of `levin_quadrature.py` matches the working file;
- check that the set-aside work is as hashed in §A.1;
- re-score §2.2's two rules and §2.3's verdict from its outputs, by a scratch reader of your own.

Then relay its report. If what it reports and what its outputs hold disagree, say so.

### A.3 The review — eleven checks, on the laptop

1. **Scope.**
   - `git show --stat <code commit>` lists only §A.2's paths.
   - `git diff --stat <dispatch SHA> <code commit> --` on §A.2's untouched list is empty.
   - `git status --short` is clean, and `find var -newer <marker>` lists nothing of the agent's.
   - The stashes and `var/levin-phase-floor-07/wip/` equal §A.1's hashes.
   - `git worktree list` shows the main checkout alone.
2. **Nothing else moved**, by AST with docstrings removed, at the dispatch SHA and the code commit:
   - `adaptive_levin_sincos`'s body is unchanged. Location is not in the facade;
   - `_roundoff_floor`, `_levin_G0_G1`, `_cc_roundoff_floor`, `_trust_noise_yardstick` and the
     trust rule in `_check_leaf` are unchanged;
   - `DEFAULT_LEVIN_MAX_DEPTH`, `DEFAULT_LEVIN_RELTOL`, `SIX_PI` and `_LEVIN_TRUST_MAX_RATIO` are
     unchanged;
   - in `QuadSourceIntegral.py`, the only change is the four counts.

   `git diff <dispatch SHA> <code commit> -- LiouvilleGreen ComputeTargets/phase_groups.py` is
   empty.
3. **The construction, read** against prompt §1:
   - **the interface:** two new basis methods, with `_Basis_SinCos` reading `theta["theta"]` and
     `theta["theta_deriv"]`, "cannot" without it, and a basis without the methods treated as
     "cannot";
   - **location:** in `_adaptive_levin`, once per phase, before the loop:
     - 4,097 points through `_sample_vectorized`;
     - roots by `brentq` with `xtol` $\le 4\varepsilon\max(|x_a|, |x_b|)$;
     - minima by bounded Brent, with $\lambda$ as the prompt defines it and $\theta'''$ by a central
       difference of the rate;
     - the end and merge rules, and each point's phase recorded;
   - **windows:**
     - edges by `brentq` on the raw phase, merged on overlap;
     - pieces of span at most $2\pi$;
     - each piece a forced Clenshaw–Curtis leaf in the one budget;
     - the seeding, and with nothing located today's root leaf unchanged;
     - never floor-limited;
   - **the guard:**
     - a sign change of any rate on the leaf's own Chebyshev nodes;
     - never returned as a Levin leaf, and chosen before unguarded leaves;
     - at `depth_max` the forced Clenshaw–Curtis solve, `guard_at_depth_max` flagged and
       `converged` False;
   - **§1.3** as the user ruled, and nothing more;
   - **the two constants** are named module constants, each with a comment citing §2, and their
     values are the ones the user settled;
   - **the bypass switch** is private, named as test-only, off by default, and read nowhere but the
     location;
   - **the result** carries `stationary_points`, the three counts, and `used_interval`'s two flags;
   - **the docstrings** prompt §1.5 lists say what is new. The module docstring no longer calls
     gate-and-bisect the structural response to a stationary point.
4. **The roots, re-run here.**
   - Five roots equal their baselines.
   - `AdaptiveLevin` and `ComputeTargets` are up by exactly the test methods added, counted from
     the diff.
   - All OK, with the flake re-run alone.
5. **The tests, read.** Prompt §3's tests 1–12 each exist, or are accounted for (test 10 only if a
   rule is in), and are named in the log.
   - **References:** closed forms, contour integrals or subdivided `mpmath`, split at every
     stationary point.
   - **Honesty:** every test asserts both that the value is within `abserr` and that
     `abserr` $\ge$ the true error.
   - **Test 1's comment** records the parent's value, abserr, region count and `converged`. Run test
     1's problem at the dispatch SHA yourself, in a scratch worktree, and compare with the comment.
   - **Test 8** compares by bit pattern through the bypass switch, over every stationary-free
     problem of `test_levin_quadrature.py`.
   - **Test 9** goes through `_adaptive_levin` directly, on a basis with Clenshaw–Curtis methods.
   - **Every changed existing assertion** is in a test whose problem has a stationary point and
     supplies `theta_deriv`, and is listed with its numbers under both codes.
   - **Run the new module alone** yourself, with `-v`, and keep its printed numbers.
6. **The breakages.** (i)–(viii) each name the test that failed, and each was reverted.
   - **Re-apply (i) and (iii) yourself in any case.** They are the defect and the exit the header's
     issue is about.
   - **If a record is thin,** re-apply that breakage too.
   - Each time: in the working tree, run the one module, and revert with `git checkout -- <file>`.
     `git status --short` is clean afterwards.
7. **§2, re-scored.** From `docs/levin-phase-floor/07/calibration/out/`, by your own reader:
   - each rule applied as written picks the rung the log says;
   - the check on (C)'s cases below $L$ holds;
   - the constants equal what the user settled;
   - the recorded SHA-256 of `levin_quadrature.py` is the committed file's, or §2 was re-run on the
     final code and its rungs did not move.
8. **The real calls, re-run here.** Replay z12 and one w call yourself at the code commit, from the
   captures, each in its own process. Compare with the agent's recorded values in `float.hex`, and
   score each against its reference.
9. **The earlier records.** Every file under `docs/levin-phase-floor/07/earlier-attempts/` equals
   the latest tarball's by SHA-256, against your own extraction into the scratchpad.
10. **The records.**
    - The log has README §5.1's headings, and classifies every deviation.
    - Its **Predictions and verdicts** holds §2's rules, tables, constants and §2.3's verdict, and
      prompt §5's and §6's predictions word for word (check by substring), marked not yet scored.
    - It states what location did on the LiouvilleGreen draws, and every piece taken from the
      set-aside work.
    - The board's 07 row says the code landed and Part B is pending.
    - The header's issue has not moved. `docs/OPEN_ISSUES.md` changed only if an issue was opened,
      with its count and date correct by counting.
    - `black --check` is clean on every `.py` file touched outside `calibration/scripts/` and
      `earlier-attempts/`.
    - The commit message follows `CLAUDE.md`'s form.
11. **Nothing else landed unaccounted.** `git log --oneline <dispatch SHA>..HEAD` shows the code
    commit, and anything another session landed, each accounted for.

### A.4 Landing

If the review passes, ask the user to land it, with the review's findings. On their word, push the
code commit. Then write §B.5's addendum, commit it and push it. That commit is Part B's dispatch
SHA. Part B is started only on the user's word.

---

## B. Part B — R27, R28, the census and the references, on the Mac Studio

### B.0 When it starts

Part B starts only when all of these hold, each checked by the laptop over `ssh macstudio-tunnel`,
read-only:
- Part A has landed, and §B.5's commit is pushed;
- on the Mac Studio:
  - the main checkout's `git status --short` is empty;
  - `runregistrykit list` shows nothing `running` or stale;
  - the lock is free, and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none;
  - there is no `var/levin-phase-floor-07*` and no `var/datastores/levin-phase-floor-07-*`;
  - the load and free disk are recorded;
- the scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/` matches its certification
  file for file;
- 06m's census run directory and R26's store are intact, by the hashes 06m recorded.

Then the laptop, which is the only party that writes git on the Mac Studio, does three things:
1. It fetches in the main checkout (`git fetch origin`), which moves no working tree.
2. It makes two detached worktrees, with no venv in either:
   - `SecondaryGWKit-lpf07-<parent short SHA>` at the code commit's parent (R27, V0's K0 run);
   - `SecondaryGWKit-lpf07-<dispatch short SHA>` at Part B's dispatch SHA (R28, $g$, the
     references, the comparisons and the readers).
3. It starts the Mac Studio orchestrator: a Claude Code session over the saved `MacStudio` SSH
   connection, in the code worktree, with this note as its brief. The folder must be trusted in the
   app first. The session reports to the laptop session at each planned stop and at the end.

### B.1 Facts, to be re-checked at Part B's dispatch

- **The scratch copy's certification:** `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`.
  06e's after-hashes are `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`. Its
  fingerprint, and A3 v2's, is `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06m's census:** `levin-phase-floor-06m-census-20261106T102153`, 150 files, 4.41 GB, checkpoint
  `ea5bc3ab6b618d562b9dee0a7ed4bdba5c1999e8348b2ad33bdadb6135e63c2e`. Its file list and hashes are in
  `docs/levin-phase-floor/06m/census/KEPT_ON_MAC_STUDIO.txt`. Its harness is
  `docs/levin-phase-floor/06m/census/scripts/`, byte-identical to the run.
- **R26:** `var/datastores/levin-phase-floor-06m-r26*.sqlite`, five files, primary `e90e26c2…`, the
  rest in `docs/levin-phase-floor/06m/rehearsal/KEPT_ON_MAC_STUDIO.txt`.
- **The fingerprint** R27 and R28 must both give is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R23–R26 gave it.
- **06m's rehearsal record:** its command line, launcher (`b0135218…`), readers script
  (`run_readers_worktree.sh`, `ec96cf09…`), 05's `classify_non_qsi.py` (`26690e61…`) and the
  tabulation are in `docs/levin-phase-floor/06m/rehearsal/`.
- **06g's captures** on the Mac Studio: `var/levin-phase-floor-06g/references/state/capture/`
  (19 MB). Its V1 capture scripts are under `docs/levin-phase-floor/06g/references/v1/`.

### B.2 What the Mac Studio orchestrator is told

**Where things are.**
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, holds the venv, the stores,
  `var/` and the registry. The records go in its `var/levin-phase-floor-07/`. **No tree runs here,
  and it is not moved.**
- **The parent worktree** runs R27 and V0's K0 run. **The code worktree** runs R28, $g$, the
  references, the comparisons and the readers. This note and the prompt are read there.
- **In both worktrees:** every command calls the main checkout's `venv/bin/python` by its absolute
  path, with `PYTHONPATH` set to that worktree and `-B` or `PYTHONDONTWRITEBYTECODE=1`. Nothing is
  written in either.
- **No git command that writes,** in any tree, `fetch` included. Git is read with
  `--no-optional-locks`.
- `--cpus 6` and at most six worker processes, always.

**Two agents, one at a time, each fresh-context, on Opus.** Each is given the prompt, with its own
section as its scope, Part B's dispatch SHA, both worktrees' paths, §B.1 and the addendum. Neither
reads `orchestrator/`.

**The rehearsal agent: prompt §5.**
- **Where it writes:** `var/levin-phase-floor-07/rehearsal/`; the stores
  `var/datastores/levin-phase-floor-07-r27*` and `-r28*`; their run directories and launch logs;
  the lock; and `/tmp/ray-sgk`.
- **The launches** are 06m's, with `--job-name levin-phase-floor-07`, slugs `r27-parent` and
  `r28-guard`, `--campaign levin-phase-floor --prompt 07`, and a `--purpose` naming the host, the
  tree's SHA, the code commit and the fingerprint it must give.
- **The lock:** `acquire` once before R27; `set-pid` with each driver's pid; held through the
  comparisons, S2's capture and the readers; `release` once `ray-processes` is empty.
- **S2's references** are made after the lock is released, Ray-free, as one registered run with
  slug `p1-references`.
- **Planned stop 1,** after R27, R28, S0 and the R28-against-R27 comparison, before S1, S2 and the
  readers. Check yourself:
  - both fingerprints;
  - S0, by your own whole-row comparison of R26 and R27;
  - outside `QuadSourceIntegral`, the value tables identical;
  - inside it, every untouched row bit-identical, by your own reader.

**The census agent: prompt §6.**
- **Where it writes:** only `var/levin-phase-floor-07/census/` and its own run directories. It uses
  no Ray and takes no lock.
- **The harness** is 06m's, copied, with the diff saved and each change a deviation.
- **V0 runs before the launch,** all three clauses.
- **The predictions** go in `state/predictions.md`, after V0 and before the launch: prompt §6's
  text, unchanged, and V4's selection rule with its 24 calls. The mtime is the review's evidence.
- **Every checkpoint record** carries its script's SHA-256, its tree's `HEAD` and
  `git status --short`, and each unit's evidence of where its modules were loaded from.
- **The budget** is 300 s per unit, with 06a's over-budget pass.
- **Planned stop 2,** after V0, before the census: re-run two K0 units yourself at the parent and
  compare them with 06m's census. Re-run one stationary-point unit at $g$ twice and compare.
- **Planned stop 3,** after the census: `done`, 18,560 of 18,560, before any verdict is scored.
  **A call that locates nothing and differs from 06m's is a stop,** before anything else is scored.
- **Planned stop 4,** before the references run: the 24 calls, their captures checked, and the
  projection.

**Both agents:**
- They save every command and its output; the review reads the outputs, not the log.
- **Launch, one check, end the turn.** Each reports the run id, the run directory, the detached
  Python pid, the `sh` pid and the first checkpoint line. You wait on each pid once, in the
  background, and resume the agent with the final `status.json`.
- **Every deviation is numbered and classified** before an agent reports.
- **They stop and ask** at any of prompt §8's conditions.
- **Each agent's records** are `log.md`, its section for appending to the code log;
  `verification.md`, its part of `VERIFICATION.md` §7 with links in the `07/…` form; `issues.md`;
  and `out/`, `scripts/` and `state/`, with a `KEPT_ON_MAC_STUDIO.txt` listing every pickle, run
  directory and file over 2 MB with SHA-256 and size.

**The marker:** `touch` a file in the session scratchpad immediately before the first dispatch, and
hash A3 v2's scratch copy, 06m's census and R26 before and after.

### B.3 The review — on the Mac Studio, after both agents report

Use scratch readers of your own, never the agents' scripts unless a check says to re-run one.
Compare floats by bit pattern (`float.hex`), never `isclose`.
1. **Scope.**
   - In all three trees `git status --short --ignored` is empty and each `HEAD` is as dispatched.
   - `find var -newer <marker>` lists only §B.2's paths.
   - A3 v2, the scratch copy, 06m's census and R26 are unchanged, by your before and after hashes.
2. **The runs.** `runregistrykit list` shows R27, R28, `p1-references`, `k0`, `census` and
   `references` `done`. Each manifest names `levin-phase-floor`, `07` and its slug, with
   `git_dirty: false`, and the SHA of the tree it ran from.
3. **The fingerprints.** Both `1b332ffa…`. Re-run `store fingerprint` on each yourself, without
   `--write`.
4. **The rehearsal values.**
   - Re-run `tools.store_value_diff` on R27 and R28 yourself. It must reproduce the saved JSON.
   - Outside `QuadSourceIntegral`, every value table identical.
   - Inside it, by your own reader: every untouched row bit-identical; the agent's tabulation of
     the rest complete; one moved row's `total` spot-checked in both stores.
   - S0, by your own comparison of R26 and R27.
5. **S1, S2, the readers and the lock.**
   - S1's twelve calls re-scored from R28 against `scores.json`, each with its located point.
   - S2's two rows: each located point, and the references' four readings agreeing.
   - The readers all exit 0, and R28's files are identical before and after.
   - The lock transcript shows `acquire`, `set-pid` with each driver's pid, and `release`, each
     exit 0. Run `status` and `ray-processes` yourself, each as a command of its own.
6. **V0.** Its three clauses are in the saved outputs, and V0 ran before the launch. Planned stop 2's
   re-runs are in your records.
7. **The census verdicts, re-scored** from the two checkpoints by your own readers:
   - **V1:** every untouched call against 06m's, bit for bit, at both rtols, and the untouched
     rows' totals;
   - **V2:** the returned Levin leaves holding a sign change of $\theta'$;
   - **V3:** both directions of the agreement between location and scan;
   - **V4:** every scored call against its reference, from the readings;
   - **V5:** leaves at `depth_max` in calls that located a point or guarded a leaf, with over-budget
     units counted as false.

   Each verdict is holds, fails or undecided against prompt §6's rule as written, with its numbers.
   The descriptive items are present.
8. **The predictions.** `state/predictions.md` is later than V0's output and earlier than the
   census's `manifest.json`. It quotes prompt §6 word for word, and fixes V4's 24 calls. The harness
   diff against 06m's copy matches the deviations.
9. **The records.** Each `log.md` follows campaign README §5.1 and classifies every deviation. Each
   `verification.md` is written as part of `VERIFICATION.md` §7. Each `issues.md` exists. Each
   `KEPT_ON_MAC_STUDIO.txt` lists what it should.

Write your verdict on each check to `var/levin-phase-floor-07/review.md`, with the commands you ran
and what they printed. End with a report whose first line is the verdict, pass or stop. Then give:
- R27 and R28's fingerprints, and S0;
- the rehearsal comparison: rows moved, rows located or guarded, the largest change against
  `total_abserr`, convergence changes;
- S1 and S2;
- V0–V5 with their numbers;
- the depth distribution at both rtols against 06m's;
- the over-budget units, and the census's wall and regions against 06m's;
- what the analytic calls located.

### B.4 After the review — on the laptop

The laptop copies `var/levin-phase-floor-07/` back without pickles, files over 2 MB or the readers'
extract, and checks every file by SHA-256 against a list made on the Mac Studio. It copies the small
run directories (R27, R28, `k0`, `p1-references`). It then re-checks, from the copied records:
- S1 on the twelve w calls, against `scores.json`;
- V4 on z12, and on two of the 24;
- one untouched rehearsal row and one moved row, in `float.hex`.

Then it asks the user about the header's issue and anything the agents' `issues.md` propose, and
makes **one records commit**, pushed, containing:
- the records in `docs/levin-phase-floor/07/rehearsal/`, `…/census/` and `…/review/`:
  - scripts byte-identical and unformatted, their directories added to `CLAUDE.md`'s `black`
    paragraph;
  - CSVs force-added if `.git/info/exclude` hides them, byte for byte (`.gitattributes` keeps them
    `-text`);
  - a `KEPT_ON_MAC_STUDIO.md` listing what stayed behind, with SHA-256;
- both `log.md`s, appended to `logs/07-stationary-point-guard.md`, with the verdicts in its
  Predictions and verdicts;
- both `verification.md`s, as `docs/levin-phase-floor/VERIFICATION.md` §7;
- the board, with 07's row and the Decisions row for the user's rulings;
- **the header's issue,** on the user's word:
  - with §5 and V0–V4 holding, closing it is proposed;
  - if any fails or is undecided, the user decides with the numbers in front of them;
  - V5 failing does not by itself keep it open. It reopens G2's L2 for the user (Decisions, **G2
    L2, decided**), and is recorded as a finding;
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 07, and the campaign README's.

**Kept until the user says otherwise:** the R27 and R28 stores, the census run directory (08's
"before"), the references' run directory and both worktrees. **Asked at the records commit:** the
fate of 07's four stashes and `var/levin-phase-floor-07/wip/`, and of the R23–R26 stores and the
censuses 07 no longer needs. The scratch copy of A3 v2 stays for 08.

### B.5 Addendum: Part B's dispatch

*Written 2026-11-07, at Part A's landing.*

**Part A** is the code commit **`ad732887`**, whose parent is **`91472dbe`**. It was reviewed in
§A.3's eleven checks, all passing, and pushed on the user's word: *"Yes, please land it. Push. Then,
write, commit, and push the dispatch addendum for Part B."* Two orchestrator commits came before it,
both pushed:
- `945c8668` opened `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`;
- `91472dbe` amended the prompt at §2's stop (its §9).

**The user's rulings at §2's stop** are prompt §9 points 1–4, with the user's words, and the board's
Decisions row **07 §2 stop**:
- §2.2 ran at `rtol` $10^{-9}$, and its rules gave **$\Phi_w = 2\pi$, $\Lambda_c = 4$**;
- §1.3 is QUADPACK's `iroff1` counted per window: limit 6, value tolerance $10^{-5}$, error ratio 0.99.
  A stopped window's leaves are returned floor-limited;
- a key-set assertion admits §1.5's six new result keys;
- §2 was re-run once, on the code with §1.3 built.

**The trees, by SHA:**
- **R27 and V0's K0 run** at **`91472dbe`**, in the worktree `SecondaryGWKit-lpf07-91472db`. Its
  production code is the dispatch SHA `bd21bf23`'s, which is 06m's: `git diff --stat bd21bf23
  91472dbe -- . ':!docs' ':!prompts'` is empty. Against R26's tree `38f6a828` it differs only in
  `CLAUDE.md`. So S0 (R27 against R26) is a reproducibility check, as written.
- **R28, the census $g$, V4's references, the comparisons and the readers** at **this addendum's
  commit**, in the worktree `SecondaryGWKit-lpf07-<this commit's short SHA>`. Its code is
  `ad732887`'s. `git diff --stat 91472dbe ad732887 -- . ':!docs' ':!prompts'` touches only:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_stationary.py` (new) and `test_levin_trust_yardstick.py`;
  - `ComputeTargets/QuadSourceIntegral.py`, the four counts;
  - `ComputeTargets/tests/test_quadsource_levin_guard_counts.py` (new);
  - `CLAUDE.md`.

**What Part A's review found that Part B must carry:**
- **Every number in the review was re-run on the laptop:**
  - the seven roots, at 103, 781, 39, 156, 280, 127 and 64, all OK;
  - §2.2's 124 units, bit-identical on the committed module;
  - z12 and w03 replayed bit for bit. z12 is honest at 0.718 of its error, against 1,681× at the
    parent.
- **Windows stop while some of their leaves are still falling** (the agent's O8). This happens on 8
  of §2.3's 14 calls at $2\pi$, and all 8 are honest and converged. **The census reports it, as
  descriptive:** for each call that locates a point, every window `stationary_windows` records as
  stopped, with `leaves_falling_at_stop`, by rtol, and whether the call converged.
- **Location costs time** (the agent's O5). A rate that does not vectorise is evaluated 4,097 times
  per call. The ComputeTargets root took 442 s here against 291 s at dispatch. The census's 300 s
  per-unit budget may be reached more often than in 06m's, which had none over. Over-budget units
  are counted under V5 as the prompt says, and **reported with their wall time against 06m's**. That
  is a finding, not a stop.
- **The bar on window leaves protects no measured value.** Under breakage (iii) test 5's value stays
  honest (0.0057 of its error). Window leaves carry the Clenshaw–Curtis floor, which has no
  $1/\theta'$ in it. Nothing for Part B to do. If V2 finds a returned leaf holding a sign change,
  check whether it is guarded or a window leaf before scoring.
- **The value tables differ from R27 only inside `QuadSourceIntegral`,** and there only in rows
  whose group or analytic calls located a point or guarded a leaf, plus the four new counts. §0.1
  already excludes those counts from "value-identical".
- **Not measured here:** `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`. Production spans
  are non-dyadic, so the parent's leaves carry that floor too. The census does not score it.

**§B.0 and §B.1, re-checked read-only on the Mac Studio, 22:47–22:50 UTC, 2026-11-07:**
- **The main checkout** is `handover-remedial` at `28ebaf07`, with `git status --short` empty.
  `git worktree list` shows the main checkout alone; 06g's and 06m's worktrees were removed
  (`a8c7854c`).
- **The registry** holds 62 runs, all finished, nothing `running` or stale.
- **The lock and Ray:**
  - `rehearsal_lock status`: free (`~/.raypoolkit/rehearsal-lock` does not exist);
  - `ray-processes --ray-temp-dir /tmp/ray-sgk`: none. `/tmp/ray-sgk` holds only session
    directories from 2026-10-25.
- **Load and disk:** the load is 2.75 / 3.04 / 3.17, and 136 GiB is free (85 % used).
- **No 07 paths yet:** there is no `var/levin-phase-floor-07/` and no
  `var/datastores/levin-phase-floor-07-*`. **But `var/levin-phase-floor-07-leaves/` exists** (28
  files, 8 MB, last written 2026-10-29). It is the working directory of 06e's leaf study of 07's
  fourth stop, whose records are committed under `docs/levin-phase-floor/06e/investigations/07-leaves/`.
  It is not Part B's, and Part B's paths (`var/levin-phase-floor-07/…`) do not collide with it.
  **No agent writes or reads it.** Whether it is kept is asked at 07's records commit.
- **The inputs, by SHA-256 against their recorded lists,** 160 files, all OK:
  - the scratch copy's five files, against `06e/review/checks/a3v2_copy_after.sha256`, with no
    `-wal` or `-shm`;
  - 06m's census run directory `levin-phase-floor-06m-census-20261106T102153`, all 150 files
    against `06m/census/KEPT_ON_MAC_STUDIO.txt`, checkpoint `ea5bc3ab…` among them;
  - R26's five store files against `06m/rehearsal/KEPT_ON_MAC_STUDIO.txt`, primary `e90e26c2…`.

  R25's and R26's run directories are present.
- **06g's captures:** `var/levin-phase-floor-06g/references/state/capture/` holds 102 entries,
  19 MB.
- **The venv** is Python 3.13.16, with `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
  0.2.0, `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy` 2.2.4, `scipy` 1.16.3 and `mpmath` 1.3.0. The
  laptop's are the same.

**The worktrees** are made by the laptop on the user's word to dispatch, after this commit is
pushed, detached and with no venv: `SecondaryGWKit-lpf07-91472db` and
`SecondaryGWKit-lpf07-<this commit's short SHA>`. The Mac Studio's main checkout fetches first,
which moves no working tree. §B.0's checks are repeated then. The Mac Studio orchestrator session
is then started in the code worktree, through the saved `MacStudio` connection, with this note as
its brief. Its folder must be trusted in the app first.

### B.6 Addendum: Part B's first stop, and the amendment it led to

*Written 2026-11-08, at the hold before the census.*

**Dispatch.** On the user's word (*"Please start Part B on the Mac Studio"*):
- §B.0's checks were repeated at 23:56 UTC on 2026-11-07, all read-only. The checkout was clean at
  `28ebaf07`, all 62 runs finished, the lock free with no Ray process, no Part B path, load about
  2.6 and 136 GiB free. The 160 input files matched their hashes.
- The laptop fetched and made `SecondaryGWKit-lpf07-91472db` at `91472dbe` and
  `SecondaryGWKit-lpf07-7460035` at `74600350`, detached, with no venv.
- It started the Mac Studio orchestrator session in the second. The marker was touched and 181
  files hashed before the first dispatch.

**The rehearsal agent, to stop 1 and past it:**
- **The rehearsals.** R27 (`levin-phase-floor-07-r27-parent-20261108T000634`) and R28
  (`-r28-guard-20261108T001916`) both give `1b332ffa…`.
- **The comparisons,** each checked by the Mac Studio orchestrator's own readers:
  - S0 holds: R27 equals R26 in all 143,831 `QuadSourceIntegral` fields and in every other value
    table, and the label differences are only the job name and the stamp.
  - Outside `QuadSourceIntegral`, R28 equals R27.
  - The 640 rows that located nothing are bit-identical.
  - 17 rows went from unconverged to converged, none the other way, and no leaf was guarded.
- **Stop 1:**
  - **The agent's question:** the store keeps only the counts, so how should it get each located
    point's kind and $z$, which §5 item 4, S1 and S2 need?
  - **The user:** *"I approve the replay. Also, add a reference to check 2 or 3 of the `q-r`
    calls, to settle whether R28 is honest. Hold Part B before the census."*
- **After stop 1:**
  - The replay reproduced all 470 calls that located a point, each bit for bit against R28.
  - S1 holds: 12 of 12 honest, the worst 0.218.
  - S2 holds: R28 at 0.020 and 0.263 of its error, R27 at $5.6\times10^5$ and $4.1\times10^4$.
  - The readers exit 0 and the lock was released.
- **The zero-rate case.** 26 `q-r` calls with $q = r$ located 4,097 points each. The three
  referenced show both stores honest (`[07-location-on-an-identically-zero-rate]`).
- **The census agent was not dispatched.**

**The user's rulings at the hold,** after the orchestrator explained why the C2 gate does not catch a
constant phase, what the integrand is there, the partly flat case and the resonance:
- *"Yes, rule A alone, please."*
- *"Open a separate issue for the `6pi` gate. We'll deal with this afterwards, as you say."*
- *"Record the resonance issues as belonging to the one-loop build and point at the planning
  document."*
- *"Please make the fix as an amendment of 07."*

The board's Decisions row is **07 Part B stop 1**. The amendment is prompt §10.

**Part A′ — the amendment's code, on the laptop.**
- **Dispatch.**
  - A fresh-context Opus subagent, in the main checkout at the amendment's commit.
  - Its scope is prompt §10, with §1, §3, §4 and §8 as they bear on it. It does not read
    `orchestrator/`.
  - Its gate is the prompt's (the stashes and `wip/` included), with `HEAD` at the amendment's
    commit and the seven roots' counts taken there (103, 781, 39, 156, 280, 127, 64 expected).
  - No planned stop. It stops at §8's conditions, and at §10's (a rung of §2 that moves; a test
    whose assertion moves).
- **The review, on the laptop, before landing:**
  1. **Scope.** The diff against the amendment's commit touches only:
     - `AdaptiveLevin/levin_quadrature.py`;
     - `AdaptiveLevin/tests/test_levin_stationary.py`;
     - `AdaptiveLevin/tests/test_levin_trust_yardstick.py`, the key set alone;
     - §2's new output directory under `docs/levin-phase-floor/07/calibration/out/`;
     - the log, the board and `docs/OPEN_ISSUES.md`.
  2. **The construction, by AST against `ad732887`.**
     - Only `_locate_stationary_points`, the guard's handling of a skipped phase, the code reporting
       the new key, and their docstrings change. Every other function is AST-equal.
     - Rule A compares against `SIX_PI` by name. Its estimate costs no evaluation beyond the scan.
  3. **§2's re-run.** Every call with no phase skipped is bit-identical to run 2. Re-run §2.2's
     124 units myself on the committed module, against run 2's outputs. The skipped calls are
     listed.
  4. **The tests.** Read test 13 (a)–(d) against §10. Re-apply breakages (x) and (xi) myself, each
     caught by its test. The seven roots rise only by the tests added.
  5. **The records.** The log's section is in README §5.1's form, with deviations numbered on from
     Part A's. The issue is moved to §4, and the index is corrected by counting.

  The code is pushed only on the user's word. A records commit then writes this note's §B.7, Part
  B's resumed dispatch, and its SHA is Part B's new dispatch SHA.

**Part B resumed, at the new dispatch SHA** (§B.7 gives the details):
- **The worktrees.** A third worktree, `SecondaryGWKit-lpf07-<new short SHA>`, made by the
  laptop after a fetch.
  - R27's worktree and R28's (`…-7460035`) are kept, with R28's store and records.
  - The Mac Studio orchestrator session goes on, given the new SHA by message. It reads this note
    and the prompt in the new worktree.
- **The rehearsal agent, resumed or fresh.**
  - **R29**, slug `r29-guard-rule-a`, at the new worktree, under the lock, with 06m's command
    line, giving `1b332ffa…`.
  - **R29 against R28:** they differ only in Levin calls whose phase Rule A skipped and which
    located a point at R28, and those calls equal R27's.
  - **R29 against R27:** as prompt §5 item 4.
  - **S1 and S2** re-scored on R29. S2's references from `p1-references` stand, being taken at
    R27's tree.
  - **The readers** on R29, then the lock released.
  - **Planned stop 1′:** before the census, to the user.
- **The census agent:** as §B.2, at the new worktree, with V3 as prompt §10 sets it. A phase
  Rule A skipped is set aside, its calls counted and listed.
- **Kept until the records commit:** R28's store, the `-7460035` worktree and the stop-1 records
  in `var/levin-phase-floor-07/`.

### B.7 Addendum: Part B's resumed dispatch

*Written 2026-11-08, at the amendment's landing. This addendum's commit is Part B's new dispatch
SHA.*

**The amendment landed.**
- Prompt §10 is `2ddd3306`. Its code is **`adffa21b`**, "Skip locating a phase whose variation is
  below the C2 gate". Both are pushed.
- `adffa21b` passed §B.6's five checks. The board's Decisions row is **07 amendment lands**.
  - **Scope:** only the allowed paths changed.
  - **By AST against `ad732887`:** only `_locate_stationary_points`, `_nodal_guard` (with the new
    `_changes_sign`), the threading of the skipped phases through `_adaptive_levin`,
    `_adaptive_levin_subregion` and `_impl`, and the result key `location_skipped_phases` change.
  - **§2.2:** 124 units re-run, bit-identical to run 2.
  - **Breakages (x) and (xi)** re-applied and caught.
  - **The seven roots:** 107, 781, 39, 156, 280, 127, 64, all OK.
- **The user:** *"Yes, please push both, then write and push §B.7, and send to the Mac Studio to be
  executed."*
- **§2 does not exercise Rule A.** No call of §2.2, §2.3 or §2.4 has a phase below $6\pi$, so all
  of §2 is bit-identical to run 2. Test 13 and R29 are where the rule is measured.
- **O9 is on `[07-c2-gate-threshold-was-never-measured]`:** a phase Rule A skips can still reach the
  Levin solve, because the gate's node-based estimate reads high. There the nodal guard covers it.

**The trees:**
- **The parent worktree** `SecondaryGWKit-lpf07-91472db` (`91472dbe`), unchanged. It runs V0's K0
  run.
- **The new code worktree** `SecondaryGWKit-lpf07-<this commit's short SHA>`, detached at this
  commit, with no venv.
  - Its code is `adffa21b`'s.
  - `git diff --stat 74600350 <this commit> -- . ':!docs' ':!prompts'` touches only
    `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_stationary.py`,
    `AdaptiveLevin/tests/test_levin_trust_yardstick.py` and `CLAUDE.md`.
  - It runs R29, the comparisons, S1 and S2's re-scoring, the readers, the census $g$ and V4's
    references.
  - The note and the prompt are read here.
- **The old code worktree** `SecondaryGWKit-lpf07-7460035` and R28's store are kept, read-only, as
  the record of the zero-rate case. No new run is made in that worktree.

**What the Mac Studio orchestrator does, in order.** It is the same session, told by message.
1. **Re-check before the lock.**
   - Every tree is clean, and each `HEAD` is as dispatched.
   - `runregistrykit list` shows nothing running or stale.
   - The lock is free, and `ray-processes` lists nothing; each is a command of its own.
   - Touch a new marker, and re-hash the 181 protected files against the first before-hashes.
2. **The rehearsal agent, resumed or fresh, on prompt §5 with §10 as written.**
   - **R29:**
     - slug `r29-guard-rule-a`, in the new worktree;
     - `--job-name levin-phase-floor-07 --campaign levin-phase-floor --prompt 07`;
     - 06m's command line;
     - a `--purpose` naming the host, the tree's SHA, the code commit `adffa21b` and the fingerprint
       `1b332ffa…`.
   - **The lock:** `acquire`, `set-pid` with R29's driver, held through the comparisons, the replay
     and the readers, then `release`. Each lock command is a shell command of its own.
   - **R29 against R28,** by the orchestrator's own reader, as well as the agent's:
     - outside `QuadSourceIntegral`, every value table identical;
     - inside it, a Levin record differs from R28's only if R28's count of located points was
       nonzero and R29's is 0. Every such record equals R27's in value, abserr, convergence and
       region count, bit for bit. The 26 `q-r` calls with $q = r$ are expected, and any other is
       listed;
     - every other Levin record bit-identical to R28's, its four counts included;
     - rows that differ hold such a record, and are tabulated.
   - **R29 against R27:** as prompt §5 item 4, with `tools.store_value_diff --json` and 05's
     `classify_non_qsi.py`, and the rows tabulated.
   - **The replay:** the calls of R29 that locate a point, by the harness the stop-1 replay used,
     run from the new worktree. Each is checked bit for bit against R29 before use. About 444 are
     expected: 470 less the 26.
   - **S1 and S2,** re-scored on R29. S2's references from `p1-references` stand, being taken at
     R27's tree.
   - **The readers** on R29: exit 0, and R29 byte-identical before and after. Then the lock is
     released, and `ray-processes` lists nothing.
   - **S0** stands: R27 is unchanged.
3. **Planned stop 1′, before the census, to the user, through the laptop.** Report:
   - R29's fingerprint;
   - R29 against R28 and against R27;
   - the replay, S1 and S2;
   - the readers and the lock;
   - R29's wall time and summed regions against R28's.
4. **The census agent, only on the user's word at stop 1′.** As §B.2, with the census tree $g$ the
   new worktree, V0's K0 run in the parent worktree, and V3 as prompt §10 sets it. Planned stops 2,
   3 and 4 as §B.2.
5. **§B.3's review** at the end, with the run checks taking in R29. R28 is checked unchanged, not
   re-reviewed.

**Unchanged:**
- git is read-only on the Mac Studio;
- `--cpus 6`, never `ray stop`, never another session's run;
- A3 v2, its scratch copy, 06m's census and R23–R26 are never written.

### B.8 Addendum: Part B from stop 1′ to its end, the records, and two corrections to §B.7

*Written 2026-11-10, at 07's records commit. §B.7 is left as written; its corrections are at the end
of this section.*

**R29 and stop 1′** (2026-11-08, 12:33–13:35 UTC). The Mac Studio orchestrator's re-check before
the lock passed: the three trees clean at their SHAs, nothing running or stale, the lock free, the
181 protected files unchanged.
- **R29** `levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, at `e182bc7a`: `done`, exit 0,
  `1b332ffa…`, 9.6 min (R27 9, R28 16.5).
- **R29 against R28,** by the agent's comparison and the orchestrator's own
  (`review/scripts/qsi_rulea_compare.py`): outside `QuadSourceIntegral` identical; 13,087 of 13,177
  Levin records bit-identical; 90 changed, every one located at R28 and skipped by Rule A. 86 equal
  R27 (the 26 `q-r` calls and 60 analytic `2pt5` J4/Y4 blocks). **4 do not:** the J4 and Y4 blocks
  at $(3.162\times10^5)^3$, $z = 1.329\times10^9$, and $(10^6)^3$, $z = 4.018\times10^9$, where Rule
  A skips the phase ($V$ 15.113, 15.896) but the C2 gate's estimate on the root leaf (19.160,
  20.196) exceeds `SIX_PI`, so the root leaf reaches the Levin solve and the nodal guard catches it
  (O9's case). §B.7's rule failed as written, and the agent stopped, the lock held.
- **The laptop's recommendation,** relayed with the agent's question: go on, the four reported as
  findings; the rule was the orchestrator's error, corrected below; add a check on the four. The
  user: *"Yes, please proceed as you recommend."* The four blocks were replayed at R27's, R28's and
  R29's trees and referenced by composite Gauss–Legendre split at the root, QAGS beside it. All three
  stores are honest on all four, at most $6.2\times10^{-4}$ of abserr, every reference's stated error
  at least 1,990× below every abserr. R29's J4 declarations are 19× and 11× R27's.
- **Then:** the replay reproduced all 470 records; S1 and S2 hold on R29, each call equal to R28's;
  the readers exit 0 and leave R29 unchanged; the lock was released. Summed regions are back to
  R27's level: 114,460 (R27 114,826, R28 227,883).
- **Planned stop 1′.** The orchestrator's checks (`review/stop1p/notes.md`) agreed with the agent in
  every figure. The user: *"Yes, please go ahead. There seems no reason not to. We may have to come
  back to the O9 issue but that is best done after the census anyway."*

**Stop 2** (2026-11-08, 13:35–14:30 UTC). The census agent was dispatched at 13:34:50Z. Its K0 run
`levin-phase-floor-07-k0-20261108T135240` gave V0: (a), (b) and (c) all hold. Its one question:
V4's "all six $k$ that carry them" was the prompt's factual slip, since all eight $k$ carry
stationary-point calls. The orchestrator re-ran two K0 units at the parent and one stationary-point
unit at $g$ twice, by its own reader: 0 unclassified differences, deterministic
(`review/stop2.md`). The predictions were written at 14:10:10Z, after V0 and before any census. The
user: *"Please keep all 8 `k` and launch the census"*.

**The census and stop 3** (2026-11-08, 17:11:37Z to 23:30:11Z). `levin-phase-floor-07-census-20261108T171137`,
six workers, the lock held: 18,560 of 18,560 units, 22,708 s against a projection of 21,458 s, no
unit over budget, checkpoint `03dfe786…`. The orchestrator's own comparison with 06m's census, by
three scripts (`review/stop3.md`), found the 284,514 calls that locate and guard nothing and the
16,104 units made only of them equal to 06m's. 4,968 calls are active: 4,940 locate and 28 guard
without locating. The 28 are O9's case in production: J4 and Y4 in 7 rows at both rtols, $V$
15.60–18.58, each guard catching one leaf, all converged. The user: *"Yes, please do that"* (step 7).

**Stop 4** (2026-11-09). Step 7 scored V1–V3 (holding) and V5 (failing: 23 calls). V4's 24 captures
reproduced the census bit for bit, and the projection for four readings on all 24 was 775,468 s,
1,163,202 s with the margin, about 9 days: a §6 stop. The orchestrator recounted V5 itself, finding
the same 23 units and two slips in the agent's prose (2–54 leaves, not 2–14; three converged, not
two), which the census log corrects (its deviations 38 and 39). The user, on the readings: *"Take
your recommended option of FCC at both resolutions plus QAWO where it is projected to be 30 minutes
or less. It costs only a few hours for an independent check of many calls. Calls without QAWO should
be scored on whatever references we do have, either one or two FCC readings."* And, beyond §6:
*"Yes, please do the cross-check on `rtol`s and the list of calls where `06m` and `07` differ by more
than the stated error. We can decide on a sample of references to sort of what these difference
really show, later."*

**The cross-check** (2026-11-09). Its rule was fixed before any result
(`census/state/xcheck_rule_20261109.md`, mtime 00:57:12Z; first result 00:58:21Z). 2,484 pairs,
$r_g \le 1$ in 2,482 (the exceptions `3-14501`, 1.39, and `3-35006`, 1.08) and $r_{06m} \le 1$ in
2,432; 843 calls disagree beyond $\max$(abserr): (a) 31, (b) 0, (c) 786 (396 rows), (d) 0, (e) 26.
The orchestrator recounted it call by call with its own reader, exactly (`review/xcheck/notes.md`).

**The references** (2026-11-09T00:53Z to 2026-11-10T02:31Z).
`levin-phase-floor-07-references-20261109T005328`: 74 readings (FCC r1 and r2 on all 24, QAWO r1 and
r2 on 13), about 25.6 h against 89,230 s projected, the 48 h backstop never reached. V4 fails: 28 of
33 honest; v08, v09, v12, v17 and v22 at 2.21–3.16× $\text{abserr}_g$, each converged with its window
stopped by `iroff1` while 4–13 leaves still fell (06m 497–30,400×). z00 and z12 are at 0.718, Z3's
seven at most 0.0021.

**The review** (2026-11-10, `review.md`): all nine checks of §B.3 pass, with V4 and V5 re-scored
exactly and failing. Two notes: the drivers living under the main checkout's `var/`, the registry's
`git_head` is `28ebaf07` for the k0, census and references runs, and the tree each ran from is named
in the manifest's `purpose`; and an mtime change on the main checkout's `.git` during the census
agent's step 10, with no git state changed and no cause identified.

**On the laptop** (2026-11-10). `var/levin-phase-floor-07/` was copied back without pickles, files
over 2 MB or the readers' extracts, and the small run directories (R27, R28, R29, `k0`,
`p1-references`, `references`, three launch logs); every file matched the Mac Studio's SHA-256 lists.
The re-checks of §B.4, from the copied records (`VERIFICATION.md` §7.7):
- **S1** recomputed exactly from `rehearsal/out/s1_scores.json`: 12 of 12 honest, worst 0.217573
  (w11), every ratio equal to the record; R29's twelve equal R28's; all twelve `ref_30` strings
  verbatim in 06g's committed V1 (`docs/levin-phase-floor/06g/references/out/scores_addendum6/scores.json`);
- **V4** in exact Fractions with the census values read from both checkpoints on the Mac Studio: v00
  0.0196667 (06m 0.00054827), v08 3.1596 (06m 30,433.3), z12 against 06g's final four readings
  0.7181095 (06m 1,681.299), each equal to the agent's and the Mac Studio orchestrator's;
- **two rehearsal rows** read from the R27 and R29 stores in `float.hex`: the moved row $(10^5)^3$ at
  $z = 2284.251956909201$ equal to `rehearsal/r29/out/qsi_rows_r27_r29.csv`, an untouched row at
  $z = 49531657077.16097$ equal in both stores, and all 692 untouched rows equal in `total` and
  `total_abserr`.

**The rulings at the records commit** (the board's Decisions, **07 results** and **G2 L2,
reopened**):
- *"`[06-levin-accepts-a-region-across-a-stationary-point]`: I didn't address this, but yes, please
  close it"*: closed (board §4), on V2;
- O9's census calls (the 28 guarded, and 154 unguarded calls whose phase Rule A skipped yet with a
  sign change) added to the $6\pi$ issue: *"I think so"*;
- the 786 undecided calls: *"these are undecided because we don't have references for them? If so,
  then yes, we attribute them to 06m's errors for the time being, on this evidence."*;
- V5: *"I don't see what value `V5` has if there are other regions that go down to the depth cap."*
  V5 is scored fails as written. Its premise was stale: it was written on 2026-10-27 from 06a's
  census, where no Levin call reached the cap outside the stationary points, and 06g–06m's
  trust-test changes left calls at the cap outside the 720 (06m's census: 114 group calls
  unconverged there, 105 holding an untrusted leaf at `depth_max`), which the orchestrator did not
  re-check at Part B's dispatch. That was the orchestrator's error. Inside the 720, 62 of 07's 64
  unconverged group calls carry the same signature, so V5's failure says nothing specific about
  stationary points. G2's L2 is reopened as one question for 08: whether any call should reach the
  cap;
- what follows: *"I agree we fix the early roundoff stop and the O9 problem. These are issues where
  the problem and the fix are both clear."* and *"Include the cap experiment in prompt 08. That
  sounds like a better place to put it. We can make 07's records commit and the housekeeping commit
  now. Then we will need a prompt of the early roundoff stop and O9."* So
  `[07-iroff1-window-stop-can-leave-a-call-dishonest]` is opened and, with O9 of
  `[07-c2-gate-threshold-was-never-measured]`, assigned to a prompt **07a**, not yet written. 08
  gains the depth-cap experiment: the about 227 calls with a leaf at the cap (160 units, 1,352 s
  serial at cap 20) re-run at caps 22, 24 and 28, about 40 min on six workers at worst;
- the housekeeping, in its own commit: *"Yes, please go ahead with the cleanup as listed."*
  (`docs/levin-phase-floor/housekeeping-2026-11-10/README.md`). 07's stashes and `wip/` were dropped;
  the R27 and R28 stores were retired with their sidecars kept, among ten rehearsal stores; and the
  three 07 worktrees were removed. **Kept:** R29's store, the census and references run directories
  (08's "before"), `var/levin-phase-floor-07/` and A3 v2's scratch copy.

**Two corrections to §B.7,** which stays as written:
1. **§B.7 item 2 said every record R29 changes against R28 equals R27's.** That is wrong for the 4
   analytic blocks in which 07's nodal guard fires. The guard is absent from R27's code, so where
   Rule A skips a phase and the C2 gate still sends its root leaf to the Levin solve, R29 guards
   that leaf and R27 does not. Those 4 records differ from R27 in abserr, regions and, in one of
   each pair, `phase_limited`, and all three stores are honest on their references (ratios at most
   $6.2\times10^{-4}$). The rule should have read: a record R29 changes against R28 equals R27's
   unless R29 guards a leaf in it, and a guarded record is referenced.
2. **§B.7's count of located calls omitted Rule A on the analytic blocks.** It expected about 444
   located calls, 470 less the 26 `q-r` calls. Rule A also skips a phase in 64 analytic `2pt5`
   J4/Y4 blocks that R28 located (V 0.49–15.9), so R29 locates in 380 records (44 group and 336
   analytic), and 384 have a nonzero count with the 4 guarded blocks.

---

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. Stop at:
- the prompt's §8 conditions;
- any failed check in §A.3 or §B.3: report it, do not repair it;
- a held lock (exit 3).

**Never touch 07's stashes or `var/levin-phase-floor-07/wip/`,** beyond the prompt's §0.5 reads.
**Never write to A3 v2 or its scratch copy,** to 06m's census or to R23–R26. **Never move the Mac
Studio's main checkout,** never run `ray stop`, and never end another session's run.
