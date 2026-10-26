## §6 — the wider re-measurement, with §9's T1 (Mac Studio)

**Subject:** prompt 05a §6, with T1 as amended by §9. It was run on the Mac Studio (mps026298) by
the re-measurement agent, Ray-free and with no lock. It ran beside the R11/R12 agent, which held
the lock for its own builds.
- **Trees:**
  - $c$ is the code commit's parent `c4eaefd`, whose code is 05's $c$. It is in the git worktree
    `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-c4eaefd` (detached).
  - $t$ is the code commit `1fc349c`, in the main checkout at Part B's dispatch SHA `149c393`. Its
    code is `1fc349c`'s up to `bd38e50`'s comments.
- **Run:** `levin-phase-floor-05a-wider-20261026T123047`. It ended `done`, 109/109 units, exit 0,
  12:30:47Z–12:32:25Z.

**Commit:** none here. The orchestrator commits the records in 05a's records commit.

**Result:**
- **Gate T0 holds,** on all three clauses.
- **T1, as amended by §9 and scored on the reading fixed before the run, fails.** It fails in 36
  of 70 scored windows, all on the $t$ clause. Those 36 are exactly the windows the literal halving
  rule accepts while their samples step 4.9–352 rad each: aliased windows, where $\sigma_\theta$
  measures `numpy.unwrap` (issues X1).
  - In the 34 other windows, $\sigma_t \le 1.3\times10^{-5}\,\varepsilon|\theta_T|$.
  - $\sigma_c$ is 0.196–0.374 $\varepsilon|\theta_T|$ in all 70.
  - **Read by the true step** (unscored; both readings kept), T1 holds in all 70.
  - The question of which reading is the rule's went to the user after the launch. It is not yet
    answered.
- **Added after the user's decision (relayed about 13:23Z; deviation 16):** T1 is scored by the
  true step. **T1, scored by the true step: holds**, in 70 of 70 scored windows.
  - $\sigma_c$ is 0.198–0.374 $\varepsilon|\theta_T|$.
  - $\sigma_t$ is $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon|\theta_T|$, at most 0.013 of
    its bound.
  - No window is unresolved, and no re-probe was needed: every window, the 21 below
    $10^{-11}$ included, is resolved within the kept levels.
  - The literal reading, which fails in 36 windows as above, now sits beside it, unscored.
- **T2 holds** (39 of 39). The largest depth under $t$ at `1e-7` is 11.
- **T3 holds** (39 of 39). The largest $|t-c|$/`total_abserr`$_c$ is $4.7\times10^{-6}$.
- **T4 fails** (29 of 31): `mid-G-r-3` at 1.29 and `mid-G-3` at 1.70. These are the same rows and
  ratios as 05's L3 under $c$. T4 holds on all 11 rows capped under B1, largest 0.166.
- **No unit went over budget.** The largest $t$ wall/budget is 0.861 (`t-1e-8-R3`).
- **The store is byte-identical:** fingerprint `0e32f24a…`, `problems: none`.
- **No region count, depth or convergence flag changes** in any of the 219 group calls between
  $c$ and $t$.

### Predictions and verdicts

Prompt 05a §6's predictions are copied unchanged below. §6's T1 is marked "replaced before the run
by §9", with §9's T1 after it. `state/predictions.md` holds the same text, quoted word for word
with §9's T1 in place of §6's, plus its "Readings fixed before the run". It was written at
12:30:03Z: after gate T0 held (12:28:33Z) and before the launch (12:30:47Z).

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **T1 — the floor is there, and goes.** *(Replaced before the run by §9, 2026-10-26.)*
  - **The rule:** in every probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad, sampled in
    $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
    $\sigma_\theta \le \max(10^{-3}\varepsilon|\theta_T|,\ 10^{-13}\ {\rm rad})$ under $t$.
  - **Holds** if true in every such window.
  - **The $u$-sampled $\sigma_\theta$ under $t$** is not scored. Wherever it exceeds
    $0.1\,\varepsilon|\theta_T|$, report it against $\varepsilon\,\tfrac{z}{1+z}|d\theta_T/du|$, and
    propose an issue for the abscissa term in the log's "Observations not acted on".
- **T2 — the cap does not return.**
  - **The rule:** on every row (all 39), under $t$ at `1e-7`, no group has a region at depth ≥ 18.
  - **Holds** if true on every row; an over-budget unit counts as false.
- **T3 — $T$'s errors were within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_t - {\rm total}_c| \le$ `total_abserr`$_c$,
    at `1e-7`.
  - A failure measures what $T$'s two errors cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **T4 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $t$ units in budget,
    $|{\rm total}_t(10^{-7}) - {\rm total}_t(10^{-8})| \le$ `total_abserr`$_t(10^{-7})$.
  - 05's L3, the same rule under $c$, held on 29 of 31.

**T1, amended** (prompt §9, replacing §6's T1 before the run):
- **The rule:** in every resolved probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad,
  sampled in $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
  $\sigma_\theta \le \max\big(10^{-3}\varepsilon|\theta_T|,\ \varepsilon(k\max|p_z| + 2\pi),\ 10^{-13}\ {\rm rad}\big)$
  under $t$.
- **Holds** if true in every such window. **Fails** if false in any. **Undecided** if none is
  false but some window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad is unresolved.
- **Also report, unscored,** in how many windows the term $\varepsilon(k\max|p_z| + 2\pi)$ is the
  largest of the three, and $\sigma_\theta/(\varepsilon|\theta_T|)$ under $t$ in those windows.

**Descriptive, not scored:**
- $|{\rm total}_t - {\rm total}_c|/|{\rm total}_c|$ and $/$`total_abserr`$_c$ on every row;
- regions and maximum depth per group, $c$ against $t$, with the `Gqr` groups and the low-$z$ end
  of each sub-interval shown apart;
- `total_converged` under $c$ and $t$;
- **the kinds of non-convergence** under $t$ at both tolerances, by the orchestrator's
  `prompts/levin-phase-floor/measure/convergence_kinds.py`, copied unchanged, recorded by SHA-256,
  and run as `convergence_kinds.py <this run> --tree t --capped-run <05's run>`. It reads the
  `checkpoint.jsonl` and `unit_*.pkl` layout of 05's run, which the harness keeps. The same
  measurement under 05's $c$ is `prompts/levin-phase-floor/measure/out/convergence_kinds.txt`.

**Readings fixed before the run** (`state/predictions.md`):
- The "largest step of the unwrapped samples" is `max|diff(numpy.unwrap(θ))|` of the 65 $z$
  samples, as test 3's guard computes it.
- The window is $[z_{\min}, z_{\min}(1+w)]$.
- $\varepsilon = 2^{-52}$.
- T2 and T3 are taken over the 39 rows and T4 over the 31 stored at depth 20. The 11 rows capped
  under B1 are reported apart.
- An over-budget $t$@`1e-7` unit makes T3 undecided on its row.

**Verdicts** (`out/wider_05a.txt`; T1 re-scored independently from `state/probe_windows.json`,
`out/cmd/24_rescore_t1.txt`):

| | Number | Verdict |
|---|---|---|
| **T0 (a)** | `git diff --stat 120fdfe c4eaefd -- . ':!docs' ':!prompts'`: `CLAUDE.md` alone (5+, 3−) | holds |
| **T0 (b)** | $c$@1e-7 against 05's `c-1e-7-*` on `low-G+r-1` and `ctrl-low-1`, by `==` and `float.hex`: `total`, `total_abserr` and all 7 calls' value and abserr, 16 doubles per row, identical. So are the whole result summary (68 floats), the call summaries, `qs_check`, the dispatcher's counts and flags, and every region array of 05's pickles | holds |
| **T0 (c)** | two in-process $t$ runs on the same two rows: result, calls, regions, probe summaries and every probe sample (31,200 and 31,070 values) bit-identical | holds |
| **T1 (§9), as written** | 91 windows on 37 rows, 70 with $\varepsilon\lvert\theta_T\rvert \ge 10^{-11}$, 0 unresolved. True in 34, false in 36: the $c$ clause holds in all 70 ($\sigma_c$ 0.196–0.374 $\varepsilon\lvert\theta\rvert$); the $t$ clause fails in 36, $\sigma_t$ 0.016–0.23 $\varepsilon\lvert\theta\rvert$, up to 232× the bound. The 36 are exactly the scored windows whose true step per sample at the level used is ≥ π/2 (4.9–352 rad); elsewhere $\sigma_t \le 1.3\times10^{-5}\,\varepsilon\lvert\theta\rvert$ | **fails** |
| (T1 by the true step, unscored) | the same rule at the first level where the true step is also below π/2 in both trees: true in 70 of 70; $\sigma_t$ $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon\lvert\theta\rvert$ (≤ 0.013 of the bound); $\sigma_c$ 0.198–0.374 | (holds) |
| **T1 (§9), scored by the true step** *(added after the user's decision, deviation 16; this is now T1's scored verdict)* | windows resolved by the true step alone, $\lvert d\theta_T/dz\rvert(b-a)/64 < \pi/2$ under both $c$ and $t$, halving further where needed; every other clause of §9 unchanged. 91 windows, 70 scored, 0 unresolved, true in 70. $\sigma_c$ 0.198–0.374 $\varepsilon\lvert\theta_T\rvert$; $\sigma_t$ $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$, largest $\sigma_t$/bound 0.0131. The largest unwrapped step at the levels used is 1.44 rad, so these windows also pass the literal rule. Levels used: $n$ = 0 (23), 2 (12), 3 (7), 5 (8), 6 (7), 7 (2), 8 (11). These are the same levels, and therefore the same numbers, as the unscored row above. No re-probe was needed (`out/cmd/26_T1_true_step_scored.txt`) | **holds** |
| (T1 by the literal step, now unscored) | the "as written" row above, kept beside the scored verdict | (fails) |
| **T2** | 39 of 39; no group under $t$@1e-7 has a region at depth ≥ 18; largest depth 11 (and 11 on the 11 rows capped under B1) | holds |
| **T3** | 39 of 39; largest $\lvert t-c\rvert$/`total_abserr`$_c$ $4.66\times10^{-6}$ (`low-G-1`); largest relative $3.6\times10^{-12}$ (R4); bit-identical totals on 13 rows | holds |
| **T4** | 29 of 31, none excluded; false on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), neither capped under B1; true on all 11 capped (largest 0.166). Every ratio equals 05's L3 under $c$ to the digits printed | **fails** |

**T1's unscored items:**
- $\varepsilon(k\max|p_z| + 2\pi)$ is the largest of the three terms in **0 of 70** windows, so
  there are no $\sigma_t$ ratios to report for it. `1e-3 ε|θ|` is largest in 69 windows and
  `1e-13` in 1. $k\max|p_z| \le 9.8\times10^{-7}|\theta_T|$, with one table node per window
  (issues X5).
- **Sampled in $u$ under $t$, at the true-step level:**
  - 14 windows exceed $0.1\,\varepsilon|\theta_T|$, at 0.10–0.22, where the abscissa term is
    0.69–0.98 $\varepsilon|\theta_T|$;
  - over all 70 windows, $\sigma_u$ is 0.14–0.29 of the abscissa term;
  - under $c$ it is 0.21–0.40 $\varepsilon|\theta_T|$;
  - at $z < 1$ (6 windows) it is 0.0025–0.0096 under $t$.
  - At the literal level, 27 windows exceed 0.1, the aliased ones among them.
  - The issue the rule asks for is proposed under observation 2.

**Descriptive:**
- **$|t-c|$:** relative at most $3.6\times10^{-12}$, and at most $4.7\times10^{-6}$ of
  `total_abserr`$_c$; 13 rows are bit-identical (T3's table).
- **Groups:** all 219 group calls are the same sequence (regime, label, `x_span` bit for bit), with
  the same region count, maximum depth and convergence flag under $c$ and $t$.
  - That includes the 140 `Gqr` groups: 264 regions, max depth 11, in both.
  - The lowest-10 % (in $\log(1+z)$) region counts and depths of every group are the same.
  - Per group in `out/wider_05a.txt`.
- **Cost:** low band 288 → 288 regions (8.0 → 4.6 s, load-dominated); mid band 321 → 321;
  R1–R7 250 → 250; the 11 capped rows 426 → 426 at 1e-7 and 690 at 1e-8.
- **`total_converged`:** unchanged between $c$ and $t$ on every row.
  - Depth-20 rows: 21 of 31 unconverged at 1e-7, 19 at 1e-8.
  - Controls: all converged.
  - 3 rows have an infinite `total_abserr` in both trees: `low-G+r-2`, `low-G-r-1`, `low-Gqr-3`.
- **Kinds of non-convergence** under $t$ (`out/convergence_kinds_t.txt`, script SHA-256
  `49c83a72…`):
  - 1e-7: resolution 18, bound only 32, phase-limited 17 (capped rows: 12, 21, 10);
  - 1e-8: 21, 24, 27 (capped rows: 17, 13, 18).
  - These are identical to 05's $c$ tallies.

### What shipped

Nothing is committed. Everything is under `var/levin-phase-floor-05a/wider/`. The checkpoints and
the pickles, which hold every probe sample of every level, are in the run directory (21 MB).

**Scripts** (`scripts/`). 05's harness, copied; the diff against 05's copy is
`out/scripts_diff_against_05.txt` (at the launch) and `…_final.txt` (with the analysis).
- **Changed:**
  - `prov.py`: the trees and paths;
  - `worker.py`: trees, provenance, the `build_partition` capture, the probe call;
  - `run_wider.py` and `launch.py`: trees, unit sets, prompt `05a`;
  - `rec.sh`: also writes `out/commands.log`;
  - `analyse.py`: rewritten for T1–T4.
- **New:** `probe.py` (the phase probe), `t0.py` (gate T0, from 05's `l0.py`) and `rescore_t1.py`
  (re-scores T1 from `state/probe_windows.json` alone).
- **Added after the user's ruling:** `t1_true_step_only.py`, which scores T1 by the true step
  alone from `state/probe_windows.json` (`out/cmd/26_T1_true_step_scored.txt`). Beside it, in
  `state/`, is `predictions_note_T1_reading.md`. The scripts diff including it is
  `out/scripts_diff_against_05_after_ruling.txt`.
- **Unchanged:** `harness.py`, `rebuild.py`, `dispatch.py` and `hash_store.py`. `convergence_kinds.py`
  is byte-identical to `prompts/levin-phase-floor/measure/`. `draw_sample.py` and `l0.py` were
  copied and are unused.

**Outputs** (`out/`):
- `commands.log` and `cmd/00…25`: every command with its output (01 is unused);
- `T0.txt`;
- `wider_05a.txt`: every row, every group, every probe window under both readings, and the scores;
- `convergence_kinds_t.txt`;
- `fingerprint_05a_wider.txt`;
- `status_final.json`;
- the two scripts diffs.

**State** (`state/`):
- `sample.json` (05's, unchanged) and `sample.md`;
- `predictions.md`, the three `prompt_*.raw` and `predictions_attempt1_wrong_window_lines.md`;
- `T0/` and `smoke/`;
- `store_files_{before,prelaunch,after,after_fingerprint}.txt`;
- `launch.txt` and `launch_check.txt`;
- `analysis.json`;
- `probe_windows.json` (4 MB): every window's ladders under both trees, and the samples as
  `float.hex` at both readings' levels.

### Deviations from the prompt

1. **The $c$ tree runs from a worktree.** **STRUCTURALLY REQUIRED.** The prompt runs two trees,
   and the orchestrator made the worktree.
   - Code is loaded through `PYTHONPATH` with `-B` and `PYTHONDONTWRITEBYTECODE=1`.
   - Nothing was written there: `git status --short --ignored` is empty at the end, although R11
     shared the worktree.
   - The manifest's `git_head` is `149c393` (the $t$ tree). Every checkpoint payload records its
     own tree's `HEAD`, its status and its modules' files.
2. **05's trees `p`/`c` became `c`/`t`, with 05a's paths and unit sets.** **STRUCTURALLY REQUIRED**,
   as §6 allows. The changes are in `prov.py`, `run_wider.py` (budget from $c$), `launch.py`
   (prompt `05a`), and the docstrings of `worker.py` and `rec.sh`. Each change is marked "prompt
   05a"; 05's docstrings are kept below a note.
3. **The scripts live in `var/levin-phase-floor-05a/wider/scripts/`,** not `…/05a/scripts/`.
   **IMPLEMENTATION CHOICE** (the orchestrator's), mirroring 05's `wider/` layout.
4. **How the probe is built.** **IMPLEMENTATION CHOICE**, where §6 and §9 leave the method open.
   - It is a new `probe.py`.
   - `worker.py` wraps `ComputeTargets.QuadSourceIntegral.build_partition` at its module
     attribute, to keep the unit's own $T$ objects.
   - The probe runs in each tree's 1e-7 unit after `wall_s` is taken, outside the budget, with the
     dispatcher uninstalled. T0 (b) shows the unit's results are unchanged by it.
   - Each tree computes **every** halving level until its 65 $z$ samples stop being distinct. The
     joint level is chosen afterwards, in the analysis, as the first level at which both trees
     pass.
   - The window is $[z_{\min}, z_{\min}(1+w)]$. The step that drives the halving is the
     $z$-sampled one.
   - $\max|p_z|$ is recorded under both trees; the $t$ value is the one scored.
   - $\sigma_u$ is None where the 65 $u$ samples are not distinct, which happened at no level used.
5. **The halving rule is read literally,** with the true-step reading reported apart and unscored.
   **IMPLEMENTATION CHOICE**, fixed in `state/predictions.md` before the run after T0's $c$ windows
   showed the two differ.
   - The point was raised in the launch report and relayed to the user; it is unanswered.
   - All levels are kept, so either reading can be scored (issues X1).
6. **Gate T0 uses a new script, `t0.py`, derived from 05's `l0.py`.** **STRUCTURALLY REQUIRED**:
   T0's clauses are not L0's.
   - T0 (b) compares more than the clause names (the whole result and the region arrays).
   - T0 (c) includes the probe.
   - Both rows are 05's L0 rows, `low-G+r-1` and `ctrl-low-1`.
7. **The worker records more than 05's did:** `TkSourceFunctions`' file and its keyword's default,
   and each factor's `phase.exact_remainder`. **IMPLEMENTATION CHOICE.** The dispatch asked for
   both.
8. **`rec.sh` also appends every record to `out/commands.log`.** **IMPLEMENTATION CHOICE.**
9. **`state/predictions.md` was written twice before the launch.** **UNINTENDED DRIFT.**
   - The first write (12:29:30Z) quoted §9's window from line 391 instead of 388. It is kept as
     `state/predictions_attempt1_wrong_window_lines.md`; a working copy passed through the
     session scratchpad on the way.
   - The second write (12:30:03Z) fixed the range and one internal reference.
   - Both writes were before the launch, and the quoted predictions are identical in both.
10. **The smoke test** (`low-G+r-1`, `ctrl-NULL-1`, all three sets) ran after the predictions were
    written and before the launch, as 05's did. **IMPLEMENTATION CHOICE.** It needed no script
    change, so T0 was not re-run.
11. **`analyse.py` was rewritten, after the run.** **STRUCTURALLY REQUIRED**: 05's L1–L3 are not
    05a's rules. The "low-$z$ end of each sub-interval" (descriptive) is read as the regions
    starting in the lowest 10 % of the group's sub-interval in $\log(1+z)$, which is an
    **IMPLEMENTATION CHOICE**.
12. **`rescore_t1.py`, an independent re-score of T1 from the saved windows alone.**
    **IMPLEMENTATION CHOICE.** It agrees with `analyse.py` on every window and both readings.
    `analyse.py` also asserts that $\sigma$ recomputed from the saved samples equals the probe's.
13. **`convergence_kinds.py` was run from the main checkout with `-I`, from this directory's
    unchanged copy.** **IMPLEMENTATION CHOICE.**
    - Its header says "capped: tree p … in <05's run>", which is 05's $p$ (B1's code), as §6
      intends.
    - `out/convergence_kinds_t.txt` drops the per-pickle hash lines; the full output is in
      `out/cmd/22`.
14. **The work ran beside R11/R12 and other load,** with a load average of 10–35 during the setup.
    **IMPLEMENTATION CHOICE**, which the prompt allows ("It may run beside §5's builds"). Wall times
    and budgets carry that load; the closest any unit came to its budget is 0.861.
15. **05's `draw_sample.py` and `l0.py` were copied unchanged and are unused.** **IMPLEMENTATION
    CHOICE**, so that the copy is 05's harness whole.

16. **T1's scored verdict is taken by the true step, by the user's decision after the run.**
    **STRUCTURALLY REQUIRED** (the user's ruling on the rule's reading; added after the run).
    - **The ruling.** Relayed to this session at about 13:23Z from the laptop session, in answer
      to "Which reading of §9's resolution rule should be T1's scored verdict?", the user chose
      "True step (Recommended)". That option read: *"T1 is scored on windows whose true step
      |dθ/dz|(b−a)/64 is below π/2, narrowing further where it isn't. This is what §9 meant by
      'resolved'. The literal unwrapped-step reading is reported beside it, unscored. Recorded as
      an amendment of §9's wording, made before any T1 number exists."*
    - **What the files show of the timing.** `predictions.md` fixed both readings at 12:30:03Z,
      the literal one as scored. Both were first scored at 12:35:49Z (`out/wider_05a.txt`). The
      ruling reached this session at about 13:23Z. When the user made it is for the laptop
      session to record. This is set out in `state/predictions_note_T1_reading.md`.
    - **How it was applied.** `predictions.md` and its first copy are left as written.
      `scripts/t1_true_step_only.py` is new; it reads `state/probe_windows.json` only.
      - It takes the first level at which the true step alone is below $\pi/2$ in both trees.
      - That level equals `analyse.py`'s true-step level in all 91 windows, so the saved samples
        are the windows' own.
      - It re-computes $\sigma$ from those samples and scores §9's rule.
      - No window needed a re-probe, and none was run.
      - The literal reading stays reported, unscored.

Deviation 9 is the only `UNINTENDED DRIFT`.

### Verification performed

- **The trees.**
  - The main checkout was at `149c393` with an empty status, and the worktree at `c4eaefd` with an
    empty status. This holds at the start, before the launch, on every checkpoint record and at the
    end (`out/cmd/00`, `15`, `25`).
  - The worktree's `git status --short --ignored` is empty at the start and at the end.
- **Module provenance.** Each worker checked and recorded the files of `ComputeTargets`,
  `primitive_phase`, `QuadSource`, `QuadSourceIntegral`, `GkSourcePolicyData`,
  `TkSourceFunctions`, `StoreFixtures`, `main.py` and the original `theta_mod_2pi`. All of them come
  from its own tree. `TkSourceFunctions`' keyword is absent under $c$ and defaults to True under
  $t$.
- **Which phase each tree used.**
  - $c$: $G$ 98,194 `theta_mod_2pi` calls with `exact_remainder=True`, and $T$ 59,906 with `False`.
  - $t$: $G$ 268,136 with True, and $T$ 167,562 with True.
  - Every unit's $T$ factors report `exact_remainder` False under $c$ and True under $t$, and so do
    all 91 probe windows.
  - The $t$ counts equal 05's $c$ counts.
- **The sample.** Its SHA-256 is `73aa7876…`, and it is `cmp`-identical to 05's. It equals the
  `sample_sha256` in 05's run header, checked before any row ran.
- **The store.**
  - All six files' SHA-256, size and mtime are identical before, before the launch, after the run
    and after the fingerprint. No journal, `-wal` or `-shm` file appeared.
  - `store fingerprint`, without `--write`, gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`,
    `problems: none`, and "matches the fingerprint taken … at b90752d".
- **The run.**
  - `done`, 109/109, in the planned order, exit 0, no problems, none over budget, no errors and no
    probe errors.
  - Every record carries `run_wider.py`'s SHA-256 `af656ad4…`.
  - Ray was never initialised, and no lock was taken.
- **`QuadSource`.** It is recomputed in every unit, equals the stored row at its nodes, and adds
  B1's 3 nodes.
- **The probe windows.** $c$ and $t$ give the same 91 windows, starting at the same double. Their
  `raw_theta` and `theta_deriv` at that point are bit-identical, as §0.4 requires.
- **05's records were only read:** 05's harness is `diff`-identical in its `var/` and `docs/`
  copies at the end.

### Observations not acted on

All are in `issues.md`:
1. **X1:** §9's halving rule, as written, admits aliased windows. T1 fails in exactly those 36, on
   the $t$ clause, and holds in all 70 by the true step. Proposed tag:
   `[05a-t1-halving-rule-admits-aliased-windows]`. The user's ruling is pending.
   *Added after the ruling:* the user chose the true step (deviation 16). T1 is scored by it and
   holds, 70 of 70. The literal reading is reported beside it, unscored. The laptop amends §9's
   wording in the records commit. What remains of X1 is the record that the wording as first
   written admitted aliased windows; whether that is still opened as an issue is the laptop's to
   decide.
2. **X2:** the abscissa term remains in $T$'s $u$-sampled phase.
   - $\sigma_u$ under $t$ is 0.10–0.22 $\varepsilon|\theta_T|$ in 14 windows at $z \gg 1$, against
     abscissa terms of 0.69–0.98; it is 0.14–0.29 of the term everywhere.
   - Proposed issue, as T1's $u$-clause asks: `[05a-t-phase-abscissa-rounding-remains]`.
3. **X3:** T4 fails on the same two rows as 05's L3, with the same ratios (05's W5, untouched by
   $T$).
4. **X4:** $T$'s two errors change no region, depth or convergence flag on the 39 rows, and move
   the totals by at most $4.7\times10^{-6}$ of the declared error.
5. **X5:** §9's $p_z$ term never binds. The windows sit at or near table nodes, with
   $k\max|p_z| \le 10^{-6}|\theta_T|$.
6. **X6:** 15 scipy `IntegrationWarning`s, as in 05.
7. **X7:** the kinds of non-convergence under $t$ are identical to 05's under $c$.

### State handed on

- **Nothing is running.** The run is `done`, no lock was taken, and nothing was written in either
  tree.
- **The store is unchanged**, fingerprint `0e32f24a…`.
- **The worktree** `SecondaryGWKit-lpf05a-c4eaefd` is untouched by this work. It is the
  orchestrator's to remove once R11's records are taken.
- **For the orchestrator:**
  - copy `out/`, `scripts/`, `state/`, `log.md`, `verification.md` and `issues.md` to
    `docs/levin-phase-floor/05a/wider/`. `state/T0/*/*.pkl` and `state/smoke/*.pkl` may be left
    out; `state/probe_windows.json` (4 MB) is what T1's re-score needs;
  - append this section to `logs/05a-reanchor-t-phase.md`;
  - put `verification.md` in `VERIFICATION.md` §5a. Its links assume the `05a/wider/` location.
- **For the user:** X1, the reading of §9's halving rule. Under the literal reading, T1 fails.
  Under the true-step reading, it holds in every window. Both are scored from the same saved
  samples.
  - *Added after the ruling:* answered: the true step (deviation 16). T1's scored verdict holds.
  - The laptop amends §9's wording in the records commit, and records when the user decided.
  - `state/predictions_note_T1_reading.md` records what the files here show.
