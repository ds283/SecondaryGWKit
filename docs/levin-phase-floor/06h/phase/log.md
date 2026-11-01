# levin-phase-floor 06h — §5, the phase re-measurement (H1 and H2): the phase agent's record

**Subject:** prompt 06h §5, "The phase re-measurement": 06f's M2 items 1 and 4, repeated at 06h's code on 06f's 20 M2 calls, with H1 and H2 scored from it. On the Mac Studio (mps026298), 2026-11-01.
- It ran Ray-free, with no lock, no store built and no code changed.
- **Trees:**
  - $h$ = worktree `SecondaryGWKit-lpf06h-c1ec3b2` at `c1ec3b2` (code `7172a8f` + `ac1c6f4`);
  - $p$ = worktree `SecondaryGWKit-lpf06h-2c325f5` at `2c325f5` (06e's code).
  - Both were clean with `--ignored` before and after. The main checkout stayed at `28ebaf0`, clean.
- **Commit:** none. The orchestrator commits the records on the laptop.

**Result: H1 holds and H2 holds.**
- **H1:** the worst $|\delta\theta|/\texttt{theta\_abserr}_h$ is **0.493**, over 21,177 points (1,177 nodes and 20,000 dense points). The parent's declaration fails at 561 of the same points, worst **376**.
- **H2:** the worst response over declared term is **0.268**, over 27 leaves. The parent fails on 2 leaves, worst **41.6**.
- **Checks:**
  - all 40 captures (20 at $p$, 20 at $h$) reproduce their own census bit for bit;
  - the method check passes at the exact $e^u$;
  - the reference agrees with itself at 50 digits and by the Hankel formula;
  - the parent's replay reproduces 06f's M2 item 1 bit for bit (49 of 1,177 nodes, 148×).

*The predictions below were copied in at 2026-11-01T03:34:43Z, before any number was computed.*
- *The record is `state/predictions_20261101T033443Z.md`, read-only, SHA-256 `e3764cb6…`.*
- *The draft of this log that held them (03:34:48Z) is kept as `state/log_draft_prelaunch.md`.*
- *The first measured number comes from the capture, which started at 03:35:14Z (`out/cmd/30`).*

## Predictions (copied verbatim from prompt §5 before the run)

- **H1 — the declaration is honest.** **Rule:** on every one of the 20 M2 calls, at every point
  sampled, $|\delta\theta(u)| \le \texttt{theta\_abserr}_h(u)$, the reference at the exact $e^u$.
  **Holds** if no point violates it. The parent's declaration is scored on the same points beside it.
- **H2 — the declared term covers the response.** **Rule:** on every returned leaf of the 20 calls,
  the first-order response to the measured phase error is at most that leaf's declared term.
  **Holds** if no leaf violates it.

## What shipped

Nothing shipped in code. The records are in `var/levin-phase-floor-06h/phase/`:
- **Write-ups:**
  - `log.md` (this file);
  - `verification.md`, my part of `VERIFICATION.md` §6h;
  - `issues.md`;
  - `KEPT_ON_MAC_STUDIO.txt`: 80 capture pickles, each with SHA-256 and size.
- **Results:**
  - `out/scores.json` (the verdicts), `out/phase_calls.csv` and `out/phase_leaves.csv`;
  - `out/cmd/43_analyse.txt`, which prints every number;
  - `out/cmd/44_details.txt`: the parent's per-subset counts, the three worst nodes, `K_lo`, and #19's leaves.
- **Per call and tree,** in `out/replay_{h,p}/<unit>_<index>/`:
  - `result.json` and `leaves.json.gz`;
  - `nodes.json.gz` and `dense.json.gz`. Each point holds:
    - $\hat\theta$;
    - $\delta\theta$ at the exact $e^u$ and at $\hat x$;
    - `theta_abserr`;
    - the 40-digit exact reference;
    - at $h$, the declaration's three parts.
- **Cross-scoring:** `out/cross/{p_at_h,h_at_p}/<unit>_<index>/`, each with `cross.json` and `rows.json.gz`.
- **Captures:** `out/capture_{p,h}.json`, with the pickles in `state/capture_{p,h}/`.
- **Selections:** `out/sample/selection_{p,h}.json`.
- **Commands:** every command and its output is in `out/cmd/NN_*.txt` and in `out/commands.log`.
- **Scripts:** `scripts/` holds 06f's scripts and its review's `orch_m2.py`.
  - 06f's reference copies are kept unchanged in `state/scripts_06f_reference/`.
  - The diff is `out/scripts_diff_against_06f_at_launch.txt`. It is unchanged since stage 1 (`out/cmd/01`, `21`).
  - The hashes at launch are in `out/scripts_sha256_at_launch.txt`.
  - Changed: `prov.py`, `x1_capture.py`, `replay06f.py`, `method_check_06f.py` and `orch_m2.py`.
  - New: `select_06h.py`, `cross06h.py`, `run_06h.py`, `launch_06h.py` and `analyse06h.py`.
- **The registered run:** `var/runs/levin-phase-floor-06h-phase-20261101T033514`.
  - Done, 84 of 84 units, exit 0, from 03:35:14Z to 03:50:16Z.
  - Driver pid 18673; `sh` pid 18666.
  - The launch transcript is `state/launch_phase.txt`.

## Predictions and verdicts

| | Number scored | Verdict |
|---|---|---|
| **H1** | **21,177 points** on the 20 calls:<br>– 1,177 nodes: every Clenshaw–Curtis fine node and every Levin collocation node of every returned leaf at $h$;<br>– 20,000 dense points: 06f's 1,000 per call, seed 20261031.<br>**0 violate.**<br>**Worst** $\lvert\delta\theta\rvert/\texttt{theta\_abserr}_h$ = **0.4933**:<br>– where: `g-1e-7-0-12689` #19 (J4), CC leaf 1, node 11, $u = -7.596923417764045$;<br>– $\delta\theta = +4.503\times10^{-13}$ against $9.128\times10^{-13}$;<br>– the constituents' term dominates there ($9.009\times10^{-13}$, from the two ν = 5/2 factors at $y = 90.96$, just above their $x_\star = 89.58$). The abscissa term is $5.46\times10^{-15}$ and $\phi_{\rm eval}$ is $6.46\times10^{-15}$.<br>**The parent on the same points:** 561 of 21,177 violate (53 nodes, 508 dense), worst **376.5** (#19, dense point 996). Its violations fall on two calls:<br>– #19: 534, worst 376.5;<br>– `g-1e-7-3-1054` #20: 27, worst 3.14. | **holds** |
| **H2** | **27 returned leaves** on the 20 calls: 23 CC, 4 Levin.<br>**0 violate.**<br>**Worst** $\lvert\delta I_{\rm measured}\rvert$ over the declared term = **0.268**, on #19's Levin leaf 3 (depth 2): $6.15\times10^{-25}$ against $2.30\times10^{-24}$.<br>**Median** $4.5\times10^{-5}$.<br>**The parent,** on its own leaves with its own declared terms: 2 of 27 violate, worst **41.6** on the same leaf ($7.88\times10^{-25}$ against $1.90\times10^{-26}$); median $8.7\times10^{-5}$. | **holds** |

**Neither rule turned out badly posed.**

**H1's reading.** Read literally, "every point sampled" means the nodes and the dense sample together, which is how H1 is scored. Read narrowly as "every node", as 06f's F4 read it, H1 covers 1,177 points: 0 violate, worst 0.4933. Both readings give the same verdict.

**The parent's column, at the exact $e^u$:**
- Its own nodes violate at 53 points, worst 277.
- At 06f's $\hat x$ the same nodes give 06f's 49 violations, worst 148, reproduced bit for bit.
- 06f's review, Observation 1, gave 246 for the node 06f found worst at $\hat x$. This replay gives 245.76 at that node (#19, leaf 3, node 4).

**The returned partitions are the same in both trees** on every call (spans and branches). So the parent "on the same points" (`cross06h.py`) and the parent on its own points coincide: 561 violations either way.

**Descriptive: measured over declared at $h$'s points, split by which part of `_declared_error_parts` dominates** (test 4's split).

| dominant part | points | median | 90 % | max | min |
|---|---|---|---|---|---|
| constituents | 20,446 | $6.8\times10^{-4}$ | $6.3\times10^{-3}$ | 0.493 | $5.6\times10^{-8}$ |
| abscissa | 731 | 0.134 | 0.275 | 0.459 | $2.6\times10^{-4}$ |
| $\phi_{\rm eval}$ | 0 | — | — | — | — |

- **Where the abscissa term dominates, the declaration is tight.** The median is 0.13, so the declaration is within a factor of about 7 of the error. The worst such point is 0.459, at #19, Levin leaf 3, node 22.
- **$\varepsilon_{\exp}$ as used** is the code's `EXP_RELERR` = $2.23\times10^{-16}$ (`three_bessel_integrals.py:134`). It was measured on the laptop. The orchestrator's test 5 on this host gave the same worst, $1.114619\times10^{-16}$. I did not re-measure it.
- **Under 06f's $\hat x$ reference, on the same points:**
  - $h$: 0 violations, worst 0.492;
  - the parent: 49 node and 473 dense violations, as 06f found.

## Deviations from the prompt

1. **`prov.py`: the trees and their paths.** **STRUCTURALLY REQUIRED.**
   - Two trees, `h` and `p`.
   - The depth assertion is `parents[3]`.
   - `census_run()` gives 06e's census at `p` and 06h's at `h`.
2. **`x1_capture.py`: `--tree h|p`.** **STRUCTURALLY REQUIRED.**
   - Units are named `<tree>-<rtol>-<shard>-<serial>`.
   - PYTHONPATH must be the tree named.
3. **`select_06h.py` (new): each tree's captures are checked against that tree's own census.** **STRUCTURALLY REQUIRED.**
   - $h$ is checked against 06h's checkpoint, `3104c18e…`.
   - $p$ is checked against 06e's checkpoint, `a0dc4d7f…`, whose records equal 06f's selection.
   - 06f's selection equals its committed copy.
4. **`replay06f.py`: the tree and the census lookup.** **STRUCTURALLY REQUIRED.**
   - It detects which tree it runs in.
   - It finds the census record whatever prefix the census gives its units (06h's census names them `g-…`).
   - If more than one record holds a call, they must agree.
5. **The reference is taken at the exact $e^u$,** in item 1 and in item 4: `mp.exp` of the double $u$, then each constituent at $m_i e^u$. **STRUCTURALLY REQUIRED** (prompt §2 and the brief).
   - 06f's $\hat x$ reference is kept beside it, computed exactly as 06f computed it. That is how the parent was shown to reproduce 06f. **IMPLEMENTATION CHOICE.**
6. **What is kept per point:** the 40-digit reduced exact reference, the declaration's three parts, and every dense point, not only a summary. The cross-scoring and the split need them. **IMPLEMENTATION CHOICE.**
7. **`K_lo` is recorded** in each call's group record. **IMPLEMENTATION CHOICE.**
8. **`method_check_06f.py`.**
   - The group check is made at the exact $e^u$, at 30 and 50 digits. **STRUCTURALLY REQUIRED.**
   - A check is added that atan2 and arg $H^{(1)}$ + π/2 agree to $10^{-25}$, for each constituent and for the group (prompt §2 test 7). **IMPLEMENTATION CHOICE.**
9. **The method check ran after the captures** and before any replay or scoring (03:43:35Z). **STRUCTURALLY REQUIRED.**
   - The brief's sequence lists it before "capture, replay and measure".
   - It needs a captured group, so it cannot come before the captures.
   - Under either reading it ran before any $\delta\theta$ was computed.
10. **`orch_m2.py`.** **IMPLEMENTATION CHOICE.**
    - Its paths are 06h's.
    - Its points are the nodes and the dense sample together.
    - A `worst` mode runs the 50-digit check at $h$'s worst point.
    - Its method is unchanged: the Hankel formula, the exact $e^u$, 30 and 50 digits.
11. **`cross06h.py` (new)** implements "the parent's declaration is scored on the same points". **IMPLEMENTATION CHOICE.**
    - It evaluates the parent's $\hat\theta$ and `theta_abserr` at $h$'s points, against $h$'s stored exact reference.
    - The reverse direction ($h$ at $p$'s points) is descriptive.
    - It checks that the two trees' spans, groups, dense points and references are equal. They are, on all 20 calls.
12. **H1 is scored on the nodes and the dense sample together,** and each subset is also reported. The two readings give the same verdict. **IMPLEMENTATION CHOICE.**
13. **The copied replay still computes 06f's M1 and item 3** (the response to a phase error of the declared size). **IMPLEMENTATION CHOICE.**
    - Both are left unchanged, to keep the diff minimal.
    - They are not scored. Item 3's $D/\max_s|\delta I|$ is in `result.json`.
14. **`run_06h.py`, `launch_06h.py` and `analyse06h.py` are new,** and the registered slug is `phase`. **IMPLEMENTATION CHOICE.**
15. **Formatting.** **IMPLEMENTATION CHOICE.**
    - `black` was run on the new scripts, and on `replay06f.py` and `method_check_06f.py`, which were black-clean in 06f.
    - `prov.py`, `x1_capture.py` and `orch_m2.py` are left as copied.
16. **The parent's reproduction of 06f's nodes** is a check the prompt did not ask for. **IMPLEMENTATION CHOICE.**
17. **`harness.py`'s inert `job_name`** is left as `levin-phase-floor-06e`. **IMPLEMENTATION CHOICE.**
18. **All of 06f's scripts were copied,** as 06f did, though only about a dozen are used. **IMPLEMENTATION CHOICE.**
19. **The scratch copy was hashed before and after, not fingerprinted.** The brief asks for the hashes. The fingerprint is held by the census agent and the orchestrator. **IMPLEMENTATION CHOICE.**
20. **A lock command was piped.** **UNINTENDED DRIFT.**
    - In `out/cmd/10_state_before`, `raypoolkit.rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk` ran inside a shell whose output was piped to `tail`.
    - CLAUDE.md asks for each lock command to be a command of its own, with no pipe. As CLAUDE.md warns, the listing then named my own shell's processes (`18371 18368 tail -30`), not Ray processes.
    - It changed nothing. The post-run checks were made correctly, each its own command with `>>` (`out/cmd/45`, `46`).
21. **A stray redirection while writing this log.** **UNINTENDED DRIFT.**
    - One shell command tried to redirect into `/tmp/../`, which is a directory.
    - The shell refused, and nothing was written.
    - It is recorded against the no-`/tmp` rule.
22. **The run's last stage string reads "refcheck running" although the run is `done`.** **UNINTENDED DRIFT**, cosmetic.
    - `run_06h.py` records the `refcheck` unit but sets no heartbeat after it before `finish`.
    - The checkpoint's last record is `refcheck`: `done`, exit 0. `out/cmd/33` is complete.

## Verification performed

- **Before the run** (`out/cmd/10`–`13`, `22`):
  - both worktrees were clean with `--ignored`, at their SHAs;
  - the main checkout was clean at `28ebaf0`;
  - the lock was free;
  - the registry held 39 runs, all finished;
  - **the scratch copy was EQUAL to its certification**, with no journal, `-wal`, `-shm` or manifest file;
  - 06h's census, `levin-phase-floor-06h-census-20261101T005037`, was `done`: 18560 of 18560 units, exit 0, checkpoint `3104c18e…`.
- **The captures** (`out/cmd/30`, `31`):
  - **20 of 20 at $p$ match 06e's census, and 20 of 20 at $h$ match 06h's census, bit for bit**: value, abserr, regions, maximum depth, converged and label;
  - every worker reports `ray_initialized` false;
  - at $h$ every call has the same regions, depth and convergence as at $p$.
- **The replays:** all 40 hold F0 against their own census, with no errors. F0 means:
  - the replay equals the census record;
  - the terms equal `cc_floor`;
  - every leaf is reproduced;
  - the leaves sum to the value and the abserr.
- **The method check passes** (`out/cmd/32`, on $h$'s capture of #19).
  - The ν = 1/2 phase-pair error is $9.09\times10^{-18}$ at its argument and $5.94\times10^{-17}$ at $x = 10^3$, against the documented $1.11\times10^{-16}$.
  - 30 against 50 digits differ by at most $2.2\times10^{-31}$ for any constituent.
  - atan2 against arg $H^{(1)}$ + π/2 differ by at most $9.9\times10^{-32}$.
  - **The group at the lower endpoint, at the exact $e^u$:**
    - $\delta\theta = +8.618620\times10^{-16}$ at both 30 and 50 digits;
    - the references differ by $7.5\times10^{-32}$, and the two formulas by 0;
    - at $\hat x$, $\delta\theta = +8.93\times10^{-16}$;
    - `theta_abserr` is $1.0102\times10^{-12}$: constituents $1.0009\times10^{-12}$, abscissa $1.08\times10^{-15}$, $\phi_{\rm eval}$ $8.19\times10^{-15}$.
- **The 50-digit check** (`out/cmd/33`: `orch_m2.py worst h 5`, the review's Hankel method).
  - **$h$'s worst point:** #19, node leaf 1, k 11, $u$ = `-0x1.e633fe4760d6fp+2`.
    - $\hat\theta$ and `theta_abserr` were recomputed, equal to the stored values.
    - $\delta\theta$ by the Hankel formula at the exact $e^u$ is $4.502680015621786\times10^{-13}$ at both 30 and 50 digits. Its relative difference from the stored atan2 value is **0**.
  - **Four more points** (the next three worst and one drawn): the relative difference is at most $5.5\times10^{-15}$ at 50 digits, and 0 at 30.
- **The cross-scoring:** on all 20 calls the two trees have equal spans, coefficients, signs, orders, $K$, dense points and dense references (40-digit strings).
- **After the run** (`out/cmd/40`–`42`, `45`, `46`):
  - **the scratch copy is EQUAL to its certification**;
  - 06h's checkpoint is unchanged at `3104c18e…`, and 06e's at `a0dc4d7f…`;
  - nothing under `var/datastores/` is newer than the stage-2 marker;
  - outside my directory and my run, the only newer files are the orchestrator's `review/checks/` and `review/scripts/`, and the `var/runs` directory entry;
  - the worktrees are clean with `--ignored`, the main checkout is clean, and there is no `__pycache__`;
  - the registry holds 40 runs, all finished;
  - **the lock is free and `/tmp/ray-sgk` has no Ray processes**, each checked with its own command.

## Observations not acted on

1. **`K_lo` is not zero on 9 of the 20 production calls.**
   - At most $|K_lo| = 4.66\times10^{-9}$, on `g-1e-7-2-33311` #12 with $K = 4.79\times10^8$: one rounding of $K$.
   - It is zero on 06f's two violating calls, as the prompt says.
   - 06h's `sin_cos` carries it. It is noted only because the prompt's (b) says "zero on 06f's two calls; not zero in general", and this measures how general.
2. **How the calls' declared errors move.**
   - `abserr` rises by 0.3–1.9 % on 17 calls, by 5.4 % on `g-1e-7-2-33311` #12, by 14 % on `g-1e-7-1-10824` #12, and by 51 % on `g-1e-8-0-10589` #1.
   - On `g-1e-8-0-10589` #1 the declared term $D$ doubles, from $1.90$ to $3.81\times10^{-24}$.
   - All 20 calls stay converged, with the same partition.
   - On #19's tail Levin leaves the declared term rises 13× and 121×, while the call's $D$ moves under 1 %, as §0.2 predicts.
3. **These calls do not reach the near-singular class** of the LiouvilleGreen draws. All 20 are A3 v2 analytic blocks.
4. **The driver's final stage string** (deviation 22) is cosmetic.

## State handed on

- **The run** `levin-phase-floor-06h-phase-20261101T033514` is done, 84 of 84 units, exit 0. Nothing of mine is running.
- **The lock** was never taken, and is free. No Ray was started.
- **The scratch copy** is unchanged.
- **The worktrees** were never written.
- **Kept on the Mac Studio:** `KEPT_ON_MAC_STUDIO.txt` lists 80 capture pickles under `state/capture_{p,h}/`. No other file is over 2 MB. The directory totals 20 MB.
