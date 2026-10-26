## §6 — the wider re-measurement (Mac Studio)

**Subject:** prompt 05 §6, run on the Mac Studio (mps026298) by the re-measurement agent. It was
Ray-free and took no lock. It ran alongside the R9/R10 agent, which held the lock for its own
builds.
- **Trees:**
  - $p$ is the code commit's parent `e648951`, in the git worktree
    `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951` (detached);
  - $c$ is the code commit `120fdfe`, in the main checkout at the dispatch SHA `53084dc`, which adds
    only records.
- **Run:** `levin-phase-floor-05-wider-20261026T020005`. It ended `done`, 109/109 units, exit 0,
  02:00:05Z–02:11:54Z.

**Commit:** none here. The records are committed by the orchestrator in 05's records commit.

**Result:**
- **Gate L0 holds**, on both of its clauses and both rows.
- **L1 fails** (9 of 24 rows) and **L1q fails** (1 of 7), as written. Every failure is the tenfold
  region clause. 20 of the 21 failing rows are not capped under $p$ at all; the 21st is R5, capped
  by one 55-region chain.
- **L2 holds** (8 of 8, largest ratio 0.652).
- **L3 fails** (29 of 31). It fails on two uncapped mid rows, at 1.29 and 1.70.
- **On the 11 rows still capped under $p$**, the reading the user asked for:
  - the cap goes on every row: no group keeps a region at depth ≥ 18 under $c$;
  - regions fall to 0.16–4.7 % of $p$'s on 10 rows, and to 34.5 % on R5;
  - L1's rule holds on 9 of 10 and L1q's on 1 of 1;
  - L3's rule holds on 11 of 11.
- **No unit went over budget.**
- **The store is byte-identical**, fingerprint `0e32f24a…`, `problems: none`.
- **B1 alone moves 36 of 39 stored totals beyond their declared error** (issues W1).

### Predictions and verdicts

Prompt §6's predictions, unchanged. The same bytes are `state/predictions.md` and
`state/prompt_section6_predictions.raw`. `state/predictions.md` was written at 01:57:36Z, after gate
L0 held (01:57:24Z) and before the launch (02:00:05Z).

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "Scored
group" is 03's definition: at least 10 regions at depth 18 or more, here under $p$.
- **L1 — the cap goes where $G$'s anchor was the floor.**
  - **The rule:** on every depth-20 row whose top group is `G+r`, `G−r` or the lone `G`, under $c$
    at `1e-7`, every group under $p$'s scored set has fewer than 10 regions at depth ≥ 18, and the
    row's regions are at most a tenth of $p$'s.
  - **Holds** if true on every such row; an over-budget unit counts as false.
- **L1q — the same in the `Gqr` rows.**
  - **The rule:** L1's rule, on the `Gqr`-topped rows.
  - **Holds** if true on every such row. A failure is evidence that $T$'s phase carries a floor
    that matters there, which is the case for re-anchoring $T$. Report the depth-18 regions'
    $z'$ against $z_{\rm response}$ on any row that fails.
- **L2 — the controls do not move beyond their error.**
  - **The rule:** on every control row, $|{\rm total}_c - {\rm total}_p| \le$ `total_abserr`$_p$.
- **L3 — the declared error is honest after the change.**
  - **The rule:** on every depth-20 row with both $c$ units in budget,
    $|{\rm total}_c(10^{-7}) - {\rm total}_c(10^{-8})| \le$ `total_abserr`$_c(10^{-7})$.

**Descriptive, not scored:**
- $|{\rm total}_c - {\rm total}_p| /$ `total_abserr`$_p$ on the depth-20 rows. This is 04's N4,
  wider, and now with the modulus correction in it too;
- cost and regions by class and band;
- `total_converged` before and after;
- the low band against the mid band.

**Verdicts** (`out/wider_05.txt`; "depth-20 row" read as the sample's depth-20 rows, as the user
agreed):

| | Number | Verdict |
|---|---|---|
| **Gate L0 (a)** | `QuadSource` at $p$: 360/360 stored nodes bit-identical in all nine columns, exactly B1's 3 nodes added, none removed, on both rows. The same in all 109 run units | holds |
| **Gate L0 (b)** | two in-process runs: result, group calls and region arrays bit-identical, on both rows | holds |
| **L1** | 9 of 24 rows true. Clause 1 holds on all 24. Clause 2 fails on 15: `low-G+r-1`, `low-G+r-2`, `low-G-r-1`, `low-G-2`, the nine drawn mid rows and R2 (none capped under $p$; $c/p$ 0.73–1.09), and R5 (capped, $c/p$ = 0.345) | **fails** |
| **L1q** | 1 of 7 rows true (R7, $c/p$ = 0.0016). It fails on the six drawn `Gqr` rows, none capped under $p$ ($c/p$ 1.00–1.35). The depth-18 regions' $z'$ the rule asks for: none exist, with 0 regions at depth ≥ 18 under $c$ and under $p$ on all six (maximum depth 2–7). So these failures are no evidence of a floor in $T$ | **fails** |
| **L2** | 8 of 8 controls true. Ratios: 0.161, 0.652, 0.0109, 0.00409, 4.3e-12, 6.6e-10, and 0 and 0 (the `NULL` controls are bit-identical) | **holds** |
| **L3** | 29 of 31 true. False on `mid-G-r-3` (ratio 1.29, relative 2.7e-6) and `mid-G-3` (1.70, 3.0e-7), neither capped under $p$. True on all 11 rows capped under $p$ (largest 0.166) | **fails** |

**The rows capped under $p$** (not scored; the user's reading, below). There are 11 of the 31:
`low-G+r-3`, `low-G-r-2`, `low-G-r-3`, `low-G-1`, `low-G-3`, R1, R3, R4, R5, R6, R7.

| row | $p$ depth, regions | $c$ 1e-7 depth, regions | $c/p$ | depth ≥ 18 under $c$, in $p$'s scored groups | L1 rule |
|---|---|---|---|---|---|
| `low-G+r-3` | 20, 1340 | 8, 33 | 0.0246 | 0 | true |
| `low-G-r-2` | 20, 15068 | 7, 42 | 0.0028 | 0 | true |
| `low-G-r-3` | 20, 6822 | 8, 40 | 0.0059 | 0 | true |
| `low-G-1` | 20, 3902 | 11, 34 | 0.0087 | 0 | true |
| `low-G-3` | 20, 831 | 7, 39 | 0.0469 | 0 | true |
| R1 | 20, 22852 | 8, 63 | 0.0028 | 0 | true |
| R3 | 20, 10457 | 5, 18 | 0.0017 | 0 | true |
| R4 | 20, 42830 | 8, 83 | 0.0019 | 0 | true |
| R5 | 20, 55 | 5, 19 | 0.345 | (no scored group) | **false** (clause 2) |
| R6 | 20, 5130 | 8, 28 | 0.0055 | 0 | true |
| R7 (`Gqr`) | 20, 16660 | 8, 27 | 0.0016 | 0 | true (L1q) |

**Descriptive:**
- **N4, wider:** $|c-p|$/`total_abserr`$_p$ is 0.0225–0.634 on the capped rows, and 2.48 on
  `low-G-3`. On the uncapped rows it is ≤ 0.932.
- **Cost:** low band, drawn rows, 28,054 → 288 regions and 101 → 4.5 s. Mid band, drawn rows,
  352 → 321 regions and 3.7 → 3.5 s (none capped). R1–R7, 97,999 → 250 regions and 545 → 2.4 s.
- **`total_converged`:** on the depth-20 rows it is False under $c$ on 21 of 31 at 1e-7 and 19 of 31
  at 1e-8. On 9 of the 11 capped rows it stays False.
- **Low band against mid band:** every drawn row still capped under $p$ is in the low band (5 of
  12). None of the 12 drawn mid rows is capped. 6 of 03's 7 mid rows are.

**The user's decision**, relayed by the laptop orchestrator after the launch, verbatim: *"§6
re-measurement: Carry on as you propose. I'm not sure I see all the consequences of this yet. I
guess we only care about rows that are capped under the new system (which I assume is what "p"
means)"*.
- The orchestrator clarified that $p$ is the code commit's parent, B1's code without 05's change.
  So "capped under $p$" means still capped after B1 and before 05.
- L1 and L1q are scored as written over the sample's rows. The capped-under-$p$ reading is reported
  beside them, unscored, and prominently.

### What shipped

Nothing is committed. Everything is under `var/levin-phase-floor-05/wider/`, and the checkpoints and
pickles are in the run directory.

**Scripts** (`scripts/`), each with a docstring naming its source and every change:
- `prov.py`: the two trees, their SHAs, and the header every output carries;
- `hash_store.py`: 04's, plus a journal/`-wal`/`-shm` check;
- `draw_sample.py`: the rule of `state/sample.md`;
- `harness.py`: 04's, with `QuadSource`'s `compute_quad_source.remote` and `ray.wait` stood in;
- `rebuild.py`: 04's, with `QuadSource` recomputed in-process and compared with the stored row
  (`qs_check`);
- `dispatch.py`: 04's `phases.py` cut down to its pass-through dispatcher and budget. It counts
  calls by kind and by `exact_remainder`;
- `worker.py`: 04's `run_remove.py` unit loop, one tree per process;
- `run_wider.py` and `launch.py`: the registered run;
- `l0.py`: gate L0;
- `analyse.py`: the scoring;
- `rec.sh`: records each command with its output.

**Outputs** (`out/`):
- `sample.txt`;
- `L0.txt` and `L0_final.txt`;
- `wider_05.txt`: every row, every group of the capped rows, the scores, and the B1 table;
- `fingerprint_05_wider.txt`;
- `cmd/`: every command run and its output, the failed attempts included.

**State** (`state/`):
- `sample.md`, `sample.json`, `predictions.md` and `prompt_section6_predictions.raw`;
- `L0/` and `L0_final/`: the gate's spec, ledger and pickles;
- `smoke/` and `smoke_attempt1_failed/`;
- `store_files_before`, `_prelaunch`, `_after` and `_after_fingerprint` (`.txt`);
- `launch.txt`, `launch_check.txt` and `registry_after.txt`;
- `analysis.json`.

**The run directory:** `var/runs/levin-phase-floor-05-wider-20261026T020005/`, 6.6 MB.
- `manifest.json`, `status.json`, `checkpoint.jsonl` (109 records), `stdout.log`, and `stderr.log`
  (15 scipy `IntegrationWarning`s, W9);
- `spec_{p,c}.json` and `worker_{p,c}_{hello,context,bye}.json`;
- 109 `unit_*.pkl`, the region arrays of every group call.

### Deviations from the prompt

1. **The `NULL`-depth controls are drawn from the store, not the census.** **STRUCTURALLY
   REQUIRED.** The census holds only rows with a Levin sub-interval, so it has no `NULL` row. The
   store was read with `sqlite3 mode=ro`, from the same 9280-row population (`state/sample.md`).
2. **Band controls must have a group containing $G$.** **IMPLEMENTATION CHOICE.** The cheapest seam
   rows otherwise have only an `r` group, on which the change cannot act. It is fixed in
   `state/sample.md` before any row ran.
3. **The top group's class.** It follows 01's T2 (the largest `elapsed`), and `Gqr` means the top
   group's regime is `Gqr`. The census writes `G-r` with an ASCII hyphen. **IMPLEMENTATION CHOICE.**
4. **How `QuadSource` is recomputed in-process.** **IMPLEMENTATION CHOICE**, within the prompt's
   "as `main.py` would compute it", with no production change.
   - The object is the factory's miss branch with `main.py`'s arguments, then `compute()`, then
     production's `store()`. `compute_quad_source.remote` and `ray.wait` are stood in.
   - The read-back is formed as `build()` forms it from a stored row, carrying the stored row's
     serial, because the `QuadSourceIntegral` row is keyed on it.
5. **The worktree.** **STRUCTURALLY REQUIRED** (the prompt runs two trees; the orchestrator set
   it up).
   - $p$'s code ran from the worktree through `PYTHONPATH`, with `-B` and
     `PYTHONDONTWRITEBYTECODE=1`.
   - Nothing was written there: `git status --short --ignored` is empty before and after.
   - The scripts and all outputs live in the main checkout's `var/`. So the manifest's `git_head` is
     `53084dc`, the $c$ tree (04's I4).
   - Each checkpoint payload records its own tree's `HEAD` and status, and every unit's evidence of
     where its modules were loaded from.
6. **One driver and one worker per tree.** One process cannot import two trees. The registered
   driver imports neither, and starts a $p$ worker and then a $c$ worker. Their unit records come
   over a pipe. **STRUCTURALLY REQUIRED.**
7. **The launch is `nohup sh -c '<driver>; exit $?'`,** so `sh` stays the driver's parent (CLAUDE.md
   rule 5). The `sh` pid (54763) was recorded first, then the driver's (54766). **IMPLEMENTATION
   CHOICE.**
8. **The budget** is 3× the row's $p$ wall time, measured as 04 measured it (around `compute()` and
   `ray.get`, after the rebuild). It is raised from the dispatcher at `theta_mod_2pi`'s entry.
   **IMPLEMENTATION CHOICE.** Rows without a Levin call can never be over budget. None was.
9. **L0 was run twice.** **IMPLEMENTATION CHOICE.**
   - The first run held (01:57:24Z). The predictions were written after it.
   - The smoke test then found that `worker.py` crashed on a `None` max depth (the `NULL` rows).
     The fix touches only that bookkeeping.
   - L0 was re-run with the final scripts (`out/L0_final.txt`) and gave an identical body. The
     launch followed.
10. **Two attempts failed before results** and are kept in `out/cmd/`:
    - the first L0 run (a module-lookup error in the worker's evidence code);
    - the first smoke test (the `None` max depth).
    **UNINTENDED DRIFT** in the harness, caught before any scored unit.
11. **The smoke test ran $c$ units** on `low-G+r-1` and `ctrl-NULL-1` before the run. It was after
    `state/predictions.md` was written, so nothing in the predictions could be influenced.
    **IMPLEMENTATION CHOICE.**
12. **"Depth-20 row" in L1, L1q and L3** is read as the sample's depth-20 rows (stored depth 20),
    since §6's "The rows" defines them that way. The rows at depth 20 under $p$ are reported beside
    them, unscored. This was proposed at the launch report, and the user agreed ("Carry on as you
    propose"; quoted above). The user said the capped-under-$p$ reading is the one that matters.
    **Recorded as the user's decision**, made after the launch and before any scoring; it changes
    no rule.
13. **The analysis was run twice.** The second run only adds, for L1q's failing rows, the count of
    depth-18 regions (zero), which the rule asks to report. The rest is identical
    (`out/cmd/17_analyse_v1.txt`). **IMPLEMENTATION CHOICE.**
14. **R9/R10's Ray builds shared the host** with the first minutes of this work. Wall times, and so
    the budgets, carry that load. No unit came near its budget: the largest $c$ wall is 0.80 s.
    **Recorded; not a deviation of method.**

### Verification performed

- **The trees.**
  - Main checkout at `53084dc` and the worktree at `e648951`, both with empty status, at the start,
    before launch, on every checkpoint record, and at the end (`state/registry_after.txt`).
  - The worktree's `git status --short --ignored` is empty.
- **Module provenance.** Each worker checked that `ComputeTargets`, `primitive_phase`,
  `QuadSource`, `QuadSourceIntegral`, `GkSourcePolicyData`, `StoreFixtures`, `main.py` and the
  original `theta_mod_2pi` came from its own tree, and refused to run otherwise.
- **Which phase each tree used.** The dispatcher counted the `theta_mod_2pi` calls:
  - under $p$, $G$ 24,192,372 and $T$ 14,924,742, with no `exact_remainder` attribute;
  - under $c$, $G$ 268,136 with `exact_remainder=True` and $T$ 167,562 with `False`.
- **The store.**
  - All six files' SHA-256, size and mtime are identical before, before launch, after the run and
    after the fingerprint. No journal, `-wal` or `-shm` file appeared.
  - `store fingerprint`, without `--write`, gives `0e32f24a…`, `problems: none`, "matches the
    fingerprint taken … at b90752d".
- **The run.**
  - It ended `done`, 109/109, in the planned order, exit 0, with none over budget and none in
    error.
  - Ray was never initialised: each worker checks at its end.
  - No lock was taken by this work.
  - The registry shows nothing `running` after it.
- **`QuadSource`.** L0 (a)'s comparison held in every one of the 109 units, not only on the gate's
  two rows.
- **Groups.** $p$ and $c$ made the same sequence of Levin group calls (regime, label, `x_span` bit
  for bit) on every row. So the sub-intervals and groups match one to one.

### Observations not acted on

All are in `issues.md`:
- **W1:** B1 moves 36/39 stored A3 v2 totals beyond their declared error, by ratios of 5.2 to
  1.5e5. One total changes sign.
- **W2:** under B1's code, 20 of the 31 stored depth-20 rows are no longer capped, R2 among them.
- **W3:** L1 and L1q's tenfold clause fails by construction on rows not capped under $p$.
- **W4:** R5's cap under $p$ is a single 55-region chain.
- **W5:** L3 fails on two uncapped mid rows: the error declared at 1e-7 misses the change to 1e-8.
- **W6:** on `low-G-3`, $|c-p|$ is 2.48 × `total_abserr`$_p$.
- **W7:** `total_converged` stays False on most depth-20 rows under $c$.
- **W8:** $c$'s region counts on R1–R7 against 04's arm A.
- **W9:** scipy `IntegrationWarning`s in both trees.

### State handed on

- **Nothing is running.** The run is `done`. No lock was taken.
- **The store is unchanged**, fingerprint `0e32f24a…`.
- **The worktree** `SecondaryGWKit-lpf05-e648951` is untouched, and is the orchestrator's to
  remove.
- **For the orchestrator:**
  - copy `out/`, `scripts/`, `state/` (the `.pkl` files under `state/L0*/` and `state/smoke*/` may
    be left out), `log.md`, `verification.md` and `issues.md` to `docs/levin-phase-floor/05/wider/`;
  - append this section to `logs/05-reanchor-g-phase.md`;
  - put `verification.md` in `VERIFICATION.md` §5. Its links assume the `05/wider/` location.
- **For the board:**
  - on the rows still capped under $p$, layer 1(a) removes the cap on all 11, and L3 holds on all
    11;
  - the scored L1, L1q and L3 fail only on rows that B1 had already uncapped, or on R5's short
    chain;
  - W1 is to be opened as an issue, as the laptop proposed.
