# Prompt 06a — the depth census: every A3 v2 row at the current code

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-27, after 06 landed and gate G2 was answered in part, on the user's decision
the same day. Its charter is G2's L2 (b) (the board's Decisions, **G2 L2**): *"L2 (b) first, then
make a decision whether layer 2 is needed. As you say, it doesn't look to be required."* G2's
option (b) reads: re-run A3 v2's rows in-process, Ray-free, as 05, 05a and 06 did, for the
distribution of depth, then decide.

**Where it runs:** on the Mac Studio, where A3 v2 and 06's certified scratch copy of it are.
- **No code changes**, so there is no laptop part and no code commit.
- **Ray-free.** It takes no lock.
- **The tree is a worktree** at the dispatch SHA. It is made before dispatch and does not move
  while the run is live, so the main checkout stays free for 07's rehearsals.
- **Orchestrated there**, by a session started from the laptop over the saved SSH connection. That
  session commits nothing.
- **The records are committed on the laptop afterwards**, in 06a's one records commit (README §4.1).

**Closes nothing by itself.** The user's L2 decision, made on its records, closes or narrows:
- `[04-x-at-1e-8-still-reaches-depth-20]`;
- `[03-depth20-regions-are-all-clenshaw-curtis]`;
- `[05-b1-uncaps-most-cheap-depth-20-rows]`.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §2.3 (gate G2 and its evidence) and its Decisions for **G2**. Then §3's entries for
   the three issues above, and for `[06-levin-accepts-a-region-across-a-stationary-point]`.
2. 06's prompt §6 and the Mac Studio record of its §6 in
   [`logs/06-producers-store-the-residual.md`](logs/06-producers-store-the-residual.md), from
   "## §6 — the re-measurement" to the end. This prompt runs 06's harness at 06's code, on
   every row instead of 39.
3. 06's harness, [`docs/levin-phase-floor/06/wider/scripts/`](../../docs/levin-phase-floor/06/wider/scripts/):
   `run_wider.py`, `worker.py`, `rebuild.py`, `rerun.py`, `harness.py`, `dispatch.py`, `prov.py`,
   `make_copy.py`, `certify_copy.py`, `fingerprint_copy.py` and `hash_store.py`. Then
   `docs/levin-phase-floor/06/wider/issues.md` W1 (why `HEAD` cannot open A3 v2) and
   [`docs/levin-phase-floor/06/KEPT_ON_MAC_STUDIO.md`](../../docs/levin-phase-floor/06/KEPT_ON_MAC_STUDIO.md).
4. [`measure/convergence_kinds.py`](measure/convergence_kinds.py), its docstring: the four kinds
   of non-convergence.
5. 01's census script, `docs/levin-phase-floor/01/scripts/group_census.py`, for the classes and
   the top-group class of each row. Then 05's `docs/levin-phase-floor/05/wider/scripts/draw_sample.py`,
   whose `band` takes 01's bands: low $z < 9$, mid $9 \le z \le 10^5$, seam $z > 10^5$.

**Gate:**
- the Mac Studio's main checkout is clean, and `runregistrykit list` there shows nothing
  `running` or stale;
- the worktree is at the dispatch SHA, `git status --short --ignored` empty;
- `git diff --stat f5cee4c <dispatch SHA> -- . ':!docs' ':!prompts'` shows `CLAUDE.md` alone. So
  the code is 06's $s$ tree's, byte for byte. If anything else appears, stop and ask;
- `raypoolkit.rehearsal_lock status` is recorded, for the record only. This prompt neither takes
  the lock nor waits for it.

---

## 0. What this prompt is

### 0.1 The question

G1 deferred layer 2 to G2 (Decisions, **Layer 2**). Layer 2 tells the driver its input's round-off
floor, so that it stops bisecting there. The evidence G2 had was 06's §6, on 05's 39 rows:

- under 06's code no row reaches depth 18 at `1e-7` or `1e-8`;
- the largest depth is 11.

The caveat (board §2.3) is the one this prompt removes. **Those 39 rows were drawn by A3 v2's
stored depth, and that describes the code before B1** (`[05-b1-uncaps-most-cheap-depth-20-rows]`).
No census of depth has been taken at the current code. A row that B1 and layers 1–3 left deep, but
that A3 v2 stored as shallow, could not have been drawn.

**So 06a re-runs every `QuadSourceIntegral` row of A3 v2, 9,280 of them, at the current code, at
`rtol = 1e-7` and at `1e-8`, and records each Levin call's depth.** The user then decides L2.

### 0.2 What it is not

- **Not a fix.** No file outside `var/` changes, on either host.
- **Not 07.** The census runs without the stationary-point guard. Where a region is accepted across
  an interior stationary point (`[06-levin-accepts-a-region-across-a-stationary-point]`), its depth
  is what `HEAD` gives, and may be too shallow. §5 item 7 records where that happens, for 07. It is
  not corrected here.
- **Not 08.** `rtol`, the depth cap, the dead `local_atol` branch and `total_converged` are G3's.
  The census records what 08 will need as a "before" (§5), and scores none of it.
- **Not a rebuild.** Nothing is written to any store. A3 v2 and its copy are read only.

### 0.3 The method, and how it differs from 06's §6

06's tree $s$ re-ran each row's producers, assembled its `GkSource`, recomputed its
`GkSourcePolicyData` and `QuadSource`, and ran `evaluate_QuadSource_integral` in-process, on the
certified scratch copy of A3 v2 (route (B), the user's decision). 06a does the same at the same
code. It differs in four ways, each needed to cover 9,280 rows instead of 39:

1. **Each producer is re-run once, not once per row.** 06's `rerun_row` re-ran about 1,500
   `GkWKBIntegration` objects for every row, at every rtol. That took about 37 s, plus about 3 s
   of factory lookups. Done that way, the census would need about 200 CPU-hours. A3 v2 has 8 values of $k$, and
   each has 1,740 `GkWKBIntegration` objects and one `TkWKBIntegration`. The census re-runs each
   object once, keeps its values, and assembles each of the 1,160 `GkSource`s, one per
   $(k, z_{\rm response})$, from them. Each `GkSourcePolicyData` is computed once, as is each
   `QuadSource`, one per $(q, r)$.
2. **One tree, no baseline.** There is no $t_5$ set. The tree is $h$, the worktree at the dispatch
   SHA.
3. **Parallel workers.** One worker process per $k$, at most six running at once (the Mac Studio's
   `--cpus 6` rule).
4. **No phase probe and no per-row `deltaTheta` pickles.** Each unit keeps its region arrays and
   the stationary-point scan of §5 item 7.

Gate C0 (§3) shows that change 1 reproduces 06's units bit for bit before anything is trusted to
it. Prediction C4 (§4) checks the same on all of 05's 39 rows after the run.

## 1. The tree and the store

- **The tree** $h$ is a detached worktree at the dispatch SHA,
  `../SecondaryGWKit-lpf06a-<short SHA>`, run with the main checkout's venv and `PYTHONPATH` set
  to the worktree, `-B` and `PYTHONDONTWRITEBYTECODE=1`, as 06 ran its trees. Each worker
  records its modules' files, all from $h$.
- **The store** is 06's certified scratch copy of A3 v2,
  `var/levin-phase-floor-06/wider/state/a3v2_copy/`. It is opened read-only, and `prov.py` and
  `rebuild.use_store_dir` point at it.
  - **Before the run:** its five files' SHA-256 equal `copy_files_certified.txt`'s, and
    `fingerprint_copy.py` at $h$ gives `0e32f24a…` with no problems.
  - If the copy is missing, or its hashes differ, stop and ask. Do not make a new copy without
    the user's word.
- **A3 v2 itself** (`var/datastores/handover-A3-v2-lambdacdm{,-shard000[0-3]}.sqlite`) is read
  only through `sqlite3` with `mode=ro`, for each row's stored total, error, depth, regions and
  compute time. The copy holds the same rows, and either may be read for them; the log says which.
- **Every file of both** is hashed (SHA-256, size, mtime) before the gate, before the launch, after
  the run and after the closing fingerprint. All must be equal. At the end, `store fingerprint`
  (no `--write`) is run on A3 v2 from the main checkout, and `fingerprint_copy.py` on the copy.
  Both must give `0e32f24a…`.

## 2. The harness

06's harness, copied from `docs/levin-phase-floor/06/wider/scripts/` into
`var/levin-phase-floor-06a/scripts/`, with 06's copy kept unchanged in
`state/scripts_06_reference/` to diff against. Change only what follows. Save the diff against
06's copy at launch and at the end. A change not named here is a deviation, and is classified.

1. **The tree.** One tree, `h`, in place of `t5` and `s`. Route (B) as in 06.
2. **The row list.** Every `QuadSourceIntegral` row of A3 v2, read by `(shard, serial)` with its
   $(k, q, r, z_{\rm response})$, band, the class of its top group and its stored depth (01's census
   classes). Saved as `state/rows.json`, with its SHA-256 in every record. 05's 39 rows, from 05's
   `sample.json` (`73aa7876…`), are marked in it by their names.
3. **The producer cache** (§0.3 change 1). `rerun.py` gains a per-$k$ re-run of every
   `GkWKBIntegration` and of the `TkWKBIntegration`, by its existing per-object code path,
   unchanged. `rebuild.py` gains a path that assembles a row's `GkSource` and
   `GkSourcePolicyData` from that cache, through the same functions `_rebuild_row_rerun` calls,
   and caches them by $(k, z_{\rm response})$, as `Gk_cache` is already keyed.
   - **The result must not depend on the order** in which rows are run. C0 (d) tests it.
   - **The per-object comparison with the stored value** (06's W9) is kept, summarised per $k$.
4. **The driver.** `run_wider.py` becomes `run_census.py`. It starts one worker per $k$, at most
   six at once, reads every worker's pipe, and records one checkpoint per unit, as 06's driver
   did, with its heartbeat thread and its SIGTERM handling. All of the $k$ workers for
   `rtol = 1e-7` finish before the `1e-8` workers start.
5. **The units.** One per (rtol, row): `h-1e-7-<shard>-<serial>` on all 9,280 rows, then
   `h-1e-8-<shard>-<serial>` on all 9,280. `atol` is A3 v2's, `1e-32`.
6. **The budget.** A fixed 300 s per unit, raised from the dispatcher as in 04–06. No unit of
   06's took more than about a second. An over-budget unit is a result. It is re-run once, after both sets, with a
   budget of 3,600 s (§3.2).
7. **What a unit keeps.** In its checkpoint, 06's per-call summary: value, abserr, converged,
   phase-limited, regions, simple regions, maximum depth, and the depth histogram split into
   Levin and Clenshaw–Curtis. Also the row's `total`, `total_abserr`, `total_converged` and wall
   time. In a per-worker pickle, each call's region arrays. **Not** the phase probe, not the
   re-run's `deltaTheta`, not the policy's full payload. The pilot (§3.1) projects the run
   directory's size.
8. **The stationary-point scan** of §5 item 7, run after the timed section with the dispatcher
   uninstalled, as 05a ran its probe. It reads phase objects only, so it changes neither the
   unit's result nor its counts.
9. **The `!! WARNING (GkSourcePolicyData` counter** from 06 is kept, per $(k, z_{\rm response})$.

## 3. Before the launch

### 3.1 Gate C0

- **(a) The code** is 06's: §Gate's `git diff --stat`.
- **(b) The store** is certified: §1.
- **(c) The census path is 06's path.** For 06's two U0 rows, `low-G+r-1` and `ctrl-low-1`, the
  census worker's `1e-7` unit is run through the producer cache. It must equal 06's `s-1e-7-*`
  unit in run `levin-phase-floor-06-wider-20261026T223634` **bit for bit**: `total`,
  `total_abserr`, every call's value and abserr, and the region arrays. The same holds for
  `low-G+r-1` at `1e-8` against `s-1e-8-low-G+r-1`.
- **(d) Order does not matter.** One row of a $(k, z_{\rm response})$ block, run first in a fresh
  worker and run last after the block's other rows, gives bit-identical units.
- **(e) No Ray.** No unit initialises Ray. `/tmp/ray-sgk` gains no session directory during C0,
  and the same holds for the run.

If C0 fails, stop.

**The pilot.** After C0, one $(k, z_{\rm response})$ block in each band (three blocks of 8 rows),
at both rtols, in one worker, timed end to end with the producer re-run of its $k$. Project from
it:
- the run's wall time at six workers;
- the run directory's size.

Record both before the launch. **If the projection exceeds 24 hours or 20 GB, stop and ask.**

**The orchestrator's estimate,** from the per-unit timings in 06's checkpoints, before the scan of
§5 item 7. That scan is not yet costed, and the pilot measures it.

| Step | How often | Each | Total |
|---|---|---|---|
| producer re-run, with factory lookups | per $k$, ×8 | ~45 s | ~6 min |
| `GkSource`, `GkSourcePolicyData` and $T$ | per $(k, z_{\rm response})$, ×1,160 | ~0.3 s | ~6 min |
| Levin at `1e-7` | per row, ×9,280 | ~0.30 s | ~0.8 h |
| Levin at `1e-8` | per row, ×9,280 | ~0.41 s | ~1.1 h |
| reading the row and its `QuadSource` | per unit, ×18,560 | ~0.2 s | ~1 h |

That is about 3–4 CPU-hours. With one worker per $k$, 8 workers in 6 slots finish in about two
workers' time, a quarter of the total, so about one hour of wall time. The Levin times come from
05's 39 rows, chosen as deep under the old code. The 6,080 seam rows were cheap even then, so the
estimate is more likely high than low.

### 3.2 The predictions, fixed before the run

Copy §4 into `state/predictions.md` before the launch, byte for byte, and record its SHA-256 and
mtime against the launch time (README §5 rule 3). A reading found necessary after that is recorded
beside the rule, never in place of it.

**The over-budget pass.** After both sets, every over-budget unit is re-run once, with 3,600 s. If
there are more than twelve, stop before the pass and ask.

## 4. Predictions

**Depth** means a Levin call's region depth, the largest over its regions. **Row depth** means the
largest over a row's calls.

- **C1 — at production's tolerance, nothing approaches the cap.**
  - **The rule:** at `rtol = 1e-7`, no row of the 9,280 has a region at depth ≥ 18.
  - **Holds** if true on every row. **Fails** if false on any row, or if any unit is still over
    budget after the pass. **Undecided** if neither, with a row that errored.
- **C2 — nor at `1e-8`.** C1's rule at `rtol = 1e-8`, scored the same way.
- **C3 — 05's sample was representative.**
  - **The rule:** at each rtol, no row's depth exceeds 11, the largest 06 found on 05's 39 rows
    at either tolerance.
  - **Holds** if true at both rtols. **Fails** otherwise.
  - C3 bears on how far 05's and 06's numbers stand for the store. It does not bear on L2 by
    itself.
- **C4 — the census reproduces 06.**
  - **The rule:** on 05's 39 rows the census's `1e-7` units, and on its 31 depth-20 rows the
    `1e-8` units, equal 06's `s` units bit for bit: `total`, `total_abserr`, every call's value
    and abserr, and the region arrays.
  - **A failure is a stop.** It means the census is not measuring 06's code path, and C1–C3 are
    not scored until the user has seen it.

**How the user reads them.** The decision is the user's (G2 L2).
- If C1 and C2 hold, no row of the store needs layer 2 at either tolerance, at this code.
- If either fails, the rows that fail are characterised by §5 items 2–3 and 7, so that the user
  can see:
  - whether they are the floor layer 2 addresses;
  - or the stationary-point defect 07 addresses;
  - or something else.

## 5. Descriptive, not scored

Each item is computed over all rows at each rtol, and tabulated by band (low, mid, seam), by regime
(`G`, `Gr`, `Gqr`), and by the class of the row's top group (`G+r`, `G−r`, lone `G`, `Gqr`).

1. **The distribution of row depth.** A histogram, and a cross-tabulation against A3 v2's stored
   depth in three bins (≤ 10, 11–19, 20). This is `[05-b1-uncaps-most-cheap-depth-20-rows]` over
   the whole store.
2. **Where depth falls by branch.** For every depth from 12 up, the number of regions that are
   Levin and the number that are Clenshaw–Curtis. This is `[03-depth20-regions-are-all-clenshaw-curtis]`
   at the current code.
3. **The deepest rows.** The 20 deepest at each rtol, with every call's label, depth, regions,
   branch split, convergence and value against abserr.
4. **Regions.** Per row, and the store's total, against A3 v2's stored region counts.
5. **Convergence.**
   - `total_converged` counts.
   - `convergence_kinds.py` over the census, with "capped" meaning depth 20 at $h$ itself. It is
     run unchanged if it can read the census's records. If it cannot, a copy is changed, and the
     diff and SHA-256 are recorded.
   - The rows whose `total_abserr` is $+\infty$ (`[04-roundoff-floor-can-be-infinite]`).
6. **The totals against A3 v2's.** $|{\rm total}_h - {\rm total}_{\rm A3\,v2}|$ at `1e-7`, relative
   and against the stored `total_abserr`. Give the distribution, and the count beyond 1×, 10× and
   100×. This is `[05-b1-moves-stored-a3v2-totals-beyond-their-error]`, measured on 39 rows,
   taken over the whole store. Also count the `GkSourcePolicyData` rows whose `Levin_z`, type or
   quality differs from A3 v2's.
7. **The stationary-point scan, for 07.** For every Levin call at `1e-7`:
   - **The scan.** Evaluate the call's own `theta["theta_deriv"]` on 4,097 points, uniform in the
     call's abscissa over its `x_span`. Record:
     - the number of sign changes;
     - $\min|\theta'|/\max|\theta'|$;
     - the abscissa of each sign change, to the grid spacing.
   - **Accepted regions.** For each accepted region that contains a sign change, record its depth,
     branch, abserr and its value against abserr. Where the grid shows none inside a region,
     evaluate $\theta'$ at the region's two ends and its midpoint. If those disagree in sign, the
     region is listed too.
   - **The count, across the store.** How many rows and calls hold an interior stationary point,
     and how many accept a region across one. That is
     `[06-levin-accepts-a-region-across-a-stationary-point]`'s extent at the current code.
   - **Its limits.** A pair of stationary points closer together than the grid spacing is not
     seen. Record the grid spacing per call, and say so beside the counts.
8. **Cost.** Wall time per unit and in total, the producer re-run's time per $k$, and CPU-hours,
   against A3 v2's stored compute time. Cost is not an argument for any decision (README §5
   rule 4). It is recorded because 08 and the next build will need it.
9. **Disagreement warnings.** The `!! WARNING (GkSourcePolicyData` count from §2 item 9, with each
   $(k, z_{\rm response})$ that printed one. They are not a stop here. 06 saw none on 39 rows; the
   census covers all 1,160 `GkSource`s.

## 6. The run

- **One registered, detached run**, slug `census`, `--campaign levin-phase-floor --prompt 06a`,
  launched as 06's `launch.py` launched its run: manifest first, stdout and stderr into the run
  directory, the driver's pid passed to `Run.heartbeat(pid=…)` (`CLAUDE.md`, "Long-running jobs").
- **Its purpose string** names the tree's SHA, the copy's fingerprint, the row count and the two
  rtols.
- **Launch, check once that it started, and end the turn** (`CLAUDE.md` rule 4). Resume when it is
  done.
- **If the run dies,** it is ended with `runregistrykit end` on the Mac Studio, and the resume is a
  new run that skips units already recorded. The resume must not re-stamp them. The orchestrator
  is told before any resume.

## 7. Records

Under `var/levin-phase-floor-06a/` on the Mac Studio (README §4.1):
- `scripts/`, `out/` and `state/`, with `log.md`, `verification.md` and `issues.md`. Every
  command and its output goes in `out/cmd/NN_*.txt`, as 06's agent kept them.
- **`out/census_rows.csv.gz`:** one line per (rtol, row), with the row's identity, band, regime,
  class, stored depth, $h$'s depth, regions, branch split, convergence, `total`, `total_abserr`,
  the stored values, wall time and the scan's counts. It is small enough to be copied to `docs/`;
  if it is over 2 MB, say so in the log.
- **`out/census_summary.txt`:** every table of §4 and §5.
- **`state/predictions.md`**, as §3.2 says.
- **The log**, in README §5.1's form. Its **Predictions and verdicts** gives C1–C4 as written, the
  number each was scored on, and holds, fails or undecided. Every deviation is numbered and
  classified before the agent reports, failed attempts included.
- **`verification.md`** becomes `VERIFICATION.md` §6a, "The depth census".
- **`issues.md`:** observations not acted on, each with its numbers. The orchestrator decides
  which are opened on the board.

The laptop copies them to `docs/levin-phase-floor/06a/`, leaving out pickles and anything over
2 MB, as `KEPT_ON_MAC_STUDIO.md` records for 06. It then makes the one records commit: the copied
records, `logs/06a-depth-census.md`, the board and `docs/OPEN_ISSUES.md`.

## 8. Acceptance

1. Gate C0 holds, and the pilot's projection is recorded before the launch.
2. The run ends `done`, with every unit recorded and none in error, or with each error listed and
   characterised.
3. C4 holds. C1, C2 and C3 are each scored holds, fails or undecided against the rule as written,
   with their numbers.
4. §5's items 1–9 are tabulated.
5. A3 v2 and its copy are unchanged, and both give `0e32f24a…`. No session directory appeared
   under `/tmp/ray-sgk`.
6. The worktree is clean at the end. Nothing was written outside `var/levin-phase-floor-06a/` and
   the run directory.

## 9. Stop conditions — stop and ask

- The Gate or C0 fails.
- The copy is missing, or its hashes or fingerprint differ from 06's certification.
- The pilot projects more than 24 hours of wall time or 20 GB.
- More than twelve units are over budget after the two sets.
- More than 1 % of units end in error.
- C4 fails.
- Any store file changes.
- The census needs a change to any file outside `var/levin-phase-floor-06a/`, on either host, or
  needs Ray.
