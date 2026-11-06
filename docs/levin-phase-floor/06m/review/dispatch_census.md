## Your brief: the census agent — prompt 06m §5 (Gate M0, the census $m$, Gate M, the descriptive tables)

Read `var/levin-phase-floor-06m/review/dispatch_common.md` first, in full; it binds you.

**Your scope is prompt 06m §5**, and only it. §4, the rehearsals R25 and R26, is the rehearsal
agent's. It has finished: both stores gave `1b332ffa…`, the readers ran and the lock is released.
You need no lock, and you start no Ray session.

§5 re-runs 06l's census at the tree, with no variant. So also read:
- `prompts/levin-phase-floor/06l-signed-trust-yardstick.md`, its census and K0 sections in full,
  and as much of the rest as you need to read its records;
- 06l's records, `docs/levin-phase-floor/06l/`: `log.md` (all of it, its deviations above all),
  `out/results.md`, `verification.md`, `KEPT_ON_MAC_STUDIO.md`, `out/k0/`, and its harness
  `scripts/`: `run_census.py`, `run_k0.py`, `compare_k0.py`, `launch_census_06l.py`, `worker.py`,
  `harness.py`, `dispatch.py`, `rerun.py`, `prov.py`, `refs_common.py`, `variant_loader.py`,
  `extract_census.py`, `kept_list_06l.py`, and whatever these import;
- 06g's census records, `docs/levin-phase-floor/06g/census/` (`log.md`, `verification.md`, `out/`),
  for the flag-cost table's 06g column and the cost comparison.

### Where you write

Nothing outside these:
- `var/levin-phase-floor-06m/census/`, with `out/`, `scripts/` and `state/` (and your `log.md`,
  `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`);
- your registered runs' directories in `var/runs/` (`levin-phase-floor-06m-k0-…` and
  `levin-phase-floor-06m-census-…`, and nothing else unless you report it as a deviation).

### The harness

- **Copy** `docs/levin-phase-floor/06l/scripts/` into `var/levin-phase-floor-06m/census/scripts/`
  (prompt §5), and save `diff -r` of the copy against 06l's in `out/` before and after your changes.
  Every change is a deviation, marked in the code as 06l's were marked ("prompt 06m").
- **No variant is installed.** Each worker imports the code worktree's own
  `AdaptiveLevin/levin_quadrature.py` (`beaf0e23…`); the harness records the module's path and
  SHA-256, and you check it in the records. 06l's `variant_loader` and `--variant` are not used.
- **What 06l's harness reads that is gone.** On 2026-11-05 06l's large state was deleted (its
  `KEPT_ON_MAC_STUDIO.md`), among it `state/rows.json` and `state/index_06g.json`. The rows file is
  06g's `var/levin-phase-floor-06g/census/state/rows.json` (`aa523d28…`), which 06l's census used:
  copy it into your `state/` and check its SHA-256. Anything else the harness needs and lacks
  (an index of route B's checkpoint, for instance) is rebuilt by a script of yours from the kept
  inputs, read only, and recorded as a deviation.
- **Provenance:** every output has a header with the script's SHA-256, the tree, the module's
  SHA-256 and the inputs' SHA-256.

### The comparison (prompt §5), the same for M0 and M

Every call matched by unit, kind, index, label and span. In each call record, every key but
`route_06l` (route B's instrument, absent at $m$) and the timing fields 06l's `compare_k0.py`
already excludes. In each unit's region records, every field, timing aside. Floats by bit pattern
(`float.hex`), arrays by dtype, shape and bytes. The "before" is route B's census
`levin-phase-floor-06l-census-routeB-20261105T100126`: its `checkpoint.jsonl` (`5cb3d3c0…`) and its
16 region pickles, read only, each SHA-256 checked against `docs/levin-phase-floor/06l/KEPT_ON_MAC_STUDIO.txt`
before use. **Say exactly which keys and fields you exclude, and why each is timing.**

### The order

1. **Before any number:** the Gate checks you rely on (trees, module hashes, the scratch copy
   against its certification, route B's and 06g's checkpoints, the registry), and the scratch
   copy hashed (`out/`). **Copy prompt §5's Gates M0 and M** from the code log's "Predictions and
   verdicts" verbatim into `predictions.md`, before K0's first unit. Do not rewrite it afterwards (a
   correction is a new file and a deviation).
2. **Gate M0.** One registered run, `--campaign levin-phase-floor --prompt 06m`, slug `k0`, on the
   100 units of `docs/levin-phase-floor/06l/out/k0/units.json`, at most six processes. If it may
   exceed about ten minutes, launch it detached, make one check, and end your turn with the launch
   report; the orchestrator resumes you when its pid exits. Then compare all 100 against route B's
   census. **Holds** if all 100 are equal. A failure is a stop.
3. **Planned stop 2,** after M0 and before the census. End your turn and report:
   - M0's result: units, calls and region records compared, and differing (with every differing
     path if any), and where the comparison's output is;
   - the run id, where each unit's emitted record and region pickle are, and **the exact command
     that runs a chosen list of units into a directory of the caller's choosing** (the
     orchestrator re-runs two units itself and compares them with route B's census), with the
     form of the units file;
   - the harness diff against 06l's, every change classified;
   - **the census plan:** the launch command (registered, slug `census`, detached, six workers, a
     300 s budget per unit with 06a's over-budget pass, as 06l's); the cost projection from route
     B's wall (18,340 s) and 06g's (20,361 s), with the load as you find it; and the backstop: a
     run past its projection is reported with its progress and projected finish, and **stopped only
     at a projected day or more, counting the units in flight, not only those finished**;
   - your deviations so far.

   Launch no census before the orchestrator resumes you.
4. **The census.** Launch it as planned, make one check, and end your turn with the launch report.
   Do not poll it. Nothing else computes while it runs.
5. **Planned stop 3.** You are resumed after the census's process has exited and the orchestrator
   has checked that it is `done`, 18,560 of 18,560. Only then score it.
6. **Gate M, first and alone.** Compare every unit with route B's census by the comparison above.
   **A single call that differs from route B's is a stop, before anything else is scored:** end
   your turn with the first differing units and fields, and the count. A unit that ended in an
   error (not over budget) is a stop too.
7. **If Gate M holds,** 06l's verdicts L1, C1, C2, H1, S1, K1 and L2 as scored are the code's, by
   identity, not re-scored. The records say so and cite `VERIFICATION.md` §6l.
8. **The descriptive tables** (prompt §5), by your own reader:
   - **the flag's cost** at $m$, for every unconverged call, outside the 720 and in them: its
     `abserr` against its request $\max(\texttt{atol}, \texttt{rtol}\,|v|)$, whether it is
     `node_limited`, and its share of its unit's `total_abserr`. The same for 06g's census
     (`03adcbba…`), beside it. State how the 720 are identified (06a's rule, as 06l's records do).
     The orchestrator's count from 06l's extract was: outside the 720, 114 calls in 84 units; 104
     within their request; at most 1.05× over; share at most 4.7 %;
   - **cost:** the census wall (manifest `created` to last heartbeat, as 06l) and group-call
     regions, at each rtol and in all, against 06g's and route B's; and the load you saw;
   - **any firing count** the tables need is read from route B's census (`route_06l`);
   - **the open issue** `[06m-levin-floor-misses-node-rounding-far-from-the-origin]`: if your
     records hold $\Delta_1$ and $\phi_P$ for checked Levin leaves, count those with
     $\Delta_1/\phi_P > 10$ as descriptive; if not, say so and leave it for 07.
9. **After:** hash the scratch copy, route B's census and 06g's census again; each must equal its
   before. Then write `log.md`, `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt` (the
   census run directory in full, every pickle, every file over 2 MB). **The census's checkpoint and
   unit records stay on the Mac Studio in full; 07 reads them.**

**Your final report:** M0 and M with their numbers, the flag-cost table at $m$ and 06g, the cost
table, the $\Delta_1/\phi_P$ count or why there is none, the before/after hashes, every run id,
every deviation classified, anything in `issues.md`, and anything left running (there should be
nothing).
