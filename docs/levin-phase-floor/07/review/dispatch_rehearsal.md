## Your brief: the rehearsal agent — prompt 07 §5 (R27, R28, S0–S2)

Read `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/review/dispatch_common.md`
first, in full; it binds you.

**Your scope is prompt 07 §5**, and only it. §6 belongs to the census agent and runs after yours.

§5 says "As 06m §4". So also read:
- `prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md` §4, and the earlier §4s it
  follows (06g's, 06h's) as far as you need them;
- 06m's rehearsal records, `docs/levin-phase-floor/06m/rehearsal/` (`log.md`, `verification.md`,
  `issues.md`, `scripts/`, `state/`, and `out/commands.log` in full): R25 and R26 are the run you
  repeat, with R27 in R25's place and R28 in R26's;
- for S1 and S2: 06g's references, `docs/levin-phase-floor/06g/references/` (its `log.md`, `verification.md`,
  `out/scores_stage3/scores.json` with its `report.md`, and `v1/`, whose scripts captured R24's calls), the capture
  harness it used, `tools/filon_reference.py` and its tests, and P1's report
  `docs/levin-phase-floor/06/p1/report.md` for P1's two rows and their scan.

- **Where you write:** nothing outside these:
  - `var/levin-phase-floor-07/rehearsal/`, with `out/`, `scripts/` and `state/`;
  - the stores `var/datastores/levin-phase-floor-07-r27*` and
    `var/datastores/levin-phase-floor-07-r28*`;
  - your runs' directories and launch logs in `var/runs/` (R27, R28 and `p1-references`);
  - the lock;
  - `/tmp/ray-sgk`.
- **The launches** are R25's and R26's (`docs/levin-phase-floor/06m/rehearsal/state/launch_r25.txt`
  and `launch_r26.txt`, and their exact form in that `out/commands.log`):
  `RunDrivers/quadsource_atol_sweep.py --build --rehearsal`, with these arguments:
  - `--database` the absolute store path, and `--job-name levin-phase-floor-07`;
  - `--atol 1e-32 --rtol 1e-7 --cpus 6`, never more cpus;
  - `--register`, with `--campaign levin-phase-floor --prompt 07` and these slugs:
    - `r27-parent` for R27, from the parent worktree at `91472dbe`;
    - `r28-guard` for R28, from the code worktree at `74600350`;
  - a `--purpose` that names the host, the tree's SHA and the code commit (07 `ad732887`; R27's
    tree is its parent, whose production code is 06m's, R26's). It says that the fingerprint must
    equal `1b332ffa…`.
- **The lock** (`CLAUDE.md`, "The shared rehearsal lock"). The values are `--project SecondaryGWKit
  --ray-temp-dir /tmp/ray-sgk --ray-address in-process`; `status` and `ray-processes` take no
  `--project`.
  - Look first (`ray-processes`), then `acquire` once, before R27;
  - `set-pid` with each driver's Python pid, at each launch;
  - held through the comparisons, planned stop 1, S2's capture and the readers;
  - `release` once `ray-processes --ray-temp-dir /tmp/ray-sgk` is empty.

  **Each lock command is a shell command of its own,** and the only process it starts. Append its
  output by `>>` to `state/lock.log`. Never use a pipe, and never put a lock command in the command
  that launches a build. A held lock (exit 3) is a stop: do not wait, retry or remove it. Never run
  `ray stop`.

### The order

1. **R27.** Acquire the lock, launch R27, make one check that it started, and end your turn with
   the launch report.
2. **You are resumed with R27's final `status.json`.**
   - Check its outcome: `done`, exit 0, fingerprint `1b332ffa…`. Anything else is a stop.
   - **S0 — R27 against R26** (prompt §5.3): `tools.store_value_diff` with `--json` on R26's store
     and R27's, run from the **code worktree** (as 06m's comparisons were), both opened read-only.
     - The job names differ (`levin-phase-floor-06m` and `levin-phase-floor-07`): normalise the
       labels **in a copy** of the JSON, as 06m's `normalise_jobname_labels.py` did, and run 05's
       `classify_non_qsi.py` on the copy, **unchanged**. Record what the normalisation changes.
     - Every value table identical, **`QuadSourceIntegral` included**: R27's production code is
       R26's. Inside `QuadSourceIntegral`, establish identity by `float.hex` bit pattern on `total`,
       `total_abserr`, the analytic columns and every value field of every Levin record, over every
       row (06m's `qsi_identity_r24_r25.py` is the precedent), and say exactly what you compare.
     - Every other difference of the four volatile kinds, and no row in one store only.
   - **S0 holds** if so. **A value difference between R26 and R27 is a finding for the user, and a
     stop:** report it, with the first differing rows, before you launch R28.
   - If S0 holds, launch R28 the same way (`set-pid` with its driver's pid), make one check, and end
     your turn. 06m's R25 and R26 took about 15 minutes each.
3. **You are resumed with R28's final `status.json`.**
   - Check its outcome: `done`, exit 0, fingerprint `1b332ffa…`. Anything else is a stop.
   - **R28 against R27** (prompt §5.4): `tools.store_value_diff --json` on R27 and R28, from the
     code worktree; `classify_non_qsi.py` copied **unchanged** from
     `docs/levin-phase-floor/05/rehearsal/scripts/` into `scripts/`, its SHA-256 recorded beside
     05's and 06m's copy (`26690e61…`).
     - **Outside `QuadSourceIntegral`:** every value table identical, every other difference of the
       four volatile kinds, no row in one store only. A value table that differs is a stop.
     - **Inside `QuadSourceIntegral`, class every row** by R28's four counts (`stationary_points`,
       `window_pieces`, `guarded_regions`, `guard_at_depth_max`) in **every** Levin record of the
       row, group and analytic (the code log's §1.6 says where the analytic blocks' counts are
       kept). A row is **untouched** if every count in every one of its Levin records is 0.
       - **Every untouched row is bit-identical** to R27's in `total`, `total_abserr`, the analytic
         columns and every Levin record, the four counts set aside. An untouched row that is not is
         **a stop** (prompt §8), reported with its fields before anything else is scored.
       - **Every other row is tabulated**, with scripts of yours in `scripts/` (you may copy and
         adapt 06m's `tabulate_qsi_signedyardstick.py`; if you do, save the diff in `out/`): a CSV
         of every row and a summary, in `out/`. Per row: $|{\rm total}_{28} - {\rm total}_{27}|$
         against R27's and R28's `total_abserr`; the located points (kind, $z$); window pieces;
         regions; maximum depth; `total_converged` in each store. The same on the analytic columns
         where a three-Bessel call located a point.
       - **The stationary-point rows' convergence** in each store, and **the rows whose
         `total_converged` changes, in each direction**, listed.
       - How many rows are bit-identical by bit pattern, on `total` and on the analytic columns.
       - **The store keeps only the count of located points, not their kind or $z$.** If you need
         kind and $z$ (here, and for S1 and S2), say exactly how you obtain them, and check that
         whatever you run reproduces R28's stored record bit for bit. If obtaining them needs
         anything this brief or the prompt does not settle, stop and ask.
       - Record in your log and `verification.md` where each tabulated quantity is stored: table,
         column, JSON path and encoding, for the four counts in group and analytic records, region
         counts, maximum depths, `total_converged` and the analytic columns. **The orchestrator
         re-checks the classification and the bit-identity with its own read-only reader over
         every row,** so give these precisely.
       - Work counters are not values. Say which fields you treat as counters, and why.
   - Hash the stores' files (R26 and R27 at S0; R27 and R28 here) before and after each comparison;
     they must be byte-identical.
   - **Planned stop 1.** Then end your turn, **with the lock still held**, and report:
     - R27's and R28's run ids, `status.json` outcomes and fingerprints;
     - S0: the result, table by table, with the JSON's path and the normalisation's account;
     - R28 against R27 outside `QuadSourceIntegral`: the classification, with the JSON's path;
     - the `QuadSourceIntegral` summary: rows; untouched rows and how many are bit-identical; the
       other rows, how many located a point or guarded a leaf (group, analytic), the largest change
       against `total_abserr`, and `total_converged` changes in each direction, listed;
     - the storage of each quantity (table, column, path, encoding);
     - the store hashes;
     - your deviations so far.

     Do not run S1, S2 or the readers, or release the lock, until the orchestrator resumes you.
4. **You are resumed after the orchestrator's checks.**
   - **S1 — V1's twelve** (prompt §5.5), scored from R28's group records against `scores.json`'s
     `V1`, each with its located point. Holds or fails as written; a violation is a finding, not by
     itself a stop.
   - **S2's capture** (prompt §5.6): P1's two rows' `G+q+r` calls captured at **R27's tree** (the
     parent worktree), as 06g's `v1/` captured R24's, under the lock. Each capture checked bit for
     bit against the store's record.
   - **The readers** (06m's `docs/levin-phase-floor/06m/rehearsal/scripts/run_readers_worktree.sh`,
     as 06m's agent ran it) run on R28, from the code worktree, under the lock. Hash the store's
     files before and after: every reader exits 0, and the store is byte-identical. The readers'
     extract is not for copying back; list it in `KEPT_ON_MAC_STUDIO.txt`.
   - **Release the lock.** Look with `ray-processes` first; then `release`; then confirm that
     `status` says free and `ray-processes` lists nothing, each a command of its own.
   - **S2's references:** `tools/filon_reference.py`, four readings (QAWO and FCC at two
     resolutions, FCC at resolution 2 the reference), Ray-free, at most six workers, as **one
     registered run**, `--campaign levin-phase-floor --prompt 07`, slug `p1-references`, from the
     code worktree. Before launching, project its wall time; a projection beyond a day is a stop.
     If it may exceed about ten minutes, launch it detached, make one check, and end your turn
     with the launch report.
5. **You are resumed with the references' final `status.json`** (or go straight on, if they ran
   inside ten minutes).
   - **Score S2** as written: location within one scan spacing (say what the spacing is and where
     it comes from), and honesty against the reference where the four readings agree. A call that
     is one of w00–w11 is scored by S1 instead.
   - **The stores are kept.** Do not retire them.
   - Write `log.md`, `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.

**Your final report** comes after S2 is scored. It gives:
- R27 and R28's run ids, their `status.json` outcomes and their fingerprints;
- S0;
- the comparison outside `QuadSourceIntegral`;
- the `QuadSourceIntegral` summary;
- S1 with each call's located point, value, abserr and reference in `float.hex`, and its ratio;
- S2: each row's located point, the four readings, the value against its reference and abserr;
- the readers' exits and the store hashes;
- the lock transcript;
- the `p1-references` run id and outcome;
- your deviations, each classified;
- anything in `issues.md`.
