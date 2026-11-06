## Your brief: the rehearsal agent — prompt 06m §4 (R25 and R26)

Read `var/levin-phase-floor-06m/review/dispatch_common.md` first, in full; it binds you.

**Your scope is prompt 06m §4**, and only it. §5 belongs to the census agent and runs after yours.

§4 says "As 06g §4". So also read:
- `prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md` §4, and
  `prompts/levin-phase-floor/06h-group-phase-rounding.md` §4, which 06g's §4 follows;
- 06g's rehearsal records, `docs/levin-phase-floor/06g/rehearsal/` (`log.md`, `verification.md`,
  `issues.md`, `scripts/`, `state/`, and `out/commands.log` in full): R23 and R24 are the run you
  repeat, with R25 in R23's place and R26 in R24's.

- **Where you write:** nothing outside these:
  - `var/levin-phase-floor-06m/rehearsal/`, with `out/`, `scripts/` and `state/`;
  - the stores `var/datastores/levin-phase-floor-06m-r25*` and
    `var/datastores/levin-phase-floor-06m-r26*`;
  - their run directories and launch logs in `var/runs/`;
  - the lock;
  - `/tmp/ray-sgk`.
- **The launches** are R23's and R24's (`docs/levin-phase-floor/06g/rehearsal/state/launch_r23.txt`
  and `launch_r24.txt`, and their exact form in that `out/commands.log`):
  `RunDrivers/quadsource_atol_sweep.py --build --rehearsal`, with these arguments:
  - `--database` the absolute store path, and `--job-name levin-phase-floor-06m`;
  - `--atol 1e-32 --rtol 1e-7 --cpus 6`, never more cpus;
  - `--register`, with `--campaign levin-phase-floor --prompt 06m` and these slugs:
    - `r25-parent` for R25, from the parent worktree at `5ff8a5e4`;
    - `r26-signedyardstick` for R26, from the code worktree at `38f6a828`;
  - a `--purpose` that names the host, the tree's SHA and the code commit (06m `48cf87c0`; R25's
    tree is its parent, whose production code is R24's). It says that the fingerprint must equal
    `1b332ffa…`.
- **The lock** (`CLAUDE.md`, "The shared rehearsal lock"). The values are `--project SecondaryGWKit
  --ray-temp-dir /tmp/ray-sgk --ray-address in-process`; `status` and `ray-processes` take no
  `--project`.
  - Look first (`ray-processes`), then `acquire` once, before R25;
  - `set-pid` with each driver's Python pid, at each launch;
  - held through the comparisons, planned stop 1 and the readers;
  - `release` once `ray-processes --ray-temp-dir /tmp/ray-sgk` is empty.

  **Each lock command is a shell command of its own,** and the only process it starts. Append its
  output by `>>` to `state/lock.log`. Never use a pipe, and never put a lock command in the command
  that launches a build. A held lock (exit 3) is a stop: do not wait, retry or remove it. Never run
  `ray stop`.

### The order

1. **R25.** Acquire the lock, launch R25, make one check that it started, and end your turn with
   the launch report.
2. **You are resumed with R25's final `status.json`.**
   - Check its outcome: `done`, exit 0, fingerprint `1b332ffa…`. Anything else is a stop.
   - **R25 against R24** (prompt §4): `tools.store_value_diff` with `--json` on R24's store and
     R25's, run from the **code worktree** (as 06g's comparison was), both opened read-only.
     - Every value table identical, **`QuadSourceIntegral` included**: R25's production code is
       R24's.
     - Every other difference of the four volatile kinds, and no row in one store only.
     - Use 05's `classify_non_qsi.py` for the tables outside `QuadSourceIntegral`, and say how you
       establish identity inside it (by `float.hex` bit pattern on `total`, `total_abserr`, the
       oracle's value and error, and every other value column, over every row).
   - **A value difference between R24 and R25 is a finding for the user, and a stop:** report it
     before you launch R26, with the first differing rows.
   - If R25 equals R24, launch R26 the same way (`set-pid` with its driver's pid), make one check,
     and end your turn. 06g's R23 and R24 took about 5 and 10 minutes.
3. **You are resumed with R26's final `status.json`.**
   - Check its outcome: `done`, exit 0, fingerprint `1b332ffa…`. Anything else is a stop.
   - **R26 against R25:** `tools.store_value_diff --json` on R25 and R26, from the code worktree;
     `classify_non_qsi.py` copied **unchanged** from `docs/levin-phase-floor/05/rehearsal/scripts/`
     into `scripts/`, its SHA-256 recorded beside 05's and 06g's copy.
     - **Outside `QuadSourceIntegral`** (prompt §4): every value table identical, every other
       difference of the four volatile kinds, no row in one store only. A value table that differs
       is a stop.
     - **Inside `QuadSourceIntegral`, tabulate every row** (prompt §4), with scripts of yours in
       `scripts/`. You may copy and adapt 06g's `tabulate_qsi_leafvalues.py`,
       `qsi_analytic_summary.py` and `v1_violations_detail.py`
       (`docs/levin-phase-floor/06g/rehearsal/scripts/`); if you do, save the diff in `out/`.
       - A CSV of every row (on `total` and on the oracle) and a summary, in `out/`.
       - $|{\rm total}_{26} - {\rm total}_{25}|$ against `total_abserr` in each store, and the same
         on the analytic oracle (`analytic_rad` against `analytic.abserr`); region counts, maximum
         depths and `total_converged` in each; **the rows whose `total_converged` changes, in each
         direction**, listed.
       - How many rows are bit-identical by bit pattern (`float.hex`, never `isclose`), on `total`
         and on the oracle.
       - Record in your log and `verification.md` where the region counts, maximum depths,
         `total_converged` and the oracle's value and error are stored: table, column and
         encoding. **The orchestrator re-scores R1 with its own read-only reader over every row,**
         so give these precisely.
       - Work counters are not values. Say which fields you treat as counters, and why.
     - **Score R1 by its rule as written** (prompt §4), on `total` and on the oracle:
       - rows outside 06a's 720 stationary-point rows with finite errors in both stores are scored.
         Say how you identify those rows in the rehearsal store (06g's rehearsal agent did this;
         follow its method and say so), and how many of the store's rows are among them;
       - rows with an infinite error in either store are counted separately;
       - the stationary-point rows are tabulated, not scored.

       Report every violating row, if any, with both values and both errors in `float.hex`. A
       violation is a finding, not by itself a stop. Also report the largest
       $|\Delta|/({\rm err}_{25} + {\rm err}_{26})$ on each of `total` and the oracle.
   - Hash the four stores' files (R25, R26, and R24 read) before and after the comparisons; they must
     be byte-identical.
   - **Planned stop 1.** Then end your turn, **with the lock still held**, and report:
     - R25's and R26's run ids, `status.json` outcomes and fingerprints;
     - R25 against R24: the result, table by table, with the JSON's path;
     - R26 against R25 outside `QuadSourceIntegral`: the classification, with the JSON's path;
     - the `QuadSourceIntegral` summary: rows, rows moved and bit-identical rows on `total` and on
       the oracle; changes in region count, depth and `total_converged`, each direction listed;
     - R1's verdict and numbers on both;
     - the storage of each tabulated quantity (table, column, encoding) and how you identify the
       720 rows;
     - the store hashes;
     - your deviations so far.

     Do not run the readers or release the lock until the orchestrator resumes you.
4. **You are resumed after the orchestrator's checks.**
   - **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, in 06h's worktree
     adaptation `docs/levin-phase-floor/06h/rehearsal/scripts/run_readers_worktree.sh`, as 06g's
     agent ran it) run on R26, from the code worktree, under the lock. Hash the store's files before
     and after: every reader exits 0, and the store is byte-identical. The readers' extract is not
     for copying back; list it in `KEPT_ON_MAC_STUDIO.txt`.
   - **Release the lock.** Look with `ray-processes` first; then `release`; then confirm that
     `status` says free and `ray-processes` lists nothing, each a command of its own.
   - **The stores are kept.** Do not retire them.
   - Write `log.md`, `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.

**Your final report** comes after the release. It gives:
- R25 and R26's run ids, their `status.json` outcomes and their fingerprints;
- R25 against R24;
- the comparison outside `QuadSourceIntegral`;
- the `QuadSourceIntegral` summary, and R1's verdict and numbers on both;
- the readers' exits and the store hashes;
- the lock transcript;
- your deviations, each classified;
- anything in `issues.md`.
