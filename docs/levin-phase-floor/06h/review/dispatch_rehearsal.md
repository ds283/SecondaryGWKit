## Your brief: the rehearsal agent — prompt 06h §4 (R21 and R22)

**Your scope is prompt 06h §4**, and only it. §5 belongs to the census agent and the phase agent,
and runs after yours.

- **Where you write:** nothing outside these:
  - `var/levin-phase-floor-06h/rehearsal/`, with `out/`, `scripts/` and `state/`;
  - the stores `var/datastores/levin-phase-floor-06h-r21*` and
    `var/datastores/levin-phase-floor-06h-r22*`;
  - their run directories and launch logs in `var/runs/`;
  - the lock;
  - `/tmp/ray-sgk`.
- **The launches** are R19's and R20's (`docs/levin-phase-floor/06e/rehearsal/state/launch_r19.txt`
  and `launch_r20.txt`): `RunDrivers/quadsource_atol_sweep.py --build --rehearsal`, with these
  arguments:
  - `--database` the absolute store path, and `--job-name levin-phase-floor-06h`;
  - `--register`, with `--campaign levin-phase-floor --prompt 06h` and these slugs:
    - `r21-parent` for R21, from the parent worktree at `2c325f5`;
    - `r22-groupphase` for R22, from the code worktree at `c1ec3b2`;
  - a `--purpose` that names the host, the tree's SHA and the code commit (06h `7172a8f` +
    `ac1c6f4`; R21's tree is their parent, 06e's code). It says that the fingerprint must equal
    `1b332ffa…`;
  - `--cpus 6`, never more.
- **One at a time,** R21 first, then R22:
  1. Acquire the lock, launch R21, make one check that it started, and end your turn with the
     launch report.
  2. You are resumed with R21's final `status.json`. Launch R22 the same way, and end your turn
     again. 06e's R19 and R20 took about 6 minutes each.
  3. After R22's `status.json`, do the comparison, H0, the readers and the release, and then report
     in full.
- **The lock** (`CLAUDE.md`, "The shared rehearsal lock"). The values are `--project SecondaryGWKit
  --ray-temp-dir /tmp/ray-sgk --ray-address in-process`; `status` and `ray-processes` take no
  `--project`.
  - `acquire` once, before R21;
  - `set-pid` with each driver's Python pid, at each launch;
  - held through the comparison and the readers;
  - `release` once `ray-processes --ray-temp-dir /tmp/ray-sgk` is empty.

  **Each lock command is a shell command of its own.** Append its output by `>>` to a transcript in
  `state/`. Never use a pipe, and never put a lock command in the command that launches a build. A
  held lock (exit 3) is a stop: do not wait, retry or remove it. Never run `ray stop`.
- **The fingerprints:** R21 and R22 must both give `1b332ffa…` (prompt §4.2). Anything else is a
  stop.
- **The comparison** is `tools.store_value_diff` (`python -m tools.store_value_diff`) on R21 and
  R22, with `--json`. Run it from the code worktree, and save its JSON in `out/`.
  - Copy 05's `classify_non_qsi.py` **unchanged** into `scripts/`, and record its SHA-256 beside
    05's.
  - **Outside `QuadSourceIntegral`** (prompt §4.3): every value table is identical, every other
    difference is of the four volatile kinds, and no row is in one store only. A value table that
    differs is a stop.
  - **Inside `QuadSourceIntegral`, tabulate every row** (prompt §4.3).
    - Use a script of yours, saved in `scripts/`. You may copy and adapt 06e's
      `tabulate_qsi_endnodes.py`; if you do, record the diff.
    - Put its output in `out/`: a CSV of every row, and a summary.
    - Tabulate $|{\rm total}_{22} - {\rm total}_{21}|$ against `total_abserr` in each store, and the
      region counts, maximum depths and `total_converged` in each.
    - **No row is expected to be bit-identical.** Say how many are, by bit pattern (`float.hex`,
      never `isclose`).
    - Record in your log and `verification.md` where the region counts and maximum depths are
      stored: table, column and encoding. The orchestrator re-scores H0 with its own read-only
      reader over every row.
    - Work counters are not values. Say which fields you treat as counters, and why.
  - **Score H0 by its rule as written:**
    - rows outside 06a's 720 stationary-point rows are scored. Say how you identify those rows in
      the rehearsal store, from 06a's `spscan.py` or its record, and how many of the store's rows
      are among them;
    - rows with an infinite `total_abserr` in either store are counted separately;
    - the stationary-point rows are tabulated, not scored.

    Report every violating row, if any, with both totals and both `total_abserr` in `float.hex`. A
    violation is a finding, not by itself a stop. Also report the largest
    $|\Delta{\rm total}|/{\rm total\_abserr}$, in each store.
  - Open every store you read read-only (`mode=ro`).
- **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, in 06e's worktree
  adaptation) run on R22, from the code worktree.
  - Hash the store's files (SHA-256) before and after.
  - Every reader exits 0.
- **Release the lock.** Confirm that `ray-processes` lists nothing and `status` says free, each a
  command of its own.
- **The stores are kept.** Do not retire them.

**Your final report** comes after the release. It gives:
- R21 and R22's run ids, their `status.json` outcomes and their fingerprints;
- the comparison outside `QuadSourceIntegral`;
- the `QuadSourceIntegral` summary: rows, rows moved, bit-identical rows, and changes in region
  count, depth and `total_converged`;
- H0's verdict and numbers;
- the readers' exits and the store hashes;
- the lock transcript;
- your deviations;
- anything in `issues.md`.
