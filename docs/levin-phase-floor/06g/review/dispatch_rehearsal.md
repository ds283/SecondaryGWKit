## Your brief: the rehearsal agent — prompt 06g §4 (R23 and R24)

**Your scope is prompt 06g §4**, and only it. §5 belongs to the census agent and the references
agent, and runs after yours.

- **Where you write:** nothing outside these:
  - `var/levin-phase-floor-06g/rehearsal/`, with `out/`, `scripts/` and `state/`;
  - the stores `var/datastores/levin-phase-floor-06g-r23*` and
    `var/datastores/levin-phase-floor-06g-r24*`;
  - their run directories and launch logs in `var/runs/`;
  - the lock;
  - `/tmp/ray-sgk`.
- **The launches** are R21's and R22's (`docs/levin-phase-floor/06h/rehearsal/state/launch_r21.txt`
  and `launch_r22.txt`): `RunDrivers/quadsource_atol_sweep.py --build --rehearsal`, with these
  arguments:
  - `--database` the absolute store path, and `--job-name levin-phase-floor-06g`;
  - `--register`, with `--campaign levin-phase-floor --prompt 06g` and these slugs:
    - `r23-parent` for R23, from the parent worktree at `4f4f9ed`;
    - `r24-leafvalues` for R24, from the code worktree at `1c5336e`;
  - a `--purpose` that names the host, the tree's SHA and the code commit (06g `3ae6f3a`; R23's
    tree is its parent, 06h's code). It says that the fingerprint must equal `1b332ffa…`;
  - `--cpus 6`, never more.
- **One at a time,** R23 first, then R24:
  1. Acquire the lock, launch R23, make one check that it started, and end your turn with the
     launch report.
  2. You are resumed with R23's final `status.json`. Launch R24 the same way, and end your turn
     again. 06h's R21 and R22 took about 6 minutes each.
  3. After R24's `status.json`, do the comparison, V1, the readers and the release, and then report
     in full.
- **The lock** (`CLAUDE.md`, "The shared rehearsal lock"). The values are `--project SecondaryGWKit
  --ray-temp-dir /tmp/ray-sgk --ray-address in-process`; `status` and `ray-processes` take no
  `--project`.
  - `acquire` once, before R23;
  - `set-pid` with each driver's Python pid, at each launch;
  - held through the comparison and the readers;
  - `release` once `ray-processes --ray-temp-dir /tmp/ray-sgk` is empty.

  **Each lock command is a shell command of its own.** Append its output by `>>` to a transcript in
  `state/`. Never use a pipe, and never put a lock command in the command that launches a build. A
  held lock (exit 3) is a stop: do not wait, retry or remove it. Never run `ray stop`.
- **The fingerprints:** R23 and R24 must both give `1b332ffa…` (prompt §4). Anything else is a
  stop.
- **The comparison** is `tools.store_value_diff` (`python -m tools.store_value_diff`) on R23 and
  R24, with `--json`. Run it from the code worktree, and save its JSON in `out/`.
  - Copy 05's `classify_non_qsi.py` (`docs/levin-phase-floor/05/rehearsal/scripts/`) **unchanged**
    into `scripts/`, and record its SHA-256 beside 05's and 06h's copy.
  - **Outside `QuadSourceIntegral`** (prompt §4): every value table is identical, every other
    difference is of the four volatile kinds, and no row is in one store only. A value table that
    differs is a stop.
  - **Inside `QuadSourceIntegral`, tabulate every row** (prompt §4).
    - Use scripts of yours, saved in `scripts/`. You may copy and adapt 06h's
      `tabulate_qsi_groupphase.py` and `qsi_analytic_summary.py`
      (`docs/levin-phase-floor/06h/rehearsal/scripts/`); if you do, record the diff.
    - Put their output in `out/`: a CSV of every row (on `total` and on the oracle), and a summary.
    - Tabulate $|{\rm total}_{24} - {\rm total}_{23}|$ against `total_abserr` in each store, and
      the same on the analytic oracle (`analytic_rad` against `analytic.abserr`); the region
      counts, maximum depths and `total_converged` in each.
    - Say how many rows are bit-identical, by bit pattern (`float.hex`, never `isclose`), on
      `total` and on the oracle.
    - Record in your log and `verification.md` where the region counts, maximum depths and the
      oracle's value and error are stored: table, column and encoding. The orchestrator re-scores
      V1 with its own read-only reader over every row.
    - Work counters are not values. Say which fields you treat as counters, and why.
  - **Score V1 by its rule as written** (prompt §4), on `total` and on the oracle:
    - rows outside 06a's 720 stationary-point rows with finite errors in both stores are scored. Say
      how you identify those rows in the rehearsal store, from 06a's `spscan.py` or its record, and
      how many of the store's rows are among them;
    - rows with an infinite error in either store are counted separately;
    - the stationary-point rows are tabulated, not scored.

    Report every violating row, if any, with both values and both errors in `float.hex`. A
    violation is a finding, not by itself a stop. Also report the largest
    $|\Delta|/({\rm err}_{23} + {\rm err}_{24})$ on each of `total` and the oracle.
  - Open every store you read read-only (`mode=ro`).
- **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, in 06h's worktree
  adaptation `docs/levin-phase-floor/06h/rehearsal/scripts/run_readers_worktree.sh`) run on R24,
  from the code worktree, under the lock.
  - Hash the store's files (SHA-256) before and after.
  - Every reader exits 0, and the store is byte-identical before and after.
- **Release the lock.** Confirm that `ray-processes` lists nothing and `status` says free, each a
  command of its own.
- **The stores are kept.** Do not retire them.

**Your final report** comes after the release. It gives:
- R23 and R24's run ids, their `status.json` outcomes and their fingerprints;
- the comparison outside `QuadSourceIntegral`;
- the `QuadSourceIntegral` summary: rows, rows moved and bit-identical rows on `total` and on the
  oracle, and changes in region count, depth and `total_converged`;
- V1's verdict and numbers, on both;
- the readers' exits and the store hashes;
- the lock transcript;
- your deviations;
- anything in `issues.md`.
