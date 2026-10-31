## Your brief: the rehearsal agent — prompt 06e §4 (R19 and R20)

**Your scope is prompt 06e §4**, and only it. §5 (the census and the Z3 references) is the other
agent's, and runs after yours: it prepares while your rehearsals run, without computing.

- **Where you write:** `var/levin-phase-floor-06e/rehearsal/` (with `out/`, `scripts/`, `state/`);
  the stores `var/datastores/levin-phase-floor-06e-r19*` and `var/datastores/levin-phase-floor-06e-r20*`;
  their run directories and launch logs in `var/runs/`; the lock; and `/tmp/ray-sgk`. Nothing else.
- **The launches** are R17's and R18's (`docs/levin-phase-floor/06c/rehearsal/state/launch_r17.txt`,
  `launch_r18.txt`), i.e. `RunDrivers/quadsource_atol_sweep.py --build --rehearsal`, with these
  arguments:
  - `--database` the absolute store path, `--job-name levin-phase-floor-06e`;
  - `--register` with slugs `r19-parent` (R19, from the parent worktree at `50277f3`) and
    `r20-endnodes` (R20, from the code worktree at `6d81fad`), and
    `--campaign levin-phase-floor --prompt 06e`;
  - a `--purpose` that names the host, the tree's SHA and the code commit (06e `f4d8e17`; R19's
    tree is its parent, 06d's code), and says that the fingerprint must equal `1b332ffa…`;
  - `--cpus 6`, never more.
- **One at a time.** R19 first, then R20. Acquire the lock, launch R19, make one check that it
  started, and end your turn with the launch report. You will be resumed with R19's final
  `status.json`. Then launch R20 the same way, and end your turn again. 06c's R17 and R18 took a few
  minutes and about 20 minutes. After R20's `status.json` you do the comparison, Z1, the readers and
  the release, and then report in full.
- **The lock** (`CLAUDE.md`, "The shared rehearsal lock"; values `--project SecondaryGWKit
  --ray-temp-dir /tmp/ray-sgk --ray-address in-process`, with `status` and `ray-processes` taking no
  `--project`):
  - `acquire` once before R19;
  - `set-pid` with each driver's Python pid, at each launch;
  - held through the comparison and the readers;
  - `release` once `ray-processes --ray-temp-dir /tmp/ray-sgk` is empty.

  **Each lock command is a shell command of its own,** with its output appended by `>>` to a
  transcript in `state/`, never a pipe, and never in the same command that launches a build. A held
  lock (exit 3) is a stop: do not wait, retry or remove it. Never run `ray stop`.
- **The fingerprints:** both R19 and R20 must give `1b332ffa…` (prompt §4.2; anything else is a
  stop).
- **The comparison** is `tools.store_value_diff` (`python -m tools.store_value_diff`) on R19 and
  R20 with `--json`, run from the code worktree, its JSON saved in `out/`.
  - 05's `classify_non_qsi.py` is copied **unchanged** into `scripts/`, and recorded by SHA-256
    beside 05's.
  - **Outside `QuadSourceIntegral`:** every value table identical, every other difference of the
    four volatile kinds, no row in one store only (prompt §4.3; a value table that differs is a
    stop).
  - **Inside `QuadSourceIntegral`, every row is tabulated** (prompt §4.3), by a script of yours
    saved in `scripts/` (06c's `tabulate_qsi_global.py` may be copied and adapted; record the diff),
    its output in `out/` (a CSV of every row, and a summary): $|{\rm total}_{20} - {\rm total}_{19}|$
    against `total_abserr` in each, the region counts, maximum depths and `total_converged` in each.
    **No row is expected to be bit-identical**; say how many are, by bit pattern (`float.hex`, never
    `isclose`). Record where the region counts and maximum depths are stored (table, column,
    encoding) in your log and `verification.md`; the orchestrator re-scores Z1 with its own
    read-only reader over every row. Work counters are not values; say which fields you treat as
    counters and why.
  - **Z1** is scored by its rule as written: rows outside 06a's 720 stationary-point rows (say how
    you identify them in the rehearsal store, from 06a's `spscan.py` or its record, and how many of
    the store's rows are among them), rows with an infinite `total_abserr` in either store counted
    separately, the stationary-point rows tabulated and not scored. Report every violating row (if
    any) with both totals and both `total_abserr` in `float.hex`; a violation is a finding, not by
    itself a stop. Also report the largest $|\Delta{\rm total}|/{\rm total\_abserr}$ (in each store).
  - Open every store you read for this read-only (`mode=ro`).
- **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, in 06c's worktree
  adaptation) run on R20, from the code worktree, with the store's files hashed (SHA-256) before and
  after. Every one exits 0.
- **Release the lock**, and confirm `ray-processes` lists nothing and `status` says free, each a
  command of its own.
- **The stores are kept.** Do not retire them.

**Your final report** (after the release) gives: R19 and R20's run ids, `status.json` outcomes and
fingerprints; the comparison outside `QuadSourceIntegral`; the `QuadSourceIntegral` summary (rows,
rows moved, bit-identical rows, region-count and depth changes, `total_converged` changes); Z1's
verdict and numbers; the readers' exits and the store hashes; the lock transcript; your deviations;
and anything in `issues.md`.
