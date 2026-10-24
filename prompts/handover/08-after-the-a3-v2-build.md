# Prompt 08 — after the A3 v2 build: the store's record, its checks, its readers, its depth census

**Campaign:** [`README.md`](README.md) · **Board item:** **08** (outside workstreams A–E) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25 at `216e44d`, after the build ended, to the charter fixed in 07 §7.
**Where it runs:** on the Mac Studio, in the Claude Code session that ran 06 and 07. **It commits
nothing there**: everything it writes is under `var/handover-08/` (§7), and the orchestrator
commits it on the laptop.
**Gate (each read by the orchestrator on 2026-10-25, read-only):**
- run `handover-07-a3-v2-20261022T211155` is `done`, exit code 0; the launch log ends
  `** driver exit status 0`; pid 84436 is gone. Last heartbeat 2026-10-24T21:56:27+01:00, about
  48 h 45 m after the launch;
- its `status.json` fingerprint is format 6, `encoding_format` 1, digest
  **`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`**, 21 classes, no problems,
  taken at `b90752d` with `git_dirty` false; `QuadSourceIntegral` 9280, `GkSource` 1160,
  `QuadSource` 36; `fingerprint_sidecar` "not written … (there is none)";
- `var/datastores/handover-A3-v2-lambdacdm.sqlite` and four shards of ~106 MB, last written
  21:56:16, with no journal, WAL or shm file;
- `list`: five runs, nothing `running` or stale. The lock is **held** by SecondaryGWKit, purpose
  "… 07: A3 v2 build …", owner pid 84436 (**not alive**). No Ray process of this checkout;
- the checkout at `b90752d`, `git status --short` empty.

**Opens:** anything you find, **without fixing it**.
**Recommended model:** **Opus**.

**Read first:**

1. `CLAUDE.md`, "Long-running jobs — the run registry" and "The shared rehearsal lock".
2. [`07-launch-a3-v2.md`](07-launch-a3-v2.md) §7, this prompt's charter, and
   [`docs/a3-v2-readiness-verification.md`](../../docs/a3-v2-readiness-verification.md) §2.4, its
   last three items ("`store create` …", "After the build", "Every reader").
3. `docs/handover/A3-V2-VERIFICATION.md` §1.2–§1.4 and the scripts R4 ran them with,
   `docs/handover/a3-v2/rehearsal/scripts/checks.sh` and `after_run.sh`. **This prompt runs the same
   tools on one store**, the A3 v2 store, as they stand.
4. This board's §3 entries `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`
   (the table at its head is the baseline's census, and the method §4 reproduces) and
   `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`; and
   [`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md`](../../docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md)
   lines 60–88 (the same table, the squeezed triples) and 237–250 (the bands and the projection).
5. `RunDrivers/quadsource_atol_sweep.py` `report()` (`:804-879`): the read-only pattern §4 follows,
   shard by shard, `mode=ro`, with the `redshift`, `wavenumber_exit_time`/`wavenumber` and
   `tolerance` maps from shard 0. **Read only; do not edit or run it.**

---

## 0. What this prompt is

A3 v2 has ended. This prompt makes it a **registered, verified, read store**: the sidecar and its
fingerprint; the after-build checks; every reader; and the one scientific number 07 §7 asked for —
**whether `rtol = 1e-7` removed the depth-20 cliff** that cost the A3 baseline 97.5 of its 98.2
CPU-hours. It then releases the lock, which ends the freeze on the Mac Studio's checkout.

**The archive is the Mac Studio** (the user, 2026-10-25): the store stays at
`var/datastores/handover-A3-v2-lambdacdm*` with its sidecar, and the run at
`var/runs/handover-07-a3-v2-20261022T211155/`. Nothing is copied.

---

## 1. Before anything

Make `var/handover-08/{out,scripts,state}`. Save every command and its output, in order, in
`out/prechecks_08.txt`, and **run every command that takes arguments from a variable through
`bash`, not zsh** (06's log, deviations 8–10).

1. `git rev-parse HEAD` is `b90752d…`; `git status --short` empty.
2. The Gate's run, status, fingerprint and store files, as listed: confirm each.
3. `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status` and `ray-processes`: held, owner
   84436 not alive; no Ray process. **Do not release yet** (§6).
4. The SHA-256, size and mtime of the primary and of every shard: `state/store_files_before.txt`.

---

## 2. The store's record

1. `./venv/bin/python -m runregistrykit store create --purpose "A3 v2: the LambdaCDM store rebuilt
   from scratch at QuadSourceIntegral atol 1e-32 rtol 1e-7, 9280 items, 8 k over 1e5-3e8/Mpc, zend
   0.1; built by handover-07-a3-v2-20261022T211155 at b90752d on the Mac Studio. Kept; archive on the
   Mac Studio." var/datastores/handover-A3-v2-lambdacdm.sqlite`
2. `./venv/bin/python -m runregistrykit store fingerprint --write
   var/datastores/handover-A3-v2-lambdacdm.sqlite`. Its digest **must equal `0e32f24a…`**, the run's
   recorded fingerprint, with `compare_fingerprints` empty.
3. `./venv/bin/python -m runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite`:
   live, the run named, the fingerprint the sidecar's.
4. Every store file's SHA-256, size and mtime again: `store create` and `fingerprint --write` write
   the sidecar only. Any store file changed is a stop.

Save as `out/store_record_08.txt`.

---

## 3. The after-build checks

Each as R4 ran it on its stores (`checks.sh`), on the A3 v2 primary alone; each output in `out/`:
- `docs/a3-v2-readiness/rehearsal/closed_store_state.py` (`closed_store_state_08.txt`): one
  `version` row per shard, `2025.1.1`; no journal, WAL or shm file; no `replication_in_flight` row;
- `docs/datastore-integrity-audit/replicated_serial_census.py` (`census_08.txt`): verdict
  **`no divergence`**;
- `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` (`store_soundness_08.txt`): `=> clean`,
  which is `PRAGMA integrity_check` `['ok']` and `foreign_key_check` 0 rows on the primary and every
  shard, and `read_inventory` over shards 0–3 with no problem in any class and 0 `duplicate`; its
  counts equal the Gate's;
- the store files byte-identical before and after (`state/store_files_after_checks.txt`).

**The readers.** `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh
var/datastores/handover-A3-v2-lambdacdm.sqlite var/handover-08/extract`, all six with
`--ray-address local --models LambdaCDM`, then `extract_Gk_data` once with `nomodels`, as R4 did
(`readers_08.txt`). Every reader exits 0 and leaves every store file as it was; no Ray process is
left after each. The `QuadSourceIntegral` extractor finds every stored row of each triple it draws.
Record each reader's wall time and output size: this store is about eight times R4's. Afterwards
`store fingerprint` (without `--write`) equals the sidecar's. **If `var/handover-08/extract` would
exceed 20 GB, stop the reader that is running, by its own Ctrl-C equivalent `kill -INT <pid>` of that
reader, and ask** — the volume had 120 GB free.

---

## 4. The depth census — whether `rtol = 1e-7` removed the cliff

**Method: the A3 baseline's, reproduced exactly.** The baseline's figures were an ad-hoc read of
its stored instrumentation (`9c8e608`; this board, the head of
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`); no script was kept, and
the store was retired on 2026-10-04, so it cannot be re-read. Write the script that reproduces the
method, `scripts/depth_census.py`, read-only, no Ray, every file `mode=ro`, following `report()`:

- **Population.** Every `QuadSourceIntegral` row on the four shards with `atol_serial`/`rtol_serial`
  resolving (through `tolerance.log10_tol`, which is log₁₀ of the value) to **(1e-32, 1e-7)**.
  Report the count; it must be 9280, and any row at another pair is listed.
- **Bins on `WKB_Levin_max_depth`**: `NULL` (no Levin call), 1–19, and exactly 20
  (`DEFAULT_LEVIN_MAX_DEPTH`; read the constant from `AdaptiveLevin/levin_quadrature.py` and print
  it). For each: rows, mean and total `compute_time`, and the count with `total_converged = 0`.
  Also the maximum `WKB_Levin_max_depth` and `WKB_Levin_num_regions`, and `SUM(WKB_Levin_elapsed)`
  against `SUM(compute_time)` and `SUM(numeric_quad_compute_time)`.
- **CPU-hours** = `SUM(compute_time)/3600`, for the store and per bin.
- **Unconverged** = `total_converged = 0`, store-wide and within the depth-20 bin. Also
  `total_phase_limited = 1`, store-wide.
- **Rows over 60 s**: count, and the fraction unconverged.
- **Bands** (`QUADSOURCE-TOLERANCE-SWEEP.md:246-250`): seam $z_{\rm response} > 10^5$, mid
  $9 \le z \le 10^5$, low $z < 9$, each with its row count (6080 / 2112 / 1088 expected) and every
  figure above.
- **The rows at depth 20, and the 20 most expensive rows**: $(k, q, r)$ in /Mpc, $z_{\rm response}$,
  `compute_time`, `WKB_Levin_num_regions`, `total_converged`, `total_abserr/|total|`.

**The comparison.** Print, beside v2's store-wide figures, the baseline's: 7600 rows at
(1e-32, **1e-8**); NULL 2489 at 0.062 s, 1–19 3054 at 0.776 s, **20: 2057 at 170.6 s**; 98.2
CPU-hr, 97.5 of it at depth 20; **889** unconverged at depth 20; 195 rows over 60 s, 74.4 %
unconverged; up to 1,530,190 regions. **State plainly that the comparison is not like-for-like**:
the pair differs (`rtol` 1e-8 → 1e-7, which is the change under test); the baseline lacked 1680
items, all at $z_{\rm response} \le 174.1$, including every low-band item; and no per-band baseline
figure exists. The one population both stores share completely is $z_{\rm response} > 174.1$:
report v2's figures there separately, as the closest like-for-like set, without a baseline number
beside them, because none was recorded.

Save the script's output as `out/depth_census_08.txt`. Its test, before the real store: run it on
R4's `var/datastores/handover-06-reference.sqlite`, where it must count **864** QSI rows, all at
(1e-32, 1e-7) (`docs/handover/A3-V2-VERIFICATION.md` §1.3.5), and change no file. Save
that as `out/depth_census_r4_08.txt`.

**Two more measurements, from the run's own logs**, saved as `out/run_logs_08.txt`:
- **The tail.** From `stdout.log`'s `RayWorkPool` progress lines for `CALCULATE QUADRATIC SOURCE
  INTEGRALS`, the time at which fewer than six items remained in flight, and the wall time from
  then to the stage's end, against the stage's total. The session reported about two hours of one
  worker on the last ~52 items. Say which items they were, from §4's table if `compute_time`
  identifies them.
- **The `IntegrationWarning`s** in `stderr.log` (13 on 2026-10-25: five "probably divergent, or
  slowly convergent", five "maximum number of subdivisions (100)", three roundoff): each with its
  Ray task or worker, the stage, and the `QuadSource` row it belongs to where the log says.
  Nothing is to be recomputed.

Each of these that shows a defect is opened as a §3 issue, not fixed: a tail that holds more than
10 % of the stage's wall time is one (the work list's order), and a warning that can be tied to a
stored `QuadSource` row is another.

---

## 5. The run's work

`./venv/bin/python -m runregistrykit list` (five runs, the build `done`), and the build's wall time
and CPU time: the manifest's `created` to the final heartbeat, and `SUM(compute_time)` from §4
beside the 250 CPU-hr projection and the 39–44 h estimate. Save as `out/run_record_08.txt`.

---

## 6. The lock

After §3's readers, with `ray-processes` empty: `PYTHONPATH=. ./venv/bin/python -m
tools.rehearsal_lock release`. Its owner pid is dead and its project is this one, so it releases;
if it refuses, record it and stop. Then `status` (no lock) and `ray-processes` (nothing). Save
every command and exit status as `out/lock_08.txt`. **The freeze on the checkout ends here**; it is
not moved by this prompt.

---

## 7. The records, and what this prompt does not do

**On the Mac Studio, under `var/handover-08/` only:** `out/`, `scripts/` (`depth_census.py` and
any helper), `state/`; `log.md`, in `qcd-background-audit` README §5.1's form, classifying every
deviation; `verification.md`, **§3 "After the build"** of `docs/handover/A3-V2-VERIFICATION.md`;
`issues.md`, a board §3 entry for each issue opened. `extract/` stays on the Mac Studio.

**The orchestrator** copies all but `extract/` to `docs/handover/a3-v2/after/`, the log to
`logs/08-after-the-a3-v2-build.md`, and §3 into the verification document; updates this board and
`docs/OPEN_ISSUES.md`; and makes **one commit**. Whether
`[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` (its surviving question: "only if
the adopted pair still reaches the depth cap") or
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` narrows or closes on
§4's numbers is the orchestrator's to propose and the user's to decide; this prompt states the
numbers and does not touch the board.

**It does not:** edit any tracked file; recompute, prune, `--drop` or resume anything; retire any
store (R4's two are the user's); copy the store elsewhere; move the checkout; or fix what it finds.

## 8. Stop conditions — stop and ask

- Any Gate item is not as stated, or the checkout is not clean at `b90752d`.
- `store fingerprint --write` gives a digest other than `0e32f24a…`, or any store file changes
  under §2, §3 or §4.
- A §3 check fails: a divergence, an `integrity_check` other than `ok`, a foreign-key row, a
  journal, a `replication_in_flight` row, a reader that exits non-zero or writes to the store.
- §4's population is not 9280 rows at (1e-32, 1e-7).
- The lock's `release` refuses, or a Ray process of this checkout is alive before it.
- A step would need anything outside `var/handover-08/` and the lock written.
