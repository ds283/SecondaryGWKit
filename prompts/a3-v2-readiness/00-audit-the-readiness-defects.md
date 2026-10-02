# Prompt 00 — audit the readiness defects

**Campaign:** [`README.md`](README.md) · **Board items:** **R1**–**R3**, **S1**–**S4**, **V1**–**V2**,
**M1**–**M2** · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none; `git status` must be clean. `python -m RunRegistry list` shows nothing `running`.
**Closes:** nothing. **Narrows or widens:** `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]`
(expected to widen, §2 R) and `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`
(its inferred mechanism, §2 S), each on the `datastore-integrity` board with its index hook.
**Opens:** whatever it finds that §0 of the README does not list.
**Recommended model:** **Opus**. Three of the measurements decide a design, and two of the
mechanisms are inferred from a shared object's strings, not from code.

**Read first:**

1. [`README.md`](README.md): §0, §1, §2, §3, §4 and §5.
2. The board entries for the four issues 02–04 will close, on
   `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` §3.3:
   `[08-extraction-script-object-get-…]`, `[02-the-version-row-…]`, `[10-a-sigterm-…]` and
   `[10-extraction-scripts-exit-1-…]`.
3. `docs/datastore-integrity-verification.md` §2.1.1, §2.3 and §2.6, and its probes
   `docs/datastore-integrity-verification/sigterm_in_ray_wait_probe.py`,
   `sigterm_alarm_probe.py` and `sigterm_probes.txt`.
4. `Datastore/SQL/ShardedPool.py`: `__init__`, `_reconcile_replicated_tables`, `object_get`,
   `_get_impl_replicated_table`, `_replicated_write`. `Datastore/SQL/Datastore.py`: `__init__`,
   `_ensure_tables`, `_validate_on_startup`, `_drop_actions`, the prune path, `object_get`,
   `_insert`. Cite both files by function name, never by line.
5. `Datastore/store_reader.py` in full: the read-only opener the fingerprint uses.
6. The six `extract_*.py` scripts' setup up to their first value read, and
   `config/model_list.build_model_list`.
7. `docs/gktk-remedial/scoped_pipeline_run.py` `terminal_state`, `register`, `main`; and
   `docs/handover/quadsource_atol_sweep.py` `run_build`.

---

## 1. What is wanted

Three of the held prompts rest on claims nobody has measured. 03 assumes it knows every write a
reader makes; the issue it closes was recorded for one call in two scripts, and is wider. 04
assumes a handler can be put back after `ray.init` and survive `ray.wait`, and that Ray's abort
about 3 s after a `SIGTERM` is an `alarm` that can be cancelled; prompt 10 inferred the first from
behaviour and the second from `_raylet.so`'s strings. 02 assumes a read-only open will be able to
find the version row. This prompt measures each, and changes no code.

---

## 2. What to establish

Every finding states how it was established: **measured** (with the command and its output),
**read** (with the function), or **inferred** (with what it was inferred from). Nothing is
promoted from inferred to measured by argument.

**R — every write a reader makes.**
- **R1.** A table of every `pool.` call each of the six scripts makes before and during its value
  reads, plus `build_model_list` and every helper they call (`create_k_exit_work` and the like):
  the class, whether it is in `config/sharding.py`'s replicated list, the factory method it
  reaches, and whether that method **inserts on a miss**. Read from the code.
- **R2.** Measure on a stand-in pool (README §5 rule 9), with no Ray: run the lookup sequence one
  script makes, against a store that holds every row it asks for, and count what is written to
  each file — rows inserted per table, and `replication_in_flight` commits on the primary. Then
  against a store missing one row of each replicated class the script asks for. The instrument is
  the probe's choice (a wrapped `_replicated_write`, `sqlite3`'s `total_changes`, a row count per
  table before and after); it must be shown to count a known write before it is trusted to count
  zero.
- **R3.** From the code, not a run: does A3 v2's store hold a `QCD_Cosmology` row? (`run_build`'s
  `_filtered_build` calls the real `build_model_list` before filtering.) And what does each
  script do, call by call, on a store that lacks a model's `BackgroundModel` — which is where
  `[10-extraction-scripts-exit-1-…]` exits.

**S — the `SIGTERM` path.** Probes under `docs/a3-v2-readiness/`, each run by hand with a real
local Ray instance, its output committed beside it as `<probe>.txt`. After every probe, show that no
`gcs_server`, `raylet` or Ray worker it started is still running (`pgrep -lf` is acceptable *here*,
because the question is whether a process exists, not whether a run is alive), and kill any that
is, with the pid in the log.
- **S1.** A Python `SIGTERM` handler installed **after** `ray.init`: is it the handler that runs
  when the signal arrives during `ray.wait`? What exception, with what code, reaches the frame
  around `ray.wait`? Compare with `sigterm_in_ray_wait_probe.py`, which keeps Ray's.
- **S2.** Ray's abort: identify what kills the process about 3 s after the signal. Ray's version,
  the source of the `*** SIGTERM received` handler if it can be found (installed package source or
  the release's published source; never a download outside `pip`'s own cache without asking), and
  whether `signal.alarm(0)` from the Python handler stops it. Measure: a cleanup that sleeps 8 s,
  with and without `alarm(0)`.
- **S3.** What leaves the `gcs_server` behind: the abort alone, or any exit without
  `ray.shutdown()`? Measure both.
- **S4.** How long `Run.finish(..., fingerprint=True)` takes on a real-sized store. Time the
  fingerprint read alone on the A3 baseline, read-only, through the function `finish` uses
  (README §3 permits this read and no other). Record the size of the store and the time.

**V — the version row.**
- **V1.** The order of events in `ShardedPool.__init__` today: the file check, the check at open,
  the first actor and its version insert, the later actors and theirs. Where exactly a crash
  leaves a version row on some shards and not others, and what the next open does with it.
- **V2.** For 02: can the version row be written through `_replicated_write` after the actors
  exist, given that each actor's constructor currently needs the serial? State each form that
  works, with what it changes in `Datastore.__init__`, and recommend one. And: how a read-only open
  (§2 M) finds the row by label without inserting it, and what it should do when the row is
  absent.

**M — what a read-only pool needs.**
- **M1.** Every write a `Datastore` actor and `ShardedPool.__init__` make at open, before any
  lookup: DDL, `_ensure_tables`, `_validate_on_startup`, `_drop_actions`, the prune, the version
  insert, the check at open's repair, the hot-journal rollback, any serial manager. For each: skip
  it, refuse, or keep, under `read_only=True`, with the reason.
- **M2.** Whether an actor can serve every lookup the six scripts make from an engine opened
  `mode=ro` (`store_reader.read_only_url`), and whether a store with a journal beside it is refused
  the way `store_reader` refuses one. Measure on a stand-in: open a closed store `mode=ro`, run R2's
  sequence, show it succeeds on a full store and that an attempted insert raises from SQLite.

---

## 3. Verification

1. Every R2 and M2 count is reproduced by the committed probe, and each probe shows its
   instrument counting a known write first.
2. Every S probe's output is committed, with the Ray version and the no-orphan check after it.
3. The six suites at `HEAD`, unchanged: this prompt adds no test.

---

## 4. Acceptance

1. `docs/a3-v2-readiness-audit.md`, with sections R, S, V and M, each finding marked measured,
   read or inferred, and a closing section **What 02, 03 and 04 can now be written against**: for
   each, the facts its method rests on and any decision the user must take first, each with a
   recommendation (README §6.2).
2. The probes and their outputs under `docs/a3-v2-readiness/`.
3. On the `datastore-integrity` board: a **Widened (2026-10-02)** or **Narrowed** line on
   `[08-…]`, and on `[10-a-sigterm-…]` wherever S changes what is inferred; the index hooks in
   `docs/OPEN_ISSUES.md` §1.17 corrected to match.
4. Any new defect opened on this campaign's board §3, with an index row. If it blocks A3 v2, say so
   in its first line.
5. `black --check` clean on every new script. On this board: the §1 row, items R1–M2, §3.

---

## 5. Stop conditions — stop and ask the user

- A measurement would need to write anything under `var/`, or to open any store there other than
  the A3 baseline, read-only, for S4.
- A probe leaves a Ray process that `kill` and then `kill -9` do not remove.
- Ray's abort cannot be stopped by anything a Python handler can do. 04's charter assumes a
  terminal state can be written before the process dies; if it can only be written by racing the
  abort, that is the user's call.
- R finds a reader writing a **sharded** row (not only replicated ones) on a full store.

---

## 6. What this prompt does not do

- It changes no production file, no driver and no test.
- It writes none of 02–04; it states what they can be written against.
- It does not run a pipeline.

---

## 7. The log and the board

`logs/00-audit-the-readiness-defects.md`, using README §5.1. In addition: the Ray version, the
machine, and each probe's wall time.

`IMPLEMENTATION_STATE.md`: the §1 row for 00, items R1–M2, §3. Leave 02–04 held; their release is
the orchestrator's, once the user has read the audit.
