# Log — prompt 00: audit the readiness defects

**Subject:** *Audit what the A3 v2 readiness prompts rest on* · **Model:** Claude Opus 5.5 ·
**Date:** 2026-10-02 · **Base:** `5707851`, clean · **Result:** landed, read-only. No production
file, driver or test changed. The audit is
[`docs/a3-v2-readiness-audit.md`](../../../docs/a3-v2-readiness-audit.md), and its probes and their
outputs are under [`docs/a3-v2-readiness/`](../../../docs/a3-v2-readiness/). No stop condition of
the prompt's §5 was met.

## What shipped

- `docs/a3-v2-readiness-audit.md`: sections R, S, V and M, each finding marked measured, read or
  inferred; the issues opened and changed; and the closing section, **What 02, 03 and 04 can now be
  written against**.
- Five probes under `docs/a3-v2-readiness/`, each `black`-clean, each with its output committed beside
  it:

  | Probe | Items | Ray | Output(s) |
  |---|---|---|---|
  | `reader_writes_probe.py` | R2, R3 | no | `reader_writes_probe.txt` |
  | `readonly_open_probe.py` | M1, M2, V2 | no | `readonly_open_probe.txt` |
  | `version_window_probe.py` | V1 | no | `version_window_probe.txt` |
  | `sigterm_probe.py` | S1, S2, S3 | yes | `sigterm_probe_S1.txt`, `_S2.txt`, `_S3.txt` |
  | `fingerprint_time_probe.py` | S4 | no | `fingerprint_time_probe.txt` |

- This campaign's board: §1's row for 00; items R1–M2; two user decisions put by the audit (U1, U2);
  two issues opened in §3.2; and §5.1.
- On the `datastore-integrity` board, one dated line each, added and nothing rewritten:
  - **Widened (2026-10-02)** on `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]`;
  - **Widened (2026-10-02)** on `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`.
- `docs/OPEN_ISSUES.md`: those two hooks in §1.17, two rows added for the issues opened, and the
  header at **106 open** (104 + 2).

## Deviations from the prompt

1. **One script for S1–S3, one output per item.** `sigterm_probe.py S1|S2|S3` runs a child per
   variant, and writes `sigterm_probe_S1.txt` and so on. The prompt asked for probes in the plural.
   — IMPLEMENTATION CHOICE: the three share their child and their process bookkeeping, and one copy
   of that bookkeeping is easier to review than three.
2. **S1 measures more than the prompt's comparison.** Besides a handler raising `SystemExit` it
   measures one raising `KeyboardInterrupt`, one raising another `BaseException`, one that returns,
   and one blocked in `ray.get`. — IMPLEMENTATION CHOICE: "what exception reaches the frame"
   turned out to depend on what the handler raises, and 04 needs to know which exception survives.
3. **S3 measures five ways of ending, with repeats**, not two: the abort; `os._exit` at once and
   3 s after the signal; `sys.exit` with Ray's `atexit` shutdown; and an explicit `ray.shutdown()`.
   — IMPLEMENTATION CHOICE: the first two trial runs gave the abort and `os._exit` different
   outcomes, so a single run of each could not answer "the abort alone, or any exit without
   `ray.shutdown()`".
4. **V1 is measured, not only read**, and its probe adds a fourth case, the prune at open.
   — IMPLEMENTATION CHOICE. Log 02's demonstration of the window was a scratch script that was not
   committed. The prune case found the sibling defect opened below.
5. **R2's transcription of the reader.** Where the script draws a random subsample, the probe takes
   the first element. For `ModelProxy`/`GkSourceProxy` it passes a stand-in `Proxy` with the same
   attributes and no `ray.put`. — STRUCTURALLY REQUIRED: the probe must not need Ray (README §5 rule
   9), and the script cannot be imported (it calls `ray.init` at module scope).
6. **R2 also counts the four sharded lookups only the other readers make** (`GkNumericIntegration`,
   `GkWKBIntegration`, `TkNumericIntegration`, `TkWKBIntegration`), as misses. — IMPLEMENTATION
   CHOICE: it puts the "no sharded write" finding of R1 on a measurement for every reader's sharded
   class, not only QSI's. The tolerance objects it needs are built from the store's serials, not
   looked up, so that those steps count only the sharded lookups.
7. **M2 patches `ShardedPool._create_engine` and `Datastore._create_engine`** to open `mode=ro`, and
   for M2b–c replaces `_replicated_write` with a one-shard call that makes no record. — STRUCTURALLY
   REQUIRED: no read-only mode exists yet, and the prompt asks whether an actor can serve every
   lookup from a `mode=ro` engine. The patches live in the probe and are scoped by `mock.patch`.
8. **S4 calls `fingerprint_store(..., write=False)` twice**, a first call and a warm one, each timed
   alone. — IMPLEMENTATION CHOICE. Both calls are the one read README §3 permits.
9. **The suites' counts were read from the full output in a file**, not through `| tail -40`.
   — IMPLEMENTATION CHOICE: for `AdaptiveLevin`, `| tail -40` showed only the suite's own printed
   lines and no verdict (observation 1).

No deviation is UNINTENDED DRIFT.

## Verification performed

**Machine and versions.** macOS, Darwin 25.5.0, arm64, 10 CPUs; Python 3.12.14; SQLite 3.53.4;
**Ray 2.43.0** (`ray --version` equivalent: `ray.__version__` printed at the head of each S output;
`--ray-commit=ecdcdc6a6e63dc4bcd6ea16aae256ce4d32a7e2c` in each `gcs_server`'s command line).
`RAY_ADDRESS` unset; no Ray process existed before the first probe.

**Discovery.** `python -m RunRegistry list` at the start (12:25) and again before S4 (13:01): 12
runs, 10 finished, 2 unknown (`a3-pilot`, `a3-pilot-resume`), nothing `running`.

**Probe wall times and load.**

| Probe | Wall | Load average around it |
|---|---|---|
| `reader_writes_probe.py` | 27.3 s | 9.7 (13:04) |
| `readonly_open_probe.py` | 8.9 s | 15.4 (13:05) |
| `version_window_probe.py` | 1.9 s | — |
| `sigterm_probe.py S1` | 186.6 s (12:50–12:53) | 12.6 → 7.4; each variant's is printed |
| `sigterm_probe.py S2` | 114.1 s (12:53–12:55) | 7.7 → 6.9 |
| `sigterm_probe.py S3` | 219.1 s (12:56–13:00) | 9.3 → 8.6 |
| `fingerprint_time_probe.py` | 3.9 s for the process; the calls 2.480 s and 0.683 s | 5.1 |

**`var/`.** Only `var/datastores/handover-A3-baseline-lambdacdm.sqlite`, its four shards and its
sidecar were read, by `fingerprint_store(primary, write=False)` alone. SHA-256 before and after, from
`fingerprint_time_probe.txt`, all `unchanged`:

```
fdbe93a541083e8f4c9a1e4aa392b565a97046d7404400af60e589e213e7376b  handover-A3-baseline-lambdacdm.sqlite
a10f3c3ce33ad0563b6479a3443e7f1e4159e35c4dacb4134e2933de9e5a1df6  handover-A3-baseline-lambdacdm-shard0000.sqlite
9a5a4827a996786f84146dfeb3e454e34b90a2097e1be7decf8109d6e06f92f4  handover-A3-baseline-lambdacdm-shard0001.sqlite
d196e8a34167aa3f4514c00ed51385af678b5da90de01e5aa6365d8d658ac7fe  handover-A3-baseline-lambdacdm-shard0002.sqlite
13398fec0071e0d3f3250f5102dd7ba56ed3dfeab1c024c24649be9bffeb2083  handover-A3-baseline-lambdacdm-shard0003.sqlite
3e43aee7aa17b02f2d2f7041a382a520a9ef5693553b0df40aa222ae8a3a7eb3  handover-A3-baseline-lambdacdm.manifest.json
```

HEAD's fingerprint did not refuse the format-1 store, so log 05's 6.66 s is quoted as the earlier
figure, not as a substitute.

**Processes.** Each S variant records its own descendants and kills only its own survivors, `kill`
then `kill -9`; each output ends with the Ray processes before and after and the pids killed. Every
kill was of a `gcs_server` the probe's own child had started, and each was gone after its `SIGKILL`:

| Output | Pids killed (`SIGTERM`, then `SIGKILL`) |
|---|---|
| `sigterm_probe_S1.txt` | none |
| `sigterm_probe_S2.txt` | 35008, 36025 |
| `sigterm_probe_S3.txt` | 38432, 38923, 39209, 40655, 41570 |

Trial runs of S1 and S3, made while the probe was written, also ran under the same runner, with
their outputs in the session scratch area. Their kills were also of their own `gcs_server`s only:
23258 in the first S3 trial, and 27017, 28122, 28452, 29484 and 29773 in the second. `pgrep -lf
'gcs_server|raylet|ray::'` was empty before the first probe, after every trial and every committed
run, and at the end (below).

**Tests.** The six suites, serially, at the end, each to a file (deviation 9): AdaptiveLevin 32 OK,
ComputeTargets 665 OK, CosmologyModels 39 OK, Datastore 551 OK, LiouvilleGreen 148 OK (1 skipped),
RunRegistry 199 OK, from 13:12 to 13:19 under load averages of 10 to 27. Every count is unchanged
from the baselines at dispatch, and none of the known load flakes fired.

**Formatting.** `./venv/bin/python -m black --check docs/a3-v2-readiness/*.py`: 5 files unchanged.

## The deliberate-breakage record

An audit makes no mutation. Its counterpart is that each instrument was first shown to count a
known write (A2):

- **R2's instruments** (`reader_writes_probe.txt`, `## Run 0`). A deliberate miss,
  `object_get("tolerance", tol=1e-99)`, gave `shard000N: 1 commit(s); rows tolerance +1` on each
  of the three shards, `primary: 3 commit(s)`, and `get tolerance x1 (1 inserted …)` from the
  `_replicated_write` wrapper. Repeating it as a hit gave `primary: 2 commit(s)` and nothing on the
  shards. A deliberate sharded `object_store` gave `shard0002: 1 commit(s); rows GkSourcePolicyData
  +1`. Each run also prints a whole-run delta, open to close, taken independently of the per-step
  deltas, which agrees with their totals.
- **M1's instrument** is the same file-state delta. Each write at open it reports — the prune, the
  drops, the in-flight table's DDL, a shard's missing tables, a new version label, a new store — is
  a known write, and each was counted, so the clean open's `no file changed` is a measured zero.
- **M2's instruments.** M2a and M2c count the attempted write as an exception raised by SQLite on a
  `mode=ro` file. M2b's `no file changed` uses R2's instrument, already shown above to count. M2d
  shows the one write the change counter does not see: a hot-journal rollback leaves `counter 191 ->
  191` but `changed`, so the SHA-256 is what catches it.
- **V1's instrument** reads each shard's `version` rows `mode=ro` before and after each crash.

## Observations not acted on

1. **README §7's suite command can hide the verdict.** With `2>&1 | tail -40`, unittest's summary
   (stderr, unbuffered) is written before the suite's buffered stdout, which is flushed at exit. So
   `AdaptiveLevin`'s last 40 lines were its own `-- region:` prints, with no `Ran`/`OK`. Writing the
   output to a file and grepping for `^Ran`/`^OK` gives the verdict. Not opened as an issue: the
   instruction belongs to the campaign plan, and this note is for the orchestrator.
2. **`sqla_version_factory.build` returns a `store_tag` object**, carrying the version row's serial
   and label (read: `Datastore/SQL/ObjectFactories/version.py`). Every consumer reads only
   `store_id`, so nothing goes wrong today. Not opened; 02 will touch this code.
3. **A reader given a path where no store exists creates one** (M1). Not opened separately: 03's
   read-only pool refuses the new-store branch (M1's table, first row), which closes it.
4. **`sqla_wavenumber_factory.build` and `sqla_redshift_factory.build` write on a hit** (they update
   the `source`/`response` flags). No reader reaches either. Recorded in M1 for 03, not opened.
5. **During the S probes, each raylet warned that `/tmp/ray/session_…` is "over 95% full"** (33 GB
   free of 926 GB). This is the machine's state, not a probe's.
6. **The serials in the stand-in outputs vary from run to run** (broker lease batches). The
   committed outputs are the runs this log cites; a re-run reproduces every count and every
   exception, but not necessarily every serial.

## State handed to the next prompt

- 02, 03 and 04 stay **held**. The audit's closing section is what the orchestrator writes them
  from. Two decisions wait on the user, each with a recommendation: **U1**, whether 02 also puts the
  prune at open of a replicated row through a record (recommended yes); **U2**, what is done about
  a new store's first-open window (recommended: record only).
- Two issues were opened on this board's §3.2. Neither blocks A3 v2.
- No store under `var/` was written. No Ray process is left (`pgrep -lf 'gcs_server|raylet|ray::'`
  empty at the end).
- Baselines after this commit: board §5.1.
