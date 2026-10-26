## §5 — rehearsals R11 and R12 on the Mac Studio

**Prompt:** [`05a-reanchor-t-phase.md`](../05a-reanchor-t-phase.md) §5, under the orchestrator's
Part B dispatch, its §B.1 and §B.5 (§B.5 governing).
**Host:** mps026298, the Mac Studio.
- The main checkout was at the dispatch SHA `149c393`. Its code is the code commit `1fc349c`'s, up
  to `bd38e50`'s comment-only changes.
- The worktree `SecondaryGWKit-lpf05a-c4eaefd` was detached at `c4eaefd`, the code commit's parent.

**Model:** Claude Opus 5.5, as the Mac Studio orchestrator's rehearsal subagent, in three turns:
- the gate, the lock and R11's launch;
- R11's checks and R12's launch;
- R12's checks, the comparison, the classification and tabulation, the readers, the release and
  these records.

**Date:** 2026-10-26, 12:20–12:33 UTC.
**Commit:** none. The records are committed on the laptop in 05a's records commit.
**Result:** COMPLETE WITH DEVIATIONS. None of the deviations stopped the work, and none changed a
computation.
- R11 and R12 both ended `done` with exit 0 and fingerprint `1b332ffa…`. Each is equal to R8's
  class by class, and R12's `classes` object equals R11's.
- Every value table of every class is bit-identical between them: 7 tables, 360,462 rows.
- Every row-column difference outside `QuadSourceIntegral` is one of the four volatile kinds. 0 are
  unclassified, and no row is in one store only.
- `QuadSourceIntegral` differs:
  - `total` moved in 424 of 864 rows, all through `WKB_Levin`;
  - $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}\rvert$ is at most 2.9e-11;
  - $\lvert\Delta{\rm total}\rvert/$`total_abserr` is at most 8.8e-5;
  - no depth, region count or convergence flag changed;
  - the `handover_continuity` metadata moved in every row.
- The readers exited 0, and the stores were left byte-identical.
- The lock was released.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§5a "Layer 1(a) for $T$".

### What shipped

Nothing in the repository: no code, and no commit. Everything was written on the Mac Studio, in
the main checkout:
- **The records directory** `var/levin-phase-floor-05a/rehearsal/`, holding:
  - `out/`;
  - `scripts/`: 05's `classify_non_qsi.py` and `tabulate_qsi.py`, unchanged;
  - `state/`;
  - `extract/`, the readers' output, 177 MB, not for copying back;
  - `log.md`, `verification.md` and `issues.md`.
- **The stores** `var/datastores/levin-phase-floor-05a-r11*.sqlite` and `-r12*.sqlite`, five files
  each.
- **The runs** `levin-phase-floor-05a-r11-parent-20261026T122041` and
  `levin-phase-floor-05a-r12-reanchor-t-20261026T122434`, with their launch logs
  `var/runs/levin-phase-floor-05a-r{11,12}.launch.log`.

The stores are kept, as prompt §5.7 says.

### Predictions and verdicts

§5 fixes no scored predictions; §6 holds them. The acceptance checks of §7.2, and the dispatch's
own:

| check | number | verdict |
|---|---|---|
| R11 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 class digests = R8's | holds |
| R12 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 = R8's; `classes` object = R11's | holds |
| every value table outside `QuadSourceIntegral` identical | all 7 value tables bit-identical (360,462 rows) | holds |
| every other difference there of the four kinds; no row in one store only | 0 unclassified; 0 only-A, only-B or unmatched in any class | holds |
| `QuadSourceIntegral` differences tabulated | 424/864 `total`s; max rel 2.9e-11; max \|Δ\|/abserr 8.8e-5 | done |
| the readers exit 0 | 6 × exit 0; store byte-identical | holds |
| the lock is released | `release` exit 0; `status` free; `ray-processes` none | holds |
| (dispatch) R11 ≠ R12 in `QuadSourceIntegral` | not value-identical, so each tree ran its own code | holds |

### Deviations from the prompt

#### 1. R11 ran from a worktree, on the main checkout's venv — STRUCTURALLY REQUIRED

Prompt §5.2 calls for a worktree, and the dispatch gave it. The worktree has no venv, so R11 used
the main checkout's `venv/bin/python` by absolute path, with its cwd and `PYTHONPATH` set to the
worktree. Its `--database` and launch log were absolute paths into the main checkout. As for any
worktree run, the registry recorded it under the main checkout's `var/runs/`.

The evidence that R11 ran the worktree's code:
- the manifest records `git_head` `c4eaefda…`, `git_dirty` false, and the worktree's launcher;
- `lsof` showed that the driver's cwd, and that of the `Datastore` actor 22808, was the worktree;
- R11 and R12 differ in `QuadSourceIntegral`.

#### 2. `PYTHONDONTWRITEBYTECODE=1` on R11 only — IMPLEMENTATION CHOICE

It keeps `__pycache__` out of the worktree, which must not be written, and it changes no
computation. This follows R9 in 05. R12 used R10's launch form. After both runs the worktree's
`git status --short --ignored` was empty, and `find -newer state/r11_launch_marker` listed
nothing.

#### 3. R12's `--database` and launch log were given as absolute paths — IMPLEMENTATION CHOICE

R10 in 05 gave both as paths relative to the main checkout. The dispatch asked for absolute paths
for both stores. Both forms name the same files from the main checkout's cwd. The manifest's
`results` is relative either way.

#### 4. A shared `--job-name levin-phase-floor-05a` — IMPLEMENTATION CHOICE

This is the dispatch's name, as 05 shared `levin-phase-floor-05`. The job name reaches only
`label`, which differs anyway by its timestamp.

#### 5. A text form of the comparison, with `--max-rows 1000000` — IMPLEMENTATION CHOICE

This was run beside the `--json` form that §5.4 asks for, as 05 did. The JSON carries every row
whatever that setting is.

#### 6. `/usr/bin/time -p` wrapped the comparison — IMPLEMENTATION CHOICE

Its three lines (`real`, `user`, `sys`) land in `out/store_value_diff.{json,txt}.stderr`, which
hold nothing else: the tool wrote nothing to stderr. 05's files have the same form.

#### 7. Two inline read-only Python summaries, not saved as scripts — IMPLEMENTATION CHOICE

The per-class summary `out/store_value_diff_summary.txt` reads only the JSON; its text is in
`out/commands.log` verbatim. `out/qsi_extra_counts.txt` reads only `out/qsi_rows.csv` and
`out/classify_non_qsi.txt`. Neither opens a store. 05's summary was made the same way. 05's other
two scripts, `gkwkb_reuse_rows.py` and `parent_serials.py`, were not re-run; §5 does not ask for
them.

#### 8. The two copied scripts' docstrings still name R9 and R10 — IMPLEMENTATION CHOICE

They are copied **unchanged**, as §5.4 says, and their SHA-256 equal 05's (`26690e61…`,
`b7508351…`). Both take their stores and JSON as arguments, and neither prints a store's name: the
outputs say A and B. **Here A = R11 (parent `c4eaefd`) and B = R12 (code `1fc349c`).** Neither
needed a change to read R11 and R12, and both exited 0 on their first run.

#### 9. The gate record lists this agent's own directory — UNINTENDED DRIFT

The command that wrote `out/gate.txt` began with `mkdir -p` of `var/levin-phase-floor-05a/rehearsal/`.
So its `ls var/levin-phase-floor-05a` shows `rehearsal`, and cannot show the directory's absence
before the run. The other gate lines are unaffected:
- `ls var/datastores | grep levin-phase-floor-05a` found nothing;
- the registry held 16 runs, all finished.

The orchestrator's §B.5 recorded at 12:12 GMT that the directory did not exist. Prompt §6's
`wider/` was not there at the gate either.

#### 10. The stores' fingerprints were not re-taken — IMPLEMENTATION CHOICE

Both fingerprints are the launcher's, taken at the end of each build and read from `status.json`.
- R11's five files are byte-identical from just after its build (12:24) to the end.
- R12's files were first hashed at 12:28, after its fingerprint and before any read. They are
  identical to the end.

### Verification performed

| check | result | where |
|---|---|---|
| gate | both trees at their SHAs and clean; 16 runs finished; lock free; no 05a store; the launcher `5b984c75…` in both trees | `out/gate.txt` |
| R11 | `done`, exit 0, pid 22684, `1b332ffa…`, `c4eaefda…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r11_end_check.txt`, `out/r11_status.json` |
| R12 | `done`, exit 0, pid 24897, `1b332ffa…`, `149c3938…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r12_end_check.txt`, `out/r12_status.json` |
| worktree unwritten | `status --ignored` empty; `find -newer` empty, after both runs and at the end | `out/r11_end_check.txt`, `out/r12_end_check.txt`, `out/final_check.txt` |
| scripts | the two copies, the tool and `run_readers.sh` all have 05's SHA-256 | `out/script_sha256.txt` |
| value tables | 7 tables, 360,462 rows, all bit-identical | `out/store_value_diff_summary.txt` |
| non-QSI row columns | 4 kinds, 0 unclassified | `out/classify_non_qsi.txt` |
| QSI tabulation | pairing = the tool's identities; 0 `total` mismatches against the tool | `out/qsi_tabulation.txt`, `out/qsi_rows.csv` |
| readers | 6 × exit 0; lookups equal 05's; store identical | `out/readers.txt` |
| stores | 10 files identical from before the comparison to the end | `state/store_sha256_before_diff.txt`, `state/stores_sha256_final.txt` |
| lock | acquire 0; set-pid 0 ×2; ray-processes 0 ×3; release 0; status 0 | `out/lock.log` |

### Observations not acted on

- **The change is small on this store.** 05's change to $G$ moved 542 `total`s by up to 4.0e-5
  relative and 28× `total_abserr`, and changed depths and regions. This change to $T$ moves 424
  `total`s by at most 2.9e-11 relative and 8.8e-5 × `total_abserr`. It changes no depth, region
  count or convergence flag.
  - By depth, the median $\lvert\Delta\rvert/$`total_abserr` is 2e-9 to 1.1e-8 at depths 2–8, and
    1.5e-6 to 1.7e-5 at depths 10–14.
  - This is recorded, not judged. §6 scores the effect on A3 v2's rows.
- **`handover_continuity` moves in every row,** even where nothing else does. Its keys
  `partition.handover.T{q,r}.handover_continuity[i].d{T,Tprime}_rel` differ in all 864 rows, as
  §B.5 expected. In 292 rows they are the only non-wall-clock keys that differ: the 291 with no
  WKB part, and one at depth 0. This is a diagnostic, not a value that enters `total`.
- **26 rows have `total_abserr` = inf in both stores,** at depths 0–8. 12 of their `total`s differ,
  by 1.1e-16 to 1.0e-15 relative. The tabulation prints their ratio to `total_abserr` as 0. This is the open issue
  `[04-roundoff-floor-can-be-infinite]`, and it is not this prompt's.
- **The GkWKB residual-memo counters** differ in 31 rows here and in 25 rows in 05's pair. This is
  further circumstantial support for 05's P2: the counters depend on scheduling. No same-code pair
  exists to prove it.
- **The readers' file counts differ from 05's** (Gk 387 against 450, GkSource 162 against 142, QSI
  73 against 74). The lookups are equal. Each reader draws an unseeded `random.sample`: the open
  issue `[10-the-readers-draw-their-work-lists-unseeded]`.
- **`[05-store-value-diff-compares-volatile-columns]`** recurs exactly as 05 found it. It is
  already open.
- **Prompt §6's run `levin-phase-floor-05a-wider-20261026T123047`** started at 12:30:47 and was
  `done` (109/109) by the final check. It is not this agent's, and it was not touched.
- **The size of `out/`.** `store_value_diff.json` is 41 MB and `.txt` 10 MB, as in 05. The
  orchestrator decides whether to copy them back whole. `store_value_diff_summary.txt`,
  `classify_non_qsi.txt` and `qsi_rows.csv` carry what the records cite.

### State handed on

- The lock is free: `status` reports the lock directory does not exist, and there are no Ray
  processes of `/tmp/ray-sgk`. `~/.raypoolkit/` is empty.
- The registry holds 19 runs, all finished: R11 and R12 `done`, and §6's `wider` `done`.
- **R11's store** has primary `85538005…` and **R12's** has primary `94dae1cc…`. Both are in
  `var/datastores/`, both carry fingerprint `1b332ffa…`, and neither has a sidecar. They are kept.
- The main checkout is at `149c393` and clean. The worktree is at `c4eaefd`, clean and unwritten.
- The records' links assume they are copied to `docs/levin-phase-floor/05a/rehearsal/`.
