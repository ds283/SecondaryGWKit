## §5 — rehearsals R9 and R10 on the Mac Studio

**Prompt:** [`05-reanchor-g-phase.md`](../05-reanchor-g-phase.md) §5, under the orchestrator's
dispatch, §B.1 and §B.5.
**Host:** mps026298 (Mac Studio). The main checkout was at the dispatch SHA `53084dc`, whose code
is the code commit `120fdfe`'s. The worktree `SecondaryGWKit-lpf05-e648951` was detached at
`e648951`, the code commit's parent.
**Model:** Claude Opus 5.5, as the Mac Studio orchestrator's rehearsal subagent, in three turns:
- the gate, the lock and R9's launch;
- R9's checks and R10's launch;
- the comparison, the stop and question, then after the answer the tabulation, the readers and
  the release.

**Date:** 2026-10-26, 01:46–02:10 UTC.
**Commit:** none. The records are committed on the laptop in 05's records commit.
**Result:** COMPLETE WITH DEVIATIONS.
- R9 and R10 both ended `done` with exit 0 and fingerprint `1b332ffa…`, equal to R8's class by
  class.
- Every value table of every class is bit-identical between them.
- `QuadSourceIntegral` differs: 542 of 864 `total`s moved, all through `WKB_Levin`.
- The readers exited 0, and the stores were left byte-identical.
- The lock was released.
- §8's "any value outside QuadSourceIntegral differs" fired as written, on wall-clock, label,
  serial-digest and work-counter columns. The agent stopped and asked. The user ruled the prompt
  wrongly worded (deviation 1).

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md` §5
"Layer 1(a)".

### What shipped

Nothing in the repository: no code, and no commit. Written on the Mac Studio, all in the main
checkout:
- `var/levin-phase-floor-05/rehearsal/`, holding `out/`, `scripts/` (four read-only scripts),
  `state/`, `extract/` (the readers' output, not for copying back), `log.md`, `verification.md`
  and `issues.md`;
- the stores `var/datastores/levin-phase-floor-05-r9*.sqlite` and `-r10*.sqlite`, five files
  each;
- the runs `levin-phase-floor-05-r9-parent-20261026T014729` and
  `levin-phase-floor-05-r10-reanchor-20261026T015120`, and their launch logs
  `var/runs/levin-phase-floor-05-r{9,10}.launch.log`.

The stores are the user's to keep or retire.

### Predictions and verdicts

§5 fixes no scored predictions; §6 holds them. The acceptance checks of §7.2:

| check (§7.2) | number | verdict |
|---|---|---|
| R9 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 class digests = R8's | holds |
| R10 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`; `classes` object = R9's | holds |
| every non-`QuadSourceIntegral` value identical | every value table bit-identical; 0 rows in one store only, 0 unmatched. Row-column differences only of the four kinds the user ruled out of scope | holds, on the user's ruling (deviation 1) |
| `QuadSourceIntegral` differences tabulated | 542/864 `total`s; median \|Δ\|/abserr 8.7e-5, max 28.2 (one row) | done |
| the readers exit 0 | 6 × exit 0; store byte-identical | holds |
| the lock is released | `release` exit 0; `status` free, no Ray processes | holds |
| (dispatch) R9 ≠ R10 in `QuadSourceIntegral` | not value-identical, so each tree ran its own code | holds |

### Deviations from the prompt

#### 1. §8's stop on "any value outside QuadSourceIntegral" fired as written, and was ruled not to apply — STRUCTURALLY REQUIRED

`store_value_diff` reported eight classes besides `QuadSourceIntegral` as not value-identical. All
eight differ only in row columns that cannot agree between two builds of any code:
- wall-clock timings;
- `label`, which is the job name plus `datetime.now()`;
- `GkSource.parents_digest`. This is the SHA-256 of the compact JSON of the sorted (z_source
  serial, GkNumericIntegration serial or None, GkWKBIntegration serial or None) triples of the
  row's parent set (`ObjectFactories/GkSource.py` `GkSource_parents_digest`). It is a function of
  serials, and the serials of 1346 of 1368 numeric and 3466 of 3477 WKB rows differ between the
  stores;
- the GkWKB residual-reuse counters `rho_evals`, `rho_reused` and `stage_1_RHS_evaluations`, in 25
  rows.

Every value table of every class is bit-identical, as the tool's JSON shows.

The agent stopped with the lock held and asked. The user's decision, relayed verbatim: *"R9/R10
differences: Those columns cannot be expected to be identical. So I think that's just an
incorrectly-worded prompt. Do whatever you were already planning to do with the lock. Only this
project is running code on the Mac Studio right now."* The orchestrator's note sets the boundary:
these four kinds are not the stop, and any other kind outside `QuadSourceIntegral` would still be.
None occurred: `scripts/classify_non_qsi.py` leaves 0 differences unclassified. The stop lasted
from about 01:58 to 02:05 UTC.

#### 2. R9 ran from a worktree — STRUCTURALLY REQUIRED

Prompt §5.2 calls for a worktree, and the dispatch gave it. The worktree has no venv, so R9 used
the main checkout's `venv/bin/python` by absolute path, with cwd and `PYTHONPATH` set to the
worktree. Its `--database` and launch log were absolute paths into the main checkout, so nothing
was written in the worktree.
- The registry recorded the run under the main checkout's `var/runs/`, as a worktree run is.
- The manifest records `git_head` `e6489519…` and the worktree's script path.
- `lsof` showed the cwd of the driver and of a Ray actor to be the worktree.
- The QSI values differ from R10's, which confirms that each tree ran its own code.

#### 3. `PYTHONDONTWRITEBYTECODE=1` on R9 only — IMPLEMENTATION CHOICE

This keeps `__pycache__` from being written into the worktree, which the dispatch says must not be
written. It changes no computation. R10 used R8's launch as written. Afterwards the worktree's
`git status --short --ignored` was empty, and `find -newer` the launch marker listed nothing.

#### 4. A shared `--job-name levin-phase-floor-05` — IMPLEMENTATION CHOICE

R8 used `handover-12`. R9 and R10 use the same job name, so that it could not become a value
difference. The job name reaches only `label`, which differs anyway, by its timestamp.

#### 5. The comparison's text form was run with `--max-rows 1000000` — IMPLEMENTATION CHOICE

The default prints 20 rows per table. The JSON, which §5.4 asks for, carries every row whatever the
setting.

#### 6. Three read-only scripts beyond the tabulation — IMPLEMENTATION CHOICE

These were written to characterise deviation 1's differences for the question:
- `classify_non_qsi.py`;
- `gkwkb_reuse_rows.py`;
- `parent_serials.py`, run after the answer to state precisely what `parents_digest` digests, as
  the orchestrator asked.

Each opens the stores sqlite `mode=ro`. The ten store files were hashed before the comparison and
at the end, and are identical.

#### 7. The first tabulation run failed — IMPLEMENTATION CHOICE (a fix to the agent's own script)

`tabulate_qsi.py` raised `TypeError` summing `WKB_Levin_num_regions`, which is NULL on the 291
rows without a WKB part. It was fixed to count NULL as 0 and re-run, exit 0. Only the second run's
output is kept.

#### 8. A lock `status` look made inside a compound command — UNINTENDED DRIFT

While stopped, one `status --ray-temp-dir /tmp/ray-sgk` was run in a command with `diff` and `git
status`, and not by `>>`. It is a look; it changed nothing. It was repeated correctly as row 5 of
the lock transcript.

#### 9. A stray `touch -r … /dev/null` — UNINTENDED DRIFT

A final `find -newer` command carried a leftover `touch -r state/r9_launch_marker /dev/null`. It
failed with "Permission denied" and wrote nothing.

#### 10. The stores' fingerprint was not re-taken — IMPLEMENTATION CHOICE

The dispatch allowed `store fingerprint` without `--write` "if useful". It was not run. Both
fingerprints come from the launcher at the end of each build. R9's files are byte-identical from
just after its build to the end. R10's files had their first hash taken at the start of turn 3,
after its fingerprint and before any read, and are identical to the end.

### Verification performed

| check | result | where |
|---|---|---|
| gate | both trees at their SHAs, clean; 13 runs finished; lock free; no 05 store | `out/gate.txt` |
| R9 | `done`, exit 0, pid 47770, `1b332ffa…`, `e6489519…`, `git_dirty` false | `out/r9_end_check.txt` |
| R10 | `done`, exit 0, pid 49752, `1b332ffa…`, `53084dcb…`, `git_dirty` false | `out/r10_end_check.txt` |
| worktree unwritten | `status --ignored` empty; `find -newer` empty, after both runs | `out/r9_end_check.txt`, `out/r10_end_check.txt` |
| value tables | 7 tables, 360,462 rows, all bit-identical | `out/store_value_diff_summary.txt` |
| non-QSI row columns | 4 kinds, 0 unclassified | `out/classify_non_qsi.txt` |
| `parents_digest` | a digest of serials; recomputed 291/291 in each store | `out/parent_serials.txt` |
| QSI tabulation | pairing = tool's identities; 0 `total` mismatches against the tool | `out/qsi_tabulation.txt`, `out/qsi_rows.csv` |
| readers | 6 × exit 0; store identical | `out/readers.txt` |
| stores | 10 files identical from before the comparison to the end | `state/store_sha256_before_diff.txt`, `state/stores_sha256_final.txt` |
| lock | acquire 0, set-pid 0 ×2, ray-processes 0 ×2, status 0 ×2, release 0 | `out/lock.log` |

### Observations not acted on

- **The tool has no notion of non-value columns.** Proposed as
  `[05-store-value-diff-compares-volatile-columns]` in `issues.md`. The laptop opens the issue for
  the prompt's wording and the tool.
- **One row moves by 28× its declared error.** At
  $(k,q,r,z_{\rm resp}) = (10^6, 3.162\times10^5, 10^6, 329.64)$, $\lvert\Delta{\rm total}\rvert$
  is 28.2 × `total_abserr`$_A$ and 29.4 × `total_abserr`$_B$, a relative 4.0e-5. Both are
  unconverged, at depth 6 → 7. Every other row moves by less than its declared error under either store (the next largest is 0.60 × `total_abserr`$_A$). Recorded, not
  judged. §6's L3 is where the declared error's honesty is scored.
- **The residual-memo counters depend on scheduling.** That is inferred from the two-way swaps. No
  same-code pair exists to prove it (`issues.md` P2).
- **A `levin-phase-floor-05-wider` run was `alive` in the registry at the end.** It is §6's
  Ray-free run, which is not this agent's. It was not touched.
- **The size of `out/`.** `store_value_diff.json` is 41 MB and `.txt` 10 MB. Whether to copy them
  back whole is the orchestrator's call. `store_value_diff_summary.txt`, `classify_non_qsi.txt` and
  `qsi_rows.csv` carry what the records cite.

### State handed on

- The lock is free (`status`: the lock directory does not exist; no Ray processes of
  `/tmp/ray-sgk`).
- The registry holds 16 runs. R9 and R10 are `done`; §6's `wider` is alive and is not this agent's.
- R9's store, primary `cfba7ac4…`, and R10's, primary `557066be…`, are in `var/datastores/`. Both
  carry fingerprint `1b332ffa…`, and neither has a sidecar.
- The main checkout is at `53084dc`, clean. The worktree is at `e648951`, clean and unwritten.
- The records' links assume they are copied to `docs/levin-phase-floor/05/rehearsal/`.
