## §5 — rehearsals R13 and R14 on the Mac Studio

**Prompt:** [`06-producers-store-the-residual.md`](../06-producers-store-the-residual.md) §5, with
its §9, under the orchestrator's Part B dispatch (`f5cee4c`), its §B.1 and §B.5. §B.5 governs.
**Host:** mps026298, the Mac Studio.
- **R13** ran in the worktree `SecondaryGWKit-lpf06-9f66622`, detached at `9f66622`, the code
  commit's parent.
- **R14**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06-f5cee4c`,
  detached at `f5cee4c`, whose code is the code commit `dc9d28c`'s.
- **The main checkout** stayed at `28ebaf0` and clean. It ran nothing, and was only the home of the
  venv, the stores, the registry and these records.

**Model:** Claude Opus 5.5, as the Mac Studio orchestrator's rehearsal subagent, in three turns:
- the gate, the lock and R13's launch;
- R13's checks and R14's launch;
- R14's checks, the comparison, the tabulations, the readers, the release and these records.

**Date:** 2026-10-26, 21:55–22:15 UTC.
**Commit:** none. The records are committed on the laptop in 06's records commit.
**Result:** COMPLETE WITH DEVIATIONS. None of the deviations stopped the work, and none changed a
computation.
- **The builds.** R13 and R14 both ended `done` with exit 0 and fingerprint `1b332ffa…`. Each
  equals R8's class by class, and R14's `classes` object equals R13's.
- **The warning.** R14 printed no `!! WARNING (GkSourcePolicyData` line (0, in its launch log,
  `stdout.log`, `stderr.log` and Ray's worker logs), and `_check_phase_sign` did not raise.
- **Outside the five classes of §0.3:**
  - every value table is bit-identical: 4 tables, 40,958 rows;
  - every other difference is of the volatile kinds: 0 unclassified;
  - no row and no column is in one store only.
- **Inside them:**
  - `theta_residual` is in R14 only;
  - `theta_mod_2pi` moved in 146,251 $G$ and 2,162 $T$ rows, by a median of 0.3–0.4
    $\varepsilon|\theta|$;
  - **no `theta_div_2pi` changed;**
  - `G_WKB` and `T_WKB` moved by the remainder's change alone;
  - `GkSourcePolicyData` moved only in `Levin_z_dtheta_dlogz`;
  - 554 of 864 `QuadSourceIntegral` totals moved. Two of them, at $z_{\rm resp} = 250$, moved by
    1.8e4 and 484 × R13's `total_abserr`, with their depth halved.
- **The readers** exited 0, and the stores were left byte-identical.
- **The lock** was released.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6 "The stored phase".

### What shipped

Nothing in the repository: no code, and no commit. Everything was written on the Mac Studio, in
the main checkout:
- **The records directory** `var/levin-phase-floor-06/rehearsal/`, holding:
  - `out/`;
  - `scripts/`:
    - 05's `classify_non_qsi.py` and `tabulate_qsi.py`, unchanged;
    - this agent's `tabulate_phase_classes.py` (§5.4's per-class tabulation),
      `examine_outliers.py`, `summarise_diff.py`, `qsi_extra_counts.py`, `peek_structure.py` and
      `run_readers_worktree.sh`;
  - `state/`;
  - `extract/`, the readers' output, 176 MB, not for copying back;
  - `log.md`, `verification.md` and `issues.md`.
- **The stores** `var/datastores/levin-phase-floor-06-r13*.sqlite` and `-r14*.sqlite`, five files
  each.
- **The runs** `levin-phase-floor-06-r13-parent-20261026T215550` and
  `levin-phase-floor-06-r14-residual-20261026T215945`, with their launch logs
  `var/runs/levin-phase-floor-06-r1{3,4}.launch.log`.

The stores are kept, as prompt §5.7 says.

### Predictions and verdicts

§5 fixes no scored predictions; §6 holds them. These are the acceptance checks of §7.2, and the
dispatch's own:

| check | number | verdict |
|---|---|---|
| R13 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 class digests = R8's | holds |
| R14 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 = R8's; `classes` object = R13's | holds |
| outside the five classes, every value table identical | 4 tables, 40,958 rows, bit-identical | holds |
| outside them, every other difference of the four kinds; no row in one store only | 0 unclassified outside; 0 only-A, only-B or unmatched in any class; no column in one store only outside | holds |
| inside them, the differences tabulated (§5.4's six items) | `out/phase_classes_tabulation.txt`, `out/qsi_tabulation.txt` | done |
| `theta_div_2pi` changes: none, or one cycle within 4ε\|θ\| of 0 or −2π | 0 changes in all three value tables; 0 `raw_theta_div_2pi` changes | holds |
| the readers exit 0, store byte-identical | 6 × exit 0; 5 files identical | holds |
| the lock is released | `release` exit 0; `status` free; `ray-processes` none | holds |
| (dispatch) R13 ≠ R14 in `GkWKBIntegration`'s values | 146,251 of 147,642 `GkWKBValue` rows differ | holds |
| (dispatch, §8) no `!! WARNING (GkSourcePolicyData` line in R14 | 0 (launch log, stdout, stderr, 24 Ray worker logs) | holds |
| (dispatch) `_check_phase_sign` does not raise in R14 | no Traceback, `Error` or `Exception` anywhere; `done` exit 0 | holds |

### Deviations from the prompt

#### 1. Both builds ran from worktrees, on the main checkout's venv — STRUCTURALLY REQUIRED

Prompt §5.2 puts R13 in a worktree, and the dispatch puts R14 in one too, so that the main checkout
does not move.
- **The python.** Neither worktree has a venv, so each build used the main checkout's
  `venv/bin/python` by absolute path, with cwd and `PYTHONPATH` set to its worktree.
- **The paths.** The `--database` and the launch log were absolute paths into the main checkout.
  The registry recorded both runs under the main checkout's `var/runs/`.

The evidence that each ran its own tree:
- the manifests record `git_head` `9f66622d…` and `f5cee4c7…`, both with `git_dirty` false;
- `lsof` showed each driver's cwd, and its `Datastore` actor's cwd (97834 and 133), as its own
  worktree;
- the two stores differ in `GkWKBValue`, which the parent cannot write differently from itself.

#### 2. `PYTHONDONTWRITEBYTECODE=1` on both builds, and `-B` on the in-process tools — IMPLEMENTATION CHOICE

These keep `__pycache__` out of the worktrees, which must not be written, and change no
computation. This is R11's form from 05a. Every check of the worktrees found them clean: `git
status --short --ignored` was empty, and `find -newer` the pre-R13 marker listed nothing, after
each build, after the readers and at the end.

#### 3. The comparison ran with the code worktree as its cwd — STRUCTURALLY REQUIRED

`python -m` puts the cwd first on `sys.path`. The code worktree's `tools/store_value_diff.py`
(`7997af85…`) is the only one that reads a store lacking `theta_residual`. The main checkout's is
05's `5c92d8e3…`, so run from there `-m tools.store_value_diff` would have imported it.
- The dispatch's form names `PYTHONPATH=<code worktree>`. The cwd made it bind.
- The store paths were absolute.

#### 4. No text form of the comparison — IMPLEMENTATION CHOICE

05 and 05a also ran the text form with `--max-rows 1000000`. Here the JSON alone is 1.09 GB, against
41 MB in 05a, because 146,251 value rows differ in each of three tables. §5.4 asks for `--json`
only. Every summary was taken from the JSON, or from the tool's own reader.

#### 5. The `!! WARNING` search was extended to Ray's worker logs — IMPLEMENTATION CHOICE

The dispatch asks for a `grep` of R14's launch log and stdout. In both builds the launch log holds
the `Datastore` actor's output only, and no line from any task. `_build_phase` and
`_check_phase_sign` run in Ray tasks, so their prints land first in
`/tmp/ray-sgk/session_*/logs/worker-*.out`.
- The same patterns were searched there too, for both sessions: `!! WARNING (GkSourcePolicyData`,
  any `!! WARNING`, `Traceback`, `_check_phase_sign`, `error|exception`.
- Every count is 0 ([`out/ray_worker_logs_check.txt`](out/ray_worker_logs_check.txt)).
- These logs are read-only, and are kept by Ray, not by this record. The count in the launch log
  and stdout is 0 as well.

#### 6. `classify_non_qsi.py`, copied unchanged, exits 1 with 185 "UNCLASSIFIED" — IMPLEMENTATION CHOICE in reading it

05 wrote the script for a change to `QuadSourceIntegral` alone, so it exempts only that class.
§5.4 says it sorts the differences outside the five classes of §0.3. All 185 problems it reports
are inside them:
- 182 are `GkSourcePolicyData` rows whose `metadata` key `Levin_z_dtheta_dlogz` moved;
- 3 are the value tables of `TkWKBIntegration`, `GkWKBIntegration` and `GkSource`.

Outside the five classes it reports 0. The script was not edited. Its output was read class by
class, and `summarise_diff.py` checks the outside/inside split from the JSON independently.
- **Why this was not a stop.** The dispatch lists "an `UNCLASSIFIED` difference" among its stops,
  and §8 words the stop as "outside those classes".
- I read the dispatch's stop as §8's. Every one of the 185 is a change §0.3 expects and §5.4
  tabulates.
- If the orchestrator reads it otherwise, this is the point to ask at. Nothing after it wrote to a
  store.

#### 7. The readers ran through a three-line copy of `run_readers.sh` — STRUCTURALLY REQUIRED

`run_readers.sh` changes directory to its own tree, then calls `./venv/bin/python`.
- **From the code worktree** there is no venv.
- **From the main checkout** it would run the main checkout's code (`28ebaf0`, without the column)
  against R14, not the code commit's.

`scripts/run_readers_worktree.sh` takes the tree and the python from the environment
(`READERS_TREE`, `READERS_PYTHON`) and sets `PYTHONDONTWRITEBYTECODE=1`. Nothing else differs:
[`out/run_readers_worktree.diff`](out/run_readers_worktree.diff), 3 lines. The readers'
`--database` and `--output` were absolute.

#### 8. More read-only scripts than the one the dispatch names — IMPLEMENTATION CHOICE

The dispatch names one new script, for §5.4's per-class tabulation: `tabulate_phase_classes.py`.
Four more were written, each saved in `scripts/` with its SHA-256 in
[`out/script_sha256.txt`](out/script_sha256.txt), and each stamps its own hash and its inputs':
- `summarise_diff.py` is 05a's inline summary made a script, with the inside/outside split and the
  columns in one store only;
- `peek_structure.py` is a first look at the JSON's shape;
- `examine_outliers.py` covers the rows above 4 ε|θ| and above A|Δθ|;
- `qsi_extra_counts.py` gives counts from `qsi_rows.csv`.

None writes a store.

There were also two exploratory runs, inline in Python, that printed the tool reader's column lists
and key shapes. Their output was not saved, and no number in these records comes from them; their
form is in `out/commands.log`.

#### 9. `examine_outliers.py` failed once — UNINTENDED DRIFT

The first run exited 1 with a `ValueError`. The CSV holds k as a `float.hex` string, and the script
parsed it with `float()`. The script was corrected to use `float.fromhex` and re-run, and it exited
0.
- The failed run's output is kept, as `out/examine_outliers.attempt1.{txt,stderr}`.
- It had printed only the header lines before failing. No number came from it.

#### 10. The per-class tabulation pairs rows with the tool's own reader — IMPLEMENTATION CHOICE

`tabulate_phase_classes.py` calls the code worktree's `tools.store_value_diff.read_store`, with
`config.datastore`'s factories. It does not use raw SQLite, as `tabulate_qsi.py` does.
- **What that gives.** The pairing is exactly the tool's identity matching, and every column of
  both sides is in hand, not only the differing ones the JSON carries. The JSON omits
  `theta_div_2pi` and `theta_residual` because they do not differ.
- **The check.** Its differing-row counts per column equal the JSON's for all three tables.

#### 11. The scale ε|θ| was taken from R14's own stored cycle count, per table — IMPLEMENTATION CHOICE

§5.4 forms $|\theta|$ "from the stored cycle count".
- **For the producer tables** that is the producer's own count, anchored at its `z_init`.
- **For `GkSourceValue`** it is the rectified count, so the same remainder is measured against a
  larger $|\theta|$ there.

Both are reported. The 54 producer rows above 4 ε|θ| are re-expressed against `GkSource`'s
$|\theta|$ in `out/examine_outliers.txt`.

#### 12. The fingerprints were not re-taken — IMPLEMENTATION CHOICE

Both are the launcher's, taken at the end of each build and read from `status.json`, as in 05a.
- R13's five files are byte-identical from just after its build (21:59) to the end.
- R14's five were first hashed at 22:04, after its fingerprint and before any read, and are
  identical to the end.

#### 13. `/usr/bin/time` wrapped the comparison and four scripts — IMPLEMENTATION CHOICE

- **`-p`** wrapped the comparison, `classify_non_qsi.py` and `tabulate_qsi.py`.
- **`-l`**, which adds the peak memory, wrapped `peek_structure.py` and
  `tabulate_phase_classes.py`. The peaks were 3.5 and 4.6 GB.

Its lines are the whole of each `.stderr` file beside these outputs: no script wrote anything else
to stderr.

#### 14. The gate's `ls var/levin-phase-floor-06` shows `rehearsal` — IMPLEMENTATION CHOICE

I created the records directory a moment before the gate re-check, so that the gate's output
could be written into it. There was no `var/levin-phase-floor-06/` before that, and no
`levin-phase-floor-06-*` store: the gate's `grep` exited 1.

### Verification performed

| check | result | where |
|---|---|---|
| gate | three trees at their SHAs and clean; `dc9d28c^` = `9f66622`; `dc9d28c..f5cee4c` touches only `prompts/`; 21 runs finished; lock free; no 06 store; launcher `5b984c75…` in all trees | `out/gate.txt` |
| R13 | `done`, exit 0, pid 97740, `1b332ffa…`, `9f66622d…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r13_end_check.txt`, `out/r13_status.json` |
| R14 | `done`, exit 0, pid 99938, `1b332ffa…`, `f5cee4c7…`, `git_dirty` false; 0 Tracebacks; `stderr.log` empty | `out/r14_end_check.txt`, `out/r14_status.json` |
| warning / sign check | 0 in launch logs, stdout, stderr and 24 worker logs per build | `out/r14_end_check.txt`, `out/ray_worker_logs_check.txt` |
| worktrees unwritten | `status --ignored` empty; `find -newer` empty, after each build, after the readers, and at the end | `out/r13_end_check.txt`, `out/r14_end_check.txt`, `out/final_check.txt` |
| scripts | the two copies equal 05's (`cmp`); `run_readers.sh` and the launcher equal 05's; the tool is `dc9d28c`'s | `out/script_sha256.txt` |
| value tables | outside: 4 tables, 40,958 rows, identical; inside: 3 tables differ, `theta_residual` in R14 only | `out/store_value_diff_summary.txt` |
| row columns outside | 0 unclassified; every kind timing or `label` | `out/classify_non_qsi.txt` |
| per-class tabulation | the counts equal the JSON's; 0 cycle changes | `out/phase_classes_tabulation.txt`, `out/examine_outliers.txt` |
| QSI tabulation | pairing = the tool's identities; 0 `total` mismatches against the tool | `out/qsi_tabulation.txt`, `out/qsi_extra_counts.txt` |
| readers | 6 × exit 0; lookups equal 05a's; R14 identical | `out/readers.txt`, `state/r14_store_sha256_{before,after}_reads.txt` |
| stores | 10 files identical from before the comparison to the end | `state/store_sha256_before_diff.txt`, `state/store_sha256_after_diff.txt`, `state/stores_sha256_final.txt` |
| lock | acquire 0; set-pid 0 ×2; ray-processes 0 ×4; release 0; status 0 | `out/lock.log` |

### Observations not acted on

- **Two `QuadSourceIntegral` rows moved far beyond R13's declared error.**
  - **Where:** at $z_{\rm resp} = 250$, at $(k, q, r) = (3.162\times10^5)^3$ and
    $(10^5, 3.162\times10^5, 3.162\times10^5)$.
  - **How far:** `total` moved by 1.6e-3 and 2.8e-5 relative. Against `total_abserr`$_A$ that is
    1.8e4 and 484×; against `total_abserr`$_B$, 1.3e4 and 555×.
  - **What else changed:** depth 12 → 6 and 14 → 6, and the region count was roughly halved. Both
    rows are unconverged in both stores.
  - **Elsewhere:** 2 more rows exceed 1 × `total_abserr`$_A$ (1.25 and 1.02), and 55 exceed 0.1.
  - This is the rehearsal scope's counterpart of §6's S4. It is recorded here, not judged, and is
    proposed in `issues.md` for the board to weigh.
- **ε|θ| understates the old error near the producer's anchor.**
  - Every producer row above 4 ε|θ| (52 $G$, 2 $T$; max 272) has $|\theta| < 1$ rad and
    $|\Delta| \le$ one ulp of $2\pi$.
  - The old sum cancelled terms of size $k\tau$, so its error there is set by them, not by
    $|\theta|$.
  - The 99th percentile overall is 1.2–1.4 ε|θ|.
- **`G_WKB` and `T_WKB`.** Every change is within $A|\Delta\theta_{\rm mod}| + 4\varepsilon A$, the
  remainder's change plus $G$'s own rounding. The relative changes up to 3.6e-4 are at zeros of the
  oscillation.
- **`Levin_z` did not move in any row.** Only the diagnostic `Levin_z_dtheta_dlogz` moved, in 182 of
  291 `GkSourcePolicyData` rows, by at most 2.6e-6 relative.
- **`total_phase_limited` flipped in 13 rows** (4 0→1, 9 1→0). No `total_converged` flag changed.
- **The GkWKB residual-memo counters** differ in 25 rows. That is 05's P2 again: 25 in 05 and 31 in
  05a.
- **26 rows have `total_abserr` = inf in both stores.** 9 of their `total`s differ. This is
  `[04-roundoff-floor-can-be-infinite]`.
- **The readers' file counts differ from 05a's** (Gk 396 against 387, GkSource 176 against 162,
  QSI 70 against 73), and the lookups are equal. This is
  `[10-the-readers-draw-their-work-lists-unseeded]`.
- **Each reader logs one `!! WARNING`** about the number of shards read against the number
  specified. It is not the §8 warning, and no reader failed.
- **The size of `out/`.** `store_value_diff.json` is 1.09 GB, against 41 MB in 05a. The orchestrator
  decides whether to copy it back. The summaries, `phase_classes_rows.csv` (297,493 rows) and
  `qsi_rows.csv` carry what the records cite.
- **`[05-store-value-diff-compares-volatile-columns]`** recurs exactly as 05 found it. It is already
  open.
- **Prompt §6's re-measurement** ran beside these builds under its own agent. It was not touched.

### State handed on

- **The lock is free.** `status` reports that the lock directory does not exist, and there are no
  Ray processes of `/tmp/ray-sgk`. `~/.raypoolkit/` is empty.
- **The registry** holds 23 runs, all finished. R13 and R14 are `done`.
- **The stores.** R13's primary is `766839d6…` and R14's is `fff6cf7f…`. Both are in
  `var/datastores/`, both carry fingerprint `1b332ffa…`, and neither has a sidecar. They are kept.
- **The trees.** The main checkout is at `28ebaf0` and clean. Both worktrees are at their SHAs,
  clean and unwritten.
- **The links.** The records' links assume they are copied to `docs/levin-phase-floor/06/rehearsal/`.
