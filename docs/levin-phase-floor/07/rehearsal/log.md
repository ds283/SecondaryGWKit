## §5 — rehearsals R27 and R28 on the Mac Studio

**Prompt:** [`07-stationary-point-guard.md`](../07-stationary-point-guard.md) §5 ("As 06m §4"), under the
orchestrator's Part B dispatch (`74600350`, addendum B.5), with the user's addendum at planned stop 1. The code
measured is 07's code commit `ad732887` (location, windows, the nodal guard and §1.3's per-window `iroff1` rule in
`_adaptive_levin`; the four counts in `QuadSourceIntegral`'s Levin records), against its parent `91472dbe`, whose
production code is 06m's.

**Host:** mps026298, the Mac Studio.
- **R27, S2's capture and the q-r captures at p** ran in the worktree `SecondaryGWKit-lpf07-91472db`, detached at
  `91472dbe` (`levin_quadrature.py` `beaf0e23…`, `QuadSourceIntegral.py` `962424c0…`; against R26's tree
  `38f6a828` only `CLAUDE.md` differs; `bd21bf23..91472dbe` outside `docs/` and `prompts/` is empty).
- **R28, the comparisons, the replay, the q-r references, S1, the readers and S2's references** ran in the
  worktree `SecondaryGWKit-lpf07-7460035`, detached at `74600350` (`levin_quadrature.py` `dc05596c…`,
  `QuadSourceIntegral.py` `33cb161e…`). `git diff --stat 91472dbe 74600350` outside `docs/` and `prompts/` touches
  exactly the six files the dispatch lists; `ad732887..74600350` outside `prompts/` is empty.
- **The same file in both trees:** the launcher `RunDrivers/quadsource_atol_sweep.py` (`b0135218…`), `main.py`
  (`a5d47f5f…`), `LiouvilleGreen/three_bessel_integrals.py` (`fea67d56…`), `tools/store_value_diff.py`
  (`7997af85…`), `ComputeTargets/phase_groups.py` (`795a0119…`), `config/fingerprint.py` (`9fb94d77…`).
- **The main checkout** stayed at `28ebaf07`, clean; it ran no tree code.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in six turns: (1) the gate, the
lock, R27's launch; (2) R27's checks, S0, R28's launch; (3) R28's checks, R28 against R27, the tabulation
(planned stop 1, lock held); (4) the replay's launch; (5) the replay's checks, the q-r references, S1, the readers,
the release, S2's references' launch; (6) S2's scoring and these records.

**Date:** 2026-11-08, 00:05–10:50 UTC.

**Commit:** none. The records are committed on the laptop, in 07's records commit.

**Result:** COMPLETE. **S0, S1 and S2 HOLD**; outside `QuadSourceIntegral` every value table is identical; inside it
every untouched row is bit-identical and the rest are tabulated. No §8 stop condition fired. One finding for the
user: the location on an identically zero rate (the q-r groups with q = r), where R28 is honest but does about
4,100–5,500 Clenshaw–Curtis pieces per call.
- **The fingerprints.** R27 and R28 both `done`, exit 0, `1b332ffa…`; `classes` equal R26's and R25's; `problems {}`.
- **S0 — R27 against R26: value-identical.** 7 value tables, 360,462 rows, identical; all 864 `QuadSourceIntegral`
  rows bit-identical in every value and counter field. Labels differ only by the job-name prefix and stamp
  (normalised in a copy, as 06m did).
- **R28 against R27, outside `QuadSourceIntegral`:** every value table identical (360,462 rows); every other
  difference a timing, label, `parents_digest` (291) or `rho_reuse` (26); 0 unclassified; no row in one store only.
- **Inside it:** 640 rows untouched (every count 0 in every Levin record), **all 640 bit-identical**; 224 rows
  tabulated (70 with a group call that located a point, 200 with an analytic block that did). No leaf guarded.
  `total_converged` 845 → 862: 17 rows gained, none lost. The largest move is 7.4e4 × R27's `total_abserr`.
- **The replay** (the user's approval): 470 of 470 touched Levin records reproduced bit for bit at R28's tree, giving
  every located point's kind and location: G+q+r roots at z = 10211.32 (42) and 275.009 (2); analytic J4/Y4 roots
  (400); each q-r call with q = r 4,097 points, every scan sample, θ′ ≡ 0.
- **The q-r references** (the user's addendum): on three of the 26 calls, the worst included, R28 is honest (0.179,
  0, 0.069 of its abserr) and so is R27 (0.0021, 0.0014, 0.0006). On the worst call R28's verdict rests on the
  Gauss–Legendre sequence; against the reference's stated error, set by QAGS, it is undecided (below).
- **S1 holds:** w00–w11 each locate one root at z = 10211.3179 ± 9e-6, and each is honest (largest 0.218, w11).
- **S2 holds:** both P1 rows locate their point (z 10211.318, 275.009; |x − x_s| ≤ 7e-9 against a spacing of 0.0041)
  and are honest against four agreeing readings (0.020, 0.263). R27 reproduces P1's failure (5.6e5, 4.1e4).
- **The readers** exit 0, six of six; R28 byte-identical before and after. **The lock** is released.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md` §7.

### What shipped

No code. The records are under the main checkout's `var/levin-phase-floor-07/rehearsal/`, for the laptop to
commit as `docs/levin-phase-floor/07/rehearsal/`:
- `log.md` (this), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.
- **`scripts/`:**
  - `rec.sh` (`b0952204…`): 06m's (`0a13627b…`), the record path changed by `sed`;
  - `classify_non_qsi.py` (`26690e61…`): 05's, unchanged, byte-identical to 05's and 06m's copy;
  - `normalise_jobname_labels.py` (`b02fd3fd…`): 06m's (`08dc45a8…`), its prefixes and docstring changed
    (`out/normalise_jobname_labels.diff`);
  - `qsi_identity_r26_r27.py` (`244840d0…`): 06m's `qsi_identity_r24_r25.py` (`e63aae0c…`), R24/R25 → R26/R27
    and a provenance header (`out/qsi_identity_r26_r27.diff`);
  - `s0_labels_and_qsi_tool_report.py` (`88357082…`), new;
  - `tabulate_qsi_guard.py` (`a1eaba31…`), `qsi_guard_records.py` (`e327f3f9…`), `qsi_rows_summary.py`
    (`aa317123…`): new; the store reader and field classification copied verbatim from 06m's (06g's);
  - `capture/`: 06g's `references/v1/scripts/` harness (`worker.py`, `dispatch.py`, `harness.py`, `rebuild.py`,
    `prov.py`, `run_census.py`, `launch.py`, `rerun.py`, `spscan.py`), copied (hashes equal to 06g's,
    `state/capture_harness_06g_v1_sha256.txt`); `prov.py` (`f84a1771…`) and `worker.py` (`ba140a73…`) changed
    where marked "prompt 07 rehearsal" (`out/capture_harness_diff.txt`); `replay07.py` (`68f985eb…`) and
    `run_replay_pipeline.sh` (`11139d79…`), new;
  - `replay_summary.py` (`8c978d0b…`), `s1_score.py` (`21cfa56f…`), `qr_reference.py` (`add90893…`),
    `run_qr_reference.sh`, `qr_score.py` (`f6d6282a…`), `qr_worst_detail.py` (`a2e7131e…`), `p1_references.py`
    (`d3770e03…`), `s2_score.py` (`a2bcd258…`): new;
  - `run_readers_worktree.sh` (`ec96cf09…`): 06m's (= 06g's), unchanged; its diff against the original,
    `out/run_readers_worktree.diff`, equals 06m's.
- **`out/`:** `commands.log`; the gate (`00-gate-trees.txt`, `01-gate-registry-lock.txt`, `01-registry-list.txt`,
  `02-r26-vs-06m-kept.txt`); each build's launch and end checks and `status.json`; S0's (`classify_non_qsi_r26_r27*`,
  `normalise_jobname_labels.*`, `qsi_identity_r26_r27.*`, `s0_*`); R28 against R27 (`classify_non_qsi.*`,
  `qsi_tabulation.*`, `qsi_rows.csv`, `qsi_guard_records.*`, `qsi_rows_summary.txt`, `stop1_check.txt`,
  `depth_max_and_counters.txt`); the replay (`replay_rows.txt`, `replay_dev1*`, `replay_pipeline.log`,
  `replay_run.txt` … `capture_check.txt`, `replay_calls.csv`, `replay_calls.checks.json`, `captures.json`,
  `replay_summary.txt`, `replay_store_check.txt`, `replay_launch_check.txt`); the q-r references
  (`qr_reference/` with its two failed attempts kept, `qr_scores.*`, `qr_worst_call_detail.txt`); S1
  (`s1_scores.*`); S2 (`p1_references/` small files, `p1_references_status.json`, `p1refs_*`, `s2_scores.*`); the
  readers (`readers.txt`, `readers_check.txt`); `final_check.txt`, `final_registry_list.txt`.
- **`state/`:** `lock.log`; the launch lines (`launch_r27.txt`, `launch_r28.txt`, `launch_replay.txt`,
  `launch_p1_references.txt`); the markers; every store-hash file (R26 at the gate; R26/R27 before and after S0;
  R27 at R28's launch; R27/R28 before and after the comparison, after the replay; R28 before and after the readers;
  `stores_sha256_final.txt`); the replay's row list and specs.
- **Not for copying back** (`KEPT_ON_MAC_STUDIO.txt`): the three diff JSONs, the readers' `extract/`, `state/capture/`
  and `state/inputs/` (the pickles), the tool's two reading JSONs (2.5 MB each) and their `.log` copies.
- **The stores are kept:** `var/datastores/levin-phase-floor-07-r27*` and `…-r28*`, five files each.

### Predictions and verdicts

| Rule (prompt 07 §5, as written) | Scored on | Verdict |
|---|---|---|
| R27 and R28 with 06m's command line, slugs `r27-parent`, `r28-guard`, `--prompt 07`; **both fingerprints `1b332ffa…`** | both `done`, exit 0, `1b332ffa…`; `classes` equal R26's and R25's; `problems {}` | **holds** |
| **S0 — R27 against R26:** every value table identical, every `QuadSourceIntegral` value field identical, every other difference of the four volatile kinds | 7 tables, 360,462 rows identical; 864 / 864 rows bit-identical in 118,776 value and 27,163 counter fields; other: timing, label (prefix and stamp, normalised in a copy), `parents_digest` 291, `rho_reuse` 26 | **holds** |
| R28 against R27, outside `QuadSourceIntegral`: every value table identical, every other difference of the four volatile kinds, no row in one store only | 360,462 rows identical; 0 unclassified; 0 only-in-one, unmatched or unresolvable | **holds** |
| Inside: a row in which nothing was located or guarded is bit-identical in `total`, `total_abserr`, the analytic columns and every Levin record, the four counts set aside | 640 untouched rows; 640 bit-identical (value and counter fields) | **holds** |
| Inside: any other row tabulated (change against both `total_abserr`, located points (kind, z), window pieces, regions, maximum depth, `total_converged`) | 224 rows in `qsi_rows.csv`; 470 records in `qsi_guard_records.csv`; kind and location for each from the replay (`replay_calls.csv`, 470 / 470 bit for bit) | **done** |
| The stationary-point rows' convergence in each store, and the rows whose `total_converged` changes | rows with a group call that located: 52 / 70 converged (R27), 69 / 70 (R28); all rows: 845 → 862; 17 gained, 0 lost (listed) | **done** |
| **S1 — V1's twelve:** each locates its point at z ≈ 10211 and \|v28 − ref\| ≤ abserr28 against 06g's reference | 12 / 12 one root at z = 10211.31789148–50 (within 8.5e-6 of 10211.3179; spacing 31–44); ratios 0.00037–0.218 | **holds** |
| **S2 — P1's two rows:** each locates its point within one scan spacing; each G+q+r call within its abserr of a reference taken afresh (captured at R27's tree, four readings) | row 1: \|x − x_s\| 7.1e-9 (spacing 0.0041), ratio 0.020; row 2: 5.6e-9, 0.263; readings agree (scatter 6.1e-37, 1.2e-36 against 6.6e-33, 9.2e-33) | **holds** |
| The readers run on R28, exit 0, the store byte-identical | 6 / 6 exit 0; five files identical by SHA-256, size, mtime; no journal | **holds** |
| The lock is released, `ray-processes` lists nothing | released; `status` free; `ray-processes` empty before and after | **holds** |
| The stores are kept | R27 and R28 in place, unchanged since the comparison | **holds** |

**The user's addendum at planned stop 1** (not a prompt rule; reported): **R28 is honest on the three referenced
q-r calls**, and so is R27:

| call (q = r) | R27 \|v−ref\|/abserr | R28 \|v−ref\|/abserr | reference's stated error / R28 abserr |
|---|---|---|---|
| (1e5, 1e6, 1e6, z 6.985e9), the worst | 0.00215 | 0.179 | 5.8 |
| (3.162e5, 1e6, 1e6, z 1.363e10) | 0.00142 | 0 | 0.15 |
| (1e5, 3.162e5, 3.162e5, z 3.047e9) | 0.000618 | 0.0695 | 0.022 |

**On the worst call R28's "honest" rests on the Gauss–Legendre sequence alone.** GL at m = 8, 10, 12, 14 agree to
5.9e-39 (GL12 = GL14), and R28 is 0.179–0.186 of its abserr (7.89e-37) from each. The reference's stated error,
4.56e-36, is |QAGS − GL14|: QAGS sits 4.9 times its own estimate (9.29e-37) from GL14, and R28 is 5.6 of its abserr
from QAGS. Against the stated error the verdict is undecided. On the other two calls it is honest outright. R27 is
honest on all three under any reading.

### Deviations from the prompt

1. **The reading for S1 and S2 was deferred until after planned stop 1** (06g's `scores.json` and `report.md`, `v1/`
   scripts, the capture harness, `tools/filon_reference.py`). The brief listed them among what to read; nothing
   before the stop depended on them. IMPLEMENTATION CHOICE. 06g's and 06h's §4 were not read: 06m's records gave the
   launch's exact form.
2. **At the gate one read-only check wrote and removed a scratch file `/tmp/x_never`** (outside the write scope),
   and its diff of R26's hashes exited 1 (the grep also matched R26's run-directory line). Redone properly
   (`out/02-r26-vs-06m-kept.txt`: equal). UNINTENDED DRIFT.
3. **S0's labels were normalised in a copy of the JSON** (06m's deviation 1): R26's job name is
   `levin-phase-floor-06m`, R27's `levin-phase-floor-07`; the raw classifier reports 5,148 UNCLASSIFIED, the
   normalised copy 0. The verdict is the same either way. STRUCTURALLY REQUIRED.
4. **S0's identity inside `QuadSourceIntegral`** used 06m's identity script, adapted, and a new corroboration script
   (the labels; the tool's own `QuadSourceIntegral` report). IMPLEMENTATION CHOICE.
5. **A mistyped parent pid in `pgrep -P`** when finding R28's driver (read-only; printed nothing). UNINTENDED DRIFT.
6. **The tabulation inside `QuadSourceIntegral` is new scripts**, not 06m's `tabulate_qsi_signedyardstick.py`: the
   classification by R28's four counts and the untouched-row test are new; the reader and field classification are
   06m's (06g's), verbatim. IMPLEMENTATION CHOICE.
7. **Two inspections were printed and not saved** (R28's metadata structure at stop 1; a tally over
   `qsi_guard_records.csv`), and two more later (a q-r pickle's keys and timing; `scores.json`'s structure). Every
   figure used is in a saved output. IMPLEMENTATION CHOICE.
8. **A two-unit development run of the replay** (`dev1`), to check the harness on 07's code and time it; kept.
   IMPLEMENTATION CHOICE.
9. **The replay ran every touched row at R28's tree only** (enough for kind and z); R27's tree ran only P1's rows and
   the q-r rows, for the captures. IMPLEMENTATION CHOICE.
10. **The harness lives in `scripts/capture/`** (so `prov.py`'s `parents[4] == MAIN` holds unchanged), the stores are
    reached through symbolic links in `state/stores/` (as 06g's `v1/stores/`), and a worker runs several units (one
    k cache per process). Kind and location come from `guard_07`, a record the worker reads from each call's own
    result after the timed section; for a call of more than 64 points it records a summary (counts, x range, whether
    every point is a grid sample) and re-samples θ′ and θ on the grid. IMPLEMENTATION CHOICE.
11. **The q-r references failed twice before they ran:** first a zsh loop did not split its arguments (six processes
    failed on a path); then scipy refused QAGS at epsrel 1e-14 (it must exceed 50 eps), fixed to 1.2e-14. Both
    attempts' outputs are kept under `out/qr_reference/failed_attempt_*`; all six runs were redone on the fixed
    script. UNINTENDED DRIFT.
12. **The q-r reference method** (the user's terms leave it to the agent): direct composite Gauss–Legendre at five
    resolutions with QAGS as a second rule, on the integrand captured at R28's tree; each capture replayed bit for
    bit at its tree, and the two trees' integrands compared bit for bit at 1,001 points. IMPLEMENTATION CHOICE.
13. **`p1-references`' manifest records the main checkout's `git_head` (28ebaf07)**, because its launcher lives under
    `var/` there; its cwd and the tool it ran are the code worktree's (74600350). STRUCTURALLY REQUIRED (the
    registry records the script's repository; the scripts' records live in `var/`).
14. **Two readings the prompt leaves open:** S2's count tolerance is min(abserr_R27, abserr_R28)/100 (06g's form, as §6
    V4 states it for the census); S1's "at z ≈ 10211" is read as one root within one scan spacing (in z, at the
    point) of 10211.3179, P1's value. Either verdict is the same at any reasonable reading (|z − 10211.3179| ≈ 9e-6).
    IMPLEMENTATION CHOICE.
15. **S2's references ran at the R27-tree captures from the code worktree** (as 06g read captures across trees); the
    integrand code is the same in both trees (`QuadSourceIntegral.py` differs only in the four counts; `phase_groups.py`
    and `LiouvilleGreen/` are equal). IMPLEMENTATION CHOICE.

### Verification performed

**The gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`): the trees as stated above; 62 runs, all
finished; `~/.raypoolkit` empty, no old lock; `auth_token` mode 600; 135 GB free; venv 3.13.16 with datastorekit 0.5.0,
raypoolkit 0.3.0, runregistrykit 0.2.0, ray 2.59.0, SQLAlchemy 2.0.55, numpy 2.2.4, scipy 1.16.3, mpmath 1.3.0; no 07
store or run; R26's five files equal to 06m's record.

**R27 and R28** (`out/r2{7,8}_{launch,end}_check.txt`): R27 launched 00:06:33Z (sh 83000, driver 83008), `done`
00:15:41Z (about 9 min); R28 launched 00:19:15Z (sh 83821, driver 83825), `done` 00:35:54Z (about 16.5 min); each
manifest at its tree with `git_dirty` false and the launcher `b0135218…`; no Traceback and no `!! WARNING`; each
driver's cwd its worktree.

**S0 and R28 against R27:** `tools.store_value_diff --json` from the code worktree (16.9 s each); the stores
byte-identical before and after each comparison (`state/s0_store_sha256_*`, `state/store_sha256_*_diff.txt`).

**The tabulation's assertions** (`out/qsi_tabulation.txt`): equal key sets, columns and tolerances; the four counts
present as ints exactly in R28's Levin records; the group regions summing to `WKB_Levin_num_regions` and
`total_converged` the AND of the group flags, in every row of both stores; `DEFAULT_LEVIN_MAX_DEPTH` 20 in both trees.

**The replay and captures** (`out/replay_check.txt`, `out/capture_check.txt`, `out/replay_store_check.txt`): 470 / 470
records; 8 / 8 captures (S2's two at p against R27; three q-r at p against R27 and at g against R28); Ray never
initialised; R27 and R28 unchanged; both worktrees clean, nothing newer than the markers.

**The q-r references** (`out/qr_scores.txt`): the replays reproduce their captures; the two trees' integrands
bit-identical at 1,001 points with equal x_span and kw; θ′ and θ exactly 0 at every sample.

**S2's references** (run `levin-phase-floor-07-p1-references-20261108T103132`, `done`, exit 0, 2 / 2; row 1 191 s, row
2 425 s): four readings each, Route B, one non-monotone segment direct (QAWO) and three (FCC), as the tool's rule says
beside a stationary point; the readings agree within the tolerance by four orders of magnitude.

**The readers** (`out/readers_check.txt`): six of six exit 0; R28 identical before and after; 0 Tracebacks; six
`!! WARNING … shards (=4) … (=10)`, as 06m's.

**The lock** (`state/lock.log`), ten commands, each its own, appended by `>>`: ray-processes (empty); acquire;
set-pid 83008; ray-processes (empty); set-pid 83825; ray-processes (empty); ray-processes (empty); release; status
(free, no Ray processes); ray-processes (empty). Each exited 0.

**Final** (`out/final_check.txt`): R26, R27 and R28 equal to their earlier hashes; 65 runs, all finished;
`~/.raypoolkit` empty; the main checkout and both worktrees clean and unmoved, nothing written in either; no
`__pycache__` in the records.

### Observations not acted on

1. **The location on an identically zero rate** (issue `[07-location-on-an-identically-zero-rate]`): the 26 q-r
   calls with q = r locate every scan sample; they cost about 20 times a G+q+r call and double the rehearsal's
   regions; R28 stays honest where referenced.
2. **17 rows gained convergence, none lost;** the one row still unconverged among the touched (3.162e5, 3.162e5,
   3.162e5, z 573.11) has one unconverged call (R27: two), at depth 20 in both.
3. **R27 is wrong where 07 says it should be:** 16 rows move by more than both stores' errors (up to 7.4e4); on the
   referenced ones (S1's w rows, S2's rows) R28 is honest and R27 is 4.1e4–5.6e5 of its abserr out (S2) — P1's
   defect reproduced at the parent's code.
4. **No minimum was located and no leaf guarded** anywhere in the rehearsal (`guarded_regions` 0 throughout, as §1.5
   expects with `theta_deriv` supplied).
5. **The analytic blocks locate a root in the 2pt5 J4 and Y4 blocks** of 200 rows (k = q = r), at a fixed η per k;
   `analytic_rad` moved by at most 1.5e-5 of its abserr.
6. **QAGS on the worst q-r call** is 4.9 times its own estimate from the Gauss–Legendre sequence; the reference rests
   on GL there (Predictions and verdicts).
7. **The tool writes its JSON to stdout too,** so each `p1_row*.log` duplicates its `.json` (kept on the Mac Studio).

### State handed on

- **R27** `levin-phase-floor-07-r27-parent-20261108T000634` and **R28** `levin-phase-floor-07-r28-guard-20261108T001916`:
  both `done`, `1b332ffa…`. **`p1-references`** `levin-phase-floor-07-p1-references-20261108T103132`: `done`, exit 0.
- **Their stores are kept** (`state/stores_sha256_final.txt`): `var/datastores/levin-phase-floor-07-r27*`, primary
  `d6fd3023…`; `…-r28*`, primary `00468d55…`. R26 (06m's, read only) unchanged.
- **The lock is released;** `~/.raypoolkit` is empty and `ray-processes` lists nothing. Nothing of this agent runs.
- **The trees** are clean and unmoved; nothing was written in either worktree.
- **Held, for the user:** the zero-rate case, before the census (the orchestrator's hold). Nothing of prompt §6 was done.
- **Kept on the Mac Studio, not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`,
  `state/capture/`, `state/inputs/`, the tool's reading JSONs).
