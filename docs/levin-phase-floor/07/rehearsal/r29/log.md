## §5 resumed under §10 (Rule A) — rehearsal R29 on the Mac Studio

**Prompt:** [`07-stationary-point-guard.md`](../07-stationary-point-guard.md) §10 (Part B after Rule A) and §5, under the
orchestrator's resumed dispatch (`e182bc7a`), with the user's word at R29's stop. The code measured is `adffa21b`
("Skip locating a phase whose variation is below the C2 gate"), Rule A on 07's `ad732887`. R29 takes R28's place in §5;
R28 (`74600350`) is kept as the record of the zero-rate case.

**Host:** mps026298. **Trees:** R29, the comparisons, the replay, S1, S2 and the readers in the worktree
`SecondaryGWKit-lpf07-e182bc7` (detached at `e182bc7a`; `levin_quadrature.py` `4e5da371…`; `QuadSourceIntegral.py`,
the launcher, `main.py` and `store_value_diff.py` unchanged from R28's tree); the four-block replays also at
`SecondaryGWKit-lpf07-91472db` (R27's) and `SecondaryGWKit-lpf07-7460035` (R28's), read-only replays. `74600350..e182bc7a`
outside `docs/` and `prompts/` touches `levin_quadrature.py`, the two test modules and `CLAUDE.md` only.

**Model:** Claude Opus 5.5, the rehearsal subagent, in four turns (7–10 of this agent): the gate and R29's launch; R29's
checks and the comparisons (a stop on 4 records); the user's word, the replay and the four-block captures; the
references, S1, S2, the readers, the release and these records.

**Date:** 2026-11-08, 12:33–13:35 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE (planned stop 1′). R29 `done`, `1b332ffa…`. **S1 and S2 hold on R29.** Against R28, 90 Levin
records changed, every one a call R28 located and R29's Rule A skipped: 86 equal R27 bit for bit (26 `q-r`, 60
analytic `2pt5` J4/Y4 blocks), and 4 analytic blocks were caught by the nodal guard instead (a stop, ruled findings by
the user). **On those 4 blocks all three stores are honest** against references with stated errors 1,990× or more below
every abserr. Against R27, 692 untouched rows are all bit-identical. Regions are back to R27's level (114,460 against
114,826; R28 227,883), wall time 9.6 min (R27 9, R28 16.5).

### What shipped

No code. Records under `var/levin-phase-floor-07/rehearsal/r29/` (committed by the laptop as
`docs/levin-phase-floor/07/rehearsal/r29/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
- **`scripts/`:** `rec.sh` (b5279dd9, the stop-1 copy with the record path changed; `out/rec_sh.diff`);
  `classify_non_qsi.py` (26690e61, 05's, unchanged); `tabulate_qsi_guard.py` (a1eaba31, the stop-1 script unchanged)
  and `tabulate_qsi_guard_r29.py` (47e18cbe; `out/tabulate_qsi_guard_r29.diff`); `compare_r29_r28.py` (f6ba1f2e),
  `stop_detail.py` (93517fee), `block_reference.py` (fc2c20df), `run_block_reference.sh` (0a5372da), `block_score.py`
  (e62946d1), `replay_summary_r29.py` (49097a96), `s2_score_r29.py` (e7430b57): new; `s1_score_r29.py` (6f60c8d1; the
  stop-1 `s1_score.py` by sed, `out/s1_score_r29.diff`); `run_readers_worktree.sh` (ec96cf09, unchanged);
  `capture/`: the stop-1 harness (hashes equal, `state/capture_harness_stop1_sha256.txt`) with `prov.py` (1ec5f339)
  and `worker.py` (6bcb4a2d) changed where marked "prompt 07 r29" (`out/capture_harness_r29_diff.txt`), and the new
  `replay07_r29.py` (192c9c42) and `run_r29_pipeline.sh` (984105f7).
- **`out/`:** `commands.log`; the gate and launch/end checks; `r29_status.json`; both diff JSONs and their classifier
  outputs; `qsi_tabulation_r27_r29.*`, `qsi_rows_r27_r29.csv`; `r29_vs_r28.*`, `r29_vs_r28_records*.csv`,
  `stop_detail.txt`; the replay (`replay_rows.txt`, `r29_pipeline.log`, stage outputs, `replay_calls.csv`,
  `replay_calls.checks.json`, `replay_summary.txt`, `blocks.json`); `block_reference/`, `block_reference_run.txt`,
  `block_scores.*`; `s1_scores.*`, `s2_scores.*`; `readers.txt`, `readers_check.txt`; `final_check.txt`,
  `final_registry_list.txt`.
- **`state/`:** `lock.log`, the launch lines and markers, every store-hash file, the replay's rows and specs.
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`, `state/capture/`, `state/inputs/`).
- **The store is kept:** `var/datastores/levin-phase-floor-07-r29*`.

### Predictions and verdicts

| Rule (as the resumed dispatch and prompt §5/§10 write it) | Scored on | Verdict |
|---|---|---|
| R29 with R28's command line, slug `r29-guard-rule-a`; fingerprint `1b332ffa…` | `done`, exit 0, `1b332ffa…`, `classes` equal R28's/R27's/R26's | **holds** |
| R29 against R28, outside `QuadSourceIntegral`: every value table identical | 360,462 rows identical; 0 unclassified | **holds** |
| R29 against R28, inside: a record may differ only if R28 located and R29 did not; such a record equals R27 (value, abserr, converged, phase_limited, regions) with R29's four counts 0; every other record bit-identical to R28 | 13,087 of 13,177 bit-identical; 90 changed, all R28-located / R29-skipped; **86 equal R27**; **4 do not** (guarded) | **fails as written on 4 records** — the user ruled them findings; the laptop corrects the rule (§10's guard clause) |
| Rows that differ hold such a record; tabulated | 54 rows, each holding a changed record; `total`, `total_abserr` equal R27's in all 54 | **holds** |
| R29 against R27 (§5 item 4): untouched rows bit-identical; the rest tabulated; `total_converged` changes listed | 692 / 692; 172 tabulated; 17 gained, 0 lost | **holds** |
| The four-block check (the user's addition): each store's value against a reference | 12 of 12 store-blocks honest outright (largest 6.2e-4 of abserr) | **reported** |
| S1 on R29 | 12 / 12 located at z 10211.3179 ± 9e-6, honest, largest 0.218 | **holds** |
| S2 on R29 (the standing `p1-references` readings) | located within one spacing on both; 0.020 and 0.263 | **holds** |
| The readers on R29, exit 0, store byte-identical | 6 / 6; identical | **holds** |
| The lock released, `ray-processes` empty | released; free; empty | **holds** |

**The four blocks** (`out/block_scores.txt`; float.hex in `verification.md`): the stated error of each reference is
set by QAGS's distance from GL14 (QAGS within its own estimate), and GL's sequence agrees to 0–2e-25. Ratios
|v − ref|/abserr: J4 at (3.162e5)³ R27 7.4e-5, R28 4.4e-5, R29 1.5e-6; Y4 there 1.6e-5, 5.8e-4, 3.7e-4; J4 at (1e6)³
1.0e-4, 2.1e-4, 3.1e-6; Y4 there 0, 6.2e-4, 5.1e-5. V = 15.113 and 15.896 (below SIX_PI); the gate's span estimate on
the root leaf = 19.160 and 20.196 (above it). R29's J4 declarations are 19× and 11× R27's; its Y4 declarations smaller.

### Deviations from the prompt

(Numbered on from the stop-1 rehearsal's 15.)

16. **Rule A's key is not in the store**, so V_i comes from the replay (`guard_07` records `location_skipped_phases`).
    IMPLEMENTATION CHOICE.
17. **The gate check printed the main checkout's `status --short --ignored` listing** (read-only; `status --short` is
    empty). UNINTENDED DRIFT.
18. **A mistyped parent pid in `pgrep -P`** when finding R29's driver (read-only). UNINTENDED DRIFT.
19. **`tabulate_qsi_guard.py` adapted for R29 by two sed passes** (R28 → R29, then the `_28` key suffixes); and
    `s1_score_r29.py` by one, which also changed the word in its own new comment ("R29 -> R29") and left one key,
    `store_value28`, named for R28 (it holds R29's stored value). Cosmetic. IMPLEMENTATION CHOICE.
20. **The record-by-record comparison is a new script** (`compare_r29_r28.py`), which also reports `simple_regions` and
    `evaluations`. IMPLEMENTATION CHOICE.
21. **The replay also covers the 90 records that changed against R28** (for V_i): 224 rows, 470 records (384 with a
    nonzero count, the dispatch's "about 444" being the addendum's oversight). IMPLEMENTATION CHOICE.
22. **The four-block check ran as stages of the replay pipeline** (pass 1 and pass 2 through the harness at each of the
    three trees), then each capture replayed in its own process at its tree. IMPLEMENTATION CHOICE.
23. **The four-block references:** composite 20-point Gauss–Legendre on 2^m x-panels each side of the root (located by
    brentq on θ′ sampled at 40,001 points; one root each), m = 6–14, with QAGS split at the root as a second rule, on
    R29's tree's captures; V by the trapezoidal rule on the driver's 4,097-point scan (equal to R29's
    `location_skipped_phases`, bit for bit); the gate's estimate as `build_Levin_data` forms it on the root leaf. The
    user's terms left the method to the agent. IMPLEMENTATION CHOICE.
24. **A stop was taken on the 4 guarded records** (the dispatch's rule as written), reported before anything else, and
    resumed on the user's word. STRUCTURALLY REQUIRED (the rule as written).

### Verification performed

- **The gate** (`out/00-gate.txt`): the four trees at their SHAs, clean; 65 runs, all finished; lock free.
- **R29** (`out/r29_launch_check.txt`, `out/r29_end_check.txt`): launched 12:34:28Z (sh 14524, driver 14528), `done`
  12:44:05Z, manifest at `e182bc7a`, clean, launcher `b0135218…`; no Traceback or WARNING.
- **The comparisons** from the new worktree, `tools.store_value_diff` (16.5–16.7 s each); R27, R28, R29 byte-identical
  before and after (`state/store_sha256_{before,after}_diff.txt`), and after the replays (`state/after_replay_body.txt`).
- **The replay and captures** (`out/replay_check.txt`, `out/blockcheck.txt`): 470 / 470 records; the 12 block calls
  reproduce their stores in pass 1 and pass 2, each capture's replay reproduces its value and abserr; Ray never
  initialised (224 + 6 + 6 workers, 12 block processes).
- **The readers** (`out/readers_check.txt`): 6 / 6 exit 0; R29 identical before and after; 0 Tracebacks.
- **The lock** (`state/lock.log`), eight commands each its own: ray-processes, acquire, set-pid 14528, ray-processes,
  ray-processes, release, status (free, no Ray processes), ray-processes. Each exited 0.
- **Final** (`out/final_check.txt`): R27, R28, R29 and R26 unchanged; 66 runs, all finished; `~/.raypoolkit` empty; the
  three worktrees and the main checkout clean, nothing written in any; the stop-1 records outside `r29/` untouched.

### Observations not acted on

1. **The gate's root-leaf estimate exceeds V on the four blocks** (19.16, 20.20 against 15.11, 15.90), which §10
   anticipates; the guard catches the root, at the cost of a wider declaration on J4 (`issues.md`).
2. **Rule A skips phases widely** — 423 replayed calls across every group and analytic label, V 0–18.70 — but only the
   90 records that R28 located differ from R28.
3. **R29's values on S1's and S2's calls equal R28's bit for bit;** the G+q+r roots are unchanged by Rule A.

### State handed on

- **R29** `levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, `done`, `1b332ffa…`; its store kept
  (`state/stores_sha256_final.txt`). R27, R28 (read-only record) and R26 unchanged.
- **The lock is released;** nothing of this agent runs. All trees clean and unmoved.
- **Planned stop 1′:** the census agent is not dispatched; nothing of prompt §6 was done.
