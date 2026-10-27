# levin-phase-floor 06a — the depth census: every A3 v2 row at the current code

**Subject:** prompt [`06a-depth-census.md`](../06a-depth-census.md), G2's L2 (b), on the Mac Studio,
2026-10-27. Ray-free, no lock, in the worktree `SecondaryGWKit-lpf06a-e05c34f` at the dispatch SHA
`e05c34f`, whose code is 06's $s$ tree's (`f5cee4c`). **Commit:** no code; the records are in 06a's
one records commit, made on the laptop. **Orchestration:**
[`orchestrator/prompt-06a.md`](../orchestrator/prompt-06a.md), by one Mac Studio session, which
dispatched one agent and reviewed its work.

**Result:**
- **C0 holds.**
- **C1 holds:** at `1e-7`, no row of the 9,280 has a region at depth ≥ 18. The deepest is 12.
- **C2 holds:** at `1e-8`, none either. The deepest is 13.
- **C3 fails:** 19 rows exceed 11 at `1e-7`, and 29 at `1e-8`. Every one of them holds an interior
  stationary point, and its deep regions lie at that point (issues X1).
- **C4 holds**, 70 of 70.
- **Stationary points:** 720 Levin calls in 720 rows hold an interior stationary point of θ′, all in
  `G+q+r`, and each accepts a region across it, 623 of them at depth 0 (X2). This is the extent of
  `[06-levin-accepts-a-region-across-a-stationary-point]` at the current code.
- **A3 v2's stored totals** move beyond their stored error on 6,187 of 8,722 rows at the current
  code (X3). That is B1's effect, which 05 measured on 39 rows, taken over the whole store.
- **The run:** 18,560 of 18,560 units in 45.5 minutes, none over budget or in error.
- **The stores:** A3 v2 and its copy are unchanged, both `0e32f24a…`.

**The review** ([`docs/levin-phase-floor/06a/review.md`](../../../docs/levin-phase-floor/06a/review.md)) passed nine of ten checks and stopped on check 9 alone.
`out/cmd/17_C0_check.txt` records a `c0.py` SHA-256, `b33229d8…`, that no kept file has. It was the
first `c0.py`, whose comparator counted timings and reported C0 (d) failing (deviation 15). It was
fixed in place, and its text was not kept. C0's verdict rests on `out/cmd/18`, whose `c0.py`
(`2c1b8067…`) is the file in `scripts/`.

**The user's ruling on check 9 (2026-10-27):** *"Yes, just accept as it stands."* The record is
committed as it stands. The orphaned hash is a record of a superseded check that no verdict uses.
The text cannot be reconstructed, and re-stamping `out/cmd/17` would falsify it (`CLAUDE.md` rule
7). Deviation 15 stays `UNINTENDED DRIFT`. To it is added: the first `c0.py`, SHA-256 `b33229d8…`,
was not kept.

**The laptop's re-checks** (orchestrator note §5; `docs/levin-phase-floor/06a/laptop_rechecks/`, run on the Mac Studio
read-only):
- **Check 4's comparison on two more of 05's rows:** `mid-G-3` at `1e-7` and `R4` at `1e-8`. Each
  census unit equals 06's `s` unit in `float.hex`, in `total`, `total_abserr`, and every call's
  value, abserr, region count and depth.
- **One line of `census_rows.csv.gz`:** `0-1971` at `1e-8`, the deepest row. Its depth 13, 38
  regions, `total` and `total_abserr` equal its checkpoint record.

**For the user (G2 L2):** C1 and C2 hold, so no row of A3 v2 needs layer 2 at either tolerance at
this code. What depth is left sits at stationary points, which are 07's. The three issues the
census bears on, `[04-x-at-1e-8-still-reaches-depth-20]`,
`[03-depth20-regions-are-all-clenshaw-curtis]` and `[05-b1-uncaps-most-cheap-depth-20-rows]`, close
or narrow with the user's decision.

---

# The Mac Studio record


**Subject:** prompt 06a on the Mac Studio (mps026298), Ray-free, no lock. Tree $h$: the worktree
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06a-e05c34f`, detached at the dispatch SHA
`e05c34f2fcc6ef20fc3323cb3a6f81d130c5a01c`. **Run:** `levin-phase-floor-06a-census-20261027T023143`,
launched 02:31:43Z (sh 24951, driver 24956). It ended `done` at 03:17:10Z, after 45.5 min:
18,560 of 18,560 units, exit 0, none over budget, none in error. **Commit:** none here; the
orchestrator commits the records in 06a's one records commit.

**Result:**
- **C0 holds**, all five clauses.
- **C1 holds:** at 1e-7, 0 of 9,280 rows reach depth 18, and the largest row depth is 12.
- **C2 holds:** at 1e-8, 0 of 9,280 rows reach depth 18, and the largest row depth is 13.
- **C3 fails:** 19 rows are deeper than 11 at 1e-7 and 29 at 1e-8. Every one of them holds an
  interior stationary point (issues X1).
- **C4 holds:** 70 of 70 units are bit-identical to 06's.
- **The stores:** A3 v2 and its copy are byte-identical throughout, and both fingerprint to
  `0e32f24a…`. No Ray session directory appeared.

## Before the launch (in the prompt's order; every command and its output in `out/cmd/`)

1. **Store hashes before the Gate** (`out/cmd/00`, `01`): A3 v2's six files equal 06's
   `store_files_after_fingerprint.txt` (`03`, EQUAL); the copy's five equal
   `copy_files_certified.txt` (`02`, EQUAL).
2. **The Gate** (02:13Z): main checkout `28ebaf0`, `handover-remedial`, status empty, two worktrees
   (`04`); registry: 24 runs, all finished, none running or stale (`05`); worktree at `e05c34f`,
   `status --short --ignored` empty (`06`); `git diff --stat f5cee4c e05c34f -- . ':!docs'
   ':!prompts'` shows `CLAUDE.md` alone (`07`); lock free, recorded (`11`); `/tmp/ray-sgk` holds
   59 entries, the newest from 2026-10-26 22:14 (`10`). **Holds.**
3. **The row list** (`12`): `state/rows.json`, sha256 `aa523d28…`, 9,280 rows read from A3 v2 itself
   (`var/datastores`, sqlite3 `mode=ro`), 8 k × 1,160, 145 blocks of 8 per k; 05's 39 rows marked.
4. **C0** (02:24–02:25Z): (a) `07`; (b) hashes `02` and `fingerprint_copy.py` at $h$:
   `0e32f24a…`, problems none (`15`); (c)–(e) `16` (run), `18` (check): **holds** — `low-G+r-1`
   at 1e-7 and 1e-8 and `ctrl-low-1` at 1e-7 bit-identical to 06's `s` units; `low-G+r-1` run first
   in a fresh worker and last after its block's seven other rows identical in every computed value
   (and the seven others too); no new entry under `/tmp/ray-sgk`, no worker initialised Ray.
5. **The pilot** (02:27–02:28Z; `19`, `20`): k `3:201` (9.70e6/Mpc, the largest stored compute
   time), the median-cost block of each band, 24 rows × 2 rtols in one worker: 81.8 s end to end,
   producer re-run 47.8 s, peak RSS 416 MiB. **Projection: 3,038 s ≈ 0.84 h wall at six workers,
   3.4 CPU-h; run directory ≈ 266 MB.** Within the 24 h / 20 GB rule.
6. **`state/predictions.md`** (`21`, `22`): §4 byte for byte (verified equal to the prompt's bytes
   13468–15146), sha256 `3ad9b43de04d005164c705a4116d608b32788932e17a71abd2f428a998f3f405`, mtime
   2026-10-27T02:28:40.042Z, before the launch at 02:31:43Z.
7. **Before the launch** (`23`–`28`): a driver smoke test (2 rows, 2 workers, both rtols); A3 v2
   and the copy unchanged (`26`, EQUAL); scripts diff against 06's saved
   (`out/scripts_diff_against_06_at_launch.txt`); every script the computation uses identical to
   the ones C0 and the pilot ran (`28`; only `launch.py` changed after them).
8. **Launch** (`29`) and **one check** (`30`): `running`, 16 records after 54 s, six workers, no
   stderr, `/tmp/ray-sgk` still 59 entries.

## What shipped

Nothing is committed. Under `var/levin-phase-floor-06a/`:
- **`scripts/`:** 06's harness, changed as deviations 3–12 say. New: `make_rows.py`, `spscan.py`,
  `run_census.py`, `c0.py`, `pilot.py`, `analyse06a.py`, `checks_after.py`, `extras_after.py`,
  `convergence_kinds_06a.py`, `compare_hashes.sh`, `extract_predictions.sh`. 06's other scripts
  are copied unchanged and unused. Diffs against 06's: `out/scripts_diff_against_06_at_launch.txt`
  and `…_final.txt`.
- **`out/`:**
  - every command and its output, `cmd/00`–`45`, and `commands.log`;
  - `census_summary.txt` and `census_rows.csv.gz` (1.65 MB);
  - `scan_listed_regions.csv` (720 regions);
  - `convergence_kinds_h.txt` and `convergence_kinds_06a.diff`.
- **`state/`:** `rows.json`, `predictions.md`, `analysis.json`, `scripts_06_reference/`, `C0/`,
  `pilot/`, `smoke/` and `dev/`.
- **The run directory** holds the checkpoint, the 16 per-worker pickle streams, the events and the
  worker logs (220 MB).

## Predictions and verdicts

`state/predictions.md` (sha256 `3ad9b43d…`, mtime 02:28:40Z, before the launch at 02:31:43Z) is
§4 byte for byte. The rules, as written:

- **C1 — at production's tolerance, nothing approaches the cap.**
  - **The rule:** at `rtol = 1e-7`, no row of the 9,280 has a region at depth ≥ 18.
  - **Holds** if true on every row. **Fails** if false on any row, or if any unit is still over
    budget after the pass. **Undecided** if neither, with a row that errored.
- **C2 — nor at `1e-8`.** C1's rule at `rtol = 1e-8`, scored the same way.
- **C3 — 05's sample was representative.**
  - **The rule:** at each rtol, no row's depth exceeds 11, the largest 06 found on 05's 39 rows
    at either tolerance.
  - **Holds** if true at both rtols. **Fails** otherwise.
- **C4 — the census reproduces 06.**
  - **The rule:** on 05's 39 rows the census's `1e-7` units, and on its 31 depth-20 rows the
    `1e-8` units, equal 06's `s` units bit for bit: `total`, `total_abserr`, every call's value
    and abserr, and the region arrays.

| | Number | Verdict |
|---|---|---|
| **C0 (a)–(e)** | see "Before the launch", 4 | holds |
| **C1** | 0 of 9,280 rows with a region at depth ≥ 18 at 1e-7; largest row depth 12; 0 over budget, 0 errored (`out/cmd/40`) | **holds** |
| **C2** | 0 of 9,280 at 1e-8; largest row depth 13; 0 over budget, 0 errored | **holds** |
| **C3** | rows deeper than 11: 19 at 1e-7 (all depth 12), 29 at 1e-8 (28 at 12, 1 at 13) | **fails** |
| **C4** | 70 of 70 units bit-identical (39 at 1e-7, 31 at 1e-8; `out/cmd/39`) | **holds** |

**C3's rows** (§4: "characterised by §5 items 2–3 and 7"; issues X1):
- all 48 of them hold an interior stationary point of θ′ in their `G+q+r` call;
- their regions at depth ≥ 11 lie within 1–3.5 grid spacings of it;
- every region at depth ≥ 12 is in a `Gqr` call (Levin 13 / Clenshaw–Curtis 25 at 1e-7; 17 / 42,
  plus 0 / 2 at depth 13, at 1e-8).

They are 07's stationary-point rows, not the floor that layer 2 addresses.

## Deviations from the prompt (numbered and classified)

1. **`out/cmd/08`: `rehearsal_lock status` given `--project`, which it does not take** (exit 2);
   re-run with the right arguments as `11`. UNINTENDED DRIFT.
2. **`out/cmd/11` piped its output through `sed`**, so the lock listed that `sed` as a process of
   `/tmp/ray-sgk` (the case `CLAUDE.md` warns of). A record artefact only; the lock was free and
   no Ray process existed. UNINTENDED DRIFT.
3. **`prov.py`: one tree `h`, 06a's paths, 06's run and records named, the copy under 06's
   `state/`; the header also records the main checkout.** STRUCTURALLY REQUIRED (§2 item 1).
4. **`rec.sh`'s comments name 06a.** IMPLEMENTATION CHOICE.
5. **New `compare_hashes.sh`** (data lines of two `hash_store.py` outputs). IMPLEMENTATION CHOICE.
6. **New `make_rows.py`** for §2 item 2: rows read from A3 v2 itself (sqlite3 `mode=ro`), the
   class of the top group from 01's `groups_01.csv` by 05's `draw_sample.read_census` rule, 05's
   names from 06's copy of 05's `sample.json` (sha256 checked `73aa7876…`); rows ordered by k, then
   z_response descending, then q, r. IMPLEMENTATION CHOICE.
7. **`rerun.py` gains `rerun_k_G`, `assemble_Gk` and `rerun_T_mode`** (§2 item 3), each a copy,
   statement for statement, of a section of `rerun_row` (which is unchanged and unused). Choices:
   `rerun_k_G` re-runs every `GkWKBIntegration` row of k's shard with k's exit-time serial and the
   model's serial (1,740 per k), in chunks of 200 objects, and keeps every value at a response
   redshift; the assembly checks that each parent-set member's WKB serial is cached and is that
   member's; `atan2`'s capture is kept as the one-call check, its `deltaTheta` dropped (§0.3
   change 4); the policy's metadata is not kept in the event. STRUCTURALLY REQUIRED, with these
   IMPLEMENTATION CHOICEs.
8. **`rebuild.py`: `rebuild_row(..., cache=)`, `CensusCache`, `_rebuild_row_census`.** Every
   serial-only lookup, check and the existence query stay per row. The `GkSourcePolicyData` cache
   holds the current (k, z_response) block only (rows run block by block). The `TkNumericIntegration`
   full lookups, the `TkWKBIntegration` re-runs and the `QuadSource` (lookup and recomputation) are
   cached **per worker**: each worker re-runs the T modes and recomputes the (q, r) sources its rows
   need, so across the 16 workers a T or a `QuadSource` is computed up to 16 times, each once per
   worker, all from the same inputs. IMPLEMENTATION CHOICE (C0 (d) tests the order-independence).
9. **`worker.py`'s `main()` rewritten:** no phase probe and no `build_partition` wrapper; the
   producer cache; the scan; a per-worker append-only pickle stream (offset, length and sha256 in
   each checkpoint record); a slimmer checkpoint (06's per-call summary, the totals and wall time,
   the scan's counts; the full result summary goes to the pickle); the cache's events. The region
   arrays add `relerr`, `phase_err` and `phase_limited` to 06's five. Required parts STRUCTURALLY
   REQUIRED (§2 items 3, 7, 8, 9); the pickle layout and the extra arrays IMPLEMENTATION CHOICE.
10. **New `spscan.py`** (§5 item 7): θ′ evaluated by scalar calls on 4,097 points (`numpy.linspace`,
    endpoints included); a region listed "grid" if a sign change lies between two grid points both
    inside it, else "ends" if θ′ at its ends and midpoint disagree in sign. **A region's value is
    not kept by the driver**, so "its value against abserr" is recorded as `1/relerr` (relerr is
    abserr over max(min(|estimate|, |refined|), local atol)). STRUCTURALLY REQUIRED for the value;
    the rest IMPLEMENTATION CHOICE.
11. **`run_census.py`** (§2 item 4): workers started longest first; each worker's stdout to
    `worker_<name>.log` in the run directory; `--skip-recorded-in` for a resume (§6); the order
    check is per worker; as in 06, any unit in error or any problem ends the run `failed` with
    exit 2. IMPLEMENTATION CHOICE.
12. **`launch.py`**: prompt `06a`, slug `census`, `run_census.py`, purpose and scope per §6,
    18,560 expected units. STRUCTURALLY REQUIRED.
13. **Development runs:** before C0, a one-row smoke test (`out/cmd/13`, `state/dev/smoke1`) and
    its comparison with 06's unit (`14`, `state/dev/dev_compare_smoke1.py`); after the
    predictions, a driver smoke test (`23`, `state/smoke`). A temporary file
    `state/_worker_main_new.py` was written and deleted while editing `worker.py`. IMPLEMENTATION
    CHOICE.
14. **C0 (c)–(e) by a new `c0.py`**: three fresh workers (W1: `low-G+r-1` first then its block's
    seven other rows, then `low-G+r-1` at 1e-8; W2: the seven then `low-G+r-1`; W3: `ctrl-low-1`);
    the seven others compared between W1 and W2 as well. IMPLEMENTATION CHOICE.
15. **The first C0 check (`out/cmd/17`) said (d) FAILS**: its comparator compared the per-group
    `elapsed` timings nested inside the full result summary. Diagnosed with
    `state/dev/dev_diff_full.py` (only `elapsed` and `compute_time` differed); the comparator now
    drops timing keys at any depth; re-run `18`: holds. No worker was re-run. UNINTENDED DRIFT.
    *(Added on committing, by the user's ruling on review check 9, 2026-10-27: the first `c0.py`,
    SHA-256 `b33229d8…`, which `out/cmd/17` records, was changed in place and its text was not
    kept. The record stands as it is.)*
16. **The pilot's choice** (k `3:201`, the largest stored compute time; the median-cost block per
    band; both rtols in one worker) and the projection method (two rounds of eight workers on six
    slots per rtol). The projection's output printed the 48 pilot units' maximum depths before
    `predictions.md` was written; the predictions are the prompt's §4 verbatim, so nothing in them
    could depend on it. IMPLEMENTATION CHOICE.
17. **`out/cmd/21` wrote `predictions.md` and then failed** at its `stat` calls (GNU `stat` is
    first on this host's PATH); the file was written once, by the `awk` before them. Its sha256,
    mtime and byte equality with §4 were recorded by `22`. UNINTENDED DRIFT.
18. **`launch.py` changed after C0 and the pilot**; every other script they ran is identical at
    the launch (`28`). IMPLEMENTATION CHOICE.

19. **No over-budget pass was run:** no unit was over budget, so §3.2's pass had nothing to
    re-run. Recorded for completeness; it is not a departure from the prompt.
20. **New `analyse06a.py`** for §5 and C1–C3. A row's regime is the regime of its deepest Levin
    call (none without a call); a row's class is 01's stored top-group class. IMPLEMENTATION CHOICE.
21. **`convergence_kinds.py` could not read the census's records** (one pickle per worker; no
    `role`), so it was run from a changed copy, `convergence_kinds_06a.py`. The copy reads each
    unit's record by offset, checks its sha256, and prints `role` with `.get`. Diff
    `out/convergence_kinds_06a.diff`; sha256 `e04a0a1e…`; the original is `49c83a72…` and is
    `cmp`-equal to `prompts/levin-phase-floor/measure/`. As the prompt allows. Its first run
    (`out/cmd/42`) stopped on `KeyError: 'role'`, was fixed, and was re-run as `43`. UNINTENDED
    DRIFT for the first run.
22. **New read-only cross-checks `checks_after.py` (`41`) and `extras_after.py` (`45`).**
    `checks_after.py` reads 05's run, read only, for the size check of X3. IMPLEMENTATION CHOICE.
23. **`out/convergence_kinds_h.txt`** is `43`'s output without its per-record input lines (the
    full output, 3.9 MB, is `43`). IMPLEMENTATION CHOICE.

24. **`out/scripts_diff_against_06_final.txt` was taken twice.** The first was at `44`; it was
    taken again at the very end, after `extras_after.py` had been added, so that it covers every
    script. Both worktree and main-checkout status were re-read empty at that point.
    IMPLEMENTATION CHOICE.

Deviations 1, 2, 15, 17 and the first run in 21 are the UNINTENDED DRIFT. None of them touched a
store, a tree, or a record outside `var/levin-phase-floor-06a/` and the run directory.

## Verification performed
- **The tree** is $h$ at `e05c34f`, `status --short --ignored` empty at the Gate, before the launch,
  on all 18,560 checkpoint records and at the end. All records carry `run_census.py` `83c4deec…`.
  The main checkout is at `28ebaf0`, status empty, throughout.
- **C0** as above. Every script the computation used at the launch is the one C0 and the pilot ran
  (`28`).
- **The run:** 18,560 records, no duplicates, every worker's units recorded in its planned order,
  no problem reported, `stderr.log` empty. Every pickle record matches the sha256 its checkpoint
  names (`40`).
- **No Ray:** `/tmp/ray-sgk` holds 59 entries before the Gate, during C0, at the launch and at the
  end. No new session directory; no worker reported Ray initialised.
- **A3 v2:** SHA-256, size and mtime of its six files are identical before the Gate, before the
  launch, after the run and after the fingerprint. `store fingerprint` (no `--write`) gives
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, problems none
  (`34`, `38`).
- **The copy:** its five files are identical to `copy_files_certified.txt` at every check.
  `fingerprint_copy.py` at $h$ gives `0e32f24a…`, problems none, before (`15`) and after (`35`).
- **06's records were only read:** its run directory still holds 124 files, its checkpoint sha256
  is unchanged (`134ef9ae…`), and its scripts in `var/` equal those in `docs/` (`44`).

## Observations not acted on

All are in `issues.md`, each with its numbers:
- X1: the deep rows are stationary-point rows;
- X2: the scan's two counts are equal by construction, and 623 of the regions are at depth 0;
- X3: the stored totals move beyond their error on 6,187 rows;
- X4: 634 rows have an infinite error;
- X5: convergence, the "before" for 08;
- X6: `Levin_z` differs on 22 rows;
- X7: the producers against the stored values;
- X8: 1e-7 against 1e-8;
- X9: regions;
- X10: record notes.

## State handed on
- **Nothing is running.** The run is `done`. No lock was taken. Nothing was written in the worktree
  or the main checkout outside `var/levin-phase-floor-06a/` and the run directory.
- **A3 v2 and the scratch copy are unchanged**, both `0e32f24a…`. The copy is still kept for the
  user to delete.
- **For the orchestrator:**
  - copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/` (without `cmd/43`, which is
    3.9 MB) and `state/` (without the pickles under `C0/`, `pilot/`, `smoke/` and `dev/`) to
    `docs/levin-phase-floor/06a/`;
  - this log becomes `logs/06a-depth-census.md`;
  - `verification.md` becomes `VERIFICATION.md` §6a; its links assume `06a/`.
- **For the user (G2 L2):** C1 and C2 hold, so no row of the store needs layer 2 at either
  tolerance at this code. C3 fails only on stationary-point rows, which are 07's.
