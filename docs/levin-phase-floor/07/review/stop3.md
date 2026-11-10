# Planned stop 3: levin-phase-floor 07 Part B (Mac Studio orchestrator)

Written 2026-11-08, after the census exited and before step 7. Neither stop condition is met.

## The user's word at stop 2, verbatim (relayed by the laptop session)

> Please keep all 8 `k` and launch the census

## The census

- **Run:** `levin-phase-floor-07-census-20261108T171137`, registered, slug `census`, launched by the census agent at
  17:11:37Z from tree g `e182bc7a`.
  - The detached `nohup sh -c` was pid 28796 and the driver `run_census.py` was pid 28799.
  - It ran six workers at once, 300 s per unit, with 06a's over-budget pass in the same driver.
  - The census agent held the rehearsal lock (set-pid 28799) and released it after the readers.
- **My one check at launch:** both pids answered `kill -0`; `status.json` read running, 0/18,560; the registry showed 1 alive and 67 finished.
- **End:** one background `kill -0` loop on 28796, which exited at 23:30:11Z.
  - `status.json`: state `done`, exit_code 0, 18,560 of 18,560, heartbeat 23:30:05Z.
  - stdout ends `over budget in the two sets: 0 []` and `== run ends: 18560 of 18560 units recorded; units in error 0; problems []`.
  - stderr is empty.
  - `checkpoint.jsonl` has 18,560 lines and no duplicate unit. Its SHA-256 is `03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10`.
- **Wall time:** 22,708 s (6 h 18 min 28 s).
  - That is against the projection of 21,458 s (+5.8 %) and 06m's 18,465 s.
  - The one-day backstop logged three past-projection lines; their largest projected total was 22,686 s. Nothing was stopped.
- **Over-budget pass:** no unit exceeded 300 s in either set, so it had nothing to re-run.

## My own comparison against 06m's census

This is independent of the census agent's `compare_07.py`. I used three scripts under `review/scripts/`; the outputs are in `review/stop3/`.

### 1. `stop3_compare.py` (sha `a8c322c1…`)

- **How it reads the records.** Every census unit is diffed leaf by leaf with 06m's record and region pickle.
  - 06m's line is accepted only if it matches the index's SHA-256 and names the same unit.
  - 06m's pickle record is accepted only if it matches its line's own SHA-256, and the census pickle likewise.
  - Floats are compared by `float.hex`, arrays by dtype, shape and bytes.
- **Re-keying.** Before the diff, each call's entries are re-keyed by the call's index, so that a call whose lists change length cannot shift another call's entries. This covers calls, other_calls, every pickle list carrying `index`, and the pickle's positional `scan` keyed by group-call order.
- **Classes.** Each differing path is classified as new 07 key, timing, provenance or `cache_built`. Anything else is attributed to its call, or to unit level.
- **Quiet and active.** A call is QUIET if it has no located point, no window, no window piece, no guarded leaf, no guard at depth max, and no leaf flagged in `window_07`, `guarded_07` or `guard_at_depth_max_07`. Otherwise it is ACTIVE.
- **Result:**
  - 18,560 units and 289,482 calls; the call sets match 06m's in every unit.
  - **284,514 quiet calls: 0 differ.**
  - 4,968 active calls (4,940 locate a point and 28 guard only), in 2,456 units.
  - **0 failing units.** All 16,104 units with no active call are equal whole: apart from new keys, timing, provenance and `cache_built`, there is no difference at all. Their `total` and `total_abserr` equal 06m's in hex.
- **What remained at unit level in the 2,456 active units:**
  - aggregates: totals, `WKB_Levin`, `analytic_rad`, dispatcher G/T counts, W3 and deep-line counts, `analytic_06h`;
  - two per-call structures left positional, handled next.

### 2. `stop3_supplement.py` (sha `c17d38c6…`)

- It attributes `deep_lines.lines` by index, and the store-level group records (`result_full.WKB_Levin_subintervals[].groups[]`) by order.
  - The match by order is checked: each group record's `label` equals its call's `group` in both records.
- **Result:** 10,688 quiet group records equal 06m's except 07's four counts (absent at 06m) and `elapsed`. 78 quiet calls' deep lines are equal. **0 bad.**
- All 63 distinct unit-level keys are listed in `supplement.txt`.

### 3. `stop3_emptylists.py` (sha `854310a1…`)

- It covers the per-call lists that went empty↔non-empty, so the re-keying could not apply: deep lines, the 06e deep lines, and W3.
- **Result:** every entry on either side belongs to an active call. **0 bad.**

### Conclusion

No call that locates nothing and guards nothing differs from 06m's.

## The census agent's report: it agrees with mine

- Its `compare_07.py calls` gives the same counts: all 18,560 units equal, 284,514 quiet calls bit for bit, and 16,104 units whole with totals equal in hex.
  - Locating calls: 4,940, which is 956 group and 1,514 analytic at each rtol, none guarding.
  - Guarding calls: 28, none locating.
- **The only excluded key that differs** beyond timing, provenance and the new keys is `cache_built[].mode_serial`, in 5 units. That is 06m's observation 1, the producer build order.
- **Evidence:** every unit loads g's `levin_quadrature.py`, with no foreign module and no variant. All 16 workers show `4e5da371…`, tree e182bc7a clean, and Ray never started.
- **The 28 guard-only calls are the O9 case** (`[07-c2-gate-threshold-was-never-measured]`).
  - They come from seven rows, J4 (#19) and Y4 (#20), at both rtols: 1-20502, 0-10584, 2-21488, 1-10792, 0-33732, 3-748 and 3-1102.
  - Rule A skipped phase 0 with V from 15.60 to 18.58, below 6π.
  - Each guarded one leaf; the leaf was bisected, and none was returned guarded.
  - All 28 converged, with 2–3 regions at depth 1–2.
  - My active count (4,968 = 4,940 + 28) agrees.
- **New deviation 15:** `scripts/stop3_guarded_07.py`, a read-only reader. IMPLEMENTATION CHOICE.

## State at stop 3

- **Protected files:** 181/181 OK (`checks/protected_at_stop3.txt`).
- **Lock:** free, with no Ray processes of `/tmp/ray-sgk`.
- **Registry:** 68 runs, all finished.
- **Trees:** g `e182bc7a`, p `91472dbe` and `-7460035` at `74600350` are all clean with `--ignored`. The main checkout's HEAD is `28ebaf07`, unmoved.
- **Nothing committed.** Step 7 (scoring V1–V3 and V5, capturing V4's 24 calls, the projection) has not started.
