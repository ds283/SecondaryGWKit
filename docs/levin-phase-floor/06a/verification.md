## 6a. The depth census

*Prompt 06a, on the Mac Studio, 2026-10-27, Ray-free, no lock. The records are under
[`06a/`](06a/). The pickles, the run directory and every file over 2 MB stay on the Mac Studio.*

**What was measured.** Every `QuadSourceIntegral` row of A3 v2, 9,280 of them, was re-run
in-process at the current code, at `rtol = 1e-7` and at `1e-8` (`atol` 1e-32), and each Levin
call's depth was recorded.
- **The tree:** $h$, the worktree at the dispatch SHA `e05c34f`. Its code outside `docs/` and
  `prompts/` is 06's $s$ tree's (`f5cee4c`), byte for byte.
- **The store:** 06's certified scratch copy of A3 v2, opened read-only.
- **The method:** 06's harness, with each producer re-run once per worker instead of once per row
  ([`06a/scripts/`](06a/scripts/); diff against 06's in
  [`06a/out/scripts_diff_against_06_final.txt`](06a/out/scripts_diff_against_06_final.txt)).
- **The run:** `levin-phase-floor-06a-census-20261027T023143`, 02:31:43Z–03:17:10Z (45.5 min). It
  ended `done`, 18,560 of 18,560 units, exit 0. No unit went over budget (300 s) or errored, and
  `stderr.log` is empty. The run directory is 220 MB.

### 6a.1 Gate C0 (before the launch): holds

| | | Evidence |
|---|---|---|
| (a) | The code is 06's: `git diff --stat f5cee4c e05c34f -- . ':!docs' ':!prompts'` shows `CLAUDE.md` alone | [`out/cmd/07`](06a/out/cmd/07_gate_diff_stat.txt) |
| (b) | The copy's five files equal `copy_files_certified.txt`; `fingerprint_copy.py` at $h$ gives `0e32f24a…` with no problems | [`02`](06a/out/cmd/02_copy_vs_certified.txt), [`15`](06a/out/cmd/15_C0b_fingerprint_copy_at_h.txt) |
| (c) | Census units bit-identical to 06's `s` units: `low-G+r-1` at 1e-7 and at 1e-8, `ctrl-low-1` at 1e-7 (total, total_abserr, every call's value and abserr, the region arrays) | [`18`](06a/out/cmd/18_C0_check_rerun.txt) |
| (d) | `low-G+r-1` run first in a fresh worker, and run last after the seven other rows of its block, is identical in every computed value; so are the seven | [`18`](06a/out/cmd/18_C0_check_rerun.txt) |
| (e) | No session directory appeared under `/tmp/ray-sgk` (59 entries before and after, here and for the run); no worker initialised Ray | [`18`](06a/out/cmd/18_C0_check_rerun.txt), [`44`](06a/out/cmd/44_final_state.txt) |

**The pilot** ([`19`](06a/out/cmd/19_pilot_run.txt), [`20`](06a/out/cmd/20_pilot_projection.txt)):
three blocks of k = 9.70e6/Mpc, at both rtols, in one worker. It projected 0.84 h of wall time and
266 MB. The run took 0.76 h and 220 MB.

**The predictions** were written to [`state/predictions.md`](06a/state/predictions.md):
sha256 `3ad9b43d…`, mtime 02:28:40Z, byte-equal to the prompt's §4, before the launch at 02:31:43Z.

### 6a.2 Predictions and verdicts

| | Rule (prompt §4) | Number | Verdict |
|---|---|---|---|
| **C1** | at `1e-7`, no row has a region at depth ≥ 18 | 0 of 9,280 rows; largest row depth 12; no unit over budget, none errored | **holds** |
| **C2** | the same at `1e-8` | 0 of 9,280; largest row depth 13; none over budget, none errored | **holds** |
| **C3** | at each rtol, no row's depth exceeds 11 | 19 rows at 1e-7 (all at depth 12); 29 at 1e-8 (28 at 12, one at 13) | **fails** |
| **C4** | 05's 39 rows at `1e-7`, and its 31 depth-20 rows at `1e-8`, bit-identical to 06's `s` units | 70 of 70 | **holds** |

C3's failure is characterised in [`06a/issues.md`](06a/issues.md) X1:
- every one of the 48 deeper rows holds an interior stationary point;
- their regions at depth ≥ 11 lie within 1–3.5 grid spacings of it;
- every region at depth ≥ 12 is in a `Gqr` call.

### 6a.3 Descriptive (§5), from [`out/census_summary.txt`](06a/out/census_summary.txt)

1. **Row depth.** 2,489 rows have no Levin call.

   | depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
   | 1e-7 | 291 | 530 | 1,786 | 654 | 99 | 96 | 511 | 2,010 | 732 | 29 | 3 | 31 | 19 | 0 |
   | 1e-8 | 284 | 516 | 1,790 | 567 | 69 | 91 | 396 | 529 | 1,565 | 923 | 8 | 24 | 28 | 1 |

   - The 2,265 rows stored at depth 20 fall to depths 1–12 at 1e-7; 1,248 of them are at depth 7.
   - The largest depth by band is low 12, mid 12, seam 9 at 1e-7; low 12, mid 13, seam 10 at 1e-8.
2. **Depth ≥ 12 by branch** (Levin / Clenshaw–Curtis): 13 / 25 at 1e-7; 17 / 42, plus 0 / 2 at
   depth 13, at 1e-8. All are in `Gqr` calls.
3. **The deepest rows:** listed in the summary with every call. They are mostly equilateral
   q = r = k at k = 1e5 and 3.14e5/Mpc, mid band, class `Gqr`.
4. **Regions.** 108,879 at 1e-7 and 157,229 at 1e-8, against 145,313,734 stored.
5. **Convergence.**
   - `total_converged` is False on 2,578 rows at 1e-7 and 2,638 at 1e-8 (2,262 in the store).
   - Kinds of unconverged call: bound only 4,014 / 3,745; resolution 1,737 / 2,182; phase-limited
     1,213 / 1,720; capped 0 / 0.
   - 634 rows have `total_abserr = +inf` at both rtols (558 in the store).
6. **Totals against A3 v2 at 1e-7.**
   - The change exceeds the stored error on 6,187 of 8,722 rows: 5,677 beyond 10× and 4,818 beyond
     100×; median 286×.
   - `Levin_z` differs on 22 of 1,160 `GkSourcePolicyData`; type and quality on none.
7. **Stationary points (1e-7).**
   - 720 rows, in 720 calls (all `Gqr`/`G+q+r`), hold an interior stationary point. In every one a
     region was accepted across it: 623 of them at depth 0 (the whole call accepted as one region),
     677 Levin, 238 phase-limited.
   - The grid spacing h runs from 1.7e-5 to 7.1e-3 (median 2.5e-3). A pair closer than h is not
     seen.
8. **Cost.** Levin wall sums 0.67 h at 1e-7 and 0.77 h at 1e-8; the median unit takes 0.25 s and
   0.27 s. Each k's producer re-run takes 53–58 s. A3 v2 stored 286 h of compute time.
9. **Disagreement warnings:** 0 (unit counters, the 2,320 assemblies, and a grep of the logs).

### 6a.4 The stores

A3 v2's six files and the copy's five keep the same SHA-256, size and mtime at every check: before
the gate, before the launch, after the run and after the fingerprints
([`38`](06a/out/cmd/38_closing_compare.txt)). No journal, `-wal` or `-shm` file appeared.
- `store fingerprint` on A3 v2 gives `0e32f24a…`, problems none
  ([`34`](06a/out/cmd/34_fingerprint_A3v2.txt)).
- `fingerprint_copy.py` on the copy gives `0e32f24a…`, problems none
  ([`35`](06a/out/cmd/35_fingerprint_copy_after.txt)).

The worktree and the main checkout are clean at the end ([`44`](06a/out/cmd/44_final_state.txt)).
