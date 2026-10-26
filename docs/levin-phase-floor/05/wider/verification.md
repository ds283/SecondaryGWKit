### 5.x The wider re-measurement (prompt 05 §6)

**Prompt 05 §6, run 2026-10-26 on the Mac Studio (mps026298).** There was no Ray, no lock and no
store write (route (a), as in §3 and §4). Two trees took part:
- $p$, the code commit's parent `e648951` (B1's code without 05's change), in the git worktree
  `SecondaryGWKit-lpf05-e648951`;
- $c$, the code commit `120fdfe`, in the main checkout at `53084dc`, which adds only records.

Each worker loaded its packages from its own tree only. The `hello` records in
[`state/L0/ledger.jsonl`](05/wider/state/L0/ledger.jsonl) and the run's `worker_*_hello.json` show
it. Under $c$, every $G$ phase had `exact_remainder=True` and every $T$ phase `False`.

**The run.** `levin-phase-floor-05-wider-20261026T020005` ended `done`, 109/109 units, exit 0, in
12 minutes. Its checkpoints and region pickles stay in that run directory on the Mac Studio.
- Every record carries `run_wider.py`'s SHA-256 (`491d2608…`), and its tree's `HEAD` and empty
  `git status --short`.
- No unit went over budget, and none raised.

**The store** was byte-identical before, before launch, after the run and after the fingerprint
([`before`](05/wider/state/store_files_before.txt),
[`after`](05/wider/state/store_files_after_fingerprint.txt)). No journal, `-wal` or `-shm` file
appeared. `store fingerprint`, run without `--write`, gives
`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` and `problems: none`
([`out/fingerprint_05_wider.txt`](05/wider/out/fingerprint_05_wider.txt)).

**The rows** ([`state/sample.md`](05/wider/state/sample.md), fixed before any row ran;
[`out/sample.txt`](05/wider/out/sample.txt)). There are 39:
- 24 depth-20 rows, 3 per cell of {`G+r`, `G−r`, lone `G`, `Gqr`} × {low, mid}, the cheapest at
  distinct $k$ with stored `compute_time` ≤ 900 s;
- 03's R1–R7;
- 8 controls: two per band with depth < 20 and a group containing $G$, and two with `NULL` depth.

The `NULL` controls come from the store (read-only), because the census holds no `NULL` rows. The
rule prefers cheap rows.

**`QuadSource` is recomputed in-process**, as `main.py` computes it at this tree. The read-back is
formed as the factory forms it, with no production change
([`scripts/rebuild.py`](05/wider/scripts/rebuild.py)).

**Scripts** ([`scripts/`](05/wider/scripts/)):
- `worker.py`, `harness.py`, `rebuild.py` and `dispatch.py` are 04's harness, adapted;
- `run_wider.py` and `launch.py` are the registered run;
- `l0.py` is gate L0;
- `analyse.py` scores the run.

**Outputs:**
- [`out/L0.txt`](05/wider/out/L0.txt) and [`out/L0_final.txt`](05/wider/out/L0_final.txt);
- [`out/wider_05.txt`](05/wider/out/wider_05.txt), which holds every unit, every group and the
  scores;
- [`out/cmd/`](05/wider/out/cmd/), every command with its output.

#### Gate L0

Taken on `low-G+r-1` (2, 38302) and `ctrl-low-1` (2, 2095), under $p$. Everything was compared by
`==` and `float.hex`.

| | Number | Verdict |
|---|---|---|
| **(a)** `QuadSource` at $p$ against the stored row | 360 of 360 stored nodes bit-identical in all nine value columns; 3 nodes added, the switch node and its 2 overlap nodes, which is exactly B1's; none removed. The same holds in all 109 units of the run | holds |
| **(b)** two in-process runs of the same unit | the result, every group call and every region array (by bytes) identical, on both rows | holds |

L0 was re-run with the final scripts after one fix to the worker's bookkeeping, and gave an
identical body ([`out/L0_final.txt`](05/wider/out/L0_final.txt)).
[`state/predictions.md`](05/wider/state/predictions.md) was written at 01:57:36Z, after L0 held and
before the 02:00:05Z launch.

#### Verdicts, scored as written

| | Rule | Number | Verdict |
|---|---|---|---|
| **L1** | every depth-20 row of the sample topped by `G+r`, `G−r` or lone `G`: under $c$ at 1e-7, $p$'s scored groups each < 10 regions at depth ≥ 18, and regions ≤ $p$'s / 10 | 9 of 24 rows. Clause 1 holds on all 24. Clause 2 fails on 15: 14 rows not capped under $p$, where $c/p$ is 0.73–1.09, and R5 at 0.345 | **fails** |
| **L1q** | the same, on the `Gqr`-topped rows | 1 of 7 (R7, $c/p$ = 0.0016). The 6 failures are the drawn `Gqr` rows, none capped under $p$: $c/p$ 1.00–1.35. None has a region at depth ≥ 18 under $c$ or $p$ | **fails** |
| **L2** | every control: $\lvert{\rm total}_c - {\rm total}_p\rvert \le$ `total_abserr`$_p$ | 8 of 8; largest ratio 0.652 (`ctrl-low-2`); the two `NULL` controls bit-identical | **holds** |
| **L3** | every depth-20 row with both $c$ units in budget: $\lvert c(10^{-7}) - c(10^{-8})\rvert \le$ `total_abserr`$_c(10^{-7})$ | 29 of 31; fails on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), neither capped under $p$; holds on all 11 rows capped under $p$ (largest 0.166) | **fails** |

No unit went over budget.

#### The rows capped under $p$ (not scored; the reading the user asked for)

$p$ is B1's code without 05's change. 11 of the 31 depth-20 rows are still at depth 20 under it.
The other 20 are not capped under $p$, R2 among them (05's `issues.md` W2).

| row | class | band | $p$: depth, regions | $c$ 1e-7: depth, regions | $c/p$ | $p$'s scored groups (regions at depth ≥ 18, $p$ → $c$) | L1 clause 1 | clause 2 |
|---|---|---|---|---|---|---|---|---|
| `low-G+r-3` | `G+r` | low | 20, 1340 | 8, 33 | 0.0246 | `G` 1199 → 0 | holds | holds |
| `low-G-r-2` | `G−r` | low | 20, 15068 | 7, 42 | 0.0028 | `G` 11741 → 0 | holds | holds |
| `low-G-r-3` | `G−r` | low | 20, 6822 | 8, 40 | 0.0059 | `G` 1671 → 0; `G−r` 4840 → 0 | holds | holds |
| `low-G-1` | `G` | low | 20, 3902 | 11, 34 | 0.0087 | `G` 3676 → 0 | holds | holds |
| `low-G-3` | `G` | low | 20, 831 | 7, 39 | 0.0469 | `G` 689 → 0 | holds | holds |
| R1 | `G+r` | mid | 20, 22852 | 8, 63 | 0.0028 | `G` 8908, `G+r` 6597, `G−r` 6075 → 0 | holds | holds |
| R3 | `G+r` | mid | 20, 10457 | 5, 18 | 0.0017 | `G+r` 9635 → 0 | holds | holds |
| R4 | `G−r` | mid | 20, 42830 | 8, 83 | 0.0019 | `G` 7545, `G+r` 15539, `G−r` 17307 → 0 | holds | holds |
| R5 | `G−r` | mid | 20, 55 | 5, 19 | 0.345 | none (`G+r` has 7) | holds | **fails** |
| R6 | `G` | mid | 20, 5130 | 8, 28 | 0.0055 | `G` 3115, `G+q+r` 1725 → 0 | holds | holds |
| R7 | `Gqr` | mid | 20, 16660 | 8, 27 | 0.0016 | `G` 11393, `G+q+r` 4647 → 0 | holds | holds |

**On these rows, the change removes the cap everywhere.**
- No group of any capped row keeps a region at depth ≥ 18 under $c$, at 1e-7.
- The maximum depth under $c$ is 5–11 at 1e-7, and 5–11 at 1e-8.
- L1's rule holds on 9 of 10 and L1q's on 1 of 1. R5 misses the tenfold clause because its cap
  under $p$ was one chain of about 50 regions (05's `issues.md` W4).
- L3's rule holds on all 11.

**Cost on these rows.** The $p$ wall time is 0.4–262 s per row, 644 s in all; under $c$ it is
0.2–0.55 s per row, 4.4 s in all. The budgets were never approached.

#### Descriptive

- **$\lvert{\rm total}_c - {\rm total}_p\rvert$ / `total_abserr`$_p$** (04's N4, wider, now with the
  modulus correction too):
  - on the capped rows: 0.0225–0.634, and 2.48 on `low-G-3` (05's `issues.md` W6);
  - on the uncapped rows: ≤ 0.932.
  - At $p$, R3, R6 and R7 are within their error: 0.406, 0.403 and 0.17. At the pre-B1 tree, 04
    found 1.16, 1.53 and 1.94.
- **By band:**

  | Rows | Capped under $p$ | Regions, $p$ → $c$ | Wall, $p$ → $c$ |
  |---|---|---|---|
  | Low band, drawn | 5 of 12 | 28,054 → 288 | 101 s → 4.5 s |
  | Mid band, drawn | 0 of 12 | 352 → 321 | 3.7 s → 3.5 s |
  | R1–R7 (mid) | 6 of 7 | 97,999 → 250 | 545 s → 2.4 s |

  The controls are unchanged: 42 regions under both trees.
- **`total_converged`:**
  - On the depth-20 rows it is False under $c$ on 21 of 31 at 1e-7 and 19 of 31 at 1e-8.
  - On the capped rows, 9 of 11 stay unconverged although the cap is gone (05's `issues.md` W7).
  - 3 rows have an infinite `total_abserr` under both trees.
- **B1 against the stored rows** (05's `issues.md` W1): $\lvert{\rm total}_p - {\rm total}_{\rm stored}\rvert$
  exceeds the stored `total_abserr` on 36 of 39 rows. The ratios are 5.2 to 1.5e5. The three rows
  within error are `low-Gqr-3` (stored error infinite) and the two `NULL` controls (bit-identical).
