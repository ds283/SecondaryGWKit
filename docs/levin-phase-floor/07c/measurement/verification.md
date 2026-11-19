# levin-phase-floor 07c, Part A — the measuring agent's part of VERIFICATION.md §7c

Mac Studio (mps026298), Python 3.13.16. The worktree is `SecondaryGWKit-lpf07c-2fd979f`, at the dispatch SHA `2fd979fd`,
whose code is 07b's (`5e19c4b6`, `AdaptiveLevin/levin_quadrature.py` `8956c7b6…`). Every command is in `out/cmd/` and
`out/commands.log`, under the record number given here. These measurements were taken on this tree. A later re-run adds a
subsection and leaves this one as written.

## 7c.A.1 The tree and the gate

- **The worktree was unchanged.** `git --no-optional-locks status --short --ignored` gave no lines at every check,
  the last at 06:18Z on 19 Nov. The main checkout stayed at `28ebaf07`.
- **The gate held before and after.** Records 01 and 47 hash 07b's census, 07b Part A's records, R31's records and A3 v2's
  scratch copy: `out/gate/gate_hashes_{before,after}.json`. After the run every pair is equal: census 151/151, Part A
  166/166, R31 6/6, scratch copy 5/5. The census checkpoint is `396784bc…`, and no file is extra or missing. The
  after-file's SHA-256 is `81a26513…`.
- **The code** is recorded by record 02.

## 7c.A.2 Section 2.1: captures and replays at the tree

- **Captures** (records 06 and 11; `out/captures/`):
  - The 198 chase calls each equal 07b's census record bit for bit, 198 of 198.
  - R5's 43 calls each equal it bit for bit, 43 of 43.
- **Replays at the tree** (run `-base-`, record 17; `out/replays/tree_summary_base.json`): 454 of 454 reproduce their
  census record (value, abserr, regions, depth, converged and refusal counts, in hex), with rc 0 and no problem listed.
  - The 454 are:
    - the 198 chase calls and their 8 counterparts at the other rtol;
    - 07b Part A's 187;
    - R31's record, against 07b's rehearsal capture (deviation 4);
    - R5's 43 and their 17 counterparts.
- **The instrument** `levin_quadrature_07c_instr.py` (`7c0a5b86…`) reproduces 454 of 454 too
  (`out/replays/instr_summary_base.json`).

## 7c.A.3 Section 2.4: the variants

- **Each variant's diff against the tree** is in `out/variants/<v>.diff`, and its SHA-256 is in `out/variants/variants.json`.
  Each was registered and smoke-tested (records 34–36).
- **Each variant was replayed on all 454 calls** (run `-variants-`, record 37), with rc 0 on all 454.
  - The calls that do not reproduce their census record are exactly the calls `variant_reach_07c.py` finds touched:
    - S 328 (126 reproduce);
    - S2 311;
    - N 438;
    - Eg 394;
    - Ea 404;
    - SEg 405;
    - SEa 411.
- **Each instrumented copy `<v>_i` equals its plain variant bit for bit** on all 259 calls it ran
  (`out/d1/variants_descent.json` `_identity`, record 38): 259 of 259 for each of the seven.
- **An untouched call is bit-identical to the tree under every variant**, in its replay and its region arrays' SHA-256
  (`out/variants/reach.md`, record 40).

## 7c.A.4 Section 2.5: the references

- **The seeds were written before any draw.** `state/seeds_07c.json` (`05f428f3…`) was written at 10:24:13Z on 18 Nov:
  R4 707304, R5 707305, D2 707302.
- **The selections were saved before any reading or solve:** `state/selection_r4.json`, `selection_r5.json` and
  `selection_d2.json`.
- **The rules were written before any reading:** `state/refs_rule.txt`, `r4_rule.txt` and `r5_rule.txt`.
- **The first run took no reading** (`levin-phase-floor-07c-refs-20261118T112553`, state killed, 0 of 123; deviation 23).
- **The second run** (`levin-phase-floor-07c-refs2-20261118T235323`) read all 123, each with exit 0, in 6 h 22 min
  wall-clock, 23:53:23Z to 06:15:38Z (`out/refs2_results.json`, `ok` true).
  - The projection was 9.85 h at three at a time, or 11.83 h with its 1.2 margin.
  - The jobs' summed wall time is 19.09 h, against 29.56 h projected: 0.65 of the projection.
- **What the 123 readings cover:** 61 integrands (R4 30, R5 30, R31's record 1), each with split FCC at resolutions 1
  and 2.
  - QAWO was read at both resolutions for the 31 integrands projected at 30 min or less: R4 13, R5 17 and R31's record.
  - The other 30 have split FCC only.
  - By call, 61 of the 121 calls this Part's references serve have all four readings, and 60 have split FCC only.
- **Every call that one of this Part's references serves meets the scoring rule** (spread ≤ min(abserr)/100): 121 of
  121.
- **The 6 referenced calls not scored** are 07b's own, as in 07b: `a:R2-27`, `a:R2-29` and `a:R3a-0-33666-a13`, each at
  both rtols.

## 7c.A.5 Section 2.6: the scores

- **The scorer:** `score_07c.py` v2 (`579ed08e…`; the v1 correction is deviation 20), record 46.
- **The outputs:** `out/tables/scores_07c.json` (`f2f10ea3…`) and `scores_07c.md` (`5dfa4282…`).
- **Every headline number in `log.md`'s Result line was checked against these tables** by hand, against the JSON, on
  19 Nov:
  - table 4's unconverged column;
  - the qualification;
  - R5's 43 of 43 and 17 of 17;
  - R2-28's ratios;
  - table 5's ratios.

## 7c.A.6 The registry

`python -m runregistrykit list` at 06:18Z on 19 Nov showed every 07c run with a recorded end:

| run | state | readings or units |
|---|---|---|
| `-capture-chase-20261118T101845` | done | 1/1 |
| `-capture-r5-20261118T105127` | done | 1/1 |
| `-base-20261118T110242` | done | 2/2 |
| `-refs-20261118T112553` | killed | 0/123, by its own backstop |
| `-variants-20261118T113152` | done | 14/14 |
| `-refs2-20261118T235323` | done | 123/123 |

None is running or stale.
