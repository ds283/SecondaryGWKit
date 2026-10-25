## 4. The floor removed

**Prompt 04, run 2026-10-25 on the Mac Studio.** The code ran from a git worktree at `98e68d3`, the tree 03 ran at. The records are in the main checkout at `c8ca155` (`handover-remedial`, clean). There was no store write, no Ray and no lock (route (a), as in §3).

**The run.** `levin-phase-floor-04-remove-20261025T231110` ended `done`, 28/28 units, exit 0. Its checkpoints and pickles stay in that run directory on the Mac Studio.

**The store** was byte-identical at every check: step 2, the stop, the resume, before launch, after the run, and after the fingerprint ([`before`](04/state/store_files_before.txt), [`after`](04/state/store_files_after.txt)). `store fingerprint`, run without `--write`, gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` and `problems: none` ([`out/fingerprint_04.txt`](04/out/fingerprint_04.txt)).

**A stop, and an amendment.** The first dispatch stopped at step 5, where E1 (c) and E2 failed at round-off: 4e-16 to 6.5e-14 rad, against an F of 1.7e-6 to 9.2e-5 ([`out/phases_04_first.txt`](04/out/phases_04_first.txt)).
- The bounds left out two round-off terms.
  - The first is the abscissa's own rounding, z = expm1(x), at about ε|dθ/du|. Predicted alone, it matches X's residual with a correlation of 0.94–1.00 ([`out/diag_abscissa_rounding_04.txt`](04/out/diag_abscissa_rounding_04.txt)).
  - The second is the ulp of arm A's unreduced remainder.
- The user amended the two gates (prompt §8, commit `d170e6d`). The scored predictions were left unchanged, and nothing had been re-run.

- **Scripts** (all in [`scripts/`](04/scripts/)):
  - [`phases.py`](04/scripts/phases.py): the arms P, A and X as one dispatcher at `PrimitivePhase.theta_mod_2pi`, with Δ^ext in mpmath at 512 bits, Φ^arm and F^arm;
  - [`check_phases.py`](04/scripts/check_phases.py): step 5, the gates E1 and E2;
  - [`run_remove.py`](04/scripts/run_remove.py) and [`launch.py`](04/scripts/launch.py): the registered run;
  - [`analyse.py`](04/scripts/analyse.py): step 8, from the checkpoints;
  - [`harness.py`](04/scripts/harness.py), [`rebuild.py`](04/scripts/rebuild.py) and [`prov.py`](04/scripts/prov.py): 03's route (a), with both trees' provenance.

  Each output's header records the SHA-256 of the script that wrote it, and both trees' `HEAD`s.
- **Outputs:**
  - [`out/phases_04.txt`](04/out/phases_04.txt): E1, E2, the reproduction of 03, F^arm per group, A1's coverage, σ_φ and the modulus;
  - [`out/remove_04.txt`](04/out/remove_04.txt): every unit, every group, and the scores.
- **Predictions** were fixed in [`state/predictions.md`](04/state/predictions.md) at 23:10:41Z, before the 23:11:10Z launch.

### 4.1 The gates

| | Number | Verdict |
|---|---|---|
| **E1 (a)** | worst \|float(Δ^ext) − delta()\| / 2 ulp = 1.000, over 80,000 abscissae | holds |
| **E1 (b)** | worst \|wrap(ρ^X − ρ^P)\| / (4εΦ^P + 1e-15) = 0.870 | holds |
| **E1 (c)**, amended | worst \|wrap(ρ^A − ρ^X)\| / (4εΦ^A + 4·2⁻⁴⁹) = 0.379 | holds |
| **E2**, amended | 29/29 scored groups; σ_θ / bound 0.22–0.34 (P), 0.10–0.77 (A), 0.05–0.20 (X); X's σ_θ ≤ 6.5e-14 ≪ 1e-2 F | holds |
| **M0** | P bit-identical to the stored row on 7/7 rows (`analytic_rad` too; only `elapsed` differs) | holds |

- **ρ^X is exact.** It equals the correctly rounded exact remainder at all 80,000 E1 abscissae (checked with `fractions`).
- **P reproduces 03.** It reproduces 03's printed medians of σ_θ and σ_θ/F in all 29 groups, to 03's digits.

### 4.2 What removing the floor does

| row | P regions | X regions (1e-7) | X/P | P wall | X wall | most at depth ≥ 18 in a scored group, X | A = X? |
|---|---|---|---|---|---|---|---|
| R1 | 34,147 | 133 | 0.0039 | 181.8 s | 1.5 s | 0 | yes |
| R2 | 29,782 | 89 | 0.0030 | 196.9 s | 1.2 s | 0 | yes |
| R3 | 33,314 | 66 | 0.0020 | 219.7 s | 0.8 s | 0 | yes |
| R4 | 52,909 | 167 | 0.0032 | 321.6 s | 2.1 s | 2 | yes |
| R5 | 26,273 | 130 | 0.0049 | 181.0 s | 1.9 s | 3 | yes |
| R6 | 22,147 | 89 | 0.0040 | 124.3 s | 1.4 s | 3 | yes |
| R7 | 23,338 | 88 | 0.0038 | 136.5 s | 1.4 s | 3 | yes |

**With the jitter gone, the cap goes.**
- No group under X or A at rtol 1e-7 reaches depth 20.
- The few regions at depth 18–19 are in `Gqr` groups on R4–R7.
- No X or A unit came near its budget: 0.4–3.0 s, against budgets of 373–965 s.

**Layer 1(a) alone does what the exact limit does, on these rows.** A, which takes kτ_R out of G's phase and leaves T's remainders as production forms them, gives the same region count as X on every row, group by group. The totals agree to ≤ 1.2e-12 relative. That includes the `Gqr` groups, whose predicted F^A_g (a maximum over the sub-interval) is 1.1–1.3 F (04's `issues.md` I7).

**At rtol 1e-8**, X reaches depth 20 in a single short chain: 2 regions at depth 20 per affected group, 4–10 per row, on R1 and R4–R7 (04's `issues.md` I6).

**The phase that is left** ([`out/phases_04.txt`](04/out/phases_04.txt), descriptive). These errors are smooth, and P, A and X share them:
- **The modulus error of G** is 0.175 F on every row.
- **G's residual φ is noisy.** In the stored samples it carries noise at 0.15–0.21 F (median), rebuilt in mpmath from the stored div/mod. Production's quantised φ_G reaches 0.46–0.78 F in its worst windows.
- So X is the jitter-free limit, not the truth (04's `issues.md` I2).

### 4.3 Verdicts

| | Rule | Number | Verdict |
|---|---|---|---|
| **R1** | under X at 1e-7, every scored group < 10 regions at depth ≥ 18, and regions ≤ P/10; on ≥ 6 of 7 rows | 7/7 rows; X/P 0.0020–0.0049; at most 3 regions at depth ≥ 18 | **holds** |
| **A1** | under A at 1e-7, every scored group with F^A_g ≤ 1e-8 has < 10 regions at depth ≥ 18 | 14 covered groups (every scored lone `G` and `G±r`); 0 at depth ≥ 18 in all 14 | **holds** |
| **N4** | \|total_P − total_X(1e-8)\| ≤ total_abserr_P on all seven | ratio 0.440, 0.0795, **1.161**, 0.245, 0.445, **1.533**, **1.942** (R1–R7) | **fails** on R3, R6, R7 |
| **R2** | \|total_X(1e-7) − total_X(1e-8)\| ≤ total_abserr_X(1e-7) on all seven | ratio 0.405, 2.1e-4, 0, 0.305, 8.9e-7, 0.474, 2.5e-3 | **holds** |

**Q5's number,** \|total_X(1e-7) − total_X(1e-8)\| / \|total_X(1e-8)\|, beside the 2026-09 sweep's 1.9e-6 (mid band), 2.84e-6 (low) and 8.3e-7 (seam):

| R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|
| 5.36e-7 | 3.16e-9 | 0 | 6.73e-6 | 1.41e-11 | 8.21e-8 | 3.04e-10 |

**Findings §2.6 item 3 is answered.** Its refutation clause, "item 3 leaves the depth at 20", does not fire: the depth falls from 20 to at most 19, and the region count to under 0.5 % of P's.

**Its second prediction, that `total` moves by less than the row's old declared error, holds on four rows only.** That prediction is N4's rule.
- It holds on R1, R2, R4 and R5.
- It fails on R3, R6 and R7, where the stored `total_abserr` understates the jitter's effect by 1.16–1.94×. In relative terms that is 2.7e-4 (R3, a cancelling total of 7.0e-29) and 6.6–7.3e-7 (R6, R7).

**Q5 measures self-consistency.** With the floor gone, the cost of `rtol = 1e-7` against `1e-8` is at most 6.7e-6 relative (R4), and below 1e-6 on six rows. It does not bound the shared φ error.
