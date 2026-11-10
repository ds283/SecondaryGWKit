### 7.C The census at 07's code (prompt 07 §6)

On the Mac Studio (mps026298), 2026-11-08 to 2026-11-10, by Part B's census agent. The log is
[`07/census/log.md`](07/census/log.md).
- **Tree and store:** the census ran from the worktree `SecondaryGWKit-lpf07-e182bc7` at Part B's resumed dispatch SHA
  `e182bc7a`, code `adffa21b`'s (`AdaptiveLevin/levin_quadrature.py` `4e5da371…`). No variant was installed.
  - Gate V0's K0 run ran at the parent worktree `SecondaryGWKit-lpf07-91472db` (`91472dbe`, `beaf0e23…`, 06m's code).
  - Both read 06's certified scratch copy of A3 v2, read-only.
- **Constraints:** Ray-free, no store built, nothing written in a tree. The lock was held for the census and the
  references, on the orchestrator's direction.

#### The Gate and the inputs

Commands and outputs are `07/census/out/cmd/01`–`09`.
- **Trees:** the three 07 worktrees are clean with `--ignored`; the main checkout is at `28ebaf07`, clean.
  - `git diff --stat 74600350 e182bc7a` (outside docs and prompts) touches `levin_quadrature.py`, two AdaptiveLevin test
    modules and `CLAUDE.md`.
- **Registry and lock:** 66 runs, none running or stale; the lock free, no Ray process.
- **The scratch copy:** EQUAL to `copy_files_certified.txt` and OK against 06e's `a3v2_copy_after.sha256`.
- **06m's census** (`levin-phase-floor-06m-census-20261106T102153`): 150 of 150 files equal its `KEPT_ON_MAC_STUDIO.txt`,
  checkpoint `ea5bc3ab…` ([`07/census/out/runs_hashed_before.txt`](07/census/out/runs_hashed_before.txt)).
- **The predictions:** [`07/census/state/predictions.md`](07/census/state/predictions.md), written at 14:10:10Z on
  2026-11-08, after V0 and before the census, SHA-256 `92e22911…`. It holds §6's predictions and §10's last bullet
  verbatim, V4's 24 calls, and how each rule is read.

#### The harness

06m's census `scripts/`, copied, changed where marked "prompt 07". The diff is
[`07/census/out/scripts_diff_at_stop4.txt`](07/census/out/scripts_diff_at_stop4.txt).
- **Per call:** §1.5's diagnostics with Rule A's `location_skipped_phases`.
- **Per returned leaf:** the window and guard flags.
- **The scan:** every group call at both rtols and every analytic call at 1e-7.
- **The comparison:** 06m's, with the new keys set aside.
- **The census's run:** 06a's over-budget pass.
- **V4:** the captures, against 07's census.

#### Gate V0 — holds

- **(a)** `git diff --stat 48cf87c0 91472dbe` touches only a test module and `CLAUDE.md`. None of the 102 files of p the
  K0 workers loaded is among them ([`07/census/out/v0/v0a_imports.txt`](07/census/out/v0/v0a_imports.txt)).
- **(b)** Run `levin-phase-floor-07-k0-20261108T135240`: 100 of 100 K0 units at p equal 06m's census. That is 2,110 calls
  and 2,110 region records (32,518 leaves), 0 differing
  ([`07/census/out/k0/compare_v0b.json`](07/census/out/k0/compare_v0b.json)).
- **(c)** `g-1e-8-1-2174` (z12's row, one root located) was run twice at g and is identical
  ([`07/census/out/v0c/compare_det.json`](07/census/out/v0c/compare_det.json)).
- **Also, descriptive:** the same 100 units at g. 2,037 calls that locate and guard nothing equal 06m's; 73 calls locate.

#### The census

Run `levin-phase-floor-07-census-20261108T171137`: `done`, 18,560 of 18,560, exit 0, 0 errors, 0 over budget.
- Wall 22,708 s against a projection of 21,458 s; the backstop reported and stopped nothing.
- Checkpoint `03dfe786…`. All 16 workers used `4e5da371…`, with no variant and no Ray.

#### V1–V5

[`07/census/out/scores/summary.json`](07/census/out/scores/summary.json),
[`07/census/out/stop3/compare_census_calls.json`](07/census/out/stop3/compare_census_calls.json),
[`07/census/out/v4/report.md`](07/census/out/v4/report.md).
- **V1 holds:** 284,514 calls that locate and guard nothing, and 16,104 units made only of such calls, all equal to 06m's
  bit for bit, at both rtols.
- **V2 holds:** 0 returned Levin leaves hold a sign change (182,066 calls scanned).
- **V3 holds:** 0 violations in 170,582 scored calls; 11,484 set aside under §10.
- **V4 fails:** 28 of 33 honest.
  - z00, z12 and Z3's seven are honest on 06g's references.
  - 19 of the 24 are honest on references at g: FCC on all 24, QAWO on 13, as the user ruled.
  - v08, v09, v12, v17 and v22 are 2.2–3.2× their abserr. Each is converged, with its window stopped by `iroff1` while
    leaves still fell.
- **V5 fails:** 23 calls that locate a root hold 2–54 leaves at `depth_max`.

#### Descriptive and the cross-check

In the log.
- 06m was honest on 17 of the 33 V4 calls.
- The cross-check (interim, beyond §6, the user's request) is in
  [`07/census/out/xcheck/report.md`](07/census/out/xcheck/report.md): 843 calls where 07 and 06m differ beyond their
  errors, 786 of them undecided (class c).

#### After

- **The scratch copy:** EQUAL to its certification, and before equals after (`07/census/out/cmd/95`–`97`).
- **06m's census run directory:** all 150 files equal before and after (`98`, `99b`;
  [`07/census/out/runs_hashed_after.txt`](07/census/out/runs_hashed_after.txt)).
- **Trees:** g, p and `-7460035` clean with `--ignored`; the main checkout clean.
- **Registry and lock:** 69 runs, all finished; the lock free (`101`).
