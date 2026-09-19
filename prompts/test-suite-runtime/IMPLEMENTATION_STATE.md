# Test-suite runtime — implementation state

**Last updated:** 2026-09-19 · **Status: COMPLETE — 1 / 1 prompts landed.** Measured on `75db3c5`;
**prompt 01 landed at `07c6041`**, which took `LiouvilleGreen.tests.test_3bessel_analytic` from
**1121.5 s to 9.4 s** and the whole suite from ~22 min to **149.4 s**, with no tolerance, assertion
or test method changed. README §5's acceptance is met. No issue is open in §3; the one this
campaign closed belonged to `transfer-remedial` and is recorded in §4.

**This campaign is retrospective** — the code commit preceded the paperwork commit. See README §4
and §5 note 3 below.

**Campaign:** [`README.md`](README.md)

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Gate the three-Bessel diagnostic plots](01-gate-the-three-bessel-diagnostic-plots.md) | **T1** | Opus | ✍️ retrospectively | ✅ | `07c6041` *"Put the three-Bessel convergence figures behind a flag"* | [`logs/01-…`](logs/01-gate-the-three-bessel-diagnostic-plots.md) |

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| T1 | **DEFECT** | `test_3bessel_analytic` spends its wall clock drawing figures, not asserting. `plot_and_compute_3Bessel` evaluated a 250-point `logspace` grid of full three-Bessel integrals per case — **42.5 s**, against **0.14 s** for the single evaluation at `max_x` that the assertion reads, a factor of **304** — and ran 47 times per run, writing 110 files. `test_YJJ_log_scaling` contained **no assertion at all** and did 40 more evaluations plus four figures. The module was **1121.5 s of a 1321 s suite, 84.9 %**. The impact is verification, not convenience: a suite this slow does not get run, which is how `[07-abserr-bounds-truth-is-now-an-unexpected-success]` survived three prompts. | 01 | ✅ **Done, 2026-09-19.** Grid and figures behind `THREE_BESSEL_DIAGNOSTIC_PLOTS`, off by default; `seaborn` and `matplotlib` imported inside the plotting functions; `test_YJJ_log_scaling` `skipUnless` the same flag, so it reports **skipped** instead of a pass that could not fail; `mu_phase`/`nu_phase` hoisted out of `test_YJJ_log_singularity`'s eps sweep. Module **1121.5 s → 9.4 s**; suite **149.4 s**, 749 tests OK across four packages. Test set identical to `HEAD` by `ast` comparison, so `LiouvilleGreen` is **148** as before with one skipped; `test_abserr_bounds_truth`'s seven true-error figures are unchanged and bound on 7 of 7 as before. Diagnostic path exercised, not assumed: with the flag set, `test_YJJ_log_scaling` runs and 10 PDFs are written. One file in the diff. |

## 3. Active and unresolved issues

None. Prompt 01 opened no issue.

## 4. Resolved issues

- **[08-3bessel-plot-cost-dominates-the-suite]** *(opened by `prompts/transfer-remedial` prompt 08,
  2026-09-10; **closed by this campaign's prompt 01**, 2026-09-19)* — the issue belongs to
  `transfer-remedial`'s board, where its full statement and resolution live; it is repeated here
  only because this campaign exists to close it. Prompt 08 was allowed to propose and not to
  implement, and its proposal — gate the plot grid behind an environment variable or module flag
  defaulting to off, and decide whether `test_YJJ_log_scaling` belongs in `unittest` discovery — is
  what shipped, with the variable chosen over the flag and `skipUnless` chosen over relocation to
  `docs/` (log §2 gives the reasons for both). Re-measured before acting rather than taken on
  trust: prompt 08's figure was 21.2 min for `test_YJJ_log_singularity` alone, and the whole module
  measured **1121.5 s** on `75db3c5`. **Closed.** Removed from `docs/OPEN_ISSUES.md` §4 in the same
  commit as this board.

## 5. Standing notes

1. **`THREE_BESSEL_DIAGNOSTIC_PLOTS=1` is how the convergence figures are obtained**, and anyone
   re-measuring the three-Bessel oracles wants it set. It restores the 250-point grid in
   `compute_3Bessel` *and* un-skips `test_YJJ_log_scaling`, at roughly the old cost: the module
   returns to ~19 min. In particular, whoever takes on
   `[08-3bessel-chebyshev-order-is-now-the-limit]` on `transfer-remedial`'s board — the finding that
   `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` now binds the two $(0,0,0)$ oracles — will want the
   figures, and they are one environment variable away rather than gone.

2. **The suite's head is now `ComputeTargets`**, 122.3 s of 149.4 s, with `test_quadsource_integral`
   52.5 s and `test_source_grid` 21.3 s the two largest modules. **This is not known to be a
   defect**, and it is recorded as a starting point rather than as an issue: no test module outside
   `test_3bessel_analytic` imports `matplotlib` or calls `savefig` anywhere in the repository, so
   there is no second instance of the same pathology to find. Anyone opening a runtime question
   should score against the per-module baseline in log §"Verification performed" and establish
   whether that cost is diagnostic before treating it as removable.

3. **The revert boundary here is two commits, not one.** `07c6041` carries the code; the paperwork
   — this board, README, prompt, log and the `docs/OPEN_ISSUES.md` edit — is a second commit,
   because the change was made in answer to a direct request before the campaign existed. CLAUDE.md
   invariant 3 asks for one commit, and the log classifies this as `UNINTENDED DRIFT` rather than
   as a choice. Reverting the code alone leaves a board claiming a landed prompt.

4. **A lint nit, since fixed.** `set_xlabel("$\epsilon$")` at the two `test_YJJ_log_scaling`
   sites was an invalid escape sequence and made every import of the module emit
   `SyntaxWarning: invalid escape sequence '\e'`. It predated `07c6041`. Log 01 recorded it as an
   observation not acted on and this note as left in place; the author asked for it fixed, and an
   `r` prefix landed in a follow-up commit. `r"$\epsilon$"` and `"$\epsilon$"` are the same string
   — `\e` is not a valid escape, so Python already kept the backslash — so the axis label is
   unchanged and the module now imports clean. *(Log 01 says "four sites": there are two. The four
   warnings were those two lines in each of the two file versions that were parsed.)*

5. **Baselines at `75db3c5`:** AdaptiveLevin **32**, CosmologyModels **39**, LiouvilleGreen **148**,
   ComputeTargets **530**, all OK. The known flake is
   `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`, a wall-clock
   assertion — re-run that module alone before attributing a failure to a commit, and note that it
   is *more* likely to pass now that the suite is not competing with a 19-minute module.
