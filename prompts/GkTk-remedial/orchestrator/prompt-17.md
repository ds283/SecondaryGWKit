# Orchestrator prompt — Prompt 17 (the $T_k$ numeric `atol` across the $k$-grid)

You are orchestrating prompt 17 of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

Not a workstream — a measurement prompt closing out Workstream E, approved by the user
2026-09-12. Prompt 12 set `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` on evidence from **one** $k$
($10^6$) on **one** model (`RadiationModel`), and then found that at $k=3\times10^8$ an isolated
$2.56\times10^{-4}$-of-envelope excursion survives near $x\approx10.8$, which `atol = 1e-16`
removes at *fewer* evaluations. So README §6's $T_k$ row is demonstrated at one $k$ and unknown at
the other 49. Prompt 17 measures the production grid on both real backgrounds and recommends; it
changes no production code and takes no decision.

**It runs before prompt 13**, and that ordering is the point: the tolerance is part of every
`TkNumericIntegration` datastore key, so if prompt 13 builds its datastore and runs its scoped
pipeline at `1e-13` and the answer later turns out to be `1e-16`, both are invalidated.

## What to read

The prompt: [`../17-tk-numeric-atol-k-sweep.md`](../17-tk-numeric-atol-k-sweep.md), in full.
`../README.md` §2 (d), §6 (error definitions), §5; `../IMPLEMENTATION_STATE.md` row 12, M22, and
the `[12-tk-numeric-atol-largest-k-excursion]` entry in §3; [`README.md`](README.md);
`logs/12-tk-numeric-atol.md` Verification section and "State handed to the next prompt".

## Preconditions

`git status` clean; row 12 ✅/⚠️ and row 17 ⬜. Baseline **299 tests, OK**. Confirm the fixtures and
the grid the prompt names still exist, since a stale reference wastes the dispatch:

```bash
grep -n "^class RadiationModel\|^class LambdaCDMModel\|^class QCDModel\|^def envelope_relative_error" ComputeTargets/tests/wkb_reference.py
grep -n "NUMBER_SOURCE_K_VALUES" main.py
```

## Dispatching

Standard dispatch text (`workstream-A.md`), with `NN-<name>` = `17-tk-numeric-atol-k-sweep`.
Model: **Opus**.

## Reviewing prompt 17

The five checks of `../README.md` §4.3, plus:

4. Tests: `discover -s ComputeTargets/tests -t .` → **299, unchanged**. This prompt adds no test.
5. **No production code.** `git diff HEAD~1 --stat` must show nothing under `ComputeTargets/`,
   `Quadrature/`, `LiouvilleGreen/`, `config/`, and not `main.py`. In particular
   `config/defaults.py` is untouched — the constant stays at `1e-13` whatever was measured — and
   `ComputeTargets/tests/test_tk_numeric_atol.py` keeps prompt 12's thresholds even if the sweep
   found a $k$ where they would fail. Either would be a stop: the decision is the user's.
6. **The reference is demonstrated converged** (§2.1). The document carries its own table showing
   that tightening the reference a further decade moves it by at least an order of magnitude less
   than the smallest reported difference, at the worst $k$ of each model. **Without this the whole
   sweep is measuring the reference's own error** — treat a missing or failing table as a stop,
   not as a caveat.
7. **The §2.4 control reproduced**: $2.534\times10^{-6}$ at $k=10^6$ and the $2.56\times10^{-4}$
   excursion at $k=3\times10^8$, both on `RadiationModel`, to a couple of significant figures. If
   they did not, the agent should have stopped; if it continued anyway, stop here.
8. **Spike versus level is answered.** For every $k$ whose maximum exceeds $3\times10^{-6}$ the
   document gives maximum, second-largest and median. A report that gives only maxima cannot
   answer the question the prompt was written for.
9. **The floor is respected.** Errors at the $2.5\times10^{-6}$ initial-condition floor (README
   §2 (d)) are identified as such and not presented as fixable by `atol`. An agent claiming an
   accuracy below a floor is a campaign-wide stop (`../IMPLEMENTATION_STATE.md` §5 note 2).
10. **Counts, not wall time**, for the cost comparison (§5 note 14 — this machine's elapsed times
    overstate by up to 53 %).
11. `[12-tk-numeric-atol-largest-k-excursion]` is **narrowed, not closed**, and
    `docs/OPEN_ISSUES.md` updated with the count unchanged.

## Continue or stop

**Stop after this prompt regardless of outcome** and put the recommendation to the user with its
three numbers — the constant is theirs to settle, and prompt 13 should not start until it is.
Stop early on checks 5, 6, 7 or 9, or any campaign-wide condition.

## Completion criterion

Row 17 ✅/⚠️; `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` present; the suite still at 299.
Report: the recommendation and the evidence; whether prompt 13 may build its datastore at
`1e-13`; and that no production code changed, so the tree is still the one prompt 12 left.
