# Prompt 08 — Tighten the tests to the acceptance table and revalidate the fixtures

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §8.3 (separating oracle improvement from consumer re-splining), §9 Stage 5, §10 (the acceptance table)
**Reconciliation items:** C4 (the standing derivative gate), §3.3 (the fixture inventory)
**Depends on:** 07 (hard)
**Recommended model:** Opus
**Files you may touch:** `LiouvilleGreen/tests/test_bessel_phase.py`,
`LiouvilleGreen/tests/test_3bessel_analytic.py`,
`ComputeTargets/tests/test_tk_source_functions.py` and
`ComputeTargets/tests/test_phase_groups.py` — **in the latter two, tolerance constants and comments
only** — plus the log and the status board.
**Do not touch:** any production module. This prompt changes no `LiouvilleGreen/` or
`ComputeTargets/` production code. In particular: no fixture *logic*, no protocol, no
`TkSourceFunctions`, no `phase_groups`.

Read first: README §6 (the acceptance table and the required coverage) and §4.2 (the hunk
discipline in `ComputeTargets/tests/`); `RECONCILIATION.md` C4 and §3.3; `DRAFT-PLAN.md` §8.3,
§9 Stage 5, §10; `docs/lg-phase-and-handover-followup-2026-09.md` §2.4; then
`logs/05-…`, `logs/06-…` and `logs/07-…` §"State handed to the next prompt".

---

## 1. Character of this commit

The campaign's numerical claims become assertions. Until now the improvement is measured in new
tests that prompts 01–07 wrote for themselves; this commit puts it into the tests that were already
there, and re-reads the downstream fixtures whose interpretation the improvement changes.

**The trap this prompt exists to avoid.** `DRAFT-PLAN.md` §8.3:

> An accurate Bessel object makes the constant-\(w\) oracle more useful. It does not repair a
> downstream consumer that samples its full phase and then fits another coarse spline in
> \(\log(1+z)\). Re-run the workstream B comparisons with two distinct questions:
> 1. Does the Bessel amplitude–phase object reproduce independently evaluated Bessel functions?
> 2. How much error does `TkSourceFunctions` introduce when consuming the sampled fixture?

Conflating those two would let the campaign claim credit for an improvement it did not make, or
hide a consumer floor behind a better oracle. Keep them separate and report them separately.

## 2. `LiouvilleGreen/tests/test_bessel_phase.py`

Three of its four tests are set where they cannot see the campaign's result. Tighten them to
README §6.

| Test | Now | Target |
|---|---|---|
| `_test_bessel_value` (`:34`) | `REL_DIFF = 0.5` — 50 % | \(E_\theta,E_A\le10^{-11}\) for \(\nu\in\{3/2,5/2\}\), envelope-normalized. **Change the measure**, not just the number: the current test divides by `their_j`, which is meaningless near a zero of \(J_\nu\) (README §6 requires zeros and extrema be covered). Use `bessel_reference.phase_pair_error` |
| `test_high_order` (`:116`) | `relerr < 1e-3`, with the "catch garbage" comment at `:111-113` | \(\le10^{-6}\) for \(\ell\in\{2,20,100,400,1000\}\), i.e. \(\nu\in\{2.5,20.5,100.5,400.5,1000.5\}\). Same measure change. Rewrite the comment: it currently explains a threshold chosen for a construction that no longer exists |
| `test_phase_derivative` (`:139`) | `relerr < 1e-6` at \(\nu\in\{2.5,20.5,100.5\}\), from \(2x_0+1\) | \(\le10^{-9}\) at low order per README §6; keep \(10^{-6}\) at \(\nu\ge20.5\). **And extend the lower bound down to the turning-point interval**: the test currently starts at \(2x_0+1\), excluding the interval where `DRAFT-PLAN.md` §4.7 locates every derivative maximum. That exclusion made the old test easier than the quantity it claimed to measure |
| `test_bessel_J_integral` (`:145`) | `MAX_INTEGRAL_RELERR = 1e-3` | Tighten as far as the measurement supports and **say what limits it**. This is a Levin quadrature result, so the floor may well be the quadrature or the closed-form reference values (which are quoted to 10 digits at `:154-158`), not the Bessel phase. If so, that is the answer — record it and do not tighten past it |

Two standing rules for this file:

- `test_phase_derivative`'s \(10^{-6}\) has been the campaign's regression gate since prompt 01
  (`RECONCILIATION.md` C4). Tightening it is the goal; **loosening it, or any other threshold in
  this file, is a stop condition** (README §4.3).
- The docstring at `:87-92` explains the old `asin(J/m)` initial-condition failure and names the
  non-Limber use case. The failure mode is gone with the ODE. Keep the use-case sentence (it is why
  high orders are tested at all, and README §7 records tightening them as deferred) and rewrite the
  rest.

Add what README §6 requires and the file does not currently cover: **the crossover \(x_\star\)**,
**changes of interpolation interval and phase branch**, **raw and logarithmic input modes with
explicitly matched reference arguments**, and **selected large arguments through \(10^{15}\)**
against `mpmath` at the identical supplied argument. Some of these exist in prompt 05's
`test_bessel_two_region.py`; do not duplicate them — cross-reference in a comment and cover here
only what is genuinely missing from both.

## 3. `LiouvilleGreen/tests/test_3bessel_analytic.py`

Tolerances are `ABS_TOLERANCE = 1e-6`, `REL_TOLERANCE = 1e-5`, `SINGULARITY_ABS_TOLERANCE = 1e-3`,
`SINGULARITY_REL_TOLERANCE = 1e-2` (`:15-19`).

`DRAFT-PLAN.md` §9 Stage 4 warns that an improvement of this size "can expose a different limiting
error rather than simply passing more easily". So: **measure first, then set.** For each closed form,
record the achieved relative and absolute error, then set each constant to a value the measurement
supports with a small margin, and — for every constant that does **not** improve by roughly the
campaign's six-to-eight orders — say what limits it instead. Candidates, in the order they are
worth suspecting:

1. the Levin quadrature's own error, now that `theta_abserr` is declared and honest (prompt 07);
2. `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12`, chosen at a time when "accuracy is set by the phase and
   modulus splines, not by the spectral order" (`three_bessel_integrals.py:53-57`) — a statement
   this campaign has just invalidated. **If the order is now the limit, that is a genuine finding
   and it is not this prompt's to fix**: record it, open a §3 issue on the board, and leave the
   constant alone;
3. the closed-form reference values themselves, several of which are quoted to ~10 digits;
4. genuine near-singular behaviour, for the `SINGULARITY_*` cases.

Prompt 07's log may already have handed you a candidate; check it.

## 4. `ComputeTargets/tests/` — comments and tolerances only

Two fixtures consume `bessel_phase` and both carry comments that are now wrong. **Change comments
and tolerance constants; change nothing else.** Fixture logic, protocols and production code are
the `source-remediation` campaign's (README §4.2), and touching them is a stop condition.

### 4.1 `test_tk_source_functions.py`

- `:228` builds `bessel_phase(self.nu, 1.02 * self.x(z_WKB[-1]))`; `:249` cites
  `bessel_phase.py:270-282` for `J = m sin theta` (line numbers that prompt 05 has changed);
  `:261-266` defines `theta_exact = pi - raw_theta(x)` with a paragraph justifying the rotation.
  **Re-check that sign convention against the new zero-point** (`DRAFT-PLAN.md` §8.3 asks for
  exactly this): the argument at `:261-265` is that \(\vartheta\) increases with \(x\), hence
  decreases with \(z\), so \(\theta=\pi-\vartheta\) increases with \(z\) as the code's
  \(d\theta/dz=+\omega_{\rm eff}\) convention requires, and \(\sin\theta=\sin\vartheta\) keeps the
  amplitude positive. Confirm each clause still holds and say so; the new construction preserves
  \(J=A\sin\theta\) with \(\theta\) increasing in \(x\), so it should — but "should" is not a check.
- `:433` says the comparison against scipy's \(J_\nu\) "carries `bessel_phase`'s own phase-function
  error and so saturates near 2e-6". That was true and is now false. Re-measure and rewrite, and
  **tighten `:449-450`** (`err_M < 1e-8`, `err_T < 1e-7`) only as far as the *fixture's own*
  re-spline error allows — which is question 2 of §1 and is **not** improved by this campaign.
- `:29-30` records the LG truncation error as "~1e-5 in d ln M/dz at 3.5 e-folds sub-horizon
  (measured below, and grid-independent)". Physical LG truncation is distinct from numerical Bessel
  representation error (README §1.1); leave it, and make sure your rewrite does not blur them.

### 4.2 `test_phase_groups.py`

- `:26-28` and `:975` both state the fixture "carries `bessel_phase`'s own accuracy (~x * 1e-8 in
  phase, `docs/lg-phase-and-handover-followup-2026-09.md`)". Re-measure and rewrite.
- `:1415` refers to "the phase re-spline and `bessel_phase`" as combined floors. Separate them now
  that they differ by six orders, and say which dominates.
- Tolerances at `:1273` (`1.0e-10`), `:1297` (`1.0e-6`) and neighbours: tighten only where the
  measurement supports it, and only where the *re-spline* is not the binding term.

## 5. Separating the two questions — the required deliverable

Produce, and put in the log, a table with one row per fixture comparison and three columns:

| comparison | error before | error after | attribution |
|---|---|---|---|

with attribution being one of: **Bessel oracle** (improved by this campaign), **consumer
re-spline** (unchanged by this campaign — the \(h^4\) fit of the sampled fixture on the production
grid), **LG truncation** (physical, unchanged), **quadrature**, or **reference value**. A row whose
"after" barely moved and whose attribution is "Bessel oracle" is a contradiction and means the
attribution is wrong.

This table is what `docs/` inherits in prompt 09, and it is the answer to the follow-up document's
§2.4 claim that every "exact" constant-\(w\) fixture in the codebase has a phase floor of order
\(x\times10^{-8}\).

## 6. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` and
  `-s CosmologyModels/tests` pass — nothing here should touch them, so a failure means something
  strayed.
- Every threshold that was **not** tightened has a comment saying what limits it.
- `git diff HEAD~1 --stat` shows changes confined to the four test files, the log and the board;
  and within `ComputeTargets/tests/`, `git diff HEAD~1 -- ComputeTargets/tests/` shows only
  comment lines and numeric constants.
- Record the suite wall-clock against prompt 01's baseline
  (`docs/bessel-remedial/baseline-2026-09.md`). Tighter tolerances plus \(\nu=400.5\) and
  \(\nu=1000.5\) may make this suite substantially slower; if it exceeds ~30 minutes, say so
  prominently and propose (do not implement) what to skip by default.

## 7. Log and commit

Follow README §5 and §5.1. The log must contain:

- the §5 attribution table, complete;
- every threshold changed, old → new, with the measured value that justifies it;
- every threshold **not** changed, with what limits it;
- confirmation of the `theta_exact = pi - vartheta` convention check (§4.1), clause by clause;
- whether `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` is now a limiting factor, as a number, with a §3 issue
  opened on the board if it is;
- the suite runtime against the baseline.

Commit subject, or something equally specific: `Tighten the Bessel tests to the new accuracy`.
