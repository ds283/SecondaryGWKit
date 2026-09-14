# Log 08 — Re-take prompt 19's measurement: does the $T_k$ sector still need `BREAK_POINT_ALL`?

**Prompt:** prompts/qcd-background-audit/08-per-sector-policy-remeasure.md
**Commit:** *(this commit)* — Re-measure the numeric break-point policy on the corrected background
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS — **state (a) was found**, in prompt §1's words: with an
accurate background the $T_k$ sector converges at all 50 QCD wavenumbers under *either* policy
(worst 7.08e-09 under `BREAK_POINT_ALL`, 8.85e-09 under `BREAK_POINT_DISCONTINUITY`, zero
offenders in both, against 1.97e-07 and three offenders when prompt 19 measured it), the
per-sector distinction is vestigial on today's cosmology, and **the constants stay**. No
production value changed and no number moved; two call-site comment blocks rewritten, one
measurement script and its output added. Three deviations, all `IMPLEMENTATION CHOICE`, none
`UNINTENDED DRIFT`, none touching a README §2 design fact.

## What shipped

### `docs/qcd-background-audit/per_sector_policy_remeasure.py` — new, 1,493 lines

A **reduction** of `docs/gktk-remedial/tk_numeric_atol_sweep.py` as that file stood at `c2bf596`,
copied rather than extended because it belongs to another campaign (prompt §2's instruction; the
precedent is `docs/qcd-background-audit/generate_qcd_references.py`, prompt 02, and
`[13-scoped-run-driver-k-grid-literal]`). **`docs/gktk-remedial/tk_numeric_atol_sweep.py` is
untouched** — `git diff` does not name it.

Everything between the two `copied verbatim ... BEGIN/END` banners is that file's text, byte for
byte, extracted by line range rather than retyped: the imports and configuration (`:79-156`), the
`_Wavenumber`/`_KExit`/`_Proxy` stand-ins, `geometry`, `run`, `x_local`, `sample_errors`,
`summarise`, the exact-radiation oracle, the markdown helpers and `reproduce_control`
(`:162-478`); the $G_k$ half — `_SmoothCosmology`, `_UnsplitModel`, `gk_geometry`, `run_gk`,
`SECTORS`, `sector_errors`, `ACCEPTANCE_DRIFT = 3.4e-8` and the two production `atol`s
(`:1015-1156`); and prompt 19's `SECTOR_POLICY`, `SECTION_9_GK_*`, `TIMING_REPEATS = 5`,
`TIMING_K_INDEX = 38`, `_bitwise_equal` and `_matches_to_printed_precision` (`:1758-1803`). The
reduction drops §§1–8's `atol` sweep and prompts 18 and 19's entry points and reporting, which
this prompt does not re-take.

New below the second banner:

- `PROMPT_19_SECTION_10` — prompt 19's §10 figures, which "unchanged" is measured against. The
  worst drifts are §10.1's; the **grid totals are sums of §10.6's per-wavenumber right-hand-side
  evaluation counts** (401,677 / 429,178 / 1,576,030 and 637,138 / 640,213 / 666,011), which are
  integers and do not depend on the machine.
- `PROMPT_19_TK_OFFENDERS` — the three QCD $T_k$ wavenumbers that were above the criterion under
  `BREAK_POINT_DISCONTINUITY` when prompt 19 measured them (§10.6 indices 13, 23, 37), with its
  drift for each under each policy.
- `load_average() -> str`, `both_policies_sweep(name, model, cosmology, sector, declares) -> dict`,
  `unsplit_sweep(name, model, cosmology, sector) -> dict`,
  `policy_timings(models) -> list`, `_worst(result, policy) -> tuple`, and the reporters
  `report_declared_set`, `report_acceptance`, `report_unsplit`, `report_offenders`,
  `report_control`, `report_cost`, `report_shift`, `report_detail`, `main_remeasure() -> None`.

The difference from prompt 19's `per_sector_sweep`, which measured each sector under its own
policy, is that `both_policies_sweep` measures **both policies on both sectors**: prompt §2 items
1 and 2 ask for a $G_k$ column under `BREAK_POINT_ALL` that has never been taken.

### `docs/qcd-background-audit/PER-SECTOR-POLICY.md` — new, 499 lines

Sections 1–8 are the script's verbatim stdout with the cosmology banners the model constructors
print stripped from the top; the one-paragraph answer above them is written by hand, on the
precedent of `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`, and the document's own comment says so.

### `ComputeTargets/TkNumericIntegration.py:424-465` and `GkNumericIntegration.py:391-416`

The `break_point_kind=self.BREAK_POINT_KIND` call-site comment blocks, rewritten to carry this
prompt's measurement in place of prompt 19's (prompt §3). **Neither `BREAK_POINT_KIND` value
moved**: `TkNumericIntegration.BREAK_POINT_KIND` is still `BREAK_POINT_ALL` (`:118`) and
`GkNumericIntegration.BREAK_POINT_KIND` is still `BREAK_POINT_DISCONTINUITY` (`:107`). No other
line of either file changed; `git diff --stat` is 47 and 33 lines, all of them comment.

The $T_k$ block now says in terms that the justification has changed: prompt 19 chose the value
because it was *necessary*, this prompt measures that it is not, and it is retained because it
costs +0.99 % of the evaluations rather than +220 % and because the mechanism has to exist for an
equation of state that declares more non-smooth points than this one does — "the per-sector
distinction is, on today's cosmology, vestigial". It records that changing it is not free
(2.86e-04 of the envelope on QCD, and the value is in a lookup key) and names README §7 D5 as the
user's. The $G_k$ block records that its reason is unchanged and its numbers better (3.67e-09
under both policies against 8.41e-09 under its own before), and that the *cost* half of prompt
19's argument has evaporated with the set: +0.57 %, where it was +155 %.

**`T_Z_REPRESENTATION_VERSION` is 5 before and 5 after.** No `BREAK_POINT_KIND` value changed, so
prompt §3's condition for bumping it is not met, and nothing in this commit moves a number.

## Deviations from the prompt

### 1. A third column that is neither policy — IMPLEMENTATION CHOICE

Prompt §2 lists two policies. §2b of the measurement document adds a third: the same drift with
the cosmology's declaration **suppressed altogether**, via prompt 18's own `_UnsplitModel`, which
came with the copied harness.

- **Why.** The board entry assigned to this prompt,
  `[04-unsplit-tk-run-now-meets-the-criterion]`, is a measurement of exactly that quantity at
  **one** wavenumber (1.0213e-06 on prompt 04's tree, 2.2767e-08 on prompt 05's, at
  $k = 4.972\times10^7$), and its "next step" is this prompt. Answering it without the column
  would have meant reasoning from the two-policy table rather than measuring.
- **What it cost.** 180 s of a 744 s run; QCD only, both sectors.
- **What it found**, which is why it was worth taking: unsplit, QCD $T_k$ is above the criterion at
  **19 of the 50** wavenumbers, worst **9.61e-06** at $k = 4.223\times10^7$. The board entry's
  wavenumber is one of the ones that now passes (2.91e-08 here, against 2.28e-08 on prompt 05's
  tree — the same statement), but it is **not representative**. Splitting at the equation of
  state's genuine jumps is load-bearing; only the distinction between the two *kinds* has become
  vestigial. Had this column not been taken, this log would have narrowed `[04-…]` on one
  wavenumber and been wrong about the general case.
- **The alternative** was to leave it and record the question. Rejected because the prompt's own
  §1 asks which of three states of the world holds, and a reader who cannot see the unsplit column
  cannot tell state (a) — "the knots were a proxy for the representation's defect" — from the much
  stronger and false claim that nothing needs splitting.

### 2. Which comment block "the `BREAK_POINT_KIND` comment block" is — IMPLEMENTATION CHOICE

Each file has two comments about `BREAK_POINT_KIND`: one above the class constant (`Tk:105-118`,
`Gk:94-107`), which says why the policy is declared once rather than three times and defers the
value to the call site, and one at the `numeric_with_phase_cut` call site, which argues the value
and carries prompt 19's numbers. Prompt §3 says to rewrite "both `BREAK_POINT_KIND` comment
blocks to carry *this* prompt's measurement in place of prompt 19's" — so the block that carries a
measurement is the one meant, and **only the two call-site blocks were rewritten**. The two
class-constant blocks are untouched: they contain no measurement, their statement that the value
is argued at the call site is still true, and editing them would have been change for its own sake.

### 3. The wall-time rows are reported as ratios against a null control — IMPLEMENTATION CHOICE

Prompt §2 item 4 asks for "right-hand-side evaluations **and wall time** per object". The
evaluation counts are reported as the measure and the seconds are reported beside them, each with
the one-, five- and fifteen-minute load averages at which it was taken, because this machine was
under heavy external load for most of the session (load average above 140 earlier in the day;
4.6–8.4 during the run that produced the shipped document). Absolute seconds are therefore **not**
compared with prompt 19's, which were taken in a different session on a quiet machine. What is
compared is the *ratio* between the two policies, timed back to back in the same process, against
the four smooth-model rows where the two policies are literally the same computation and the
ratio is therefore exactly 1 by construction. Those read 1.028, 0.985, 1.024 and 1.024, so ±3 %
is the noise floor; QCD reads 1.001 ($T_k$) and 0.984 ($G_k$), i.e. nothing, consistent with the
+0.99 % and +0.57 % in evaluations. **No conclusion in this log rests on a wall-clock comparison.**

## Verification performed

Everything below is from one run,
`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/per_sector_policy_remeasure.py`,
**744 s**, 50 wavenumbers × 3 models × 2 sectors × 2 policies plus the QCD-only unsplit column,
emitted as `docs/qcd-background-audit/PER-SECTOR-POLICY.md`. **The full matrix prompt §2 asks for
was run; nothing was reduced.** Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2, load average
4.6 / 8.4 / 10.2 at the start and 8.0 / 11.0 / 11.9 at the end. An earlier, identical-in-form run
under heavier load (37.3 / 22.4 / 29.8 at the start) reproduced every drift and every evaluation
count in the tables below **to each digit printed**, which is the internal check that the
measurement does not depend on the machine.

### Prompt §4's acceptance table

| Quantity | Prompt 19's figure | Measured | Verdict |
|---|---|---|---|
| QCD $T_k$ worst drift, `BREAK_POINT_ALL` | 8.72e-09 (with 404 knots) | **7.08e-09** at $k=7.105\times10^5$, median 1.96e-09, **0** of 50 above the criterion | — |
| QCD $T_k$ worst drift, `BREAK_POINT_DISCONTINUITY` | 1.97e-07, **3** of 50 above | **8.85e-09** at $k=1.387\times10^5$, median 1.96e-09, **0** of 50 above | — |
| QCD $G_k$ worst drift, `BREAK_POINT_DISCONTINUITY` | 8.41e-09 | **3.67e-09** at $k=1.608\times10^6$, 0 of 50 above | ≤ 3.4e-08 ✔ |
| QCD $G_k$ worst drift, `BREAK_POINT_ALL` | *(never measured)* | **3.67e-09** at the same $k$, 0 of 50 above | ≤ 3.4e-08 ✔ |
| Radiation / LambdaCDM, both sectors, both policies | — | **bit-identical**, and the grid totals are prompt 19's integers | ✔ |
| $T_k$ RHS evaluations per QCD object | ~31.5k under `all`, ~9.8k under jumps | **8,986** under `all`, **8,897** under jumps | — |

### Which state of the world — **(a)**

Prompt §1's three states, answered with the numbers that decide them:

- **(a) holds.** QCD $T_k$ converges at all 50 wavenumbers under **both** policies. The three
  wavenumbers prompt 19's decision actually turned on now read, with the jumps alone,
  **8.89e-10** ($k=8.366\times10^5$, was 6.1e-08), **2.03e-09** ($k=4.287\times10^6$, was
  2.0e-07) and **5.60e-10** ($k=4.223\times10^7$, was 3.5e-08) — factors of 69, 98 and 63 without
  any change to the integrator. The knots were a proxy for the representation's own defect.
- **(b) does not hold.** There is no wavenumber, in either sector, at which the declared set is
  insufficient: 0 of 50 above the criterion in all four (sector, policy) columns on QCD. The new
  representation does not need to declare knots, and G1's claim does not narrow.
- **(c) does not hold.** $G_k$ has moved, but *downwards*: worst 8.41e-09 → **3.67e-09**, median
  2.05e-09 → **2.16e-10**, 0 of 50 above the criterion before and after. The wavenumber carrying
  the worst moved from $6.034\times10^5$ to $1.608\times10^6$ because the whole distribution fell
  by an order of magnitude and a different member of it is now the maximum — at prompt 19's
  $6.034\times10^5$ the drift now reads **6.0e-11**, 140× better (§8's per-wavenumber table).
  Nothing in prompts 04–07 did more than it was supposed to; a better background gives a
  better-converged Green's function.

### Prompt §2 item 3 — the control, and it is bit-identical

`RadiationModel` and `LambdaCDMModel` declare nothing, so both policies reach the same
single-`solve_ivp` call. Measured, not assumed:

| sector | model | max segments | prompt 19 worst drift | now | prompt 19 grid evals | `all` | `discontinuity` |
|---|---|---|---|---|---|---|---|
| Tk | RadiationModel | 1 | 4.21e-11 | **4.21e-11** | 401,677 | **401,677** | **401,677** |
| Tk | LambdaCDMModel | 1 | 5.70e-11 | **5.70e-11** | 429,178 | **429,178** | **429,178** |
| Gk | RadiationModel | 1 | 1.94e-11 | **1.94e-11** | 637,138 | **637,138** | **637,138** |
| Gk | LambdaCDMModel | 1 | 2.10e-11 | **2.10e-11** | 640,213 | **640,213** | **640,213** |

and at **all 50** wavenumbers of each the production run under one policy is `==` to the run under
the other, sample by sample plus the evaluation count (`_bitwise_equal`). Prompt 17's two control
figures reproduce under both policies: **2.53e-06 in 7,403** evaluations and **2.56e-04 in 8,483**,
the same numbers and the same integers under each. The module default is still
`BREAK_POINT_DISCONTINUITY`: the run with the argument omitted is bit-identical to the run with it
named at all 50 wavenumbers of all six (model, sector) pairs. **No control number moved.**

### Prompt §2 item 4 — the cost

Right-hand-side evaluations per object, averaged over the 50-wavenumber grid — the reproducible
measure, and the one this campaign has used elsewhere:

| sector | model | prompt 19, `discontinuity` | prompt 19, `all` | now, `discontinuity` | now, `all` | `all` / `discontinuity` |
|---|---|---|---|---|---|---|
| Tk | RadiationModel | 8,034 | 8,034 | 8,034 | 8,034 | +0.00 % |
| Tk | LambdaCDMModel | 8,584 | 8,584 | 8,584 | 8,584 | +0.00 % |
| **Tk** | **QCDModel** | **9,843** | **31,521** | **8,897** | **8,986** | **+0.99 %** |
| Gk | RadiationModel | 12,743 | 12,743 | 12,743 | 12,743 | +0.00 % |
| Gk | LambdaCDMModel | 12,804 | 12,804 | 12,804 | 12,804 | +0.00 % |
| **Gk** | **QCDModel** | **13,320** | **13,320** | **13,343** | **13,419** | **+0.57 %** |

What prompt 07 changed, stated plainly: a QCD $T_k$ object under `BREAK_POINT_ALL` costs **8,986**
evaluations where it cost **31,521** — a **3.51× reduction**, and 1.0 % above what the jumps-only
policy costs rather than 220 % above it. The QCD $G_k$ figures move by +0.17 % and +0.74 % of
prompt 19's because the background itself moved (prompts 04–07); that model is allowed to move and
the two that are not did not.

Wall time, best of 5 single-core runs at $k=4.972\times10^7$/Mpc, **reported as a ratio against a
null control** (deviation 3): smooth-model rows 1.028 / 0.985 / 1.024 / 1.024, QCD $T_k$ **1.001**,
QCD $G_k$ **0.984**, at load averages 8.0–8.4. The policy difference is invisible in wall time,
which is what +0.99 % predicts.

### Prompt §2b — the column beyond the prompt (deviation 1)

With `integration_break_points` suppressed altogether:

| sector | worst drift | at $k$ | median | above 3.4e-08 | at $k=4.972\times10^7$ | evals/object |
|---|---|---|---|---|---|---|
| Tk, QCD | **9.61e-06** | $4.223\times10^7$ | 1.80e-08 | **19 of 50** | 2.91e-08 | 8,788 |
| Gk, QCD | 3.52e-09 | $1.608\times10^6$ | 2.26e-10 | 0 of 50 | 4.33e-11 | 13,320 |

So the $T_k$ sector still needs the *jumps*, emphatically — 19 offenders and a worst three orders
above the criterion — and `[04-unsplit-tk-run-now-meets-the-criterion]`'s single wavenumber is one
of the 31 that pass. The $G_k$ sector, interestingly, does not need them either: its declared
split buys 3.67e-09 against 3.52e-09 unsplit, i.e. nothing. That is recorded, not acted on.

### Prompt §2 item 5's consequence for the datastore

The two policies still give different numbers on QCD: the production run under
`BREAK_POINT_DISCONTINUITY` scored against the one under `BREAK_POINT_ALL` differs by up to
**2.86e-04** of the envelope in $T_k$ (median 4.55e-08, worst at $k=3.139\times10^5$) and
**2.11e-08** in $G_k$ (median 4.25e-10). On the two models that declare nothing it is identically
zero. So changing either `BREAK_POINT_KIND` would invalidate every stored QCD numeric object in
that sector, which is why README §7 D5 is the user's and why this prompt did not decide it.

### The suites

| suite | before (`c2bf596`) | after |
|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK**, 0.57 s |
| `ComputeTargets/tests` | 359 OK | **359 OK**, 153.3 s |
| `LiouvilleGreen/tests` | 143 OK on the fast set | **143 OK**, 14.5 s |

The `LiouvilleGreen` run is the **fast set**: every module except `test_3bessel_analytic`, which
is the ~1,400 s one, invoked by name rather than by `discover`. It was not run; no file this
commit touches is reachable from it. No count falls and no test changed expectation — this commit
changes comments, adds a `docs/` script and adds its output.

`black --check` clean on `docs/qcd-background-audit/per_sector_policy_remeasure.py`,
`ComputeTargets/TkNumericIntegration.py` and `ComputeTargets/GkNumericIntegration.py`.
`git diff --stat` names those three plus `docs/qcd-background-audit/PER-SECTOR-POLICY.md`, this
log, the board and `docs/OPEN_ISSUES.md`. **Not** `docs/gktk-remedial/tk_numeric_atol_sweep.py`,
**not** `Quadrature/integrators/numeric_with_phase_cut.py`, **not** `Datastore/`, **not** any
tolerance, Gauss order or `main.py`.

## Observations not acted on

1. **The JSON's `convergence` block was not regenerated, and this prompt could not do it.**
   `[01-convergence-block-has-a-separate-generator]` names prompt 08 as the natural place to
   re-run `docs/gktk-remedial/residual_convergence.py`, and one tolerance is still owed on that
   account: `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` in `ComputeTargets/tests/test_background_tau.py`,
   measured 1.418851e-04 at `T_120_MEV`. **Prompt 08's "Files you may touch" does not include
   `ComputeTargets/tests/wkb_reference_data.json` or `ComputeTargets/tests/test_background_tau.py`,
   and its §4 acceptance table contains no tolerance row**, so regenerating the block and taking
   that constant back is out of scope here (README §5 rule 5). The figure is unchanged from what
   prompt 07 re-measured (1.418851e-04 at `T_120_MEV`, 1.728034e-05 at `T_LO`, 1.060594e-06 at
   `EOS_T_LO`) because nothing in this commit moves a number. `[02-qcd-reference-floor]` on the
   `GkTk-remedial` board says the same thing about the recorded floors. **Next step:** whichever
   prompt next has the JSON and that test module in scope — prompt 09 is the obvious candidate,
   since it is the close-out and already re-scores the consumers. The issue is re-pointed there on
   the board rather than closed.
2. **The $G_k$ sector's own declared split buys nothing either** (§2b): 3.52e-09 unsplit against
   3.67e-09 under `BREAK_POINT_DISCONTINUITY`, i.e. within the noise of the measure, at 13,320
   evaluations against 13,343. That is not an argument for removing the split — it costs 0.17 %
   and it is the same mechanism the $T_k$ sector genuinely needs — but it is a fact about this
   sector that no document previously recorded. Opened as
   `[08-gk-declared-split-buys-nothing-measurably]` on the board.
3. **`docs/qcd-background-audit/measure_T_z_representation.py` §5's prose is still wrong**
   (`[09-audit-script-section-5-prose-counts-the-wrong-set]`). `docs/qcd-background-audit/` **is**
   in this prompt's file list, but that script is not "a measurement script and its output" for
   *this* prompt, and correcting a sentence in another prompt's reproduction is exactly the scope
   creep README §5 rule 5 forbids. Left, with the issue unchanged.

## State handed to the next prompt

**The state found is (a), in prompt §1's words: "The knots were a proxy for the representation's
own defect."** With an accurate background the $T_k$ sector converges at all 50 QCD wavenumbers
under either policy — worst drift 7.08e-09 under `BREAK_POINT_ALL`, 8.85e-09 under
`BREAK_POINT_DISCONTINUITY`, median 1.96e-09 either way, zero offenders in both — where prompt 19
measured 1.97e-07 and three offenders with the jumps alone. The per-sector distinction is
**vestigial on today's cosmology** and the constants stay: `TkNumericIntegration.BREAK_POINT_KIND
= BREAK_POINT_ALL` and `GkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_DISCONTINUITY`, both
unchanged, both re-argued in their call-site comments. **`T_Z_REPRESENTATION_VERSION` is 5, the
value prompt 07 left**, and nothing in this commit moves a number: no bump was due and none was
made.

**README §7 D5 is not discharged, and is not urgent.** It remains the user's, but the decision has
lost its cost argument in both directions: keeping `BREAK_POINT_ALL` costs the $T_k$ sector
**+0.99 %** of its right-hand-side evaluations (8,897 → 8,986 per object), against +220 % when the
knot lattice was declared, so there is nothing to save by changing it — while changing it would
move every stored QCD $T_k$ value by up to **2.86e-04** of the envelope and, because the value is
in the lookup key (`GkTk-remedial` prompt 20), demand a regeneration of that sector and everything
downstream of it. **The recommendation this prompt makes, and does not act on, is to leave both
constants where they are.** If the user decides otherwise, what must be regenerated is every
`TkNumericIntegration` row on a QCD model and the whole transfer chain below it; `GkNumericIntegration`
rows are unaffected, and `LambdaCDMModel` and `RadiationModel` rows are unaffected in both sectors.

**Prompt 09 inherits three things it does not otherwise have.**

1. **The $T_k$ sector's cost fell by 3.51× and prompt 09's throughput tables will show it.** A QCD
   $T_k$ object at production tolerances is **8,986** right-hand-side evaluations under the shipped
   policy where prompt 19 measured **31,521**; the grid total is 449,281 against 1,576,030. That is
   prompt 07's doing, not this prompt's — the cause is that `BREAK_POINT_ALL` returns 3 points
   rather than 407 — and README §6.4's "a moved number with no cause is a stop" is satisfied by
   this sentence. $G_k$ moves by +0.17 % (13,320 → 13,343 per object under its own policy), which
   is the background itself moving, not the policy.
2. **The convergence block is still stale and is now prompt 09's to take** (observation 1).
   `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` in `ComputeTargets/tests/test_background_tau.py` is
   the one tolerance still owed to `[01-convergence-block-has-a-separate-generator]`, measured
   1.418851e-04 at `T_120_MEV` and unchanged by this commit. Re-running
   `docs/gktk-remedial/residual_convergence.py` is what closes it, and this prompt's file list did
   not permit it. `[02-qcd-reference-floor]` on the `GkTk-remedial` board waits on the same run.
3. **Splitting at the jumps is still load-bearing, and the fifty-wavenumber sweep says so where
   one wavenumber did not.** Unsplit, QCD $T_k$ is above the criterion at **19 of 50** wavenumbers,
   worst **9.61e-06** at $k = 4.223\times10^7$. `[04-unsplit-tk-run-now-meets-the-criterion]` is
   therefore **narrowed, not closed**: its premise is false at the wavenumber the test in
   `ComputeTargets/tests/test_numeric_break_points.py` uses ($k = 4.972\times10^7$, where the
   unsplit run reads 2.91e-08 and passes) and **true at nineteen others**. The test's weaker
   assertion (`UNSPLIT_PENALTY_FACTOR = 5.0`) still holds; whoever next has that module in scope
   can restore the original, stronger statement simply by re-pointing it at
   $k = 4.223\times10^7$, where the unsplit drift is 9.61e-06 against a split 5.60e-10 — a factor
   of 17,000. That is a test change and was not made here.

**The measurement is reproducible in one command**, 744 s, no Ray and no datastore:

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/per_sector_policy_remeasure.py
```

Its stdout is §§1–8 of `docs/qcd-background-audit/PER-SECTOR-POLICY.md` (the cosmology banners the
model constructors print are stripped by hand, and the one-paragraph answer is written by hand —
the document's own comment says so). `docs/gktk-remedial/tk_numeric_atol_sweep.py` was **copied,
not edited**, so §§9–10 of `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` remain reproducible from
their own file, and this document is additive: a later re-measurement appends a section.
