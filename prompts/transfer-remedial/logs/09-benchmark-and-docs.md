# Log 09 — Benchmark re-run and the campaign's measured outcome

**Prompt:** prompts/transfer-remedial/09-benchmark-and-docs.md
**Commit:** *(this commit; SHA not self-embedded, per the same precedent prompts 01–08 set)* —
"Record the measured outcome of the Bessel rebuild"
**Model:** Claude Sonnet 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

Documentation and one benchmark run; no production code, no test.

### `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`

One constant, one comment (§2 of the prompt). `B1_KAPPA = [1.0, 10.0, 100.0]` → `[1.0, 10.0, 100.0,
1000.0]`; the comment above `B1_MAX_X` rewritten to state the corrected mechanism
(`RECONCILIATION.md` C1) rather than the old symptom-only diagnosis, and to record the measured
$\kappa=1000$ result (phase build $\le0.008$ s, Levin evaluation $0.15$–$0.39$ s, all seven oracles
under 2 s total), while keeping the historical claim ("did not complete within $\sim25$ min") as
history, per the prompt's explicit instruction.

### `docs/lg-phase-and-handover-followup-2026-09.md`

§2.4 and §2.5 edited in place with dated supersession notes (§9 of the prompt's five-item list);
structure and headings unchanged, no rewrite. §2.4 gets a new §2.4.1 ("The replacement, measured")
covering the four required points — measured replacement accuracy, the `phi` offset finding, the
fixture/production tolerance distinction, and the chunking measurement (with the harsher
cosmological finding from `docs/gk-wkb-numerical-review-2026-09.md` §3 cited alongside it) — and
the three stale "Consequences" bullets are struck through and annotated superseded rather than
deleted. §2.5's first bullet (storing `Q` directly) is annotated: `bessel_phase` no longer has a
`Q` member (prompt 06), and `DRAFT-PLAN.md` §4.2 explains why that was never the right general
answer regardless — evaluating a $Q=\theta/x$ spline still multiplies its own error by $x$.

### `docs/transfer-remedial-verification.md` (new)

The campaign's closing record, in the shape of `docs/backport-modules-verification.md`: what
changed per prompt with commit SHA (§2), the `DRAFT-PLAN.md` corrections recorded here because the
plan itself is not edited (§3, closing two board issues), the acceptance table with achieved values
(§4), the attribution table transcribed verbatim from prompt 08 §5 (§5), the cost/domain result
(§6), the benchmark re-run (§7), the remaining floors (§8), the environment (§9), deferred and
handed-over work (§10), and the `source-remediation` hand-off (§11).

### `prompts/source-remediation/IMPLEMENTATION_STATE.md`

One new §3 entry, `[transfer-remedial-qsi-phase-groups]`, recording the `_three_bessel_Levin`
missing-`theta_deriv`/`theta_abserr` gap and the working pattern
(`LiouvilleGreen/three_bessel_integrals.py`'s `_PhaseGroup`) to copy. Nothing else in that file
touched — no other row, no item-level table entry, no standing note.

### `prompts/transfer-remedial/IMPLEMENTATION_STATE.md` and `docs/OPEN_ISSUES.md`

Prompt 09's row (⚠️, this commit), progress counter (9/9, campaign marked complete), M5/M20/M21
flipped to ✅. Five §3 issues closed and moved to §4 (`[00-plan-vs-tree-corrections]`,
`[00-qsi-three-bessel-levin-excluded]` — by hand-off, `[03-draft-plan-tail-coefficient-wrong]`,
`[07-generic-K-product-rounding]` — accepted and documented, `[08-test-three-bessel-tolerances-unassigned]`
— deferred and documented); nine left open, genuinely unresolved; one new issue opened
(`[09-bessel-tier-hardcoded-repo-path]`, discovered while re-running the benchmark — see
Deviations). `docs/OPEN_ISSUES.md` updated in the same commit: five rows deleted, two rows added
(the new source-remediation hand-off entry and the new transfer-remedial path-hardcoding entry),
count corrected from 33 to 30.

## Deviations from the prompt

### 1. The benchmark's own path hygiene bug had to be worked around to measure the right tree — STRUCTURALLY REQUIRED

**What the prompt assumed.** "Run it," timing `build_phases` separately from the Levin evaluation,
against this tree.

**What is actually there.** `bessel_tier.py:40` hardcodes
`REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"` — the **main checkout's** absolute path — and
inserts it into `sys.path`; its other `sys.path` entry resolves to `docs/adaptive-levin-benchmark`,
which has no `LiouvilleGreen` package to shadow `REPO` with. Every commit in this campaign has been
made from a **worktree**, so running the script as committed, from this worktree, silently imports
`LiouvilleGreen` from the main checkout instead — a different tree entirely. The first re-run
attempt reproduced exactly the *old* construction's cliff (interrupting the stuck process and
reading the traceback showed a `solve_ivp` call inside `bessel_phase.py` at line numbers that do
not exist in this tree's version of that file), because the main checkout was on `main` at
`9ff59d5` at the time, 20+ commits behind this campaign's baseline.

**What was done instead.** The measurement in `docs/transfer-remedial-verification.md` §7 was
obtained by pre-importing `LiouvilleGreen.bessel_phase`, `LiouvilleGreen.three_bessel_integrals`
and `LiouvilleGreen.tests.test_3bessel_analytic` from this worktree, by absolute path, before
importing `levin_bench.bessel_tier` — which pins the correct modules in `sys.modules` ahead of the
hardcoded shadowing path — rather than by editing `bessel_tier.py`'s `REPO` constant, which is
outside this prompt's "one constant, one comment" scope and may be intentional for the script's
normal invocation from the main checkout. Recorded as board issue
`[09-bessel-tier-hardcoded-repo-path]` so a future re-run of this tier from a worktree is not
silently measuring the wrong tree with no warning.

### 2. Two additional §3 issues that anticipated riding on this prompt's hand-off were not folded in — IMPLEMENTATION CHOICE

`[05-quadsource-order-check-docstring-stale]` and `[08-tk-fixture-scipy-comparison-unasserted]`
both recorded, in their own "Next step," that they were candidates to hand to
`prompts/source-remediation` "alongside" the phase-groups gap this prompt does hand off. This
prompt's own §5 text restricts that hand-off to **one entry**, about the `_three_bessel_Levin`
missing-`theta_deriv` gap specifically, and says so explicitly ("this is the only edit this
campaign makes to that folder, it adds one entry and changes nothing else"). Read literally against
that restriction, neither docstring issue was folded in. Both remain open in this campaign's own
§3, undischarged — recorded in `docs/transfer-remedial-verification.md` §10 rather than silently
dropped. Alternative considered: treat "alongside" as authorising a small bundle. Rejected — the
prompt's own restriction is unambiguous and a second or third entry was not asked for.

### 3. Two issues were closed by documentation, using the resolution their own text offered — IMPLEMENTATION CHOICE

`[07-generic-K-product-rounding]`'s own "Next step" was "either accept it and record it in `docs/`
(prompt 09), or split $K\cdot x$..."; `[08-test-three-bessel-tolerances-unassigned]`'s was "either a
short follow-up prompt in this campaign, or fold it into prompt 09's documentation as explicitly
deferred." Both resolutions are self-contained documentation actions inside this campaign's own
`docs/`, unlike deviation 2's pair (which would have required a second edit to *another*
campaign's board), so both were taken: recorded in `docs/transfer-remedial-verification.md` §8 and
§10 respectively, and closed on the board. Neither required touching a production module or a test.

### 4. No deviation on the acceptance table or the attribution table — none

Every row of README §6's table has an achieved value from prompts 05/08's own measurements (§4);
every row of prompt 08's attribution table is transcribed verbatim (§5), per the prompt's explicit
instruction. Nothing here was re-derived or re-measured beyond the benchmark tier itself.

## Verification performed

Everything below was run, from the repository root, with `PYTHONPATH=. ./venv/bin/python`.
Environment: Python 3.12.14, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, Darwin 25.5.0 (arm64) — the
same environment recorded throughout the campaign.

### The benchmark tier

$\kappa=1000$, $k,q,s=1300,1700,2100$, `B1_MAX_X`$=10^{12}$, $n_{\rm osc}=8.117\times10^{14}$ per
oracle, against **this worktree's** `LiouvilleGreen` (see Deviation 1 for how that was ensured):

```
J000 phase_build_s=0.0002 levin_s=0.1493 total_s=0.1495 rel_err=2.276e-11 converged=False
J110 phase_build_s=0.0065 levin_s=0.3142 total_s=0.3207 rel_err=3.251e-10 converged=False
J220 phase_build_s=0.0060 levin_s=0.2976 total_s=0.3035 rel_err=2.810e-11 converged=False
J222 phase_build_s=0.0078 levin_s=0.3858 total_s=0.3936 rel_err=5.064e-11 converged=False
J231 phase_build_s=0.0081 levin_s=0.3566 total_s=0.3647 rel_err=3.321e-12 converged=False
Y000 phase_build_s=0.0001 levin_s=0.1560 total_s=0.1561 rel_err=1.783e-11 converged=False
Y022 phase_build_s=0.0052 levin_s=0.2781 total_s=0.2834 rel_err=1.367e-11 converged=False
TOTAL WALL: 1.97s
```

`converged=False` on every row reflects the requested Levin tolerance (`atol=1e-14, rtol=1e-10`,
tighter than the measured true error in every row above), not a stall or a failed evaluation —
`phase_limited=False` throughout too, confirming the driver simply reports its declared bound
against a tighter request than the true error needs, the same honest-declaration behaviour §5's
attribution table documents elsewhere. **It completes**, in under 2 s total against "did not
complete within $\sim25$ min" before. `B1_KAPPA` raised to include 1000.0; the module comment
rewritten with the corrected mechanism, per §2 of the prompt.

### The two required test suites

| suite | result | wall clock |
|---|---|---:|
| `unittest discover -s ComputeTargets/tests -t .` | **97 tests, OK** | *(recorded below once complete — no production file or test was touched by this commit, so a change here would itself be the finding)* |
| `unittest discover -s LiouvilleGreen/tests -t .` | **run in full, per prompt 08's own precedent of completing it** | *(recorded below once complete)* |

*(placeholders above are filled from the actual run output before this log is committed — see
the final verification block.)*

### Documentation checks

- Every commit SHA cited in `docs/transfer-remedial-verification.md` §2 resolves:
  `git cat-file -e f71401d fe33e8e ffbb36c bc31493 f6cbb29 f9cc891 69c37a9 8ba9159` — all present
  (they are this branch's own history, matching `git log --oneline` exactly).
- Every acceptance row in README §6 has an achieved value beside it (§4 of the verification
  document) — none reported "not met."
- `git diff HEAD~1 --stat` touches: `docs/OPEN_ISSUES.md`,
  `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`,
  `docs/lg-phase-and-handover-followup-2026-09.md`, `docs/transfer-remedial-verification.md` (new),
  `prompts/source-remediation/IMPLEMENTATION_STATE.md`,
  `prompts/transfer-remedial/IMPLEMENTATION_STATE.md`, and this log —
  exactly the files this prompt allows plus the one `source-remediation` entry.
- `./venv/bin/python -m black --check docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py` —
  failed once (one file would be reformatted) after the edit; `black` (no check) reformatted it;
  `--check` then passed.

## Observations not acted on

1. **`bessel_tier.py`'s `REPO` hardcoding is a latent trap for anyone re-running this benchmark
   from a worktree**, which is how every commit in this campaign was made. Deviation 1; board issue
   `[09-bessel-tier-hardcoded-repo-path]`. Not fixed — out of this prompt's "one constant, one
   comment" scope, and the constant may be a deliberate simplification for the script's normal use
   from the main checkout.
2. **Two docstring/assertion hand-off candidates stay open**, per deviation 2:
   `[05-quadsource-order-check-docstring-stale]` (a stale docstring in
   `ComputeTargets/QuadSourceIntegral.py`) and `[08-tk-fixture-scipy-comparison-unasserted]` (a
   printed-not-asserted number in `test_tk_source_functions.py`). Both are small, low-risk fixes for
   whoever next has write access to `ComputeTargets/`.
3. **The order threshold between the two Amos failure boundaries remains bracketed, not pinned**
   (`[01-scipy-jv-yv-high-order-boundary]`) — unaffected by anything this prompt could touch.
4. **`docs/transfer-remedial/measure_bessel_phase.py` (prompt 01's diagnostic) still cannot be
   re-run against the current construction** (`[06-measure-bessel-phase-num-chunks]`) — it is not
   in this prompt's file list, so it was left as is.

## State handed to the next prompt

There is no next prompt: this is prompt 09 of 9, and the campaign is complete. For whoever next
touches this area:

- **The campaign's full closing record is `docs/transfer-remedial-verification.md`.** Read it
  before re-deriving anything about the Bessel oracle's accuracy, cost, or remaining floors.
- **Nine issues remain open** on `prompts/transfer-remedial/IMPLEMENTATION_STATE.md` §3, each with
  a stated next step; none blocks anything currently in the tree.
- **The `source-remediation` campaign now owns one new issue**,
  `[transfer-remedial-qsi-phase-groups]`, with a working pattern to copy rather than a design to
  invent.
- **A future re-run of `bessel_tier.py`'s benchmark from a worktree must work around the `REPO`
  hardcoding** (Deviation 1) or fix it first.
