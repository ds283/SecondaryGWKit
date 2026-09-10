# Log 02 — Pin the SciPy Bessel domain boundaries as tests

**Prompt:** prompts/transfer-remedial/02-domain-boundary-tests.md
**Commit:** *(this commit; SHA not self-embedded)* — Pin the SciPy Bessel domain boundaries as tests
**Model:** Claude Sonnet 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

New file `LiouvilleGreen/tests/test_scipy_bessel_domain.py` (unittest module,
`TestScipyBesselDomain`, 7 tests), pinning the four SciPy/Amos facts `RECONCILIATION.md` C1 and §1
measure, with the module docstring stating plainly that these tests are about SciPy, not this
repository, and that a failure means the supported domain must be re-derived, not that a bound
should be loosened.

- `test_hankel1e_returns_exact_negative_zero_and_is_finite` — Fact 1. Asserts
  `hankel1e(100.5, 1e9) == -0j`, `np.isfinite(...)` is `True`, `log(abs(...))` is `-inf`,
  `angle(...)` is `-0.0`.
- `test_hankel1e_usable_boundary_low_order` / `_high_order` — Fact 2. Helper
  `_contiguous_usable_boundary(values_at, nu, x_hi=1e16, n=6000)` reproduces the prompt's §3 script
  exactly (geometric grid from `1.5*sqrt(nu^2-1/4)` to `1e16`, 6000 points, finite-and->1e-3
  contiguous-from-below boundary) and is asserted within a factor of 2 of 2.247e15 at
  \(\nu\in\{0.5,2.5,20.5\}\) and of 7.13e8 at \(\nu\in\{100.5,1000.5\}\).
- `test_crossover_stays_far_below_hankel1e_boundary` — the README §2(e) consequence, asserted
  directly: `boundary / (100*nu) >= 1e3` at every supported order; prints the worst-case ratio.
- `test_jv_yv_rhs_identity_good_below_boundary` / `_bad_above_boundary` — Fact 3. The exact identity
  `(2/pi)/(x*(J_0.5^2+Y_0.5^2)) == 1` is asserted to hold to `1e-9` at \(x\in\{10^{12},10^{14},2\times10^{15}\}\)
  and to depart by more than `1e-2` somewhere among five adjacent doubles near `x=5e15`.
- `test_jv_yv_failure_is_not_monotonic_in_x` — Fact 4, with a substituted concrete pair; see
  "Deviations" below.

Two module-level helpers, `_hankel1e_amplitude(nu, xs)` and `_jv_yv_amplitude(nu, xs)`, wrap
`hankel1e`/`jv`,`yv` with `np.errstate(all="ignore")` so the plausibility-band arithmetic (which
deliberately evaluates in the region where these routines misbehave) does not print runtime
warnings.

No production code touched. `IMPLEMENTATION_STATE.md` updated: prompt 02's row (✅, this commit),
progress counter (2/9), and M4/M5 in §2 moved from ⬜ to 🟡 (partially discharged — 03/04 and
03/05/09 respectively still own the rest). §3/§4 unchanged: no issue opened, narrowed or closed by
this prompt, so `docs/OPEN_ISSUES.md` needed no edit (checked — its
`[01-scipy-jv-yv-high-order-boundary]` line already says "not yet a test", which remains accurate;
see "Observations not acted on").

## Deviations from the prompt

### Fact 4's literal example did not reproduce; a different concrete pair was substituted (STRUCTURALLY REQUIRED)

`DRAFT-PLAN.md` §4.4 states `jv`/`yv` "recover at \(10^{16}\) for \(\nu=1000.5\) while failing at
\(10^{10}\)". Measured directly: at \(x=10^{10}\), \(a=\sqrt{\pi x/2}\hypot(J,Y)=0.487462\)
(badly wrong, as the plan says), but at \(x=10^{16}\), \(a=0.516961\) — also far from 1, not a
recovery. The literal pair in the plan does not reproduce as a "fails then recovers" example on
this tree/SciPy version.

The prompt anticipates exactly this ("If it does not reproduce, record it as a deviation... an
assertion that the non-monotonicity exists at some specific \((\nu,x)\) pair is fine... Prefer the
concrete pair"), so this is classified `STRUCTURALLY REQUIRED` rather than an open question: the
plan's specific numbers were not available to build the test on, and the prompt's own fallback
instruction was followed.

I searched a grid of round \(x\) values at \(\nu=1000.5\) between \(10^{10}\) and \(10^{11}\) and
found a clean, decade-local pair: \(x=10^{10}\) gives \(a=0.487462\) (\(|a-1|=0.513\)), and
\(x=3\times10^{10}\) gives \(a=0.999070\) (\(|a-1|=0.00093\)) — badly wrong, then accurate, at a
*larger* \(x\), which is the qualitative claim the campaign needs (a monotone ceiling on `jv`/`yv`
accuracy cannot be assumed). The test asserts `|a-1| > 0.1` at \(x=10^{10}\) and `|a-1| < 1e-2` at
\(x=3\times10^{10}\), both measured and reproduced. The test's docstring records both the plan's
original claim, that it did not reproduce, and the substituted pair, so a later reader does not
have to re-derive this.

No other deviations. Every other numeric claim in the prompt (Facts 1–3, the crossover-margin
consequence) reproduced within the stated or a tighter margin; see "Verification performed" for the
measured values against each stated target.

## Verification performed

Reproduced the prompt's §3 script directly before writing any assertion (all printed values match
`RECONCILIATION.md` C1 and §1 to the stated precision):

```
nu=0.5: hankel1e to 2.248e+15, jv/yv to 1e+16
nu=2.5: hankel1e to 2.247e+15, jv/yv to 1e+16
nu=20.5: hankel1e to 2.247e+15, jv/yv to 1e+16
nu=100.5: hankel1e to 7.128e+08, jv/yv to 6.305e+13
nu=1000.5: hankel1e to 7.143e+08, jv/yv to 1e+16
x=1.0e+12: ['1.000000', '1.000000', '1.000000', '1.000000', '1.000000']
x=1.0e+15: ['1.000000', '1.000000', '1.000000', '1.000000', '1.000000']
x=2.0e+15: ['1.000000', '1.000000', '1.000000', '1.000000', '1.000000']
x=3.0e+15: ['0.988677', '0.948322', '1.028585', '0.978680', '1.033255']
x=5.0e+15: ['1.676372', '1.653327', '1.077128', '1.389636', '1.665364']
x=1.0e+16: ['0.750691', '1.094350', '1.035889', '0.734835', '1.060665']
```

- **Fact 1**: `hankel1e(100.5, 1e9)` → `np.complex128(-0j)`; `== -0j` → `True`; `isfinite` → `True`;
  `log(abs(...))` → `-inf`; `angle(...)` → `-0.0`. Exact match to `RECONCILIATION.md` §1. SciPy
  1.15.2, NumPy 2.2.4, Python 3.12.14 confirmed at test time — same environment the campaign's
  numbers were measured on.
- **Fact 2**: hankel1e usable boundary — 2.248e15/2.247e15/2.247e15 at \(\nu\in\{0.5,2.5,20.5\}\)
  (target 2.247e15, within the asserted ×2 margin, in fact within 0.1%); 7.128e8/7.143e8 at
  \(\nu\in\{100.5,1000.5\}\) (target 7.13e8, within 0.2%).
- **Crossover margin**: measured ratio boundary/\(x_\star\) at \(\nu=1000.5\) (the worst case) is
  **7139**, against the required `>=1000` and the prompt's own cited "at least \(10^3\)
  (\(7.13\times10^8\) vs \(10^5\))" — consistent (`7.13e8/1e5=7130`).
- **Fact 3**: good-below values agree with 1 to six places at \(x\in\{10^{12},10^{14},2\times10^{15}\}\),
  well inside the `1e-9` tolerance (deviation is exactly `0.0` to six printed digits, i.e. below
  float printing resolution — the identity is exact there). Bad-above: worst departure from 1 among
  the five adjacent doubles near \(x=5\times10^{15}\) is `0.676372` (at the first offset), far past
  the required `>1e-2`.
- **Fact 4**: see "Deviations" — measured `|a-1|=0.513` at \(x=10^{10}\) (`>0.1`, asserted) and
  `|a-1|=0.00093` at \(x=3\times10^{10}\) (`<1e-2`, asserted).
- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_scipy_bessel_domain -v` — **7
  tests, OK, 0.034 s** (well under the ~30 s target).
- Per-module runs (the discovery run is not attempted whole, per board note 13 — it does not finish
  in 50 min, dominated by `test_3bessel_analytic.py`, which this prompt does not touch):
  `test_bessel_phase` OK, `test_bessel_reference` OK, `test_range_reduce` OK (4 tests), `test_three_bessel`
  OK, `test_scipy_bessel_domain` OK (7 tests, 0.034 s) — all exit 0, matching prompt 01's recorded
  per-module baseline.
- `./venv/bin/python -m black --check LiouvilleGreen/tests/test_scipy_bessel_domain.py` — failed
  once (one file would be reformatted), ran `black` to reformat, `--check` then passed; re-ran the
  test module afterward to confirm the reformat did not change behaviour (still 7 tests, OK).

## Observations not acted on

- `[01-scipy-jv-yv-high-order-boundary]`'s recorded "Next step" (`IMPLEMENTATION_STATE.md` §3) says
  "prompt 02 pins both boundaries for `jv`/`yv` as well as for `hankel1e`, and narrows the order
  threshold if it is cheap to do so." Prompt 02's actual text (§2, Facts 1–4) does not ask for a
  `jv`/`yv` contiguous-usable-boundary test at high order analogous to Fact 2's `hankel1e` table, or
  for narrowing the \([85.5, 88.5]\) order-threshold bracket — only Fact 3's RHS-identity check
  (at \(\nu=0.5\) only) and Fact 4's non-monotonicity pair (at \(\nu=1000.5\)) touch `jv`/`yv`. I
  followed the prompt's actual text rather than the board issue's anticipatory note, per the
  instruction to execute the prompt exactly. The issue remains open and unchanged; I did not edit
  it or `docs/OPEN_ISSUES.md`, since its current wording ("the order threshold is bracketed
  [85.5, 88.5], not pinned, and not yet a test") is still accurate after this commit. A future
  prompt (04, which the board says calibrates its plausibility band against
  `scipy_reference_max_x(nu)`) is the natural place to decide whether that narrowing is worth doing,
  since it is the first prompt after this one to depend on the order threshold rather than merely on
  the `hankel1e` boundary this prompt pins.
- The reproduction script in the prompt's §3 iterates `nu` only over
  `(0.5, 2.5, 20.5, 100.5, 1000.5)` for the boundary table, and I kept the same set for the shipped
  tests (`DRAFT-PLAN.md` §12.2 additionally includes `50.5`, which the prompt's own §3 script drops).
  Not acted on — the prompt's §3 script, not §12.2's, is what this prompt asks to be reproduced, and
  `50.5` adds no new information for the four facts being pinned.

## State handed to the next prompt

- New test module: `LiouvilleGreen/tests/test_scipy_bessel_domain.py`,
  `TestScipyBesselDomain`, 7 tests, ~0.03 s. No new public names outside the test module (two
  private helpers, `_contiguous_usable_boundary`, `_hankel1e_amplitude`, `_jv_yv_amplitude`, are
  module-local and not part of any public surface).
- Pinned numeric bounds, for prompt 04's plausibility band and declared domain (calibrate against
  these rather than re-measuring):
  - `hankel1e` usable-boundary (finite and amplitude `>1e-3`, contiguous from below): **~2.247e15**
    at \(\nu\in\{0.5,2.5,20.5\}\); **~7.13e8** at \(\nu\in\{100.5,1000.5\}\). Tests assert these
    within a factor of 2 — treat that factor as the margin already "spent" by this pin, not as
    slack still available.
  - Crossover margin: `boundary / (100*nu) >= 1e3` at every supported order, measured worst case
    (at \(\nu=1000.5\)) **7139**.
  - `jv`/`yv` RHS identity `(2/pi)/(x*(J^2+Y^2))`: good to `1e-9` of 1 up to \(x=2\times10^{15}\)
    (at \(\nu=0.5\)); departs by `>1e-2` among adjacent doubles near \(x=5\times10^{15}\).
  - `jv`/`yv` non-monotonic recovery pair at \(\nu=1000.5\): \(a=0.487462\) at \(x=10^{10}\)
    (`|a-1|=0.513`), \(a=0.999070\) at \(x=3\times10^{10}\) (`|a-1|=0.00093`) — usable if a later
    prompt wants a second worked non-monotonicity example without re-deriving one.
- `LiouvilleGreen.tests.bessel_reference.SCIPY_REFERENCE_MAX_X` (2.0e15),
  `SCIPY_REFERENCE_HIGH_ORDER_NU` (85.5) and `SCIPY_REFERENCE_HIGH_ORDER_MAX_X` (7.13e8) are now
  regression-gated by this commit's tests (Fact 2 pins the same numbers `scipy_reference_max_x`
  encodes) — a future change to those constants should keep this test's tolerances in mind, and a
  future SciPy upgrade that fails this module is the trigger to revisit both together.
- The order threshold between "hankel1e usable to 2.247e15" and "usable to 7.13e8" remains
  bracketed at \([85.5, 88.5]\) and is not itself a test (see "Observations not acted on"). Prompt
  04's plausibility band should use `scipy_reference_max_x(nu)`'s existing 85.5 cutoff rather than
  assuming it is tight.
