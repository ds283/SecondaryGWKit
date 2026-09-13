# Log 01 — Make `WKB_mod_2pi`'s cycle count consistent with its remainder

**Prompt:** [`prompts/phase-representation/01-wkb-mod-2pi-cycle-count.md`](../01-wkb-mod-2pi-cycle-count.md)
**Commit:** *(this commit)* — Derive the WKB cycle count from the exact remainder
**Model:** Claude Opus 5
**Date:** 2026-09-13
**Baseline:** `9daa2cb` · **Parent:** `7bca2cc`
**Result:** COMPLETE

## What shipped

### 1. `LiouvilleGreen/WKBtools.py` — `WKB_mod_2pi`

One expression changed, plus the comment block stating the error budget:

```python
theta_mod_2pi = fmod(theta, TWO_PI)                            # unchanged, exact
theta_div_2pi = int(round((fabs(theta) - fabs(theta_mod_2pi)) / TWO_PI))
```

replacing `int(floor(fabs(theta) / TWO_PI))`. `fmod` is exact, so
$|\theta| - |\theta_{\rm mod}|$ is mathematically $n\cdot{\tt TWO\_PI}$ for the integer
$n = \lfloor |\theta| / 2\pi \rfloor$ that is wanted, rather than a second, independently rounded
division of the same quantity. Everything downstream of that line is untouched: the sign flip for
$\theta<0$, the `theta_mod_2pi > 0` branch that increments the count and subtracts `TWO_PI`, and
the returned tuple. The negative-remainder convention $\theta_{\rm mod}\in(-2\pi,0]$ is unchanged
(prompt §2 item 2; README §2 (a)) and is asserted by a new test.

The unused `floor` import was dropped from the module's `from math import` line.

**The error budget, as the comment states it.** With $n \approx |\theta|/2\pi$: the subtraction
rounds by at most ${\rm ulp}(|\theta|)/2 \le |\theta|\cdot 2^{-53}$, i.e. $n\cdot 2^{-53}$ cycles
once divided by `TWO_PI`; the division itself rounds by at most $n\cdot2^{-53}$. So `round()`'s
argument is within $n\cdot2^{-52}$ of the exact integer and round-to-nearest recovers $n$ exactly
while $n < 2^{51}$, i.e. while $|\theta| < 2\pi\cdot2^{51} \approx 1.4\times10^{16}$ — beyond which
${\rm ulp}(\theta)\approx2$ rad and the phase has no fractional information left to represent. The
largest $|\theta|$ this code sees is $\approx5\times10^{12}$ ($n\approx8\times10^{11}$), where the
bound is $1.8\times10^{-4}$ cycles. The comment also records why replacing `floor` by `round` on
the *old* quotient is not the fix (prompt §2 item 1): it moves the failure from quotients just
below an integer to quotients just above one.

### 2. `LiouvilleGreen/range_reduce_mod_2pi.py` — `simple_mod_2pi`

The identical construction, fixed the identical way, keeping *its* convention — a remainder
carrying the sign of `num`, which is deliberately **not** `WKB_mod_2pi`'s (prompt §2 item 3;
`WKBtools.py:9-14`). The two functions were not unified.

```python
mod_2pi = fmod(fabs(num), TWO_PI)                              # unchanged
div_2pi = int(round((fabs(num) - mod_2pi) / TWO_PI))
```

The unused `floor` import was dropped, and the one phrase of the module docstring that described
the now-removed construction ("plain fmod plus a floor") was corrected — deviation 1.

### 3. `LiouvilleGreen/tests/test_wkbtools.py` — new, seven tests

Contains `_legacy_WKB_mod_2pi` and `_legacy_simple_mod_2pi`, verbatim copies of the pre-fix bodies,
used as the reference the "remainder does not move" test scores against and as the subject of the
"the adversarial sweep really is adversarial" test. They are labelled as not-production.

| test | what it pins |
|---|---|
| `test_WKB_mod_2pi_reconstructs_theta` | over 3,716 $\theta$, both signs, $\|\theta\|$ from 0 to $9.3\times10^{12}$: `div*TWO_PI + mod` is never off by a whole cycle, and is within 1 ulp of $\theta$ |
| `test_simple_mod_2pi_reconstructs_num` | the same for `simple_mod_2pi` |
| `test_production_regression_sample` | $\theta = -3832989103139.361$ by value: `div == -610039162581` and `div*TWO_PI + mod - theta == 0.0` exactly |
| `test_adversarial_abscissae_are_actually_adversarial` | the pre-fix body must still fail on the sweep — guards against the sweep silently ceasing to probe the half-ulp band |
| `test_remainder_is_bit_identical_to_the_old_implementation` | `mod.hex()` equality against both legacy bodies, for both functions, over the whole sweep |
| `test_negative_remainder_convention_is_preserved` | $\theta_{\rm mod}\in(-2\pi,0]$ for `WKB_mod_2pi`; sign-of-argument for `simple_mod_2pi` |
| `test_cycle_count_matches_the_exact_floor` | the recovered count equals $\lfloor\|\theta\|/2\pi\rfloor$ computed in exact rational arithmetic (`fractions.Fraction(TWO_PI)`) |

The abscissae are **constructed, not sampled** (prompt §3 item 2): `_adversarial_thetas()` walks
`nextafter` downwards 24 steps from $N\cdot{\tt TWO\_PI}$ for six cycle counts $N$ at each decade of
$|\theta|$ from $10^0$ to $10^{13}$ — 3,504 values. Random draws find the defect at the half-ulp
rate (25 in 400,000 at $|\theta|\sim4\times10^{12}$); this sweep finds it 64 times in 3,504.

### 4. `ComputeTargets/tests/test_tk_source_functions.py` — one comment

The `stored_values` docstring at `:542` said "`WKB_mod_2pi` uses `fmod`, which is exact". True of
the remainder, never true of the pair. Corrected to say so, naming the issue and this prompt. **No
assertion, tolerance or other line in that file was touched** — it is otherwise
`transfer-remedial`'s (`[10-transfer-remedial-tolerance-comments-stale]`).

### 5. Boards and index

This campaign's `IMPLEMENTATION_STATE.md` (row 01, item P1, §4); the `GkTk-remedial` board, where
`[13-wkb-mod-2pi-cycle-count-inconsistent]` moved from §3 to §4 with the measured result and the
`[10-wrap-theta-loop-at-large-phase]` clause that refers to it was corrected; and
`docs/OPEN_ISSUES.md`, where the §1.3 row was deleted, the `[10-wrap-theta-loop-at-large-phase]`
hook corrected, and the count taken 54 → 53.

## Deviations from the prompt

1. **`IMPLEMENTATION CHOICE` — corrected one phrase of the `range_reduce_mod_2pi` module
   docstring.** §2 of that docstring described `simple_mod_2pi` as "plain fmod plus a floor". After
   this commit there is no floor; the sentence now reads "an exact fmod, plus a cycle count
   recovered from that remainder". The file is on the prompt's "may touch" list and the phrase is a
   direct description of the line being changed, so leaving it would have shipped a docstring
   falsified by its own commit. Nothing else in the docstring was edited — in particular the
   house rule on not re-reducing, and the historical note on the removed prime-factorisation
   scheme, are untouched.

2. **`IMPLEMENTATION CHOICE` — dropped the now-unused `floor` import from both modules.** Neither
   file uses `math.floor` any more. Mechanical; `black` and the suites confirm.

3. **`IMPLEMENTATION CHOICE` — two tests beyond the three the prompt lists.**
   `test_negative_remainder_convention_is_preserved` and `test_cycle_count_matches_the_exact_floor`.
   The first turns prompt §2 item 2's stop condition ("it is a stop condition if the convention
   moves") into an assertion rather than an expectation; the second scores the count against exact
   rational arithmetic rather than against reconstruction, so it would catch a count that is wrong
   in a way the 1-ulp reconstruction test happens to absorb. No production behaviour depends on
   either.

No deviation touches a README §2 design fact. Nothing is tagged `UNINTENDED DRIFT`.

## Verification performed

### The acceptance table, measured

`PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py`, **unedited**, run at
`HEAD~1` (the baseline) and at this commit; 49.6 s and 47.7 s.

| acceptance row | before | after | target | verdict |
|---|---|---|---|---|
| `div*2π + mod == θ`, production $G_k$, LambdaCDM $k=3\times10^8$ | **1** of 77,975 (1.282e-5) | **0** of 77,975 (0.000e+00) | 0 of 77,975 | **met** |
| the eleven other (model, sector, $k$) rows of §6 | 0 | 0 | 0 | **met** |
| uniform control, $\|\theta\|\sim4\times10^{12}$ | **25** of 400,000 (6.250e-5) | **0** of 400,000 | 0 of 400,000 | **met** |
| uniform controls at $10^9$ and $10^{11}$ | 0 / 0 | 0 / 0 | 0 | **met** |
| stored `theta_mod_2pi`, every model and $k$ | — | **bit-identical** | bit-identical | **met** |
| consumer phase, LambdaCDM $G_k$ $k=3\times10^8$ | **6.1748 rad** at $z=33295.6$ (12646.00 ulp) | **0.0000e+00 rad** at $z=4.09953\times10^{12}$ (0.00 ulp) | falls to the floor | **met** |
| consumer phase, LambdaCDM, the other rows, both sectors | 1.00 ulp | 1.00 ulp, identical text | unchanged | **met** |

The half-ulp width at $|\theta|\sim4\times10^{12}$ is 6.104e-05 cycles and the pre-fix rate was
6.250e-05 — the mechanism's own prediction, and it is that whole band that is now empty.

### Nothing else in the script's output moved

`diff before.txt after.txt` is 57 lines. Every hunk is one of:

- the three §6 `cycles` rows quoted above (LambdaCDM $G_k$ $3\times10^8$ and the $4\times10^{12}$
  control);
- the LambdaCDM $G_k$ $k=3\times10^8$ **consumer** row and its `!! 1 of 1361 stored samples carry a
  whole-cycle phi outlier (z = ['33226'])` warning line, which is gone; `phi in [-6.283, 0] rad`
  becomes `phi in [0, 0] rad`;
- the `theta_deriv vs omega, Gk` block for that same case — **5.6245e-08 at $z=34000$ →
  1.8060e-13 at $z=4.1081\times10^{12}$**, deep interior 5.6245e-08 → 1.4363e-13. Not a target of
  this prompt; it is the same single sample, seen through the spline's derivative. Its $z=34000$
  is the grid neighbour of the $z=33226$ outlier;
- **wall-clock only**: the two table-build lines and the two `tau.delta` per-call cost lines of §0,
  the four wall-time columns of the §1 producers table, the §5 throughput line and the total. The
  §1 producers table's *integrand-evaluation* columns — the reproducible measure — are identical
  to the digit (7392/468, 11376/5540, 8380/472, 12896/5608).

Every **number that is not a timing** is character-identical between the two runs outside the four
bullets above: §0's primitive tables ($\tau$, $\Delta\tau$, $c_s\tau$, $F$, $\rho$ at the production
nodes), §1's phase tables, §2 ($F$ and $\rho_T$ as stored), §4 (margins), the eleven other §6 rows
and every other §3 consumer row, the three QCD ones included.

### Tests

| suite | result |
|---|---|
| `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` | **Ran 339 tests in 155.6 s — OK** |
| `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` | **Ran 148 tests in 1264.4 s — OK** |

The `ComputeTargets` count **did not fall**: 339, the same number the `GkTk-remedial` board records
for prompt 20, with no test removed and none added there (this commit's only change in that package
is a docstring). `LiouvilleGreen` goes **141 → 148**, the seven of the new `test_wkbtools.py`;
nothing was removed. (141 OK in 1438 s at `HEAD~1` is the orchestrator's own measurement on this
machine, quoted here because that suite takes 18–30 minutes and is easy to mis-capture: a first
attempt of mine piped it through `tail -5`, which returned the *pipeline's* exit status and the
last five lines of interleaved stdout rather than the summary, and had to be re-run unpiped.)

**The new tests fail on the old code, loudly.** Re-running `TestWKBModTwoPi` with the two
production functions monkeypatched back to the legacy bodies: **4 of 7 fail**, printing

```
FAIL: test_WKB_mod_2pi_reconstructs_theta
AssertionError: 6.283203125 not less than 3.141592653589793 : whole-cycle inconsistency at
theta=-3832989103139.361: div=-610039162582, mod=-6.282807898877543,
recon - theta = -6.283203125 (-1.000003 cycles)

FAIL: test_cycle_count_matches_the_exact_floor
AssertionError: 610039162582 != 610039162581 : WKB_mod_2pi cycle count wrong at
theta=-3832989103139.361

FAIL: test_production_regression_sample
AssertionError: -610039162582 != -610039162581

FAIL: test_simple_mod_2pi_reconstructs_num
AssertionError: 6.283203125 not less than 3.141592653589793 : whole-cycle inconsistency at
num=-3832989103139.361 ...
```

Counted over the adversarial sweep alone: **64 of 3,504 abscissae** are whole-cycle inconsistent on
the pre-fix code and **0 of 3,504** after, with the failures spanning $|\theta|$ from
$1.70\times10^2$ to $1.00\times10^{12}$ — the defect is not confined to the largest phases, only
its *random* rate is.

The three tests that pass on the old code are the two convention tests (the convention did not
move — that is the point) and the "remainder is bit-identical" test, which compares the old code to
itself.

### Cost

Microbenchmark, $10^6$ reductions of $\theta$ drawn uniformly in $[-4\times10^{12}, -1]$, best of 5:

| | total for $10^6$ | per call |
|---|---|---|
| before | 0.2254 s | **0.2254 µs** |
| after | 0.2731 s | **0.2731 µs** |

**+21 %, or +48 ns per call.** The reduction runs **once per stored sample**, so the largest
(model, sector, $k$) sample set in §6 — QCD $G_k$ at $3\times10^8$, 79,809 samples across 117
objects — pays **3.8 ms more in total**, against 0.2382 s of wall time for a *single* one of those
objects in the §1 producers table. The verification script's total wall time went *down*
(49.6 → 47.7 s), i.e. the difference is far inside this machine's timing noise at the whole-run
level. There is no threshold on this row; the number is recorded because the prompt requires it to
be known.

## Observations not acted on

1. **`wrap_theta` still reduces by a loop** — `[10-wrap-theta-loop-at-large-phase]`, already open on
   the `GkTk-remedial` board, and its "*`WKB_mod_2pi` uses `fmod`, is exact*" clause is the one this
   prompt was told to correct (done, in both the board and the index). The loop itself is
   out of scope here and the entry stays open with its own next step.

2. **The `GkSource` rectifier's trigger is still one-sided.** `GkSource.py:200`'s condition is
   `theta > last_theta` — a cycle count jumping *up* — so it never could have caught this defect,
   which pushes the phase one cycle *down*. With the producer fixed there is nothing for it to
   catch from this source, but the asymmetry is a property of the rectifier, not of
   `WKB_mod_2pi`, and `GkSource.py` is on this prompt's do-not-touch list. Not opened as an issue:
   the rectifier exists for a different mechanism and prompt 13 did not measure it firing.

3. **`WKB_product_mod_2pi` is still not used by production** and still carries the `while` loops
   and the `DEFAULT_ABS_TOLERANCE` guard that `WKBtools.py:33-45` documents. Its
   `simple_mod_2pi` call now returns a consistent pair, which is all this prompt owed it.

4. **`black --check .` is not clean on this tree** — 54 files would be reformatted, 53 of them in
   `docs/` and one in `AdaptiveLevin/`. None is touched by this commit and both directories are on
   this prompt's do-not-touch list; the four files this commit does touch are clean. `CLAUDE.md`
   says the tree is clean under `--check`, so either the statement has drifted or the installed
   `black` has. Not acted on and not opened as an issue: it is a statement about the repository's
   formatting policy, not about this campaign, and reformatting 54 files would destroy the
   revert-per-prompt property for nothing.

5. **The §1 producers table's wall-time columns swing by up to 40 %** run to run on this machine
   (QCD $T_k$ at $3\times10^8$: 1.0737 s → 0.6744 s cold, with *identical* evaluation counts). This
   is campaign board §5 note 14 restated and is why every cost figure here is a best-of-N; noted so
   that a later reader diffing two runs of the script does not read it as a change.

## State handed to the next prompt

**The defect is closed.** `WKB_mod_2pi` and `simple_mod_2pi` now derive their cycle count from the
exact `fmod` remainder. On the production geometry the LambdaCDM $k=3\times10^8$ Green's-function
set is **0 inconsistent of 77,975** (was 1), the $|\theta|\sim4\times10^{12}$ uniform control is
**0 of 400,000** (was 25), and the LambdaCDM $k=3\times10^8$ consumer row has fallen from
**6.1748 rad to exactly 0.0000e+00 rad**, i.e. from 12,646 ulp of the span to 0.00. All **six**
production consumer cases on LambdaCDM (three wavenumbers × two sectors) are now at or below
**1.00 ulp** of their span, and ten of the twelve across both models are at exactly 1.00 or 0.00 —
the two that are not are prompt 02's, both on QCD.

**Prompt 02 inherits an unchanged `PrimitivePhase`.** Nothing in `ComputeTargets/` was touched
except one docstring in a test file. The three QCD consumer rows are bit-identical to `9daa2cb` —
$G_k$ at $k=10^5$ is still **1.9073e-06 rad (8.00 ulp) at $z=4.24388\times10^7$** and $T_k$ at
$k=10^5$ still **3.1859e-06 rad (427.60 ulp) at the same $z$** — so prompt 02's before-column is
exactly the one `docs/gktk-remedial-verification.md` §3.5 records. The `theta_deriv` figures prompt
02 must separate are likewise unmoved **on QCD**; the one `theta_deriv` row that did move is
LambdaCDM $G_k$ at $k=3\times10^8$ (5.6245e-08 → 1.8060e-13), which was this defect and not a knot
placement, so prompt 02's §3.6 QCD baseline stands.

### The datastore consequence (prompt §5; README §7 D3)

**A datastore written before this commit is served, silently, with the old cycle count at the
affected samples.** `theta_div_2pi` is a stored `nullable=False` column on `GkWKBValue` and
`TkWKBValue` and it is in **no lookup key**, so there is no schema change to raise a `RuntimeError`
on and no key that misses. No migration was invented and no column or key was added: both are out
of scope, and adding a key would invalidate every row of every existing datastore for a defect that
touches one sample in 78,000. **A louder mechanism is not recommended, but it is the user's call
and it is recorded as theirs.**

**Regeneration list** — what a pre-01 datastore gets wrong, in the form log 13's list uses:

| quantity | affected? |
|---|---|
| `theta_div_2pi` on `GkWKBValue` / `TkWKBValue` | **yes** — one cycle too many at the half-ulp samples; 1 in 77,975 at LambdaCDM $k=3\times10^8$, rate $\propto k$ |
| the reconstructed unwrapped phase in `build_phi_samples` | **yes**, through `theta_div_2pi` |
| `PrimitivePhase.raw_theta`, and its spline of $\varphi$ | **yes**, through `build_phi_samples` — the 6.17 rad excursion, plus the fifteen grid intervals either side that a cubic spline spreads it over |
| `QuadSourceIntegral`'s `_ClampedPhase` | **yes**, through `PrimitivePhase` |
| `theta_mod_2pi` | **no** — it is the `fmod`, unchanged, bit-identical, asserted by `test_remainder_is_bit_identical_to_the_old_implementation` |
| every stored $G$ and $T$ (`G_WKB`, `T_WKB`) | **no** — they are built from the remainder |
| `friction_F`, $\rho$, $\tau$, $c_s\tau$, and every `BackgroundModel` column | **no** |

**No dated line was appended to `docs/gktk-remedial-verification.md` §8.** This is the first prompt
of the campaign to need one, and prompt §5 item 2 says in that case to say so here and leave the
document alone; the campaign close-out writes §8. The regeneration table above is what §8 should
carry for this commit, and nothing at or above §7 of that document was read as editable or edited.
