# Prompt 01 — Refuse or report: finiteness, validation, and an honest aggregate

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §1.2 (C1), §1.4 fix 1 (C3), §1.6 (C6), §1.8 (C8), §1.9 (C9); §5 recommendations 1–4
**Depends on:** nothing — this is the first prompt
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_quadrature.py`,
plus the log and the status board.

---

## Character of this commit

Seven small changes that share one purpose: **the module must either return a meaningful result or
refuse.** None of them changes the value returned for valid input. All of them change what happens
for input that is invalid, degenerate, or non-finite — where the module currently returns a number
that looks fine and is not.

This is the highest value-for-effort commit in the campaign. It is also the foundation for
everything after it: prompt 02 rewrites the solve and must carry the finiteness guards into the
complex path, and prompt 03 introduces a second cell type that needs them too.

Work through the items in order. Each is self-contained.

---

## Item 1 — C1: a non-finite amplitude must raise, not silently contribute zero

**This is the most important change in the campaign.** Read audit §1.2 in full before starting.

### The mechanism

`_adaptive_levin_subregion_impl` checks `np.isfinite(LevinL).all()` at `levin_quadrature.py:662`,
but **nothing checks `f_Cheb`** (built at `:645`). A NaN in the sampled amplitude propagates to `p`,
hence to `p_ratios` (`:759`), and then:

```python
p_use = [r > rtol for r in p_ratios]      # NaN > rtol is False
```

so `p_use == [False, False]`, so `lower_limit = upper_limit = 0.0` (`:772-777`) **and**
`p_endpoint_l1 = 0.0` (`:782-789`), so `phase_err = 0.0`. The driver then sees `estimate = 0`,
`refined = 0`, `abserr = 0 < atol`, and accepts the region.

### Reproduced at HEAD

```
amplitude f0(x) = NaN if x > 1.9 else 1,  theta = 1e5 x,  span (1, 2), default tolerances
  ->  value = -5.191968760785887e-07,  abserr = 1.025e-15,  4 regions
all-NaN amplitude
  ->  value = 0.0,  abserr = 0.0,  1 region
```

The first is a **partial integral certified at 10⁻¹⁵**. The second is the most dangerous possible
response to a completely broken integrand.

### What to do

Four checks. The existing `LevinL` finiteness check at `:662` is the right model for all of them —
print a warning naming the region, then raise `ValueError` with a message that says *what* was
non-finite.

1. **`f_Cheb`, immediately after sampling** (`:645`). This is the one that matters; it is the direct
   cause. The message must name the amplitude, not the super-operator — see item 6 on the current
   message misattributing the cause for a NaN endpoint.
2. **The sampled `θ′`** (`theta_prime_Cheb`, built in `_Basis_SinCos.build_Levin_data` at `:474` or
   `:478`). Note this is *not* covered by the `LevinL` check in the way it looks: a NaN in `θ′` does
   propagate into `AmatT` and hence into `LevinL`, so it is caught — but the message is wrong, and
   an `inf` in `θ′` with a `0` in the right place is not something to rely on. Check it where it is
   sampled, so the diagnostic names the phase derivative.
3. **The solved `p`**, after the solve succeeds and before `P = p.reshape(...)` at `:756`. The
   direct-solve fast path already tests `np.isfinite(p_direct).all()` (`:695`); the `lstsq` and
   `pinv` paths do not. Add a single check covering all three exits.
4. **`p_use` must reject non-finite ratios explicitly**, not by accident:

```python
p_use = [np.isfinite(r) and r > rtol for r in p_ratios]
```

Even with check 3 in place this is worth doing: it makes the intent legible, and it stops the
`NaN > rtol is False` coincidence from being load-bearing. **Do not** treat this as an alternative
to check 3 — a non-finite `p` must raise, not be quietly dropped.

Also guard `p_mean_max` at `:758`: `p_ratios = [pm / p_mean_max ...]` divides by zero when every
component of `p` is identically zero, which is reachable (`QuadSourceIntegral.py` passes
`f = [Levin_f, lambda x: 0.0]`, and a region where the amplitude underflows gives `p ≡ 0`). It
currently yields `nan` and a `RuntimeWarning`. Decide what a zero solution should report — an
all-ones ratio vector and an all-`True` `p_use` is the behaviour-preserving choice, since a zero
region genuinely contributes zero — and record the decision as an implementation choice.

**Cost.** One `np.isfinite(...).all()` on an `mN` vector per solve, ~0.5 µs at N = 12, against a
~56 µs subregion evaluation. Negligible.

### Verification for this item

Both reproducers above must now raise `ValueError` with a message naming the amplitude. Add both as
tests (see *Tests* below).

## Item 2 — C3 (reporting half): compare the aggregate with the request

The audit calls this "the single most valuable change in the audit relative to effort".

Acceptance at `:1083` is per region, and `abserr_total` (`:1173-1180`) is the sum over accepted
regions, so the delivered absolute error scales like `N_regions × atol`. **Measured at HEAD:**
`∫₀³ e^{−400(x−1.1)²} sin(3×10⁴x) dx` at `atol = 1e-10` reports `abserr = 1.26e-10` — which exceeds
the requested `atol`, and nothing says so.

After the aggregate is computed (`:1173-1183`):

- compute `requested = max(atol, rtol * abs(val))`;
- add a `converged` key to the returned dictionary: `abserr_total <= requested`;
- when it is `False`, emit a warning in the same style as the two existing health checks at `:1197`
  and `:1213`, stating both numbers and both tolerances.

Do **not** raise. A caller that asked for more than the problem can give still wants the number and
the honest error bar. Do **not** change the acceptance logic — prompt 05 does that, and it is what
makes `converged` usually true.

`converged` is a new key, so the benchmark harness is unaffected (standing note 8).

## Item 3 — C6: update `max_depth` on every popped region

The direct-quadrature branch `continue`s at `:972`, before the
`if current_region.depth > max_depth` update at `:1112`. So a run that bisects to the `6π` floor and
terminates in direct quadrature never raises `max_depth`, and the depth-limit health warning at
`:1197` cannot fire — precisely the non-convergence mode the warning exists to catch.

**Reproduced at HEAD:** a run that is entirely direct-quadrature reports `max_depth = 0` at
`depth_max = 3`.

Move the update to immediately after `regions.pop()` at `:884`, and delete the copy at `:1112`.

Note prompt 03 deletes the fallback branch entirely — but this fix is still correct and still
needed after that, because the CC branch that replaces it will also `continue` past `:1112` unless
the update is at the top of the loop. Do it here so prompt 03 inherits it.

## Item 4 — C8: reject `atol <= 0`

With `atol = 0` the relative-error denominator floor at `:1059` is inert, `relerr` becomes `0/0 =
nan` for a region with a vanishing contribution, and the `phase_limited` branch cannot fire because
`phase_err > atol` is `0 > 0`.

**Reproduced at HEAD:** identically-zero integrand, `atol = 0`, `depth_max = 8` →
**256 regions, 1023 solves**, against 1 region / 3 solves at `atol = 1e-15`. At the default
`depth_max = 20` that is 2²⁰ regions.

A caller wanting pure relative control would naturally write `atol = 0`. The audit offers two
options: reject `atol <= 0`, or floor it at ~`1e-300` and floor `relerr_denom` at a positive number
derived from the running total. **Take the rejection** — it is honest about the fact that this
module cannot deliver a purely relative contract (the phase floor is absolute by nature), and a
caller who is told so can pick a real `atol`. Note the choice and the rejected alternative in the
log; if you conclude the floor is better, say why, but do not do both.

## Item 5 — C9: input validation

Add validation at the top of `adaptive_levin_sincos`, and in `_Basis_SinCos.__init__` for the phase
dictionary. Reproduced at HEAD, all eight rows exactly as the audit reports:

| Input | Currently | Required |
|---|---|---|
| `len(f) != 2` | `ValueError: operands could not be broadcast together with shapes (12,12) (24,24)` | clear error naming the 2-component contract of `_Basis_SinCos` |
| `f = []` | `ValueError: need at least one array to concatenate` | clear error |
| `len(x_span) != 2` | **silently integrates `(x_span[0], x_span[1])`** | error |
| `atol <= 0`, `rtol < 0` | silently accepted | error (item 4) |
| `depth_max < 0` | silently accepted; every region accepted at depth 0 | error |
| `chebyshev_order < 8` | silently clamped to 8 | warn, and say it was clamped |
| `theta` as a bare callable | `TypeError: argument of type 'function' is not iterable` | clear error |
| NaN endpoint | `ValueError` from the `LevinL` check | fine, but the message misattributes the cause (item 6) |

The `theta`-as-a-bare-callable case is an easy mistake to make: the parameter is *named* `theta` and
the module's own tests pass `theta={"theta": …}`. The error must say so.

**Where to put the `len(f) != 2` check.** `_adaptive_levin` and `_adaptive_levin_subregion_impl` are
written for general `m` and must stay that way (standing note 10) — the two-component contract
belongs to `_Basis_SinCos`, which is what `adaptive_levin_sincos` constructs. Validate there, and
make the message say which basis imposes it. Do not add an `m == 2` assumption to the generic code.

Also validate `x_span` entries are finite. A NaN endpoint should be caught by name, not by the
super-operator check downstream (item 6).

## Item 6 — C9 (message quality) and the missing docstring

`adaptive_levin_sincos` has **no docstring**. Write one. It must state, at minimum:

- what the routine computes: `∫ Σᵢ fᵢ(x) wᵢ(x) dx` with `w = (sin θ, cos θ)`;
- the two-component contract on `f`, and which slot is sine and which is cosine;
- the three recognised keys in the `theta` dict (`theta`, and the optional `theta_mod_2pi`,
  `theta_deriv`), what each is for, and the accuracy consequence of supplying or omitting them;
- that `atol` is currently a **per-region** tolerance and the returned `abserr` is a sum over
  regions — **and** that `converged` (item 2) reports whether the aggregate met the request. Prompt
  05 changes this; update the docstring then;
- the meaning of every key in the returned dictionary;
- that the reported `abserr` is an estimate, not a proven bound.

Do not document behaviour that later prompts will change as though it were permanent; where a
statement is scheduled to change, say so briefly.

## Item 7 — tests

Add to `AdaptiveLevin/tests/test_levin_quadrature.py`:

- `test_nan_amplitude_raises` — the partial-NaN reproducer above raises `ValueError`.
- `test_all_nan_amplitude_raises` — the all-NaN reproducer raises `ValueError`.
- `test_input_validation` — one assertion per row of the table in item 5, each checking the
  exception *type* and that the message names the offending parameter.
- `test_atol_zero_rejected`.
- `test_converged_flag` — a run that comfortably meets its tolerance reports `converged=True`; the
  `∫₀³ e^{−400(x−1.1)²} sin(3×10⁴x) dx` case at `atol = 1e-10` reports `converged=False`.

Keep them fast. The whole suite is currently 0.008 s; do not push it past a second or two.

---

## Do not

- **Do not change the weakly-oscillatory gate.** That is prompt 03, and it is the largest change in
  the campaign. Leave `:924` alone.
- **Do not change `theta_scale` or `_phase_error`.** That is prompt 04.
- **Do not change the acceptance test or distribute `atol` by length.** That is prompt 05. Item 2
  only *reports*; it does not change what is accepted.
- **Do not change how `p_use` selects modes** beyond rejecting non-finite ratios. Gating on endpoint
  magnitudes is prompt 06.
- **Do not touch the solve.** Prompt 02 rewrites it; a conflict here costs both prompts.
- **Do not move the `seaborn`/`matplotlib` imports.** Prompt 07.
- **Do not "fix" anything else you notice.** Record it under *Observations not acted on*.

---

## Verification

1. `AdaptiveLevin/tests/` passes:
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .`
   Baseline is 4 tests / OK; you should have ~9.
2. `black --check AdaptiveLevin/levin_quadrature.py` clean, if `black` is available.
3. **Numerical no-change check (required, README §5 rule 9).** On valid input this commit must not
   move any digit. Pick three problems — one from the existing tests, `∫_{1/3}^{7/3} e^{−x} sin(ωx)
   dx` at ω = 10⁶ against its closed form, and one three-Bessel oracle from
   `LiouvilleGreen/tests/test_3bessel_analytic.py` — and record `value`, `abserr` and `num_regions`
   before and after. They must be **exactly equal**, not merely close: nothing in this commit
   touches arithmetic. If any of them moves, you have changed something you did not intend — find
   it before committing.
4. Both C1 reproducers raise. Both C8 reproducers behave: `atol = 0` is rejected; `atol = 1e-15`
   still gives 1 region / 3 solves.
5. The C6 reproducer (an all-fallback run at `depth_max = 3`) now reports a non-zero `max_depth`.
6. The C3 reproducer reports `converged = False` and prints a warning.

---

## Finish

1. Write `prompts/levin-refactor/logs/01-refuse-or-report.md` using the template in `README.md`
   §5.1. This prompt contains **three explicit judgement calls** — the `p_mean_max == 0` behaviour
   (item 1), reject-vs-floor for `atol <= 0` (item 4), and where to place the `len(f) != 2` check
   (item 5) — and each must appear under deviations as an implementation choice with its reasoning,
   even where you took the recommended option. Include the item-3 numerical no-change table under
   *Numerical evidence*.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 01 row, the C1/C3/C6/C8/C9 and rec 1–4 item rows,
   the progress count, "last updated", and any §3 issues.
3. Commit in one commit. Suggested message:

```
Make Levin quadrature refuse bad input rather than certifying it

The adaptive Levin driver had three ways to return a confident number that
was not an answer to the question asked. None of them changes behaviour for
valid input; all of them replace a plausible-looking result with a refusal or
an honest report.

  * A non-finite sample of the amplitude propagated through the solve into
    p_ratios, where "NaN > rtol" is False, so every mode was discarded, the
    region contributed exactly 0.0 and its estimated error was exactly 0.0.
    The caller received a partial integral certified at machine precision; an
    entirely non-finite integrand returned 0.0 with an error bar of 0.0.
    f_Cheb, the sampled phase derivative and the solved p are now all checked
    for finiteness and raise, and the mode filter rejects non-finite ratios
    explicitly rather than by coincidence.

  * The aggregate error estimate was never compared against what the caller
    asked for. Because atol is a per-region tolerance, the summed error
    scales with the region count and was measured exceeding the request with
    no indication. The returned dictionary now carries a "converged" flag and
    the driver warns when it is false.

  * max_depth was updated after the direct-quadrature branch had already
    continued, so a run that bisected to the phase floor and terminated in
    direct quadrature reported depth 0 and could not trigger the depth-limit
    health check -- exactly the non-convergence mode that check exists to
    catch. The update now happens when the region is popped.

atol <= 0 is now rejected. With atol = 0 the relative-error denominator floor
is inert and the phase-limited acceptance branch cannot fire, so an integrand
with a vanishing contribution subdivided to the depth limit: 256 regions and
1023 linear solves at depth 8, against 1 region and 3 solves at atol = 1e-15.

Argument validation and a docstring are added. Previously a three-element
x_span was silently truncated to its first two entries, an f-vector of the
wrong length failed with a numpy broadcast error naming matrix shapes, and
passing theta as a bare callable rather than a dict -- an easy mistake, since
the parameter is named theta -- failed with "argument of type 'function' is
not iterable".

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match what you actually did.
