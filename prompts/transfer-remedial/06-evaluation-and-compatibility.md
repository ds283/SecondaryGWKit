# Prompt 06 — The compatibility adapter, `theta_abserr`, and consumer migration

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §8.1 (the interface and `theta_abserr`), §9 Stage 3
**Reconciliation items:** C3 (`plot_besssel_phase.py` is already broken), §1 (`theta` is a required Levin key), §3.2 (why `QuadSourceIntegral.py` is excluded), §3.3 (the complete consumer inventory)
**Depends on:** 05 (hard)
**Recommended model:** Opus
**Files you may touch:** `LiouvilleGreen/bessel_phase.py`, `main.py` (**the Bessel construction
stage only**, `main.py:516-528`), `ComputeTargets/QuadSourceIntegral_debug.py`,
`plot_besssel_phase.py`, `LiouvilleGreen/tests/test_bessel_two_region.py` (add serialization and
adapter tests), plus the log and the status board.
**Do not touch:** `ComputeTargets/QuadSourceIntegral.py`, `ComputeTargets/QuadSource.py`,
`ComputeTargets/phase_groups.py`, `ComputeTargets/TkSourceFunctions.py`, anything under
`Datastore/`, `AdaptiveLevin/`, `LiouvilleGreen/three_bessel_integrals.py` (prompt 07),
`LiouvilleGreen/tests/test_bessel_phase.py` or `test_3bessel_analytic.py` (prompt 08).

Read first: README §1.1 and §4.2 (the scope boundary and why); `RECONCILIATION.md` C3 and §3.2–3.3;
`DRAFT-PLAN.md` §8.1 and §9 Stage 3; then `logs/05-two-region-construction.md`
§"State handed to the next prompt"; then, in the code, `AdaptiveLevin/levin_quadrature.py:930-975`
(the phase-dict contract), `:1030-1095` (what is actually evaluated), `:2350-2370` (the
`theta_abserr` rationale) and `:2748-2770` (the docstring); `ComputeTargets/QuadSourceIntegral.py:121`
(`BesselPhaseProxy`) and `main.py:516-528`.

---

## 1. Character of this commit

Plumbing, plus two judgement calls. A better construction is useless if evaluation immediately
rounds its correction away, or if it reports an artificially small error to the quadrature that
consumes it.

The two judgement calls, both flagged for the orchestrator:

- **`Q` and `phi`** — what to do with two members whose meaning no longer exists.
- **`plot_besssel_phase.py`** — repair or delete a script that has been dead for at least two
  interface changes.

## 2. The phase adapter

`DRAFT-PLAN.md` §8.1: keep the constructor entry point and the useful members `phase`, `mod`,
`bessel_j`, `bessel_y`, `min_x`, `max_x`. The `phase` object **need not remain a `phase_spline`
instance** — under the new construction it will not be one — but must implement the behaviour its
consumers require, including raw/log inputs and ordinary/log derivatives.

Required surface, verified against the actual call sites (`RECONCILIATION.md` §3.3):

| Method | Called by | Notes |
|---|---|---|
| `raw_theta(x, x_is_log=False)` | `three_bessel_integrals.py:179-181`, `QuadSourceIntegral.py:1226…`, `test_bessel_phase.py:37`, `ComputeTargets/tests/test_tk_source_functions.py:266` | \(\varepsilon x\)-limited; docstring must say so |
| `theta_mod_2pi(x, x_is_log=False)` | `three_bessel_integrals.py:188-190`, `QuadSourceIntegral.py:1236…`, `bessel_j`/`bessel_y` | must be the **bounded-angle** accessor of prompt 05 §2.5, not `fmod(raw, TWO_PI)` |
| `theta_deriv(x, x_is_log=False, log_derivative=False)` | `three_bessel_integrals.py:199-201`, `test_bessel_phase.py:135` | \(e^{-2\ell}\) route; both derivative flavours |

**Audit every caller before finalizing the adapter** (§8.1 says so explicitly). `RECONCILIATION.md`
§3.3 is the complete inventory; confirm it is still complete with your own grep and record any
addition.

### 2.1 `theta_abserr` — declare it

`adaptive_levin_sincos` already accepts a caller-declared phase error
(`levin_quadrature.py:962`), and the comment at `:2360` says it exists for exactly this case:

> where the phase's own construction is the real limit (e.g. a fitted spline), a declared
> `theta_abserr` so the caller sees an honest number instead of an artificially small one.

Nothing supplies one today. Since this campaign changes achieved accuracy by six to eight orders,
the object must report its own achieved phase accuracy and consumers must pass it through
(`DRAFT-PLAN.md` §8.1). Add a `theta_abserr` accessor to the phase object, fed from prompt 05's
achieved-accuracy fields. It may be a scalar or a callable of \(x\) — `levin_quadrature.py:967-975`
accepts either — and a callable is the better answer here, because the near region and the tail
have genuinely different errors and the tail's is far smaller.

Note two contract facts confirmed in `RECONCILIATION.md` §1 and §3.3:

- `theta` is a **required** key: `_Basis_SinCos.__init__` raises without it
  (`levin_quadrature.py:948-952`). "`raw_theta` is compatibility-only" means Levin never *evaluates*
  it when both `theta_mod_2pi` and `theta_deriv` are supplied (`:1038`); it does not mean the key is
  optional.
- The derivative carries **double duty** — basis conditioning *and* subdivision, since `phase_span`
  comes from `theta_prime_Cheb` (`:1090`). That is a further reason prompt 04/05's derivative budget
  had to be met rather than finessed, and it is worth a comment at the adapter.

Also: the docstring at `levin_quadrature.py:2750` says `theta` is "always used to decide
subdivision". That is **stale**, not a contradiction of the plan (§8.1 checked this). You may not
edit `AdaptiveLevin/`, so record it in the log's "Observations not acted on" for a future
`AdaptiveLevin` prompt.

### 2.2 `Q` — the first judgement call

The current `Q` is the **pre-offset ODE state**, whereas the returned phase incorporates `phi` and
cycle rebasing. There is no ODE any more, so there is no `Q`.

`DRAFT-PLAN.md` §8.1 is prescriptive about the hazard, not the answer:

> Do not silently replace `Q` with a different diagnostic quantity under the same undocumented
> meaning: either update those consumers to use the residual, or provide a clearly documented
> compatibility quantity and a deprecation path.

Two live consumers: `ComputeTargets/QuadSourceIntegral_debug.py:55` (a real diagnostic, which
plots `Q` alongside the phase) and `plot_besssel_phase.py:22` (already broken — §2.4).

The recommended resolution, which you may depart from with reasons: **remove `Q` and update the
debug consumer to plot the residual \(r_\nu\)** instead, which is the quantity that now plays
`Q`'s role of "the smooth thing the construction actually represents". Expose `r_nu(x, x_is_log=False)`
and `log_amplitude(x, ...)` for diagnostics, and make the removal of `Q` a `KeyError` with a
message naming the replacement rather than a silent absence. Under no circumstances keep the key
`Q` mapped to something that is not \(\theta/x\).

### 2.3 `phi` — the second, easier one

§4.3 establishes that `phi` should be reported as **identically zero** for \(\nu>1/2\), *not
quietly dropped*. For \(\nu=1/2\) the old match point was \(x=1\), an interior point, so that case
was genuinely different and is not covered by §4.3's argument — but the new construction has no
offset for any order.

Keep the key, set it to `0.0`, and document in one sentence that it exists only so that a consumer
reading `phi` gets the right answer rather than a `KeyError`, and that the old non-zero values were
a root-solve artefact of `xtol=1e-6, rtol=1e-4`. Removing it instead is defensible; if you do,
grep for readers first (there are none outside diagnostics as of `RECONCILIATION.md` §3.3) and say
so.

### 2.4 `plot_besssel_phase.py` — already broken, so this is a decision

It cannot currently run (`RECONCILIATION.md` C3): `:15` reads `data["x_min"]`, a key `bessel_phase`
has never returned (the dict has `min_x`), so the function raises `KeyError` immediately; and `:30`
calls `phase(x)`, but `phase_spline` defines no `__call__`. Nothing imports it.

So "migrate it" is not a port. **Repair it** — fix `x_min`→`min_x`, replace `phase(x)` with
`raw_theta`, replace `Q` with the residual, and confirm it actually produces its plots — **or
delete it**. Repairing is the default. Deleting is a stop condition (README §4.3): the user decides.
Whichever you do, state it in the log and make the orchestrator surface it.

### 2.5 `ComputeTargets/QuadSourceIntegral_debug.py`

Reads `Q` (`:55`), `mod` (`:254-255`), `phase`, `bessel_j`, `bessel_y`, `min_x`, `max_x`. Migrate
`Q` per §2.2 and leave everything else alone. This is a live diagnostic; check it imports and that
its plotting path is exercised at least as far as building its grids. It is **not** in the
prohibited set, but note that the file sits beside `QuadSourceIntegral.py`, which is — do not let
an edit stray across.

## 3. Production migration — `main.py:516-528`

Currently:

```python
Bessel_0pt5 = bessel_phase(0.5 + b_value, largest_x_with_clearance, atol=1e-25, rtol=5e-14)
Bessel_2pt5 = bessel_phase(2.5 + b_value, largest_x_with_clearance, atol=1e-25, rtol=5e-14)
```

with `b_value = 0.0` (`main.py:510`). Replace the tolerance arguments with the explicit accuracy
settings prompt 05 introduced, at values that meet README §6's low-order row (\(10^{-11}\)) with
margin. **Update the caller rather than relying on the deprecation shim** — the shim exists for
third-party and script callers, not for production.

The comment at `main.py:516-519` explains the old tolerances in terms of "keeping \(Q\) very
accurately close to 1" and names the Dormand–Prince stepper. It is now wrong in every particular.
Replace it with a short, accurate comment: what accuracy is requested, and that the construction is
two-region with a closed-form tail so the cost no longer depends on `largest_x_with_clearance`.

**`main.py` hunk discipline** (README §4.2): the in-flight `source-remediation` campaign edits the
QuadSource and QuadSourceIntegral stages of the same file. Your hunks must be confined to the Bessel
construction stage. A hunk anywhere else fails review.

## 4. Serialization through Ray

`BesselPhaseProxy` (`ComputeTargets/QuadSourceIntegral.py:121`) does `ray.put(obj)` on the whole
returned dict and `ray.get` in the worker. **Verify serialization of the new representation and its
interpolants** (§8.1).

You may not edit `QuadSourceIntegral.py`, so test the round trip directly: `ray.put` / `ray.get` (or
`ray.cloudpickle.loads(ray.cloudpickle.dumps(obj))`, which needs no cluster and is what to prefer in
a unit test) and then assert that every accessor on the revived object returns bit-identical values
to the original at a grid of \(x\). Closures over local functions are the usual hazard; `scipy`
`BSpline` objects pickle, bare closures over module-level functions pickle, closures over
locals inside a factory function pickle under cloudpickle but *not* under plain pickle. If anything
fails to round-trip, that is a `STRUCTURALLY REQUIRED` deviation and it needs a design fix (make the
interpolant carrier a module-level class), not a workaround in the test.

Also verify scalar and log-input behaviour survives the round trip (§9 Stage 3), and that the
`theta_abserr` accessor does too — a callable that does not pickle would silently become the thing
the Levin quadrature cannot get an honest number from.

## 5. Tests

Add to `LiouvilleGreen/tests/test_bessel_two_region.py` (or a new
`test_bessel_compatibility.py` if that file is getting long — your choice, state it):

1. **The adapter surface.** Every method in §2's table exists with the stated signature, accepts
   both raw and log input, and agrees between the two modes at matched arguments (§7.5: matched
   means the log mode is scored at \(e^u\), the same double).
2. **`theta_abserr`.** Present; returns a finite positive number (or a callable that does); its
   value is \(\ge\) the measured \(E_\theta\) at the same \(x\) for a grid of \(x\) across both
   regions. An estimator that under-reports is worse than none, so assert the inequality in the
   safe direction and quote the ratio.
3. **A real Levin call using it.** Run `adaptive_levin_sincos` on \(\int A\sin\theta\,dx\) with the
   new object supplying `theta`, `theta_mod_2pi`, `theta_deriv` and `theta_abserr`, and assert (a)
   it converges, (b) the reported `abserr` is now dominated by the declared phase error rather than
   being implausibly small, and (c) `need_theta_Cheb` did not fire — most simply by asserting the
   result matches a run with a deliberately broken `theta` callable, which is only possible if
   `theta` is never evaluated. That last check is worth having: it is the campaign's evidence for
   `RECONCILIATION.md` §1's reading of `:1038`.
4. **`phi` is 0.0** for every order tested (or the key is absent and documented).
5. **`Q` is gone cleanly** — a `KeyError` naming the replacement, not a silent absence.
6. **The Ray round trip** of §4, bit-identical on a grid.
7. **The deprecation shim.** Passing `atol`/`rtol` warns, names the new arguments, and produces the
   documented behaviour; passing both old and new gives the new precedence and says so.
8. **`XSplineWrapper` still imports** from `LiouvilleGreen.bessel_phase`
   (`test_three_bessel.py:10` needs it and is not yours to edit).

## 6. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes,
  including `test_bessel_phase.py`'s four tests with `test_phase_derivative` unchanged at
  \(10^{-6}\) (`RECONCILIATION.md` C4).
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes —
  the workstream-B fixtures consume `mod` and `raw_theta` and are the canary for an adapter slip.
- `PYTHONPATH=. ./venv/bin/python -c "import main"` succeeds (an import check on the edited stage;
  a full run needs Ray and a datastore and is prompt 09's business, not yours).
- `PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets.QuadSourceIntegral_debug"` succeeds.
- `plot_besssel_phase.py` either runs to completion producing its figures, or is deleted with the
  user's agreement.
- `git diff HEAD~1 --stat` shows `main.py` hunks only in the Bessel stage.

## 7. Log and commit

Follow README §5 and §5.1. Your log must state, unambiguously:

- the **decision on `Q`** and the alternatives weighed;
- the **decision on `phi`**;
- the **decision on `plot_besssel_phase.py`**, and that the orchestrator was told;
- the final adapter surface, verbatim, including `theta_abserr`'s type and semantics;
- the deprecation translation and precedence as shipped;
- the measured `theta_abserr`-to-\(E_\theta\) ratio across both regions;
- whether the Ray round trip was bit-identical, and what carries the interpolants;
- the Levin call's reported `abserr` before and after declaring `theta_abserr`, as numbers — this
  is the concrete demonstration that the campaign's accuracy claim reaches the consumer.

In "State handed to the next prompt": the residual and leading-coefficient accessors prompt 07 needs
to assemble \(Kt+C+R(t)\), with exact names and signatures.

Commit subject, or something equally specific: `Adapt the Bessel phase interface to its consumers`.
