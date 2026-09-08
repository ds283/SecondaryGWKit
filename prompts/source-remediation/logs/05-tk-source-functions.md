# Log 05 — `TkSourceFunctions`: a two-region representation of $T_k$ for consumers (A2, part 1)

**Prompt:** prompts/source-remediation/05-tk-source-functions.md
**Commit:** *(this commit)* — "Add a two-region LG representation of T_k for source consumers"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Opus 5
**Date:** 2026-09-08
**Result:** COMPLETE WITH DEVIATIONS

> **Partially superseded (2026-09-08, orchestrator review).** The measurements in this log are
> correct, but two of their interpretations are not, and are replaced by
> [`docs/lg-phase-and-handover-followup-2026-09.md`](../../../docs/lg-phase-and-handover-followup-2026-09.md):
>
> 1. The numeric-region midpoint residual quoted under deviation 7 and in §3 issue
>    `[05-numeric-region-is-now-the-accuracy-floor]` (7.4e-06 in $T$, 2.7e-04 in $dT/dz$) is
>    mostly a cubic-spline *end-interval* effect at the hand-over node, not grid-density fit error:
>    the last two intervals carry 7.4e-06 and 2.9e-06 while the interior at the same node density
>    is ~5e-07. The hand-over is therefore the worst point for a boundary reason, and the fix is an
>    overlap past the stop point, not a denser grid (followup §1).
> 2. "State handed to the next prompt" item 5 says the LG-branch phase floor is set by the phase
>    range per chunk, not the total cycle count. That is wrong. The phase-spline error scales as
>    $h^4 x/384$ and grows linearly with $x$ (measured 6.9e-07 at $x=10^3$, 6.2e-06 at $x=10^4$;
>    ~1e-02 rad extrapolated to production $x\sim10^7$). Chunking protects floating-point precision,
>    not interpolation error (followup §2). The same document records that `bessel_phase`, the
>    "exact" oracle behind these fixtures, is itself accurate only to ~$x\times10^{-8}$ in phase
>    (followup §2.4); prompts 07, 08 and 12 should read §4 of the followup before setting tolerances.
>
> The relaxed tolerances of deviation 7 were accepted by the user on 2026-09-08 with these two
> issues deferred to a later remediation.

## What shipped

- **new `ComputeTargets/TkSourceFunctions.py`** (447 lines, of which ~110 are the module
  docstring). A non-persisted class `TkSourceFunctions(model, k, Tk_numeric, Tk_WKB)` that
  describes $T_k(z)$ over the whole source-redshift range in two regions, plus the exact
  $T=1$ region above them. Public surface, verbatim:

  | member | signature / type | meaning |
  |---|---|---|
  | `crossover_z` | `float` | hand-over redshift; `float(Tk_WKB.z_init)` |
  | `numeric_region` | `(z_max, z_min)` | evaluable range of `T`, `dT_dz` |
  | `WKB_region` | `(z_max, z_min)` | evaluable range of `M`, `dlnM_dz`, `omega`, `T_WKB`, `friction`, `phase` |
  | `sin_coeff` | `float` | the stored LG sine coefficient (already inside `M`) |
  | `phase` | `LiouvilleGreen.phase_spline.phase_spline` | $\theta$ over the WKB region |
  | `T(z, z_is_log=False)` | `float` | numeric-region $T$; exactly `1.0` above `numeric_region[0]`; `RuntimeError` below `numeric_region[1]` |
  | `dT_dz(z, z_is_log=False)` | `float` | numeric-region $dT/dz$, splined from the stored `Tprime`; exactly `0.0` above the region |
  | `M(z, z_is_log=False)` | `float` | LG amplitude, signed (carries `sin_coeff`) |
  | `dlnM_dz(z, z_is_log=False)` | `float` | $d\ln M/dz$, closed form |
  | `omega(z, z_is_log=False)` | `float` | $d\theta/dz=\sqrt{\texttt{Tk\_omegaEff\_sq}}$, closed form |
  | `T_WKB(z, z_is_log=False)` | `float` | $M\sin(\theta \bmod 2\pi)$ |
  | `friction(z, z_is_log=False)` | `float` | the splined LG friction integral $F$ (extra; see deviation 8) |

  Module constants `MIN_SPLINE_DATA_POINTS = 5` (as `GkSourcePolicyData.py:16`) and
  `PHASE_SPLINE_CHUNK_LOGSTEP = 125`.

  - Amplitude: `M = sin_coeff * sqrt(H_init/H(z)/omega_eff(z)) * exp(F(z))` with
    `H_init = model.functions.Hubble(crossover_z)`, `omega_eff` from
    `WKB_Tk.Tk_omegaEff_sq`, and `F` the only splined ingredient (`ZSplineWrapper` over
    $\log(1+z)$ through the stored `friction` samples). The product is never splined.
  - Derivative, closed form (module docstring cites `TkWKBIntegration.friction_RHS`
    `TkWKBIntegration.py:25-49`, return value at `:49`, `dF/dz = (3/2)(1+c_s^2)/(1+z)`, and
    $d\ln H/dz=\epsilon/(1+z)$):
    `dlnM_dz = -eps/(2(1+z)) - Tk_d_ln_omegaEff_dz/2 + (3/2)(1+cs2)/(1+z)`.
  - Phase: `phase_spline(log(1+z) samples, theta_div_2pi, theta_mod_2pi, x_is_log=True,
    x_is_redshift=True, chunk_step=None, chunk_logstep=125, increasing=True)`.
  - Construction-time consistency checks, both raising `RuntimeError`: `cos_coeff` must vanish
    (`|cos_coeff| <= DEFAULT_FLOAT_PRECISION`); and if a `z_exit` is available (from `k`, else
    from `Tk_numeric`) together with `stop_deltaz_subh`, then
    `|z_exit - stop_deltaz_subh - crossover_z| <= DEFAULT_FLOAT_PRECISION * max(1, |crossover_z|)`.
- **`ComputeTargets/__init__.py`**: `from .TkSourceFunctions import (TkSourceFunctions,)`
  inserted between the `TkNumericIntegration` and `TkWKBIntegration` blocks.
- **new `ComputeTargets/tests/test_tk_source_functions.py`** (12 tests, 0.5 s). Stand-in
  `FakeModel` (constant $w$, copied from `docs/spec-code-audit/scripts/TK_03_numeric_vs_analytic.py`),
  stand-in value/integration/wavenumber objects supplying only the duck-typed protocol, and two
  WKB fixtures (deviation 5). `ComputeTargets/tests/__init__.py` already existed (prompt 03).

Nothing in `TkNumericIntegration.py`, `TkWKBIntegration.py`, `QuadSource.py` or
`QuadSourceIntegral.py` was touched, and no Datastore factory was added.

## Deviations from the prompt

### 1. A class, not a `namedtuple` — IMPLEMENTATION CHOICE

§2 specifies the constructor `TkSourceFunctions(model, k, Tk_numeric, Tk_WKB)`; §3 asks for a
"`namedtuple`/dataclass … (mirror `GkSourceFunctions`'s style)". A `namedtuple` cannot have that
constructor. The alternatives were (a) a `namedtuple` plus a module-level
`build_Tk_source_functions(...)` factory, mirroring `GkSourcePolicyData._create_functions()`
exactly, and (b) a plain class whose `__init__` is the stated constructor and whose fields are
properties. I chose (b): the prompt's constructor is then literally what callers write, the
access syntax prompts 07/08 program against is identical (`f.numeric_region`, `f.T(z)`,
`f.phase`), and there is no host compute target here to own a `_create_functions()`, which is
the only reason `GkSourceFunctions` is a separate record. Cost of the choice: the object is
mutable in principle and cannot be used as a dict key. If a later prompt wants the record shape,
`TkSourceFunctions` can be split into a builder plus a `namedtuple` without changing any call
site.

### 2. `increasing=True`, not `increasing=False`, in the phase spline — STRUCTURALLY REQUIRED

§5 says to build `phase` "exactly as `GkSourcePolicyData._create_functions` does at `:657-671`",
which passes `increasing=False`. That flag is not a formula: it tells `phase_spline` how to sort
its chunks (`phase_spline.py:297-302, 367-370`), and the sort must come out in ascending $x$
order or the constructor raises "chunks are not in increasing min_x-order"
(`phase_spline.py:392-399`). For the Green's function the stored phase at fixed response
redshift *decreases* with the source redshift, so descending `div_2pi` order is ascending $x$;
for the transfer function a single `TkWKBIntegration` integrates $d\theta/dz=+\omega$ towards
smaller $z$ from $\theta(z_{\rm init})=0$, so $\theta$ is negative and *increases* with $z$
(audit `TK-report.md` TK-8(a)) and ascending `div_2pi` is ascending $x$. `increasing=True` is
therefore the correct flag here. `chunk_step`, `chunk_logstep`, `x_is_log`, `x_is_redshift` are
as the prompt specifies. The board's §5 note 5 (every stored phase is negative and decreasing as
$z$ falls) is unaffected and is what forces this.

### 3. Region ends are the sampled ranges, clipped at `crossover_z` — STRUCTURALLY REQUIRED

§3 defines `numeric_region = (first numeric sample, hand-over redshift)` and
`WKB_region = (hand-over redshift, last WKB sample)`. The second cannot be honoured as an
*evaluable* range. `main.py:695-697` builds the WKB grid as
`z_source_sample.truncate(min(z_exit_subh_e3, z_init), keep="lower")`, so its largest sample is
the largest grid point at or below `z_init`, i.e. up to one grid step *below* the hand-over; and
`phase_spline` refuses to be evaluated more than 0.1 % (in $\log(1+z)$) outside its own sampled
range (`phase_spline.py:88-91`), which on `main.py`'s 100-per-decade grid is a fifth of a grid
step. Declaring the region up to `crossover_z` would therefore hand prompt 08 a range in which
the phase raises. Shipped instead:

    numeric_region = (largest numeric sample, smallest numeric sample >= crossover_z)
    WKB_region     = (largest WKB sample <= crossover_z, smallest WKB sample)

so `WKB_region[0] <= crossover_z <= numeric_region[1]`, with equality whenever the grid contains
the hand-over point (which is what the test fixtures do, and what `TkNumericIntegration`'s
`mode="stop"` sampling gives at the numeric end). Nothing is extrapolated. The gap between
`WKB_region[0]` and `numeric_region[1]` is at most one grid step, contains `crossover_z`, and is
documented in the module docstring; `crossover_z` remains a separate field and is still the
nominal split. A test (`test_WKB_grid_starting_below_the_hand_over`) pins this behaviour on a
fixture built in the production shape.

### 4. `k` is duck-typed; `z_exit` is resolved, not assumed — STRUCTURALLY REQUIRED

§2 gives the signature `k: wavenumber` and §3 asks for a check against `k.z_exit`. A
`wavenumber` has no `z_exit` (`CosmologyConcepts/wavenumber.py:19-66`); `z_exit` lives on
`wavenumber_exit_time` (`:225`) and, as a float, on `TkNumericIntegration.z_exit` (`:209`). The
constructor therefore uses `float(k)` for $k/a_0$ (valid for both classes' `__float__`/`.k`) and
resolves `z_exit` from `k` first, then from `Tk_numeric`; if neither carries one the consistency
check is skipped rather than failing. Recorded in the module docstring's protocol.

### 5. Two WKB fixtures, not one — STRUCTURALLY REQUIRED

§6's assertion list cannot be satisfied by a single fixture. Item 3 asks for the stored
`friction` to be chosen so that the code's $M$ formula reproduces the **exact** Bessel envelope
$2^{3/2+b}\Gamma(\tfrac52+b)x^{-3/2-b}m(x)$; items 4b/4c then ask that `M` match that envelope to
$10^{-8}$ *and* that `dlnM_dz` match a finite difference of $\ln M$ to $10^{-6}$, and that
`omega` match `phase.theta_deriv` to $10^{-6}$. But the exact envelope is not the
Liouville–Green amplitude and the exact Bessel phase is not $\int\omega_{\rm eff}$: they differ
by the LG truncation error, which the audit measures at $3.7\times10^{-4}$–$5.6\times10^{-3}$ of
envelope at 3 e-folds sub-horizon (`TK-report.md` TK-8(e)). Measured here on the exact fixture:
`dlnM_dz` vs finite difference $8.5\times10^{-6}$ ($w=1/3$), $7.0\times10^{-5}$ ($w=0.2$), and
$1.0\times10^{-5}$ at both 100 and 300 samples per decade — grid-independent, so truncation and
not spline error. Shipped:

- **fixture "exact"** — `friction` backed out per sample from the exact envelope,
  $\theta=\pi-\vartheta(x)$ with $(m,\vartheta)$ from `bessel_phase`. Carries the $M$ and
  $T_{\rm WKB}$ assertions.
- **fixture "LG"** — `friction` the closed-form integral $\tfrac32(1+w)\ln\frac{1+z}{1+z_{\rm init}}$
  of `friction_RHS`, $\theta$ the DOP853 integral of `Tk_omegaEff_sq` (`atol` 1e-14, `rtol` 1e-13)
  from the hand-over. Carries the `dlnM_dz` and `omega` assertions, which are exact identities on
  it.

The grid-independent LG truncation measurement is itself kept as a test
(`test_LG_truncation_error_of_the_exact_envelope`, threshold $10^{-3}$) so the reason for the
split stays visible.

### 6. `sin_coeff` is backed out from the exact amplitude, not from the $T,T'$ matching — IMPLEMENTATION CHOICE

§6 item 3 offers both. I back it out at the hand-over as
`sin_coeff = M_exact(z_init) * sqrt(omega(z_init))`, which is the $H_{\rm ratio}=1$, $F=0$ limit
of the code's own amplitude formula, i.e. exactly the normalisation `TkWKBIntegration.store()`
arrives at (`:494-506` with `friction_sample[0] = 0`, R26). Re-deriving `raw_sin_coeff` from
`T_init`, `Tprime_init` (`:438-459`) would have tested the *matching*, which belongs to
`TkWKBIntegration` and is already verified exactly by the audit (TK-8(c)), and would have made
the fixture's amplitude only LG-accurate, defeating deviation 5's split. The exact fixture's
$\theta=\pi-\vartheta$ rather than $-\vartheta$ so that the amplitude stays positive:
$\sin(\pi-\vartheta)=\sin\vartheta$, and both rotations put $\theta$ in the code's
increasing-with-$z$ convention.

### 7. Test thresholds — STRUCTURALLY REQUIRED

Two of §6 item 4's $10^{-8}$ thresholds are below the fit error of the grids the prompt itself
prescribes. Shipped thresholds, with the measured maxima in "Verification performed":

| §6 assertion | prompt | shipped |
|---|---|---|
| `T`, `dT_dz` vs exact, numeric region, 100/decade | $10^{-8}$ | $10^{-12}$ *at the sample nodes* (wiring), and $10^{-4}$ ($T$) / $10^{-3}$ ($dT/dz$) at log-midpoints, both relative to the region's envelope |
| `T_WKB` vs exact, relative to envelope | $10^{-8}$ | $10^{-7}$ on a 300/decade WKB grid, plus $10^{-5}$ on the production 100/decade grid and a refinement test asserting $\ge10\times$ improvement from 100 → 300 |
| `M` vs exact | $10^{-8}$ | $10^{-8}$ (met: $1.3\times10^{-13}$) |
| `dlnM_dz` vs finite difference | $10^{-6}$ | $10^{-6}$ on the LG fixture (met: $8.3\times10^{-11}$) |
| `omega` vs `phase.theta_deriv` | $10^{-6}$ | $10^{-6}$ on the LG fixture (met: $4.2\times10^{-8}$) |

The numeric-region midpoint residual is the cubic-spline error of `make_interp_spline` on the
prescribed 100-per-log10(1+z) grid at 3.5 e-folds sub-horizon ($x\approx19$, ~3 cycles), fully
consistent with the audit's own table (QS-5: $1.8\times10^{-4}$ of envelope at 10 cycles). No
choice inside `TkSourceFunctions` affects it — the numeric branch is a spline through the stored
samples by construction, and the prompt forbids differentiating the $T$ spline — so the honest
test is node-exactness plus a quoted fit error. The `T_WKB` residual on the production grid is
the re-splining error of a ~160-cycle phase; see "State handed to the next prompt".

### 8. Extra `friction()` accessor — IMPLEMENTATION CHOICE

Not in §3's table. It is one line, it is the only splined ingredient of the amplitude, and a
consumer that wants to factor $M$ (or to check the LG friction against a model integral, as
prompt 12 might) would otherwise have to re-spline the samples. Alternative considered: keep it
private. Kept public because it costs nothing and prompts 07/08 assemble amplitudes by hand.

### 9. `.z_sample` is not consumed — IMPLEMENTATION CHOICE

§2's protocol lists `.z_sample` for both inputs. The implementation reads `.values` only and
sorts them itself. Reason: in `mode="stop"` a `TkNumericIntegration` holds *fewer* values than
its `z_sample` (`TkNumericIntegration.py:424-426` — "T_sample may not be as long as
self._z_sample"), so `z_sample` is not a description of what was sampled, and the WKB
`z_sample.max` may sit below `z_init`. Using `values` makes both regions self-describing and
lets stand-ins be smaller. `z_sample` remains available to callers; it is simply not required by
the protocol, which the module docstring states.

### 10. Checks raise `RuntimeError` rather than using `assert` — IMPLEMENTATION CHOICE

§2 and §3 say "assert". The codebase's convention for exactly these consistency checks is a
raised `RuntimeError` with a diagnostic message (`GkSourcePolicyData.py:584-588, 626-629`), which
also survives `python -O`. The tests assert that `RuntimeError` is raised.

## Verification performed

`PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` from the
repository root: **17 tests, 0.686 s, OK** (12 new, 5 pre-existing from prompt 03), well inside
the prompt's ~30 s budget. `./venv/bin/black` reports both new files formatted.

Measured maxima printed by the new tests (relative differences; $k/a_0=10^4$, $H_0=1$,
numeric region 5 e-folds super-horizon → 3.5 e-folds sub-horizon, WKB region from there to
$x=k c_s a_0\eta=10^3$; 370/462 numeric and 173/218 WKB samples at 100/decade):

| quantity | $w=1/3$ ($b=0$) | $w=0.2$ ($b=0.25$) |
|---|---|---|
| `T` at the numeric sample nodes | 2.220e-16 | 2.220e-16 |
| `dT_dz` at the numeric sample nodes | 1.879e-16 | 1.692e-16 |
| `T` at numeric log-midpoints | 7.400e-06 | 1.322e-06 |
| `dT_dz` at numeric log-midpoints | 2.720e-04 | 2.701e-04 |
| `M` vs exact envelope (300/decade) | 1.268e-13 | 7.708e-14 |
| `T_WKB` vs $M_{\rm exact}\sin\theta_{\rm exact}$, /local envelope (300/decade) | 3.021e-08 | 2.215e-08 |
| same, vs `scipy.special.jv` (300/decade) | 1.985e-06 | 1.550e-06 |
| same, /local envelope, 100/decade | 6.090e-06 | — |
| `dlnM_dz` vs centred finite difference, LG fixture | 6.355e-11 | 8.323e-11 |
| `omega` vs `phase.theta_deriv`, LG fixture | 4.249e-08 | 2.163e-08 |
| `dlnM_dz` vs finite difference, exact fixture (LG truncation) | 8.528e-06 (100/dec), 1.025e-05 (300/dec) | 6.979e-05 |

Read as follows.

- **The closed forms are right.** On the LG fixture — where the stored `friction` *is* the
  integral of `friction_RHS` and the stored $\theta$ *is* $\int\omega_{\rm eff}\,dz$ — the
  closed-form $d\ln M/dz$ of §4 reproduces a centred finite difference of the object's own
  $\ln M$ to $8\times10^{-11}$, and the closed-form `omega` reproduces the spline derivative of
  the phase to $4\times10^{-8}$. Both are exact identities, so these numbers are the
  finite-difference and phase-spline floors, not physics.
- **The assembly is right.** Fed an exact amplitude–phase decomposition, the object returns it:
  `M` to $10^{-13}$ and $T_{\rm WKB}$ to $3\times10^{-8}$ of the local envelope, the latter
  being the phase re-spline error (it falls by 200× from 100 to 300 samples per decade, the
  $\Delta^4$ of a cubic fit).
- **The $\sim2\times10^{-6}$ column is `bessel_phase`, not this object.** Comparing
  $T_{\rm WKB}$ against `scipy.special.jv` instead of against the fixture's own
  $m\sin\vartheta$ adds `bessel_phase`'s own phase-function error, and that comparison
  *saturates* at $2.0\times10^{-6}$ under grid refinement while the self-consistent comparison
  keeps falling. `LiouvilleGreen/` is out of scope (README §5 item 8), so this is recorded, not
  pursued; it is the same order as the audit's stated phase-spline floor ($\sim2\times10^{-8}$
  quoted for the fit, here $2\times10^{-6}$ of envelope at $x\sim10^3$).
- **Not verified here, and needing a pipeline run:** that a real `TkWKBIntegration`'s stored
  `theta_div_2pi`/`theta_mod_2pi`/`friction`/`sin_coeff` satisfy the same identities, and that
  the crossover consistency check passes on real rows (it compares
  `TkWKBIntegration.z_init` against `k_exit.z_exit - TkNumericIntegration.stop_deltaz_subh`,
  which `main.py:697` sets equal by construction). Prompt 12 owns this.

## Observations not acted on

1. **`ZSplineWrapper`'s error messages all say `GkSource.function:`** (`spline_wrappers.py:42`,
   `:52`, and the `GkWKBSplineWrapper` copies at `:101`, `:111`). Audit B10 fixed the *label*
   passed in by `QuadSource`, not this hard-coded prefix, so a range error from the new
   `"numeric T_k"` spline reads `GkSource.function: evaluated numeric T_k out of bounds`. One
   line in a file no prompt in this campaign owns; left alone.
2. **`phase_spline.theta_mod_2pi` does not preserve the stored negative-remainder convention.**
   Each chunk rebases $\theta$ by an integer number of cycles (`phase_spline.py:49-51`) and the
   accessor returns `fmod(spline value, 2pi)` (`:192`), so the remainder is congruent to
   $\theta$ mod $2\pi$ but is typically *positive*, unlike the samples produced by
   `WKB_mod_2pi`/`wrap_theta`. Harmless — it is only ever an argument to `sin`/`cos` — but a
   consumer that tests the sign of a remainder (or that compares two phases' remainders without
   reducing) will be surprised. The test asserts the invariant that actually holds
   (`sin(remainder) == sin(theta_exact)`).
3. **`TkWKBIntegration` stores `H_ratio` and `omega_WKB_sq` per value; this object recomputes
   both from the model.** That is what §4 asks for (closed forms, no splined products), and the
   two agree by construction. It does mean a caller who has already paid for the stored values
   pays again for `Hubble` and `Tk_omegaEff_sq` calls: `M` costs one `Hubble`, one
   `wPerturbations`, two `epsilon`-family and two `w`-derivative spline evaluations through
   `Tk_omegaEff_sq`. If prompt 08's per-node cost turns out to matter, the amplitude could be
   memoised per $z$ or the stored `H_ratio`/`omega_WKB_sq` splined instead; not done, because it
   would put a splined product back into the amplitude.
4. **No `TkSourcePolicyData` is needed and none was added**, confirming README §2(a) against the
   code: `main.py:697` sets `z_init = k_exit.z_exit - Tk.stop_deltaz_subh` and `main.py:686-688`
   sets `T_init`/`Tprime_init` from the same stop point, so the hand-over is a single already
   determined redshift, recoverable from `TkNumericIntegration` alone. Nothing needed to be
   persisted, so §1's instruction to stop and open a §3 issue did not trigger.

## State handed to the next prompt

Prompts 07 and 08 program against the following, verbatim.

```python
from ComputeTargets import TkSourceFunctions          # or ComputeTargets.TkSourceFunctions

f = TkSourceFunctions(model, k, Tk_numeric, Tk_WKB)   # model: BackgroundModel; float(k) = k/a0

f.crossover_z            # float
f.numeric_region         # (z_max, z_min), z_min >= f.crossover_z
f.WKB_region             # (z_max, z_min), z_max <= f.crossover_z
f.sin_coeff              # float, already included in f.M(...)
f.phase                  # LiouvilleGreen.phase_spline.phase_spline over the WKB region

f.T(z, z_is_log=False)         # numeric region; exactly 1.0 above numeric_region[0]
f.dT_dz(z, z_is_log=False)     # numeric region; exactly 0.0 above numeric_region[0]
f.M(z, z_is_log=False)         # WKB region; signed
f.dlnM_dz(z, z_is_log=False)   # WKB region; closed form
f.omega(z, z_is_log=False)     # WKB region; = d(theta)/dz, closed form
f.T_WKB(z, z_is_log=False)     # WKB region; M * sin(theta mod 2pi)
f.friction(z, z_is_log=False)  # WKB region; the splined LG friction integral F
```

1. **Three regions, and the boundary is not one point.** Above `numeric_region[0]`,
   $T=1$ and $dT/dz=0$ *exactly* (no spline is consulted). Below it, `T`/`dT_dz` are valid down
   to `numeric_region[1]` and `M`/`dlnM_dz`/`omega`/`T_WKB`/`friction`/`phase` are valid from
   `WKB_region[0]` down to `WKB_region[1]`. `WKB_region[0] <= crossover_z <= numeric_region[1]`,
   and in production the two differ by at most one grid step (deviation 3). **Partition your
   integrals on `crossover_z` but clamp your quadrature nodes to `numeric_region` and
   `WKB_region`** — every accessor raises `RuntimeError` outside its own range, and the phase
   cannot be extrapolated at all.
2. **`z_is_log=True` takes $\log(1+z)$**, matching `ZSplineWrapper.__call__`. Use it: every
   accessor otherwise recomputes `log(1+z)`.
3. **`M` is signed** and may be negative (it carries `sin_coeff`). Do not take an absolute
   value; `T = M sin(theta)` with the stored $\theta$ is the whole representation, since
   `cos_coeff` is identically zero and construction refuses data where it is not.
4. **Phase sign and remainders.** $\theta<0$, increasing with $z$ (board §5 note 5); the phase
   spline is built with `increasing=True` (deviation 2). `f.phase.theta_mod_2pi(z)` is congruent
   to $\theta$ mod $2\pi$ but its *sign is not* the stored negative-remainder convention
   (observation 2) — use it only inside `sin`/`cos`, and use `f.phase.raw_theta(z)` when you
   need to add or subtract phases. `f.omega(z) == f.phase.theta_deriv(z)` to
   $4\times10^{-8}$; prefer `f.omega` in a Levin phase-derivative slot, since it is closed-form
   and free of chunk-boundary effects.
5. **Accuracy floors, for prompt 08's error budget.** With the production 100-per-log10(1+z)
   grid, on an exact fixture at 3.5 e-folds sub-horizon down to $x=10^3$ (~160 cycles):
   - LG branch: $T_{\rm WKB}$ reproduces the exact oscillation to **6.1e-06 of the local
     envelope**, all of it the phase re-spline's cubic fit error ($\propto\Delta^4$: 3.0e-08 at
     300/decade). Crucially this floor is set by the phase *range per chunk*
     (`chunk_logstep=125`, so $\lesssim125$ cycles), not by the total cycle count, provided each
     chunk keeps $\ge30$ samples (`phase_spline.MINIMUM_SPLINE_DATA_POINTS`) — i.e. $\ge30$
     samples per 125 cycles. Compare audit A2's numeric-value spline: 0.88 of envelope at 95
     cycles and $>1$ beyond. **This is the quantitative justification for A2's fix.**
   - Numeric branch: 7.4e-06 ($T$) and 2.7e-04 ($dT/dz$) of envelope between grid points at the
     hand-over, falling steeply super-horizon. So the *numeric* region, not the LG region, is
     now the accuracy floor of the source integrand, and its worst point is the hand-over.
     Prompt 06 should keep the numeric region as short as the LG validity allows.
   - The amplitude assembly itself contributes $10^{-13}$; `dlnM_dz` and `omega` are exact
     identities to $10^{-10}$ and $4\times10^{-8}$.
6. **The LG representation is not the exact solution.** The LG amplitude and the exact envelope
   differ by the LG truncation error: $\sim10^{-5}$ relative in $d\ln M/dz$ at 3.5 e-folds
   sub-horizon, grid-independent, and $3.7\times10^{-4}$–$5.6\times10^{-3}$ in $T$ itself at
   3 e-folds (audit TK-8(e)). Do not chase residuals of that size against an exact oracle in the
   first few e-folds below the hand-over; they are the representation, not a bug.
7. **Test fixtures are reusable.** `ComputeTargets/tests/test_tk_source_functions.py` exposes
   `FakeModel`, `FakeRedshift`, `FakeNumericValue`, `FakeWKBValue`, `FakeTkNumeric`,
   `FakeTkWKB`, `FakeWavenumber`, `log_grid`, `log_midpoints` and `Fixture` (with
   `Fixture.exact_functions()` / `Fixture.LG_functions()`). Prompts 07/08 can import them
   directly for a $T_q$/$T_r$ pair at two different $k$; `Fixture(w, k=...)` is the only knob
   needed, and `x_max`/`WKB_samples` control the cycle count.
8. **No schema, no persistence, no `Datastore` change**, and `MetadataConcepts/QuadSourcePolicy`
   is untouched.
