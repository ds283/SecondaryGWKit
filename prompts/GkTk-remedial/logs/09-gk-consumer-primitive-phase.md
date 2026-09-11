# Log 09 — `PrimitivePhase`, and the Green's-function consumer

**Prompt:** prompts/GkTk-remedial/09-gk-consumer-primitive-phase.md
**Commit:** *(this commit)* — Evaluate the Green function phase from the conformal-time table
**Model:** Claude Opus 5 — **Fable was unavailable in this session**; README §3 and orchestrator
README rule 5 name Opus as the documented fallback for the three Fable prompts ("If Fable is
unavailable use Opus and review those three most closely"). This is one of them.
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

### 1. New `ComputeTargets/primitive_phase.py`

The consumer-side counterpart of prompt 06's producer rewrite: the phase is no longer splined, it
is *evaluated*.

```python
class PrimitivePhase:
    def __init__(self, k: float, leading, z_anchor: float,
                 z_samples: Sequence[float], phi_samples: Sequence[float], *,
                 sign: int, model_functions, label: str = "",
                 spline_order: int = 3)

    # the phase_spline protocol, exactly the three methods every consumer calls
    def raw_theta(self, x: float, x_is_log: bool = False) -> float
    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float
    def theta_deriv(self, x: float, x_is_log: bool = False,
                    log_derivative: bool = False) -> float
    @property
    def num_chunks(self) -> int          # always 1

    # inspection
    def phi(self, x: float, x_is_log: bool = False) -> float
    k, sign, z_anchor, z_samples, phi_samples, spline_order, label   # properties
    min_x, max_x, min_log_x, max_log_x                               # attributes


def build_phi_samples(k, leading, z_anchor, z_samples, theta_samples, *, sign) -> np.ndarray
```

* `raw_theta(x) = sign * k * leading.delta(z, z_anchor) + phi(z)`, with `leading` a
  `BackgroundModel.TablePrimitive` (prompt 03) — `functions.tau` here, `functions.cs_tau` for
  prompt 10 — reached through its double-double interval accessor, never through a difference of
  two pointwise values. `phi` is a `make_interp_spline` in `u = log(1+z)`, cubic by default,
  `spline_order=5` accepted, anything else refused.
* `sign = -1` for the Green's function at fixed `z_response` (θ decreases as `z_source` rises);
  `sign = +1` for the transfer function at fixed `z_init`. The same expression serves both because
  `CumulativeTable.delta` is antisymmetric (README §2 (c)), so prompt 10 needs no second class.
* `theta_mod_2pi` is `WKB_mod_2pi(raw_theta(...))[1]` — one reduction of one unreduced value,
  negative-remainder convention, against the **global** anchor (D6). The module docstring states
  the `eps k tau` floor this implies (1e-7 rad at `k = 1e5`, 9e-4 rad at `3e8`), why it is
  accepted (three to four orders under the consumer error it replaces and at or under the QCD LG
  truncation floor), and names `[00-consumer-anchoring-floor]` as the recorded follow-up.
* `theta_deriv` is closed form for the leading term:
  `d theta/dz = sign * k / H(z) + phi'(z)` (`d/dz leading.delta(z, z_anchor) = +f(z) = 1/H` since
  `CumulativeTable` accumulates `T(z) = int_z^{z_top} f`), and
  `d theta/d log(1+z) = (1+z) d theta/dz`. Nothing spectrally differentiates a sampled phase.
  `model_functions` is used for `Hubble` alone.
* Range discipline is `phase_spline`'s, reusing its `SPLINE_TOP_BOTTOM_CUSHION` (imported, not
  copied): clamp inside the cushion, `RuntimeError` beyond it. **Only the `phi` spline is
  clamped** — the leading term is evaluated at the requested redshift whatever happens, because
  the table is exact over the whole background grid, so `theta` stays continuous at the boundary
  instead of acquiring a kink (deviation 6).
* `build_phi_samples` is the inverse: `phi_i = theta_stored_i - sign * k * leading.delta(z_i,
  z_anchor)`. Its docstring records that `theta_stored` must come from the **rectified**
  `theta_div_2pi` and that `phi` is defined only up to the exact multiple of 2π the rectifier's
  rebasing of the first sample carries.

### 2. `ComputeTargets/GkSourcePolicyData.py`

* `:12` `from LiouvilleGreen.phase_spline import phase_spline` → `from
  ComputeTargets.primitive_phase import PrimitivePhase, build_phi_samples` plus `from
  LiouvilleGreen.constants import TWO_PI`. `grep -n "phase_spline"` on this file is now empty.
* New module-level `GK_PHASE_SIGN = -1` and `_build_phase(source, WKB_data, label) ->
  PrimitivePhase` (`:136-195`), shared by both consumers: `k = source.k.k`,
  `leading = source.model_proxy.get().functions.tau`, `z_anchor = source.z_response.z`,
  `z_points = [v.z_source.z …]`, `theta_points = [v.WKB.theta_div_2pi * TWO_PI +
  v.WKB.theta_mod_2pi …]` (the **rectified** counts), `phi` from `build_phi_samples`.
* `_classify_Levin` (`:169-186` before → `:231-236` after): the single-chunk `phase_spline` and
  its "edge effects near the chunk boundaries" comment are gone; the threshold test now calls
  `theta_phase.theta_deriv(z_source.z, log_derivative=True)` on the `PrimitivePhase`. The
  threshold semantics and `Levin_z`'s diagnostic status are unchanged.
* `_create_functions` (`:672-690` before → `:722-725` after): the chunked `phase_spline` and the
  false comment at `:676-680` ("chunking keeps the `theta_div_2pi` rebasing well-conditioned over
  many oscillation cycles. Audit B9") are deleted. `WKB_theta_spline` is renamed
  `WKB_theta_phase`; `GkSourceFunctions.phase` is now a `PrimitivePhase`.
* `GkWKBSplineWrapper` receives it unchanged (verified: it calls `theta_mod_2pi(log_z,
  x_is_log=True)` and nothing else).

### 3. `ComputeTargets/__init__.py`

Exports `PrimitivePhase` and `build_phi_samples`.

### 4. Tests

* New `ComputeTargets/tests/test_primitive_phase.py` (17 tests): the interpolation law head to
  head against a `phase_spline` of the same samples, the closed-form derivative, the mod-2π
  convention, a non-zero smooth `phi`, the protocol, and the constructor contract.
* New `ComputeTargets/tests/test_gk_source_primitive_phase.py` (6 tests): a faithful copy of the
  `GkSource` rectifier (`rectify(div_2pi, mod_2pi) -> (rectified, corrections)`), a `Sweep` class
  reproducing prompt 06's cross-object geometry, and the three statements the prompt asks for.
* `ComputeTargets/tests/test_gk_source_policy.py`: `FakeGkSource` gains `k` (a `FakeWavenumber`
  at `k = 1/Mpc`), `z_response` and `model_proxy` (a radiation `FakeModel` whose `functions.tau`
  is a real `CumulativeTable`/`TablePrimitive` on the fixture's own sample redshifts). The three
  existing assertions are unchanged; one assertion was **added**, that the recorded
  `Levin_z_dtheta_dlogz` is still exactly the fixture's `rate`.

---

## Deviations from the prompt

### 1. Test 1's accuracy threshold is below the representation floor — STRUCTURALLY REQUIRED

**The prompt assumed:** §4 `test_primitive_phase.py` test 1 — "max error at 10 points per interval
≤ 1e-8 rad … This is README §6's consumer row" — on the geometry it fixes in the same sentence:
exact radiation, `k = 1e8`, `z_r = 0.1`, `z_s ∈ [10, 1e4]`, 100 samples per decade.

**What is actually there:** on that geometry `|theta|` reaches **9.0899e7 rad** (at `z_s = 1e4`,
where `x_s` is *smallest*). One ulp of that is **1.4901e-8 rad**; the campaign's own `eps k tau`
floor (README §2 (d), §5 standing note 2, decision D6) is **2.019e-8 rad**. A double cannot carry
the answer to 1e-8 rad, let alone a computation of it: `raw_theta` is a table difference times `k`
plus a spline value, three roundings. Measured against a 50-digit mpmath reference the error is
**4.189e-8 rad = 2.81 ulp of the span** (at `z_s = 1730.75`); at the samples themselves, where the
spline contributes nothing, **3.681e-8 rad**. README §5 standing note 2 is explicit that "a test
asserting below a floor is asserting agreement between two errors".

**What was done instead:** the module asserts (i) README §6's consumer row, **≤ 1e-6 rad**, which
is the campaign acceptance table's number for this row and is met with 24× margin; and (ii) a
floor-aware **≤ 6 ulp** of the geometry's own span, which pins the result *to the floor* rather
than to a round number and would fail if the decomposition ever stopped being at the floor. The
prompt's other half of the same test — "build both and assert the ratio > 1e5" against the
`phase_spline` of the same samples — **passes as written**: 7.286e-3 / 4.189e-8 = **1.739e5**.
The docstring of `test_primitive_phase.py` states all of this at the top.

*Orchestrator note.* README §4.3 lists "a numerical acceptance threshold in the prompt or §6 is
missed, even narrowly" as a stop condition. §6's threshold is met; the prompt's own tighter number
is not, and cannot be by anything this campaign could build, because it is below the
double-precision floor of the quantity being asserted. The user may wish to correct the prompt
text rather than the code.

### 2. Prompt 06's `_sweep` could not be reused directly — STRUCTURALLY REQUIRED

**The prompt assumed:** "Using prompt 06's cross-object sweep machinery".

**What is actually there:** `ComputeTargets/tests/test_gk_wkb_phase._sweep` returns each object's
phase at the response point already *combined*, as `"stored": div2[j] * TWO_PI + mod2[j]`. The
rectifier consumes the `(div, mod)` pair, and recovering it by re-reducing a ~1e8 rad double is
exactly the anti-pattern README §2 (e) forbids. `test_gk_wkb_phase.py` is not in this prompt's
"files you may touch", so `_sweep` could not be extended.

**What was done instead:** `test_gk_source_primitive_phase.Sweep` reproduces the same geometry
(stop at the first maximum of `G` past the 4-e-fold point; `theta_i = -(3π/2 + 2πn)`; exact
initial data; 100 samples per decade; response sparseness 12) and **imports `store_algebra` from
prompt 06's module verbatim**, so the producer algebra under test is the shipped one. It returns
`div`, `mod`, `n`, `delta` and the exact phase per object.

### 3. The sweep's source samples are background-grid nodes — IMPLEMENTATION CHOICE

Prompt 06's `_sweep` draws the numeric-initialised band from its own
`geomspace(s_lim, s_e3, …)`, independent of the response grid. `Sweep` instead takes both the
source samples and the response point from the background grid `geomspace(1, s_e3, …)`, because
that is production (`main.py:476`: the background model is built on the source grid, and the
response grid is a subset), and because it makes every `tau.delta` call on-grid at both ends, so
the test measures the decomposition and not an off-grid partial.

The consequence is that the sample redshifts differ from prompt 06's log by up to one grid
spacing (2.33 %): the two stop-point transitions at `k = 1e7`, `x_r = 1e3` land at
`z_s = 321275 → 328768` and `395358 → 404579` where log 06 reports `316700 → 324331` and
`401839 → 411522`. The test asserts agreement to within one grid spacing and prints both. Every
*count* reproduces exactly: 22 objects and 2 transitions for that case, **990 objects and 90
transitions** over prompt 06's 15 × 3 sweep — log 06's figures to the object.

Alternative considered: copy `_sweep`'s independent `geomspace` and accept off-grid `tau.delta`
calls at every sample. Rejected because it would have made `phi` carry an off-grid partial's error
at every point and obscured the very thing being measured, for no gain — the transitions are a
property of the stop condition, not of the grid.

### 4. Test 3's `rectified == raw` needed the geometry chosen for it — IMPLEMENTATION CHOICE

The prompt asks, for pure-WKB objects, "assert `rectified_theta_div_2pi == raw` for every sample".
The rectifier *always* rebases the first (lowest-`z`) sample to `rectified = 0` and subtracts the
same constant from every later one, so that identity holds only when the lowest sample's raw
`div_2pi` is already 0. The pure-WKB band in `Sweep` therefore starts at the response point
itself, where `theta = 0` exactly — which is also production (`GkSource`'s `z_sample` is the
source grid restricted to `z_source >= z_response`, so its minimum is the response redshift). With
that choice the prompt's literal assertion holds, and the test asserts it.

It also asserts the statement that does not depend on the geometry and is what "the rectifier
applies zero corrections" means: the non-monotonicity branch fires **0 times**. Both are checked
for three `(k, x_r)` combinations.

### 5. `build_phi_samples`, `PrimitivePhase.phi()` and `_build_phase` are additions — IMPLEMENTATION CHOICE

The prompt specifies the class and its five members. Three small things were added:

* `build_phi_samples(...)` at module scope, so the `phi` recipe — and the requirement that the
  phase fed to it be the *rectified* one — lives next to the class that consumes it, and prompt 10
  reuses it rather than restating it. The alternative, a classmethod constructor
  `PrimitivePhase.from_stored(...)`, was rejected because the two consumers assemble their sample
  lists differently and would have had to pass them twice.
* `PrimitivePhase.phi(x, x_is_log)`, returning the residual alone. Needed by
  `test_primitive_phase` test 4: `phi` is recovered to 2.16e-10 rad, which cannot be measured
  through `raw_theta`, whose own noise is 4.19e-8 rad.
* `GkSourcePolicyData._build_phase(source, WKB_data, label)`, because `_classify_Levin` and
  `_create_functions` build the identical object and the prompt asks for the identical
  construction in both.

### 6. Only `phi` is clamped at the range boundary — IMPLEMENTATION CHOICE

`phase_spline` clamps its abscissa into the sampled range inside the cushion and evaluates the
spline there. `PrimitivePhase` clamps the `phi` spline the same way but evaluates the *leading*
term at the requested redshift, because the `CumulativeTable` behind it is exact over the whole
background grid and has its own (much wider) range check. Clamping both would freeze `theta`
inside the cushion — a kink of up to `k Δτ(cushion)` in a quantity a Levin consumer
differentiates; clamping `phi` alone leaves `theta` continuous and correct to the variation of
`phi` over the cushion, which is O(1e-3) rad. Pinned by
`TestProtocol.test_only_phi_is_clamped`.

### 7. `k` is read through the public accessor — IMPLEMENTATION CHOICE

The prompt writes `k = source._k_exit.k.k`. `GkSource.k` (`:471-473`) returns exactly
`self._k_exit.k`, so `source.k.k` is the same object by the public property; that is what
`_build_phase` uses. The model *is* available on the source — `GkSource.model_proxy` (`:475-477`)
— so nothing had to be threaded from the caller, as the prompt's fallback anticipated.

---

## Verification performed

All runs from the worktree root with `PYTHONPATH=.` and `./venv/bin/python`.

### `test_primitive_phase.py` — 17 tests, OK

| prompt §4 item | measured | asserted |
|---|---|---|
| 1. `raw_theta` vs exact, 10 pts/interval, `k = 1e8`, `z_r = 0.1`, `z_s ∈ [10, 1e4]`, 296 samples, 2890 interior points | **4.189e-8 rad** at `z_s = 1730.75` (2.81 ulp of the 9.0899e7 rad span; at the samples 3.681e-8 rad) | ≤ 1e-6 rad (README §6) and ≤ 6 ulp — see deviation 1 |
| 1. the same `phase_spline` (`chunk_logstep=125`, `increasing=False`) | **7.286e-3 rad** (review §5 scaled: 8.26e-3 at `x = 1e8`) | ≥ 1e-3 rad |
| 1. ratio | **1.739e5** | > 1e5 ✅ |
| 2. `theta_deriv`, both `log_derivative` values, vs `-k/H` and `-k(1+z)/H` | **0.0 relative**, both (the `phi` spline of an all-zero sample has exactly zero derivative) | ≤ 1e-12 |
| 3. `theta_mod_2pi ∈ (-2π, 0]` and `== WKB_mod_2pi(raw_theta)[1]` | exact equality at every point | bitwise |
| 4. `phi = 0.3 sin(log(1+z))` recovered | **2.157e-10 rad** against `h^4 max|phi''''|/384 = 2.19e-10` (`h = 0.02302`) | ≤ 1e-8 rad and ≤ 10× the law |
| 4. `theta_deriv` includes `phi'` | `d theta/d log(1+z)` minus the closed-form leading term reproduces `0.3 cos(u)` to **2.88e-8** | ≤ 1e-6 |
| 5. protocol | `num_chunks == 1`; `x_is_log=True` calls at `min_log_x`, mid, `max_log_x` finite; clamped inside the cushion; `RuntimeError` at 1 % outside, both ends, for `raw_theta` and `theta_deriv` | — |
| extra | quintic `phi` spline: **1.37e-14 rad**; `build_phi_samples` round trip at the samples **0.0 rad** exactly, recovering a planted 3-cycle offset to 1e-6 | — |

Floors printed by the test for the record: `eps*k*tau = 2.019e-8 rad`, 1 ulp = 1.490e-8 rad.

### `test_gk_source_primitive_phase.py` — 6 tests, OK

The rectifier copy, the `Sweep` geometry, and:

| prompt §4 item | measured |
|---|---|
| 1. the named case (`k = 1e7`, `x_r = 1e3`, `z_response = 10350.8`) | **22 objects** in `z_s ∈ [301973, 497870]`; **2** `phi` jumps before rectification, each **+1.0000 cycles**, at `z_s = 321275 → 328768` and `395358 → 404579`, which are **exactly** the 2 stop-point transitions; **2 corrections** applied; after rectification **max \|Δphi\| = 2.27e-13 rad**, `phi = 929.911 rad = 148.000000 cycles` (constant) |
| 1. the full sweep, 15 `k` × 3 `x_r` | **990 objects, 90 `phi` jumps = 90 stop-point transitions = 90 rectifier corrections**; after rectification **max \|Δphi\| = 3.64e-12 rad** (< 0.1 rad required). Log 06's figure: "90 steps among 990 objects" |
| 2. `raw_theta` vs the exact radiation phase, 210 interior points | `raw_theta − exact = 148 cycles + 4.55e-13 rad`; the offset is whole cycles to 1e-9 of a cycle. `\|theta\|` up to 945.931 rad |
| 2. extra: `theta_mod_2pi` at the samples vs the stored remainder | **1.58e-13 rad** |
| 3. pure-WKB band, three `(k, x_r)` | `delta == 0.0` exactly for every object; **0 rectifier corrections**; `rectified == raw` for every sample; max \|phi\| **4.55e-13** (`k=1e7`, 147 objects), **2.13e-14** (`k=1e6`, 51), **3.64e-12** (`k=1e8`, 243) |
| 3. extra: `raw_theta` vs exact over the pure-WKB band, 1460 interior points | **3.41e-13 rad** |

The `+1 cycle` direction confirms `RECONCILIATION.md` §2 item 6: the step is in the
monotonicity-violating direction, which is the only direction the rectifier can detect.

### `test_gk_source_policy.py` — 3 tests, OK

The three original assertions hold unchanged. The recorded `Levin_z_dtheta_dlogz` is **5000.0000**
against the fixture's `rate = 5000`, i.e. the closed-form leading term plus the splined residual
recover an exactly linear phase to better than 1e-9 relative — the new assertion.

### Full suite

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
Ran 249 tests in 125.4s
OK
```

`test_quadsource_integral.py` and `test_phase_groups.py` are **unmodified** (`git status`) and pass
inside that run — the protocol check the prompt asks for.

### Acceptance checks

* `grep -n "phase_spline" ComputeTargets/GkSourcePolicyData.py` → **empty**.
* `./venv/bin/python -m black --check ComputeTargets LiouvilleGreen Quadrature` → 76 files
  unchanged. (A repository-wide `black --check` reports 54 files needing reformatting, all under
  `docs/`; that is pre-existing and none of them is touched here.)

### Smoke test of `_create_functions` (reasoned + run, not a shipped test)

`_create_functions` cannot be unit-tested without a `GkSource` (it was not covered before this
prompt either), so it was driven once from a scratch script on duck-typed stand-ins carrying real
`redshift` objects, a radiation `CumulativeTable` and exact-radiation stored phases at
`k = 1e7`: `functions.phase` is a `PrimitivePhase` with `num_chunks == 1`; `raw_theta` at a sample
returns `-829.8334679744298` against the stored `-829.8334679744298` (bit-identical);
`theta_mod_2pi` returns the stored remainder bit-identically; `GkWKBSplineWrapper` evaluates; and
`theta_deriv(log_derivative=True)` gives `-155.75178648966406` against the closed form
`-155.7517864896702`. The script was not kept (it is not in this prompt's file list and duplicates
what `test_gk_source_primitive_phase` measures with a real rectifier).

### Cost of the new evaluation pattern

Measured on the radiation stand-in: `PrimitivePhase.raw_theta` costs **0 integrand evaluations**
when the abscissa is a table node and **exactly 4** (one order-4 panel) when it is not; the anchor
never costs anything, because `z_response` is a background-grid node. Wall time **3.11 µs**
on-grid / **5.85 µs** off-grid per call, best of 3 over 20 × 286 calls. See the observations for
what this means for `[01-offgrid-accessor-cost-on-qcd]`.

---

## Observations not acted on

1. **`[01-offgrid-accessor-cost-on-qcd]` is narrowed, not closed.** The issue's next step reads
   "closes when prompt 09 confirms its evaluation pattern is on-grid". It is not: a Levin region
   evaluates the phase at Chebyshev abscissae, which are off-grid by construction. What *is*
   confirmed is that the pattern is always **one** off-grid endpoint, never two — the anchor
   `z_response` is a background-grid node and costs nothing — so the relevant figure from log 03
   is **26.1 µs on `QCD_Cosmology`** and 4.6 µs on LambdaCDM, not the 52.0 µs both-off-grid
   number the issue was opened against. Recorded on the board.

2. **The anchor partial is free only while the anchor is on-grid.** `CumulativeTable.delta`
   re-integrates an off-grid endpoint's panel on every call, so a future `PrimitivePhase` whose
   anchor is off the grid (prompt 10's `z_init` is the candidate — it is a `root_scalar` root,
   `RECONCILIATION.md` §2 item 5) would pay one extra order-4 panel per evaluation. That is
   precisely item 2 of `[07-tk-per-object-cost-is-all-setup]`, and the same
   `nearest_table_node` split prompt 14 applied to the residual table would remove it. Not in
   this prompt's scope; flagged for prompt 10.

3. **`spline_wrappers.GkWKBSplineWrapper.__init__`'s type hint still says
   `theta_spline: phase_spline`.** It is duck-typed and works unchanged (verified: it calls only
   `theta_mod_2pi`), but the annotation is now wrong for the production path. `spline_wrappers.py`
   is "verify, do not edit" in this prompt's header, so it was left alone. A one-line annotation
   change whenever that file is next opened.

4. **`extract_GkSource_data.py` needs no change.** It reads `functions.phase.raw_theta(z)` and
   `functions.phase.theta_deriv(z)` (`:275, :285, :291, :618`) — protocol methods only — so it
   keeps working against a `PrimitivePhase`. Recorded because the campaign's other consumer
   changes did strand `docs/` scripts (`[06-docs-scripts-reference-removed-ode]`,
   `[08-docs-scripts-reference-removed-chunking]`); this one does not.

5. **`phi` carries a constant that is an exact multiple of 2π.** The rectifier rebases the first
   sample into the fundamental block, so `phi` is the physical residual plus `2π N₀` — 148 cycles
   (929.9 rad) in the measured case. It is constant, so it costs the spline nothing and
   `theta_mod_2pi` is unchanged by it; but it means `phi` is *not* small in absolute terms, and
   anyone reading `phi_samples` as "the residual `-Δρ`" should subtract its mean first. Documented
   in `build_phi_samples`.

6. **`_classify_Levin` still builds its phase object unconditionally.** It has no
   `MIN_SPLINE_DATA_POINTS` guard, so a WKB region with fewer than four samples raised inside
   `make_interp_spline` before and raises inside `PrimitivePhase` now (with a clearer message
   naming the spline order and the count). Behaviour is equivalent; no guard was added, because
   adding one would change which sources are classified.

---

## State handed to the next prompt

**`PrimitivePhase`'s full interface (verbatim), `ComputeTargets/primitive_phase.py`:**

```python
from ComputeTargets.primitive_phase import PrimitivePhase, build_phi_samples
# also exported from ComputeTargets/__init__.py

class PrimitivePhase:
    def __init__(self, k: float, leading, z_anchor: float,
                 z_samples: Sequence[float], phi_samples: Sequence[float], *,
                 sign: int, model_functions, label: str = "",
                 spline_order: int = 3)
    def raw_theta(self, x: float, x_is_log: bool = False) -> float
    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float
    def theta_deriv(self, x: float, x_is_log: bool = False,
                    log_derivative: bool = False) -> float
    def phi(self, x: float, x_is_log: bool = False) -> float
    num_chunks -> int            # property, always 1
    k, sign, z_anchor, spline_order, label            # properties
    z_samples, phi_samples                            # properties, np.ndarray, ascending in z
    min_x, max_x, min_log_x, max_log_x                # attributes (floats)

def build_phi_samples(k, leading, z_anchor, z_samples, theta_samples, *, sign) -> np.ndarray
    # phi_i = theta_stored_i - sign * k * leading.delta(z_i, z_anchor)
```

**The formula and the `sign` convention.**
`raw_theta(x) = sign * k * leading.delta(z, z_anchor) + phi(z)`.
`sign = -1` for the Green's function at fixed `z_response`;
**`sign = +1` for the transfer function at fixed `z_init`** — prompt 10 passes
`leading = model.functions.cs_tau`, `z_anchor = z_init`, `sign=+1`, and gets
`theta_T(z) = +k cs_tau.delta(z, z_init) = -k cs_tau.delta(z_init, z)`, which is what prompt 07's
producer stores. `theta_deriv` then returns `+k/H(z) + phi'` — note the sign follows `sign`
automatically; nothing else changes. `PrimitivePhase` is one object for both sectors and prompt 10
should not subclass or copy it.

**How `model` was obtained in `_create_functions`.** From the source itself:
`source.model_proxy.get()` (`GkSource.model_proxy`, `ComputeTargets/GkSource.py:475-477`), and
`k = source.k.k`, `z_anchor = source.z_response.z`. Nothing is threaded from the caller, and
`GkSourcePolicyData` did not need a new field. `TkSourceFunctions` (prompt 10) has its own model
access; check it the same way before threading anything.

**Constructor requirements prompt 10 must satisfy.** `z_samples` must be distinct in `log(1+z)`
and at least `spline_order + 1` long (4 for the default cubic); `sign` must be exactly `-1` or
`+1`; `model_functions` must expose `Hubble`; `leading` must expose `delta(z_a, z_b)`. A stand-in
`ModelFunctions` in a test fixture therefore needs **both** `Hubble` and the `TablePrimitive` — a
`cs_tau` with `.delta` is not enough on its own.

**Measured errors (prompt 09's §4 test 1 and the ratio).**
`k = 1e8`, `z_r = 0.1`, `z_s ∈ [10, 1e4]`, 100/decade, 10 points per interval, mpmath at 50
digits: `PrimitivePhase` **4.189e-8 rad** (2.81 ulp of the 9.0899e7 rad span; 3.681e-8 rad at the
samples), `phase_spline` of the same samples **7.286e-3 rad**, **ratio 1.739e5**. The `eps k tau`
floor there is 2.019e-8 rad and one ulp is 1.490e-8 rad, so prompt 09's stated 1e-8 rad threshold
is below the floor — **prompt 10 should not copy that number**; README §6's row is 1e-6 rad and is
the one to score against. A `phi`-only measurement (test 4) gives **2.157e-10 rad** against the
cubic law `h^4 max|phi''''|/384 = 2.19e-10`, which is the accuracy the representation actually
delivers once the leading term is taken out.

**`phi` statistics on the stand-in.** Numeric-initialised band, `k = 1e7`, `x_r = 1e3`, 22
objects: after rectification `phi = 929.911 rad` **constant** = exactly 148 cycles, `max |Δphi|`
between neighbours **2.27e-13 rad**; before rectification `max |Δphi| = 2π` at each of the 2
stop-point transitions. Over the 990-object sweep, `max |Δphi| = 3.64e-12 rad` after
rectification. Pure-WKB band: `delta = 0` exactly, `phi` **≤ 3.64e-12 rad** (identically zero in
radiation), 0 rectifier corrections.

**Cost.** `raw_theta` costs 0 integrand evaluations at a table node and exactly 4 (one order-4
panel) off-grid; the anchor is free while it is a node. 3.11 µs / 5.85 µs per call on the
radiation stand-in. On `QCD_Cosmology` read log 03's one-endpoint-off-grid figure, **26.1 µs**.

**`chunk_logstep` in production is now gone from the Green's-function path.** Nothing in
`ComputeTargets/GkSourcePolicyData.py` imports `LiouvilleGreen.phase_spline`. The remaining
production constructor of a `phase_spline` is `TkSourceFunctions._build_WKB` (`:272-281`, with
`PHASE_SPLINE_CHUNK_LOGSTEP = 125` at `:127`) — prompt 10's. `LiouvilleGreen/bessel_phase.py` no
longer builds one on this branch (`transfer-remedial` prompt 05 removed it, merged at `e01c31d`),
so after prompt 10 the only `chunk_logstep=125` left in the tree will be in test fixtures
(`test_quadsource_integral.py:266`, `test_phase_groups.py:336, :1137`, and
`test_primitive_phase.py:182`, which passes it deliberately to build the object it is beating) and
in `docs/` reproduction scripts. The frozen signature (D4) is what keeps those working.
