# Prompt 01 — Reference harness, primitive prototype and throughput benchmark

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §1 (production geometry table), §4 and §12.3 (baseline errors), §6 (the
split), §7 (the primitive, measured), §13.3 (interval accessor, throughput risk), §13.5 items 5–6
**Depends on:** nothing.
**Recommended model:** Opus
**Files you may touch:** new `ComputeTargets/tests/wkb_reference.py`, new
`ComputeTargets/tests/wkb_reference_data.json`, new `ComputeTargets/tests/test_wkb_reference.py`,
new files under `docs/gktk-remedial/` (scripts and one results document), plus the log and the
status board.
**Do not touch:** any production module. This prompt changes no production code.

Read first: README §2 (a)–(d), §5, §5.1, §6 (the error definitions); `RECONCILIATION.md` §1
items 11–12 and §2 items 1, 5, 7; review §7 and §13.3 in full. You may read and reuse the review's
own scripts in `docs/gk-wkb-review-fable-2026-09-09/` (`common.py`, `realbg.py`, `t4_primitive.py`,
`t4b_production_real.py`, `tk2_production_real.py`) — they are the measurements this campaign is
built on, but they are **not importable machinery**: copy what you need into the test package.

---

## 1. Character of this commit

Measurement infrastructure that every later prompt is scored against, plus a prototype of the
design that is measured for the one property the review did not measure — **evaluation
throughput** (review §13.3: "a consumer-side throughput benchmark belongs early in the campaign; it
is the one place in the §7 design that could disappoint"). Nothing here is consumed by production.

A reference built from the object under test conceals common error. `wkb_reference.py` must
import nothing from `Quadrature/integrators/WKB_phase_function.py`, `LiouvilleGreen/phase_spline.py`
or (after they exist) `ComputeTargets/cumulative_table.py`, `phase_residual.py`,
`primitive_phase.py`. It may import the cosmology models, `ModelFunctions`, `redshift`/`redshift_array`,
and the frequency modules `WKB_Gk`/`WKB_Tk` (those are what the references are *for*).

## 2. What to build

### 2.1 Stand-in backgrounds (`wkb_reference.py`)

Three duck-typed models exposing `.functions` as a `ModelFunctions`, in the pattern of
`test_tk_source_functions.FakeModel` and the review's `realbg.py`:

- **`RadiationModel`** — exact radiation, $H=(1+z)^2$, $\epsilon=2$, $w=c_s^2=1/3$, with closed
  forms $\tau(z)=1/(1+z)$ (in the code's $a_0\eta$ sense, up to the additive constant the
  campaign's tables fix at the top of the grid — state which convention you use),
  $\tau_s=\tau/\sqrt3$, $F(z)-F(z_i)=2\ln\frac{1+z}{1+z_i}$, $\theta_G=k(1/s_i-1/s)$,
  $\rho_G\equiv0$, $\rho_T=1/x_i-1/x$ with $x=kc_s\tau$ (review §12.4).
- **`LambdaCDMModel`** — `LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())` with the
  analytic `d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3`, `d_wPerturbations_dz`, `d2_wPerturbations_dz2`
  wired into `epsilon`, `d_epsilon_dz`, `d2_epsilon_dz2` exactly as `realbg.py` does. Its
  `tau` field may be `None` here (nothing in this prompt needs the production accessor).
- **`QCDModel`** — `QCD_Cosmology(store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20)`
  (the production `max_z`, `config/model_list.py`). It supplies no analytic derivatives, so build
  its `ModelFunctions` by calling the **undecorated** `compute_background` on the production grid
  (the pattern of `test_background_derivatives.py`) and splining the returned derivative samples as
  `BackgroundModel._create_functions` does. Record how long that takes.

Grids: a helper producing the production grids — 100 per decade of $1+z$ from the 5-e-fold
super-horizon point of the largest $k$ down to $z=0.1$ for the source grid, and its 12-fold winnow
for the response grid — mirroring `wavenumber_exit_time.populate_z_sample` and
`redshift_array.winnow` (`main.py:411-425`). Use `CosmologyConcepts.redshift`/`redshift_array` so
that later prompts can pass these into production functions.

### 2.2 References (`generate_references.py` → `wkb_reference_data.json`)

A generator script under `docs/gktk-remedial/` (long runtime allowed; the test reads the JSON in
under 10 s) producing, for each of the three models and at ~12 checkpoint nodes spanning the grid
(include the top node, the bottom node, and nodes at $z\approx10^9, 10^6, 10^3, 10, 1$):

- $\tau(z_j)-\tau(z_{\rm top})$, $\tau_s(z_j)-\tau_s(z_{\rm top})$, $F(z_j)-F(z_{\rm top})$;
- for $k\in\{10^5, 10^7, 3\times10^8\}\,{\rm Mpc}^{-1}$: $\rho_G(z_j;z_{\rm top})$ and
  $\rho_T(z_j;z_{\rm top})$ using the rationalised integrands of review §6 and §12.2 with $C$ and
  $C_T$ taken **directly** from the non-leading terms of `Gk_omegaEff_sq` / `Tk_omegaEff_sq`
  (`RECONCILIATION.md` §2 item 3: re-implement the `B + C` terms in the generator; never form
  $\omega^2-\omega_0^2$);
- three **short-baseline** references per model: $\Delta\tau$ over one grid interval near
  $z=10^6$, $z=10^2$ and $z=1$, and over a *fractional* interval (an off-grid endpoint 37 % of the
  way through the interval).

Reference method, stated in the JSON's `"method"` field per model: **mpmath** (`mp.dps = 40`,
`mp.quad` in $u=\log(1+z)$ against the analytic $H$, as `realbg.py`'s `tau_increment_mp`) for
`RadiationModel` and `LambdaCDMModel`; for `QCDModel`, whose $H(z)$ is itself a double-precision
spline evaluation, a **converged adaptive quadrature of the double-precision integrand** — SciPy
`quad` in $u$ per production interval with `epsabs=0, epsrel=1e-14`, summed with `math.fsum`,
cross-checked against the same at `epsrel=1e-12` and against composite Gauss–Legendre order 40 with
bisection until two successive refinements agree to $10^{-14}$; record the agreement achieved as the
reference's own floor. Every reference is evaluated **at the supplied double** (`mpf(float(z))`).

Also record the two **baseline** numbers this prompt re-measures (§3).

### 2.3 Error definitions (`wkb_reference.py`)

Functions, used unchanged by every later prompt:

```python
def phase_error(theta, theta_ref) -> float            # |theta - theta_ref| in rad, unwrapped
def difference_error(delta, delta_ref) -> float       # |delta - delta_ref| / |delta_ref|
def envelope_relative_error(value, value_ref, envelope) -> float
```

with docstrings stating README §6's definitions. Provide `load_references()` returning the JSON as
nested dicts keyed by model → quantity → checkpoint.

### 2.4 The prototype (`docs/gktk-remedial/prototype_primitive.py`)

Implement, **outside the production tree**, the design of review §7 and §13.3:

- nodes $u_i=\log(1+z_i)$ on the production source grid; per-interval Gauss–Legendre of order
  $N\in\{4,8,12\}$ in $u$ of the integrand $f(z)\,e^u$ for $f=1/H$, $c_s/H$, $\tfrac32(1+c_s^2)/(1+z)$;
- cumulative sums as **(hi, lo)** pairs: `hi = fsum(increments[:j])`, `lo = fsum([*increments[:j], -hi])`;
- a pointwise accessor `value(z)` = nearest-node (hi + lo) + local Gauss partial over the fraction
  of one interval, returned as a float;
- an interval accessor `delta(z_a, z_b)` = partial(a→node) + [(hi_b − hi_a) + (lo_b − lo_a)] +
  partial(node→b), returned as a float, with the sign convention of README §2 (c);
- a *control* variant with a single-double table (no `lo`), to demonstrate the floor.

Measure and tabulate, for `LambdaCDMModel` and `QCDModel`, at orders 4, 8, 12:

| measurement | what to record |
|---|---|
| build | Hubble evaluations, wall time |
| node accuracy | max relative error of `value` at the checkpoints vs the JSON reference |
| off-grid accuracy | max relative error of `value` at 25 random off-grid points (LambdaCDM: vs mpmath; QCD: vs the adaptive reference) |
| short baseline | relative error of `delta` over one interval and over the 37 % fraction, **double-double vs single-double** table; convert to phase at $k=3\times10^8$ |
| **throughput** | µs per call of `delta(z_a, z_b)` for $10^5$ random endpoint pairs (both on-grid and off-grid mixes), and of `value(z)`, against a `make_interp_spline` cubic lookup on the same nodes; both models |

The throughput row is the one that can change the design: review §13.3 estimates 8 Hubble
evaluations per off-grid call, "perhaps 50–100× per call" against a spline. **On-grid calls should
cost no Hubble evaluation at all** — confirm the prototype short-circuits them.

### 2.5 The baseline re-measurement (`docs/gktk-remedial/baseline_k1e5.py`)

Reproduce, on this tree and through the production `integrate_phase_function` exactly as the
review's `t4b_production_real.py` and `tk2_production_real.py` do, the two cheap baselines:
$\theta_G$ error at $z=0.1$ for $k=10^5$ (review: 13.9 rad, 1.3 s) and $\theta_T$ for $k=10^5$
(review: 2.01 rad, 1.1 s), against the JSON references. Expect agreement with the review to 5 %. Do
**not** run $k=3\times10^8$ (64 s and 58 s per object; the review's numbers stand as the baseline
for that row of README §6).

## 3. Tests (`test_wkb_reference.py`)

Fast (<10 s), reading the JSON:

1. `RadiationModel` closed forms agree with the JSON to $10^{-15}$ relative — the generator's
   self-test.
2. The JSON carries every checkpoint, every quantity and every $k$ listed in §2.2 for all three
   models, and each model's `"method"` and reference floor are present.
3. `difference_error` and `phase_error` behave as documented on constructed inputs.
4. The `LambdaCDMModel` and `QCDModel` stand-ins construct, and their `epsilon(z)` at $z=10^{10}$
   is within $10^{-6}$ of 2 (radiation era) — a smoke test that the wiring is right.

The prototype and the baseline scripts are **not** tests; they run once and their outputs go into
`docs/gktk-remedial/PROTOTYPE-MEASUREMENTS.md` and the log.

## 4. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_wkb_reference -v` passes in
  under 10 s and does not import `mpmath` at test time.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` still passes
  (no production code changed).
- Prototype order 4 reaches $\le2\times10^{-14}$ relative at the LambdaCDM checkpoints (review §7:
  $5.7\times10^{-15}$). Report the QCD figure without a threshold; if orders 4 and 8 disagree by
  more than $10^{-13}$ on QCD, say so prominently — prompt 02 is about exactly this.
- The double-double `delta` over one interval agrees with the reference to $\le10^{-13}$ relative;
  the single-double control shows the $\sim10^{-12}$ Mpc absolute floor (≈ $9\times10^{-4}$ rad at
  $k=3\times10^8$).
- Throughput measured and tabulated for both models. **If `delta` costs more than 50 µs per off-grid
  call on `QCDModel`, say so in the first line of the log's Result section** — README §4.3 makes
  it a stop condition for the orchestrator.
- Baselines reproduce the review to 5 %.

## 5. Log and commit

Follow README §5 and §5.1. In "State handed to the next prompt", give **verbatim**: the module
path and every public name in `wkb_reference.py` with signatures; the JSON schema (keys, units,
sign conventions, which model uses which reference method and its floor); the production-grid
helper's signature; the prototype's measured table (all rows of §2.4); the two baseline numbers;
the time `compute_background` took on `QCDModel`.

Commit subject, or something equally specific: `Add WKB phase references and a measured primitive prototype`.
