# Log 03 — The conformal-time primitive in `BackgroundModel`

**Prompt:** prompts/GkTk-remedial/03-tau-primitive.md
**Commit:** *(this commit)* — Build conformal time as a double-double Gauss-Legendre table
**Model:** Claude Fable 5.1
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

`BackgroundModel.compute_background` no longer integrates $d\tau/dz=-1/H$ with RK45. It builds a
Gauss–Legendre cumulative table of order 4 on the sample grid, split at the cosmology's break
points, holds every node as a double-double `(hi, lo)` pair, and persists both limbs.
`functions.tau` is a `TablePrimitive` with the pointwise `tau(z)` every existing consumer calls and
the interval accessor `tau.delta(z_a, z_b)` the rest of the campaign is built on. Measured on the
production grid (1,732 nodes, $z=2.06\times10^{16}\to0.1$): **LambdaCDM $\tau$ at the nodes
3.8e-16 relative** (target $\le2\times10^{-14}$), **one-interval $\Delta\tau$ 1.6e-16–2.5e-16**
(target $\le10^{-13}$); **QCD nodes 2.1e-14** against a reference whose own floor is 1.88e-14.
The retired accessor, reconstructed, is **3.08 rad** off at $k=10^5$, $z_s=10^6$, $z_r=0.1$
(review §13.2(b) predicted ~2 rad). Build: 6,924 Hubble evaluations, 0.064 s (LambdaCDM);
8,552, 0.141 s (QCD). Every deviation is `STRUCTURALLY REQUIRED` or `IMPLEMENTATION CHOICE`; none
is `UNINTENDED DRIFT`. Three files outside the prompt's list were touched for the break-point API
the board assigned to this prompt (deviation 1).

## What shipped

### `ComputeTargets/cumulative_table.py` (new, 444 lines)

Model-agnostic kernel. Imports `math`, `typing`, `numpy` only.

```python
def two_sum(a, b) -> (s, e)          # Knuth error-free transformation; elementwise on arrays
def quick_two_sum(a, b) -> (s, e)    # Dekker, |a| >= |b|

class CumulativeTable:
    def __init__(self, z_nodes, f, order: int, *,
                 hi=None, lo=None, break_points=None, label: str = "")
    # T(z) = int_z^{z_top} f dz on nodes z_0 > z_1 > ... (index 0 = top), Gauss-Legendre of
    # `order` per interval in u = log(1+z), each panel split at the `break_points` (ascending u)
    # strictly inside it; hi[j] = fsum(inc[:j]), lo[j] = fsum([*inc[:j], -hi[j]]).
    # hi=..., lo=... reconstructs with no quadrature (f may then be None for on-grid-only use).
    def value(self, z) -> float                  # node hi+lo, plus one local partial off-grid
    def delta(self, z_a, z_b) -> float           # T(z_b) - T(z_a); NEVER value(b) - value(a)
    def node_index(self, z) -> Optional[int]     # exact float lookup, else None
    def shifted(self, offset) -> CumulativeTable # T + offset, added in double-double, renormalised
    z_nodes, u_nodes, hi, lo : np.ndarray        # node order (descending z)
    order : int ; break_points : np.ndarray ; label : str
    evaluations : int        # integrand calls spent at build (0 for a reconstructed table)
    total_evaluations : int  # build plus every off-grid partial since
    __len__
```

`delta` is formed as `table = (hi_b - hi_a) + (lo_b - lo_a)`; `ends = p_b - p_a`;
`return table + ends`, with `p = 0.0` for an on-grid endpoint and a signed local panel from the
nearest node otherwise. That arrangement is what makes `delta(a, b) == -delta(b, a)` and
`delta(z, z) == 0.0` hold to the bit. The docstring carries the review §13.3 argument (9e-4 rad at
$k=3\times10^8$ from a pointwise difference) and the Sterbenz remark on why the hi-with-hi
difference is exact whenever it matters.

**Panel parametrisation** (deviation 3, and the reason the LambdaCDM figures beat prompts 01/02):
a panel is described by its lower redshift and its width
`W = log1p((z_hi - z_lo)/(1 + z_lo))`, not by two rounded `u` values, and its Gauss abscissae are
placed as `1 + z_x = (1 + z_lo) e^t`, `t in [0, W]`. With break points, the sub-panel widths come
from the break positions except the last, which is `W` minus the others, so a break's own rounding
moves only a sub-panel boundary (cost $\delta u\times$ the jump of the integrand there).
Off-grid evaluation beyond one grid interval past either end raises `RuntimeError`.

### `ComputeTargets/BackgroundModel.py`

| | before | after |
|---|---|---|
| module constants | `A0_TAU_INDEX = 0`, `EXPECTED_SOL_LENGTH = 1` | deleted; `TAU_GAUSS_ORDER = 4`, `TAU_SOLVER_LABEL_BASE = "cumulative-GL"`, `TAU_SOLVER_LABEL = "cumulative-GL-stepping4"` |
| imports | `solve_ivp`, `fabs` | dropped; `CumulativeTable` added |
| `_cosmology_break_points(cosmology, z_lo, z_hi)` | — | new module helper; `getattr(cosmology, "integration_break_points", None)`, empty array when absent |
| `compute_background(cosmology, z_sample, atol, rtol)` `:129-183` | `solve_ivp(RK45, t_eval=z_sample)`, sample-point validation loop | `CumulativeTable(z_nodes, 1/H, 4, break_points=…)` inside the `IntegrationSupervisor`, each Hubble call under `RHS_timer`; `tau_init` closed form kept and added by `table.shifted(tau_init)`. `atol`/`rtol` retained in the signature (datastore key; `compute()` passes them) and documented as unused by the table |
| payload keys | `"a0_tau_sample"`, `"solver_label": "solve_ivp+RK45-stepping0"` | `"tau_hi_sample"`, `"tau_lo_sample"`, `"tau_order"`, `"solver_label": TAU_SOLVER_LABEL`; `IntegrationData.compute_steps` = node count, `RHS_evaluations` = Hubble evaluations |
| `class TablePrimitive` | — | new: `__call__(z) -> float`, `delta(z_a, z_b) -> float`, `table`, `label` |
| `BackgroundModel` | — | class attributes `TAU_GAUSS_ORDER`, `TAU_SOLVER_LABEL_BASE`, `TAU_SOLVER_LABEL` (so `main.py` needs one hunk and no new import); docstring no longer says $\tau$ is integrated |
| `_create_functions` `:404` | `tau_func = _build_func("tau")` (cubic spline, `hasattr` shortcut) | `tau_func = self._build_tau_primitive()`: values sorted descending, `CumulativeTable(z_nodes, 1/H, 4, hi=[v.tau], lo=[v.tau_lo], break_points=…)`, wrapped in `TablePrimitive(table, "tau")`. `_build_func` unchanged for the other fields |
| `store()` `:501-533` | inline value construction from `a0_tau_sample` | `self._values = self.values_from_payload(self._z_sample, data)` |
| `values_from_payload(z_sample, data)` | — | new `@staticmethod`, the value-construction loop, reading both limbs; shared with the tests |
| `BackgroundModelValue.__init__` | 12 fields | `tau_lo: float = 0.0` appended as a keyword; property `tau_lo`; `tau` documented as the high limb |

`ModelFunctions` gained no field. `grep -n solve_ivp ComputeTargets/BackgroundModel.py` is empty.

### `Datastore/SQL/ObjectFactories/BackgroundModel.py`

- Module docstring: the schema note the prompt asks for (a datastore without `tau_lo_Mpc` predates
  this change and must be regenerated; no migration).
- `BackgroundModelValue` table: `sqla.Column("tau_lo_Mpc", sqla.Float(64), nullable=False)`
  immediately after `tau_Mpc`.
- `sqla_BackgroundModelFactory.build()`: selects `tau_lo_Mpc`, reads `tau_lo=row.tau_lo_Mpc * Mpc`;
  the sample-row query is wrapped so that an `SQLAlchemyError` mentioning `tau_lo_Mpc` becomes a
  `RuntimeError` naming the regeneration (README §7 D1, "the factory says so").
- `store()`: writes `"tau_lo_Mpc": value.tau_lo / Mpc`.
- `sqla_BackgroundModelValue_factory.build()`: reads `payload["tau_lo"]`, selects and writes the
  column, passes `tau_lo=` to the constructor. **The two latent defects on this path were left as
  they are** (`"wkb_serial"` at the fresh-insert dict, `row_data.Hubble` at the consistency check);
  confirmed and opened as `[03-backgroundmodelvalue-build-path]`.

### `main.py` — one hunk, `:2810-2834` on this tree

One `pool.object_get("IntegrationSolver", label=BackgroundModel.TAU_SOLVER_LABEL_BASE,
stepping=BackgroundModel.TAU_GAUSS_ORDER)` added to the existing `ray.get([...])`, and
`BackgroundModel.TAU_SOLVER_LABEL: cumulative_GL_tau` added to `solvers`. The `IntegrationSolver`
factory does not validate label text (`String(256)`), so no adaptation was needed.

### `CosmologyModels/GenericEOS/` — the break-point API (deviations 1, 2)

- `GenericEOS.py`: `GenericEOSBase.break_temperatures_GeV -> tuple`, a non-abstract property
  returning `()`.
- `QCD_EOS.py`: overrides it with `(T_LO, EOS_T_LO, T_120_MEV, T_HI)`.
- `LambdaCDM_GenericEOS.py`: `_build_T_z_spline` records `self._T_z_spline_knots_log1pz =
  np.unique(spline.t)`; new `_temperature_crossing_log1pz(T, u_lo, u_hi) -> Optional[float]`
  (`root_scalar`, `xtol=rtol=1e-15`, `None` when the crossing is not strictly inside); new public
  `integration_break_points(z_lo, z_hi) -> np.ndarray` returning the ascending $u$ values strictly
  inside the range: the interior knots plus every temperature crossing.

### Tests

- `ComputeTargets/tests/test_cumulative_table.py` (new, 18 tests, 0.13 s): closed form
  $f=(1+z)^{-2}$ on 1,301 nodes; every prompt §6 item plus `shifted`, the error-free
  transformations, range refusal, constructor validation, and a `TestCumulativeTableBreakPoints`
  class with a jump integrand.
- `ComputeTargets/tests/test_background_tau.py` (new, 14 tests, 6.2 s): both production models on
  the JSON grid through `values_from_payload` and `_create_functions`; every prompt §6 item, the
  break-point count, bit-for-bit reconstruction, the throughput benchmark, the oracle note.
- `test_background_derivatives.py` untouched: it reads no renamed key. Passes.

## Deviations from the prompt

### 1. Three `CosmologyModels/GenericEOS/` files touched — STRUCTURALLY REQUIRED

The prompt's file list predates prompt 02's finding that order 4 is at the floor on `QCD_Cosmology`
**only** when every production interval is split at the cosmology's break points, and the board
issue `[02-cosmology-break-point-api]` assigned the API to this prompt ("prompt 03 chooses the API
and implements it"; the orchestrator's dispatch note repeats it). The knots and the branch
temperatures live in the cosmology and the EOS; the only alternative that stays inside the file
list is to read `cosmology._T_z_spline._spline.t` from `BackgroundModel.py`, which the board
explicitly rules out for production code. The three files are not in README §0.2's
`transfer-remedial` list nor in any other forbidden list.

### 2. Shape of the break-point API — IMPLEMENTATION CHOICE

Chosen: `integration_break_points(z_lo, z_hi) -> np.ndarray` of **ascending $u=\log(1+z)$**
values strictly inside the range, on `LambdaCDM_GenericEOS`; the EOS contributes
`break_temperatures_GeV`; `compute_background` and `_create_functions` reach it duck-typed via
`_cosmology_break_points`. Alternatives considered:

- *A default on `BaseCosmology`* returning an empty array. Cleaner contract, one more file; the
  duck-typed form matches how `compute_background` already treats optional analytic derivatives
  (`hasattr(cosmology, attr)`) and lets the test stand-ins (`_HideAnalyticDerivatives`, the
  `wkb_reference` models) work unchanged. Rejected on footprint.
- *Return $z$ instead of $u$.* $u$ is the integration variable and the crossings are solved in it;
  returning $z$ would force a lossy $\log(1+z)\to z$ round trip and back (README §5 rule 9). The
  board's suggestion was $u$.
- *Reach into `ZSplineWrapper._spline.t`* from the cosmology. Avoided by recording the knot vector
  where the spline is built.
- *Put the temperature constants in `LambdaCDM_GenericEOS`.* They are the EOS's, and a future EOS
  with different branches would have to edit the cosmology; the property on the EOS base makes
  the contract explicit and empty by default.

### 3. Panels parametrised by an exact redshift-derived width — IMPLEMENTATION CHOICE

The prototype (log 01) forms each panel from two rounded `u` values. A node's `u = log1p(z)`
carries half an ulp of $u$ — 1.8e-15 at $u\approx27$, 3.5e-15 at $u\approx37$ — and a width formed
as `u_hi - u_lo` inherits it as a *relative* error of $\sim10^{-13}$ on a production interval
($\Delta u\approx0.023$). Interior prefix sums do not see it (a shared edge cancels), but the first
increment below the top node and every off-grid partial do. Measured on the closed-form test
before the change: nodes 1.03e-13, adjacent-node `delta` 1.48e-13 (both at the top of the grid),
which would have failed the prompt's $2\times10^{-15}$ / $10^{-14}$ thresholds. After: 5.1e-16 /
6.2e-16. Cost: one extra `log1p` per panel. This is also why the LambdaCDM checkpoint figure is
3.8e-16 where prompts 01/02 measured 1.81e-15 with the prototype.

### 4. `CumulativeTable` interface additions and one relaxed cost bound — IMPLEMENTATION CHOICE

Beyond the prompt's sketch: the `break_points=` keyword (required by log 02's scheme, which the
prompt's signature predates), `shifted(offset)` (the double-double addition of `tau_init` the
prompt describes, as a method rather than inline in `compute_background`, so prompt 04 can reuse
it), `total_evaluations` alongside the build-time `evaluations`, `u_nodes`, `label`, `__len__`.
Partials use the **build** order, as the prompt says, and are break-aware: an off-grid endpoint
whose partial straddles a break costs `(1 + breaks) * order` integrand calls rather than the
prompt's "≤ `order`". Without that, an anchor $z_{\rm init}$ landing in one of the 404 knot-bearing
QCD intervals would carry the unconverged $N^{-2}$ error the whole scheme exists to remove. On the
grid the bound is unchanged: zero calls.

### 5. The single-double floor is $\sim10^{-14}$, not $\gtrsim10^{-6}$, on the prompt's integrand — STRUCTURALLY REQUIRED

Prompt §6 test 2 expects the single-double `delta` on an adjacent-node pair near $z=1$ to be
$\gtrsim10^{-6}$ relative for $f=(1+z)^{-2}$. There $T\approx0.5$ and one increment is $\approx0.0115$,
so the single-double floor is ${\rm ulp}(T)/\Delta T\approx10^{-14}$; measured **1.507e-15**
single-double against **1.507e-16** double-double (10×). The $10^{-6}$ regime is the review §13.3
one — a primitive that dwarfs the increment — and is demonstrated in the same test by attaching a
$10^8$ offset with `shifted`: double-double unchanged at 1.507e-16, single-double **1.016e-6**.
The test asserts single-double $\ge3\times$ double-double at the natural scale and $>10^{-7}$ with
the offset, and prints all four numbers.

### 6. QCD short-baseline threshold is README §6's $10^{-13}$, not 3× the JSON's recorded agreement — STRUCTURALLY REQUIRED

Prompt §6 test 2 scores QCD "within 3× the reference's recorded floor". For the checkpoints the
floor is `[02-qcd-reference-floor]` (1.879e-14, read from the JSON's `convergence` block) and the
test asserts 3× that. For the short baselines the JSON's recorded self-agreement (`gauss40_bisect`
$\le1.6\times10^{-15}$) does **not** include the error the references actually carry: they were
integrated by `quad` in $u$ between rounded `log1p(z)` endpoints, i.e. up to
${\rm ulp}(u)/W$ relative — 2.1e-13 for the 37 % fraction at $z=10^6$. Verified directly: a `quad`
in the exact-width parametrisation $1+z=(1+z_{\rm lo})e^t$ agrees with the shipped table to
$\le8.8\times10^{-16}$ on all six baselines, while the JSON values differ from it by 4.4e-15 /
3.3e-14 ($z=10^6$, full / fraction), 9.4e-15 / 2.0e-14 ($z=10^2$), 1.9e-15 / 9.6e-15 ($z=1$),
each inside its bound. Asserting 3× 1.6e-15 would be asserting agreement with a reference error.
The test asserts $10^{-13}$ (README §6's row) and the mechanism is
`[03-qcd-short-baseline-reference-endpoint-rounding]`.

### 7. Names, structure and small policies — IMPLEMENTATION CHOICE

- The accessor object is `TablePrimitive` (not `TauPrimitive`): one class serves `tau` now and
  `cs_tau`, `friction_F` in prompt 04.
- `values_from_payload` extracted from `store()` so the offline tests exercise the production
  value construction rather than a copy of it.
- `BackgroundModel.TAU_*` class attributes so that `main.py`'s hunk references the label through a
  name it already imports, rather than a string literal that can drift or a new import line.
- The factory's `RuntimeError` on a missing `tau_lo_Mpc` column, in addition to the module comment
  the prompt asks for: README §7 D1 says the factory "says so", and the raw SQLAlchemy
  "no such column" would not.
- Out-of-range policy: the retired `ZSplineWrapper` clamped softly within 1 % of the range and
  raised beyond. The table evaluates a genuine partial up to one grid interval past either end
  (mathematically correct, unlike a clamped spline) and raises beyond that. No production caller
  leaves the grid.

### 8. Line numbers — STRUCTURALLY REQUIRED (noted, no effect)

The `transfer-remedial` merge moved the `main.py` solver block from `:2805-2822` (on `9ff59d5`) to
`:2810-2834`; `compute_background`'s `solve_ivp` was at `:129-183` as the prompt says. Located by
content.

## Verification performed

**I ran these, and they printed:**

`PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_cumulative_table` —
18 tests, OK, 0.13 s. Closed form $f=(1+z)^{-2}$, 1,301 nodes, $z=10^{12}\to0.1$, order 4:

| quantity | measured | at | threshold |
|---|---|---|---|
| nodes, relative | **5.129e-16** | $z=0.155$ | $\le2\times10^{-15}$ |
| adjacent-node `delta`, relative to $\Delta T$ | **6.154e-16** | $z=3.16$ | $\le10^{-14}$ |
| 37 %-fraction `delta` (one off-grid endpoint) | **5.952e-16** | $z=1000$ | $\le10^{-14}$ |
| off-grid `value` | **5.779e-16** | $z=0.735$ | $\le10^{-15}$ |
| single-double vs double-double `delta`, adjacent pair at $z=1.0$ ($T=0.5$) | **1.507e-15 vs 1.507e-16** | — | demonstration |
| same with a $10^8$ offset | **1.016e-6 vs 1.507e-16** | — | demonstration |
| jump integrand, interval 650, unsplit / split | **7.140e-2 / 1.697e-14** | — | $>10^{-3}$ / $\le10^{-13}$ |
| build integrand calls, split | 5,200 → 5,204 | — | `+ order` |

Also asserted: `node_index(z_k) == k` for all 1,301 nodes and `None` off-grid; `delta(z, z) == 0.0`
at four points; `delta(a, b) == -delta(b, a)` to the bit on five pairs including off-grid ones;
zero integrand calls for on-grid `delta`/`value` and exactly 4 for one off-grid endpoint;
reconstruction from `(hi, lo)` bit-identical on 5×5 point pairs; `shifted` low limb below an ulp
of the high limb everywhere.

`PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_background_tau` —
14 tests, OK, 6.2 s. Production grid (JSON `z_init` = 2.0636395964161516e16, 1,732 nodes; every
checkpoint found by exact node lookup at its JSON index):

| quantity | LambdaCDM | QCD | threshold |
|---|---|---|---|
| `tau.delta(z_top, z_j)` at the 12 non-trivial checkpoints, relative | **3.810e-16** at $z=1002$ | **2.104e-14** at $z=1.005\times10^7$ | $\le2\times10^{-14}$ / $\le3\times1.879\times10^{-14}=5.64\times10^{-14}$ |
| `tau(z_j) - tau(z_top)` (pointwise difference, the consumers' contract) | 3.810e-16 | 2.104e-14 | same |
| one-interval `delta` at $z=10^6$ / $10^2$ / $1$ | **1.616e-16 / 1.990e-16 / 2.494e-16** | **4.200e-15 / 9.354e-15 / 1.496e-15** | $\le10^{-13}$ |
| 37 %-fraction `delta` (off-grid endpoint) at $z=10^6$ / $10^2$ / $1$ | **4.400e-16 / 4.049e-16 / 1.686e-16** | **3.408e-14 / 1.984e-14 / 9.443e-15** | $\le10^{-13}$ (deviation 6) |
| break points in $(0.1, 2.06\times10^{16})$ | 0 (no method) | **407** = 404 knots + 3 crossings, each within $10^{-9}$ of log 02's $u$ | exact count |
| Hubble evaluations at build | **6,924** ($=4\times1731$) | **8,552** | — |
| table build time (`IntegrationData.compute_time`) | **0.064 s** | **0.141 s** | $\le0.5$ s (LambdaCDM) |
| whole `compute_background` (with the derivative splines) | 0.090 s | 0.313 s | — |
| mean Hubble evaluation | 0.28 µs | 7.35 µs | — |
| `tau(z_top) == tau_init` closed form; top `tau_lo == 0.0` | exact | exact | — |
| `(tau/Mpc)*Mpc == tau`, `(tau_lo/Mpc)*Mpc == tau_lo`, all 1,732 nodes; $|{\tt tau\_lo}|\le{\rm ulp}({\tt tau})$ | exact | exact | — |
| reconstruction through `_create_functions` vs a direct build, 5×5 points incl. off-grid | bit-identical | bit-identical | — |

**Throughput of `delta` as shipped** (prompt 01's benchmark re-run on `model.functions.tau`,
20,000 calls, best of 3):

| | LambdaCDM | QCD |
|---|---|---|
| both endpoints on-grid (the production case) | **0.44 µs** | **0.33 µs** |
| one on-grid, one off-grid | 4.56 µs | 26.12 µs |
| both off-grid | 6.63 µs | **51.98 µs** |
| pointwise `tau(z)` on-grid | 0.22 µs | 0.22 µs |

Prompt 01's prototype: 3.0 / 8.2 / 13.1 µs (LambdaCDM) and 4.4 / 47.7 / 102.2 µs (QCD) with
order-8 partials. The both-off-grid QCD figure remains marginally above README §4.3's 50 µs line;
the issue `[01-offgrid-accessor-cost-on-qcd]` is narrowed with these numbers and stays open until
prompt 09 confirms its evaluation pattern.

**Oracle improvement note (prompt §7):** at $k=10^5$/Mpc, $z_s=1.004\times10^6$, $z_r=0.1$ on
LambdaCDM, $k\,\Delta\tau=1.372755\times10^9$ rad. The retired accessor — RK45 at
`atol=1e-10, rtol=1e-8` with `t_eval` on the nodes, cubic-splined in $\log(1+z)$, rebuilt in the
test exactly as `compute_background`/`_create_functions` used to build it — differs by
**−3.080 rad** (2.24e-9 relative), and `compute_analytic_G` evaluated with it differs from the new
one by **0.909 of the local envelope**. A cubic spline of the *new* nodes is exact at nodes and off
by −2.99e-4 rad (7.2e-10 relative) at an off-grid $z=9.92\times10^4$ against the accessor. Review
§13.2(b) predicted ~2 rad; the measured 3.1 rad is the same order.

`PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` —
**143 tests, OK, 122.7 s** (includes `test_background_derivatives`, 5 tests OK, unedited).
`… discover -s CosmologyModels/tests -t .` — 11 tests, OK.

`grep -n "solve_ivp" ComputeTargets/BackgroundModel.py` — empty.

`black --check` on the nine touched files — "9 files would be left unchanged". (Repo-wide
`black --check .` flags 54 pre-existing `docs/…` scripts, none touched here.)

`main.py` registration, checked statically with `ast` (the module cannot be imported): the
`object_get("IntegrationSolver", label=BackgroundModel.TAU_SOLVER_LABEL_BASE,
stepping=BackgroundModel.TAU_GAUSS_ORDER)` call is present; `solvers` has the key
`BackgroundModel.TAU_SOLVER_LABEL`; evaluated, `f"{label}-stepping{stepping}"` ==
`TAU_SOLVER_LABEL` == the `solver_label` `compute_background` returns == `"cumulative-GL-stepping4"`.
`BackgroundModelValue` table columns, from `register()`: `tau_lo_Mpc` directly after `tau_Mpc`.

**Hypothesis check behind deviation 6** (ad-hoc script, not committed): for each of the six QCD
short baselines, a `quad` (`epsrel=1.5e-14`) in $t\in[0,W]$ with $1+z=(1+z_{\rm lo})e^t$ against
(i) the shipped table: 1.62e-16, 8.79e-16, 0.0, 2.70e-16, 3.74e-16, 1.69e-16; (ii) the JSON:
4.36e-15, 3.32e-14, 9.35e-15, 2.01e-14, 1.87e-15, 9.61e-15; bounds
$\tfrac12[{\rm ulp}(u_{\rm hi})+{\rm ulp}(u_{\rm lo})]/W$: 7.71e-14, 2.08e-13, 3.90e-14, 1.05e-13,
9.69e-15, 2.62e-14. The table's disagreement with the JSON is order-independent (3.41e-14 /
3.36e-14 / 3.36e-14 at orders 4 / 8 / 16 for the worst case), as a reference-endpoint effect must be.

**I reasoned that this is correct:** `delta`'s bit-exact antisymmetry (each of the two summands
negates exactly under argument reversal; asserted on five pairs, argued in the docstring for all);
the Sterbenz exactness of `hi_b - hi_a` for nodes within a factor of two; that `values_from_payload`
is the same loop `store()` ran (it is the moved code). `store()` itself was not run — it needs Ray.

**Needs a run the user must do:** a scoped pipeline run on a fresh datastore
(`docs/source-remediation-verification/scoped_pipeline_run.py`), which prompt 13 owns. It will
exercise `store()`, the `IntegrationSolver` registration and the `tau_lo_Mpc` column live. **Every
existing datastore must be regenerated** (README §7 D1): `tau_Mpc` values change and the schema
gains a column; the factory refuses an old one by name.

## Observations not acted on

- **`[03-backgroundmodelvalue-build-path]`** (confirmed, opened): `sqla_BackgroundModelValue_factory.build()`
  inserts with key `"wkb_serial"` where the column is `model_serial`, and compares `row_data.Hubble`
  where the select provides `Hubble_GeV`. Production inserts through `store()`; neither has fired.
  Edited around, not repaired.
- **`[03-qcd-short-baseline-reference-endpoint-rounding]`** (opened, inert): deviation 6.
- **`[03-integrationsolver-stepping-minimum-lookup]`** (opened, inert): the `IntegrationSolver`
  factory matches `stepping >= requested` and returns the first row; harmless while every table is
  order 4, a trap if a second order is registered under `cumulative-GL`.
- **`[01-offgrid-accessor-cost-on-qcd]`** narrowed with the shipped figures (above).
- The QCD off-grid cost is four `QCD_Cosmology.Hubble` evaluations at 7.35 µs each per off-grid
  endpoint (each a `T(z)` spline evaluation plus `QCD_EOS.G`); the table's own overhead is
  $<0.5$ µs. A caching partial, if prompt 09 ever needs one, would target the Hubble call, not the
  table.
- `docs/gktk-remedial/prototype_primitive.py` still uses `u`-edge panel widths; it is the prototype
  and was left as measured. Its `PROTOTYPE-MEASUREMENTS.md` figures (1.81e-15 nodes) are
  superseded by this log's 3.8e-16 for the production object, for the reason in deviation 3.
- `wkb_reference.QCDModel` still calls `compute_background._function(cosmology, z_sample, atol,
  rtol)` and now pays the 0.14 s table build it does not use; unchanged (prompt 01's file).
- README §0.4 / §4.3 stop conditions were checked: nothing here splines $\tau$, keeps any part of
  the phase ODE, or changes an `*_omegaEff_sq` value or a WKB column.

## State handed to the next prompt

**`ComputeTargets/cumulative_table.py` — full public interface:**

```python
from ComputeTargets.cumulative_table import CumulativeTable, two_sum, quick_two_sum

two_sum(a, b) -> (s, e)          # a + b = s + e exactly; elementwise on numpy arrays
quick_two_sum(a, b) -> (s, e)    # requires |a| >= |b|

CumulativeTable(z_nodes, f, order, *, hi=None, lo=None, break_points=None, label="")
    # z_nodes: strictly descending floats (index 0 = top). f: f(z) -> float, the integrand in z
    # (the table integrates f(z)(1+z) in u = log(1+z)). order: Gauss-Legendre order per panel,
    # build and partials. hi/lo: persisted limbs (both or neither); with them no quadrature runs
    # and f may be None if only on-grid evaluation is wanted. break_points: ascending u values;
    # every panel is split at those strictly inside it.
    .value(z) -> float                 # T(z); node hi+lo (0 calls) or nearest node + one partial
    .delta(z_a, z_b) -> float          # T(z_b) - T(z_a) = int_{z_b}^{z_a} f dz; > 0 for z_b < z_a, f >= 0
    .node_index(z) -> Optional[int]    # exact float lookup
    .shifted(offset) -> CumulativeTable  # T + offset in double-double, renormalised
    .z_nodes .u_nodes .hi .lo -> np.ndarray (node order)
    .order -> int ; .break_points -> np.ndarray ; .label -> str
    .evaluations -> int (build-time integrand calls; 0 when reconstructed)
    .total_evaluations -> int (build + partials so far)
    len(table) -> number of nodes
    # raises RuntimeError more than one grid interval beyond either end, or on an off-grid
    # evaluation of a table reconstructed with f=None; ValueError on malformed construction
```

**The accessor object:** `ComputeTargets.BackgroundModel.TablePrimitive(table, label)` —
`__call__(z) -> float` (absolute value pointwise), `delta(z_a, z_b) -> float`, `.table`, `.label`.
`model.functions.tau` is a `TablePrimitive` with `label="tau"`; `tau.table.break_points` is the
cosmology's break set. Convention: `tau.delta(z_a, z_b) = tau(z_b) - tau(z_a)`, positive for
$z_b<z_a$; `tau(z)` is the absolute $a_0\eta$ with the `tau_init` closed form at the top node.

**`compute_background` payload keys** (unchanged keys omitted): `"tau_hi_sample"`,
`"tau_lo_sample"` (lists of floats in `z_sample` order), `"tau_order"` (= 4),
`"solver_label"` (= `"cumulative-GL-stepping4"`); `"a0_tau_sample"` is gone. `"data"` is an
`IntegrationData` with `compute_steps` = node count and `RHS_evaluations` = Hubble evaluations.
The signature `compute_background(cosmology, z_sample, atol=..., rtol=...)` is unchanged.

**Module-level names in `ComputeTargets/BackgroundModel.py`:** `TAU_GAUSS_ORDER = 4`,
`TAU_SOLVER_LABEL_BASE = "cumulative-GL"`, `TAU_SOLVER_LABEL = "cumulative-GL-stepping4"`,
`_cosmology_break_points(cosmology, z_lo, z_hi) -> np.ndarray` (duck-typed; empty when the
cosmology has no `integration_break_points`), `TablePrimitive`,
`BackgroundModel.values_from_payload(z_sample, data) -> List[BackgroundModelValue]`,
`BackgroundModel._build_tau_primitive()`, and the class attributes `BackgroundModel.TAU_GAUSS_ORDER`,
`.TAU_SOLVER_LABEL_BASE`, `.TAU_SOLVER_LABEL`. `BackgroundModelValue(..., tau_lo: float = 0.0)`
keyword-last, property `.tau_lo`; `.tau` is the high limb.

**Column:** `BackgroundModelValue.tau_lo_Mpc`, `Float(64)`, `nullable=False`, after `tau_Mpc`;
written as `value.tau_lo / Mpc`, read as `row.tau_lo_Mpc * Mpc`; exact in `Mpc_units`.
Prompt 04's columns (`cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F`) are not yet there.

**Solver:** label `"cumulative-GL"`, `stepping=4`, registered in `main.py` `:2810-2834` as
`BackgroundModel.TAU_SOLVER_LABEL`. If prompt 04 keeps order 4 for $\tau_s$ and $F$ it can reuse the
same `IntegrationSolver` (the `BackgroundModel` row has one `solver_serial`); a different order
under the same label would hit `[03-integrationsolver-stepping-minimum-lookup]`.

**Break-point API:** `LambdaCDM_GenericEOS.integration_break_points(z_lo, z_hi) -> np.ndarray`
(ascending $u$, strictly inside); `GenericEOSBase.break_temperatures_GeV -> tuple` (`()` default;
`QCD_EOS`: `(1e-5, 2e-3, 0.12, 1e16)`); `LambdaCDM_GenericEOS._T_z_spline_knots_log1pz`. On the
production grid `QCD_Cosmology` gives 407 points; `LambdaCDM` has no method. Pass
`break_points=_cosmology_break_points(cosmology, z_min, z_max)` to every `CumulativeTable` built on
a cosmology — the $\tau_s$ and $F$ tables of prompt 04 need it exactly as $\tau$ does (log 02:
`cs_tau` 4.55e-15, `friction` 7.53e-16 max increment error at order 4 under `branch+knots`).

**Measured on the production grid (1,732 nodes), order 4:**

- Build: LambdaCDM 6,924 Hubble evaluations, 0.064 s table / 0.090 s whole payload; QCD 8,552,
  0.141 s / 0.313 s (plus ~0.3 s to construct `QCD_Cosmology` itself).
- $\tau$ at the nodes vs the JSON: LambdaCDM 3.81e-16 (mpmath reference); QCD 2.10e-14 (reference
  floor 1.88e-14; do not assert below $3\times$ that).
- One-interval $\Delta\tau$: LambdaCDM $\le2.5\times10^{-16}$; QCD $\le9.4\times10^{-15}$. Fraction
  baselines (one off-grid endpoint): LambdaCDM $\le4.4\times10^{-16}$; QCD $\le3.4\times10^{-14}$,
  of which the table's share is $\le8.8\times10^{-16}$ and the rest is the JSON's rounded-$u$
  endpoint (`[03-qcd-short-baseline-reference-endpoint-rounding]`) — **assert $10^{-13}$ for QCD
  short baselines, not the JSON's self-agreement.**
- `delta` throughput on the production object: on-grid 0.44 / 0.33 µs (LambdaCDM / QCD); one
  off-grid endpoint 4.6 / 26.1 µs; both off-grid 6.6 / 52.0 µs; pointwise on-grid 0.22 µs.
- Oracle: `compute_analytic_G` with the retired accessor was 3.08 rad off at $k=10^5$,
  $z_s=10^6$, $z_r=0.1$ (0.91 of the envelope). Anything prompt 05+ scores against
  `compute_analytic_G/T` in-tree is now scored against a reference at the floor.

**Datastore:** every existing datastore predates this commit and must be regenerated; the factory
raises a `RuntimeError` naming the missing `tau_lo_Mpc` column on an old one.
