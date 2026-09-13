# Prompt 03 — The conformal-time primitive in `BackgroundModel`

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §7 (the primitive, measured), §13.2 (`model.functions.tau` is the rejected
representation; the primitive belongs in `BackgroundModel`; the persisted schema), §13.3 (what an
interval accessor entails; the double-double constraint; the interface)
**Design facts:** README §2 (b), (c); decision §7 D1.
**Depends on:** 01 (references, error definitions), 02 ($N_\tau$).
**Recommended model:** **Fable** (Opus if unavailable; review most closely)
**Files you may touch:** new `ComputeTargets/cumulative_table.py`, `ComputeTargets/BackgroundModel.py`,
`Datastore/SQL/ObjectFactories/BackgroundModel.py`, `main.py` **solver registration hunk only**
(`:2805-2822` on `9ff59d5`), new `ComputeTargets/tests/test_cumulative_table.py`, new
`ComputeTargets/tests/test_background_tau.py`, `ComputeTargets/tests/test_background_derivatives.py`
(only if it reads payload keys this prompt renames), plus the log and the status board.
**Do not touch:** `WKB_phase_function.py`, the integration objects, `phase_spline.py`, anything in
README §0.2's `transfer-remedial` list.

Read first: README §2 (b), (c), §5 rule 7 (stand-ins must keep constructing), §5 rule 9, §7 D1;
`RECONCILIATION.md` §1 items 10–13, §2 items 2, 5, 11, 12; review §7, §13.2, §13.3 in full;
`logs/01-…` and `logs/02-…` "State handed to the next prompt".

**Precondition (orchestrator):** the user has confirmed decision D1 (README §7).

---

## 1. Character of this commit

The foundation. After it, `functions.tau(z)` is at the double-precision floor instead of
$1.4\times10^{-9}$ relative, `functions.tau.delta(z_a, z_b)` exists and carries short-baseline
differences without the $\varepsilon\tau$ floor, and every downstream consumer of `tau` —
`compute_analytic_G/T`, `QuadSourceIntegral`'s η-limits, `main.py`'s Bessel $x_{\max}$ — is
improved without knowing it. Prompts 04–10 are built on the two accessors defined here.

It is also the commit that **changes stored data**: `tau_Mpc` values change (RK45 → Gauss–Legendre)
and the schema gains a column. A datastore created before this commit cannot be read afterwards, and
that is intended (D1).

## 2. `ComputeTargets/cumulative_table.py`

A generic, model-agnostic kernel. No import of Ray, the datastore, or any `ComputeTargets` object.

```python
class CumulativeTable:
    """
    T(z) = int_z^{z_top} f(z') dz', accumulated on a fixed grid of nodes z_0 > z_1 > ... > z_n
    by per-interval Gauss-Legendre in u = log(1+z) (integrand f(z) e^u), with the prefix sums
    held as double-double (hi, lo) pairs. T is non-decreasing as z falls when f >= 0.
    """
    def __init__(self, z_nodes, f, order: int, *, hi=None, lo=None, label: str = ""): ...
    # build from f (quadrature), or reconstruct from persisted (hi, lo) arrays without any f call
    @property
    def z_nodes(self) -> np.ndarray: ...
    @property
    def hi(self) -> np.ndarray: ...
    @property
    def lo(self) -> np.ndarray: ...
    @property
    def order(self) -> int: ...
    @property
    def evaluations(self) -> int: ...      # integrand evaluations spent at build time
    def value(self, z: float) -> float: ...            # pointwise T(z)
    def delta(self, z_a: float, z_b: float) -> float:  # T(z_b) - T(z_a); never value(b) - value(a)
    def node_index(self, z: float) -> Optional[int]:   # exact-node lookup, or None
```

Requirements:

- **Accumulation.** Increments per interval from the Gauss rule; `hi[j] = fsum(inc[:j])`,
  `lo[j] = fsum([*inc[:j], -hi[j]])` (README §2 (c)). $O(n^2)$ `fsum` work at $n\approx1300$ is
  fine (measure it); if you prefer an $O(n)$ compensated running sum, show it agrees with the
  `fsum` form to the last bit on the prompt-01 references and state the choice.
- **Node lookup is exact.** Production samples are nodes (`RECONCILIATION.md` §1 item 11); look
  them up by exact `z` equality (or an index map keyed on the float), **not** by a tolerance in
  $z$ and **never** through a $\log(1+z)\to z$ round trip (README §5 rule 9). An off-grid `z` gets
  a bracketing interval by `searchsorted` on $u=\log(1+z)$.
- **Partials.** For an off-grid endpoint, a local Gauss rule of the same order over
  $[u_{\rm node}, u]$ from the nearest node — at most one interval, ≤ `order` integrand calls.
  An on-grid endpoint costs **zero** integrand calls.
- **`delta` is a sum of small things.** partial(a → node_a) + [(hi_b − hi_a) + (lo_b − lo_a)] +
  partial(node_b → b), with the two table differences formed hi-with-hi and lo-with-lo before
  adding. The docstring must say why (review §13.3's 9e-4 rad).
- **Reconstruction.** `CumulativeTable(z_nodes, f, order, hi=..., lo=...)` rebuilds from persisted
  arrays with no quadrature; `f` is still required for partials.
- **Vectorised evaluation** is welcome but not required; the interface above is scalar.

## 3. `ComputeTargets/BackgroundModel.py`

1. **`compute_background`**: delete the `solve_ivp` (`:129-183`) and `A0_TAU_INDEX`/`EXPECTED_SOL_LENGTH`.
   Build `CumulativeTable(z_nodes=z_sample as floats (descending), f=lambda z: 1.0/cosmology.Hubble(z),
   order=N_tau)`. Absolute $\tau$ at the nodes is `tau_init + T(z)`, with `tau_init` the existing
   radiation-era closed form (`:138-140`, an author convention — keep it, and keep it as the
   **hi** limb's offset added in double-double: `hi' , lo' = two_sum(hi, tau_init)` then fold `lo`).
   Return `"tau_hi_sample"`, `"tau_lo_sample"` (replacing `"a0_tau_sample"`), `"tau_order"`,
   and an `IntegrationData` whose `RHS_evaluations` is the Hubble-evaluation count and
   `compute_steps` the node count. Solver label: `"cumulative-GL-stepping{N}"` — see §5.
2. **`BackgroundModelValue`**: add `tau_lo: float = 0.0` (keyword, default 0 so stand-ins and the
   `build()` path construct).
3. **`_create_functions`**: replace `_build_func("tau")` with a `TauPrimitive` (name it as you
   like; state the name) — an object with `__call__(z) -> float` returning the absolute $\tau$
   pointwise, and `delta(z_a, z_b) -> float` per README §2 (c), wrapping a `CumulativeTable`
   reconstructed from the stored (hi, lo) values with `f = 1/self._cosmology.Hubble`. Assign it to
   `ModelFunctions.tau`. **Drop the `hasattr(cosmology, "tau")` shortcut for `tau`** (no cosmology
   defines it; `RECONCILIATION.md` §2 item 2); keep `_build_func` for the other splined fields.
   `ModelFunctions` gains **no** new field in this prompt.
4. **`store()`**: read the two limbs into the values.
5. Update the class docstring (`:284-290`) which says the model "integrates" $\tau$.

## 4. `Datastore/SQL/ObjectFactories/BackgroundModel.py`

- `BackgroundModelValue` table: add `sqla.Column("tau_lo_Mpc", sqla.Float(64), nullable=False)`
  after `tau_Mpc` (`:605`). `tau_Mpc` is now the high limb.
- Write both limbs (`store()`, `:399`; `build()`, `:680`) as `value.tau / Mpc`, `value.tau_lo / Mpc`;
  read both (`:288`, `:695`) as `row.tau_Mpc * Mpc`, `row.tau_lo_Mpc * Mpc`; select both.
- The `BackgroundModel` `solver_serial` continues to point at the registered solver; the label
  changes (§5).
- **Do not repair** the two latent defects on the `build()` existing-row branch
  (`RECONCILIATION.md` §2 item 11: `"wkb_serial"` at `:674`, `row_data.Hubble` at `:705`). Record
  them in the log and open `[03-backgroundmodelvalue-build-path]` on the board if you confirm them.
- Add a module-level comment: a datastore whose `BackgroundModelValue` table lacks `tau_lo_Mpc`
  predates this change and must be regenerated; there is no migration.

## 5. `main.py` — solver registration only

At `:2809-2821` add one `pool.object_get("IntegrationSolver", label="cumulative-GL", stepping=N_tau)`
and the matching `solvers["cumulative-GL-stepping{N_tau}"]` entry, following the existing pattern
(label + `-stepping{n}`). Verify `BackgroundModel.store()`'s
`self._solver_labels[data["solver_label"]]` resolves. **No other `main.py` hunk.** If the
`IntegrationSolver` factory rejects the label format, adapt and record it as `STRUCTURALLY REQUIRED`.

## 6. Tests

### `test_cumulative_table.py` (no cosmology; closed forms)

1. $f(z)=(1+z)^{-2}$ ($T=1/(1+z)-1/(1+z_{\rm top})$) on a 100/decade grid from $10^{12}$ to $0.1$:
   nodes to $\le2\times10^{-15}$ relative; `delta` between adjacent nodes to $\le10^{-14}$
   relative to $\Delta T$; `delta` over a 37 % fraction of an interval likewise; `value` off-grid
   to $\le10^{-15}$.
2. **The floor is demonstrated, not asserted away**: the same `delta` from a single-double table
   (construct one by zeroing `lo`) has relative error $\gtrsim10^{-6}$ on an adjacent-node pair near
   $z=1$ where $T\approx1$ — i.e. the test shows what `lo` buys.
3. Exact-node lookup: `node_index(z_nodes[k]) == k` for every node; an off-grid `z` returns `None`;
   `delta(z, z) == 0.0` exactly; `delta(a, b) == -delta(b, a)` to the bit; on-grid `delta` makes
   **zero** integrand calls (count them through a wrapped `f`).
4. Reconstruction from `(hi, lo)` reproduces `value`/`delta` bit-for-bit.

### `test_background_tau.py` (stand-ins from prompt 01; undecorated `compute_background`)

1. `LambdaCDMModel` on the production grid: `functions.tau(z_j) - functions.tau(z_top)` at the
   JSON checkpoints to $\le2\times10^{-14}$ relative; `functions.tau.delta` over the three
   short-baseline references to $\le10^{-13}$ relative to $\Delta\tau$; the off-grid fractional
   baseline likewise.
2. `QCDModel`: same, against the adaptive references, to within 3× the reference's recorded floor.
3. `functions.tau(z)` is still callable with a float and returns a float — the pointwise contract
   every existing consumer relies on (`RECONCILIATION.md` §1 item 10).
4. Persisted round trip: `(tau / Mpc) * Mpc == tau` and likewise for `tau_lo`, exactly, in
   `Mpc_units` (`RECONCILIATION.md` §2 item 12).
5. Build cost recorded (Hubble evaluations, seconds) for both models; assert $\le0.5$ s on
   LambdaCDM.

`test_background_derivatives.py` must still pass; edit it only if it names the removed payload key.

## 7. Verification and acceptance

- Both new test modules pass; `discover -s ComputeTargets/tests -t .` passes.
- README §6 rows for $\tau$ (nodes $\le2\times10^{-14}$; adjacent-node `delta` $\le10^{-13}$) met
  on LambdaCDM; QCD figures reported against the reference floor.
- `grep -n "solve_ivp" ComputeTargets/BackgroundModel.py` is empty.
- A note on the **oracle improvement**: evaluate `compute_analytic_G` at one production $(k, z_s, z_r)$
  with the old cubic-spline `tau` (reconstruct it in the test from the stored nodes) and the new
  accessor and report the phase-equivalent difference (review §13.2(b) predicts ~2 rad at
  $k=10^5$). Information for the log, not a threshold.
- `black --check` clean.

## 8. Log and commit

README §5, §5.1. "State handed to the next prompt", verbatim: `CumulativeTable`'s full public
interface; the name and interface of the `tau` accessor object; the `compute_background` payload
keys; the new column name; the solver label and `stepping` value; measured build cost and
accuracy on both models; the throughput of `delta` as shipped (re-run prompt 01's benchmark against
the production object).

Commit subject, or something equally specific: `Build conformal time as a double-double Gauss-Legendre table`.
