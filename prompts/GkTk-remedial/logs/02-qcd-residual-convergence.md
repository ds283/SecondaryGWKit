# Log 02 — Gauss-order convergence across the QCD model's spline knots

**Prompt:** prompts/GkTk-remedial/02-qcd-residual-convergence.md
**Commit:** *(this commit)* — Measure Gauss-order convergence of the WKB primitives on both models
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

**No adaptive fallback for $\rho$ is required.** A fixed Gauss–Legendre rule of order 4 reaches
$3.75\times10^{-16}$ rad on `QCD_Cosmology` and $4.16\times10^{-17}$ rad on LambdaCDM, against the
$10^{-7}$ rad target — but **only once each production interval is split at the break points of the
cosmology**. Without that subdivision the rule never converges, and what fails is not $\rho$ (which
is small enough to survive an unconverged rule) but $\tau$.

## What shipped

**`docs/gktk-remedial/residual_convergence.py`** (new, 1,760 lines). Runs from the repository root
in 328 s; regenerates `RESIDUAL-CONVERGENCE.md` in full and writes a `"convergence"` block into
`ComputeTargets/tests/wkb_reference_data.json`. Public structure:

```python
ORDERS = (2, 4, 6, 8, 12, 16)
SCHEME_ORDER = ("plain", "branch", "branch+knots")
QUAD_EPSREL = 1.5e-14 ; QUAD_CROSSCHECK_EPSREL = 1e-12
GAUSS_CROSSCHECK_RTOL = 1e-14 ; GAUSS_CROSSCHECK_MAX_LEVEL = 8
RHO_TARGET_RAD = 1e-7 ; RHO_RADIATION_CONTROL_RTOL = 1e-14 ; FLOOR_FACTOR = 3.0
QCD_BREAK_TEMPERATURES_GEV = (("T_LO", 1e-5, ...), ("EOS_T_LO", 2e-3, ...),
                              ("T_120_MEV", 0.12, ...), ("T_HI", 1e16, ...))

gauss_panel(f, a, b, order) ; gauss_interval(f, a, b, order, breaks=())
qcd_branch_boundary_u(cosmology, T_in_GeV, u_lo, u_hi) -> Optional[float]
qcd_break_points(cosmology, u_lo, u_hi) -> (branch_list, knot_array)
assign_breaks(edges, break_u) -> list[tuple[float, ...]]
quad_over_interval / gauss40_over_interval / build_reference(f, edges, break_lists)
measure_orders(f, edges, ref_parts, break_lists, checkpoint_index, json_cumulative)
measure_model(name, model, z_nodes, u_ascending, references, schemes, reference_breaks)
smoothness(functions, u_ascending, u_anchor, maker, k) -> dict
smallest_within_factor(errors_by_order, factor) ; smallest_meeting(errors_by_order, target)
decide(lam_results, qcd_results, controls) -> dict
build_cost(models, u_ascending, references, decision, scheme_breaks) -> dict
radiation_controls(rad, z_nodes_rad, refs) -> dict
```

**`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md`** (new, 446 lines, machine-generated). §1 the
decision, §2 the QCD geometry, §3 per-interval convergence under all three schemes, §4 cumulative
error at the checkpoints (relative and absolute), §5 residual smoothness, §6 build cost, §7 the
exact-radiation controls, §8 what prompts 03–07 must do.

**`ComputeTargets/tests/wkb_reference_data.json`**: one new top-level key `"convergence"`
(37,142 → 247,844 bytes). No existing key altered — `schema_version`, `models`, `baselines` and the rest
are byte-identical, and `wkb_reference.load_references()` reads it unchanged.

Nothing else was touched. No production module was opened.

## Deviations from the prompt

### 1. The reference had to be made break-aware (STRUCTURALLY REQUIRED)

The prompt says to score the fixed-order increment against "the converged adaptive reference
(prompt 01's method)". Prompt 01's method is `quad` per *production interval*. `QCD_Cosmology`'s
`H(z)` is **discontinuous** inside two of those intervals (below), so on exactly the intervals the
measurement is about, prompt 01's per-interval reference is the least trustworthy number available:
its Gauss-40-bisection cross-check does not converge there (it runs to the bisection limit), so a
per-interval error quoted against it would be an error against an unknown.

`build_reference` therefore splits each production interval at the break points *before* the
adaptive rule runs, and forms three independent values per interval (`quad` at 1.5e-14, `quad` at
1e-12, composite Gauss–Legendre 40 with bisection). All three schemes are then scored against that
one reference, so the columns of §3 are comparable. The JSON's own cumulative references are
retained unchanged for §4 — the prompt asks for the cumulative to be scored against them — and the
agreement between the two is recorded per integrand as `json_vs_reference_max_rel`.

### 2. A third QCD break point that prompt 01 did not find (STRUCTURALLY REQUIRED)

Prompt 01 named two break points, both from `QCD_EOS.G`/`Gs`. There is a third inside the
production range: `QCD_EOS.w` clamps its argument to `EOS_T_LO = 2e-3` GeV below `EOS_T_LO`
(`QCD_EOS.py:243-246`), which kinks $c_s^2$ at $z=1.187\times10^{10}$ and therefore reaches
$\tau_s$, $F$ and $\rho_T$. Measured there: the *value* of $c_s^2$ is continuous to
$7.3\times10^{-15}$ but its $u$-derivative jumps by 99.5 %. It is included in the break set.
`T_HI = 1e16` GeV lies far above the production range and is correctly absent.

### 3. $N_\rho$ needs both of the prompt's criteria, not only §3's (STRUCTURALLY REQUIRED)

§3 fixes $N_\rho$ as the smallest order with cumulative $|\delta\rho|\le10^{-7}$ rad. Applied
alone that rule returns **order 2**, on every scheme, because $\rho$ is $10^{-3}$ rad on QCD and
$10^{-10}$ rad on LambdaCDM and an absolute target that loose is met by a rule that has not
converged at all. Order 2 then **fails §4's own acceptance test** — "$\rho_T = 1/x_i-1/x$ to
$10^{-14}$ relative at order $N_\rho$" — by three orders ($7.7\times10^{-11}$).

Rather than report an $N_\rho$ that fails a stated acceptance criterion, `decide()` takes
$N_\rho$ as the smallest order satisfying §3's target *and* §4's exact-radiation control. That is
order 4, which is at the floor on both ($3.75\times10^{-16}$ rad; $2.0\times10^{-16}$ relative) and
which no higher order improves. Both criteria and both per-order tables are in the JSON
(`decision.N_rho_detail`) so a later reader can apply either rule.

### 4. $\rho_T$ is scored against the closed form, not against $1/x_i-1/x$ (STRUCTURALLY REQUIRED)

§4 asks for "$\rho_T = 1/x_i - 1/x$ to $10^{-14}$ relative". That identity is the **large-$x$
asymptote** of the exact radiation primitive, not the primitive, and it carries the opposite sign
in this campaign's convention (prompt 01's log and `wkb_reference.RadiationModel` docstring both
say so). At the production anchor $x_{T,i}=11.6$ the asymptote departs from the exact answer by
$1.24\times10^{-3}$ relative — a hundred billion times the requested tolerance — so scoring against
it could never pass. The control is therefore scored against `RadiationModel.rho_T`, the exact
closed form, and the asymptote's departure is reported beside it (§7 of the document).

### 5. The build scheme is chosen by the script, not asserted (IMPLEMENTATION CHOICE)

The prompt's §3 fallback clause anticipates one binary decision (fixed order, or adaptive for
$\rho$). The measurement makes a third option available and better: deterministic subdivision at
break points the cosmology knows in closed form. Rather than hard-code that conclusion, `decide()`
measures all three schemes and picks the cheapest that (i) admits a fixed order $\le16$ for $\rho$
and (ii) loses nothing on any primitive relative to the best scheme measured. Alternatives
considered: hard-coding `branch+knots` (rejected — it would make the recommendation an assertion
rather than a measurement, and a later re-run on a changed cosmology would not notice); requiring
only the $\rho$ criterion (rejected — that selects `plain`, see the finding below).

### 6. Relative, not absolute, cumulative error decides the three primitive orders (IMPLEMENTATION CHOICE)

§3 says "cumulative error at every checkpoint within a factor of 3 of the reference floor" without
naming the measure. The absolute error at a checkpoint is dominated by the lowest checkpoint, where
$\tau\approx1.4\times10^4$ Mpc; the high-$z$ checkpoints, where the cumulative is small, would then
be invisible. The relative measure scores every checkpoint against its own size, which is also
README §6's definition of a difference error, and is the more demanding of the two. Both are
recorded in the JSON and both are tabulated in §4 of the document; the choice does not change the
answer (order 4 either way).

### 7. "The reference floor" is read as the best any order achieves (IMPLEMENTATION CHOICE)

§3 does not define the floor. `smallest_within_factor` takes it as
$\min_{N\in\{2,\dots,16\}}$ of the error — the level at which raising the order stops buying
anything, whatever sets it. On LambdaCDM that is the `Hubble` rounding floor
(`[01-lambdacdm-hubble-rounding-floor]`); on QCD it is the JSON reference's own accuracy (new issue
`[02-qcd-reference-floor]`). Neither is a property of the Gauss rule, which is the point.

## Verification performed

Everything below was **run**, not reasoned. `PYTHONPATH=. ./venv/bin/python
docs/gktk-remedial/residual_convergence.py`, 328.3 s wall (`QCDModel` construction 0.398 s of it),
Python 3.12.14 / NumPy 2.2.4 / SciPy 1.15.2.

**The decision.** $N_\tau = N_{\tau_s} = N_F = N_\rho = \mathbf{4}$; recommended scheme
`branch+knots`; **adaptive fallback not required**.

**Prompt §4, item 2 — the review §7 reproduction.** LambdaCDM $\tau$, relative cumulative error at
the checkpoints: order 4 **1.81e-15**, order 8 1.96e-15, order 12 1.96e-15, order 16 2.11e-15.
Order 4 is $\le6\times10^{-15}$ as review §7 predicts and orders 8 and 12 are no better — they are
*worse*, by the last bit. Per-interval, order 4 is 7.63e-16 (worst at $z=1.268\times10^{10}$).

**Prompt §4, item 3 — the exact-radiation controls.** `RadiationModel`, all 6 orders, all three $k$,
every one of the 1,000–1,400 Gauss increments of $\rho_G$: **bit-exactly `0.0`** (asserted with
`!= 0.0`, not toleranced; `C = -\epsilon'/2s + (3\epsilon/2-\epsilon^2/4-2)/s^2` evaluates to `0.0`
exactly at $\epsilon=2,\epsilon'=0$). $\rho_T$ against the closed form at $N_\rho=4$:
**2.01e-16** ($k=10^5$), **1.64e-16** ($10^7$), **9.53e-16** ($3\times10^8$) — all $\le10^{-14}$.
At order 2 the same numbers are 7.26e-11, 7.17e-11, 7.68e-11, which is what excludes order 2
(deviation 3). The asymptote $1/x-1/x_i$ departs from the closed form by 1.24e-03 at every $k$
($x_{T,i}=11.6$).

**Prompt §4, item 4 — `test_wkb_reference.py` with the extended JSON.** The whole suite:
`PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` → **Ran 107
tests in 249.5 s, OK.**

**Item 1 — per-interval convergence.** Maximum relative error of one production interval's
increment, order 4, with the interval's $z$:

| integrand | LambdaCDM `plain` | at $z$ | QCD `plain` | QCD `branch` | QCD `branch+knots` | at $z$ |
|---|---|---|---|---|---|---|
| `tau` | 7.63e-16 | 1.268e10 | **6.33e-05** | 7.18e-09 | 4.75e-15 | 8.779e11 |
| `cs_tau` | 7.21e-16 | 2.828e04 | **6.33e-05** | 7.64e-09 | 4.55e-15 | 8.779e11 |
| `friction` | 6.04e-16 | 3.175e05 | 2.76e-05 | 2.20e-10 | 7.53e-16 | 5.039e07 |
| `rho_G@3e8` | 3.95e-06 | 6.363e12 | 1.70e-04 | 8.34e-09 | 1.06e-09 | 1.236e07 |
| `rho_T@3e8` | 1.58e-15 | 1.234e04 | 5.48e-05 | 7.83e-09 | 6.74e-12 | 8.384e11 |

The QCD `plain` worst interval is at $z=4.191\times10^7$ for every integrand — the interval
containing the `T_LO` boundary — and prompt 01's figures reproduce exactly: 6.33e-5 (order 4),
1.69e-5 (8), 8.20e-6 (12), and 301 / 161 / 38 intervals of 1,731 above $10^{-12}$ at orders 4 / 8 /
12. Under `branch+knots` that count is **0 / 1731** for all three primitives at every order $\ge4$.

**Item 2 — cumulative error at the checkpoints, order 4.**

| integrand | LambdaCDM rel | LambdaCDM abs | QCD `branch+knots` rel | QCD abs |
|---|---|---|---|---|
| `tau` | 1.81e-15 | 1.82e-12 Mpc | 1.89e-14 | 3.64e-12 Mpc |
| `cs_tau` | 1.82e-15 | 1.14e-13 Mpc | 1.90e-14 | 7.96e-13 Mpc |
| `friction` | 2.35e-16 | 1.42e-14 rad | 3.01e-16 | 1.42e-14 rad |
| `rho_G@1e5 / 1e7 / 3e8` | — | 1.74 / 2.26 / 5.62 e-18 rad | — | 4.68e-17 / 8.20e-19 / 2.71e-16 rad |
| `rho_T@1e5 / 1e7 / 3e8` | — | 4.16 / 3.47 / 4.16 e-17 rad | — | 1.11e-16 / 1.80e-16 / 3.75e-16 rad |

$\tau$ meets README §6's $\le2\times10^{-14}$ on both models. The QCD 1.89e-14 is **at the JSON
reference's own accuracy** (`json_vs_reference_max_rel` = 1.88e-14), not a quadrature error; see
`[02-qcd-reference-floor]`.

**The finding that changes the recommendation's reasoning.** Review §11 predicted that a fixed rule
"for $\rho$ could converge slowly across [the knots]". It does — but harmlessly. Under `plain` on
QCD the cumulative $\rho$ error is 1.5e-09 rad at order 2 and 4.7e-10 at order 12, i.e. `plain`
**passes** the prompt's $10^{-7}$ rad $\rho$ target at every order. What `plain` cannot do is the
*leading* term: its cumulative $\tau$ error is 1.60e-08 Mpc at order 4 and never better than
2.06e-09 Mpc at any order $\le16$ — **4.79 rad and 0.62 rad of phase at $k=3\times10^8$**, against
1.1e-03 rad under `branch+knots`. The test the review asked for gives the right answer for a
different reason than it expected, and the `plain` figures reproduce prompt 01's 4.78 rad exactly.

**Item 3 — QCD geometry on the production grid** (1,732 nodes, 1,731 intervals,
$\Delta u \in [2.116\times10^{-3},\,2.303\times10^{-2}]$):

| break point | $T$ | $z$ | interval | rel. jump in $H$ | in $c_s^2$ | in $dc_s^2/du$ |
|---|---|---|---|---|---|---|
| `T_LO` | 1e-5 GeV | 4.25344e7 | 862 | 4.435e-04 | 7.08e-08 | 8.93e-04 |
| `EOS_T_LO` | 2e-3 GeV | 1.18721e10 | 1107 | 4.00e-10 | 7.32e-15 | 9.95e-01 |
| `T_120_MEV` | 0.12 GeV | 8.64355e11 | 1293 | 1.038e-04 | 7.46e-04 | 1.99e-01 |

`T(z)` spline: **404 interior knots** in the production range, median $\Delta u$ 0.0929, i.e.
**0.233 per production interval — one knot every 4.3 intervals**, and 404 of 1,731 intervals
contain exactly one. The prompt's estimate of ~20 knots per interval is wrong by two orders of
magnitude in the reassuring direction; it came from review §6's *quintic $\epsilon$ spline at 2000
points per decade*, which is the review's own reference construction, not the model's `T(z)`
spline. $\epsilon$ reaches its minimum 1.850811 at $z=1.299\times10^{12}$ (interval 1310),
reproducing review §6's "departs from 2 by up to 0.150 at $z\approx1.2\times10^{12}$". The steepest
node-to-node $|dw/du|$ is at $z=8.779\times10^{11}$ (interval 1293), i.e. at the QCD transition
beside `T_120_MEV`, **not** at matter–radiation equality.

**Item 4 — smoothness of what prompts 09 and 10 will spline.** Worst case over both models, both
sectors and all three $k$: $h^4\max|\rho''''|/384 = \mathbf{3.12\times10^{-7}}$ rad, on QCD,
`rho_T@3e8`, at $z=8.384\times10^{11}$ (the QCD transition). LambdaCDM is far smaller: 6.60e-11 rad
(`rho_T@1e5`, $z=2.202\times10^9$) and $\le1.22\times10^{-18}$ rad for $\rho_G$. On QCD the finite-
difference stencil straddles the three break points at a handful of nodes, so these are upper
bounds. All are inside README §6's $10^{-6}$ rad consumer target.

**Item 5 — build cost.** Order 4, one full table on the 1,731-interval grid:

| | $\tau$ | $\tau_s$ | $F$ | three together | $\rho$ at $k=3\times10^8$ |
|---|---|---|---|---|---|
| LambdaCDM `plain` | 6,924 ev / 0.007 s | 6,924 / 0.008 s | 6,924 / 0.006 s | **20,772 ev / 0.020 s** | 5,536 / 0.010–0.013 s |
| QCD `branch+knots` | 8,552 / 0.063 s | 8,552 / 0.168 s | 8,552 / 0.102 s | **25,656 ev / 0.305 s** | 6,904 / 0.113–0.281 s |

Subdivision costs **24 % more integrand evaluations** than `plain` and buys seven orders of
magnitude. The LambdaCDM figure reproduces prompt 01's 6,924 evaluations exactly.

## Observations not acted on

1. **The break points are reachable only through private attributes.** The script obtains the
   `T(z)` knots as `cosmology._T_z_spline._spline.t`. Acceptable in a `docs/` measurement script;
   not acceptable in `ComputeTargets/cumulative_table.py`, which prompt 03 writes and which needs
   the same list. Opened as `[02-cosmology-break-point-api]`.
2. **`QCD_Cosmology`'s `T(z)` spline nodes carry ~2e-5 relative error of their own.**
   `LambdaCDM_GenericEOS._solve_T_z` solves $T\,g_S(T)^{1/3} = $ const with
   `root_scalar(..., xtol=1e-6, rtol=1e-4)`. Re-solving eight sampled nodes at `rtol=1e-15` moves
   the answer by up to **2.08e-05** relative (at $z=10^{13}$; 1.17e-05 at $z=4.2\times10^7$, 0.0 at
   $z\le10^5$). This is a property of the cosmology, not of any quadrature — the tables converge to
   the integral of the function the model actually defines — but it bounds what a QCD $\tau$ table
   *means*. Opened as `[02-qcd-T-z-spline-node-tolerance]`. Not touched: it is a production module
   and outside this prompt.
3. **$\rho_G$'s per-interval relative error does not reach $10^{-12}$ on QCD even under
   `branch+knots`** (41–46 intervals of ~1,400 stay above it at order 4). Those increments are
   $10^{-15}$ rad in absolute terms, formed from a $C$ that is a difference of $O(1)$ spline values,
   so the relative measure is asking for accuracy below the integrand's own rounding. The absolute
   cumulative is 2.71e-16 rad. Recorded in §3.5 of the document; no action.
4. **Order 2 is erratic rather than merely inaccurate.** Its cumulative $\rho$ error on QCD is
   1.53e-09 rad under `plain` but 2.03e-08 rad under `branch+knots` — subdividing makes it *worse*,
   because at order 2 the answer is dominated by cancelling truncation errors. Another reason not to
   let §3's absolute target select it alone.

## State handed to the next prompt

**The four orders, and the fallback flag.**

```
N_tau    = 4
N_cs_tau = 4          (i.e. N_{tau_s})
N_F      = 4
N_rho    = 4
rho_adaptive_fallback_required = False
```

**Prompts 03, 04 and 05 must also carry the build scheme, which is not optional.** Order 4 is only
at the floor on `QCD_Cosmology` when each production interval is **split at every break point of the
cosmology that falls strictly inside it**, and each sub-panel integrated with its own order-4 Gauss
rule (`fsum` of the sub-panels). The break-point set, all known in closed form before any
integration:

| source | value | $u=\log(1+z)$ of the crossing | affects |
|---|---|---|---|
| `QCD_EOS.T_LO` | 1e-5 GeV | $z = 4.25344\times10^{7}$ | `Hubble` — jumps 4.4e-4 |
| `QCD_EOS.EOS_T_LO` | 2e-3 GeV | $z = 1.18721\times10^{10}$ | `wPerturbations` — value continuous, slope jumps 99.5 % |
| `QCD_EOS.T_120_MEV` | 0.12 GeV | $z = 8.64355\times10^{11}$ | `Hubble` (1.0e-4) and `wPerturbations` (7.5e-4) |
| `QCD_EOS.T_HI` | 1e16 GeV | outside the production range | — |
| `cosmology._T_z_spline._spline.t` | 404 interior knots | median $\Delta u = 0.0929$ | everything through `T(z)` |

Each temperature crossing is found by `root_scalar` on
$\log T_{\rm photon}(e^u-1) - \log(T\cdot\texttt{units.GeV})$, bracketed by the grid ends,
`xtol=rtol=1e-15` (`residual_convergence.qcd_branch_boundary_u`). `LambdaCDM` has an empty break
set and the scheme reduces to review §7's rule as written. **The knots are the load-bearing half:**
splitting only at the three temperatures leaves $\tau$ at 1.81e-13 relative, 25× worse than the
floor.

**Per-integrand maximum increment error at order 4, both models** (relative, one production
interval, against the break-aware converged reference; the QCD column is `branch+knots`):

| integrand | LambdaCDM | at $z$ | QCD | at $z$ |
|---|---|---|---|---|
| `tau` = $1/H$ | 7.63e-16 | 1.268e10 | 4.75e-15 | 8.779e11 |
| `cs_tau` = $c_s/H$ | 7.21e-16 | 2.828e04 | 4.55e-15 | 8.779e11 |
| `friction` = $\tfrac32(1+c_s^2)$ | 6.04e-16 | 3.175e05 | 7.53e-16 | 5.039e07 |
| `rho_G@1e5` | 1.47e-09 | 1.918e09 | 1.06e-09 | 1.236e07 |
| `rho_G@1e7` | 8.18e-08 | 1.965e11 | 1.06e-09 | 1.236e07 |
| `rho_G@3e8` | 3.95e-06 | 6.363e12 | 1.06e-09 | 1.236e07 |
| `rho_T@1e5` | 1.19e-15 | 1.417e04 | 1.00e-12 | 4.095e07 |
| `rho_T@1e7` | 1.12e-15 | 1.520e06 | 6.42e-13 | 4.095e07 |
| `rho_T@3e8` | 1.58e-15 | 1.234e04 | 6.74e-12 | 8.384e11 |

The large $\rho_G$ *relative* numbers are on increments of $10^{-15}$ rad or less; the cumulative
absolutes are in the table above (worst 2.71e-16 rad).

**Cumulative accuracy prompts 03–05 should expect to reproduce** (order 4, at the JSON checkpoints):
$\tau$ 1.81e-15 relative / 1.82e-12 Mpc on LambdaCDM and 1.89e-14 / 3.64e-12 Mpc on QCD; $\tau_s$
1.82e-15 / 1.90e-14; $F$ 2.35e-16 / 3.01e-16; $\rho$ $\le5.62\times10^{-18}$ rad (LambdaCDM) and
$\le3.75\times10^{-16}$ rad (QCD). **Do not assert below 2e-14 relative for QCD $\tau$** — that is
the JSON reference's own floor (`[02-qcd-reference-floor]`), not the table's.

**Predicted $\varphi$ spline error for prompts 09 and 10** ($h^4\max|\rho''''|/384$ on the
production grid): worst **3.12e-07 rad**, QCD, $\rho_T$ at $k=3\times10^8$, at
$z=8.384\times10^{11}$; 1.73e-07 rad for $\rho_G$ at the same $k$ and $z$; LambdaCDM
$\le6.60\times10^{-11}$ rad. All inside README §6's $10^{-6}$ rad target, with the QCD figure set by
the QCD transition and contaminated upward by the break-point stencils.

**Build cost at order 4**, per model, for the three $k$-independent tables together:
LambdaCDM **20,772 integrand evaluations, 0.020 s** (`plain`); QCD **25,656 evaluations, 0.305 s**
(`branch+knots`), plus 0.398 s to construct `QCDModel` itself. One $\rho$ table per $k$ costs 5,536
evaluations / 0.010–0.013 s (LambdaCDM) and 6,904 / 0.113–0.281 s (QCD) from the 3-e-fold anchor at
$k=3\times10^8$.

**JSON field names prompt 05's tests will read.** Everything is under the new top-level key
`"convergence"` of `ComputeTargets/tests/wkb_reference_data.json`; `load_references()["convergence"]`.

```
convergence.orders                      -> [2, 4, 6, 8, 12, 16]
convergence.schemes                     -> ["plain", "branch", "branch+knots"]
convergence.schema                      -> prose, per block
convergence.decision.N_tau | .N_cs_tau | .N_F | .N_rho          -> 4, 4, 4, 4
convergence.decision.recommended_scheme                          -> "branch+knots"
convergence.decision.rho_adaptive_fallback_required               -> false
convergence.decision.rho_target_rad | .rho_radiation_control_rtol -> 1e-7, 1e-14
convergence.decision.<N_x>_detail.<model>.{scheme,floor_rel,order,errors_rel,errors_abs}
convergence.decision.N_rho_detail.<model>.{scheme,criterion,order,errors_abs|errors_rel}
convergence.decision.scheme_summary.<scheme>.{primitive_floor_rel,primitive_floor_abs,
                                              primitive_choice, <key>_errors_rel,
                                              <key>_errors_abs, rho_errors_abs, rho_min_order}
convergence.decision.lambdacdm_rho_magnitude_rad | .qcd_rho_magnitude_rad
convergence.geometry.QCDModel.{num_nodes,num_intervals,du_min,du_max,branch_boundaries,
                               T_spline_knots_in_range,T_spline_knot_du,
                               knots_per_production_interval,epsilon_departure,cs2_transition}
convergence.models.<model>.<scheme>.<integrand key>.orders.<N>.{
        max_increment_rel_error, max_increment_at_z, max_increment_abs_error,
        max_increment_as_rad_at_3e8, intervals_above_1e-12, offending_interval_z,
        max_cumulative_abs_error, max_cumulative_abs_at_z,
        max_cumulative_rel_error, max_cumulative_rel_at_z }
convergence.models.<model>.<scheme>.<integrand key>.{reference_total, num_intervals,
        crosscheck_quad_1e-12_max_rel, crosscheck_gauss40_max_rel, json_vs_reference_max_rel,
        quad_max_reported_abserr, quad_intervals_with_warnings, k, rho_anchor_z, label, scheme}
convergence.smoothness.<model>.<integrand key>.{max_d2_rho_du2, max_d2_at_z, max_d4_rho_du4,
        max_d4_at_z, max_cubic_spline_error_rad, max_cubic_spline_error_at_z}
convergence.build_cost.<model>.{scheme, <table>.{order,integrand_evaluations,seconds,value},
        three_tables_total_evaluations, three_tables_total_seconds}
convergence.radiation_controls.rho_G_exactly_zero.<k key>          -> true
convergence.radiation_controls.rho_G_max_abs_increment.<k key>     -> 0.0
convergence.radiation_controls.rho_T.<k key>.{orders.<N>.{max_rel_error_vs_closed_form, at_z,
        max_rel_error_vs_json}, at_N_rho, x_init, x_final, closed_form,
        asymptote_1_over_x_minus_1_over_x_i, asymptote_relative_departure}
```

Integrand keys are `"tau"`, `"cs_tau"`, `"friction"`, and `"rho_G@<k>"` / `"rho_T@<k>"` with `<k>`
the existing `f"{k:.6e}"` key (`"1.000000e+05"`, `"1.000000e+07"`, `"3.000000e+08"`).
`LambdaCDMModel` carries only the `"plain"` scheme. **No pre-existing JSON key was modified.**

**Whether $\rho_G$ can be omitted on LambdaCDM.** It can, numerically: $|\rho_G|$ over the whole WKB
range is 2.588e-07 rad ($k=10^5$), 3.434e-09 ($10^7$), 1.353e-10 ($3\times10^8$) — the first is at
the $\varepsilon k\tau$ floor and the last two are far below it. On QCD it is 5.420e-04, 3.003e-05
and 1.180e-03 rad, and $|\rho_T|$ is 8.6e-02 rad on *both* models at every $k$. The recommendation
stands as the prompt anticipated: **carry it regardless** — the machinery is required for $T_k$ and
for QCD, it costs 5.5k–6.9k evaluations and 10–280 ms per $k$, and one code path is better than two.
(These reproduce review §6 and §12.2: −1.26e-3 vs 1.180e-3 for $\rho_G$ on QCD at $3\times10^8$, and
−0.0863 / −0.0933 vs 8.634e-02 / 9.310e-02 for $\rho_T$ on LambdaCDM / QCD.)
