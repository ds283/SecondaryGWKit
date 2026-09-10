# Prompt 02 — Gauss-order convergence across the QCD model's spline knots

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §6 (the residual and its size), §7 last paragraph ("not measured here and
needed before adoption"), §11 "Where its caution still applies" ("the first test of any
implementation"), §12.2 and §12.7 ($\rho_T$, $\tau_s$ through the $c_s$ transition)
**Depends on:** 01 (references, stand-ins, prototype).
**Recommended model:** Opus
**Files you may touch:** new `docs/gktk-remedial/residual_convergence.py`, new
`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md`, `ComputeTargets/tests/wkb_reference_data.json` (add
fields only), `docs/gktk-remedial/generate_references.py` (extend), plus the log and the status
board.
**Do not touch:** any production module.

Read first: README §2 (a), (b), §7 D3; `RECONCILIATION.md` §2 item 3; review §6 (including the
paragraph on $\epsilon$ from a quintic spline at 2000 points per decade), §11's last two
paragraphs, §12.2, §12.7 bullets 1–2; `logs/01-reference-harness-and-prototype.md` §"State handed
to the next prompt".

---

## 1. Character of this commit

One measurement, whose outcome is a set of integers. The review's design assumes a fixed-order
Gauss–Legendre rule per production interval for every table. That is measured at the floor for
$\tau$ on LambdaCDM (review §7) and **unmeasured** for: $\tau$ on `QCD_Cosmology` (whose $H$ is
built on a $T(z)$ spline with knots not aligned to the production grid), $\tau_s$ and $F$ on both
models (the $c_s^2$ transition), and $\rho_G$, $\rho_T$ on both models (whose integrands contain
$\epsilon'$ from a differentiated spline on QCD). If a fixed order does not converge, the review's
fallback is adaptive quadrature **for the residual alone** — never a return to the ODE.

Prompts 03–07 read four integers from this prompt's log: $N_\tau$, $N_{\tau_s}$, $N_F$, $N_\rho$,
and one boolean: whether $\rho$ needs an adaptive fallback on QCD.

## 2. What to measure

Using the stand-ins and references of prompt 01, on the production source grid, for both
`LambdaCDMModel` and `QCDModel`:

1. **Per-interval convergence of the increments.** For each integrand
   $f\in\{1/H,\ c_s/H,\ \tfrac32(1+c_s^2)/(1+z),\ C/(\omega+k/H),\ C_T/(\omega_T+kc_s/H)\}$ (the
   last two for $k\in\{10^5,10^7,3\times10^8\}$) and each production interval $[u_i,u_{i+1}]$,
   compute the Gauss–Legendre increment at orders $N\in\{2,4,6,8,12,16\}$ and the converged
   adaptive reference (prompt 01's method). Tabulate, per integrand and order, the **maximum
   relative increment error over all intervals** and the **interval where it occurs** (its $z$, and
   whether a $T(z)$ spline knot or the $c_s^2$ transition falls inside it).
2. **Cumulative error at the checkpoints**, per order, against the JSON references — the quantity
   the tables actually deliver.
3. **Where the QCD transition sits on the grid**: the intervals containing $z\approx1.2\times10^{12}$
   (the $\epsilon$ departure, review §6) and the $c_s^2$ transition; how many $T(z)$-spline knots
   each production interval contains (2000 per decade vs 100 per decade → ~20; confirm).
4. **Smoothness of what later prompts will spline.** For $\rho_G$, $\rho_T$ (and hence $\varphi$
   of README §2 (g)): the maximum of $|d^2\rho/du^2|$ and $|d^4\rho/du^4|$ estimated by finite
   differences on the grid, and the resulting predicted cubic-spline interpolation error
   $h^4\max|\rho^{(4)}|/384$. This is what bounds prompts 09 and 10.
5. **Cost.** Integrand evaluations and wall time to build each full table at the recommended order,
   for both models.

## 3. Decide, and write the decision down

In `docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` and the log:

- $N_\tau$, $N_{\tau_s}$, $N_F$: the smallest order whose cumulative error at every checkpoint is
  within a factor of 3 of the reference floor on **both** models. Review §7 predicts 4 for $\tau$
  on LambdaCDM; the QCD answer is the point.
- $N_\rho$: the smallest order with cumulative $|\delta\rho|\le10^{-7}$ rad on both models for all
  three $k$ (README §6 target for $\rho$ is $10^{-6}$; take a decade of margin).
- **Fallback flag:** if no order $\le16$ meets the $\rho$ target on QCD, state so, identify the
  offending intervals, and specify the fallback prompt 05 must implement: SciPy `quad` on the
  offending interval class only, with the tolerance and the expected cost. This is a stop condition
  for the orchestrator (README §4.3), so put it in the first line of the log's Result section.
- Whether $\rho_G$ can be **omitted** on LambdaCDM (review §6: $\le2.5\times10^{-7}$ rad, below the
  $\varepsilon k\tau$ floor). Recommendation: carry it regardless — the machinery is required for
  $T_k$ and for QCD and one code path is better than two — but record the number so a later reader
  knows what it buys.

Extend the JSON with the per-order convergence tables (compact: per integrand, per order, max
relative increment error and the cumulative checkpoint errors) so that prompt 05's tests can assert
against them without recomputing.

## 4. Verification and acceptance

- `residual_convergence.py` runs from the repository root and regenerates
  `RESIDUAL-CONVERGENCE.md`'s tables; runtime recorded.
- The `LambdaCDMModel` $\tau$ row reproduces review §7 (order 4: $\le6\times10^{-15}$ relative at
  the nodes; orders 8 and 12 no better).
- The exact-radiation controls hold: $\rho_G\equiv0$ at every order (assert exactly zero from the
  integrand, not to a tolerance), $\rho_T=1/x_i-1/x$ to $10^{-14}$ relative at order $N_\rho$.
- `test_wkb_reference.py` still passes with the extended JSON.
- Every number in the decision section has its (model, integrand, $k$, interval $z$) attached.

## 5. Log and commit

README §5 and §5.1. "State handed to the next prompt" must contain, verbatim: the four orders and
the fallback flag; the per-integrand maximum increment error at those orders on both models; the
predicted $\varphi$ spline error from item 4; the build cost from item 5; and the JSON field names
prompt 05's tests will read.

Commit subject, or something equally specific: `Measure Gauss-order convergence of the WKB primitives on both models`.
