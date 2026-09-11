# Prompt 09 — `PrimitivePhase`, and the Green's-function consumer

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §5 (the $h^4x/384$ law), §7 "The consumer need not spline a growing phase",
§8.3 (the rectifier), §8.4 (jitter is a smooth bias), §13.3 "Splining" and "Consumers … anchor",
§13.4 (the pair, the anchoring floor)
**Design facts:** README §2 (c), (e), (g); decisions §7 D5, D6.
**Depends on:** 06 (exactly stored $\theta+\delta$; the $\delta$-wrap case from its log), 03, 05.
Recommended after 08.
**Recommended model:** **Fable** (Opus if unavailable)
**Files you may touch:** new `ComputeTargets/primitive_phase.py`, `ComputeTargets/GkSourcePolicyData.py`,
`ComputeTargets/tests/test_gk_source_policy.py`, new `ComputeTargets/tests/test_primitive_phase.py`,
new `ComputeTargets/tests/test_gk_source_primitive_phase.py`, `ComputeTargets/__init__.py` (export),
plus the log and the status board.
**Do not touch:** `GkSource.py` (D5 — the rectifier's logic is a stop condition),
`QuadSourceIntegral.py`, `phase_groups.py`, `spline_wrappers.py` (duck-typed; verify, do not edit),
`AdaptiveLevin/`.

Read first: README §2 (c), (e), (g), §0.4 (what is *not* done: per-region anchoring), §7 D5, D6;
`RECONCILIATION.md` §1 item 6 (every protocol consumer), §2 items 6, 7, 13; review §5, §7, §8.3,
§13.3, §13.4; `logs/06-…` "State handed to the next prompt" (the $\delta$-wrap case);
`logs/08-…` (the interpolation error to beat).

---

## 1. The problem and the decomposition

`GkSourcePolicyData._create_functions` (`:587-702`) splines $\theta(z_s)$ at fixed $z_r$ through
the stored `(theta_div_2pi, theta_mod_2pi)` of ~1300 independent objects with a cubic spline in
$\log(1+z_s)$. Its error is $h^4x_s/384$: $8.3\times10^{-3}$ rad at $x_s=10^7$ and $O(1)$–$O(10)$
rad at production $x_s\sim10^9$–$10^{12}$ (review §5, §12.6). No density fixes that; the
representation must change.

With the tables (README §2 (g)):
$$\theta(z_s)=-k\,\Delta\tau(z_s\to z_r)+\varphi(z_s),\qquad
\varphi(z_s)\equiv\theta_{\rm stored}(z_s)+k\cdot\texttt{tau.delta}(z_s, z_r),$$
where $\theta_{\rm stored}$ is the **rectified** unwrapped phase `theta_div_2pi*TWO_PI +
theta_mod_2pi` from `GkSourceValue`. $\varphi$ is $-\Delta\rho$ ($\le10^{-3}$ rad) for pure-WKB
objects and a smooth $O(1)$-rad function for numeric-initialised ones (`RECONCILIATION.md` §2
item 7); it carries $\varepsilon k\tau$ rounding noise ($\sim10^{-7}$ rad at $k=10^5$,
$9\times10^{-4}$ at $3\times10^8$ — the D6 floor). $\varphi$ is what gets splined.
$\theta'=d\theta/dz_s=-k/H(z_s)+\varphi'$ in closed form (the leading term; `tau.delta`'s
derivative with respect to its first argument is $+1/H(z_s)$ — check the sign against README §2
(c) and the exact-radiation case).

## 2. `ComputeTargets/primitive_phase.py`

```python
class PrimitivePhase:
    """
    theta(x) = sign * k * leading.delta(x, x_anchor) + phi(x), evaluated from a per-model
    CumulativeTable accessor and a spline of the small residual phi. Implements the phase_spline
    protocol used by AdaptiveLevin (through the phase dict), phase_groups._OscillatoryG,
    QuadSourceIntegral._ClampedPhase and spline_wrappers.GkWKBSplineWrapper.
    """
    def __init__(self, k: float, leading, z_anchor: float, z_samples, phi_samples, *,
                 sign: int, model_functions, label: str = "", spline_order: int = 3): ...
    def raw_theta(self, x: float, x_is_log: bool = False) -> float: ...
    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float: ...
    def theta_deriv(self, x: float, x_is_log: bool = False, log_derivative: bool = False) -> float: ...
    @property
    def num_chunks(self) -> int: return 1
    # plus whatever range attributes RECONCILIATION §1 item 6's consumers read (grep them)
```

- `x_is_log` means $x=\log(1+z)$, as `phase_spline`. Convert with `expm1`, and note README §5 rule 9:
  the recovered $z$ is a *quadrature endpoint* here (harmless), never compared for equality.
- `theta_mod_2pi` reduces `raw_theta` with `WKB_mod_2pi` (negative-remainder convention) — a
  global anchor, D6. Docstring: the floor this implies and why it is accepted; per-region anchoring
  is the recorded follow-up `[00-consumer-anchoring-floor]`.
- `theta_deriv(log_derivative=True)` returns $d\theta/d\log(1+z)$ $=(1+z)\,d\theta/dz$.
- `phi` spline: `make_interp_spline` in $\log(1+z)$, cubic by default; `spline_order=5` allowed.
  Enforce the same range discipline as `phase_spline` (cushion + raise); reuse
  `SPLINE_TOP_BOTTOM_CUSHION` semantics.
- `sign`: $-1$ for the Green's function at fixed $z_r$ (θ decreases with $z_s$); $+1$ will be used
  by prompt 10 for $T_k$ at fixed $z_{\rm init}$ — write the formula so both are one object.

## 3. `ComputeTargets/GkSourcePolicyData.py`

1. `_create_functions`: replace the `phase_spline` (`:671-680`) with a `PrimitivePhase` built from
   `k = source._k_exit.k.k`, `leading = model.functions.tau`, `z_anchor = source.z_response.z`,
   the WKB samples' $z_s$ and $\varphi$ computed as in §1 from the **rectified** `v.WKB.theta_div_2pi`
   and `v.WKB.theta_mod_2pi`. Obtain `model` through the `GkSource`'s model proxy (verify the
   attribute name on `GkSource`; if the source object does not carry one, that is a
   `STRUCTURALLY REQUIRED` note and you thread it from the caller). Delete the false comment
   (`:666-670`, "chunking keeps the rebasing well-conditioned"); `import phase_spline` goes if
   unused. `GkWKBSplineWrapper` receives the `PrimitivePhase` (duck-typed — verify it only calls
   `theta_mod_2pi`).
2. `_classify_Levin` (`~:160-200`): replace its single-chunk `phase_spline` with the same
   `PrimitivePhase`'s `theta_deriv(..., log_derivative=True)`. `Levin_z` is diagnostic only (its
   own comment); the semantics of the threshold test are unchanged.
3. `GkSourceFunctions.phase` is now a `PrimitivePhase`. Nothing in `QuadSourceIntegral`,
   `phase_groups` or `AdaptiveLevin` changes — assert by running their tests.

## 4. Tests

### `test_primitive_phase.py`

Exact radiation, closed-form `tau` accessor (a `CumulativeTable` on the production grid from
$f=1/H$), $\varphi\equiv0$ samples, $k=10^8$, $z_s\in[10,10^4]$ at fixed $z_r=0.1$:

1. `raw_theta` vs $\theta=k(1/s_r-1/s_s)$... (fix the sign against README §2 (c)): max error at
   10 points per interval $\le10^{-6}$ rad — README §6's consumer row — **and $\le6$ ulp of the
   span**, against $8.26\times10^{-3}$ for the `phase_spline` of the same samples (prompt 08's
   test 1 geometry, scaled to $k=10^8$ — build both and assert the ratio $>10^5$).

   > **Threshold corrected 2026-09-11, after the prompt ran.** This item originally asked for
   > $\le10^{-8}$ rad. On this geometry $|\theta|$ reaches $9.09\times10^7$ rad, so one ulp is
   > $1.49\times10^{-8}$ rad and the $\varepsilon k\tau$ floor (README §2 (d)) is
   > $2.02\times10^{-8}$ rad: the stated threshold was **0.67 ulp** of the quantity it measures,
   > and asserting it would be the error `IMPLEMENTATION_STATE.md` §5 note 2 names. The $10^{-8}$
   > came from two places where it *is* reachable — prompt 08's test 1 is the same $z_s$ band at
   > $k=10^6$ (86 ulp), and README §6's prompt-06 radiation control is at a $10^7$ rad span
   > (5.4 ulp) — and the $k$ was scaled to $10^8$ here to reproduce the review's
   > $8.26\times10^{-3}$ rad `phase_spline` figure while the tolerance was not. Prompt 09
   > measured $4.189\times10^{-8}$ rad = 2.81 ulp and asserted the two bounds above instead;
   > the 6-ulp bound is the load-bearing one, since it pins the result to the representation
   > floor where $10^{-6}$ rad would not catch a regression. See
   > `[09-consumer-threshold-below-representation-floor]` and log 09 deviation 1.
2. `theta_deriv` (both `log_derivative` values) vs the closed form to $10^{-12}$ relative.
3. `theta_mod_2pi ∈ (-2π, 0]` and equals `WKB_mod_2pi(raw_theta)`.
4. With a non-zero smooth $\varphi$ (e.g. $\varphi=0.3\sin(u)$ sampled on the grid): recovered to
   the cubic-spline law of $\varphi$ itself ($h^4\max|\varphi^{(4)}|/384\approx10^{-9}$ rad), and
   `theta_deriv` includes $\varphi'$.
5. Protocol: `_ClampedPhase`-style calls (`x_is_log=True`), `num_chunks == 1`, range checks raise
   outside the samples.

### `test_gk_source_primitive_phase.py` (the stand-in `GkSource`)

Using prompt 06's cross-object sweep machinery: fake `GkSourceValue`s for the numeric-initialised
band with exact initial data at extrema, run through the producer algebra (`apply_phase_offset`,
no rebase), then through **a faithful copy of the `GkSource` rectifier** (do not import
`assemble_GkSource_values` — it is a Ray remote over datastore objects; copy the `:166-233` logic
as the review's `t6_sweep.py` did and say so), then build $\varphi$ and the `PrimitivePhase`:

1. $\varphi$ has **no** $2\pi$ jumps after rectification (max $|\Delta\varphi|$ between neighbours
   $<0.1$ rad), and has the expected jumps *before* rectification at exactly the $\delta$-wrap
   points prompt 06's log identifies — this documents D5.
2. `PrimitivePhase.raw_theta` vs the exact $\theta(z_r;z_s)$ at 10 points per interval
   $\le10^{-8}$ rad for $k=10^7$ (radiation, where the exact phase is known).
3. On pure-WKB objects ($z_s$ below $\sqrt{z_{e3}z_{e4}}$) the rectifier applies **zero**
   corrections (assert `rectified_theta_div_2pi == raw` for every sample).

### `test_gk_source_policy.py`

Update the `_classify_Levin` stand-ins (`FakeGkSource` etc.) to carry a `k`, a model with a `tau`
accessor and a `z_response`; the three existing assertions must still hold with their meaning
unchanged.

## 5. Verification and acceptance

- The three test modules pass; `discover -s ComputeTargets/tests -t .` passes — in particular
  `test_quadsource_integral.py` and `test_phase_groups.py` **unchanged and passing** (their
  Green's-function fixtures construct their own phase objects, so this is a protocol check).
- README §6 consumer row met (the $>10^5$ ratio of test 4.1).
- `grep -n "phase_spline" ComputeTargets/GkSourcePolicyData.py` empty.
- `black --check` clean.

## 6. Log and commit

"State handed to the next prompt", verbatim: `PrimitivePhase`'s full interface and the `sign`
convention; how `model` was obtained in `_create_functions`; the measured errors of tests 4.1 and
the ratio; the $\varphi$ statistics (max $|\varphi|$, max $|\Delta\varphi|$) on the stand-in.

Commit subject, or something equally specific: `Evaluate the Green function phase from the conformal-time table`.
