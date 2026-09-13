# Prompt 02 — Stop `PrimitivePhase` splining across the cosmology's break points

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[13-consumer-spline-crosses-eos-break-points]` on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3
**Measurements:** [`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md)
§3.5 (the consumer table) and §3.6 (`theta_deriv` against $\omega$)
**Design facts:** README §2 (d) — **read it before you design anything** — plus (e), (f), (g)
**Depends on:** 01 (not for an interface; for a quiet tree).
**Recommended model:** Opus — the implementation choice is the substance, and one of the two
candidates may not construct at all.

**Files you may touch:** `ComputeTargets/primitive_phase.py`,
`ComputeTargets/GkSourcePolicyData.py` and `ComputeTargets/TkSourceFunctions.py` **only** at their
`PrimitivePhase(...)` call sites, `ComputeTargets/BackgroundModel.py` **only** if §2 item 1 forces
it and only as §2 item 1 describes, `ComputeTargets/tests/test_primitive_phase.py`,
`test_gk_source_primitive_phase.py`, `test_tk_source_functions.py`, plus this campaign's log and
board, the `GkTk-remedial` board entry, and `docs/OPEN_ISSUES.md`.

**Do not touch:** `CosmologyModels/` (the break points are already declared and already measured —
prompts 02, 18 and 19 of `GkTk-remedial` did that work), `Quadrature/`, `LiouvilleGreen/`,
`WKB_phase_function.py`, either producer, any `Datastore/` factory, `main.py`, and everything
README §4 lists.

Read first: `ComputeTargets/primitive_phase.py` — the **module docstring**, `__init__`
(`:132-220`), `raw_theta` / `theta_mod_2pi` / `theta_deriv` (`:273-330`), `build_phi_samples`
(`:359-`); `ComputeTargets/BackgroundModel.py:196-226` (`_cosmology_break_points` and its
duck-typing docstring) and `:125-149` (`ModelFunctions`);
`CosmologyModels/GenericEOS/GenericEOS.py:25-38` (`BREAK_POINT_ALL` /
`BREAK_POINT_DISCONTINUITY`, and the comments distinguishing them);
`docs/gktk-remedial-verification.md` §3.5, §3.6; the `GkTk-remedial` board entry
`[13-consumer-spline-crosses-eos-break-points]` and `[02-qcd-T-z-spline-node-tolerance]`.

---

## 1. What is wrong

`PrimitivePhase.__init__` (`primitive_phase.py:212`) builds

```python
self._spline = make_interp_spline(self._u_points, self._phi_points, k=spline_order)
```

with `make_interp_spline`'s **default knots** — a $C^2$ cubic, knots at the data sites, smooth
everywhere by construction. On `QCD_Cosmology` the residual $\varphi$ is **not** smooth: `QCD_EOS`
declares branch boundaries, the effective degrees of freedom step across them, $H(z)$ jumps by
$4.4\times10^{-4}$ at $z=4.24\times10^7$ (the `T_LO` boundary, measured in `GkTk-remedial` log 02),
the Liouville–Green frequency $\omega$ depends on $H$, and so $\varphi$ **kinks** there. A $C^2$
cubic cannot turn a corner. It interpolates straight across.

Measured at production geometry (§3.5), ten points per grid interval:

| model | $k$ | sector | max error | at $z$ | in ulp of span |
|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 2.384e-7 rad | 1.15e6 | **1.00** |
| LambdaCDM | 1e5 | $T_k$ | 7.451e-9 rad | 147.3 | **1.00** |
| **QCD** | **1e5** | **$G_k$** | **1.907e-6 rad** | **4.24e7** | **8.00** |
| **QCD** | **1e5** | **$T_k$** | **3.186e-6 rad** | **4.24e7** | **427.6** |

Ten of the twelve (model, $k$, sector) cases sit at 1.00 ulp — the representation floor, where no
algorithm can do better. The two QCD $k=10^5$ rows are 8× and 428× that, **both with their maximum
at the same $z$**, which is what identifies the cause beyond argument. $k=10^5$ is the wavenumber
that matters because there the floor $\varepsilon k\tau = 3.05\times10^{-7}$ rad is below the
$10^{-6}$ rad target; at $10^7$ and $3\times10^8$ the floor is already above it.

The same non-smoothness makes `theta_deriv` miss $\omega$ by **2.3e-7 to 3.3e-4 relative, uniformly
across the QCD interior** (§3.6) — uniformly, not at the ends, which is why
`[10-residual-spline-end-condition]` was closed at the cubic rather than escalated to
`spline_order=5`: a quintic would not have touched this.

## 2. The change

1. **Get the break points to `PrimitivePhase`.** `_cosmology_break_points(cosmology, z_lo, z_hi,
   kind)` (`BackgroundModel.py:202`) already returns exactly what is needed: an ascending array **in
   $u=\log(1+z)$**, the variable `PrimitivePhase` splines in, empty for any cosmology that does not
   implement `integration_break_points`.

   Prefer an **explicit keyword-only parameter** on `PrimitivePhase` — `break_points=None`,
   appended last so no positional index moves — over a sixteenth `ModelFunctions` field. That is
   README §2 (f) and the prompt-15 precedent: an explicit parameter rather than a quantity
   smuggled through `model_functions`. The two call sites (`GkSourcePolicyData.py`,
   `TkSourceFunctions.py`) then pass it. If you find a reason the other shape is better, take it
   and argue it in the log as an `IMPLEMENTATION CHOICE`; README §7 D1 records that this is open.

   Touch `BackgroundModel.py` **only** if `_cosmology_break_points` must be exported or its
   signature widened to serve a consumer, and say so as a deviation.

   `BREAK_POINT_ALL` is the kind to ask for: $\varphi$ loses smoothness at knots as well as jumps,
   which is precisely the $T_k$/quadrature lesson of `GkTk-remedial` prompts 02 and 19. Measure it;
   do not assume it.

2. **Use them, and read README §2 (d) first.** Two candidates, and the campaign does **not** dictate
   which:

   - **A repeated-knot vector** passed as `make_interp_spline(..., t=...)`: one spline object, one
     `derivative()`, continuity dropped only at the break points. The knot vector must satisfy the
     **Schoenberg–Whitney conditions** against the data sites actually present on a production grid,
     and this is **not guaranteed** — a break point with too few samples between it and its
     neighbour will not admit a repeated knot.
   - **Per-segment splines**, one per interval between break points, with a dispatch on evaluation.
     Always constructs; needs the dispatch; and is **textually close to the chunking prompt 08
     deleted**, which is a stop condition.

   If you take the segments, your log must say explicitly why this is not what §2 (d) forbids —
   physically declared boundaries rather than an arbitrary `logstep`, a bounded residual rather
   than the growing phase, no rebasing of ordinates, and no switch discontinuity — and your tests
   must **demonstrate** the last of those, not assert it.

   **If the knot vector cannot be made to construct on a real production grid, stop and report.**
   Do not fall back silently; README §7 D2 reserves that decision.

3. **A cosmology that declares nothing must be bit-identical.** LambdaCDM, `RadiationModel` and
   every stand-in pass no break points, take the unchanged `make_interp_spline` call, and produce
   **bit-identical** numbers. This is an acceptance test (§3 item 2) and a stop condition.

4. **Do not change what is splined.** $\varphi$ alone, never the leading term, never the full
   phase. `raw_theta`, `theta_mod_2pi`, `theta_deriv`, `num_chunks` and `build_phi_samples` keep
   their signatures and their meanings. **`num_chunks` keeps returning 1** — it reports the
   *phase-spline chunking* that prompt 08 deleted and that `QuadSourceIntegral` persists as
   `WKB_phase_spline_chunks`; segmenting the residual spline is not that, and if you make that
   column read anything but 1 you have changed a stored value and must stop.

5. **`spline_order` stays 3 by default.** The quintic was considered and rejected on the real
   background (`[10-residual-spline-end-condition]`, closed); this prompt is not a second attempt
   at it. Keep the parameter and its 3/5 validation as they are.

## 3. Tests

1. **The kink is resolved.** On a fixture with a *declared* break point and a genuine kink in
   $\varphi$ — a stand-in cosmology exposing `integration_break_points`, as
   `ComputeTargets/tests/test_numeric_break_points.py` builds — the interpolation error at and
   around the break falls from the $O(10)$-ulp regime to $O(1)$ ulp. **Show it fails on the old
   code.**
2. **Smooth cosmologies are bit-identical.** A `PrimitivePhase` built with `break_points=None`, and
   one built on a cosmology declaring none, are bit-identical to `HEAD~1` in `raw_theta`,
   `theta_mod_2pi` and `theta_deriv` over a dense grid, in both sectors (`sign=-1` and `sign=+1`).
3. **No switch discontinuity** (if you took the segments): $\varphi$ and $\varphi'$ across each
   segment boundary agree to the floor. This is the defect that killed chunking; prove it absent.
4. **Degenerate geometry.** A break point outside the sample range, one coinciding with a sample,
   two closer together than a sample spacing, and a segment with fewer samples than the spline
   order — each must either work or raise something that names the problem. None may produce a
   silently wrong spline.

Both suites pass; the `ComputeTargets` count does not fall:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3
```

## 4. Verification and acceptance

Re-run `docs/gktk-remedial/verify_production_path.py` (do not edit it) and quote, before and after:

- **§3.5's twelve-row consumer table.** The two QCD $k=10^5$ rows must meet
  $\le10^{-6}$ rad **and** $\le2$ ulp of the span. The ten rows already at 1.00 ulp must stay at
  1.00 ulp, and the LambdaCDM rows must be **bit-identical**.
- **§3.6's `theta_deriv` against $\omega$.** Target $\le10^{-6}$ relative across the QCD interior.
  **Separate the two contributions**: how much of the 2.3e-7–3.3e-4 was the knots, and how much is
  `[02-qcd-T-z-spline-node-tolerance]`'s scatter of $\omega^2$ itself between neighbouring nodes at
  the top of the grid. Report them as two numbers. If the residue is the latter, that is a
  **narrowing** of this issue and a note on that one, not a miss to hide — say so, and the Result
  is `COMPLETE WITH DEVIATIONS`.
- **Cost.** Per-`PrimitivePhase` build, wall time and integrand evaluations, LambdaCDM and QCD.
  README §6: measured and recorded; **stop if it exceeds 2×**. The $G_k$ figure to beat is
  0.0010 s / 468 evaluations per object.

**Do not loosen a target** (README §6). A miss is an issue and `COMPLETE WITH DEVIATIONS`.

## 5. Log and commit

Log to `logs/02-primitive-phase-break-point-knots.md` per README §5.1. The log must state, in
"Deviations" or "What shipped", **which of the two constructions you took and why**, and — if the
segments — the §2 item 2 argument that this is not chunking. Update this campaign's board, move the
`GkTk-remedial` §3 entry `[13-consumer-spline-crosses-eos-break-points]` to that board's §4, and
update `docs/OPEN_ISSUES.md` — all in this commit.

Commit subject, or something equally specific:
`Break the consumer phase spline at the cosmology's declared knots`.
