# Prompt 05 — Spline the entropy factor, not the temperature: fix the median (T3)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §8 recommendation **2**, the first of its two halves ·
**Measurements:** audit §3 (T3), §4 · **Decisions:** README §7 **D2** (the object's shape) and
**D3** (node count and order)
**Depends on:** 04. **Blocks 06** — segmentation is applied to *this* representation, not to the
shipped one.
**Recommended model:** **Opus** — the numerics are simple and the interface contract is not.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`,
`ComputeTargets/spline_wrappers.py` **only if README §7 D2 forces it and only as §2 item 2
describes**, `ComputeTargets/tests/wkb_reference_data.json` (**via the prompt-02 generator only**),
the test modules prompt 02's map identifies (**tolerances, thresholds and comments only**),
`CosmologyModels/tests/test_T_z_representation.py` and `test_temperature_spline.py`, plus this
campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `integration_break_points`'s *logic* (prompt 07) — but you **must** keep
`_T_z_spline_knots_log1pz` populated with the knots of whatever you build, because that method
still reads it and prompt 07 owns its removal; `QCD_EOS.py`; `phase_residual.py`; any
`ComputeTargets/` compute target; `Quadrature/`; `Datastore/`; `main.py`.

**Read first:** `LambdaCDM_GenericEOS.py:189-220` (`_build_T_z_spline`, the 5 % buffer comment and
the knot-recording comment), `:340-400` (`_rho_fluid`, `rho`, `Hubble`, `wBackground`,
`wPerturbations` — every consumer of `self._T_z_spline`), `T_photon` (`:140`);
`ComputeTargets/spline_wrappers.py` `ZSplineWrapper` in full, including `_outward` and the negative
`min_z` comment; audit §3's T3 paragraph and §4's table;
`docs/qcd-background-audit/measure_T_z_representation.py` `build_entropy`;
`CosmologyModels/tests/test_temperature_spline.py`.

---

## 1. What is wrong

`_build_T_z_spline` interpolates **$T$ against $u=\log(1+z)$**. Over twenty decades $T$ is
dominated by the $(1+z)$ ramp, which is known in closed form, so the spline spends its degrees of
freedom re-deriving something exact. Write

$$T(z) = T_{\rm CMB}\,(1+z)\,\exp F(u), \qquad
  F(u) = \log\!\frac{T}{T_{\rm CMB}(1+z)} = -\tfrac13\log\!\frac{g_s(T)}{g_s^{\rm CMB}}$$

and the ramp is exact by construction while $F$ — the only part that needs approximating — is
bounded, $O(1)$, and **exactly constant wherever $g_s$ is**, which is most of the range.

Measured at 500 nodes with accurate node values (audit §4): splining $F$ instead of $T$ improves
the **median** from 1.071e-07 to **2.599e-10**, a factor of 400, and the p90 from 1.936e-07 to
8.912e-08, at no runtime cost — 2.07–2.19 µs per call against the shipped 2.26–2.44 µs.

The **max** does not move (7.236e-04 against 7.261e-04). That is correct and expected: the max is
pinned at the jump height and only segmentation touches it. **Prompt 06.** Do not chase it here,
and do not report its not-moving as a failure.

## 2. The change

1. **Build $F$, evaluate $T$.** Replace the spline's ordinates with
   `entropy_factor(u)` — computed from the **tightened** `_solve_T_z` of prompt 04, in the same
   loop, so no second solve is introduced — and make the evaluation return
   `T_CMB * (1+z) * exp(F(u))`. Keep the 5 % buffer in $(1+z)$ and its comment exactly as they are
   (`test_temperature_spline.py` asserts that behaviour, and the sign bug it documents was a real
   defect).

2. **The object contract — README §7 D2.** `T_photon` and `_rho_fluid` both call
   `self._T_z_spline(z)`, and `ZSplineWrapper` supplies the range check, the error messages naming
   the label, and the soft clamp at each end. Two shapes:

   - **keep `ZSplineWrapper`** and hand it a composed callable in $u$ that already contains the
     $\exp$ and the ramp — minimal diff, but the wrapper's `log_z=True` / `deriv` branches then
     mean something slightly different from what its docstring says;
   - **a small `TemperatureRepresentation` class** in `LambdaCDM_GenericEOS.py` honouring the same
     `__call__(z, z_is_log=False)` contract, including the bounds behaviour and the error text —
     more code, but the range logic and the representation live in one readable place, which is
     what prompt 06 then segments.

   **Pick, implement, and argue it in the log as an `IMPLEMENTATION CHOICE`.** Whichever you take,
   the out-of-range `RuntimeError`s must still fire at the same bounds with a message naming the
   quantity, because `test_temperature_spline.test_genuinely_out_of_range_still_raises` asserts it,
   and `docs/qcd-background-audit/measure_T_z_representation.py` monkey-patches
   `cosmology._T_z_spline` with a bare callable, so **whatever you build must still be replaceable
   by a plain `f(z)`** — that script is the campaign's reproduction and must keep running unedited.
   Verify that it does.

3. **Node count and order — README §7 D3.** Measure at least **two** of the audit's candidates and
   report both: 500 / $k=3$ (median 2.599e-10), 2,000 / $k=3$ (1.476e-13), 2,000 / $k=5$
   (2.320e-16). The recommended end state is 3,000 / $k=5$, but that is prompt 06's segmented
   figure; here choose what serves the unsegmented representation and **say what the extra nodes
   cost at build time** (8.3 µs per node after prompt 04: 2,000 nodes is ~17 ms).

4. **Keep `_T_z_spline_knots_log1pz` populated** with the knots of whatever spline you build, so
   `integration_break_points` keeps returning what it returns. The **number** of declared break
   points will change with the node count, which changes how `BackgroundModel` splits its panels —
   **quantify that** (the count before and after, and the effect on the QCD background build time)
   and hand it to prompt 07, which owns the removal.

5. **Bump `T_Z_REPRESENTATION_VERSION` to 3** and add this prompt's row to the constant's table.

6. **Regenerate the QCD reference fixture** with prompt 02's generator, in this commit; quote the
   largest relative move per key; re-score the tests prompt 02's map lists, under prompt 04 §2
   item 5's rule (a tolerance may tighten or stay; one that must loosen is a finding; more than one
   is a stop).

## 3. Tests

1. **The median improves, measured.** Tighten prompt 01's test 1 to README §6.1's "After 05"
   column and **show it fails on `HEAD~1`** by the procedure in prompt 04 §3.
2. **A constant-$g_s$ equation of state is now exact.** Prompt 01's test 6, tightened:
   `PureRadiationEOS` has constant $g_s$, so $F$ is identically constant, the spline of a constant
   is that constant, and $T = T_{\rm CMB}(1+z)$ should come out to **a few ulp**. Assert it and say
   what "a few" measured as. This is the cleanest available demonstration that the new shape is
   doing what it claims (README §2 (g)).
3. **The range behaviour is unchanged.** `test_temperature_spline.py` passes untouched except for
   its `INTERPOLATION_FLOOR`, which may be tightened. Its four range tests must not need editing;
   if one does, that is a `STRUCTURALLY REQUIRED` deviation and must be argued.
4. **LambdaCDM and `RadiationModel` are bit-identical.** Dense comparison of `T_photon`, `Hubble`,
   `rho`, `wBackground`, `wPerturbations` against `HEAD~1`. Quote the comparison.
5. **The reproduction script still runs unedited**, and its §3 table now shows the entropy-factor
   row where the shipped row used to be.

## 4. Acceptance

| Quantity | Before (after 04) | Target |
|---|---|---|
| `T(z)` error, **median** | ~1.07e-07 | **≤ 3.0e-10** at 500 nodes, better with more |
| `T(z)` error, p90 | ~1.94e-07 | **≤ 9.0e-08** |
| `T(z)` error, max | ~7.26e-04 | unchanged — **prompt 06's** |
| `T_photon` cost per call | 2.26–2.44 µs | **≤ 2.5 µs**; a regression is a stop (README §2 (c)) |
| `_build_T_z_spline` wall time | ~4 ms | **≤ 100 ms**, quoted, with the node count |
| `BREAK_POINT_ALL` count on the production grid | 407 | **quoted**; hand it to prompt 07 |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/05-entropy-factor-representation.md` per README §5.1. It must state: **which object
shape was taken and why** (README §7 D2); **which node count and order, with both candidates'
numbers** (D3); the measured cost per call and per build; and the new `BREAK_POINT_ALL` count.
Narrow `[01-genericeos-tz-spline-floor]` on the `source-remediation` board §3 with your measured
figures — it asks "whether the `T(z)` spline grid is adequately defined", and you now have the
answer for two of its three defects — and update `docs/OPEN_ISSUES.md` in the same commit. Update
this campaign's board §1 version table and §2 row T3.

Commit subject, or something equally specific:
`Spline the entropy factor rather than the temperature itself`
