# Prompt 07 — Re-derive `integration_break_points`: the knot lattice is not cosmology (G1)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §8 recommendation **3** · **Closes:**
`[19-cosmologymodels-docstrings-predate-per-sector-policy]` on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3 · **Unblocks:** prompt 10 and
`[13-consumer-spline-crosses-eos-break-points]`
**Measurements:** audit §7 · **Design facts:** README §2 (f), (g), (i)
**Depends on:** 06. **Blocks 08, 10.**
**Recommended model:** **Opus** — the widest blast radius in the campaign. Every quadrature and
every ODE in the tree reads this set.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`,
`CosmologyModels/GenericEOS/GenericEOS.py` (**the two docstrings named in §2 item 4 only**),
`ComputeTargets/BackgroundModel.py` **only** if `_cosmology_break_points`'s docstring must change to
stay true (it names the knots explicitly at `:211`), `Quadrature/integrators/numeric_with_phase_cut.py`
**docstrings and comments only**, `ComputeTargets/tests/wkb_reference_data.json` (**via the prompt-02
generator only**), `ComputeTargets/tests/test_numeric_break_points.py`,
`ComputeTargets/tests/test_background_tau.py`, `test_background_cs_tau_friction.py`, plus whatever
else prompt 02's map identifies (**tolerances, thresholds and comments only**), and this campaign's
log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `TkNumericIntegration.BREAK_POINT_KIND` or `GkNumericIntegration.BREAK_POINT_KIND`
— **prompt 08 owns them, and changing one here without prompt 08's measurement is a stop condition**
(README §2 (f)); `_segmented_solve`, `_separated_boundaries`, `BREAK_POINT_STANDOFF` or any other
*logic* in `numeric_with_phase_cut.py`; `QCD_EOS.py`; `phase_residual.py`'s logic; `main.py`;
`Datastore/` beyond what prompt 03 left.

**Read first:** audit §7; `LambdaCDM_GenericEOS.integration_break_points` (`:256-330`) and
`_temperature_crossing_log1pz` (`:222`); `GenericEOS.py:20-40` (the `BREAK_POINT_*` comment block)
and `:78-108` (the two property docstrings); `ComputeTargets/BackgroundModel.py:203-226`;
`Quadrature/integrators/numeric_with_phase_cut.py`'s module docstring and `_separated_boundaries`'s
(`:166-192`) — its measurement that "the closest a declared temperature crossing comes to a knot
anywhere in $(z=0.1, 10^{14})$ is 3.4e-03" is about to become vacuous; `prompts/GkTk-remedial/logs/19-*`
and `prompts/phase-representation/logs/02-*`; board entries
`[19-cosmologymodels-docstrings-predate-per-sector-policy]` and
`[13-consumer-spline-crosses-eos-break-points]`.

---

## 1. What is wrong

`integration_break_points(..., kind=BREAK_POINT_ALL)` returns, on the production source grid
(1,732 samples, median spacing 2.3032e-02 in $u$):

| kind | count | median spacing | relative to grid |
|---|---|---|---|
| `BREAK_POINT_ALL` | **407** | 9.2936e-02 | 4.04 × |
| `BREAK_POINT_DISCONTINUITY` | 2 | 9.9194e+00 | 430.68 × |

**404 of the 407 are knots of the `T(z)` interpolant** — `linspace(..., 500)` intersected with the
production range. Not cosmology; an implementation artefact of an auxiliary object. Two costs:

- `BackgroundModel` splits a Gauss–Legendre panel **every 4.04 grid intervals** throughout all
  three cumulative tables, for an artefact;
- `prompts/phase-representation` prompt 02 was **blocked entirely** by it: a multiplicity-3 knot at
  each of 407 points cannot satisfy Schoenberg–Whitney against ~1,400 samples, and none of the 407
  coincides with a sample.

Prompts 05 and 06 have replaced the representation. Whatever knots it now has, the function they
approximate is smooth to 1e-15 across them rather than to 1e-4, and the declaration has to be
re-derived on measurement rather than inherited.

## 2. The change

1. **Measure before you remove.** The claim to establish is that the new representation's knots are
   not worth declaring. Quantify, at a knot of the new $F$ spline, the size of the discontinuity in
   the highest derivative that any consumer actually uses — and, more usefully, the *observable*:
   the relative size of the residual in $H(z)$, $\mathrm{d}\ln H/\mathrm{d}z$ and $c_s^2$ across a
   knot, against the $10^{-4}$-level defect the old lattice carried. **Quote both.** A claim that
   "the knots are now smooth" without a number is not acceptable in this prompt's log.

2. **Stop declaring the knot lattice.** `BREAK_POINT_ALL` becomes the crossings of
   `break_temperatures_GeV`; `BREAK_POINT_DISCONTINUITY` stays the crossings of
   `discontinuity_temperatures_GeV`. On the production range that is **3** and **2**. Remove
   `_T_z_spline_knots_log1pz` and the code that populates it, unless prompt 06's representation
   needs it for something else — in which case say what.

3. **The crossings themselves.** `_temperature_crossing_log1pz` root-finds on
   $\log T_{\rm photon}(z) - \log T_{\rm break}$, which README §2 (b) says is the wrong tool *for a
   segment edge*. Now that the representation is segmented, `T_photon` **is** discontinuous at
   exactly these temperatures, so this function is bracketing a root across a step. Two honest
   resolutions: return the segment edges prompt 06 already bisected — they are the same points, to
   machine precision, and are already known exactly — or keep the solve and prove it lands on the
   edge. **Prefer the first**: prompt 06 located these points correctly and re-deriving them by a
   method §2 (b) warns against is a regression waiting to happen. Argue the pick in the log, and
   **assert agreement with prompt 06's 17-digit edges in a test**.

4. **Correct the docstrings that are now false** — this closes
   `[19-cosmologymodels-docstrings-predate-per-sector-policy]`:
   - `GenericEOS.py:78-90` (`break_temperatures_GeV`) and `:92-108`
     (`discontinuity_temperatures_GeV`) — the latter says an adaptive ODE solver "only has to be
     split at a jump" and that the two sets "differ by three orders of magnitude on the production
     range (404 spline knots against 3 temperature crossings)". The first clause was refuted by
     `GkTk-remedial` prompt 19's measurement; the second is refuted by this prompt.
   - `GenericEOS.py:20-40`, the `BREAK_POINT_*` comment block, same reason.
   - `LambdaCDM_GenericEOS.integration_break_points`'s docstring, which says "the knots are the
     load-bearing half".
   - `ComputeTargets/BackgroundModel.py:211`, which names the knots as part of what the QCD
     cosmology declares.
   - `numeric_with_phase_cut.py`'s module docstring (`:24-45`) and `_separated_boundaries`'s
     (`:166-192`), whose "~125 boundaries inside one production numeric range" and "3.4e-03 to the
     nearest knot" measurements no longer describe anything. **Correct the text; do not touch the
     guard**, which exists for a future equation of state and stays.

   Every rewritten docstring must carry **this prompt's own measurement**, not a promise.

5. **Bump `T_Z_REPRESENTATION_VERSION` to 5.** The break-point set changes what `BackgroundModel`
   computes and is exactly the sort of thing the constant was introduced to cover (prompt 03 §2
   item 1).

6. **Regenerate the QCD reference fixture**, in this commit, and quote the largest relative move per
   key. Re-score under prompt 04 §2 item 5's rule. **Expect the `tau`/`cs_tau`/`friction_F`
   references to move**: the panel structure changes even though the integrand does not.

## 3. Tests

1. **The set is what §2 item 2 says.** On `QCD_Cosmology` over the production range:
   `BREAK_POINT_ALL` returns **3** points and `BREAK_POINT_DISCONTINUITY` **2**, none of them a
   knot of anything, and the three agree with prompt 06's segment edges to machine precision.
   **Show the count fails on `HEAD~1`** (407).
2. **The cumulative tables do not lose accuracy.** `tau`, `cs_tau` and `friction_F` on QCD, scored
   against the regenerated references, are **no worse** than before — this is the test that
   establishes the 404 splits were buying nothing. Quote all three, before and after.
3. **The build gets cheaper.** `BackgroundModel`'s QCD cumulative-table build: wall time and
   integrand evaluations, before and after. The audit expects a speed-up; **quote it**, and if
   there is none, say so plainly rather than omitting the row.
4. **A cosmology declaring nothing is bit-identical**, and the two `kind` values still differ for a
   stand-in that declares both a jump and a kink — `test_numeric_break_points.py`'s
   `_ManyBreakCosmology` and its eight `TestPerSectorPolicy` cases must keep their meaning. Any
   case that now needs editing is a `STRUCTURALLY REQUIRED` deviation with an argument.
5. **`prompts/phase-representation` prompt 02's blocker is gone.** A `make_interp_spline` knot
   vector with multiplicity `spline_order` at each of the 3 break points **constructs** against a
   real production grid, for all six (model, $k$) combinations that prompt 02 measured as singular.
   Assert construction only — **do not build the consumer spline here**, which is prompt 10.
6. **Reproduction.** `docs/qcd-background-audit/measure_T_z_representation.py` runs unedited and its
   §5 table now reads 3 / 2 with 0 knots.

## 4. Acceptance

| Quantity | Before | Target |
|---|---|---|
| `BREAK_POINT_ALL` on the production grid | 407 | **3** |
| Of those, knots of an interpolant | 404 | **0** |
| `BREAK_POINT_DISCONTINUITY` | 2 | **2**, unchanged |
| QCD `tau` / `cs_tau` / `friction_F` against the regenerated reference | — | **no worse**, quoted |
| QCD `BackgroundModel` build, wall time and evaluations | — | **quoted**, speed-up expected |
| Knot-vector construction on all six production grids | singular | **constructs** |
| LambdaCDM, `RadiationModel`, stand-ins | — | **bit-identical** |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/07-rederive-break-points.md` per README §5.1, carrying §2 item 1's two measurements,
the three break points to 17 digits, the build-cost comparison, and the list of docstrings
rewritten. Move `[19-cosmologymodels-docstrings-predate-per-sector-policy]` to the `GkTk-remedial`
board's §4; **narrow, do not close, `[13-consumer-spline-crosses-eos-break-points]`** — its blocker
is gone but the defect is prompt 10's — and update `docs/OPEN_ISSUES.md` in the same commit.

Commit subject, or something equally specific:
`Declare only the cosmology's own break points, not the spline's knots`
