# Prompt 13 — Segment every background spline at the cosmology's break points

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[12-background-derivative-fit-grid-rings-at-a-step]` on this campaign's board §3
**Implements:** the principle the rest of the campaign established piecewise — **a smooth
interpolant may not run across a point at which the cosmology declares it is not smooth**. Prompt 06
applied it to $F(u)$, prompt 07 to the cumulative tables' Gauss panels; this is the last place in
the tree where it is still violated.
**Measurements:** [`docs/qcd-background-verification.md`](../../docs/qcd-background-verification.md)
**§10.4** — read it before anything else; it is the whole prompt
**Depends on:** 12 (which measured it). **Unblocks:** the production half of prompt 11's fix.
**Recommended model:** **Opus** — small diff, wide blast radius, and the acceptance test is a
four-way table that is easy to score in the wrong configuration.

**Files you may touch:** `ComputeTargets/BackgroundModel.py`,
`ComputeTargets/tests/test_background_derivatives.py`,
`ComputeTargets/tests/test_background_tau.py` and `test_background_cs_tau_friction.py`
(**tolerances, thresholds and comments only**), `ComputeTargets/tests/wkb_reference_data.json`
(**via `docs/qcd-background-audit/generate_qcd_references.py` only — never by hand**), a new test
module if you want one, plus this campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `CosmologyModels/` — the break points are correct and prompt 07 owns them, and
`LambdaCDM_GenericEOS`'s own $F$ spline is already segmented (`:227`); `CosmologyConcepts/` and
`main.py` — the source grid is prompt 11's and its density is prompt 12's open recommendation, not
this prompt's; `QCD_EOS.py`; `Quadrature/`; any compute target other than `BackgroundModel`; any
`Datastore/` factory.

**Read first:** `docs/qcd-background-verification.md` §10.4 **in full**, and §9.2 for the
configuration trap it describes; `ComputeTargets/BackgroundModel.py` — `_build_derivative_fit_grid`
(`:66-110`), `_build_derivative` (`:340-372`), `_create_functions._build_func` (`:583-600`), and
`_cosmology_break_points` (`:203-226`), which already returns exactly the ascending array in
$u=\log(1+z)$ this prompt needs and is empty for any cosmology that declares nothing;
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`'s `build_segmented_entropy_spline` and
`SegmentedEntropyFactor`, which are prompt 06's implementation of the same idea and the shape to
follow; `prompts/qcd-background-audit/logs/06-segment-at-the-jumps.md` for the node-allocation and
degenerate-geometry problems already solved there.

---

## 1. What is wrong

`QCD_Cosmology` supplies no `d_lnH_dz`, so `compute_background` builds one: `_build_derivative`
(`:367`) fits `make_interp_spline(fit_x, y_data, k=DERIVATIVE_SPLINE_ORDER)` over
`_build_derivative_fit_grid(z_sample)` — a padded, 3× refined copy of the **source grid** — and
differentiates it. `d2_lnH_dz2`, `d3_lnH_dz3` and `d_wPerturbations_dz` are stacked on top of the
same lattice. **That lattice is not split at `integration_break_points`**, and $H$ genuinely
**steps** at two of the three declared crossings. So a quintic runs straight across a step, and
`epsilon`, `d_epsilon_dz`, `d2_epsilon_dz2` — hence $\omega_{\rm eff}$, hence every stored phase —
ring there.

Measured (§10.4), against a central difference of the cosmology's own pointwise `Hubble`:

| crossing | base-grid background | shipped-grid background |
|---|---|---|
| `T_LO`, $u = 17.565806941870026$ | **3.66e-02** | **2.04e-02** |
| `EOS_T_LO`, $u = 23.197460552819653$ | 1.9e-09 | 1.6e-09 |
| `T_120_MEV`, $u = 27.485391822044257$ | **6.48e-03** | **1.03e-03** |

**The middle row is the control and it is the campaign's own physics**: `EOS_T_LO` is where $g_s$
is continuous to 1.8e-11 and only $w$ kinks (README §7 D4), so $H$ does not step and nothing rings.
Away from a crossing the agreement is **3.9e-09 max / 8.1e-10 median**. That contrast is the
evidence that this is a spline ringing at a step and not an error of the cosmology.

**Why it matters now.** Prompt 11 refined the source grid around each crossing and measured a 35×
improvement in the consumer's $\varphi$ — with the background held on the *base* grid. **Production
rebuilds the background on the new grid** (`main.py` passes `z_sample=z_source_sample`), and in
that configuration refining the lattice at a step makes the ringing lower **and narrower**, so a
cubic through closer samples resolves it *worse*:

| background built on | samples from | QCD $T_k$, $k=10^5$ |
|---|---|---|
| base | base | 1.4055e-05 |
| base | shipped | **3.9744e-07** ← prompt 11's harness |
| shipped | base | 1.1851e-04 |
| shipped | shipped | **2.4859e-05** ← **production** |

Until this is repaired, prompt 11's fix does not reach production at the crossings, and **no source
grid density can reach it** — which is why prompt 12's criterion excludes a 0.15-in-$u$ halo around
each crossing.

## 2. The change

1. **Segment `_build_derivative`'s fit at `_cosmology_break_points(cosmology, z_lo, z_hi)`.** One
   spline per branch, each fitted only on the fit-grid points strictly inside its own branch, with a
   pad that keeps every node off the step — prompt 06's `SEGMENT_EDGE_PAD_LOG1PZ` is the precedent
   and its argument is in log 06. Evaluate by dispatching on $u$ against the ascending edge list.
   **Compare $u$ against $u$, never a recovered $z$ against a $z$** (README §2 (i)).

2. **Segment `_create_functions._build_func` (`:592`) the same way.** It splines the *stored*
   values over the sample grid for any attribute the cosmology does not supply as a method, so it
   carries the same defect for every such quantity. If you conclude that one of the two sites does
   not in fact need segmenting, **say so with a measurement**, not by assertion.

3. **The break points must reach `compute_background`.** `_cosmology_break_points` is already
   duck-typed and already imported in this module; a cosmology that declares nothing must take a
   code path that is **numerically indistinguishable** from today's, which is this prompt's central
   acceptance test and README §2 (g).

4. **Degenerate geometry.** A branch too narrow to hold `order + 1` fit points must raise something
   that names the problem, never silently drop to a lower order or produce a spline fitted across
   the step after all. Prompt 06 solved exactly this and its log records the cases; reuse the
   shape. Decide and state what the **padding** does at a segment edge — the existing pad extends
   the *outer* ends of the fit grid, and whether each branch also needs its own end treatment is a
   design question this prompt must answer explicitly rather than inherit.

5. **Bump `T_Z_REPRESENTATION_VERSION` to 6** on `LambdaCDM_GenericEOS` and add a row to the table
   in the comment block above it. This moves every stored QCD $\omega_{\rm eff}$ and therefore every
   stored phase: it is a keyed change with a regeneration attached, which is precisely what prompt
   03 built the constant for.

6. **Regenerate the QCD reference fixture** with prompt 02's generator, in this commit, and quote
   the largest relative move per key. Re-score the tests prompt 02's map lists under prompt 04 §2
   item 5's rule — a tolerance may tighten or hold; one that must loosen is a finding; **more than
   one is a stop and ask**.

## 3. Tests

1. **The ringing is gone, measured the way §10.4 measured it.** `epsilon` from the model against a
   central difference of the cosmology's own `Hubble`, at offsets either side of each declared
   crossing. Target: the two genuine steps come down to the **3.9e-09 / 8.1e-10** regime that holds
   away from a crossing. Quote all three crossings, and keep `EOS_T_LO` as the control — it must
   stay at ~1.6e-09 and **must not get worse**.

2. **The four-way table, re-taken.** Background built on {base, shipped} × samples from {base,
   shipped}, QCD $T_k$ at $k=10^5$ and at least one other (sector, $k$). The production row —
   **shipped / shipped** — is the one that matters, and the acceptance is that it comes down from
   **2.4859e-05 rad** to at or below prompt 11's **3.9744e-07 rad**. `docs/qcd-background-audit/grid_density_criterion.py`
   is prompt 12's tool and already builds all four configurations; reuse it rather than writing a
   third scorer, and say if you had to extend it.

3. **Show it fails on `HEAD~1`.** Check out the previous `ComputeTargets/BackgroundModel.py`, run
   the suite, and confirm the ringing test fails, naming it. A test that passes both before and
   after has measured nothing (README §0.2).

4. **A cosmology that declares nothing is bit-identical.** `LambdaCDM`, `RadiationModel` and every
   stand-in build one segment and must be **byte-identical** to `HEAD~1` — dump `Hubble`, `rho`,
   `epsilon`, `d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3`, `wPerturbations` and `d_wPerturbations_dz`
   as `float.hex()` on a dense grid and `cmp`. Quote the comparison; prompt 03's log has the recipe
   and the worktree caveat.

5. **Cost.** `compute_background`'s QCD wall time and integrand evaluations, before and after,
   **evaluation counts leading**. Prompt 07 left it at 6,936 evaluations per cumulative table.
   Segmenting adds splines, not evaluations; a large regression means something other than the
   measured design was built.

## 4. Acceptance

| Quantity | Before | Target |
|---|---|---|
| `epsilon` ringing at `T_LO` | 2.04e-02 | **≤ 1e-08**, the away-from-crossing regime |
| `epsilon` ringing at `T_120_MEV` | 1.03e-03 | **≤ 1e-08** |
| `epsilon` at `EOS_T_LO` (control) | 1.6e-09 | **unchanged, and no worse** |
| QCD $T_k$, $k=10^5$, **production configuration** | 2.4859e-05 rad | **≤ 3.9744e-07 rad** |
| LambdaCDM, `RadiationModel`, stand-ins | — | **bit-identical** |
| `T_Z_REPRESENTATION_VERSION` | 5 | **6** |
| QCD `compute_background` evaluations | 6,936 per table | **quoted**; no large regression |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/13-segment-background-derivative-splines.md` per README §5.1, carrying: which of the
two spline sites you segmented and what each bought, separately; the padding decision of §2 item 4;
the ringing table for all three crossings; the four-way table; and the fixture's largest move per
key.

Close `[12-background-derivative-fit-grid-rings-at-a-step]` on this campaign's board §3 → §4 if your
measurements close it, and update `docs/OPEN_ISSUES.md` in the same commit with its count and date
corrected. Update the board's §1 row, its version table and its header — the campaign reopens at
13 prompts and closes again here.

**Two things to record on the board that are not yours to decide:**

- **README §7 D5 is settled: the user's decision is to keep both `BREAK_POINT_KIND` values as they
  are** (`TkNumericIntegration.BREAK_POINT_ALL`, `GkNumericIntegration.BREAK_POINT_DISCONTINUITY`),
  which is prompt 08's recommendation. Record it on the board and in §7 D5's entry; **change no
  value.**
- **Prompt 12's density recommendation stays open and unimplemented.** It is
  `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` and it is the
  user's, with a full regeneration attached. Do not act on it here, and do not let this prompt's
  grid-adjacent measurements be read as acting on it.

Commit subject, or something equally specific:
`Segment the background derivative splines at the declared break points`
