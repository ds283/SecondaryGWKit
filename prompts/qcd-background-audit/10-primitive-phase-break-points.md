# Prompt 10 — `PrimitivePhase` on the 3-point break set

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[13-consumer-spline-crosses-eos-break-points]` on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3
**Implements:** audit §8 recommendation **6** — *"Re-run `prompts/phase-representation` prompt 02,
whose blocker is removed by 3 and whose premise is corrected by §2."*
**Supersedes:** [`prompts/phase-representation/02-primitive-phase-break-point-knots.md`](../phase-representation/02-primitive-phase-break-point-knots.md),
which stopped — **read it and its log in full before designing anything**
**Depends on:** 07 (the blocker), 09 (a verified base). **Gated: README §7 D7.**
**Recommended model:** **Opus** — this design stopped once already.

**Files you may touch:** `ComputeTargets/primitive_phase.py`, `ComputeTargets/GkSourcePolicyData.py`
and `ComputeTargets/TkSourceFunctions.py` **only** at their `PrimitivePhase(...)` call sites,
`ComputeTargets/BackgroundModel.py` **only** if §2 item 1 forces it,
`ComputeTargets/tests/test_primitive_phase.py`, `test_gk_source_primitive_phase.py`,
`test_tk_source_functions.py`, plus this campaign's log and board, the `GkTk-remedial` board entry,
and `docs/OPEN_ISSUES.md`.

**Do not touch:** `CosmologyModels/` — the break points are now correct and prompt 07 owns them;
`Quadrature/`; `LiouvilleGreen/`; either producer; any `Datastore/` factory; `main.py`.

**Read first, in this order:** `prompts/phase-representation/02-primitive-phase-break-point-knots.md`;
`prompts/phase-representation/logs/02-primitive-phase-break-point-knots.md` **in full** — it is the
record of what a knot vector does and does not buy, and it contains three ranked options and a
warning that none of the three schemes may be scored at $k=10^5$ alone;
`prompts/phase-representation/README.md` §2 (d) — **splitting at declared break points is not
chunking, and a reviewer must not confuse them**; `prompts/phase-representation/IMPLEMENTATION_STATE.md`
§5 note 6; `ComputeTargets/primitive_phase.py`; `docs/gktk-remedial-verification.md` §3.5, §3.6;
`docs/qcd-background-verification.md` §3 as prompt 09 wrote it.

---

## 1. What changed since prompt 02 stopped

Prompt 02 of `prompts/phase-representation` was `BLOCKED` on three findings, and **two of them are
now false**:

| Prompt 02's finding | State after prompt 07 |
|---|---|
| A repeated-knot vector is **singular on all six production grids** — 226–325 break points inside a 1,016–1,401 sample range, none coinciding with a sample | **False.** `BREAK_POINT_ALL` is now **3** points. Prompt 07 §3 item 5 asserts the knot vector constructs on all six |
| At `BREAK_POINT_DISCONTINUITY` the knot vector made the measured consumer error **2× worse** | **Re-measure.** That was against a background carrying a $10^{-4}$-level defect at every knot |
| The kink at the declared discontinuity is only **1.6e-08 / 1.4e-07 rad** — 1 % and 4 % — of the 1.907e-06 / 3.186e-06 rad the issue was charged with; the rest is $\varphi$'s own structure on a $\pm3$-interval scale | **Probably still true, and it is the thing to establish first.** Prompt 02 measured it honestly and nothing since should have changed it |

So this prompt's first job is **not** to build anything. It is to re-take prompt 02's §4
measurements on the corrected background and find out how much of the 1.907e-06 / 3.186e-06 rad
survives at all. **If the error is already at the floor, the right outcome is to close the issue
with a measurement and change no code**, and that is a `COMPLETE` result, not a failure.

## 2. The change, if a change is warranted

1. **Get the break points to `PrimitivePhase`**, as prompt 02 §2 item 1 describes:
   `_cosmology_break_points(cosmology, z_lo, z_hi, kind)` already returns an ascending array in
   $u=\log(1+z)$, the variable `PrimitivePhase` splines in, and empty for any cosmology that
   declares nothing. Prefer an **explicit keyword-only `break_points=None` parameter** appended last
   (the prompt-15 precedent) over a sixteenth `ModelFunctions` field.
   **Which `kind`:** after prompt 07 the two sets are 3 and 2 points and differ only by
   `EOS_T_LO`, where $w$ kinks and $g_s$ does not. **Measure which serves $\varphi$ better; do not
   assume.**

2. **The construction.** A **repeated-knot vector** (`make_interp_spline(..., t=...)`) is now the
   default, because prompt 07 removed the reason it could not be used: one spline object, one
   `derivative()`, continuity dropped only at the break points. Per-segment splines remain the
   fallback and remain **textually close to the chunking `GkTk-remedial` prompt 08 deleted**. If you
   take them, your log must give `phase-representation` README §2 (d)'s argument — physically
   declared boundaries rather than an arbitrary `logstep`, a bounded residual rather than the
   growing phase, no rebased ordinates, no switch discontinuity — and your tests must
   **demonstrate** the last of those, not assert it.

3. **`num_chunks` keeps returning 1.** It reports the *phase-spline chunking* prompt 08 deleted and
   `QuadSourceIntegral` persists it as `WKB_phase_spline_chunks`; segmenting the residual spline is
   not that. Anything else is a changed stored value and a **stop**.

4. **Do not change what is splined** ($\varphi$ alone, never the leading term, never the full
   phase), `spline_order`'s default of 3, or any signature of `raw_theta`, `theta_mod_2pi`,
   `theta_deriv`, `build_phi_samples`.

5. **A cosmology declaring nothing must be bit-identical.** LambdaCDM, `RadiationModel` and every
   stand-in take the unchanged `make_interp_spline` call.

## 3. Measurement, and the trap prompt 02 recorded

**Score at all three wavenumbers, never at $k=10^5$ alone.** That is prompt 02's own warning and
`[02-consumer-phi-below-the-storage-granularity]` is why: the residual $\varphi$ is recovered as a
difference of two numbers of size $k\tau$, so at $k=3\times10^8$ its whole range is **2.0 ulp** of
the stored phase — 3 distinct values over 1,377 samples — and differentiating that staircase makes
`theta_deriv` **3× and 10× worse than omitting $\varphi$ altogether**. A scheme that looks good at
$k=10^5$ can be a regression at $3\times10^8$, and **that issue is not this prompt's to fix**
(README §0.5): it is per-region anchoring, and no break-point treatment touches it.

Report, before and after, for **both sectors at all three wavenumbers**:

- the consumer phase error in rad and in ulp of the span (§3.5's twelve rows);
- `theta_deriv` against $\omega$ (§3.6), **split** into the break-point component and
  `[02-consumer-phi-below-the-storage-granularity]`'s;
- cost per `PrimitivePhase` build, wall time and evaluations, both models. The $G_k$ figure to beat
  is **0.0010 s / 468 evaluations**; **stop if it exceeds 2×**.

## 4. Tests

1. **The kink is resolved** on a stand-in with a declared break point and a genuine kink in
   $\varphi$ (`test_numeric_break_points.py` builds the stand-in). **Show it fails on the old code.**
2. **Smooth cosmologies are bit-identical** in `raw_theta`, `theta_mod_2pi` and `theta_deriv` over a
   dense grid, in both sectors.
3. **No switch discontinuity**, if you took per-segment splines: $\varphi$ and $\varphi'$ across
   each boundary agree to the floor. Prove it; do not assert it.
4. **Degenerate geometry**: a break point outside the sample range, one coinciding with a sample,
   two closer than a sample spacing, a segment with fewer samples than the spline order. Each works
   or raises something that names the problem; none produces a silently wrong spline.

## 5. Acceptance

| Quantity | Base (`9daa2cb`) | After prompt 09 | Target |
|---|---|---|---|
| Consumer phase, QCD $k=10^5$, $G_k$ / $T_k$ | 1.907e-6 / 3.186e-6 rad (8 / 428 ulp) | measured by 09 | **≤ 1e-6 rad and ≤ 2 ulp**, *or* a measurement showing the residue is not the break points' |
| The ten rows at 1.00 ulp | 1.00 ulp | — | **unchanged** |
| `theta_deriv` vs $\omega$, QCD interior | 2.3e-7 – 3.3e-4 | measured by 09 | **≤ 1e-6**, *or* split and attributed |
| `PrimitivePhase` build cost | 0.0010 s / 468 evals | — | **≤ 2×** |
| `num_chunks` | 1 | 1 | **1** |

**A residue attributed to `[02-consumer-phi-below-the-storage-granularity]` is a narrowing,
reported, not a miss hidden.** Say so and the Result is `COMPLETE WITH DEVIATIONS`.

## 6. Log and commit

Log to `logs/10-primitive-phase-break-points.md` per README §5.1. The log must state **which
construction was taken and why**, and — if per-segment — the "this is not chunking" argument. Move
`[13-consumer-spline-crosses-eos-break-points]` to the `GkTk-remedial` board's §4 and update
`docs/OPEN_ISSUES.md` in the same commit. Append a dated section to
`docs/qcd-background-verification.md` (additive) with the before/after tables.

Commit subject, or something equally specific:
`Break the consumer phase spline at the cosmology's own break points`
