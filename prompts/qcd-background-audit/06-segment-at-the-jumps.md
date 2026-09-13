# Prompt 06 — Segment the representation at the jumps: fix the max, and close T1

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §8 recommendation **2**, its second half — and **closes T1**, the campaign's
critical finding
**Measurements:** audit §2, §4, §5 · **Decisions:** README §7 **D3** (nodes and order) and **D4**
(which temperatures are edges)
**Design facts:** README §2 (b) — **this prompt *is* §2 (b); read it twice**
**Depends on:** 05. **Blocks 07.**
**Recommended model:** **Opus** — the highest-risk prompt in the campaign. The failure mode is
silent: an edge misplaced by one node leaves the full error in place and every other number still
improves.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`,
`ComputeTargets/spline_wrappers.py` **only as prompt 05 §2 item 2 left it**,
`ComputeTargets/tests/wkb_reference_data.json` (**via the prompt-02 generator only**), the test
modules prompt 02's map identifies (**tolerances, thresholds and comments only**),
`CosmologyModels/tests/test_T_z_representation.py`, `test_temperature_spline.py`, plus this
campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `integration_break_points`'s logic (prompt 07); `QCD_EOS.py` — in particular its
`break_temperatures_GeV` and `discontinuity_temperatures_GeV`, which are the input to this prompt,
not its output; anything in `ComputeTargets/` beyond the fixture and tolerances.

**Read first:** **audit §2 in full**, then §4; `CosmologyModels/tests/T_z_reference.py`'s
`jump_locations` (prompt 01 built it, and its bisection is the algorithm this prompt needs);
`docs/qcd-background-audit/measure_T_z_representation.py` `build_segmented`;
`LambdaCDM_GenericEOS.py` as prompt 05 left it; `_temperature_crossing_log1pz` (`:222`) — **note
that it root-finds on a difference, which is exactly what §2 (b) forbids for an edge**, and that
prompt 07 owns whether it survives.

---

## 1. What is wrong

$T(z)$ is **genuinely discontinuous**, not merely kinked. Across the lowest crossing
$z_c = 4.25337\times10^7$, $g_s$ steps $3.940\to3.931$, so on each branch $T\propto(1+z)$
*exactly*, and

$$F(u) = \log\!\big(T/[T_{\rm CMB}(1+z)]\big)$$

is flat to twelve decimals on both sides with a step of **7.614e-04** between them (audit §2; prompt
01's test 5 pins it). **No single smooth approximant can represent a step at any node count** — the
audit's table shows the max pinned near the jump height at 500, 2,000 and 5,000 nodes alike. A
representation *segmented at the step* reproduces it **exactly**.

This is the change that closes **T1**: the $3.461\times10^{-8}$ systematic error in
$\int\mathrm{d}z/H$, worth 47.5 / 4.75e3 / **1.43e5 rad** at $k=10^5/10^7/3\times10^8$ against
1-ulp floors of 3.05e-7 / 3.05e-5 / 9.15e-4 rad — and which is common mode, so **only prompt 01's
guard test can see it move**.

## 2. The change

1. **Locate the edges by bisection.** README §2 (b), and the audit's own warning:

   > The segment edges must be located by bisecting the monotone $T(z)$ against $T_{\rm break}$,
   > **not** by root-finding on $T(z)-T_{\rm break}$: that difference need not have a root at a
   > discontinuity, and a bracketing solver lands beside the jump. An edge misplaced by one node
   > leaves the full error in place — measured at **5.7e-04** on a first attempt that made exactly
   > this mistake. This is the one detail an implementation must get right.

   Geometric bisection on $(1+z)$ to a relative $10^{-15}$, on the accurate `_solve_T_z` of prompt
   04 — **not** on the spline being built. Return the edges in $u$.

   **The edge-finding code belongs in production**, not in the test tree; prompt 01's
   `jump_locations` is the reference implementation and the test that scores yours against it.

2. **One $F$ spline per branch.** Nodes distributed across segments in proportion to their width in
   $u$, each segment's node set held **strictly inside** its own branch by a small pad
   (the audit uses `1e-12` in $u$) so that no segment interpolates across the discontinuity. Each
   segment needs at least `order + 1` nodes; a segment too narrow to hold them must raise something
   that names the problem, never silently produce a lower-order fit.

3. **Dispatch on evaluation** by $u$ against the ascending edge list. Keep it cheap — the audit
   measures the whole call at 2.19–2.21 µs including the dispatch, against 2.26–2.44 µs shipped, so
   a linear scan over three or four edges is affordable and a `bisect` is better. **Compare $u$
   against $u$, never a recovered $z$ against a $z$** (README §2 (i)).

4. **README §7 D4 — which temperatures are edges.** The audit's measured representation segments at
   **`break_temperatures_GeV`** (all four, including `EOS_T_LO = 0.002` GeV where $g_s$ is
   continuous to 1.751e-11 and only $w$ kinks). Keep that unless you can show a reason not to, and
   **say which you used**. Segmenting at a continuous point is harmless; missing a discontinuous
   one is not.

5. **README §7 D3 — nodes and order.** The audit's recommendation is **3,000 nodes at $k=5$**,
   measured at max 6.807e-11 / p90 3.123e-15 / median 1.773e-16, building in ~25 ms. Measure at
   least one cheaper candidate and report what the difference bought.

6. **Bump `T_Z_REPRESENTATION_VERSION` to 4**; add this prompt's row to the constant's table.

7. **Regenerate the QCD reference fixture** with prompt 02's generator, in this commit. **This is
   the regeneration that matters**: after it, the QCD reference is no longer circular, because the
   representation now agrees with the defining equation to 1e-15 and the accurate root solve is a
   genuine oracle. Update the fixture's `method` string to say so, and re-measure
   `[02-qcd-reference-floor]` and `[03-qcd-short-baseline-reference-endpoint-rounding]` — both may
   now be closable, and if they are, say so on the `GkTk-remedial` board and in
   `docs/OPEN_ISSUES.md`. Re-score the tests prompt 02's map lists under prompt 04 §2 item 5's rule.

## 3. Tests

1. **The max collapses.** Prompt 01's test 1, tightened to README §6.1's "After 06": max ≤ 1e-10,
   p90 ≤ 1e-14, median ≤ 1e-15. **Show it fails on `HEAD~1`.**

2. **T1 is closed.** Prompt 01's test 3 (`test_conformal_time_matches_the_exact_background`),
   tightened from 4e-08 to **≤ 1e-15**, and to bit-identity with the exact-background integral if
   that is what it achieves — the audit reports agreement at all 17 digits. Print the equivalent
   phase at the three wavenumbers and assert each is **below its 1-ulp floor**. **Show it fails on
   `HEAD~1`**; this is the campaign's headline check and the only test in the repository that can
   fail because the background is wrong.

3. **The edge-misplacement trap, tested.** Build a representation with one edge deliberately moved
   by one node and assert the probe-set max **returns to the $10^{-4}$ regime**. This is the
   audit's 5.7e-04 failure, turned into a guard. A test that cannot reproduce it has not
   demonstrated that the edges are where they must be.

4. **The step is reproduced exactly.** Evaluate the representation immediately either side of each
   edge and compare against `accurate_T`: agreement to the floor on *both* sides, with the relative
   step between them equal to the audit's measured value for that crossing (7.614e-04 at the
   lowest). A representation that smooths the step by even one node interval fails this.

5. **Degenerate geometry.** An edge outside the tabulated range; two edges closer together than a
   node spacing; a segment with fewer nodes than `order + 1`; an edge exactly at a tabulation
   bound. Each must either work or raise something that names the problem. **None may produce a
   silently wrong representation** — that is the failure mode this whole prompt is about.

6. **Smooth cosmologies are bit-identical**, and a constant-$g_s$ equation of state stays exact
   (prompt 05's test 2). `LambdaCDM`, `RadiationModel` and the stand-ins declare no break
   temperatures, so they build one segment and take a code path that must be numerically
   indistinguishable from prompt 05's. Assert it densely and quote the comparison.

7. **The reproduction script still runs unedited** and its §3 table's segmented row now matches
   what the production class does.

## 4. Acceptance

| Quantity | Before (after 05) | Target |
|---|---|---|
| `T(z)` error, **max** | ~7.24e-04 | **≤ 1e-10** (audit measures 6.807e-11) |
| `T(z)` error, p90 | ~8.9e-08 | **≤ 1e-14** (3.123e-15) |
| `T(z)` error, median | ~2.6e-10 | **≤ 1e-15** (1.773e-16) |
| $H(z)$ on the production grid, max / p90 / median | 1.278e-03 / 2.895e-05 / 3.424e-07 | **≤ 2e-10 / 1e-14 / 1e-15** (1.690e-10 / 6.314e-15 / 2.803e-16) |
| **$\int\mathrm{d}z/H$ relative error** | 3.461e-08 | **≤ 1e-15** — T1 closed |
| Equivalent phase at $k=10^5/10^7/3\times10^8$ | 47.5 / 4.75e3 / 1.43e5 rad | **below 3.05e-7 / 3.05e-5 / 9.15e-4 rad** |
| `T_photon` cost per call | ≤ 2.5 µs | **≤ 2.5 µs** — the audit measures the segmented form *cheaper* than shipped |
| `_build_T_z_spline` wall time | ~17 ms | **≤ 100 ms** (audit: ~25 ms for 3,000 nodes) |

**Do not loosen a target** (README §6). A miss is an issue and `COMPLETE WITH DEVIATIONS`, and a
miss on the $\int\mathrm{d}z/H$ row is a **stop**, because that row is what the campaign is for.

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/06-segment-at-the-jumps.md` per README §5.1. It must carry:

- **the segment edges to 17 digits**, and which temperatures produced them (D4);
- **the node count and order**, with the cheaper candidate's numbers (D3);
- **how the edges were located**, in enough detail that a reader can confirm no bracketing solver
  was applied to $T(z)-T_{\rm break}$ anywhere in the path;
- the before/after of every row in §4;
- what happened to `[02-qcd-reference-floor]` and
  `[03-qcd-short-baseline-reference-endpoint-rounding]`;
- **the `BREAK_POINT_ALL` count**, which prompt 07 consumes.

Close `[01-genericeos-tz-spline-floor]` on the `source-remediation` board if your measurements
close it — the question it asks is answered by §4's table — and update `docs/OPEN_ISSUES.md` in the
same commit. Update this campaign's board §1 version table and §2 rows T1 and T4.

Commit subject, or something equally specific:
`Segment the QCD temperature at its genuine discontinuities`
