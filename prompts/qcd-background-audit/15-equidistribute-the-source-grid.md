# Prompt 15 — Build the source grid by the measured criterion, at the cap that never coarsens

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` on this
campaign's board §3 — prompt 12's recommendation, which the user has now decided to take
**Implements:** audit §8 recommendation **5**, its research half, measured by prompt 12 and decided
by the user · **Measurements:** [`docs/qcd-background-verification.md`](../../docs/qcd-background-verification.md) **§10**
**Depends on:** 13 (so the background no longer rings at a crossing and the measurement means
something) and 14 (so the construction change is keyed before it lands).
**Recommended model:** **Opus** — the numerics are prompt 12's and already measured; the risk is
scope.

**Files you may touch:** `CosmologyConcepts/wavenumber.py` (`populate_z_sample` and the grid
construction prompt 11 added), `main.py` (**the grid-construction and tagging hunks only**),
`ComputeTargets/tests/test_source_grid.py`, `ComputeTargets/tests/test_main_plumbing.py`,
`ComputeTargets/tests/wkb_reference_data.json` (**via the prompt-02 generator only**), the test
modules prompt 02's map identifies (**tolerances, thresholds and comments only**), plus this
campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `CosmologyModels/`; `ComputeTargets/BackgroundModel.py`; `extract_*.py` — prompt
14's exception was for prompt 14 and the general stop condition is back in force; `QCD_EOS.py`;
`Quadrature/`; any compute target; any `Datastore/` factory beyond the tag labels `main.py` builds.

**Read first:** `docs/qcd-background-verification.md` **§10 in full**, especially §10.0 (the
recommendation and the ceiling it carries), §10.3 (the criterion), §10.5 (the cap ladder) and §10.6
(the two constraints); `docs/qcd-background-audit/grid_density_criterion.py`, which **already
implements the criterion** and is the tool, not a starting point to rewrite;
`prompts/qcd-background-audit/logs/12-grid-density-criterion.md`; prompt 11's grid construction as
it stands.

---

## 1. The decision this prompt implements

Prompt 12 measured that a uniform `source_samples_per_log10z` is wrong **in both directions at
once**: the grid misses by **7.84×** in the top decade of each Liouville–Green band at $k=10^5$,
and is over-resolved by up to **$10^{19}$** in error at the bottom of the range. It offered a cap
ladder rather than a number, because the criterion measures $\varphi$ only and is therefore a
**lower bound on density, never an upper one**.

**The user's decision, which sets the cap and is not open here:**

> *This is a science code. We want to use compute resource sensibly, efficiently, and without
> extravagance, but not take any short cuts or risks: there is no reward for doing so. The only
> thing we get is an unreliable published result. I would much rather have the grid slightly more
> dense than needed, than have it underdense.*

So: **take the cap-1× column — "never coarser than today, anywhere" — and do not take the saving.**
At the same sample count (QCD 1,761 against the shipped 1,773; LambdaCDM 1,634 against 1,732) it
puts every production row inside its floor where the shipped grid misses two of them, and it
triggers `[03-derivative-pad-clamp-on-coarse-grids]` **not at all**. The 1.75×/2.06× saving at
cap 2× is measured, is real, and is **declined**: the whole of it is coarsening at low $z$, the
derivative-fit padding clamps the moment that coarsening begins, and the criterion cannot see the
numeric ODE's samples, the four `CumulativeTable`s' Gauss panels or `QuadSourceIntegral`'s
abscissae, none of which is measured anywhere.

**Regeneration cost is not a constraint.** The user has stated that everything is currently
development, there is no production data to curate, and a superseded datastore retains archival
value. Do not trade accuracy against regeneration in any decision in this prompt, and do not
present a cheaper grid as a virtue.

## 2. The change

1. **Construct the grid by fourth-derivative equidistribution**, $h(u)^4\,|\varphi''''(u)|/384 \le
   \varepsilon$ in $u=\log(1+z)$, with $\varphi'$ the integrand of
   `ComputeTargets.phase_residual.phase_residual_integrand` times $(1+z)$ and $\varphi''''$ by the
   five-point stencil prompt 12 used. **Reuse `grid_density_criterion.py`'s implementation**; do not
   re-derive it. It needs $H$, $c_s^2$, their $z$-derivatives and $k$ and nothing else — in
   particular **no `BackgroundModel`**, which is what makes it usable before the grid exists.

2. **The cap is "never coarser than today, anywhere."** State it as an explicit constant with its
   own comment, so that a later prompt can move it deliberately rather than by accident, and so the
   declined saving is visible in the code rather than only in a log.

3. **Prompt 11's protected set and break neighbourhoods survive unchanged.** The crossings, the
   straddling pairs and the ±5-interval refinement are prompt 11's and are *local*; this prompt
   changes the *base* density between them. Do not fold one into the other, and assert that the
   protected points are still present and still straddled.

4. **The response grid stays a decimation of the source grid** — prompt 12 checked this on all
   twenty candidate grids and it held. Assert it, and **quote the resulting response-grid size**:
   prompt 12 measured 147 at cap 1× against the 148 a blind stride gives today, and warned that the
   ladder's coarse end reaches 44, which is "almost certainly not usable". You are at the safe end;
   say so with the number.

5. **Bump the source-grid construction version** prompt 14 introduced, to **2**, and add its row to
   that constant's table. This is exactly the change that constant exists to record.

6. **Regenerate the QCD reference fixture** with prompt 02's generator, in this commit, and quote
   the largest relative move per key. Re-score under prompt 04 §2 item 5's rule — a tolerance may
   tighten or hold; one that must loosen is a finding; **more than one is a stop and ask**.

## 3. Tests

1. **The grid is denser where the criterion says it must be, and nowhere coarser.** Assert the cap
   directly: no interval on the new grid exceeds the corresponding interval on the shipped grid, on
   either model. This is the user's decision expressed as a test and it is the one that must not be
   weakened.
2. **The rows that missed now pass.** The two QCD $T_k$ rows at $k=10^5$ that read 7.84× their
   floor come inside it. Score with `grid_density_criterion.py` in the **production
   configuration** — background rebuilt on the new grid, samples from the new grid — which is
   §10.4's four-way table and the trap prompt 11 fell into. **Show it fails on `HEAD~1`.**
3. **A cosmology declaring nothing.** LambdaCDM's grid changes here (it is a density change, not a
   break-point change), so the prompt-11 bit-identity test no longer applies and **must be updated
   rather than deleted** — say so as a `STRUCTURALLY REQUIRED` deviation. What must still hold is
   that the construction consults no equation-of-state module and that a cosmology with no break
   points gets no protected points.
4. **`[03-derivative-pad-clamp-on-coarse-grids]` does not bind.** Assert the lowest grid interval
   stays below $-\log(0.9)/12 = 8.7800\times10^{-3}$ in $u$. This is the measured consequence of
   choosing cap 1× and it should fail loudly if a later prompt coarsens.
5. **Response grid a subset of the source grid**, and its size quoted.

## 4. Acceptance

| Quantity | Shipped | Target |
|---|---|---|
| QCD source samples | 1,773 | **~1,761**, none coarser anywhere |
| LambdaCDM source samples | 1,732 | **~1,634**, none coarser anywhere |
| Worst row against its floor | **7.84×** (miss) | **≤ 1×**, measured 0.14× |
| `[03-derivative-pad-clamp-on-coarse-grids]` | does not bind | **still does not bind** |
| Response grid | 156 | **quoted**, and still a subset |
| Source-grid construction version | 1 | **2** |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/15-equidistribute-the-source-grid.md` per README §5.1, carrying: the cap constant and
its argument; the per-model sample counts and the response-grid size; the four-way table in the
production configuration; and the fixture's largest move per key.

**Restate, where the result is and not in a footnote** (`docs/OPEN_ISSUES.md` §5): no pipeline has
ever been run on any grid in this line of work, **including the one that ships**, and no
verification run has reached production $x$. This prompt changes the grid on the strength of a
Gauss-quadrature oracle, not a pipeline run, and the log must say so plainly.

Close `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` on the board
§3 → §4 and update `docs/OPEN_ISSUES.md` in the same commit.

Commit subject, or something equally specific:
`Equidistribute the source grid on the phase residual's curvature`
