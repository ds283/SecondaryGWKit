# Prompt 02a — Make the source grid buildable at every production anchor

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** board item **T13**; unblocks **T4**, **T5**, **T6**, **T7** and **T11**
**Closes:** `[01-v2-density-raises-at-the-qcd-production-anchor]`
**Narrows:** `[01-density-criterion-imposed-outside-the-wkb-region]` — turns it from a hard
failure into a measured cost, which is what makes it **T7**'s to decide rather than a blocker
**Depends on:** prompts 01 and 02. It uses prompt 01's named grid generations
(`wkb_reference.source_grid`) as its whole test surface.
**Recommended model:** **Opus** — the fix is four lines, and the reason it is not a four-line
prompt is that it is the first production change this campaign makes, it is made under a widened
§0.5 boundary (**§7 D6**), and its acceptance is a *bit-identity* claim against two published
digests. Getting the guard right is easy; keeping it inert on everything already measured is the
work.

**Files you may create or touch:**
`main.py` — `source_grid_spacing_profile` **only**;
`ComputeTargets/tests/wkb_reference.py` — the production-anchor constants **only**;
`ComputeTargets/tests/test_source_grid.py` — new assertions, existing ones unchanged;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `config/defaults.py`; `CosmologyConcepts/wavenumber.py` (`build_z_sample`,
`SOURCE_GRID_*`, `_solve_horizon_exit`); `ComputeTargets/phase_residual.py` — including
`RESIDUAL_WKB_REGION_MARGIN` and `residual_node_range`; any `Datastore/` file; any other function
in `main.py`. **Do not tighten the anchor solve and do not touch the grid digest** — both are real
and both belong to prompts 03 and 05 (§4, and `[02a-grid-digest-not-reproducible]`).

**Read first:** README §0.5 **as amended by D6**, §2 (b), §5 rules 4, 6 and 8; board §3's two
`[01-…]` entries **in full**, including the orchestrator's narrowing block; the comment above
`RESIDUAL_WKB_REGION_MARGIN` (`ComputeTargets/phase_residual.py:220-239`); the docstring of
`residual_node_range`; `main.source_grid_spacing_profile` in full.

---

## 1. Why this prompt exists

`main.source_grid_spacing_profile` **raises** on `QCD_Cosmology` when the source grid is anchored
where a QCD production run anchors it, so there is no version-2 production grid on QCD at QCD's own
$z_{\rm init}$. Prompts 03, 04 and 06 are each chartered to measure "over the production grids on
all three models". Without this prompt every one of them must measure QCD at **LambdaCDM's**
anchor, which is the defect `[00-three-production-grid-reproductions]` named and the reason prompt
01 exists. The campaign would publish, three more times, the thing it was convened to stop.

Reproduced at `main.py`'s own configuration — `z_init = k_exit_earliest.z_exit_suph_e5` from
`_solve_horizon_exit(QCD, k = 3\times10^8, -5)`, all fifty wavenumbers of `main.py:3587`, both
Liouville–Green sectors, `zend = 0.1`, 100 samples per decade:

| cosmology | anchor | version 2 | version 1 |
|---|---|---|---|
| QCD | own, 3.30033444460513e+16 | **`ValueError`** | 1793 |
| QCD | LambdaCDM's, 2.0636395964161516e+16 | 1996, `4849552b` | 1773 |
| LambdaCDM | own | 1778, `60a3205a` | 1732 |

> `ValueError: phase_residual[Gk]: the Liouville-Green frequency is not positive at
> z = 8.6447769e+11 for k = 266544.64 (leading = 3.8377293e-26, correction = -2.2713588e-21,
> omega^2 = -2.2713204e-21)`

**The board's open question is now answered: `main.py` does reach the raise.** Perturbing `z_init`
relatively, the construction raises at $10^{-16}$, $10^{-14}$, $10^{-12}$, $10^{-10}$ and $10^{-8}$
and first builds at $10^{-6}$. The anchor's own solve is nowhere near that loose, so no re-solve of
it escapes. Record this in the log: it is the measurement that closes the issue, and it is the
reason this is a blocker rather than a curiosity.

## 2. The mechanism — and why the crossing mask is not the fix

The orchestrator's review attributed the raise to an **ordering** defect: the five-point stencil
runs over every node of `inside` before `usable &= |u_profile - u_break| > SOURCE_GRID_CROSSING_MASK_U`
is applied, so a node the mask would discard is still evaluated. That is true and it is what makes
QCD's anchor differ from LambdaCDM's. **It is not the mechanism, and reordering is not the fix.**

`residual_node_range` establishes its band by testing $\omega^2$ **at the grid's nodes**, and it is
correct there. The stencil then evaluates `dphi_du` at $u\pm\delta$ and $u\pm2\delta$ — **off-node**
points, at `SOURCE_GRID_CURVATURE_STEP_U`. Where $H$ steps, $\omega^2$ can be negative *between*
two nodes that both pass the margin test. The declared crossing is one instance; the second case
recorded in `RESIDUAL_WKB_REGION_MARGIN`'s own comment, on QCD at $z = 3.61\times10^{15}$ away from
any declared crossing, is another, and no crossing mask would ever catch it. A steep step does not
have to be *declared* to defeat a node-wise test.

So the defect is that the profile assumes the expansion exists everywhere its band's *interior*
reaches, when what was established is that it exists at the band's *nodes*.

**You may not reorder the mask**, and the reason is evidential rather than stylistic: moving it
ahead of the loop changes which nodes are evaluated, hence `usable`, hence the log-interpolation
fit, hence the profile — so it could move a published grid, and nothing has measured whether it
does. Record it in "Observations not acted on".

## 3. What to do

**A node at which the Liouville–Green expansion does not exist is exactly a node to mark
`usable = False`.** The criterion already carries that mask and already fills a masked node by
log-interpolation from either side. Use the machinery that is there:

1. Guard the stencil evaluation in `source_grid_spacing_profile`. Where the four `dphi_du` calls
   raise `ValueError`, leave `d4[i] = 0.0` and `usable[i] = False` and continue to the next node.
   Catch `ValueError` and nothing wider — a `TypeError` or a `KeyError` there is a defect, not a
   region boundary.
2. **Count the guarded nodes and report them.** A silent `except` buries the evidence **T7** needs:
   53 guarded nodes on QCD is not noise, it is
   `[01-density-criterion-imposed-outside-the-wkb-region]` expressed as a number. Accumulate the
   count across the $(k, \text{sector})$ loop and surface it — `main.py:963` already prints a line
   about the criterion, and that is the natural place.
3. **Refuse above a fraction of the band.** A guard that can silently absorb an arbitrarily wrong
   band is a worse defect than the raise, because the raise at least stops. Add a constant beside
   the other `SOURCE_GRID_*` values, choose its value from the measured figure with a stated
   margin, and raise a `ValueError` naming the count, the band size and the constant when the
   fraction is exceeded. The prompt does not fix the number; **you** fix it, from your own
   measurement, and the log records why.
4. **Name QCD's own anchor in the test tree.** `wkb_reference.PRODUCTION_Z_INIT` is a single
   constant whose comment claims it is "the top of the universal source grid on **both** production
   cosmologies". That is false — it is LambdaCDM's, and QCD's is 3.30033444460513e+16. Make the
   anchor per-cosmology, correct the comment, and leave `PRODUCTION_Z_INIT` resolving to the
   LambdaCDM value if that keeps existing callers bit-identical. Prompts 03, 04 and 06 need to be
   able to *say* which anchor a figure was taken at, in the same way §2 (b) makes them say which
   grid generation.

Nothing else. In particular the band stays exactly as `residual_node_range` returns it: this prompt
makes the grid buildable, it does not decide where the criterion should apply.

## 4. Acceptance

1. **QCD at its own anchor builds.** The version-2 grid at 3.30033444460513e+16, fifty production
   wavenumbers, both sectors, builds and is strictly descending, duplicate-free and no closer than
   `SOURCE_GRID_MIN_SEPARATION` anywhere. The probe measured **2034 samples** and **53 guarded
   nodes**; quote your own figures and reconcile any difference rather than asserting the probe's.
2. **Both published grids are bit-identical.** QCD at LambdaCDM's anchor is **1996 samples, digest
   `4849552b`**; LambdaCDM at its own anchor is **1778 samples, digest `60a3205a`**; both with
   **zero** nodes guarded. This is the load-bearing test: it is what says the guard is inert on
   every figure already in the record. A digest that moves is a **stop**, not a finding to write up.
3. **The refusal fires.** A synthetic case whose guarded fraction exceeds the constant of §3.3
   raises, with the count in the message. Build it by narrowing the band or widening the stencil in
   the test, not by editing the production constant.
4. **`test_source_grid.py`'s existing 38 tests still pass**, unchanged, and the four assertions its
   docstring calls the ones that must not be weakened are untouched.
5. `ComputeTargets` and `CosmologyModels` suites green at the counts in the board header, allowing
   for `[computetargets-suite-flake]`'s wall-clock assertion.

## 5. Out of scope, and where each piece went

- **The band.** Whether the criterion should run over a horizon-based band of its own rather than
  over `residual_node_range`'s anchor-coverage band is `[01-density-criterion-imposed-outside-the-wkb-region]`
  and is **T7**, prompt 04. Your guarded-node count is the evidence that issue has been waiting
  for; put it in the log's "State handed to the next prompt" in the form prompt 04 can use.
- **The crossing mask's ordering.** §2 above. Observation only.
- **The anchor solve.** `_solve_horizon_exit`'s convergence is `xtol + rtol \cdot |u|`, and with
  `rtol = 1e-8` at $u \approx 37.6$ the binding term is $3.8\times10^{-7}$ — not the
  `xtol = 1e-10` the record quotes. That is **T6**, prompt 03, and it is the prerequisite for the
  digest work; do not pre-empt either. See `[02a-grid-digest-not-reproducible]`.
- **The grid digest and `DEFAULT_REDSHIFT_RELATIVE_PRECISION`.** Prompt 05, for the reason §4 gives:
  05 already invalidates the datastore, and folding the tag change in costs one invalidation
  instead of two.

## 6. The log

`logs/02a-source-grid-density-guard.md`, on `GkTk-remedial` §5.1's template, classifying every
deviation. It must carry, beyond the template:

- the `z_init` sensitivity scan of §1, as the measurement that closes
  `[01-v2-density-raises-at-the-qcd-production-anchor]`;
- the guarded-node count per cosmology **and per sector**, with the band size beside it, in the form
  **T7** can consume;
- the two bit-identity results of §4.2, quoted as digests, with the grid generation named (§5 rule 6);
- the justification for the §3.3 constant, with the margin stated;
- and, under "State handed to the next prompt", the per-cosmology anchor constants of §3.4 by name,
  since prompts 03, 04 and 06 cite them.
