# Prompt 09 — Re-run the benchmark tier and record the campaign's measured outcome

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §9 Stage 5, §11 (boundaries and deferred work)
**Reconciliation items:** C1 (the cliff's true location and mechanism), §3.2 (the hand-off to `source-remediation`)
**Depends on:** 08 (hard)
**Recommended model:** Sonnet
**Files you may touch:** `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`,
`docs/lg-phase-and-handover-followup-2026-09.md`, new `docs/transfer-remedial-verification.md`,
plus the log and the status board.
**Do not touch:** any production module, any test. This is documentation plus one benchmark run.

Read first: `RECONCILIATION.md` C1 and §3.2; `DRAFT-PLAN.md` §9 Stage 5 and §11;
`docs/lg-phase-and-handover-followup-2026-09.md` §2.4 and §2.5;
`docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:255-285`; `docs/transfer-remedial/baseline-2026-09.md`
(prompt 01's baseline); and the "State handed to the next prompt" section of every log in
`prompts/transfer-remedial/logs/`.

---

## 1. Character of this commit

Close-out. Three deliverables, in this order.

## 2. The benchmark tier — the one measurement

`bessel_tier.py:273-277`:

```python
B1_MAX_X = 1.0e12
# kappa=1000 (x_max ~ 8e15) does not complete a bessel_phase build within ~25 min:
# the phase layer, not the Levin core, is the scalability limit.  Capped at 100.
B1_KAPPA = [1.0, 10.0, 100.0]
```

`DRAFT-PLAN.md` §9 Stage 5 asks for the \(\kappa=1000\) tier to be re-run and the note updated if
the phase layer is no longer the limit.

**Run it.** `PHASE_HEADROOM * mom * B1_MAX_X` with \(\kappa=1000\) puts \(x_{\max}\approx8.6\times10^{15}\),
which is where the old construction stalls. Time the `build_phases` call separately from the Levin
evaluation — `bessel_tier.py:76-90` already returns `phase_time`, so the split is available for
free. Then:

- If \(\kappa=1000\) completes, **raise the cap** to include it and rewrite the note with the
  measured phase-build time and the Levin time, so a reader can see which layer now dominates. Keep
  the historical claim as history ("before the two-region construction, a build at this scale did
  not complete in ~25 min") rather than deleting it — `DRAFT-PLAN.md` §9 Stage 5 asks that historical
  measurements be retained as historical evidence.
- If it does not complete, **that is a stop condition** (README §4.3). Report the failure mode
  precisely: is it the phase build again, or is it now the Levin core, or is it `sin`/`cos` at
  \(8.6\times10^{15}\)? Do not raise the cap on a hopeful basis and do not extend the timeout past
  ~30 minutes to force it through.

**Correct the note's diagnosis while you are there.** The current comment attributes the failure to
"the phase layer" being "the scalability limit", which reads as a cost statement. It is not:
`RECONCILIATION.md` C1 shows the build is ~0.1 s across five decades of \(x_{\max}\) and then
*stalls* above \(x\approx2.5\times10^{15}\), because SciPy/Amos `jv`/`yv` lose argument-reduction
accuracy there, the ODE right-hand side becomes O(1)-relatively noisy, and DOP853 at `rtol=5e-14`
cannot pass its error test on noise. Whatever the re-run shows, the replacement note must state the
mechanism, not just the symptom.

Do not restructure the benchmark, retune its orders or change any other tier. One constant, one
comment, and whatever the run requires.

## 3. `docs/lg-phase-and-handover-followup-2026-09.md`

§2.4 and §2.5 are the project's standing record of `bessel_phase`'s accuracy, and they are now
superseded in part. `DRAFT-PLAN.md` §9 Stage 5 asks for four specific updates plus one removal:

1. **The measured replacement accuracy**, from prompt 08's attribution table and prompt 05's log.
2. **The offset finding** — that `phi` was a pure artefact of a loose root solve (`xtol=1e-6,
   rtol=1e-4`) at a match point where the phase was already exact, and that it *was* the whole
   tight-tolerance error: \(\phi=-4.836537\times10^{-8}\) and \(E_\theta=4.873\times10^{-8}\) at
   \(\nu=5/2\).
3. **The fixture/production tolerance distinction.** §2.4 reasons throughout from the
   `config/defaults.py` values (`rtol=1e-8, atol=1e-10`), but `main.py:520-528` uses
   `rtol=5e-14, atol=1e-25`. So the document's blanket \(x\times10^{-8}\) figure was correct only at
   fixture tolerances; at production tolerances the dominant error was the constant offset, and the
   two regimes have different mechanisms.
4. **The chunking measurement** of `DRAFT-PLAN.md` §4.6: chunking had **no** measurable effect on
   accuracy (identical to four significant figures at every \(x_{\max}\) tested), the shipped
   `chunk_logstep=125` cannot reduce the splined dynamic range by design, and the code contradicted
   its own comment. Note that this measurement is about the *Bessel* phase and that the
   cosmological \(\theta(u)\) must be measured on its own rows before anything changes there
   (README §1.1).
5. **Remove the stale blanket statements** that the Bessel oracle necessarily has an
   \(x\times10^{-8}\) floor — specifically the three bullets at §2.4 — while **retaining the
   historical measurements as historical evidence.** Mark them as superseded and dated, do not
   delete the numbers.

Also update §2.5's first bullet, which suggests storing \(Q\) and notes that "`bessel_phase` already
keeps `Q` as a spline (`"Q"` in its returned dict) and could evaluate \(\theta=xQ\) from it
directly." That option no longer exists (prompt 06 removed or repurposed `Q`), and — more
importantly — `DRAFT-PLAN.md` §4.2 explains why it was never the right answer: evaluating the \(Q\)
spline still multiplies its state and interpolation errors by \(x\). Say that; the remaining bullets
about the *cosmological* phases stand untouched and are explicitly out of scope (README §1.1).

Edit in place with dated supersession notes. Do not rewrite the document's structure.

## 4. `docs/transfer-remedial-verification.md`

The campaign's verification record, in the style of `docs/adaptive-levin-verification.md` and
`docs/backport-modules-verification.md` — read one of them first and follow its shape.

It must contain, and be checkable without reading any log:

- **What changed**, per prompt, with the commit SHA and one sentence each.
- **The acceptance table** (README §6) with the **achieved** value in each row beside the target,
  and the \((\nu,x)\) of each maximum. This is the campaign's headline result; if any row was not
  met, say so plainly here rather than only in a log.
- **The attribution table** from prompt 08 §5, verbatim.
- **The cost and domain result**: build time versus \(x_{\max}\) before and after, from prompt 01's
  baseline and prompt 05's measurements; the largest \(x_{\max}\) at which construction completes,
  before and after; and the benchmark tier outcome from §2. State the honest version of the
  performance claim (`RECONCILIATION.md` C1): a hard cliff was removed, not a cost curve.
- **The remaining floors**, named and quantified, each with what would move it: the consumer
  re-spline error on the production grid; the physical LG truncation; the Levin quadrature and
  `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` if prompt 08 found it binding; input-coordinate error in
  \(x=k\eta\) at large \(x\) (`DRAFT-PLAN.md` §7.5); and the SciPy/Amos boundaries, which are now
  properties pinned by tests (prompt 02) rather than assumptions.
- **The environment**: SciPy, NumPy, mpmath versions and the platform. §4.4's boundaries are
  properties of the bundled Amos library, not guarantees, so the record is worthless without it.
- **Deferred and handed-over work**, from README §7, plus the one cross-campaign hand-off in §5
  below.

## 5. The hand-off to `source-remediation`

`RECONCILIATION.md` §3.2 records a finding this campaign deliberately did not act on:
`ComputeTargets/QuadSourceIntegral.py`'s `_three_bessel_Levin` (`:1175-1442`) makes **eight**
`adaptive_levin_sincos` calls whose phases are signed sums of three `bessel_phase` `raw_theta`
values (`:1226, :1267, :1308, :1349`), supplying **no `theta_deriv`** — so `need_theta_Cheb` is `True`
there and Levin obtains \(\theta'\) by spectral differentiation of the raw phase, the very route
`three_bessel_integrals._phase_group`'s docstring was written to avoid. It also has the phase-group
cancellation problem prompt 07 fixed in the sibling module.

It was excluded because that file belonged to `source-remediation`, whose prompts 08–10 rewrote it
(README §1.1, §4.2). **The gap survived that rewrite**, and is now anomalous within its own file: the
new phase-group route passes `theta_deriv` (`:1008`, `LEVIN_USE_THETA_DERIV = True` at `:93`) while
these eight calls do not, and the file's own comment at `:74` notes they are the odd ones out.
Checked at planning time: that campaign's board records only its item B6 (`atol`/`rtol` forwarding)
against these call sites, **not** the missing derivative — so verify that is still so before writing
the hand-off, and if it has since been recorded, say so and do not duplicate it.

Record the hand-off in **two** places, because a finding in one campaign's docs is invisible to the
other's agents:

1. a section of `docs/transfer-remedial-verification.md`, with the line numbers, the missing
   `theta_deriv`, the missing `theta_abserr`, and what prompt 07 did in the sibling module as the
   template;
2. an entry in `prompts/source-remediation/IMPLEMENTATION_STATE.md` §3 (Active issues), in that
   file's existing format — `**[NN-shortname]** *(opened by …)* — description. **Impact:** …
   **Next step:** …`. Use a shortname that makes the origin obvious, e.g.
   `[transfer-remedial-qsi-phase-groups]`. **This is the only edit this campaign makes to that
   folder**, it adds one entry and changes nothing else, and it must not alter that campaign's
   status rows, item table or standing notes.

Also note for that campaign that `LiouvilleGreen/three_bessel_integrals.py` now supplies
`theta_abserr` and assembles \(Kt+C+R(t)\), so there is a working pattern to copy rather than a
design to invent.

## 6. Verification and acceptance

- The benchmark ran, and its result — completion or the specific failure — is recorded with times.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` and
  `-s ComputeTargets/tests` still pass. Nothing here touches code, so a failure means something
  strayed.
- Every commit SHA cited in `docs/transfer-remedial-verification.md` resolves
  (`git cat-file -e <sha>`).
- Every acceptance row in README §6 has an achieved value beside it, or an explicit "not met" with
  a reason.
- `git diff HEAD~1 --stat` touches only the files this prompt allows, plus the one entry in
  `prompts/source-remediation/IMPLEMENTATION_STATE.md`.

## 7. Log and commit

Follow README §5 and §5.1. Mark the campaign complete on
[`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md), move every resolved entry from its §3 to §4,
and leave §3 containing only what is genuinely still open.

Commit subject, or something equally specific: `Record the measured outcome of the Bessel rebuild`.
