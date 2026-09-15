# Prompt 12 — A measured criterion for the grid's density

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §8 recommendation **5**, its research half — *"its density should be set by a
measured criterion on $\varphi$'s curvature rather than by a uniform `samples_per_log10z`"*
**Depends on:** 11. **Gated: README §7 D7. This prompt stops for the user by design.**
**Recommended model:** **Opus** — it is a design question, and its answer costs the user a full
datastore regeneration.

**Files you may touch:** `docs/qcd-background-audit/` (measurement scripts and their output),
`docs/qcd-background-verification.md` (a new dated section), this campaign's log and board, and
`docs/OPEN_ISSUES.md`.

**Do not touch: any production file.** This prompt **measures and recommends**. Changing
`source_samples_per_log10z` or the grid's construction is the user's decision, taken with the
regeneration cost in front of them, and is not in this prompt's scope even if the answer is
obvious.

**Read first:** audit §7's closing paragraphs and §9's last two bullets; prompt 11's log;
`docs/gktk-remedial-verification.md` §3.5 and §3.6; `ComputeTargets/primitive_phase.py` as prompt 10
left it; `docs/OPEN_ISSUES.md` §3's `[03-derivative-pad-clamp-on-coarse-grids]` (the padding binds
at 50 samples per decade) and §5 (the standing caveat that no verification run ever reached
production $x$).

---

## 1. The question

`source_samples_per_log10z` is a command-line number, uniform across twenty decades, chosen by
nobody in particular and currently 100. The audit's position is that the density should follow
**$\varphi$'s curvature** — the residual phase the consumers spline — rather than being uniform in
$\log_{10} z$.

Three sub-questions, and the prompt must answer all three or say why it could not:

1. **What does the current grid actually deliver?** Interpolation error of the consumer's
   $\varphi$ spline as a function of local sample spacing, per decade, per model, at all three
   reference wavenumbers. Where is the grid over-sampled and where is it under-sampled? A uniform
   grid over a non-uniform curvature is wrong in *both* directions, and the cheaper half of the
   answer is usually the over-sampling.
2. **What criterion would a non-uniform grid use?** State it as something computable **before** the
   grid is built — from $H(z)$, $c_s^2(z)$ and $k$, which are all available at that point — and
   show that it predicts the measured error. A criterion that needs $\varphi$ to already exist is
   not usable by `populate_z_sample`.
3. **What would it cost or save?** Sample count at fixed accuracy, and accuracy at fixed sample
   count, for at least two candidate criteria. Every stored object is keyed on the grid, so a
   change here is a full regeneration; the user needs the trade in numbers.

## 2. Two constraints the answer has to respect

- **The response grid is a decimation of the source grid** and must stay a subset of it
  (`main.py`'s own comment, and prompt 11's invariant). A criterion that produces a source grid no
  useful response grid can be winnowed from is not an answer.
- **`[02-consumer-phi-below-the-storage-granularity]` is the floor at large $k$**, not the spacing.
  At $k=3\times10^8$ the recovered $\varphi$ spans **2.0 ulp** of the stored phase — three distinct
  values over 1,377 samples. **No grid density improves that**, and a criterion derived from a
  measurement at that wavenumber will be measuring rounding. Derive the criterion where $\varphi$
  has dynamic range, and say explicitly where it stops meaning anything.

## 3. What to record, whatever the answer

The audit §9 lists two things it did not examine and that this prompt must not lose:

- **The response grid and the $k\tau$ oscillation** an $\Omega_{\rm GW}$ post-processing step would
  need to resolve to fit an RMS amplitude. Not yet designed; write down what would be required, so
  that a future grid decision is taken with that consumer in view rather than after it.
- **The standing caveat that no verification run has reached production $x$**
  (`docs/OPEN_ISSUES.md` §5). Any density criterion validated at $x\sim5\times10^5$ carries that
  ceiling; say so where the recommendation is stated, not in a footnote.

## 4. Deliverable

A dated section in `docs/qcd-background-verification.md`, and a §3 issue on this campaign's board
carrying the recommendation, its evidence and its cost, so that it survives the campaign's close.
**The Result is `COMPLETE` when the measurement is made and the recommendation is stated** — not
when a grid changes. If the measurement says the current uniform grid is adequate, that is a
perfectly good answer and must be stated as plainly as any other.

**Stop and report to the user.** Do not proceed to implement a criterion, and do not open a
follow-up prompt; the campaign's remaining scope is the user's to set.

## 5. Log and commit

Log to `logs/12-grid-density-criterion.md` per README §5.1. Its "State handed to the next prompt" is
the decision packet: the criterion, the numbers, the regeneration cost, and what is still unmeasured.

Commit subject, or something equally specific:
`Measure what the source grid's density buys and what it wastes`
