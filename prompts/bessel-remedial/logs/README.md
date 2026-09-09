# Logs — Bessel amplitude and phase campaign

One log per prompt, named `NN-<name>.md` matching the prompt file, committed **in that prompt's own
commit**. The template is [`../README.md`](../README.md) §5.1 and it is mandatory.

| # | Prompt | Log |
|---|---|---|
| 01 | [Reference harness](../01-reference-harness.md) | `01-reference-harness.md` |
| 02 | [SciPy domain boundaries](../02-domain-boundary-tests.md) | `02-domain-boundary-tests.md` |
| 03 | [Closed-form tail](../03-closed-form-tail.md) | `03-closed-form-tail.md` |
| 04 | [Near-region sampler](../04-near-region-sampler.md) | `04-near-region-sampler.md` |
| 05 | [Two-region construction](../05-two-region-construction.md) | `05-two-region-construction.md` |
| 06 | [Evaluation and compatibility](../06-evaluation-and-compatibility.md) | `06-evaluation-and-compatibility.md` |
| 07 | [Bessel phase groups](../07-bessel-phase-groups.md) | `07-bessel-phase-groups.md` |
| 08 | [Fixture revalidation](../08-fixture-revalidation.md) | `08-fixture-revalidation.md` |
| 09 | [Benchmark and docs](../09-benchmark-and-docs.md) | `09-benchmark-and-docs.md` |

## What a log is for

A later agent or human must be able to tell **what shipped and why it differs from the prompt**
without reconciling against the code. That is the standard, and it is higher than "a summary of the
diff" — the diff is already in git. What the diff cannot say is *why*.

So every difference between the prompt and the shipped code must be classified:

- **STRUCTURALLY REQUIRED** — the prompt could not be implemented as written. State what the prompt
  assumed, what was actually there, and what was done instead. A name that differed, an ordering
  constraint, a numerical fact that turned out otherwise.
- **IMPLEMENTATION CHOICE** — the prompt left it open and you picked. Give the alternatives
  considered and the reason, in enough detail that a later reader can **disagree on the merits
  without re-doing the analysis**. This is the tag that carries the most weight in this campaign:
  prompts 04, 05 and 06 each leave real choices open.
- **UNINTENDED DRIFT** — noticed after the fact, not deliberate. Say so plainly, and say whether it
  was reverted or kept. Kept drift is a stop condition for the orchestrator
  ([`../README.md`](../README.md) §4.3), so do not use this tag to launder a choice you would rather
  not justify.

"None" is an acceptable and expected answer for the small prompts.

## Campaign-specific requirements

Beyond the template:

1. **Quote numbers, not verdicts.** Every acceptance threshold in the prompt gets its measured
   value, and every maximum gets the \((\nu,x)\) at which it occurred. "Passes" is not a
   verification record; `E_theta = 3.04e-14 at nu=3/2, x=1.9e1` is.
2. **"State handed to the next prompt" must be verbatim.** Prompts 03 through 09 are programmed
   against names, signatures and field lists that earlier logs settle. A log that says "as in the
   prompt" fails review, because the prompt is a specification and the log is the record of what was
   actually built.
3. **Distinguish three kinds of verification**: "I ran this and it printed X", "I reasoned that this
   is correct", and "this needs a run the user must do". Do not blur them. The campaign's
   credibility rests on this distinction being observed — see
   [`../README.md`](../README.md) §6 on estimators versus supremum bounds.
4. **Two claims must never appear**, in a log any more than in the code
   ([`../IMPLEMENTATION_STATE.md`](../IMPLEMENTATION_STATE.md) §5 notes 2 and 3):
   - that the residual "never exceeds a cycle" — false above \(\nu\approx630\);
   - that the campaign's motivation is construction speed — the old build is ~0.1 s across the whole
     production range; what it removes is a hard cliff.
5. **Observations not acted on are part of the deliverable.** This campaign deliberately leaves
   several real problems alone (`../README.md` §1.1, §7): `phase_spline`'s chunking and its
   progress-guard bug, `QuadSourceIntegral._three_bessel_Levin`, the stale `AdaptiveLevin`
   docstring, the high-order accuracy target. If you notice another, record it here rather than
   fixing it — scope creep destroys the revert-per-prompt property, which is the campaign's only
   rollback mechanism.
