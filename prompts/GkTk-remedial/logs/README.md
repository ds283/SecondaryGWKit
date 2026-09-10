# Logs — Gk/Tk WKB phase remedial campaign

One log per prompt, named `NN-<name>.md` matching the prompt file, committed **in that prompt's own
commit**. The template is [`../README.md`](../README.md) §5.1 and it is mandatory.

| # | Prompt | Log |
|---|---|---|
| 01 | [Reference harness and prototype](../01-reference-harness-and-prototype.md) | `01-reference-harness-and-prototype.md` |
| 02 | [QCD residual convergence](../02-qcd-residual-convergence.md) | `02-qcd-residual-convergence.md` |
| 03 | [τ primitive](../03-tau-primitive.md) | `03-tau-primitive.md` |
| 04 | [Sound-horizon and friction tables](../04-sound-horizon-and-friction-tables.md) | `04-sound-horizon-and-friction-tables.md` |
| 05 | [Phase residual](../05-phase-residual.md) | `05-phase-residual.md` |
| 06 | [Gk WKB phase from the primitive](../06-gk-wkb-phase-from-primitive.md) | `06-gk-wkb-phase-from-primitive.md` |
| 07 | [Tk WKB phase from the primitive](../07-tk-wkb-phase-from-primitive.md) | `07-tk-wkb-phase-from-primitive.md` |
| 08 | [`phase_spline` de-chunk](../08-phase-spline-dechunk.md) | `08-phase-spline-dechunk.md` |
| 09 | [Gk consumer on `PrimitivePhase`](../09-gk-consumer-primitive-phase.md) | `09-gk-consumer-primitive-phase.md` |
| 10 | [Tk consumer on the tables](../10-tk-consumer-primitive-phase.md) | `10-tk-consumer-primitive-phase.md` |
| 11 | [Numeric diagnostics and units](../11-numeric-diagnostics-and-units.md) | `11-numeric-diagnostics-and-units.md` |
| 12 | [Tk numeric `atol`](../12-tk-numeric-atol.md) | `12-tk-numeric-atol.md` |
| 13 | [Verification and docs](../13-verification-and-docs.md) | `13-verification-and-docs.md` |

## What a log is for

A later agent or human must be able to tell **what shipped and why it differs from the prompt**
without reconciling against the code. The diff is in git; what the diff cannot say is *why*. Every
difference between the prompt and the shipped code is classified:

- **STRUCTURALLY REQUIRED** — the prompt could not be implemented as written. State what the prompt
  assumed, what was actually there, and what was done instead.
- **IMPLEMENTATION CHOICE** — the prompt left it open and you picked. Give the alternatives and the
  reason, in enough detail that a later reader can **disagree on the merits without re-doing the
  analysis**. Prompts 03, 06, 08, 09 and 11 each leave real choices open and name them.
- **UNINTENDED DRIFT** — noticed after the fact, not deliberate. Say so, and say whether it was
  reverted or kept. Kept drift is an orchestrator stop condition; do not use this tag to launder a
  choice you would rather not justify.

"None" is an acceptable and expected answer for the small prompts.

## Campaign-specific requirements

1. **Quote numbers, not verdicts.** Every acceptance threshold gets its measured value and every
   maximum its (model, $k$, $z$). "Passes" is not a verification record;
   `phase error 3.1e-6 rad at LambdaCDM, k=1e5, z=0.1` is.
2. **"State handed to the next prompt" is verbatim.** Later prompts are programmed against the
   names, signatures, column names, solver labels and Gauss orders earlier logs settle.
3. **Three kinds of verification, never blurred:** "I ran this and it printed X"; "I reasoned that
   this is correct"; "this needs a run the user must do".
4. **Two statements must never appear** in a log, a docstring or a commit message: that a target
   below a floor of `IMPLEMENTATION_STATE.md` §5 note 2 was "achieved"; and that the campaign's
   motivation is the per-object cost alone — the cost saving is real (64 s → milliseconds) but the
   finding is that the stored phase was wrong by tens to thousands of cycles.
5. **Observations not acted on are part of the deliverable.** This campaign deliberately leaves
   the hand-over, the LG order, per-region anchoring, the initial-condition series and the
   `transfer-remedial` files alone. Record what you notice; open a §3 issue and the
   `docs/OPEN_ISSUES.md` row; do not fix it.
