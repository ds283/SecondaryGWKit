# Implementation logs

One log per prompt, written by the agent that executed it and committed alongside its code changes.

Format and the mandatory deviation classification are in [`../README.md`](../README.md) §5.1.

The short version: a log must be good enough that a later reader can tell **what shipped** and **why
it differs from the prompt** without re-deriving anything from the code. Every difference is one of

- **structurally required** — the prompt could not be implemented as written;
- **an implementation choice** — the prompt left it open, with enough justification that a later
  reader can disagree on the merits without redoing the analysis;
- **unintended drift** — noticed after the fact.

Prompts that change numerics additionally carry a *Numerical evidence* section
([`../README.md`](../README.md) §5 rule 9): a before/after comparison against a closed form, with
actual numbers. "The existing tests still pass" is not evidence — those tests assert to `1e-10` on a
module that delivers `1e-16`.

| Log | Prompt |
|---|---|
| `01-refuse-or-report.md` | [01](../01-refuse-or-report.md) |
| `02-complexified-solve.md` | [02](../02-complexified-solve.md) |
| `03-total-variation-gate.md` | [03](../03-total-variation-gate.md) |
| `04-roundoff-floor.md` | [04](../04-roundoff-floor.md) |
| `05-global-tolerance.md` | [05](../05-global-tolerance.md) |
| `06-mode-filter.md` | [06](../06-mode-filter.md) |
| `07-diagnostics-hygiene.md` | [07](../07-diagnostics-hygiene.md) |
| `08-order-and-sampling.md` | [08](../08-order-and-sampling.md) |
| `09-caller-propagation.md` | [09](../09-caller-propagation.md) |
| `10-test-matrix.md` | [10](../10-test-matrix.md) |
