# Logs — the QCD background campaign

One log per prompt, named `NN-<name>.md` matching the prompt file, committed **in that prompt's own
commit**. The template is [`../README.md`](../README.md) §5.1 and it is mandatory: every deviation
is classified `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`, and every
acceptance threshold gets its measured value with the (model, $k$, $z$) of the maximum.

Two campaign-specific additions to the template, because a later reader cannot recover either from
the diff:

- **`T_Z_REPRESENTATION_VERSION` before and after**, in "What shipped". It is the only thing that
  tells a datastore its QCD rows are stale (README §2 (d)).
- **The exact command that regenerates the QCD reference fixture**, in "State handed to the next
  prompt", together with the largest relative move per key that this prompt caused (README §2 (e)).

| # | Prompt | Workstream | Log |
|---|---|---|---|
| 01 | [The background-against-background harness](../01-background-reference-harness.md) | A | `01-background-reference-harness.md` |
| 02 | [Make the QCD reference fixture regenerable](../02-regenerable-qcd-references.md) | A | `02-regenerable-qcd-references.md` |
| 03 | [Key the `T(z)` representation](../03-key-the-representation.md) | A | `03-key-the-representation.md` |
| 04 | [Tighten `_solve_T_z`](../04-tighten-node-solve.md) | A | `04-tighten-node-solve.md` |
| 05 | [Spline the entropy factor](../05-entropy-factor-representation.md) | A | `05-entropy-factor-representation.md` |
| 06 | [Segment at the jumps](../06-segment-at-the-jumps.md) | A | `06-segment-at-the-jumps.md` |
| 07 | [Re-derive `integration_break_points`](../07-rederive-break-points.md) | B | `07-rederive-break-points.md` |
| 08 | [Re-measure the per-sector policy](../08-per-sector-policy-remeasure.md) | B | `08-per-sector-policy-remeasure.md` |
| 09 | [Close-out verification](../09-close-out-verification.md) | C | `09-close-out-verification.md` |
| 10 | [`PrimitivePhase` on the 3-point break set](../10-primitive-phase-break-points.md) | D | `10-primitive-phase-break-points.md` |
| 11 | [A cosmology-aware source grid](../11-cosmology-aware-source-grid.md) | D | `11-cosmology-aware-source-grid.md` |
| 12 | [A measured grid-density criterion](../12-grid-density-criterion.md) | D | `12-grid-density-criterion.md` |
