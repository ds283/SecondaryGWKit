# levin-phase-floor 06h — §5, the phase re-measurement (H1, H2): the phase agent's record

*Draft. The predictions below were copied in at 2026-11-01T03:34Z, before any number was computed (`state/predictions_20261101T033443Z.md`, SHA-256 `e3764cb6…`).*

## Predictions (copied verbatim from prompt §5 before the run)

- **H1 — the declaration is honest.** **Rule:** on every one of the 20 M2 calls, at every point
  sampled, $|\delta\theta(u)| \le \texttt{theta\_abserr}_h(u)$, the reference at the exact $e^u$.
  **Holds** if no point violates it. The parent's declaration is scored on the same points beside it.
- **H2 — the declared term covers the response.** **Rule:** on every returned leaf of the 20 calls,
  the first-order response to the measured phase error is at most that leaf's declared term.
  **Holds** if no leaf violates it.
