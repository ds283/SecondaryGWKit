# levin-phase-floor 05 §6 — the predictions, fixed before the run

Copied unchanged from prompt 05 §6 (`prompts/levin-phase-floor/05-reanchor-g-phase.md` lines 292–315 at
`53084dc`, by `git show 53084dcb:prompts/levin-phase-floor/05-reanchor-g-phase.md | sed -n 292,315p`;
the same bytes are `state/prompt_section6_predictions.raw`). Written after gate L0 held
(`out/L0.txt`) and before the run is launched. The text between the rules below is the prompt's.

---

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "Scored
group" is 03's definition: at least 10 regions at depth 18 or more, here under $p$.
- **L1 — the cap goes where $G$'s anchor was the floor.**
  - **The rule:** on every depth-20 row whose top group is `G+r`, `G−r` or the lone `G`, under $c$
    at `1e-7`, every group under $p$'s scored set has fewer than 10 regions at depth ≥ 18, and the
    row's regions are at most a tenth of $p$'s.
  - **Holds** if true on every such row; an over-budget unit counts as false.
- **L1q — the same in the `Gqr` rows.**
  - **The rule:** L1's rule, on the `Gqr`-topped rows.
  - **Holds** if true on every such row. A failure is evidence that $T$'s phase carries a floor
    that matters there, which is the case for re-anchoring $T$. Report the depth-18 regions'
    $z'$ against $z_{\rm response}$ on any row that fails.
- **L2 — the controls do not move beyond their error.**
  - **The rule:** on every control row, $|{\rm total}_c - {\rm total}_p| \le$ `total_abserr`$_p$.
- **L3 — the declared error is honest after the change.**
  - **The rule:** on every depth-20 row with both $c$ units in budget,
    $|{\rm total}_c(10^{-7}) - {\rm total}_c(10^{-8})| \le$ `total_abserr`$_c(10^{-7})$.

**Descriptive, not scored:**
- $|{\rm total}_c - {\rm total}_p| /$ `total_abserr`$_p$ on the depth-20 rows. This is 04's N4,
  wider, and now with the modulus correction in it too;
- cost and regions by class and band;
- `total_converged` before and after;
- the low band against the mid band.
