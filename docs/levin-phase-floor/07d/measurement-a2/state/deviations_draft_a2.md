(draft, folded into log.md at the stop)

6. The new band's first draw rested on a misreading (UNINTENDED DRIFT).
   - The misreading: I told the orchestrator that no returned leaf has V ≥ 20π. In fact Part A's population.json leaves out
     out-of-band leaves by construction.
   - The fix: the band was redrawn from the returned leaves (section 2.4's construction), as band_20_32. The approved
     "visited, not returned" set was kept as band_20_32v.
   - The first, mixed draw (state/superseded/, out/leaves_a2/superseded/) was superseded before any solve.
7. The first reproduction compared a field on a different convention (UNINTENDED DRIFT).
   - What differed: 80 of 488 leaves, on the "rule" field only. I6 records a rule only where it refuses; the reproduction
     recorded the rule it applied.
   - The fix: the reproduction records `rule` only where refused, and `rule_applied` beside it. After that, 488 of 488
     are equal. v1 is kept.
8. CC₉₃ is computed by np.dot(w₉₃, g₉₃) in item 1 and in V93; the largest |dot − fsum|/|CC₉₃| is recorded.
   IMPLEMENTATION CHOICE, approved (go-ahead, item 2).
9. Tables 1–6 score every leaf of Part A's that this Part measured, not only Part A's scored prefix. IMPLEMENTATION CHOICE.
10. Table 3 scored each CC₉₃ candidate with the 47-point floor, max(E, floor₄₇) + κ s₉₃ (tables_a2.py decls93), where
    V93 declares with its floor and node term on the 93-point grid (go-ahead, item 2). UNINTENDED DRIFT.
    - The check: decl93_a2.py scores the variants' own declarations on v4.
11. Item 4 measures the jumps on f₀, f₁ and θ_raw as the integrand receives them, not per table ingredient. IMPLEMENTATION
    CHOICE.
12. V93tail (tail93) is an agent's choice: the second built V93 declaration beside pair93. IMPLEMENTATION CHOICE.
13. A seventh process ran briefly (UNINTENDED DRIFT, as Part A's deviation 11).
    - While batch 1 held six workers, two reader commands ran, each about a second:
      - one read Part A's census index's first row;
      - one read item 4's jump records, for a first look at the nearest-node midpoint jumps.
    - Neither wrote anything.
    - Earlier, while the leaf run held six workers, only `ps` and `ls` ran, not Python.
14. The patch to make_variants_a2.py was truncated by a nested triple quote, and repaired before any build. It had no
    effect on any variant (each diff is saved). UNINTENDED DRIFT.
15. G's basis text was first written with unchecked figures. It was recomputed and corrected before G was built.
    UNINTENDED DRIFT.
16. (K), (KT) and (G) take a call's knots from leaf_a2.typed_knots (production_edges' walk), supplied by the harness. On
    analytic calls the walk raises and no knots are supplied, so those calls bisect and gate as the tree does.
    STRUCTURALLY REQUIRED.
