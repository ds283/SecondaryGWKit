# 07b Part A: the agent's working plan (written 2026-11-13T14:10Z, before any reading)

Fixed before any reference reading; changes are logged as deviations.

1. **Gate** (done, `out/gate/`): code AST-equal, captures and scratch copy by hash, 07a checkpoint `461df2ec…`.
2. **§2.1 item 4** (done, `out/w2/w2_rescore.json`): 4 of 06b's 24 dishonest at 07a, all converged:
   `h-1e-7-0-12977` #5 (2.20), `h-1e-7-1-2174` #2 (1.36), `h-1e-7-2-14345` #3 (6.01), `h-1e-7-2-34473` #3 (1.39).
3. **Selections** (done before any reading): seeds `state/seeds_07b.json` (written first), `state/selection_07b.json`.
4. **Captures:** item 4's four and R2/R3 (not the pairs) captured afresh at h by `x1_capture_07b.py`, each matched
   to 07a's census by x1's own check; R1 from 07's cross-check captures, the pairs and the no-window calls from 07a
   Part A's, V4 from 07's inputs: every one re-checked by a replay at the tree against 07a's census record (value,
   abserr, regions, depth, converged, region arrays, section 1.5 entries). Each capture also replayed at the other
   rtol's (rtol, atol) against 07a's census there: a capture reproducing both is the integrand of both, and one
   reference scores both.
5. **§2.2:** for each chosen leaf, `leafprobe_07b.py` re-derives the trust check solve by solve (checked against the
   replay's leaf bit for bit) and solves the eighths and the M3 perturbations; `leafref_07b.py` reads Route B FCC r1
   and r2 on the leaf's span with every subdivision point as an edge, so one reading gives the leaf, its halves,
   quarters and eighths. M1's window variants (b4, b8) replayed, and their touching leaves' spans read in the same
   reading as edges. **Pilot first:** v08's carrying leaf.
6. **§2.4:** one reference per integrand, by 07's V4 method (`one_reading.py`: FCC r1, r2; QAWO r1, r2 where projected
   at 30 min or less), projected first with 07a's refitted model (Route B) and a pilot for Route A.
7. **§2.3 variants** written after §2.2's findings for (c), (d), (e), (f); (b4), (b8) now.
8. **Stop at §2.6.**
