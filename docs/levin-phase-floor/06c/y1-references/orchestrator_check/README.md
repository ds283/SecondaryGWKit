# Orchestrator's re-check of the Y1 references

- **Host:** mps026298
- **Time:** 2026-10-29, 21:37–21:46Z
- **Trees:**
  - `e5764a6` (R18's code), clean;
  - `fef401c` (R17's code), clean;
  - the main checkout, `28ebaf0`, clean.
- **Script:** `orch_ref.py`, sha256 `8a0d291d09d74d444b47ff1f34b0f9000e4a045c4b94c36e5dec3158f2ea863b`.
  It reads only.

This quadrature does not share code with the agent's `refquad.py`, and it is not the Levin driver.

- **Panels.** I tabulated the raw phase θ, not θ′, on uniform 1e-3 cells, and cut each cell into
  ceil(|Δθ|/128) equal sub-panels.
- **Nodes.** I used numpy's `leggauss` at 64 and at 96 nodes per sub-panel.
- **Summation.** Each chunk is summed with `fsum`, and the chunk sums are summed with `fsum` again.

## The outputs

**`01-Y01-GqR-R18tree.txt`: Y01's G+q+r call, where R18 is the wrong code.**
- **Replay:** reproduces the capture. Scaled, it gives 8.89521217897e-18, which matches the rehearsal's own tabulation.
- **The reference:** S96 = `0x1.035b115081929p-69`, with |S96 − S64| = 5.0e-33.
- **Against the agent's reference:** they differ by 1.1e-4 of min(abserr).
- **R18:** misses by 2.44e5 of its abserr.
- **Stationary point:** θ has one extremum between cells, at x = 9.23159, where θ′ runs from −372 to +610 across ±1e-3.

**`02-R17tree-replays.txt`: the R17 pickles of Y01 and Y08, replayed at `fef401c`.**
- Both reproduce the capture.
- Scaled, the values and abserrs match the rehearsal's tabulation.
- Y01's R17 call misses by 2.27 of its abserr.

**`03-Y08-GqR-R18tree.txt`: Y08's G+q+r call, where R17 is the wrong code.**
- **The reference:** S96 = `0x1.da71eaa6b71a4p-70`, over 287,242 sub-panels and 3.5e7 rad, with |S96 − S64| = 6.8e-33.
- **Against the agent's reference:** they differ by 8.3e-5 of min(abserr).
- **The codes:** R17 misses by 1.62e6 of its abserr; R18 by 0.283.
- **Stationary point:** at x = 9.23141.

**`04-verdicts-recomputed.txt`: all 30 rows' verdicts, re-derived from `results.json`.**
- The arithmetic is exact, in rationals.
- Every verdict equals the agent's.
- T17 − T18 is matched by the moved calls' deltas to at most 2.3e-9 of the row errors, so no moved call is missing.

**Stores.** All ten originals re-hashed at 21:38Z are equal to `out/00`.
