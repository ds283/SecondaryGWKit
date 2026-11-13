# levin-phase-floor 07a §6 — the user's ruling at STOP 2 (relayed by the laptop orchestrator), recorded 2026-11-12

The user, verbatim: "Yes, A for Q1 and call's outputs for Q2; launch the census"

The orchestrator's reading, as relayed:
- Q1, reading A: every solve counts as "a region whose route or LU choice differs" in W1 and W7, the trust check's
  auxiliary solves (quarters, comparison halves) included. W1 covers only `untouched` calls; W7 excludes any call with an
  auxiliary or own solve whose route or LU choice differs. The `aux_only` calls are tabulated as descriptive: their abserr
  moves (distribution; how many shrink and by how much), their value moves, any `converged` changes, and their honesty
  where referenced.
- Q2, the call's outputs: "equals 07's call bit for bit" compares the call's own outputs (value, abserr, converged, every
  returned region with its value and error). Differences in the `location` category (the skipped-phase record's form,
  newly reported unlocated minima with no point located and no window) and in the `harness` category (after-call re-solves
  under h's routing) are set aside like phase_span; the count per category and the strict comparison are reported beside
  it. W1 holds if no untouched call differs in the `call` category. A `location` difference that comes with a located point
  or a window is not set aside: such a call is not untouched.
- Stop 3's rule applies under these readings: an untouched call that differs in the `call` category is a stop.
- W1 and W7 are still scored under both readings, as proposed; the verdict is the one above.
