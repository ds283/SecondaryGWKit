## 2026-11-02 ~19:00Z — the rerun batch (relayed by the laptop orchestrator)

- **r24 qawo r2 is not added.** r24's two FCC readings agree to 2.1e-14 relative and are monotone.
  It is added only if r24's QAWO r1, which runs first in stage 3, disagrees with FCC beyond the
  agreement tolerance; that is reported, and nothing is added without asking.
  The orchestrator set a background check on `r24-qawo-r1-rerun.json`.
- **Stage 3 runs as planned.** The user's words: "leave Stage 3 to run as planned, and we can
  curtail it if the machine is needed sooner."
- **Scoring:** after stage 2 or after the whole batch is still open. Nothing is scored until the
  user rules.

## 2026-11-02 — when to score (relayed by the laptop orchestrator)

**The ruling:** score V1, V3 and V4, and the V8 sample's table, once stage 2 of the rerun batch ends.
The user's words: "we can use the result to inform our view on whether we should let Stage 3 run to
completion. We should then check whether anything in stage 3 would change those scorings, after it
lands."

**The stage-2 scoring:**
- each reference is scored with the readings it has at that point;
- r24, r26 and r53 score as FCC-only, and r47 with three readings, each marked as such;
- the scores are reported to the laptop before stage 3 goes much further.

**When stage 3 lands:** r24, r26, r47 and r53 are rechecked. Any change is recorded as an addition
beside the stage-2 scores, which are not rewritten.

## 2026-11-02 22:23Z — r24 qawo r1 (rerun) landed

- **The reading:** 0x1.3a013cf6f38aep-92 in 6,098 s, against an estimate of 6,402–10,669 s,
  all_monotone True.
- **Against FCC:** it agrees with FCC r1 to 4.2e-6 of the tolerance and with FCC r2 to 5.6e-5.
- **Per the ruling,** r24 qawo r2 is therefore **not** added. The check is in
  `checks/r24_qawo_r1_vs_fcc.txt`.

## 2026-11-02 ~23:35Z — the FCC fix at b5626ecb is run on the laptop

**The user's words:** "This machine is quiet for now. Can you run the FCC batch here, leaving Stage 3
still running on the Mac Studio?"

**On the Mac Studio:**
- stage 3 is not stopped;
- no `references-fcc` batch is launched here.

**The laptop's run:** `levin-phase-floor-06g-fcc-laptop-20261102T233229`, 32 units, tool at
`b5626ecb`.
- **Checks first:** four bit-for-bit re-reads (w06 qawo r1 from the rerun batch; r55 fcc r1, fcc r2
  and qawo r1 from part 2), which also test cross-host identity.
- **Then:** FCC r1 and r2 on w00–w11, z00 and z12.

**When it ends:** the laptop copies its readings to `references/out/readings_fcc_b5626ec_laptop/`.
They replace the stage-2 FCC readings of the 14 samples, which are recorded as superseded. Those
samples' stage-2 QAWO readings stand. The rescoring is an addition beside the stage-2 scores.
