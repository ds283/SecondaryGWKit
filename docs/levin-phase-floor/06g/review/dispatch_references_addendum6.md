## Addendum 6 to the references agent's brief — rescoring with the laptop's FCC readings at `b5626ecb` (2026-11-03)

**The source.** The user ruled that FCC near a stationary point is "worth fixing", and that V1's
FCC readings be re-run at both resolutions "because we don't trust them". The fix is commit
`b5626ecb`: FCC integrates directly any segment whose slope ratio is above 2, and QAWO is
unchanged. On the user's word ("Can you run the FCC batch here, leaving Stage 3 still running on the
Mac Studio?") the laptop ran the batch: run `levin-phase-floor-06g-fcc-laptop-20261102T233229`,
done, exit 0, 32 of 32.

**The files** are in `references/out/readings_fcc_b5626ec_laptop/`.

**The orchestrator's checks** (`review/checks/fcc_laptop_check.txt`):
- all 32 SHA-256 entries verify;
- the tool's SHA-256 is `d37e633b…`, equal to `SecondaryGWKit-lpf06g-b5626ec/tools/filon_reference.py`;
- every reading's `pickle_sha256` equals the Mac Studio's readings of the same sample;
- the four checks (w06 qawo r1 from the rerun batch; r55 fcc r1, fcc r2 and qawo r1 from part 2)
  are bit-identical on `value_hex`, `value_30` and `imag_30`;
- in every one of the 28 FCC readings, the non-monotone segment is integrated directly
  (`direct` True). Every direct segment is either non-monotone or has slope ratio > 2.

### The usable rule, extended (clause iii)

A Route B **FCC** reading made by the tool at `b5626ecb` is usable if every segment with `monotone`
False has `direct` True. This is a third clause, beside (i) all_monotone and (ii) `ff3e8394` with
`direct_segments` equal to the non-monotone count. State the clause per reading.

### What is superseded

The stage-2 FCC readings of the 14 stationary-point samples (w00–w11, z00, z12; the
`*-fcc-r*-rerun.json` files) are **superseded** by the laptop's. They stay on disk, unscored, with
that reason recorded. Those samples' stage-2 QAWO readings (`*-qawo-r*-rerun.json`, tool
`ff3e8394`) stand: QAWO's code is unchanged at `b5626ecb`.

### The rescoring

Make it an **addition** beside the stage-2 scores, in its own timestamped section and files. Do not
rewrite the stage-2 scores. Cover:
- **V1's table:**
  - R23 and R24 against these references, with each code's |v − ref|/abserr;
  - which is nearer;
  - whether R24's declared error covers its distance, on all 12.
- **V3:** with z00 and z12, both by the rule as written and as the ruling scores it.
  - As written, the predictions require every Route B reading to be all_monotone, so z00 and z12
    still cannot count. Say so, and give the verdict.
  - Beside it, give the verdict if clauses (ii) and (iii) are accepted in place of all_monotone, so
    the user sees both readings of the rule.
- **The count rule's verdict for each of the 14:** whether the four readings agree within 1/100 of
  the minimum abserr, with the spread.
- Everything else from the stage-2 scoring, unchanged except where these readings enter.

### The open check: are z00's and z12's captured integrands equal?

The captured pickles differ (`1b870b17…` and `7b8792e5…`). Compare the two calls' groups by the
criterion you used for 06i's 13:
- coefficients, orders, signs and spans;
- each factor's spline and table bytes;
- the integrand the tool reads, at 65 points, plus K, C, the breaks and the default edges.

Report every field that differs, if any. The orchestrator will spot-check your result.

**Process limits.** At most one process: stage 3 holds six cores. Do not touch the running batch.
**Stop and ask** only if something here cannot be done as described.
