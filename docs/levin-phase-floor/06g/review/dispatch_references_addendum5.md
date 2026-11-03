## Addendum 5 to the references agent's brief — the new tool at `ff3e8394`, and the rerun batch after part 2 (supersedes addendum 4's §2 and amends its §1)

**The source.** The user's ruling of 2026-11-02 on the stationary-point segments, relayed by the
laptop orchestrator: "I agree with your implementation (1) and (2), so please go ahead with that."

**The new tool.** Commit `ff3e8394fa3cf81a3d27087708f6fc8acbc3fb63` ("Integrate Route B's
stationary-point segments directly in the tool"). The laptop fetched it on this host and added the
detached worktree **`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-ff3e839`**. The orchestrator
checked that worktree: HEAD `ff3e8394`, clean with `--ignored`. `git diff --stat 1c5336e ff3e8394`
touches only `tools/filon_reference.py`, `tools/tests/test_filon_reference.py` and
`prompts/levin-phase-floor/IMPLEMENTATION_STATE.md`.

**What changed in the tool:**
- **Direct integration:** a Route B segment with a stationary point (`monotone` False) is
  integrated directly in y against the same smooth phase. FCC uses Gauss–Legendre in mpmath; QAWO
  uses QUADPACK's QAG in double.
- **Refusal:** both inversions refuse such a segment, and refuse dp/dy = 0, with a ValueError.
- **New fields:** readings report `tmap.direct_segments`. QAWO stats gain `direct_panels` and
  `ier_messages`, QUADPACK's flag messages counted by kind.
- **Monotone segments** run the old code unchanged.

### The batch: after part 2 has ended, one registered run

**How:** slug `references-rerun` (`--campaign levin-phase-floor --prompt 06g`), at most 6
processes, one reading at a time with 6 workers. **Every reading uses the tool from the
`ff3e839` worktree**, with the same harness on `PYTHONPATH` as each sample's original readings:
- g's census harness for the z, r and v samples;
- V1's rehearsal-store harness (`v1/scripts/`) for the w samples.

Record each reading's tool path and SHA-256.

**1. The stopped QAWO readings,** as addendum 4 §2 planned them:
- each estimated from its phase span (the sum of its FCC readings' `t_span`) at 0.06–0.10 µs/rad
  for r1 and about 4× that for r2;
- each listed in `out/rerun_plan.md` with its span, estimate, `all_monotone` and whether it runs.

The candidates and skips are:
- **Candidates:** r47 qawo r2; r26 qawo r1 and r2; r24 qawo r1 if under a day; r53 qawo r1 (part 2
  stopped it at 10:16Z) and r53 qawo r2 if part 2 stops it.
- **Skips:** r52 qawo r2; r24 qawo r2; anything estimated at a day or more; any call whose FCC and
  QAWO r1 already agree as tightly as r52's.

These calls are monotone, so the new tool reads them exactly as the old one would.

**2. All four readings of each of the 14 stationary-point samples,** 56 readings:
- the samples are w00–w11 (V1's 12), z00 and z12;
- these replace their old readings;
- the old readings stay on disk, unscored, with the reason recorded: "made by the tool at
  `1c5336e`, which inverted through a non-monotone segment; unsound by the tool's own rule".

z00 took about 37 minutes in all on the laptop, and the w samples should be cheaper.

**3. Bit-for-bit re-reads under the new tool,** before items 1 and 2:
- at least five clean readings from parts 1 and 2: two Route B QAWO, two Route B FCC and one
  Route A;
- choose cheap ones, say which before running them, and record why each is representative;
- each must reproduce its old `value_hex` (and imaginary part) **bit for bit**. A reading that does
  not reproduce is a stop.

**4. z00 across two hosts.** Compare z00's four new readings with the laptop's values:

| Reading | Laptop value | Laptop wall s |
|---|---|---|
| fcc r1 | 0x1.0c5f18f4cb423p-86 | 160 |
| fcc r2 | 0x1.0c5f18f86f178p-86 | 735 |
| qawo r1 | 0x1.0c5f18f86f93bp-86 | 309 |
| qawo r2 | 0x1.0c5f18f86f3e1p-86 | 1,023 |

Report whether each is bit-identical across the hosts. If one is not, report the difference. That
alone is not a stop.

**The backstop** is addendum 4's, the user's earlier ruling, not part 2's mechanism:
- no reading is stopped for passing 30 minutes or for a multiple of its estimate;
- progress and projected finish are logged every 5 minutes from 30 minutes, counted out of order
  (`imap_unordered` for progress only, results reassembled in chunk order) or else estimated from
  phase span;
- a reading is stopped only at a projected finish of **a day or more**, or 24 h run;
- if the progress wrapper changes, item 3's re-reads cover it, and the change is a numbered
  deviation.

### Scoring (amends addendum 4 §1)

**A Route B reading is usable** if either:
- `tmap.all_monotone` is True; or
- it was made by the tool at `ff3e8394` with `tmap.direct_segments` equal to its count of
  non-monotone segments.

Nothing else is scored as a reference. Addendum 4's hold is lifted for readings that meet this rule,
and only for those. State, for every scored Route B reading, which clause it meets.

**Once the new readings exist:**
- V1's table (addendum 1) is scored on them;
- z00 and z12 return to V3;
- every other rule of addenda 1–3 stands, including the scoring both ways for V3 and V4.

**The question V1's table answers stands.** For each of the 12 calls, which of R23 and R24 is nearer
the reference, and does that code's declared error cover the distance? Report the numbers per call;
interpretation stays with the laptop.

**Stop and ask** only if something here cannot be done as described.
