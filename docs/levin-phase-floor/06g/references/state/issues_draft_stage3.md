# Issues from the references agent (06g Part B §5) — draft, for the board's §3 and docs/OPEN_ISSUES.md

## Proposed: `[06g-rerun-backstop-projection-ignores-chunks-in-flight]`

**Found by:** the references agent, 2026-11-03, after the rerun batch
(`levin-phase-floor-06g-references-rerun-20261102T185324`), at the coordinator's request.

**What:** the batch's backstop (`var/levin-phase-floor-06g/references/scripts/run_rerun.py`, a
harness script, not production code) projected a reading's total as `elapsed × total / done`, with
`done` the chunks back out of order. At `done` = 1 this ignores the other five chunks in flight on
the six workers, folds the setup time in, and treats chunks as equal, while a Route B QAWO chunk's
cost follows its phase span, which varies by orders of magnitude. All three stops fired at the first
chunk back: r47 qawo r2 (453,016 s projected), r53 qawo r2 (290,165 s), r26 qawo r1 (2,296,531 s).

**Measured** (`out/stage3_projection.json`): a six-worker schedule of chunk costs proportional to
their phase span, calibrated on the first chunk back, reproduces the monitor's chunk counts for the
two stage-3 readings that finished (r24, r53 qawo r1) at every check, and projects r47 r2 3.0 h, r53
r2 8.9 h, r26 r1 13.7 h: none a day or more. The three references stay as they are (r47 three
readings, r53 three, r26 FCC-only).

**Impact:** three readings stopped that the user's rule ("stopped only at a projected day or more")
would have let finish; no scored verdict depends on them (the stage-3 recheck: none changes).

**Next step:** a phase-span-weighted projection in any later backstop, or none until a few chunks
per worker are back; whether to rerun the three readings is for the user.
