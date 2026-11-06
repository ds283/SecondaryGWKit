# Issues from 06m §4 (rehearsal agent)

**None to open.** No §7 stop condition fired. R1 holds on `total` and on the oracle. R25 reproduces
R24 in every value.

For the record, not opened:
- **05's `classify_non_qsi.py` treats a label difference as a `label` only when the two runs share
  a job name** (log, deviation 1, observation 3).
  - A comparison across prompts, here R24 (`levin-phase-floor-06g`) against R25
    (`levin-phase-floor-06m`), reports every label as UNCLASSIFIED.
  - The script is a record, kept byte-identical to 05's, and the workaround here leaves it
    unchanged.
  - If cross-prompt comparisons become routine, a later prompt may want a classifier that takes the
    job names. Nothing in this prompt depends on it.
- **The readers' unseeded `random.sample`** makes their outputs differ file for file between runs,
  as 06g recorded (log, observation 4). No store is affected.
