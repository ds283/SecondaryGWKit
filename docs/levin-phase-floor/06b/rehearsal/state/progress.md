# Rehearsal agent progress (levin-phase-floor 06b §4)

- Turn 1, 2026-10-28 13:52–13:54Z: gate re-checked (out/00..02), lock acquired (state/lock.log),
  R15 launched from P at da5c24d: run levin-phase-floor-06b-r15-parent-20261028T135349, driver pid
  96498, sh pid 96490, store var/datastores/levin-phase-floor-06b-r15*.sqlite; set-pid 96498; one
  check out/r15_launch_check.txt. Turn ended; awaiting R15's final status.json.

## Deviations so far (numbered; classified)

1. Lock transcript at `state/lock.log` (06's was `out/lock.log`): the brief names `state/` for lock
   transcripts. IMPLEMENTATION CHOICE.
2. A `ray-processes` look before `acquire` (not in the protocol's list). Read-only.
   IMPLEMENTATION CHOICE.
3. The driver ran with both `PYTHONDONTWRITEBYTECODE=1` and `-B` (R13 used the environment
   variable alone); the brief asks for either. No effect on values. IMPLEMENTATION CHOICE.
4. Read-only checks are recorded through `scripts/rec.sh` (command, cwd, times, output, exit code
   into `out/NAME.txt`). IMPLEMENTATION CHOICE.
- Turn 2, 13:57Z: R15 done, exit 0, fingerprint 1b332ffa (classes equal R8's), out/r15_end_check.txt;
  no Ray processes left; R15 store hashed (state/r15_store_sha256_after_build.txt). R16 launched
  from C at 92bc62f: run levin-phase-floor-06b-r16-premise-20261028T135745, driver 97360, sh 97356;
  set-pid 97360; one check out/r16_launch_check.txt. Turn ended; awaiting R16's final status.json.
- Turn 3, 14:01–14:12Z: R16 done, exit 0, 1b332ffa (classes equal R8's and R15's). Comparison
  (exit 1, QSI only), classify_non_qsi (0 unclassified), per-row QSI tabulation (attempt 1 failed
  on identity columns; attempt 2: 782 non-phase-limited rows bit-identical, 82 tabulated), readers
  exit 0 with R16 byte-identical, lock released, records written.
