# Issues found by the rehearsal agent, levin-phase-floor 06e §4 (Mac Studio)

These are proposals for the orchestrator. The board and `docs/OPEN_ISSUES.md` are not edited here.

## Proposed

None. §4 holds in every clause, and Z1 holds with no violation.

## Observed, no issue proposed

- **Z1's "outside 06a's 720 stationary-point rows" excludes nothing on R8's rehearsal scope.** No
  rehearsal row is an A3 v2 row: only $k=10^5$ is shared, and none of the 96 response redshifts.
  Any rehearsal-scope rule that excludes "06a's stationary-point rows" is therefore scored on every
  row. A future prompt that wants the exclusion at rehearsal scope needs its own scan of the
  rehearsal store, or must name the rows another way. (Log, deviation 4 and Observations 1.)
- **06c's twelve large Y1 disagreements stay where 06c put them.** These are the equilateral
  `G+q+r` calls, where R17 (06b) and R18 (06c + 06d) disagreed by up to 16,763 times their summed
  declared errors. 06e moves R18's totals by at most $1.8\times10^{-6}$ of the summed errors there.
  Whatever 06c's records opened for them is neither answered nor changed by §4.
- **`python -m runregistrykit list | head -3`** raised `BrokenPipeError` with a traceback once
  `head` had closed the pipe (`out/final_check.txt`). The listing is otherwise correct. This is
  RunRegistryKit's CLI, not this repository's code. It is cosmetic, and recorded only in case the
  orchestrator wants it reported upstream.
