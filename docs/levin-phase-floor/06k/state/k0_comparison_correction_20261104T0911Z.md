# levin-phase-floor 06k — a correction to K0's comparison (section 3 of the rules file), before any K0 unit

Written at 2026-11-04T09:11Z, after one development unit outside K0's 100 (`g-1e-7-2-1449`, run alone
under the identity variant, `out/cmd/14`, compared by `out/cmd/15`) and before any K0 unit was run.

**The correction.** The key `cache_built` (the unit record's list of what the worker's producer cache
built while that unit ran) is added to the excluded keys. It is a property of the worker's history,
not of the unit's computation: the first unit a worker runs for a k builds that k's producers, and
later units find them built. A K0 worker runs a subset of a census worker's units, so its first unit
records the builds that 06g recorded on its own worker's first unit. On the development unit
`cache_built` is the only key, outside the timing and provenance keys, that differs (4 builds against
0). It is still compared and reported, as every excluded key is, and never counted against the gate.

The rules file itself is not edited.
