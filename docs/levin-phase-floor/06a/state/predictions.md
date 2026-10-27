## 4. Predictions

**Depth** means a Levin call's region depth, the largest over its regions. **Row depth** means the
largest over a row's calls.

- **C1 — at production's tolerance, nothing approaches the cap.**
  - **The rule:** at `rtol = 1e-7`, no row of the 9,280 has a region at depth ≥ 18.
  - **Holds** if true on every row. **Fails** if false on any row, or if any unit is still over
    budget after the pass. **Undecided** if neither, with a row that errored.
- **C2 — nor at `1e-8`.** C1's rule at `rtol = 1e-8`, scored the same way.
- **C3 — 05's sample was representative.**
  - **The rule:** at each rtol, no row's depth exceeds 11, the largest 06 found on 05's 39 rows
    at either tolerance.
  - **Holds** if true at both rtols. **Fails** otherwise.
  - C3 bears on how far 05's and 06's numbers stand for the store. It does not bear on L2 by
    itself.
- **C4 — the census reproduces 06.**
  - **The rule:** on 05's 39 rows the census's `1e-7` units, and on its 31 depth-20 rows the
    `1e-8` units, equal 06's `s` units bit for bit: `total`, `total_abserr`, every call's value
    and abserr, and the region arrays.
  - **A failure is a stop.** It means the census is not measuring 06's code path, and C1–C3 are
    not scored until the user has seen it.

**How the user reads them.** The decision is the user's (G2 L2).
- If C1 and C2 hold, no row of the store needs layer 2 at either tolerance, at this code.
- If either fails, the rows that fail are characterised by §5 items 2–3 and 7, so that the user
  can see:
  - whether they are the floor layer 2 addresses;
  - or the stationary-point defect 07 addresses;
  - or something else.

