# levin-phase-floor 05 §6 — the rule that draws the rows

Fixed 2026-10-26 on the Mac Studio (mps026298), before any row is drawn or run. The script that
applies it is `scripts/draw_sample.py`; its output is `state/sample.json` and `out/sample.txt`.
Prompt 05 §6's text for the rows is quoted first, then the rule as applied, then its bias.

## The prompt's text (§6, "The rows")

> Draw them from `var/levin-phase-floor-01/out/groups_01.csv` on the Mac Studio, by a rule fixed
> in `state/sample.md` before any row runs.
> - **Depth-20 rows.** In each cell of {top group `G+r`, `G−r`, lone `G`, `Gqr`} × {low band, mid
>   band}, up to 3 rows: those of smallest stored `compute_time` at distinct $k$, with
>   `compute_time` ≤ 900 s. That is at most 24 rows. Also 03's seven rows, R1–R7.
> - **Controls.** 8 rows with `WKB_Levin_max_depth` < 20 or `NULL`, two per band (low, mid and
>   seam) and two with `NULL`, each with `compute_time` ≤ 60 s.
> - Record the rule's bias: it prefers cheap rows.

## The rule

**Source.** `var/levin-phase-floor-01/out/groups_01.csv` (01's census of A3 v2 at the population
pair `(atol, rtol) = (1e-32, 1e-7)`; its SHA-256 is recorded in `out/sample.txt`). It holds one
record per phase group of every row that has a Levin sub-interval. A row is a distinct
`(shard, serial)`. Its `band`, `k`, `WKB_Levin_max_depth` and `compute_time` are the census's
columns, which are the stored row's values (computed at A3 v2's build tree, before `handover` B1).

**The top group** of a row is 01's T2 definition: the group record of the row with the largest
`elapsed`, over every Levin sub-interval of the row. A tie is broken by the first in file order.
Its **class** is:
- `G+r` if its label is `G+r`; `G−r` if its label is `G-r` (the census writes ASCII `-`);
- lone `G` if its label is `G`;
- `Gqr` if its sub-interval's regime is `Gqr` (labels `G+q+r`, `G+q-r`, `G-q+r`, `G-q-r`; 03's R7,
  top group `G+q+r`, is of this class);
- none of the four otherwise (labels `q±r`, `r`), and such a row is in no cell.

**Depth-20 rows.** For each band in (low, mid) and each class in (`G+r`, `G−r`, lone `G`, `Gqr`),
in that order:
1. the candidates are the rows of that band and class with `WKB_Levin_max_depth` = 20 and
   `compute_time` ≤ 900 s;
2. they are sorted by `compute_time` ascending, ties by `(shard, serial)` ascending;
3. walking that order, a row is taken if its $k$ (the census's `k`, compared exactly as the string
   the census wrote) is not already taken in this cell, until 3 are taken or the candidates run out.

A cell with fewer than 3 candidates at distinct $k$ contributes what it has, and the count is
reported. Then **03's seven rows** are added: R1 (2, 37527), R2 (0, 1797), R3 (1, 37670),
R4 (2, 37720), R5 (0, 37966), R6 (2, 34728), R7 (2, 34820). A row drawn by the cells that is also
one of R1–R7 is run once and carries both names.

**Controls in a band.** For each band in (low, mid, seam), in that order:
1. the candidates are the rows of that band with `WKB_Levin_max_depth` < 20 (not `NULL`; the census
   has no `NULL`), `compute_time` ≤ 60 s, and **at least one group record whose label contains
   `G`**, so that $G$'s phase, the one the code commit changes, is used by the row's Levin
   integration;
2. they are sorted by `compute_time` ascending, ties by `(shard, serial)`;
3. walking that order, a row is taken if its $k$ is not already taken among this band's controls,
   until 2 are taken.

**Controls with `NULL`.** The census holds only rows with a Levin sub-interval, so it has no row
whose `WKB_Levin_max_depth` is `NULL`: those are the rows with no Levin sub-interval at all. They
are drawn from the store itself, read-only (`sqlite3`, `mode=ro`, as 01's census read it), from
the same population (the `QuadSourceIntegral` rows whose `atol` and `rtol` are `1e-32` and `1e-7`):
1. the candidates are the rows with `WKB_Levin_max_depth` `NULL` and `compute_time` ≤ 60 s, in any
   band;
2. sorted by `compute_time` ascending, ties by `(shard, serial)`;
3. 2 rows taken at distinct $k$ as above. Their band (01's: low $z_{\rm response} < 9$, mid $9$ to
   $10^5$, seam $> 10^5$) is recorded.

That gives at most 24 + 7 depth-20 rows and 8 controls.

**The order of the rows** within each unit set is: the cells' rows in the order above (low band
first; within a band `G+r`, `G−r`, lone `G`, `Gqr`; within a cell by the walk), then R1–R7 not
already drawn, then the controls (low, mid, seam, `NULL`).

**Gate L0's two rows** are the first depth-20 row of that order (the cheapest drawn `G+r` row of
the low band) and the first control (the cheapest low-band control).

## The rule's bias

- **It prefers cheap rows.** In every cell it takes the rows of smallest stored `compute_time`.
  The census's depth-20 rows run from under a second to many minutes, and the cheapest of them are
  rows where the bisection reaches depth 20 in a short chain and little else. So the drawn
  depth-20 rows under-represent the expensive tail, where the cap holds tens of thousands of
  regions. That tail is represented only by 03's R1–R7 (134–355 s stored), which are in the mid
  band.
- **The controls prefer cheap, shallow rows** for the same reason: the cheapest rows below the cap
  are mostly rows whose Levin work stops at depth 0–3. Requiring a group containing `G` keeps every
  band control one on which the change can act; the `NULL` controls are rows on which it cannot act
  at all (no Levin sub-interval), and are expected to be bit-identical between $p$ and $c$.
- **Distinct $k$ within a cell** spreads each cell over wavenumbers, so $F = \varepsilon k\eta_R$
  varies within it; it does not control $z_{\rm response}$ or geometry.
- **The classes and depths are the stored rows'.** They were computed before `handover` B1. Under
  $p$ (which includes B1), the `QuadSource` spline reaches three nodes further and the partition may
  differ, so a drawn depth-20 row need not reach depth 20 under $p$. That is measured, not assumed:
  L1 and L1q are scored against $p$'s own scored groups and region counts.
