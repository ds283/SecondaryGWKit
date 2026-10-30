# depth20-tiling: the sampling rule (fixed before any result beyond the inventory was read)

Written 2026-10-29 on mps026298, after `out/00_inventory.txt` (identity, tolerance, block, unit
`wall_s`, call `elapsed`, `x_span`) and before any residual, floor, tiling, capture or re-solve
was read or computed. Its SHA-256 is recorded in `log.md`.

**Population.** The 262 calls of `census/out/deep_lines_06c.json`, as listed in
`state/inventory.json`. A call is identified by `(unit, index)`. Its tolerance is the unit's
rtol (`1e-7`, `1e-8`); its block is the `notify_label`'s group (`G+q+r`, `G+q-r`, `G-q+r`,
`G-q-r`) or analytic label (`J1`–`J4`, `Y1`–`Y4`). There are 24 cells (2 tolerances × 12 blocks),
each non-empty.

**Hash order.** Each call's key is `sha256(f"{unit}:{index}".encode()).hexdigest()`, compared as a
string; smaller first.

**Eligibility (short units preferred, stated in advance).** A call is *eligible* if its unit's
census `wall_s` is at most 120 s.

**Rule.**
1. **One per cell.** In each of the 24 cells, take the eligible call with the smallest key. If a
   cell has no eligible call, take its call with the smallest unit `wall_s` (ties: smallest key).
2. **Stratified fill to 30.** Order the cells by their number of calls, largest first; ties by
   tolerance (`1e-7` before `1e-8`), then by block in the order `G+q+r, G+q-r, G-q+r, G-q-r, J1, J2,
   J3, J4, Y1, Y2, Y3, Y4`. Walk that order and, from each cell, take its next eligible call by key
   (not already taken); skip a cell with none left. Stop when the sample holds 30 calls.
3. The sample is the 30 calls so drawn. Nothing is replaced after the draw: a call whose capture
   does not reproduce the census is reported (and the brief's stop rule applies), not swapped.

**Use.** The cheap pass (step 1) runs on all 262 calls if the spans can be reconstructed without a
capture. Steps 2–4 (capture, classification re-solves, counterfactual) run on the 30. References
(step 4) are computed for as many of the 30 as the budget allows, in sample order (the order in
which rule 1 then rule 2 drew them).

The draw is made by `scripts/draw_sample.py` and written to `state/sample.json`.
