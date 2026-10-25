**On the 72 `mixed` rows of A3 v2 (ΛCDM, `SourceGridConstruction_2`, fingerprint `0e32f24a…`), `crossover_z` has
6 / 308 / 481 / 664 / 664 grid intervals of numeric clearance (min / 5th percentile / median / 95th / max) and
1 / 82 / 82 / 82 / 82 of WKB clearance, counted by node index on the stored source grid. One row (1 of 72, 1.4 %) has fewer than
two on one side: shard 2, serial 460, the `minimal` row ($k = 3.091682\times10^6$/Mpc, $z_{\rm response} = 3.963729\times10^{10}$,
`crossover_z` $= 4.2472907\times10^{10}$), with 6 numeric and 1 WKB interval. The other 71, all `mixed`/`complete`, have 301 to 664 numeric
intervals and exactly 82 WKB intervals, with `crossover_z` the same grid node ($3.121867\times10^{15}$) at every $k$ and every
$z_{\rm response}$.**

Two things that sentence leaves out, because they bear on how it is read. The counts are measured from the **inner** ends of the two
splines (`numeric_smallest_z`, `primary_WKB_largest_z`), the ends the clearance is measured from. And the 82 is not a finding about
the grid's end intervals so much as the width of `CLEARANCE_GOOD` = 0.05 in intervals at the top of this grid, to the node (§4).
Whether the end-interval error matters is not measured here (§10).
