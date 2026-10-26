# Issues found by handover prompt 04 (A4)

These are proposed for the board's §3. The records commit decides.

1. **[04-table-cs-tau-fixture-does-not-cover-z-init-when-the-first-sample-is-dropped]**: in
   `ComputeTargets/tests/test_g_phase_reanchor.py:448`, `_TableCsTauFixture.exact_envelope_model`
   builds its `cs_tau` `CumulativeTable` on the fixture's own `z_WKB`.
   - **The failure.** With `drop_first_WKB_sample=True`, `z_WKB` starts one grid step below
     $z_{\rm init}$, but `TkSourceFunctions` anchors $T$'s phase at $z_{\rm init}$.
     `build_phi_samples` then evaluates `cs_tau.delta(z, z_init)`, $z_{\rm init}$ is exactly one
     interval above the top node, and `CumulativeTable._nearest_node` refuses anything more than
     one interval above. One ulp of `log1p` decides it.
   - **Where it fails.** With A2's fixtures at $b = 0$, it builds at $x_{\rm resp} = 980$, $10^7$
     and $10^8$ (margin $-1$ or $0$ ulp). It raises at $10^4$, $10^5$ and $10^6$ (margin $+1$ ulp),
     for $T_q$ and $T_r$ on `together` and `T-first` and for $T_r$ on `q-smooth`.
   - **Measurement:** `var/handover-04/out/probe_table_margin.txt` and
     `out/check_construction.txt`.
   - **Impact:** A4 cannot build its realistic $T$ as written at those rungs, and any later
     instrument using this fixture with production's seam (B2) will hit the same wall.
     Production is not affected, because its `functions.cs_tau` covers the whole background grid.
   - **Next step:** the user's answer to A4's stop question. In the fixture itself, build the
     table on a grid that covers $z_{\rm init}$, for example the undropped `z_WKB`.

   *Added on resume:* `_TableCsTauFixture`'s numeric grid ends exactly at $z_{\rm init}$
   (`z_numeric[-1] == crossover_z` bit for bit). So any table that includes the numeric nodes
   makes $z_{\rm init}$ a node, unlike production, where $z_{\rm init}$ is a root off the
   background grid. Measured in `out/probe_union_nodes.txt` and `out/probe_union_table.txt`.

2. **[04-production-phase-gk-tau-table-has-a-zero-width-bottom-interval]**: this is what stopped
   A4's run.
   - **The object.** `ComputeTargets/tests/test_g_phase_reanchor.py:652`, `_ProductionPhaseGk`,
     builds its `tau` `CumulativeTable` on `sorted({z_grid} ∪ {z_resp}, reverse=True)`.
   - **The near-duplicate.** A4 passes `test_quadsource_integral.Case`'s own $G$ grid,
     `log_grid(1 + G_crossover_z, 1 + z_resp, 100/decade)`. Its last node is
     `exp(log(1 + z_resp)) - 1`, which is $z_{\rm resp} - 1$ ulp at 13 of A4's 15 rungs (equal at
     `together` $10^8$ and `q-smooth` $10^6$). The set union keeps both values, so the table's two
     lowest nodes are one ulp apart in $z$ and **identical** in $u = \log(1+z)$. The bottom
     interval is zero-wide in $u$.
   - **The refusal.** `CumulativeTable._nearest_node` refuses any abscissa more than one grid
     interval below the bottom node, which is now any $u$ even one ulp below it
     (`out/probe_tau_table_margin.txt`: refused at 1 to 5 ulps on `together` $10^4$ and
     `q-smooth` 980; accepted at 1 to 5 ulps on `together` $10^8$, whose bottom interval is
     $2.3\times10^{-2}$).
   - **What triggers it.** The Levin driver forms such an abscissa at the bottom end of the
     $z_{\rm resp}$ sub-interval in some cells, so `theta_mod_2pi`, through `end_partial`, raises:
     `RuntimeError: CumulativeTable[tau]: z=… lies more than one grid interval below the table`.
   - **Which cells.** It raised in six A4 cells: P and P₀ at `q-smooth` 980, `together` $10^4$ and
     `T-first` $10^4$. Twenty more were skipped. The cells it did not raise in (P and P₀ at
     `together` 980 and `T-first` 980) have the same degenerate table; their Levin abscissae
     happened not to fall below it.
   - **Whose it is.** It belongs to the arm's construction through a test fixture: the
     `_ProductionPhaseGk` set union, fed a grid whose end point round-trips to $z_{\rm resp}$ minus
     one ulp. It is not production's builder. `_build_phase` reads the background model's own
     `functions.tau`, on which $z_{\rm resp}$ is one node and which continues below it.
   - **Can production be affected?** Not by this route as found. One related property is noted
     and not checked further: `CumulativeTable.__init__` requires the nodes to be strictly
     descending in $z$ (`cumulative_table.py:111`), but not distinct in $u = \log(1+z)$. So two
     nodes one ulp apart are accepted, and they make a zero-width interval. Whether production's
     background grid can ever hold such a pair was not checked here.
   - **Measurement:** `out/probe_tau_table_bottom.txt`, `out/probe_tau_table_margin.txt`, and the
     raised cells' tracebacks in the run directory's `cells/*.json`.
   - **Impact:** A4's arms P₀ and P cannot be run on these grids as built, so A4's readings are
     unanswered. B2, or anything that uses `_ProductionPhaseGk` on a `log_grid` that ends at
     $z_{\rm resp}$, will meet the same wall.
   - **Next step:** the user's decision on how A4 proceeds. In a fixture, build the node set so
     that the grid's end point *is* $z_{\rm resp}$ (no near-duplicate), or deduplicate in $u$.

   *Run 2 (2026-10-26):* A4's own script works around item 2 by setting the grid's last node
   exactly to $z_{\rm resp}$, on the user's choice. The fixture `_ProductionPhaseGk` is
   unchanged, and the issue stands for any other caller.

---

**Run 2 (ended `done`, 2026-10-26):** no new issue. Items 1 and 2 stay proposed, and A4 works
around both in its own script, on the user's choices. For the board's narrowings, as measurements
rather than issues:
- **`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`.** A4 measures, over the ladder, the
  growth of arm P (production's $G$ phase): 6.26 (`together`), 3.29 (`T-first`) and 2.16
  (`q-smooth`). Arm E's is 1.29 / 1.86 / 0.94 and arm S's (A2's $G$) is 6.8 / 46.9 / 40.2. The
  un-anchored `PrimitivePhase` (P₀) grows 8,930 / 2,360 / 619. The memo reads this as **neither
  reading cleanly**. Whether the issue closes or narrows is the user's decision.
- **`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`.** A4 builds production's
  $G$ phase on the fixture's grid, in its own script. It needed the grid's end node set exactly to
  $z_{\rm resp}$ (item 2). The test module is unchanged.
- **`[12-realistic-large-x-reads-the-retired-clamp-records]`.** `a4_cells.py` reads
  `partition["handover"]` in place of the clamp records, in its own copy. `realistic_large_x.py`
  is unchanged.
