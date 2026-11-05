# Planned stop 1 — checked by the orchestrator, 2026-11-05 ~10:10Z: PASS, census cleared

- **The variant's diff** (`checks/s1_ast_compare.txt`, own script): identity d89c959e… byte-identical to the
  tree's module; route B 5aa6178a… differs only in `_adaptive_levin`: `route_records_06l = []`, `_check_leaf`
  (the trust statement replaced by nu_B and B's record), and the return dict gains `route_06l` (all other
  keys' values AST-equal). Signature and decorators unchanged.
- **nu_B read against prompt §0.2:** phi_P = region.phase_err (whole floor); + each Levin half's phi; on a
  Clenshaw–Curtis parent only, + each CC half's `phase_err_less_node` + |sum of the CC halves'
  `node_error_signed`|; a CC half adds nothing to a Levin parent's yardstick; a CC half missing either
  field raises. All four cases as the table.
- **K0** (`out/k0/summary.json`): identity 100/100 units equal, compared whole, counters included; the
  excluded keys that differ are timing, provenance hashes, worker ids and the route keys only. Route B:
  1,862 non-firing calls of 2,110 equal 06g bit for bit, 0 failing; 0 withdrawals.
- **L3** (`checks/s1_partL_recheck.txt`, own reader of `out/partL_leaves.csv`): nu_B recomputed from each
  row's floors, E and kinds agrees with the variant's to 1e-15 on all 2,046 rows; B's and route 1's
  decisions agree with the table's; nu_B <= nu1(1+1e-12) on every leaf (max ratio 1 + 2.2e-16); no leaf
  trusted by B and not by route 1. The CC halves' phim equals phi − |E| to 3.3e-16.
- **L2's leaves:** B trusts 63 of 75 (L2 fails, scored, not a stop: prompt §4 "None is a stop"). The 12
  untrusted are all CC/CC/CC, r 0.491–0.501, delta1/nu_B 1.0004–1.0041, E_P and E_L+E_R of opposite sign on
  every one, delta1/|E_P − (E_L+E_R)| 1.0020–1.0061: the triangle bound is tight and delta1 exceeds the node
  difference by more than the rest of nu_B. Agent's account confirmed. L1: 0 of 17 (22 evaluations) trusted.
- **The census plan:** projection 17,347 s, inside [16,758, 20,361]; backstop 06k's code unchanged (diff of
  `run_census.py` against 06k's committed copy: only the `routeB` choice and a header), P_o counts one unit
  in flight per running worker, stops only at P_o >= 86,400 s.
