## 6j. Prompt 06j — where 06g's slow leaves come from, and which code is right on V2's rows (2026-11-03, Mac Studio)

This was measured on the Mac Studio (mps026298), Ray-free, with no lock and no store built. The tree is the
worktree at `dfa19d98`: production code `1c5336e`'s (06g's `3ae6f3a`), and `tools/filon_reference.py` from
`b5626ecb`.

It read 06's certified scratch copy of A3 v2, read-only. It also read 06g's census
(`levin-phase-floor-06g-census-20261101T114315`, checkpoint `03adcbba…`) and 06h's (`3104c18e…`).

The records are in [`06j/`](06j/); the log is `prompts/levin-phase-floor/logs/06j-slow-leaves-and-v2-rows.md`.

### 6j.1 Gate

All clauses pass ([`06j/out/cmd/00_gate_trees.txt`](06j/out/cmd/00_gate_trees.txt) to `07b`):
- **Commits:** `fdc3a2b` is on `origin/handover-remedial`.
- **The main checkout** is clean at `28ebaf0`.
- **The registry** holds 46 runs, all finished.
- **The scratch copy** is EQUAL to its certification, with fingerprint `0e32f24a…` and no problems.
- **The checkpoints:** 06g's is `03adcbba…` and 06h's is `3104c18e…`.
- **The V8 inputs:** `v8_calls.json` is `1e9a62b0…` and `v8_slow_leaves.json` is `b4643033…`.
- **The tree:** `git diff --stat 1c5336e dfa19d98` outside docs and prompts shows only `CLAUDE.md`, the tool and
  its test.

### 6j.2 Predictions, samples and runs

- **The predictions, rules and seeds** were written at 09:04:38Z, before any probe, draw or reading, in
  [`06j/state/predictions_and_rules_20261103T0905Z.md`](06j/state/predictions_and_rules_20261103T0905Z.md)
  (read-only, `b41740de…`).
- **The samples** were drawn at 09:06:41Z ([`06j/out/sample/`](06j/out/sample/)):
  - **Part S:** 24 lost calls with 142 scoped slow leaves (seed 20261103).
  - **W-5:** rows 0-12778, 0-12959, 0-1873, 0-2108 and 0-33730, call 1, at both rtols.
  - **W-12:** 12 calls from the call-level set, which reproduced 62 and 80.
- **The readings plan** ([`06j/state/readings_plan_20261103T0935Z.md`](06j/state/readings_plan_20261103T0935Z.md))
  was checked by the orchestrator before any reading. Its header gives an impossible correction time (log,
  deviation 15).
- **Two registered runs,** both `done`, exit 0:
  - `levin-phase-floor-06j-run1-20261103T091504`: captures, J0, Part S's probes and W's preparation;
  - `levin-phase-floor-06j-readings-20261103T093537`: 68 reference readings, 2 h 38 min, none stopped.

### 6j.3 Gate J0 — holds, no replacement

- **Captures:** all 46 reproduce 06g's census bit for bit: value, abserr, label, converged, regions and region
  arrays ([`06j/out/check_captures.json`](06j/out/check_captures.json)).
- **Replays:** all 46, each call replayed alone, equal the census including every checked leaf
  ([`06j/out/replay/`](06j/out/replay/)).
- **The tool:** 06g's r51 QAWO r1, re-read at this tree, is bit-identical in value and imaginary part
  ([`06j/out/cmd/40_j0_tool_compare.txt`](06j/out/cmd/40_j0_tool_compare.txt)).
- **Determinism:** s00, replayed twice in fresh processes, gives identical checked-leaf tables and returned leaves,
  value and $r$ ([`06j/out/cmd/41_j0_determinism.txt`](06j/out/cmd/41_j0_determinism.txt)).

### 6j.4 Part S — S1, S2, S3 and S4 fail

Every sampled leaf is a Clenshaw–Curtis leaf at depth 20. The isolated leaf rule reproduces the census's
$\Delta_1$, $\Delta_2$, $r$ and $\phi_P$ on all 142 leaves.

| | Numbers | Verdict |
|---|---|---|
| S1 | 1 of 142 hold a seam (0.70 %); null 0.119 %; trusted depth ≥ 15 leaves 1.7 % against their null 1.6 % | fails |
| S2 | 1 seam leaf; $r$ = 0.494, 0.152, 0.571, 0.078 | fails (n = 1) |
| S3 | 0 of 142 with $\Delta_1$ > 10 × scatter; $\Delta_1$/scatter median 0.039, max 1.81 | fails |
| S4 | 1 seam leaf; neither side trusted | fails (n = 1) |

**Descriptive, after scoring** ([`06j/out/partS_nodes.json`](06j/out/partS_nodes.json)):
- **On 75 leaves** ($\Delta_1/\phi_P < 2$), $\Delta_1$ equals $|E_P - (E_L + E_R)|$, the difference of the three
  solves' 06e node errors, to 0.6 %. Correcting each value by $-E$ removes 99.8 % of $\Delta_1$.
- **On the other 67,** $\Delta_1$ lies inside the integrand's few-ulp input sensitivity (31–278 × $\Delta_1$).
- **At the one seam** (s07, $z = 1.59\times10^8$) the integrand shows no jump: its stencil equals the control's
  ([`06j/out/partS_se_control.json`](06j/out/partS_se_control.json)).

The cause is (N), not (J).

### 6j.5 Part W — W1, W2, W3 and W4 hold

- **The readings:** 17 references, 68 readings, all usable (all `all_monotone`).
- **The W-5 twins:** each row's 1e-8 call is equal in integrand to its 1e-7 call and uses its reference.

| | Numbers | Verdict |
|---|---|---|
| W1 | 22 of 22 agree within min(e)/100 (spread/tol ≤ 8.2e-3) | holds |
| W2 | W-5: 06g within its error on 10 of 10, worst 0.0463 | holds |
| W3 | W-5: 06h outside on 10 of 10, 10.6–20.3 | holds |
| W4 | W-12: 06g within on 12 of 12 (worst 0.071); 06h outside on 12 of 12 (2.40–18.4) | holds |

The tables are [`06j/out/w_tables.md`](06j/out/w_tables.md) and [`06j/out/w_scores.json`](06j/out/w_scores.json).

**Where 06h's error sits:**
- **06h's leaf values,** recomputed and added in the code's own order, reproduce 06h's value on 22 of 22 calls.
- **One 06h leaf** carries 97.6–100 % of the move (c06: 117 %). On the `G+r` calls it is the depth-1 leaf
  $[23.539, 25.255]$, with $v_{06h} - {\rm ref}$ equal to the move to 0.2 %.
- **Every carrying leaf** was trusted by 06h at $0.26 \le r \le 0.67$ with $\Delta_1/\phi_P$ from 5.8 to
  $1.2\times10^7$; rule (b) untrusts every one.

### 6j.6 Unchanged and untouched

- **The scratch copy** is EQUAL to its certification after the runs and after the final fingerprint, and its
  fingerprint is `0e32f24a…` after ([`06j/out/cmd/71_copy_after_vs_certified.txt`](06j/out/cmd/71_copy_after_vs_certified.txt),
  `72`, `74`).
- **Both checkpoints and 06g's inputs** match by SHA-256 after (`75`). Nothing was created in `var/datastores/`.
- **No Ray:** Ray was never initialised in any process (`81`, `62`). The lock is free and no Ray process is listed
  (`78`, `79`).
- **The trees:** the worktree is clean with `--ignored` before and after, and the main checkout is clean (`76`).
  06g's worktrees, 07's records and every stash were not touched.
