### 6g.x The census: 06h's census re-run under 06g's driver (Mac Studio, 2026-11-01)

**What was run.** Run `levin-phase-floor-06g-census-20261101T114315` re-ran every A3 v2 `QuadSourceIntegral` row
(9,280) at `rtol` 1e-7 and 1e-8, in-process and Ray-free.
- **Tree:** $g$, the worktree at `1c5336e`, whose code is 06g's `3ae6f3a`.
- **Store:** 06's certified scratch copy of A3 v2, read-only. It was byte-identical to its certification before
  and after; fingerprint `0e32f24a…`.
- **Comparison:** against 06h's run `levin-phase-floor-06h-census-20261101T005037` (checkpoint `3104c18e…`,
  unchanged).
- **Outcome:** 18,560 of 18,560 units `done`; none in error, none over budget, none past 300 s.
- **Cost:** the run took 11:43–17:23Z, **5 h 40 min against 06h's 2 h 33 min**. Unit wall ×3.43 (1e-7) and ×2.96
  (1e-8). See [`06g/census/log.md`](06g/census/log.md), "Cost".
- **Harness:** 06h's, changed as the log's deviations 1–15 say. The diff is
  [`06g/census/out/scripts_diff_against_06h_final.txt`](06g/census/out/scripts_diff_against_06h_final.txt).

**Gate V holds** ([`06g/census/out/cmd/20`](06g/census/out/cmd/20_gate_v_check.txt),
[`21`](06g/census/out/cmd/21_gate_v_git_clauses.txt)):
1. **The parent reproduces 06h.** At `4f4f9ed`, `low-G+r-1` and `ctrl-low-1` at 1e-7, and `0-1971` at 1e-8, equal
   06h's units bit for bit in every value 06h recorded, the oracle's value, abserr and converged included. The
   totals are `0x1.5e6b398fe7c40p-101`, `-0x1.e3272e29c8b71p-102` and `0x1.6ad248e2d3ab7p-66`.
2. **The integrand is 06h's.** `c1ec3b2..3ae6f3a` outside `docs/` and `prompts/` touches only `AdaptiveLevin/`,
   `tools/`, test files and `CLAUDE.md`. Also, `c1ec3b2..4f4f9ed` changes only `CLAUDE.md`, and
   `3ae6f3a..1c5336e` changes nothing.
3. **The new code is deterministic.** `g-1e-7-0-14164` is identical in two fresh workers on every key. It
   qualifies: 2 CC calls have ΣE ≠ 0, and 107 leaves return their quarters.

**The predictions.** [`06g/census/state/predictions.md`](06g/census/state/predictions.md), sha256 `930ce6fd…`, was
written at 11:42:36Z, before the launch. Part A is the prompt's V2–V8 byte for byte. Part B fixes the readings,
V8's five cause classes among them.

**The note's §5 stop on V8 was reached** (`review/stop_v8.md`). It was put to the user, who ruled to score the rest:
*"(b) with the V8 sample, please"*. V8's verdict stands.

| | Number | Verdict |
|---|---|---|
| V2, on `total` | 8,560 rows per rtol; 5 violations at each, worst ratio 2.164 (`0-12959`). On the oracle: 0 violations, max 0.499 | **fails** |
| V5 | 966 (1e-7) and 996 (1e-8) group calls outside the 720 at depth 20 (06h: 0, 1); 0 over budget | **fails** |
| V6 | `g-1e-8-0-33995` #5 ends `node_limited` at depth 20 (abserr 7.16e-34 against request 6.69e-34; 963 regions; ΣE −5.71e-35, Σ\|E\| 6.51e-34) | **fails** |
| V7 | `g-1e-8-3-1072` #13 converges (1.54e-26 against 1.63e-26; 2 regions, depth 1) | **holds** |
| V8 | Lost 801 calls / 470 rows (1e-7), 830 / 476 (1e-8); all group calls. By class: (b) refined to the depth cap 801 / 827; node_limited 0 / 3. Every (b) call meets its request on abserr. Gained: 1 (V7's call) | **fails** |

[`06g/census/out/v8.md`](06g/census/out/v8.md) holds V8's tables; the other scores are in
[`06g/census/out/cmd/49`](06g/census/out/cmd/49_analyse06g.txt).

**The slow leaves behind V8** (the user's addendum, descriptive;
[`06g/census/out/v8_slow_leaves.md`](06g/census/out/v8_slow_leaves.md)):
- **Scope:** 4,342 (1e-7) and 4,478 (1e-8) untrusted (b) leaves returned by lost calls, all at depth 20.
- **r:** median 0.501, with 41 % within 0.01 of 1/2.
- **Δ1/φ_P:** median 2.3, p90 10, max 45.
- **r along the recorded ancestor chains:** it wanders, and settles near 0.5 only over the last few levels.
- **Position:** median 2 clusters per call, max 29, none at the call's interval ends.
- **By group:** the same r distribution in every group.

**Descriptive** (outside the 720):
- On the same partition, the values move by at most abserr_g; no call without quarters moves.
- Group regions ×4.95 and ×3.81. Exit firings 90,509 → 402,760 and 127,498 → 461,974.
- Σ|E|/|ΣE| has median 1 and p99 about 100 where Σ|E| is material.
- A3 v2 reaches the near-singular class (435 rows): 43 group calls lost there at each rtol, no analytic call lost.
