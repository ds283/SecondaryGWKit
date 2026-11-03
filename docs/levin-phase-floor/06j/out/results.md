# levin-phase-floor 06j — results

<!-- provenance: written 2026-11-03 by the 06j agent on mps026298 from out/partS_summary.json (score_s.py, out/cmd/42),
out/partS_nodes.json (probe_s_nodes.py, out/cmd/43-44), out/partS_se_control.json (out/cmd/61),
out/w_scores.json (score_w.py, out/cmd/60), out/w_prep_v2/ (out/cmd/45), out/cmd/51, out/cmd/63. Tree dfa19d98
(code 1c5336e's, tool b5626ecb's); scratch copy fingerprint 0e32f24a…; 06g census checkpoint 03adcbba…, 06h's
3104c18e…. Every number below is in those files. This file proposes no code. -->

## Gate J0 — holds, with no replacement

- **The captures match the census.** All 46 captures reproduce 06g's census bit for bit: value, abserr, label,
  converged, regions and the region arrays (`out/check_captures.json`).
- **The replays match too.** All 46 replays, each call alone in a fresh process, reproduce the census record,
  including the region arrays and every checked leaf (`out/replay/`).
- **The tool reproduces.** Re-reading 06g's r51 QAWO r1 (a monotone Route B call) at this tree gives the same
  `value_hex`, `value_30` and `imag_30` (`out/cmd/40`).
- **The replay is deterministic.** s00, replayed twice in fresh processes, gives identical checked-leaf tables (164
  leaves) and identical returned leaves with their value and $r$ (83 leaves; `out/cmd/41`).

## Part S — the slow leaves

### The verdicts

The sample is 24 lost calls, 142 scoped slow leaves.
- Every leaf is a Clenshaw–Curtis leaf at depth 20.
- The probe on each leaf's own span reproduces the census $\Delta_1$, $\Delta_2$, $r$ and $\phi_P$ bit for bit on
  all 142.

| | Rule | Numbers | Verdict |
|---|---|---|---|
| **S1** | ≥ 80 % hold a seam, and ≥ 5× the null | 1 of 142 (0.70 %); the null is 0.119 %, so 5.9× it | **fails** |
| **S2** | ≥ 80 % of the seam leaves keep $r \in [0.4, 0.6]$ at depths 21–24 on the seam branch | 1 seam leaf (s07), $r$ = 0.494, 0.152, 0.571, 0.078 | **fails** (n = 1) |
| **S3** | ≥ 80 % have $\Delta_1$ > 10 × the larger S-c scatter | 0 of 142; $\Delta_1$/scatter has median 0.039 and maximum 1.81 | **fails** |
| **S4** | ≥ 80 % of the seam leaves have both sides trusted under (b) | s07: the sides have $r$ = 20.4 and 2.42, neither trusted | **fails** (n = 1) |

### What the measurements show

**S-a: the slow leaves are not at representation seams.**
- 141 of 142 hold none of the 1,981–12,459 seams per call (phase and amplitude spline knots, table nodes, switch
  midpoints).
- Their distance to the nearest seam has median 206 leaf widths (range 7–1,141).
- **The comparison leaves.** The calls' trusted returned leaves at depth ≥ 15 hold a seam at 1.7 %, against a null
  of 1.6 % at their depths. So slow and trusted leaves sit on seams at the rate chance gives.
- **Check:** the seams merged by the tool's own rule equal `production_edges` on all 24 calls.

**S-b: $r$ stays near ½ wherever the leaf goes.**
- On the 141 no-seam leaves, the median $r$ is 0.49–0.52 at every depth 21–24, on both the larger-$\Delta_1$ and
  the smaller-$\Delta_1$ branch.
- **$r$ wanders around ½ rather than staying in the band.** Its p10–p90 range is 0.16–1.5 (max branch) and up to
  0.15–6.2 (min branch), and no leaf stays in $[0.4, 0.6]$ at every depth on both branches.
- **$\Delta_1/\phi_P$ alternates between about 0.5 and about 1.5** along the branches.
- This is what rounding that scales with the leaf gives. A localised feature would give $r = ½$ only on the branch
  that holds it.

**S-c: $\Delta_1$ is at or below the rounding scatter.**
- Moving the nodes, or the integrand's inputs at the nodes, by a few ulp changes $V_P$ by an rms of 0.34–795 × $\Delta_1$.
- Every one of the 16 draws changes the value, on every leaf.

**S-d and S-e (s07, the one seam leaf).**
- **The seam:** a table node of `tau`, `cs_tau` and `friction_F` coinciding with spline knots, at $z = 1.59\times10^8$.
- **No jump in the integrand.** Its one-sided "jump" falls as $h^3$ and equals the control stencil's at every step
  (2.52e-33, 3.93e-35 and 5.84e-37 at $h = w/16$, $w/64$, $w/256$; control 2.57e-33, 4.01e-35, 6.26e-37;
  $|F| = 5.2\times10^{-26}$).
- **A tiny phase step.** `theta_mod_2pi` shows an apparent step of 5e-13 to 1.1e-12 rad there that does not shrink
  with $h$; the control's falls to 3e-15. Its effect on $F$, at most 1e-12 of $|F|$, is below the stencil's noise.
- **Splitting at the seam does not restore trust** (S4 above).

**Descriptive, added after S1–S4 were scored: what $\Delta_1$ is** (`out/partS_nodes.json`). The leaves fall into
two populations.
- **75 leaves with $\Delta_1/\phi_P < 2$ (median 1.49):**
  - $\Delta_1 = |E_P - (E_L + E_R)|$ to 0.6 %, where $E$ is each solve's own signed node-displacement error (06e's
    node term).
  - Correcting each solve's value by $-E$ removes 99.8 % of $\Delta_1$ (median remainder 0.22 %).
  - $r$ is 0.489–0.506, and $\phi_P$ is 99.9 % node term.
  - **The mechanism.** Δ1 sees the difference of three solves' node errors; $\phi_P$ declares only the parent's
    $|E_P|$. Each $E$ is proportional to the leaf's width, which fixes $r$ at ½. And $\phi_P$ understates
    $\Delta_1$ by the factor of about 1.5 that defeats the noise clause $\Delta_1 \le \phi_P$.
- **67 leaves with $\Delta_1/\phi_P \ge 2$:**
  - The node errors explain about half of $\Delta_1$ (median $|E$-difference$|/\Delta_1$ is 0.46).
  - The few-ulp scatter is 31–278 × $\Delta_1$ (p10–p90), and $\phi_P$ here is mostly the non-node rounding terms.
  - So $\Delta_1$ is well inside the integrand's rounding sensitivity at the nodes, the same jitter of
    `theta_mod_2pi` 06i described.

**Reading.** Explanation (J), a non-smooth point, is not what happens: 141 of 142 slow leaves hold no seam. The
slow leaves are explanation (N), rounding committed per node that scales with the leaf's width. For half of them
it is exactly the node displacement 06e's node term models, and the trust test's noise clause compares it with
the parent's term alone.

**What a remedy would have to do (Part S).**
- Rule (b)'s noise clause must compare $\Delta_1$ and $\Delta_2$ with a noise level that covers every solve they
  are differences of, not $\phi_P$ alone. That means at least $|E_P| + |E_L| + |E_R|$ for the node term, the
  quarters' for $\Delta_2$, and the evaluation jitter of the nodes for the other population. Or $\Delta_1$ and
  $\Delta_2$ must be formed from values corrected by $-E$, with the second population's jitter then declared.
- Passing seams to Levin as breakpoints would not address these leaves.
- **The Part W evidence below constrains any remedy.** Rule (b) is right where $\Delta_1 \gg \phi_P$:
  $\Delta_1/\phi_P$ was 5.8 to 1.2e7 on 06h's wrong leaves, against 1.0–25 on these slow leaves. A remedy must keep
  refining such leaves.

## Part W — V2's rows

### The verdicts

All readings were usable and agreed.
- Seventeen references were read, 68 readings, all four usable on every call, all `all_monotone` with no direct
  segment.
- The five W-5 rows' 1e-8 calls use their 1e-7 twins' readings: the integrands are equal by 06g's criterion and the
  65-point check, and only `rtol` differs.

| | Rule | Numbers | Verdict |
|---|---|---|---|
| **W1** | every W call's four readings agree within min(e_g, e_06h)/100 | 22 of 22; spread/tolerance from 3.1e-8 (c05) to 8.2e-3 (c03) | **holds** |
| **W2** | every W-5 call has $\|v_g - {\rm ref}\| \le e_g$ | 10 of 10; worst 0.0463 (0-2108), others 5.6e-5 to 0.0203 | **holds** |
| **W3** | every W-5 call has $\|v_{06h} - {\rm ref}\| > e_{06h}$ | 10 of 10; 10.6, 13.0, 13.1, 16.7 and 20.3 | **holds** |
| **W4** | over W-12, 06g within on ≥ 11 and 06h outside on ≥ 6 | 06g within on 12 of 12 (worst 0.071, c10); 06h outside on 12 of 12 (2.40 to 18.4) | **holds** |

Per call, in units of each code's own declared error (`out/w_tables.md`; f05–f09 equal f00–f04):

| call | group | g (regions, depth) | 06h (regions, depth) |
|---|---|---|---|
| f00 0-12778 | G+r | 6.3e-5 (16, 7) | 12.98 (2, 1) |
| f01 0-12959 | G+r | 0.0203 (12, 5) | 13.05 (2, 1) |
| f02 0-1873 | G+r | 5.6e-5 (15, 6) | 20.28 (2, 1) |
| f03 0-2108 | G+r | 0.0463 (14, 6) | 16.65 (2, 1) |
| f04 0-33730 | G+r | 6.4e-5 (16, 7) | 10.61 (2, 1) |
| c00 1-2142 #5 | G-q+r | 0.0050 (122, 16) | 3.55 (9, 5) |
| c01 2-33416 #6 | G-q-r | 0.0014 (23, 6) | 3.94 (1, 0) |
| c02 1-35497 #5 | G-q+r | 0.0012 (69, 15) | 5.02 (8, 4) |
| c03 0-12727 #4 | G+q-r | 0.0216 (74, 14) | 18.4 (2, 1) |
| c04 1-35052 #6 | G-q-r | 0.0154 (59, 15) | 5.30 (6, 4) |
| c05 0-33892 #1 | G+r | 0.0215 (13, 5) | 9.18 (2, 1) |
| c06 3-34272 #4 | G-q-r | 0.0329 (144, 15) | 2.40 (14, 7) |
| c07 3-37323 #1 | G+q+r | 0.0133 (42, 11) | 3.79 (3, 2) |
| c08 1-35136 #3 | G+q+r | 0.0005 (90, 15) | 10.69 (1, 0) |
| c09 1-35446 #4 | G-q-r | 0.0040 (52, 12) | 3.40 (5, 3) |
| c10 0-12965 #4 | G+q-r | 0.0708 (26, 10) | 5.73 (7, 3) |
| c11 0-33648 #1 | G+r | 4.2e-5 (14, 5) | 15.61 (2, 1) |

Both codes report converged on every W call.

### Where 06h's error sits (`out/cmd/51`, `out/cmd/63`)

**06h's leaf values were reconstructed exactly.** They come from solving each 06h leaf's span with the tree's
subregion solve, whose value path is 06h's. Adding them in the code's own ascending order reproduces 06h's value
bit for bit on all 22 calls.

**The move is in one 06h leaf on every call.** One 06h leaf, the coarse cell of the common refinement, carries
97.6–100 % of $v_g - v_{06h}$ (c06: 116.7 %, offset by its other cells).
- **On the `G+r` calls** it is the left depth-1 leaf $[23.539, 25.255]$. The move is
  $7.04$–$7.21\times10^{-33}$, and $v_{06h} - {\rm ref}$ equals it to 0.2 %.
- **Inside that leaf,** 06g's 11–13 leaves (depths 3–7, all trusted, $r \le 0.234$) sum to within
  $2\times10^{-38}$–$1.3\times10^{-35}$ of the reference over the whole call.

**06h trusted every carrying leaf, and rule (b) would not have.**
- 06h trusted each carrying leaf at $1/4 \le r < 1$ (0.258–0.671), with $\Delta_1/\phi_P$ between 5.8 and
  $1.2\times10^7$.
- Every one would be untrusted under 06g's rule (b).
- This is `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`'s mechanism: a leaf extrapolated on an $r$ that is
  not yet asymptotic. 06g's rule (b) is what corrects it.

**Reading.** On V2's rows the parent is the one in error, by 2.4 to 20 times its declared error, and 06g is honest
on all 22 calls (worst 0.071). This is the same answer 06g's references gave on V1's 12 calls and on 3 of the 12 V8
calls.

**What a remedy would have to do (Part W).** No remedy is called for on 06g's side for these rows. 06g's values
and declared errors are right. What V2 exposed is that 06h's trust test (r < 1) accepted leaves with large
$\Delta_1/\phi_P$. Any change made for Part S, such as a wider noise level in the trust test, must keep rule (b)'s
refinement of leaves like these:
- $\Delta_1/\phi_P$ from 5.8 to $1.2\times10^7$, and $r$ from 0.26 to 0.67;
- the node-limited slow leaves of Part S, by contrast, sit at $\Delta_1/\phi_P$ of 1.0–25.

The two ranges overlap only at the low end: c07's carrying leaf has $\Delta_1/\phi_P = 5.8$.
