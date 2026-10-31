# levin-phase-floor 06f — results

*Written 2026-10-31 on mps026298.phys.susx.ac.uk.*

*Provenance:*
- Tree $g$ = worktree `SecondaryGWKit-lpf06f-f4770d1` at `f4770d13`, clean; its code is `f4d8e17`'s.
- Main checkout `28ebaf0`, clean.
- Census checkpoint `a0dc4d7f…`.
- Scratch copy of A3 v2, fingerprint `0e32f24a…`. It is EQUAL to its certification before and after (`cmd/15`, `18`, `41`).
- Registered run `levin-phase-floor-06f-measure-20261031T204803`: done, 44/44 units, exit 0.

*Sources:*
- Every number below is printed by `cmd/43_analyse.txt` (`analyse06f.py`), from `replay/*/result.json`.
- The F4 attribution is in `cmd/44`, the census-wide declared term in `cmd/45`, and the method check in `cmd/31`.
- Tables: `f0.csv`, `m1_calls.csv`, `m1_z2_by_depth.csv`, `m1_z2_subtrees.csv`, `m2_calls.csv`, `m2_leaves.csv`, `scores.json`.

## Verdicts

| | Rule (prompt §4) | Number | Verdict |
|---|---|---|---|
| F0 | captures and replays reproduce 06e; terms reproduce; Z2's call deterministic | 41/41 calls equal to the census record bit for bit, and their summed terms equal `cc_floor`; Z2's call twice (pids 55078, 55415) gave identical digests, per-leaf tables and per-solve $E$ | **holds**, no replacements |
| F1 | $\sum\lvert E\rvert \ge 10\lvert\sum E\rvert$ on Z2's call | $\sum\lvert E\rvert = 1.3123\times10^{-33}$ against $\lvert\sum E\rvert = 2.4313\times10^{-35}$: ratio **54.0** | **holds** |
| F2 | signed-combination error $\le \max(\texttt{atol}, \texttt{rtol}\lvert v\rvert)$ on Z2's call | **$1.759\times10^{-34}$** ≤ $6.689\times10^{-34}$ (0.26 of the request). The linear form is $4.47\times10^{-35}$ | **holds** |
| F3 | median $\sum\lvert E\rvert/\lvert\sum E\rvert \ge 10$ over node-dominated calls with $N_{\rm CC}\ge100$ (≥5 calls) | 11 calls. Ratios 7.2, 7.8, 8.25, 10.4, 17.6, 20.8, 20.8, 36.3, 54.0, 59.2, 80.2. Median **20.8** (19.2 without Z2's call) | **holds** |
| F4 | $\lvert\delta\theta\rvert \le \texttt{theta\_abserr}$ at every node sampled, every M2 call | 1,177 nodes (every node of every returned leaf): **49 violate**, all on 2 calls, largest ratio **148** | **fails** (a finding, not a stop) |
| F5 | median $D/\max_s\lvert\delta I\rvert \ge 10$ | range 1.07–21.2, median **6.0** | **fails** |
| F6 | median $\sum_\ell\lvert\delta I_\ell\rvert/\max_s\lvert\delta I\rvert \ge 10$ | 18 calls at 1.000, then 1.08 and 1.57; median **1.0** | **fails** |

All three acceptance clauses hold:
- **Method check passes** (`cmd/31`). The ν = 1/2 constituent's phase-pair error against `atan2(J, −Y)` at 30 digits is $9.1\times10^{-18}$ at its argument in the call and $5.9\times10^{-17}$ at $x = 10^3$. `EVALUATION_FLOOR`'s comment documents 1.11e-16. The 30- and 50-digit references agree to $<10^{-30}$.
- **Item 3 converges** to two figures on every call: the worst relative change on doubling the Gauss–Legendre resolution is $1.9\times10^{-7}$.
- **Item 4 was computed on every call**, since item 1 sampled every node.

## M1 — the node errors

**Z2's call** (`g-1e-8-0-33995` call 5) reproduces 06e exactly: 8,706 leaves (8,600 Clenshaw–Curtis), depth 20, unconverged, 30,707 solves, 107 s.

| Quantity | Value |
|---|---|
| $N_{\rm CC}$ | 8,600 ($\sqrt{N_{\rm CC}} = 92.7$) |
| $\sum\lvert E\rvert$ | $1.3123\times10^{-33}$ |
| $\lvert\sum E\rvert$ | $2.431\times10^{-35}$ ($\lvert\sum E\rvert/(\sqrt N\,{\rm rms}\,E) = 0.60$) |
| $\sum(R_g+R_2+R_c)$ | $5.99\times10^{-36}$ |
| request | $6.689\times10^{-34}$ (atol; $\texttt{rtol}\lvert v\rvert = 2.73\times10^{-34}$) |
| declared error | $1.3327\times10^{-33}$ |
| everything but the $\lvert E\rvert$ | $1.516\times10^{-34}$ |
| signed combination | $1.759\times10^{-34}$ |
| per sibling pair | $1.114\times10^{-33}$ (2,484 pairs, 3,632 leaves unpaired) |

How the signed-combination error is formed. Each Clenshaw–Curtis leaf's floor is taken without its $\lvert E\rvert$, max'd with the leaf's resolution term as the code max's them, plus its truncation term. These are summed, and then $\lvert\sum E\rvert$ is added once.

- **The cancellation is not local.** Combining $E$ within sibling pairs takes away only 15 % of $\sum\lvert E\rvert$. Combined over the whole call, $\sum E$ is 1/54 of it.
- **Signs come in runs.**
  - Along the call there are 3,129 runs of one sign, against 4,289 expected for independent signs ($z = -25$). The longest run is 49.
  - So $E$ has a systematic component from leaf to leaf, yet its sum is no larger than a random walk's: 0.60 of $\sqrt N\,{\rm rms}$.
- **Where $\sum\lvert E\rvert$ sits by depth** (`m1_z2_by_depth.csv`).
  - Depth 13 holds $4.3\times10^{-34}$, depth 16 holds $2.0\times10^{-34}$, and depth 10 holds $1.9\times10^{-34}$.
  - The 6,182 leaves at depths 17–20 hold $1.0\times10^{-34}$ together.
- **Subtrees against their root's own $\lvert E\rvert$**, from the trial solves (8,191 all-Clenshaw–Curtis subtrees, `m1_z2_subtrees.csv`).
  - The median ratio $\sum\lvert E\rvert_{\rm subtree}/\lvert E_{\rm root}\rvert$ is 2–27 for roots at depths 5–12 and 14, 4,600 at depth 15, and 0.36–0.75 at depths 13 and 16–19. The p10–p90 spread is two to four decades.
  - Splitting therefore does not lower $\sum\lvert E\rvert$ in any systematic way. It sometimes raises it by orders, and it lowers it by about 2× only near the bottom.

**The other 20 M1 calls** (`m1_calls.csv`):
- Their $N_{\rm CC}$ runs from 1 to 149. At 1e-7 only 4 eligible node-dominated calls have $N_{\rm CC}\ge100$, and all 4 were taken.
- On the 10 calls with $N_{\rm CC}\ge100$ the ratio $\sum\lvert E\rvert/\lvert\sum E\rvert$ is 7.2–80.2, and $\lvert\sum E\rvert/(\sqrt N\,{\rm rms})$ is 0.08–1.05.
- On every one of them the signed combination changes the declared error by under 1 % (ratios 0.996–1.0009). On these calls the node term is the largest floor term, but the floor is not what sets their error. All 20 converged.
- Z2's call is the only sampled call where the node term's combination decides convergence: there the error falls from 1.99× the request to 0.26×.

**What a remedy would have to do for the node term, on this evidence.** No code is proposed here.
1. It would have to carry each leaf's first-order node error $E$ **signed** into the call's total, so that the call carries $\lvert\sum E\rvert$ and not $\sum\lvert E\rvert$. On Z2's call that is a factor of 54, and on the larger sampled calls a median factor of 21.
2. It must do so **across the whole call**: pairing siblings recovers only 15 %.
3. It must **not** assume the signs are independent and substitute a $\sqrt N$ estimate. The signs run ($z=-25$), so only the actual signed sum is evidence.
4. It can keep $R_g$, $R_2$ and $R_c$ as absolute bounds. They are $6\times10^{-36}$ on Z2's call, 0.9 % of its request.
5. Since $E$ would then belong to the call rather than to a leaf, the leaf-level floor-limited test, which compares a leaf's residual with a floor that contains $\lvert E\rvert$, would need a consistent counterpart. That is a design question, not a measurement made here.
6. The F2 number is a statement about Z2's **returned partition**. The driver was not re-run under another combination (prompt §0.3).

## M2 — the declared phase error

The 20 M2 calls are analytic blocks (J1–Y4). 18 have one or two leaves, all Clenshaw–Curtis, at depth 0–1. Two have Levin leaves: `g-1e-7-3-1054` #20 has 1 CC leaf and 1 Levin leaf, and `g-1e-7-0-12689` #19 has 1 CC leaf and 3 Levin leaves. All 20 converged.

**Item 1, the declaration's tightness** (`m2_calls.csv`, nodes in `replay/*/nodes.json.gz`).
- Sample points: on Clenshaw–Curtis leaves, every one of the 47 fine nodes; on Levin leaves, every one of the 24 collocation nodes, endpoints included.
- Typical case:
  - Where a ν = 5/2 constituent is in its near region, `theta_abserr` is $1.0009\times10^{-12}$.
  - There the measured $\lvert\delta\theta\rvert$ has median $2\times10^{-16}$–$8\times10^{-15}$ per call, and $\max\lvert\delta\theta\rvert/\texttt{theta\_abserr}$ is $5\times10^{-4}$–0.08.
- **Violations: 49 of 1,177 nodes** (descriptive dense sample: 473 of 20,000 points, largest ratio 166).
  - They are all on the Levin leaves of the two calls above, in the tail, where `theta_abserr` is about $3\times10^{-15}$. Four of the 49 are leaf endpoints, where a Levin leaf actually evaluates θ.
  - The largest is 148×, on `g-1e-7-0-12689` #19, where $K\eta = -6.85\times10^3$, $\delta\theta = -3.94\times10^{-13}$ and `theta_abserr` is $2.66\times10^{-15}$.
- **Attribution** (`cmd/44`):
  - The rounding of the product $K\eta$ in `BesselPhaseGroup.sin_cos` explains each violation to within 0.27 `theta_abserr`. $K$ itself is exact on both calls.
  - This is the "eps |K| x product floor" that the class docstring names as a genuine limit. `theta_abserr`, the sum of the constituents' declared errors, does not include it.

**Items 2–4 per call** (`m2_calls.csv`; per leaf in `m2_leaves.csv`).
- $D/\max_s\lvert\delta I\rvert$: 1.07–21.2, median 6.0. $\delta I(s=-1) = -\delta I(s=+1)$ exactly.
- $\sum_\ell\lvert\delta I_\ell\rvert/\max_s\lvert\delta I\rvert$: median 1.0. One-leaf calls give 1 by construction, and so do the two Levin calls, whose leaves' responses all share one sign.
- **The proxy itself is nearly honest.** With the per-endpoint proxy $\tfrac12 f_{\rm scale}h$ replaced by $\tfrac12\sum_k\lvert w_k\rvert\lvert f_k\rvert$, $D/D_{\rm honest}$ has median 1.05 (range 1.00–1.52), and $D_{\rm honest}/\max_s\lvert\delta I\rvert$ has median 5.25.
- **So the factor of about 6 is cancellation within the leaf.** It is $\int\lvert f\rvert$ against $\lvert\int f\,b'(\theta)\rvert$ over a leaf with 0.2–68 rad of phase (8,130 rad on the Levin call). It is neither the proxy's slack nor the combination across leaves.
- **Response to the measured error (item 4).** $\lvert\delta I_{\rm measured}\rvert/\max_s\lvert\delta I\rvert$ has median $4.8\times10^{-4}$. The declaration, not the proxy or the combination, is the large factor between $D$ and the error the phase actually commits.
  - Exception: on the tail Levin leaves of `g-1e-7-0-12689` the measured response exceeds the declared term. Leaf 3 gives $1.7\times10^{-25}$ against $1.9\times10^{-26}$, and leaf 2 gives $2.3\times10^{-25}$ against $2.3\times10^{-25}$. That is F4's under-declaration, visible in the value.
  - That call's total $D = 2.1\times10^{-20}$, set by its CC leaf, covers both.
- **Cross-check of item 3.** The rule's own first-order response to $\delta\theta = \texttt{theta\_abserr}$ at its nodes is compared with the Gauss–Legendre integral:
  - on Clenshaw–Curtis leaves without a kink of `theta_abserr`, agreement to $\le 5\times10^{-14}$ relative;
  - on the one CC leaf that holds a kink, $3\times10^{-4}$;
  - on Levin leaves, differences of 1.5 % (no kink) and 10 % (kink): the endpoint formula omits the response through θ′.
- **$D$ never holds a call back.**
  - On the sample, $D$ is the call's declared error (median ratio 0.994), and $D/\text{request}$ has median $1.4\times10^{-4}$ and maximum $6\times10^{-3}$.
  - Across 06e's whole census (`cmd/45`), the declared term summed over floor-attributed CC leaves reaches the request on **0** of 5,224 (1e-7) and 6,055 (1e-8) calls with a non-zero term. It reaches 0.1 of the request on one call and 0.01 on 23, and none of these calls is unconverged.
  - D4's 99.4 % share is a share of floors that are themselves far below the request.

**What a remedy would have to do for the declared term, on this evidence.** No code is proposed here.
1. **For tightness.** Replacing $\texttt{theta\_abserr}\cdot\int\lvert f\rvert$ by the leaf's signed first-order response would gain a median factor of 6. That response is $\int f\,\texttt{theta\_abserr}\,b'(\theta)$; on the CC branch $\sum_k w_k\,g'_k\,\texttt{theta\_abserr}_k$ gives it to $5\times10^{-14}$ (or $3\times10^{-4}$ across a kink) from data the solve already holds. Replacing only the proxy by $\sum\lvert w\rvert\lvert f\rvert$ would gain 5 %, and combining across leaves gains nothing on these calls.
2. **For whether tightness matters.** On 06e's census the term never reached a request, so a tighter term would change no call's convergence there.
3. **For honesty.** The larger defect is F4. `theta_abserr` under-declares wherever $\varepsilon\lvert K\rvert\eta$ exceeds the constituents' tail declarations, up to 148× here. A remedy for the declared term must not be built on the assumption that `theta_abserr` bounds $\delta\theta$. How `theta_abserr` is constructed is outside 06f (§0.3), and is the user's to assign.
