# The residue the gap was masking, by term

**Measured:** 2026-10-27, on the laptop (`mps026088`; Python 3.13.16), in the worktree
`SecondaryGWKit-handover-13` at `28ade78` (B2's Part A), with a clean tree. Offline: no Ray, no
datastore, no lock.
**Run:** `handover-13-residue-20261027T110832`, registered, `done`, 208/208 cells `ok`, 11:08–11:18
UTC. Each cell ran in a child process, at most two at once. The 1-minute load was 6–27, because
other projects' Ray work shared the host.
**Scripts:** [`b2/scripts/`](b2/scripts/). Each is a record of its own SHA-256, and the run's
checkpoint records hash the four that ran it: `b2_cells.py` `0a8ae0eb…`, `b2_driver.py` `dc8772e3…`,
`b2_checks.py` `6bbda526…` and `b2_launch.py` `c8fb8bf0…`. §3 is the stdout of `b2_tables.py`.
The run's records are in the main checkout's `var/runs/`, which is not committed.
**Instrument:** `ComputeTargets/tests/test_quadsource_integral.py`'s `Case`, whose realistic
flavour is production's construction since Part A. A2's helpers are imported from
[`realistic_large_x.py`](realistic_large_x.py) (`c1cd3598…`), which is not edited.
**Reference:** Kohri & Terada eq. (22) and the head, both at 50 digits (`reference_of`).
**Campaign:** [`prompts/handover`](../../prompts/handover/README.md), board item **B2**, prompt 13.

---

## 0. What is being compared, and whose each object is

| | Object | Provenance |
|---|---|---|
| **E** | `Case(exact=True)`: the pipeline's partition and Levin integrator on `ExactTkFunctions`/`ExactGk`/exact-$f$ stand-ins. Each $T$ breaks at its $z_{\rm init}$ | **This repository's pipeline**, ingredients exact to rounding. The control. It is A2's exact closed cell, re-run at `HEAD` |
| **R** | `Case(exact=False)` at `28ade78`. The stand-ins are for the stored samples. $G$'s phase is built by `GkSourcePolicyData._build_phase`; each $T$ by `TkSourceFunctions` (no keyword) on a table `cs_tau`; production's seam (Liouville–Green samples from the grid node below $z_{\rm init}$); $f$ is splined on the 100-per-decade grid | **This repository's pipeline and its production constructors.** What is not production's is listed in §7 |
| **E′** | R's `Case` with **every** accessor of the five terms below delegated to its exact counterpart. R's regions, switch nodes, partition and $z_{\rm init}$ are kept | **New, written for this document** (`b2_cells.apply_hybrid`). It swaps accessors only |
| **E′+i**, **R−i** | E′ with term $i$'s accessors realistic; R with term $i$'s accessors exact | Likewise |
| **Ref** | KT eq. (22) and the head $0\to\bar x_{\rm min}$ at 50 digits | The `radiation-oracle` campaign's transcription. It shares no machinery with the rest. It is common to every cell of a rung, so it cancels in every difference below |

**The five terms.** The mapping was read from the code at `HEAD` and fixed in `b2/out/terms.md`
before the run. Each accessor is in exactly one term.

| term | accessors made exact | read where |
|---|---|---|
| `f` | `QuadSource.functions.source` | the all-smooth sub-interval only (`numeric_quad_integral`) |
| `Tnum` | `T.T`, `T.dT_dz` ($T_q$ and $T_r$) | Levin sub-intervals where that $T$ is smooth, including $T=1$ above its numeric grid |
| `Trep` | `T.M`, `T.phase` | Levin sub-intervals where that $T$ is oscillatory |
| `TLG` | `T.omega`, `T.dlnM_dz` | the same; `omega` is also the Levin phase derivative of every group containing that $T$ |
| `G` | `Gk.phase`, `Gk.sin_amplitude` | Levin sub-intervals where $G$ is oscillatory |

`Gk.numeric_Gk` is `compute_analytic_G` in R and E alike, so it is not a term.

**What each comparison guards, and what it does not.**
- **Geometry: $N_{E'}-N_E$.** It guards the decomposition: R's partition, with exact ingredients,
  must reproduce the exact flavour's own. It does not test any ingredient.
- **The residue: $D = N_R - N_{E'}$.** This is everything in which R's ingredients differ from
  exact ones, on R's geometry. It is A2's "representation term", now unmasked, and taken on
  production's construction.
- **Add-one and remove-one terms: $A_i$, $B_i$.** These split $D$. They agree, and they sum to
  $D$, only where the terms do not interact.
- **What none of these guards.** They are statements about this fixture, at $b=0$ and on the
  uniform grid, not about a production row (§7).

---

## 1. The headline

**The residue has five terms. They are additive at all 16 rungs by the rule fixed before the run,
and two of them, `f` and `Tnum`/`TLG`, carry essentially all of it.** `Trep` and `G` are
individually three to eight orders below the largest term at every rung.

| shape | $D = N_R - N_{E'}$ | `f` ($A_f$) | `Tnum` ($A_{\rm Tnum}$) | `Trep` | `TLG` ($A_{\rm TLG}$) | `G` ($A_G$) | additive? |
|---|---|---|---|---|---|---|---|
| `together` (6 rungs) | $-1.090\times10^{-3}$ … $+3.623\times10^{-3}$ | $-1.049\times10^{-3}$ … $+3.375\times10^{-3}$ (0.88–1.02 of $D$) | $-4.90\times10^{-5}$ … $+1.37\times10^{-4}$ | $\le9\times10^{-11}$ | $-3.33\times10^{-5}$ … $+1.11\times10^{-4}$ | $\le1.6\times10^{-9}$ | yes, all 6 |
| `T-first` (5) | $-6.04\times10^{-5}$ … $-8.18\times10^{-6}$ | $-7.36\times10^{-5}$ … $-1.03\times10^{-5}$ (1.22–1.26 of $D$) | $+9.8\times10^{-7}$ … $+8.23\times10^{-6}$ | $\le7\times10^{-12}$ | $+9.5\times10^{-7}$ … $+4.99\times10^{-6}$ | $\le4.2\times10^{-11}$ | yes, all 5 |
| `q-smooth` (5) | $-9.08\times10^{-5}$ … $+7.5\times10^{-7}$ | $\le2.4\times10^{-8}$ | $-1.43\times10^{-4}$ … $+3.72\times10^{-5}$ | $\le6\times10^{-13}$ | $-9.03\times10^{-5}$ … $+1.41\times10^{-4}$ | $\le1.4\times10^{-8}$ | yes, all 5, but see below |

**Errors.** Every entry carries the sum of its cells' declared errors converted to $N$ (§3 Table 2).
These are $10^{-11}$–$10^{-7}$, at least two orders below every term quoted as non-zero here.

- **`together`'s $3.5\times10^{-3}$ at $x_{\rm resp}=980$ is the source spline.** $D = +3.623\times10^{-3}$
  ($\pm3\times10^{-8}$). It splits as:
  - $A_f = B_f = +3.375\times10^{-3}$, **93%**;
  - $A_{\rm Tnum} = +1.373\times10^{-4}$, 3.8%;
  - $A_{\rm TLG} = +1.106\times10^{-4}$, 3.1%;
  - $A_{\rm Trep} = +9.2\times10^{-11}$ and $A_G = +6.0\times10^{-12}$;
  - interaction $9.8\times10^{-10}$.

  It is the interior residual of the $f$ spline on the all-smooth sub-interval. B1 removed the
  spline's end interval at the seam; what is left is the spline's interior.
- **`T-first`:** the same `f` term dominates, at $-1.0\times10^{-5}$ to $-7.4\times10^{-5}$. The
  $T$ terms are a smaller correction of the opposite sign.
- **`q-smooth`:** `f` is absent. Its all-smooth sub-interval ends above the $T$ seam, as B1 found.
  What is left is `Tnum` and `TLG`, which are of opposite sign and largely cancel.
- **The additivity rule passes everywhere, but not by the same margin.** The add-one interaction
  $|\sum A_i - D|$ is 50–62% of the largest $|A_i|$ on every `q-smooth` rung, and 4–7% at
  `together` $10^5$ and $10^7$. The remove-one sum matches $D$ to $\le1.3\times10^{-7}$ on
  `q-smooth`, but on `together` it differs from $D$ by the same amount as the add-one sum.
- **So `TLG` is reported jointly with `Tnum` on `q-smooth`,** and `G`'s remove-one term ($B_G$ up
  to $4.05\times10^{-5}$ at `together` $10^7$) is an interaction, not a $G$ term. A post-run
  probe (§4.6) puts both interactions in one pair of terms each: `TLG`×`Tnum` and `TLG`×`G`.
  Each pair's interaction coincides with a change in the Levin subdivision of 10–20 regions to
  several hundred. That is opened as
  `[13-the-tlg-term-depends-on-the-levin-subdivision]`.
- **No term scales monotonically with $x_{\rm resp}$** over the ladder, except $A_G$ on `q-smooth`
  (slope +0.89, at $\le1.4\times10^{-8}$). Every other term changes sign from rung to rung
  (§4.5).

---

## 2. The controls, and the stationary-point scan

All six of prompt 13 §2.5's checks were saved before the launch, in `b2/out/check_*.txt`, and all
hold. `terms.md` is 13 s older than the run manifest (11:08:19 against 11:08:32 UTC).

1. **Construction**, at every rung, for all 13 objects (208 builds):
   - R's $G$ phase is a `PrimitivePhase` on a `TablePrimitive` with `exact_remainder` on. It
     evaluates at $\log(1+z_{\rm resp})$ and 1–8 ulp below it.
   - Each R $T$ phase is the same kind of object, and each $T$'s Liouville–Green region starts at
     its switch node, below $z_{\rm init}$.
   - Every hybrid's accessors come from the object its delegated set names.
2. **Completeness** at `together` 980, bit for bit in `total`, `total_abserr`, `numeric_quad` and
   `WKB_Levin`:
   - **(i)** the hybrid with nothing delegated equals unwrapped R
     (`total` `0x1.03e5aa4fb72dep-40`);
   - **(ii)** the hybrid with everything delegated equals an independent construction of the exact
     flavour on R's geometry, `exact_with_the_geometry_of(R)` (`0x1.04bc9b11398a2p-40`).
   - **Disjointness:** nine accessors, each in one term.
3. **Determinism.** E, E′ and R at `together` 980, run twice in one process, are bit-identical in
   totals and in every group's region count and depth (7, 7 and 44 regions).
4. **The wrapper is transparent.** R at `together` 980 with and without it is bit-identical.
5. **Controls.**
   - **E against A2's exact closed column** (`REALISTIC-LARGE-X.md` Table 2): within the sum of the
     two declared errors at all 16 rungs. $|\Delta N|\le4.4\times10^{-14}$, against bounds of
     $2.3\times10^{-12}$–$7.9\times10^{-8}$, so $N$ agrees to 13–15 significant figures.
     $N+9/8$ itself is at the rounding level, so its four printed figures agree 0–4 times (4 of 4
     at seven rungs).
   - **Geometry, $N_{E'}-N_E$:** $-3.73\times10^{-14}$ at `together` 980 against
     $1.84\times10^{-11}$. Over the run it is at most $2.7\times10^{-10}$ in magnitude, within the
     declared errors at every rung (§3 Table 2, last column).
   - **R against A4's arm P** (`28ebaf0`, before `levin-phase-floor` 06), recorded and not judged:
     $+3.622668\times10^{-3}$ against $+3.623\times10^{-3}$ at `together` 980. R reproduces arm P's
     $N+9/8$ to the four printed figures of A4 Table H at every rung the two share.
6. **The stationary-point scan.** It covered 108 (rung, Levin sub-interval, phase group) rows, with
   $\theta'$ from the exact closed forms on 4001 points each. **No sign change**, so no cell falls
   under `levin-phase-floor`'s `[06-levin-accepts-a-region-across-a-stationary-point]`.

   | shape | nearest approach | min $\lvert\theta'\rvert$ per $\log(1+z)$ | relative to the largest constituent |
   |---|---|---|---|
   | `together` | 980, $G+q+r$ | 5.67 | 0.152 |
   | `T-first` | $10^7$, $G-q+r$, at $z=2.55\times10^6$ | 0.626 | **0.0226** (the closest anywhere) |
   | `q-smooth` | $10^7$, $G+q+r$ | $1.22\times10^3$ | 0.359 |

---

## 3. The tables

<!-- generated by docs/handover/b2/scripts/b2_tables.py from the run's checkpoint -->

**Table 1 -- N + 9/8 per object and rung**, with each cell's declared error in N (and, for E and R, the reference's own: eq. (22) and the head at 50 digits, negligible against these).

| shape | x_resp | E | R | E' | E'+f | E'+Tnum | E'+Trep | E'+TLG | E'+G | R-f | R-Tnum | R-Trep | R-TLG | R-G |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | +6.9944e-14 (1e-11) | +3.6227e-03 (3e-08) | +3.2641e-14 (9e-12) | +3.3747e-03 (3e-08) | +1.3733e-04 (9e-12) | +9.1534e-11 (9e-12) | +1.1060e-04 (9e-12) | +6.0665e-12 (9e-12) | +2.4792e-04 (9e-12) | +3.4853e-03 (3e-08) | +3.6227e-03 (3e-08) | +3.5121e-03 (3e-08) | +3.6227e-03 (3e-08) |
| together | 10000 | +3.1086e-15 (2e-12) | -1.0349e-04 (3e-09) | +0.0000e+00 (2e-12) | -9.2384e-05 (3e-09) | -8.5269e-06 (2e-12) | +3.8993e-12 (2e-12) | -2.2191e-06 (2e-12) | -4.3165e-13 (2e-12) | -1.1102e-05 (2e-12) | -9.4959e-05 (3e-09) | -1.0349e-04 (3e-09) | -1.0091e-04 (3e-09) | -1.0313e-04 (3e-09) |
| together | 100000 | +1.7231e-13 (2e-11) | +1.4976e-04 (2e-09) | +3.2063e-13 (2e-11) | +1.5298e-04 (2e-09) | +1.8564e-06 (2e-11) | +1.0348e-11 (2e-11) | +5.7593e-06 (2e-11) | +7.5095e-12 (2e-11) | -3.2249e-06 (2e-11) | +1.4790e-04 (2e-09) | +1.4976e-04 (2e-09) | +1.5484e-04 (2e-09) | +1.6060e-04 (2e-09) |
| together | 1e+06 | -1.5614e-12 (2e-10) | -8.7873e-05 (2e-09) | +1.9500e-12 (1e-10) | -7.7258e-05 (2e-09) | -7.8857e-06 (1e-10) | +8.6218e-12 (1e-10) | -2.7215e-06 (1e-10) | -1.3179e-11 (1e-10) | -1.0615e-05 (1e-10) | -7.9987e-05 (2e-09) | -8.7873e-05 (2e-09) | -8.5144e-05 (2e-09) | -8.7865e-05 (2e-09) |
| together | 1e+07 | -3.1011e-10 (1e-08) | -1.0903e-03 (2e-08) | -8.4247e-11 (1e-08) | -1.0485e-03 (2e-08) | -4.8977e-05 (1e-08) | -1.0431e-10 (1e-08) | -3.3309e-05 (1e-08) | -1.5590e-10 (1e-08) | -4.1785e-05 (1e-08) | -1.0413e-03 (2e-08) | -1.0903e-03 (2e-08) | -1.0975e-03 (2e-08) | -1.1308e-03 (2e-08) |
| together | 1e+08 | -1.4614e-10 (4e-08) | -3.2500e-04 (4e-08) | +1.2307e-10 (4e-08) | -3.0701e-04 (4e-08) | -1.7606e-05 (4e-08) | +1.2161e-10 (4e-08) | -9.1981e-06 (4e-08) | -1.4590e-09 (4e-08) | -1.7989e-05 (4e-08) | -3.0740e-04 (4e-08) | -3.2500e-04 (4e-08) | -3.2462e-04 (4e-08) | -3.3382e-04 (4e-08) |
| T-first | 980 | -2.9310e-14 (5e-12) | -6.0420e-05 (4e-09) | -3.7303e-14 (5e-12) | -7.3641e-05 (4e-09) | +8.2334e-06 (5e-12) | -5.3415e-12 (5e-12) | +4.9875e-06 (5e-12) | -2.9843e-13 (5e-12) | +1.3221e-05 (5e-12) | -6.8654e-05 (4e-09) | -6.0420e-05 (4e-09) | -6.5408e-05 (4e-09) | -6.0420e-05 (4e-09) |
| T-first | 10000 | -2.2204e-16 (1e-12) | -1.3323e-05 (3e-09) | +1.1102e-15 (1e-12) | -1.6516e-05 (3e-09) | +1.6948e-06 (1e-12) | -5.3955e-12 (1e-12) | +1.3528e-06 (1e-12) | +6.1950e-14 (1e-12) | +3.1935e-06 (1e-12) | -1.5018e-05 (3e-09) | -1.3323e-05 (3e-09) | -1.4821e-05 (3e-09) | -1.3469e-05 (3e-09) |
| T-first | 100000 | +6.4615e-14 (2e-12) | -1.3810e-05 (3e-09) | +6.5947e-14 (2e-12) | -1.7107e-05 (3e-09) | +1.7626e-06 (2e-12) | -5.3448e-12 (2e-12) | +1.3907e-06 (2e-12) | +1.9451e-13 (2e-12) | +3.2964e-06 (2e-12) | -1.5573e-05 (3e-09) | -1.3810e-05 (3e-09) | -1.5344e-05 (3e-09) | -1.3953e-05 (3e-09) |
| T-first | 1e+06 | -2.3104e-12 (9e-11) | -3.1036e-05 (2e-09) | -2.3079e-12 (9e-11) | -3.7900e-05 (2e-09) | +4.1378e-06 (9e-11) | -8.8209e-12 (9e-11) | +2.7258e-06 (1e-10) | -4.3733e-11 (1e-10) | +6.8636e-06 (1e-10) | -3.5174e-05 (2e-09) | -3.1036e-05 (2e-09) | -3.3762e-05 (2e-09) | -3.1036e-05 (2e-09) |
| T-first | 1e+07 | +4.6736e-12 (3e-10) | -8.1768e-06 (1e-09) | +4.6714e-12 (3e-10) | -1.0288e-05 (1e-09) | +9.8392e-07 (3e-10) | -3.3462e-13 (3e-10) | +9.5288e-07 (3e-10) | +1.5330e-11 (3e-10) | +2.1111e-06 (3e-10) | -9.1607e-06 (1e-09) | -8.1768e-06 (1e-09) | -9.3040e-06 (1e-09) | -8.3512e-06 (1e-09) |
| q-smooth | 980 | -3.6438e-13 (6e-12) | -5.6384e-05 (9e-11) | +4.3299e-14 (6e-12) | +6.4723e-09 (6e-12) | -7.4234e-05 (9e-11) | +3.9613e-13 (6e-12) | +5.4634e-05 (6e-12) | +3.0962e-12 (6e-12) | -5.6391e-05 (8e-11) | +1.7904e-05 (1e-11) | -5.6384e-05 (9e-11) | -7.4227e-05 (9e-11) | -5.6384e-05 (9e-11) |
| q-smooth | 10000 | -1.0880e-13 (1e-11) | -9.0812e-05 (2e-10) | +7.7738e-13 (1e-11) | -4.3596e-09 (2e-11) | -1.4318e-04 (2e-10) | +1.2188e-12 (1e-11) | +1.4111e-04 (1e-11) | -7.1175e-11 (1e-11) | -9.0808e-05 (2e-10) | +5.2242e-05 (2e-11) | -9.0813e-05 (2e-10) | -1.4319e-04 (2e-10) | -9.0812e-05 (2e-10) |
| q-smooth | 100000 | -3.8547e-13 (2e-11) | -1.1355e-05 (1e-10) | +1.2101e-13 (2e-11) | +2.0081e-08 (2e-11) | +1.3311e-05 (1e-10) | -2.4314e-13 (2e-11) | -5.9602e-05 (2e-11) | +2.8616e-10 (2e-11) | -1.1375e-05 (1e-10) | -2.4660e-05 (3e-11) | -1.1355e-05 (1e-10) | +1.3331e-05 (1e-10) | -1.1356e-05 (1e-10) |
| q-smooth | 1e+06 | +2.0028e-13 (6e-10) | -8.7804e-05 (8e-10) | -1.5039e-12 (6e-10) | -3.1614e-09 (6e-10) | -1.3691e-04 (8e-10) | -9.5635e-13 (6e-10) | +1.3265e-04 (6e-10) | +2.7947e-09 (6e-10) | -8.7801e-05 (8e-10) | +4.9108e-05 (6e-10) | -8.7804e-05 (8e-10) | -1.3691e-04 (8e-10) | -8.7807e-05 (8e-10) |
| q-smooth | 1e+07 | -6.2861e-13 (5e-09) | +7.4618e-07 (5e-09) | +2.6839e-12 (5e-09) | +2.3794e-08 (5e-09) | +3.7196e-05 (5e-09) | +2.3452e-12 (5e-09) | -9.0269e-05 (5e-09) | -1.3448e-08 (5e-09) | +7.2239e-07 (5e-09) | -3.6450e-05 (5e-09) | +7.4619e-07 (5e-09) | +3.7206e-05 (5e-09) | +7.5960e-07 (5e-09) |

**Table 2 -- the residue and its terms per rung**, each with the sum of the cells' declared errors in N.

| shape | x_resp | D = N_R - N_E' | A_f | A_Tnum | A_Trep | A_TLG | A_G | sum A - D | B_f | B_Tnum | B_Trep | B_TLG | B_G | sum B - D | additive? | N_E' - N_E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | +3.623e-03 (3e-08) | +3.375e-03 (3e-08) | +1.373e-04 (2e-11) | +9.150e-11 (2e-11) | +1.106e-04 (2e-11) | +6.034e-12 (2e-11) | +9.760e-10 (5e-08) | +3.375e-03 (3e-08) | +1.373e-04 (5e-08) | +9.152e-11 (5e-08) | +1.106e-04 (5e-08) | +6.027e-12 (5e-08) | -9.760e-10 (3e-07) | yes | -3.730e-14 (2e-11) |
| together | 10000 | -1.035e-04 (3e-09) | -9.238e-05 (3e-09) | -8.527e-06 (4e-12) | +3.899e-12 (3e-12) | -2.219e-06 (3e-12) | -4.317e-13 (3e-12) | +3.562e-07 (6e-09) | -9.238e-05 (3e-09) | -8.527e-06 (6e-09) | +5.723e-12 (6e-09) | -2.575e-06 (6e-09) | -3.563e-07 (6e-09) | -3.562e-07 (3e-08) | yes | -3.109e-15 (3e-12) |
| together | 100000 | +1.498e-04 (2e-09) | +1.530e-04 (2e-09) | +1.856e-06 (4e-11) | +1.003e-11 (4e-11) | +5.759e-06 (4e-11) | +7.189e-12 (4e-11) | +1.084e-05 (5e-09) | +1.530e-04 (2e-09) | +1.856e-06 (4e-09) | +1.949e-11 (4e-09) | -5.081e-06 (4e-09) | -1.084e-05 (4e-09) | -1.084e-05 (2e-08) | yes | +1.483e-13 (4e-11) |
| together | 1e+06 | -8.787e-05 (2e-09) | -7.726e-05 (2e-09) | -7.886e-06 (3e-10) | +6.672e-12 (3e-10) | -2.721e-06 (3e-10) | -1.513e-11 (3e-10) | +7.697e-09 (5e-09) | -7.726e-05 (2e-09) | -7.886e-06 (4e-09) | +6.510e-12 (4e-09) | -2.729e-06 (4e-09) | -7.775e-09 (4e-09) | -7.697e-09 (2e-08) | yes | +3.511e-12 (3e-10) |
| together | 1e+07 | -1.090e-03 (3e-08) | -1.049e-03 (3e-08) | -4.898e-05 (2e-08) | -2.007e-11 (2e-08) | -3.331e-05 (2e-08) | -7.165e-11 (2e-08) | -4.050e-05 (1e-07) | -1.049e-03 (3e-08) | -4.898e-05 (4e-08) | -4.811e-11 (4e-08) | +7.192e-06 (4e-08) | +4.050e-05 (4e-08) | +4.050e-05 (2e-07) | yes | +2.259e-10 (2e-08) |
| together | 1e+08 | -3.250e-04 (8e-08) | -3.070e-04 (8e-08) | -1.761e-05 (8e-08) | -1.455e-12 (8e-08) | -9.198e-06 (8e-08) | -1.582e-09 (8e-08) | -8.816e-06 (5e-07) | -3.070e-04 (8e-08) | -1.761e-05 (9e-08) | -6.402e-12 (9e-08) | -3.822e-07 (9e-08) | +8.814e-06 (9e-08) | +8.816e-06 (5e-07) | yes | +2.692e-10 (8e-08) |
| T-first | 980 | -6.042e-05 (4e-09) | -7.364e-05 (4e-09) | +8.233e-06 (1e-11) | -5.304e-12 (1e-11) | +4.988e-06 (1e-11) | -2.611e-13 (1e-11) | +6.888e-11 (8e-09) | -7.364e-05 (4e-09) | +8.233e-06 (8e-09) | -7.175e-12 (8e-09) | +4.987e-06 (8e-09) | +2.287e-14 (8e-09) | -6.888e-11 (4e-08) | yes | -7.994e-15 (1e-11) |
| T-first | 10000 | -1.332e-05 (3e-09) | -1.652e-05 (3e-09) | +1.695e-06 (2e-12) | -5.397e-12 (2e-12) | +1.353e-06 (2e-12) | +6.084e-14 (2e-12) | -1.459e-07 (7e-09) | -1.652e-05 (3e-09) | +1.695e-06 (7e-09) | -5.347e-12 (7e-09) | +1.499e-06 (7e-09) | +1.459e-07 (7e-09) | +1.459e-07 (3e-08) | yes | +1.332e-15 (2e-12) |
| T-first | 100000 | -1.381e-05 (3e-09) | -1.711e-05 (3e-09) | +1.763e-06 (4e-12) | -5.411e-12 (4e-12) | +1.391e-06 (4e-12) | +1.286e-13 (4e-12) | -1.431e-07 (7e-09) | -1.711e-05 (3e-09) | +1.763e-06 (7e-09) | -5.284e-12 (7e-09) | +1.534e-06 (7e-09) | +1.432e-07 (7e-09) | +1.431e-07 (3e-08) | yes | +1.332e-15 (4e-12) |
| T-first | 1e+06 | -3.104e-05 (2e-09) | -3.790e-05 (2e-09) | +4.138e-06 (2e-10) | -6.513e-12 (2e-10) | +2.726e-06 (2e-10) | -4.143e-11 (2e-10) | -8.439e-11 (4e-09) | -3.790e-05 (2e-09) | +4.138e-06 (3e-09) | -6.693e-12 (3e-09) | +2.726e-06 (3e-09) | +8.027e-11 (3e-09) | +8.438e-11 (2e-08) | yes | +2.442e-15 (2e-10) |
| T-first | 1e+07 | -8.177e-06 (1e-09) | -1.029e-05 (1e-09) | +9.839e-07 (7e-10) | -5.006e-12 (7e-10) | +9.529e-07 (7e-10) | +1.066e-11 (7e-10) | -1.743e-07 (5e-09) | -1.029e-05 (1e-09) | +9.839e-07 (2e-09) | -4.971e-12 (2e-09) | +1.127e-06 (2e-09) | +1.743e-07 (2e-09) | +1.743e-07 (1e-08) | yes | -2.220e-15 (7e-10) |
| q-smooth | 980 | -5.638e-05 (9e-11) | +6.472e-09 (1e-11) | -7.423e-05 (9e-11) | +3.528e-13 (1e-11) | +5.463e-05 (1e-11) | +3.053e-12 (1e-11) | +3.679e-05 (2e-10) | +6.472e-09 (2e-10) | -7.429e-05 (1e-10) | +1.276e-10 (2e-10) | +1.784e-05 (2e-10) | +3.627e-12 (2e-10) | -5.471e-08 (9e-10) | yes | +4.077e-13 (1e-11) |
| q-smooth | 10000 | -9.081e-05 (2e-10) | -4.360e-09 (3e-11) | -1.432e-04 (2e-10) | +4.414e-13 (2e-11) | +1.411e-04 (2e-11) | -7.195e-11 (2e-11) | +8.873e-05 (5e-10) | -4.360e-09 (4e-10) | -1.431e-04 (2e-10) | +1.912e-10 (4e-10) | +5.238e-05 (4e-10) | -6.195e-11 (4e-10) | +1.301e-07 (2e-09) | yes | +8.862e-13 (2e-11) |
| q-smooth | 100000 | -1.136e-05 (1e-10) | +2.008e-08 (4e-11) | +1.331e-05 (1e-10) | -3.642e-13 (4e-11) | -5.960e-05 (4e-11) | +2.860e-10 (5e-11) | -3.492e-05 (4e-10) | +2.008e-08 (2e-10) | +1.330e-05 (1e-10) | +9.946e-12 (2e-10) | -2.469e-05 (2e-10) | +2.862e-10 (2e-10) | -6.308e-09 (1e-09) | yes | +5.065e-13 (4e-11) |
| q-smooth | 1e+06 | -8.780e-05 (1e-09) | -3.160e-09 (1e-09) | -1.369e-04 (1e-09) | +5.476e-13 (1e-09) | +1.327e-04 (1e-09) | +2.796e-09 (1e-09) | +8.354e-05 (8e-09) | -3.160e-09 (2e-09) | -1.369e-04 (1e-09) | +1.811e-10 (2e-09) | +4.911e-05 (2e-09) | +2.793e-09 (2e-09) | -9.234e-12 (1e-08) | yes | -1.704e-12 (1e-09) |
| q-smooth | 1e+07 | +7.462e-07 (1e-08) | +2.379e-08 (1e-08) | +3.720e-05 (1e-08) | -3.386e-13 (1e-08) | -9.027e-05 (1e-08) | -1.345e-08 (1e-08) | -5.381e-05 (6e-08) | +2.379e-08 (1e-08) | +3.720e-05 (1e-08) | -1.132e-11 (1e-08) | -3.646e-05 (1e-08) | -1.342e-08 (1e-08) | +8.490e-11 (7e-08) | yes | +3.312e-12 (1e-08) |

**Table 3 -- the cells not computed** (`not reached`: killed over the 1 h budget, with its elapsed time; `raised` and `skipped` with the recorded reason).

| shape | x_resp | object | status | elapsed / reason |
|---|---|---|---|---|
| -- | -- | -- | none | -- |

**Table 4 -- cost of R per rung**, recorded without a reading: Levin regions, maximum depth, integral time and the 1-minute load at the cell's start.

| shape | x_resp | Levin regions | max depth | integral_time | load |
|---|---|---|---|---|---|
| together | 980 | 44 | 5 | 0.88 s | 21.2 |
| together | 10000 | 66 | 6 | 1.15 s | 20.1 |
| together | 100000 | 80 | 8 | 1.35 s | 10.7 |
| together | 1e+06 | 102 | 9 | 1.71 s | 7.1 |
| together | 1e+07 | 313 | 14 | 4.82 s | 9.4 |
| together | 1e+08 | 286 | 13 | 4.51 s | 14.7 |
| T-first | 980 | 34 | 7 | 0.53 s | 16.3 |
| T-first | 10000 | 44 | 7 | 0.74 s | 14.7 |
| T-first | 100000 | 53 | 7 | 0.87 s | 9.4 |
| T-first | 1e+06 | 46 | 7 | 0.85 s | 6.0 |
| T-first | 1e+07 | 114 | 9 | 1.63 s | 8.2 |
| q-smooth | 980 | 686 | 11 | 5.32 s | 12.7 |
| q-smooth | 10000 | 1190 | 14 | 9.80 s | 12.1 |
| q-smooth | 100000 | 1204 | 14 | 10.16 s | 7.8 |
| q-smooth | 1e+06 | 1478 | 14 | 13.00 s | 9.1 |
| q-smooth | 1e+07 | 1244 | 13 | 11.18 s | 7.7 |

**Table 5 -- each term per shape over the ladder**: the range of A_i (and of D), its signs, the largest |A_i| / |D| over the rungs, and a least-squares slope of log|A_i| against log x_resp where |A_i| is monotone in x_resp (A2's Table 4 convention); '--' where it is not.

| shape | quantity | min | max | signs | max over rungs of abs(A_i)/abs(D) | monotone? | slope |
|---|---|---|---|---|---|---|---|
| together | D | -1.090e-03 | +3.623e-03 | +-+--- |  | no | -- |
| together | A_f | -1.049e-03 | +3.375e-03 | +-+--- | 1.02 | no | -- |
| together | A_Tnum | -4.898e-05 | +1.373e-04 | +-+--- | 0.0897 | no | -- |
| together | A_Trep | -2.007e-11 | +9.150e-11 | ++++-- | 7.59e-08 | no | -- |
| together | A_TLG | -3.331e-05 | +1.106e-04 | +-+--- | 0.0385 | no | -- |
| together | A_G | -1.582e-09 | +7.189e-12 | +-+--- | 4.87e-06 | no | -- |
| T-first | D | -6.042e-05 | -8.177e-06 | ----- |  | no | -- |
| T-first | A_f | -7.364e-05 | -1.029e-05 | ----- | 1.26 | no | -- |
| T-first | A_Tnum | +9.839e-07 | +8.233e-06 | +++++ | 0.136 | no | -- |
| T-first | A_Trep | -6.513e-12 | -5.006e-12 | ----- | 6.12e-07 | no | -- |
| T-first | A_TLG | +9.529e-07 | +4.988e-06 | +++++ | 0.117 | no | -- |
| T-first | A_G | -4.143e-11 | +1.066e-11 | -++-+ | 1.33e-06 | no | -- |
| q-smooth | D | -9.081e-05 | +7.462e-07 | ----+ |  | no | -- |
| q-smooth | A_f | -4.360e-09 | +2.379e-08 | +-+-+ | 0.0319 | no | -- |
| q-smooth | A_Tnum | -1.432e-04 | +3.720e-05 | --+-+ | 49.8 | no | -- |
| q-smooth | A_Trep | -3.642e-13 | +5.476e-13 | ++-+- | 4.54e-07 | no | -- |
| q-smooth | A_TLG | -9.027e-05 | +1.411e-04 | ++-+- | 121 | no | -- |
| q-smooth | A_G | -1.345e-08 | +2.796e-09 | +-++- | 0.018 | increasing | +0.89 |

**Table 6 -- the interactions against the additivity rule** (the rule: not additive where max(abs(sum A - D), abs(sum B - D)) > max_i abs(A_i)).

| shape | x_resp | abs(sum A - D) | abs(sum B - D) | max_i abs(A_i) | ratio | verdict |
|---|---|---|---|---|---|---|
| together | 980 | 9.760e-10 | 9.760e-10 | 3.375e-03 | 0.000 | yes |
| together | 10000 | 3.562e-07 | 3.562e-07 | 9.238e-05 | 0.004 | yes |
| together | 100000 | 1.084e-05 | 1.084e-05 | 1.530e-04 | 0.071 | yes |
| together | 1e+06 | 7.697e-09 | 7.697e-09 | 7.726e-05 | 0.000 | yes |
| together | 1e+07 | 4.050e-05 | 4.050e-05 | 1.049e-03 | 0.039 | yes |
| together | 1e+08 | 8.816e-06 | 8.816e-06 | 3.070e-04 | 0.029 | yes |
| T-first | 980 | 6.888e-11 | 6.888e-11 | 7.364e-05 | 0.000 | yes |
| T-first | 10000 | 1.459e-07 | 1.459e-07 | 1.652e-05 | 0.009 | yes |
| T-first | 100000 | 1.431e-07 | 1.431e-07 | 1.711e-05 | 0.008 | yes |
| T-first | 1e+06 | 8.439e-11 | 8.438e-11 | 3.790e-05 | 0.000 | yes |
| T-first | 1e+07 | 1.743e-07 | 1.743e-07 | 1.029e-05 | 0.017 | yes |
| q-smooth | 980 | 3.679e-05 | 5.471e-08 | 7.423e-05 | 0.496 | yes |
| q-smooth | 10000 | 8.873e-05 | 1.301e-07 | 1.432e-04 | 0.620 | yes |
| q-smooth | 100000 | 3.492e-05 | 6.308e-09 | 5.960e-05 | 0.586 | yes |
| q-smooth | 1e+06 | 8.354e-05 | 9.234e-12 | 1.369e-04 | 0.610 | yes |
| q-smooth | 1e+07 | 5.381e-05 | 8.490e-11 | 9.027e-05 | 0.596 | yes |

208 records; git_head ['28ade78c3dd038f2d84e664918b78a318a4a150d']; driver sha256 ['dc8772e3c7b5dfe345296abc1dfffa78d303afc6de179149028920289d6d2278']

---

## 4. The attribution

### 4.1 `f`: the source spline

$A_f = B_f$ to within the declared errors at every rung. `f` enters only on the all-smooth
sub-interval, where nothing else realistic is read except `numeric_Gk`, which is exact, so it
cannot interact.

| shape | $\lvert A_f\rvert$ | share of $D$ | sign |
|---|---|---|---|
| `together` | $7.7\times10^{-5}$–$3.4\times10^{-3}$ | 0.88–1.02 | follows $D$: $+-+---$ |
| `T-first` | $1.0\times10^{-5}$–$7.4\times10^{-5}$ | 1.22–1.26 (the $T$ terms offset it) | negative throughout |
| `q-smooth` | $\le2.4\times10^{-8}$ | — | — |

It is not monotone in $x_{\rm resp}$, so no slope is fitted.

**`[06-source-spline-residual-vs-handover]`, on this fixture.** After B1 removed the end interval,
the interior residual of the spline of $f$ on the 100-per-decade grid is **the dominant term of
the seam's residue** in two of three shapes: 88–102% of $D$ on `together` and 122–126% on
`T-first`. That is up to $3.4\times10^{-3}$ in $N$. It is the 4.5e-04-of-envelope floor that entry
records, integrated: $f$ oscillates at twice the $T$ phase, and 100 per decade is too few nodes per
half-cycle near the seam.

### 4.2 `Tnum`: the numeric $T$ branch

$A_{\rm Tnum} = B_{\rm Tnum}$ to within $6\times10^{-8}$ at every rung.

| shape | range of $A_{\rm Tnum}$ | relation to $D$ |
|---|---|---|
| `together` | $-4.9\times10^{-5}$ … $+1.4\times10^{-4}$ | 1.2–9% of $D$, the same sign |
| `T-first` | $+1.0\times10^{-6}$ … $+8.2\times10^{-6}$ | opposite sign to $D$, 12–14% of $\lvert D\rvert$ |
| `q-smooth` | $-1.43\times10^{-4}$ … $+3.7\times10^{-5}$ | the largest single term, up to 50× $D$, cancelled by `TLG` |

**`[05-numeric-region-is-now-the-accuracy-floor]`, on this fixture.** With the numeric spline
running past the switch node since B1, its interior error integrated over the Levin sub-intervals
where that $T$ is smooth is $10^{-6}$–$1.4\times10^{-4}$ in $N$. It dominates nothing except
`q-smooth`. There $T_q$ is smooth over the whole range and its spline enters every Levin
sub-interval. `source_samples_per_log10z` remains the lever.

### 4.3 `TLG`: the Liouville–Green closed forms

| shape | $A_{\rm TLG}$ | $B_{\rm TLG}$ |
|---|---|---|
| `together` | $-3.3\times10^{-5}$ … $+1.1\times10^{-4}$ | $-5.1\times10^{-6}$ … $+1.1\times10^{-4}$ |
| `T-first` | $+9.5\times10^{-7}$ … $+5.0\times10^{-6}$ | close to $A_{\rm TLG}$ |
| `q-smooth` | $-9.0\times10^{-5}$ … $+1.41\times10^{-4}$ | $-3.6\times10^{-5}$ … $+5.2\times10^{-5}$ |

**The two disagree where the interactions are**, so the per-term value of `TLG` is not
well-defined on `q-smooth` (§4.6).

**`[07-lg-derivative-truncation-at-handover]` and `[00-tk-lg-truncation-floor]`, on this fixture.**
The derivative pieces (`omega`, `dlnM_dz`), at the fixture's fixed hand-over $x_T\approx19.1$,
contribute $10^{-6}$–$1.4\times10^{-4}$ in $N$. That is the same order as `Tnum`, and below `f`
wherever `f` is present. This measures the derivative half only. The fixture's stored amplitude and
phase are the **exact** Bessel ones, so the amplitude and phase truncation of
`[00-tk-lg-truncation-floor]` (the $x_i^{-3}$ law) is **not** measured here. Nothing here is a
depth scan; that is E2's.

### 4.4 `Trep` and `G`: the stored WKB representations

$|A_{\rm Trep}|\le9.2\times10^{-11}$ over the whole ladder; $|A_G|\le1.6\times10^{-9}$ on
`together` and `T-first` and $\le1.4\times10^{-8}$ on `q-smooth`. At every rung that is three to
eight orders below the largest term. With `levin-phase-floor` 05a and 06 and Part A, $T$'s stored phase
(`PrimitivePhase` on a table `cs_tau`, `exact_remainder` on) and $G$'s (`_build_phase`) contribute
nothing measurable at these $x$. The amplitude halves (`bessel_phase`'s modulus in both) are
included in these numbers. **`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`'s
question is answered:** A2's representation term was an upper bound by the `phase_spline`, and $G$'s
construction is not where the residue lives.

### 4.5 Scaling with $x_{\rm resp}$

Unlike A2's Table 4 representation term (slopes −0.06 to −0.15), no term here is monotone over the
ladder except $A_G$ on `q-smooth` (slope +0.89, at $\le1.4\times10^{-8}$; §3 Table 5). The large
terms change sign from rung to rung on `together` ($+-+---$) and `q-smooth`, with no trend in
magnitude. This is consistent with a seam error whose sign follows the phase of the integrand at
the hand-over relative to the response. It is not a fitted law, and none is given.

### 4.6 The interactions, and a post-run probe

The additivity rule fixed before the run passes at every rung. Two interactions are nonetheless
visible:
- on `q-smooth`, $\sum A_i - D$ is 0.50–0.62 of the largest $|A_i|$, with $\sum B_i - D\le1.3\times10^{-7}$;
- on `together` $10^5$, $10^7$ and $10^8$, $B_G$ differs from $A_G$ by $1.1\times10^{-5}$,
  $4.05\times10^{-5}$ and $8.8\times10^{-6}$.

A **post-run probe**, not part of the registered run (`b2/scripts/b2_probe_pairs.py`,
`b2/out/probe_pairs.txt`), took pairwise hybrids $E'+i+j$, interaction
$I_{ij} = (N_{E'+i+j}-N_{E'}) - A_i - A_j$:

| place | $I_{\rm TLG,G}$ | $I_{\rm TLG,Tnum}$ | $I_{\rm TLG,f}$, $I_{\rm TLG,Trep}$ |
|---|---|---|---|
| `q-smooth` 980 | $-3.674\times10^{-5}$ | $-3.679\times10^{-5}$ | $0$ and $3\times10^{-15}$ |
| `q-smooth` $10^7$ | $+5.381\times10^{-5}$ | $+5.381\times10^{-5}$ | — |
| `together` $10^7$ | $+4.050\times10^{-5}$ | — | — |

At `together` $10^7$, $I_{G,f}$, $I_{G,\rm Tnum}$ and $I_{G,\rm Trep}$ are all at most
$3\times10^{-11}$.

**Two things are measured:**
- the interaction is `TLG` with **either** `G` **or** `Tnum` realistic, and it is the same size
  for both;
- `G` alone and `Tnum` alone do one common thing to the cell's integration: they raise its Levin
  region count. At `q-smooth` 980, E′+TLG uses 10 regions and R−TLG 674. At `together` $10^7$, 9
  against 304 (§3, and the cells' records).

A plausible mechanism, **not demonstrated here**: the Levin driver is given $T$'s phase derivative
as `omega`, the Liouville–Green closed form, while the phase it oscillates with is the stored one.
On this fixture the stored phase is the exact Bessel phase. If the two disagree by the
Liouville–Green truncation, the Levin result depends on how finely the range is cut, so `TLG`'s
contribution changes with whatever else drives the subdivision.

**Consequence for this document:** on `q-smooth`, and for $B_G$ on `together`'s upper rungs, the
attribution between `TLG`, `Tnum` and `G` is reported jointly, not per term. Opened as
`[13-the-tlg-term-depends-on-the-levin-subdivision]`.


### 4.7 After the run: §4.6's mechanism confirmed (2026-10-27)

*Added after this document was written, from a laptop probe outside the registered run:*
`b2/scripts/probe13.py`, outputs in `b2/probe13/`. It changes none of §3's numbers.

§4.6's proposed mechanism holds. Taking the Levin driver's $T$ phase derivative from the phase
it oscillates with (`phase.theta_deriv`), instead of from `omega`, makes $A_{\rm TLG} = B_{\rm TLG}$
to $\le4\times10^{-10}$ at `q-smooth` 980 and $10^7$ and `together` 980 and $10^7$. Letting the
driver differentiate $\theta$ itself gives the same values at the two `q-smooth` places. On this
fixture the realistic $T$'s stored phase is the exact Bessel phase while `omega` and `dlnM_dz` are
Liouville–Green, so the two disagree by construction. Production's stored phase is integrated
from the same $\omega_{\rm eff}$, so it is consistent by construction.

| place | $A_{\rm TLG}$, as run | $B_{\rm TLG}$, as run | $A_{\rm TLG} = B_{\rm TLG}$, derivative from the phase | $N_R+9/8$, as run | $N_R+9/8$, derivative from the phase |
|---|---|---|---|---|---|
| `q-smooth` 980 | $+5.463\times10^{-5}$ | $+1.784\times10^{-5}$ | $+1.784\times10^{-5}$ | $-5.638\times10^{-5}$ | $-5.638\times10^{-5}$ |
| `q-smooth` $10^7$ | $-9.027\times10^{-5}$ | $-3.646\times10^{-5}$ | $-3.650\times10^{-5}$ | $+7.46\times10^{-7}$ | $+7.08\times10^{-7}$ |
| `together` $10^7$ | $-3.331\times10^{-5}$ | $+7.19\times10^{-6}$ | $+7.79\times10^{-6}$ | $-1.0903\times10^{-3}$ | $-1.0897\times10^{-3}$ |
| `together` 980 | $+1.106\times10^{-4}$ | $+1.106\times10^{-4}$ | $-4.07\times10^{-5}$ | $+3.623\times10^{-3}$ | $+3.471\times10^{-3}$ |

So this document's `TLG` column, and the interactions §4.6 reports jointly, carry a property of
the fixture, not of production: up to about $1.5\times10^{-4}$ in $N$. **The `f` attribution is
unaffected:** `f` is read only on the all-smooth sub-interval, where no Levin call is made, and it
is 97 % of `together`'s residue at 980 once the derivative is made consistent
($3.375\times10^{-3}$ of $3.471\times10^{-3}$; 93 % as run). `[13-the-tlg-term-depends-on-the-levin-subdivision]`
carries the next step: make the fixture's $T$ self-consistent before E1.
---

## 5. Against the earlier records

Context, each on its own tree, not columns of this measurement.

- **A2's representation term** (`REALISTIC-LARGE-X.md` §4, pre-B1 tree `c414451`) was
  $4.08\times10^{-5}$–$2.68\times10^{-3}$. It included the two splines' end intervals at the seam
  and A2's `phase_spline` $G$. B1 removed the first; §4.4 shows the second contributes
  $\le1.6\times10^{-9}$.
- **HANDOVER-B1.md §2.2** (B1 at 980, still with the `phase_spline` $G$ and a closed-form `cs_tau`):

  | shape | B1 at 980 | R here |
  |---|---|---|
  | `together` | $+3.470\times10^{-3}$ | $+3.623\times10^{-3}$ |
  | `T-first` | $-6.04\times10^{-5}$ | $-6.042\times10^{-5}$ |
  | `q-smooth` | $-5.41\times10^{-5}$ | $-5.638\times10^{-5}$ |

  B1's "+5.1e-3 from the source overlap" on `together` was the end interval. This document's
  $A_f = 3.375\times10^{-3}$ is the interior that remains.
- **A4 Table H**, arm P at `28ebaf0` (before `levin-phase-floor` 06): R equals arm P to every
  printed figure at all 15 rungs the two share. So 06 and Part A together moved nothing visible in
  $N$, which matches §4.4.

---

## 6. Cost

Recorded without a reading. R per rung (§3 Table 4) beside A4's arm P (A4 Table D):

| | Levin regions | maximum depth | integral |
|---|---|---|---|
| `together` | 44→286 | 5–14 | 0.9–4.8 s |
| `T-first` | 34→114 | 7–9 | 0.5–1.6 s |
| `q-smooth` | 686→1,244 | 11–14 | 5.3–13.0 s |

These match arm P's counts (A4: 46→288, 34→112, 686→1,480 over its shorter `q-smooth` ladder). The
whole run took 456 s of cell wall time, in 10 minutes. `q-smooth` at $10^7$, the rung neither A2
nor A4 reached in the realistic flavour, took 11.2 s.

---

## 7. What this does not cover

- **$b = 0$ only.** A1's oracle makes $b=0.2$ possible; it was not B2's configuration.
- **The hand-over at the fixture's fixed $x_T\approx19.1$** (3.5 e-folds inside the horizon), so
  nothing here is a depth scan. That is E2's. The `f`, `Tnum` and `TLG` terms all depend on it.
- **The fixture's uniform 100-per-decade grid,** not production's curvature-criterion grid
  (`qcd-background-audit` prompt 15). `f` and `Tnum` are spline terms and depend on it directly.
- **$G$'s numeric branch is exact** (`compute_analytic_G`), and its crossover sits at the shape's
  `G_cross_x`, not at production's super-horizon node
  (`[03-the-g-crossover-is-one-super-horizon-node-on-every-complete-row]`). So D1's consumer
  spline is not measured.
- **$G$'s amplitude comes from `bessel_phase`,** not from a spline of a stored amplitude.
- **$f$ is splined from exact samples,** not from samples of the stored $T$.
- **The stored $T$ amplitude and phase are exact Bessel,** not Liouville–Green-integrated, so
  `[00-tk-lg-truncation-floor]`'s amplitude and phase truncation is absent (§4.3).
- **$x_{\rm resp}\le10^8$,** not production's $\sim4\times10^{12}$.
- **The Levin driver before `levin-phase-floor` 07.**
- **The interactions of §4.6 are diagnosed, not explained.** On `q-smooth`, `TLG`'s per-term
  value is not given.
