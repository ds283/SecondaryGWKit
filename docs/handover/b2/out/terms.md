# B2 — the five terms and the additivity rule (prompt 13 §2.2, §2.3, §2.5 item 7)

Written after checks 1–6 (`state/check_*.txt`, all HOLD) and **before** the launch. This file's
mtime against the run manifest's is the review's evidence that the terms were fixed before the
run. The mapping below is the one coded in `scripts/b2_cells.py` (`TERMS`, `T_ACCESSORS`,
`G_ACCESSORS`, `ACCESSORS`, `apply_hybrid`). It was written before any number was computed, and
is recorded by hash in `state/mapping_before_smoke.txt`. A smoke test of all 13 objects at
`together` 980 then ran before this file was written; the log says so. The mapping has not
changed since.

## Objects

| object | construction |
|---|---|
| E | `Case(exact=True)`: the exact flavour on its own geometry (`ExactTkFunctions` break at $z_{\rm init}$) |
| R | `Case(exact=False)` at `28ade78`, unwrapped: production's construction and seam |
| E′ | R's `Case` with all five terms delegated (`apply_hybrid(case, all)`) |
| E′+i | R's `Case` with every term but $i$ delegated |
| R−i | R's `Case` with term $i$ delegated |

**Delegation** swaps accessors only. It applies to $T_q$ and $T_r$ together, wrapping each
`TkSourceFunctions` in `HybridT`. R's regions, switch nodes, $z_{\rm init}$, hand-over record and
partition are kept.

## Which accessor each regime reads, at `HEAD` (read from the code)

| where | code | accessors read |
|---|---|---|
| all-smooth sub-interval (`method == "quad"`) | `numeric_quad_integral` | `source.functions.source` (f), `Gk.numeric_Gk` |
| Levin sub-interval, $G$ smooth | `phase_groups._SmoothG` | `Gk.numeric_Gk` |
| Levin sub-interval, $G$ oscillatory | `_OscillatoryG` | `Gk.sin_amplitude`; `Gk.phase.raw_theta`, `.theta_mod_2pi`, `.theta_deriv` (the Levin phase derivative) |
| Levin sub-interval, $T$ smooth | `_SmoothT` | `T.T`, `T.dT_dz` |
| Levin sub-interval, $T$ oscillatory | `_OscillatoryT` | `T.M`, `T.dlnM_dz`, `T.omega`; `T.phase.raw_theta`, `.theta_mod_2pi`. **The Levin phase derivative of $T$ is `T.omega`**, not `phase.theta_deriv` |
| partition only (no values) | `build_partition`, `evaluate_QuadSource_integral` | `crossover_z`, `z_init`, `numeric_region`, `WKB_region`, `handover_continuity`, `Gk.numeric_region`, `Gk.WKB_region`, `source.numeric_region`, `crossover_z_q/r`; kept from R in every object |

Some of these are read in more than one regime or belong to no term:
- `TkSourceFunctions.T` and `dT_dz` **never dispatch to the WKB form**. They return 1 and 0
  above the numeric grid and the numeric spline within it, so name-delegation is
  region-delegation: `Tnum` is every smooth-$T$ evaluation, including the exact-unity region above
  the grid.
- The source spline is read **only** on the all-smooth sub-interval. A Levin sub-interval with
  both $T$ smooth recomputes $f$ from `T.T` and `T.dT_dz` (`Tnum`).
- `Gk.numeric_Gk` is `compute_analytic_G` in R and in E alike (`BesselPhaseGk.numeric_Gk`,
  `ExactGk.numeric_Gk`). It is not a term, and its delegation would be the identity.

## The five terms (the final mapping)

| term | accessors made exact when delegated | regions where they are read | exact counterpart |
|---|---|---|---|
| `f` | `QuadSource.functions.source` | the all-smooth sub-interval only | `Case.f_exact`, through `exp(x) - 1` exactly as `FakeQuadSource(exact=True)` |
| `Tnum` | `T.T`, `T.dT_dz` (both $T$) | Levin sub-intervals where that $T$ is smooth: from $z_{\rm source,max}$ down to its switch node, including $T = 1$ above its numeric grid | `test_phase_groups.ExactTk` |
| `Trep` | `T.M`, `T.phase` (both $T$) | Levin sub-intervals where that $T$ is oscillatory: below its switch node | `ExactTk` (`phase` is an `_ExactPhase`) |
| `TLG` | `T.omega`, `T.dlnM_dz` (both $T$) | the same, and `omega` also as the Levin phase derivative of every group containing that $T$ | `ExactTk` |
| `G` | `Gk.phase`, `Gk.sin_amplitude` | Levin sub-intervals where $G$ is oscillatory: below `G_crossover_z` | `test_phase_groups.ExactGk`, by `GkSourceFunctions._replace` |

**Disjointness**, by listing (`state/check_completeness.txt`): nine accessors, each in exactly one
term. They are `QuadSource.functions.source` (f), `T.T` and `T.dT_dz` (Tnum), `T.M` and
`T.phase` (Trep), `T.omega` and `T.dlnM_dz` (TLG), `Gk.phase` and `Gk.sin_amplitude` (G).

**Completeness**, bit for bit at `together` 980 (`state/check_completeness.txt`):
- **(i) E′ + all five ≡ R:** the hybrid with no term delegated reproduces unwrapped R in `total`,
  `total_abserr`, `numeric_quad` and `WKB_Levin`.
- **(ii) R − all five ≡ E′:** the hybrid with every term delegated reproduces an independent
  construction of the exact flavour on R's geometry, `exact_with_the_geometry_of(R)`, in all four.

So no accessor that bears on a value lies outside the five terms. **No term needed a production
change to isolate.**

## What is computed, and the additivity rule (fixed here, before the run)

At each rung, with $N = K\,{\rm total}/{\rm predicted}$, $K = -9/8$ (A2's `N_of`):

| quantity | definition |
|---|---|
| residue | $D = N_R - N_{E'}$ |
| add-one term | $A_i = N_{E'+i} - N_{E'}$ |
| remove-one term | $B_i = N_R - N_{R-i}$ |
| interactions | $\sum_i A_i - D$ and $\sum_i B_i - D$ |
| geometry control | $N_{E'} - N_E$ |

- **Errors.** Each quantity carries the sum of the cells' declared errors converted to $N$,
  $|N|\cdot{\rm declared}$ per cell (A2's `term`), where
  ${\rm declared} = \max({\rm total\_abserr},\ {\rm REF\_RTOL}\,(|{\rm numeric\_quad}| + |{\rm WKB\_Levin}|))/|{\rm total}|$.
- **The reference.** It is common to every cell of a rung and cancels in every difference, so it
  adds nothing to these errors. $N_R$ and $N_E$ against the reference carry its own error beside
  them (README §5 rule 8).
- **Additivity.** **A rung where $|\sum_i A_i - D|$ or $|\sum_i B_i - D|$ exceeds the largest
  $|A_i|$ is reported *not additive at that rung*.** No per-term attribution is given there
  without saying so. There is no other threshold.
- **Missing cells.** A rung with any of its 13 cells `not reached` gives only the quantities whose
  cells exist, and no additivity verdict.
