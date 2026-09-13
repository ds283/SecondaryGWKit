# Prototype measurements — the shared conformal-time primitive

**Campaign:** [`prompts/GkTk-remedial/README.md`](../../prompts/GkTk-remedial/README.md) · **Prompt:** 01
**Date:** 2026-09-10 · **Tree:** `gktk-remedial` branched from `main` at `9ff59d5`
**Scripts:** [`prototype_primitive.py`](prototype_primitive.py), [`generate_references.py`](generate_references.py),
[`baseline_k1e5.py`](baseline_k1e5.py), [`reference_lib.py`](reference_lib.py)
**Environment:** Python 3.12, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, macOS/arm64

This document records what the prompt-01 prototype measured. It is additive: a later re-run adds
a new dated section rather than editing these tables.

---

## 0. Headlines

1. **The design reproduces review §7 on `LambdaCDM`, to the digit.** Gauss–Legendre order 4 on the
   production grid gives $\tau$ at the nodes to $1.81\times10^{-15}$ relative in 6,924 Hubble
   evaluations and 0.04 s; off-grid, nearest-node-plus-8-point-Gauss gives $5.8\times10^{-16}$
   against a cubic spline of the same nodes at $7.3\times10^{-10}$ — a factor of $1.3\times10^{6}$,
   and 3,080 rad against $2.5\times10^{-3}$ rad at $k=3\times10^8$.
2. **Two production intervals of `QCD_Cosmology` defeat any fixed-order Gauss rule**, and they are
   not spline knots: they are the two branch boundaries of `QCD_EOS.G(T)`, at
   $T=T_{\rm LO}=10^{-5}$ GeV ($z\approx4.19\times10^7$) and $T=T_{120\,{\rm MeV}}=0.12$ GeV
   ($z\approx8.58\times10^{11}$). At order 4 the worse of the two carries $6.3\times10^{-5}$
   relative, i.e. $1.6\times10^{-8}$ Mpc absolute — **4.8 rad at $k=3\times10^8$** for any phase
   difference straddling it. Order 20 only reaches 0.26 rad. **This is prompt 02's question and it
   is answered "no" for a fixed-order rule on this grid.** (§3)
3. **The interval accessor costs 102 µs per call on `QCD_Cosmology` when both endpoints are
   off-grid — above README §4.3's 50 µs stop threshold.** But the production case is not that
   case: the background model is built on the source grid (`main.py:476`) and every WKB sample is
   a node, so both endpoints are on-grid and the call costs **4.4 µs and zero Hubble
   evaluations**. Only the per-object anchor $z_{\rm init}$ is off-grid, one endpoint per object.
   (§4)
4. **The double-double table earns its keep only where $\tau$ is large.** Over one production
   grid interval it is indistinguishable from the single-double control at $z\sim10^6$ (where
   $\tau\approx0.046$ Mpc) and 4–60× better at $z\sim1$ (where $\tau\approx1.1\times10^4$ Mpc). Its
   real value is against the *pointwise-difference* accessor, which saturates at
   $1.2$–$2.2\times10^{-4}$ rad at $k=3\times10^8$ **however short the baseline**, exactly as
   review §13.3 predicts. (§2)
5. **The two cheap baselines reproduce the review exactly**: 13.90 rad vs 13.9 ($G_k$),
   2.012 rad vs 2.01 ($T_k$). (§5)

---

## 1. Build cost and node accuracy

Production source grid: 1,732 nodes, $z=2.0636\times10^{16}$ (5 e-folds outside the horizon for
$k=3\times10^8$) down to $z=0.1$, 100 per decade of $z$. `value(z)` at the JSON checkpoints,
relative error against the reference; "as rad" multiplies by $k\tau(0.1)=3\times10^8\times1.37\times10^4$.

### `LambdaCDMModel` (reference: mpmath, `mp.dps = 40`)

| Gauss order | Hubble evaluations | build | max relative error at nodes | as rad at $k=3\times10^8$ |
|---|---|---|---|---|
| 4 | 6,924 | 0.037 s | **1.81e-15** | 7.4e-3 |
| 8 | 13,848 | 0.041 s | 1.96e-15 | 8.1e-3 |
| 12 | 20,772 | 0.047 s | 1.96e-15 | 8.1e-3 |

Worst case at $z=1.005\times10^7$ in every row. Order 4 is at the floor; the residual is the
double-precision evaluation of $1/H$, not the quadrature (§2.1). Review §7 measured
$5.7\times10^{-15}$ on a shorter grid; **acceptance $\le2\times10^{-14}$ — PASS.**

### `QCDModel` (reference: converged adaptive quadrature of the double-precision integrand)

| Gauss order | Hubble evaluations | build | max relative error at nodes | as rad at $k=3\times10^8$ |
|---|---|---|---|---|
| 4 | 6,924 | 0.084 s | **3.45e-07** | 1.4e6 |
| 8 | 13,848 | 0.129 s | 9.21e-08 | 3.8e5 |
| 12 | 20,772 | 0.222 s | 4.46e-08 | 1.8e5 |

Orders 4 and 8 disagree by $2.5\times10^{-7}$, seven orders above prompt 01 §4's
"say so prominently" threshold of $10^{-13}$. §3 identifies the cause.

`QCDModel` construction — the undecorated `compute_background` (RK45 over 1,732 nodes) plus the
derivative splines — takes **0.19–0.43 s**.

---

## 2. Off-grid evaluation and short baselines

### 2.1 Off-grid, 25 random points inside random intervals

| method | LambdaCDM | as rad at $k=3\times10^8$ | QCD | as rad |
|---|---|---|---|---|
| nearest node + Gauss 8, table order 4 | **5.84e-16** | 2.5e-3 | 2.16e-07 | 9.1e5 |
| nearest node + Gauss 8, table order 8 | 3.89e-16 | 1.6e-3 | 5.78e-08 | 2.4e5 |
| nearest node + Gauss 8, table order 12 | 3.89e-16 | 1.6e-3 | 7.49e-08 | 3.2e5 |
| cubic spline of the nodes | 7.33e-10 | 3.1e3 | 7.56e-08 | 3.2e5 |
| quintic spline of the nodes | 9.91e-15 | 4.2e-2 | 7.50e-08 | 3.2e5 |

The LambdaCDM column reproduces review §7's three rows (2.1e-16 / 1.4e-9 / 1.8e-14). The QCD
column is *uninformative about the accessor*: every method there is dominated by the same
accumulated table error of §3, which is why the local-Gauss, cubic and quintic rows agree with
each other to 30 %.

### 2.2 One production grid interval, and its 37 % fraction

`delta(z_hi, z_lo)` against the reference interval; "dd" is the double-double table, "sd" the
single-double control. **Acceptance: dd $\le10^{-13}$ relative — PASS, worst 8.8e-15.**

| model | $z_{\rm hi}$ | baseline | $\Delta\tau$ [Mpc] | dd rel | dd [rad] | sd rel | sd [rad] |
|---|---|---|---|---|---|---|---|
| LambdaCDM | 1.004e6 | full | 1.0734e-2 | 4.69e-15 | 1.5e-8 | 4.04e-15 | 1.3e-8 |
| LambdaCDM | 1.004e6 | 37 % | 3.9429e-3 | 3.37e-14 | 4.0e-8 | 3.37e-14 | 4.0e-8 |
| LambdaCDM | 100.19 | full | 17.850 | 8.76e-15 | 4.7e-5 | 6.77e-15 | 3.6e-5 |
| LambdaCDM | 100.19 | 37 % | 6.5801 | 2.02e-14 | 4.0e-5 | 2.02e-14 | 4.0e-5 |
| LambdaCDM | 1.0006 | full | 56.977 | 2.24e-15 | 3.8e-5 | 1.05e-14 | 1.8e-4 |
| LambdaCDM | 1.0006 | 37 % | 21.068 | 9.61e-15 | 6.1e-5 | 9.61e-15 | 6.1e-5 |
| QCD | 1.004e6 | full | 1.0740e-2 | 4.85e-16 | 1.6e-9 | 1.29e-15 | 4.2e-9 |
| QCD | 100.19 | full | 17.850 | 5.97e-16 | 3.2e-6 | 2.39e-15 | 1.3e-5 |
| QCD | 1.0006 | full | 56.977 | 2.49e-16 | 4.3e-6 | 1.52e-14 | 2.6e-4 |

Two things this table says that the prompt did not anticipate.

* **The single-double floor only bites where $\tau$ is large.** At $z\sim10^6$, $\tau=0.046$ Mpc
  and one ulp of it is $7\times10^{-18}$ Mpc — six orders below review §13.3's
  $1.5\times10^{-12}$ Mpc, which is a statement about $\tau\approx1.4\times10^4$ Mpc at low $z$.
  The QCD rows show the expected ordering ($z\sim1$: 2.5e-16 dd against 1.5e-14 sd, a factor 60);
  the LambdaCDM rows do not, because there the dd result is limited by something else:
* **The double-precision integrand is the accuracy floor for LambdaCDM, at 2–9e-15 relative.**
  Raising the Gauss order from 4 to 20 on a single interval does not move the number
  (4.687e-15 → 4.525e-15 at $z=10^6$; 8.757e-15 unchanged at $z=100$). It is the rounding of
  `LambdaCDM.Hubble`'s $\sqrt{(\rho_m s^3+\rho_r s^4+\rho_{cc})/3M_P^2}$, which no quadrature and
  no storage width can repair. In phase this is $3.8$–$6.1\times10^{-5}$ rad at $k=3\times10^8$
  over one grid interval — below README §6's $5\times10^{-3}$ rad target for $\theta_G$, and above
  the $9\times10^{-4}$ rad $\varepsilon k\tau$ floor only in the sense that it is a *different*
  error.

### 2.3 Baselines far shorter than one grid interval — review §13.3's actual claim

Near $z=1$, where $\tau\approx1.08\times10^4$ Mpc. Errors in radians at $k=3\times10^8$. The third
column is `value(z_b) - value(z_a)`, the pointwise-difference anti-pattern.

| model | width / interval | $\Delta\tau$ [Mpc] | `delta`, dd | `delta`, sd | pointwise difference |
|---|---|---|---|---|---|
| LambdaCDM | 1e-1 | 5.6926 | 2.5e-5 | 2.5e-5 | 1.2e-4 |
| LambdaCDM | 1e-3 | 5.6920e-2 | 4.2e-6 | 4.2e-6 | 2.0e-4 |
| LambdaCDM | 1e-5 | 5.6920e-4 | 1.2e-5 | 1.2e-5 | 2.2e-4 |
| QCD | 1e-1 | 5.6926 | 2.7e-7 | 2.7e-7 | 1.5e-4 |
| QCD | 1e-3 | 5.6920e-2 | 4.2e-9 | 4.2e-9 | 6.7e-5 |
| QCD | 1e-5 | 5.6920e-4 | **0.0** | 0.0 | 4.9e-5 |

The pointwise column does not fall as the baseline shrinks: it sits at $10^{-4}$ rad, which is
${\rm ulp}(1.08\times10^4\,{\rm Mpc})\times3\times10^8\approx5\times10^{-4}$ rad to within a
factor of a few. That is the whole of review §13.3, measured. `delta` falls with the baseline
until it hits its own floor.

dd and sd coincide here because both endpoints lie inside a single interval, so `delta` is one
local Gauss partial and the table never enters. **A caveat for prompt 03's tests:** the LambdaCDM
1e-5 row *rises*, because a baseline that narrow is no longer resolvable in $u=\log(1+z)$ —
${\rm ulp}(u)/\Delta u = 1.1\times10^{-16}/2.3\times10^{-7}=4.8\times10^{-10}$ of the width. Below
about $10^{-9}$ of a decade the endpoint's own representation, not the accessor, sets the error.

---

## 3. `QCD_EOS.G(T)` branch boundaries defeat fixed-order Gauss (prompt 02's question)

Per-production-interval relative error of a fixed-order Gauss–Legendre rule against the adaptive
reference on the same interval.

| order | LambdaCDM: intervals $>10^{-12}$ | worst | QCD: intervals $>10^{-12}$ | worst | at $z$ | abs [Mpc] | as rad at $k=3\times10^8$ |
|---|---|---|---|---|---|---|---|
| 4 | 0 / 1731 | 7.6e-16 | **301 / 1731** | 6.33e-05 | 4.191e7 | 1.59e-08 | **4.78** |
| 8 | 0 | 5.8e-16 | 163 | 1.69e-05 | 4.191e7 | 4.25e-09 | 1.28 |
| 12 | 0 | 1.2e-15 | 38 | 8.20e-06 | 4.191e7 | 2.06e-09 | 0.62 |
| 20 | 0 | 5.8e-16 | 8 | 3.47e-06 | 4.191e7 | 8.72e-10 | 0.26 |

The two worst intervals are at $z=4.191\times10^7$ and $z=8.579\times10^{11}$. Evaluating
`QCD_Cosmology.T_photon` there gives $T=9.85\times10^{-6}$ GeV and $T=0.1193$ GeV: they are
`QCD_EOS.T_LO = 1e-5` and `QCD_EOS.T_120_MEV = 0.12`
(`CosmologyModels/GenericEOS/QCD_EOS.py:130-143`), where `G(T)` and `Gs(T)` switch between the
Saikawa–Shirai fit and the asymptotic constants. The switch is not smooth, so the integrand has a
low-order derivative discontinuity *inside* a production interval and Gauss convergence is
algebraic: the error falls only as $N^{-2}$, from 6.3e-5 at order 4 to 3.5e-6 at order 20.

The 500-point $T(z)$ spline (`LambdaCDM_GenericEOS._build_T_z_spline`) contributes the long tail —
its knots are every ~4 production intervals, which is why 301 intervals exceed $10^{-12}$ at
order 4 — but the knots are individually 1e-9-level and only the two branch boundaries are
1e-5-level.

**What this leaves for prompt 02.** Raising the order is not a fix. The candidates it must weigh
are (a) adaptive quadrature on the intervals that contain a branch boundary or a $T(z)$ knot —
there are two of the former and ~430 of the latter, so this is cheap; (b) subdividing the table's
intervals at the branch boundaries, which are two known constants; (c) accepting the error, which
at order 4 is 4.8 rad of phase at $k=3\times10^8$ across the $z=4.19\times10^7$ interval and
therefore not acceptable. Nothing here decides between them.

---

## 4. Throughput (review §13.3 item 5; README §4.3 stop condition)

$10^5$ calls, best of three, build order 4, partial order 8, timed on a table built with the raw
integrand (the call-counting wrapper is used only for the counts).

| call | LambdaCDM | QCD | integrand calls |
|---|---|---|---|
| `delta`, both endpoints on-grid | **3.03 µs** | **4.39 µs** | **0** |
| `delta`, one endpoint off-grid | 8.20 µs | 47.7 µs | 8 |
| `delta`, both endpoints off-grid | 13.1 µs | **102.2 µs** | 16 |
| `value`, on-grid | 1.47 µs | 1.44 µs | 0 |
| `value`, off-grid | 7.15 µs | 46.7 µs | 8 |
| cubic spline lookup (comparator) | 3.29 µs | 2.38 µs | — |

A Hubble evaluation costs 0.23 µs on `LambdaCDM` and 8.7 µs on `QCD_Cosmology` (its
$\rho$ goes through the $T(z)$ spline), which accounts for the whole difference: 16 × 8.7 ≈ 140 µs
of the 102 µs measured (the loop is faster than the isolated micro-benchmark).

**On-grid calls are short-circuited: zero integrand evaluations, confirmed by the counter.**

Against a cubic spline lookup the off-grid interval accessor is 4× (LambdaCDM) and 43× (QCD) — at
the optimistic end of review §13.3's estimate of "perhaps 50–100×". Against the ODE it replaces,
review §4 measures $2.5\times10^6$ RHS evaluations for one `GkWKBIntegration` object at
$k=3\times10^8$; an object with ~1,300 samples and one off-grid anchor costs 1,300 on-grid `delta`
calls plus one partial, i.e. ~5 ms on QCD.

---

## 5. Baseline re-measurement (prompt 01 §2.5)

`LambdaCDM`, $k=10^5$/Mpc, production tolerances (`atol=1e-10`, `rtol=1e-8`), from the 3-e-fold
sub-horizon point $z_{\rm init}=2.30758\times10^9$ down to $z=0.1$ on the production source grid
(1,037 samples), through the production `integrate_phase_function`.

| quantity | this tree | review (`f06f587`) | agreement |
|---|---|---|---|
| $\theta_G$ error at $z=0.1$ | **13.90 rad** | 13.9 rad | 0.0 % |
| $G_k$ stage 1 / resets / stage 2 | 46,984 / 676 / 1,298 | 46,702 / 676 / 1,271 | — |
| $G_k$ wall clock | 1.72 s | 1.3 s | — |
| $\theta_T$ error at $z=0.1$ | **2.012 rad** | 2.01 rad | 0.1 % |
| $T_k$ stage 1 / resets / stage 2 / friction | 35,653 / 508 / 1,061 / 1,814 | 35,626 / 508 / 1,061 / 1,811 | — |
| $T_k$ wall clock | 1.12 s | 1.1 s | — |
| $T_k$ friction $\delta F$ at $z=0.1$ | $-2.26\times10^{-7}$ | $-2.3\times10^{-7}$ | 2 % |

**Acceptance: 5 % — PASS on both.** The RHS-evaluation counts differ by 0.6 % because the sample
grid here is the production *source* grid rather than the review's 100/12-per-decade construction;
the phase error is a property of the ODE and is unchanged.

$k=3\times10^8$ was not run (63.7 s and 58 s per object); the review's 7,366 rad and
$5.1\times10^3$ rad stand as the baseline for those rows of README §6.
