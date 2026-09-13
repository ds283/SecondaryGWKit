# Gauss-order convergence of the WKB primitives on both production models

**Campaign:** [`prompts/GkTk-remedial/README.md`](../../prompts/GkTk-remedial/README.md) ·
**Prompt:** [`02-qcd-residual-convergence.md`](../../prompts/GkTk-remedial/02-qcd-residual-convergence.md)
**Generated:** 2026-09-10 by `docs/gktk-remedial/residual_convergence.py` in 328 s
(Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2).

> This file is **regenerated in full** by the script; do not hand-edit it. Every number below is a
> measurement, and the decision in §1 applies the rules of the prompt's §3 and §4 to those numbers.

Review §11 names this measurement "the first test of any implementation": whether the fixed-order
Gauss–Legendre-per-production-interval rule of review §7 converges across the QCD model's spline
knots. It does not, and the reason is stronger than the review anticipated — **`QCD_EOS.G(T)` and
`Gs(T)` jump discontinuously at their branch boundaries, so `QCD_Cosmology`'s `H(z)` is itself
discontinuous** — and a third break point, the clamp inside `QCD_EOS.w(T)` at `EOS_T_LO`, kinks
`c_s^2` (§2). The remedy is deterministic and costs 24 % more integrand evaluations:
split the containing production interval at the break point, which is known in closed form from
the cosmology before any integration happens.

---

## 1. Decision

| quantity | order | how it was chosen |
|---|---|---|
| $N_\tau$ | **4** | smallest order within 3× the floor of the *relative* cumulative checkpoint error, on both models |
| $N_{\tau_s}$ | **4** | same rule, integrand $c_s/H$ |
| $N_F$ | **4** | same rule, integrand $\tfrac32(1+c_s^2)/(1+z)$ |
| $N_\rho$ | **4** | smallest order with cumulative $|\delta\rho|\le10^{-7}$ rad on both models for all three $k$ **and** reproducing the exact-radiation $\rho_T$ to $10^{-14}$ relative (§7) |

**Adaptive fallback for $\rho$: not required** — a fixed order $\le16$ reaches the target on both
models under the recommended scheme, so prompt 05 needs no adaptive path.

The two $\rho$ criteria are not redundant, and the second is what fixes the answer. $\rho$ is
$10^{-3}$ rad on QCD and $10^{-10}$ rad on LambdaCDM, so an *absolute* $10^{-7}$ rad target is
met by rules that have not converged at all: order 2 passes it under every scheme
(2.03e-08 rad on QCD) while carrying
7.68e-11 relative error on a control whose
answer is known in closed form. Order 4 is at the floor on both
(3.75e-16 rad and
9.53e-16 relative), and no higher
order improves either.

**Recommended build scheme: `branch+knots`** — one order-$N$ Gauss panel per production interval on
LambdaCDM, and on `QCD_Cosmology` the same panel split at every break point of the cosmology that
falls inside it. The scheme was chosen from the measurement, not in advance: the cheapest of the
three that admits a fixed order $\le16$ for $\rho$ *and* loses nothing on the primitives relative
to the best scheme measured.

| QCD scheme | floor, `tau` | floor, `cs_tau` | floor, `friction` | smallest $N$ meeting §3's $\rho$ target | verdict |
|---|---|---|---|---|---|
| `plain` | 4.46e-08 | 4.46e-08 | 5.76e-10 | 2 | rejected: primitive floor 3913120× the best |
| `branch` | 1.81e-13 | 1.88e-13 | 3.68e-15 | 2 | rejected: primitive floor 25× the best |
| `branch+knots` | 1.86e-14 | 1.87e-14 | 1.47e-16 | 2 | **selected** |

The three floor columns are the smallest relative cumulative checkpoint error any order in
[2, 4, 6, 8, 12, 16] achieves under that scheme. **Note what disqualifies `plain`: not $\rho$, but
$\tau$.** The residual is so small that even an unconverged rule delivers it to $10^{-9}$ rad
absolute, so review §11's worry — "a fixed Gauss rule for $\rho$ could converge slowly across
[the knots]" — is real but harmless. What is *not* harmless is the same roughness in the leading
term: `plain` leaves the cumulative $\tau$ at best
2.06e-09 Mpc from the reference however
high the order — 0.62
rad of phase at $k=3\times10^8$, and
4.79 rad at order 4 —
against 1.1e-03 rad
under `branch+knots`. The test the review asked for gives the right answer for a different reason than
it expected.

Magnitude of $\rho$ over the whole WKB range, so that a later reader knows what carrying it buys
(review §6: on LambdaCDM $\rho_G$ is below the $\varepsilon k\tau$ floor, but the machinery is
required for $T_k$ and for QCD, so one code path is carried — README §7 D3):

| integrand | $|\rho|$, LambdaCDM [rad] | $|\rho|$, QCD [rad] |
|---|---|---|
| `rho_G@1.000000e+05` | 2.588e-07 | 5.420e-04 |
| `rho_G@1.000000e+07` | 3.434e-09 | 3.003e-05 |
| `rho_G@3.000000e+08` | 1.353e-10 | 1.180e-03 |
| `rho_T@1.000000e+05` | 8.634e-02 | 8.867e-02 |
| `rho_T@1.000000e+07` | 8.634e-02 | 8.903e-02 |
| `rho_T@3.000000e+08` | 8.634e-02 | 9.310e-02 |

---

## 2. Where the QCD model's features sit on the production grid

The production source grid has 1732 nodes and 1731 intervals, with
$\Delta u$ from 0.002116 to 0.02303 in $u=\log(1+z)$.

**`QCD_EOS` break points.** `G(T)`, `Gs(T)` switch between the Saikawa–Shirai fit and an asymptotic
constant at fixed temperatures and the two pieces do not match, so `H(z)` *jumps*. `w(T)` clamps
its argument to `EOS_T_LO` below `EOS_T_LO`, so $c_s^2$ is continuous there but its slope is not:
that one is a kink, not a jump, which the last two columns separate. `T_HI = 1e16` GeV is far above
the production range and does not appear.

| constant | $T$ | $z$ | production interval | breaks | rel. jump in $H$ | rel. jump in $c_s^2$ | rel. jump in $dc_s^2/du$ |
|---|---|---|---|---|---|---|---|
| `T_LO` | 1e-05 GeV | 4.25344e+07 | 862 | G(T), Gs(T) -> H(z) | 4.43e-04 | 7.08e-08 | 8.93e-04 |
| `EOS_T_LO` | 0.002 GeV | 1.18721e+10 | 1107 | w(T) -> c_s^2 | 4.00e-10 | 7.32e-15 | 9.95e-01 |
| `T_120_MEV` | 0.12 GeV | 8.64355e+11 | 1293 | G(T), Gs(T) -> H(z) | 1.04e-04 | 7.46e-04 | 1.99e-01 |

Prompt 01 reported the two `G(T)` boundaries as $z=4.191\times10^7$ and $8.579\times10^{11}$: those
are the *lower edges of the containing production intervals*, which §3.2 below reproduces exactly
as the worst-converging intervals. The $z$ column here is the break point itself, from a
`root_scalar` solve of $T_{\rm photon}(z)=T$ on the model's own spline to $10^{-15}$, and is what
the subdivision splits at.

**`T(z)` spline knots.** The model builds `T(z)` as a 500-point
`make_interp_spline` in $\log(1+z)$ (`LambdaCDM_GenericEOS._build_T_z_spline`);
404 of its knots lie inside the production range, a median $\Delta u$
of 0.09294 apart. That is
0.233 knots per production interval — **one knot every
4.3 intervals** — and
404 of 1731
intervals contain one. (The prompt's estimate of ~20 knots per interval came from review §6's
*quintic $\epsilon$ spline at 2000 points per decade*, which is the review's own reference
construction, not the model's `T(z)` spline. The estimate is wrong by two orders of magnitude in
the reassuring direction.)

**The QCD transition.** $\epsilon$ departs from 2 by its largest amount,
$\epsilon=1.850811$, at $z=1.299e+12$
(production interval 1310), reproducing review §6's
"$\epsilon$ departs from 2 by up to 0.150 at $z\approx1.2\times10^{12}$".

**The $c_s^2$ transition.** The node-to-node $|dw/du|$ is largest at
$z=8.779e+11$ (production interval
1293), i.e. at the QCD transition and within a few production
intervals of the `T_120_MEV` boundary, not at matter–radiation equality — the fall of $c_s^2$
towards equality is spread over many decades and is gentle per interval. Once `T_120_MEV` is a
split point the remaining variation is smooth and needs no further treatment; the `friction` rows
of §3.4 and §4 confirm it.

---

## 3. Per-interval convergence of the increments

The largest relative error of a *single production interval's* Gauss increment, against the
converged reference. **The reference is break-aware**: within each production interval it is
`quad` (`epsabs=0`, `epsrel=1.5e-14`, `limit=400`) on each sub-panel between break points,
cross-checked per interval against `quad` at `epsrel=1e-12` and against
composite Gauss–Legendre order 40 with bisection. Without that, the reference would be as wrong as
the rule it is meant to score across a discontinuity.

### 3.1 LambdaCDM (`plain`; the model has no break points)

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 | worst interval at $z$ (N=4) |
|---|---|---|---|---|---|---|---|
| `tau` | 6.51e-11 | 7.63e-16 | 7.68e-16 | 5.76e-16 | 1.15e-15 | 4.84e-16 | 1.268e+10 |
| `cs_tau` | 6.51e-11 | 7.21e-16 | 7.50e-16 | 4.50e-16 | 9.97e-16 | 5.34e-16 | 2.828e+04 |
| `friction` | 2.51e-12 | 6.04e-16 | 4.79e-16 | 4.28e-16 | 4.52e-16 | 6.42e-16 | 3.175e+05 |
| `rho_G@1.000000e+05` | 1.73e-09 | 1.47e-09 | 9.73e-10 | 7.13e-10 | 7.22e-10 | 4.04e-10 | 1.918e+09 |
| `rho_G@1.000000e+07` | 1.54e-07 | 8.18e-08 | 9.12e-08 | 9.59e-08 | 7.18e-08 | 6.53e-08 | 1.965e+11 |
| `rho_G@3.000000e+08` | 3.47e-06 | 3.95e-06 | 1.85e-06 | 2.82e-06 | 3.07e-06 | 1.40e-06 | 6.363e+12 |
| `rho_T@1.000000e+05` | 8.51e-11 | 1.19e-15 | 9.89e-16 | 7.77e-16 | 6.87e-16 | 8.65e-16 | 1.417e+04 |
| `rho_T@1.000000e+07` | 8.42e-11 | 1.12e-15 | 1.18e-15 | 8.50e-16 | 8.54e-16 | 7.85e-16 | 1.52e+06 |
| `rho_T@3.000000e+08` | 8.45e-11 | 1.58e-15 | 1.10e-15 | 7.79e-16 | 7.72e-16 | 7.92e-16 | 1.234e+04 |

### 3.2 QCD, `plain` — one panel per production interval

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 | worst interval at $z$ (N=4) |
|---|---|---|---|---|---|---|---|
| `tau` | 6.32e-05 | 6.33e-05 | 4.01e-05 | 1.69e-05 | 8.20e-06 | 1.90e-05 | 4.191e+07 |
| `cs_tau` | 6.32e-05 | 6.33e-05 | 4.01e-05 | 1.69e-05 | 1.36e-05 | 1.90e-05 | 4.191e+07 |
| `friction` | 3.11e-05 | 2.76e-05 | 9.42e-06 | 3.12e-07 | 8.50e-06 | 1.41e-06 | 8.579e+11 |
| `rho_G@1.000000e+05` | 6.09e-01 | 1.70e-04 | 1.44e-04 | 5.07e-05 | 2.93e-05 | 6.46e-05 | 4.191e+07 |
| `rho_G@1.000000e+07` | 6.09e-01 | 1.70e-04 | 1.44e-04 | 5.07e-05 | 2.93e-05 | 6.46e-05 | 4.191e+07 |
| `rho_G@3.000000e+08` | 6.09e-01 | 1.70e-04 | 1.44e-04 | 5.07e-05 | 2.93e-05 | 6.46e-05 | 4.191e+07 |
| `rho_T@1.000000e+05` | 5.76e-04 | 5.48e-05 | 3.18e-05 | 1.42e-05 | 6.51e-06 | 1.53e-05 | 4.191e+07 |
| `rho_T@1.000000e+07` | 5.76e-04 | 5.48e-05 | 3.18e-05 | 1.42e-05 | 6.51e-06 | 1.53e-05 | 4.191e+07 |
| `rho_T@3.000000e+08` | 5.76e-04 | 5.48e-05 | 3.18e-05 | 1.42e-05 | 6.51e-06 | 1.53e-05 | 4.191e+07 |

### 3.3 QCD, `branch` — split at the `QCD_EOS` break points only

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 | worst interval at $z$ (N=4) |
|---|---|---|---|---|---|---|---|
| `tau` | 2.22e-07 | 7.18e-09 | 1.42e-09 | 4.56e-10 | 9.07e-11 | 4.06e-11 | 8.007e+11 |
| `cs_tau` | 2.39e-07 | 7.64e-09 | 1.51e-09 | 4.85e-10 | 9.64e-11 | 4.34e-11 | 8.779e+11 |
| `friction` | 7.04e-09 | 2.20e-10 | 4.09e-11 | 1.31e-11 | 2.61e-12 | 1.26e-12 | 8.779e+11 |
| `rho_G@1.000000e+05` | 6.09e-01 | 1.57e-09 | 4.27e-10 | 5.69e-10 | 3.45e-10 | 1.69e-10 | 3.486e+07 |
| `rho_G@1.000000e+07` | 6.09e-01 | 1.57e-09 | 4.27e-10 | 5.69e-10 | 3.45e-10 | 1.69e-10 | 3.486e+07 |
| `rho_G@3.000000e+08` | 6.09e-01 | 8.34e-09 | 1.57e-09 | 5.69e-10 | 3.45e-10 | 1.69e-10 | 8.007e+11 |
| `rho_T@1.000000e+05` | 5.76e-04 | 1.41e-09 | 4.39e-10 | 1.29e-10 | 2.51e-11 | 4.93e-12 | 3.486e+07 |
| `rho_T@1.000000e+07` | 5.76e-04 | 1.41e-09 | 4.39e-10 | 1.29e-10 | 2.51e-11 | 4.93e-12 | 3.486e+07 |
| `rho_T@3.000000e+08` | 5.76e-04 | 7.83e-09 | 1.48e-09 | 4.76e-10 | 9.45e-11 | 4.33e-11 | 8.779e+11 |

### 3.4 QCD, `branch+knots` — also split at the `T(z)` spline's knots

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 | worst interval at $z$ (N=4) |
|---|---|---|---|---|---|---|---|
| `tau` | 8.05e-09 | 4.75e-15 | 3.75e-15 | 5.40e-15 | 3.33e-15 | 3.09e-15 | 8.779e+11 |
| `cs_tau` | 8.20e-09 | 4.55e-15 | 3.68e-15 | 5.00e-15 | 3.49e-15 | 3.14e-15 | 8.779e+11 |
| `friction` | 5.19e-10 | 7.53e-16 | 6.00e-16 | 4.73e-16 | 4.58e-16 | 4.73e-16 | 5.039e+07 |
| `rho_G@1.000000e+05` | 6.09e-01 | 1.06e-09 | 4.27e-10 | 5.69e-10 | 3.45e-10 | 1.69e-10 | 1.236e+07 |
| `rho_G@1.000000e+07` | 6.09e-01 | 1.06e-09 | 4.27e-10 | 5.69e-10 | 3.45e-10 | 1.69e-10 | 1.236e+07 |
| `rho_G@3.000000e+08` | 6.09e-01 | 1.06e-09 | 4.27e-10 | 5.69e-10 | 3.45e-10 | 1.69e-10 | 1.236e+07 |
| `rho_T@1.000000e+05` | 5.76e-04 | 1.00e-12 | 6.34e-15 | 5.36e-15 | 5.36e-15 | 6.07e-15 | 4.095e+07 |
| `rho_T@1.000000e+07` | 5.76e-04 | 6.42e-13 | 5.99e-15 | 5.42e-15 | 5.42e-15 | 5.97e-15 | 4.095e+07 |
| `rho_T@3.000000e+08` | 5.76e-04 | 6.74e-12 | 6.31e-15 | 5.43e-15 | 5.43e-15 | 6.33e-15 | 8.384e+11 |

### 3.5 How many intervals are worse than $10^{-12}$ relative

QCD, `plain`:

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 1675 / 1731 | 301 / 1731 | 235 / 1731 | 161 / 1731 | 38 / 1731 | 13 / 1731 |
| `cs_tau` | 1676 / 1731 | 341 / 1731 | 272 / 1731 | 179 / 1731 | 39 / 1731 | 15 / 1731 |
| `friction` | 783 / 1731 | 42 / 1731 | 13 / 1731 | 5 / 1731 | 5 / 1731 | 4 / 1731 |
| `rho_G@1.000000e+05` | 1008 / 1040 | 188 / 1040 | 136 / 1040 | 88 / 1040 | 38 / 1040 | 29 / 1040 |
| `rho_G@1.000000e+07` | 1210 / 1242 | 233 / 1242 | 169 / 1242 | 113 / 1242 | 45 / 1242 | 30 / 1242 |
| `rho_G@3.000000e+08` | 1369 / 1401 | 272 / 1401 | 201 / 1401 | 137 / 1401 | 57 / 1401 | 38 / 1401 |
| `rho_T@1.000000e+05` | 1001 / 1040 | 191 / 1040 | 149 / 1040 | 94 / 1040 | 14 / 1040 | 6 / 1040 |
| `rho_T@1.000000e+07` | 1203 / 1242 | 233 / 1242 | 184 / 1242 | 119 / 1242 | 21 / 1242 | 7 / 1242 |
| `rho_T@3.000000e+08` | 1362 / 1401 | 271 / 1401 | 215 / 1401 | 143 / 1401 | 33 / 1401 | 14 / 1401 |

QCD, `branch`:

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 1675 / 1731 | 300 / 1731 | 234 / 1731 | 160 / 1731 | 36 / 1731 | 11 / 1731 |
| `cs_tau` | 1676 / 1731 | 339 / 1731 | 270 / 1731 | 177 / 1731 | 36 / 1731 | 12 / 1731 |
| `friction` | 781 / 1731 | 39 / 1731 | 10 / 1731 | 2 / 1731 | 2 / 1731 | 1 / 1731 |
| `rho_G@1.000000e+05` | 1008 / 1040 | 188 / 1040 | 136 / 1040 | 88 / 1040 | 38 / 1040 | 28 / 1040 |
| `rho_G@1.000000e+07` | 1210 / 1242 | 233 / 1242 | 169 / 1242 | 113 / 1242 | 45 / 1242 | 29 / 1242 |
| `rho_G@3.000000e+08` | 1369 / 1401 | 271 / 1401 | 200 / 1401 | 136 / 1401 | 56 / 1401 | 36 / 1401 |
| `rho_T@1.000000e+05` | 1001 / 1040 | 191 / 1040 | 149 / 1040 | 94 / 1040 | 13 / 1040 | 5 / 1040 |
| `rho_T@1.000000e+07` | 1203 / 1242 | 232 / 1242 | 183 / 1242 | 118 / 1242 | 19 / 1242 | 5 / 1242 |
| `rho_T@3.000000e+08` | 1362 / 1401 | 269 / 1401 | 213 / 1401 | 141 / 1401 | 30 / 1401 | 11 / 1401 |

QCD, `branch+knots`:

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 1658 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 |
| `cs_tau` | 1669 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 |
| `friction` | 684 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 | 0 / 1731 |
| `rho_G@1.000000e+05` | 1007 / 1040 | 41 / 1040 | 27 / 1040 | 17 / 1040 | 20 / 1040 | 22 / 1040 |
| `rho_G@1.000000e+07` | 1209 / 1242 | 44 / 1242 | 30 / 1242 | 20 / 1242 | 21 / 1242 | 23 / 1242 |
| `rho_G@3.000000e+08` | 1368 / 1401 | 46 / 1401 | 30 / 1401 | 20 / 1401 | 21 / 1401 | 23 / 1401 |
| `rho_T@1.000000e+05` | 1000 / 1040 | 1 / 1040 | 0 / 1040 | 0 / 1040 | 0 / 1040 | 0 / 1040 |
| `rho_T@1.000000e+07` | 1202 / 1242 | 0 / 1242 | 0 / 1242 | 0 / 1242 | 0 / 1242 | 0 / 1242 |
| `rho_T@3.000000e+08` | 1361 / 1401 | 1 / 1401 | 0 / 1401 | 0 / 1401 | 0 / 1401 | 0 / 1401 |

The `rho_G` rows do not reach zero under `branch+knots` and this is not a convergence failure. On
QCD a single production interval's $\rho_G$ increment is $10^{-6}$–$10^{-15}$ rad, formed from a
$C$ that is itself a difference of $O(1)$ spline values, so a *relative* $10^{-12}$ on the increment
is being asked of a quantity already at its own rounding floor. The absolute cumulative error is
what matters and §4.2 gives it as 2.71e-16 rad.

---

## 4. Cumulative error at the checkpoints

The quantity the tables actually deliver: the largest error of the cumulative value at a
`wkb_reference_data.json` checkpoint, scored against the JSON references — mpmath at 40 digits for
LambdaCDM, converged adaptive quadrature of the double-precision integrand for QCD. The LambdaCDM
rows therefore include the double-precision evaluation floor of `LambdaCDM.Hubble`
(`[01-lambdacdm-hubble-rounding-floor]`, 2–9e-15 relative) and the QCD rows do not.

Two floors are visible in these tables and neither is a property of the Gauss rule. On LambdaCDM
the $\tau$ and $\tau_s$ rows bottom out at $\approx2\times10^{-15}$, the `Hubble` rounding floor.
On QCD they bottom out at $\approx1.9\times10^{-14}$, which is exactly the disagreement between
this script's break-aware reference and the JSON reference it is scored against
(`json_vs_reference_max_rel` in the JSON block); the QCD $\tau$ column therefore *cannot* resolve
below $2\times10^{-14}$, and 1.9e-14 should be read as "at the reference's own floor", not as a
quadrature error. Both are below README §6's $2\times10^{-14}$ target for $\tau$ at the nodes.

### 4.1 Relative

LambdaCDM (`plain`):

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 6.51e-11 | 1.81e-15 | 1.96e-15 | 1.96e-15 | 1.96e-15 | 2.11e-15 |
| `cs_tau` | 6.51e-11 | 1.82e-15 | 1.95e-15 | 2.09e-15 | 1.95e-15 | 2.22e-15 |
| `friction` | 1.48e-14 | 2.35e-16 | 2.35e-16 | 2.11e-16 | 2.22e-16 | 2.90e-16 |
| `rho_G@1.000000e+05` | 9.67e-11 | 2.93e-11 | 2.34e-10 | 2.46e-11 | 2.22e-10 | 1.57e-10 |
| `rho_G@1.000000e+07` | 2.90e-08 | 2.25e-09 | 1.24e-08 | 1.29e-09 | 1.97e-08 | 1.16e-08 |
| `rho_G@3.000000e+08` | 2.17e-07 | 2.14e-07 | 9.48e-08 | 1.35e-07 | 9.73e-08 | 1.32e-07 |
| `rho_T@1.000000e+05` | 7.59e-11 | 8.54e-16 | 8.54e-16 | 9.97e-16 | 8.54e-16 | 1.14e-15 |
| `rho_T@1.000000e+07` | 7.58e-11 | 7.13e-16 | 8.55e-16 | 9.98e-16 | 8.55e-16 | 9.98e-16 |
| `rho_T@3.000000e+08` | 7.09e-11 | 4.82e-16 | 3.22e-16 | 1.63e-16 | 1.63e-16 | 1.61e-16 |

QCD, `plain`:

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 3.44e-07 | 3.45e-07 | 2.19e-07 | 9.21e-08 | 4.46e-08 | 1.03e-07 |
| `cs_tau` | 3.44e-07 | 3.45e-07 | 2.19e-07 | 9.21e-08 | 4.46e-08 | 1.03e-07 |
| `friction` | 5.73e-08 | 5.09e-08 | 1.74e-08 | 5.76e-10 | 1.57e-08 | 2.61e-09 |
| `rho_G@1.000000e+05` | 8.59e-07 | 5.06e-07 | 4.28e-07 | 1.51e-07 | 8.72e-08 | 1.92e-07 |
| `rho_G@1.000000e+07` | 1.12e-06 | 9.13e-08 | 7.72e-08 | 2.73e-08 | 1.57e-08 | 3.47e-08 |
| `rho_G@3.000000e+08` | 1.21e-06 | 9.12e-07 | 3.38e-07 | 2.05e-08 | 3.05e-07 | 5.20e-08 |
| `rho_T@1.000000e+05` | 1.73e-08 | 2.09e-08 | 1.21e-08 | 5.41e-09 | 2.48e-09 | 5.83e-09 |
| `rho_T@1.000000e+07` | 7.49e-10 | 2.06e-10 | 1.20e-10 | 5.40e-11 | 2.44e-11 | 5.79e-11 |
| `rho_T@3.000000e+08` | 1.26e-08 | 1.24e-08 | 3.96e-09 | 5.33e-11 | 3.58e-09 | 5.86e-10 |

QCD, `branch+knots` (the recommended scheme):

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 1.73e-10 | 1.89e-14 | 1.89e-14 | 1.88e-14 | 1.88e-14 | 1.86e-14 |
| `cs_tau` | 1.73e-10 | 1.90e-14 | 1.90e-14 | 1.89e-14 | 1.89e-14 | 1.87e-14 |
| `friction` | 5.52e-12 | 3.01e-16 | 2.36e-16 | 1.47e-16 | 2.01e-16 | 2.13e-16 |
| `rho_G@1.000000e+05` | 1.90e-05 | 8.64e-14 | 5.60e-15 | 6.00e-15 | 5.80e-15 | 6.20e-15 |
| `rho_G@1.000000e+07` | 1.64e-05 | 2.73e-14 | 4.09e-14 | 1.06e-14 | 8.35e-15 | 8.99e-15 |
| `rho_G@3.000000e+08` | 8.91e-06 | 2.29e-13 | 8.83e-15 | 5.19e-15 | 4.33e-15 | 1.73e-15 |
| `rho_T@1.000000e+05` | 2.02e-07 | 1.86e-15 | 1.73e-15 | 1.73e-15 | 1.73e-15 | 1.60e-15 |
| `rho_T@1.000000e+07` | 9.67e-09 | 2.03e-15 | 2.03e-15 | 1.88e-15 | 2.03e-15 | 1.88e-15 |
| `rho_T@3.000000e+08` | 2.21e-07 | 4.07e-15 | 7.15e-15 | 7.01e-15 | 7.01e-15 | 6.86e-15 |

### 4.2 Absolute (Mpc for $\tau$, $\tau_s$; radians for $F$ and the residuals)

LambdaCDM (`plain`):

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 1.28e-07 | 1.82e-12 | 1.82e-12 | 2.27e-13 | 1.82e-12 | 1.82e-12 |
| `cs_tau` | 1.24e-09 | 1.14e-13 | 1.14e-13 | 2.84e-14 | 2.84e-14 | 1.14e-13 |
| `friction` | 8.38e-13 | 1.42e-14 | 1.42e-14 | 7.11e-15 | 1.42e-14 | 1.42e-14 |
| `rho_G@1.000000e+05` | 2.71e-18 | 1.74e-18 | 5.06e-18 | 1.72e-18 | 5.12e-18 | 3.74e-18 |
| `rho_G@1.000000e+07` | 6.55e-18 | 2.26e-18 | 3.11e-18 | 1.30e-18 | 4.93e-18 | 2.99e-18 |
| `rho_G@3.000000e+08` | 5.68e-18 | 5.62e-18 | 2.50e-18 | 3.54e-18 | 2.58e-18 | 3.46e-18 |
| `rho_T@1.000000e+05` | 6.19e-12 | 4.16e-17 | 4.16e-17 | 4.86e-17 | 4.16e-17 | 5.55e-17 |
| `rho_T@1.000000e+07` | 6.19e-12 | 3.47e-17 | 4.16e-17 | 4.86e-17 | 4.16e-17 | 4.86e-17 |
| `rho_T@3.000000e+08` | 6.11e-12 | 4.16e-17 | 2.78e-17 | 1.39e-17 | 1.39e-17 | 1.39e-17 |

QCD, `plain`:

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 1.12e-07 | 1.60e-08 | 1.01e-08 | 4.25e-09 | 2.06e-09 | 4.77e-09 |
| `cs_tau` | 9.85e-09 | 9.32e-09 | 5.83e-09 | 2.46e-09 | 1.19e-09 | 2.76e-09 |
| `friction` | 1.38e-06 | 1.23e-06 | 4.19e-07 | 1.39e-08 | 3.78e-07 | 6.30e-08 |
| `rho_G@1.000000e+05` | 4.65e-10 | 2.74e-10 | 2.32e-10 | 8.18e-11 | 4.73e-11 | 1.04e-10 |
| `rho_G@1.000000e+07` | 1.95e-11 | 2.74e-12 | 2.32e-12 | 8.19e-13 | 4.73e-13 | 1.04e-12 |
| `rho_G@3.000000e+08` | 1.43e-09 | 1.08e-09 | 3.99e-10 | 2.42e-11 | 3.60e-10 | 6.14e-11 |
| `rho_T@1.000000e+05` | 1.53e-09 | 1.84e-09 | 1.07e-09 | 4.78e-10 | 2.19e-10 | 5.15e-10 |
| `rho_T@1.000000e+07` | 4.58e-11 | 1.83e-11 | 1.07e-11 | 4.80e-12 | 2.18e-12 | 5.16e-12 |
| `rho_T@3.000000e+08` | 1.16e-09 | 1.14e-09 | 3.64e-10 | 4.96e-12 | 3.29e-10 | 5.40e-11 |

QCD, `branch+knots`:

| integrand | N=2 | N=4 | N=6 | N=8 | N=12 | N=16 |
|---|---|---|---|---|---|---|
| `tau` | 8.39e-08 | 3.64e-12 | 1.82e-12 | 1.82e-12 | 1.82e-12 | 1.82e-12 |
| `cs_tau` | 3.59e-08 | 7.96e-13 | 6.82e-13 | 6.82e-13 | 6.82e-13 | 5.68e-13 |
| `friction` | 2.37e-10 | 1.42e-14 | 1.42e-14 | 7.11e-15 | 1.42e-14 | 1.42e-14 |
| `rho_G@1.000000e+05` | 1.03e-08 | 4.68e-17 | 3.04e-18 | 3.25e-18 | 3.14e-18 | 3.36e-18 |
| `rho_G@1.000000e+07` | 3.17e-10 | 8.20e-19 | 8.91e-19 | 1.83e-19 | 2.51e-19 | 1.56e-19 |
| `rho_G@3.000000e+08` | 1.05e-08 | 2.71e-16 | 6.51e-19 | 1.73e-18 | 1.08e-18 | 1.30e-18 |
| `rho_T@1.000000e+05` | 1.79e-08 | 1.11e-16 | 9.02e-17 | 9.02e-17 | 9.02e-17 | 8.33e-17 |
| `rho_T@1.000000e+07` | 5.92e-10 | 1.80e-16 | 1.80e-16 | 1.67e-16 | 1.80e-16 | 1.67e-16 |
| `rho_T@3.000000e+08` | 2.03e-08 | 3.75e-16 | 6.66e-16 | 6.52e-16 | 6.52e-16 | 6.38e-16 |

---

## 5. Smoothness of the residual — the bound prompts 09 and 10 inherit

$\rho(u)=\int_u^{u_{\rm anchor}}g$, so $\rho''=-g'$ and $\rho''''=-g'''$; both are central
finite differences of the integrand on the production grid's own local spacing $h$. The last column
is $h^4\max|\rho''''|/384$, the cubic-spline interpolation error of the *residual* spline that
`PrimitivePhase` (prompt 09) and `TkSourceFunctions` (prompt 10) will build. On QCD the stencil
straddles the break points of §2 at a handful of nodes, so the QCD maxima are upper bounds
contaminated by those few points rather than a statement about the smooth part. Even so, the worst
predicted $\varphi$-spline error is $3\times10^{-7}$ rad — well inside README §6's $10^{-6}$ rad
consumer target, and set by the QCD transition, not by the residual's generic smoothness.

| model | integrand | $\max|\rho''|$ | $\max|\rho''''|$ | $h^4\max|\rho''''|/384$ [rad] | at $z$ |
|---|---|---|---|---|---|
| LambdaCDMModel | `rho_G@1.000000e+05` | 3.86e-09 | 3.26e-09 | 1.22e-18 | 3726 |
| LambdaCDMModel | `rho_G@1.000000e+07` | 3.86e-11 | 3.26e-11 | 1.22e-20 | 3726 |
| LambdaCDMModel | `rho_G@3.000000e+08` | 1.29e-12 | 5.10e-12 | 3.74e-21 | 6.219e+12 |
| LambdaCDMModel | `rho_T@1.000000e+05` | 8.31e-02 | 9.01e-02 | 6.60e-11 | 2.202e+09 |
| LambdaCDMModel | `rho_T@1.000000e+07` | 8.13e-02 | 8.78e-02 | 6.43e-11 | 2.154e+11 |
| LambdaCDMModel | `rho_T@3.000000e+08` | 8.19e-02 | 8.86e-02 | 6.49e-11 | 6.512e+12 |
| QCDModel | `rho_G@1.000000e+05` | 9.36e-03 | 4.86e+01 | 3.56e-08 | 4.388e+07 |
| QCDModel | `rho_G@1.000000e+07` | 1.97e-03 | 1.98e+00 | 1.45e-09 | 2.057e+11 |
| QCDModel | `rho_G@3.000000e+08` | 7.52e-02 | 2.36e+02 | 1.73e-07 | 8.384e+11 |
| QCDModel | `rho_T@1.000000e+05` | 8.13e-02 | 8.44e+01 | 6.19e-08 | 4.388e+07 |
| QCDModel | `rho_T@1.000000e+07` | 8.05e-02 | 3.62e+00 | 2.66e-09 | 1.965e+11 |
| QCDModel | `rho_T@3.000000e+08` | 1.28e-01 | 4.26e+02 | 3.12e-07 | 8.384e+11 |

---

## 6. Build cost at the recommended orders

Integrand evaluations (one background evaluation each) and wall time for one full table on the
1731-interval production grid. `QCDModel` construction itself
(`compute_background` plus the derivative splines) costs 0.398 s and is
not counted here. The `rho` rows are per $k$ and per anchor, at $k=3\times10^8$ (the longest
range); the three tables are built once per model.

| model | table | order | integrand evaluations | seconds |
|---|---|---|---|---|
| LambdaCDMModel | `tau` | 4 | 6924 | 0.006 |
| LambdaCDMModel | `cs_tau` | 4 | 6924 | 0.008 |
| LambdaCDMModel | `friction` | 4 | 6924 | 0.006 |
| LambdaCDMModel | `rho_G@3.000000e+08` | 4 | 5536 | 0.011 |
| LambdaCDMModel | `rho_T@3.000000e+08` | 4 | 5536 | 0.013 |
| QCDModel | `tau` | 4 | 8552 | 0.055 |
| QCDModel | `cs_tau` | 4 | 8552 | 0.149 |
| QCDModel | `friction` | 4 | 8552 | 0.101 |
| QCDModel | `rho_G@3.000000e+08` | 4 | 6904 | 0.114 |
| QCDModel | `rho_T@3.000000e+08` | 4 | 6904 | 0.277 |

Three tables together: 20772
evaluations in 0.020 s (LambdaCDM),
25656 in
0.305 s (QCD, `branch+knots`).

---

## 7. Exact-radiation controls

`RadiationModel` has $C\equiv0$ for the Green's function, so **every** Gauss increment of $\rho_G$
must be bit-exactly zero at every order. Measured, at all 6 orders and all three $k$:
all bit-exactly zero.

$\rho_T$ is scored against `RadiationModel`'s closed-form primitive
$g(s)=-2s/(\sqrt{a^2-2s^2}+a)+\sqrt2\arcsin(\sqrt2 s/a)$, $a=k/(\sqrt3 H_0)$. Review §12.4's
$1/x_i-1/x$ is that primitive's *large-$x$ asymptote*, and carries the opposite sign in this
campaign's convention (`wkb_reference.RadiationModel` docstring), so its departure from the closed
form is reported rather than used as the reference — at $x_{T,i}\approx12$ that departure is a
few parts in $10^3$, far above any quadrature error.

| $k$ | $x_{T,\rm init}$ | $x_{T,\rm final}$ | rel. error at $N_\rho$ | departure of the asymptote from the closed form |
|---|---|---|---|---|
| 1.000000e+05 | 11.6 | 5.249e+04 | 2.01e-16 | 1.24e-03 |
| 1.000000e+07 | 11.6 | 5.249e+06 | 1.64e-16 | 1.24e-03 |
| 3.000000e+08 | 11.6 | 1.575e+08 | 9.53e-16 | 1.24e-03 |

---

## 8. What prompts 03–07 must do

1. Build every table with the orders in §1: $N_\tau=4$, $N_{\tau_s}=4$,
   $N_F=4$, $N_\rho=4$.
2. On a cosmology that has break points — `QCD_Cosmology` does, through `QCD_EOS.T_LO`,
   `QCD_EOS.EOS_T_LO`, `QCD_EOS.T_120_MEV` and the `T(z)` spline's knots — **split the production
   interval at each break point it contains** and integrate each sub-panel separately. On
   LambdaCDM the break-point list is empty and the scheme reduces to review §7's as written.
   Note that `EOS_T_LO` breaks `c_s^2` only, so it matters for $\tau_s$, $F$ and $\rho_T$ and not
   for $\tau$ or $\rho_G$; splitting on the union costs nothing measurable and is simpler.
3. The cumulative errors of §4 under `branch+knots` are the accuracies prompt 13 should expect to
   reproduce; the `plain` rows are what the review's design as written would have given.
