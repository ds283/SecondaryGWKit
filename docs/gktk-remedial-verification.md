# Gk/Tk WKB phase campaign — verification and close-out

**Campaign:** [`prompts/GkTk-remedial/README.md`](../prompts/GkTk-remedial/README.md)
**Status board:** [`prompts/GkTk-remedial/IMPLEMENTATION_STATE.md`](../prompts/GkTk-remedial/IMPLEMENTATION_STATE.md)
**Source review:** [`docs/gk-wkb-review-fable-2026-09-09.md`](gk-wkb-review-fable-2026-09-09.md) §4, §12.3, §13.5
**Reconciliation:** [`prompts/GkTk-remedial/RECONCILIATION.md`](../prompts/GkTk-remedial/RECONCILIATION.md)
**Prompt:** [`prompts/GkTk-remedial/13-verification-and-docs.md`](../prompts/GkTk-remedial/13-verification-and-docs.md)
**Date:** 2026-09-13
**Tree:** every measurement below was taken on branch `gktk-remedial` at **`ff9ee29`** (prompt 20's
commit, the campaign tip), clean. This commit sits on top of it and changes no production code.

---

## 1. What this document is

Twenty prompts landed the campaign. This document records the two layers prompt 13 asks for — the
production-path measurements offline on both production background models, then one scoped,
live pipeline run per model on a **fresh** datastore — and says plainly what each settled and what
it did not.

**Headline results.**

1. **The stored WKB phase is at the double-precision floor on both production models, in both
   sectors.** At $z=0.1$ the review's two headline errors are gone: $\theta_G$ on LambdaCDM goes
   from **13.9 rad to 0.0 rad** at $k=10^5$/Mpc and from **7366 rad to $9.77\times10^{-4}$ rad** at
   $k=3\times10^8$/Mpc, the latter being exactly one ulp of the $4.1\times10^{12}$ rad phase it
   carries; $\theta_T$ goes from **2.01 rad to $1.49\times10^{-8}$ rad** and from
   **$5.1\times10^3$ rad to $9.16\times10^{-5}$ rad**. On `QCD_Cosmology`, where the review took no
   measurement at all, the same figures are $4.77\times10^{-7}$ / $4.88\times10^{-4}$ rad
   ($\theta_G$) and $7.45\times10^{-8}$ / $2.44\times10^{-4}$ rad ($\theta_T$).
2. **The cost is gone with it.** A `GkWKBIntegration` object at $k=3\times10^8$/Mpc on LambdaCDM
   costs **0.0010 s and 468 integrand evaluations** against the ODE's 63.7 s and
   $2.5\times10^6$ — a factor $6\times10^4$ in time — and a `TkWKBIntegration` object
   **0.018 s and 5,540** against 58 s and $1.94\times10^6$.
3. **The consumers interpolate nothing that grows.** At fixed $z_r=0.1$ over the production source
   grid, scored at ten points per grid interval against the producer evaluated there,
   `PrimitivePhase` and `TkSourceFunctions.phase` are at **1.00 ulp of their own span in ten of the
   twelve (model, $k$, sector) cases**. The two that are not are both on `QCD_Cosmology` at
   $k=10^5$ and both have their maximum at $z=4.24\times10^7$, the `T_LO` branch boundary of the
   equation of state (8 and 428 ulp, i.e. 1.9e-6 and 3.2e-6 rad —
   `[13-consumer-spline-crosses-eos-break-points]`); the LambdaCDM $k=3\times10^8$ row is item 4.
   The review's $h^4x/384$ term, 8.3e-3 rad at $x=10^7$ and $O(1)$–$O(10)$ rad at production $x$, is
   nowhere visible above the representation floor.
4. **One defect was found, and is not fixed here** (prompt 13's own instruction).
   `LiouvilleGreen.WKBtools.WKB_mod_2pi` forms its cycle count from a **rounded** division while
   its remainder comes from an exact `fmod`, so at large $|\theta|$ the two can disagree by a whole
   cycle and the stored pair no longer reconstructs its own phase. Measured on the production
   geometry: **1 of 77,975** Green's-function samples at $k=3\times10^8$/Mpc on LambdaCDM, and it
   costs the consumer **6.17 rad** there — four orders above the $9.1\times10^{-4}$ rad floor
   everything else sits at. Board issue `[13-wkb-mod-2pi-cycle-count-inconsistent]`.
5. **Live, on a fresh datastore per model, the stored rows are the Layer 1 numbers.** Both scoped
   runs completed every stage prompt 13 §2 requires — background, $T_k$ numeric and WKB, $G_k$
   numeric and WKB — plus `QuadSource`, `GkSource` and `GkSourcePolicyData`; the QCD run **did not
   fail**, finishing in 11 minutes. Re-running the production producer offline on each row's own
   stored initial data and applying the production `store()` algebra reproduces
   `(theta_div_2pi, theta_mod_2pi)` **bit-identically at 13,430 sampled values across the two
   models**, the persisted double-double limbs agree with an independent reference to 4.6e-16
   relative, `WKB_phase_spline_chunks` is **1** on all 1,388 rows that populate it, and every
   stage-2 column is NULL on all 15,940 WKB rows.

6. **Three open issues close on measurements the board had assigned to this prompt.** The interval
   accessor's Levin-side cost in bulk on `QCD_Cosmology` is **29.3 µs per call**, below the
   campaign's 50 µs stop threshold (`[01-offgrid-accessor-cost-on-qcd]`); the residual table's
   cut-to-anchor margin never falls below **1.735 e-folds** over the whole production $k$ range on
   either model in either sector (`[14-residual-range-top-margin]`); and `theta_deriv` reproduces
   $\omega$ on the real background inside the wider of the two windows prompt 10 shipped and within
   1.7× of the tighter one, so **the cubic residual spline stands and `spline_order=5` is not
   taken** (`[10-residual-spline-end-condition]`).

Nothing in production code was changed by this prompt.

---

## 2. Reconciliation

### 2.1 Git history

The campaign's commits on `gktk-remedial`, in the order they landed. The three `transfer-remedial`
and `qsi-phase-groups` commits that were merged in at `e01c31d` are marked; they are another
campaign's and are listed only so that the tree this document measures is fully described.

| Prompt | Commit | Subject |
|---|---|---|
| — | `4980185` | Plan the Gk/Tk WKB phase remediation as thirteen prompts |
| 01 | `dcd99fd` | Add WKB phase references and a measured primitive prototype |
| 02 | `e912593` | Measure Gauss-order convergence of the WKB primitives on both models |
| — | `e01c31d` | Merge the Bessel phase campaigns into `gktk-remedial` (`transfer-remedial`, `qsi-phase-groups`) |
| — | `c5e835e` | Point the last four Gk WKB review references at its new name |
| 03 | `83ef7c5` | Build conformal time as a double-double Gauss-Legendre table |
| 04 | `680ed84` | Tabulate the sound horizon and the LG friction integral per model |
| 05 | `cf986b5` | Add the WKB phase residual as a per-k table |
| 06 | `243cb84` | Compute the Green function WKB phase from the conformal-time table |
| 14 | `b183cb8` | Build the WKB phase residual once per wavenumber |
| 07 | `2873e15` | Compute the transfer-function WKB phase and friction from tables |
| 08 | `bb6a4c8` | Drop the chunked phase spline in favour of one rebased spline |
| 09 | `c1c3717` | Evaluate the Green function phase from the conformal-time table |
| 10 | `8ba58e7` | Evaluate the transfer-function phase and friction from tables |
| 15 | `2ed3632` | Give `PrimitivePhase` an explicit leading-rate callable |
| 11 | `8f606c4` | Test oscillation resolution on the sample grid, off the RHS |
| 16 | `45da2cd` | Summarise unresolved-oscillation warnings per wavenumber |
| 12 | `25e5b6f` | Give the transfer-function numeric run its own absolute tolerance |
| 17 | `6037cf3` | Measure the transfer-function numeric tolerance across the k-grid |
| 18 | `2f5664e` | Split the numeric ODE at the cosmology's declared discontinuities |
| 19 | `5b84d06` | Let each numeric sector choose which declared break points it splits at |
| 20 | `ff9ee29` | Put the numeric break-point policy in the datastore lookup key |

Orchestrator and prompt-text commits (`2e634fd`, `c0a0a9e`, `42773ba`, `b65539e`, `79e0757`,
`837e909`, `7f6c393`, `a2ea069`, `e6f88f4`, `aae6ac5`, `866eeee`, `9689949`, `0038f28`, `003ad5c`,
`121de53`, `340abe9`, `a38c200`, `dcdf51c`, `243c239`, `891d84a`, `2b4f30c`, `fa87cd8`) touch only
`prompts/` and `docs/OPEN_ISSUES.md`.

### 2.2 Environment

Python 3.12, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, Ray 2.43.0 — the review's environment
(`RECONCILIATION.md` §0), plus Ray, which the review did not need. macOS (Darwin 25.5.0), 10 cores,
16 GB.

### 2.3 What "before" means in the tables below

The **before** column is the review's own measurement on `f06f587` of the two-stage phase ODE this
campaign deleted. That ODE no longer exists in the tree, so it cannot be re-run here; prompt 01
reproduced the two $k=10^5$ figures on `9ff59d5` before deleting it and recorded them in
`ComputeTargets/tests/wkb_reference_data.json`'s `baselines` block — **13.905 rad** ($\theta_G$) and
**2.0123 rad** ($\theta_T$) against the review's 13.9 and 2.01, i.e. 0.04 % and 0.11 % — which is
what licenses quoting the review's other rows as the before column.

The review's $z$ rows and prompt 01's JSON checkpoints coincide only at $z=0.1$. The tables below
are at the checkpoints, and the review's own figure is printed beside a row only where the two
redshifts agree to 1 %.

---

## 3. Layer 1 — production-path measurements, offline

Script: [`docs/gktk-remedial/verify_production_path.py`](gktk-remedial/verify_production_path.py),
46 s for all six sections. Everything runs through the production functions — the undecorated
`WKB_phase_function`, `compute_background`'s `TablePrimitive` accessors, `TkWKBIntegration.store()`,
`PrimitivePhase` and `TkSourceFunctions` — with the reference harness's stand-in *models*
(`ComputeTargets/tests/wkb_reference.py`) and the stand-in *fixtures* of
`ComputeTargets/tests/test_gk_wkb_phase.py` and `test_tk_wkb_phase.py`. No Ray, no datastore.

Error definitions are prompt 01's, unchanged (README §6): **phase error** is the absolute
difference in radians of the unwrapped phase against the reference at the supplied double `z`;
**difference error** is relative to the interval quantity, never to the absolute.

### 3.1 The primitives at the production nodes

Against prompt 01's references at the JSON checkpoints, on the 1,732-node production grid.

| Quantity | LambdaCDM | at $z$ | QCD | at $z$ |
|---|---|---|---|---|
| $\tau$, relative | **3.810e-16** | 1002.5 | **2.104e-14** | 1.005e7 |
| $\tau_s$, relative | 2.496e-16 | 1.009e13 | 2.108e-14 | 1.005e7 |
| $F$, relative | 3.314e-16 | 1.005e7 | 3.340e-16 | 1.005e7 |
| $\Delta\tau$ over one production interval, relative | 2.494e-16 | 1.0006 | 9.354e-15 | 100.19 |
| $\Delta\tau$ over a 37 % fraction (one off-grid endpoint) | 4.400e-16 | 1.004e6 | 3.408e-14 | 1.004e6 |
| $\rho$ vs reference, absolute (worst over both sectors, all three $k$) | 4.163e-17 rad | `Tk`, 3e8, $z=1.006\times10^9$ | 3.608e-16 rad | `Tk`, 3e8, $z=1.007\times10^{11}$ |

Every figure reproduces the prompt that produced it (logs 03, 04, 05) to the printed digits. The
QCD $\tau$ and $\tau_s$ rows are **at** the JSON reference's own floor of 1.88e-14 — not above it —
which is `[02-qcd-reference-floor]`, and the QCD fraction row carries the reference's rounded-$u$
endpoint error, `[03-qcd-short-baseline-reference-endpoint-rounding]`. Neither is the table's error.

**Interval-accessor cost per call** (best of 5, 200 calls per configuration):

| | on-grid | one endpoint off-grid | both off-grid |
|---|---|---|---|
| LambdaCDM | 0.32 µs | 3.54 µs | 6.56 µs |
| `QCD_Cosmology` | 0.33 µs | **33.3 µs** | 64.7 µs |

Log 03 measured 0.44 / 4.6 / 6.6 and 0.33 / 26.1 / 52.0 µs for the same three. The production and
consumer patterns are on-grid or one endpoint off-grid (§3.5); both-off-grid is above README §4.3's
50 µs line on QCD and no production path reaches it.

### 3.2 Review §4 re-measured — $\theta_G$ on the real background

Anchored at prompt 01's 3-e-fold residual anchor; the sample set is the JSON checkpoints below it.
"after" is the phase error in radians of the stored `theta_div_2pi * 2π + theta_mod_2pi`.

**LambdaCDM, $k=10^5$/Mpc** (span 1.3728e9 rad; $\varepsilon|\theta|$ floor 3.05e-7 rad):

| $z$ | $\theta_{\rm ref}$ [rad] | after [rad] | before (review §4) |
|---|---|---|---|
| 1.006e9 | −2.598e1 | 3.55e-15 | |
| 1.005e7 | −4.592e3 | 9.09e-13 | |
| 1.004e6 | −4.609e4 | 7.28e-12 | |
| 1.003e4 | −4.283e6 | 0.00 | |
| 1002.5 | −2.984e7 | 1.12e-8 | |
| 100.19 | −1.330e8 | 2.98e-8 | |
| 10.012 | −4.523e8 | **1.19e-7** | |
| 1.0006 | −1.076e9 | 0.00 | |
| **0.1** | **−1.3728e9** | **0.00** | **+13.9** |

**LambdaCDM, $k=3\times10^8$/Mpc** (span 4.1184e12 rad; floor 9.15e-4 rad):

| $z$ | $\theta_{\rm ref}$ [rad] | after [rad] | before (review §4) |
|---|---|---|---|
| 1.007e11 | −1.360e3 | 2.27e-13 | |
| 1.006e9 | −1.382e5 | 2.91e-11 | |
| 1.005e7 | −1.384e7 | 0.00 | |
| 1.004e6 | −1.383e8 | 2.98e-8 | |
| 1.003e4 | −1.285e10 | 3.81e-6 | |
| 1002.5 | −8.952e10 | 3.05e-5 | |
| 100.19 | −3.991e11 | 6.10e-5 | |
| 10.012 | −1.357e12 | 2.44e-4 | |
| 1.0006 | −3.229e12 | **9.77e-4** | |
| **0.1** | **−4.1184e12** | **9.77e-4** | **+7366** |

$9.77\times10^{-4}$ rad is exactly one ulp of $4.1\times10^{12}$: the phase is delivered as
accurately as a double can hold it. The intermediate $k$ and the QCD model, in one line each
(maximum over the checkpoints, and the row at $z=0.1$):

| model | $k$ [1/Mpc] | span [rad] | max error [rad] | at $z$ | at $z=0.1$ | $\varepsilon|\theta|$ floor |
|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | 1.3728e9 | 1.192e-7 | 10.012 | 0.00 | 3.05e-7 |
| LambdaCDM | 1e7 | 1.3728e11 | 3.052e-5 | 1.0006 | 3.05e-5 | 3.05e-5 |
| LambdaCDM | 3e8 | 4.1184e12 | 9.766e-4 | 1.0006 | 9.77e-4 | 9.15e-4 |
| QCD | 1e5 | 1.3728e9 | 4.768e-7 | 1.0006 | 4.77e-7 | 3.05e-7 |
| QCD | 1e7 | 1.3728e11 | 4.578e-5 | 1.0006 | 3.05e-5 | 3.05e-5 |
| QCD | 3e8 | 4.1184e12 | 9.766e-4 | 1.0006 | 4.88e-4 | 9.15e-4 |

Every maximum is within a factor 1.6 of the floor of its own span. README §6's two $\theta_G$ rows
($\le10^{-5}$ rad at $k=10^5$, $\le5\times10^{-3}$ rad at $3\times10^8$) are met with two orders
and a factor five to spare.

### 3.3 Review §12.3 re-measured — $\theta_T$ and the friction integral

Same geometry, `sector="Tk"`, `friction=True`.

**LambdaCDM, $k=10^5$/Mpc** (span 6.1747e7 rad; floor 1.37e-8 rad):

| $z$ | $\theta_{T,\rm ref}$ [rad] | after [rad] | before (review §12.3) |
|---|---|---|---|
| 1.006e9 | −1.495e1 | 3.55e-15 | |
| 1.005e7 | −2.651e3 | 0.00 | |
| 1.004e6 | −2.659e4 | 7.28e-12 | |
| 1.003e4 | −2.297e6 | 0.00 | |
| 1002.5 | −1.163e7 | 1.86e-9 | |
| 100.19 | −2.787e7 | 7.45e-9 | |
| 10.012 | −4.510e7 | 7.45e-9 | |
| 1.0006 | −5.816e7 | 7.45e-9 | |
| **0.1** | **−6.1747e7** | **1.49e-8** | **+2.01** |

| model | $k$ [1/Mpc] | span [rad] | max $\theta_T$ error | at $z$ | at $z=0.1$ | floor | max $|\delta F|/|F|$ |
|---|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | 6.1747e7 | 1.490e-8 | 0.1 | 1.49e-8 | 1.37e-8 | 6.535e-16 |
| LambdaCDM | 1e7 | 6.1747e9 | 9.537e-7 | 100.19 | 9.54e-7 | 1.37e-6 | 3.763e-16 |
| LambdaCDM | 3e8 | 1.8524e11 | 9.155e-5 | 0.1 | 9.16e-5 | 4.11e-5 | 3.964e-16 |
| QCD | 1e5 | 6.1745e7 | 7.451e-8 | 10.012 | 7.45e-8 | 1.37e-8 | 1.345e-15 |
| QCD | 1e7 | 6.1745e9 | 7.629e-6 | 1.0006 | 7.63e-6 | 1.37e-6 | 1.782e-15 |
| QCD | 3e8 | 1.8523e11 | 2.441e-4 | 1.0006 | 2.44e-4 | 4.11e-5 | 2.665e-14 |

The review's own $k=3\times10^8$ row is 5.1e3 rad at $z=0.1$; this is 9.16e-5 rad, a factor
$5.6\times10^7$. README §6's two $\theta_T$ rows ($\le10^{-4}$ and $\le5\times10^{-3}$ rad) are met.
The friction integral, which the ODE delivered to 2.3e-7–4.1e-7 relative, is now at 4e-16 to
2.7e-14 — the table's own accuracy, seven orders below the $T_k$ Liouville–Green truncation floor
that bounds what any of it means (`[00-tk-lg-truncation-floor]`).

### 3.4 $F$ and $\rho_T$ as stored (prompt 13 §1 item 3)

Through the production `TkWKBIntegration.store()`, reading the persisted columns back. $\rho_T$ is
recovered from the stored phase by removing the constant initial-data offset $\delta$ and the
leading sound-horizon term, so its error inherits the $\varepsilon k\tau_s$ floor of the
`div * 2π + mod` representation, which is what the 2.2e-5 rad at $k=3\times10^8$ is.

| model | $k$ [1/Mpc] | max $|F_{\rm stored}/F_{\rm ref}-1|$ | max $|\rho_T-\rho_{T,\rm ref}|$ [rad] | $\rho_T(z=0.1)$ [rad] | `sin_coeff` | `cos_coeff` |
|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | 6.535e-16 | 2.421e-9 | −0.086343 | 7.169e-8 > 0 | 0.0 |
| LambdaCDM | 1e7 | 3.763e-16 | 2.017e-7 | −0.086341 | 7.169e-9 > 0 | 0.0 |
| LambdaCDM | 3e8 | 3.964e-16 | 2.212e-5 | −0.086341 | 1.309e-9 > 0 | 0.0 |
| QCD | 1e5 | 1.345e-15 | 1.012e-8 | −0.088672 | 6.896e-8 > 0 | 0.0 |
| QCD | 1e7 | 1.782e-15 | 9.471e-7 | −0.089025 | 6.615e-9 > 0 | 0.0 |
| QCD | 3e8 | 2.665e-14 | 3.650e-5 | −0.093103 | 1.061e-9 > 0 | 0.0 |

$\rho_T$ reproduces logs 05 and 07 to six figures (−0.086343 / −0.093103), and it is **not**
negligible: it is 0.086–0.093 rad, four orders above the phase error the campaign now delivers, so
review §12.2's insistence that it be carried is confirmed on the production path.
`sin_coeff = B > 0` and `cos_coeff` exactly zero in every case — the removed sign fix was indeed a
no-op (review §8.1, M10).

### 3.5 The consumers at production $x$ (prompt 13 §1 item 2)

**Geometry.** For the Green's function: the production pure-WKB band of `main.py:1588`
($z_{\rm source} < \sqrt{z_{e3}z_{e4}}$, where `G_init = 0`, `Gprime_init = 1`, $z_{\rm init} =
z_{\rm source}$ and hence $\delta = {\rm atan2}(0, +) = 0$ exactly), at the production
`--zend 0.1`, with the stored phase produced by the production producer and $\varphi$ recovered by
the production `build_phi_samples`. For the transfer function: a real `TkWKBIntegration` built and
stored by `build_and_store` over the whole source grid below the hand-over, fed to the production
`TkSourceFunctions` constructor with a stand-in numeric region. Both are scored at **ten points per
production grid interval** against the producer evaluated at those same points — the producer's
leading term is the double-double table's exact interval and its residual the order-4 table's, so
what the comparison isolates is exactly what the consumer approximates, the cubic spline of
$\varphi$.

| model | $k$ [1/Mpc] | sector | samples | scored | max error [rad] | at $z$ | span [rad] | in ulp | $\varepsilon k\tau$ |
|---|---|---|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 1014 | 10130 | **2.384e-7** | 1.15e6 | 1.3728e9 | 1.00 | 3.05e-7 |
| LambdaCDM | 1e7 | $G_k$ | 1214 | 12130 | **0.00** | — | 1.3728e11 | 0.00 | 3.05e-5 |
| LambdaCDM | 3e8 | $G_k$ | 1361 | 13600 | 6.175 † | 3.33e4 | 4.1184e12 | 12646 | 9.15e-4 |
| QCD | 1e5 | $G_k$ | 1016 | 10150 | **1.907e-6** | 4.24e7 | 1.3728e9 | 8.00 | 3.05e-7 |
| QCD | 1e7 | $G_k$ | 1218 | 12170 | **1.526e-5** | 1.20e11 | 1.3728e11 | 1.00 | 3.05e-5 |
| QCD | 3e8 | $G_k$ | 1377 | 13760 | **4.883e-4** | 5.11e12 | 4.1184e12 | 1.00 | 9.15e-4 |
| LambdaCDM | 1e5 | $T_k$ | 1037 | 10360 | **7.451e-9** | 147.3 | 6.1746e7 | 1.00 | 1.37e-8 |
| LambdaCDM | 1e7 | $T_k$ | 1236 | 12350 | **9.537e-7** | 12.19 | 6.1746e9 | 1.00 | 1.37e-6 |
| LambdaCDM | 3e8 | $T_k$ | 1384 | 13830 | **3.052e-5** | 8.81 | 1.8524e11 | 1.00 | 4.11e-5 |
| QCD | 1e5 | $T_k$ | 1040 | 10390 | **3.186e-6** | 4.24e7 | 6.1744e7 | 427.6 | 1.37e-8 |
| QCD | 1e7 | $T_k$ | 1242 | 12410 | **9.537e-7** | 12.74 | 6.1744e9 | 1.00 | 1.37e-6 |
| QCD | 3e8 | $T_k$ | 1401 | 14000 | **3.052e-5** | 8.24 | 1.8523e11 | 1.00 | 4.11e-5 |

† The one row that is not at the floor is the `WKB_mod_2pi` defect of §3.7, not an interpolation
error: **one** of the 1,361 stored samples carries a $\varphi$ that is a whole cycle from its
neighbours, at $z=33{,}226$. Excluding the fifteen grid intervals either side of it — where a cubic
spline's response to one bad ordinate has decayed by $(\sqrt3-2)^{15}$ — the maximum over the
remaining 13,300 points is **exactly 0.00 rad**.

$x_{\rm source}$ at the foot of the band is $4.6\times10^8$ ($k=10^5$) to $1.4\times10^{12}$
($3\times10^8$), and $x_T$ at $z=0.1$ is $4.8\times10^6$ to $1.4\times10^{10}$ — so these are the
production $x$ values the review says no sample density can spline. README §6's consumer row
(interpolation error $\le10^{-6}$ rad at $x=10^7$, 100/decade, against 8.3e-3 rad before) is met:
the two rows whose span is comparable to $x=10^7$ are the $k=10^5$ ones, at 2.4e-7 and 7.5e-9 rad,
both one ulp.

The two QCD rows that are several ulp — $G_k$ at $k=10^5$ (1.9e-6 rad, 8 ulp) and $T_k$ at $k=10^5$
(3.2e-6 rad, 428 ulp) — **exceed README §6's $10^{-6}$ rad target**, and both have their maximum at
$z=4.24\times10^7$, which is `QCD_EOS`'s `T_LO` branch boundary, where $H(z)$ jumps by 4.4e-4
(log 02). The residual $\varphi$ has a kink there and a cubic spline of it does not: this is the
cosmology's own non-smoothness showing through the consumer, it is a factor 500 below the
Liouville–Green truncation floor of ~1e-3 rad that bounds any QCD phase claim, and it is recorded as
`[13-consumer-spline-crosses-eos-break-points]` rather than fixed.

### 3.6 `theta_deriv` against `omega` — `[10-residual-spline-end-condition]` on the real background

The board assigns this identity to prompt 13, which is to "re-measure it on the real background for
both sectors and then either close this issue at the cubic or escalate to `spline_order=5`", and to
"report the error at the second and third samples, not only the window maxima".

Relative error of $|{\rm phase.theta\_deriv}(z)|$ against $\omega(z)$ from `*_omegaEff_sq`, over the
consumer's own sample set:

| model | $k$ | sector | max, all samples | at $z$ | `[3:-3]` | `[3:-5]` | `[3:-8]` | deep interior |
|---|---|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 5.503e-10 | 1.39e9 | 5.135e-10 | 4.904e-10 | 4.577e-10 | 4.371e-10 |
| LambdaCDM | 1e5 | $T_k$ | 1.764e-8 | 2.31e9 | 3.031e-10 | **1.663e-11** | 1.020e-11 | 8.828e-12 |
| LambdaCDM | 1e7 | $T_k$ | 1.681e-8 | 2.26e11 | 2.890e-10 | 1.580e-11 | 9.689e-12 | 8.389e-12 |
| LambdaCDM | 3e8 | $T_k$ | 1.709e-8 | 6.82e12 | 2.938e-10 | 1.608e-11 | 9.865e-12 | 8.538e-12 |
| QCD | 1e5 | $G_k$ | 5.128e-7 | 1.45e9 | 2.312e-7 | 2.312e-7 | 2.312e-7 | 2.312e-7 |
| QCD | 1e7 | $G_k$ | 1.098e-5 | 1.49e11 | 7.043e-6 | 6.768e-6 | 6.540e-6 | 6.084e-6 |
| QCD | 3e8 | $G_k$ | 3.309e-4 | 5.17e12 | 3.309e-4 | 3.309e-4 | 1.748e-4 | 1.748e-4 |
| QCD | 1e5 | $T_k$ | 1.100e-6 | 1.75e9 | 1.100e-6 | 1.100e-6 | 1.100e-6 | 6.992e-7 |

The windows are the shipped test's, read in its sense: `z_WKB` is in redshift order, so its
`[3:-3]` and `[5:-3]` trim 3 and 5 samples from the **high-$z$ (hand-over) end**, which is where the
end condition lives; the columns above are trimmed the same way (3 from the low-$z$ end throughout,
$m$ from the high-$z$ end).

**The end condition is real and it decays exactly as the issue says.** The last five $T_k$ samples,
at the hand-over where $\varphi\sim-1/x$ varies fastest, run 9.2e-11, 2.9e-10, 1.2e-9, 4.5e-9,
1.7e-8 on LambdaCDM at **every** $k$ — a factor ~3.8 per sample inwards, against the ~3× prompt 10
measured on its closed-form fixture — and the deep interior is 8.5e-12.

**On LambdaCDM, against the two bounds prompt 10 shipped:**

- `< 1e-9` over `[3:-3]` — **met**, at 2.9–3.0e-10 (the fixture gave 1.0492e-10 for the same window,
  a 4.9 % miss of the 1e-10 prompt 10 §3 item 3 asked for; the real background is 3× larger there
  and comfortably inside the shipped bound).
- `< 1e-11` over `[5:-3]` — **missed by 1.7×** on the real background, at 1.66e-11; it is met from
  about the seventh sample in (1.02e-11 at `[3:-8]`) and the deep interior is 8.5e-12. The shipped
  assertion is on the *fixture*, where it passes, and the whole suite passes unchanged; what this
  says is that the real background's residual is a little less smooth near the hand-over than the
  constant-$w$ closed form is, by under a factor two.

**The cubic therefore stands and `spline_order=5` is not taken.** It would need six samples against
`TkSourceFunctions.MIN_SPLINE_DATA_POINTS = 5`, it would make the $T_k$ consumer's representation
differ from prompt 09's $G_k$ consumer, and — decisively — it does not address the error that
actually dominates on the real background:

**On `QCD_Cosmology` the identity is missed by orders, and not at the ends.** 2.3e-7 to 3.3e-4
relative, *uniformly* across the interior — `[3:-3]`, `[3:-8]` and the deep interior are the same
number — driven by the same `T(z)` spline knots and equation-of-state branch boundaries as §3.5, and
at the QCD $G_k$ maximum ($z=5.2\times10^{12}$) by $\omega^2$ itself scattering between neighbouring
nodes (`[02-qcd-T-z-spline-node-tolerance]`). No spline order fixes that; a knot vector would.

`[10-residual-spline-end-condition]` is therefore **closed at the cubic** (§6), and what it turned
up is carried forward as `[13-consumer-spline-crosses-eos-break-points]`.

### 3.7 The stored `(div, mod)` pair at large $|\theta|$ — a defect found here

`LiouvilleGreen.WKBtools.WKB_mod_2pi` (`:15-27`) forms

```python
theta_mod_2pi = fmod(theta, TWO_PI)                     # exact
theta_div_2pi = int(floor(fabs(theta) / TWO_PI))        # a ROUNDED division, then floor
```

`fmod` is exact, but `fabs(theta) / TWO_PI` is a correctly-rounded double division. When the exact
quotient lies within half an ulp *below* an integer, the division rounds up across it, `floor`
returns one cycle too many, and the returned pair no longer reconstructs its own phase:
`div * 2π + mod == theta - 2π`. Worked example, from the LambdaCDM $k=3\times10^8$ consumer set:

```
theta            = -3832989103139.361
|theta| / TWO_PI = 610039162582.0        (rounded)     -> floor = 610039162582
exact quotient   = 610039162581.99994    (frac 0.99994) -> floor = 610039162581
fmod(theta, TWO_PI) = -6.282807898877543
div * TWO_PI + mod - theta  =  -6.283203125     (= -2 pi)
```

**What is and is not affected.** The stored `theta_mod_2pi` is correct — it is the `fmod`, and it
is what `G_WKB` and `T_WKB` are built from, so no stored *value* moves. What is wrong is
`theta_div_2pi`, and therefore every consumer that reconstructs the unwrapped phase from the pair,
which since prompts 09 and 10 is both of them (`build_phi_samples`, then
`PrimitivePhase.raw_theta`), and `QuadSourceIntegral`'s `_ClampedPhase` through them.

**Rate, measured on the production geometry.** The Green's-function exposure is the whole
(source × response) rectangle, since there is one object per $(k, z_{\rm source})$ and each holds
the response grid below its source; the transfer function is one object per $k$ over the source
grid.

| model | sector | $k$ [1/Mpc] | samples | inconsistent | rate | max $|\theta|$ |
|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 1e5 | 43,434 | 0 | 0 | 1.373e9 |
| LambdaCDM | $G_k$ | 1e7 | 62,117 | 0 | 0 | 1.373e11 |
| LambdaCDM | $G_k$ | **3e8** | **77,975** | **1** | **1.28e-5** | 4.118e12 |
| LambdaCDM | $T_k$ | 1e5 / 1e7 / 3e8 | 1037 / 1236 / 1384 | 0 | 0 | 1.852e11 |
| QCD | $G_k$ | 1e5 / 1e7 / 3e8 | 43,604 / 62,525 / 79,809 | 0 | 0 | 4.118e12 |
| QCD | $T_k$ | 1e5 / 1e7 / 3e8 | 1040 / 1242 / 1401 | 0 | 0 | 1.852e11 |

Uniform controls over 400,000 draws each: **0** at $|\theta|\sim10^9$, **0** at $10^{11}$, **25**
($6.25\times10^{-5}$) at $|\theta|\sim4\times10^{12}$, where half an ulp of $|\theta|/2\pi$ is
$6.1\times10^{-5}$ cycles — so the rate is the half-ulp width, as the mechanism predicts, and it
scales linearly with $|\theta|$, i.e. with $k$.

**Cost when it fires:** §3.5's 6.17 rad, against a 9.15e-4 rad floor. The `GkSource` rectifier does
**not** repair it: its trigger is `theta > last_theta` (a cycle count jumping *up* as the source
redshift rises), and this defect makes the stored phase one cycle *more negative*, so the condition
is false at the offending sample. Opened as `[13-wkb-mod-2pi-cycle-count-inconsistent]`; not fixed
here, because prompt 13 may not touch production code.

### 3.8 Two more open issues that close on measurements taken here

**`[01-offgrid-accessor-cost-on-qcd]`** — its recorded next step is "closes if a Levin-side
measurement on `QCD_Cosmology` (prompt 13) shows the per-call cost acceptable in bulk". Measured:
`PrimitivePhase.raw_theta` over 4,000 Chebyshev-like abscissae drawn uniformly in $u=\log(1+z)$
across the production band, best of 3:

| model | off-grid µs/call | integrand evaluations/call | on-grid µs/call | evaluations |
|---|---|---|---|---|
| LambdaCDM | 6.86 | 4.00 | 3.40 | 0.00 |
| `QCD_Cosmology` | **29.3** | 4.27 | 3.56 | 0.00 |

Exactly one order-4 panel per off-grid evaluation, as prompt 09 predicted, and **below the 50 µs
stop threshold** README §4.3 sets. The anchor ($z_{\rm response}$) is a grid node and costs nothing.
**Closed.**

**`[14-residual-range-top-margin]`** — its recorded next step is "prompt 13 measures the
cut-to-anchor margin across the production $k$ range on both models and both sectors ... and
records it; if it is ever below ~1 e-fold, the margin constant is what to revisit". Measured over
the 50-point production $k$ grid:

| model | sector | worst cut/anchor | e-folds | at $k$ [1/Mpc] | cut $z$ | anchor $z$ | fewest nodes |
|---|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 2981 | 8.000 | 3.000e8 | 2.064e16 | 6.923e12 | 1732 |
| LambdaCDM | $T_k$ | **5.668** | **1.735** | 9.704e6 | 1.269e12 | 2.239e11 | 1113 |
| QCD | $G_k$ | 47.48 | 3.860 | 1.160e6 | 1.392e12 | 2.931e10 | 1243 |
| QCD | $T_k$ | **5.666** | **1.735** | 3.696e5 | 5.286e10 | 9.329e9 | 1117 |

The tightest margin over the whole production range is **1.735 e-folds**, in the $T_k$ sector on
both models, against log 14's 1.74 at a single wavenumber. Nothing approaches one e-fold.
**Closed**, with the figure recorded so that a change to `RESIDUAL_WKB_REGION_MARGIN`, to the
production grid or to a cosmology can be checked against it.

### 3.9 Cost per object (prompt 13 §1 item 1)

Best of 5 for the cached figure; the "build" figure is the single call that builds the per-$k$
residual table, which in the $G_k$ sector is amortised over ~1,700 objects of the same $k$ and in
the $T_k$ sector is never amortised, there being one object per $k$
(`[07-tk-per-object-cost-is-all-setup]`). Integrand evaluations are the reproducible measure; the
seconds are one machine's (board §5 note 14).

| model | sector | $k$ [1/Mpc] | samples | build [s] | cached [s] | build evals | cached evals | before (review) |
|---|---|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 3e8 | 116 | 0.0405 | **0.0010** | 7392 | **468** | 63.7 s, 2.54e6 |
| LambdaCDM | $T_k$ | 3e8 | 1384 | 0.0511 | **0.0181** | 11376 | **5540** | 58 s, 1.93e6 |
| QCD | $G_k$ | 3e8 | 117 | 0.1791 | **0.0085** | 8380 | **472** | — |
| QCD | $T_k$ | 3e8 | 1401 | 0.6559 | **0.3257** | 12896 | **5608** | — |

The LambdaCDM figures reproduce logs 06, 07 and 14 exactly (0.0010 s / 468, 0.0181 s / 5540).
README §6's $G_k$ cost row ($\le0.05$ s per object) is met with a factor 50 to spare on LambdaCDM
and a factor 6 on QCD. The $T_k$ row is a *stage* claim, not a per-object bound (README §6's note):
50 objects per model at 0.018–0.33 s is **0.9 s (LambdaCDM) to 16 s (QCD) for the whole stage**
against the ODE's ~48 minutes.

---

## 4. Layer 2 — a scoped pipeline run per model

Live, through `main.py`'s own pipeline on a locally bootstrapped Ray cluster (8 CPUs of 10) and a
**fresh** datastore per model, both outside the repository. Neither datastore is committed; the
paths are in §7. The driver is `docs/gktk-remedial/scoped_pipeline_run.py` — see §7 and
`[13-scoped-run-driver-k-grid-literal]` for why it is not the `source-remediation` one.

### 4.1 The commands, verbatim

```bash
PYTHONPATH=. ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py \
    --k-min 1e5 --k-max 1e7 --k-count 5 --cpus 8 --models LambdaCDM \
    -- --database <scratch>/layer2/lambdacdm.sqlite --job-name gktk-verify-lambdacdm \
       --shards 4 --zend 0.1 --source-samples-log10z 100

PYTHONPATH=. ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py \
    --k-min 1e5 --k-max 1e7 --k-count 5 --cpus 8 --models QCD_Cosmology \
    -- --database <scratch>/layer2/qcd.sqlite --job-name gktk-verify-qcd \
       --shards 4 --zend 0.1 --source-samples-log10z 100 --no-quad-source-integral-queue
```

The wavenumber set is 5 modes log-spaced over $10^5$–$10^7$/Mpc; the redshift geometry is
production (`--zend 0.1`, `--source-samples-log10z 100`, the default response sparseness), which
gives a 1,584-node background grid from $z=6.8788\times10^{14}$ down to 0.1. Every tolerance, tag,
stage and work-queue parameter is `main.py`'s. `--no-quad-source-integral-queue` on the QCD run is
explained in §4.4.

### 4.2 LambdaCDM — every stage, and its wall time

| stage | items | wall time | note |
|---|---|---|---|
| horizon exit times (source / response) | 5 / 5 | 1.28 s / 0.016 s | |
| **`BackgroundModel`** | 1 | **1.14 s** | the three Gauss–Legendre tables, 1,584 nodes; 6,336 `BackgroundModelValue` rows |
| Bessel splines | 1 | 0.015 s | `transfer-remedial`'s, unchanged here |
| **`TkNumericIntegration`** | 5 | **1.15 s** | one object per $k$ |
| **`TkWKBIntegration`** | 5 | **0.669 s** | one object per $k$; the ODE it replaces was ~58 s *per object* |
| `QuadSource` | 15 | 0.714 s | |
| **`GkNumericIntegration`** | 2455 | **54.9 s** | |
| **`GkWKBIntegration`** | 7920 | **3 m 59.9 s** | 30 ms per object end to end, of which the phase itself is ~1 ms (§3.9); the rest is lookup, store and validation |
| `GkSource` | 660 | 1 m 18.2 s | |
| `GkSourcePolicyData` | 660 | 19.2 s | |
| `QuadSourceIntegral` | 3300 | stopped after ~3 h | §4.4 |

`GkSourcePolicyData` completed **660 of 660** with no `fail` and no `incomplete`: 195 `numeric`,
44 `mixed`, **421 `WKB`**, every one "complete". This is the stage that builds a `PrimitivePhase`
per Green's function, so it is the live exercise of prompt 09's consumer on real stored rows.

**`has_unresolved_osc` in production terms** (README §7 D2, prompt 16's per-$k$ summary):

```
-- UNRESOLVED-OSCILLATION SUMMARY | matter transfer functions, numerical part
|  no object reported unresolved oscillations (5 objects over 5 wavenumbers)

-- UNRESOLVED-OSCILLATION SUMMARY | tensor Green's functions, numerical part
|  k = 1e+05/Mpc: 591 of 591 objects flagged | e-folds inside horizon at first unresolved sample: 3.36 to 4.19
|  k = 3.1623e+05/Mpc: 541 of 541 objects flagged | ... 3.41 to 4.24
|  k = 1e+06/Mpc: 491 of 491 objects flagged | ... 3.46 to 4.01
|  k = 3.1623e+06/Mpc: 441 of 441 objects flagged | ... 3.5 to 4.05
|  k = 1e+07/Mpc: 391 of 391 objects flagged | ... 3.27 to 4.1
|  TOTAL: 2455 of 2455 objects flagged, over 5 of 5 wavenumbers
```

**Eight printed lines instead of 4,910.** Prompt 11 predicted 2,149 of 2,149 $G_k$-like objects
firing and 0 of 6 $T_k$-like; production gives **2,455 of 2,455 and 0 of 5**, and the depth at which
the response grid first fails to resolve the mode is 3.27–4.24 e-folds inside the horizon at every
wavenumber — the hand-over window, exactly as `[00-unresolved-osc-print-policy]`'s resolution
records. That semantic question is the hand-over campaign's; what Layer 2 confirms is that prompt
16's wiring survives a real `RayWorkPool` with `store_results=False`.

### 4.3 What the stored rows say — `docs/gktk-remedial/analyse_scoped_run.py`

```bash
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/analyse_scoped_run.py \
    --shards-glob '<scratch>/layer2/lambdacdm-shard*.sqlite' --model LambdaCDMModel --objects 12
```

**(1) The persisted limbs, against an independent reference on the run's own grid.** The scoped
run's grid is not prompt 01's — `z_init` follows the largest wavenumber of the run — so the JSON
checkpoints do not fall on its nodes and the reference is rebuilt here: mpmath at 40 digits for
LambdaCDM, converged adaptive Gauss–Legendre of the double integrand for QCD, both from
`docs/gktk-remedial/reference_lib.py`. Three nodes spread over the grid:

| $z$ | `tau_Mpc` (hi limb) | `tau_lo_Mpc` (lo limb) | $\tau-\tau(z_{\rm top})$ | reference | relative |
|---|---|---|---|---|---|
| 1.8062e13 | 2.5661227474240417e-08 | 8.0779356694631609e-25 | 2.4987432774324529e-08 | 2.4987432774324525e-08 | 1.324e-16 |
| 8.1989e06 | 0.056525073710321769 | −2.36332813153768e-18 | 0.056525073036527068 | 0.056525073036527054 | 2.455e-16 |
| 3.8085e00 | 6968.059186706053 | −2.6492109264279863e-13 | 6968.059186705379 | 6968.0591867053772 | 2.610e-16 |

Worst over the three: $\tau$ **2.611e-16**, $\tau_s$ **2.204e-16**, $F$ **2.440e-16** relative. The
low limb is exactly what README §7 D1 says it is — 1e-17 to 1e-13 of the high limb, i.e. the bits a
single double cannot hold — and it round-trips through `Float(64)` unchanged, `Mpc = 1.0` in
`Mpc_units`.

**(2) The stored phases are bit-identical to the offline producer.** For each sampled row the
object's own `z_init`, `G_init`/`T_init` and derivative are read out of the datastore, the three
background tables are **reconstructed from the persisted limbs** (zero quadrature, exactly as
`BackgroundModel._build_tau_primitive` does on load), `WKB_phase_function` is re-run offline, and
the production `store()` algebra and `apply_phase_offset` are applied:

```
Gk: 12 objects, 1080 samples; 0 not bit-identical to the offline producer
Tk:  5 objects, 5620 samples; 0 not bit-identical to the offline producer
```

Exact equality of both `theta_div_2pi` (an integer) and `theta_mod_2pi` (a double) at every one of
6,700 samples. This is what closes Layer 1 to Layer 2: the numbers in §3 are the numbers in the
datastore.

**(3) The consumers, and the chunk count.** 660 of 660 `GkSourcePolicyData` rows;
2,363 `QuadSourceIntegral` rows with `WKB_phase_spline_chunks` **1 on all 1,388 rows that populate
it** and NULL on the 975 whose Green's function has no WKB phase at that response redshift (the
column is `getattr(Gk_f.phase, "num_chunks", None)`). Never 2 or more: prompt 08's de-chunking and
prompt 09's `PrimitivePhase`, which reports `num_chunks == 1` by construction, are what production
now records.

**(4) Solver provenance, and the retired stage 2.**

```
BackgroundModel.solver_serial:              ['cumulative-GL']
GkWKBIntegration.solver_serial:             ['wkb-primitive']
TkWKBIntegration.phase_solver_serial:       ['wkb-primitive']
TkWKBIntegration.friction_solver_serial:    ['wkb-primitive']
GkWKBIntegration.stage_2 NULL:              7920 of 7920
TkWKBIntegration.stage_2 NULL:              5 of 5
```

Every WKB row points at the primitive's `IntegrationSolver`, not at an ODE; every stage-2 column is
NULL, which is the ODE's second stage having no successor rather than a failure to record one
(prompt 06 §4 item 6). The background model's solver is prompt 03's `cumulative-GL` at stepping 4.

### 4.4 The `QuadSourceIntegral` stage, and why it was stopped

The stage **was reached and did run** on LambdaCDM: all 3,300 work items were dispatched
(`0/5 work items remaining = 100.00% complete`), 2,363 were computed and stored, and the run was
stopped after **about three hours** with 937 compute tasks still in flight on eight cores. The
stored rows are the ones checked in §4.3(3), and they are spread over all five wavenumbers and all
fifteen $(q,r)$ pairs (463 / 450 / 450 / 450 / 450 rows by $k$), so the sample is representative
rather than a prefix.

Progress was 2,263 rows in the first ~2 minutes and 100 more in the following ~2.5 hours, so the
residue is a small number of pathologically expensive source integrals, not a uniform slowdown.
**That cost is not this campaign's**: `QuadSourceIntegral` is explicitly out of scope (README §1.1),
this campaign changed nothing in it beyond what the duck-typed phase protocol carries, and the
`source-remediation` board already owns the cost of its Levin/Clenshaw–Curtis fallback
(`[10-levin-wholesale-cc-fallback]`: "2.65–3.56× wall clock for 1.00–1.44× integrand evaluations").
Prompt 13 §2 asks for this stage "if the scoped run reaches them"; it did, and what it had to say
about this campaign — `WKB_phase_spline_chunks` — it said 1,388 times.

The QCD run was therefore started with `--no-quad-source-integral-queue`, so that every stage
prompt 13 §2 requires plus `GkSource` and `GkSourcePolicyData` complete on that model in bounded
time.

### 4.5 `QCD_Cosmology`

**The run did not fail.** Prompt 13 §2 anticipates that it might ("if the QCD run fails for a reason
unrelated to this campaign, record it as a §3 issue and run the LambdaCDM half"); it did not, and
every stage it was asked for completed, in **11 minutes** end to end.

| stage | items | wall time | LambdaCDM, for comparison |
|---|---|---|---|
| horizon exit times (source / response) | 5 / 5 | 1.7 s / 0.024 s | 1.28 s / 0.016 s |
| **`BackgroundModel`** | 1 | **1.14 s** | 1.14 s — 1,604 nodes here, 6,416 `BackgroundModelValue` rows |
| Bessel splines | 1 | 0.015 s | 0.015 s |
| **`TkNumericIntegration`** | 5 | **2.04 s** | 1.15 s — `BREAK_POINT_ALL`, prompt 19's +220 % on QCD |
| **`TkWKBIntegration`** | 5 | **1.97 s** | 0.669 s |
| `QuadSource` | 15 | 1.56 s | 0.714 s |
| **`GkNumericIntegration`** | 2538 | **1 m 50.6 s** | 54.9 s |
| **`GkWKBIntegration`** | 8020 | **7 m 5.2 s** | 3 m 59.9 s |
| `GkSource` | 670 | 1 m 15.4 s | 1 m 18.2 s |
| `GkSourcePolicyData` | 670 | 20.3 s | 19.2 s |
| `QuadSourceIntegral` | — | not run (§4.4) | 2,363 of 3,300 |

`GkSourcePolicyData`: **670 of 670**, no `fail` — 203 `numeric`, 45 `mixed`, 422 `WKB`. One of the
45 `mixed` rows is `minimal` rather than `complete`, which is the band `source-remediation`'s audit
§4.2 established is reachable (1 of 462 there, 1 of 670 here), not a defect of this campaign.

**`has_unresolved_osc`**: 0 of 5 for $T_k$; **2,538 of 2,538** for $G_k$, first firing at 3.27–4.22
e-folds inside the horizon at every wavenumber — the same picture as LambdaCDM, and the same
hand-over-window statement.

**Read-back** (`analyse_scoped_run.py --model QCDModel --objects 12`):

- **The persisted limbs agree with an independent break-aware reference to 4.6e-16 relative**
  ($\tau$; $\tau_s$ 4.0e-16, $F$ 2.4e-16), at nodes $z=2.73\times10^{13}$, $1.03\times10^7$ and
  3.99. *The reference had to be rebuilt to get that number, and the first attempt is worth
  recording*: a break-**unaware** composite Gauss–Legendre rule over the whole range, refined to
  4,096 panels of order 40, puts $\tau$ at **3.9e-9 relative** — stalled, not converged. Splitting
  at the cosmology's 375 declared break points inside the range (`integration_break_points`, prompt
  03) and at each decade of $1+z$ takes the same reference to 4.6e-16. That is prompt 02's
  `[01-qcd-eos-branch-boundaries]` finding reproduced from the other side: on `QCD_Cosmology` it is
  the *reference* that needs the break points, and the stored table — which has them — was right
  all along.
- **Bit-identical stored phases**: `Gk` 12 objects / 1,092 samples, `Tk` 5 objects / 5,638 samples,
  **0 not bit-identical** to the offline producer through the production `store()` algebra.
- **Solver provenance**: `cumulative-GL` for the background, `wkb-primitive` for all three WKB
  solver columns; `stage_2` NULL on **8,020 of 8,020** `GkWKBIntegration` and 5 of 5
  `TkWKBIntegration` rows.

### 4.6 What Layer 2 settles that Layer 1 could not

1. The schema change of prompts 03 and 04 **round-trips through SQLite**: the low limb is a
   `Float(64)` between 1e-24 and 1e-13 of the high limb and comes back bit-for-bit, so the
   double-double table survives persistence, which is the one thing README §2 (c) could not be
   checked for offline.
2. The producers **reproduce offline exactly**, on 13,430 stored samples across both models, which
   ties every Layer 1 figure to stored data.
3. The consumer stage runs on real assembled `GkSource` rows — including the numeric-initialised
   band that Layer 1's pure-WKB geometry does not cover — and completes for **1,330 of 1,330**
   Green's functions over the two models, with no `fail`.
4. Prompt 16's per-$k$ unresolved-oscillation summary works in a real `RayWorkPool`: **16 printed
   lines** across both models where the per-object form would have printed 9,986.
5. `WKB_phase_spline_chunks` is 1 wherever it is populated, over 1,388 stored `QuadSourceIntegral`
   rows — prompt 08's de-chunking as production records it.

---

## 5. The README §6 acceptance table, measured

Every row, with the value measured for this document where it could be, and the prompt that
measured it where it could not. "Now" is the review's figure, reproduced in `RECONCILIATION.md` §1.
**No target was loosened.**

| Quantity | Now | Target | Measured | Verdict |
|---|---|---|---|---|
| $\tau$ at production nodes, LambdaCDM, relative | 1.4e-9 | $\le2\times10^{-14}$ | **3.810e-16** (QCD 2.104e-14, at its reference's own 1.88e-14 floor) | met |
| $\Delta\tau$ between adjacent nodes, relative | — | $\le10^{-13}$ | **2.494e-16** (LambdaCDM), **9.354e-15** (QCD) | met |
| Interval-accessor cost per call, LambdaCDM / QCD | — | measured; stop if $>50\,\mu$s on QCD | **0.32 / 0.33 µs** on-grid, **3.54 / 33.3 µs** one endpoint off-grid, 6.56 / 64.7 µs both off-grid; in bulk through `raw_theta`, **6.86 / 29.3 µs** | met (production never evaluates both-off-grid; §3.8) |
| $\tau_s$, $F$ at nodes, relative | $F$: 2.3e-7–4.1e-7 (ODE) | $\le2\times10^{-14}$ / $\le10^{-13}$ | $\tau_s$ **2.496e-16 / 2.108e-14**; $F$ **3.314e-16 / 3.340e-16** | met |
| $\rho_G$, $\rho_T$ vs converged reference, absolute | — | $\le10^{-6}$ rad | **4.163e-17 rad** (LambdaCDM), **3.608e-16 rad** (QCD) | met |
| $\theta_G$ at $z=0.1$, LambdaCDM, $k=10^5$ | 13.9 rad | $\le10^{-5}$ rad | **0.0 rad**; max over the checkpoints 1.192e-7 rad at $z=10.012$ | met |
| $\theta_G$ at $z=0.1$, LambdaCDM, $k=3\times10^8$ | 7366 rad | $\le5\times10^{-3}$ rad | **9.766e-4 rad** = 1 ulp of 4.118e12 rad | met |
| $\theta_G$ exact-radiation control, span $10^7$ rad | 9.7e-3 rad | $\le10^{-8}$ rad | 3.7e-9 rad (prompt 06; pinned by `test_gk_wkb_phase.TestRadiationControl`) | met |
| Cost per `GkWKBIntegration` object, $k=3\times10^8$ | 63.7 s, 2.54e6 RHS | $\le0.05$ s | **0.0010 s / 468 evaluations** (LambdaCDM), **0.0085 s / 472** (QCD), best of 5 | met |
| Cross-object cycle consistency (t6-style sweep) | 60–165 of 330 rebased −1 | 0 rebase offsets | 0 of 990 objects rebased; 90 cycle steps at 90 stop-point transitions, all repaired by the rectifier (prompts 06, 09) | met |
| $\theta_T$ at $z=0.1$, LambdaCDM, $k=10^5$ / $3\times10^8$ | 2.0 / 5.1e3 rad | $\le10^{-4}$ / $\le5\times10^{-3}$ rad | **1.490e-8** / **9.155e-5 rad** | met |
| $T_{\rm WKB}$ radiation control from $x_i=24$ / $400$ | 3.8e-5 / 7.8e-9 | assert $\le5\times10^{-5}$ / $\le2\times10^{-8}$ | 3.8118e-05 / 7.7760e-09 (prompt 07; `test_tk_wkb_phase.TestRadiationValue`) | met (this **is** the floor) |
| Cost per `TkWKBIntegration` object, $k=3\times10^8$ | 58 s, 1.94e6 RHS | measured and recorded, per object and per stage | **0.0511 s / 11,376** cold, **0.0181 s / 5,540** with the table cached (LambdaCDM); **0.6559 / 12,896** and **0.3257 / 5,608** (QCD). Per stage, 50 objects per model: **0.9 s** (LambdaCDM) to **16 s** (QCD) against the ODE's ~48 min | recorded |
| Consumer phase interpolation error at production $x$ | 8.3e-3 rad ($h^4x/384$) | $\le10^{-6}$ rad | **2.384e-7 rad** ($G_k$) and **7.451e-9 rad** ($T_k$) on LambdaCDM at $k=10^5$, both 1.00 ulp of their span; 1.00 ulp in ten of the twelve (model, $k$, sector) cases | met on LambdaCDM; **the two QCD $k=10^5$ rows are 1.9e-6 and 3.2e-6 rad**, above the target, at the `T_LO` branch boundary (`[13-consumer-spline-crosses-eos-break-points]`), and the LambdaCDM $k=3\times10^8$ row carries §3.7's single whole-cycle sample |
| Chunk-switch discontinuity | 1.4e-4 rad | none (single spline) | `phase_spline` has one chunk by construction (prompt 08); in stored data, `WKB_phase_spline_chunks` is **1** on every row that populates it | met |
| Numeric RHS diagnostic overhead | 45 % of run | 0 %; the three fields still populated | the `*_omegaEff_sq` call is gone from both `RHS` functions; LambdaCDM $G_k$ object 0.1299 s → 0.0773 s, RHS evaluations bit-identical (prompt 11) | met |
| $T_k$ numeric $\delta T/{\rm env}$, production initial data | 1.1e-5 | $\le3\times10^{-6}$ | 2.534e-6 on review §12.5's geometry (prompt 12); **across the production $k$ grid** the median of the per-$k$ maximum is 3.8e-7 / 4.5e-7 / 1.0e-6 with 3 / 13 / 8 of 50 wavenumbers above 3e-6, worst 8.64e-4 (prompt 17) | met at the typical $k$, **missed at 3–13 of 50** — `[12-tk-numeric-atol-largest-k-excursion]`, assigned to `prompts/tolerance-convergence` |

**Two rows do not pass outright, and neither target was loosened.**

1. **The consumer row is missed on `QCD_Cosmology` at $k=10^5$** — 1.9e-6 rad ($G_k$) and 3.2e-6 rad
   ($T_k$) against $\le10^{-6}$ rad — and missed badly on LambdaCDM at $k=3\times10^8$ (6.17 rad,
   from one sample). Both are recorded, neither is interpolation of the growing phase, and both sit
   under the `QCD_Cosmology` Liouville–Green truncation floor of ~1e-3 rad in the first case and are
   a representation defect in the second:
   `[13-consumer-spline-crosses-eos-break-points]` and `[13-wkb-mod-2pi-cycle-count-inconsistent]`.
   The row's own stated geometry — $x=10^7$, 100/decade — is met: 2.4e-7 and 7.5e-9 rad.
2. **The $T_k$ numeric tolerance row is met at the typical wavenumber and not at all fifty.** The
   user settled the constant on 2026-09-12 and the remaining lever, `rtol`, belongs to
   `prompts/tolerance-convergence`: `[12-tk-numeric-atol-largest-k-excursion]`.

---

## 6. What remains open

Every entry of the board's §3 at close, with a one-line status. The board holds the measurements;
this is an index of where each one stands after the verification.

**Opened here.**

| Issue | Status |
|---|---|
| `[13-wkb-mod-2pi-cycle-count-inconsistent]` | New. The cycle count is a rounded division, the remainder an exact `fmod`; they disagree by a whole cycle on 1 of 77,975 production $G_k$ samples at $k=3\times10^8$ and cost the consumer 6.17 rad there. The stored `theta_mod_2pi`, and so every stored $G$ and $T$, is unaffected. Not fixed: production code. §3.7. |
| `[13-consumer-spline-crosses-eos-break-points]` | New. `PrimitivePhase`'s cubic spline of $\varphi$ interpolates straight across `QCD_EOS`'s declared break points, where $\varphi$ kinks: 1.9e-6 / 3.2e-6 rad at $z=4.24\times10^7$ and the QCD `theta_deriv` miss of §3.6. The remedy is the knot vector prompts 02/03/18/19 already built for the quadrature and the ODE. §3.5, §3.6. |
| `[13-scoped-run-driver-k-grid-literal]` | New. `docs/source-remediation-verification/scoped_pipeline_run.py` matches a `main.py` k-grid literal that `f17f2d4` renamed, so it now raises rather than running. Another campaign's file; this campaign copied it instead (§7). |

**Closed here.**

| Issue | Why |
|---|---|
| `[01-offgrid-accessor-cost-on-qcd]` | Its stated closing condition met: 29.3 µs per call in bulk on QCD at off-grid abscissae, against the 50 µs stop threshold. §3.8. |
| `[14-residual-range-top-margin]` | Its stated closing condition met: worst cut-to-anchor margin 1.735 e-folds over the whole production $k$ range, both models, both sectors. §3.8. |
| `[10-residual-spline-end-condition]` | Re-measured on the real background as the board assigned. On LambdaCDM the wider shipped bound is met (`[3:-3]` 2.9e-10 against `< 1e-9`) and the tighter one missed by 1.7× (`[5:-3]` 1.66e-11 against `< 1e-11`, met from the seventh sample in); the end effect decays ~3.8× per sample inwards, as the issue records. **The cubic stands and `spline_order=5` is not taken** — it needs six samples against `MIN_SPLINE_DATA_POINTS = 5`, it would diverge the two consumers, and it does not touch the error that dominates on the real background. That error is QCD's, it is uniform rather than at the ends, and it is carried forward as `[13-consumer-spline-crosses-eos-break-points]`. §3.6. |

**Narrowed, still open.**

| Issue | Where it stands |
|---|---|
| `[00-consumer-anchoring-floor]` | The floor is now the *whole* consumer error on the real background at production $x$: 1.00 ulp of the span in ten of twelve (model, $k$, sector) cases. Below the QCD Liouville–Green truncation and three to four orders below the numeric region's, so nothing downstream is limited by it. Next step unchanged. |
| `[07-tk-per-object-cost-is-all-setup]` | Both figures quoted, as the issue asks: 11,376 evaluations cold, 5,540 cached, of which the leading table's anchor panel is most. Next step unchanged. |
| `[14-rhs-evaluations-depend-on-build-order]` | The cost table gives cold and cached separately and says which is which. Next step unchanged. |
| `[10-wrap-theta-loop-at-large-phase]` | One clause of its text — "`WKB_mod_2pi` … is exact" — is now known to be true of the *remainder* only; both entries say so. |

**Unchanged by this prompt** (recorded, not re-measured): `[02-qcd-reference-floor]`,
`[02-qcd-T-z-spline-node-tolerance]`, `[01-lambdacdm-hubble-rounding-floor]`,
`[11-stop-point-root-tolerance]`, `[00-tk-lg-truncation-floor]`, `[00-tk-superhorizon-ic-series]`,
`[03-backgroundmodelvalue-build-path]`, `[03-qcd-short-baseline-reference-endpoint-rounding]`,
`[03-integrationsolver-stepping-minimum-lookup]`, `[04-background-rhs-evaluations-count]`,
`[06-metadata-column-headroom]`, `[08-docs-scripts-reference-removed-chunking]`,
`[06-docs-scripts-reference-removed-ode]`, `[10-transfer-remedial-tolerance-comments-stale]`,
`[12-tk-numeric-atol-largest-k-excursion]` (owned by `prompts/tolerance-convergence`),
`[19-cosmologymodels-docstrings-predate-per-sector-policy]`,
`[20-wkb-gauss-orders-not-in-lookup-key]`, `[20-wkb-rows-consume-numeric-initial-data]`.

**Out of scope throughout, and still out of scope** (README §0.3, §0.4): the numeric→WKB hand-over —
where the switch sits, the overlap width, the clamp error and the Liouville–Green derivative
truncation there — which `docs/OPEN_ISSUES.md` §1.1 owns; raising the LG order; per-region
anchoring; the Wronskian two-solution construction; the super-horizon initial-condition series.

---

## 7. Reproduction

Every command below is verbatim, run from the repository root on `gktk-remedial` at `ff9ee29`.

**Layer 1** (46 s, no Ray, no datastore):

```bash
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py
# or one section at a time:
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py --section consumers
```

**Layer 2** — the commands are in §4. The driver is
[`docs/gktk-remedial/scoped_pipeline_run.py`](gktk-remedial/scoped_pipeline_run.py), a copy of
`docs/source-remediation-verification/scoped_pipeline_run.py` differing only in the two
wavenumber-grid literals it substitutes in `main.py`: since `f17f2d4` those read
`np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)` and
`… NUMBER_RESPONSE_K_VALUES)`, so the original's exact-text match on `…, 50)` finds nothing and it
refuses to run. The original belongs to the `source-remediation` campaign and was not edited;
`[13-scoped-run-driver-k-grid-literal]`.

**The datastore paths**, verbatim, both outside the repository:

```
/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/14496f20-8692-4e03-957d-57834f5e600c/scratchpad/layer2/lambdacdm.sqlite
/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/14496f20-8692-4e03-957d-57834f5e600c/scratchpad/layer2/qcd.sqlite
```

with four shards each beside them (`…-shard0000.sqlite` … `…-shard0003.sqlite`). They are **not
committed** and are not expected to survive; rebuild them with the commands in §4. **A datastore
built before this campaign cannot be used**: `BackgroundModelValue` gained four columns in prompts
03 and 04, the `TkNumericIntegration` lookup key gained a tolerance in prompt 12 and a
`break_point_kind` column in prompt 20, and the QCD numeric values moved in prompts 18 and 19. Each
of those raises a `RuntimeError` naming the prompt rather than silently returning a stale row.

**The reference data** both layers score against is
`ComputeTargets/tests/wkb_reference_data.json`, regenerated by
`docs/gktk-remedial/generate_references.py` (prompt 01) and extended by
`docs/gktk-remedial/residual_convergence.py` (prompt 02). Neither was re-run here.

**The unit tree**, which pins most of Layer 1's figures as assertions rather than printed numbers.
Both were run on this tree after every edit and both pass; no test was added, removed or changed by
this prompt, which touches no production code and no test module.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
    Ran 339 tests in 148.977s      OK
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .
    Ran 141 tests in 1065.977s     OK
```

---

## 8. Post-close-out: what the `phase-representation` campaign moved (2026-09-13)

**Nothing at or above §7 is edited.** This section is appended by the orchestrator of
[`prompts/phase-representation`](../prompts/phase-representation/README.md) at its close, per
`CLAUDE.md` (verification documents are additive). §§1–7 remain the record of the tree at
`ff9ee29`/`9daa2cb` and were correct for it.

That campaign ran two prompts against the two defects §3.5 and §3.7 found and prompt 13 was
forbidden to fix. **It closed at 1 / 2.**

### 8.1 Prompt 01 — `WKB_mod_2pi`'s cycle count (`0c61799`)

`[13-wkb-mod-2pi-cycle-count-inconsistent]` is **resolved**. The cycle count now comes from the
exact `fmod` remainder, `int(round((fabs(theta) - fabs(theta_mod_2pi)) / TWO_PI))`, in both
`WKB_mod_2pi` and `simple_mod_2pi`, each keeping its own remainder convention. Measured with
`docs/gktk-remedial/verify_production_path.py`, unedited, before → after:

| measurement | §3 / §3.7 value | after prompt 01 |
|---|---|---|
| `div*2π + mod == θ`, production $G_k$, LambdaCDM $k=3\times10^8$ | 1 of 77,975 | **0 of 77,975** |
| the other eleven (model, sector, $k$) rows of §3.7 | 0 | 0 |
| uniform control at $\|\theta\|\sim4\times10^{12}$ | 25 of 400,000 | **0 of 400,000** |
| §3.5 consumer row, LambdaCDM $G_k$ $k=3\times10^8$ | 6.175 rad, 12,646 ulp, at $z=3.33\times10^4$ | **0.0000e+00 rad, 0.00 ulp** |
| §3.6 `theta_deriv`, same case | 5.6245e-08 | **1.8060e-13** |
| stored `theta_mod_2pi`, every model and $k$ | — | **bit-identical** |

Nothing else in that script's output moved but wall-clock timings; the §3.6 row above is the same
single sample seen through the spline's derivative, its $z=34{,}000$ maximum being the grid
neighbour of the $z=33{,}226$ outlier. Cost 0.2254 → 0.2731 µs per reduction, once per stored
sample. Suites at that commit: `ComputeTargets` 339 OK, `LiouvilleGreen` 141 → 148 OK.

**Datastore regeneration.** `theta_div_2pi` is a stored `nullable=False` column on `GkWKBValue` and
`TkWKBValue` and is in **no lookup key**, so unlike the columns §7 lists there is no schema change
to raise a `RuntimeError` on: **a datastore written before `0c61799` is served silently with the
old cycle count at the affected samples.** No migration was invented and none is recommended.

| quantity | affected by a pre-`0c61799` datastore? |
|---|---|
| `theta_div_2pi` on `GkWKBValue` / `TkWKBValue` | **yes** — one cycle too many at the half-ulp samples |
| the reconstructed unwrapped phase in `build_phi_samples` | **yes**, through `theta_div_2pi` |
| `PrimitivePhase.raw_theta` and its spline of $\varphi$ | **yes**, plus the ±15 grid intervals a cubic spreads one bad ordinate over |
| `QuadSourceIntegral`'s `_ClampedPhase` | **yes**, through `PrimitivePhase` |
| `theta_mod_2pi` | **no** — it is the `fmod`, bit-identical |
| every stored $G$ and $T$ (`G_WKB`, `T_WKB`) | **no** — built from the remainder |
| `friction_F`, $\rho$, $\tau$, $c_s\tau$, every `BackgroundModel` column | **no** |

### 8.2 Prompt 02 — `PrimitivePhase` break-point knots (`b3e3769`): stopped, no production change

`[13-consumer-spline-crosses-eos-break-points]` **remains open**. Prompt 02 stopped on its own §2
item 2 (README §7 D2: report rather than fall back silently) and **changed no production file** —
its commit is documentation only, so §3.5's three QCD rows and §3.6's QCD columns stand exactly as
printed above.

What it established, which supersedes that issue's recorded remedy:

- a multiplicity-3 knot vector at `BREAK_POINT_ALL` is **singular on all six production grids**
  (226–325 break points across 1,016–1,401 samples; 1–3 segments hold no sample, so
  Schoenberg–Whitney fails for any placement);
- at `BREAK_POINT_DISCONTINUITY` it constructs and is **2× worse** — $G_k$ 1.907e-06 → 3.815e-06 rad
  (8 → 16 ulp), $T_k$ 3.186e-06 → 6.790e-06 rad (428 → 911 ulp);
- the kink at the declared discontinuity is **1.60e-08 / 1.44e-07 rad**, i.e. **1 % and 4 %** of the
  1.907e-06 / 3.186e-06 rad §3.5 attributes to it;
- §3.6's two failing rows (QCD $G_k$ at $10^7$ and $3\times10^8$) are **neither** the knots **nor**
  `[02-qcd-T-z-spline-node-tolerance]`: the recovered $\varphi$ spans 6.0 and 2.0 ulp of the stored
  phase there, so including its spline derivative is 3× and 10× *worse* than omitting $\varphi$
  entirely. That is `[00-consumer-anchoring-floor]`, and it is opened as
  `[02-consumer-phi-below-the-storage-granularity]`.

### 8.3 A caveat that applies to §3.5 and §3.6 as printed

`docs/qcd-background-audit-2026-09.md` (2026-09-13), written after prompt 02 stopped, establishes
that the 407 `BREAK_POINT_ALL` points prompt 02 foundered on are **404 knots of the `T(z)` spline**
— a uniform lattice of an auxiliary 500-point interpolant inside
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, not a feature of the cosmology — and that the
QCD background's `T(z)` carries a relative error of **3.461e-08** in $\int\mathrm{d}z/H$.

**This matters for how the tables above are read.** §3.5 and §3.6 score a *consumer* against a
*producer*, and both are built from the same `BackgroundModel`, hence the same $H$ and the same
$\tau$. A background error is **common mode** and cancels exactly in that comparison. So those
tables legitimately read 1.00 ulp while the background underneath both sides carries, at
$k=3\times10^8$/Mpc, of order $10^5$ radians of systematic phase error. Nothing in §§1–7 is wrong;
what they measure is self-consistency, and self-consistency is blind to this class of defect. The
audit's §10 script is the first measurement in the tree that scores the background against an
independent background.

`[13-consumer-spline-crosses-eos-break-points]` therefore now belongs with that audit's work, not
with a knot vector.

---

## 9. Post-close-out: what the `qcd-background-audit` campaign moved (2026-09-14)

**Nothing at or above §8 is edited.** This section is appended by the close-out of
[`prompts/qcd-background-audit`](../prompts/qcd-background-audit/README.md), per `CLAUDE.md`
(verification documents are additive). §§1–7 remain the record of the tree at `ff9ee29`/`9daa2cb`
and §8 the record of `0c61799`/`b3e3769`; all were correct for the trees they were taken on.

**The full record is [`qcd-background-verification.md`](qcd-background-verification.md).** What
follows is only what moved *here*, re-measured with `docs/gktk-remedial/verify_production_path.py`
run **unedited** at the campaign base `e8f746d` and at its close. (The base run was taken at
`2a5e0fa`, the campaign's planning commit, which differs from `e8f746d` in 21 files all of them
under `docs/` or `prompts/` — no Python file, no fixture, and not that script — so the two trees are
numerically identical.)

### 9.1 What §8.3 warned about has happened

§8.3 records that §3.5 and §3.6 score a consumer against a producer built from the same
`BackgroundModel`, so a background error is common mode and cancels in them — and that the QCD
background's `T(z)` carried a relative **3.461e-08** in $\int\mathrm{d}z/H$, worth $10^5$ radians at
$k=3\times10^8$/Mpc, which those tables could not see. **That error is now zero**: the shipped
background's $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$ is bit-identical to the same integral
over an independently root-solved exact background, `1.3320002507788795e+03` at all 17 digits, and
the equivalent phase is 0.000e+00 rad at all three production wavenumbers.

**So the figures in §§1–7 were correct for the tree they were taken on, and the background
underneath them has since changed.** That is precisely the fact §6 could not have seen, because no
measurement in this document scores the background against an independent background. The 407
`BREAK_POINT_ALL` points §8.2 and §8.3 discuss are now **3**, none of them a knot of any
interpolant.

### 9.2 §3.5 — six rows bit-identical, four unchanged in value, **two worse**

| model | $k$ | sector | §3.5 / base [rad] | now [rad] | |
|---|---|---|---|---|---|
| LambdaCDM | all three | $G_k$, $T_k$ | — | **bit-identical**, all six rows | |
| QCD | 1e5 | $G_k$ | 1.907e-6 (8.00 ulp) | **8.1062e-6 (34.00 ulp)** | **4.25× worse** |
| QCD | 1e5 | $T_k$ | 3.186e-6 (427.60 ulp) | **1.3982e-5 (1876.61 ulp)** | **4.39× worse** |
| QCD | 1e7 / 3e8 | $G_k$, $T_k$ | 1.526e-5 / 4.883e-4 / 9.537e-7 / 3.052e-5 | unchanged, 1.00 ulp | |

Both risen rows have their maximum at $z=4.24\times10^7$ — the same `T_LO` branch boundary §3.5
already attributes them to — and the cause is measured in
[`qcd-background-verification.md`](qcd-background-verification.md) §3.3: **the corrected background
presents a sharper feature there.** The old 500-node `T(z)` spline had a knot spacing 4.04× the
production grid, so it smeared the equation of state's genuine step over about four grid intervals
and presented only **25.55 %** of the total deviation in $\mathrm{d}\ln H/\mathrm{d}u$ inside
any one of them; the corrected background puts **99.84 % of a
step 2.78× taller inside the single interval** containing the crossing. §3.5's attribution was
right and its magnitude was an under-reading: it is `[13-consumer-spline-crosses-eos-break-points]`,
seen without the background's smoothing on top.

### 9.3 §3.6 — `theta_deriv`

All six LambdaCDM rows bit-identical, decay ladders included. On QCD:

| $k$ | sector | §3.6 / base (max / `[3:-3]` / deep) | now |
|---|---|---|---|
| 1e5 | $G_k$ | 5.128e-7 / 2.312e-7 / 2.312e-7 | **1.0232e-6 / 1.0232e-6 / 1.0232e-6** (worse, at the `T_LO` node) |
| 1e5 | $T_k$ | 1.100e-6 / 1.100e-6 / 1.100e-6 | **3.0630e-6** throughout (worse, same node) |
| 1e7 | $G_k$ | 1.098e-5 / 7.043e-6 / 6.084e-6 | 1.0969e-5 / 6.7435e-6 / 6.0229e-6 |
| **1e7** | **$T_k$** | 4.890e-7 / 4.890e-7 / 4.329e-7 | **1.8853e-8 / 3.4112e-10 / 3.0630e-10** — **1,413× better** |
| 3e8 | $G_k$ | 3.309e-4 / 3.309e-4 / 1.748e-4 | 2.5418e-4 throughout |
| 3e8 | $T_k$ | 8.419e-6 throughout | 7.3236e-6 throughout |

§3.6 concludes that on `QCD_Cosmology` the identity "is missed by orders, and not at the ends …
driven by the same `T(z)` spline knots and equation-of-state branch boundaries as §3.5". **The
$T_k$ row at $k=10^7$ separates those two causes and settles them**: with the `T(z)` representation
corrected, its last five samples at the hand-over become a clean geometric ladder rising ~3.8× per
sample inwards — `8.63e-11 3.41e-10 1.29e-09 4.94e-09 1.89e-08`, which is `[10-residual-spline-end-condition]`'s
signature and matches LambdaCDM at the same $k$ to within 20 % — where before they were flat and the
size of the interior. What §3.6 measured on that row was the representation; what is left is the end
condition, and §3.6's closure of `[10-residual-spline-end-condition]` at the cubic stands.

The two QCD $G_k$ rows at $k\ge10^7$ barely moved, which is `prompts/phase-representation` prompt
02's finding confirmed: they are `[02-consumer-phi-below-the-storage-granularity]` and not the
knots.

### 9.4 Elsewhere in this document

- **§3.1** — QCD `tau` 2.104e-14 → **2.254e-15** and `cs_tau` 2.108e-14 → **2.212e-15**, from above
  their references' own 1.88e-14 floor to an order below it; `friction_F` and $\Delta\tau$
  unchanged; `rho` worst 3.608e-16 → 2.248e-15 rad against a 1e-6 rad target, both at the round-off
  floor of the quantity, its reference having been regenerated.
- **§3.2, §3.3** — all six QCD $\theta_G$ and $\theta_T$ rows improved; all six were above the
  script's printed `eps*|theta|_max` floor at the base (1.07× to 5.94×) and three are now below it
  with the other three within 11 % of it, i.e. 1.5–2 ulp of the accumulated phase. LambdaCDM
  unchanged.
- **§3.7** — unchanged: 0 inconsistent samples in all twelve rows and 0 of 400,000 in each of the
  three uniform controls. §8.1's repair holds.
- **§3.9 / §5** — QCD per-object build costs fall: $G_k$ 8,380 → **6,892** integrand evaluations,
  $T_k$ 12,896 → **11,532**, with both LambdaCDM rows and all four cached-evaluation counts exactly
  unchanged; the QCD off-grid `raw_theta` accessor needs **4.00** integrand evaluations per call
  where it needed 4.27. The QCD `BackgroundModel` cumulative tables fall 16,580 → **6,936**
  evaluations each, 0.17 % above LambdaCDM's break-free 6,924.
- **§4** — the residual-table cut-to-anchor margin moves on QCD, 47.48 → 14.88 (3.860 → 2.700
  e-folds) for $G_k$ and 5.666 → 4.677 for $T_k$, at different wavenumbers in each case: with the
  corrected background the top of the residual region is pinned at production grid node 439,
  $z=8.384\times10^{11}$, 1.33 grid intervals below the `T_120_MEV` crossing, rather than wandering
  with the old representation's scatter. Both remain far above the ~1 e-fold that would make
  `RESIDUAL_WKB_REGION_MARGIN` worth revisiting; the LambdaCDM rows are unchanged.
- **§5's acceptance table** — the consumer row's two QCD $k=10^5$ entries become 8.1e-6 and 1.4e-5
  rad against its $\le10^{-6}$ rad target; the row's own stated geometry ($x=10^7$, 100/decade) is
  still met at 2.4e-7 and 7.5e-9 rad on LambdaCDM. No other verdict in that table changes.

### 9.5 The standing caveat of §8.3 is discharged; one is added

§8.3 said the audit's script was "the first measurement in the tree that scores the background
against an independent background". It is now a **test** in the tree:
`CosmologyModels/tests/test_T_z_representation.py`, scored against
`CosmologyModels/tests/T_z_reference.py`, with `CosmologyModels` rising 11 → 30 across the campaign
and `ComputeTargets` 339 → 359. Nothing in this document's numbers can drift out from under a
corrected background again without a test failing.

**Added:** `QCD_EOS`'s branch joins are still unrepaired — `[00-eos-branch-joins-do-not-match]` —
and the $10^{-5}$ GeV join is the origin both of §3.5's worst QCD errors and of the 4.4e-4 jump in
$H(z)$ at $z=4.24\times10^7$ that `GkTk-remedial` log 02 measured without attribution. This campaign
deliberately did not repair it: it is an upstream data fixture, and a segmented representation
reproduces a discontinuous fixture exactly. All four joins are now pinned in a test so that a later
correction announces itself.
