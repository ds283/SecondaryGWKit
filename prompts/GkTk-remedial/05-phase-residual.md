# Prompt 05 — The phase residual $\rho$ and the leading/correction split of $\omega^2$

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §6 (the rationalised residual, its size, the LG truncation floor), §12.2
($\rho_T$, not negligible), §12.4 (radiation control $\rho_T=1/x_i-1/x$), §11 (fallback is
adaptive for $\rho$ alone)
**Design facts:** README §2 (a), (d), (f).
**Depends on:** 04 (the `cs_tau` accessor for the $\rho_T$ control), 02 ($N_\rho$, fallback flag,
JSON convergence tables).
**Recommended model:** Opus
**Files you may touch:** new `ComputeTargets/phase_residual.py`, `ComputeTargets/WKB_Gk.py`,
`ComputeTargets/WKB_Tk.py`, new `ComputeTargets/tests/test_phase_residual.py`, new
`ComputeTargets/tests/test_omega_eff_split.py`, plus the log and the status board.
**Do not touch:** `WKB_phase_function.py`, the integration objects.

Read first: README §2 (a), (d), (f) — **`*_omegaEff_sq` return values must not change**;
`RECONCILIATION.md` §2 item 3; review §6 (all), §12.2, §12.4; `logs/02-…` and `logs/04-…` "State
handed to the next prompt".

---

## 1. Character of this commit

Two small things that must be exactly right:

1. **Expose the non-leading part of $\omega^2$ without subtraction.** `Gk_omegaEff_sq` returns
   `A + B + C` with `A = (k/H)^2` (`WKB_Gk.py:15-19`); `Tk_omegaEff_sq` returns `A + B + C` with
   `A = w (k/H)^2` (`WKB_Tk.py:16-24`). Refactor each into a leading function and a correction
   function, and make the existing function return their sum **bit-identically** to today:

   ```python
   def Gk_omegaEff_sq_leading(model, k, z) -> float      # (k/H)^2
   def Gk_omegaEff_sq_correction(model, k, z) -> float   # B + C  (k-independent for Gk)
   def Gk_omegaEff_sq(model, k, z) -> float              # leading + correction, as today
   ```
   and likewise `Tk_*` (the Tk correction depends on $k$ only through nothing — check: `B`, `C`
   in `Tk_omegaEff_sq` are $k$-free; the leading term carries $w$). The `*_d_ln_omegaEff_dz`
   functions are untouched. Note the `k`-independence in the docstrings: it is why one residual
   table per $k$ is cheap and why the leading term is a per-model table.
2. **The residual as a `CumulativeTable`.** With $\omega_0=\sqrt{\rm leading}$ and
   $\omega=\sqrt{\rm leading+correction}$,
   $$\rho(z;z_i)=\int_{z_i}^{z}\frac{\rm correction}{\omega+\omega_0}\,dz'$$
   (review §6; the rationalised form avoids $\omega-\omega_0$). Provide

   ```python
   def build_phase_residual(model, k: float, z_nodes, sector: str, order: int = N_rho) -> CumulativeTable
   ```
   with `sector in ("Gk", "Tk")` selecting the pair of functions. Single-limb (`lo` zero) — $\rho$
   is $\le0.1$ rad. The consumer forms $\Delta\rho=$ `table.delta(z_i, z)` with prompt 03's sign
   convention, so that $\theta(z;z_i)=-[k\,$`tau.delta(z_i, z)`$\,+\,$`rho.delta(z_i, z)`$]$ — write
   this identity in the module docstring with the exact-radiation check.
   If prompt 02 set the fallback flag, implement the adaptive rule for the identified interval
   class exactly as its log specifies, and nothing more.

The integrand needs `omega + omega_0 > 0`; raise with the model, $k$, $z$ if `omega_sq <= 0`
(the WKB region has $\omega^2>0$ by construction; `WKB_phase_function` already raises on it).

## 2. Tests

### `test_omega_eff_split.py`

For all three stand-ins, $k\in\{10^5,10^7,3\times10^8\}$, 200 log-spaced $z$ across the grid:
`Gk_omegaEff_sq(model,k,z) == Gk_omegaEff_sq_leading(...) + Gk_omegaEff_sq_correction(...)`
**exactly** (`==`, not `assertAlmostEqual`) — the refactor must not move a bit, because
`omega_WKB_sq` is a stored column. Same for `Tk`. Also: `Gk_omegaEff_sq_correction` is
independent of `k` (evaluate at two $k$, assert equal); on `RadiationModel` it is exactly `0.0`
(with $\epsilon=2$, $\epsilon'=0$: $B=0$, $C=(3-1-2)/s^2=0$ — assert `== 0.0`).

### `test_phase_residual.py`

1. **Radiation controls.** `Gk`: the table is identically zero (every `hi` exactly `0.0`). `Tk`:
   $\rho_T(z;z_i)=1/x_i-1/x$ with $x=kc_s\tau$, $c_s=1/\sqrt3$, using the stand-in's closed-form
   $\tau$, to $\le10^{-13}$ absolute over $x\in[24, 10^4]$ (review §12.4 table's last column).
2. **Against the references.** `LambdaCDMModel` and `QCDModel`, both sectors, three $k$:
   `rho.delta(z_top, z_j)` at the JSON checkpoints within $10^{-7}$ rad of prompt 01's references
   (README §6 target $10^{-6}$ with a decade of margin, as prompt 02 chose $N_\rho$). Quote the
   measured maximum and its $(k, z)$.
3. **Size sanity** (documents the review's §6 and §12.2 tables): on LambdaCDM,
   $|\rho_G|$ over the whole range $\le3\times10^{-7}$ rad for $k=10^5$; on QCD, $|\rho_G|$ for
   $k=3\times10^8$ is $O(10^{-3})$ rad (assert between $5\times10^{-4}$ and $3\times10^{-3}$);
   $\rho_T\approx-0.09$ rad on both models (assert between $-0.12$ and $-0.06$). These bounds
   protect against a sign or factor slip in the integrand.
4. **Cost.** Integrand evaluations per table $\le N_\rho\times$(intervals); wall time recorded per
   model.

## 3. Verification and acceptance

- Both new modules pass; `discover -s ComputeTargets/tests -t .` passes (nothing else changed).
- README §6 row for $\rho$ met.
- `black --check` clean.

## 4. Log and commit

"State handed to the next prompt", verbatim: the six new function names and signatures;
`build_phase_residual`'s signature and the sign identity; measured $\rho$ accuracy and cost per
model and $k$; whether the fallback was implemented.

Commit subject, or something equally specific: `Add the WKB phase residual as a per-k table`.
