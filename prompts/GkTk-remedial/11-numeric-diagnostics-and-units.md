# Prompt 11 — The numeric region: diagnostic off the RHS, grid units, stop-point repairs

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §10.1 (the numeric region is sound), §10.2 (the six observations), §12.5
(the same items apply to $T_k$), **§13.1 (the flag is a live warning; the unit slip is a
precondition; the sampling change must be stated)**
**Design facts:** README §2 (h); decision §7 D2.
**Depends on:** nothing (independent of A–D).
**Recommended model:** Opus
**Files you may touch:** `Quadrature/integrators/numeric_with_phase_cut.py`,
`Quadrature/supervisors/numeric.py`, `LiouvilleGreen/integration_tools.py`,
`ComputeTargets/GkNumericIntegration.py`, `ComputeTargets/TkNumericIntegration.py`, `main.py`
**comment-only hunks at `:595-598` and `:1160-1165`** (the `0.85` truncation comments), new
`ComputeTargets/tests/test_numeric_phase_cut.py`, plus the log and the status board.
**Do not touch:** the stop-search window attributes (`z_exit_subh_e3`/`_e6`) or the
$\sqrt{z_{e3}z_{e4}}$ limit (README §0.3); `main.py`'s `delta_logz=` arguments (they stay as they
are — see §2.1); the Datastore factories.

Read first: README §2 (h), §7 D2, §0.3; `RECONCILIATION.md` §1 items 8, 9 and §2 items 8, 9;
review §10, §12.5, §13.1 in full.

---

## 1. Character of this commit

Cleanup of a region the review found **sound** ($2\times10^{-7}$ of the envelope at 0.1 s per
object, §10.1), so every change must leave the returned $G$, $G'$, $T$, $T'$ samples and the stop
point **bit-identical** for the same inputs, except where a repair changes them by design and the
test says so. One item is a semantic decision the user has to see (D2): this prompt implements the
faithful version, measures its consequence, and the orchestrator stops.

## 2. What to build

### 2.1 The oscillation-resolution test, off the RHS, with the right grid

Review §13.1: the flag tells the caller "the sample grid *you supplied* is too coarse to resolve
the oscillations the integrator stepped through". Implement exactly that, on the sample grid,
after the solve:

- `numeric_with_phase_cut` gains a parameter `omega_sq` (the sector's `*_omegaEff_sq`); both
  integrators pass theirs. Remove the `Gk_omegaEff_sq`/`Tk_omegaEff_sq` call and the
  `report_wavelength` call from `RHS` (`GkNumericIntegration.py:68-76`,
  `TkNumericIntegration.py:82-91`).
- After `solve_ivp`, for each pair of consecutive **returned** samples $(z_i, z_{i+1})$, compute the
  local wavelength $\lambda=2\pi/\sqrt{\omega^2(z_i)}$ (skip where $\omega^2\le0$) and flag the first
  $i$ with $\lambda<|z_i-z_{i+1}|$. Set `has_unresolved_osc`, `unresolved_z`, `unresolved_efolds_subh`
  from it, and print the same single warning line the supervisor prints today. Keep
  `NumericIntegrationSupervisor.report_wavelength` for compatibility but make the supervisor's
  `delta_logz`-based path **document** that it is superseded (or delete it — state which); keep the
  `delta_logz` constructor argument accepted so `main.py` need not change.
- Because the test now uses the actual sample spacing, the $\ln10$ unit slip is moot for the
  spacing; but **also** fix `report_wavelength` if you keep it: `grid_spacing = (1+z) * delta_logz * ln(10)`
  with the parameter documented as $\Delta\log_{10}(1+z)$ (the value `main.py` passes).
- **State the sampling change** in the code comment and the log, as review §13.1 requires: on the
  RHS the test saw every internal step; on the sample grid it sees only output points, so a
  wavelength minimum between two samples is no longer caught — and the subject of the test is the
  output grid, so this is the more faithful form.

### 2.2 Measure the consequence (D2)

On `RadiationModel` and `LambdaCDMModel` stand-ins (prompt 01 — if Workstream A has not run,
build the two stand-ins locally in the test module and say so), with the **production grids**
(source: 100/decade; response: its 12-fold winnow, truncated to $0.85\,z_{e6}$ as `main.py` does),
for $k\in\{10^5,10^7,3\times10^8\}$ and the production source-redshift bands: what fraction of
`GkNumericIntegration`-like runs and of `TkNumericIntegration`-like runs would now set the flag,
and at what $x$ it fires. README §7 D2 predicts "essentially every $G_k$ object above
$x\approx22$" and "none for $T_k$ below $x\approx273$… i.e. none, since the run stops at
$x\approx403$" — hmm: check the $T_k$ case explicitly (source grid, stop at $z_{e6}$). **Put the
measured fractions in the first paragraph of the log's Result section.** Do not change the print
policy yourself; that is the user's decision.

### 2.3 The other repairs (review §10.2)

1. `mode.lower()` before the `None` check (`numeric_with_phase_cut.py:50`): guard it. The
   `mode != "stop"` branch stays (`RECONCILIATION.md` §3).
2. `find_phase_minimum` (`integration_tools.py`): (a) it finds a **maximum** of $G$ ($G'$ changing
   from negative to positive as $z$ decreases; $G_{\rm stop}/{\rm env}=+1.000000$ in every review
   run) — fix the docstring and the comments in both integrators (`GkNumericIntegration.py:312-314`,
   the `TkNumericIntegration` twin) and the "minimum … to avoid jitter" motivation, which is
   obsolete because `store()` rotates arbitrary $(G,G')$ into a pure sine; keep the function's
   name or rename it to `find_phase_extremum` with a compatibility alias — state which; (b) step
   in **phase**, not in relative $z$: with `omega_sq` now available, choose the step so the phase
   advances by $\approx2\pi/16$ per step ($\Delta z=2\pi/(16\,\omega)$), falling back to the current
   $10^{-3}z$ where $\omega^2\le0$. Review §10.2: the current step gives 15 samples per cycle at
   $x=403$ and fewer than one at $x>6283$, so it is safe only inside the window; the window is
   **not** widened here.
3. The `0.85·z_e6` truncation comment in `main.py` (`:595-598`, `:1160-1165`): the trailing
   samples are requested but never produced in stop mode (the ODE terminates at $z_{e6}$ and the
   `expected_values` check is skipped). Rewrite the two comments to say so. **Comment-only**; the
   constant stays (changing it is a hand-over decision).
4. Stale comments about the stop being a minimum, wherever they occur in the touched files.

## 3. Tests (`test_numeric_phase_cut.py`)

Undecorated `numeric_with_phase_cut` with the two RHS functions and stand-ins:

1. **Bit-identity.** For `RadiationModel`, $k=10^7$, production stop window, `mode="stop"`: the
   returned `value_sample`, `deriv_sample` and the stop point agree **exactly** with a run of the
   pre-change code (capture the pre-change outputs in the test file as constants from a run at
   `HEAD~1`, with the commit SHA in a comment) — except the stop point, which item 2.3(2b) may move
   by one root-bracketing step; assert the stop point is an extremum of the same sign with
   $|G'|<10^{-12}\,|G|\omega$ and that $G_{\rm stop}/{\rm env}=+1$ to $10^{-6}$.
2. **The flag semantics.** A run whose sample grid is deliberately coarse (one sample per decade)
   sets `has_unresolved_osc=True` with `unresolved_z` at the first pair violating the test; a run
   on a grid finer than the wavelength everywhere sets `False`; the warning line is printed once
   (capture stdout).
3. **Cost.** RHS evaluations unchanged; wall time per run on `LambdaCDMModel` falls by $\ge30\,\%$
   (review §10.2: 0.13 s → 0.09 s, 45 %). Assert the RHS count; report the time.
4. `mode=None` runs to the end of the grid without raising; `mode="STOP"` is accepted; `mode="x"`
   raises `ValueError`.
5. `find_phase_minimum` finds the same extremum from the same start when stepping in phase, on a
   dense-output solution of the exact radiation ODE, and finds it at $x=6\times10^3$ where the old
   step would have skipped cycles (construct the case; this is why 2.3(2b) exists).
6. Both sectors: the `TkNumericIntegration` RHS path produces identical $T$, $T'$ samples before
   and after.

## 4. Verification and acceptance

- New tests pass; `discover -s ComputeTargets/tests -t .` passes.
- `grep -n "omegaEff_sq\|report_wavelength" ComputeTargets/GkNumericIntegration.py ComputeTargets/TkNumericIntegration.py`
  shows no call inside `RHS`.
- README §6 row "Numeric RHS diagnostic overhead" met; flag fields still populated and persisted
  (the factories are untouched; the payload keys are unchanged — assert).
- `black --check` clean.
- **The D2 measurement is in the log's first paragraph.** The orchestrator stops after this prompt
  regardless, to put it to the user.

## 5. Log and commit

"State handed to the next prompt": the new `numeric_with_phase_cut` signature; the D2 fire
rates; the chosen name for the extremum finder; the measured speed-up.

Commit subject, or something equally specific: `Test oscillation resolution on the sample grid, off the RHS`.
