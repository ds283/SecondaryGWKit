# Log 11 — The numeric region: diagnostic off the RHS, grid units, stop-point repairs

**Prompt:** prompts/GkTk-remedial/11-numeric-diagnostics-and-units.md
**Commit:** *(this commit)* — Test oscillation resolution on the sample grid, off the RHS
**Model:** Claude Opus 5
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

**The D2 measurement (README §7 D2, `[00-unresolved-osc-print-policy]`) — the decision the user
must take.** With the corrected test — evaluated against the actual spacing of the grid the caller
supplied, which removes both the $\ln10$ slip and the source-grid/response-grid mismatch —
**every single `GkNumericIntegration`-like object flags: 2,149 of 2,149** over the full production
source-redshift band ($z_{\rm exit,subh}^{e4} \le z_s \le z_{\rm exit,suph}^{e5}$, every point of the
universal 100-per-decade source grid), on both `RadiationModel` (1,075 objects) and
`LambdaCDMModel` (1,074), at $k = 10^5$, $10^7$ and $3\times10^8$/Mpc. The flag first trips at
$x = 26.5$–$66.6$ (min 26.47 at LambdaCDM $k=10^7$, max 66.59 at LambdaCDM $k=10^5$), i.e. at the
first response-grid pair past $x \approx 2\pi/(10^{12/100}-1) = 19.7$, matching README §7 D2's
predicted $x\approx22$. Today the flag fires on **none** of them, so keeping the per-object printed
warning would add **two printed lines per `GkNumericIntegration` object** — of order $6.5\times10^4$
objects per model, i.e. ~130,000 lines per model where production prints none. That is a factor far
beyond the ten the campaign README §4.3 names as the stop threshold, so **the orchestrator stops
here and the print policy is the user's to choose**; this prompt implements the faithful test and
keeps the per-object line exactly as it is today (option (i)), and changes no policy.
**`TkNumericIntegration`-like runs flag on none: 0 of 6.** README §7 D2 was unsure of this case and
the prompt asked for it explicitly. The transfer function is sampled on the *source* grid itself
(not its 12-fold winnow) and its effective frequency carries $c_s = 1/\sqrt3$, so the trip point is
at $x_T \approx 2\pi/(10^{1/100}-1) = 270$, i.e. $x \approx 467$; the run terminates at $z_{e6}$,
$x = 403$, and its deepest *returned* sample sits at $x = 385$–$403$, $x_T = 223$–$228$. The worst
spacing/wavelength ratio reached over all six runs is **0.807–0.822** — a 22 % margin below firing,
comfortable but not large.

## What shipped

### `Quadrature/integrators/numeric_with_phase_cut.py`

- **New module-level function** `scan_sample_grid_for_unresolved_osc(model, k, k_float, omega_sq,
  sampled_z, object_label) -> dict` (keys `has_unresolved_osc`, `unresolved_z`,
  `unresolved_efolds_subh`). Walks consecutive **returned** samples $(z_i, z_{i+1})$, skips any
  where $\omega^2(z_i)\le0$, and flags the first $i$ with $2\pi/\omega(z_i) < |z_i - z_{i+1}|$,
  printing the same two-line warning the supervisor printed, verbatim. Its docstring states the
  sampling change review §13.1 requires: on the RHS the test saw every internal step; on the
  sample grid it sees only output points, so a wavelength minimum strictly between two samples is
  no longer caught — and the subject of the test *is* the output grid, so this is the more faithful
  form.
- **New parameter** `omega_sq: Optional[Callable[[BackgroundModel, float, float], float]] = None`,
  placed after `RHS`. Both integrators pass theirs by keyword. It serves two purposes: the scan
  above, and the phase-stepped extremum search below.
- `:50` **`mode.lower()` guard fixed**: the `None` test now comes first
  (`if mode is not None: mode = mode.lower(); if mode not in ["stop"]: raise`). The
  `mode != "stop"` branch stays (`RECONCILIATION.md` §3).
- `find_phase_minimum` → `find_phase_extremum`, called with
  `omega_sq=lambda z: omega_sq(model, k_float, z)` when a frequency was supplied.
- The payload's three flag fields now come from the scan instead of the supervisor. `delta_logz`
  is still accepted and still handed to the supervisor, with a comment saying the diagnostic no
  longer uses it, so `main.py` needs no change.
- Comment at the stop-point search rewritten: the extremum is a **maximum**, and the "fixed phase
  to avoid jitter" motivation is obsolete because `store()` rotates arbitrary $(G,G')$ into a pure
  sine.

### `Quadrature/supervisors/numeric.py`

- `report_wavelength` **kept** and documented as superseded, with the caller and the replacement
  named. Its unit slip is fixed in place: `grid_spacing = (1.0 + z) * self._delta_logz * LN_10`
  (new module constant `LN_10 = log(10.0)`), with the parameter documented as
  $\Delta\log_{10}(1+z)$ — the value `main.py:630, :1199` passes. No production caller remains.

### `LiouvilleGreen/integration_tools.py`

- `find_phase_minimum` → **`find_phase_extremum`**, with `find_phase_minimum = find_phase_extremum`
  retained at module scope as a compatibility alias (the choice the prompt asks to be stated).
  New optional parameter `omega_sq: Optional[Callable[[float], float]] = None`.
- **Steps in phase** when a positive $\omega^2$ is available: `step_from(z)` returns
  $-2\pi/(16\,\omega(z))$, i.e. `PHASE_STEPS_PER_CYCLE = 16` samples per cycle, falling back to
  `-DEFAULT_RELATIVE_STEP * z` ($10^{-3}z$, the old step) where `omega_sq` is `None` or
  $\omega^2\le0$. Both constants are module-level and named.
- Docstring rewritten: the extremum is a maximum; nothing depends on which extremum; the window is
  not widened.

### `ComputeTargets/GkNumericIntegration.py`, `ComputeTargets/TkNumericIntegration.py`

- `Gk_omegaEff_sq` / `Tk_omegaEff_sq` and `supervisor.report_wavelength` **removed from `RHS`**
  (`GkNumericIntegration.py:68-76`, `TkNumericIntegration.py:82-91`), replaced by a comment saying
  where the test went and why. The unused `pi` import is dropped from `GkNumericIntegration`
  (`TkNumericIntegration` still uses it in its super-horizon warning).
- `compute()` passes `omega_sq=Gk_omegaEff_sq` / `omega_sq=Tk_omegaEff_sq`.
- The "cutting at a point of fixed phase where G' = 0 at a minium" comments
  (`GkNumericIntegration.py:312-314` and the `TkNumericIntegration` twin) rewritten: the point is a
  maximum, and which extremum it is does not matter because `store()` rotates the initial data.

### `main.py` — comment only

- `:604-611` and `:1178-1189`: both `0.85 * z_exit_subh_e6` truncation comments now say that in
  `"stop"` mode the ODE terminates on an event at $z_{e6}$ and the `expected_values` check is
  skipped, so the samples requested between $z_{e6}$ and $0.85z_{e6}$ are **never produced** and
  the returned list is silently shorter; consumers cope; the constant stays because changing it is
  a hand-over decision. `git diff main.py` is comment-only.

### `ComputeTargets/tests/test_numeric_phase_cut.py` — new, 15 tests

Stand-ins `_Wavenumber`, `_KExit`, `_Proxy` (the pattern of `test_gk_wkb_phase.py`), geometry
helper `_geometry(model, k, efolds_suph=5.0)` reproducing review §10's production geometry from
prompt 01's `wkb_reference`, and `_quiet(fn, ...)` capturing stdout. No Ray, no datastore.

## Deviations from the prompt

### 1. The prompt's stop-point assertion $|G'|<10^{-12}|G|\omega$ cannot hold, before or after — STRUCTURALLY REQUIRED

Prompt §3 item 1 asks the bit-identity test to "assert the stop point is an extremum of the same
sign with $|G'|<10^{-12}|G|\omega$". The stop point is a root located by
`root_scalar(..., xtol=1e-6, rtol=1e-4)` inside `find_phase_extremum` — untouched by this prompt,
and tightening it would move the stop point, which is hand-over-adjacent (README §0.3). That
tolerance places the root to $\sim10^{-4}z$, and $G''\approx|G|\omega^2$ there, so the residual
derivative is $O(|G|\omega^2\cdot10^{-4}z)$, not $10^{-12}|G|\omega$. Measured at
`RadiationModel`, $k=10^7$: **$|G'|/(|G|\omega) = 9.76\times10^{-6}$ with the pre-change code** and
$6.52\times10^{-5}$ with the shipped code, against the $10^{-12}$ asked for — the *pre-change* code
misses the prompt's threshold by six orders, so it is a property of the existing root finder, not
of this change.

What shipped instead is the bound `root_scalar`'s own tolerance implies:
$|G'| < |G|\,\omega^2\,(\mathrm{xtol} + \mathrm{rtol}\cdot z)$. Measured: $|G'| = 797.7$ against a
bound of $2.88\times10^4$ (radiation $G_k$), $9.78\times10^{-12}$ against $7.94\times10^{-10}$
(radiation $T_k$). The other two clauses of the prompt's assertion ship unchanged and pass: the
extremum has the same (positive) sign, and value/envelope $= +1$ to $2.1\times10^{-9}$ (worst of
six cases), inside the prompt's $10^{-6}$. The envelope is taken as
$\mathrm{hypot}(G, G'/\omega)$, the local Liouville–Green envelope, so the test needs no analytic
normalisation.

### 2. The diagnostic lives in the integrator, not in the supervisor — IMPLEMENTATION CHOICE

The prompt says "set `has_unresolved_osc`, … from it, and print the same single warning line the
supervisor prints today", which leaves open whether the new test writes through the supervisor or
replaces it. Alternatives considered:

- *(a)* add a second method to `NumericIntegrationSupervisor` that takes an explicit grid spacing
  and sets the same private state, leaving the payload reading `supervisor.has_unresolved_osc`;
- *(b, shipped)* a module-level `scan_sample_grid_for_unresolved_osc` in
  `numeric_with_phase_cut.py`, with the payload reading its return value.

(a) was rejected because the supervisor's three properties gate on `self._delta_logz is None` —
they return `None` when no `delta_logz` was supplied — and the new test does not use `delta_logz`
at all, so the gate would have been a lie: a caller supplying `omega_sq` but not `delta_logz` would
have run the test and then been told `None`. It is also a supervisor of a *running* integration,
and the scan happens after `__exit__`. (b) keeps the gate honest: the fields are `None` exactly
when `omega_sq` is `None`, i.e. when the diagnostic was not requested. Consequence to note: the
"unknown" sentinel is now keyed on `omega_sq` rather than on `delta_logz`. In production both are
always supplied (`main.py:614-630`, `:1182-1199`), so the persisted values and their types are
unchanged; a test asserts the `None` path explicitly.

### 3. `find_phase_extremum` with an alias, rather than keeping the old name — IMPLEMENTATION CHOICE

The prompt offers either. The rename is shipped, with `find_phase_minimum = find_phase_extremum` at
module scope, because the name is the thing review §10.2 calls wrong and the alias costs one line.
The alias is not deprecated-with-a-warning: two `docs/` reproduction scripts and
`prompts/source-remediation` refer to the old name in prose, and a warning would be noise. A test
asserts `find_phase_minimum is find_phase_extremum`.

### 4. The coarse-grid test grid is not "one sample per decade below the source" — IMPLEMENTATION CHOICE

Prompt §3 item 2 asks for "a run whose sample grid is deliberately coarse (one sample per decade)".
Built downward from the source redshift, such a grid spends its first three points *outside* the
horizon, where $\lambda \gg \Delta z$ and the test correctly does not fire — the coarse grid must
be coarse *where the mode oscillates*. The shipped grid is one sample per decade starting at
$z_{e3}$ (the top of the production search window, $x = e^3$) and running two decades deeper, where
$\lambda \le 0.32 z$ against a spacing of $0.9z$. It runs with `mode=None`, because such a grid
ends below $z_{e6}$ and there is no termination event to hit; the flag has nothing to do with the
stop point. The test still asserts what the prompt asks: the flag is `True`, `unresolved_z` is the
first violating pair (checked against an independently recomputed scan), and the warning is printed
exactly once.

### 5. The $\ge30\%$ wall-time drop is reported, not asserted — IMPLEMENTATION CHOICE

Prompt §3 item 3 says "wall time per run on `LambdaCDMModel` falls by $\ge30\,\%$ … Assert the RHS
count; report the time", which is self-consistent only if the time is not also asserted; a
wall-clock threshold is not a property of the tree. The RHS count is asserted exactly (12,854).
The time is printed by the test and is recorded below.

### 6. The unused `pi` import in `GkNumericIntegration` — IMPLEMENTATION CHOICE

Removing the wavelength computation from `Gk_RHS` left `pi` unused in that module; it is dropped.
`TkNumericIntegration` keeps its `pi` (used by the super-horizon warning at `:335`). This is inside
the prompt's file list and is the direct consequence of the edit it asks for, not a separate fix.

## Verification performed

Everything below was **run**, on this tree, from the worktree root.

### Bit-identity (prompt §3 items 1 and 6) — ran

Pre-change outputs were captured at `2ed3632` (the parent of this commit) on a clean tree, before
any edit, for `RadiationModel`, $k=10^7$/Mpc, source 5 e-folds outside the horizon, production
response grid, `mode="stop"`, $(\mathrm{atol},\mathrm{rtol}) = (10^{-10},10^{-8})$. They are
embedded in the test module as `GK_RAD_VALUE_SAMPLE` / `GK_RAD_DERIV_SAMPLE` /
`TK_RAD_VALUE_SAMPLE` / `TK_RAD_DERIV_SAMPLE` (40 samples each) with the SHA in a comment.

| | samples returned | `value_sample` | `deriv_sample` | RHS evaluations |
|---|---|---|---|---|
| `RadiationModel`, $G_k$, $k=10^7$ | 40 of 41 requested | **bit-identical** | **bit-identical** | 12,770 → 12,770 |
| `RadiationModel`, $T_k$, $k=10^7$ | 40 of 41 | **bit-identical** | **bit-identical** | 6,611 → 6,611 |
| `LambdaCDMModel`, $G_k$, $k=10^7$ | 39 of 41 | **bit-identical** | **bit-identical** | 12,854 → 12,854 |

(The 40-of-41 is exactly the `main.py` `0.85 z_e6` behaviour the comment hunks now describe.)

The stop point moves, as prompt §2.3(2b) anticipates, because the sign change is bracketed one step
differently: `stop_deltaz_subh` changes by **1.042e-07 relative** (radiation $G_k$), 4.503e-08
(radiation $T_k$), 1.042e-07 (LambdaCDM $G_k$) — all far inside `root_scalar`'s own
$(\mathrm{xtol},\mathrm{rtol}) = (10^{-6},10^{-4})$. At the new point, value/envelope is
**0.999999997875** (both $G_k$ cases) and **0.999999999819** (radiation $T_k$), against the prompt's
$10^{-6}$; see deviation 1 for the derivative clause.

### The flag (prompt §3 item 2) — ran

- Coarse grid (one per decade from $z_{e3}$, two decades): `has_unresolved_osc` is `True`,
  `unresolved_z` equals the first violating pair recomputed independently in the test, the warning
  is printed exactly **once**.
- Fine grid (2,000 log-spaced points between $z_{e3}$ and $z_{e6}$): `False`, both companion fields
  `None`, nothing printed.
- $\omega^2\le0$ samples are skipped, not raised on.
- No `omega_sq` → all three fields `None`, and the payload keys are asserted to be exactly the ten
  the factories persist (`EXPECTED_PAYLOAD_KEYS`), so nothing in the Datastore path changed.

### Cost (prompt §3 item 3; README §6 row "Numeric RHS diagnostic overhead") — ran

RHS evaluations are **unchanged to the evaluation** (table above), so the solver does exactly the
work it did. Wall time per run, same machine, same geometry, $k=10^7$:

| | pre-change (`2ed3632`) | shipped | drop |
|---|---|---|---|
| `LambdaCDMModel`, $G_k$ (best of 3) | 0.1299 s | 0.0773 s | **40.5 %** |
| `LambdaCDMModel`, $G_k$ (in-suite, best of 3) | 0.1299 s | 0.0833 s | 35.8 % |
| `RadiationModel`, $G_k$ | 0.0880 s | 0.0760 s | 13.6 % |
| `RadiationModel`, $T_k$ | 0.0497 s | 0.0376 s | 24.3 % |

Review §10.2 measured 0.13 s → 0.09 s (45 %) on LambdaCDM; the pre-change 0.1299 s reproduces its
0.13 s exactly and the shipped 0.0773 s beats its 0.09 s. README §6's target ("0 %;
`has_unresolved_osc`, `unresolved_z`, `unresolved_efolds_subh` still populated") is met: the
per-RHS `*_omegaEff_sq` call is gone, the three fields are still populated, and
`grep -n "omegaEff_sq\|report_wavelength" ComputeTargets/GkNumericIntegration.py
ComputeTargets/TkNumericIntegration.py` shows **no call inside `RHS`** — only the import, the
explanatory comment, the `omega_sq=` argument in `compute()`, and the pre-existing `store()` uses
at `GkNumericIntegration.py:419` / `TkNumericIntegration.py:445`.

### The mode guard (prompt §3 item 4) — ran

`mode=None` integrates the whole 40-point grid (`len(value_sample) == len(z_sample)`, the three
stop fields `None`); `mode="STOP"` is accepted and produces a stop point; `mode="x"` raises
`ValueError`.

### The extremum search (prompt §3 item 5) — ran

On a dense-output DOP853 solution of the exact radiation Green's-function ODE
($H=(1+z)^2$, $G=0$, $G'=1$), $k=10^7$:

- **Inside the production window** ($x$ from $e^3 = 20$ to $e^6 = 403$): phase-stepping and the old
  $10^{-3}z$ step find the **same** maximum, separated by less than 0.2 of a half cycle. Both
  return a positive value, confirming the extremum is a maximum.
- **At $x = 6\times10^3$** (review §10.2's "fewer than one sample per cycle at $x>6283$"): the old
  step covers **0.955 of a cycle** at the start of the search. Phase-stepping lands within
  **0.1 cycle** of the first maximum below the start (established with a 100× finer phase step);
  the old step lands **more than a full cycle** away — it skipped at least one maximum, which is
  why §2.3(2b) exists. The window is not widened.

### D2 (prompt §2.2) — ran; the numbers are in the first paragraph

Full sweep, stride 1 over the production source-redshift band, both models, three $k$: **2,149 of
2,149 $G_k$-like runs flag**; **0 of 6 $T_k$-like runs flag**. Per $(model, k)$:

| model | $k$ /Mpc | $G_k$ objects | flagged | first-fire $x$ (min–max) | $T_k$ objects | flagged |
|---|---|---|---|---|---|---|
| RadiationModel | $10^5$ | 358 | 358 | 27.32–62.61 | 1 | 0 |
| RadiationModel | $10^7$ | 359 | 359 | 32.75–56.94 | 1 | 0 |
| RadiationModel | $3\times10^8$ | 358 | 358 | 26.99–61.87 | 1 | 0 |
| LambdaCDMModel | $10^5$ | 358 | 358 | 29.06–66.59 | 1 | 0 |
| LambdaCDMModel | $10^7$ | 359 | 359 | 26.47–60.65 | 1 | 0 |
| LambdaCDMModel | $3\times10^8$ | 357 | 357 | 28.81–66.01 | 1 | 0 |

The $T_k$ near-miss, measured directly as the worst spacing/wavelength ratio over the returned
samples (firing needs $>1$): 0.8220, 0.8068, 0.8182 (radiation, $k=10^5,10^7,3\times10^8$) and
0.8207, 0.8197, 0.8136 (LambdaCDM), at $x = 385$–$394$, $x_T = 223$–$228$. A version of this sweep
at stride 40 (106 $G_k$ objects, 6 $T_k$) ships in the test module as `TestD2FireRate` and asserts
both halves.

### Suites — ran

- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_numeric_phase_cut`:
  **Ran 15 tests … OK** (8.7 s).
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .`:
  **Ran 273 tests in 146.7 s … OK**.
- `./venv/bin/python -m black --check` on all seven touched files: clean. (The repository as a
  whole is **not** clean under `black --check` — 55 files, all untouched by this commit, mostly
  under `docs/`; that predates this prompt.)
- `ast.parse(open("main.py").read())` succeeds; `git diff main.py` is comment-only.

## Observations not acted on

1. **The two-line warning is two `print` calls, not one line.** The prompt says "the same single
   warning line"; the supervisor has always printed a header line and a detail line. Both are
   reproduced verbatim, so the fire-rate figures above mean ~2 lines per flagged object. Worth
   knowing when the print policy is chosen.
2. **`NumericIntegrationSupervisor.report_wavelength` now has no caller in the tree.** It is kept,
   documented as superseded and with its $\ln10$ fixed, as the prompt directs. If the user picks
   print policy (iii) ("pass the intended grid explicitly"), it is the natural place to put the
   per-step form back; if they pick (ii), it should probably go. Not opened as an issue because
   `[00-unresolved-osc-print-policy]` already owns the decision.
3. **`unresolved_efolds_subh` costs one extra `Hubble` call**, made only when the flag fires. On
   the RHS it was free (`k_over_H` was already in hand). Negligible: one call per object.
4. **The scan is $O(n)$ in returned samples and stops at the first violation.** On the production
   response grid that is $\le$ 41 `*_omegaEff_sq` calls against the 12,770 the RHS used to make,
   and in practice 1–3 because the first pair inside the horizon already trips. No measurable cost.
5. **`find_phase_extremum`'s `root_scalar` tolerances (`xtol=1e-6, rtol=1e-4`) are loose** — they
   are what makes deviation 1 necessary, and they place the stop point to $\sim10^{-4}z$, i.e.
   $\sim10^{-4}$ of a cycle at $x\sim6\times10^3$ but $\sim6\%$ of a cycle at $x=403$. Nothing
   downstream depends on it (`store()` rotates the initial data), but it is the reason the stop
   point is not reproducible to better than 1e-7 relative across a change of step. **Opened as
   `[11-stop-point-root-tolerance]`.**
6. **The `mode != "stop"` branch remains untested in production** and its `expected_values` check is
   the only thing that would catch a short return. Kept per `RECONCILIATION.md` §3; now covered by
   two tests.

## State handed to the next prompt

- **`numeric_with_phase_cut` signature** (undecorated call is `numeric_with_phase_cut._function`):

  ```python
  numeric_with_phase_cut(
      model_proxy, k, z_init, z_sample, initial_value, initial_deriv, RHS,
      omega_sq=None,                       # NEW: callable (model, k_float, z) -> omega^2
      atol=DEFAULT_ABS_TOLERANCE, rtol=DEFAULT_REL_TOLERANCE,
      delta_logz=None,                     # accepted, handed to the supervisor, no longer used
      mode=None, stop_search_window_z_begin=None, stop_search_window_z_end=None,
      task_label="numeric_with_phase_cut", object_label="(object)",
  ) -> dict
  ```

  `omega_sq` is positional-or-keyword and sits **after** `RHS`; both integrators pass it by
  keyword. The ten payload keys are unchanged. `has_unresolved_osc`, `unresolved_z`,
  `unresolved_efolds_subh` are `None` iff `omega_sq is None` (previously: iff `delta_logz is None`).

- **New public symbol** `scan_sample_grid_for_unresolved_osc(model, k, k_float, omega_sq,
  sampled_z, object_label) -> dict` in the same module.

- **The extremum finder is `LiouvilleGreen.integration_tools.find_phase_extremum(sol, start_z,
  stop_z, value_index, deriv_index, omega_sq=None)`**, with `find_phase_minimum` retained as an
  alias to the same object. Module constants `PHASE_STEPS_PER_CYCLE = 16` and
  `DEFAULT_RELATIVE_STEP = 1e-3`.

- **D2 fire rates** (the orchestrator puts these to the user): $G_k$ **2,149 / 2,149 = 100 %**,
  first firing at $x = 26.5$–$66.6$; $T_k$ **0 / 6**, peaking at 0.807–0.822 of the trip threshold
  ($x_T = 223$–$228$ against $\approx270$). Today's rate is 0 / 2,155. Keeping the per-object line
  turns 0 printed warnings into ~$1.3\times10^5$ lines per model.

- **Measured speed-up**: LambdaCDM $G_k$ object 0.1299 s → 0.0773 s (40.5 %), RHS evaluations
  identical at 12,854. Radiation $T_k$ 0.0497 s → 0.0376 s.

- **Prompt 12** (the $T_k$ numeric `atol`) touches `config/defaults.py` and every
  `pool.object_get("TkNumericIntegration", …)` in `main.py`. This commit changed neither: its
  `main.py` hunks are the two `0.85 z_e6` comments at `:604-611` and `:1178-1189`, and
  `TkNumericIntegration.compute()` gained only `omega_sq=Tk_omegaEff_sq` in its
  `numeric_with_phase_cut.remote(...)` call — the `atol=self._atol.tol` line directly below it is
  untouched and is where prompt 12's change lands.

- **Test constants**: `ComputeTargets/tests/test_numeric_phase_cut.py` pins the returned samples
  and RHS counts for `RadiationModel`/`LambdaCDMModel` at $k=10^7$. A later prompt that changes the
  numeric solver, its tolerances, or the production grid geometry will have to re-capture them; a
  prompt that does not touch the solver must leave them passing.
