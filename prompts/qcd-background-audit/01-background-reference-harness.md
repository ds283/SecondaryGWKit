# Prompt 01 — The background-against-background harness, and the guard that would have caught T1

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §8 recommendation **4** — *"the only recommendation here that adds a permanent
guard rather than a fix"*.
**Measurements:** [`docs/qcd-background-audit-2026-09.md`](../../docs/qcd-background-audit-2026-09.md)
§1, §2, §3, §5, §6 · **Reproduction:** `docs/qcd-background-audit/measure_T_z_representation.py`
**Design facts:** README §2 (a), (b), (g), (i) — **read (a) and (b) before writing a line**
**Depends on:** nothing. This is the first prompt.
**Recommended model:** **Opus** — the substance is making the reference genuinely *independent*,
which is the one thing the existing test tree failed to do for six months.

**Files you may create or touch:** `CosmologyModels/tests/T_z_reference.py` (new),
`CosmologyModels/tests/test_T_z_representation.py` (new), this campaign's log and board, and
`docs/OPEN_ISSUES.md`.

**Do not touch:** any production file. **This prompt changes no production code at all** — it adds
tests that pass on the tree as it stands. `CosmologyModels/GenericEOS/`, `ComputeTargets/`,
`Quadrature/`, `Datastore/`, `main.py` and every existing test module are out of bounds.

**Read first:** the audit, all of it, but especially §0.2, §1, §2, §6 and §10;
`docs/qcd-background-audit/measure_T_z_representation.py` (it is the prototype for everything here,
and its `Background` class is the shape to follow);
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` `_solve_T_z` (`:144`) and
`_build_T_z_spline` (`:189`); `CosmologyModels/GenericEOS/QCD_EOS.py`
`break_temperatures_GeV` / `discontinuity_temperatures_GeV`;
`CosmologyModels/tests/test_temperature_spline.py` (the existing style in this package, and
`PureRadiationEOS` in `test_wPerturbations.py`); `ComputeTargets/tests/wkb_reference.py`
(`production_source_grid`, `load_references`).

---

## 1. Why this prompt exists

Every measurement in `docs/gktk-remedial-verification.md` §3.5–§3.6 scores a **consumer against a
producer**. Both are built from the same `BackgroundModel`, hence the same $H$, hence the same
$\tau$. An error in the background **cancels exactly** in that comparison. That is how §3.5 reads
1.00 ulp of the span while the background underneath both sides carries $1.4\times10^5$ radians at
$k=3\times10^8$/Mpc (audit §5, §6).

The repository therefore contains no test that can fail because the background is wrong. This
prompt builds one. It is deliberately first, so that prompts 04, 05 and 06 have something to be
scored against that is **in the test tree**, runs in the suite, and is not a script under `docs/`
that nobody executes.

## 2. What to build

### 2.1 `CosmologyModels/tests/T_z_reference.py` — the reference

A module holding the measurement infrastructure, with **no dependency on anything it measures**.
It may import `QCD_EOS`, `QCD_Cosmology`, `LambdaCDM_GenericEOS`, `HIGH_T_GSTAR` and the units and
parameter blocks. It must **not** call `_solve_T_z`, `_build_T_z_spline`, `T_photon`,
`_T_z_spline` or `integration_break_points` anywhere in the reference path — README §2 (a). State
that rule in the module docstring, as `ComputeTargets/tests/wkb_reference.py` states its own.

Public surface, at minimum:

- `accurate_T(cosmology, z, rtol=1.0e-14) -> float` — the defining equation
  $T\,g_s(T)^{1/3} = T_{\rm CMB}\,g_s(T_{\rm CMB})^{1/3}(1+z)$, bracketed exactly as the audit
  script brackets it (`hi = 1.05 T_CMB (1+z)`, `lo = 0.95 T_CMB (1+z)/g_*^{1/3}`) and solved with
  `root_scalar(..., xtol=1e-300, rtol=rtol)`. **The reference for everything in this campaign.**
- `entropy_factor(cosmology, u, rtol=1.0e-14) -> float` — $F(u)=\log\!\big(T/[T_{\rm CMB}(1+z)]\big)$,
  which is $-\tfrac13\log\!\big(g_s(T)/g_s^{\rm CMB}\big)$: bounded, $O(1)$, and *exactly constant*
  wherever $g_s$ is.
- `jump_locations(cosmology) -> list[float]` — the $u$ at which $T(z)$ crosses each declared break
  temperature, **found by bisecting the monotone $T(z)$**, never by root-finding on
  $T(z)-T_{\rm break}$ (README §2 (b)). Geometric bisection on $(1+z)$ to a relative
  $10^{-15}$, as the audit script does. Returned in ascending $u$.
- `probe_set(...) -> np.ndarray` — the audit's 640-point probe geometry: production source-grid
  nodes **and the midpoints between them**, because a spline is worst between its knots. Restricted
  to $z\in(1, 10^{16})$ and decimated by 5, exactly as the audit did, so the numbers in README §6.1
  are reproduced to the digit. The grid itself may come from
  `ComputeTargets.tests.wkb_reference.production_source_grid`; if importing across test packages is
  awkward under `discover -s CosmologyModels/tests -t .`, reproduce the geometry locally from the
  documented constants and **say which you did** in the log.
- `relative(candidate, reference)`, and a small `Stats` (max / p90 / median) so that every later
  prompt reports the same three numbers in the same order.

The **cost** of `accurate_T` is 8.3 µs per call, so a 640-point probe reference is ~5 ms. Cache it
at class level in `setUpClass`; do not rebuild it per test.

### 2.2 `CosmologyModels/tests/test_T_z_representation.py` — the tests

Six cases. **Each one must pass on the tree as it stands**, with its threshold set at the *shipped*
value and a comment naming the prompt that tightens it. This is a characterisation harness on
2026-09-13 and an accuracy guard from prompt 06 onward; write it so that the transition is a
one-line threshold edit and nothing else.

1. **`test_T_z_matches_the_defining_equation`** — the shipped `T_photon` against `accurate_T` on
   the probe set. Assert max / p90 / median at **7.2e-04 / 1.4e-05 / 2.0e-07**. Print all three.
   *Tightened by prompts 04, 05, 06 to README §6.1.*
2. **`test_the_node_solve_converges`** — the shipped `_solve_T_z` against `accurate_T` at the probe
   points. Assert max **2.5e-05**. *Tightened by prompt 04 to 1e-14.* This is the one test that may
   call `_solve_T_z`, because `_solve_T_z` is its subject.
3. **`test_conformal_time_matches_the_exact_background`** — **the T1 guard, and the reason this
   prompt exists.** $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$, computed twice: once with the
   cosmology as shipped, once with its temperature replaced by `accurate_T`. Integrate in
   $u=\log(1+z)$ with `scipy.integrate.quad`, `epsabs=0`, `epsrel=1e-11`, `limit=400`, and
   `points=` the interior `jump_locations()`; suppress `IntegrationWarning` with the audit's
   justification (the exact integrand root-solves on every call, so `quad` reports roundoff before
   it reaches `epsrel`; the figure is converged — 3.4744e-08, 3.4602e-08, 3.4605e-08, 3.4605e-08 at
   `epsrel` = 1e-9 … 1e-12). Assert the relative difference is **≤ 4e-08** and print it together
   with the equivalent phase at $k=10^5$, $10^7$ and $3\times10^8$ against the 1-ulp floors.
   *Tightened by prompt 06 to ≤ 1e-15.*
   **Budget:** the exact integral is the expensive one. Keep the whole test under ~20 s; if it is
   slower, reduce the range and say so, but do not reduce it below four decades.
4. **`test_the_branch_joins_are_where_the_fixture_puts_them`** — README §7 D6 and issue
   `[00-eos-branch-joins-do-not-match]`. For each of the four `break_temperatures_GeV`, evaluate
   $g$ and $g_s$ at $T(1\pm10^{-9})$ and assert the relative jumps against audit §1's table to
   three significant figures: $10^{16}$ GeV **+1.454e-02 / +1.395e-02**, 0.12 GeV
   **−2.075e-04 / −3.744e-04**, 0.002 GeV **|·| ≤ 1e-10** (it matches to 1.751e-11), $10^{-5}$ GeV
   **+8.876e-04 / −2.284e-03**. The docstring must say this is a **characterisation** test of an
   upstream data fixture that this campaign deliberately does not repair, and that a failure means
   the fixture changed and the campaign's segment edges must be re-derived — **not** that the code
   regressed.
5. **`test_T_z_is_a_step_at_the_lowest_crossing`** — audit §2. $F(u)$ at
   $z = z_c\times(0.95, 0.99, 0.999)$ and $z_c\times(1.001, 1.01, 1.05)$ with
   $z_c = 4.25337\times10^7$: flat to twelve decimals on each side, and a step of
   **7.614e-04** between them. This is the test that makes "a step, not a kink" a fact in the tree
   rather than a claim in a document, and it is what justifies segmentation in prompt 06.
6. **`test_a_constant_gs_equation_of_state_is_an_exact_ramp`** — `PureRadiationEOS` (from
   `CosmologyModels/tests/test_wPerturbations.py`) has constant $g_s$, so $F(u)$ is identically
   constant and $T=T_{\rm CMB}(1+z)$ exactly. Measure what the **shipped** representation does to
   that exact ramp and assert it. *Tightened by prompt 05, where the new representation should be
   exact to a few ulp.* README §2 (g).

### 2.3 What the harness must also expose, because prompt 06 needs it

`jump_locations()` is the function prompt 06 builds its segment edges from, and README §2 (b) is
the reason it lives here rather than in production code first. Add a seventh test:

7. **`test_a_segment_edge_bisected_and_one_root_found_disagree`** — solve for the lowest crossing
   both ways: by `jump_locations()`'s bisection, and by a `root_scalar` bracket on
   $T(z)-T_{\rm break}$ as a naive implementation would. Assert that the two **do not** agree to
   better than a grid interval, or — if the bracketing solver raises or returns a non-root — that
   it fails. Either outcome documents the trap; a test that finds them equal is a **stop**, because
   it would mean audit §2's central claim does not hold on this tree and prompt 06's design rests
   on it.

## 3. Verification and acceptance

- All three suites pass; quote the counts before and after. The `CosmologyModels` count rises by
  seven.
  ```bash
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -3
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3
  ```
- **Every threshold in §2.2 gets its measured value quoted in the log**, next to the audit's figure.
  Where they differ by more than the last digit, say so: the audit was taken on `b3e3769` and this
  tree is `e8f746d`, and no production file changed in between, so they should agree.
- **Wall time of the new module**, reported. If `test_conformal_time_matches_the_exact_background`
  dominates, say by how much; prompt 06 re-runs it.
- `black` clean.

## 4. Log and commit

Log to `logs/01-background-reference-harness.md` per README §5.1. The log's **"State handed to the
next prompt"** must carry: the exact public signatures of `T_z_reference.py`; the seven test names;
the measured value behind every asserted threshold; the three `jump_locations()` values **to 17
digits** (prompt 06 needs them); and the module's wall time.

Open `[00-eos-branch-joins-do-not-match]` on this campaign's board §3 — the text is already drafted
there by the planning commit; confirm its numbers against your own run and correct any that differ
— and check that `docs/OPEN_ISSUES.md` §1.7 carries its row.

Commit subject, or something equally specific:
`Add a background-against-background test for the QCD temperature`
