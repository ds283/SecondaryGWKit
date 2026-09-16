# Prompt 01 — The equality-solve characterisation test

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(a)** and **(d)**
**Measurements:** [`AUDIT.md`](AUDIT.md) §2.2, §2.3, §3.2, §4.1 ·
[`RECONCILIATION.md`](RECONCILIATION.md) §2
**Depends on:** nothing. This is the first prompt of the campaign.
**Recommended model:** **Opus** — the production change is nil, but the design decision is the one
the whole campaign rests on: which facts are *invariants that must outlive prompt 02* and which are
*today's behaviour, which prompt 02 deliberately changes*. Assert the second kind and prompt 02
cannot ship without rewriting your test, which destroys README §2 (e).

**Files you may create or touch:** `CosmologyModels/tests/test_rho_equality.py` (**new**), plus this
campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** any production file — **this prompt changes no production code at all**.
Not `LambdaCDM_GenericEOS.py`, not `main.py`, not any other test module (including
`test_wPerturbations.py`: §4 item 3 says *measure* it, not fix it).

**Read first:** `AUDIT.md` §2 and §3 in full; `RECONCILIATION.md` §2 and §9.3;
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:499-520` (the two call sites and the prints)
and `:999-1021` (`_find_rho_equality`, `_rho_fluid` above it);
`CosmologyModels/tests/test_wPerturbations.py:1-120` (the stand-in pattern, `PureRadiationEOS` and
`lambdaCDM_gstar`); and `prompts/background-solver-robustness/measure_rho_equality.py`, which
already implements most of what you need.

---

## 1. Why this prompt exists first

The campaign's primary stop condition is **"the two equality redshifts do not move"** (README §2
(a)). A stop condition that is checked by reading a `print` with `:.4g` is not a stop condition:
four significant figures cannot see a change of $10^{-11}$, and audit §3.1 shows the shipped solve
returning errors up to 2.95e-11 the moment its guess is displaced.

There is also a trap. `AUDIT.md` §2.2 measures the shipped solve at **−4.00e-16** relative — the
double-precision floor — and §2.3 explains that this is because the caller hands it the answer.
**A test that simply pins today's output would therefore pass after prompt 02 no matter what
prompt 02 did**, because the correct answer and the accidentally-correct answer are the same
number. The test has to score against something *independent*: a bracketed `brentq` at Brent's own
floor, and the closed forms.

## 2. What must be asserted, and what must only be recorded

**Assert** — these are true before prompt 02 and must still be true after it:

1. The two equality redshifts, on both models, agree with an independent bracketed reference.
2. Both density ratios are strictly monotone over the range their own root lives in. This is the
   property a bracket-expansion policy rests on, and prompt 02 is about to rest on it.
3. The $\Lambda$ closed form is exact: $\rho_m/\rho_\Lambda\propto(1+z)^3$ with **no temperature
   dependence at all**, so $(\Omega_\Lambda/\Omega_m)^{1/3}-1$ is the root by construction on every
   equation of state, not just a flat-$g_*$ one.

**Record, in the log, do not assert** — these are today's behaviour and prompt 02 changes them:

4. The exception types and messages at guess offsets of −30 %, −50 % and −80 % (audit §3.2).
5. The `_rho_fluid` evaluation counts at the production call sites (audit §2.2: 3, 1, 1, 1).
6. Whether `_find_rho_equality`'s guard at `:1015` fires on any of those paths (it does not).

Items 4–6 become prompt 02's "show it fails on `HEAD~1`" baseline. Put them in the log's **State
handed to the next prompt** section, verbatim, with the exact exception text.

## 3. The test module

Create `CosmologyModels/tests/test_rho_equality.py`. No Ray, no datastore. Build the two models
exactly as `measure_rho_equality.py:186-197` does — `QCD_Cosmology(store_id=…, units=Mpc_units(),
params=Planck2018(), max_z=1e12)` and a `LambdaCDM_GenericEOS` over `PureRadiationEOS(units,
lambdaCDM_gstar(params.Neff))` at `max_z=1e6`. Construct each **once** per test class, in
`setUpClass`: a `QCD_Cosmology` costs 3,000 node root solves plus the two equality solves
(`test_T_z_representation.py:207` makes the same point), and the suite is 0.62 s today.

Required contents:

1. **A module docstring** that states, in prose a later reader can act on: what the solve is, that
   its accuracy today is *inherited from its caller's guess and not from its tolerances*, and that
   this module exists so that the bracketing change cannot move either root. Cite `AUDIT.md` §2.3
   for the flat-$g_*$ argument. **Say explicitly that the module is written to pass both before and
   after prompt 02** — that is the point of it — and that the assertions prompt 02 adds are the
   ones that distinguish the trees.

2. **The reference.** A helper that brackets and solves with `brentq(f, lo, hi, xtol=1e-300,
   rtol=8.9e-16)`. `measure_rho_equality.bracketed_reference` is the implementation; reproduce it
   in the test module rather than importing from `prompts/` — a test must not depend on a campaign
   directory. Comment the `8.9e-16` as $4\varepsilon$, Brent's own floor, with a note that asking
   for less is asking `scipy` for something it cannot deliver.

3. **`test_equality_redshifts_match_a_bracketed_reference`** — for each of the two models and each
   of the two pairs, call `model._find_rho_equality(...)` with exactly the guess `__init__` passes,
   and compare against the reference. **Threshold: 2 ulp**, expressed as
   `abs(got - ref) <= 2 * np.spacing(abs(ref))`, not as a relative tolerance — the quantity is at
   the floor and a relative tolerance invites someone to loosen it later. Print all four pairs of
   17-digit values.

4. **`test_the_density_ratios_are_monotone_where_their_roots_live`** — $\rho_m/\rho_r$ strictly
   decreasing over audit §4.1's probe set $z\in[33, 3.4\times10^5]$, $\rho_m/\rho_\Lambda$ strictly
   increasing over $z\in[0,10]$. Use the probe sets `measure_rho_equality.py` §4.1 uses so the two
   agree. **The docstring must say why this test exists**: it is the precondition for bracketing,
   and an equation of state that broke it would make a bracket-expansion policy unsound rather than
   merely inaccurate.

5. **`test_the_lambda_equality_is_closed_form_on_any_equation_of_state`** — assert
   $(\Omega_\Lambda/\Omega_m)^{1/3}-1$ equals the bracketed reference to 2 ulp on **both** models.
   This is the oracle that does not depend on $g_*$ being flat, and it is what tells a later reader
   that the $\Lambda$ root is safe on any equation of state while the matter–radiation root is not.

6. **`test_the_matter_radiation_closed_form_is_exact_only_because_g_star_is_flat`** — assert
   $g_*(T)$ is equal at $z_{\rm eq}/10$, $z_{\rm eq}$ and $10z_{\rm eq}$ on `QCD_Cosmology` (audit
   §2.3 measures 3.38 at all three), and that the closed form matches the reference there.
   **The docstring is the finding**: the guess is the root *because* of this, so this test is the
   standing statement of the assumption nothing in the production code checks.

## 4. Also measure, and put in the log

1. **Audit §3.2's failure boundary, re-run at your `HEAD`.** All six offsets, with the exact
   exception class and message text. `RECONCILIATION.md` §4 says the message now names
   `TemperatureRepresentation`; confirm it, and confirm the guard at `:1015` still never fires.

2. **The evaluation counts.** Use `measure_rho_equality.counted_match_rho`'s approach. Four numbers.

3. **`RECONCILIATION.md` §9.3's stale comment.** `CosmologyModels/tests/test_wPerturbations.py:34-41`
   describes a "500-point spline" and quotes ~1.3e-9 and ~4e-7; the representation has been a
   segmented entropy factor at 3,000 nodes of order 5 since `qcd-background-audit` prompt 06.
   **Measure what the two figures actually are now** — the `PureRadiationEOS` model against
   `LambdaCDM`'s closed forms at `max_z=1e4` and at `max_z=1e20`, which is what `AGREEMENT_RTOL =
   1.0e-8` was set against. **Do not edit that file.** Report the two numbers and open a §3 issue
   `[01-agreement-threshold-comment-predates-the-representation]`; prompt 08 is where it is fixed
   if the user wants it fixed (README §7 D3).

## 5. Acceptance

| Check | Threshold |
|---|---|
| All four equality redshifts against the bracketed reference | ≤ 2 ulp, all four values quoted at 17 digits |
| Both monotonicity statements | strict at every probe |
| The $\Lambda$ closed form on both models | ≤ 2 ulp |
| `CosmologyModels` suite | 30 → 30 + *n*, OK; quote *n* and the new wall time |
| `ComputeTargets` suite | 447 → 447, OK |
| Production files in the diff | **zero** |
| Runtime of the new module | quote it; if it exceeds ~5 s, say why and whether `setUpClass` is being re-entered |

## 6. Stop conditions

- **Any of the three assertions fails.** That would mean audit §2 or §4.1 does not hold on this
  tree, which `RECONCILIATION.md` §2 says it does. Stop and report the number; do not loosen the
  threshold to make it pass.
- **The monotonicity probe is not strict somewhere.** Prompt 02's whole design rests on it. Report
  where, and stop.
- **You find yourself wanting to change `_find_rho_equality` to make a test cleaner.** That is
  prompt 02. Stop and say so.

## 7. Deliverables

1. `CosmologyModels/tests/test_rho_equality.py`.
2. `logs/01-equality-solve-characterisation.md` per README §5.1 — including the **two equality
   redshifts** section, §4's three measurement blocks, and a **State handed to the next prompt**
   section giving prompt 02 the reference values at 17 digits, the failure-boundary table with
   exact exception text, and the evaluation counts.
3. Board row 01, item table rows for (a) and (d), and the §3 issue from §4 item 3 — with
   `docs/OPEN_ISSUES.md` updated in the same commit, count and date corrected.
4. One commit, README §5 rule 2.
