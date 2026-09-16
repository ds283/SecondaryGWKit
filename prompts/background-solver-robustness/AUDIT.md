# Audit — the root solves in `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`

**Taken:** 2026-09-16 at `be21f5c` (`tolerance-convergence`, on the tree
`qcd-background-audit` left at `acd5b8e`).
**Subject:** every root solve in `LambdaCDM_GenericEOS.py`, and in particular
`_find_rho_equality` at **`:1008`**.
**Measurements:** [`measure_rho_equality.py`](measure_rho_equality.py) — needs neither Ray nor a
datastore. **Every figure below comes from that script; re-run it rather than trusting this
document.**
**Status:** audit only. No production code has been changed and no campaign has been launched.

---

## 0. The one-paragraph version

`LambdaCDM_GenericEOS.py` contains three root solves. Two of them —
`_solve_T_z` (`:578`) and `_temperature_crossing_log1pz` (`:864`) — were audited by
`prompts/qcd-background-audit`, are bracketed, carry tolerances at the representable floor, and
carry comments explaining the choice. The third, `_find_rho_equality` (`:1008`), was not
audited by anything: it runs an **unbracketed secant** at `xtol=1e-6, rtol=1e-4`, tolerances
two orders looser than anything else in the file. **Measured, it returns the right answer to
4.0e-16 relative** — because the analytic initial guess the caller supplies is already the root
to rounding — and **both quantities it produces are used only in two `print` statements**. So
there is no defect in any computed number today. What there is, is a solve that is correct by
accident rather than by construction, whose failure mode escapes its own convergence guard, and
whose provenance no campaign record can state. That is a robustness and provenance problem, and
this document scopes it as one. **Anyone launching a campaign from this audit should read §5
first: the honest impact of fixing it is zero change to any computed quantity.**

---

## 1. The three solves, and which one is the subject

| # | Site | Method | Tolerances | Bracketed? | Audited by | Verdict |
|---|---|---|---|---|---|---|
| 1 | `_solve_T_z`, `:578` (`prompts/tolerance-convergence` cites the argument line, `:579`) | Brent | `xtol=1e-300`, `rtol=1e-14` | **yes** | `qcd-background-audit` prompt 04 | **sound** — `rtol` sits just above Brent's own floor of $4\epsilon \approx 8.9\times10^{-16}$; `xtol=1e-300` disables the absolute component deliberately, because $T$ spans twenty decades and any finite absolute tolerance binds at the cold end before the hot end resolves. The reasoning is in the code at `:569-578`. |
| 2 | `_temperature_crossing_log1pz`, `:864` | Brent | `xtol=1e-15`, `rtol=1e-15` | **yes** | `qcd-background-audit` prompts 06, 07 | **sound, and not on the production path.** Nothing in production calls it; `integration_break_points` uses the *geometric bisection* at `:754` instead, because $T(z)$ genuinely jumps at the crossings and a bracketing solver reports `converged` on a non-root there. The docstrings at `:754-796` and `:823-855` record the trap and name the standing test that demonstrates it. |
| 3 | **`_find_rho_equality`, `:1008`** | **secant** (`x0=`, no `bracket=`) | **`xtol=1e-6`, `rtol=1e-4`** | **no** | **nothing** | **the subject of this audit** |

Solve 3 is the only one left in the file at the tolerances `qcd-background-audit` prompt 04
removed from solve 1. The campaign that tightened `:578` did not touch `:1008`, and its logs do
not say why — most likely because `:1008` is outside the $T(z)$ representation that campaign
owned.

`prompts/tolerance-convergence` records `:1008` in its board §3 under *"Recorded by the rebase,
**not owned here**"*, with the note: *"Prompt 02 records what it is and what it feeds; if it turns
out to matter, that is an issue for whoever owns that file and not a repair to make in passing."*
**§2 and §3 below are the answer to that conditional.**

---

## 2. What `:1008` is, and what it achieves

### 2.1 What it feeds

```python
root = root_scalar(match_rho, x0=init_z, xtol=1e-6, rtol=1e-4)
```

`_find_rho_equality` is called **exactly twice**, both inside `LambdaCDM_GenericEOS.__init__`:

| Call site | Pair | Initial guess supplied by the caller |
|---|---|---|
| `:496` | matter = radiation | `self.omega_m / self.omega_r - 1.0` |
| `:499` | matter = $\Lambda$ | `pow(self.omega_cc / self.omega_m, 1/3) - 1.0` |

Both results land in **local variables** (`matter_radiation_equality`, `matter_cc_equality`) that
are printed at `:508-509` with `:.4g` and then discarded. They are never stored on the object,
never persisted to the datastore, never returned, and never read by anything downstream. A
repository-wide grep for `_find_rho_equality` returns three hits: the two call sites and the
definition.

**The blast radius of this solve is two banner lines.**

### 2.2 What the shipped tolerances actually achieve

Against an independent `brentq` reference at `xtol=1e-300, rtol=8.9e-16`
(`measure_rho_equality.py` §2 output):

| Model | Pair | Shipped root | `_rho_fluid` evals | Relative error |
|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.6689742499` | 3 | **−4.00e-16** |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.30342303299641` | 1 | **−3.66e-16** |
| pure-radiation stand-in | matter = radiation | `3403.1059638279` | 1 | **−1.34e-16** |
| pure-radiation stand-in | matter = $\Lambda$ | `0.30342303299641` | 1 | **−3.66e-16** |

**One to three evaluations, and the double-precision floor.** The printed values are correct to
every one of the four significant figures displayed. The nominal `rtol = 1e-4` — which at
$z \sim 3400$ would permit a displacement of $\pm0.34$ — never binds.

### 2.3 Why: the initial guess is already the root

The matter–radiation equality sits at $z \sim 3.4\times10^3$, where the photon temperature is
$T \sim 8\times10^{-10}$ GeV — far below every QCD threshold. $g_*(T)$ is flat there:

| $z$ | $T$ | $G(T)$ |
|---|---|---|
| $3.40\times10^{2}$ | 8.00566e-11 GeV | 3.38 |
| $3.41\times10^{3}$ | 8.00566e-10 GeV | 3.38 |
| $3.41\times10^{4}$ | 8.00566e-09 GeV | 3.38 |

With $g_*$ constant, $\rho_r \propto (1+z)^4$ **exactly**, so $\rho_m = \rho_r$ has the closed
solution $1+z = \Omega_m/\Omega_r$ — which is precisely the guess the caller passes. Secant
evaluates, finds $f(x_0)$ at the rounding floor, and stops. The $\Lambda$ case is closed-form for
the same reason: $\rho_m/\rho_\Lambda \propto (1+z)^3$ with no temperature dependence at all, so
the cube-root guess is exact by construction, in **one** evaluation.

**This is the finding.** The solve is not accurate because its tolerances are adequate. It is
accurate because it is handed the answer and asked to confirm it. Nothing in the code says so,
and nothing enforces it.

---

## 3. What is actually wrong

### 3.1 The tolerance is loose but not binding

Displacing the initial guess so the solver has to do real work (`measure_rho_equality.py` §3.1,
QCD, matter = radiation, reference $z = 3406.66897425$):

| Guess offset | Shipped rel. err | evals | At `xtol=1e-300, rtol=1e-14` | evals |
|---|---|---|---|---|
| +0 % | −4.00e-16 | 3 | −4.00e-16 | 3 |
| +1 % | 1.89e-13 | 9 | 1.33e-16 | 15 |
| +5 % | 6.14e-15 | 12 | −1.33e-16 | 15 |
| +20 % | 2.88e-13 | 15 | 0.00e+00 | 21 |
| +50 % | 2.95e-11 | 18 | −2.67e-16 | 24 |
| +200 % | 1.45e-11 | 27 | 0.00e+00 | 33 |

Secant's superlinear convergence overshoots the stopping test by seven to twelve orders, so even
a badly displaced guess lands far inside the requested tolerance. **The tolerance is a poor
description of what the solve delivers, in both directions**: it neither binds the error nor
predicts it. Note also that `xtol = 1e-6` is an *absolute* tolerance in $z$ and is meaningless at
$z \sim 3400$ while being the only thing that acts at $z \sim 0.3$ — the two roots differ by four
decades, which is the same argument `qcd-background-audit` prompt 04 wrote into the comment at
`:575-578` for `_solve_T_z`.

**Cost of tightening: 6 to 9 extra `_rho_fluid` calls, twice, at model construction.** Each is one
spline evaluation. It is free, and the campaign should not pretend otherwise in either direction.

### 3.2 The solve is unbracketed, and its failure escapes its own guard

This is the more serious half. `root_scalar` with `x0=` and no `bracket=` selects secant, which is
free to iterate anywhere. Walking the guess *down* (`measure_rho_equality.py` §3.2):

| Guess offset | $z_0$ | Outcome |
|---|---|---|
| −5 % | 3236.34 | `converged=True`, root 3406.668974 |
| −10 % | 3066.00 | `converged=True`, root 3406.668974 |
| −20 % | 2725.34 | `converged=True`, root **3406.668983** — rel. err **2.7e-9** |
| −30 % | 2384.67 | **`ValueError: math domain error`** |
| −50 % | 1703.33 | **`RuntimeError: … evaluated T(z) out of bounds @ z=-0.25719`** |
| −80 % | 681.33 | **`RuntimeError: … evaluated T(z) out of bounds @ z=-0.38344`** |

Three things follow:

1. **The iterate leaves the tabulated range.** At −50 % and beyond the secant has walked to
   *negative* $z$, below the $T(z)$ spline's floor of $z = -0.24$.
2. **Neither failure is `converged=False`.** `_find_rho_equality`'s own guard at `:1010`,
   `if not root.converged: raise RuntimeError(...)`, **never fires** — the exception comes out of
   `_rho_fluid` instead. The guard is dead code on every path that actually fails. A
   `LambdaCDM_GenericEOS` constructor that dies this way reports a temperature-spline bounds error
   with no mention of equality-finding, which is a bad first message for a user whose real mistake
   was a cosmology with an unusual $\Omega_r$.
3. **Both roots are trivially bracketable** — see §4.1 — so there is no reason to accept any of
   this.

### 3.3 Provenance

No campaign record anywhere states who chose `xtol=1e-6, rtol=1e-4` here, or why.
`prompts/tolerance-convergence` README §1.2 requires that, when it closes, **no accuracy parameter
in the pipeline is unexplained**, and lists this site among those "nobody has ever chosen". As
things stand its provenance entry can only read *unestablished, and still loose*.

---

## 4. What a fix would be

### 4.1 Bracketing is available, and the monotonicity argument is measured

Both ratios are strictly monotone across the range their own root lives in
(`measure_rho_equality.py` §4.1), so a bracket expanded geometrically from the analytic guess is
guaranteed to straddle the root and Brent applies:

- $\rho_m/\rho_r \propto 1/\big((1+z)\,g_*(T)\big)$ — **strictly decreasing**, confirmed over
  $z \in [33, 3.4\times10^5]$. ($g_*$ rises with $T$ and hence with $z$, so the two effects add
  rather than competing.)
- $\rho_m/\rho_\Lambda \propto (1+z)^3$ — **strictly increasing**, confirmed over
  $z \in [0, 10]$, with no temperature dependence at all.

The two probe ranges differ because the two roots do ($z \sim 3.4\times10^3$ against
$z \sim 0.3$); a fix must not assume one bracket serves both.

### 4.2 The shape of the change

1. Bracket both solves, expanding geometrically from the analytic guess, and use `brentq`.
2. Tighten to `xtol=1e-300, rtol=1e-14`, matching `_solve_T_z` at `:578`, with a comment to the
   same standard — the measurement that chose the value, in the file, at the point of use.
3. Make the existing `if not root.converged` guard reachable: a failure to bracket should raise
   `_find_rho_equality`'s own error naming the species pair and the range searched, not a
   temperature-spline bounds error from two frames down.
4. A test in `CosmologyModels/tests/` — no Ray, no datastore, the stand-in pattern
   `test_wPerturbations.py` already uses — asserting (i) both equality redshifts unchanged to
   ~1 ulp against a `brentq` reference, and (ii) that the displaced-guess path now raises the
   module's own `RuntimeError` rather than `ValueError` or a spline bounds error.

### 4.3 What must stay out of scope

- **`:578` and `:864`.** Both audited, both sound (§1). Touching them re-opens
  `qcd-background-audit`'s work.
- **`find_phase_extremum`, `LiouvilleGreen/integration_tools.py:95`.** It is
  `root_scalar(xtol=1e-6, rtol=1e-4)` and looks identical to `:1008`. **It is not the same
  problem**: it is bracketed, it is on the production path, and it sets a *computed* quantity —
  the numeric→WKB hand-over point. It is `[11-stop-point-root-tolerance]`, owned by the hand-over
  campaign (`docs/OPEN_ISSUES.md` §1.1), and `prompts/tolerance-convergence` §0.5 explicitly
  declines to retune it. A campaign launched from this audit must not absorb it.
- **Any change to the printed diagnostics themselves**, to `_rho_fluid`, or to the $T(z)$
  representation.

---

## 5. Impact, stated plainly

**Fixing this changes no computed quantity in the pipeline.** The two numbers involved are
diagnostics; they are already right to the double-precision floor; and the datastore is not
affected, so no regeneration is implied.

What a fix buys is:

- a solve that is correct **by construction** rather than because its caller happens to hand it
  the answer — the current behaviour is one exotic equation of state away from silently degrading,
  since the flat-$g_*$ argument of §2.3 is a property of *this* EOS at *this* redshift and nothing
  checks it;
- a reachable convergence guard and a comprehensible failure message (§3.2);
- a provenance entry that `docs/TOLERANCE-PROVENANCE.md` can state rather than record as
  unestablished (§3.3);
- the last loose tolerance removed from a file whose other two solves were tightened for exactly
  these reasons.

**This is a robustness and provenance fix, not a numerics fix, and a campaign launched from this
audit should say so in its README §0.** An agent told to "fix the precision problem at `:1008`"
without that framing will look for an impact, fail to find one, and either inflate the change or
stall.

---

## 6. Observations not acted on

Recorded here because they were found while taking the measurements above, not because this audit
proposes to fix them. Each would need its own decision.

1. **`TemperatureRepresentation.__call__` raises with the wrong class name.** Both bounds errors,
   at `:321` and `:333`, are formatted
   `f"GkSource.function: evaluated {self._label} out of bounds @ z=…"` — a label copied from an
   unrelated class. Every out-of-bounds $T(z)$ failure in the tree therefore reports itself as a
   `GkSource` problem. Visible in §3.2's table. One-line fix, but it is a different file region
   from `_find_rho_equality` and it should be a separate commit.
2. **`_find_rho_equality`'s `init_z` is computed by the caller, not the method.** Both call sites
   derive the closed-form guess inline at `:496-503`. Since the guess *is* the root whenever
   $g_*$ is flat (§2.3), a method that derived its own guess could also assert that property and
   fail loudly when an EOS violates it. That is a design change, not a repair, and it is out of
   scope for a fix scoped as §4.2.
3. **`ComputeTargets/QuadSourceIntegral.py:1550`** still says the pipeline supplies
   `DEFAULT_QUADRATURE_ATOL = 1e-25`; the constant has been `1e-32` since `source-remediation`
   prompt 12. Already recorded by `prompts/tolerance-convergence` board §3; repeated here only so
   that a campaign launched from this audit does not re-discover it and treat it as new.

---

## 7. Relationship to `prompts/tolerance-convergence`

The two do not overlap in files. `tolerance-convergence` §0.5 does not own
`CosmologyModels/GenericEOS/`, and its only contact with this file is **prompt 02's read-only
inventory**, which lists `:579`, `:864` and `:1008` among the hard-coded tolerance literals it
must record.

**Sequencing matters in one direction only.** If a fix lands before `tolerance-convergence`
prompt 02 runs, prompt 02 records a settled provenance entry and
`docs/TOLERANCE-PROVENANCE.md` can state it. If it lands after, the note records
*unestablished* and the fix becomes a follow-up amendment. `tolerance-convergence` has not
started — there are no numbered prompt files in it yet — so the cheap moment is now.

A campaign from this audit should branch from `main`, not from `tolerance-convergence`. The file
sets are disjoint, so merging `main` into the campaign branch afterwards is conflict-free and no
second rebase of `RECONCILIATION.md` is needed.
