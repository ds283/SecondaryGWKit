# Prompt 02 — Bracket the equality solve

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(b)**, **(c)**, **(e)** · **Implements:** `AUDIT.md` §4.2 items 1–4
**Closes:** `[00-equality-solve-is-unbracketed-and-loose]` on this board
**Measurements:** `AUDIT.md` §3.1, §3.2, §4.1 · log 01
**Depends on:** **01** — its test is your acceptance and its log holds your baseline.
**Recommended model:** **Opus** — this is the campaign's only production numerics change. The
substance is not the `root_scalar` call; it is the bracket-expansion policy for two roots four
decades apart, the behaviour when the guess is already the root to rounding, and an acceptance that
is bit-for-bit.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`
(**`_find_rho_equality` only**, `:999-1021`), `CosmologyModels/tests/test_rho_equality.py` (add
tests; do not weaken prompt 01's), plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `_solve_T_z` (`:539-595`), `_temperature_crossing_log1pz` (`:828-880`),
`_bisect_temperature_crossing_log1pz` (`:759-826`), `_rho_fluid` (`:971-997`), `__init__`'s call
sites at `:501-506` or the prints at `:516-517` — **`_find_rho_equality`'s signature does not
change and its callers are not edited** (audit §6 observation 2 is README §7 D2, prompt 03's, and
explicitly not yours); `main.py`; `ComputeTargets/`; any other test module.

**Read first:** `AUDIT.md` §3 and §4 in full; `LambdaCDM_GenericEOS.py:569-595` — **the comment
`qcd-background-audit` prompt 04 wrote for `_solve_T_z` is the standard your comment is held to**;
`logs/01-equality-solve-characterisation.md` "State handed to the next prompt"; and
`CLAUDE.md`'s redshift-arithmetic paragraph.

---

## 1. What is wrong

```python
root = root_scalar(match_rho, x0=init_z, xtol=1e-6, rtol=1e-4)      # :1013

if not root.converged:                                              # :1015
    raise RuntimeError(...)
```

Three defects, in increasing order of seriousness.

**(i) The tolerances are wrong in kind, not only in size.** `rtol = 1e-4` at $z\sim3400$ would
permit a displacement of $\pm0.34$; it never binds, because secant's superlinear convergence
overshoots the stopping test by seven to twelve orders (audit §3.1). So the tolerance neither
bounds the error nor predicts it. `xtol = 1e-6` is an **absolute** tolerance in $z$, and the two
roots are at $z\approx3.4\times10^3$ and $z\approx0.3$ — four decades apart. It is meaningless at
the first and is the only thing acting at the second. This is the same argument
`qcd-background-audit` prompt 04 put in the comment at `:574-583` for `_solve_T_z`.

**(ii) The solve is unbracketed, and its failure escapes its own guard.** `root_scalar` with `x0=`
and no `bracket=` selects secant, free to iterate anywhere. Audit §3.2, reproduced at `f023eb8`:

| Guess offset | $z_0$ | Outcome |
|---|---|---|
| −20 % | 2725.34 | `converged=True`, root off by 2.7e-9 |
| −30 % | 2384.67 | `ValueError: math domain error` |
| −50 % | 1703.33 | `RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.25719` |

The iterate walks to **negative $z$**, below the $T(z)$ spline's floor of $z=-0.24$. **Neither
failure is `converged=False`**, so the guard at `:1015` is dead on every path that actually fails,
and a constructor that dies this way reports a temperature-spline bounds error with no mention of
equality-finding — a bad first message for a user whose real mistake was an unusual $\Omega_r$.

**(iii) It is correct by accident.** Audit §2.3: $g_*$ is flat at $z_{\rm eq}$, so
$\rho_r\propto(1+z)^4$ exactly and $1+z=\Omega_m/\Omega_r$ is the closed solution — which is the
guess the caller passes. The solve evaluates once to three times and stops at the rounding floor.
Nothing in the code says so and nothing enforces it.

## 2. The change

### 2.1 Bracket, then Brent

Both roots are bracketable: audit §4.1 measures $\rho_m/\rho_r$ **strictly decreasing** over
$z\in[33,3.4\times10^5]$ and $\rho_m/\rho_\Lambda$ **strictly increasing** over $z\in[0,10]$, and
prompt 01 pins both. Expand a bracket geometrically about `init_z` until `match_rho` changes sign,
then call `brentq` (or `root_scalar(..., bracket=...)`, which selects Brent).

Design points you must settle and **justify in the log**, because the prompt deliberately leaves
them open:

1. **The expansion variable.** `CLAUDE.md`: integration is in $\log(1+z)$, $z\to\log(1+z)$ is safe,
   the reverse is irreducibly lossy at large $z$. Here the roots are at $z\sim3.4\times10^3$ and
   $z\sim0.3$, where the reverse conversion is harmless, and the caller's guess and the callers'
   prints are both in $z$. **Expanding in $1+z$ multiplicatively is the natural choice** — it is
   scale-free, so one policy serves both roots, and it never produces $1+z\le0$. Expanding in $z$
   additively does not have either property. Say which you chose and why.
2. **The expansion factor and the iteration cap.** A factor of $\sqrt2$ or 2 in $1+z$, capped at
   enough doublings to cover the tabulated range, is ample: the guess is the root to rounding in
   every production case, so the first bracket almost always straddles. **The cap must be a named
   constant with a comment saying what range it covers**, not a magic number in a `while`.
3. **The floor.** The expansion must never propose $z$ below the $T(z)$ representation's own floor
   (`DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2`, with the 5 % buffer `_build_T_z_spline` applies).
   Clamping there and then failing to bracket is **correct and is the point** — see §2.3. Walking
   past it and letting `_rho_fluid` raise is defect (ii) preserved under a new name.
4. **The guess is already the root.** In every production case `match_rho(init_z)` is at the
   rounding floor. A bracket expansion that starts by evaluating the guess will find
   `match_rho(init_z)` tiny but non-zero, and both expanded endpoints will straddle it immediately.
   Confirm this is what happens and quote the evaluation count. **Do not add a short-circuit that
   returns `init_z` when the residual is small** — that reintroduces defect (iii) in explicit form,
   and it is the one thing this prompt exists to remove.

### 2.2 The tolerances

Ship **`xtol=1e-300, rtol=1e-14`** (README §7 **D1**), identical to `_solve_T_z:583`. If `scipy` on
this version rejects `xtol=1e-300` for `brentq`, say what it accepted and why.

Replace the absent comment with one **held to the standard of `:569-583`**: the measurement that
chose the value, in the file, at the point of use. It must say

- that `rtol=1e-14` sits just above Brent's own floor of $4\varepsilon\approx8.9\times10^{-16}$ and
  is therefore the tightest tolerance `root_scalar` can actually resolve;
- that `xtol=1e-300` disables the absolute component **deliberately**, because the two roots this
  method serves are four decades apart and any finite absolute tolerance binds at one of them;
- that the cost is 6 to 9 extra `_rho_fluid` calls, twice, at model construction — each one spline
  evaluation (audit §3.1);
- **and the finding of audit §2.3**: that until this change the solve was accurate because its
  caller handed it the closed-form root, not because its tolerances were adequate, and that the
  bracket is what makes it correct by construction on an equation of state where $g_*$ is *not*
  flat at equality. This sentence is the most valuable line in the diff. Cite `AUDIT.md` §2.3 and
  this campaign.

### 2.3 Make the guard reachable

The `if not root.converged` guard stays, but it is no longer the only guard. A **failure to
bracket** must raise `_find_rho_equality`'s own `RuntimeError` naming

- the two species, as passed (`species_A`, `species_B`);
- the initial guess;
- the range actually searched, as $z$ endpoints;
- the residual at each endpoint, so a reader can see it did not change sign.

Follow the house style of `_solve_T_z:587-590` and `_temperature_crossing_log1pz:871-875`: the
message begins `LambdaCDM_GenericEOS._find_rho_equality: `. Keep the existing `converged` guard's
message and add the species pair to it, so both failure modes identify themselves.

**Do not catch and re-raise `_rho_fluid`'s bounds error.** The bracket must be constructed so that
`_rho_fluid` is never called out of range in the first place; a `try/except` around it would hide
the very coupling this prompt removes.

## 3. Tests

Add to `CosmologyModels/tests/test_rho_equality.py`. **Do not weaken or delete anything prompt 01
wrote** — its assertions are the "nothing moved" half of the acceptance and they must still pass
unchanged.

1. **`test_a_displaced_guess_raises_the_methods_own_error`** — call `_find_rho_equality` with the
   matter–radiation guess displaced by −50 % (audit §3.2: today a `TemperatureRepresentation` bounds
   error). Assert `RuntimeError`, and use `assertRaisesRegex` against the **species names** and the
   string `_find_rho_equality` — not against the whole message, which is prose and will be
   reworded.
2. **`test_a_displaced_guess_does_not_raise_from_the_temperature_representation`** — the same call
   at −80 % must not produce a message mentioning `TemperatureRepresentation`, and must not raise
   `ValueError`. This is the assertion that distinguishes the trees.
3. **`test_the_bracket_does_not_leave_the_tabulated_range`** — with a guess so bad that no bracket
   exists, the failure is the method's own error and the search range reported is inside the
   representation's own bounds. Choose the guess from your measurements, not from this prompt.

**README §2 (e): show items 1 and 2 failing on `HEAD~1`.** Run them against the parent commit and
**quote the failure output in the log**. A test that passes both before and after proves nothing,
and this campaign's other stop condition is "nothing moved" — which is also what a test that tests
nothing reports. This is the single most important review step in the workstream.

## 4. Acceptance

| Check | Threshold |
|---|---|
| The four equality redshifts, against prompt 01's bracketed reference | **bit-identical**; if not, ≤ 1 ulp with both floats at 17 digits and an explanation |
| Prompt 01's four test methods | pass **unchanged** |
| §3 items 1 and 2 on `HEAD~1` | **fail**, output quoted |
| `_rho_fluid` evaluation count at each of the four production call sites | quoted before and after; audit §3.1 predicts +6 to +9 |
| Wall-clock cost of constructing a `QCD_Cosmology` | quoted before and after; the expectation is "unmeasurable against 3,000 node solves" |
| `CosmologyModels` suite | 30 + *n* → 30 + *n* + 3, OK |
| `ComputeTargets` suite | **447 → 447**, OK |
| `T_Z_REPRESENTATION_VERSION` | **6** before and after |
| The two printed banner lines | character-identical at `:516-517` |
| `black --check` | clean |

## 5. Stop conditions

- **Either root moves by more than 1 ulp.** Report both floats at 17 digits and stop. Do not adjust
  a threshold to accommodate it; README §2 (a) is the campaign's primary stop condition and a move
  here means the bracket has changed the answer, which is exactly what must not happen.
- **A bracket cannot be constructed for either production case within the cap.** Audit §4.1 says it
  can. Report the endpoints and residuals, and stop.
- **`ComputeTargets` moves at all.** Nothing there reads this method; if a number moves, something
  else changed and you need to find out what before committing.
- **You need to change `__init__`, `_rho_fluid`, or the signature** to make the prompt work. Stop
  and ask — that is README §7 D2 and prompt 03's subject.

## 6. Deliverables

1. `_find_rho_equality`, bracketed, tightened, guarded, and commented to the `:569-583` standard.
2. Three new tests in `CosmologyModels/tests/test_rho_equality.py`.
3. `logs/02-bracket-the-equality-solve.md` per README §5.1 — including the **two equality
   redshifts** section at 17 digits, the `HEAD~1` failure output, the evaluation counts, and an
   `IMPLEMENTATION CHOICE` entry for every one of §2.1's four design points.
4. Board row 02, item rows (b), (c), (e), and `[00-equality-solve-is-unbracketed-and-loose]` moved
   to §4 — with `docs/OPEN_ISSUES.md` updated in the same commit, count and date corrected.
5. One commit, README §5 rule 2.
