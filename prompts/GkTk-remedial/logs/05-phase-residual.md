# Log 05 — The phase residual ρ and the leading/correction split of ω²

**Prompt:** prompts/GkTk-remedial/05-phase-residual.md
**Commit:** *(this commit)* — Add the WKB phase residual as a per-k table
**Model:** Claude Opus 5
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### `ComputeTargets/WKB_Gk.py` — the leading/correction split

Three public functions where there was one, plus one private helper. `Gk_d_ln_omegaEff_dz` is
untouched.

| symbol | returns |
|---|---|
| `_Gk_correction_terms(model, k, z) -> Tuple[float, float]` | `(B, C)` separately: `B = -eps'/(2 s)`, `C = (3 eps/2 - eps²/4 - 2)/s²` |
| `Gk_omegaEff_sq_leading(model, k, z) -> float` | `(k/H)²` |
| `Gk_omegaEff_sq_correction(model, k, z) -> float` | `B + C`; **independent of `k`** (the argument is accepted only for a shared signature) |
| `Gk_omegaEff_sq(model, k, z) -> float` | `Gk_omegaEff_sq_leading(...) + B + C` — the *same summation order* as before, so the stored `omega_WKB_sq` does not move by a bit |

### `ComputeTargets/WKB_Tk.py` — the same split

`_Tk_correction_terms`, `Tk_omegaEff_sq_leading` (`w (k/H)²`, whose square root is `k c_s/H`, the
sound-horizon leading term), `Tk_omegaEff_sq_correction` (`B + C`, `k`-independent, and in exact
radiation `-2/s²` — it does **not** vanish, unlike the Green's function's), and `Tk_omegaEff_sq`
returning `leading + B + C` in the original order. `Tk_d_ln_omegaEff_dz` is untouched.

Both modules' docstrings record the `k`-independence and why the correction must never be formed
as `omega_sq - leading` (`RECONCILIATION.md` §2 item 3).

### `ComputeTargets/phase_residual.py` (new, 200 lines)

```python
RHO_GAUSS_ORDER = 4                       # log 02's N_rho
RHO_ADAPTIVE_FALLBACK_REQUIRED = False    # log 02's flag, pinned by a test
SECTORS = ("Gk", "Tk")

def phase_residual_integrand(model, k: float, sector: str) -> Callable[[float], float]
def build_phase_residual(model, k: float, z_nodes, sector: str,
                         order: int = RHO_GAUSS_ORDER) -> CumulativeTable
```

* The integrand is `correction / (sqrt(omega_sq) + sqrt(leading))` with
  `omega_sq = leading + correction` — the rationalised form of review §6, so `omega - omega_0` is
  never formed. It raises `ValueError` naming the sector, `z`, `k`, the two parts and the
  cosmology if `omega_sq <= 0` (outside the WKB region) or if the leading term is negative
  (a `c_s² < 0` stand-in in the `Tk` sector).
* `build_phase_residual` accumulates `R(z) = ∫_z^{z_top} C/(ω+ω₀) dz'` into a `CumulativeTable`
  on the caller's nodes, splitting every panel at the cosmology's break points through
  `BackgroundModel._cosmology_break_points` (duck-typed; empty for `LambdaCDM` and the stand-ins).
  A `redshift_array` is accepted as well as a sequence of floats. The table is `CumulativeTable`'s
  usual double-double, but effectively single-limb: ρ ≤ 0.1 rad, so the low limb is ~1e-18.
* The module docstring carries the identity
  `theta(z; z_i) = -[ k tau.delta(z_i, z) + rho.delta(z_i, z) ]`, the sign convention, the
  exact-radiation check and the measured sizes and costs.

### `ComputeTargets/tests/test_omega_eff_split.py` (new, 8 tests)

Three stand-ins × three `k` × 200 log-spaced `z` across the production grid (1,800 samples per
sector): the return value is bit-identical to a verbatim copy of the pre-refactor expression;
`leading + correction` reproduces it to ≤ 1 ulp of the largest term; each correction is
`k`-independent; `Gk`'s correction is exactly `0.0` in radiation and `Tk`'s is exactly `-2/s²`.

### `ComputeTargets/tests/test_phase_residual.py` (new, 10 tests)

Radiation controls, the twelve (model, sector, `k`) cases against prompt 01's JSON references,
the review's §6/§12.2 size windows, the build cost, and three guards.

## Deviations from the prompt

### 1. `*_omegaEff_sq` keeps its summation order; the exact-equality test is split in two — STRUCTURALLY REQUIRED

**What the prompt assumed.** §1 item 1 asks that the refactored `*_omegaEff_sq` "return their sum
**bit-identically to today**", and §2 asks for a test that
`Gk_omegaEff_sq(model,k,z) == Gk_omegaEff_sq_leading(...) + Gk_omegaEff_sq_correction(...)`
**exactly**, "because `omega_WKB_sq` is a stored column".

**What is actually there.** Floating-point addition is not associative, and the two demands are
not simultaneously satisfiable. Today's code computes `(A + B) + C`; `leading + correction` is
`A + (B + C)`. Measured over 1,800 samples (three stand-ins × three `k` × 200 `z` across the
production grid):

| | exactly equal | worst difference |
|---|---|---|
| `Gk` | 1,574 / 1,800 | 1 ulp of max(\|A\|,\|B\|,\|C\|) |
| `Tk` | 1,571 / 1,800 | 1 ulp of max(\|A\|,\|B\|,\|C\|) |

so 13 % of samples differ, by one ulp. (Scored against max(\|A\|,\|B+C\|) instead, the worst is
4 ulp, at `QCDModel`, `k = 1e5`, `z = 2.06e16`: five e-folds outside the horizon, where
`3 eps/2 - eps²/4 - 2` nearly cancels and `B`, `C` are each ten times `B + C`. That is the same
one-ulp rounding measured on a smaller yardstick.)

**What was done instead.** The return value is preserved, because README §4.3 makes "an agent
proposes to change a `*_omegaEff_sq` return value" a stop condition and
`IMPLEMENTATION_STATE.md` standing note 4 says the value "must not move by a bit". So
`Gk_omegaEff_sq` is `Gk_omegaEff_sq_leading(model, k, z) + B + C` with `(B, C)` from a private
helper — the original order, evaluated from the same expressions, so the result is bit-identical.
The prompt's single test becomes two:

* `TestReturnValueUnchanged` — exact `==` against `_reference_Gk_omegaEff_sq` /
  `_reference_Tk_omegaEff_sq`, verbatim copies of the pre-refactor function bodies held in the
  test module. This is the strongest form of the claim the prompt's test was standing in for, and
  it *is* the guarantee the stored column needs.
* `TestSplitSumsToTheReturnValue` — `|(leading + correction) - *_omegaEff_sq| ≤ 2 ulp` of the
  largest of `A`, `B`, `C`, with the measured maximum printed (1.000 ulp in both sectors).

No stop was raised, because nothing about the returned value changed: this is a deviation in the
*test*, not in the code's behaviour. The alternative — defining `*_omegaEff_sq` as
`leading + correction` — would have moved the stored column by one ulp on ~13 % of evaluations
and is the thing README §4.3 forbids without asking.

### 2. The `rho_T` radiation control is scored against the exact primitive, not `1/x_i - 1/x` — STRUCTURALLY REQUIRED

**What the prompt assumed.** §2 test 1: "`Tk`: ρ_T(z;z_i) = 1/x_i − 1/x with x = k c_s τ … to
≤ 1e-13 absolute over x ∈ [24, 1e4] (review §12.4 table's last column)."

**What is actually there.** Two things.

* **Sign.** `1/x_i - 1/x` is review §12.4's convention, in which ρ is `theta - (x_i - x)`. In the
  campaign's convention (README §2 (a), (c), and the convention of `wkb_reference.py` and of the
  `rho_T` block of `wkb_reference_data.json`), the same quantity is `1/x - 1/x_i`, **negative**
  for `x > x_i`. The prompt's own §2 test 3 confirms the campaign sign, asking that ρ_T lie
  between −0.12 and −0.06. `wkb_reference.RadiationModel.rho_T`'s docstring already flags the
  difference.
* **Exact vs asymptotic.** `1/x - 1/x_i` is the *asymptote* of the exact primitive, in error by
  `2/(3x³)`: measured 2.903e-04 relative over `x ∈ [24, 1e4]`. A 1e-13 absolute agreement against
  it is not attainable by any quadrature.

**What was done instead.** The control is scored against `RadiationModel.rho_T`, the exact
closed-form primitive prompt 01 supplies (`g(s) = -2s/(√(a²-2s²)+a) + √2 asin(√2 s/a)`), at
1e-13 absolute as asked — measured **2.6888e-17 rad** (1.55e-14 relative), at `z = 2244.0`,
`k = 1e5`, 271 nodes at 100 per decade. A second test,
`test_rho_T_matches_the_review_asymptote`, records the whole-window value −4.157873557184e-02 rad
against the asymptote −4.156666666667e-02 and asserts the departure lies in (1e-5, 1e-3), which
is review §12.4's "ρ_T = 0.0416 (= 1/x_i − 1/x)" row reproduced in magnitude. Prompt 02's
convergence block scores the same way (`max_rel_error_vs_closed_form` alongside
`asymptote_relative_departure`), so this follows the campaign's established practice.

### 3. The cost bound is `order × panels`, not `order × intervals`, on QCD — STRUCTURALLY REQUIRED

**What the prompt assumed.** §2 test 4: "Integrand evaluations per table ≤ N_ρ × (intervals)".

**What is actually there.** Log 02 makes break-point subdivision mandatory on `QCD_Cosmology`
("not optional"), and each sub-panel costs its own `order` evaluations: 24 % more than
`order × intervals`, exactly as log 02's build-cost table says. The bound as written can only be
met by dropping the subdivision.

**What was done instead.** The test asserts `evaluations == order × intervals` **exactly** on
LambdaCDM (no break points) and `intervals × order < evaluations ≤ 1.30 × order × intervals` on
QCD, printing the measured ratio (1.224–1.232) and the wall time for all twelve tables.

### 4. Two private helpers `_Gk_correction_terms` / `_Tk_correction_terms` — IMPLEMENTATION CHOICE

The prompt names three public functions per module. A fourth, private, one exists so that
`*_omegaEff_sq` can sum `A + B + C` in the original order (deviation 1) without duplicating the
`B` and `C` expressions between the two entry points. The alternatives considered were (i)
duplicating the expressions in `*_omegaEff_sq` and in `*_omegaEff_sq_correction`, which invites
the two copies to drift, and (ii) making `*_omegaEff_sq_correction` return a tuple, which would
have changed the signature the prompt specifies. `test_omega_eff_split.py` imports the private
helper in one place only, to set the scale of the re-association bound.

### 5. `build_phase_residual` reads break points from `model.cosmology` — IMPLEMENTATION CHOICE

The prompt's signature is `(model, k, z_nodes, sector, order)` and says nothing about break
points, but log 02 makes the break-point scheme mandatory. The builder therefore asks
`getattr(model, "cosmology", None)` and, when there is one, calls
`BackgroundModel._cosmology_break_points(cosmology, min(z_nodes), max(z_nodes))` — the same
private, duck-typed accessor `compute_background` and `_create_functions` use for `tau`, `cs_tau`
and `friction_F`. The alternative, a `break_points=` parameter on `build_phase_residual`, would
have made every caller (prompts 06 and 07) responsible for remembering a decision log 02 calls
"not optional"; re-implementing the two-line `getattr` locally would let the two copies drift.
Importing a private name across modules inside one package is the cost.

`SECTORS`, `RHO_GAUSS_ORDER` and `RHO_ADAPTIVE_FALLBACK_REQUIRED` are module constants the prompt
did not name; `RHO_GAUSS_ORDER` is the default of `order`, and the other two exist so that log
02's decision is visible in the code and pinned by a test.

## Verification performed

Everything below was **run**, on the worktree at `680ed84` plus this commit's changes, with
`PYTHONPATH=. ./venv/bin/python`.

### `python -m unittest ComputeTargets.tests.test_omega_eff_split` — 8 tests, OK, 0.73 s

| assertion | threshold | measured |
|---|---|---|
| `Gk_omegaEff_sq` == pre-refactor expression | exact | exact on all 1,800 samples |
| `Tk_omegaEff_sq` == pre-refactor expression | exact | exact on all 1,800 samples |
| `Gk`: `leading + correction` vs the return value | ≤ 2 ulp | **1.000 ulp**, `LambdaCDMModel`, `k = 1e5`, `z = 451.17`; exact on 1,574/1,800 |
| `Tk`: `leading + correction` vs the return value | ≤ 2 ulp | **1.000 ulp**, `LambdaCDMModel`, `k = 1e5`, `z = 0.1`; exact on 1,571/1,800 |
| `Gk` correction independent of `k` | exact | exact, all models, all `z` |
| `Tk` correction independent of `k` | exact | exact, all models, all `z` |
| `Gk` correction on `RadiationModel` | `== 0.0` | exactly `0.0` at every `k`, every `z` |
| `Tk` correction on `RadiationModel` vs `-2/s²` | ≤ 4e-16 | **0.0** (bit-exact at all 200 `z`) |

### `python -m unittest ComputeTargets.tests.test_phase_residual` — 10 tests, OK, 1.69 s

**Test 1, radiation controls.** `rho_G`: every `hi` **and** every `lo` limb is exactly `0.0` at
all three `k`, and `delta(z_top, z_end) == 0.0`. `rho_T`, `k = 1e5`, `x ∈ [24, 1e4]`, 271 nodes:
max **2.6888e-17 rad** absolute (1.5527e-14 relative) at `z = 2244.0007652843915`, threshold
1e-13. Whole window −4.157873557184e-02 rad; asymptote `1/x − 1/x_i` = −4.156666666667e-02;
departure 2.9027e-04 relative; `x_i = 24` and `x = 1e4` to 9 and 6 decimal places.

**Test 2, against prompt 01's references.** `rho.delta(z_anchor, z_j)` at the JSON checkpoints,
`z_anchor` the 3-e-fold sub-horizon root (off-grid, so the top partial panel is exercised in every
one of the twelve cases). Threshold 1e-7 rad.

| model | sector | k [1/Mpc] | max abs error [rad] | at z |
|---|---|---|---|---|
| LambdaCDM | Gk | 1e5 | 3.5546e-18 | 1002.47 |
| LambdaCDM | Tk | 1e5 | 2.7756e-17 | 1.004e6 |
| LambdaCDM | Gk | 1e7 | 2.4718e-18 | 100.185 |
| LambdaCDM | Tk | 1e7 | 4.1633e-17 | 1.006e9 |
| LambdaCDM | Gk | 3e8 | 6.0772e-18 | 1.00062 |
| LambdaCDM | Tk | 3e8 | 4.1633e-17 | 1.006e9 |
| QCD | Gk | 1e5 | 4.7054e-17 | 1.005e7 |
| QCD | Tk | 1e5 | 1.3184e-16 | 1.006e9 |
| QCD | Gk | 1e7 | 2.9138e-19 | 1.007e11 |
| QCD | Tk | 1e7 | 1.9429e-16 | 1.006e9 |
| QCD | Gk | 3e8 | 2.7062e-16 | 1.006e9 |
| QCD | Tk | 3e8 | **3.6082e-16** | 1.007e11 |

**Worst 3.6082e-16 rad, `QCDModel`, `Tk`, `k = 3e8`, `z = 1.00742e11`** — nine orders inside the
1e-7 rad threshold and ten inside README §6's 1e-6 rad target. This reproduces log 02's predicted
cumulative accuracies (≤ 5.62e-18 rad LambdaCDM, ≤ 3.75e-16 rad QCD) to the digit.

**Test 3, size sanity.** Whole WKB range (3-e-fold anchor → `z = 0.1`):

| quantity | measured | window asserted | review |
|---|---|---|---|
| ρ_G, LambdaCDM, `k = 1e5` | −2.588116e-07 rad | ≤ 3e-7 | §6: −2.5e-7 |
| ρ_G, QCD, `k = 3e8` | −1.180206e-03 rad | 5e-4 … 3e-3 | §6: −1.26e-3 |
| ρ_T, LambdaCDM, 1e5 / 1e7 / 3e8 | −8.634258e-02 / −8.634111e-02 / −8.634109e-02 | −0.12 … −0.06 | §12.2: −0.0863 |
| ρ_T, QCD, 1e5 / 1e7 / 3e8 | −8.867182e-02 / −8.902549e-02 / −9.310315e-02 | −0.12 … −0.06 | §12.2: −0.0887 / −0.0979 (1e8) / −0.0933 |

Every figure agrees with `convergence.decision.{lambdacdm,qcd}_rho_magnitude_rad` in the JSON to
all printed digits.

**Test 4, cost.**

| model | k | nodes | evaluations | × order·intervals | seconds (Gk / Tk) |
|---|---|---|---|---|---|
| LambdaCDM | 1e5 | 1037 | 4,144 | 1.000 | 0.017 / 0.018 |
| LambdaCDM | 1e7 | 1236 | 4,940 | 1.000 | 0.022 / 0.024 |
| LambdaCDM | 3e8 | 1384 | 5,532 | 1.000 | 0.026 / 0.028 |
| QCD | 1e5 | 1040 | 5,088 | 1.224 | 0.082 / 0.160 |
| QCD | 1e7 | 1242 | 6,100 | 1.229 | 0.108 / 0.224 |
| QCD | 3e8 | 1401 | 6,900 | 1.232 | 0.125 / 0.255 |

Log 02 predicted 5,536 / 6,904 at `k = 3e8`; the two-evaluation difference is the node count at
the anchor. Against the 2.5e6 RHS evaluations and 63.7 s per object of the phase ODE (review §4),
a residual table is 5.5k–6.9k evaluations and 17–255 ms, built once per `(model, k, sector)`.

**Guards.** `phase_residual_integrand(..., "Tk")` raises `ValueError` at `x_T = 1`
(`omega_T² < 0`); `"Qk"` raises in both entry points; `RHO_GAUSS_ORDER` and
`RHO_ADAPTIVE_FALLBACK_REQUIRED` are asserted equal to the JSON's
`convergence.decision.N_rho` and `.rho_adaptive_fallback_required`.

### `python -m unittest discover -s ComputeTargets/tests -t .` — **Ran 178 tests, OK**, 121.2 s

The whole package suite passes (prompt §3). No pre-existing test was changed.

### `black`

`./venv/bin/python -m black --check` on the five touched/added files: clean. (`black --check`
over the whole tree reports 54 files it would reformat, all of them pre-existing and none of them
touched here — chiefly `docs/spec-code-audit/scripts/`.)

### Reasoned, not run

Nothing. Every number above was measured.

## Observations not acted on

1. **No break point of `QCD_Cosmology` falls between a table's top node and its anchor.** The
   builder declares break points over the node range only, matching `compute_background` and
   `_create_functions`, so a break inside the off-grid anchor partial would go undeclared. It was
   checked explicitly: for all three production `k`, zero of the 407 break points lie in
   `(u(z_top node), u(z_anchor))` (233, 284 and 325 lie inside the declared range). Not a defect
   today, and the convention is prompt 03's, not this prompt's; recorded so that a later reader
   who changes the anchor knows to re-check.

2. **The low limb of a residual table is dead weight.** Prompt 05 §1 calls the table
   "single-limb (`lo` zero)". `CumulativeTable` has no single-limb mode, so the low limb is
   computed (it is ~1e-18) and carried. Adding a mode to `CumulativeTable` would have meant
   editing prompt 03's file, which is not in this prompt's file list. The cost is the
   `O(n²)` `fsum` of `_accumulate` — ~10 ms per table at n = 1400, inside the measured build
   times above — and nothing else: `delta` differences both limbs regardless. If prompt 06 or 07
   finds the per-`k` build cost matters on QCD, a single-limb path in `CumulativeTable` is where
   to look first.

3. **`Gk_omegaEff_sq_correction` and `Tk_omegaEff_sq_correction` take a `k` they never use.**
   That is the prompt's signature, and it is what lets `_SECTOR_FUNCTIONS` hold uniform pairs. It
   also means a caller cannot be warned if it passes the wrong `k`. Documented in both docstrings
   rather than changed.

Nothing here is actionable enough to open a §3 issue, and §3/§4 of the board are therefore
unchanged (so `docs/OPEN_ISSUES.md` is unchanged too).

## State handed to the next prompt

**The six new function names and signatures** (all in `ComputeTargets/`):

```python
# WKB_Gk.py
Gk_omegaEff_sq_leading(model, k: float, z: float) -> float        # (k/H)^2
Gk_omegaEff_sq_correction(model, k: float, z: float) -> float     # B + C; k-independent
Gk_omegaEff_sq(model, k: float, z: float) -> float                # unchanged, bit for bit

# WKB_Tk.py
Tk_omegaEff_sq_leading(model, k: float, z: float) -> float        # w (k/H)^2; sqrt = k c_s/H
Tk_omegaEff_sq_correction(model, k: float, z: float) -> float     # B + C; k-independent
Tk_omegaEff_sq(model, k: float, z: float) -> float                # unchanged, bit for bit
```

`_Gk_correction_terms` / `_Tk_correction_terms` return `(B, C)` and are private; do not use them
outside the two modules and their split test.

**`*_omegaEff_sq` did not move by a bit** — `TestReturnValueUnchanged` asserts exact equality
against verbatim copies of the pre-refactor bodies. But **`leading + correction` is not bit-equal
to `*_omegaEff_sq`**: it differs by one ulp on ~13 % of the production range (deviation 1). Any
prompt that needs `omega_WKB_sq` for the stored column must call `*_omegaEff_sq`; any prompt that
needs the residual must call the two parts. Do not "simplify" `*_omegaEff_sq` to
`leading + correction`.

**`build_phase_residual`'s signature and the sign identity:**

```python
from ComputeTargets.phase_residual import build_phase_residual, RHO_GAUSS_ORDER  # = 4

rho = build_phase_residual(model, k, z_nodes, sector, order=RHO_GAUSS_ORDER)
#   sector in ("Gk", "Tk");  z_nodes strictly descending (a redshift_array is accepted);
#   returns a ComputeTargets.cumulative_table.CumulativeTable

theta(z; z_i) == -( k * tau.delta(z_i, z)    + rho.delta(z_i, z) )      # sector "Gk"
theta_T(z; z_i) == -( k * cs_tau.delta(z_i, z) + rho.delta(z_i, z) )    # sector "Tk"
```

with `rho.delta(z_i, z) = ∫_z^{z_i} C/(ω+ω₀) dz`, **negative** for `z < z_i` on both sectors of
both production models. No sign flip anywhere: `tau`/`cs_tau` (log 03, log 04) and `rho` all use
`X.delta(z_a, z_b) = X(z_b) - X(z_a)`. Exact-radiation check: ρ_G ≡ 0 bit-exactly, so
`theta_G = -k tau.delta(z_i, z) = k(1/s_i - 1/s)` with `tau = 1/(H₀ s)`.

**Which nodes to build over.** The caller chooses. `z_nodes` must lie inside the WKB region —
the integrand raises `ValueError` if `ω² ≤ 0`, which it is above the horizon-crossing region for
the smaller `k`. The per-object anchor (`z_init`, a `root_scalar` root) is *not* a node, and is
reached through `delta`'s off-grid partial; it must lie within one grid interval of the top node
or `CumulativeTable` refuses. In the tests, the nodes are the production source grid filtered to
`z <= z_anchor`, which puts the anchor 0.1–1.5 % of a decade above the top node.

**Measured accuracy and cost** (production grid, order 4, anchored at the 3-e-fold root):

* **Accuracy vs prompt 01's references: worst 3.61e-16 rad** over all twelve (model, sector, `k`)
  cases, at `QCDModel`, `Tk`, `k = 3e8`, `z = 1.007e11`. LambdaCDM is ≤ 4.2e-17 rad, QCD
  ≤ 3.6e-16 rad. README §6's ρ row (≤ 1e-6 rad) is met with ten orders of margin; the physical
  floor is the LG truncation, 1e-3 rad on QCD.
* **Radiation control:** ρ_T to 2.69e-17 rad absolute against the exact primitive over
  `x ∈ [24, 1e4]`; ρ_G bit-exactly zero.
* **Cost:** 4,144–5,532 integrand evaluations and 17–28 ms per table on LambdaCDM (exactly
  `4 × intervals`); 5,088–6,900 evaluations and 82–255 ms on `QCD_Cosmology` (1.22–1.23 ×
  `4 × intervals`, the break-point subdivision). One table per `(model, k, sector)`.

**Sizes prompts 06 and 07 should expect** (whole range, anchor → `z = 0.1`): ρ_G = −2.59e-07
(LambdaCDM 1e5), −3.43e-09 (1e7), −1.35e-10 (3e8); −5.42e-04, +3.00e-05, −1.18e-03 (QCD).
ρ_T = −0.08634 on LambdaCDM at every `k`; −0.08867, −0.08903, −0.09310 on QCD. **ρ_T is not
negligible and must be carried** (review §12.2); ρ_G is negligible on LambdaCDM and is carried
anyway, because one code path is better than two and QCD needs it.

**The fallback was not implemented.** Log 02 set `rho_adaptive_fallback_required = False`, and
`phase_residual.RHO_ADAPTIVE_FALLBACK_REQUIRED = False` records it; a test asserts it agrees with
the JSON's `convergence.decision.rho_adaptive_fallback_required`. Break-point subdivision is what
makes order 4 sufficient, and it is applied automatically by `build_phase_residual`.
