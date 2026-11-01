# levin-phase-floor 06h — the Bessel group phase's rounding

> **§1–§3 only** (the code, the tests and these records), written by the executing agent on the
> laptop. §4 (rehearsals R21 and R22) and §5 (the census and the phase re-measurement) run later on
> the Mac Studio; the orchestrator appends their record to this log.

**Subject:** prompt 06h §1–§3, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `2c325f5`; the code there is 06e's (`f4d8e17`). `git log --oneline 2c325f5..HEAD`
  was empty at the start and before staging. The stashes were not touched, and
  `var/levin-phase-floor-07/` was not read.
- **The gate** held: the 06f row ✅, `git status --short` empty, `runregistrykit list` nothing
  `running` or stale (61 runs: 59 finished, 2 `unknown`, pre-registry). The seven roots at
  `2c325f5` are the orchestrator's (Verification).
- No Ray was started, no run was registered, no store was opened, and nothing was written under
  `var/`.

**Commit:** the code commit `7172a8f`, and the follow-up `ac1c6f4` on the user's ruling on
deviation 10; both reviewed by the orchestrator and pushed 2026-10-31.

**Result:** `BesselPhaseGroup.sin_cos` carries the rounding of the product $\hat K\hat x$
(TwoProduct) and of $K$ itself ($K_{\rm lo}$) exactly in its small angle, and `theta_abserr`
declares the rounding of the abscissa, $\varepsilon_{\exp}|d\theta/du|$, and the group's own
evaluation $\phi_{\rm eval}$. Tests 1–7 hold on ten cells; 06h's worst error is at most 0.499 of
its declaration on every cell, where the parent's reached $4.4\times10^{10}$ of its own. The three
breakages are caught. The 20 seeded LiouvilleGreen draws pass under both codes, with no Levin call's
convergence changed and none at depth 20. One existing assertion changed (the definition test). The
two product-floor bounds measure the raw `theta`, which the declaration does not cover; on the
user's ruling (deviation 10) they keep that check with its product term, and `ac1c6f4` adds the
declaration's own check on `theta_mod_2pi` beside it. Two issues opened.

## What shipped

**$K_{\rm lo}$** (`BesselPhaseGroup.__init__`): `K_lo = math.fsum(signed coefficients + [-K])`, i.e.
$K - \hat K$ correctly rounded, in error by at most $u|K_{\rm lo}|$. Zero whenever $k + e_\nu q +
e_\sigma s$ is representable, which is so on every production cell checked (06f's two calls) and on
every test cell except the one built for it.

**TwoProduct** (`_two_product_error(a, b, product)`): `math.fma(a, b, -product)` on a 0-d argument,
Dekker's algorithm with Veltkamp's split (`_two_product_split`, splitter $2^{27}+1$) on an array.
Both are exact barring overflow and underflow, so they agree bit for bit; test 6 checks exactness in
`Fraction` arithmetic at every point of every cell and the scalar/array agreement.

**The small angle** (`sin_cos`): $C_{\rm red} + e_1r_1 + e_2r_2 + e_3r_3 + p_{\rm lo} +
\mathrm{fl}(K_{\rm lo}\hat x)$, added left to right; the big angle is ${\rm fl}(\hat K\hat x)$,
unreduced, as before. `theta_mod_2pi` is unchanged code and inherits it. The residuals are added in
`sin_cos` itself rather than through `self.residual(x)`, which uses the builtin `sum()` (deviation 7).

**The declaration** (`theta_abserr`, which sums the three parts `_declared_error_parts` returns):
$$\texttt{theta\_abserr}(u) = \underbrace{\textstyle\sum_i \texttt{theta\_abserr\_at}_i(m_i\hat x)}_{\rm constituents}
+ \underbrace{\varepsilon_{\exp}\big[(|\hat K| + |K_{\rm lo}|)\hat x + 2\textstyle\sum_i|r'_i|\big]}_{\rm abscissa}
+ \phi_{\rm eval},$$
with $r'_i = dr_i/d\log y$ at $m_i\hat x$, every sum left to right (`_ordered_sum`), and

| term of $\phi_{\rm eval}$ | value | basis |
|---|---|---|
| the small angle's sum | $\gamma_5\,(|C_{\rm red}| + \sum_i|r_i| + u|\hat K|\hat x + |K_{\rm lo}|\hat x)$ | six summands, five additions (Higham §4.2); $|p_{\rm lo}| \le u|\hat K\hat x|$ |
| $C_{\rm red}$'s own fsum | $u\,|C_{\rm red}|$ | one correctly rounded sum, declared by no constituent |
| $K_{\rm lo}$ | $2u\,|K_{\rm lo}|\hat x$ | the fsum that forms it, and the product |
| the arguments $m_i\hat x$ | $u\cdot 2\sum_i|r'_i|$ | a relative $u$ in $y_i$ moves $r_i$ by $u|r'_i|$; allowance 2 |
| the angle addition | $12u$ | $4\varepsilon_t + 2\sqrt2u$ with $\varepsilon_t = 2u$ (two ulps of sin/cos) $= 10.83u$, rounded up |
| atan2 in `theta_mod_2pi` | $8u$ | two ulps of a value in $(-\pi,\pi]$, ulp $\le 4u$ |

Each is a named constant or expression in `three_bessel_integrals.py` with its derivation in a
comment. The roundings of ${\rm fl}(\hat K\hat x)$ and of $K$ are not declared: `sin_cos` carries
both exactly. The constituents' `c_nu_reduced` roundings are not declared again: their
`EVALUATION_FLOOR` covers them (deviation 3).

**$\varepsilon_{\exp}$ = `EXP_RELERR` = $2.23\times10^{-16}$ (2.009 u), measured.** The platform
documents no accuracy for `exp` (macOS's exp(3) page states none; numpy 2.2.4 on aarch64 calls the
platform's exp for float64). On $10^6$ seeded points in each of $u\in[-30,5]$ (production's
$\log\eta$; 06f's calls span $[-17.38,-1.79]$) and $u\in[-2,\log10^{12}]$, against mpmath at 30
digits, the worst relative error is $1.1146\times10^{-16}$ = 1.004 u (0.512 ulp of the result), at
$u = 0.696347627688485$. The scalar path (`np.exp` on a Python float or an `np.float64`) and the
array path are bit-identical to each other and to `math.exp` at all $2\times10^6$ points. Taken at
`EXP_RELERR_MULTIPLE` = 2 times the worst, rounded up; that is at least the $2u$ of any faithful exp.
Script `docs/levin-phase-floor/06h/calibration/scripts/measure_exp_relerr.py`, output
`out/exp_relerr.txt`; re-measured on the same sample by test 5.

**The same script measured sin, cos and atan2** (descriptive, the basis for $\varepsilon_t$): on
$10^5$ seeded arguments each, `math.sin`/`np.sin` worst 0.840 ulp (large arguments, $10^{-2}$ to
$10^{13}$) and 0.836 ulp (small, $|a|\le20$), cos 0.847 and 0.830, atan2 0.722 ulp; math and numpy
bit-identical. They are faithful, **not correctly rounded**, so two ulps are assumed rather than one.

**Docstrings and comments (§0.2 item 3):** the class docstring's product-floor paragraph rewritten
(the floor is now carried in the small angle; what remains is the abscissa and the evaluation; `theta`
still rounds); `sin_cos` and `theta_abserr`, each stating what it contains and why; `_coefficient_sum`'s
note on per-point sums; `BesselIntegralResult`'s "What it includes"; and one sentence in
`bessel_phase.py`'s "Accuracy of the supplied argument" pointing to the group's declaration of the
abscissa term. `bessel_phase.py`'s code is unchanged. **`ComputeTargets/QuadSourceIntegral.py` is
unchanged:** its comment above the four groups describes the group construction and the measurement
of prompt `qsi-phase-groups` 01, which stay true; nothing there describes the declared error in a way
06h makes wrong.

**Unchanged:** `theta`, `theta_deriv`, `residual`, `arguments`, `levin_theta`'s keys; every constituent;
`AdaptiveLevin/`; `ComputeTargets/` code; `ComputeTargets/phase_groups.py`.

**New test module** `LiouvilleGreen/tests/test_group_phase_rounding.py` (8 tests, about 18 s, of
which test 5 is about 17 s).

## Predictions and verdicts

06h §2 carries no prediction of its own. The following are §4's and §5's, copied before they run.
**Not yet scored** — each is scored on the Mac Studio, in 06h's records commit.

§4, the value comparison:

   - **H0 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
     `total_abserr` in both stores has $|{\rm total}_{22} - {\rm total}_{21}| \le {\rm
     total\_abserr}_{21} + {\rm total\_abserr}_{22}$. Rows with an infinite `total_abserr` in either
     are counted separately, and the stationary-point rows are tabulated, not scored. **Holds** if
     no row violates it. A violation is a finding for the user, not by itself a stop.
     — **not yet scored**

§5, the census and the phase re-measurement:

**Gate H, before scoring:**
- **The parent reproduces 06e.** `git diff --stat 6d81fad <code commit's
  parent> -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` and files whose ASTs without
  docstrings equal the dispatch commit's. At the parent, the harness reproduces 06e's units **bit
  for bit** on `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on `0-1971` at `1e-8`.
- **The new code is deterministic.** Two runs of one $h$ unit are bit-identical.

— **not yet scored**

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **H1 — the declaration is honest.** **Rule:** on every one of the 20 M2 calls, at every point
  sampled, $|\delta\theta(u)| \le \texttt{theta\_abserr}_h(u)$, the reference at the exact $e^u$.
  **Holds** if no point violates it. The parent's declaration is scored on the same points beside it.
  — **not yet scored**
- **H2 — the declared term covers the response.** **Rule:** on every returned leaf of the 20 calls,
  the first-order response to the measured phase error is at most that leaf's declared term.
  **Holds** if no leaf violates it. — **not yet scored**
- **H3 — the rows agree.** **Rule:** outside the stationary-point rows, every row with finite
  `total_abserr` under both has $|{\rm total}_h - {\rm total}_{06e}| \le {\rm total\_abserr}_{06e} +
  {\rm total\_abserr}_h$. **Holds** if true at both rtols. — **not yet scored**
- **H4 — no convergence is lost.** **Rule:** outside the stationary-point rows, no Levin call that
  06e reported converged is unconverged under $h$. **Holds** if true at both rtols. A failure is a
  finding for the user: §0.2's estimate of what the declaration costs the driver is wrong.
  — **not yet scored**

## Deviations from the prompt

1. **$\varepsilon_{\exp}$ measured, with its ranges and multiple:** $u\in[-30,5]$ and
   $[-2,\log10^{12}]$, $10^6$ points each, multiple 2, rounded up to $2.23\times10^{-16}$. §0.2
   allows measurement when the accuracy is undocumented; the ranges and the rounding are mine.
   **IMPLEMENTATION CHOICE.**
2. **sin, cos and atan2 taken at two ulps, on a measurement the prompt did not ask for.** The angle
   addition's and atan2's bounds need an accuracy for the library's trigonometric functions, which is
   not documented; `bessel_phase.py` says "correctly rounded", but the measurement in
   `measure_exp_relerr.py` finds up to 0.85 ulp. One ulp would leave a margin of 1.2. **IMPLEMENTATION
   CHOICE.**
3. **The angle addition is declared, not credited to the constituents' `EVALUATION_FLOOR`.** §0.2 asks
   for it "counted against what each constituent's `EVALUATION_FLOOR` already covers". That floor is
   each constituent's allowance for its own angle addition in `sin_cos_theta` (and its
   `c_nu_reduced`'s rounding); the group never calls `sin_cos_theta`, so it inherits the
   `c_nu_reduced` roundings (left to the constituents, not declared twice) and performs one angle
   addition of its own on different arguments, which no constituent declares and which is declared
   here (not omitted). The other reading — the two `EVALUATION_FLOOR`s of angle additions the group
   does not perform pay for the one it does — would rest on slack, not on a rounding declared. Cost:
   $12u = 1.3\times10^{-15}$. **IMPLEMENTATION CHOICE.**
4. **atan2 declared** ($8u$). §0.2's list for $\phi_{\rm eval}$ names the small sum, the angle addition
   and the arguments; the driver integrates `theta_mod_2pi`, so atan2's rounding is in the phase it
   is given. **IMPLEMENTATION CHOICE.**
5. **The small sum has six summands and five additions** ($\gamma_5$), not the "four or five terms" of
   §0.2: $R$ is three residuals. Two further terms: $u|C_{\rm red}|$ (C_red's fsum) and
   $2u|K_{\rm lo}|\hat x$ ($K_{\rm lo}$'s fsum and product). **IMPLEMENTATION CHOICE.**
6. **$|d\theta/du|$'s allowance:** a factor 2 on $\sum_i|r'_i|$ (in the near region $r'$ is an
   interpolant's derivative, its relative error undeclared); none on $(|\hat K|+|K_{\rm lo}|)\hat x$,
   whose rounding is relative $u$ of a term already multiplied by $\varepsilon_{\exp}$. The same
   factor 2 is used in the argument term. **IMPLEMENTATION CHOICE.**
7. **Sums left to right, not `sum()`.** Python 3.12+ `sum()` compensates (Neumaier) a sum of exact
   Python floats but not of arrays or `np.float64`, so its scalar and array results can differ in the
   last bit. The parent's `theta_abserr` did: at 2–18 of about 50 points per cell (test 6 prints it).
   §0.2 requires `theta_abserr`'s and `sin_cos`'s scalar and array results bit-identical, so both
   now add left to right (`_ordered_sum`; an explicit loop in `sin_cos`), and `sin_cos` forms $R$
   itself instead of calling `residual`, which §0.3 says does not change. **STRUCTURALLY REQUIRED.**
8. **A private helper `_declared_error_parts`** returns the declaration's three parts; tests 1–4
   score against them. §0.3 allows private helpers for $\phi_{\rm eval}$. **IMPLEMENTATION CHOICE.**
9. **The definition test** (`test_three_bessel.py`, `test_every_levin_call_receives_a_declared_phase_error`)
   asserts the three parts term by term, each rebuilt in the test from the module's constants, to a
   relative $10^{-14}$, and their sum. One of §2 test 9's two allowed kinds. Its other assertions are
   unchanged. **IMPLEMENTATION CHOICE** within the allowance.
10. **The two product-floor bounds were not tightened.** §2 test 9 allows `theta_abserr + 2 EPS |K| x`
    to become `theta_abserr` alone, "since the declaration now carries what the extra term stood
    for". Both tests measure **`group.theta(log_x)`, the raw value**, which 06h does not change and
    which still rounds $\hat K\hat x$ into one double: its error is up to about $3u|\hat K\hat x|$, and
    the declaration does not declare it (the product rounding is removed from `sin_cos`, not declared).
    Tightened, both pass — but only because $\varepsilon_{\exp}|\hat K|\hat x \approx 2u|\hat K\hat x|$
    happens to exceed the raw value's rounding at the grids' points: worst error/bound **0.9646**
    (`K/max ~ 1e-6`) and **0.9087** (`generic`) in `test_three_bessel.py`, and **0.9739** (`generic`)
    in `test_quadsource_integral.py`. That is not a bound on the quantity measured, so the assertions
    are left as written (worst error/bound under them: parent 0.4812, 0.4845, 0.4812, 0.4594 and QSI
    0.4877, 0.4895; 06h 0.442, 0.442, 0.442, 0.3873 and QSI 0.4677, 0.3752). The allowance is a
    permission, and declining it is the smaller change; but it departs from the prompt's "become".
    The orchestrator may prefer to tighten them, or to have them measure `theta_mod_2pi`, which would
    be a change of a kind the prompt does not allow. **IMPLEMENTATION CHOICE** — for the orchestrator.

    **Ruled 2026-10-31, after the orchestrator's review of `7172a8f`.** The orchestrator recommended
    separating the two questions the old tests mixed: (1) whether the $Kx + C + R$ restructure beats
    the legacy three-raw-phase route, which can only be compared on the raw single-double `theta`
    and whose bound rightly keeps its one rounding of $Kx$; and (2) whether `theta_abserr` is an
    honest declaration, which must be tested on `theta_mod_2pi`, the phase it declares. The user:
    *"That recommendation sounds OK. Let's do that."* — and, on how to weigh it: *"I really don't
    like these narrow legalistic readings of what prompts 'allow'. We are trying to produce a code
    that does the right thing for the right reason. ... If the prompt doesn't allow the right
    answer, the prompt is wrong."* Done in a follow-up commit on top of `7172a8f`:
    - in both tests the raw-`theta` assertion, its `+ 2 EPS |K| x` term and the before/after
      measurement are kept, and described as a check on the plain-double phase, which still rounds
      $Kx$ and which the driver never evaluates when `theta_mod_2pi` and `theta_deriv` are supplied;
    - at every point both tests already sample, a new assertion
      $|\texttt{theta\_mod\_2pi} - \Theta| \le \texttt{theta\_abserr}$ alone, as angles modulo
      $2\pi$ against the unrounded mpmath reference (each reference helper gains `as_mpf`: a double
      holding a phase of $10^{11}$ rad is itself off by up to $7.6\times10^{-6}$). The reference is at
      the evaluated $\hat x$, so the abscissa term is slack there and the constituents and
      $\phi_{\rm eval}$ are what is exercised; a comment says so;
    - in `test_the_bounded_angle_is_the_angle_of_the_group_not_a_sum_of_angles`, the generic case's
      `worst_new <= worst_old` is replaced by a per-point bound on the new route's (sin, cos) error,
      derived from the declaration: $|s - \sin\Theta| \le |\rho - 1| + |\phi - \Theta|$ with
      $|\phi - \Theta| \le$ constituents + $\phi_{\rm eval}$, $|\rho - 1| \le 12u$ (the angle
      addition's radial error, derived as its tangential one is) and $u$ for rounding the reference's
      sin and cos. It is asserted in every case; the cancelling cases keep their gain assertions;
    - the docstrings and comments that called the declared bound "the assertion that matters" or
      the $Kx$ rounding one "no double-valued phase of that size can avoid" are corrected, and
      `test_the_four_groups_reconstruct_the_triple_product`'s docstring, which attributed its
      $4\,\varepsilon\max_i|K_i|x$ bound to the groups' product floor, now attributes it to the
      direct side's rounded arguments ${\rm fl}(m_i x)$ (its bound unchanged).
    Numbers are under Verification, test 9. The deviation is resolved.
11. **The test cells add one:** "three non-zero residuals", orders $(5/2, 7/2, 3/2)$ (J231's) at
    $(1.3, 1.7, 2.1)$, signs $(+,-,-)$, since every listed cell has an order-1/2 factor whose residual
    is identically zero, leaving two terms in $R$ (where `sum()` and a plain sum agree). The
    $\hat K\ne K$ cell is $(2.0, 0.3, 0.1)$: $\hat K = 1.6$, $K_{\rm lo} = -8.33\times10^{-17}$.
    **IMPLEMENTATION CHOICE.**
12. **06f's cells rebuilt from their records.** 06f's capture pickles were removed (the board's
    Decisions, **06f results**), so the cells are built from `result.json`'s coefficients, signs,
    orders and spans, with production's `phase_atol = amplitude_rtol = 1e-12` (`main.py`) and
    `max_x` = $1.1\max_i m_i\,e^{u_{\rm hi}}$. The reconstruction is checked by an extra test,
    `test_the_06f_cells_reproduce_06f`: $\hat K$ bit for bit, the scalar declarations
    ($8.88\times10^{-16}$, $5.000000000000008\times10^{-13}$), and the parent's `theta_mod_2pi` at
    each call's three worst M2 nodes **bit for bit** against what 06f recorded. Each cell also
    evaluates those three nodes. **STRUCTURALLY REQUIRED** (the pickles are gone); the extra test is
    an **IMPLEMENTATION CHOICE.**
13. **Tests 1–3 use 48 seeded points per cell plus the span's two ends** (and the three nodes on
    06f's cells), references at 30 digits; 50 digits only in test 7's check of the reference.
    Test 1 scores `theta_mod_2pi` as well as the pair's angle. **IMPLEMENTATION CHOICE.**
14. **Test 7's documented errors:** $\nu = 1/2$ at $x = 10^3$, at most $1.11\times10^{-16}$
    (`EVALUATION_FLOOR`'s comment); $\nu = 3/2$ at $10^3$, at most $1.1\times10^{-16}$ (`sin_cos_theta`'s
    docstring); $\nu = 5/2$ at $10^7$, at most the $8.9\times10^{-16}$ that `theta_abserr_at`'s
    docstring says the tail declares there (a documented declaration, not a measured error, the only
    statement the module makes for 5/2 at a point). **IMPLEMENTATION CHOICE.**
15. **The draws harness swaps the module, not the driver:** in mode "parent" the dispatch commit's
    `three_bessel_integrals.py` is put in `sys.modules` before `test_3bessel_analytic` imports from it;
    the driver is the working tree's in both modes. It also records each three-Bessel result for the
    per-oracle table. 06e's DEBUG capture of the driver is dropped (the driver does not change).
    **STRUCTURALLY REQUIRED** (what is swapped); the rest **IMPLEMENTATION CHOICE.**
16. **The breakages are applied by a script,** `calibration/scripts/breakages.py`, as 06e's were,
    with output `out/breakages.txt`. **IMPLEMENTATION CHOICE.**
17. **The first draws run was discarded.** I ran `breakages.py`, which rewrites
    `three_bessel_integrals.py` for about 70 s, while the first draws run was still going, and its
    workers import that module at each unit's start; so some 06h units may have run a broken module.
    It was stopped at 128 of 160 units, nothing of it was kept, and the run was repeated from the
    start with the module unchanged throughout (its SHA-256 `fea67d56…` checked before and after).
    That second run's header then carried the harness's SHA-256 from before I edited its report
    functions (`8d2bd6dd…`, the run code unchanged); rather than leave a mismatched stamp, the run was
    repeated a third time on the committed tree (clean, harness `34190dc8…`), with every count and
    every number of the report identical. The recorded run is that one. **UNINTENDED DRIFT**,
    corrected.
18. **The two issues opened** (Observations 2 and 3) are on the board and in `docs/OPEN_ISSUES.md`,
    as `CLAUDE.md` asks of an observation not acted on. **IMPLEMENTATION CHOICE.**

## Verification performed

**The seven roots,** each run on its own with `CLAUDE.md`'s commands, one at a time:

| root | at `2c325f5` (orchestrator) | after 06h |
|---|---|---|
| AdaptiveLevin | 68 | 68, OK |
| ComputeTargets | 780 | 780, OK (249 s) |
| CosmologyModels | 39 | 39, OK |
| LiouvilleGreen | 148 (1 skipped) | **156** (1 skipped), OK — +8, the new module's tests |
| ObjectFactories | 280 | 280, OK |
| RunDrivers | 127 | 127, OK |
| tools | 58 | 58, OK |

No count fell; the one rise equals the tests added. Neither known wall-clock flake fired.

**The new module, `test_group_phase_rounding.py`** (ten cells; parent = the dispatch commit's module):

- **06f's cells reproduce 06f** (`test_the_06f_cells_reproduce_06f`): $\hat K$, $K_{\rm lo} = 0$, the
  scalar declarations, and the parent's `theta_mod_2pi` at the six worst M2 nodes bit for bit.
- **Test 1 — the product's rounding is gone** (reference at the float $\hat x$; bound constituents +
  $\phi_{\rm eval}$, no $u|\hat K\hat x|$ term). Worst error, 06h (fraction of the bound) | parent
  (multiple of its declaration):

  | cell | 06h | parent |
  |---|---|---|
  | GROUP_CASES $K=0$ | $1.97\times10^{-12}$ (0.499) | $1.97\times10^{-12}$ (0.50) |
  | GROUP_CASES $K/\max\sim10^{-6}$ | $8.68\times10^{-13}$ (0.497) | $8.69\times10^{-13}$ (0.499) |
  | GROUP_CASES $K/\max\sim10^{-10}$ | $5.85\times10^{-14}$ (0.132) | $5.94\times10^{-14}$ (0.235) |
  | GROUP_CASES generic | $5.45\times10^{-13}$ (0.497) | $6.42\times10^{-6}$ ($2.41\times10^{9}$) |
  | $\hat K\ne K$ | $1.76\times10^{-15}$ (0.097) | $9.21\times10^{-5}$ ($3.46\times10^{10}$) |
  | three non-zero residuals | $4.99\times10^{-13}$ (0.086) | $3.05\times10^{-5}$ ($1.15\times10^{10}$) |
  | QSI near-resonant | $1.24\times10^{-12}$ (0.498) | $1.24\times10^{-12}$ (0.499) |
  | QSI generic | $1.83\times10^{-15}$ (0.136) | $7.51\times10^{-9}$ ($2.82\times10^{6}$) |
  | 06f #19 (J4) | $7.44\times10^{-14}$ (0.469) | $4.34\times10^{-13}$ (163) |
  | 06f #20 (Y4) | $3.80\times10^{-13}$ (0.493) | $3.81\times10^{-13}$ (1.81) |

  **Holds** on every cell.
- **Test 2 — $K$'s rounding is gone** ($\hat K = 1.6$, $K_{\rm lo} = -8.33\times10^{-17}$, checked
  against exact rational arithmetic): 06h's worst $1.71\times10^{-15}$; the parent's
  $9.21\times10^{-5}$ at $x = 10^{12}$, where $|K_{\rm lo}|x = 8.33\times10^{-5}$ and $u|\hat K x| =
  1.78\times10^{-4}$. **Holds.**
- **Test 3 — the declaration bounds the phase error as a function of u** (reference at the exact
  $e^u$). Worst $|\delta\theta|/\texttt{theta\_abserr}$, 06h | parent (points violating):
  $K=0$ 0.499 | 0.500 (0/50); $10^{-6}$ 0.497 | $1.03\times10^{4}$ (20/50); $10^{-10}$ 0.158 | 1.47
  (2/50); generic 0.496 | $3.49\times10^{9}$ (35/50); $\hat K\ne K$ 0.335 | $1.73\times10^{10}$
  (44/50); three residuals 0.354 | $4.38\times10^{10}$ (43/50); QSI near-resonant 0.498 | 0.499
  (0/50); QSI generic 0.378 | $6.03\times10^{6}$ (40/50); **06f #19 0.452 | 277 (29/53)**; **06f #20
  0.488 | 3.14 (5/53)**. **Holds** on every cell; the parent fails on eight of ten, both 06f cells
  among them.
- **Test 4 — not loose by orders where the new terms dominate** (descriptive), measured/declared at
  the exact $e^u$ by dominant part: constituents 146 points, median $5.2\times10^{-4}$, max 0.499;
  **abscissa 229 points, median 0.154, 90 % 0.316, max 0.470, min $2.5\times10^{-3}$**; $\phi_{\rm eval}$
  131 points, median 0.035, 90 % 0.080, max 0.131. Where the abscissa term dominates the
  declaration is within a factor of about 6 at the median, as $\varepsilon_{\exp} = 2\times$ the worst
  exp error makes it.
- **Test 5 — $\varepsilon_{\exp}$'s basis:** on the $2\times10^6$-point sample (SHA-256 of each range
  checked), the worst relative error is $1.114619\times10^{-16}$ = 1.004 u $\le$
  `EXP_RELERR`/2 = $1.115\times10^{-16}$; scalar and array paths bit-identical. **Holds.**
- **Test 6 — scalar and array agree bit for bit:** `sin_cos` (both components), `theta_mod_2pi` and
  `theta_abserr` at every point of every cell; TwoProduct exact (`Fraction`) at every point, its two
  forms identical; `_detect_vectorized` true for `theta_mod_2pi`, `theta_deriv` and `theta_abserr`.
  **Holds.** (The parent's `theta_abserr` differed at 2–18 points per cell; its `theta_mod_2pi` at
  none.)
- **Test 7 — the method check:** phase-pair error $5.94\times10^{-17}$ ($\nu=1/2$, $x=10^3$,
  documented $1.11\times10^{-16}$), $8.41\times10^{-17}$ ($\nu=3/2$, $10^3$, documented
  $1.1\times10^{-16}$), $1.48\times10^{-16}$ ($\nu=5/2$, $10^7$, declared $8.9\times10^{-16}$);
  $|\mathrm{atan2}(J,-Y) - (\arg H^{(1)} + \pi/2)| \le 2.0\times10^{-31}$; 30 against 50 digits
  $\le 8.3\times10^{-32}$. **Holds.**

**The three breakages** (`calibration/out/breakages.txt`; the module's SHA-256 `fea67d56…` equal to
the clean file's after each revert and at the end):
- **(i)** $p_{\rm lo}$ dropped from the small angle: test 1 fails on 5 cells (GROUP_CASES generic,
  $\hat K\ne K$, three residuals, QSI generic, 06f #19; e.g. #19's pair error $1.46\times10^{-13}$
  against constituents + $\phi_{\rm eval}$ of $9.07\times10^{-15}$), and test 2 fails. Caught.
- **(ii)** the abscissa term dropped: test 3 fails on 6 cells (GROUP_CASES $K/\max\sim10^{-6}$ and
  generic, $\hat K\ne K$, three residuals, QSI generic, 06f #19; e.g. GROUP_CASES generic,
  $2.88\times10^{-6}$ against $8.02\times10^{-15}$). Caught.
- **(iii)** the array path's $p_{\rm lo}$ set to zero: test 6 fails on 6 cells ("sin array and scalar
  results differ"). Caught.

**Test 8, the LiouvilleGreen draws** (`calibration/out/lg_seeded_draws_report.txt`, run
`lg_seeded_draws.jsonl`; 160 units, 160 s on 8 processes; parent module SHA-256 `862a18fe…`, the
dispatch commit's file):
- **Outcomes:** all 80 (test, seed) pairs pass under both codes. **No test fails under 06h that passes
  at the parent** — no stop.
- **Levin calls:** 4,320; none bit-identical (every value moves at round-off). Converged (parent,
  06h): T,T 3,830, F,F 490, **T,F 0, F,T 0**. Calls at depth 20: 0 and 0. Regions 68,772 → 61,126;
  solves 468,216 → 431,180; phase-limited calls 3,351 → 3,394.
- **Three-Bessel results:** 1,080; converged T,T 884, F,F 196, no change. Largest
  $|v_{06h} - v_{\rm parent}|/({\rm abserr}_{\rm parent} + {\rm abserr}_{06h})$ = 0.352; none above 1.
- **The honesty table** (dishonest = true error > reported abserr):

  | test | part | code | cases | dishonest | dishonest & converged | worst ratio | worst among converged |
  |---|---|---|---|---|---|---|---|
  | `test_YJJ_log_singularity` | near-singular ($\epsilon\le10^{-4}$) | parent | 560 | 438 | 438 | $8.131\times10^5$ | $8.131\times10^5$ |
  | | | 06h | 560 | 436 | 436 | $7.947\times10^5$ | $7.947\times10^5$ |
  | | rest | parent | 240 | 0 | 0 | 0.9343 | 0.9343 |
  | | | 06h | 240 | 0 | 0 | 0.9342 | 0.9342 |
  | `test_JJJ` | all | parent | 100 | 0 | 0 | 0.2767 | 0.2444 |
  | | | 06h | 100 | 0 | 0 | 0.2776 | 0.2437 |
  | `test_YJJ` | all | parent | 40 | 0 | 0 | 0.4096 | 0.4096 |
  | | | 06h | 40 | 0 | 0 | 0.5495 | 0.5495 |
  | `test_abserr_bounds_truth` | all | parent | 140 | 0 | 0 | 0.1827 | — |
  | | | 06h | 140 | 0 | 0 | 0.246 | — |

  One case moves from honest to dishonest, near-singular: `test_YJJ_log_singularity` seed 2 case 26,
  Y022 at $\epsilon = 10^{-4}$ (lo), true error $7.954\times10^{-8}$ under both, reported
  $3.126\times10^{-7}$ → $6.611\times10^{-8}$, converged under both. Three near-singular cases move
  the other way. Outside the near-singular cases none moves.
- **Per oracle** (value, abserr and the true error, both codes; the true errors are in the honesty
  table, and for the fixed triple below). Over every unit, max $|v_{06h} - v_{\rm parent}|$ over the
  combined abserr: J000 $4.4\times10^{-6}$, J110 $2.6\times10^{-5}$, J220 $6.2\times10^{-4}$, J222
  $5.6\times10^{-6}$, J231 $3.1\times10^{-5}$, Y000 0.346, Y022 0.352. Largest abserr, parent → 06h:
  J000 $5.86\to5.16\times10^{-11}$, J110 $3.308\to3.313\times10^{-11}$, J220 $2.914\to2.919\times10^{-11}$,
  J222 $3.555\to3.560\times10^{-11}$, J231 $4.866\to4.869\times10^{-11}$, Y000 $1.513\to1.290\times10^{-7}$,
  Y022 $1.0952\to1.0965\times10^{-6}$. At the fixed triple $(1.3, 1.7, 2.1)$ (identical at every
  seed under each code):

  | oracle | parent value | abserr | true | 06h value | abserr | true |
  |---|---|---|---|---|---|---|
  | J000 | 0.16923037349941747 | 1.9815e-11 | 2.8761e-12 | 0.1692303734994175 | 1.9960e-11 | 2.8762e-12 |
  | J110 | 0.006508860519099907 | 6.3245e-12 | 2.1667e-15 | 0.006508860519099914 | 6.3032e-12 | 2.1736e-15 |
  | J220 | -0.08423967556415518 | 6.3700e-12 | 3.2137e-13 | -0.08423967556415517 | 6.5207e-12 | 3.2138e-13 |
  | J222 | 0.08912515270056783 | 8.4513e-12 | 5.4678e-15 | 0.08912515270056785 | 8.4592e-12 | 5.4401e-15 |
  | J231 | -0.06216498240328321 | 6.3914e-12 | 4.1699e-13 | -0.06216498240328323 | 6.3975e-12 | 4.1700e-13 |
  | Y000 | -0.1142137220352884 | 9.9352e-11 | 1.8153e-11 | -0.1142137220352885 | 7.3791e-11 | 1.8153e-11 |
  | Y022 | 0.0783164899825905 | 6.1333e-12 | 2.1372e-14 | 0.0783164899825905 | 6.0809e-12 | 2.1372e-14 |

**Test 9, existing tests.** One existing assertion changed, the definition test (deviation 9):
`group.theta_abserr(log_x)` at $(1.3, 1.7, 2.1)$, signs $(+,-,-)$, $x = 5\times10^5$ was
$2.66\times10^{-15}$ (the constituents' sum, asserted to $10^{-18}$) under the parent and is
$2.79\times10^{-10}$ under 06h, almost all the abscissa term ($\varepsilon_{\exp}|\hat K|\hat x$,
$|\hat K|\hat x = 1.25\times10^6$); the test now rebuilds each of the three parts. The product-floor
bounds are unchanged (deviation 10). No convergence assertion moved.

After the ruling on deviation 10 (the follow-up commit), worst measured/bound, parent → 06h. The
parent's numbers are scored by putting the dispatch commit's module in place of the working tree's,
its declaration being the constituents' sum and its $\phi_{\rm eval}$ zero:

| assertion | case | parent | 06h |
|---|---|---|---|
| `test_three_bessel` cancellation test, raw `theta` ≤ `theta_abserr` + 2 eps\|K\|x (kept) | $K=0$ / $10^{-6}$ / $10^{-10}$ / generic | 0.4812 / 0.4845 / 0.4812 / 0.4594 | 0.442 / 0.442 / 0.442 / 0.3873 |
| same test, **new**: `theta_mod_2pi` ≤ `theta_abserr` | $K=0$ / $10^{-6}$ / $10^{-10}$ / generic | 0.480 / 0.4796 / 0.4788 / **$2.41\times10^9$** (33 of 41 points violate) | 0.441 / 0.4406 / 0.4398 / 0.4113 |
| bounded-angle test, **new** per-point (sin, cos) bound (replaces generic new ≤ old) | $K=0$ / $10^{-6}$ / $10^{-10}$ / generic | 0.4684 / 0.4684 / 0.4684 / **$1.45\times10^9$** | 0.4312 / 0.4312 / 0.4312 / 0.3147 |
| `TestThreeBesselPhaseGroups`, raw `theta` ≤ `theta_abserr` + 2 eps\|K\|η (kept) | near-resonant / generic | 0.4877 / 0.4895 | 0.4677 / 0.3752 |
| same test, **new**: `theta_mod_2pi` ≤ `theta_abserr` | near-resonant / generic | 0.4876 / **$2.58\times10^6$** (23 of 31 violate) | 0.4676 / 0.4135 |

The replaced generic assertion: 06h's pair error $1.41\times10^{-13}$ against the summed route's
$1.49\times10^{-5}$ (the old assertion asked only new ≤ old). `test_the_four_groups_reconstruct_the_triple_product`
(docstring only): worst $|{\rm direct} - {\rm groups}|$ $1.25\times10^{-4}$ (0.056 of the bound) at the
parent, $3.00\times10^{-5}$ (0.033) under 06h. LiouvilleGreen 156 (1 skipped) and ComputeTargets 780,
both OK and unchanged; no test method was added.

**`black`** on `three_bessel_integrals.py`, `bessel_phase.py`, `test_group_phase_rounding.py` and
`test_three_bessel.py`; the calibration scripts are left unformatted, as §3 says.

## Observations not acted on

1. **The product-floor tests measure the raw `theta`** (deviation 10): a tightened bound passes at
   0.91–0.97 by coincidence of sizes. Acted on, on the user's ruling, in `ac1c6f4` (deviation 10).
2. **The platform's sin and cos are not correctly rounded.** `bessel_phase.py` says, in its
   `MAX_SUPPORTED_X` comment and in `sin_cos_theta`'s docstring, that they "were verified correctly
   rounded to 1e16"; on this laptop (macOS, numpy 2.2.4) they reach 0.85 ulp, atan2 0.72 ulp.
   `EVALUATION_FLOOR`'s $8u$ is measured, not derived, so whether it still covers a constituent's
   angle addition and `c_nu_reduced` rounding at 0.85 ulp is not shown. Opened
   `[06h-platform-sin-cos-are-not-correctly-rounded]`.
3. **The builtin `sum()` is compensated on scalars only.** `BesselPhaseGroup.residual` and
   `theta_deriv` (unchanged, §0.3) add three terms with `sum()`, so their scalar and array results can
   differ in the last bit, and `_detect_vectorized`'s two-point probe cannot rule it out; the parent's
   `theta_abserr` did differ (deviation 7). Not seen for `theta_deriv` or `residual` on 2,000 points
   of each of the ten cells. `_coefficient_sum`'s docstring reasons as if the per-point sums were
   plain. Opened `[06h-builtin-sum-is-compensated-on-scalars]`.
4. **$\varepsilon_{\exp}$ was measured on the laptop.** Production runs on the Mac Studio; if its
   libm differs, `measure_exp_relerr.py` there would say so. §5 could run it.
5. **The near-singular honesty flip** (test 8): its reported abserr falls 4.7× with the true error
   unchanged; the near-singular class is 438 of 560 dishonest at the parent already, the input
   conditioning `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` records.

## State handed on

- One commit, not pushed. The header issue `[06f-theta-abserr-omits-the-product-rounding]` stays
  open; it moves in 06h's records commit, after §4 and §5.
- §4 (R21 at this commit's parent, R22 at this commit) and §5 (the census at the code commit, 06f's
  M2 phase measurement repeated) are the Mac Studio's. For §5's reference at the exact $e^u$, the
  new module's `_group_reference` and `_wrap` are the method used here.
- `docs/levin-phase-floor/06h/calibration/`: `scripts/` (`prov.py`, `measure_exp_relerr.py`,
  `breakages.py`, `lg_seeded_draws.py`) and `out/` (`exp_relerr.txt`, `breakages.txt`,
  `lg_seeded_draws.jsonl`, `lg_seeded_draws_report.txt`).

## Part B: the Mac Studio record (appended by the orchestrator, 2026-11-01)

Part B ran on 2026-11-01 on the Mac Studio (`mps026298`): R21 and Gate H's reproduction at
`2c325f5` (06e's code), R22, the census and the phase measurement at $h$ = `c1ec3b2` (`7172a8f`'s
and `ac1c6f4`'s code). The Mac Studio orchestrator ran the read-only preflight
(`06h/review/preflight/`), dispatched a rehearsal agent, a census agent and a phase agent, and
reviewed all three in ten checks: **the review passes, and no §7 stop condition was met.** The full
records are in `docs/levin-phase-floor/06h/`: `review.md` (the review), `rehearsal/`, `census/` and
`phase/` (each agent's log, verification, issues, scripts and outputs). What stays on the Mac
Studio is listed in `06h/KEPT_ON_MAC_STUDIO.md`.

**The gate test on the Mac Studio.** `test_group_phase_rounding` passes, 8 tests. Its test 5
measures $\varepsilon_{\exp}$'s basis on that host's `np.exp` as $1.114619\times10^{-16}$
(1.0040 u), identical to the laptop's, so `EXP_RELERR` = $2.23\times10^{-16}$ holds there too
(Observation 4 above).

**§4.** R21 `levin-phase-floor-06h-r21-parent-20261101T001852` and R22
`levin-phase-floor-06h-r22-groupphase-20261101T002423`, both `done`, both `1b332ffa…`,
re-fingerprinted by the review from the worktree at `c1ec3b2`. Outside `QuadSourceIntegral` the
seven value tables are identical (360,462 rows). Inside, 864 rows: `total` and `total_abserr`
bit-identical in all 864. **H0 holds, trivially**: 06h cannot reach `total` (below). Scored on the
analytic oracle instead (`analytic_rad` against `analytic.abserr`), it holds too: 464 rows move, the
worst at $2.5\times10^{-5}$ of the bound. Six readers exit 0, R22 byte-identical before and after;
the lock released.

**§5.** Census `levin-phase-floor-06h-census-20261101T005037`: 18,560 of 18,560 units `done` in
2 h 33 min, none in error or over budget. Phase measurement `levin-phase-floor-06h-phase-20261101T033514`:
84 of 84 units. A3 v2's scratch copy and 06e's checkpoint `a0dc4d7f…` unchanged by SHA-256.

| | `1e-7` | `1e-8` | verdict |
|---|---|---|---|
| Gate H | three units equal to 06e's census at the parent; $h$ deterministic | | holds |
| H1 | 21,177 points on 06f's 20 calls, 0 violations, worst 0.4933 (the parent: 561, worst 376.5) | | holds |
| H2 | 27 returned leaves, 0 violations, worst 0.268 (the parent: 2, worst 41.6) | | holds |
| H3 (`total`) | 8,560 rows, 0 violations | 8,560, 0 | holds |
| H3 (the oracle) | 8,560 rows, 0 violations, worst 0.0756 | 8,560, 0, worst $1.35\times10^{-4}$ | holds |
| H4 | 95,896 analytic-block calls, none lost | 1 lost: `g-1e-8-3-1072` #13 | fails |

**What H0 and H3 measure.** `total` is `numeric_quad + WKB_Levin`. `BesselPhaseGroup`, the only class
06h changes, reaches `QuadSourceIntegral` only through the analytic oracle, so every `total` is
bit-identical to 06e's and the two rules as written cannot fail. Their second reading, on the
oracle, is the meaningful one, and holds at both rtols. The prompt was wrong to expect `total` to
move (prompt §8, amendment 1).

**H1 and H2.** H1's worst point is `g-1e-7-0-12689` #19 (J4), Clenshaw–Curtis leaf 1, node 11:
$\delta\theta = 4.50\times10^{-13}$ against `theta_abserr` $9.13\times10^{-13}$, the review
recomputing $\delta\theta$ in `mpmath` at the exact $e^u$ at 30 and 50 digits. At the parent's worst
point (#19, dense point 996) the parent is 376.47× its declaration and $h$ 0.387×. By dominant term
at $h$: constituents 20,446 points (median $6.8\times10^{-4}$ of the declaration), abscissa 731
(median 0.134), $\phi_{\rm eval}$ none. `K_lo` is non-zero on 9 of 20 calls, at most
$4.66\times10^{-9}$.

**H4's failure** (`06h/census/issues.md` D1). `g-1e-8-3-1072` call 13, analytic J1, group (+,+,+),
$k = 10^5$, $q = r = 3\times10^8$:
- **under 06e,** the round-off exit's premise failed at the root, the root was split once, and both
  halves converged as Levin leaves: abserr $7.16\times10^{-27}$ against a request of
  $1.63\times10^{-26}$;
- **under $h$,** the root's floor is the eq. (151) floor $1.94\times10^{-26}$ plus the declared term
  $4.14\times10^{-27}$, almost all of it the abscissa's. The premise held, so the exit accepted the
  root at depth 0, unconverged at $2.36\times10^{-26}$.

The declared term does not fall under bisection, so it pushed the halves' summed floor past half
the parent's, where 06b's test says bisection cannot help. Bisection would still have helped. Under
06e the halves' floors summed to $9\times10^{-27}$; adding 06h's whole declared term gives about
$1.3\times10^{-26}$, still below the request. The halves' floors under $h$ were not recorded; the
reading rests on the premise counters and 06e's halves. The row's `total` and oracle value are bit-identical to
06e's. It is the only lost call of $2\times129{,}621$ scored.

**Descriptive.** Unit wall ×0.974 and ×0.981 of 06e's. Analytic-block regions fall ×0.876 and
×0.870 (625,878 → 576,425 regions in all at `1e-7`; 799,186 → 738,153 at `1e-8`); group calls
unchanged. The declared term reaches at most 0.341 of a call's request (median $5.3\times10^{-6}$
and $6.3\times10^{-5}$). On the Clenshaw–Curtis branch, calls whose largest floor term is the
declared one rise 2,493 → 5,137 and 2,785 → 5,939. The 720 stationary-point rows: no `total` moved,
the oracle moved in 462 and 421, within the bound; no convergence changed. The near-singular class
(435 rows, all (+,+,−) with $q = r \gg k$) did not move.

**The user's rulings** (2026-11-01, the board's Decisions, **06h Part B lands**):
- **H4 counts as failing.** Its call narrows `[07-roundoff-exit-assumes-bisection-cannot-help]`:
  the premise test compares the halves' floors with the parent's, so a part of the floor that does
  not fall under bisection can make it hold where bisection would reach the request.
- **The header issue closes.** `[06f-theta-abserr-omits-the-product-rounding]`'s condition, §2's
  tests and §5's H1 and H2, is met.
- **D3 narrows `[06e-cc-declared-phase-term-is-a-crude-proxy]`**: the proxy now multiplies a larger
  declaration.
- **Nothing else opens.** The rehearsal agent's question about the readers is answered: they start
  Ray and ran under the lock, as the prompt requires, and the brief that said otherwise was
  imprecise. The `BrokenPipeError` of `runregistrykit list | head` is Python's ordinary behaviour
  on a closed pipe.
- **On the Mac Studio:** the user will retire R21 and R22. The census run directory (2.3 GB) stays
  in full as 06g's "before".
