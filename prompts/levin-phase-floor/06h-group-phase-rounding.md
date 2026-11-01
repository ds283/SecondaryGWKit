# Prompt 06h — the Bessel group phase's rounding: remove what can be removed, declare the rest

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-31, on the user's decisions after 06f's results (the board's Decisions,
**06f results**):
- on `[06f-theta-abserr-omits-the-product-rounding]`: *"We should fix before 07 (I assume) but it
  doesn't matter where that happens. We can have an 06f if needed."*;
- on the order: *"Yes, I agree with 06h first and the design in point 2."* So **06h lands first,
  then 06g** (the trust check's value and the signed node-error combination), **then 07 resumes**.

**Where it runs:** in three parts, as 06e does.
- **§1–§3, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§4, rehearsals R21 and R22:** on the Mac Studio, after the code commit is pushed, under the
  lock. R21 runs at the code commit's parent and R22 at the code commit, each in a worktree.
  06g's rehearsals become R23 and R24, and 07's R25 and R26.
- **§5, the census and the phase re-measurement:** on the Mac Studio, Ray-free, with no lock, on
  06's certified scratch copy of A3 v2. 06e's census is re-run at the code commit, with 06e's run
  as the "before", and 06f's M2 measurement of the phase error is repeated at the code commit.
  **This census becomes 06g's "before".**

The records of §4 and §5 are committed on the laptop afterwards, in 06h's records commit.

**Closes, when §2's tests and §5's H1 and H2 hold:** `[06f-theta-abserr-omits-the-product-rounding]`.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entry `[06f-theta-abserr-omits-the-product-rounding]`, and
   `[06e-cc-declared-phase-term-is-a-crude-proxy]` for how the declared term enters a
   Clenshaw–Curtis leaf.
2. `LiouvilleGreen/three_bessel_integrals.py`: `_coefficient_sum` and its docstring;
   `BesselPhaseGroup`, its class docstring (the paragraph that calls the product's rounding "a
   genuine limit"), `__init__`, `residual`, `sin_cos`, `theta_mod_2pi`, `theta_deriv`,
   `theta_abserr` and `levin_theta`.
3. `LiouvilleGreen/bessel_phase.py`: the module docstring's "Accuracy of the supplied argument",
   `EVALUATION_FLOOR`, `BesselPhaseFunction.theta_abserr_at`, `sin_cos_theta` and
   `residual_log_deriv`.
4. `ComputeTargets/QuadSourceIntegral.py`, `_three_bessel_Levin`: how the groups are built
   (coefficients `k.k`, `q.k * cs`, `r.k * cs`) and called, with integration variable
   $u = \log\eta$.
5. `AdaptiveLevin/levin_quadrature.py`: `_phase_error`, `_declared_endpoint_phase_err`, the
   comment above `endpoint_proxy` in `_adaptive_levin_subregion_cc`, `_cc_roundoff_floor`'s
   paragraph "What it does not carry", and `_detect_vectorized` / `_sample_vectorized` (a phase
   callable's array result must be bit-identical to its scalar results).
6. The tests that bound the group phase today, by the declaration **plus** $2\varepsilon|K|x$:
   `LiouvilleGreen/tests/test_three_bessel.py`
   (`test_the_group_phase_survives_cancellation_of_its_leading_term`, and the test near line 640
   asserting that `theta_abserr` equals the constituents' sum) and
   `ComputeTargets/tests/test_quadsource_integral.py` (`TestThreeBesselPhaseGroups`).
7. 06f's records, `docs/levin-phase-floor/06f/`: `out/results.md` (M2), `out/m2_calls.csv`,
   `out/cmd/44_f4_attribution.txt`, `review.md` (its §4 and Observation 1, the exact-$e^u$
   reference), and the scripts `scripts/replay06f.py`, `scripts/method_check_06f.py`,
   `scripts/f4_attribution_06f.py`, `scripts/declared_vs_request_06f.py` and
   `review/scripts/orch_m2.py`.

**Gate:**
- 06f's records commit has landed (the board's 06f row is ✅);
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

`BesselPhaseGroup` holds a sum-and-difference group phase as
$\theta(x) = Kx + C + R(x)$, with $x = \hat x = {\rm fl}(e^u)$ computed from the integration
variable $u$. `theta_abserr` declares the sum of the three constituents' declared errors at their
own arguments, and nothing else. What the group's own arithmetic adds is not declared:

- **(a) The product $Kx$.** `sin_cos` hands ${\rm fl}(K\hat x)$ to `sin` and `cos`. Its rounding
  is up to $u|K\hat x|$ rad. The class docstring calls it "a genuine limit" (1.5e-5 rad at
  $K = 0.1$, $x = 10^{12}$), and two tests already add $2\varepsilon|K|x$ to the declaration before
  asserting it. The driver is never told.
- **(b) $K$ itself**, when $k + e_\nu q + e_\sigma s$ is not representable: one rounding,
  $|K - \hat K|\,\hat x$ rad. Zero on 06f's two calls; not zero in general.
- **(c) The abscissa.** The driver integrates in $u$, and the phase it receives is
  $\hat\theta(u) = \theta({\rm fl}(e^u))$. The rounding of `exp` displaces the point at which
  $\theta$ is evaluated by a relative $\delta_e$, which moves the phase by about
  $|d\theta/du|\,\delta_e$. `bessel_phase.py` defines its own evaluation to be "at the computed
  $\exp(u)$", which is right for a single factor's representation. It is not what the driver
  needs: `levin_quadrature.py` says an error in the phase function's value belongs in the caller's
  `theta_abserr`, and the phase function the driver integrates is a function of $u$.
- **(d) The rest of the group's evaluation:** the sum forming the small angle, the angle addition,
  and the rounding of each constituent's argument $m_i\hat x$ as it reaches $r_i$.

**Measured** by 06f's M2 (`VERIFICATION.md` §6f): 49 of 1,177 nodes above the declaration, on
the tail Levin leaves of two calls, up to 148× (`g-1e-7-0-12689` #19, $K\eta = -6.85\times10^3$,
$\delta\theta = -3.94\times10^{-13}$ against `theta_abserr` $2.66\times10^{-15}$). Term (a) explains
each one to within 0.27 of the declaration. With the reference at the exact $e^u$ (the review's
Observation 1), term (c) changes the ratios at the three worst nodes to 246, 17.9 and 29.2: still
violations, and of the same size as (a). On one Levin leaf the response to the measured phase error
is $1.7\times10^{-25}$ against that leaf's declared term of $1.9\times10^{-26}$.

### 0.2 The remedy

**1. Remove (a) and (b) at their source.** Both are exactly computable:
- $\hat K$'s rounding error $K_{\rm lo} = K - \hat K$, from the float coefficients and signs, by
  `math.fsum` of the terms and $-\hat K$, computed once in `__init__`. The terms are exact (a float
  times $\pm1$), so this is $K - \hat K$ correctly rounded, with an error of at most
  $u|K_{\rm lo}|$;
- the product's rounding error $p_{\rm lo} = \hat K\hat x - {\rm fl}(\hat K\hat x)$, by
  TwoProduct (`math.fma`, or Dekker's split for arrays), exact barring overflow and underflow.

Both are added to the **small** angle in `sin_cos`, which becomes
$C_{\rm reduced} + R(\hat x) + p_{\rm lo} + K_{\rm lo}\,\hat x$, and the big angle stays
${\rm fl}(\hat K\hat x)$, unreduced as now. The group phase that `sin_cos` and `theta_mod_2pi`
represent is then $K\hat x + C + R(\hat x)$ to the rounding of the small sum, which is a few $u$
of $|C_{\rm reduced}| + |R| + |p_{\rm lo}| + |K_{\rm lo}\hat x|$.
- **Scalar and array results must be bit-identical** (`_detect_vectorized`). TwoProduct is exact
  either way, so `math.fma` on scalars and Dekker's split on arrays agree. The test asserts it.
- `theta` (the raw value, which the driver does not evaluate when `theta_mod_2pi` and
  `theta_deriv` are supplied) and `theta_deriv` are **not** changed. `theta_deriv`'s relative error
  from the product's rounding is $u$, inside the driver's $\eta_\theta = 2^{-10}$ (06e).

**2. Declare (c) and (d)** in `theta_abserr`, which becomes
$$\texttt{theta\_abserr}(u) = \sum_i \texttt{theta\_abserr\_at}_i(m_i\hat x)
\;+\; \varepsilon_{\exp}\,\Bigl|\frac{d\theta}{du}\Bigr|
\;+\; \phi_{\rm eval}(u),$$
- **$\varepsilon_{\exp}$** bounds the relative error of the `exp` the class calls (`np.exp` on
  scalars and arrays). **Derived** from the platform library's documented accuracy if it is
  documented; otherwise **measured** on a seeded sample of at least $10^6$ arguments over the
  production range of $u$ and the LiouvilleGreen range (up to $\log 10^{12}$) against `mpmath`, and
  taken at a stated conservative multiple of the worst measured, at least 2: the generous end,
  with its basis stated. The log states the value and its basis.
- **$|d\theta/du|$** is bounded from data the class holds: $|\hat K|\hat x + \sum_i
  |dr_i/d\log y|$ at $m_i\hat x$, with a stated allowance for its own rounding. Not `theta_deriv`
  itself, whose signed sum can cancel.
- **$\phi_{\rm eval}$** bounds (d), term by term, each with its derivation in a comment, as 06d's
  and 06e's constants have:
  - the small angle's sum (four or five terms, $\gamma_n$ times the sum of their magnitudes);
  - the angle addition, as an equivalent phase error, counted against what each constituent's
    `EVALUATION_FLOOR` already covers. A rounding already declared by a constituent is not
    declared twice, and one not declared by any is not omitted;
  - the rounding of each argument $m_i\hat x$, at most $u\,|dr_i/d\log y|$.
- **Combined linearly**, as the constituents are, and for the same reason.
- **Scalar and array results bit-identical**, as for the phase.

**What it costs the driver.** On a Levin leaf the declared term is
$\texttt{theta\_abserr}\cdot\sum_i|p_i|$ at each endpoint, and away from a stationary point
$|p| \approx |f|/|\theta'|$. So the new abscissa term contributes about $\varepsilon_{\exp}|f|$
per endpoint, whatever $x$ is. On a Clenshaw–Curtis leaf the proxy gives about
$\varepsilon_{\exp}|\theta'|h\,f_{\rm scale}$, and the C2 gate keeps $|\theta'|h$ below about
$6\pi$. **The declaration should therefore rise by orders in the tail while the declared error of
a call barely moves.** §5's H3 and H4 test that.

**3. Docstrings and comments:**
- `BesselPhaseGroup`'s class docstring: the paragraph on the product floor says it is now carried
  exactly in the small angle, and says what remains;
- `sin_cos` and `theta_abserr`, each stating what it now contains and why;
- `_coefficient_sum`'s note that the per-point sums do not use `fsum`, where it describes the
  product's rounding as the dominant error;
- `BesselIntegralResult`'s "What it includes";
- in `bessel_phase.py`, a sentence in "Accuracy of the supplied argument" pointing to the group's
  declaration of the abscissa term. **That file's code does not change.**

### 0.3 What this changes, and what it does not

- **It changes:**
  - `BesselPhaseGroup.__init__` (the stored $K_{\rm lo}$), `sin_cos` (and so `theta_mod_2pi`) and
    `theta_abserr`;
  - private helpers for TwoProduct, $\varepsilon_{\exp}$ and $\phi_{\rm eval}$, with named
    constants;
  - the docstrings and comments of §0.2 item 3.
- **It does not change:**
  - `theta`, `theta_deriv`, `residual`, `arguments`, `levin_theta`'s keys;
  - anything in `bessel_phase.py`'s code, any constituent's `theta_abserr_at`, `EVALUATION_FLOOR`;
  - `AdaptiveLevin/` in any way, including how the declared term enters a leaf;
  - `ComputeTargets/` code, including how `_three_bessel_Levin` forms its coefficients. A comment
    there may be corrected if it describes the group's phase error;
  - `ComputeTargets/phase_groups.py`, the cosmological groups, which are a different object.
- **What moves:** every value that passes through `sin_cos` moves at the round-off level, and every
  declared error of a call with a supplied group. So every analytic-block Levin call and every
  LiouvilleGreen three-Bessel call can move in value, regions, depth, `abserr` and `converged`.
  Every `QuadSourceIntegral` row can move (§4).

### 0.4 What it does not fix

- **How the declared term enters a leaf and combines across leaves.** The Clenshaw–Curtis proxy,
  the per-leaf absolute endpoint terms, and the median 6× 06f measured
  (`[06e-cc-declared-phase-term-is-a-crude-proxy]`) are unchanged.
- **The signed combination of the node errors**, which is 06g's.
- **Input coefficient uncertainty.** `q.k * cs` and `r.k * cs` are rounded upstream. The group is
  exact for the coefficients it is given; their rounding is a property of the inputs, as the class
  docstring says.
- **The modulus.** The abscissa's rounding also moves $f$, by a relative
  $\delta_e\,|d\log f/du|$, and `adaptive_levin_sincos` has no channel for an error in $f$
  (`BesselIntegralResult`, "What it still does not include" (i)). Not declared here.
- **The cosmological phase groups** (`[01-cosmological-group-declares-no-phase-error]`).

## 1. The code

1. **$K_{\rm lo}$** in `__init__`, as §0.2 item 1.
2. **TwoProduct** as a private helper, scalar and array, bit-identical; the small angle in
   `sin_cos` as §0.2 item 1.
3. **The declaration** as §0.2 item 2, with $\varepsilon_{\exp}$ and each term of $\phi_{\rm eval}$
   a named constant or helper, its derivation in a comment. A constant measured rather than derived
   says so, with the measurement's script and numbers.
4. **Docstrings and comments**, per §0.2 item 3.

## 2. Tests

A new module, `LiouvilleGreen/tests/test_group_phase_rounding.py`, no Ray, no datastore.
References are `mpmath` at 30 digits, with 50 digits where a check of the reference is asked. **The
reference for a phase is at the exact $e^u$**: `mpmath.exp` of the float $u$, then each constituent's
phase at $m_i e^u$ exactly, composed with the group's signs. A constituent's reference phase is
$\arg H^{(1)}_\nu(y) + \pi/2$, or `atan2(J, −Y)`; the method check confirms them equal at one
argument. The parent is `HEAD`'s module (`git show <dispatch SHA>:LiouvilleGreen/three_bessel_integrals.py`,
imported under another name). A test that quotes the parent's numbers records them in its comment.

**The cells.** Seeded grids in $u$ over:
- `test_three_bessel.py`'s `GROUP_CASES` (exact, near and absent cancellation), up to
  $x = 10^{12}$;
- `test_quadsource_integral.py`'s `TestThreeBesselPhaseGroups` coefficients;
- 06f's two violating calls' groups, by their coefficients, signs and spans
  (`g-1e-7-0-12689` #19, J4; `g-1e-7-3-1054` #20, Y4), from `docs/levin-phase-floor/06f/`;
- a case with $\hat K \neq K$, built from coefficients whose signed sum is not representable.

1. **The product's rounding is gone.** In every cell, the phase `sin_cos` represents, compared as an
   angle modulo $2\pi$ against the reference at the **float** $\hat x$, is within the constituents'
   declarations plus $\phi_{\rm eval}$: no $u|K\hat x|$ term is needed. The parent's error at the
   same points, and its excess over the constituents' declarations, are recorded.
2. **$K$'s rounding is gone.** On the $\hat K \neq K$ case, the same against the reference with the
   exact $K$; the parent's error, which carries $|K_{\rm lo}|\hat x$, is recorded.
3. **The declaration bounds the phase error as a function of $u$.** In every cell, at every point,
   $|\hat\theta(u) - \theta(e^u)| \le \texttt{theta\_abserr}(u)$, the reference at the exact
   $e^u$. The parent fails it on 06f's cells; the log records where and by how much.
4. **The declaration is not loose by orders where its new terms dominate.** Descriptive, not
   asserted: the distribution of measured over declared, split by which term dominates.
5. **$\varepsilon_{\exp}$'s basis.** If measured: the measurement as a test, asserting the worst
   relative error over the sample is at most $\varepsilon_{\exp}$ divided by its stated multiple.
6. **Scalar and array agree bit for bit** for `sin_cos`, `theta_mod_2pi` and `theta_abserr`, and
   `_detect_vectorized` still finds the group's callables vectorising.
7. **The method check.** The constituent reference reproduces, at one argument per order, the error
   `bessel_phase.py` documents there, and the two reference formulas agree to $10^{-25}$.
8. **The LiouvilleGreen draws.** `test_3bessel_analytic.py` over seeds 0–19, as 06e's test 7, at
   the parent and at 06h:
   - no test fails under 06h that passes under the parent;
   - the log carries the honesty table (dishonest, dishonest-and-converged, worst ratio) under
     both, converged counts, and depth-20 calls, with the near-singular cases reported separately;
   - per oracle of `test_3bessel_analytic.py`, value, `abserr` and the true error under both.
   - The harness lives in `docs/levin-phase-floor/06h/calibration/scripts/`.
9. **Existing tests.** Every test in `LiouvilleGreen/tests/`, `ComputeTargets/tests/` and
   `AdaptiveLevin/tests/` passes. Two kinds of change to an existing assertion are allowed, each
   listed with its numbers under both codes:
   - **the two product-floor bounds** (`test_three_bessel.py` and `test_quadsource_integral.py`),
     `theta_abserr + 2 EPS |K| x`, become `theta_abserr` alone: a tightening, since the
     declaration now carries what the extra term stood for;
   - **the definition test** near `test_three_bessel.py` line 640, which asserts that
     `theta_abserr` equals the constituents' sum, asserts the new sum, term by term.

   Any other change, and in particular a convergence assertion that would go from converged to
   unconverged, is a stop (§7).

**Deliberate breakages,** each recorded with the test that catches it, then reverted:
- **(i)** $p_{\rm lo}$ dropped from the small angle: test 1 must fail;
- **(ii)** the abscissa term dropped from the declaration: test 3 must fail;
- **(iii)** the array path computing $p_{\rm lo}$ differently from the scalar path (e.g. set to
  zero): test 6 must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records of the code

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06h-group-phase-rounding.md`, in README §5.1's form. Its **Predictions and verdicts** holds
  §5's predictions, copied before they run and marked not yet scored.
- The board's 06h row.
- **One commit.** The header's issue moves only in 06h's records commit, after §4 and §5.

## 4. Rehearsals R21 and R22 — on the Mac Studio, after the code commit is pushed

As 06e §4. Records go under `var/levin-phase-floor-06h/rehearsal/`.
1. **The lock:** `acquire`, `set-pid` at each launch, `release` after the readers. Each lock
   command is a shell command of its own (`CLAUDE.md`).
2. **R21** runs at the code commit's parent and **R22** at the code commit, each in a worktree.
   - Each runs `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R19 and R20 ran it.
   - Each is registered with `--campaign levin-phase-floor --prompt 06h`, with slugs
     `r21-parent` and `r22-groupphase`.
   - **Both fingerprints must equal `1b332ffa…`.**
3. **The value comparison** is `tools.store_value_diff` on R21 and R22, with `--json`, with 05's
   `classify_non_qsi.py` copied unchanged.
   - **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
     the four volatile kinds, and no row is in one store only.
   - **Inside `QuadSourceIntegral`, every row is tabulated:** $|{\rm total}_{22} - {\rm
     total}_{21}|$ against `total_abserr` in each, the region counts, maximum depths and
     `total_converged` in each.
   - **H0 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
     `total_abserr` in both stores has $|{\rm total}_{22} - {\rm total}_{21}| \le {\rm
     total\_abserr}_{21} + {\rm total\_abserr}_{22}$. Rows with an infinite `total_abserr` in either
     are counted separately, and the stationary-point rows are tabulated, not scored. **Holds** if
     no row violates it. A violation is a finding for the user, not by itself a stop.
4. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) run on R22. Every one
   exits 0, and the store is byte-identical before and after.
5. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
6. **The stores are kept** until the user says otherwise.

## 5. The census and the phase re-measurement — on the Mac Studio, Ray-free

**The tree** $h$ is a worktree at Part B's dispatch SHA, whose code is the code commit's. **The
store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its certification
and fingerprint, read-only, and hashed before and after. Nothing else in that directory is touched.

**The census.** 06e's census, re-run at $h$, with 06e's run
(`levin-phase-floor-06e-census-20261030T081318`, checkpoint `a0dc4d7f…`) as the "before".
- **The harness** is 06e's, copied from `docs/levin-phase-floor/06e/census/scripts/` into
  `var/levin-phase-floor-06h/census/scripts/`. Change only the tree and its paths. Save the diff
  against 06e's copy, and record each change as a deviation.
- **The units:** every row at `1e-7` and at `1e-8`, in one registered, detached run with slug
  `census`. **The budget:** 300 s per unit, with 06a's over-budget pass.
- **Its checkpoint and per-unit records are kept in full on the Mac Studio.** 06g's census compares
  against them.

**The phase re-measurement.** 06f's M2 item 1 and item 4, repeated at $h$ on 06f's 20 M2 calls
(`docs/levin-phase-floor/06f/out/sample/`), with 06f's scripts copied into
`var/levin-phase-floor-06h/phase/scripts/` and the diff saved:
- each call captured at $h$ as `x1_capture.py` captures, and replayed;
- $\delta\theta$ at every fine node of every returned Clenshaw–Curtis leaf and every collocation node
  of every returned Levin leaf, plus 06f's seeded dense sample of 1,000 points per call;
- **the reference at the exact $e^u$**, as §2;
- per returned leaf, the first-order response to the measured phase error (06f's item 4), against
  that leaf's declared term.

**Gate H, before scoring:**
- **The parent reproduces 06e.** `git diff --stat 6d81fad <code commit's
  parent> -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` and files whose ASTs without
  docstrings equal the dispatch commit's. At the parent, the harness reproduces 06e's units **bit
  for bit** on `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on `0-1971` at `1e-8`.
- **The new code is deterministic.** Two runs of one $h$ unit are bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **H1 — the declaration is honest.** **Rule:** on every one of the 20 M2 calls, at every point
  sampled, $|\delta\theta(u)| \le \texttt{theta\_abserr}_h(u)$, the reference at the exact $e^u$.
  **Holds** if no point violates it. The parent's declaration is scored on the same points beside it.
- **H2 — the declared term covers the response.** **Rule:** on every returned leaf of the 20 calls,
  the first-order response to the measured phase error is at most that leaf's declared term.
  **Holds** if no leaf violates it.
- **H3 — the rows agree.** **Rule:** outside the stationary-point rows, every row with finite
  `total_abserr` under both has $|{\rm total}_h - {\rm total}_{06e}| \le {\rm total\_abserr}_{06e} +
  {\rm total\_abserr}_h$. **Holds** if true at both rtols.
- **H4 — no convergence is lost.** **Rule:** outside the stationary-point rows, no Levin call that
  06e reported converged is unconverged under $h$. **Holds** if true at both rtols. A failure is a
  finding for the user: §0.2's estimate of what the declaration costs the driver is wrong.

**Descriptive, not scored:**
- **What the declaration costs:** per analytic-block call, the declared term's sum over leaves under
  06e and $h$, by branch, and against the request. 06f's `declared_vs_request_06f.py` method, on
  both checkpoints.
- **Convergence and depth:** converged calls and rows, `total_converged`, and the depth
  distribution under 06e and $h$, at both rtols.
- **Cost:** regions, solves and wall time against 06e. Cost is not an argument (README §5 rule 4).
- **The stationary-point rows:** how they moved. They are 07's, and not scored here.

## 6. Acceptance

1. **§1–§3 as written:**
   - every test passing, the three breakages caught;
   - every changed existing assertion of the two kinds §2 test 9 allows, and listed;
   - the LiouvilleGreen draws with no new failure;
   - the seven roots rising only by the tests added.
2. **§4:** R21 and R22 both `done` with `1b332ffa…`; outside `QuadSourceIntegral`, every value table
   identical; inside it every row tabulated and H0 scored; the readers exit 0; the lock released.
3. **§5:** Gate H holds; H1–H4 are each scored holds, fails or undecided against the rule as
   written, with their numbers; the store is unchanged.
4. **Records:**
   - `logs/06h-group-phase-rounding.md`, with §4 and §5's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §6h, "The group phase's rounding";
   - the records in `docs/levin-phase-floor/06h/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **A term cannot be removed or bounded as written:** TwoProduct is not exact on some cell (report
  it); $\varepsilon_{\exp}$ can be neither derived nor measured; or a rounding in (d) has no bound
  that can be stated.
- **A test cannot pass as written:** test 3 fails at 06h on any cell (report the point, the measured
  error and each term of the declaration); scalar and array results differ.
- **The construction reaches outside its files,** beyond test files and comments: to
  `bessel_phase.py`'s code, `AdaptiveLevin/`, `ComputeTargets/` code or `phase_groups.py`.
- **A test moves beyond §2 test 9's allowance,** a LiouvilleGreen test fails under 06h that passes
  under the parent, or a seven-roots count falls.
- **The rehearsals:** R21 or R22 does not give `1b332ffa…`, or outside `QuadSourceIntegral` a value
  table differs.
- **The census:** Gate H fails, or a unit ends in an error (not over budget).
- **Any store file** changes.
- **The lock** is held, or will not release.

## 8. Amendments after the run (2026-11-01)

1. **H0 and H3 score `total`, which 06h cannot move.** `total` is `numeric_quad + WKB_Levin`;
   `BesselPhaseGroup` reaches `QuadSourceIntegral` only through the analytic oracle (`analytic_rad`,
   `analytic.*`). Every `total` stayed bit-identical, so both rules held trivially. The test that
   bears on the change is the same rule applied to the oracle's value and declared error; it was
   scored beside each as a second reading, and holds (log, Part B). A future prompt that changes
   the analytic blocks scores that reading.
