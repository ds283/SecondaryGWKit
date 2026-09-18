# Radiation oracle — implementation state

**Last updated:** 2026-09-18 · **Status: COMPLETE — 1 / 1 prompts landed.** The audit
[`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
was measured at `2033cfc`; **prompt 01 has landed the oracle** as
`ComputeTargets/tests/kohri_terada.py` with `ComputeTargets/tests/test_kohri_terada_oracle.py`, and
README §5's acceptance is met. Two issues remain open in §3, one of them opened by prompt 01.

**Campaign:** [`README.md`](README.md)

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [The Kohri–Terada radiation oracle](01-kohri-terada-radiation-oracle.md) | **R1** | Opus | ✍️ yes | ✅ | *"Land the Kohri-Terada radiation oracle with its tests"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/01-…`](logs/01-kohri-terada-radiation-oracle.md) |

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| R1 | **FACILITY** | The radiation-era closed form of Kohri & Terada eq. (22) in the tree as a module, with eq. (25), the `Cin`-regularised resonance, and tests pinning it against `scipy.quad` **and** against the code's own `total` at $N = -9/8$. The measurement is done — `KOHRI-TERADA-ORACLE.md` §7, nine $b=0$ cases, $N$ constant to 4.7e-13 and worst deviation from $-9/8$ 3.09e-13 against a quadrature of KT's integrand, 3.18e-10 against eq. (22) itself (§7.1) — so the prompt productionises a verified result rather than discovering one. Acceptance turns on whether the tests would catch the paper's three errata, not on whether the numbers agree. | 01 | ✅ **Done, 2026-09-18.** The module is `ComputeTargets/tests/kohri_terada.py` (eqs. 15, 16 at $w=1/3$, 19, 20, 22, 25, the `Cin` regularisation, `total_from_I_RD`), with nine test methods in `test_kohri_terada_oracle.py` covering prompt §3's six tests. Test 6 runs **all nine** $b=0$ cases at `(1e-45, 1e-12)` in 2.7 s and asserts $N=-9/8$ against eq. (22) minus the head: worst $\lvert N+9/8\rvert$ **1.08e-10** (q-smooth), 6.9e-14 elsewhere, each under a per-case bound built from the two sides' declared errors. **The 3.18e-10 this row quotes "against eq. (22) itself" is eq. (22)'s own double-precision rounding at $u=0.01$, not the head subtraction** — `[01-the-eq22-figure-is-eq22s-own-rounding-not-the-head]`. The resonance returns 0.1391877548291586 at $x=15$, equal to quadrature. Every one of the four traps was broken deliberately and caught by a named test (log §3.2); the inverted Ci/Si pairing by tests 1, 2, 3, 4 and 6. `ComputeTargets` 521 → **530**, `CosmologyModels` **39**; no existing `.py` file in the diff. |

## 3. Active and unresolved issues

- **[01-general-w-normalisation-is-predicted-not-measured]** *(opened 2026-09-18 with the audit)* —
  the oracle covers $b = 0$ only. KT give closed forms for $I$ in pure radiation (eq. 22) and pure
  matter (eq. 37) domination and none in between, and the code's $b$ maps to their $w$ by
  `w_of_b(b) = (1-b)/(3(1+b))`, so only $b = 0$ has an oracle at all. The nine $b = 0.2$ fixture
  cases are unchecked. The natural extrapolation is
  $N(b) = -(3+2b)^2/\big(2(2+b)^2\big)$, which reproduces $-9/8$ at $b = 0$ — **predicted, never
  measured**, and recorded so that nobody reads it as established. Closing it needs either a
  general-$w$ closed form (KT §3.3 discusses transitions but produces none) or an independent
  general-$w$ quadrature oracle built from the code's own $\Phi$, which is a weaker check because it
  shares the transfer function. **Next step:** decide whether a general-$w$ quadrature oracle is
  worth having given that it shares machinery with the thing it checks.
  Indexed at `docs/OPEN_ISSUES.md` §1.9.

- **[01-the-eq22-figure-is-eq22s-own-rounding-not-the-head]** *(opened 2026-09-18 by prompt 01)* —
  the audit's §1 table ("Limited by: the head subtraction") and §7.1 preamble, campaign README §1,
  item R1 above and prompt 01 §3.6 and §6 all attribute the eq.-(22) comparison's worst figure,
  3.18e-10, to the head $0\to\bar x_{\rm min}$ being quadratured and subtracted. **It is eq. (22)'s
  own double-precision rounding at $u=q/k=0.01$**, where the $1/(u^3v^3)$ prefactor multiplies terms
  that nearly cancel: the head's declared quad error is 1e-20 to 1.4e-18 of $I$ on all nine cases,
  while eq. (22)'s double-precision value at the three q-smooth points is off a 50-digit `mpmath`
  evaluation of the same formula by 9.67e-11, 6.62e-11, 2.50e-11 — and the q-smooth deviations of
  $N$ in `test_kohri_terada_oracle` are 1.08e-10, 7.45e-11, 2.82e-11, the same numbers (the audit's
  larger 3.18e-10 is its script's summation order; `fsum` in the module lowers it). Removing it
  leaves the pipeline agreeing with eq. (22) on q-smooth to ~1e-11. **Impact:** documentary — no
  number is wrong, but a reader who believes the head limits the comparison will try to improve the
  head, which changes nothing, and will misread 3.18e-10 as a property of the code. **Next step:** an
  additive subsection in `KOHRI-TERADA-ORACLE.md` (CLAUDE.md invariant 6) recording the attribution,
  by whoever next owns `docs/radiation-oracle/`; a small-$u$ form of eq. (22) would tighten test 6 on
  q-smooth by two orders and is optional. Measurement: `logs/01-kohri-terada-radiation-oracle.md`
  §3.1. Indexed at `docs/OPEN_ISSUES.md` §1.9.

## 4. Resolved issues

*(none yet)*

## 5. Standing notes

1. **$-9/8$ is a property of the whole prefactor chain, not of the quadrature.** It factorises as
   $-\tfrac12\times\tfrac94$: the $\tfrac94$ is $1/c^2$ with $c = (2+b)/(3+2b)$
   (`docs/spec/03-source-term.md` §0.1), which Kohri & Terada fold into their $f$ and the code
   strips; the sign is the orientation convention of `docs/spec/04-source-integral.md` §0(3); and
   the $\tfrac12$ is the author's $h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$. Any future change that
   perturbs a factor of two or a sign anywhere in that chain moves $N$ off $-9/8$, which is what
   makes prompt 01's test 6 a regression test on the chain rather than on the integrator.

2. **This closes `docs/spec/cross-spec-check.md` §3 item 9 for the Kohri–Terada half** — the
   $h_{ij}$ normalisation, which `docs/spec/05-one-loop.md` §539 records as transcribed with the
   transcriber explicitly not having checked it, and which the cross-spec check lists under "no spec
   allows a check". The paper allows one and it holds to ten digits. **The spec has not been edited
   and must not be by this campaign** (README §2): `prompts/spec-transcription` owns it and sign-off
   is the author's. The Adshead half of that item is untouched and still unchecked.

3. **The paper carries three errata**, all in `KOHRI-TERADA-ORACLE.md` §2 and all found by
   disagreement with a quadrature of the paper's own eq. (15): the Ci/Si arguments pair with the difference rather
   than the sum; the stated small-$x$ limit $x^2/2$ is wrong and the truth is $2x^2/9$; and eq. (20)
   cannot be evaluated below $x\approx0.1$ in double precision. The first produces a plausible
   wrong function, so a reader "correcting" the module back to the PDF would break it silently.

4. **The resonance $u+v=\sqrt3$ is where the oracle is most wanted and where eq. (22) as written
   fails.** The $\mathrm{Cin}$ regularisation is algebraically exact, not an approximation, and
   returns the finite value at the resonance itself (0.1391877548292 at $x=15$, matching quadrature
   to 4.0e-16).

5. **The oracle exercises the Levin path, and this is the campaign's least obvious benefit.**
   `total = numeric_quad + WKB_Levin`; on the nine $b=0$ cases the Levin half runs over 2–3 regions
   apiece and carries 0.099 to 3.846 of $|total|$, exceeding it in four. The halves cancel, so the
   agreement bounds the Levin half's own relative error at 8.0e-14 to 3.1e-12. **What it does not
   reach:** `analytic_rad`'s `_three_bessel_Levin`, a separate path and the subject of
   `qsi-phase-groups`' own issue; and it bounds the *combination*, so a Levin error cancelling
   against a compensating `numeric_quad` error would pass.

6. **Baselines at the audit commit `2033cfc`:** `ComputeTargets` **521**, `CosmologyModels` **39**,
   both OK; `config/defaults.py` at blob `76bab78…`. The known flake is
   `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`, a wall-clock
   assertion that passes roughly one run in three — confirm by re-running that module alone before
   attributing a failure to a commit.
