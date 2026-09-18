# Radiation oracle — implementation state

**Last updated:** 2026-09-18 · **Status: not started — 0 / 1 prompts landed.** The audit
[`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
is complete and measured at `2033cfc`; prompt 01 is written against it and has not been dispatched.

**Campaign:** [`README.md`](README.md)

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [The Kohri–Terada radiation oracle](01-kohri-terada-radiation-oracle.md) | **R1** | Opus | ✍️ yes | ⬜ | | |

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| R1 | **FACILITY** | The radiation-era closed form of Kohri & Terada eq. (22) in the tree as a module, with eq. (25), the `Cin`-regularised resonance, and tests pinning it against `scipy.quad` **and** against the code's own `total` at $N = -9/8$. The measurement is done — `KOHRI-TERADA-ORACLE.md` §7, nine $b=0$ cases, $N$ constant to 4.7e-13 and worst deviation from $-9/8$ 3.09e-13 against a quadrature of KT's integrand, 3.18e-10 against eq. (22) itself (§7.1) — so the prompt productionises a verified result rather than discovering one. Acceptance turns on whether the tests would catch the paper's three errata, not on whether the numbers agree. | 01 | ⬜ |

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
