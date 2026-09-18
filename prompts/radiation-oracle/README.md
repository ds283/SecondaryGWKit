# The radiation oracle — a one-prompt campaign

**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) ·
**Audit:** [`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)

## 1. Why this exists

Kohri & Terada ([arXiv:1804.08577](https://arxiv.org/abs/1804.08577), Phys. Rev. D **97**, 123532)
give a **closed form** for the source time integral in a radiation-dominated universe — their
eq. (22), in $\mathrm{Si}$, $\mathrm{Ci}$, sines and logarithms. It is the same object
`evaluate_QuadSource_integral` computes: their eq. (15) integrand is the Green's function against
the quadratic source, and spec 04 §0(2) says the code's $G$ and theirs are the same function up to
$-a_0H(z')$.

The audit measured the relation on 2026-09-18, offline, through the fixture
`ComputeTargets/tests/test_quadsource_integral.py` builds:

$$\text{total} = -\frac{9}{8}\,\frac{1}{k_{\rm phys}^2}\;I_{\rm RD}(r/k,\,q/k,\,k\tau(z_{\rm resp}))$$

on **all nine $b = 0$ cases**, spanning $x$ from 4.33 to 1556 and $u$ from 0.01 to 12, with $N$
constant to **4.7e-13** against a quadrature of KT's integrand and to **3.18e-10** against their
closed form itself (audit §1 — the two are not interchangeable and the difference is the head
subtraction, not the agreement). That measurement is the premise of this campaign; this prompt lands
it as code with tests.

> **Corrected 2026-09-18:** the difference between the two figures is eq. (22)'s own double-precision
> rounding at $u = 0.01$, not the head subtraction. With eq. (22) at 50 digits the code agrees with it
> to 3.67e-13 (audit §7.2; `[01-the-eq22-figure-is-eq22s-own-rounding-not-the-head]`, closed).

**The audit's §0 is a precondition for reading any of it.** Two of the three objects compared there
are the audit's own transcriptions of Kohri & Terada, who published no code, and those two share the
Green's function, the measure and the source — so the only comparison that can detect a misread
kernel is the one against this repository's pre-existing pipeline. The $N = -9/8$ constancy is
therefore **mutual**: it is the evidence that the transcription is right *and* the first absolute
accuracy check the source integral has had.

### Why it is worth having

**The tree has no fixed oracle for this sector.** `analytic_rad` is the code's own closed form, but
it reduces to three-Bessel integrals that are quadratured *at the caller's own tolerances*
(`QuadSourceIntegral.py:1523`, comment at `:1547`), so it moves when a tolerance moves — up to
×1.25e+06 further than `total` does, which is `[06-analytic-rad-is-computed-at-the-callers-tolerance]`
on `qsi-phase-groups`' board. `prompts/tolerance-convergence` prompt 06 had to freeze it at a
reference pair and fall back on **self-convergence** as its primary statistic
(`docs/tolerance-convergence/QUADSOURCE-READONLY.md` §2.1), with an independent `scipy.quad`
cross-check on a short range at only 6 of 18 cases. $I_{\rm RD}$ is fixed by construction and moves
with no tolerance, and it shares no machinery with the pipeline it scores — though, per §1 above, it
shares plenty with the audit's own quadrature, which is why the tie to the pipeline is the load-
bearing test and not a formality.

**And it closes a spec question no spec could close.** $-9/8$ factorises as $-\tfrac12\times\tfrac94$:
the $\tfrac94$ is $1/c^2$ with $c=(2+b)/(3+2b)$ (spec 03 §0.1), which KT fold into their $f$; the
sign is spec 04 §0(3)'s orientation convention; and the $\tfrac12$ is the author's
$h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$ relative to Kohri & Terada, which `spec/05-one-loop.md` §539
records as transcribed and unverified and which `spec/cross-spec-check.md` §3 item 9 lists under
**"no spec allows a check"**. The paper allows one, and it now checks out to ten digits.

## 2. Scope

**In scope**

- A new module for the closed form — KT eqs. (22) and (25) — with the `Cin` regularisation the
  resonance needs, and its tests. The prompt chooses the location and justifies it.
- `ComputeTargets/tests/` — the tests, including the $N = -9/8$ tie to the fixture.

**Out of scope, and none of it is to be touched**

- `ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/`.
  This campaign adds an oracle; it does not change the thing being measured. If the oracle and the
  code disagree, that is a finding and a §3 issue, **not** a licence to edit either one.
- `ComputeTargets/tests/test_quadsource_integral.py` — **imported from, never edited**. It is
  `prompts/source-remediation`'s acceptance fixture and `prompts/tolerance-convergence` prompt 06
  ran under the same restriction.
- `config/defaults.py`, `main.py`, any factory or schema, the six `extract_*.py`.
- `docs/spec/` — the audit's finding about spec 05 Q11 is **reported**, not written into the spec.
  The spec is `prompts/spec-transcription`'s and its sign-off procedure is the author's.
- Anything `prompts/tolerance-convergence` still owns: prompt **06a** has not landed.

## 3. The prompt

| # | Prompt | Covers | Model |
|---|---|---|---|
| 01 | [The Kohri–Terada radiation oracle](01-kohri-terada-radiation-oracle.md) | Board item **R1**; opens `[01-general-w-normalisation-is-predicted-not-measured]` | Opus |

## 4. Rules

The invariants in the repository's [`CLAUDE.md`](../../CLAUDE.md) apply unchanged: one commit for
the prompt; a log at `logs/01-<name>.md` classifying every deviation as `STRUCTURALLY REQUIRED`,
`IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`; `IMPLEMENTATION_STATE.md` and
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) updated in that same commit; and no fixing of
things the prompt did not ask for — record those in the log's "Observations not acted on" and open
a §3 issue instead.

Conventions this campaign inherits and must not "correct":

- **$a_0$ is absorbed, never "set to 1".** The mapping is written in $1/k_{\rm phys}^2$ for exactly
  this reason and must stay invariant under $a_0 \to \lambda a_0$ (spec 04 §0(1)).
- **The sign is a convention, not an error.** $N$ is negative because of spec 04 §0(3)'s orientation
  choice. Do not "fix" it into agreement.
- **A number without its reference's own error beside it is not a measurement.** Every figure the
  prompt reports carries the error of whatever it was scored against.

### 4.1 Log format

As `prompts/qsi-phase-groups/README.md` §4.1: front matter with **Prompt**, **Commit**, **Model**,
**Date**, **Result**; then `## What shipped`, `## Deviations from the prompt`,
`## Verification performed` (quote the numbers, not pass/fail), `## Observations not acted on`,
`## State handed to the next prompt`.

## 5. Acceptance

The campaign is done when prompt 01's row is ✅ or ⚠️; the closed form is in the tree with tests that
pin it against `scipy.quad` **and** against the code's own `total` at $N = -9/8$; the resonance is
finite and correct at $u+v=\sqrt3$ rather than `inf`; and `ComputeTargets/tests` and
`CosmologyModels/tests` both pass at no lower a count than the campaign started with.
