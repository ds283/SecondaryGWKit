# Prompt 01 — the Kohri–Terada radiation oracle

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **R1**. **Opens:** `[01-general-w-normalisation-is-predicted-not-measured]`.
**Recommended model:** **Opus** — the code is short and the transcription is unforgiving. Three of
the paper's own statements are wrong in ways that produce a plausible function, and the prompt's
value is entirely in whether the tests would catch a wrong one.

**Read first:**
[`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
**in full** — it is the measurement this prompt implements, and its §2 is three errata you will
otherwise rediscover the hard way;
[`docs/radiation-oracle/kt_verification.py`](../../docs/radiation-oracle/kt_verification.py), which
already contains a correct implementation and is the thing you are productionising;
`docs/spec/04-source-integral.md` §0(1) and §0(2) — the $a_0$ convention and the Green's-function
relation the mapping rests on; `docs/spec/03-source-term.md` §0.1;
`ComputeTargets/tests/test_quadsource_integral.py`'s module docstring, for the fixture you import;
and campaign README §2 and §4.

**It changes no production code and no number.** No file under README §2's out-of-scope list appears
in the diff.

---

## 1. What to build

A module exposing the radiation-era closed form, and nothing more:

- **`I_RD(v, u, x)`** — KT eq. (22), regular at $u+v=\sqrt3$;
- **`I_RD_asymptotic(v, u, x)`** — KT eq. (25), the $x\to\infty$ limit;
- **`Phi(x)`, `dPhi(x)`** — KT eq. (19) and its derivative, with the series branches;
- **`f_RD(v, u, x)`** — the source, eq. (16) at $w=1/3$ (see §2 item 3 on why **not** eq. 20);
- **the normalisation constant** $N = -9/8$ and the mapping, as a documented function
  `total_from_I_RD(...)` or equivalent, so that the relation is in one place rather than
  re-derived in each test.

**Where it lives is your choice and you justify it in the log.** `ComputeTargets/tests/` keeps it
out of the shipped package but makes it awkward to reach from elsewhere; a top-level module treats
it as a library. Say which you picked and why. It is an oracle, not a pipeline component: nothing in
`main.py` calls it.

**Docstrings carry the provenance.** Every function says which KT equation it is, and the three
errata of §2 are recorded at the point where an implementer would trip on them — not only in the
log. A later reader comparing your code against the PDF must find out from your code that the PDF is
wrong, or they will "fix" it back.

## 2. The four things that will go wrong

The audit found all four. They are stated here so that you verify rather than rediscover, and §3's
tests are what prove you got them right.

1. **The Ci and Si arguments pair with the difference, not the sum.** $+\mathrm{Ci}$ and
   $-\mathrm{Si}$ take $(v-u)$; $-\mathrm{Ci}$ and $+\mathrm{Si}$ take $(v+u)$. Inverting them
   produces a smooth, plausible function that agrees with nothing. **Take the arguments from
   `kt_verification.py`, not from a re-reading of the paper**, and if you do re-read it, use the
   ar5iv `alttext` rather than the rendered equation.

2. **The paper's small-$x$ remark is wrong by $4/9$.** It says $I_{\rm RD}\simeq x^2/2$; the truth
   is $2x^2/9$. **Do not use the paper's remark as an acceptance test** — it will reject a correct
   implementation. §3's small-$x$ test asserts $2x^2/9$.

3. **Eq. (20) is numerically unusable at small $x$** — it cancels to $O(x^6)$ against an $x^6$
   denominator and is out by nine orders at $x=0.03$. Build the source from eq. (16) with a
   series-guarded $\Phi$. If you also implement eq. (20), it is for the equivalence test of §3 and
   it is documented as unusable below $x\approx0.1$.

4. **The resonance $u+v=\sqrt3$ must be regularised, not avoided.** $\mathrm{Ci}$ and the $\log$
   each diverge there and their sum does not; use $\mathrm{Cin}(z)=\gamma+\ln z-\mathrm{Ci}(z)$ as
   the audit §2.1 sets out, with its own series branch at small $z$. **A guard that raises, returns
   `nan`, or nudges the argument off the resonance is a failure of this prompt**: the resonance is
   the physically interesting point and an oracle that cannot be evaluated there is not usable where
   it is most wanted.

## 3. The tests

In `ComputeTargets/tests/`, `unittest`, no Ray and no datastore, runnable from the repository root.
Each test says what it is scoring against and that reference's own error.

1. **Against `scipy.quad` of KT eq. (15)**, over at least the audit §5 grid. The audit gets
   **3.5e-15** worst; assert a bound you can defend, not the number you happen to observe, and put
   `quad`'s own declared error in the failure message.
2. **Small $x$: $2x^2/9$**, not $x^2/2$. The audit reaches 0.99999 of $2x^2/9$ at $x=0.01$.
3. **Large $x$: eq. (25)**, approached like $1/x$. Assert the *trend*, not a fixed tolerance at one
   $x$ — the audit shows 1.03e-01 at $x=200$ falling to 7.06e-04 at $x=20000$.
4. **At the resonance exactly.** $u=v=\sqrt3/2$ must return a finite value matching quadrature; the
   audit gets 0.1391877548292 to 4.0e-16 at $x=15$. **This test must fail against an
   unregularised implementation** — check that it does before you keep it, as a test that passes
   either way is testing nothing.
5. **Eq. (16) at $w=1/3$ against eq. (20)**, away from the cancellation region, which is what
   licenses using the first as the second.
6. **The tie to the pipeline: $N = -9/8$.** Import `Case`, `SHAPES`, `X_RESP_VALUES` from
   `test_quadsource_integral.py` and assert the relation on the $b=0$ cases. **There are two
   figures and they are not interchangeable** (audit §1): against a quadrature of KT's integrand
   between the pipeline's own limits the audit gets **3.09e-13**; against eq. (22) itself, with the
   head $0\to\bar x_{\rm min}$ subtracted, it gets **3.18e-10**, the difference being the head
   subtraction and not the agreement. **Test against eq. (22)** — the closed form is the thing being
   landed, and a test that only ever compares against a quadrature would pass even if eq. (22) were
   transcribed wrongly. Assert at the looser bound and say in the log which you used and why.

   **This is also the only test that can catch a misread kernel** (audit §0): tests 1–5 compare two
   of your own transcriptions against each other and share the Green's function, the measure and the
   source. Do not let its cost, which the next paragraph is about, tempt you into dropping it.

   **And it is a Levin test, which is not obvious and is worth saying in the test's own docstring.**
   `total = numeric_quad + WKB_Levin`, and on the nine $b=0$ cases the Levin half runs over 2–3
   regions apiece and carries between **0.099 and 3.846** of $|total|$ — it *exceeds* the total in
   four of them, so the two halves cancel. An error $\varepsilon_L$ in the Levin half enters as
   $\varepsilon_L|L|/|total|$, so the agreement bounds it at **8.0e-14 to 3.1e-12** depending on the
   case: the cancellation tightens the constraint rather than hiding behind it. Three limits on that
   claim, all of which belong in the log rather than being quietly assumed away:
   it bounds the **combination**, so a systematic Levin error cancelling against a compensating
   `numeric_quad` error would pass; it exercises the `total` path and **not** `analytic_rad`'s
   `_three_bessel_Levin`, which is `qsi-phase-groups`' open issue and is untouched by any of this;
   and it is the **exact** flavour only, so the Levin algorithm is scored on exact ingredients with
   no representation floor present.

**Test 6 is the one the campaign exists for and it is also the expensive one.** Nine cases at
`(1e-45, 1e-12)` is minutes, not seconds. If that is too slow for the suite, reduce the *number of
cases*, not the tolerance — a loose tolerance moves what is being measured, whereas fewer cases
measures the same thing less often. Say in the log what you ran and what you kept.

## 4. What this prompt does not do

- **It does not change `QuadSourceIntegral.py` or anything it computes.** If the oracle and the code
  disagree anywhere, that is a §3 issue and a stop, not a repair. The audit found them agreeing on
  every case tried — to 3.09e-13 against a quadrature of the integrand, 3.18e-10 against eq. (22) —
  so a disagreement is new information.
- **It does not edit `test_quadsource_integral.py`.** Import from it. If something you need is local
  to a test method, say so in the log and work around it in your own module.
- **It does not write to `docs/spec/`.** The audit closes what `cross-spec-check.md` §3 item 9 calls
  a check no spec allows, and that is worth recording — but the spec is
  `prompts/spec-transcription`'s and its sign-off is the author's. Report it; do not edit it.
- **It does not extend the oracle to $b\neq0$.** KT give no general-$w$ closed form. The predicted
  $N(b) = -(3+2b)^2/(2(2+b)^2)$ goes in the issue, not the code.

## 5. Acceptance

1. The module exists, `I_RD` matches `scipy.quad` to the bound §3.1 sets, and **is finite and
   correct at $u+v=\sqrt3$**.
2. All six tests of §3 pass, and §3.4's fails against an unregularised implementation.
3. The three errata of §2 are recorded in the code, at the point of use.
4. `ComputeTargets` must not fall below its count at the parent commit; `CosmologyModels` 39. Both
   OK. Record both counts in the log, taken at the parent **and** at your commit.
5. No file under README §2's out-of-scope list is in the diff; `config/defaults.py` byte-identical.
6. `black --check` clean on every `.py` in the diff.
7. Board row 01, item **R1**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and date corrected —
   all in the **same commit**.

## 6. Stop conditions

- **The oracle and the pipeline disagree** on any $b=0$ case by a margin you cannot attribute to
  your own quadrature. **Compare against the right figure**: 3.09e-13 if you scored against a
  quadrature of KT's integrand between the pipeline's limits, **3.18e-10** if you scored against
  eq. (22) with the head subtracted, which §3 test 6 asks you to do. Exceeding 3.09e-13 while
  testing against eq. (22) is expected and is **not** a stop. Report the case and both numbers.
- **You need to edit any file under README §2's out-of-scope list.**
- **The $N=-9/8$ relation does not reproduce.** The audit measured it at `2033cfc`; if it has moved,
  something in the prefactor chain has moved with it and that is a report about the tree.
- **You conclude the regularisation of §2 item 4 is wrong or insufficient.** Say what you found;
  the audit's form is algebraically exact but a better-conditioned one may exist.

## 7. The log

`logs/01-kohri-terada-radiation-oracle.md`, on the campaign's template. Beyond it:

- **Where you put the module and why**, per §1.
- **Which of the four traps of §2 your tests would actually have caught**, tested by breaking the
  implementation deliberately and recording what failed. A test suite that would pass an inverted
  Ci argument has not done its job.
- **The numbers**, each with its reference's own error.
- **What you did about §3's test 6 cost**, and what you kept.
