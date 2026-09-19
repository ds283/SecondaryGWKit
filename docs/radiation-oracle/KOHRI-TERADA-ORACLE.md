# Kohri & Terada as an exact oracle for the source integral

**Measured:** 2026-09-18, on the tree at `2033cfc`, offline (no Ray, no datastore)
**Script:** [`kt_verification.py`](kt_verification.py) — everything from §4 down is its stdout
**Invocation:** `PYTHONPATH=. ./venv/bin/python docs/radiation-oracle/kt_verification.py`
**Paper:** Kazunori Kohri and Takahiro Terada, *Semianalytic calculation of gravitational wave
spectrum nonlinearly induced from primordial curvature perturbations*,
[arXiv:1804.08577](https://arxiv.org/abs/1804.08577), Phys. Rev. D **97**, 123532 (2018)
**Instrument:** `ComputeTargets/tests/test_quadsource_integral.py`, imported and **not edited**
**Grid generation:** *none — this sector has no source grid*; the exact flavour has no sampling
parameter at all

---

## 0. What is being compared, and whose each object is

Three distinct objects appear below. **Two of them are new and were written for this audit**, and
saying which is which is the difference between a cross-check and a tautology. Kohri & Terada
published no code; everything attributed to them here is this audit's transcription of their
published equations.

| | Object | Provenance |
|---|---|---|
| **A** | `evaluate_QuadSource_integral` (`ComputeTargets/QuadSourceIntegral.py:733`), driven offline through the fixture `ComputeTargets/tests/test_quadsource_integral.py` builds | **This repository's pipeline.** Pre-existing, and written with no knowledge of this paper. Called "the code" below, as the specs use that phrase |
| **B** | KT eq. (22), the radiation-era closed form | **This audit's transcription**, `kt_verification.py:I_RD` |
| **C** | `scipy.quad` of KT eq. (15), their defining integral, built from eqs. (16), (18) and (19) | **This audit's transcription**, `kt_verification.py:I_by_quadrature` |

**What each comparison does and does not guard.** B and C share the Green's function
$\sin(x-\bar x)$, the measure $\bar x/x$ and the source $f$, so §5's agreement between them is a
check on the *transcription of eq. (22)* and not on the reading of the kernel:

- **§4 (eq. 16 against eq. 20)** guards the **source**. These are two independent formulas in the
  paper for the same function, so a misreading of $f$ shows up here. It does not.
- **§5 (B against C)** guards **eq. (22)**: given the same kernel and source, the closed form and the
  quadrature agree. A misreading of the kernel would make both wrong together and §5 would still
  pass.
- **§7 (A against C, and §7.1 A against B)** is the only comparison in which anything independent of
  this audit appears, and it is therefore the **only** check on the kernel. A wrong measure or a
  wrong Green's function would not produce a constant $N$ across nine cases spanning $x$ from 4.33
  to 1556 and $u$ from 0.01 to 12; it would drift with $x$. It does not drift.

So §7 is **mutual validation**, and should be read that way in both directions: it is the evidence
that this audit's KT implementation has the kernel right, *and* it is the first absolute accuracy
check the repository's source integral has ever had.

---

## 1. The headline

**The code's source integral reproduces Kohri & Terada's exact radiation-era result on all nine
$b = 0$ fixture cases**, with

$$\text{total} \;=\; -\frac{9}{8}\,\frac{1}{k_{\rm phys}^{2}}\;I_{\rm RD}\big(r/k,\;q/k,\;k\,\tau(z_{\rm resp})\big).$$

The cases span $x = k\tau$ from **4.33 to 1556** and $u = q/k$ from **0.01 to 12**, across all three
fixture shapes and all three response redshifts.

**Two figures, and they measure different things — do not quote the tighter one for the looser
claim.** The code's range starts at $\tau(z_{\rm source\,max})$ and KT's integral starts at
$\bar x = 0$, so there are two ways to score the code and §7 reports the first:

| What is compared | Worst deviation from $-9/8$ | Limited by |
|---|---|---|
| code against **quadrature of KT's integrand between the code's own limits** (§7) | **3.09e-13** | the code's own quadrature |
| code against **KT's closed form eq. (22), minus the head $0\to\bar x_{\rm min}$** (§7.1) | **3.18e-10** | the head subtraction |

> **Corrected 2026-09-18 (§7.2):** the second row is limited by eq. (22)'s own double-precision
> rounding at $u = 0.01$, not by the head subtraction. With eq. (22) at 50 digits it is **3.67e-13**.

The first is the sharper measurement of the *normalisation*; the second is the one that actually
touches eq. (22), and it is three orders looser because the head has to be quadratured and then
subtracted from a number $10^{5}$ times larger. Both give $-9/8$. Neither is a measurement of
eq. (22) itself — that is §5, which scores the closed form against the quadrature directly and gets
**3.5e-15**.

**This is an oracle of a kind the tree did not have.** `analytic_rad`, the code's own closed form, is
a reduction to three-Bessel integrals that are themselves quadratured *at the caller's own
tolerances* (`QuadSourceIntegral.py:1523`, comment at `:1547`) — which is why
`docs/tolerance-convergence/QUADSOURCE-READONLY.md` §2.1 had to freeze it at a reference pair and
why prompt 06's primary statistic was self-convergence rather than an oracle comparison. $I_{\rm RD}$
is Si, Ci, sines and logarithms. It is fixed by construction and moves with no tolerance, and it
shares no machinery with the pipeline it is scoring — it is new code (§0 object **B**), written from
the paper, not a rearrangement of anything the pipeline already does.

### 1.1 What $-9/8$ closes

$-9/8$ factorises as $-\tfrac12 \times \tfrac94$:

- $\tfrac94$ is $1/c^2$ with $c = (2+b)/(3+2b) = 2/3$ at $b = 0$
  ([`spec/03-source-term.md`](../spec/03-source-term.md) §0.1), the constant KT fold into their $f$
  while the code strips it;
- $\tfrac12$ is the author's $h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$ relative to Kohri & Terada;
- the sign is the orientation convention of
  [`spec/04-source-integral.md`](../spec/04-source-integral.md) §0(3), which says in terms that
  "the relative minus sign between the code's source integral and the `MAIN` 14 target is this
  orientation convention".

[`spec/05-one-loop.md`](../spec/05-one-loop.md) §539 records the $h_{ij}$ statement as **transcribed
as written, with the transcriber noting they had not checked it**, and
[`spec/cross-spec-check.md`](../spec/cross-spec-check.md) §3 item 9 lists it among the things where
**"no spec allows a check"**. The paper allows one. It is now checked, numerically, to ten digits,
and the factorisation is exact rather than fitted: $9/8 = 1.125$ and the measurement is
$1.125000000000$.

### 1.2 What it does not cover

**Radiation only.** KT give closed forms for $I$ in pure RD (their eq. 22) and pure MD (eq. 37) and
for nothing in between; §3.3 of the paper treats transitions but does not produce a general-$w$
closed form. The code's $b$ is exactly KT's $w$ reparametrised — `w_of_b(b) = (1-b)/(3(1+b))`, so
$b = 0 \Leftrightarrow w = 1/3$ and $c = (2+b)/(3+2b)$ matches KT's $3(1+w)/(5+3w)$ identically —
but only $b = 0$ has an oracle. The nine $b = 0.2$ fixture cases cannot be checked this way.

The natural extrapolation, **untested**, is

$$N(b) \;=\; -\,\frac{(3+2b)^2}{2\,(2+b)^2},$$

which reproduces $-9/8$ at $b = 0$. It is recorded as
`[01-general-w-normalisation-is-predicted-not-measured]` rather than asserted.

## 2. Three errata in the paper, each found by disagreement with their own eq. (15)

Each was found by the closed form disagreeing with `scipy.quad` of KT's own eq. (15) — object **C**
of §0, not an independent source — and each is stated here because an implementer transcribing from
the PDF will hit all three.

1. **The Ci and Si arguments pair with the difference, not the sum.** In eq. (22) the terms carrying
   $+\mathrm{Ci}$ and $-\mathrm{Si}$ are the $(v-u)$ ones and those carrying $-\mathrm{Ci}$ and
   $+\mathrm{Si}$ are the $(v+u)$ ones. Reading them the other way round — which is the natural
   reading of the rendered equation, where the grouping of $\frac{v\pm u}{\sqrt3}$ inside
   $\left(1\pm\frac{v\pm u}{\sqrt3}\right)$ is easy to invert — gives a function that agrees with
   nothing, misses the quadrature by O(1) at every $x$, and returns $-12.08$ instead of $0$ as
   $x\to0$. The authoritative form is the ar5iv `alttext`, reproduced in `kt_verification.py`.

2. **The stated small-$x$ limit is wrong by $4/9$.** The paper says, below eq. (24), that
   "for small $x$, the leading term is independent of $u$ and $v$, $I_{\rm RD} \simeq x^2/2$". The
   correct limit is $\mathbf{2x^2/9}$. Their own eq. (22) gives $2x^2/9$ and so does direct
   quadrature; §5 below shows both converging on it to ten digits. The elementary check agrees:
   $f \to 4/3$ as $x\to0$, so
   $I \to \tfrac43\int_0^x \tfrac{\bar x}{x}(x-\bar x)\,d\bar x = \tfrac43\cdot\tfrac{x^2}{6}
   = \tfrac{2x^2}{9}$. **The formula is right; only the remark is wrong.** An implementer who uses
   the remark as an acceptance test will reject a correct implementation.

3. **Eq. (20) cannot be evaluated at small $x$ in double precision.** Its bracket cancels to
   $O(x^6)$ against an explicit $x^6$ denominator, so it returns noise below $x \approx 0.1$: used
   as the integrand of eq. (15) it puts the integral out by **nine orders** at $x = 0.03$
   ($4.5\times10^{5}$ against a true $2.0\times10^{-4}$). Eq. (16) with a series-guarded $\Phi$ has
   no such cancellation and is what should be integrated. §3 is what licenses the substitution.

### 2.1 And one thing that is not an erratum

**The resonance $u+v=\sqrt3$ is removable, but eq. (22) as written does not remove it.** At the
resonance $\mathrm{Ci}\big(|1-\tfrac{u+v}{\sqrt3}|x\big)$ and
$\log\big|\tfrac{3-(u+v)^2}{3-(u-v)^2}\big|$ each diverge and their sum does not, so a literal
transcription returns $\pm\infty$ on the resonance and loses accuracy approaching it. Writing
$\mathrm{Cin}(z) = \gamma + \ln z - \mathrm{Ci}(z)$, which is analytic and starts at $z^2/4$,

$$-\mathrm{Ci}(|c|x) + \log\left|\frac{3-(u+v)^2}{3-(u-v)^2}\right|
 = -\gamma - \ln x + \mathrm{Cin}(|c|x) + \log e - \log|ab|$$

with $c = 1-\tfrac{v+u}{\sqrt3}$, $e = 1+\tfrac{v+u}{\sqrt3}$, $a = 1-\tfrac{v-u}{\sqrt3}$,
$b = 1+\tfrac{v-u}{\sqrt3}$. The $\ln|c|$ cancels **analytically**, and §6 shows the regularised form
finite *at* the resonance and matching quadrature to 4.0e-16 there. This is an implementation
requirement, not a correction to the paper: the resonance is the physically interesting point (it is
the Kohri–Terada resonance the scaffolding documents are built around) and an oracle that returns
`inf` there is not usable at the one place it is most wanted.

## 3. The mapping, and why it is forced

From [`spec/04-source-integral.md`](../spec/04-source-integral.md) §0(2), the code's Green's function
is $G_{\rm code}(z,z') = -a_0H(z')\,\mathrm{Gr}_k(\eta(z),\eta(z'))$, and KT's $G_k$ of their eq. (10)
**is** $\mathrm{Gr}_k$ — same operator, same $+\delta$ source, same unit jump in $d/d\eta$. The code's
target is

$$\text{total} = (1+z_{\rm resp})\int d\log(1+z')\;G_{\rm code}(z_{\rm resp},z')\,\frac{f(z')}{H(z')^2}$$

(`QuadSourceIntegral.py:756`). Substituting $d\log(1+z') = -H a_0\,d\eta'/(1+z')$ and
$(1+z_{\rm resp})/(1+z') = a(\bar\eta)/a(\eta)$, the two minus signs cancel and so do the two powers
of $H$, leaving $a_0^2\int d\eta'\,\frac{a(\bar\eta)}{a(\eta)}\mathrm{Gr}_k f_{\rm code}$. Against
KT eq. (15), $I = k^2\int d\bar\eta\,\frac{a}{a}\mathrm{Gr}_k f_{\rm KT}$, so

$$\text{total} = \frac{a_0^2}{k^2}\cdot\frac{f_{\rm code}}{f_{\rm KT}}\cdot I_{\rm RD},\qquad
a_0^2/k^2 = 1/k_{\rm phys}^2 .$$

$1/k_{\rm phys}^2$ is $a_0$-invariant, so the mapping passes the covariance test
[`spec/04-source-integral.md`](../spec/04-source-integral.md) §0(1) imposes: $a_0$ is absorbed, not
set to one, and no stray power survives. Everything except the pure number $f_{\rm code}/f_{\rm KT}$
is fixed by the specs; that number is what §7 measures.

The lower limit differs — KT integrate from $\bar x = 0$, the code from $\tau(z_{\rm source\,max})$ —
so §7 scores against a quadrature between **the code's own limits**, and the truncation is not
charged to $N$.

---

<!-- everything below is kt_verification.py's stdout, 2026-09-18 -->

## 4. KT eq. (16) at `w = 1/3` against KT eq. (20)

The general-`w` source against the explicit radiation-era one. They share no code.

| v | u | x | eq. (16) | eq. (20) | rel |
|---|---|---|---|---|---|
| 0.7 | 1.1 | 0.5 | +1.302182785045e+00 | +1.302182785043e+00 | 1.54e-12 |
| 0.7 | 1.1 | 2.0 | +9.075894098121e-01 | +9.075894098121e-01 | 1.35e-15 |
| 0.7 | 1.1 | 7.3 | +1.360492305650e-01 | +1.360492305650e-01 | 2.04e-16 |
| 0.7 | 1.1 | 31.0 | +1.682621639467e-03 | +1.682621639467e-03 | 3.87e-16 |
| 0.7 | 1.1 | 100.0 | +3.720068933855e-04 | +3.720068933855e-04 | 0.00e+00 |
| 1.0 | 1.0 | 0.5 | +1.296786708319e+00 | +1.296786708319e+00 | 3.60e-13 |
| 1.0 | 1.0 | 2.0 | +8.540082525516e-01 | +8.540082525516e-01 | 1.30e-15 |
| 1.0 | 1.0 | 7.3 | +2.298614013349e-01 | +2.298614013349e-01 | 1.21e-16 |
| 1.0 | 1.0 | 31.0 | +6.951289815464e-03 | +6.951289815464e-03 | 1.25e-16 |
| 1.0 | 1.0 | 100.0 | +1.059161375103e-03 | +1.059161375103e-03 | 2.05e-16 |
| 0.3 | 1.6 | 0.5 | +1.284840276314e+00 | +1.284840276314e+00 | 6.31e-14 |
| 0.3 | 1.6 | 2.0 | +6.841752767931e-01 | +6.841752767931e-01 | 7.14e-15 |
| 0.3 | 1.6 | 7.3 | +3.568617627551e-02 | +3.568617627551e-02 | 1.75e-15 |
| 0.3 | 1.6 | 31.0 | +5.313656071257e-03 | +5.313656071257e-03 | 0.00e+00 |
| 0.3 | 1.6 | 100.0 | +2.373358907119e-03 | +2.373358907119e-03 | 0.00e+00 |
| 1.2 | 0.5 | 0.5 | +1.302322676407e+00 | +1.302322676409e+00 | 1.58e-12 |
| 1.2 | 0.5 | 2.0 | +9.005706389958e-01 | +9.005706389958e-01 | 3.33e-15 |
| 1.2 | 0.5 | 7.3 | -3.044104344623e-02 | -3.044104344623e-02 | 9.12e-16 |
| 1.2 | 0.5 | 31.0 | +2.288666328927e-03 | +2.288666328927e-03 | 3.79e-16 |
| 1.2 | 0.5 | 100.0 | -2.410111732667e-04 | -2.410111732667e-04 | 1.12e-16 |
| 0.9 | 0.87 | 0.5 | +1.304618001703e+00 | +1.304618001704e+00 | 9.12e-13 |
| 0.9 | 0.87 | 2.0 | +9.402189990350e-01 | +9.402189990350e-01 | 2.36e-16 |
| 0.9 | 0.87 | 7.3 | +2.624509213319e-01 | +2.624509213319e-01 | 0.00e+00 |
| 0.9 | 0.87 | 31.0 | +4.981656814431e-05 | +4.981656814431e-05 | 3.81e-15 |
| 0.9 | 0.87 | 100.0 | +5.605040177786e-06 | +5.605040177786e-06 | 3.78e-15 |

**Worst relative disagreement: 1.579e-12.** The general-`w` transcription is right, and with it the reading of `xbar d_etabar Phi` as `xbar d_xbar Phi`.

## 5. KT eq. (22) against `scipy.quad` of KT eq. (15)

The load-bearing check. The closed form knows nothing of the quadrature.

| v | u | x | eq. (22) | quadrature | rel | quad's own error |
|---|---|---|---|---|---|---|
| 0.7 | 1.1 | 1.0 | +2.053469779911e-01 | +2.053469779911e-01 | 1.35e-15 | 2.3e-15 |
| 0.7 | 1.1 | 5.0 | +3.520250466749e-01 | +3.520250466749e-01 | 1.10e-15 | 8.7e-15 |
| 0.7 | 1.1 | 20.0 | -2.023764954439e-01 | -2.023764954439e-01 | 0.00e+00 | 5.3e-15 |
| 0.7 | 1.1 | 60.0 | +1.593532404202e-01 | +1.593532404202e-01 | 5.23e-16 | 3.0e-15 |
| 1.0 | 1.0 | 10.0 | +2.416399669992e-01 | +2.416399669992e-01 | 6.89e-16 | 8.7e-15 |
| 0.3 | 1.6 | 12.0 | -2.570546556294e-01 | -2.570546556294e-01 | 2.38e-15 | 4.1e-15 |
| 1.2 | 0.5 | 25.0 | -1.770406660654e-01 | -1.770406660654e-01 | 1.25e-15 | 4.4e-15 |
| 0.9 | 0.87 | 40.0 | +4.501650138327e-02 | +4.501650138327e-02 | 3.55e-15 | 4.3e-15 |
| 0.86 | 0.87 | 15.0 | +1.368699956249e-01 | +1.368699956249e-01 | 1.62e-15 | 8.5e-15 |
| 0.9 | 0.9 | 15.0 | +2.026653071225e-01 | +2.026653071225e-01 | 8.22e-16 | 8.2e-15 |

**Worst relative disagreement: 3.545e-15.**

## 6. The limits, and the paper's small-`x` remark

KT state below eq. (24) that `I_RD ~ x^2/2` for small `x`. Both their own closed form and the quadrature give `2 x^2 / 9`, a factor `4/9` smaller. The formula is right; the remark is not.

| x | eq. (22) | quadrature | ratio to `x^2/2` | ratio to `2x^2/9` |
|---|---|---|---|---|
| 0.3 | +1.9859393962e-02 | +1.9859393962e-02 | 0.441320 | 0.9929696981 |
| 0.1 | +2.2204820229e-03 | +2.2204820226e-03 | 0.444096 | 0.9992169103 |
| 0.03 | +1.9998591558e-04 | +1.9998590039e-04 | 0.444413 | 0.9999295779 |
| 0.01 | +2.2221913772e-05 | +2.2222048149e-05 | 0.444438 | 0.9999861197 |

And the large-`x` limit against eq. (25), which should approach it like `1/x`:

| v | u | x | eq. (22) | eq. (25) | rel |
|---|---|---|---|---|---|
| 0.7 | 1.1 | 200.0 | -1.7946266745e-02 | -2.0013946052e-02 | 1.03e-01 |
| 0.7 | 1.1 | 2000.0 | +1.4761120238e-03 | +1.4712387968e-03 | 3.30e-03 |
| 0.7 | 1.1 | 20000.0 | -3.6260339761e-04 | -3.6285971249e-04 | 7.06e-04 |
| 1.2 | 0.5 | 200.0 | +3.5260522628e-02 | +3.3706875791e-02 | 4.41e-02 |
| 1.2 | 0.5 | 2000.0 | -3.5309303603e-03 | -3.5896969057e-03 | 1.64e-02 |
| 1.2 | 0.5 | 20000.0 | -2.2487801076e-04 | -2.2463012478e-04 | 1.10e-03 |
| 0.3 | 1.6 | 200.0 | -2.1598600071e-02 | -2.3484330103e-02 | 8.03e-02 |
| 0.3 | 1.6 | 2000.0 | +2.3164459896e-03 | +2.3034746130e-03 | 5.60e-03 |
| 0.3 | 1.6 | 20000.0 | +8.0779496812e-06 | +8.0182662871e-06 | 7.39e-03 |

### 6.1 The resonance `u + v = sqrt(3)`

Eq. (22) as written returns `+-inf` exactly at the resonance and loses accuracy approaching it: `Ci` and the `log` each diverge and their sum does not. With the `Cin` regularisation of this module the closed form is finite AT the resonance and matches the quadrature there.

| u + v - sqrt(3) | eq. (22), regularised | quadrature | rel |
|---|---|---|---|
| +1.0e-02 | +1.496913642140e-01 | +1.496913642140e-01 | 4.08e-15 |
| +1.0e-04 | +1.392944452597e-01 | +1.392944452597e-01 | 1.20e-15 |
| +1.0e-06 | +1.391888218935e-01 | +1.391888218935e-01 | 5.98e-16 |
| +1.0e-08 | +1.391877654998e-01 | +1.391877654998e-01 | 9.97e-16 |
| +0.0e+00 | +1.391877548292e-01 | +1.391877548292e-01 | 3.99e-16 |

## 7. The code's `total` against the oracle

From spec 04 section 0 (2), `G_code(z,z') = -a0 H(z') Gr_k`, and KT's `G_k` IS `Gr_k`; with `dlog(1+z') = -H a0 deta'/(1+z')` and `(1+z_resp)/(1+z') = a(etabar)/a(eta)` the two signs and the two powers of `H` cancel, leaving

        total = N * I_RD(r/k, q/k, k tau(z_resp)) / k_phys^2

with `N` a pure number. `a0^2/k^2 = 1/k_phys^2` passes the spec's `a0`-covariance test. `I_trunc` integrates between the code's OWN limits so that the code's finite `z_source_max` is not charged to `N`.

| shape | x_resp | u = q/k | v = r/k | x = k tau | code `total` | `I_trunc` | N |
|---|---|---|---|---|---|---|---|
| together | 980 | 0.9091 | 1.0909 | 1.5560e+03 | +9.2632299e-13 | -9.963118e-05 | **-1.125000000000** |
| together | 100 | 0.9091 | 1.0909 | 1.5877e+02 | -1.4546904e-10 | +1.564600e-02 | **-1.125000000000** |
| together | 30 | 0.9091 | 1.0909 | 4.7631e+01 | -1.9958706e-10 | +2.146670e-02 | **-1.125000000000** |
| T-first | 980 | 10.0000 | 12.0000 | 1.4145e+02 | +7.4690422e-11 | -6.639149e-05 | **-1.125000000000** |
| T-first | 100 | 10.0000 | 12.0000 | 1.4434e+01 | -8.4352153e-09 | +7.497969e-03 | **-1.125000000000** |
| T-first | 30 | 10.0000 | 12.0000 | 4.3301e+00 | +2.3437795e-08 | -2.083360e-02 | **-1.125000000000** |
| q-smooth | 980 | 0.0100 | 1.1000 | 1.5431e+03 | -8.5701377e-12 | +7.617900e-04 | **-1.125000000000** |
| q-smooth | 100 | 0.0100 | 1.1000 | 1.5746e+02 | +1.5578804e-10 | -1.384783e-02 | **-1.125000000000** |
| q-smooth | 30 | 0.0100 | 1.1000 | 4.7238e+01 | +7.4905262e-10 | -6.658246e-02 | **-1.125000000000** |

**N is constant at -1.125 = -9/8**: min -1.125000000000, max -1.125000000000, spread 4.67e-13 about the mean -1.125000000000. Worst deviation from -9/8 over the nine cases: 3.09e-13.

`-9/8` factorises as `-(1/2) * (9/4)`. The `9/4` is `1/c^2` with `c = (2+b)/(3+2b) = 2/3` at `b = 0` (spec 03 section 0.1), which KT fold into their `f`. The `1/2` is the author's `h^us_ij = h^them_ij / 2` relative to Kohri-Terada. The sign is the orientation convention of spec 04 section 0 (3).

### 7.1 The same nine cases against eq. (22) itself

The table above scores the code against a QUADRATURE of KT's integrand. This one scores it against the closed form, with the head `0 -> x_min` subtracted. It is the only place the code and eq. (22) are put side by side, and it is three orders looser -- not because the agreement is worse, but because the head must be quadratured and then subtracted from a number about 1e5 times larger.

| shape | x_resp | code `total` | eq. (22) - head | head / eq. (22) | N |
|---|---|---|---|---|---|
| together | 980 | +9.26322986810291e-13 | -9.96311834702685e-05 | 1.25e-04 | **-1.125000000000** |
| together | 100 | -1.45469041491689e-10 | +1.56460035737728e-02 | 1.02e-05 | **-1.125000000000** |
| together | 30 | -1.99587057601236e-10 | +2.14666968619996e-02 | 1.20e-05 | **-1.125000000000** |
| T-first | 980 | +7.46904219832232e-11 | -6.63914862073081e-05 | 1.76e-06 | **-1.125000000000** |
| T-first | 100 | -8.43521529922964e-09 | +7.49796915487059e-03 | 1.86e-06 | **-1.125000000000** |
| T-first | 30 | +2.34377947708663e-08 | -2.08335953518810e-02 | 2.16e-06 | **-1.125000000000** |
| q-smooth | 980 | -8.57013770491575e-12 | +7.61790017999320e-04 | 1.15e-05 | **-1.125000000318** |
| q-smooth | 100 | +1.55788036198175e-10 | -1.38478254378743e-02 | 4.21e-06 | **-1.125000000160** |
| q-smooth | 30 | +7.49052621673936e-10 | -6.65824552569145e-02 | 8.71e-07 | **-1.125000000051** |

**Worst deviation from -9/8 against the closed form: 3.18e-10**, against 3.09e-13 against the quadrature. Both give -9/8.

### 7.2 What limits §7.1: eq. (22)'s own rounding, not the head *(added 2026-09-18)*

**This corrects an attribution; no measurement above changes.** §1's table ("Limited by: the head
subtraction") and §7.1's preamble ("because the head must be quadratured and then subtracted from a
number about 1e5 times larger") both attribute the 3.18e-10 to the head $0\to\bar x_{\rm min}$. That is
wrong. The head is good to $\le1.4\times10^{-18}$ of $I$ on every case, declared or actual. The whole
3.18e-10 is **eq. (22)'s own double-precision rounding on the `q-smooth` shape**: at $u = q/k = 0.01$
the $1/(u^3v^3)$ prefactor is $\sim10^6$, and it multiplies terms that nearly cancel. With eq. (22)
evaluated at 50 digits, the code agrees with Kohri & Terada's closed form to **3.67e-13**. That is
the same order as §7's 3.09e-13 against the quadrature. The two comparisons are therefore **not**
three orders apart in what they say about the code; the gap was the reference's rounding.

The finding is prompt 01's (`prompts/radiation-oracle/logs/01-kohri-terada-radiation-oracle.md`,
Result and §3.1). The table below re-measures it independently. It uses a 50-digit `mpmath`
transcription of eq. (22) that shares no code with `kt_verification.py` or
`ComputeTargets/tests/kohri_terada.py`, and a 50-digit head. The "as §7.1" column uses
`kt_verification.I_RD` and §7.1's own `quad` call for the head, so it reproduces §7.1's $N$ exactly.
"eq. (22) error" is `kt_verification.I_RD`'s relative error; the head columns are relative to $I$.
Tree `63511aa`; `PYTHONPATH=. ./venv/bin/python docs/radiation-oracle/eq22_rounding.py` prints what
follows (4 s).

<!-- generated by docs/radiation-oracle/eq22_rounding.py -->

| shape | x_resp | u | x | eq. (22) error | head / I | head's quad error | head error | N - (-9/8), as §7.1 | N - (-9/8), exact eq. (22) |
|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 0.9091 | 1.5560e+03 | 9.96e-16 | 1.25e-04 | 1.4e-18 | 3.6e-19 | +6.88e-14 | +6.99e-14 |
| together | 100 | 0.9091 | 1.5877e+02 | 4.21e-16 | 1.02e-05 | 1.1e-19 | 1.4e-21 | -2.00e-15 | -2.66e-15 |
| together | 30 | 0.9091 | 4.7631e+01 | 2.60e-15 | 1.20e-05 | 1.3e-19 | 1.2e-20 | -1.33e-15 | -4.44e-15 |
| T-first | 980 | 10 | 1.4145e+02 | 3.72e-16 | 1.76e-06 | 2.0e-20 | 1.1e-19 | -2.42e-14 | -2.38e-14 |
| T-first | 100 | 10 | 1.4434e+01 | 3.34e-17 | 1.86e-06 | 2.1e-20 | 3.3e-22 | -2.98e-14 | -3.00e-14 |
| T-first | 30 | 10 | 4.3301e+00 | 5.40e-16 | 2.16e-06 | 2.4e-20 | 3.7e-22 | -1.07e-14 | -1.13e-14 |
| q-smooth | 980 | 0.01 | 1.5431e+03 | 2.82e-10 | 1.15e-05 | 1.3e-19 | 3.0e-19 | -3.18e-10 | -3.67e-13 |
| q-smooth | 100 | 0.01 | 1.5746e+02 | 1.42e-10 | 4.21e-06 | 4.7e-20 | 8.2e-21 | -1.60e-10 | -2.71e-14 |
| q-smooth | 30 | 0.01 | 4.7238e+01 | 4.49e-11 | 8.71e-07 | 9.7e-21 | 4.1e-21 | -5.05e-11 | +1.04e-14 |

Worst |N + 9/8| with section 7.1's double-precision eq. (22) and head: **3.18e-10**. With eq. (22) and the head both at 50 digits: **3.67e-13**. The head's error, declared or actual, never exceeds **1.4e-18** of I.

`ComputeTargets/tests/kohri_terada.py` sums eq. (22)'s terms with `math.fsum`. Its error at the three
`q-smooth` points is 9.7e-11, 6.6e-11 and 2.5e-11, against the 2.8e-10, 1.4e-10 and 4.5e-11 above,
which is why `test_kohri_terada_oracle`'s test 6 measures 1.08e-10 where §7.1 has 3.18e-10. Either
way the limit is the closed form at small $u$. A form of eq. (22) expanded in small $u$ would remove
it; nothing here needs one.

## 8. The pipeline at large x *(added 2026-09-19)*

**Why this section exists.** §7 and prompt 01's test 6 stop at $x \approx 1.6\times10^3$, because
the fixture's Liouville–Green stand-ins do ("the LG fixtures end at x = 1e3",
`test_quadsource_integral.py:330`). That is only a few hundred periods, and plain quadrature can
still act as referee there (Table 8.2). The Liouville–Green representation and Levin quadrature
exist for a much larger regime. Production takes source $k$ up to $3\times10^8\,{\rm Mpc}^{-1}$
(`main.py`) to $z_{\rm end} = 0.1$ (`DEFAULT_ZEND`), which is $x$ of order $4\times10^{12}$ (our
estimate, from $a_0\tau \approx 1.4\times10^4$ Mpc at $z = 0.1$), or about $10^{12}$ periods.
Eq. (22) costs the same at any $x$, so it can referee the pipeline far beyond where quadrature can.

**How.** `docs/radiation-oracle/large_x.py` runs the real `evaluate_QuadSource_integral`, in the
**exact** flavour, at the reference pair `(1e-45, 1e-12)`, at $x_{\rm resp}$ up to $10^8$. No
repository file is changed. $k$, $q$ and $r$ are multiplied by a common factor
$\lambda = x_{\rm resp}/980$, which leaves $u$ and $v$ and hence $I(v,u,x)$ unchanged and keeps
the response time at the same redshift. The fixture cache is seeded with `Fixture` objects whose
Liouville–Green region reaches it. Every case is scored against eq. (22) and the head, both at 50
digits (`eq22_rounding.py`), so §7.2's small-$u$ rounding does not enter. Tree `bcdb459` plus
the script, on an
Apple M1 Pro, one core; `PYTHONPATH=. ./venv/bin/python docs/radiation-oracle/large_x.py` prints
what follows (about 70 s).

<!-- generated by docs/radiation-oracle/large_x.py -->

**Table 8.1 -- the pipeline against eq. (22).**

| shape | x_resp | lam | x = k tau | z_resp | N + 9/8 | pipeline's declared error | N + 9/8, head omitted | Levin / abs(total) | eq. (22) vs eq. (25) | integral | fixture set-up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 1 | 1.5560e+03 | 6.07 | +6.99e-14 | 8.7e-12 | +1.41e-04 | 3.85 | 9.7e-02 | 0.54 s | 0.0 s |
| together | 10000 | 10.2 | 1.5877e+04 | 6.07 | +2.89e-15 | 1.3e-12 | +4.30e-06 | 0.13 | 1.9e-04 | 0.48 s | 0.0 s |
| together | 100000 | 102 | 1.5877e+05 | 6.07 | +1.72e-13 | 1.7e-11 | +1.39e-05 | 0.39 | 6.3e-05 | 0.71 s | 0.0 s |
| together | 1e+06 | 1.02e+03 | 1.5877e+06 | 6.07 | -1.56e-12 | 1.4e-10 | +4.89e-06 | 0.15 | 3.9e-06 | 0.60 s | 0.0 s |
| together | 1e+07 | 1.02e+04 | 1.5877e+07 | 6.07 | -3.10e-10 | 9.9e-09 | -3.31e-05 | 0.86 | 1.5e-06 | 0.60 s | 0.0 s |
| together | 1e+08 | 1.02e+05 | 1.5877e+08 | 6.07 | -1.46e-10 | 3.5e-08 | -4.10e-06 | 0.09 | 9.9e-09 | 0.54 s | 0.0 s |
| T-first | 980 | 1 | 1.4145e+02 | 6.07 | -2.38e-14 | 4.8e-12 | +1.98e-06 | 2.89 | 1.1e-01 (no cos term) | 0.18 s | 0.0 s |
| T-first | 10000 | 10.2 | 1.4434e+03 | 6.07 | -2.22e-16 | 1.0e-12 | +2.24e-06 | 0.15 | 4.8e-05 (no cos term) | 0.26 s | 0.0 s |
| T-first | 100000 | 102 | 1.4434e+04 | 6.07 | +6.48e-14 | 1.6e-12 | +2.24e-06 | 0.18 | 8.9e-05 (no cos term) | 0.29 s | 0.0 s |
| T-first | 1e+06 | 1.02e+03 | 1.4434e+05 | 6.07 | -2.31e-12 | 7.8e-11 | +2.24e-06 | 1.11 | 1.9e-05 (no cos term) | 0.28 s | 0.0 s |
| T-first | 1e+07 | 1.02e+04 | 1.4434e+06 | 6.07 | +4.67e-12 | 3.1e-10 | +2.24e-06 | 0.13 | 1.0e-07 (no cos term) | 0.33 s | 0.0 s |
| q-smooth | 980 | 1 | 1.5431e+03 | 5.48 | -3.67e-13 | 5.2e-12 | -1.30e-05 | 2.10 | 3.8e-02 | 0.25 s | 0.0 s |
| q-smooth | 10000 | 10.2 | 1.5746e+04 | 5.48 | -1.01e-13 | 1.0e-11 | -1.22e-05 | 4.70 | 1.3e-02 | 0.24 s | 0.0 s |
| q-smooth | 100000 | 102 | 1.5746e+05 | 5.48 | -3.87e-13 | 2.1e-11 | -1.27e-05 | 1.62 | 3.5e-04 | 0.49 s | 0.0 s |
| q-smooth | 1e+06 | 1.02e+03 | 1.5746e+06 | 5.48 | +2.26e-13 | 5.4e-10 | -1.24e-05 | 4.45 | 5.0e-04 | 0.24 s | 0.0 s |
| q-smooth | 1e+07 | 1.02e+04 | 1.5746e+07 | 5.48 | -6.49e-13 | 4.9e-09 | -1.27e-05 | 2.59 | 4.1e-05 | 0.26 s | 0.0 s |

**Table 8.2 -- plain `scipy.quad` of eq. (15), together shape (scipy 1.15.2).**

| x | periods of the fastest oscillation | source evaluations | time | error against eq. (22) | quad's declared error / abs(I) |
|---|---|---|---|---|---|
| 1e+02 | 18 | 1596 | 0.00 s | 1.1e-15 | 4.7e-14 |
| 1e+03 | 184 | 11844 | 0.02 s | 1.2e-13 | 1.0e-12 |
| 1e+04 | 1838 | 89061 | 0.12 s | 3.2e-13 | 1.6e-11 |
| 3e+04 | 5513 | 200235 | 0.26 s | 1.8e-02 | 1.4e+00 |
| 1e+05 | 18378 | 127869 | 0.19 s | 4.1e-03 | 7.5e-02 |

**Table 8.3 -- `wrap_theta` at large theta.**

| theta | time per call | error of wrap_theta | error of theta - div * 2pi | theta * eps |
|---|---|---|---|---|
| 1.000e+03 | 0.0003 ms | 0.0e+00 | 0.0e+00 | 2.2e-13 |
| 1.000e+05 | 0.0004 ms | 0.0e+00 | 3.7e-12 | 2.2e-11 |
| 1.000e+07 | 0.0004 ms | 0.0e+00 | 7.3e-10 | 2.2e-09 |

**What the tables show.**

1. **The integral's cost does not grow with $x$.** One $(k, q, r)$ integral takes **0.2–0.7 s** of
   wall time on one core, from $x = 1.4\times10^2$ to $1.6\times10^8$, including building the
   Bessel-phase splines it needs. This is the working figure for estimating the cost of a
   one-loop evaluation scheme: the number of $(k, q, r)$ triples it needs, times about half a
   second, at these tolerances and with exact ingredients. Plain quadrature (Table 8.2) costs
   50–90 source evaluations per period. It loses accuracy as $x$ grows, from cancellation across
   periods, and fails at $x \approx 3\times10^4$: the breakpoint list is
   capped at 4,000 there, and quad's 200-subdivision limit runs out. It does report the failure,
   with a declared error of order $\lvert I\rvert$.

2. **$N = -9/8$ holds at every $x$ tried.** $\lvert N + 9/8\rvert \le 6.5\times10^{-13}$ on
   `q-smooth` to $x = 1.6\times10^7$, $\le 4.7\times10^{-12}$ on `T-first` to $1.4\times10^6$,
   and $\le 3.1\times10^{-10}$ on `together` to $1.6\times10^8$. Every deviation is below the
   pipeline's own declared error, by a factor of at least 14. That declared error grows roughly like
   $x\epsilon$ above $x \sim 10^5$, which is the price of a phase that is known only to relative
   precision $\epsilon$: $x\epsilon$ is $4\times10^{-8}$ at $x = 1.6\times10^8$, and would be
   of order $10^{-3}$ at production's $4\times10^{12}$.

3. **The head does not become negligible as $x$ grows.** Kohri & Terada integrate from
   $\bar x = 0$, i.e. from $z = \infty$; the pipeline starts at `z_source_max`, five e-folds
   outside the horizon. Omitting the head leaves $N$ off by $2\times10^{-6}$ to
   $1.4\times10^{-4}$ at **every** $x$. At large $x$ both the head,
   $\approx \tfrac23\bar x_{\rm min}^2 \sin x/x$, and $I$ fall like $1/x$. Their ratio is set by
   $\bar x_{\rm min}^2 = (k\tau_{\rm start})^2$, about $(6\times10^{-3})^2$ for `together`, which is how early the
   integral starts, not how late it ends.

4. **Eq. (22) approaches eq. (25) — except where eq. (25) does not apply.** For `together` the
   relative difference falls from $10^{-1}$ at $x = 1.6\times10^3$ to $10^{-8}$ at
   $1.6\times10^8$. Single-$x$ figures are dominated by where the zeros fall; §6 and prompt 01's
   test 3 measure the trend. **Eq. (25) as printed assumes $\lvert v - u\rvert < \sqrt3$.** As
   $x\to\infty$, $\mathrm{Si}(ax)\to\operatorname{sign}(a)\,\pi/2$. For
   $\lvert v-u\rvert > \sqrt3$ the argument $1-(v-u)/\sqrt3$ is negative, the four Si terms'
   limits cancel, and eq. (25)'s $-\pi d\,\Theta(v+u-\sqrt3)\cos x$ term is absent. `T-first`
   ($v - u = 2$) is such a case, and the table scores it against eq. (25) with the cos term
   dropped; with the term kept the difference is $O(1)$ at every $x$. Eq. (22) is unaffected: it
   takes Ci of $\lvert\cdot\rvert$ and Si is odd, and test 1 already checks it at a `T-first`
   point. No physical configuration is affected either: momentum conservation requires
   $\lvert v - u\rvert \le 1$. `T-first` is a test shape that is not a closable triangle, with
   $\lvert q - r\rvert = 2000 > k = 1000$.

5. **The fixture set-up time does not grow with $x$, and phase reduction is exact.** Set-up is
   0.0 s at every row of Table 8.1, including $x = 1.6\times10^8$, and the whole script runs in
   about nine seconds. Both are consequences of how the phase is range-reduced.
   `Fixture.exact_functions()` passes the full, unreduced Bessel phase -- about $x$ -- at each of
   its roughly 570 WKB samples, and that phase is reduced by `WKB_mod_2pi`, whose remainder is
   an `fmod`: one step, exact, independent of $|\theta|$. `LiouvilleGreen.WKBtools.wrap_theta`
   reduces the same way (Table 8.3): 0 error and ~0.3 µs per call at $\theta$ from $10^3$ to
   $10^7$. **Production is unaffected either way.** It calls `wrap_theta` only through
   `apply_phase_offset`, on `mod + deltaTheta` with `mod` in $(-2\pi, 0]$ and `deltaTheta` from
   `atan2`, so it never reduces by more than one cycle. What Table 8.3's fourth column still
   warns against is **reassembling** the phase: recomputing the remainder as the double
   expression $\theta - {\rm div}\cdot2\pi$ rounds twice and is $7.3\times10^{-10}$ rad out at
   $\theta = 10^7$, which is why the $({\rm div}, {\rm mod})$ pair is carried rather than
   rebuilt.

**What this does not cover.** The exact flavour only: the realistic flavour's representation
floors (splined phases and amplitudes) are not exercised at large $x$. $b = 0$ only (§1.2).
And not production's $x$: the largest here is $1.6\times10^8$, four decades short of
$4\times10^{12}$, where $x\epsilon$ alone is of order $10^{-3}$ for any method that carries the
phase as a double.
