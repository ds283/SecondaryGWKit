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
