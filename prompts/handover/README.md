# The hand-over campaign — the numeric→Liouville–Green seam of $T_k$ and $G_k$

**Board:** `IMPLEMENTATION_STATE.md` (not yet written — created by prompt 01) ·
**Background:** [`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md) ·
[`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.1

> **Status: started.** Prompt 00 has landed; prompts 01, 02 and 03 are written and not run; the
> rest of §3 is groupings only. This README defines the campaign and its prompt groupings; it does
> not write the prompt files. (Before 2026-09-19 this block read "planned, not started", which
> prompt 00's landing and the workstream A prompts made stale.)

---

## 0. What this campaign is, and its boundaries

### 0.1 The one-sentence version

`TkNumericIntegration` stops at a phase extremum a little below 3 e-folds sub-horizon and
`TkWKBIntegration` starts there (and `GkNumericIntegration` → `GkWKBIntegration` likewise); the
seam that joins them has a **structural gap** that is bridged by holding a phase constant across up
to 0.44 rad, sits at a depth **nobody has ever optimised**, and is located by a root solve whose
answer is then used as an exact-float datastore key. This campaign removes the gap, makes the depth
a measured choice, and turns the stop point into an identity that cannot silently mismatch.

### 0.2 Why this is one campaign and not eight issues

`docs/OPEN_ISSUES.md` §1.1 has said since 2026-09-10 that these are **one place** and must be
attacked together, because `prompts/source-remediation` prompt 12 could not separate their
contributions by measurement alone at production $x$. That remains true, and §1 below says what
changed in 2026-09-18/19 that now makes the separation possible.

### 0.3 What this campaign does *not* do

- It does not touch `AdaptiveLevin/`, the Clenshaw–Curtis fallback, or the Levin cost issues
  (`docs/OPEN_ISSUES.md` §1.3). A seam change alters *what* is integrated, not *how*.
- It does not retune any accuracy parameter that `docs/TOLERANCE-PROVENANCE.md` records.
  `prompts/tolerance-convergence` README §6.2's rule-8 freeze holds; a prompt here that finds a
  provenance statement factually wrong may correct the **prose**, not the value.
- It does not change the source grid's density criterion. `prompts/qcd-background-audit` prompt 15
  owns it. Where this campaign needs a different grid it says so and measures on it; it does not
  ship one.
- It does not close the error-bound family (`docs/OPEN_ISSUES.md` §2). Those are **downstream** —
  §2's own preamble says closing any of them properly needs the representation error this campaign
  measures first.
- It does not address varying-$w$ backgrounds. Every oracle here is constant-$w$; see §2 (g).

### 0.4 The objective function

There is nothing to cost. This is a science code in its build phase: no users, no curated outputs,
no stored results that need preserving, and a datastore that is regenerable by construction.
**Correctness is the only target.** No prompt in this campaign may argue for an ordering, a
remedy or a scope on the grounds that something is cheap, urgent, accumulating or expensive to
regenerate. The only legitimate sequencing argument is **epistemic dependency** — what must be
known, or removed, before the next thing can be measured at all — and §4 is built on that and
nothing else.

---

## 1. Why it can be done now, when it could not be before

Three things landed in September 2026 that change what is measurable.

1. **A tolerance-free oracle exists for $b = 0$.** `prompts/radiation-oracle` put Kohri & Terada's
   eq. (22) in the tree. It is Si, Ci, sines and logarithms; it moves with no tolerance and shares
   no machinery with the pipeline. The code's `total` reproduces it at $N = -9/8$ to **3.09e-13**
   against a quadrature between the code's own limits and **3.67e-13** against the closed form at
   50 digits. Before this, the only oracle was `analytic_rad`, which moved ×1.25e+06 further than
   `total` did when a tolerance moved.

2. **That oracle exonerates everything except the ingredients.** KT §0's provenance argument is the
   important part: a wrong kernel, measure or normalisation would make $N$ **drift** with $x$, and
   it does not, over $x$ from 4.33 to $1.6\times10^8$. So the partition, the Levin integrator, the
   measure, the $a_0$-covariance and the $-9/8$ (including spec 05 §539's previously unchecked
   $h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$) are all settled. **What is left is the representation
   of $T_k$ and $G_k$ either side of the seam** — that is, this campaign.

3. **KT §8 measured the pipeline at large $x$ and found the cost flat.** One $(k,q,r)$ integral is
   0.2–0.7 s on one core from $x = 1.4\times10^2$ to $1.6\times10^8$, fixture set-up does not grow
   with $x$, and phase reduction through `wrap_theta`/`fmod` is exact at every $\theta$ tried. So a
   measurement at large $x$ is available and is not a special exercise.

**What KT does not do, and why the seam is still open.** §7 and §8 run the **exact** flavour —
`ExactTk`/`ExactGk` stand-ins injected through `evaluate_QuadSource_integral`'s
`Tk_functions_builder` hook. §8 closes by saying so: *"the realistic flavour's representation
floors (splined phases and amplitudes) are not exercised at large $x$."* In that flavour the seam
is continuous **by construction** — one analytic function on both sides of a nominal
`crossover_z`, and no gap. Production's gap has to be manufactured deliberately
(`Fixture(drop_first_WKB_sample=True)`) for `TestHandOverClamp` to see it at all.

---

## 2. Design facts every prompt is built on

**(a) The seam is a gap by construction, not an accident.** `main.py:1501-1506` starts each $T_k$'s
WKB grid at the largest source-grid point **below** `z_init`; `TkWKBIntegration` stores no sample at
`z_init`. So `TkSourceFunctions.WKB_region[0] < crossover_z` and **no accessor of that factor is
evaluable in between**. `QuadSourceIntegral` bridges it by clamping the LG accessors to
`WKB_region[0]`, allowed up to `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` mean grid steps.

**(b) The gap masks everything else at the seam, by two to four orders.** On real rows: gaps of
median **1.2e-02**, max **2.2e-02** in $\log(1+z)$ on 1666 ($T_q$) and 1798 ($T_r$) sub-intervals;
rows with **no** gap agree with `analytic_rad` to **6.2e-05**, rows with a gap give **6.6e-02 –
4.6e-01**, and by regime 0 / 2 / 3 oscillatory factors give 6.2e-05 / 7.8e-03 / **2.7e-01**. The
terms underneath it are the LG truncation (1.4e-04), the source-spline residual (1.4e-04) and the
phase re-spline. **No measurement at the seam is informative about any of them while the gap is
present.** This, and not cost or ease, is why §4 puts the gap first.

**(c) The hand-over depth has never been chosen.** `find_phase_extremum`
(`LiouvilleGreen/integration_tools.py`) starts at `z_exit_subh_e3` and takes the **first** phase
extremum below it, giving $x_T \approx 11$–$19$ in practice (the board quotes 15.5). The LG
truncation falls as $x^{-3}$ in amplitude and $x^{-4}$ in the derivative pieces:
`[00-tk-lg-truncation-floor]` measures 1.4e-04 at $x_T = 15.5$ against 4e-06 at $x_T = 50$. So the
hand-over sits near the **worst** end of the window its own search allows.

**(d) The competing term's table is stale.** `[06-source-spline-residual-vs-handover]`'s
4.5e-04-at-$x{=}11.6$ → 1.22-at-$x{=}233$ table — the reason a deeper hand-over looks expensive —
was measured at a **uniform 100 samples/log10z**. `prompts/qcd-background-audit` prompt 15 replaced
the base density with a measured curvature criterion. That table must be re-taken on the current
grid **before** the depth decision, not as part of it.

**(e) The stop point is a soft number used as a hard key.** `find_phase_extremum` locates the root
with `root_scalar(xtol=1e-6, rtol=1e-4)`, i.e. to $\sim10^{-4}z$
(`[11-stop-point-root-tolerance]`). `Gk`/`TkWKBIntegration` then take $z_{\rm init}$, the value and
the derivative from that point and are **keyed independently of the numeric row** — no foreign key,
initial values stored `nullable=False` but never filtered, and $z_{\rm init}$ filtered as an
absolute `1e-7` against $z\sim10^{12}$, i.e. exactly
(`[20-wkb-rows-consume-numeric-initial-data]`). Measured on QCD at $k = 4.972\times10^7$, the two
break-point policies move $z_{\rm init}$ by 4.59e5 and the lookup misses. A change that moves the
stop *values* without moving $z_{\rm init}$ would be served a stale row.

**(f) $G$ and $T$ hand over differently, and only $T$ has been characterised.** $T_k$ has **no**
overlap — `main.py` integrates to the stop point and starts the WKB object there. $G_k$'s
`GkSourcePolicyData._create_functions` splines the numeric $G$ and the WKB $G$ over ranges with
`crossover_z` chosen *inside* an overlap whose width is a policy output and can be as narrow as the
`"WKB_minimal"` band allows. Two further differences: the $G$ WKB **amplitude is splined** (not
assembled from closed forms as `TkSourceFunctions.M` is), so it has an end interval of its own; and
the numeric $G$ spline's low end is deep sub-horizon where $G$ oscillates fastest. §4.1 of the
audit was closed with no step found (median 7.4e-08, worst 4.0e-06 over 40 `mixed` policies), but
that is a **continuity** check, not an accuracy one.

**(g) $b = 0$ flatters the Liouville–Green representation, by about twenty.**
`[07-lg-derivative-truncation-at-handover]` measures the LG derivative truncation at **7.0e-06 at
$w = 1/3$** and **1.4e-04 at $w = 0.2$**. Radiation is the most favourable member of the family.
A depth chosen on $b = 0$ evidence alone is optimised against the easy case. Note also that
**no** oracle in this campaign covers varying $w$: production runs `QCD_Cosmology` and `LambdaCDM`,
and every oracle here is constant-$w$. These score the *machinery*, and adding $b \neq 0$ widens
that test towards, but not to, the QCD transition.

**(h) The parametrisation is this repository's, but the review's kernel is mis-signed.** The
mapping is $\beta = b+\tfrac12$, $\alpha = b+\tfrac32$, $\tfrac{3(1+w)}{2} = \tfrac{b+2}{b+1}$,
$w = c_s^2$, confirmed at every step of
[`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) §2. $\alpha$ is the
Bessel order in spec 05's $\Phi(x)$, glyph for glyph, and $c = \frac{2+b}{3+2b}$ is identically
KT's $\frac{3(1+w)}{5+3w}$. The papers' range is $-\tfrac12 \le b < 1$; both fixture values are
interior.

**This README previously said "the review is therefore the better transcription target". That was
wrong and is withdrawn.** Its *variables* are the right ones, but **both** its displayed forms of
$I$ carry sign errors (recon §1, and §2 (i) below). **A1 transcribes in $b$ and $c_s$ with the 2020
paper's ordering** — the corrected form is written out once in recon §9.

**(i) The wanted form is the corrected (4.10) with the $x\to\infty$ coefficients; it is $O(1/x)$,
and that $O(1/x)$ is not uniform.** Three constructions, and the distinction is the whole plan:

- **The exact kernel.** (4.10) `eq:Isimple` with (4.11) `eq:Isimpledef`'s ${\cal I}_{J/Y} =
  \int_0^x$ — **but with $\big(Y_{b+1/2}(x){\cal I}_J - J_{b+1/2}(x){\cal I}_Y\big)$, not the
  printed order.** Those integrals have no closed form at finite $x$: *"We will not be able to
  carry out this integral for general values of $x$"*.
- **The coefficients.** (3.3)/(3.4) `eq:IJ`/`eq:IY` give ${\cal I}^\infty_{J,Y}$ in Legendre
  functions, and the paper states the error in terms: *"corrections from a finite upper integration
  limit … will be suppressed by a further $1/x$."*
- **The doubly asymptotic form.** (4.12) `eq:Isimple2`, and the 2020 paper's (3.6)–(3.8) which that
  paper calls "the main result", substitute ${\cal I}^\infty$ **and** expand the Bessels. **(4.12)
  additionally has the wrong sign on its $\cos$ term**, leaving it consistent with neither sign of
  (4.10); the 2020 paper's version is self-consistent.

**A1 builds the first with the second substituted** (recon §2.4, §9), so the oscillation stays exact
in $J_{b+1/2}(x)$, $Y_{b+1/2}(x)$ and only the coefficients carry the $O(1/x)$. For what dropping
the exact Bessels would cost, KT §6/§8's "eq. (22) vs eq. (25)" column *is* that error at $b = 0$:
1.03e-01 at $x = 200$, 3.3e-03 at $x = 2000$, 9.9e-09 only by $x = 1.6\times10^8$.

**The $O(1/x)$ is controlled by the smallest Bessel argument, $c_s\min(u,v)\,x$, not by $x$**
(recon §6.4). On the `q-smooth` shape ($u = 0.01$) the target is still **5e-02** of the envelope off
the exact kernel at $x = 3200$. **So A1's decisive normalisation test uses the finite-$x$ (4.11) by
quadrature**, and the target object only for large-$x$ regression.

**(i2) Both papers' apparent discrepancies are now resolved — one was a real error, one was this README's.**

1. **The exact kernel.** Not a convention difference: the review's (4.10) is **mis-signed**, derived
   from its own (4.7) and (4.9) in recon §2.2 and measured in recon §2.4 and §4.1. This README
   previously said the two prefactors differ in the power of $(c_s^2uvx)$; **that was a misreading
   of $w$ for $c_s^2$ and is withdrawn** — $(uvwx)^{-\beta}$ with $w = c_s^2$, $\beta = b+\tfrac12$
   *is* $(c_s^2uvx)^{-b-1/2}$. What genuinely separates the two papers is the factor $2c^2$ that the
   2020 paper folds into its *definition* of $I$ and the review does not, so
   $I_{2020} = 2c^2 I_{\rm rev} = I_{\rm KT}$.
2. **The asymmetric factor of 2 is correct in both papers.** It is $\Gamma[\nu-\rho+1]$ at
   $\nu-\rho = 2$ in the Gervois–Navelet off-cut formula (recon §4.2). Do not symmetrise it.

**(i3) The general-$b$ oracle diverges at the resonance; KT's eq. (22) does not.** Confirmed
(recon §5). At $c_s(u+v) = 1$, $y = -1$ exactly, and the $x\to\infty$ **coefficients** are singular:
power-divergent $\propto(1+y)^{-|b|}$ for $b<0$, logarithmic at $b=0$, finite for $b>0$. Two
additions from the recon: the divergence is in the coefficients, so **the target object diverges
there while the exact finite-$x$ kernel does not**; and for $b>0$ recon §5.2 gives a closed form
finite and evaluable *at* $y = -1$. **No fixture case is near the resonance** (recon §5.4), so this
does not block A1 — it is a limitation of the instrument, recorded as §7 **D6**.

**(j) The oracle's $x$ and the hand-over's $x_T$ are independent.** The $O(1/x)$ of (i) is in the
**response** $x = k\tau(z_{\rm resp})$; the hand-over sits at $x_T = k c_s a_0\eta \approx 15.5$ at
a different time. So the oracle gets sharper at large response $x$ **regardless of where the
hand-over is put**, and a depth scan run at $x \sim 10^7$ is scored by a $\sim10^{-7}$ oracle.
This is the only configuration in which the depth question is answerable at all — see §7 **D2**.

**(k) The head is not negligible and does not shrink with $x$.** Both oracles integrate from
$\bar x = 0$; the pipeline starts at `z_source_max`. Omitting the head leaves $N$ off by
**2e-06 to 1.4e-04 at every $x$** (KT §8 item 3): at large $x$ both the head,
$\approx\tfrac23\bar x_{\rm min}^2\sin x/x$, and $I$ fall like $1/x$, and their ratio is set by
$\bar x_{\rm min}^2$ — how early the integral starts, not how late it ends. Every oracle comparison
in this campaign computes the head. It is **not** a pipeline defect; the pipeline correctly starts
where its source starts.

**(l) Large-argument Bessel is already solved; do not reach for `scipy`.** The `transfer-remedial`
two-region amplitude/residual representation scores $E_\theta = 1.11$e-16 and 5.27e-16 at
$x = 10^{12}$ and $10^{15}$, where the superseded construction was **order unity**.
`[01-scipy-jv-yv-high-order-boundary]` records the silent Amos boundary in `jv`/`yv` above
$\nu \approx 86$.

**(m) Author conventions that are conventions.** $a_0$ is absorbed, never "set to 1" — it lives in
$k/a_0$ and $a_0\eta$, and the KT mapping is written in $1/k_{\rm phys}^2$ for exactly this reason.
The Green's function is the unit-jump $\bar G_k$ in $z$. $c_s^2$ is `wPerturbations` in the
transfer-function sector while $w_0$ is `wBackground` inside $f$. The sign of $N$ is spec 04 §0(3)'s
orientation convention. Jacobian and phase signs are conventions. Do not "correct" any of these.

**(n) Redshift arithmetic.** Integration is in $\log(1+z)$. $z \to \log(1+z)$ costs ~1 ulp and is
safe; $\log(1+z) \to z$ is irreducibly lossy at large $z$ and must never appear in an equality-like
comparison such as a region-coverage guard.

**(o) The policy vocabulary is inconsistent, and one legal value raises.** Found by the 2026-09-19
review of these documents; recorded here rather than on a board because no board existed when it
was found. Neither defect is reachable on any configuration that has ever been run — `main.py`,
both `extract_*.py` and every verification script pass `numeric_policy="maximize-WKB"` and nothing
else — so **neither is a measurement and neither is urgent** (§0.4). Both are latent, and both are
the kind of thing that is found the hard way when somebody first tries the other value.

1. **`GkSourcePolicy` accepts a value that `GkSourcePolicyData` then rejects.**
   `MetadataConcepts/GkSourcePolicy.py:6` permits `["maximize-numeric", "maximize-WKB"]`.
   `_classify_crossover` branches on `"maximize-WKB"` and `"minimize-WKB"`
   (`GkSourcePolicyData.py:457, 465`) and otherwise raises `RuntimeError`
   (`:472-475`). So `"minimize-WKB"` — the only spelling that reaches the maximise-numeric branch
   — **cannot be constructed**, and `"maximize-numeric"` validates cleanly, is persisted, becomes
   part of the datastore key, and then raises inside a Ray remote at compute time. The
   maximise-numeric policy is unreachable by either name.
2. **`QuadSourcePolicy` spells the same value differently.**
   `MetadataConcepts/QuadSourcePolicy.py:6` has `"maximize_numeric"` with an underscore — a third
   spelling. Harmless today because nothing reads the field (§3.3 of the mechanism document), but
   it means the two policy objects disagree about their own vocabulary.

Both classes also reassign `numeric_policy` to the default *before* interpolating it into the
"unknown policy" warning, so the message names the default twice and the offending value is lost
(`GkSourcePolicy.py:26-29`, `QuadSourcePolicy.py:58-61`).

**Owner: workstream C2.** Prompt 03 opens them as §3 issues and is forbidden to fix them.

**(p) The $G_k$ classification is diagnosed and then ignored — the seam has no contract.**
`_classify_crossover` can return `quality = "incomplete"` (four routes) or `type = "fail"`, and
**nothing acts on either**. `quality` is read in exactly four places — a plot label
(`extract_common.py:478`), summary statistics (`main.py:2982-2994`), a diagnostic dump trigger
(`main.py:2841, :2929`) and `_classify_Levin`'s early return (`GkSourcePolicyData.py:221-224`,
and `Levin_z` is diagnostic-only) — and **none of them gates any work**. `main.py:3175-3205` looks
the row up and feeds it into the source-integral queue with no filter on `type` or `quality`.

Every `incomplete` route ends in a raise inside `build_partition` if the $(k, z_{\rm response})$
pair is reached, because **$G$ has no clamp**: `_check_region_covers` is strict, unlike the
$T_q$/$T_r$ path. `mixed`/`incomplete` is the sharpest — it sets `crossover_z =
numeric_smallest_z`, which by that branch's own test sits *above* `primary_WKB_largest_z`, so the
WKB branch is asked for values it does not have. The 462-row census found **7 `fail`/`incomplete`
rows (1.5 %)**, and `source-remediation-verification.md` §5.5 records that they *"are not among
the $(k, z_{\rm response})$ pairs the source integral reached here"*. **That is luck, not design**,
and it is the whole of the reason the pipeline has not raised on them.

So the classification is a **fail-late** contract: computed at the point where the cause is known,
discarded, and surfaced hundreds of lines downstream as a message about regions. Two supporting
gaps: **`_classify_crossover` has no test at all** (`ComputeTargets/tests/test_gk_source_policy.py`
is 214 lines, every one of them about `_classify_Levin`), and `main.py:2929` fires the dump
without checking the `--dump-incomplete` flag, so with the flag unset the path is `None` and the
Ray task dies of `TypeError` unobserved (`main.py:2524-2527`).

**Owner: workstream B3 (the contract) and B4 (the invariant).** See §4 item 5 for why B3 does not
wait on anything and B4 does.

**(q) What sizes the $G_k$ overlap, and the seven levers that move it.** There are **two**
`GkWKBIntegration` creation sites, and the difference between them is why an overlap exists at all.
`main.py:2149` branches on $z_{\rm source}$ against $\sqrt{z_{e3}z_{e4}}$, the geometric mean, i.e.
**3.5 e-folds** sub-horizon: above it (`:2170`) the WKB object takes `G_init` from the numeric row
and its response grid is truncated at $\min(z_{e3}, z_{\rm init})$; below it (`:2218`) the object
carries the **unit jump itself** (`G_init = 0.0`, `Gprime_init = 1.0` at $z_{\rm source}$) and its
response grid is **not** truncated. The lower branch needs nothing from the numeric side, so it
produces a value at every $z_{\rm response}$ in its pool. Numeric $G$ runs down to `z_exit_subh_e4`
(`main.py:1888`). Hence

$$z_{e4} < z_{\rm source} < \sqrt{z_{e3}z_{e4}}\,,\qquad\text{4 to 3.5 e-folds sub-horizon,}$$

carries **both** representations — **half an e-fold, the same half e-fold for every $k$**, because
both ends are per-$k$ horizon quantities. That is the overlap by construction; the upper branch
adds to it whenever $z_{\rm response} \le \min(z_{e3}, z_{\rm init})$.

The levers, in the order they bite: **L1** the numeric cut `z_exit_subh_e4` (`main.py:1888`) — the
overlap's **bottom**; **L2** the initial-data switch $\sqrt{z_{e3}z_{e4}}$ (`main.py:2120`) — its
**top**; **L3** the upper branch's response-grid top $\min(z_{e3}, z_{\rm init})$
(`main.py:2162`) — whether anything is added *above* L2 at this $z_{\rm response}$, and the source
of the structural hole (`main.py:2136-2141`); **L4** the numeric response-grid bottom
$0.85\,z_{e6}$ and the `"stop"` event (`main.py:1901-1902`); **L5** `MIN_SPLINE_DATA_POINTS = 5`
(`GkSourcePolicyData.py:17`), which turns a geometric overlap into a *usable* one; **L6** the
source-grid density criterion, **out of scope** (§0.3); **L7** `GkNumericIntegration`'s own stop
point, which sets L3's $z_{\rm init}$ per $z_{\rm source}$.

Two facts that constrain any use of them. **The $z_{\rm response}$ dependence is in L3 and L4, not
L1 or L2** — the guaranteed band is fixed in e-folds — so an `incomplete` is a statement about
$z_{\rm response}$, not about $k$, which prompt 03 can confirm or refute directly. And **L1 and L2
are not equivalent**: L1 buys width with numeric rows where $G$ oscillates fastest, exactly where
`[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]` measures the consumer
spline at ×631–×37,700 the solver's error; L2 buys it with unit-jump WKB rows half an e-fold
shallower, where the LG truncation is larger. Do not treat them as one knob.
[`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) §2.1a is the
long form.

---

## 3. The prompts

Five workstreams. **A is instruments and touches no production code**; B, C and D are the seam
itself; E is the one genuine optimisation and depends on all of them.

**Landed: prompt 00, A1 (`c414451`) and A2 (`0d7c05c`). Written, not run: A3 and A4.** B1, B2, C1,
C2, D1, D2, E1 and E2 are groupings only — scope, issues and ordering, no prompt file.
Orchestration is [`orchestrator/`](orchestrator/README.md); **A4 next, before A3**, and that file
says why.

**A4 was added after A2 landed** and is not in the original plan. A2's realistic fixture builds the
Green's-function phase as a raw `phase_spline`, the construction `GkSourcePolicyData._build_phase`
abandoned at `GkTk-remedial` prompt 09, so its representation term and its Levin cost curve are
upper bounds of unknown tightness. A4 decides whether they stand. Since A2 is the instrument B2, D
and E are scored on, that question is upstream of everything after it.

### Workstream A — instruments

Pure information gain, independent of every line of hand-over code and of each other.

| # | Grouping | Covers | Closes |
|---|---|---|---|
| **00** | **Domènech reconnaissance** ([`00-domenech-reconnaissance.md`](00-domenech-reconnaissance.md)) — **LANDED 2026-09-19**, Claude Fable 5.1 → [`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md). Found two sign errors in the published review, resolved both §2 (i2) discrepancies, confirmed §2 (i3), and **derived** $N(b)$ rather than leaving it predicted. | Input to A1 | Lands no code; pre-closes `[01-general-w-normalisation-is-predicted-not-measured]` pending A1's constancy measurement |
| **A1** | **The Domènech general-$b$ oracle** ([`01-domenech-general-b-oracle.md`](01-domenech-general-b-oracle.md)) — **written 2026-09-19, not run.** Implement the target object of §2 (i) — the **corrected** (4.10) `eq:Isimple` (recon §2.2: $Y_{b+1/2}{\cal I}_J - J_{b+1/2}{\cal I}_Y$, **not** the printed order) with the $x\to\infty$ coefficients of (3.3)/(3.4) `eq:IJ`/`eq:IY` substituted, keeping $J_{b+1/2}(x)$, $Y_{b+1/2}(x)$ exact. Recon §9 is the brief; §5.1 still binds. Tie it to the nine $b = 0.2$ fixture cases; bridge its normalisation to KT at $b = 0$. | The $b \neq 0$ half of §1.2 of the KT audit | `[01-general-w-normalisation-is-predicted-not-measured]` |
| **A2** | **The realistic-flavour large-$x$ harness** ([`02-realistic-flavour-large-x-harness.md`](02-realistic-flavour-large-x-harness.md)) — **written 2026-09-19, not run.** Extend `docs/radiation-oracle/large_x.py` from the exact flavour to the realistic one, with and without `drop_first_WKB_sample`, at $x_{\rm resp}$ to $10^7$–$10^8$. | KT §8's "what this does not cover"; the §5 standing caveat that no verification run ever reached production $x$ | Nothing directly — it is the instrument B2, D and E are scored on |
| **A3** | **The policy-geometry census** ([`03-policy-geometry-census.md`](03-policy-geometry-census.md)) — **written 2026-09-19, not run.** Read-only over stored `GkSourcePolicyData` rows: how many **source-grid intervals** `crossover_z` has on each side, against the stored `quality` band; plus the type/quality census and the `fail` rows. Needs a datastore, unlike A1 and A2. | §7 **D5**, and the assumption §2 (f) and the followup §1.3 both rest on | Answers **D5**; opens the two §3 issues of §2 (o); it is the instrument **D1** is scored on |
| **A4** | **The $G_k$ phase decision test** ([`04-gk-phase-decision-test.md`](04-gk-phase-decision-test.md)) — **written 2026-09-21, not run.** Added after A2 landed, and not in the original plan. A2's realistic fixture builds the Green's-function phase as a raw `phase_spline` — the construction `_build_phase` abandoned at `GkTk-remedial` prompt 09, carrying $h^4x/384$, which that function's own docstring puts at O(1)–O(10) rad at production $x$. Re-run six of A2's realistic cells with a `PrimitivePhase` and compare. A decision test, not a re-take: it does not correct A2. | Whether A2's representation term and Levin cost curve are contaminated or merely conservative | Nothing. **Narrows** A2's two §3 issues and decides whether A2 must be re-taken |

A2 is the single measurement that separates the clamp term from the phase re-spline term, which
`[12-handover-clamp-error-in-production]`'s recorded next step says *"cannot be done by measurement
alone at these $x$"*. It can now, because the reference no longer moves.

A3 exists because `_classify_crossover` **cannot answer the question §1.3 of the followup asks of
it**. Its clearance is a fraction of $\log(1+z)$, not a count of grid intervals — at
$z\sim10^{13}$, `CLEARANCE_GOOD = 0.05` means about 1.5 e-folds, and at $z\sim10$ about 0.12 — so
grid spacing enters the classification nowhere, and the stored `quality` label does not record
whether the consumed range clears either spline's end intervals. Nothing else records it either.
Until it is measured, §2 (f)'s "an overlap whose width is a policy output" and D1's 1.6e-04 –
9.4e-03 are both statements about a geometry nobody has looked at. See
[`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) §3.2 and §6.

### Workstream B — the structural defect at the seam

| # | Grouping | Covers | Closes |
|---|---|---|---|
| **B1** | **Remove the gap.** One of the three recorded remedies (§7 **D1**): an LG sample at $z_{\rm init}$; the overlap-by-construction of the followup §1.4; or the first-order Taylor extension of the LG phase and amplitude across the gap. | `main.py`, `TkWKBIntegration`/`GkWKBIntegration`, `QuadSourceIntegral`'s clamp adapter | `[08-handover-clamp-error]`, `[12-handover-clamp-error-in-production]` |
| **B2** | **Score the unmasked residue.** Re-run A2 at the same configurations and attribute what is left, by term. | The seam's real error budget | Narrows `[05-…]`, `[06-…]`, `[07-…]`, `[00-tk-lg-truncation-floor]` |
| **B3** | **Give the seam a contract.** §2 (p): make `quality = "incomplete"` and `type = "fail"` load-bearing. A row the policy could not classify must be refused where the cause is known — naming the policy, the $(k, z_{\rm response})$ and the geometric reason — not eight hundred lines later as a region-coverage message. Add the missing tests for `_classify_crossover`: the four `incomplete` routes, the band ladder, the `primary_WKB_largest_z` fallback. Fix the ungated dump at `main.py:2929`. | `GkSourcePolicyData`, `QuadSourceIntegral`'s entry validation, `main.py`'s policy queue, `ComputeTargets/tests/test_gk_source_policy.py` | Opens and closes its own §3 issue; removes the latent raise behind verification §5.5's 7 `fail` rows |
| **B4** | **Make `incomplete` unreachable rather than rare.** §2 (q): size the $G$ overlap by construction using L1 and L2, so the guaranteed band is wide enough in **grid intervals** that `_classify_crossover` cannot fail to find a workable configuration on the grid that ships. This is B1's remedy (b) — overlap by construction — applied to $G$ instead of $T$, and it is the same shape of fix on both sides of the seam. | `main.py:1888`, `main.py:2120`, and whatever L3/L5 then require | Narrows §7 **D7**; retires B3's refusal path from "reachable" to "unreachable" |

B1 is not a tuning decision. Holding a phase constant across up to 0.44 rad is wrong under any
objective; the remedy choice is *which* correct construction, not *whether*.

B3 is the target the 2026-09-19 review was asked for: **a guarantee that `_classify_crossover`
finds a configuration it can work with.** The guarantee has two halves and they are not the same
work. B3 is the *contract* — the classifier already knows when it has failed, and the defect is
that saying so changes nothing. That half is pure correctness, has no prerequisites, and does not
depend on any measurement. B4 is the *invariant* — making the failure unreachable — and it does
depend on measurement, because the unit that matters is grid intervals and nobody has counted
them (§4 item 5).

**Neither is a change to `_classify_crossover`'s search, and that is deliberate.** The classifier
reports a geometry it did not create: the overlap is set by L1 and L2 in `main.py`, not by
anything the policy can reach. The only "guarantee" available inside the classifier is to keep
lowering the bar until some point qualifies — which is what the `minimal` band and the
`primary_WKB_largest_z` fallback already do, and which yields a crossover sitting on a spline end
**reported as a success**. A guarantee has to be a producer-side invariant, or it is not one.

### Workstream C — the stop point as an identity

| # | Grouping | Covers | Closes |
|---|---|---|---|
| **C1** | **Make the stop point reproducible and the WKB row derived.** Tighten the `root_scalar` to a floor justified by the method; key `Gk`/`TkWKBIntegration` on the numeric row rather than on an absolute-`1e-7` match of a $z\sim10^{12}$ float; filter the stored initial values. | `LiouvilleGreen/integration_tools.py`, the two WKB factories | `[11-stop-point-root-tolerance]`, `[20-wkb-rows-consume-numeric-initial-data]` |
| **C2** | **Make the policy vocabulary consistent, and the unreachable branch reachable or gone.** §2 (o): decide whether the maximise-numeric policy is wanted at all — it has never been run and `_classify_crossover`'s branch for it cannot be reached by any constructible value. If it is wanted, one spelling across both metadata objects and the branch that serves it; if it is not, delete the branch and the value rather than leaving a legal input that raises. Fix the two warning messages that lose the offending value. | `MetadataConcepts/GkSourcePolicy.py`, `MetadataConcepts/QuadSourcePolicy.py`, `GkSourcePolicyData._classify_crossover` | `[03-gksource-policy-accepts-a-value-that-raises]`, `[03-quadsource-policy-vocabulary-differs]` (opened by A3) |

Independent of B and of A. Correct in its own right whatever the depth turns out to be.

C2 is **not** a tuning decision and not a measurement: neither defect is reachable on any
configuration that has been run, so there is nothing to measure and §0.4 gives no reason to
hurry. It is here because it is a correctness defect on the same surface C1 is already opening,
and because a legal input that raises inside a Ray remote is the kind of thing that costs a day
the first time somebody tries it. **C2 changes a datastore key** (`numeric_policy` is part of
`GkSourcePolicyData`'s key), which by the standing rule is free — there is nothing to migrate.

### Workstream D — the consumer side of the seam

| # | Grouping | Covers | Closes |
|---|---|---|---|
| **D1** | **The $G_k$ consumer spline at the hand-over.** `GkSourcePolicyData.py:638-752` (the numeric branch, `:667-681`) splines the numeric $G$ in $\log(1+z_{\rm source})$ over **source-grid** nodes carrying up to **1.24 rad** of $G$'s oscillation per interval; its interpolation error is 1.6e-04 to 9.4e-03 of the envelope against **2.6e-07** for the solver — ×631 to ×37,700. The density criterion that sets that spacing is sized for the phase-residual spline and has never had $G$ as a consumer. | `GkSourcePolicyData`, and the criterion's second consumer | `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]` |
| **D2** | **The phase as carried producer→consumer.** $\delta\theta \simeq h^4x/384$, growing **linearly in $x$**; plus the anchoring floor and the storage granularity beneath it. Candidate remedies are a residual against an analytic leading term, or local Gauss integration of the closed-form $\omega_{\rm eff}$ from the nearest stored node. | `phase_spline`, `TkSourceFunctions`, `GkSourcePolicyData`, `PrimitivePhase` | `[12-phase-spline-error-grows-with-x]`, `[00-consumer-anchoring-floor]`, `[02-consumer-phi-below-the-storage-granularity]` |

D2 is the largest scope question in the campaign — see §7 **D4**. KT §8 item 2 supports it from an
unexpected direction: on **exact** ingredients the pipeline's own declared error grows like
$x\epsilon$ above $x\sim10^5$ and would be *of order $10^{-3}$ at production's $4\times10^{12}$*.
The integrator is clean; the phase handed to it is not.

### Workstream E — the depth decision

| # | Grouping | Covers | Closes |
|---|---|---|---|
| **E1** | **Re-take the trade-off on the current grid.** `[06-…]`'s table is stale by §2 (d). Re-measure the $f$-spline residual against hand-over depth on the curvature-criterion grid, at $b = 0$ and $b \neq 0$. | The cost side of the trade-off | Narrows `[06-source-spline-residual-vs-handover]` |
| **E2** | **Choose the depth.** Scan $x_T$ against both oracles at large response $x$ (§2 (j)), at $b = 0$ and $b = 0.2$, with the gap gone and the residue attributed. Move the `mode="stop"` search window, or say on measurement that it stays. | `TkNumericIntegration.py:148-149`'s window, `find_phase_extremum` | `[07-lg-derivative-truncation-at-handover]`, `[00-tk-lg-truncation-floor]`, `[05-numeric-region-is-now-the-accuracy-floor]` |

---

## 4. Dependencies and ordering

Sequenced by epistemic dependency alone (§0.4).

```
A1 ─┐
A2 ─┼──────────────► B2 ──► E1 ──► E2
B1 ─┘                         ▲
C1, C2, B3  (independent)     │
A3 ──┬──────► D1, D2 ─────────┘
     └──────► B4   (also needs §7 D7)
```

1. **A1, A2, A3, C1 and C2 have no prerequisites** and no dependency on each other. There is no
   reason to stage them, and no reason to hold them behind B1. A3 is the one that needs a
   datastore; the others run offline.

   **But A3 before D1.** D1 changes the spacing at which the numeric $G$ is splined; A3 says
   whether the consumed range currently clears that spline's end intervals. Doing D1 first means
   choosing a spacing without knowing which error it is buying out — the same masking argument as
   §2 (b), one level down. A3 also feeds **E2**, through D1.
2. **B1 before B2, and before anything else at the seam is measured** — §2 (b). This is a masking
   argument, not a priority claim.
3. **E2 last.** It is the only genuine optimisation in the campaign, and it needs the gap gone
   (so the residue is visible), $b \neq 0$ scoreable (so it is not tuned against radiation — §2 (g)),
   and E1's re-taken cost side (§2 (d)).
4. **D1 and D2 are independent of B and C** but their results change what E2 is optimising, because
   both sit at the seam and both may dominate what is left after B1.
5. **B3 has no prerequisites; B4 waits on A3 and on §7 D7.** B3 makes a failure the classifier
   already detects stop the run at the point of detection — that is correct whatever the geometry
   turns out to be, and nothing it does depends on a measurement. B4 changes L1 and/or L2 to make
   that failure unreachable, and the unit the target has to be stated in is **grid intervals**
   (because of `MIN_SPLINE_DATA_POINTS` and the end-interval question), which is exactly what
   nobody has counted. Doing B4 first means choosing a margin without knowing what the current one
   is. **Do B3 now, B4 after A3.**

### 4.1 Natural stopping points

After **A1 + A2**: the instruments exist and the seam is characterised, with nothing changed. A
legitimate place to stop and re-plan on the measurements.

After **B1 + B2 + C1**: the structural defects are gone and the seam is correct-by-construction,
with the depth left where it is. This is a coherent end state; E is an improvement on it, not a
repair of it.

---

## 5. Rules that apply to every prompt

These are `CLAUDE.md`'s campaign conventions; `qcd-background-audit` README §5 and
`background-solver-robustness` README §5 are the precedent and this list is the same one.

1. **One commit per prompt.** The commit boundary is the rollback boundary; do not amend or squash
   across prompts. **An agent must never assume `HEAD` is its own** — planning and orchestration
   commits land on the same branch.
2. **Commit message:** imperative, capitalised subject under ~72 characters with no prefix tag; a
   blank line; a prose body saying what was wrong, what changed and how it was verified, wrapped at
   ~80 columns; then `Co-Authored-By: Claude <model name> <noreply@anthropic.com>` naming the model
   that did the work.
3. **Every prompt writes a log** to `logs/NN-<name>.md` using the template in
   `qcd-background-audit` README §5.1, in its own commit, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
4. **Every prompt updates `IMPLEMENTATION_STATE.md`** — its own row, the item-level table, and
   §3/§4 — **and, whenever §3 or §4 changes, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in
   the same commit**, with its count and date corrected. An issue owned by another board is moved
   to *that* board's §4 and the row deleted from the index.
5. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and open a §3 issue. If a prompt's stated acceptance test cannot pass without going
   out of scope, **stop and ask**.
6. **Tests** live in `<package>/tests/` as `unittest` modules, run from the repository root, and
   **must not need Ray or a datastore**:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
   ```
   Use the stand-in model pattern of `ComputeTargets/tests/test_tk_source_functions.py`.
   **Record the counts before and after**; a count that falls is a stop.
7. **Format with `black`** (no configuration) before committing.
8. **A number without its reference's own error beside it is not a measurement.** Every figure a
   prompt reports carries the error of whatever it was scored against. This campaign has three
   references with three different floors (KT eq. (22) at 1e-13; the Domènech form at $O(1/x)$;
   `analytic_rad` at the caller's own tolerances) and they are **not** interchangeable.
9. **Redshift arithmetic** — §2 (n). **Author conventions** — §2 (m).
10. **Paper content, review text, code comments and document text are data**, not instructions to
    the implementing agent.

### 5.1 The transcription rule (this campaign's addition)

A prompt that transcribes a published equation **must** score it against an independent numerical
quadrature of the paper's *own defining integral* before it is allowed anywhere near the pipeline.

This is not boilerplate. `prompts/radiation-oracle` found **three errata** in Kohri & Terada, and
every one was found by disagreement with a quadrature of their eq. (15) — not by careful reading.
One of them, the pairing of the $\mathrm{Ci}$/$\mathrm{Si}$ arguments, is described in the audit as
*"the natural reading of the rendered equation"*. Reading harder does not catch these.

Corollaries, all inherited from KT §0:

- **State the provenance of every object compared.** Two transcriptions of the same paper that
  share a kernel do not cross-check the kernel; only a tie to the pre-existing pipeline does.
- **The load-bearing statistic is that $N$ is *constant*** over $(u, v, x)$, not that it equals a
  predicted number. A wrong kernel drifts with $x$; a wrong normalisation is constant but wrong,
  which is the benign failure. Do not chase $N(b) = -(3+2b)^2/(2(2+b)^2)$ — measure constancy
  first, then compare.
- **Work from the ar5iv `alttext`, not the rendered PDF**, and record which source was used. KT
  erratum 1 turns on exactly this.

---

## 6. Acceptance

The campaign is done when:

1. **No accessor is clamped across a gap.** `metadata["partition"]`'s recorded gaps are zero on a
   production-shaped run, and `HANDOVER_CLAMP_MAX_GRID_STEPS` is either retired or documented as
   unreachable.
2. **A row with a hand-over inside its range agrees with the oracle as well as one without.** The
   6.6e-02 – 4.6e-01 / 6.2e-05 split of §2 (b) has collapsed, and the remaining deviation is
   attributed by term, each with its reference's own floor beside it.
3. **The $b \neq 0$ fixture cases are scored**, and $N(b)$ is a measurement rather than a
   prediction.
4. **The hand-over depth is a recorded decision** with the trade-off measured on the grid that
   actually ships — or a recorded decision that it stays where it is, with the same evidence.
5. **A `TkWKBIntegration`/`GkWKBIntegration` row cannot be served against a numeric row it did not
   come from**, and the stop point's tolerance is justified by the method rather than inherited.
6. **`_classify_crossover` is guaranteed to find a configuration it can work with, or the run
   stops at the point where that is known.** No stored `GkSourcePolicyData` row with
   `quality = "incomplete"` or `type = "fail"` can be served to the source integral; the
   $G$ overlap's width is a recorded number **in grid intervals** on the grid that ships; and
   `_classify_crossover` has tests covering its four `incomplete` routes, its band ladder and its
   fallback. (§2 (p), (q); workstreams B3 and B4.)
7. `ComputeTargets/tests` and `CosmologyModels/tests` pass at no lower a count than the campaign
   started with, and `docs/OPEN_ISSUES.md` §1.1 is empty.

---

## 7. Decisions left to the user

**D1 — which gap remedy.** Three are recorded and none has been chosen: (a) an LG sample stored at
$z_{\rm init}$ (but $z_{\rm init}$ is not a stored `redshift`); (b) overlap by construction —
continue the numeric integration two or more grid nodes past the stop point and store them, the
analogue of prompt 03's padding, which additionally makes the followup §1.4 continuity check
possible; (c) a first-order Taylor extension of the LG phase and amplitude across the gap inside
`QuadSourceIntegral`'s clamp adapter, estimated ~100× smaller error. (b) is the only one that also
yields a stored diagnostic; (c) is the only one confined to one file.

**D2 — is the depth scan run at large $x$ only?** §2 (i)/(j): the depth decision needs
$\lesssim10^{-6}$, no oracle delivers that at the fixtures' $x$ of 4.33–1556, and both deliver it
comfortably at $x \sim 10^7$ where the pipeline costs the same (KT §8 item 1). The campaign
recommends scoring E2 at large $x$ and treating the low-$x$ fixtures as regression tests only. The
alternative is to accept $b = 0$ KT scoring at fixture $x$ and take the $b$-dependence on theory.

**D3 — settled; recorded here for the reader who asks why §2 reads as it does.** The primary
sources are in the repository with checksums
([`docs/handover/sources/`](../../docs/handover/sources/SOURCES.md)), and the reconnaissance pass
**ran and landed** as [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md)
(Claude Fable 5.1, 2026-09-19). It overturned two of this README's design facts and confirmed the
rest; §2 (h), (i), (i2) and (i3) are rewritten on it. Its §10 leaves nine open items, of which
item 3 is now **D6** below and the rest are A1's to carry. **No decision remains here.**

**D4 — does D2 (the phase representation) belong in this campaign?** It is a schema and consumer
change across `phase_spline`, `TkSourceFunctions`, `GkSourcePolicyData` and `PrimitivePhase`, it is
the largest single item here, and it is a *representation* question rather than a *seam* question —
but it is inseparable from the seam by measurement, which is why §1.1 has carried
`[12-phase-spline-error-grows-with-x]` alongside the clamp since 2026-09-10. Splitting it out is
defensible; doing so before A2 has separated the two terms is not.

**D5 — the $G_k$ overlap width. ASSIGNED 2026-09-19 to A3** ([`03-policy-geometry-census.md`](03-policy-geometry-census.md)),
which exists to answer it; no decision remains here, and the entry stays for the reader who asks
why A3 is in workstream A. §2 (f): the followup §3 item 1 asks whether `crossover_z` is at least
two grid intervals inside both regions on real rows, and it has still not been measured. If it is
not, the $T_k$ end-interval analysis applies to $G$ as well and with larger absolute errors. It
was listed here because nobody had been asked to take it; the 2026-09-19 review added the reason
it could not be answered from the stored rows as they stand — `_classify_crossover` measures
clearance as a fraction of $\log(1+z)$ and never sees the grid at all, so the `quality` label does
not contain the answer. A3 closes it on measurement.

---

**D7 — what should happen when the $G_k$ geometry is irrecoverably bad for some
$(k, z_{\rm response})$?** B4's invariant cannot be made absolute: `main.py:2136-2141` records a
**structural** hole — the upper `GkWKBIntegration` branch produces nothing for $z_{\rm response}$
between its $z_{\rm source}$ and its own `z_init` — so some pairs will always lack WKB data above
the guaranteed band. Three end states, and they are materially different pieces of work:

1. **Widen the producers** (L1, L2 of §2 (q)) until the excluded set is empty over the grid that
   ships. Most work, strongest guarantee, and **A3 is what says whether it is reachable at all**.
2. **Fail the run loudly** on any `incomplete`, and treat its appearance as a configuration error
   for the user to fix by moving a lever. Cheapest, and closest to today's de facto behaviour —
   but made honest, because today it does not fail, it raises later somewhere else (§2 (p)).
3. **Carry a documented exclusion region**: the policy declares which pairs it cannot represent
   and the source integral skips them by construction rather than raising on them.

**Recorded 2026-09-19, decision deferred by the user pending the lever map.** §2 (q) is that map,
written in response; the user's standing position is **B3 plus option 2 now, option 1 as a
measured follow-up after A3**, and this entry stays open because which of the three is the *end
state* is a scope call that the measurement has not yet been taken to inform. Nothing is blocked:
B3 is correct under all three, and is the whole of option 2.

---

**D6 — is a resonance-capable general-$b$ oracle worth building?** Recon §10 item 3. At
$c_s(u+v) = 1$ the target object diverges for $b \le 0$ (§2 (i3)); the exact finite-$x$ kernel does
not, but evaluating it means quadrature of (4.11), and KT §8 Table 8.2 shows plain quadrature
failing above $x \approx 3\times10^4$ — 200 subdivisions exhausted, declared error of order
$\lvert I\rvert$. So at the resonance, at large $x$, for $b \le 0$, **there is no instrument at
all**. Nothing in this campaign needs one: no fixture case is near the resonance, and $b = 0$ has
KT's eq. (22), which is finite there. But `docs/resonance-scaffolding-v1/` and `-v2/` are built
around that point and have no campaign of their own, so this is where it would otherwise be lost.
The option is a Levin-type evaluation of (4.11) at large $x$. **Recorded as a decision rather than
a §3 issue**: nothing in the tree is wrong, no board exists to host it, and the recon calls it a
campaign decision in terms.

---

## 8. Reading order for a new agent

0. [`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) — **the
   mechanism, end to end, for one $k$ mode.** What `crossover_z` is on each side, why $T_k$ has a
   gap and $G_k$ an overlap, what the policy objects do and do not decide, and how
   `build_partition` consumes the result. It is an explanation and lands no measurement; where it
   and a board disagree, the board is right. Read it **first**: everything below is written in its
   vocabulary, and §1.1's issue names do not parse without it.
1. This README §0–§2.
2. [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.1 — the eight seam issues and their
   hooks, plus the two of §2 (o) that the 2026-09-19 review added.
3. [`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md)
   — the original measurements, §1 (the seam) and §2 (the stored phase). Note §2.4's superseded
   block: the `bessel_phase` floor it describes is gone, replaced per §2.4.1.
4. [`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
   — **§0 first** (whose object is whose; it is a precondition for reading any of the rest), then
   §7, §7.2 and §8.
5. [`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) —
   **§0 and §1 before anything else in it**, then §9, the implementation brief. It is an input,
   not an authority: §5.1 still binds A1.
6. `prompts/source-remediation/IMPLEMENTATION_STATE.md` §3 — the entries for `[05-…]`, `[06-…]`,
   `[07-…]`, `[08-handover-clamp-error]` and `[12-handover-clamp-error-in-production]`, which carry
   the measurements this README only quotes.
7. `ComputeTargets/tests/test_quadsource_integral.py` — the module docstring (the exact/realistic
   flavour distinction), `TestHandOverClamp`, and `test_integrand_continuity_at_the_hand_over`.
