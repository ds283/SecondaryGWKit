# Prompt 06i — Filon references for Z3's calls (a measurement)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-01, on the user's word (the board's Decisions, **Z3 references**). The
orchestrator proposed a Filon-type reference in place of 06e's Gauss–Legendre one, validated first
where Gauss–Legendre resolved and then used where it did not. The user: *"Yes, I think that would
be a very good check."*

**Where it runs:** on the laptop, Ray-free, with no lock and no store. **It changes no code.** The
13 captured inputs are copied from the Mac Studio by the orchestrator before dispatch (§1). One
agent, in a worktree; the orchestrator reviews and makes the records commit.

**When:** it depends on nothing after 06e, and runs before 06g's Part B, whose census re-scores
Z3's samples against these references (§0.4).

**Closes:** nothing. It bears on `[06e-trusted-leaf-returns-its-coarsest-value]` and
`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`. It is evidence for 06g, not a remedy.

**Recommended model:** **Opus**.

**Read first:**
1. `docs/levin-phase-floor/VERIFICATION.md` §6e.3–§6e.4, and 06e's prompt §5, Z3: the sample, the
   counting rule, the scoring rule.
2. `docs/levin-phase-floor/06e/census/references/`: `orch_ref.py` (the Gauss–Legendre reference),
   `score_z3.py`, and `out/z3_scores.json` (every reference at every resolution, the census's value
   and abserr under 06c and 06e, by sample).
3. `docs/levin-phase-floor/06e/investigations/d3-leaf-values/README.md` and `d3_probe.py`: how a
   captured call is loaded and replayed.
4. `LiouvilleGreen/three_bessel_integrals.py`: `BesselPhaseGroup` (its class docstring for
   $\theta = K x + C + R(x)$, `__init__`, `residual`, `theta_deriv`) and `_three_bessel_Levin`'s
   amplitudes; `LiouvilleGreen/bessel_phase.py`'s "Accuracy of the supplied argument".
5. `ComputeTargets/phase_groups.py`: how a cosmological group composes `theta`, `theta_mod_2pi` and
   `theta_deriv` over $\log(1+z)$.
6. `scipy.integrate.quad`'s `weight='cos'|'sin'` (QUADPACK QAWO), and Filon–Clenshaw–Curtis
   quadrature (e.g. Domínguez, Graham and Smyshlyaev, Numer. Math. 118 (2011) 299; Olver, IMA J.
   Numer. Anal. 27 (2007) 213 for the moment-free form).

**Gate:**
- 06e's records commit has landed (the board's 06e row is ✅);
- the 13 inputs are on the laptop, each equal by SHA-256 to `06e/KEPT_ON_MAC_STUDIO.md`'s list;
- `git status --short` is empty in the main checkout.

---

## 0. What this prompt is

### 0.1 Why Gauss–Legendre could not settle 7 of Z3's 13 references

`orch_ref.py` integrates the call's integrand on panels a fixed number of radians wide, with
Gauss–Legendre nodes, summed by `fsum`. Its rounding is about $\varepsilon\int|{\rm integrand}|$,
and these calls cancel by many orders: they declare relative errors of $10^{-9}$ to $10^{-12}$ of
values near $10^{-27}$. On samples 1, 10, 13, 22, 23, 25 and 27 the three resolutions disagree by
more than $1/100$ of the declared error, so Z3 dropped them. Two of the dropped (13 and 25) read
near $10\times$ their declared error, with scatter of $18\times$ and $66\times$: no reading at all.
Thirteen references took about 19 hours, 2–3.5× the estimates, and the other 17 of Z3's 30 were not
attempted (estimated at 20 h or more each).

### 0.2 The Filon idea

For $\int g(y)\,e^{i\omega y}\,dy$ with $g$ smooth, a Filon-type rule interpolates $g$ alone and
integrates the interpolant against $e^{i\omega y}$ exactly, through moments
$\int T_k(s)\,e^{i\omega h s}\,ds$. The cancellation between oscillations happens inside the exact
moments, so the rounding is about $\varepsilon\,|g|/\omega$, the size of the answer, not of
$\int|g|$; and the cost depends on $g$'s smoothness, not on the number of oscillations. QUADPACK's
QAWO is the adaptive form, with Chebyshev moments (`scipy.integrate.quad(..., weight=...,
wvar=...)`).

### 0.3 Making the phase linear

**Route A, the analytic Bessel blocks** (samples 8, 9, 10, 22, 23, 25, 27). The integration
variable is $u = \log\eta$ and the phase is $\theta = K\eta + C + R(\eta)$ exactly
(`BesselPhaseGroup`). In $\eta$,
$$\int f(u)\,e^{i\theta(u)}\,du = \int \Bigl[\,\frac{f(\log\eta)}{\eta}\,e^{i(C + R(\eta))}\Bigr]\,e^{iK\eta}\,d\eta,$$
a linear phase with $\omega = K$, and the bracket smooth (the residual $R$ varies slowly). It needs
no inverse map, and it does not need the phase to be monotone: near resonance the bracket varies
faster and needs more nodes. $K$ is taken **exactly** as $\hat K + K_{\rm lo}$ from the group's
coefficients and signs (`math.fsum`, as 06h's `__init__` does), so Route A is a reference for the
intended integrand, independent of how any version of `sin_cos` rounds $K\eta$.

**Route B, the cosmological group calls** (samples 1, 2, 3, 13, 14, 15). There is no exact linear
part. The phase itself is the variable, $t = \theta(x) - \theta(a)$, giving
$\int (f/\theta')(x(t))\,e^{i(t + \theta(a))}\,dt$, linear with $\omega = 1$. It needs $x(t)$ at
every node (a root solve with $\theta'$ supplied), and it fails where $\theta' = 0$, where $f/\theta'$
is singular. **Its precision is the open question:** an error $\delta$ in $t(x)$ is a phase error
$\delta$ in the integrand, and $\theta$'s span here reaches $10^{6}$ rad and more. The agent states
how $t(x)$ is formed (from `theta`, from integrating `theta_deriv`, or corrected node by node by
`theta_mod_2pi`) and bounds its error, before scoring.

**Product rounding.** QAWO in double precision forms $\omega y$ internally, the same product
rounding 06h removed from `sin_cos`. Route A shifts the variable panel by panel,
$\eta = \eta_a + s$, with $e^{iK\eta_a}$ formed exactly (TwoProduct reduction, or `mpmath`), or
uses a Filon–Clenshaw–Curtis rule written in `mpmath` (§1).

### 0.4 What it is for

- **Validated:** whether the Filon references reproduce the Gauss–Legendre ones where those
  resolved (samples 2, 3, 8, 9, 14, 15).
- **Then used:** a reading for the seven dropped samples, scored by Z3's rule against 06e's census.
- **For 06g:** the references 06g's census re-scores. 06g returns the finest value computed by a
  trusted leaf, and predicts that samples 2, 8, 9 and 14, at 0.998–0.9995 today, fall well below 1.

### 0.5 What it does not do

- It does not test the **phase representation**: every reference here is built from the call's own
  amplitude and phase (Route A from the group's own $K$, $C$ and residuals). 06h's test module
  tests the phase against `mpmath` Bessel functions.
- It does not reach the 17 Z3 samples never referenced, unless §4's optional item is run.
- It changes no code and proposes none.

## 1. The harness

**The tree:** a worktree at the dispatch SHA, for the scripts. **The integrand is loaded at 06e's
census tree** `6d81fad` (code `f4d8e17`'s), in a second worktree. The captured `BesselPhaseGroup`
objects predate 06h, whose `sin_cos` and `theta_abserr` read an attribute (`K_lo`) those objects lack.
Route A does not call `sin_cos`, but every cross-check that calls `theta_mod_2pi` must run there.

**The inputs:** Z3's 13 pickles, copied by the orchestrator from the Mac Studio's
`var/levin-phase-floor-06e/census/references/inputs/` into
`var/levin-phase-floor-06i/inputs/`, loaded as `d3_probe.py` loads them, with
`var/levin-phase-floor-06e/census/scripts/` on the path.

**Two implementations, each checked against the other:**
1. **QAWO** (`scipy.integrate.quad` with `weight` and `wvar`), in double precision, with the
   panel-by-panel shift of §0.3 and `limit`, `epsabs` and `epsrel` stated;
2. **Filon–Clenshaw–Curtis in `mpmath`**: on each panel, the amplitude interpolated at $N$
   Chebyshev points (evaluated in double, by the call's own callables), the moments computed in
   `mpmath` at 40 digits, panels summed in `mpmath`. Two values of $N$ and two panellings.

**The method check, before any sample (Gate R0):**
- on closed forms, e.g. $\int_a^b y^n e^{-\alpha y} e^{i\omega y}\,dy$ and
  $\int_a^b e^{i(\omega y + \beta/y)}\,dy$ against `mpmath.quad` at 50 digits, for
  $\omega \in \{10, 10^3, 10^6, 10^9\}$ and spans from 1 to $10^3$ wavelengths and beyond: each
  implementation within $10^{-13}\max|g|/\omega$ of the closed form;
- on one sample of each route, Route A's integrand at 20 points compared with the captured
  integrand $f_0\sin(\theta_{\rm mod}) + f_1\cos(\theta_{\rm mod})$ at the census tree: agreement to
  the phase declaration times $|f|$.

## 2. Route A — the analytic blocks

For each of samples 8, 9, 10, 22, 23, 25 and 27:
- the reference by both implementations of §1, at two resolutions each;
- **its own scatter** (max minus min over the four readings), against $\min({\rm abserr}_{06c},
  {\rm abserr}_{06e})$;
- **the counting rule, Z3's:** a reference counts if its scatter is at most $1/100$ of that
  minimum;
- for samples 8 and 9, the difference from 06e's Gauss–Legendre reference, against the sum of the
  two scatters;
- wall time per reading.

## 3. Route B — the cosmological groups

For each of samples 1, 2, 3, 13, 14 and 15, as Route A, with:
- **$t(x)$'s construction and its error bound**, stated and measured on each sample before the
  reference is formed;
- the stationary points of $\theta$ on the span listed (from `theta_deriv`'s sign changes); a sample
  with one is split there, and the piece about it integrated by `mpmath.quad` at 30 digits, or the
  sample reported as not referenced by Route B;
- for samples 2, 3, 14 and 15, the difference from 06e's Gauss–Legendre reference.

## 4. Optional, if §2 and §3 are fast

On the Mac Studio, the 17 Z3 samples never referenced, by whichever route R1 validated, with their
inputs captured afresh at `6d81fad` as `x1_capture.py` does. **Only on the user's word** after
§2–§3 are reported.

## 5. Predictions, fixed before the run

Copied into the log unchanged before any sample is run. Each is scored **holds**, **fails** or
**undecided** against the rule as written.

- **R0 — the method check.** **Rule:** every closed-form cell of §1 is met by both implementations,
  and the integrand comparison holds on both samples. **Holds** if all do. A failure is a stop.
- **R1 — Filon reproduces Gauss–Legendre where it resolved.** **Rule:** on each of samples 2, 3, 8,
  9, 14 and 15, the Filon reference counts, and differs from 06e's Gauss–Legendre reference by at
  most the sum of the two scatters plus $1/100$ of the minimum abserr. Scored per route: **Route A
  holds** if samples 8 and 9 meet it; **Route B holds** if samples 2, 3, 14 and 15 do.
- **R2 — Filon resolves where Gauss–Legendre did not.** **Rule:** for a route that holds R1, each of
  its dropped samples (Route A: 10, 22, 23, 25, 27; Route B: 1, 13) counts. **Holds** if all do.
- **R3 — Z3 re-scored.** **Rule:** for every reference that counts under a route holding R1,
  $|v_{06e} - {\rm ref}| \le {\rm abserr}_{06e}$. **Holds** if none violates it. Fewer than five
  newly counted is **undecided**. The 06c column is reported beside it.
- **R4 — the cost.** **Rule:** every Route A reading takes under 10 minutes on the laptop.
  **Holds** if all do. Descriptive beside it: Route B's times, and Gauss–Legendre's from 06e.

## 6. Records

- **`var/levin-phase-floor-06i/log.md`** in README §5.1's form, every deviation classified; the
  orchestrator copies it to `prompts/levin-phase-floor/logs/06i-filon-references.md`.
- **`docs/levin-phase-floor/06i/`:** the scripts, the method check, per-sample tables (every
  reading, its scatter, the counting verdict, the 06c and 06e ratios, the Gauss–Legendre
  difference), and a `results.md` stating each verdict with its numbers. The references are given
  as hex doubles, and as 30-digit strings where `mpmath` formed them.
- **Provenance** in every output header (README §5 rule 5): the scripts' SHA-256, the tree, and each
  input's SHA-256.
- `docs/levin-phase-floor/VERIFICATION.md` §6i, written by the orchestrator.

## 7. Acceptance

1. R0 holds; R1–R4 each scored against the rule as written, with their numbers.
2. Every sample of §2 and §3 tabulated, referenced or reported as not referenced with the reason.
3. No code changed, no store opened, no Ray started; the inputs unchanged by SHA-256.
4. The records of §6.

## 8. Stop conditions — stop and ask

- **The Gate** fails, or an input does not load or does not match its SHA-256.
- **R0 fails** on any cell: report the cell, both implementations' errors and the closed form.
- **The two implementations disagree** on a sample by more than the larger scatter: report both.
- **Route B's $t(x)$ cannot be bounded** below $1/100$ of a sample's minimum abserr. Report the
  bound, and run Route B no further on that sample.
- **A code change** looks necessary. Report it; do not make it.

## 9. Amendments

**1 — R0's integrand rule (2026-11-01, after the stop at Gate R0).** As written, the rule compared
the captured integrand with Route A's to "the phase declaration times $|f|$" at 06e's census tree.
That declaration is the one 06h found incomplete: it omits the rounding of $\hat K\hat x$, of $K$
and of the abscissa, which Route A excludes by design. The cosmological groups declare no phase
error at all. So the rule could not hold, and the fault was the prompt's. The user ruled
(*"Yes, (A), and your two additions"*):
- R0 is recorded as failing as written, the integrand cell badly posed. The closed forms are met;
  Route A's integrand agrees once the three roundings are removed; Route B's agrees up to the
  phase jitter.
- **Route B integrates the least-squares smooth phase**, the only well-defined target: every method
  integrates the jitter at different nodes.
- **Added:** each Route B call reports the jitter's estimated contribution, so an R3 violation can be
  attributed; and the all-order-½ samples (8, 10, 22, 23, 25) are compared with a closed form in
  Ci and Si.
- The jitter finding narrows `[05a-t-phase-abscissa-rounding-remains]`; no new issue.

**2 — sample 8 (2026-11-01).** §8's implementation check exceeded 1 on sample 8 (5.1×, $3.5\times10^{-25}$,
$1.3\times10^{-8}$ of the minimum abserr, the closed form between the two). The user: *"I accept
sample 8."*

**3 — §4 (2026-11-01).** The 17 never-referenced Z3 samples are not run here. The user: *"I think
we should leave the 17 Z3 samples that could not be computed for 06g."*
