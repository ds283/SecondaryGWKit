# Tolerance and convergence campaign — one convergence test, five compute targets

**Origin:** [`prompts/GkTk-remedial/`](../GkTk-remedial/README.md) prompts 12 and 17, and the
review [`docs/gk-wkb-review-fable-2026-09-09.md`](../../docs/gk-wkb-review-fable-2026-09-09.md)
§10.1 and §12.5.
**Planned:** 2026-09-12, at `gktk-remedial` `891d84a`. **Not yet started.**
**Target branch:** to be cut from `main` **after `gktk-remedial` merges**, i.e. after GkTk-remedial
prompt 13 (§4.2).
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) · **Logs:** [`logs/`](logs/)

---

## 0. What this campaign is, and its boundaries

### 0.1 The one-sentence version

Every compute target in the pipeline that integrates or quadratures something has an accuracy
nobody has systematically established, and four of the five share two tolerance constants that
were never chosen for any of them: this campaign **builds one reusable convergence test, applies
it to all five targets on all three models, anchors it to the constant-$w$ closed forms already in
the tree, and decouples the tolerances so that each target can be converged on its own terms.**

It exists because GkTk-remedial prompt 12 set one constant on one wavenumber of one model, prompt
17 then measured the production grid and found the target missed at 3, 13 and 8 of 50 wavenumbers
on the three models, and the lever turned out to be a constant — `rtol` — that prompt 17 was not
allowed to touch because it is shared by four object types. The same review that measured the
transfer function says independently of the Green's function that "the error is set by `rtol`"
(§10.1). Nobody has checked the rest.

### 0.2 What "convergence test" means here, and why it is not an oracle comparison

There is no closed-form solution on `LambdaCDMModel` or `QCDModel` — the Hubble rate is a spline —
so for most (target, model) pairs the reference is **a converged run of the same integrator**, and
the test is a self-convergence one: build the reference at a tolerance far tighter than any
candidate, build it again a decade tighter still, and require that the reference move by at least
an order of magnitude less than the smallest difference the measurement intends to report. If it
does not, the measurement is measuring its own reference.

That surrogate is **calibrated, not assumed**. On a constant-$w$ background the closed forms in
[`ComputeTargets/analytic_Gk.py`](../../ComputeTargets/analytic_Gk.py) and
[`analytic_Tk.py`](../../ComputeTargets/analytic_Tk.py) are exact — `compute_analytic_T(k, w, tau)`
is the Bessel form $2^{n}\Gamma(n+1)(kc_s\tau)^{-n}J_{n}(kc_s\tau)$ with $n=\tfrac32+b$, reducing
at $w=\tfrac13$ to $3(\sin x - x\cos x)/x^3$ — so on the radiation control both the drift statistic
and the distance to truth can be computed and compared. GkTk-remedial prompt 17 did exactly this
for $T_k$ and found them the same order (4.21e-11 drift against 2.3–4.3e-11 from the oracle). Every
prompt here repeats that calibration for its own target before trusting the drift figure anywhere
else. **Radiation domination is the anchor, and the general constant-$w$ form is what the code
already provides.**

### 0.3 Boundary with `prompts/GkTk-remedial` (must be finished first)

That campaign owns the WKB phase representation and the numeric region's diagnostics, and it is
still running. Two of its results are **preconditions** here, not things to re-derive:

- **Its prompt 18** wires the cosmology's declared discontinuities to the ODE driver
  (`Quadrature/integrators/numeric_with_phase_cut.py`). Until that lands, the reference on
  `QCDModel` does not converge at four wavenumbers
  (`[17-qcd-reference-not-converged]`), and a tolerance measured against a reference that does not
  converge is worthless. **This campaign cannot start before it.**
- **Its prompt 06** removed the phase ODE. `GkWKBIntegration` and `TkWKBIntegration` no longer have
  tolerances with a referent (§2 (a)); their accuracy is a Gauss-order question, which changes what
  this campaign can even ask of them.

Do not edit anything that campaign's `IMPLEMENTATION_STATE.md` still shows as in flight; if the two
disagree about the state of a file, it is right and this README is stale.

### 0.4 Boundary with `AdaptiveLevin` and `QuadSourceIntegral`

`ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py` and `AdaptiveLevin/`
belong to [`prompts/levin-refactor`](../levin-refactor/) and
[`prompts/qsi-phase-groups`](../qsi-phase-groups/), and `docs/OPEN_ISSUES.md` §1.2 and §1.3 park
work against them. `QuadSourceIntegral` is also the one target whose tolerances are **already
decoupled** — `DEFAULT_QUADRATURE_ATOL`/`_RTOL`, distributed per sub-interval by log-width
(`QuadSourceIntegral.py:816-819`) — and whose `atol` was already chosen by measurement against an
analytic oracle (`prompts/source-remediation`, `[12-atol-too-loose-for-the-source-integral]`).

**This campaign does not edit any of those files.** It applies its convergence test to
`QuadSourceIntegral` **read-only**, as prompt 05, and hands the result to whoever owns it. If a
prompt here finds it must change one of those files to proceed, that is a stop.

### 0.5 What this campaign does *not* do

- **Does not change any integrator's algorithm.** No method swaps, no order changes, no
  re-derivations. It measures what the existing code does and chooses the constants it is given.
- **Does not revisit the initial conditions.** The $T=1,T'=0$ super-horizon condition holds a
  2.52e-6-of-envelope floor (`[00-tk-superhorizon-ic-series]`); it is a *floor to measure against*,
  never a target to beat. Claiming an accuracy below a declared floor is a campaign-wide stop.
- **Does not move the numeric→WKB hand-over**, which is `docs/OPEN_ISSUES.md` §1.1's.
- **Does not decide the Gauss orders.** GkTk-remedial prompt 02 measured $N_\tau=N_{c_s\tau}=N_F=
  N_\rho=4$ and its evidence stands; prompt 03 here *audits* them against the analytic anchors and
  reports, it does not re-choose them.

---

## 1. What this campaign does

Five prompts. The first three change **no production code**; the fourth is the only one that does.

**01 — the harness.** One reusable convergence facility, in the test tree rather than under
`docs/`, that takes a (target, model, $k$) and returns a converged reference, a drift figure, and —
where the background is constant-$w$ — the distance to the analytic anchor. Every later prompt uses
it, and GkTk-remedial's one-off sweep script is folded into it.

**02 — audit the numeric sectors.** `GkNumericIntegration` and `TkNumericIntegration` over the
production grids on all three models, across an `atol`×`rtol` matrix, anchored on radiation.
Produces the evidence for the decoupled constants and recommends them. **Stops for the user.**

**03 — audit the WKB sectors.** `GkWKBIntegration` and `TkWKBIntegration` have no tolerances left
(§2 (a)), so the same test is applied to the knob they *do* have — Gauss order — and to the
question of what should become of their vestigial `atol`/`rtol` key columns. **Stops for the user.**

**04 — decouple.** Per-sector tolerance constants and the `main.py` plumbing to carry them, on the
pattern GkTk-remedial prompt 12 established for a single constant, with the same `ast`-based
structural guard. The only prompt here that touches production code, and the one that invalidates
the datastore.

**05 — `QuadSourceIntegral`, read-only, and close-out.** The harness applied to the one target this
campaign does not own, the campaign document, and the hand-off.

### 1.1 Explicitly out of scope

Everything in §0.3–§0.5, plus: the datastore migration that prompt 04 implies (regeneration is a
user decision with a compute cost, §7 D2); any change to `GkSourcePolicy`, the rectifier, or the
`*Value` schemas; and the question of whether `QuadSourceIntegral`'s per-sub-interval `atol`
distribution is the right one, which is `levin-refactor`'s.

---

## 2. Design facts every prompt is built on

**(a) Two of the five targets have no tolerance to converge.** GkTk-remedial prompt 06 replaced the
WKB phase ODE with Gauss–Legendre tables. `GkWKBIntegration`/`TkWKBIntegration` keep `self._atol`
and `self._rtol` **only because they are part of the datastore lookup key**
(`ComputeTargets/GkWKBIntegration.py:334-336`; `RECONCILIATION.md` §10: "The primitive has no
tolerances. The columns stay… and the payload `metadata` records the Gauss orders actually used").
A prompt that proposes to "tighten the WKB tolerance" has misread the tree.

**(b) `QuadSourceIntegral` is already decoupled, and is not ours.** `DEFAULT_QUADRATURE_ATOL =
1e-32`, `DEFAULT_QUADRATURE_RTOL = 1e-8`, chosen by the `source-remediation` campaign against an
analytic oracle, with `atol` distributed across sub-intervals by log-width. §0.4.

**(c) In both numeric sectors the error is set by `rtol`, not `atol` — measured twice,
independently.** Review §10.1, on $G_k$: "The error is set by `rtol`… tightening `rtol` by $10^3$
costs 2.3× and buys $10^3$." GkTk-remedial prompt 17 §7, on $T_k$: at fixed `atol = 1e-13`, one
decade of `rtol` takes the worst wavenumber of every model from 2.5e-4/8.6e-4/2.8e-4 to
1.0e-7/7.4e-8/9.8e-7 for +23–25 % evaluations, while two decades of `atol` fix nothing. This is the
campaign's central prior, and prompt 02 exists to confirm or destroy it.

**(d) `atol` is not uniform across sectors because the quantities are not.** $|G|\sim10^{10}$ at the
hand-over in `Mpc_units`, so an absolute floor of 1e-10 never binds; $|T|\sim10^{-5}$ deep inside
the horizon, so the same floor acts as a $10^{-5}$ *relative* tolerance. That asymmetry is why
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` exists (`config/defaults.py:12-33`), and it is the argument for
decoupling generalised: **a shared absolute tolerance is a statement about magnitudes, and the five
targets do not share magnitudes.**

**(e) The floors, which are not targets.** $T_k$ numeric: the $T=1,T'=0$ initial condition holds
2.52e-6 of the envelope, $k$-independent, confirmed against the exact $T$ at all 50 wavenumbers
(GkTk-remedial prompt 17 §8). $T_k$ WKB: LG truncation, 3.8e-5 of the envelope from $x_i=24$.
Phase: $\varepsilon k\tau$, 3e-7 rad at $k=10^5$ to 9e-4 rad at $3\times10^8$. On `QCDModel`,
whatever prompt 18 leaves of the $H(z)$ discontinuity floor. **An agent that reports an accuracy
below a floor has made an error, and it is a campaign-wide stop.**

**(f) Decoupling a tolerance is a datastore change, and the plumbing is the risky half.** Every
tolerance is part of its object's lookup key, so a new constant makes every existing row of that
type unreachable. GkTk-remedial prompt 12 did this once, for one constant on one target, and still
needed a `STRUCTURALLY REQUIRED` deviation to repair a `RayWorkPool` batch that dispatched over the
wrong list once two objects carried different tolerances. `main.py` names the five targets 20, 29,
13, 16 and 25 times respectively. Prompt 04 must carry the `ast`-based site-classification guard
prompt 12 built (`ComputeTargets/tests/test_main_plumbing.tk_numeric_tolerance_sites`), generalised.

**(g) The analytic anchors are constant-$w$, not radiation-only.** `compute_analytic_G`,
`compute_analytic_Gprime`, `compute_analytic_T`, `compute_analytic_Tprime` all take $w$. Radiation
($w=\tfrac13$) is the production-relevant case and the one the stand-in `RadiationModel` provides,
but a prompt may use another constant $w$ to separate a $w$-dependent error from a solver one.

**(h) Counts, not wall time.** This machine's elapsed times overstate by up to 53 %
(GkTk-remedial `IMPLEMENTATION_STATE.md` §5 note 14). Every cost figure in this campaign is
RHS evaluations, integrand evaluations or Hubble calls.

---

## 3. The prompts

| # | Prompt | Covers | Files | Production code? | Model |
|---|---|---|---|---|---|
| 01 | The convergence harness | §2 (g); GkTk-remedial `[17-qcd-reference-not-converged]`'s method | new `ComputeTargets/tests/convergence_reference.py` + its test; folds in `docs/gktk-remedial/tk_numeric_atol_sweep.py` | **No** | Opus |
| 02 | Audit the numeric sectors | §2 (c), (d), (e); review §10.1, §12.5 | new `docs/tolerance-convergence/numeric_sweep.py`, `NUMERIC-CONVERGENCE.md` | **No** | Opus |
| 03 | Audit the WKB sectors | §2 (a); GkTk-remedial prompt 02's Gauss orders | new `docs/tolerance-convergence/wkb_order_sweep.py`, `WKB-CONVERGENCE.md` | **No** | Opus |
| 04 | Decouple the tolerances | §2 (d), (f); §7 D1 once settled | `config/defaults.py`, `main.py` (every `object_get` of the retuned targets), the four `ComputeTargets/*Integration.py` constructors if a signature moves, `ComputeTargets/tests/test_main_plumbing.py` | **Yes — the only one** | Opus |
| 05 | `QuadSourceIntegral` and close-out | §0.4, §2 (b) | new `docs/tolerance-convergence/TOLERANCE-CONVERGENCE.md`; `docs/OPEN_ISSUES.md` | **No** | Opus |

### 3.1 Prompt 01 — the convergence harness

The facility GkTk-remedial prompt 17 needed and did not have: its convergence test is inline in
`sweep_model`, so it could not be reused for the second sector without copying. Build it properly,
in `ComputeTargets/tests/` beside `wkb_reference.py` (so it is importable by tests and by `docs/`
scripts alike, and subject to the suite), with at least:

- a converged reference for a given (target, model, $k$, geometry), at a reference tolerance pair
  the caller supplies;
- the **drift** statistic — the same reference one decade tighter, differenced in the caller's
  error measure — with the criterion (drift $\le\frac1{10}$ of the smallest difference to be
  reported) evaluated, not just reported;
- the **anchor** comparison against `compute_analytic_{G,T}{,prime}` wherever the model is
  constant-$w$, so the drift statistic is calibrated at every use and not only in prompt 17;
- the envelope-relative, phase and difference error measures, reusing `wkb_reference`'s definitions
  rather than restating them.

It must need no Ray and no datastore (call remotes through their undecorated `_function`, use the
prompt 01 stand-ins), and it must reproduce GkTk-remedial prompt 17's published drift figures for
$T_k$ exactly — that is its acceptance test, and it is why the campaign starts here.

### 3.2 Prompt 02 — audit the numeric sectors

Both numeric targets, three models, the production grids, across a matrix in `atol` and `rtol` —
the point being that prompt 17 held `rtol` fixed and so could only see one axis. Report per
(target, model, $k$, `atol`, `rtol`): maximum, second-largest and median envelope-relative error,
the location of the maximum, and the RHS-evaluation count; and per (target, model) the distribution
over the grid, since prompt 17 established that a single $k$ is not characteristic.

Answer explicitly: **is §2 (c) right** — is `rtol` the lever in both sectors? What does each sector
cost at the tolerance that first reaches its floor? And what would the decoupled constants be, with
the evidence? Recommend; do not decide (§7 D1).

### 3.3 Prompt 03 — audit the WKB sectors

`GkWKBIntegration` and `TkWKBIntegration` have no live tolerances (§2 (a)). The same convergence
test applies to the knob they do have: the Gauss orders $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$,
all currently 4. Sweep the order against the analytic anchors and the converged-reference drift, on
all three models, and answer: are they converged at 4 at every production $k$, or was prompt 02 of
the other campaign right at the $k$ it measured and lucky at the rest? Then the API question this
campaign inherits: the vestigial `atol`/`rtol` columns are part of the lookup key but describe
nothing — **recommend** keeping, dropping or repurposing them, with the schema-churn cost of each
(§7 D3). No production code either way.

### 3.4 Prompt 04 — decouple the tolerances

The only production change in the campaign, and only after the user has settled D1. Per-sector
constants in `config/defaults.py` with the same standard of comment `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`
already carries — the measurement that chose the number, in the file, where the next reader will
find it. Then the `main.py` plumbing: one `tolerance` object per sector in the same `ray.get`, and
**every** `object_get` of that target switched, with an `ast`-based guard that enumerates the sites
and fails when an unclassified one appears (§2 (f)). Expect the batch-dispatch hazard prompt 12
hit; the guard exists because it is silent.

### 3.5 Prompt 05 — `QuadSourceIntegral`, read-only, and close-out

Apply the harness to `QuadSourceIntegral` **without editing it or anything else in §0.4**: is it
converged at `quad_atol = 1e-32`, `quad_rtol = 1e-8` on the production configuration, measured
against the analytic oracle the `source-remediation` campaign used? Report to `levin-refactor` and
`qsi-phase-groups` through `docs/OPEN_ISSUES.md` §1.2/§1.3 rather than acting. Then the campaign
document: the five targets, the constants each ended at, the evidence, and the floors each is now
limited by.

---

## 4. Dependencies and ordering

```
01 ──▶ 02 ──▶ [user settles D1] ──▶ 04 ──▶ 05
   └──▶ 03 ──▶ [user settles D3] ──┘
```

02 and 03 both consume 01 and are otherwise independent; run 02 first, since D1 is the decision
with a compute cost attached. 04 must not start until D1 is settled — it is the prompt that
invalidates the datastore, and settling the constant afterwards would invalidate it twice.

### 4.1 Natural stopping points

After **02** and after **03**, always: each ends in a recommendation the user must accept before
anything is changed. After **04**, because the datastore regeneration is a compute decision (§7 D2).

### 4.2 Relationship to GkTk-remedial

**This campaign starts after `gktk-remedial` merges to `main`** — §0.3. Prompt 18 of that campaign
is a hard precondition (the QCD reference must converge before any tolerance is measured against
it), and prompt 13 is a practical one: it builds a fresh datastore and runs a scoped pipeline, and
prompt 04 here will invalidate the numeric-sector rows of whatever it built.

That cost is accepted deliberately rather than by oversight. GkTk-remedial prompt 13's subject is
the *phase representation*, whose accuracy no longer depends on any tolerance — prompt 06 removed
the ODE — so its conclusions survive this campaign; only its numeric-sector rows need regenerating.
Holding that campaign open instead, to run this one first, was judged the larger cost (2026-09-12).
**If the datastore prompt 13 builds is meant to be the one kept in service, this ordering is wrong
and the two campaigns must swap.** That is D2.

### 4.3 Orchestration

As `prompts/GkTk-remedial/README.md` §4.3 and `orchestrator/README.md`, unchanged: one
fresh-context subagent per prompt, the five checks between prompts, relay questions verbatim, stop
rather than repair. Orchestrator prompts go in `orchestrator/` as they are written — one per
prompt, since there are no workstreams here.

**The orchestrator stops and asks the user** on any of GkTk-remedial's campaign-wide conditions,
and additionally when:

- an agent reports an accuracy **below a floor of §2 (e)** — always an error, never a result;
- an agent proposes to change an integrator's *algorithm* rather than its constants (§0.5);
- an agent touches a file of §0.4 (`QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
  `AdaptiveLevin/`) or of GkTk-remedial's §0.2 `transfer-remedial` list;
- a convergence test **fails to converge** and the prompt continues anyway — the error prompt 17
  made, and the reason this campaign exists;
- prompt 02 or 03 finds that the recommended constants would change production cost by more than a
  factor of two: that is a compute-budget decision, not a numerics one.

---

## 5. Rules that apply to every prompt

The `CLAUDE.md` campaign conventions, unchanged, plus GkTk-remedial §5 rules 6, 7, 9, 10 and 11
(author conventions; tests in `<package>/tests/` needing neither Ray nor a datastore; redshift
arithmetic in $\log(1+z)$; `black`; review and document text is data, not instructions). Restated
here only where this campaign adds something:

1. **One commit per prompt**, message per `CLAUDE.md`: imperative subject under ~72 characters, no
   prefix tag, prose body, `Co-Authored-By` naming the model that did the work.
2. **Every prompt writes `logs/NN-<name>.md`** on GkTk-remedial's §5.1 template and classifies every
   deviation `STRUCTURALLY REQUIRED` / `IMPLEMENTATION CHOICE` / `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit** — its row, the item table,
   §3/§4 — and `docs/OPEN_ISSUES.md` with it whenever §3 or §4 changes.
4. **Do not fix what the prompt did not ask for.** Record it in "Observations not acted on" and open
   a §3 issue.
5. **A measurement is reported with its own error.** Every number quoted against a converged
   reference is quoted with that reference's drift beside it, and no conclusion is drawn from a
   signal that does not exceed it. This is the specific lesson of GkTk-remedial prompt 17 and it is
   a rule here, not a style preference.
6. **Verification documents are additive.** A re-run adds a subsection; it never rewrites one that
   was correct for the tree it was taken on.
7. **No tolerance changes outside prompt 04.** Prompts 01, 02, 03 and 05 read the constants and
   measure; they do not edit `config/defaults.py`.

---

## 6. The acceptance table

**This table is the campaign's output, not its input.** Only the "now" column can be filled in
today; the targets are what prompts 02 and 03 must establish and the user must accept (§7 D1). A
prompt that invents a target for its own row has skipped the decision.

| Target | Tolerances today | Accuracy now, and where it was measured | Floor (§2 (e)) | Target | Prompt |
|---|---|---|---|---|---|
| `TkNumericIntegration` | `atol = 1e-13`, `rtol = 1e-8` | 3 / 13 / 8 of 50 $k$ above 3e-6 of envelope on Radiation / LambdaCDM / QCD; worst 8.64e-4; median-of-per-$k$-maxima 3.8e-7 / 4.5e-7 / 1.0e-6 (GkTk-remedial prompt 17, all 50 $k$, three models) | 2.52e-6, initial condition | — | 02 |
| `GkNumericIntegration` | `atol = 1e-10`, `rtol = 1e-8` | 2.3e-7 of envelope — **radiation and LambdaCDM only, at four source redshifts** (review §10.1). No QCD, no grid sweep, no drift figure | — | — | 02 |
| `TkWKBIntegration` | none live (§2 (a)) | $T_{\rm WKB}$ radiation control 3.8e-5 of envelope from $x_i=24$ (GkTk-remedial prompt 07) | LG truncation | — | 03 |
| `GkWKBIntegration` | none live (§2 (a)) | $\theta_G$ 13.9 rad at $k=10^5$ and 7366 rad at $3\times10^8$ against target ≤1e-5 / ≤5e-3 rad (GkTk-remedial README §6) | $\varepsilon k\tau$ | — | 03 |
| `QuadSourceIntegral` | `quad_atol = 1e-32`, `quad_rtol = 1e-8`, decoupled already | `atol` chosen against an analytic oracle by `source-remediation` log 12; `rtol` confirmed non-binding (1e-8 → 1e-11 bit-identical on 159 items) | — | read-only | 05 |

Error definitions are GkTk-remedial README §6's, unchanged and deliberately: **envelope-relative**
divides by the local Liouville–Green envelope, not by the value; **phase error** is absolute
radians of the unwrapped phase at the supplied double $z$; **difference error** is relative to the
interval quantity, never to the absolute.

---

## 7. Decisions left to the user

**D1 — the decoupled constants.** What `atol` and `rtol` each retuned target gets. Prompt 02
recommends with evidence; prompt 04 may not start until this is settled. The likely shape, from
§2 (c), is that `rtol` tightens by a decade in both numeric sectors at +23–25 % evaluations — a
recurring compute cost on a sector with ~65,000 objects per model, which is why it is a decision
and not a measurement.

**D2 — the datastore, and the campaign ordering.** Prompt 04 makes every row of the retuned targets
unreachable by key. If the datastore GkTk-remedial prompt 13 builds is meant to stay in service,
this campaign must run *before* that prompt instead of after it, and GkTk-remedial stays open
longer (§4.2). Settle before prompt 04, and ideally before GkTk-remedial 13.

**D3 — the vestigial WKB key columns.** `GkWKBIntegration`/`TkWKBIntegration` carry `atol`/`rtol`
columns that are part of the lookup key and describe nothing (§2 (a)). Keep (schema churn avoided,
a permanently misleading column), drop (a migration), or repurpose to record the Gauss orders that
actually set the accuracy. Prompt 03 recommends.

**D4 — whether `QuadSourceIntegral` is in or out.** §0.4 puts it out, read-only, because two other
campaigns own those files. If it should instead be retuned here, that has to be agreed with those
campaigns first, and prompt 05 changes character entirely.
