# Implementation state — the phase-representation campaign

**Campaign:** [`README.md`](README.md) · **Source measurements:**
[`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md) §3.5, §3.6, §3.7
**Baseline commit:** `9daa2cb` (`gktk-remedial`, clean — the `GkTk-remedial` close-out)
**Last updated:** 2026-09-13 — **prompt 02 stopped; the campaign is at 1 / 2 and needs a user
decision.** Prompt 01 landed: `WKB_mod_2pi` and `simple_mod_2pi` derive their cycle count from the
exact `fmod` remainder instead of a second rounded division, so on the production geometry
LambdaCDM $G_k$ at $k=3\times10^8$ is **0 inconsistent of 77,975** (was 1), the
$|\theta|\sim4\times10^{12}$ uniform control **0 of 400,000** (was 25), and that case's consumer
phase error **0.0000e+00 rad, 0.00 ulp of the span** (was 6.1748 rad, 12,646 ulp), with the
remainder bit-identical so no stored $G$ or $T$ moved. **Prompt 02 is `BLOCKED` on its own §2
item 2** (README §7 D2): a repeated-knot vector is **singular on all six production grids** at
`BREAK_POINT_ALL`, and at `BREAK_POINT_DISCONTINUITY` it makes the measured consumer error **2×
worse** in both sectors. It also narrows the issue it was to close — the kink at the declared
discontinuity is **1.6e-08 / 1.4e-07 rad**, 1 % and 4 % of the 1.907e-06 / 3.186e-06 rad it was
charged with. **No production file changed.**

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

| # | Prompt | Closes | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [`WKB_mod_2pi` cycle count](01-wkb-mod-2pi-cycle-count.md) | `[13-wkb-mod-2pi-cycle-count-inconsistent]` | Opus | ✅ | *"Derive the WKB cycle count from the exact remainder"* (SHA not embedded, per the campaign convention) | [`logs/01-wkb-mod-2pi-cycle-count.md`](logs/01-wkb-mod-2pi-cycle-count.md) |
| 02 | [`PrimitivePhase` break-point knots](02-primitive-phase-break-point-knots.md) | `[13-consumer-spline-crosses-eos-break-points]` — **not closed** | Opus | ⛔ | *"Stop prompt 02: the repeated-knot vector does not construct"* (SHA not embedded, per the campaign convention) | [`logs/02-primitive-phase-break-point-knots.md`](logs/02-primitive-phase-break-point-knots.md) |

**Progress:** 1 / 2 complete, 1 blocked. The campaign cannot proceed without the README §7 D2
decision, which is the user's; the three ways forward are ranked in log 02's "State handed to the
next prompt" item 3.

---

## 2. Mechanism-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| P1 | **DEFECT, accuracy** | `WKB_mod_2pi`'s cycle count is a rounded division while its remainder is an exact `fmod`, so at large $\|\theta\|$ the stored pair reconstructs $\theta-2\pi$. 1 of 77,975 production $G_k$ samples at $k=3\times10^8$ on LambdaCDM; 6.17 rad of consumer error against a 9.15e-4 rad floor; rate = the half-ulp width of $\|\theta\|/2\pi$, growing linearly with $k$. `simple_mod_2pi` shares the construction. | 01 | ✅ |
| P2 | **DEFECT, accuracy** | `PrimitivePhase` splines $\varphi$ with `make_interp_spline`'s default knots, interpolating across `QCD_EOS`'s declared break points. 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$, 428 ulp) at $z=4.24\times10^7$, against 1.00 ulp elsewhere; and `theta_deriv` misses $\omega$ by 2.3e-7–3.3e-4 relative across the QCD interior. **Re-attributed by prompt 02:** the kink at the declared discontinuity is only 1.6e-08 / 1.4e-07 rad of that, the rest is $\varphi$'s own structure on a $\pm3$-interval scale; and the `theta_deriv` misses at $k\ge10^7$ are the 2–6 ulp storage granularity of $\varphi$ itself (`[00-consumer-anchoring-floor]`), not the knots. | 02 | ⛔ |

**Out of scope (do not schedule):** the numeric→WKB hand-over (`docs/OPEN_ISSUES.md` §1.1); the
$T_k$ numeric tolerance and `rtol` (`prompts/tolerance-convergence`); raising the LG order;
per-region anchoring (`[00-consumer-anchoring-floor]`); the super-horizon series initial condition;
`[02-qcd-T-z-spline-node-tolerance]`, which prompt 02 must *separate out* but may not fix.

---

## 3. Active and unresolved issues

None opened by prompt 01 — its log's "Observations not acted on" lists five, none of which is a
new defect: an existing entry on another board (`[10-wrap-theta-loop-at-large-phase]`), a property
of a file this prompt may not touch (`GkSource.py`'s one-sided rectifier trigger), the unchanged
non-production status of `WKB_product_mod_2pi`, a restatement of the `GkTk-remedial` board's §5
note 14 on this machine's wall-clock noise, and the pre-existing `black --check` state of `docs/`
and `AdaptiveLevin/`.

Two opened by prompt 02:

- **[02-consumer-phi-below-the-storage-granularity]** *(opened by prompt 02, 2026-09-13)* — the
  residual $\varphi$ a `PrimitivePhase` splines is recovered as
  $\theta_{\rm stored} - {\rm sign}\,k\,{\rm leading}.\delta$, a difference of two numbers of size
  $k\tau$, so its own dynamic range shrinks as $k$ grows while the rounding it inherits does not.
  On `QCD_Cosmology` in the $G_k$ sector the whole range of $\varphi$ is **6.0 ulp** of the stored
  phase at $k=10^7$ (7 distinct values over 1,218 samples) and **2.0 ulp** at $3\times10^8$
  (**3** distinct values over 1,377), every one an exact multiple of ${\rm ulp}(\theta)$. The
  consequence is measurable and it is a *regression against doing nothing*: scored against
  $\omega$ over the deep interior, `theta_deriv` misses by **5.9403e-06** at $k=10^7$ and
  **1.7475e-04** at $3\times10^8$ with the $\varphi$ spline's derivative included, against
  **1.9982e-06** and **1.7392e-05** with it set to zero — a factor 3 and a factor 10 *worse* than
  omitting $\varphi$ entirely. (At $k=10^5$, where $\varphi$ spans 1,112 ulp, the spline earns its
  place: 6.8497e-06 → 2.3123e-07.) **Impact:** the two rows of
  `docs/gktk-remedial-verification.md` §3.6 that miss README §6's $10^{-6}$ relative target are
  these, and prompt 02 measured that they belong here rather than to the break-point knots or to
  `[02-qcd-T-z-spline-node-tolerance]`. Nothing downstream is limited by it today — the figures sit
  under the QCD Liouville–Green truncation floor — but any tightening of the QCD phase-derivative
  claims meets it first. **Next step:** this is `[00-consumer-anchoring-floor]` seen in the
  derivative, and the principled fix is the same one — per-region anchoring, so that $\varphi$ is
  formed against a local anchor and keeps its dynamic range. A cheaper containment, if that
  campaign is far off, is for `PrimitivePhase` to refuse to contribute $\varphi'$ when the sampled
  $\varphi$ spans fewer than a few ulp of the phase; that is a production change and a policy
  decision, not a repair. Measurements in
  [`logs/02-primitive-phase-break-point-knots.md`](logs/02-primitive-phase-break-point-knots.md)
  §4.

- **[02-verify-script-builds-its-own-Gk-consumer]** *(opened by prompt 02, 2026-09-13; not this
  campaign's file)* — `docs/gktk-remedial/verify_production_path.py` constructs `PrimitivePhase`
  directly at `:557` (the §3.5 $G_k$ consumer) and `:1173` (the §5 throughput section), rather than
  going through `GkSourcePolicyData._build_phase`. Its $T_k$ half does go through the production
  `TkSourceFunctions`. **Impact:** any change to what the $G_k$ production call site *passes* to
  `PrimitivePhase` is invisible to six of §3.5's twelve rows and to both `theta_deriv` $G_k$
  columns, so half of prompt 02's own acceptance table could not have moved however the prompt was
  executed. Prompt 02 therefore took its §3.5/§3.6 measurements on a harness reproducing that
  script's geometry exactly and varying only the knot vector. **Next step:** a few lines in the
  script for whoever next has `docs/gktk-remedial/` in scope — build the $G_k$ phase through
  `_build_phase` with a stand-in `GkSource`, or at minimum forward whatever the production call
  site forwards. Same class as `[13-scoped-run-driver-k-grid-literal]`: a verification driver that
  has drifted from the thing it verifies.

**Not opened, but seen:** `test_tk_wkb_phase.TestCost.test_wall_time_per_object` fails on this
machine at `0c61799` with no `.py` file changed — 0.0638–0.0979 s against its 0.06 s limit, four
runs of four. That is `[07-tk-per-object-cost-is-all-setup]` on the `GkTk-remedial` board, already
open, now straddling the test's threshold as well as prompt 07's. Log 02, observation 5.

The defect this campaign exists to fix — `[13-consumer-spline-crosses-eos-break-points]`,
prompt 02's — **remains open** on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3, which owns its measurements
and its history and carries the `**Assigned (2026-09-13):**` line naming this campaign
(README §0.2). Prompt 02 narrowed it and re-attributed it rather than closing it; the entry there
now says so. Issues **opened** by a prompt here are recorded in this section, with a row in
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit.

---

## 4. Resolved issues

- **`[13-wkb-mod-2pi-cycle-count-inconsistent]`** — closed by **prompt 01**, 2026-09-13. The
  closure itself is recorded on the
  [`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §4, which opened it and owns
  its history. What prompt 01 measured, `docs/gktk-remedial/verify_production_path.py` unedited,
  before → after:

  | measurement | before | after | target |
  |---|---|---|---|
  | `div*2π + mod == θ`, production $G_k$, LambdaCDM $k=3\times10^8$ | 1 of 77,975 | **0 of 77,975** | 0 |
  | the other eleven (model, sector, $k$) rows | 0 | 0 | 0 |
  | uniform control, $\|\theta\|\sim4\times10^{12}$ | 25 of 400,000 | **0 of 400,000** | 0 |
  | stored `theta_mod_2pi`, every model and $k$ | — | **bit-identical** | bit-identical |
  | consumer phase, LambdaCDM $G_k$ $k=3\times10^8$ | 6.1748 rad (12,646 ulp) at $z=33{,}296$ | **0.0000e+00 rad (0.00 ulp)** | the floor |
  | consumer phase, LambdaCDM, all other rows | 1.00 ulp | 1.00 ulp, unchanged | unchanged |
  | cost per reduction (best of 5 over $10^6$) | 0.2254 µs | 0.2731 µs (+48 ns) | recorded, no threshold |

  The fix is `int(round((fabs(theta) - fabs(theta_mod_2pi)) / TWO_PI))` in place of
  `int(floor(fabs(theta) / TWO_PI))`, in both `WKB_mod_2pi` and `simple_mod_2pi`, each keeping its
  own remainder convention. **Datastore:** `theta_div_2pi` is in no lookup key, so a pre-01
  datastore is served silently with the old count at the affected samples; no migration was
  invented and none is recommended (README §7 D3). The regeneration list is in the log's "State
  handed to the next prompt"; `docs/gktk-remedial-verification.md` §8 is **not yet written** —
  prompt 01 was the first to need it and, per its §5 item 2, left the document alone for the
  close-out.

---

## 5. Standing notes for implementers

1. **The remainder is exact; only the cycle count may move.** Every stored `theta_mod_2pi`, and so
   every stored $G$ and $T$, must be bit-identical across prompt 01 (README §2 (b)).
2. **A cosmology that declares no break points must be bit-identical across prompt 02.** LambdaCDM,
   `RadiationModel` and every stand-in take the unchanged code path (README §2 (e)). **Moot as of
   2026-09-13**: prompt 02 stopped without changing production code, so every model is
   bit-identical, LambdaCDM and QCD alike.
3. **Splitting at declared break points is not chunking.** README §2 (d). Prompt 02's log must say
   which of the two it did, and prove the absence of the switch discontinuity that killed chunking.
   **As of 2026-09-13 it did neither**, and said so: neither a repeated-knot vector nor per-segment
   splines shipped, so nothing was split and nothing was chunked.
4. **The floor is $\varepsilon k\tau$** — 3.05e-7 rad at $k=10^5$/Mpc to 9.15e-4 rad at
   $3\times10^8$. Ten of the twelve production consumer cases are already there. A test that
   asserts below that floor is asserting agreement between two errors
   (`GkTk-remedial` board §5 note 2).
5. **`theta_div_2pi` is in no lookup key.** Prompt 01 moved it at a handful of samples and a
   pre-01 datastore is served silently with the old value. README §7 D3; prompt 01 §5. **Live as
   of 2026-09-13**: prompt 01 landed and no louder mechanism was added, on the prompt's own
   instruction; the regeneration list is in `logs/01-wkb-mod-2pi-cycle-count.md`.
6. **A knot vector cannot resolve a break the sample grid does not resolve.** Prompt 02's central
   measurement. `BREAK_POINT_ALL` on `QCD_Cosmology` is one break point per 4.5 production samples
   (226–325 inside a 1,016–1,401 sample range), **none** of which coincides with a sample; a
   multiplicity-`spline_order` knot there is singular on all six production grids, and where it
   does construct it must steal the knots of the samples adjacent to the break, coarsening the
   spline exactly where $\varphi$ is least smooth. Any future attempt has to change the *grid* or
   the *representation*, not the knots. The quadrature (prompts 02, 03) and the ODE (prompts 18,
   19) escaped this because they **choose their own abscissae**; an interpolating spline does not.
