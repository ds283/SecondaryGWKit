# Implementation state — the phase-representation campaign

**Campaign:** [`README.md`](README.md) · **Source measurements:**
[`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md) §3.5, §3.6, §3.7
**Baseline commit:** `9daa2cb` (`gktk-remedial`, clean — the `GkTk-remedial` close-out)
**Last updated:** 2026-09-13 — **prompt 01 landed.** `WKB_mod_2pi` and `simple_mod_2pi` derive
their cycle count from the exact `fmod` remainder instead of a second rounded division. On the
production geometry, LambdaCDM $G_k$ at $k=3\times10^8$ is **0 inconsistent of 77,975** (was 1),
the $|\theta|\sim4\times10^{12}$ uniform control **0 of 400,000** (was 25), and that case's
consumer phase error **0.0000e+00 rad, 0.00 ulp of the span** (was 6.1748 rad, 12,646 ulp). The
remainder is bit-identical, so no stored $G$ or $T$ moved. Prompt 02 not dispatched.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

| # | Prompt | Closes | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [`WKB_mod_2pi` cycle count](01-wkb-mod-2pi-cycle-count.md) | `[13-wkb-mod-2pi-cycle-count-inconsistent]` | Opus | ✅ | *"Derive the WKB cycle count from the exact remainder"* (SHA not embedded, per the campaign convention) | [`logs/01-wkb-mod-2pi-cycle-count.md`](logs/01-wkb-mod-2pi-cycle-count.md) |
| 02 | [`PrimitivePhase` break-point knots](02-primitive-phase-break-point-knots.md) | `[13-consumer-spline-crosses-eos-break-points]` | Opus | ⬜ | | |

**Progress:** 1 / 2 complete.

---

## 2. Mechanism-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| P1 | **DEFECT, accuracy** | `WKB_mod_2pi`'s cycle count is a rounded division while its remainder is an exact `fmod`, so at large $\|\theta\|$ the stored pair reconstructs $\theta-2\pi$. 1 of 77,975 production $G_k$ samples at $k=3\times10^8$ on LambdaCDM; 6.17 rad of consumer error against a 9.15e-4 rad floor; rate = the half-ulp width of $\|\theta\|/2\pi$, growing linearly with $k$. `simple_mod_2pi` shares the construction. | 01 | ✅ |
| P2 | **DEFECT, accuracy** | `PrimitivePhase` splines $\varphi$ with `make_interp_spline`'s default knots, interpolating across `QCD_EOS`'s declared break points where $\varphi$ kinks. 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$, 428 ulp) at $z=4.24\times10^7$, against 1.00 ulp elsewhere; and `theta_deriv` misses $\omega$ by 2.3e-7–3.3e-4 relative across the QCD interior. | 02 | ⬜ |

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

The remaining defect this campaign exists to fix — `[13-consumer-spline-crosses-eos-break-points]`,
prompt 02's — lives on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3, which owns its measurements
and its history and carries an `**Assigned (2026-09-13):**` line naming this campaign
(README §0.2). Issues **opened** by a prompt here are recorded in this section, with a row in
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
   `RadiationModel` and every stand-in take the unchanged code path (README §2 (e)).
3. **Splitting at declared break points is not chunking.** README §2 (d). Prompt 02's log must say
   which of the two it did, and prove the absence of the switch discontinuity that killed chunking.
4. **The floor is $\varepsilon k\tau$** — 3.05e-7 rad at $k=10^5$/Mpc to 9.15e-4 rad at
   $3\times10^8$. Ten of the twelve production consumer cases are already there. A test that
   asserts below that floor is asserting agreement between two errors
   (`GkTk-remedial` board §5 note 2).
5. **`theta_div_2pi` is in no lookup key.** Prompt 01 moved it at a handful of samples and a
   pre-01 datastore is served silently with the old value. README §7 D3; prompt 01 §5. **Live as
   of 2026-09-13**: prompt 01 landed and no louder mechanism was added, on the prompt's own
   instruction; the regeneration list is in `logs/01-wkb-mod-2pi-cycle-count.md`.
