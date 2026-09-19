# Prompt 02 — the realistic-flavour large-$x$ harness

**Campaign:** [`README.md`](README.md) · **Board item:** **A2** ·
**Board:** `IMPLEMENTATION_STATE.md` — **created by prompt 01**; if 01 has not landed, see orchestrator `prompt-02.md` §1.3
**Closes:** nothing on its own. **It is the instrument B2, D and E are scored on**, and it takes
the measurement `[12-handover-clamp-error-in-production]` records as impossible.
**Recommended model:** **Opus**. The work is a measurement, not an algorithm, and its whole value
is in whether the attribution is honest.

**Read first:**

1. [`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
   **§8 in full** — the harness you are extending, its five findings, and its closing "What this
   does not cover", which is this prompt's subject.
2. [`docs/radiation-oracle/large_x.py`](../../docs/radiation-oracle/large_x.py) — the script itself.
   ~233 lines; read all of it.
3. `ComputeTargets/tests/test_quadsource_integral.py` — the **module docstring** (the exact vs
   realistic flavour distinction, which is the pivot of this prompt), `class Case`, `class Fixture`
   and `TestHandOverClamp`.
4. Campaign [`README.md`](README.md) §0.4, §2 (a)–(d), (j), (k), (l), §5.
5. `prompts/source-remediation/IMPLEMENTATION_STATE.md` §3 — `[08-handover-clamp-error]` and
   `[12-handover-clamp-error-in-production]` in full. Those are the numbers you are trying to
   reproduce and separate.

**It changes no production code.** It adds a `docs/` script and a document.

---

## 1. Why this exists, stated precisely

KT §8 ran `evaluate_QuadSource_integral` at $x_{\rm resp}$ to $10^8$ against an oracle that does not
move, and found $N = -9/8$ throughout. **It ran the `exact` flavour** — `ExactTk`/`ExactGk`
stand-ins injected through the `Tk_functions_builder` hook — and says so in its closing paragraph:
*"the realistic flavour's representation floors (splined phases and amplitudes) are not exercised at
large $x$."*

In that flavour the seam is continuous **by construction**: one analytic function on both sides of a
nominal `crossover_z`, and **no gap**. Production's gap is structural (README §2 (a)) and has to be
manufactured deliberately — `Fixture(drop_first_WKB_sample=True)`, which is what
`TestHandOverClamp` does at $x_{\rm resp} = 100$ only.

So there are two terms nobody has separated:

- the **clamp** — holding the LG accessors constant across up to 1.5 grid steps;
- the **phase re-spline** — $\delta\theta \simeq h^4x/384$, growing linearly in $x$.

`[12-handover-clamp-error-in-production]`'s recorded next step says separating them *"cannot be done
by measurement alone at these $x$"*. **It can now**, because the reference no longer moves and
because the cost of the integral does not grow with $x$ (KT §8 item 1: 0.2–0.7 s per triple from
$x = 1.4\times10^2$ to $1.6\times10^8$).

---

## 2. What to build

`docs/handover/realistic_large_x.py` — a sibling of `large_x.py`, not an edit of it. One command,
from the repository root, `PYTHONPATH=.`, no Ray, no datastore. It prints markdown tables to stdout,
as `large_x.py` and `kt_verification.py` do.

The design is a **2×2×(shape × $x_{\rm resp}$) factorial**, scored throughout against eq. (22) and
the head at 50 digits (`eq22_rounding.py`'s machinery), so KT §7.2's small-$u$ rounding never enters:

| axis | levels |
|---|---|
| flavour | `exact` (the KT §8 control) · `realistic` |
| seam | gap closed (`Fixture` as built) · gap open (`drop_first_WKB_sample=True`) |
| shape | `together`, `T-first`, `q-smooth` |
| $x_{\rm resp}$ | the KT §8 ladder, $980$ to $10^7$–$10^8$ as each shape's fixtures reach |

The `exact` × gap-closed cell **must reproduce KT §8 Table 8.1 to the digits printed**. That is the
harness's own control and the first thing to check.

Report per cell: $N + 9/8$; the pipeline's declared error; $|$Levin$|/|$total$|$; the recorded
`clamp_gaps_log1pz` and `max_clamp_gap_log1pz`; wall time. Plus, per row, the quantity the whole
prompt is for:

$$\text{clamp term} \;=\; N_{\rm gap\ open} - N_{\rm gap\ closed}\quad\text{at fixed flavour and }x,$$
$$\text{representation term} \;=\; N_{\rm realistic} - N_{\rm exact}\quad\text{at fixed seam and }x.$$

**These are the deliverable.** Everything else is scaffolding for them.

---

## 3. The four things that will go wrong

1. **The fixtures' Liouville–Green region has to reach the response time.** `large_x.py` seeds
   `Case._fixtures` with `Fixture(w, k=kk, x_max=max(1e3, 1.05 * x_resp * kk / shape.r))` for
   exactly this reason, and KT §8 item 5 measures the set-up cost as flat because the phase is
   range-reduced by `fmod`. The realistic flavour builds a **`phase_spline` through those samples**
   rather than passing closed forms, so its set-up cost is **not** guaranteed flat. Measure it;
   report it; if it becomes the binding cost, say so.
2. **`drop_first_WKB_sample` opens a gap of one grid step at the fixture's own grid density**, and
   `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` bounds what the integral will accept. Production's measured
   gaps are median 1.2e-02, max 2.2e-02 in $\log(1+z)$, i.e. about one mean source-grid step. Check
   that the gap your fixture opens is comparable, **in units of the fixture's own grid step**, and
   say so in the document — otherwise the clamp term you measure is not production's.
   `test_gap_beyond_tolerance_raises` shows what five steps does: it raises.
3. **The $q$-smooth shape at $u = 0.01$ is where eq. (22) rounds.** KT §7.2: use the 50-digit path,
   not `kt_verification.I_RD`. `ComputeTargets/tests/kohri_terada.py` is `fsum`-ed and better but
   still floors at ~1e-10.
4. **`T-first` is not a closable triangle** ($|q-r| = 2000 > k = 1000$), and eq. (25) drops a term
   there (KT §8 item 4). You are scoring against eq. (22), which is unaffected — but do not import
   an eq. (25) comparison without that caveat.

---

## 4. What to write down

`docs/handover/REALISTIC-LARGE-X.md`, in the shape of `KOHRI-TERADA-ORACLE.md` §8:

- **§0** — what is being compared and whose each object is. KT §0 is the model and is not optional.
- the tables, generated by the script, with a `<!-- generated by ... -->` marker as
  `KOHRI-TERADA-ORACLE.md` uses;
- **the attribution**: the clamp term and the representation term, per shape, per $x$, each with the
  reference's own error beside it (README §5 rule 8);
- **the $x$-scaling of each**. The prediction to test is that the clamp term is roughly
  $x$-independent (it is a held phase over a fixed $\Delta\log(1+z)$, so its effect on the integrand
  is set by the phase advanced over the gap, which grows with $x$ — state what you actually find,
  not what this sentence guesses) while the representation term grows like $h^4x/384$. **If the
  measurement disagrees with either guess, the measurement wins and the guess goes in the
  document.**
- **what this still does not cover** — the honest closing section KT §8 has. At minimum: $b = 0$
  only; the fixtures' grid is not the production grid; and production's $x \approx 4\times10^{12}$
  is still beyond reach.

---

## 5. What this prompt does not do

- It does not touch any production file. Not `main.py`, not `QuadSourceIntegral.py`, not
  `TkWKBIntegration`, not `config/`.
- It does not edit `ComputeTargets/tests/test_quadsource_integral.py` — **imported, never edited**.
- It does not edit `docs/radiation-oracle/large_x.py` or `KOHRI-TERADA-ORACLE.md`. That campaign is
  closed and its documents are its record. If §2's control cell **fails** to reproduce Table 8.1,
  that is a §3 issue and a stop condition, not a licence to edit either.
- **It does not fix the gap.** That is B1. This prompt measures the gap that is there.
- It does not add a test to `ComputeTargets/tests/`. This is a `docs/` measurement script, like
  `kt_verification.py`, `large_x.py` and `grid_density_criterion.py`. Suite counts must be
  **unchanged**.

---

## 6. Acceptance

1. One command, from the repository root, no Ray, no datastore, runtime recorded.
2. The `exact` × gap-closed control reproduces KT §8 Table 8.1 to the digits printed.
3. The clamp term and the representation term are separated, per shape and per $x$, each with the
   reference's own error.
4. `docs/handover/REALISTIC-LARGE-X.md` exists, has a §0 in KT §0's shape, and a closing "what this
   does not cover".
5. `ComputeTargets` and `CosmologyModels` **unchanged** from the baselines in the orchestrator
   prompt. This prompt adds no test.
6. `black --check` clean on the new script.
7. Board and `docs/OPEN_ISSUES.md` updated in the same commit.

---

## 7. Stop conditions — stop and ask the user

- **The control cell does not reproduce KT §8 Table 8.1.** Something moved under that campaign's
  close-out. Report which rows and by how much; do not re-baseline.
- The realistic flavour **cannot be driven at large $x$** — set-up cost explodes, the phase spline
  will not build, or an accessor raises. That is itself a finding about the representation and is
  worth reporting as one; do not work around it by falling back to the exact flavour.
- The two terms **do not separate** — the 2×2 is not additive, the cross term being comparable to
  the main effects. Report the interaction; it would mean the clamp and the representation are not
  independent, which changes what B1 can be expected to buy.
- The gap the fixture opens is **not comparable to production's** in grid-step units and cannot be
  made so.

---

## 8. The log

`logs/02-realistic-flavour-large-x-harness.md`, template as `qcd-background-audit` README §5.1.
Beyond it: the control-cell comparison against Table 8.1 row by row; the measured set-up cost of the
realistic fixtures against the exact ones; and the gap size in units of the fixture's grid step
against production's median and maximum.

Board: this prompt's row, and a §3 issue for anything it finds and does not fix. Nothing on
`docs/OPEN_ISSUES.md` closes here — but if the attribution narrows
`[08-handover-clamp-error]`, `[12-handover-clamp-error-in-production]` or
`[12-phase-spline-error-grows-with-x]`, add the narrowing note to **their own boards**
(`source-remediation`, `GkTk-remedial`) and say so in the index, as other campaigns have done.
