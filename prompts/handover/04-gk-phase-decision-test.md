# Prompt 04 — does the fixture's superseded $G_k$ phase change A2's answer?

**Campaign:** [`README.md`](README.md) · **Board item:** **A4** ·
**Board:** `IMPLEMENTATION_STATE.md` — exists; add the A4 row. The campaign README §3 table needs
an A4 row too, in workstream A.
**Closes:** nothing. **Decides** whether
`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]` and
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` are defects in
[`REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md)'s numbers or only in its
fixture's documentation.
**Recommended model:** **Opus**. The work is small; the judgement about what counts as "materially
different" is not.

**Read first:**

1. [`docs/handover/REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md) — **§4 (the
   attribution), §5 (the $x$-scaling), §7 (cost) and §8 items 5 and 6**. Those are the numbers on
   trial.
2. [`prompts/handover/IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) §3 — both `02-` issues in
   full. They are the hypothesis this prompt tests.
3. `ComputeTargets/GkSourcePolicyData.py` — `_build_phase` (`:145`) **including its docstring**,
   which quantifies what the superseded construction carries: a cubic spline of $\theta$ has
   $h^4x_{\rm source}/384$, 8.3e-3 rad at $x_{\rm source} = 10^7$ and O(1)–O(10) rad at production's
   $10^9$–$10^{12}$. Then `ComputeTargets/primitive_phase.py`'s module docstring, which has the
   full comparison.
4. `ComputeTargets/tests/test_phase_groups.py` — `BesselPhaseGk` / `OffsetBesselPhaseGk`, and the
   claim at `:299` that the fixture is built "the way `GkSourcePolicyData._create_functions` builds
   the real one". That claim is what is false.
5. `docs/handover/realistic_large_x.py` — **all of it**, in particular `run_cell`, the checkpoint
   machinery and how the realistic flavour's $G$ phase is constructed.
6. Campaign [`README.md`](README.md) §0.4, §2 (b), (l), §5 and **§5.1**.

**It changes no production code and adds no test.** It adds one `docs/` script and one short
document, and it edits neither `realistic_large_x.py` nor `REALISTIC-LARGE-X.md`.

---

## 1. Why this exists

A2 measured a representation term of 4.08e-05 to 2.68e-03 and a clamp term 11.6 to 1034 times
larger, and reported the realistic flavour's Levin cost growing 1.7×/3.6×/8.6× per decade of $x$
where the exact flavour's is flat. **Both results were taken on a fixture whose $G$ half is one
campaign behind production.** `GkSourcePolicyData._build_phase` has returned a `PrimitivePhase`
since `prompts/GkTk-remedial` prompt 09; the fixture still builds a raw `phase_spline`, which
carries the $h^4x/384$ term production has removed. At A2's top rung, $x \approx 1.6\times10^8$,
that term is of order a tenth of a radian.

So two of A2's headline numbers are, as they stand, upper bounds of unknown tightness:

- **the representation term**, which may be inflated by a phase error production does not have;
- **the cost growth**, which may be bisection chasing a representation floor that production does
  not have — and which is what forced `q-smooth` to stop at $x_{\rm resp} = 10^5$.

**This prompt is a decision test, not a re-take.** It answers one question — does correcting the
fixture's $G$ phase move either number materially — at the smallest cost that can answer it. What
follows from the answer is the user's call, not this prompt's.

---

## 2. What to build

`docs/handover/gk_phase_decision_test.py` — a **sibling** of `realistic_large_x.py`, importing
from it rather than reimplementing or editing it. One command from the repository root,
`PYTHONPATH=.`, no Ray, no datastore.

It adds one flavour, `realistic-Gk-current`: the realistic fixture in every respect except that the
Green's function phase is built as a `PrimitivePhase`, matching what `_build_phase` returns in
production. Nothing else about the case may move — same shapes, same $\lambda$ scaling, same
fixtures, same 50-digit reference, same tolerances.

**Do not invalidate A2's checkpoint.** `realistic_large_x.py` gates reuse on a SHA-256 of its own
source, so editing that file discards all 60 cells and costs a three-hour recomputation for
nothing. Use a **separate checkpoint file** under `var/runs/`, and read A2's checkpoint
**read-only** for the comparison baseline. State in the docstring how you did this.

### The cells

Two families, chosen because they are diagnostic for different things. The `exact` cells are
unaffected by the $G$ phase construction and are already on disk; reuse them, do not recompute.

| family | cells | what it decides | measured cost in A2 |
|---|---|---|---|
| `together` at $x_{\rm resp} = 10^7$ and $10^8$, closed and open | 4 | **the representation term**, where $h^4x/384$ is largest | 774 + 958 + 242 + 1516 s ≈ 58 min |
| `q-smooth` at $x_{\rm resp} = 10^5$, closed and open | 2 | **the cost growth**, on the shape that forced the ceiling | 515 + 490 s ≈ 17 min |

If the `q-smooth` cost **collapses**, add $10^6$ closed and open — the pair A2 could not reach —
and report whether the ceiling lifts. That is the one place you may extend the cell list, and you
must say in the log that you did and why.

---

## 3. The four things that will go wrong

1. **"No change" is a result, and it is the one this prompt most likely returns.** A2 already
   measured the representation term as *flat* in $x$, not growing like $h^4x/384$, which is weak
   evidence against contamination. If both numbers hold, say so plainly; do not hunt for a
   difference to justify the run.
2. **The comparison must be like-for-like.** The quantity to compare is the **representation term**
   $N_{\rm realistic} - N_{\rm exact}$ at fixed seam and $x$, not raw $N$. Recompute it from your
   own cells and A2's exact cells, through the same block reference, so the 50-digit reference
   cancels exactly as it does in A2 §4.
3. **The threshold is fixed by this prompt, not by you.** A change in the representation term is
   material if it exceeds the sum of the two cells' declared errors **and** changes the
   clamp-to-representation ratio by more than a factor of two. A cost change is material if
   `Levin_regions` moves by more than 2×. Copy these into the document verbatim before you run
   anything. **You may not adjust them.** If you believe a threshold is wrong, that is a stop: say
   why and ask, before you have the numbers. Choosing the threshold after seeing the result is how a
   decision test becomes a rationalisation, and it is the one failure this prompt cannot recover
   from.
4. **The `PrimitivePhase` needs the background model's conformal-time table.** `_build_phase` takes
   the leading term from a double-double $\tau$ table plus a spline of the residual $\phi$. Building
   that in a fixture is the only real work in this prompt. If it cannot be built offline — no Ray,
   no datastore — **stop and ask**; do not substitute an approximation and call it current.

---

## 4. What to write down

`docs/handover/GK-PHASE-DECISION-TEST.md`, short — this is a decision memo, not a survey:

- **§0** — what is being compared and whose each object is, in `KOHRI-TERADA-ORACLE.md` §0's shape.
  Three objects now: A2's realistic flavour, this prompt's corrected one, and the shared exact
  control.
- **§1 — the threshold**, stated before the tables, per §3 item 3.
- **§2 — the tables**: representation term old and new per cell with both errors; `Levin_regions`
  and `integral_time` old and new.
- **§3 — the verdict**, in one paragraph: material or not, for each of the two issues separately.
  They can come apart — the cost could collapse while the physics stands, or the reverse.
- **§4 — what this does not cover.** At minimum: two families, not the full factorial; $b = 0$;
  the $T$ half was already current so this tests only the $G$ half; and a null result at
  $x \le 1.6\times10^8$ does not extend to production's $4\times10^{12}$, where `_build_phase`'s
  own docstring puts the superseded term at O(1)–O(10) rad.

---

## 5. What this prompt does not do

- It does not edit `docs/handover/realistic_large_x.py` or `REALISTIC-LARGE-X.md`. If the verdict is
  "material", correcting A2 is a separate decision and a separate prompt.
- It does not edit `ComputeTargets/tests/test_phase_groups.py`, or fix the false claim at `:299`.
  That belongs to whoever fixes the fixture — README §7 **B2** by the issue's own next step.
- It does not touch production code, `main.py`, `config/`, or `docs/radiation-oracle/`.
- It adds no test. Suite counts must be **unchanged**: `ComputeTargets` 552, `CosmologyModels` 39,
  `LiouvilleGreen` 148 (skipped=1).
- It does not re-run the full factorial under the corrected phase. That is the thing the verdict
  decides, not the thing this prompt does.

---

## 6. Acceptance

1. One command, from the repository root, no Ray, no datastore, runtime recorded.
2. A2's checkpoint is intact and was read read-only: state its cell count before and after.
3. The threshold is stated in the document before the tables.
4. The representation term is recomputed for the compared cells, old and new, each with both
   declared errors.
5. `Levin_regions` and `integral_time` reported old and new for every cell run.
6. A verdict given separately for each of the two issues.
7. Suite counts unchanged; `black --check` clean.
8. Board and `docs/OPEN_ISSUES.md` updated in the same commit: the two `02-` issues **narrowed**
   with the measurement, not closed — closing them is for whoever acts on the verdict.

---

## 7. Stop conditions — stop and ask the user

- **The representation term moves materially.** That means A2 §4's attribution was taken on a
  contaminated fixture and the question of re-taking it is the user's. Report the size and the sign
  and stop; do not begin a re-take.
- **The `PrimitivePhase` cannot be built offline.**
- **The corrected fixture will not drive at all** — an accessor raises, the phase will not build at
  $x = 1.6\times10^8$. That is a finding about the current construction and is worth reporting as
  one.
- A2's checkpoint is found to have been modified, or its cells no longer load.

---

## 8. The log and the board

`logs/04-gk-phase-decision-test.md`, template as the campaign README §5.1. Beyond it: the threshold
and **when it was fixed relative to seeing the numbers**; A2's checkpoint cell count before and
after; and the cost comparison per cell.

Board: the A4 row in §1 and the item table, an A4 row in campaign README §3's workstream A table,
and the **narrowing** notes on both `02-` issues in §3. `docs/OPEN_ISSUES.md`: no row is added or
deleted unless you open something new; correct the date, and update the two `02-` hooks if the
narrowing changes what they say at a glance.
