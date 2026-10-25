# Prompt 04 — does the fixture's superseded $G_k$ phase change A2's answer?

**Campaign:** [`README.md`](README.md) · **Board item:** **A4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md), which carries the A4 row and item
since the refresh. Campaign README §3 carries A4 too; do not add it there again.
**Written:** 2026-09-21. **Refreshed 2026-10-25**, never having run: moved to a worktree at the
pre-B1 tree, made the only planned answer to
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` (§1.1), and given the measurements that
issue's narrowing asks for. **The threshold for "material" (§3 item 3) is unchanged, word for
word.**
**Where it runs:** on the laptop, in a git worktree of its own, cut at **`98e68d3`**
([`orchestrator/prompt-04.md`](orchestrator/prompt-04.md)). Not at `HEAD`, for two reasons, either
of which is enough:
- **It cannot run at `HEAD`.** It imports `docs/handover/realistic_large_x.py`, whose `run_cell`
  reads `clamp_gaps_log1pz` and `max_clamp_gap_log1pz` (`:332`, `:356`). B1 (`049c7c8`) removed both
  keys, so at `HEAD` every cell raises `KeyError`
  (`[12-realistic-large-x-reads-the-retired-clamp-records]`). B1 also changed
  `ComputeTargets/tests/test_quadsource_integral.py`, whose `Case` and `OffsetBesselPhaseGk` the
  harness builds on.
- **It is the right tree.** A4 judges A2's numbers, which were taken before B1, with the clamp. A
  corrected $G$ phase must be compared against them on the code that produced them, so that the
  $G$ phase is the only thing that moves.

**Uses:** no Ray, no datastore, no lock. One registered, detached run (§2.2).
**Closes:** nothing. **Decides** whether
`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]` and
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` are defects in
[`REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md)'s numbers or only in its
fixture's documentation.
**Recommended model:** **Opus**. The work is small; the judgement about what counts as "materially
different" is not, which is why the prompt fixes it.

**Read first.** Every path below is read **in the worktree, at `98e68d3`**, except where it says
`handover-remedial`; read those with `git show handover-remedial:<path>`, because the worktree's
copies are older.

1. [`docs/handover/REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md) — **§4 (the
   attribution), §5 (the $x$-scaling), §7 (cost) and §8 items 5 and 6**. Those are the numbers on
   trial.
2. The `handover` board §3, at `handover-remedial`: both `02-` issues in full, **with all their
   narrowings and the assignment**. They are the hypothesis this prompt tests.
3. `prompts/levin-phase-floor/README.md` §2, at `handover-remedial`: the paragraph "02 was stopped
   and superseded". It is why §1.1 below exists.
4. `ComputeTargets/GkSourcePolicyData.py` — `_build_phase` (`:145-195`) **including its
   docstring**, which quantifies what the superseded construction carries: a cubic spline of
   $\theta$ has $h^4x_{\rm source}/384$, 8.3e-3 rad at $x_{\rm source} = 10^7$ and O(1)–O(10) rad at
   production's $10^9$–$10^{12}$. Then `ComputeTargets/primitive_phase.py`'s module docstring, which
   has the full comparison.
5. `ComputeTargets/tests/test_phase_groups.py` — `BesselPhaseGk` (`:299`), and the claim in its
   docstring (`:301`) that the fixture is built "the way `GkSourcePolicyData._create_functions`
   builds the real one". That claim is what is false. Then
   `ComputeTargets/tests/test_quadsource_integral.py`'s `OffsetBesselPhaseGk` (`:217`), the
   subclass the realistic flavour actually uses (`:445`).
6. `docs/handover/realistic_large_x.py` — **all of it**, in particular `run_cell` (`:303`), the
   checkpoint machinery (`:153-163`, `:375-415`) and how the realistic flavour's $G$ phase is
   constructed.
7. `AdaptiveLevin/levin_quadrature.py`: what `adaptive_levin_sincos` returns — `num_regions`,
   `max_depth` and `num_phase_limited_regions` among it — and how `_aggregate_LevinData` in
   `ComputeTargets/QuadSourceIntegral.py` (`:662-680`) combines it. These are the fields §2.1 adds.
8. For the registered run: `CLAUDE.md`, "Long-running jobs — the run registry", and
   `docs/levin-phase-floor/03/scripts/launch.py` at `handover-remedial`, a launcher of a registered
   detached run from a worktree at this same tree. **The pattern to follow.**
9. Campaign [`README.md`](README.md) §0.4, §2 (b), (l), §5 and **§5.1**.

**It changes no production code and adds no test.** It adds one `docs/` script and one short
document. It edits neither `realistic_large_x.py` nor `REALISTIC-LARGE-X.md`.

**Gate** (check, and save the output in the log):
- the worktree's `HEAD` is `98e68d3` on its own branch, and `git status --short` there is empty;
- `./venv` in the worktree is a symlink to the main checkout's (Python 3.13.16);
- `runregistrykit list`, run from the **main checkout**, shows nothing `running` or stale;
- A2's checkpoint, `/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/realistic_large_x_cells.jsonl`
  in the **main checkout**, has 60 lines and the single script hash `c1cd3598c23a`;
- `sha256sum docs/handover/realistic_large_x.py` in the worktree begins `c1cd3598c23a`, the hash
  A2's cells carry.

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

### 1.1 What has changed since this was written

- **This prompt is now the only planned answer to
  `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`.** On 2026-10-25 that issue was assigned
  to `prompts/levin-phase-floor`, whose prompt 02 was to inject phase noise into A2's fixtures, the
  exact flavour included, and whose prediction N2 asked whether noise alone makes the exact flavour
  grow as the realistic one does. **02 was stopped and superseded by 03 the same day, and N2's
  original form was not tested** (that campaign's README §2; the user judged that it does not
  matter for that campaign). Nothing else on any board will run the one-cell experiment the issue's
  next step names. This prompt runs it.
- **The issue has been narrowed twice since this prompt was written**, and both narrowings change
  what a result here means:
  - **2026-09-23, by the A3 baseline run:** production has a Levin blow-up with a `PrimitivePhase`
    $G$ (up to 1,530,190 regions, `max_depth = 20`), so the strong form of the hypothesis is
    refuted. A null result here no longer means the cost curve is safe, and the experiment should
    record `max_depth` and the region count, not only `Levin_regions`.
  - **2026-10-25, by the A3 v2 findings:** a phase floor $\varepsilon k\eta$ that `phase_limited`
    cannot see is a candidate third explanation, carried by the harness's $T$ half too. The
    experiment should also record which group carries the deepest regions.

  §2.1 adds those measurements. They are recorded, **not** added to the threshold: the verdict is
  still decided by §3 item 3 alone.
- **The venv has moved since A2 ran:** SciPy 1.15.2 → 1.16.3 (`8abb3da`, 2026-10-05) and Python
  3.12 → 3.13.16 (`py313-venv`, `d4e5504`). A2's numbers were taken on the old one. §2's reproduction cells
  measure the drift, so that it cannot be mistaken for the effect of the phase.

---

## 2. What to build

`docs/handover/gk_phase_decision_test.py` — a **sibling** of `realistic_large_x.py`, importing
from it rather than reimplementing or editing it. One command from the worktree's root,
`PYTHONPATH=.`, no Ray, no datastore.

It adds one flavour, `realistic-Gk-current`: the realistic fixture in every respect except that the
Green's function phase is built as a `PrimitivePhase`, matching what `_build_phase` returns in
production. Nothing else about the case may move — same shapes, same $\lambda$ scaling, same
fixtures, same 50-digit reference, same tolerances.

**Do not invalidate A2's checkpoint.** `realistic_large_x.py` gates reuse on a SHA-256 of its own
source, so editing that file discards all 60 cells and costs a three-hour recomputation for
nothing. The worktree adds a second hazard: the module's `CHECKPOINT` is computed from its own
`__file__` (`:153-157`), so in the worktree it names a file that does not exist, and the real one is
in the main checkout. Therefore:
- A2's checkpoint is read **directly, by its absolute path in the main checkout, opened for reading
  only**. Do not set `REALISTIC_LARGE_X_CHECKPOINT`, and do not call any of the module's checkpoint
  functions that write.
- This prompt's own checkpoint is a separate file in its run directory, under the **main
  checkout's** `var/runs/` (a run launched from a worktree is recorded there, `CLAUDE.md`). Every
  record carries the script's SHA-256 and the git SHA, taken at launch.

State in the docstring how you did both, and that **after landing the script is a record of its own
SHA, not a script that runs at `HEAD`**, for the reason in the header.

### The cells

Three families, chosen because they are diagnostic for different things. The `exact` cells are
unaffected by the $G$ phase construction and are already on disk; reuse them, do not recompute.

| family | flavour | cells | what it decides | measured cost in A2 |
|---|---|---|---|---|
| `together` at $x_{\rm resp} = 10^7$ and $10^8$, closed and open | `realistic-Gk-current` | 4 | **the representation term**, where $h^4x/384$ is largest | 774 + 958 + 242 + 1516 s ≈ 58 min |
| `q-smooth` at $x_{\rm resp} = 10^5$, closed and open | `realistic-Gk-current` | 2 | **the cost growth**, on the shape that forced the ceiling | 515 + 490 s ≈ 17 min |
| `q-smooth` at $x_{\rm resp} = 10^5$, closed and open | `realistic`, **unchanged**, re-run here | 2 | **the reproduction**: the venv's drift, and §2.1's fields on the old phase | 515 + 490 s ≈ 17 min |

The reproduction cells are A2's own `realistic` cells, run through `realistic_large_x.run_cell`
unmodified, in the current venv, into **this prompt's** checkpoint. They exist for two reasons
fixed here, before anything runs:
- **The cost verdict's "old" column for `q-smooth` is the reproduction**, with A2's recorded
  `Levin_regions` beside it. That makes the old and new phases differ in nothing but the phase.
- **They test whether A2's numbers are still comparable.** If a reproduction cell's `total`
  differs from A2's by more than that cell's declared error, the representation term's "old"
  column (A2's) is not comparable with a "new" column taken in this venv: stop (§7).

If the `q-smooth` cost **collapses**, add $10^6$ closed and open in `realistic-Gk-current` — the pair
A2 could not reach — and report whether the ceiling lifts. That is the one place you may extend the
cell list, and you must say in the log that you did and why.

### 2.1 What every cell records, beyond `run_cell`'s fields

For every cell this prompt runs, from the Levin driver's own per-call results:
- `Levin_regions`, as `run_cell` sums it, **and the region count per phase group**;
- **`max_depth`**, per group and the maximum over the cell, and **which group carries it**;
- `num_phase_limited_regions` per group, which is what the issue's 2026-09-23 narrowing means by
  recording the regions "against `local_atol` rather than against `phase_err` alone". If the driver
  does not return a count you need, say so. Do not instrument `AdaptiveLevin/`.

These fields are not in A2's checkpoint, which is why the reproduction cells carry them for the old
phase. They are reported beside the verdict, as the issue asks. **They do not enter it.**

### 2.2 How it runs

The six to ten cells take one to two hours. That is a registered run, `CLAUDE.md` rules 1–7:
a launcher in the pattern of Read first item 8, `runregistrykit.begin` before anything runs
(campaign `handover`, prompt `04`), started detached from the worktree so that the registry records
`98e68d3`, stdout and stderr into the run directory, the child's pid recorded. Verify in **one**
check that it started, and end the turn. Resume when it has ended.

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
  Four objects now: A2's realistic flavour as recorded, the same flavour reproduced in this venv,
  this prompt's corrected one, and the shared exact control. Name the tree (`98e68d3`) and say why
  it is not `HEAD`.
- **§1 — the threshold**, stated before the tables, per §3 item 3.
- **§2 — the tables**: representation term old and new per cell with both errors; `Levin_regions`
  and `integral_time` old and new; the reproduction against A2's record.
- **§3 — the verdict**, in one paragraph: material or not, for each of the two issues separately.
  They can come apart — the cost could collapse while the physics stands, or the reverse.
- **§4 — the depth record**: §2.1's fields for every cell run, old phase and new, with the group
  that carries the deepest regions. What they suggest about the issue's other two candidate
  mechanisms is stated as a reading, not a verdict.
- **§5 — what this does not cover.** At minimum: two families, not the full factorial; $b = 0$;
  the $T$ half was already current, so this tests only the $G$ half; the tree is pre-B1, so the
  seam is the clamp, not B1's overlap; and a null result at $x \le 1.6\times10^8$ does not extend
  to production's $4\times10^{12}$, where `_build_phase`'s own docstring puts the superseded term at
  O(1)–O(10) rad.

---

## 5. What this prompt does not do

- It does not edit `docs/handover/realistic_large_x.py` or `REALISTIC-LARGE-X.md`. If the verdict is
  "material", correcting A2 is a separate decision and a separate prompt.
- It does not edit `ComputeTargets/tests/test_phase_groups.py`, or fix the false claim at `:301`.
  That belongs to whoever fixes the fixture — README §7 **B2** by the issue's own next step.
- It does not touch production code, `AdaptiveLevin/`, `main.py`, `config/`, or
  `docs/radiation-oracle/`.
- It adds no test. Suite counts must be **unchanged** from the orchestrator's baselines, taken in
  the worktree at `98e68d3`.
- It does not re-run the full factorial under the corrected phase. That is the thing the verdict
  decides, not the thing this prompt does.
- It does not work in the main checkout. It reads A2's checkpoint there and writes its run
  directory there, through the registry, and nothing else.

---

## 6. Acceptance

1. One command launches the run from the worktree's root, no Ray, no datastore; the run is
   registered and ended `done`, with its runtime recorded.
2. A2's checkpoint is intact and was read read-only: its line count and script-hash set before and
   after, and its SHA-256 unchanged.
3. The threshold is stated in the document before the tables.
4. The representation term is recomputed for the compared cells, old and new, each with both
   declared errors.
5. `Levin_regions` and `integral_time` reported old and new for every cell run; the reproduction set
   against A2's record.
6. §2.1's fields reported for every cell run.
7. A verdict given separately for each of the two issues.
8. Suite counts unchanged; `black --check` clean.
9. Board and `docs/OPEN_ISSUES.md` updated in the same commit: the two `02-` issues **narrowed**
   with the measurement, not closed — closing them is for whoever acts on the verdict.

---

## 7. Stop conditions — stop and ask the user

- **The representation term moves materially.** That means A2 §4's attribution was taken on a
  contaminated fixture and the question of re-taking it is the user's. Report the size and the sign
  and stop; do not begin a re-take.
- **A reproduction cell's `total` differs from A2's by more than its declared error.** The venv has
  moved A2's numbers, and the comparison has to be re-planned by the user before it means anything.
- **The `PrimitivePhase` cannot be built offline.**
- **The corrected fixture will not drive at all** — an accessor raises, the phase will not build at
  $x = 1.6\times10^8$. That is a finding about the current construction and is worth reporting as
  one.
- A2's checkpoint is found to have been modified, or its cells no longer load.
- Any Gate item is not as stated.

---

## 8. The log and the board

`logs/04-gk-phase-decision-test.md`, template as the campaign README §5.1. Beyond it: the threshold
and **when it was fixed relative to seeing the numbers**; A2's checkpoint cell count before and
after; the reproduction against A2's record; the cost comparison per cell; and the worktree and the
tree as `STRUCTURALLY REQUIRED` deviations.

**One commit, on the worktree's branch, staged by explicit path**: the script, the document, the
log, the board and `docs/OPEN_ISSUES.md`. No push, no merge, no rebase: landing is the
orchestrator's. **The worktree's board is the one at `98e68d3`**, which has no A4 row; add the row
and the item there in the refreshed form, and the orchestrator resolves the conflict on rebase by
taking yours. Board: the A4 row in §1 and the item in §2, and the **narrowing** notes on both `02-`
issues in §3. `docs/OPEN_ISSUES.md`: no row is added or deleted unless you open something new;
correct the date, and update the two `02-` hooks if the narrowing changes what they say at a glance.
