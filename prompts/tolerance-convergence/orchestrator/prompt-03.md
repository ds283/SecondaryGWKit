# Orchestrator — prompt 03, `GkNumericIntegration` and its floor

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../03-gk-numeric-and-its-floor.md`](../03-gk-numeric-and-its-floor.md)
**Model:** Opus. **Production code changed:** none. **Test code changed:**
`ComputeTargets/tests/convergence_reference.py`, **additive only**, and only if §2.2's bound check
needs it.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This prompt is the first half of a charter that was split.** README §3.3 originally gave prompt 03
three targets; on 2026-09-17 the user split it, so **03 takes `GkNumericIntegration` and the
consumer-spline floor** and **03a takes `TkNumericIntegration` and `wavenumber_exit_time`**. The
reason is that the $G_k$ sector is ~65,000 objects per model and is where **D1** actually turns, so
it gets its own commit and its own review rather than sharing a rollback boundary with a re-take and
a small new measurement. Board items: this prompt is **T4**; **T5** and **T6** are 03a's.

**It recommends and does not decide** (D1 is the user's), so it ends in a **hand-back**.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and both suite counts — **491** (`ComputeTargets`) and **39**
(`CosmologyModels`) at `90998ac`. `ComputeTargets` must not fall below **452**, the campaign floor,
and should be at 491 here.

**Then record prompt 01's acceptance, because this is the first prompt permitted to touch the
facility.** `ComputeTargets/tests/test_convergence_reference.py` is what says prompt 01's published
figures have not moved:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_convergence_reference 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

Record the count. Any movement in it after the agent's commit is a **stop** under §3.5 — the
facility is prompt 01's deliverable and this prompt may only append to it.

**Precondition:** prompts 01, 02 and 02a have landed, **including 02a's documentation correction**
(`90998ac`). That correction matters to this prompt specifically: it is what says
`[01-density-criterion-imposed-outside-the-wkb-region]` is **still unmeasured**, so an agent reading
the board will not think the band question is settled and start reasoning from a census that
measures something else.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `03-gk-numeric-and-its-floor`, model
**Opus**. **The template's standing parameter-freeze sentence applies unchanged** — unlike 02a,
this prompt changes no production code and has no D6 carve-out. Add one sentence to it:

> **You may add to `ComputeTargets/tests/convergence_reference.py` and may not change an existing
> line of it.** Prompt 01's published figures must reproduce bit-identically afterwards, and
> `ComputeTargets/tests/test_convergence_reference.py` is what checks that. If the check your prompt
> §2.2 requires cannot be built additively, stop and say so rather than editing.

Do not tell the agent what the answer is expected to be. The prompt says plainly that `unchanged` is
the likely outcome and is a result; an orchestrator that repeats that in the dispatch has told the
agent what to find, in the one sector where nobody has looked.

## 3. The review — six checks

1. **Is it a matrix, or a diagonal wearing a matrix's clothes?**

   Read the setting list in `docs/tolerance-convergence/gk_numeric_sweep.py`, not the document's
   prose. Every `(atol, rtol)` pair the sweep visited must be there, and the set must contain
   points that move **one** axis with the other held — on both axes — plus the interaction corners
   of §4. **A set whose points all lie on a line through `(1e-10, 1e-8)` is a stop**, because that
   is exactly what review §10.1 did and the reason the prompt exists. Check too that no reported
   step crosses `SCIPY_RTOL_FLOOR = 2.220446049250313e-14`; a step below it measures nothing.

2. **Does every published figure carry its drift, its grid generation and its anchor?**

   Spot-check five rows of `GK-NUMERIC-SWEEP.md` against the script's output. Each needs the
   reference's drift beside it (§5 rule 5), `SOURCE_GRID_V2` named (§5 rule 6), and — since 02a —
   **which anchor**, `PRODUCTION_Z_INIT_LAMBDACDM` or `PRODUCTION_Z_INIT_QCD`, on the QCD rows.
   A QCD figure that does not name its anchor is the defect prompt 02a was inserted to make
   impossible, and it is a stop.

   Check also that the radiation column is scored against the **closed forms** and not by
   self-convergence. Radiation has truth available; a prompt that self-converges there has thrown
   away its only calibration.

3. **Was the floor measured, or inherited with extra steps?**

   The document must carry two numbers: review §10.1's `1e-5`–`1e-4` with its citation, and the
   agent's own, with its uncertainty. Read the **method**, not the number: the spline error must be
   scored **between** the response grid's nodes, because that is where a spline's error lives and
   scoring it at the nodes it was built from returns approximately zero. If the fresh figure sits
   suspiciously close to the inherited one, check the method before believing it.

4. **Was §6.1's target rule applied, in the direction it actually points?**

   Two failure modes, opposite and both stops:
   - the agent recommends **tightening** — then check it recommended the **loosest** setting that
     clears the floor, not the tightest it measured. Rule 3. A setting two decades tighter than the
     one that first clears buys nothing and costs 65,000 objects per model.
   - the agent recommends **`unchanged`** — then check the word is actually in the cell, and that
     the **factor by which the floor dominates** is recorded beside it. Rule 4. `unchanged` without
     the factor is an assertion, not a result.

   Either way the cost must be in **evaluations times objects**, at the setting and one step either
   side, and **never in wall time** (§2 (i)).

5. **Was the $z_{\rm source}$ bound checked, and is the facility diff additive?**

   ```bash
   git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/convergence_reference.py
   ```

   Deletions must be **zero**. Then re-run `test_convergence_reference` and compare with §1's count.
   Separately, the document must state the §2.2 check's result — three wavenumbers per model, at a
   spread of interior source redshifts — **either way**. If the outermost source redshift was not
   the least favourable somewhere and the agent continued anyway, that is a stop: the bound is the
   premise the whole fifty-run sweep rests on.

6. **The suites and the bookkeeping.** `ComputeTargets` **must not fall below 452** and should read
   491 or more; `CosmologyModels` must read **39** — this prompt touches nothing in that package.
   Board row 03, item row **T4**, and `docs/OPEN_ISSUES.md` in the **same commit** with its count
   and date corrected if §3 or §4 moved. `black --check` clean on every `.py` in the diff.

## 4. What a good outcome looks like

- A matrix with both axes moved independently, an `atol` inertness claim that is a **measurement**
  rather than an inference from §2 (e), and a floor taken on this tree with its uncertainty.
- An answer to "is §2 (d)'s prior right in the $G_k$ sector?" in one paragraph with a number in it.
- Most likely: **`unchanged`**, with the factor by which the consumer spline dominates, and a cost
  figure showing what tightening would have bought. **That is the prompt succeeding, not failing.**
- A recommendation for D1 written so the user can take a decision from it rather than a summary of
  the measurement.

**What a good outcome does *not* look like:** a tightened tolerance recommended because the sweep
found the error falls when you tighten. It always does. The question is whether it falls below
something that matters, and §6.1 rule 4 says that where the floor already swamps it, no prompt may
tighten however cheap it looks.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **An accuracy reported below the floor the agent itself just measured.** §2 (f) and §6.1 rule 5 —
  an arithmetic error, never a discovery, and campaign-wide.
- **The reference did not converge** at any $(model, k)$. Prompt 17's error. Not a wavenumber to
  drop and not a criterion to loosen.
- **The recommendation moves the sector's total by more than a factor of two.** §4.3 makes that the
  user's decision, and in this sector a factor of two is ~65,000 objects per model.
- **The outermost $z_{\rm source}$ is not the least favourable.** The fifty-run bound fails and the
  prompt's shape changes; the user should decide before anything is built on it.
- **The agent proposes to touch the band, the grid, `_solve_horizon_exit`, the digest, or any
  parameter.** Each is named out of scope and each will look like the right fix.
- **The agent asks you to choose a target, a floor, or a setting.** Relay verbatim (rule 5); do not
  pick a number, and in particular do not tell it what §2 (e) predicts.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

## 6. After it lands

**Hand back to the user**, because **D1 is theirs** and prompt 05 may not start until they accept.
Report, in this order:

1. **The answer to §6 question 1** — is the prior right in the $G_k$ sector? One line, with the
   figure that decides it and its drift.
2. **The floor**, freshly measured, per model, with its uncertainty and the inherited figure beside
   it; and the factor by which it dominates the solver error at `(1e-10, 1e-8)`.
3. **The recommendation** — a pair, or the word `unchanged` — with the cost at that setting and one
   step either side in evaluations times objects. This is what the user is being asked to accept
   under D1.
4. **Whether the $z_{\rm source}$ bound held**, since prompt 03a, prompt 04 and prompt 06 all
   inherit fifty-run coverage of this sector from it.

Then stop. **Prompt 03a is next and is not written yet**; write it after the user has read this, so
that a `TkNumericIntegration` sweep is written knowing what the $G_k$ sweep found about the axes.
Do not draft it yourself (rule 1) and do not dispatch anything further.
