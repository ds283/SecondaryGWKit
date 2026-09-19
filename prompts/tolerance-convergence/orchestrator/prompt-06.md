# Orchestrator — prompt 06, the read-only measurement

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../06-quadsource-integral-read-only.md`](../06-quadsource-integral-read-only.md)
**Model:** Opus. **Production code changed: no.** Documents, one new sweep script, and
`docs/OPEN_ISSUES.md`. `config/defaults.py` must be **byte-identical**, and **no file under
README §0.4** — `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/` — may
appear in the diff at all.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**The dispatch template is used exactly as written.** Unlike 02a, 05 and 05a, this prompt needs no
replacement sentence: §5 rule 8's parameter freeze names 06 among the prompts inside it, and that is
correct. **If you find yourself editing the template for this prompt, you have misread something.**

**Why it runs now.** 05a (`a9ad6fc`) landed the last production change and every other prompt in the
campaign has reported. §7 **D11** (2026-09-18) split §3.6 into 06 and 06a: 06 measures
`QuadSourceIntegral` read-only and regenerates the stale inventory, board item **T11**; 06a assembles
the close-out document and `docs/TOLERANCE-PROVENANCE.md`, **T12**. The split is D7's and D9's, for
D9's reason — §5 rule 1 makes the commit the rollback boundary and the campaign's named deliverable
should not be reverted by a failure in a sweep.

**Precondition:** 05a has landed. **Do not dispatch 06 if `ComputeTargets` is below 521, or if
`config/defaults.py` is not at blob `76bab78…` as 05a left it** — this prompt changes no number and
a difference before it starts means something else moved.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the baselines. At `a9ad6fc` they are **521**
(`ComputeTargets`, **with the known `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
wall-clock flake** — see §5) and **39** (`CosmologyModels`). `ComputeTargets` must not fall below
**521**; it may rise only if §7 of the prompt justified an addition to the harness, and if it rises
for any other reason that is a question, not a bonus.

**Record `config/defaults.py`, which must not move at all** — this is the first prompt since 05a and
the freeze is back on:

```bash
git rev-parse HEAD:config/defaults.py
```

**Record the inventory's staleness, because the prompt's §4 turns on it.** It should report
**OUT OF DATE** before the dispatch and pass after:

```bash
PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check; echo "exit=$?"
```

**And the digests**, which this prompt has no business moving:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_source_grid 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `06-quadsource-integral-read-only`, model
**Opus**, **unmodified** — including the standing parameter-freeze sentence, which applies.

Add these two sentences, and no more:

> **You may not edit `ComputeTargets/tests/test_quadsource_integral.py`.** You import `Case`,
> `SHAPES`, `X_RESP_VALUES` and `B_VALUES` from it and build your sweep around them. If what you
> need is local to a test method rather than to the module, say so and work around it in your own
> script rather than refactoring theirs.

> **The measurement is offline and that is a decision, not a limitation** (the user, 2026-09-18).
> Do not attempt a datastore or Ray run, and do not treat the absence of one as a gap to apologise
> for: the live-run statistics this prompt needs are already in the record and your prompt tells you
> to cite them.

**Do not tell the agent what the answer is.** §1 of the prompt says the record suggests the floor
will dominate, and says so because §6.1 rule 1 requires the floor to be measured *first* — not so
that the agent can arrive at `unchanged` and work backwards. Check 3 turns on whether the two
flavours were used to separate the two quantities, not on which way the ratio went.

**Do not hint at the inventory's corrections.** §4 of the prompt names three judgement columns that
are now wrong and tells the agent to find the rest. An orchestrator who enumerates them has removed
the reading.

**Do not pre-empt the hand-off.** §5 asks the agent to determine what happened to
`[12-atol-too-loose-for-the-source-integral]` from `config/defaults.py`'s own comment. It is not your
job to tell it that the constant moved from `1e-25` to `1e-32`.

## 3. The review — seven checks

1. **Nothing under §0.4 moved, and no production code moved.**

   ```bash
   git diff --name-only HEAD~1 HEAD
   git diff --stat HEAD~1 HEAD -- ComputeTargets/QuadSourceIntegral.py ComputeTargets/QuadSource.py ComputeTargets/phase_groups.py AdaptiveLevin/ ComputeTargets/tests/test_quadsource_integral.py config/defaults.py main.py Datastore/
   ```

   The second must be **empty**; a single line in it is a **stop**, including a comment or a
   docstring. The first must contain nothing outside `docs/`, `prompts/tolerance-convergence/` and —
   only if the log justified it under prompt §7 — `ComputeTargets/tests/convergence_reference.py`
   and `test_convergence_reference.py`. Confirm the `config/defaults.py` blob is unchanged from §1.

2. **The sweep is a sweep, not a diagonal, and it reaches the production pair.** Read
   `QUADSOURCE-READONLY.md`. Both axes, production `(1e-32, 1e-8)` interior to the range with at
   least two steps either side on each. **README §2 (d) is the record of what a diagonal costs** —
   it is why prompt 03 exists — and a diagonal here would repeat it in the one sector the campaign
   does not own. Then run the script yourself:

   ```bash
   PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/quadsource_readonly.py 2>/dev/null | head -60
   ```

   It must run from the repository root with no arguments and no datastore, and its tables must be
   the document's.

3. **The two flavours were used to separate the two quantities.** This is the check the prompt
   exists for. The quadrature error is measured on the **exact** flavour, where the ingredients are
   truth; the floor is measured on the **realistic** flavour, which carries the representation error.
   An agent that swept tolerances on the realistic flavour alone has measured the two convolved and
   cannot report either — the ratio it quotes would be a ratio of one number to itself plus noise.
   Check the document says which flavour every figure came from, and that each carries its
   reference's own error (§5 rule 5).

4. **§6.1's rule was applied, and `unchanged` is an acceptable outcome stated in that word.** Rule 4
   requires the factor by which the floor dominates beside it. A recommendation to change
   `quad_atol` or `quad_rtol` is a **stop** under §0.4 — the prompt may report what the measurement
   implies and hand it over, and the difference is §6 of the prompt.

5. **The `rtol` question was answered on this tree.** `source-remediation` log 12 found
   `1e-8 → 1e-11` bit-identical on 159 items *because `atol` bound at `1e-25`*; `atol` is now
   `1e-32`. The log must say whether that still holds at the current constant and must not inherit
   it. An agent that quotes the 159-item result as though it settled the question has made the
   error §3 of the prompt names.

6. **The inventory is regenerated and its judgement columns corrected.**

   ```bash
   PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check; echo "exit=$?"
   ```

   Must pass. Then read the diff of `TOLERANCE-INVENTORY.md` **outside** the generated markers: the
   shared pair must no longer be described as keying seven and eight object types, the Gauss orders
   must no longer be described as keying none, and the six constants 05a shipped must be present with
   owners. And §5 rule 7 is an acceptance condition — a paragraph of §§1–4 **rewritten** rather than
   added to is a stop, however stale it was:

   ```bash
   git diff HEAD~1 HEAD -- docs/tolerance-convergence/TOLERANCE-INVENTORY.md | grep "^-" | grep -v "^---" | head -40
   ```

   Deletions should be confined to the generated block. Deletions in §§1–4 need the log to justify
   each one.

7. **The hand-off, the suites and the bookkeeping.** `docs/OPEN_ISSUES.md` §1.2/§1.3 carry the
   report; `[12-atol-too-loose-for-the-source-integral]` is **narrowed and still open** — closing it
   is a stop, because whether 58 % of work items still short-circuit is a live-run question and this
   prompt has no live run. Anything new is filed to the owning board, not to this campaign's. Then
   the digests, both suites — `ComputeTargets` **must not fall below 521**, `CosmologyModels` **39** —
   board row 06, item **T11**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and date corrected, all
   in the **same commit**. `black --check` clean.

## 4. What a good outcome looks like

- A measurement that says what the source integral's accuracy is actually limited by, with the
  quadrature error and the representation floor as **two separately measured numbers** rather than
  one number and an assertion — which is what the exact/realistic pair is for and what nothing else
  in the tree could have given.
- `unchanged` in §6.1 rule 4's own word, with the dominating factor beside it, if that is what the
  ratio gives — the campaign's third such finding after prompt 03's ×631–×37,700 and prompt 04's
  ×48.4–×3.03e5, and no less a result for being the expected one.
- The `rtol` question re-answered at `atol = 1e-32` instead of inherited from a measurement taken
  seven decades looser.
- An inventory that describes the tree prompts 05, 05a and 05b left, so that 06a can assemble the
  provenance note from it rather than re-reading the factories.
- A hand-off that two other campaigns can act on without reading this campaign's logs.

**What a good outcome does *not* look like:** a green `--check`, a tidy document, and a tolerance
sweep taken on the realistic flavour. Every number in it would be the representation floor with a
tolerance label on it, and the conclusion would be right by accident.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **The measurement says the quadrature tolerance is the limiting parameter.** That puts §7 **D4**
  back to the user, and it is theirs and not yours: §0.4 put `QuadSourceIntegral` out of scope on
  the premise that its tolerances were already chosen by measurement. Relay the numbers verbatim,
  and do not let the agent recommend a value.
- **The agent needs to edit a §0.4 file, or `test_quadsource_integral.py`.** Relay what it needed
  and why; that is a scope question, and this campaign's own orchestrator README records what
  happens when a scope stop is relayed as though it were settled on the merits.
- **The agent proposes to retune `quad_atol` or `quad_rtol`**, or to close
  `[12-atol-too-loose-for-the-source-integral]`.
- **`inventory.py --write` changes something no prompt in this campaign moved** — a lookup predicate,
  a keyed object type, or an unexpected constant. That is a report about the tree.
- **The agent says a parameter's provenance cannot be assembled from the logs.** That is prompt 06a's
  premise and the user should hear it now rather than at 06a.
- **`ComputeTargets` falls at all**, or rises for a reason prompt §7 did not authorise.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

**The known flake is not a stop.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
asserts a wall-clock figure against a 0.06 s limit and sits on it — observed at 0.0601–0.0612 across
consecutive runs at an unchanged commit, passing roughly one run in three. Confirm by re-running that
module alone before attributing anything to the commit.

## 6. After it lands

**Hand back to the user.** This is a stopping point in README §4.1's sense only if §9's first
condition fired — otherwise it is a hand-back, because **prompt 06a is written against what 06
leaves** and the user should see the measurement before the provenance note is chartered. Report:

1. The quadrature error and the representation floor as two numbers, with the flavour and the case
   each was taken at, and the ratio.
2. §6.1's outcome in the rule's own words, and the dominating factor.
3. Whether `rtol` binds at `atol = 1e-32`, and what that does to log 12's 159-item result.
4. What the inventory regeneration changed, and which judgement columns were wrong.
5. What went to `docs/OPEN_ISSUES.md` §1.2/§1.3, and what happened to
   `[12-atol-too-loose-for-the-source-integral]`.
6. The agent's list of parameters whose provenance **cannot** be established from the record — 06a's
   working set, and the one part of its job that cannot be assembled.
7. Which issues closed, opened and stayed open, and the counts.

> **Superseded 2026-09-18:** 06a **is** now written, at [`prompt-06a.md`](prompt-06a.md), against
> what 06 actually left. Three of the forecasts below turned out stale and the prompt corrects
> them: eleven logs not ten, five campaign documents not four, and §5.4's **44 rows** rather than
> "39 plus the six constants 05a added" — 06's regeneration put those six among the 44.

Then **prompt 06a is next and is not written yet**, deliberately and on the same principle that held
06 back since 2026-09-16: it is written against what 06 actually leaves, not against the plan's
forecast of it. Its charter is campaign README §3.6a and §1.2, its board item is **T12**, and it
assembles `docs/TOLERANCE-CONVERGENCE.md` and `docs/TOLERANCE-PROVENANCE.md` from the campaign's ten
logs, the four campaign documents and the inventory's §5.4 — whose **39 rows are the coverage
checklist**, plus the six constants prompt 05a added. **Do not dispatch it in the same turn.** Note
that §1.2's closing rule is the one acceptance condition 06a cannot satisfy by diligence: where the
provenance of a constant cannot be established from the record, the note says so **in those words**,
and an invented justification is worse than the admission.
