# Orchestrator prompts — the tolerance and convergence campaign

One orchestrator prompt per campaign prompt; there are no workstreams here (README §4.3). Each
dispatches one fresh-context subagent, reviews its commit against fixed criteria, and either
continues or stops and reports to the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 01 | The convergence harness and one production grid | [`prompt-01.md`](prompt-01.md) | **yes** | Test-tree only. The review is whether a published measurement survived being moved — §4.1's two rows are the whole safety net |
| 02 | The accuracy-parameter inventory | [`prompt-02.md`](prompt-02.md) | **yes** | Documents only. Ends in a **stop**: prompts 03 and 04 take their scope from its table, and the user reads it first |
| 02a | Make the source grid buildable | [`prompt-02a.md`](prompt-02a.md) | **yes** | The first production change, under a **widened §0.5 boundary** (campaign README §7 D6). Acceptance is **bit-identity** with two published digests, not improvement. Ends in a **hand-back**: prompts 03 and 04 are written from 02's output and 02a's together |
| 03 | `GkNumericIntegration`, and the floor that decides whether it matters | [`prompt-03.md`](prompt-03.md) | **yes** | The sector **D1** turns on: ~65,000 objects per model, never swept. Acceptance is a **matrix**, not a diagonal, and `unchanged` is a result. Ends in a **hand-back**: D1 is the user's |
| 03a | `TkNumericIntegration` and `wavenumber_exit_time` | [`prompt-03a.md`](prompt-03a.md) | **yes** | Two targets that fail differently, and the one that **closes D1**. The $T_k$ re-take is under the sector's own `BREAK_POINT_ALL`, which has never been swept; `wavenumber_exit_time` is a **location, not a value**, and is the campaign's likeliest §6.1 **rule 6** row. Ends in a **hand-back**: D1 closes on the user's acceptance |
| 04 | Audit the order-governed targets | [`prompt-04.md`](prompt-04.md) | **yes** | The first prompt to write a fixture other prompts' tests read. D5 gives it three files; **three test modules read the block and only one of them is inside that carve-out**, which is what the review is for. Ends in a **hand-back**: D3 is the user's and prompt 05 waits on it |
| 04b | Regenerate the convergence block, and repair the two tests that read it | [`prompt-04b.md`](prompt-04b.md) | **yes** | Lands what 04 measured and stopped short of writing. **D5 widened by one file (D8)**; the review turns on whether the threshold repair is a bound on the production quantity or a bigger `QCD_FLOOR_FACTOR`. Closes two issues. Ends in a short hand-back, not a decision |
| 05 | Decouple | — | **held** | Its charter is fixed (campaign README §3.5) and **both gates are now open** — D1 closed 2026-09-17, D3 settled 2026-09-18. Written after 04b lands |
| 06 | `QuadSourceIntegral`, close-out, the provenance note | — | **held** | Assembles from the earlier logs |

**02a is an insertion, not a renumbering** (campaign README §3.2a, §7 D6, 2026-09-17). It carries a
letter so that §§3.3–3.6's charters and §6.2's acceptance rows keep the numbers the rest of the tree
cites. It exists because the version-2 source grid cannot be built on `QCD_Cosmology` at the anchor a
QCD production run uses, and prompts 03, 04 and 06 are each chartered to measure over the production
grids on **all three models** — so without it every one of them must measure QCD at LambdaCDM's
anchor, the defect prompt 01 exists to close.

**Prompts 05 and 06 are deliberately not written yet**, and that is a decision of 2026-09-16,
not an omission — README §3 fixes each one's charter and §6 fixes its acceptance, so what is held
back is the *method*, not the commitment. Writing them against an inventory that prompt 02 exists to
establish would repeat the error the 2026-09-12 plan made. **Since 2026-09-17 the precondition is
02 and 02a together**, for the reason "Running one" gives. Board §1 records which are written.

**03 is split from 03a, not renumbered** (campaign README §7 **D7**, 2026-09-17). Prompt 03 takes
`GkNumericIntegration` and the consumer-spline floor — board item **T4** — and 03a takes
`TkNumericIntegration` and `wavenumber_exit_time`, **T5** and **T6**. No item is renumbered. The
split exists because §5 rule 1 makes the commit the rollback boundary and the three targets are not
comparable in weight: one is ~65,000 objects per model and has never been swept, the other two are a
re-take and a small new measurement.

## Running one

Start with, for example:

> Read `prompts/tolerance-convergence/orchestrator/prompt-01.md` and follow it.

**Take the baselines the prompt names before dispatching anything.** They cannot be reconstructed
after the fact.

Run **01 → 02 → 02a → 03 → 03a → 04 → 04b → stop**. 02, 02a, 03 and 03a each end in a hand-back, and
**04–06 are written from what has landed** (user decision, 2026-09-17): 02 says which targets the
audits own, 02a settles the anchor question that **T6** — prompt 03a's row — turns on, and 03
settled whether the two tolerance axes separate in the sector that carries the cost. **03a was
written on 2026-09-17, after 03 landed and after the user accepted D1's $G_k$ half**, which is what
§7 D7 asked for: the $T_k$ re-take is chartered knowing what the $G_k$ sweep found about the axes,
and explicitly forbidden from borrowing its conclusions (prompt 03a §4.3).

02a recommends nothing, so it is not a stopping point in campaign README §4.1's sense; it is a stop
because the next prompt was not yet written. **03 and 03a are stopping points in §4.1's own sense**:
each ends in a recommendation the user must accept under **D1**. The user accepted the
`GkNumericIntegration` half on **2026-09-17**; **D1 does not close until 03a's two targets are
accepted as well**, and prompt 05 unblocks only then. Each prompt's **§1 baseline must be taken
before dispatch** and cannot be reconstructed afterwards.

## The rules that bind the orchestrator

The same ones `prompts/GkTk-remedial/orchestrator/README.md` and
`prompts/background-solver-robustness/orchestrator/README.md` set out, unchanged:

1. **You do not write code.** Not a fix, not a test, not a docstring.
2. **One prompt, one subagent, one commit.**
3. **Do not re-derive the work.** Check that the prompt's own tests pass when *you* run them, that
   the log classifies every deviation, that the boards and `docs/OPEN_ISSUES.md` were updated in
   the same commit, and that the diff stayed inside its allowed files.
4. **Give each subagent only its own prompt.** Do not let it read the others; a prompt that knows
   what comes next starts optimising for it. This matters more than usual here — prompt 02's whole
   value is that it reads the tree rather than the plan, and an agent that has read prompt 03 will
   inventory what prompt 03 wants to find.
5. **Relay every subagent question verbatim.** Do not answer it yourself.
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says about it.
7. **An agent must never assume `HEAD` is its own** — planning and orchestration commits land on
   this branch. Every dispatch says so.

## The checks that apply after every prompt

Run these yourself, at the subagent's commit, before dispatching the next one.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -40
```

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -40
```

```bash
./venv/bin/python -m black --check $(git diff --name-only HEAD~1 HEAD -- '*.py')
```

- **Baseline is `bc6dc97`: `ComputeTargets` 452, `CosmologyModels` 39** (board §5 note 14). A
  campaign document written before 2026-09-16 19:32 will say 447 and 30; those are the
  `acd5b8e` figures and an orchestrator checking for them would stop on a healthy tree.
- `ComputeTargets` may **rise** at prompt 01 (it adds a test module) and must **not fall**.
- `CosmologyModels` must read **39** at every commit of prompts 01, 02, 02a and 03 — none of them
  touches that package, so any movement at all is unintended.
- Both suites print model banners on stdout, so **`| tail -5` will not show the verdict**. Capture
  to a file and grep it, or use `tail -40`.
- `ComputeTargets` takes ~164 s. That is normal, not a hang.

Plus, for **prompt 02 only** — it claims to change no code at all:

```bash
git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs/tolerance-convergence'
```

must be **empty**.

And for **prompt 02a only** — it is the one prompt before 05 that changes production code, and D6
bounds what it may change:

```bash
git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs'
```

must list **only** `main.py`, `ComputeTargets/tests/wkb_reference.py` and
`ComputeTargets/tests/test_source_grid.py`; and `git diff HEAD~1 HEAD -- main.py` must touch
`source_grid_spacing_profile` and no other function. See [`prompt-02a.md`](prompt-02a.md) §3.1.

And for **prompt 03 only** — it is the first prompt permitted to touch prompt 01's facility, and
only additively:

```bash
git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/convergence_reference.py
```

must show **zero deletions**, and
`PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_convergence_reference` must
read the same count as before the dispatch. See [`prompt-03.md`](prompt-03.md) §3.5.

## The dispatch template

For prompt NN, launch a subagent with **exactly** this context, with the model the campaign
README §3 names (**Opus** for 01, 02, 02a and 03). **For 02a the template's parameter-freeze sentence is
replaced**, because §7 D6 permits it one function; the replacement text is in
[`prompt-02a.md`](prompt-02a.md) §2 and must be used verbatim rather than paraphrased:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/tolerance-convergence/README.md`,
> `prompts/tolerance-convergence/RECONCILIATION.md` (**§7 is the current baseline; §§1–6 describe
> a superseded tree**), `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/tolerance-convergence/NN-<name>.md` and the files it tells you to read first.
> Execute the prompt exactly. **Do not read the other prompts in this campaign.** Follow README §5
> for the commit, the log, this campaign's board, any other board your prompt closes an entry on,
> and `docs/OPEN_ISSUES.md`. Other commits may land on this branch while you work: make exactly one
> commit, and do not amend, reset or rebase anything you did not create — if you need to change a
> commit you already made and it is no longer `HEAD`, stop and say so rather than rewriting.
> **The baseline is `bc6dc97`, with `ComputeTargets` at 452 and `CosmologyModels` at 39**; any
> document in this campaign quoting 447 and 30 was written against the superseded `acd5b8e` tree.
> **No prompt before 05 changes a parameter** (README §5 rule 8): if you find yourself needing to
> edit `config/defaults.py` or `main.py` to make your prompt pass, stop and say so rather than
> editing it. **A number without its reference's drift beside it is not a measurement, and a number
> without its grid generation beside it is not comparable** (README §5 rules 5 and 6). When you
> finish, reply with: the commit SHA, the **Result** line from your log, the "State handed to the
> next prompt" section verbatim, both suite counts, the number of production files in your diff,
> and every deviation with its classification tag.

## When to stop and ask the user

README §4.3's list, which applies to every prompt in this campaign:

- an agent reports an accuracy **below a floor of §2 (f)** — always an error, never a result,
  qualified only by §6.1 rule 5;
- an agent quotes a figure **without saying which grid generation it was taken on** (§2 (b));
- an agent proposes to change an integrator's **algorithm** rather than its parameters (§0.5);
- an agent touches `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/`
  (§0.4) or a file on `GkTk-remedial` §0.2's `transfer-remedial` list;
- an agent proposes to change `BREAK_POINT_KIND`, the source grid, or
  `RESIDUAL_WKB_REGION_MARGIN`'s value (§0.5) — **amended by §7 D6 for prompt 02a only**, whose
  carve-out is `main.source_grid_spacing_profile` alone; 02a touching the band, the margin,
  `build_z_sample`, `_solve_horizon_exit` or the digest is still a stop, and so is **any** prompt
  reporting a moved grid digest;
- a convergence test **fails to converge** and the prompt continues anyway — the error prompt 17
  made, and the reason this campaign exists;
- prompt 03 or 04 finds that the recommended parameters would change production cost by more than a
  factor of two **in a sector's total**, not per object.

And, for this batch specifically:

- **After prompt 02, always.** It is a natural stopping point (README §4.1) and prompts 03 and 04
  do not exist yet. Hand the user prompt 02's table and its §4 answers, and ask whether §2 (a) is
  to be corrected before 03 and 04 are written.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**
- **Prompt 01 reporting that §4.1's two rows did not reproduce.** That is a stop, not a tuning
  exercise.
- **Prompt 02a reporting a moved grid digest** for either published grid. A stop even if the new
  grid looks better; see [`prompt-02a.md`](prompt-02a.md) §3.2 for the likely cause.

**Prompt 04 was written on 2026-09-17, after 03a landed and after the user closed D1.** D1's closure
does not gate it — no tolerance in prompt 04 is one D1 settled — but it changes what the prompt is
for: **D3 is now the only decision prompt 05 is still waiting on**, and prompt 04's §6 is where it
comes from. The prompt is written against the tree rather than the plan in one respect the plan did
not anticipate: `RECONCILIATION.md` §5 said the re-run needs three files, and reading the tree shows
that **`test_background_cs_tau_friction.py` and `test_phase_residual.py` also read the block** and
are outside D5's grant. That collision is prompt 04 §2 and orchestrator §5's first stop, and it is
the reason the prompt tells its agent to determine the orders *before* writing the fixture.


**Prompt 04b was written on 2026-09-18, after the user read prompt 04's hand-back and settled two
things at once.** **D3** is settled as `replace with the orders`, with a standing instruction that
governs every later schema question in this campaign: *there is nothing to rebuild and nothing to
backfill, a schema change costs nothing, and a schema change that is needed is always the right
answer.* Prompt 04's §10 costed a migration and that framing is withdrawn — see campaign README §7
D3. **D8** widens D5 by one file so that 04b can land the block prompt 04 measured and declined to
write.

**The orchestrator's own error is recorded here because it is the kind that repeats.** Prompt 04's
hand-back reported the stop as though leaving a 58×-stale fixture in the tree were correct on the
merits, when it was only correct as rule-following; and it costed D3 in rows and rebuild effort,
which is not a currency this project trades in. An orchestrator relays a scope stop as a scope
stop, and asks the user whether the scope is still right.
