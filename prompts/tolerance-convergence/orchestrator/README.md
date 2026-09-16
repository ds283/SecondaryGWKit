# Orchestrator prompts — the tolerance and convergence campaign

One orchestrator prompt per campaign prompt; there are no workstreams here (README §4.3). Each
dispatches one fresh-context subagent, reviews its commit against fixed criteria, and either
continues or stops and reports to the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 01 | The convergence harness and one production grid | [`prompt-01.md`](prompt-01.md) | **yes** | Test-tree only. The review is whether a published measurement survived being moved — §4.1's two rows are the whole safety net |
| 02 | The accuracy-parameter inventory | [`prompt-02.md`](prompt-02.md) | **yes** | Documents only. Ends in a **stop**: prompts 03 and 04 take their scope from its table, and the user reads it first |
| 03 | Audit the adaptive solvers | — | **held** | Its target list is prompt 02's output |
| 04 | Audit the order-governed targets | — | **held** | Likewise. D5 is settled yes, so it may write the fixture |
| 05 | Decouple | — | **held** | Its content *is* D1 and D3, which do not exist until 03 and 04 report |
| 06 | `QuadSourceIntegral`, close-out, the provenance note | — | **held** | Assembles from the earlier logs |

**Prompts 03–06 are deliberately not written yet**, and that is a decision of 2026-09-16, not an
omission — README §3 fixes each one's charter and §6 fixes its acceptance, so what is held back is
the *method*, not the commitment. Writing 03 and 04 against an inventory that prompt 02 exists to
establish would repeat the error the 2026-09-12 plan made. Board §1 records which are written.

## Running one

Start with, for example:

> Read `prompts/tolerance-convergence/orchestrator/prompt-01.md` and follow it.

**Take the baselines the prompt names before dispatching anything.** They cannot be reconstructed
after the fact.

Run **01 → 02 → stop**. Prompt 02's completion is a hand-back to the user (README §4.1), and the
next orchestrator prompt does not exist until then.

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
- `CosmologyModels` must read **39** at every commit of prompts 01 and 02 — neither touches that
  package, so any movement at all is unintended.
- Both suites print model banners on stdout, so **`| tail -5` will not show the verdict**. Capture
  to a file and grep it, or use `tail -40`.
- `ComputeTargets` takes ~164 s. That is normal, not a hang.

Plus, for **prompt 02 only** — it claims to change no code at all:

```bash
git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs/tolerance-convergence'
```

must be **empty**.

## The dispatch template

For prompt NN, launch a subagent with **exactly** this context, with the model the campaign
README §3 names (**Opus** for both 01 and 02):

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
  `RESIDUAL_WKB_REGION_MARGIN`'s value (§0.5);
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
