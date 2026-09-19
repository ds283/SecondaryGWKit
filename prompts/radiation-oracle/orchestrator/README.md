# Orchestrator prompts — the radiation oracle campaign

One orchestrator prompt for the campaign's one prompt. It dispatches a single fresh-context
subagent, reviews its commit against fixed criteria, and either hands back or stops and reports to
the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 01 | The Kohri–Terada radiation oracle | [`prompt-01.md`](prompt-01.md) | **yes** | Test-tree and one new module; **no production file is modified**. The first prompt in the tree whose review is about whether its tests *would fail* rather than whether they pass: three of the paper's statements are wrong in ways that produce a plausible function, so the review turns on the log's deliberate-breakage record and on test 6, the only comparison with an object in it this campaign did not write |

## Running it

Start with:

> Read `prompts/radiation-oracle/orchestrator/prompt-01.md` and follow it.

**Take the baselines the prompt names before dispatching anything.** They cannot be reconstructed
after the fact, and this prompt is the one that *raises* `ComputeTargets` — so "did not fall" is not
enough; the rise has to equal the tests the agent added.

**Ordering.** `prompts/tolerance-convergence` prompt 06a had to land first (user decision,
2026-09-18), because this prompt raises `ComputeTargets` above that campaign's 521 baseline. It
landed at `300d964` and that campaign is closed. Campaign README §2's last out-of-scope bullet —
"prompt **06a** has not landed" — is therefore **satisfied**, and is not a reason to stop. That
campaign's documents stay read-only to this prompt all the same.

## The rules that bind the orchestrator

The same ones `prompts/GkTk-remedial/orchestrator/README.md` and
`prompts/tolerance-convergence/orchestrator/README.md` set out, unchanged:

1. **You do not write code.** Not a fix, not a test, not a docstring — and **not the deliberate
   breakage either**. Prompt §7 makes the agent break its own implementation and record what
   failed; you check the record, you do not reproduce it by editing the tree.
2. **One prompt, one subagent, one commit.**
3. **Do not re-derive the work.** Check that the prompt's own tests pass when *you* run them, that
   the log classifies every deviation, that the board and `docs/OPEN_ISSUES.md` were updated in the
   same commit, and that the diff stayed inside its allowed files.
4. **Give the subagent only its own prompt** and the files that prompt tells it to read.
5. **Relay every subagent question verbatim.** Do not answer it yourself.
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says about it.
7. **An agent must never assume `HEAD` is its own** — planning and orchestration commits land on
   this branch. The dispatch says so.

## The checks that apply after the prompt

Run these yourself, at the subagent's commit.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -40
```

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -40
```

```bash
./venv/bin/python -m black --check $(git diff --name-only HEAD~1 HEAD -- '*.py')
```

- **Baseline is `300d964`: `ComputeTargets` 521, `CosmologyModels` 39**, both OK, with
  `config/defaults.py` at blob `76bab78d9e9374263a91da87d86b509b2ed019d0`. These are the figures
  board §5 note 6 records at the audit commit `2033cfc`; nothing between the two moved them.
- `ComputeTargets` **must rise, by exactly the number of test methods the agent added**, and must
  not fall. `CosmologyModels` must read **39** — this prompt does not touch that package.
- Both suites print model banners on stdout, so **`| tail -5` will not show the verdict**. Capture
  to a file and grep it, or use `tail -40`.
- `ComputeTargets` takes ~200 s wall at `300d964` (198 s measured 2026-09-18). Prompt §3 test 6 adds minutes to that; record the new
  figure, do not treat it as a hang.
- **The known flake is not a stop.**
  `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object` asserts a wall-clock
  figure against a 0.06 s limit and passes roughly one run in three. Confirm by re-running that
  module alone before attributing anything to the commit. It is likelier to fire after this prompt,
  not less, because the suite is longer and the machine is busier.
