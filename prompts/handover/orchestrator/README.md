# Orchestrator prompts — the hand-over campaign

One orchestrator prompt per campaign prompt. Each dispatches a single fresh-context subagent,
reviews its commit against fixed criteria, and either hands back or stops and reports to the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 00 | Domènech reconnaissance | — | n/a | **Already landed** (`ab5c0ed`), executed outside this flow on a separate account. Read-and-report; no orchestration |
| 01 | The Domènech general-$b$ oracle | [`prompt-01.md`](prompt-01.md) | **yes** | One new module and its tests; **no production file is modified**. Like `radiation-oracle` prompt 01, the review is about whether the tests *would fail*: the source paper's own kernel is mis-signed, and a faithful transcription is wrong by an overall sign while looking entirely plausible. Turns on the deliberate-breakage record and on test 5, the only comparison with an object this campaign did not write |
| 02 | The realistic-flavour large-$x$ harness | [`prompt-02.md`](prompt-02.md) | **yes** | A `docs/` measurement script and a document; **no production file and no test**. Suite counts must be *unchanged*, not risen. The review turns on the control cell reproducing KT §8 Table 8.1 and on whether the attribution of the two terms is honest about its own error |
| 03 | The policy-geometry census | — | **not yet** | A `docs/` census script and a document; **no production file and no test**. Unlike 01 and 02 it **needs a datastore**, so its dispatch has to name one and record which source-grid construction its rows carry. The review turns on the type/quality census reproducing `source-remediation-verification.md` §5.5's shape, on the interval counts being taken by node index rather than by converting $\log(1+z)$ back to $z$, and on the agent having opened the two §2 (o) issues **without fixing them** |

## Running them

> Read `prompts/handover/orchestrator/prompt-01.md` and follow it.

**Order: 01, then 02 and 03 in either order.** All three are independent work — campaign README §4
says A1, A2 and A3 have no prerequisites and no dependency on each other — but **01 creates
`IMPLEMENTATION_STATE.md`**, which the others then update. Running 01 later means moving that
clause of prompt 01 §8 into whichever runs first. Running any two concurrently in separate
worktrees means conflicting on the board and on `docs/OPEN_ISSUES.md`. Serialise them.

**03 has a precondition the others do not**: a datastore written on the current source grid, i.e.
after `qcd-background-audit` prompt 15 replaced the base density with the measured curvature
criterion. A census on the superseded grid answers a question nobody asked, and prompt 03 §7 makes
that a stop. Establish which datastore you are dispatching against **before** you dispatch, not
after.

**Take the baselines before dispatching anything.** They cannot be reconstructed after the fact, and
prompt 01 is one that *raises* `ComputeTargets` — so "did not fall" is not enough; the rise must
equal the tests the agent added.

## The rules that bind the orchestrator

The same ones `prompts/radiation-oracle/orchestrator/README.md`,
`prompts/GkTk-remedial/orchestrator/README.md` and
`prompts/tolerance-convergence/orchestrator/README.md` set out, unchanged:

1. **You do not write code.** Not a fix, not a test, not a docstring — and **not the deliberate
   breakage either**. Prompt 01 §3 makes the agent break its own implementation and record what
   failed; you check the record, you do not reproduce it by editing the tree.
2. **One prompt, one subagent, one commit.**
3. **Do not re-derive the work.** Check that the prompt's own tests pass when *you* run them, that
   the log classifies every deviation, that the board and `docs/OPEN_ISSUES.md` were updated in the
   same commit, and that the diff stayed inside its allowed files.
4. **Give the subagent only its own prompt** and the files that prompt tells it to read.
5. **Relay every subagent question verbatim.** Do not answer it yourself.
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says about it.
7. **An agent must never assume `HEAD` is its own** — planning, reconnaissance and orchestration
   commits land on this branch. The dispatch says so.
8. **No urgency, no cost.** Campaign README §0.4. If a subagent argues for a shortcut on the grounds
   that something is cheap, slow or expensive to regenerate, that is a deviation to record, not a
   reason to accept.

## Baselines

Taken at **`0283906`**, on this branch, 2026-09-19:

| Suite | Count | Verdict | Wall |
|---|---|---|---|
| `ComputeTargets/tests` | **530** | OK | 122 s |
| `CosmologyModels/tests` | **39** | OK | 0.7 s |

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -40
```

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -40
```

```bash
./venv/bin/python -m black --check $(git diff --name-only HEAD~1 HEAD -- '*.py')
```

- Both suites print model banners on stdout, so **`| tail -5` will not show the verdict**. Capture
  to a file and grep it, or use `tail -40`.
- `LiouvilleGreen/tests` was **not** baselined here, and neither prompt touches that package. It
  was still running after ~20 minutes when these figures were taken, which is **expected, not a
  hang**: `[08-3bessel-plot-cost-dominates-the-suite]` records `test_3bessel_analytic` spending
  its whole wall clock — **21.2 min for one test** — evaluating 250-point grids to draw figures,
  not on assertions. If you want it as a control, start it before you dispatch and collect it
  afterwards; do not block on it, and do not kill it and call the suite broken.
- **The known flake is not a stop.**
  `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object` asserts a wall-clock
  figure with about a 1× margin and fails roughly one run in three. Confirm by re-running that
  module alone before attributing anything to the commit.
- **`./venv` does not exist in a fresh worktree.** Symlink the main checkout's and add it to
  `.git/worktrees/<name>/info/exclude`.
