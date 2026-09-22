# Orchestrator prompts — the hand-over campaign

One orchestrator prompt per campaign prompt. Each dispatches a single fresh-context subagent,
reviews its commit against fixed criteria, and either hands back or stops and reports to the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 00 | Domènech reconnaissance | — | n/a | **Already landed** (`ab5c0ed`), executed outside this flow on a separate account. Read-and-report; no orchestration |
| 01 | The Domènech general-$b$ oracle | [`prompt-01.md`](prompt-01.md) | **yes** | One new module and its tests; **no production file is modified**. Like `radiation-oracle` prompt 01, the review is about whether the tests *would fail*: the source paper's own kernel is mis-signed, and a faithful transcription is wrong by an overall sign while looking entirely plausible. Turns on the deliberate-breakage record and on test 5, the only comparison with an object this campaign did not write |
| 02 | The realistic-flavour large-$x$ harness | [`prompt-02.md`](prompt-02.md) | **yes** | A `docs/` measurement script and a document; **no production file and no test**. Suite counts must be *unchanged*, not risen. The review turns on the control cell reproducing KT §8 Table 8.1 and on whether the attribution of the two terms is honest about its own error |
| 03 | The policy-geometry census | — | **not yet** | A `docs/` census script and a document; **no production file and no test**. Unlike 01 and 02 it **needs a datastore**, so its dispatch has to name one and record which source-grid construction its rows carry. The review turns on the type/quality census matching `source-remediation-verification.md` §5.5's **shape** — a like-for-like comparison is impossible and prompt 03 §2.2 says why — on the interval counts being taken by node index rather than by converting $\log(1+z)$ back to $z$, and on the agent having opened the two §2 (o) issues **without fixing them** |
| 04 | The $G_k$ phase decision test | [`prompt-04.md`](prompt-04.md) | **yes** | A `docs/` script and a short memo; **no production file and no test**. Added after A2 landed, because A2's fixture turned out to carry a $G$ phase one campaign behind production. It decides whether A2's numbers stand. The review turns on one thing a green suite cannot see: whether the threshold for "material" was fixed **before** the numbers were seen — so the prompt fixes it and the dispatch restates it |

## Running them

> Read `prompts/handover/orchestrator/prompt-04.md` and follow it.

**01 and 02 have landed** (`c414451`, `0d7c05c`). **04 comes next, before 03.**

That ordering is not the one this file originally carried, and the reason it changed is worth
stating: 04 did not exist when the campaign was planned. It exists because A2, on landing, was
found to rest on a fixture whose $G_k$ phase is one campaign behind production — so **04 decides
whether A2's numbers stand**, and A2 is the instrument B2, D and E are scored on. Sequencing by
epistemic dependency rather than by the order things were written puts it first. 03 is unaffected
either way; it reads stored rows and does not touch A2's results.

The original three were independent work — campaign README §4 says A1, A2 and A3 have no
prerequisites and no dependency on each other — but **01 created `IMPLEMENTATION_STATE.md`**, which
the others update. Running any two concurrently in separate worktrees means conflicting on the
board and on `docs/OPEN_ISSUES.md`. Serialise them.

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

Measured at **`b80b0f5`**, the merge of `main`, on 2026-09-19. The campaign runs on
**`handover-remedial`**, which starts at `d6afa05`; the two commits between them touch only
markdown, so these figures hold there unchanged. Re-take them at your actual `HEAD` anyway —
the full tree is under three minutes.

| Suite | Count | Verdict | Wall |
|---|---|---|---|
| `ComputeTargets/tests` | **530** | OK | 140 s |
| `CosmologyModels/tests` | **39** | OK | 0.8 s |
| `LiouvilleGreen/tests` | **148** | OK (skipped=1) | 25 s |

**Full tree: ~2.8 minutes.** Before the merge it was ~24.5 — `LiouvilleGreen` alone measured
**1346.7 s** on this branch at `3f786ec`. Verify freely; there is no longer a reason to ration it.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -40
```

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -40
```

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -40
```

```bash
./venv/bin/python -m black --check $(git diff --name-only HEAD~1 HEAD -- '*.py')
```

- Both suites print model banners on stdout, so **`| tail -5` will not show the verdict**. Capture
  to a file and grep it, or use `tail -40`.
- **Do not set `THREE_BESSEL_DIAGNOSTIC_PLOTS`.** `main`'s `07c6041` gated
  `test_3bessel_analytic`'s convergence figures behind it, which is the whole of the 1346.7 s →
  25 s fall; setting it to anything other than `0`/`false` restores a ~22-minute run and 4.2 MB of
  figures per invocation. The skipped test is `test_YJJ_log_scaling`, which asserts nothing and is
  `skipUnless` the same flag — **`skipped=1` is the expected state, not a problem to fix.**
- Neither workstream A prompt touches `LiouvilleGreen`, so it is a control rather than a target; it
  is cheap enough now to take every time.
- **The known flake is not a stop.**
  `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object` asserts a wall-clock
  figure with about a 1× margin and fails roughly one run in three. `ComputeTargets`'s own wall
  time varies with machine load — 122 s and 140 s on two runs of the same 530 tests — so treat
  the count as the signal and the time as an aside. Confirm by re-running that
  module alone before attributing anything to the commit.
- **`./venv` does not exist in a fresh worktree.** Symlink the main checkout's and add it to
  `.git/worktrees/<name>/info/exclude`.
