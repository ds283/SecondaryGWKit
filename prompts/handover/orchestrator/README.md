# Orchestrator prompts — the hand-over campaign

One orchestrator prompt per campaign prompt. Each dispatches a single fresh-context subagent,
reviews its commit against fixed criteria, and either hands back or stops and reports to the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 00 | Domènech reconnaissance | — | n/a | **Already landed** (`ab5c0ed`), executed outside this flow on a separate account. Read-and-report; no orchestration |
| 01 | The Domènech general-$b$ oracle | [`prompt-01.md`](prompt-01.md) | **yes** | One new module and its tests; **no production file is modified**. Like `radiation-oracle` prompt 01, the review is about whether the tests *would fail*: the source paper's own kernel is mis-signed, and a faithful transcription is wrong by an overall sign while looking entirely plausible. Turns on the deliberate-breakage record and on test 5, the only comparison with an object this campaign did not write |
| 02 | The realistic-flavour large-$x$ harness | [`prompt-02.md`](prompt-02.md) | **yes** | A `docs/` measurement script and a document; **no production file and no test**. Suite counts must be *unchanged*, not risen. The review turns on the control cell reproducing KT §8 Table 8.1 and on whether the attribution of the two terms is honest about its own error |
| 03 | The policy-geometry census | [`prompt-03.md`](prompt-03.md) | **yes** (2026-10-25) | **Ran 2026-10-25 on the Mac Studio, in parallel with `levin-phase-floor` 04, and passed review 7 / 7**, check 5 on the user's ruling (the §5.5 shape difference is the old run's recorded response range, `--zend 1e7`, which prompt §2.2 omitted); three rows re-counted from the laptop. **Before it ran:** refreshed 2026-10-25. A `docs/` census script and a document; **no production file and no test**. Re-pointed from the A3 baseline store to **A3 v2 on the Mac Studio** (`0e32f24a…`, `SourceGridConstruction_2`), read-only and Ray-free at `HEAD`, orchestrated there as 10 and 11, in parallel with `levin-phase-floor` 04 and A4; it commits nothing there. It checks 11's policy count rather than re-taking it. The review turns on the interval counts being taken by node index rather than by converting $\log(1+z)$ back to $z$, on three rows recounted by hand, on the §5.5 comparison being one of **shape** (prompt 03 §2.2 says why), and on the agent having drafted the three §2 (o)/(p) issues **without fixing them** |
| 04 | The $G_k$ phase decision test | [`prompt-04.md`](prompt-04.md) | **yes**; refreshed 2026-10-25 | **Refreshed 2026-10-25, not run; dispatchable.** A `docs/` script and a short memo; **no production file and no test**. Added after A2 landed, because A2's fixture turned out to carry a $G$ phase one campaign behind production. It decides whether A2's numbers stand, and is now the only planned answer to `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`. **On the laptop, in a worktree at `98e68d3`**: its harness raises at `HEAD` since B1. Lands by rebase, as B1 did. The review turns on one thing a green suite cannot see: whether the threshold for "material" was fixed **before** the numbers were seen — so the prompt fixes it and the dispatch restates it |
| 05 | The Ray address default | [`prompt-05.md`](prompt-05.md) | **yes** | Added 2026-10-16 for an issue the user assigned here on 2026-10-12; **not seam work** and independent of every other prompt. Seven `--ray-address` defaults go from `auto` to `local`, plus one `ast` test module. The review turns on the A3 v2 launcher, `quadsource_atol_sweep.py`, being byte-identical, and on the test reading the scripts rather than restating them. Lands before the A3 v2 launch or with it |
| 06 | The A3 v2 host and R4 | [`prompt-06.md`](prompt-06.md) | **yes** | Added 2026-10-22; **not seam work**. **Runs on the Mac Studio**, in a Claude Code session over the saved `MacStudio` SSH connection, and **commits nothing there**: everything it writes is under `var/handover-06/`, which the orchestrator copies back, reviews and commits here. Before dispatch the orchestrator pushes the dispatch commit and copies R3's two recorded run directories to the Mac Studio. The review turns on the venv's `pip freeze` equalling the laptop's and on both R4 stores reproducing `39809dca…` |
| 07 | Launch A3 v2 | [`prompt-07.md`](prompt-07.md) | **yes** | Added 2026-10-22, held on 06. One launch on the Mac Studio, one check, end. Its gate includes the user's confirmation that automatic macOS update installs are off there. Its records are committed here while the build runs; **the Mac Studio's checkout is not moved until 08** |
| 08 | After the A3 v2 build | [`prompt-08.md`](prompt-08.md) | **yes** | Added 2026-10-25, when the build had ended. On the Mac Studio, as 06 and 07. The store's sidecar and fingerprint, the after-build checks, the readers, a reproduction of the A3 baseline's depth census, and the lock's release. The review turns on the sidecar's digest equalling the run's `0e32f24a…`, the store files being unchanged by everything after it, and the depth census reproducing the baseline's method on R4's store before the real one |
| 09 | `analytic_G_w` outside its domain | [`prompt-09.md`](prompt-09.md) | **yes** | **Landed `1c19998`, reviewed 2026-10-25.** Added 2026-10-25. **On the laptop**, a fresh-context subagent as 01–05. The compute-side `None` and the reader helper, with a laptop rehearsal that must stay `39809dca…`. The review turns on bit-identity inside the domain, the breakage record, and the rehearsal fingerprint. One ruling at dispatch: the signed `Gk-linear` plot is filtered too, keeping its sign |
| 10 | Re-read A3 v2 | [`prompt-10.md`](prompt-10.md) | **yes** | **Ran 2026-10-25, reviewed on the Mac Studio**; check 9 (link form) failed and was corrected on committing, by the user's decision. Added 2026-10-25, held on 09. On the Mac Studio, as 06–08; its checkout moves to 09's tree for the first time since the launch, and `raypoolkit` to `v0.3.0`. **The orchestrator runs on the Mac Studio too**, in an SSH session, and commits nothing; the bookkeeping commit is made on the laptop. The review turns on the GkWKB read-back being complete and the store unchanged |
| 11 | A3 v2's seam record | [`prompt-11.md`](prompt-11.md) | **yes** | **Ran 2026-10-25 and passed review on the Mac Studio (7 / 7).** Added 2026-10-25. On the Mac Studio, read-only and Ray-free, orchestrated there as 10, and **run in parallel with `levin-phase-floor` 01** in the same checkout: the laptop fast-forwards the checkout first, and neither session runs a git command that writes. The review turns on the policy block agreeing with the table, the gap census reading `clamp_gaps_log1pz` rather than the allowance, and the store and `stdout.log` unchanged |
| 12 | B1: remove the $T_k$ gap, by overlap | [`prompt-12.md`](prompt-12.md) | **yes** | **Ran 2026-10-25: Part A (the code, `049c7c8`) reviewed 8 / 8 in its worktree; R8 on the Mac Studio reviewed 7 / 7, check 3 on the user's ruling; records in B1's records commit, `ad23a29`.** Written 2026-10-25, dispatched with `levin-phase-floor` 03. **Two parts.** The code on the laptop, in a worktree of its own (`levin-phase-floor` 02's running script imports two files B1 changes), landing only after 02 has landed; then rehearsal R8 on the Mac Studio, only after 03's orchestrator has ended there, because 03 needs the pre-B1 tree. The review turns on the clamp's removal being complete, the switch node matched by redshift identity, the breakage record, and R8 differing from R3 in `QuadSource` and `QuadSourceIntegral` only |

## Running them

> Read `prompts/handover/orchestrator/prompt-04.md` and follow it.

and, in the Mac Studio session,

> Read `prompts/handover/orchestrator/prompt-03.md` and follow it.

**01 and 02 have landed** (`c414451`, `0d7c05c`). **03 and 04 were refreshed on 2026-10-25 and
run in parallel, on different hosts** (the user, 2026-10-25): 03 on the Mac Studio, read-only on
A3 v2 at `HEAD` ([`prompt-03.md`](prompt-03.md)); 04 on the laptop, in a worktree at `98e68d3`
([`prompt-04.md`](prompt-04.md)). `levin-phase-floor` 04 runs on the Mac Studio at the same time.
They share no file but the board and `docs/OPEN_ISSUES.md`; whichever lands second resolves the
conflict there by reading both sides. After them: B2 (after A4), a fix for
`[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]`, then `levin-phase-floor`'s G1.

**05 is outside that order** (added 2026-10-16). It shares no file with 01–04 and depends on none of
them; its only constraint is to land before the A3 v2 launch or with it. Run it whenever that
launch is near, serialised with the others only because they all update the board.

*(Until 2026-10-25 this file said "04 comes next, before 03", serialised on one host. The
paragraphs below are why 04 was put first; the refresh made the two independent of each other in
files and hosts, so they now run together.)*

That ordering is not the one this file originally carried, and the reason it changed is worth
stating: 04 did not exist when the campaign was planned. It exists because A2, on landing, was
found to rest on a fixture whose $G_k$ phase is one campaign behind production — so **04 decides
whether A2's numbers stand**, and A2 is the instrument B2, D and E are scored on. Sequencing by
epistemic dependency rather than by the order things were written puts it first. 03 is unaffected
either way; it reads stored rows and does not touch A2's results.

The original three were independent work — campaign README §4 says A1, A2 and A3 have no
prerequisites and no dependency on each other — but **01 created `IMPLEMENTATION_STATE.md`**, which
the others update. Running any two concurrently in separate worktrees means conflicting on the
board and on `docs/OPEN_ISSUES.md`. Serialise them. *(03 and 04 are the exception the user made on
2026-10-25: their commits are made apart, and the second to land resolves the board and the index.)*

**03 has a precondition the others do not**: a datastore written on the current source grid, i.e.
after `qcd-background-audit` prompt 15 replaced the base density with the measured curvature
criterion. A census on the superseded grid answers a question nobody asked, and prompt 03 §7 makes
that a stop. Establish which datastore you are dispatching against **before** you dispatch, not
after. **Established 2026-10-25:** A3 v2, whose `GkSource` rows carry
`SourceGridConstruction_2`.

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
