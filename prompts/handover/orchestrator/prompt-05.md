# Orchestrator — prompt 05, the Ray address default

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.**

**The prompt:** [`prompts/handover/05-ray-address-defaults-to-local.md`](../05-ray-address-defaults-to-local.md)
**Board item:** 05 · **Closes:** `[00-main-and-the-readers-default-to-ray-address-auto]`, held by
the `runregistry-extraction` board and assigned here by the user on 2026-10-12.

## 0. What makes this prompt unusual

It is the smallest prompt in the campaign and the only one that touches no part of the seam. The
change is seven string defaults. What can go wrong is not the change but its neighbours: the A3 v2
launcher, `docs/handover/quadsource_atol_sweep.py`, is the rehearsed artefact, and the obvious
alternative remedy — making the flag required — breaks it. Your review turns on the launcher being
byte-identical and on the test reading the defaults rather than restating them.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`.** `handover-remedial`. Record the SHA; the agent is told it and
   told that `HEAD` is not its own.
2. **Take the seven baselines.** Expected: `AdaptiveLevin` 32, `CosmologyModels` 39,
   `ObjectFactories` 279, `tools` 91, `LiouvilleGreen` 148 (`skipped=1`), `RunDrivers` 99,
   `ComputeTargets` 693.
3. **Record the launcher's hash:** `shasum -a 256 docs/handover/quadsource_atol_sweep.py`.
   Expected `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`.
4. **Check nothing long is running** (`./venv/bin/python -m runregistrykit list`, and the rehearsal
   lock). The agent starts no Ray and takes no lock, but its suite timings are worthless under a
   build.

## 2. Dispatch

One fresh-context subagent, given the prompt, the SHA, the baselines and the launcher's hash. One
commit; log at `logs/05-ray-address-defaults-to-local.md`; the `runregistry-extraction` board (the
issue moves to its §4), this campaign's board and `docs/OPEN_ISSUES.md` in the same commit;
`black`; prompt §6's stop conditions mean stop and ask.

## 3. The review — six checks

1. **Scope.** `git diff --name-only HEAD~1 HEAD` is `main.py`, the six `extract_*.py`,
   `ComputeTargets/tests/test_ray_address_default.py`, `prompts/handover/IMPLEMENTATION_STATE.md`,
   `prompts/handover/logs/05-*.md`, `prompts/runregistry-extraction/IMPLEMENTATION_STATE.md` and
   `docs/OPEN_ISSUES.md`. Nothing else.
2. **The launcher's hash is unchanged**, and `RunDrivers` reads 99.
3. **The test reads the scripts.** It parses each with `ast` and follows `main.py`'s argument to
   the `DEFAULT_RAY_ADDRESS` assignment; it asserts the one `ray.init` takes `args.ray_address`.
   A test that only greps for the string `"local"` is a stop.
4. **The breakage record** names, for each deliberate breakage, the test that failed.
5. **Counts.** Six roots unchanged; `ComputeTargets` up by exactly the methods added.
6. **Bookkeeping.** The issue in the `runregistry-extraction` board's §4 with a resolution, gone
   from its §3 and from `docs/OPEN_ISSUES.md` §1.1, the index count down by one; the 05 rows on
   this board. `black --check` clean; commit message in `CLAUDE.md`'s form.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §6 conditions; any proposal to edit the launcher or
`RunDrivers/tests/`; any failed check in §3 — report it, do not repair it.
