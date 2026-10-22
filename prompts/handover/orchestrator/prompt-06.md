# Orchestrator — prompt 06, the A3 v2 host and R4

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.**

**The prompt:** [`prompts/handover/06-a3-v2-host-and-rehearsal.md`](../06-a3-v2-host-and-rehearsal.md)
**Board item:** 06 · **Decides:** R4, which 07 needs.

## 0. What makes this prompt unusual

It is the first prompt in this repository that **runs on another machine**. The executing session
is a Claude Code session on the Mac Studio, opened over the saved `MacStudio` SSH connection in
`/Users/ds283/Documents/Code/SecondaryGWKit`. It commits nothing there. Its records come back to
the laptop, where you review them and make the one commit. From the laptop the Mac Studio is
`ssh macstudio-tunnel`; the plain `macstudio` alias times out.

## 1. Before you dispatch

1. **The branch and `HEAD`**, on the laptop: `handover-remedial`, clean. The dispatch SHA is the
   commit carrying this note. **Push it** (the user approved on 2026-10-22), and check that
   `origin/handover-remedial` equals it.
2. **Nothing long running**, on the laptop: `./venv/bin/python -m runregistrykit list`.
3. **R3's recorded runs** are staged **outside** the Mac Studio's checkout, at
   `~/handover-06-recorded/`: `var/runs/rrk-v020-adoption-03-reference-20261018T232838/` and
   `var/runs/rrk-v020-adoption-03-resumed-c-20261018T233941/`, copied whole. They are not copied
   into the checkout's `var/` as prompt §2 says, because the Mac Studio is on `main`, whose
   `.gitignore` does not ignore `var/` (it gained that line later), so `git status --short` would
   no longer be empty at §1.2. The session moves them to `var/handover-06/recorded/` after
   `git switch handover-remedial`. Record their SHA-256 sums on both machines.
4. **The Mac Studio**, read-only, as the planning found it on 2026-10-22: `main` at `9678f28`,
   clean; `venv` Python 3.13.16 with `ray` 2.45.0; no `var/`; no `~/ScienceData`; 122 GB free;
   automatic macOS update installation **on** (`AutomaticallyInstallMacOSUpdates = 1`), which
   does not bear on 06's minutes-long builds but is 07's gate.

## 2. Dispatch

One session on the Mac Studio, given the prompt, the dispatch SHA, the staging path of §1.3, and
these instructions: commit nothing, write only under `var/handover-06/` (plus the venv move of
§1.3 and the lock), and stop and report at any of prompt §5's conditions. It reports when its §7
records are complete.

## 3. The review — on the laptop, after copying `var/handover-06/` back

1. **Scope.** The Mac Studio's `git status --short` is empty and its `HEAD` is the dispatch SHA.
   Nothing outside the checkout's `var/`, `~/Documents/Code/retired-venvs` and
   `~/ScienceData/.rehearsal-lock` was written.
2. **The venv.** `pip_06.txt`'s freeze is identical to `01-pip.txt`'s second freeze; the three
   packages are at their pins and not editable.
3. **The roots** are at `01-roots.txt`'s counts.
4. **The fingerprints.** Both R4 stores are `39809dca…`, and `fingerprint_against_recorded_06.txt`
   is empty against R3's two runs. Read the comparison output itself, not the log's summary of it.
5. **The manifests.** Four runs, each with `git_dirty` `false`, the dispatch SHA, the Mac Studio's
   `host`, `datastorekit` 0.5.0 at `f47b9a2…`, and the launcher's `b2fa4ef1…`.
6. **The lock** was taken, set at each launch and released; `ray-processes` is empty at the end.
7. **The log** classifies every deviation, including §1.3's staging path (`STRUCTURALLY
   REQUIRED`), and records the stage times beside R3's.
8. **Bookkeeping**, which is yours: `out/` to `docs/handover/a3-v2/rehearsal/`, the log to
   `logs/06-a3-v2-host-and-rehearsal.md`, `verification.md` to
   `docs/handover/A3-V2-VERIFICATION.md`; the board's 06 rows, §3/§4, and `docs/OPEN_ISSUES.md`
   for anything opened, with the count and date corrected by counting. One commit, then push, so
   that 07's checkout can fast-forward to it.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate. Prompt §5's conditions; any failed check in §3. Report a
failure; do not repair it.
