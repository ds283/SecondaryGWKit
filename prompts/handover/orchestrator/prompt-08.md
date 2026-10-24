# Orchestrator — prompt 08, after the A3 v2 build

Read [`README.md`](README.md) in this directory first, and [`prompt-06.md`](prompt-06.md) and
[`prompt-07.md`](prompt-07.md), whose arrangement this prompt reuses. **You do not write code.**

**The prompt:** [`prompts/handover/08-after-the-a3-v2-build.md`](../08-after-the-a3-v2-build.md)
**Board item:** 08.

## 0. What makes this prompt unusual

It is the first prompt that writes into the store's record, through `store create` and
`fingerprint --write`, and the first to read the kept store with every reader. Everything after the
sidecar must leave the store's files byte-identical. It also produces the campaign's first
production-scale measurement, the depth census, by a method that was never written down as a
script. So the script is tested on R4's store, whose count is known, before it reads A3 v2.

## 1. Before you dispatch

1. **The Gate**, read on the Mac Studio on 2026-10-25 by the orchestrator (`ssh macstudio-tunnel`,
   read-only): the run `done`, exit 0; the fingerprint `0e32f24a…`, 21 classes, no problems; the
   store files and no journal; the lock held with a dead owner; no Ray process; the checkout clean at
   `b90752d`. Two observations came from the Mac Studio session: the QSI tail ran on one worker for
   about two hours, and `stderr.log` holds 13 SciPy `IntegrationWarning`s. They are prompt §4's
   two log measurements.
2. **The user's decisions** (2026-10-25): dispatch 08; the archive is the Mac Studio, and nothing
   is copied.
3. **The dispatch SHA** is the commit carrying this note, pushed. The Mac Studio's checkout is
   **not** moved to it: 08 runs at `b90752d`, the build's tree, and reads the prompt from
   `origin` with `git show <dispatch SHA>:prompts/handover/08-after-the-a3-v2-build.md`. No code
   changed between them (`git diff --stat b90752d <dispatch SHA> -- . ':!prompts' ':!docs'` is empty).

## 2. Dispatch

To the Mac Studio session `handover-06-r4`. The prompt, the dispatch SHA, the instruction to read
the prompt by `git fetch origin` and `git show` without moving the checkout, and 07's extras:
`bash` for commands with variable arguments; commit nothing; write only under `var/handover-08/`
and the lock. **It may take as long as the readers need**; its waits are by background command on
a pid, never a foreground poll.

## 3. The review — on the laptop, after copying `var/handover-08/` back without `extract/`

1. **The sidecar** exists, its fingerprint is `0e32f24a…` and equal to the run's, and the store
   files' SHA-256 are the same in `store_files_before.txt` and after every later step.
2. **The checks**: `no divergence`, `clean`, one `version` row per shard, no journal or in-flight
   row. Read the outputs, not the log's summary.
3. **The readers**: six plus one, each exit 0, the store unchanged, no Ray process left; the
   fingerprint afterwards the sidecar's.
4. **The depth census**: the script is read-only (every `connect` `mode=ro`, no write statement),
   its R4 test counts 864, its population is 9280 at (1e-32, 1e-7), and its bins add up to 9280.
   The comparison says plainly that it is not like-for-like and why. **Check one depth-20 row, if
   any, yourself** against the shard with `sqlite3 'file:…?mode=ro'`.
5. **The lock** is released, `status` free, `ray-processes` empty.
6. **Bookkeeping**: `docs/handover/a3-v2/after/`, `logs/08-after-the-a3-v2-build.md`, §3 of the
   verification document, the board's 08 rows and any §3 issue opened, `docs/OPEN_ISSUES.md`. One
   commit, then push. Then **propose to the user**, from §4's numbers, whether the two A3 baseline
   issues narrow or close. Do not move them on your own reading.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate. Prompt §8; any failed check in §3. Never repair a store, and
never re-run a build.
