# Prompt 01 — the run registry

**Campaign:** [`README.md`](README.md) · **Board item:** **G1** ·
**Board:** `IMPLEMENTATION_STATE.md` — **does not exist; this prompt creates it** (§8).
**Closes:** nothing. It is infrastructure, and the thing it prevents has already happened five
times (README §0).
**Recommended model:** **Opus**. The code is small and dull. The judgement is in what *not* to
build, and in being honest about what the guard cannot do.

**Read first:**

1. [`README.md`](README.md) **§0 in full** — the five failures. Every design decision below traces
   to one of them, and §5 rule 7 says a field that traces to none of them does not get built.
2. `docs/handover/realistic_large_x.py` — **the checkpoint machinery and the provenance stamp**
   (`cell_key`, `load_checkpoint`, `append_checkpoint`, and the `script_sha256` / `git_head` /
   `git_dirty` triple on each record). This is the pattern to generalise. It works; do not
   redesign it for the sake of it.
3. `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` — a manifest written by hand under
   pressure. What it carries is roughly what a manifest needs; take the content seriously and the
   format not at all.
4. `CLAUDE.md` — repository mechanics. Tests live in `<package>/tests/`, run from the repository
   root, and **must not need Ray or a datastore**.

**It changes no production compute path.** Nothing under `ComputeTargets/`, `CosmologyModels/`,
`LiouvilleGreen/`, `AdaptiveLevin/`, `Datastore/`, `main.py`, `config/` or `docs/spec/` appears in
the diff.

---

## 1. What to build

A small top-level package, `RunRegistry/`, with `RunRegistry/tests/`. Top-level because `RayTools/`
already establishes that shape for a utility package, and because both `docs/` scripts and future
callers need it without `sys.path` surgery.

### 1.1 The layout on disk

```
var/runs/<run-id>/
    manifest.json     written once at launch, never mutated
    status.json       state, units done/total, heartbeat, pid, exit code
    checkpoint.jsonl   optional: one record per completed unit, append-only
    stdout.log  stderr.log
```

`<run-id>` is `<campaign>-<prompt>-<slug>-<YYYYMMDDTHHMMSS>`, so that it sorts, and so that a
stranger reading `ls var/runs/` learns who owns each one.

`var/` is already gitignored. **Nothing in the registry may live in a session scratchpad, in
`/tmp`, or anywhere derived from a session, conversation or account** — README §0 item 4 is what
that rule is for.

### 1.2 `manifest.json`

Immutable. It must carry enough that **a stranger with no conversation history** can say what the
job is and whether it still matters:

- a one-line plain-English `purpose`, and the campaign and prompt that own it;
- the exact argv and working directory;
- `git_head`, and a dirty flag;
- a SHA-256 of the script, where there is a single script;
- the unit of work and the expected count, where there is one;
- the checkpoint path, where there is one.

### 1.3 `status.json`

The only mutable file. `state` is one of `running`, `done`, `failed`, `killed`; plus units done and
total, an ISO `heartbeat`, the pid, and an exit code once known. Written atomically — write to a
temporary file in the same directory and `os.replace`, so a reader never sees a half-written file.

### 1.4 The helper

Whatever API you like, meeting these requirements. Keep it small; README §1 says a few dozen lines
plus a convention, and means it.

- **Begin** a run: create the directory, write the manifest, write `status.json` as `running`.
  The manifest must be on disk **before** the work starts, so a crash in the first second still
  leaves a record.
- **Record** a completed unit: append one JSON object to `checkpoint.jsonl`, `flush`, `fsync`.
  Each record carries the provenance triple. A torn final line must be tolerated on read, as
  `realistic_large_x.load_checkpoint` already does.
- **Known** units: read the checkpoint back, keyed by unit, so a resume recomputes only what it
  lacks. **A record whose script hash does not match the running script is not reused.** Discard
  or refuse — pick one, implement it, and say which in the docstring; do not blend, and **never
  re-stamp an existing record with a new hash**, which would falsify the provenance the stamp
  exists to provide.
- **Finish**: set the terminal state and the exit code.
- **List**: every run, newest first, with purpose, state, progress and liveness.

### 1.5 Liveness, which is where the last attempt went wrong

A run is alive iff its pid answers `kill -0` **and** its heartbeat is recent relative to a
staleness window the caller supplies.

**The registry must never identify a process by matching a pattern against command lines.**
`pgrep -f <script-name>` matches the *polling shell's own* command line, so the condition can never
be satisfied; that single mistake produced nineteen immortal shells, fifteen of them waiting on a
job that was already dead (README §0 item 3). Write that reason into the docstring, in one line, so
that nobody reinvents it.

**`RunRegistry/tests/` must contain a regression test for exactly this**: a liveness check
performed from a process whose own command line contains the target's name must not report the
target alive. That test is the campaign's whole justification in miniature.

### 1.6 The lister

`python -m RunRegistry list`, printing one line per run: id, state, progress, age, purpose,
and — loudly — any run whose state is `running` but whose heartbeat is stale, because that is a
crashed job pretending to be alive.

---

## 2. The rules, which are the deliverable

The helper is worth little without them. Add a section to `CLAUDE.md` — short, in the voice of the
existing "Repository mechanics" section — stating:

1. **Discovery.** At the start of a session, and before launching anything long, run the lister and
   report anything `running` or stale. A job's existence must not depend on a conversation
   remembering it.
2. **Liveness.** `kill -0` plus heartbeat. Never a command-line pattern match.
3. **Launch.** Detached (`nohup`, `disown`; `setsid` is not available on macOS), manifest written
   first, stdout and stderr to the run directory, pid recorded.
4. **Do not babysit.** Launch, verify in one check that it started, and **end the turn**. Polling a
   long job re-sends an entire context per poll and learns nothing; that, not wall-clock, is what
   makes waiting expensive. Resume once, when it is done.
5. **Durability.** Never a session scratchpad, never `/tmp`.
6. **Provenance.** Script hash and git SHA on every checkpoint record; a mismatch is never blended
   away.

---

## 3. What not to build

README §1 already forbids a scheduler, a queue, a daemon and a UI. In addition:

- **No automatic cleanup.** Nothing in this package may delete a run directory. The A3 pilot logs
  are evidence; something that tidies them is a liability.
- **No process supervision.** The registry records; it does not restart, kill or reap.
- **No lock files or exclusivity.** Two runs of the same job is a user error, not something to
  engineer against, and a stale lock is worse than the problem.
- **No field that README §0 does not justify.** §5 rule 7. If you cannot name the failure a field
  would have caught, leave it out.

---

## 4. Tests

In `RunRegistry/tests/`. **No Ray, no datastore, no network**; a `tmp_path`-style temporary
directory is fine and is not what the rule prohibits.

At minimum:

1. **The self-match regression** of §1.5. Non-negotiable.
2. Manifest written before work begins, and not mutated afterwards.
3. Checkpoint round trip: record, read back, resume skips what is known.
4. **A record whose script hash differs is not reused**, and the chosen policy (discard or refuse)
   is the one that happens.
5. A torn final line in the checkpoint is tolerated.
6. `status.json` is never observed half-written — exercise the atomic replace.
7. A run whose process is gone but whose state says `running` is reported stale, not alive.

---

## 5. What this prompt does not do

- It does not modify any existing script to use the registry. That is prompt 02, and it carries a
  hazard this prompt must not trip: `realistic_large_x.py` gates checkpoint reuse on a hash of its
  own source, so editing it discards the sixty cells behind
  `docs/handover/REALISTIC-LARGE-X.md`'s published tables.
- It does not touch any production compute path, or the datastores in `var/datastores/`.
- It does not delete or tidy anything already under `var/runs/`.

---

## 6. Acceptance

1. `RunRegistry/` exists, is importable from the repository root, and is a few dozen lines plus
   tests — not a framework.
2. `python -m RunRegistry list` runs and produces sensible output against the existing
   `var/runs/` directory, including the two A3 pilot directories that predate the registry and
   have no manifest. **Degrading gracefully on those is part of the test, not an edge case.**
3. All seven tests of §4 pass, the self-match regression among them.
4. The `CLAUDE.md` section is added, stating all six rules of §2.
5. Existing suites unchanged at their baselines; `RunRegistry/tests` up by exactly the tests added.
6. `black --check` clean; board created; `docs/OPEN_ISSUES.md` updated in the same commit.

---

## 7. Stop conditions — stop and ask the user

- The helper is growing past a few hundred lines, or you find yourself wanting a dependency. That
  means the design has drifted into a framework; stop and say what pushed it there.
- You cannot write the self-match regression test honestly — for instance the check passes only
  because of an accident of how the test harness names its processes.
- Adopting the rules would require changing a production compute path.
- You conclude a field is needed that README §0 does not justify. Say which failure you think it
  prevents; it may be a sixth one worth recording.

---

## 8. The log and the board

`logs/01-the-run-registry.md`, template as README §5.1. Beyond it: the API you settled on and what
you rejected; the discard-or-refuse choice and why; and **an honest limits section** — per-unit
resolution, the need for a serialisable unit of work, what the static self-match test does not
prove, and the fact that a registry nobody reads is worse than none.

`IMPLEMENTATION_STATE.md`: create it, modelled on `prompts/handover/IMPLEMENTATION_STATE.md` — §1
prompt table, §2 item table, §3 Active and unresolved issues, §4 Resolved issues, maintenance-rule
blockquote. `docs/OPEN_ISSUES.md`: add a §1 subsection for this campaign, plus any issue opened,
and correct the count and the date.
