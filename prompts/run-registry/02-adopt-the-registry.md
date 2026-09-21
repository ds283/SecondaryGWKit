# Prompt 02 — adopt the registry, without falsifying anything

**Campaign:** [`README.md`](README.md) · **Board item:** **G2** ·
**Board:** `IMPLEMENTATION_STATE.md` — created by prompt 01; if 01 has not landed, stop and say so.
**Closes:** nothing. **Recommended model:** **Opus** — the work is small and the hazard is real.

**Read first:**

1. [`README.md`](README.md) §0 and **§4, the live state this campaign must not damage**.
2. `prompts/run-registry/logs/01-the-run-registry.md` and `RunRegistry/` — what prompt 01 actually
   built, and its stated limits.
3. `docs/gktk-remedial/scoped_pipeline_run.py` — the pipeline driver that works, and the one that
   ran for 10 h 33 m.
4. `docs/handover/realistic_large_x.py` — **read it, and read §2 below before you touch it.**
5. `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` — a hand-written manifest for a
   job whose checkpoint is a datastore rather than a JSONL file.

---

## 1. What to adopt, and what adoption means here

README §0.2's distinction governs everything in this prompt. **The registry records existence;
checkpointing is per-job and is sometimes already solved.**

- `docs/gktk-remedial/scoped_pipeline_run.py` runs for hours and **already has a checkpoint: the
  datastore.** The 2026-09-21 resume proved it works — every stage before the read-back defect
  replayed from stored results in about ninety seconds. It does not need re-plumbing. It needs to
  **register**: manifest before launch, heartbeat and status while running, terminal state on exit,
  under `var/runs/`. Its datastore path and the grid criterion belong in the manifest, because
  those are what a stranger needs to know whether the run is still relevant.
- New measurement scripts get the full pattern, registry and checkpoint both. This prompt writes
  none, so what it owes them is a **worked example** other scripts can be read against.

---

## 2. The hazard: leave `realistic_large_x.py` alone

`realistic_large_x.py` gates checkpoint reuse on a SHA-256 of its own source. Editing it by one
character discards all sixty cells in `var/runs/realistic_large_x_cells.jsonl` — the cells behind
every table in `docs/handover/REALISTIC-LARGE-X.md` — and costs a three-hour recomputation.

**Do not retrofit it.** It is a completed measurement whose provenance is load-bearing: that hash
is what proves the committed script produced the published numbers, verified in four seconds after
the fact. Its value as a fixed artefact exceeds its value as a registry client.

**And under no circumstances re-stamp its records with a new hash to preserve reuse.** That would
make the provenance stamp assert something false, which is worse than having no stamp. If you find
yourself reasoning towards it, that is a §7 stop.

Record the non-adoption in the log as a deliberate `IMPLEMENTATION CHOICE` with this reasoning,
and note in `REALISTIC-LARGE-X.md`'s campaign board entry — not the document — that the script
predates the registry by design.

`docs/source-remediation-verification/scoped_pipeline_run.py` is also left alone: it is broken by
`[13-scoped-run-driver-k-grid-literal]` and is another campaign's record.

---

## 3. What will go wrong

1. **Registering a pipeline run means wrapping a driver you must not otherwise change.** Every
   stage, tag, tolerance and queue parameter in that driver is `main.py`'s, and the driver's whole
   claim is that it changes only three things. Adding registry calls must not become a fourth.
   Prefer a thin wrapper or an opt-in flag over edits scattered through it, and say which you chose.
2. **Heartbeats from a Ray-parallel job.** The driver's work happens in Ray actors. The heartbeat
   must come from the parent, and must mean "the run is progressing", not "the parent is alive" —
   a parent blocked forever on `ray.get` is not progress. If you cannot make it mean the former,
   make it mean the latter and **say so in the manifest**, rather than implying a guarantee the
   heartbeat does not give.
3. **`units done` is not always available.** The pipeline reports progress per stage and per work
   queue, not as one total. Do not invent a denominator. An honest `stage` string beats a fabricated
   percentage.
4. **The existing run directories have no manifest.** `var/runs/a3-pilot/` and `a3-pilot-resume/`
   predate the registry. Do not retrofit manifests onto them, and do not tidy them — they are the
   evidence in README §0. The lister degrades gracefully on them, per prompt 01 §6 item 2.

---

## 4. Acceptance

1. `docs/gktk-remedial/scoped_pipeline_run.py` registers: a manifest on disk **before** the
   pipeline starts, status maintained while running, a terminal state on exit, all under
   `var/runs/`.
2. Demonstrated on a **short** run — a tiny `--k-count`, a throwaway datastore under `var/runs/`,
   killed deliberately partway — showing: the manifest exists from the first second; the lister
   shows it `running`; after the kill the lister reports it **stale or `killed`, not alive**.
   Quote the lister's output in the log before and after.
3. `realistic_large_x.py` is **untouched**, and `var/runs/realistic_large_x_cells.jsonl` still has
   60 cells and the single script hash `c1cd3598c23a`. State both before and after.
4. The A3 datastore, its backup, and both existing pilot run directories are untouched.
5. A worked example, short, that a future measurement script can be written against — in the log
   or as a docstring, not as a framework.
6. Existing suites unchanged; `RunRegistry/tests` unchanged or risen.
7. `black --check` clean; board and `docs/OPEN_ISSUES.md` updated in the same commit.

---

## 5. What this prompt does not do

- It does not edit `realistic_large_x.py`, `REALISTIC-LARGE-X.md`, or the
  `source-remediation` driver.
- It does not touch any production compute path, or `main.py`.
- It does not finish the interrupted A3 run. That is the user's, after
  `prompts/datastore-readback` prompt 01 lands.
- It does not add retry, restart or supervision behaviour.

---

## 6. Stop conditions — stop and ask the user

- Registering the driver cannot be done without changing what it computes, or without breaking its
  claim to change only three things about `main.py`.
- You conclude `realistic_large_x.py` should be retrofitted after all.
- You find yourself wanting to re-stamp, migrate or regenerate any existing checkpoint record.
- The demonstration run cannot be made short enough to be a demonstration rather than a measurement.

---

## 7. The log and the board

`logs/02-adopt-the-registry.md`, template as README §5.1. Beyond it: the lister's output before and
after the deliberate kill; the checkpoint cell count and script hash for `realistic_large_x.py`
before and after, showing it untouched; what the heartbeat actually means for a Ray-parallel job;
and the worked example.

Board: the G2 row, and the non-adoption of `realistic_large_x.py` recorded as a §3 issue if you
think a future campaign should revisit it — with the hash hazard stated, so that whoever picks it
up knows the cost before they start.
