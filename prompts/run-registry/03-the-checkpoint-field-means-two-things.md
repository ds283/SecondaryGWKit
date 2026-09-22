# Prompt 03 — the `checkpoint` field means two things

**Campaign:** [`README.md`](README.md) · **Board item:** **G3** ·
**Board:** `IMPLEMENTATION_STATE.md` — created by prompt 01; add the G3 row.
**Closes:** `[02-a-datastore-checkpoint-path-would-be-appended-to-by-record]`.
**Recommended model:** **Opus**. The change is small. The judgement is in not letting a two-field
split become a type system.

**Read first:**

1. [`README.md`](README.md) §0 items **4 and 5**, and §0.2. The distinction in §0.2 — the registry
   records *existence*, checkpointing is *per-job* — is the one the code currently blurs.
2. `RunRegistry/__init__.py`: `begin()`'s `checkpoint` argument, `Run.checkpoint_path`,
   `Run.record()` and `Run.known()`. Four places, and the whole defect is in how they read one
   field.
3. `prompts/run-registry/logs/02-adopt-the-registry.md`, "Observations not acted on" item 2, and
   the board entry for the issue this prompt closes.
4. `docs/gktk-remedial/scoped_pipeline_run.py` around its `begin()` call — the only caller in the
   tree that puts a non-ledger path in the field, and the comment there explaining why.

**It changes no production compute path.** Nothing under `ComputeTargets/`, `CosmologyModels/`,
`LiouvilleGreen/`, `AdaptiveLevin/`, `Datastore/`, `main.py`, `config/` or `docs/spec/` appears in
the diff.

---

## 1. The defect, stated exactly

`manifest["checkpoint"]` carries two different things:

- **a unit ledger** — the append-only JSON-Lines file that `record()` writes and `known()` reads,
  one object per completed unit, each stamped with the provenance triple;
- **the durable store a job's results actually live in** — for a registered pipeline run, a
  SQLite datastore, which the registry only *names* and never writes.

`Run.record()` opens whatever that field names in `"a"` and appends a JSON object. On a registered
pipeline run that is **a SQLite file**, and the run's results are the 333 MB the campaign is built
to protect.

The quieter half is worse. `Run.known()` on a binary file parses no line successfully, drops every
one as torn, and returns `{}` — *"nothing has been done yet"* — rather than *"this is not a unit
ledger"*. A resume that trusts it recomputes everything. The A3 baseline took 10 h 33 m.

Neither is reachable today: `scoped_pipeline_run.py` calls neither method, and says so in a comment
at the `begin()` call. **The exposure is the next script that copies the pipeline pattern and then
adds units** — which is exactly what prompt 02's worked example invites somebody to do.

## 2. The window, which closes

Prompt 01 §1.2 makes the manifest **immutable**: written once at launch, never mutated. That is
right, and it means a field whose meaning changes after a long run has been registered cannot be
corrected on disk without breaking the invariant that makes the manifest worth reading.

At the time of writing, the three prompt-02 demonstration directories are the only
registry-written manifests that have ever existed, and they are being deleted. **Confirm this
before you start:**

```bash
find var/runs -name manifest.json
```

If that prints nothing, the split is free: no migration, no rewritten manifest, no compatibility
shim. **If it prints a manifest belonging to a real run, stop and ask** — §6.

## 3. What to build

Give the two meanings two fields, so that `record()` and `known()` can only ever touch the ledger.

- **The ledger field** keeps the narrow meaning and keeps the name `checkpoint` unless you have a
  better reason than tidiness: `checkpoint=True` still puts one in the run directory, a path still
  puts it where the job's own convention says, and `record()`/`known()` read only this.
- **The results field** names the durable thing the job's results live in, for jobs whose
  checkpoint is not a ledger. **It traces to README §0 item 4** — a datastore written into a
  session scratchpad, with nothing on disk saying where the results were — **and to item 5**, since
  a resume must know what it is resuming into. Campaign §5 rule 7 is therefore met; say so in the
  log rather than leaving the next reader to re-derive it.
- **`record()` must refuse**, clearly and by name, when the run declares no ledger. The message is
  the deliverable: it should name the field the caller probably meant, so that somebody who copied
  the pipeline pattern and then added units is told what to do rather than told "no".
- **`known()` must not answer `{}` for a file that is not a ledger.** Silently reporting "nothing
  done" for an unreadable ledger is the failure above. Distinguish *"the ledger does not exist
  yet"*, which is ordinary and returns nothing, from *"this file exists and is not a ledger"*,
  which is not. A torn **final** line must still be tolerated — that is prompt 01 §1.4 and its
  test, and it must keep passing.
- **Update `scoped_pipeline_run.py`** to put its datastore in the results field. That is a change
  to the registration block only; it is still not a fourth change to `main.py`, and the constraint
  from prompt 02 §3 item 1 still binds.

Pick the names and the mechanism yourself. Say in the log what you rejected.

## 4. What not to build

- **No schema version, no migration path, no compatibility shim.** §2 is why: there is nothing to
  migrate. A shim for manifests that do not exist is speculative generality, and campaign §5 rule 7
  forbids it.
- **No type system for manifest fields**, no validator, no schema file. Two fields and a refusal.
- **No sniffing the contents of a datastore**, and no importing anything from `Datastore/`. The
  registry names the store; it does not open it.
- **Nothing that deletes, truncates or rewrites any file a manifest names.** The whole point is a
  method that must not write to the wrong file.
- **No new manifest field beyond the one above.** If you think you need another, that is §6.

## 5. Tests

In `RunRegistry/tests/`. **No Ray, no datastore, no network**; a temporary directory is fine.

1. `record()` on a run whose results are a store and which declares no ledger **raises**, and the
   message names the field the caller should have used.
2. That run's results file is **byte-identical** after the refusal. Write a few bytes that are not
   JSON, attempt the `record()`, and compare. This is the test that would have caught the bug.
3. `known()` on a run whose declared ledger exists but is not a ledger **does not return `{}`
   silently** — assert whatever you chose, a raise or a loud discard, but not silence.
4. `known()` on a run whose ledger does not exist yet still returns nothing, quietly. That is a
   first run, and it is not an error.
5. A torn final line is still tolerated — prompt 01's test must still pass unchanged.
6. A run with both fields set keeps them separate: `record()` writes the ledger and does not touch
   the store.
7. `scoped_pipeline_run.py`'s registration still produces a manifest a stranger can read, with the
   datastore in the results field. Extend prompt 02's `test_pipeline_adoption.py` rather than
   starting a new module.

## 6. Stop conditions — stop and ask the user

- `find var/runs -name manifest.json` finds a manifest belonging to a **real** run. Renaming or
  re-meaning a field under it would either break prompt 01 §1.2's immutability or leave the
  manifest lying. Say which run it is and stop.
- You conclude a third field is needed. Name the README §0 failure it would have caught; it may be
  a sixth failure worth recording, which is the user's to judge.
- The split cannot be made without changing a production compute path, or without breaking
  `scoped_pipeline_run.py`'s claim to change only three things about `main.py`.
- You find yourself writing a migration, a schema version or a validator. That means the change has
  outgrown its cause — say what pushed it there.

## 7. Acceptance

1. `record()` cannot append to anything but a unit ledger, and refuses by name when there is none.
2. `known()` never reports "nothing done" for a file that is not a ledger.
3. The manifest carries the two meanings as two fields, each traced in the log to a README §0
   failure.
4. `scoped_pipeline_run.py` registers its datastore in the results field; `main.py` still runs
   byte-identically with and without `--register`.
5. All seven tests of §5 pass; every test prompts 01 and 02 wrote still passes unchanged. If one
   must change, that is a finding — say so in the log and do not quietly adjust it.
6. Existing suites at their baselines; `black --check` clean; board G3 row and
   `docs/OPEN_ISSUES.md` updated in the same commit, with the closed issue's row **deleted** from
   the index and recorded in the board's §4.

## 8. The log and the board

`logs/03-the-checkpoint-field-means-two-things.md`, template as README §5.1. Beyond it: the names
you chose and what you rejected; what `known()` now does on an unreadable ledger and why that
rather than the alternative; and the output of `find var/runs -name manifest.json` before you
started, which is the evidence that no migration was owed.

Board: the G3 row, and move
`[02-a-datastore-checkpoint-path-would-be-appended-to-by-record]` from §3 to §4 with the commit
that closed it.
