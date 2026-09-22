# Log 03 — the `checkpoint` field means two things

**Prompt:** [`prompts/run-registry/03-the-checkpoint-field-means-two-things.md`](../03-the-checkpoint-field-means-two-things.md)
**Commit:** *(this commit)* — "Split the manifest's overloaded checkpoint field in two"
**Model:** Claude Opus 5
**Date:** 2026-09-22
**Result:** COMPLETE

## What shipped

**Two fields, and a refusal.** The manifest's `checkpoint` keeps the narrow meaning — the
append-only JSON-Lines **unit ledger** that `record()` writes and `known()` reads — and a new
`results` field names the **durable thing the job's results live in**, which the registry names and
never opens. `record()` and `known()` read `checkpoint` and nothing else, so neither can now reach
a datastore however a caller declares one.

| Field | Means | Read by | Traces to |
|---|---|---|---|
| `checkpoint` | the unit ledger: append-only JSON-Lines, one object per completed unit | `record()`, `known()`, `Run.checkpoint_path` | README §0 item 1 (a three-hour measurement that lived in memory) and §0.1 |
| `results` | the durable store the job's results live in — a pipeline run's SQLite datastore | nothing; `Run.results_path` exposes it, and the refusal message quotes it | README §0 item **4** — a datastore written into a session scratchpad, with nothing on disk saying where the results went — and §0 item **5**, since a resume must know what it is resuming into |

That is campaign §5 rule 7 discharged explicitly rather than left for the next reader: both fields
are justified by a §0 failure, and no third field was added.

**`record()` refuses by name.** A run that declares no ledger raises `ValueError` before opening
anything. The message is the deliverable — the caller who reaches it has copied the pipeline
pattern and then given their job units — so it says what was declared, what this method will not do
with it, and what to pass instead:

> run c-03-demo-20260922T140715 declares no checkpoint ledger in its manifest, so record() has
> nothing to append to. It names a results store (…/handover-A3-baseline-lambdacdm.sqlite), which
> this method will never write to: the registry names where a job's results live, it does not open
> them. A job with units to record needs a ledger of its own: pass checkpoint=True to begin() (or a
> path to a .jsonl file), which is the field record() and known() read.

The middle sentence is dropped when the run declares no results either, so a job that simply forgot
`checkpoint=True` is not told about a field it is not using.

**`known()` is loud about a file that is not a ledger, and still silent about one that does not
exist.** The rule that separates them is one sentence: *a line that does not parse is tolerated only
when it is the last line in the file and is the beginning of a JSON object.* So

- a **torn final line** — prompt 01 §1.4, a kill mid-append — is tolerated exactly as before, and
  prompt 01's test passes unchanged;
- a readable line **after** an unparseable one, a first line that is not `{…`, or a file that is not
  even text (a `UnicodeDecodeError` from a binary datastore) raise `ValueError`, naming the line and
  pointing at the `results` field;
- a ledger that **does not exist yet** returns `{}` with nothing printed. That is every first run,
  and it is the half that must not break.

**`scoped_pipeline_run.py` registers its datastore as `results`.** The change is six lines inside
`register()` — the keyword and the comment explaining it. It is still not a fourth change to
`main.py`: the text handed to `exec` is byte-identical to the text the driver at `815bdce` built
(verification §2), and registration remains something that happens to this driver's own process.

**Tests: `RunRegistry` 30 → 38.** Seven new methods in
`RunRegistry/tests/test_run_registry.py::TestLedgerIsNotTheResultsStore` and one in
`test_pipeline_adoption.py::TestTheDatastoreIsNotALedger`, mapping onto prompt §5 as:

| §5 | Test |
|---|---|
| 1 | `test_record_refuses_a_run_that_declares_no_ledger_and_names_the_field` |
| 2 | `test_the_results_store_is_byte_identical_after_the_refusal` |
| 3 | `test_known_is_loud_about_a_declared_ledger_that_is_not_one`, and `…_that_is_not_even_text` |
| 4 | `test_a_ledger_that_does_not_exist_yet_is_silent_and_empty` |
| 5 | prompt 01's `test_a_torn_final_line_is_tolerated`, **unchanged** — the diff to that file adds a class and deletes nothing — plus `test_tolerance_of_a_torn_line_stops_at_the_end_of_the_file`, which fixes where the tolerance ends |
| 6 | `test_a_run_with_both_fields_keeps_them_apart` |
| 7 | `test_the_datastore_is_registered_as_results_and_not_as_a_ledger`, in prompt 02's module |

## The names, and what was rejected

`checkpoint` was **kept**, as prompt §3 invites: renaming it would have touched prompt 01's tests
and its `CLAUDE.md` section for tidiness, and the name is right for what is left of it.

`results` was chosen for the new one. Rejected:

- **`results_store`, `store`, `datastore`** — all three name a *kind* of thing. The field must be
  able to hold whatever a job's results durably live in; a name that says "database" invites the
  next caller with a directory of HDF5 files to add a second field.
- **`results_path`** — `checkpoint` carries a path without saying so, and a `_path` suffix on one of
  the two would suggest the other is not.
- **`output`** — already means stdout in this package (`stdout_path`, `stderr_path`), and a run's
  stdout is not where its results live.
- **`checkpoint_kind` / `checkpoint_is_ledger`** — the shape the prompt §4 calls a type system. A
  discriminator on one field means every reader must branch, and nothing stops the two halves
  disagreeing; two fields cannot disagree because neither claims anything about the other.

Also rejected, and it is prompt 02's issue's *first* suggested repair: **`record()` refusing a
`checkpoint` that does not end `.jsonl`**. It is one line, and it guesses. A ledger under a job's own
convention need not be named `.jsonl`, and a datastore can be — the extension is not what makes a
file a ledger. The refusal is now on the declaration (*this run has no ledger*), which is a fact the
manifest states, not a fact inferred from a name.

## What `known()` does on an unreadable ledger, and why

**It raises `ValueError`.** The alternative was a loud discard — return `{}` and print a notice on
`stderr`, as the script-hash mismatch policy does. That is right for the mismatch and wrong here,
and the difference is what the caller is about to do with the answer:

- a discarded record is a unit the resume **should** recompute, so `{}` minus that unit is a true
  answer and the notice is a courtesy;
- an unreadable ledger means the registry **does not know** what has been done. `{}` is not a
  cautious answer, it is a false one, and the caller's next action is to recompute everything. A
  notice on `stderr` does not stop it; on a job launched with `nohup`, nobody is reading `stderr`
  until afterwards. The A3 baseline is 10 h 33 m, which is what that notice would have cost.

A raise is also recoverable in a way silence is not: the operator sees the path, the line and the
name of the field they probably meant, and either moves the path to `results` or deletes a file that
was never a ledger. Nothing in the registry deletes or truncates it for them (prompt §4).

## The window: `find var/runs -name manifest.json`

**Before starting** (the first command run in this prompt's working tree, at `815bdce`):

```
$ find var/runs -name manifest.json
$
```

Nothing. **After the change**, the same command still prints nothing — no test writes outside a
temporary directory, and `var/` is untouched. The three `run-registry-02-demo-*` manifests that
existed at prompt 02's close were the only registry-written manifests that had ever existed, and
they were removed at the user's direction before this prompt began. **No migration is therefore
owed, and none was written** — which is prompt §4's first prohibition satisfied rather than merely
obeyed. Had that command printed a manifest belonging to a real run, prompt §6's first stop
condition would have fired.

## Deviations from the prompt

1. **`known()` also raises on a first line that is not a JSON object, and on a file that is not
   text.** `STRUCTURALLY REQUIRED`. Prompt §5 test 3 asks that a declared ledger which is not one
   must not return `{}` silently, and prompt §5 test 5 requires the torn-final-line tolerance to
   survive. A rule of "tolerate any unparseable final line" satisfies neither at once: a one-line
   junk file — the exact shape of a mis-declared ledger — is a file whose only unparseable line *is*
   the final one, and would have been tolerated into silence. The second clause (it must begin `{`)
   is what closes that, and the binary case is the same defect arriving as a decode error rather
   than a parse error.
2. **Two tests for §5 test 3 rather than one**, and one extra for the boundary of the torn-line
   tolerance. `IMPLEMENTATION CHOICE`. The three unreadable shapes reach the same error by three
   different routes (a non-JSON first line, a mid-file tear, a decode failure), and a single test
   would leave two of them unasserted.
3. **`Run.results_path` exists although nothing in the tree reads it.** `IMPLEMENTATION CHOICE`. It
   is the symmetric accessor to `checkpoint_path`, it is two lines, and a resume — README §0 item
   5 — is the reader it exists for. It resolves the recorded path exactly as `checkpoint_path` does
   and opens nothing.
4. **The refusal message varies on whether `results` is set.** `IMPLEMENTATION CHOICE`. Prompt §3
   requires it to name the field the caller probably meant; naming a results store to a caller who
   has not declared one would be advice about a field they are not using.

No `UNINTENDED DRIFT`. Nothing under `ComputeTargets/`, `CosmologyModels/`, `LiouvilleGreen/`,
`AdaptiveLevin/`, `Datastore/`, `main.py`, `config/` or `docs/spec/` is in the diff; the four files
changed are `RunRegistry/__init__.py`, its two test modules and
`docs/gktk-remedial/scoped_pipeline_run.py`.

## Verification performed

**§1 The results file is byte-identical after the refusal.** `test_the_results_store_is_byte_identical_after_the_refusal`
writes 69 bytes that are not JSON, attempts `record()`, and compares the bytes — not merely that an
exception was raised, which would have tested the exception and not the file. It also asserts
`units_done` is still 0: a refusal is not progress. The pipeline-adoption test asserts the stronger
form, that the named datastore is not even **created** on the way to the refusal.

**§2 `main.py` runs byte-identically with and without `--register`.** The driver builds the text it
`exec`s in one place (`source = MAIN_PY.read_text()`, then the two `K_GRID_LITERALS`
substitutions), and that block does not read `run`; registration only wraps the `exec` in a
`try/except` that writes a terminal state. Measured rather than asserted: the transformed source was
computed from the driver **at `815bdce`** and from the working tree, against the same untouched
`main.py`.

```
main.py on disk       c0b93dff4b36481576456bd5d0450fc246d1e3130d00f8e859b8c8817ac21b55
exec'd source, HEAD   f3d23045690b009ea7c8ff9a9de434050ff0cf94f6beba2f61ddf5add974d8ce
exec'd source, tree   f3d23045690b009ea7c8ff9a9de434050ff0cf94f6beba2f61ddf5add974d8ce
```

`main.py` itself is not in the diff, and prompt 02's claim to change only three things about it
still holds: this prompt changed one keyword argument in the driver's own registration block.

**§3 Suites, at this tree, on this machine.** All OK, all at the baselines measured at `815bdce`
except `RunRegistry`, which gained this prompt's eight tests.

| Suite | Baseline | Here |
|---|---|---|
| `ComputeTargets` | 552 | **552 OK** (151 s; the `test_wall_time_per_object` flake did not fire) |
| `CosmologyModels` | 39 | **39 OK** |
| `LiouvilleGreen` | 148 (skipped=1) | **148 OK** (skipped=1) |
| `AdaptiveLevin` | 32 | **32 OK** |
| `Datastore` | 10 | **10 OK** |
| `RunRegistry` | 30 | **38 OK** |

`./venv/bin/python -m black --check .` — clean, 278 files.

**§4 Every test prompts 01 and 02 wrote still passes, unchanged.** The diff to
`RunRegistry/tests/test_run_registry.py` is a single hunk that adds 114 lines and deletes none, so
`test_a_torn_final_line_is_tolerated` is byte-identical to the version prompt 01 committed; the
diff to `test_pipeline_adoption.py` adds a class and two imports and deletes nothing.

**§5 Nothing under `var/` was touched.** `find var/runs -name manifest.json` prints nothing before
and after. `var/runs/realistic_large_x_cells.jsonl` is still
`f8e417cf730d9ddd2fd5cfdf5aac91d086094de65035bc223c3c517dcd4d8ceb`, 60 cells at script hash
`c1cd3598c23a`; `var/runs/` still holds exactly `a3-pilot`, `a3-pilot-resume` and the four loose
files of issue `[01-var-runs-holds-unattributable-loose-files]`; the A3 datastore, its shards and
`backup-pre-resume-20260921T091011` were not opened. Every test run directory is in a
`tempfile.TemporaryDirectory`, including the one that exercises the driver's own registration
block, which runs with `RunRegistry.DEFAULT_ROOT` patched and puts back both output streams and the
two signal handlers it takes over.

## Observations not acted on

1. **The lister does not print either field.** `python -m RunRegistry list` shows state, liveness,
   progress and stage, so a stranger reading it cannot see where a registered run's results live
   without opening the manifest — which is most of what README §0 item 4 is about. Not acted on:
   prompt §4 forbids a new manifest field, and a new *column* is a change to prompt 01's display
   contract that this prompt was not given. Not opened as an issue either: it is a suggestion for
   whatever prompt next touches the lister, not a defect, and the manifest does carry the fact.
2. **A caller can still declare a datastore as `checkpoint` if they insist.** `record()` refuses
   only when no ledger is declared; a caller who passes `checkpoint=<a datastore>` has asserted
   that it is a ledger, and the registry takes the manifest at its word. What has changed is that
   the *pattern* no longer invites it — the driver names its datastore in `results`, so the script
   that copies it inherits the correct declaration — and that `known()` on such a file now shouts
   instead of reporting "nothing done". Closing the remaining gap means inferring a file's kind
   from its name or its bytes, which is prompt §4's second and third prohibitions.
3. **Nothing in the tree calls `record()` or `known()` at all.** Both are exercised only by
   `RunRegistry`'s own tests: the pipeline driver has no units, and `realistic_large_x.py` keeps
   its own checkpoint (issue `[02-realistic-large-x-is-outside-the-registry]`). The refusal shipped
   here is therefore for the *next* script, which is what prompt §1 says it is for.

## State handed to the next prompt

- **The campaign is complete at 3 / 3**, and §4 of the board is no longer empty: this prompt closes
  `[02-a-datastore-checkpoint-path-would-be-appended-to-by-record]`, the issue prompt 02 opened.
  Two issues remain open, both recorded so that their price is known, neither work this campaign
  left half-done.
- **The window is closed by being used.** No registry-written manifest exists, so the split cost
  nothing; the next manifest written will carry both fields, and prompt 01 §1.2's immutability
  means the meaning cannot be revised again for free once one does.
- **The A3 baseline is untouched and still interrupted**, and when it is resumed with `--register`
  its datastore will be named in `results` — the field README §0 item 4 exists to have had.
