# Working notes for Claude

## Open issues — keep the project-wide index in step

[`docs/OPEN_ISSUES.md`](docs/OPEN_ISSUES.md) is the project-wide index of open issues. It exists
because issues opened by one campaign were being lost when that campaign closed.

**Whenever you add, narrow or close an entry in a campaign board's §3 (Active and unresolved
issues) or §4 (Resolved issues), update `docs/OPEN_ISSUES.md` in the same commit.** Specifically:

- **Opening an issue** — add a one-line row under the right heading, with the campaign board name
  and a hook short enough to read at a glance.
- **Closing one** — delete its row. Do not keep a "resolved" section here; the board's §4 is the
  record.
- **Assigning one to a future campaign** — move its row into the matching §1 subsection, and add
  an `**Assigned (date):**` line to the board entry saying which campaign owns it and why.
- Either way, correct the **count** and the **Last updated** date in the index header.

The index is an *index*. One line per issue, pointing at the board that holds the measurements,
the impact statement and the next step. Never copy issue content into it; if the two disagree the
board is right.

## Campaign conventions

Remediation work is organised as campaigns under `prompts/<campaign>/`: a `README.md` holding the
plan, numbered prompt files, a `logs/` directory, and `IMPLEMENTATION_STATE.md` as the status
board. `README.md` §5 of each campaign states the rules that campaign runs under. The invariants
that hold across all of them:

1. **One commit per prompt.** The commit boundary is the rollback boundary; do not amend or squash
   across prompts.
2. **Every prompt writes a log** to `logs/NN-<name>.md` using the template in that campaign's
   README §5.1, and classifies every deviation from its prompt as `STRUCTURALLY REQUIRED`,
   `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit** — its own row, the
   item-level table, and §3/§4 — plus `docs/OPEN_ISSUES.md` per the rule above.
4. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and open a §3 issue. Scope creep destroys the revert-per-prompt property. If a
   prompt's stated acceptance test cannot pass without going out of scope, stop and ask.
5. **Commit messages**: imperative, capitalised subject under ~72 characters with no prefix tag; a
   blank line; a prose body saying what was wrong, what changed and how it was verified, wrapped
   at ~80 columns; then `Co-Authored-By: Claude <model name> <noreply@anthropic.com>`.
6. **Verification documents are additive.** When a re-run supersedes a measurement, add a new
   subsection recording it; do not rewrite the original, which was correct for the tree it was
   taken on.

## The datastore layer is frozen — it is moving to DatastoreKit

Since 2026-10-07 the generic datastore layer is being extracted into a separate package,
**DatastoreKit** (`/Users/ds283/Documents/Code/DatastoreKit`, https://github.com/ds283/DatastoreKit),
which SecondaryGWKit, ChamPBH and StochasticInstantons will all depend on. Its plan is that
repository's `prompts/extraction/README.md`. The package is a copy of this repository's layer at
**`6f7f291`** (the import commit), and its first prompts check, line by line, that it still equals
that copy.

**Until SecondaryGWKit adopts the package (that plan's gate G2), these files do not change:**
- `Datastore/__init__.py`, `object.py`, `contract.py`, `replication.py`, `shard_paths.py`,
  `store_reader.py`, `store_inventory.py`;
- `Datastore/SQL/__init__.py`, `schema.py`, `ShardedPool.py`, `Datastore.py`, `ClientPool.py`,
  `SerialPoolBroker.py`, `ProfileAgent.py`;
- `Datastore/SQL/ObjectFactories/base.py`;
- `tools/sharded_store.py` and `tools/shard_key_audit.py`.

A defect found in them in the meantime is **recorded** as an issue on the board of the campaign
that found it (and in `docs/OPEN_ISSUES.md`), saying it is held by the freeze. It is **not fixed
here**. It is fixed in DatastoreKit after G2, and reaches this repository through a release. A
prompt whose acceptance test cannot pass without editing one of these files stops and asks.

The factories under `Datastore/SQL/ObjectFactories/` (other than `base.py`), `config/`, and every
caller of the layer are **not** frozen. They are this project's code, and change as usual. The
`RayTools/` files are not frozen either: they are not part of this extraction. SecondaryGWKit's
adoption will be a campaign in this repository, planned from DatastoreKit's adoption checklist;
the freeze ends when that campaign lands.

## Long-running jobs — the run registry

A job that takes hours must not be something a conversation owns. `RunRegistry/` records that one
exists, in `var/runs/<campaign>-<prompt>-<slug>-<timestamp>/`, and
`prompts/run-registry/README.md` §0 lists the five failures in one working session that are the
reason for each rule below. It records; it does not schedule, supervise, restart or lock. It
deletes nothing but a store's own files, and those only through `store retire`, which a person
runs and which leaves the store's sidecar behind as its record. It never deletes a run directory,
a sidecar or any other record.

1. **Discovery.** Run `python -m RunRegistry list` at the start of a session and before launching
   anything long, and report anything `running` or stale. A job's existence must not depend on a
   conversation remembering it.
2. **Liveness.** A run is alive iff its pid answers `kill -0` **and** its heartbeat is inside the
   staleness window. **Never a command-line pattern match**: `pgrep -f <name>` matches other
   pollers waiting on the same job, so the condition can never be satisfied — nineteen poll loops
   that could not exit, fifteen of them waiting on a job already dead.
3. **Launch.** Detached (`nohup`, `disown`; `setsid` is not on macOS), manifest written *first*,
   stdout and stderr into the run directory, pid recorded — the launcher passes a detached child's
   pid to `Run.heartbeat(pid=…)`.
4. **Do not babysit.** Launch, verify in **one** check that it started, and **end the turn**.
   Polling a long job re-sends an entire context per poll and learns nothing; that, not wall-clock,
   is what makes waiting expensive. Resume once, when it is done.
5. **Durability.** `var/runs/`, which is gitignored and in the repository. Never a session
   scratchpad, never `/tmp` — a 462-row census died with a scratchpad and cannot be re-examined.
6. **Provenance.** Script hash and git SHA on every checkpoint record, taken at launch. A record
   whose script hash does not match is discarded and recomputed, never blended and **never
   re-stamped**: re-stamping falsifies the thing the stamp exists to prove.

## Repository mechanics

- **Tests** live in `<package>/tests/` as `unittest` modules and run from the repository root:
  ```bash
  PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
  ```
  They must not need Ray or a datastore — use the stand-in model pattern in
  `ComputeTargets/tests/test_tk_source_functions.py` and `docs/spec-code-audit/scripts/`.
- **Formatting** is `black`, with no configuration. Run `./venv/bin/python -m black <files>` before
  committing; the tree is clean under `--check`.
- **`main.py` cannot be imported** — it parses `sys.argv`, opens a Ray connection and a
  `ShardedPool` at module scope, then runs the pipeline. To exercise it, use
  `docs/source-remediation-verification/scoped_pipeline_run.py`; to extract one of its functions
  for a test, use `ComputeTargets/tests/test_main_plumbing.load_main_py_functions`, which reads it
  with `ast` rather than executing it.
- **Author conventions that are conventions, not defects** — do not "correct" them: $a_0$ is
  absorbed, never "set to 1" (it lives in $k/a_0$ and $a_0\eta$); the Green's function is the
  unit-jump $\bar G_k$ in $z$; $c_s^2$ is `wPerturbations` in the transfer-function sector while
  $w_0$ is `wBackground` inside $f$; Jacobian and phase sign choices are conventions. The spec §0
  blocks under `docs/spec/` record these.
- **Redshift arithmetic.** Integration is in $\log(1+z)$. Converting $z\to\log(1+z)$ costs ~1 ulp
  and is safe; $\log(1+z)\to z$ is irreducibly lossy at large $z$ (at $z\sim5\times10^{14}$ the
  recoverable $1+z$ has a granularity of ~0.02), so it must never appear in an equality-like
  comparison such as a region-coverage guard. It is harmless where the recovered $z$ is only a
  quadrature limit.
