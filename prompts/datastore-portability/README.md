# Campaign — datastore portability

## 0. Why this campaign exists

A `ShardedPool` datastore is one primary SQLite file plus *N* shard files alongside it
(`foo.sqlite`, `foo-shard0000.sqlite`, …). The primary records where each shard is in its `shards`
table. **It records the absolute path**, taken from `Path(db_name).resolve()` at
`Datastore/SQL/ShardedPool.py:71`. Symlinks are resolved first, so a store opened through a
symlinked directory records the target directory, not the link.

That path is written once, by `_write_shard_data` (`:270`), when the store is created.
`ShardedPool` never updates it. `_read_shard_data` (`:314`) uses it exactly as stored, and
`ShardedPool` never checks that the file is still there. The one other writer is a workaround:
`docs/handover/quadsource_atol_sweep.py` `prepare()` (`:648`) issues an `UPDATE` to re-point a
copied primary at its own shards, because the pool would not do it itself.

- **A store that has been copied** opens against the **original's** shards. It reads them, writes to
  them, and inserts a `version` row into each on open. The only file of its own that it changes is
  the copy's primary, whose `shard_keys` table then disagrees with the original's about which key
  lives on which shard. **This has happened.** On 2026-09-23 the first run of the atol sweep put 54
  rows into the A3 baseline store its own docstring promised to leave alone. That incident is why
  the workaround exists, and it is recorded as
  `[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]` on the
  [`run-registry`](../run-registry/IMPLEMENTATION_STATE.md) board §3, owner "datastore code". **This
  campaign is that owner, and closes that issue.**
- **A store that has been moved or renamed**: here the record and a reading of the code disagree.
  The issue says a primary naming a nonexistent shard *raises*. A reading of the code says each path
  goes straight to a `Datastore` actor, which **creates an empty database when its file is missing**
  (`Datastore.py:216`, correct behaviour for a single store). On that reading, the pool opens with
  empty shards at the old path, every sharded lookup misses, and you see a datastore in which
  nothing has been computed rather than an error. Prompt 01 settles which is true by measurement
  before changing anything (its §2 **P0**). Either way the fix is the same, and either way the
  copied case is silent.

**0.1 A second instance, not yet recorded anywhere.** `var/datastores/backup-pre-resume-20260921T091011/`
is the retained backup of the `handover-A3-baseline-lambdacdm` store, and the
`datastore-readback` campaign and that store's manifest both treat it as a safe copy. The backup's
own primary records the **live** store's shard paths
(`/Users/…/var/datastores/handover-A3-baseline-lambdacdm-shard000N.sqlite`, checked 2026-09-24,
read-only). So opening the backup as a datastore would open the live shards and write to them. The
backup is only good for restoring by copying files over the live ones. It cannot be opened in
place, and nothing about it says so.

**0.2 Correctness is the only objective.** Sequence by epistemic dependency, never by urgency or
by what is cheap. What a missing shard does today is measured before anything is changed. The
fail-closed check on missing shards comes before the relative-path change,
because the relative-path change rests on it: without the check, a mistake in path resolution
repeats the failure in §0 silently instead of raising an error.

## 1. Scope

**In scope:** `Datastore/SQL/ShardedPool.py` (the three methods that touch the `shards` table and
the constructor branch that opens an existing store), `tools/shard_key_audit.py` (another reader of
that table), and `Datastore/tests/`. **Not** `docs/handover/quadsource_atol_sweep.py`: it is the
record of a measurement, its re-pointing workaround stays as it is, and prompt 01 §2 explains why it
keeps working unchanged.

**Out of scope:** `Datastore/SQL/Datastore.py`, whose create-when-missing behaviour is correct for a
single store. Also the object factories, `main.py`, any physics, and any change to the columns of
the `shards` table. Renaming a whole store (primary and shards together) is also out of scope; see
prompt 01 §5.

## 2. Prompts

| # | Prompt | Covers |
|---|---|---|
| 01 | [`01-relative-shard-paths.md`](01-relative-shard-paths.md) | Measure what a missing shard does today; fail closed on it; record shard paths relative to the primary; read existing absolute records safely; one resolver shared with the audit tool. Closes `run-registry`'s `[04-sharded-store-paths-are-absolute-…]` |

## 3. Datastores

Two stores exist under `var/datastores/` (gitignored), each with 4 shards of about 85 MB:
`handover-A3-baseline-lambdacdm` and `handover-atol-sweep`, plus the backup in §0.1. The sweep
store's primary was re-pointed by `prepare()`, so it names its own shards, as absolute paths. **All three
are read-only to this campaign.** The acceptance test needs a copy, placed in a **different
directory** under `var/` and under a **different stem**, and discarded afterwards. Never open an
original, and never open the backup: on the unfixed tree, opening either writes to the live shards.

Before touching any store, run `python -m RunRegistry list` and confirm nothing is `running`
against it.

## 4. Baselines

At `a0922e7` on `handover-remedial`, the per-package counts are recorded on the boards of the most
recent campaigns. Re-measure before dispatching, and record `Datastore/tests` too, which exists
now:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . 2>&1 | tail -40
```

The suites print model banners on stdout, so `| tail -5` will not show the verdict. **Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
is a known wall-clock flake. Re-run that module on its own before attributing a failure to a
change.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, unchanged, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`.
4. **Do not fix things the prompt did not ask for.** Record them and open a §3 issue.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**

### 5.1 The log template

Subject, commit, result; **What shipped**; **Deviations from the prompt** (each classified);
**Verification performed**; **Observations not acted on**; **State handed to the next prompt**.
