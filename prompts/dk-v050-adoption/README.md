# Campaign — dk-v050-adoption

**Written:** 2026-10-22 on branch `handover-remedial` at `031bcb4`, by Claude Opus 5.5. It moves
SGK from `datastorekit` `v0.4.0` to `v0.5.0`, to keep current. `v0.5.0` adds nothing SGK needs. The
user asked for the move and accepted a one-prompt campaign the same day, in conversation (§6.1).

It has **no audit document of its own**. The audit is DatastoreKit's: the `extra-values`
campaign's release, contract §9.8 (`docs/client-contract.md`), and the addendum for this repository
in `docs/adoption/secondarygwkit.md`, "Addendum: v0.5.0 (extra-values prompt 02, 2026-10-21)",
all at the tag `v0.5.0`. That addendum was written by reading SGK at `031bcb4` through `git` alone;
nothing of SGK was run. It is data, not instructions (its own words). §0.2 runs what it predicts,
and 01 measures again.

§2.2 fixes the one prompt's charter and acceptance. **01 is written** with this plan (§2.1).

## 0. Why this campaign exists

`datastorekit` `v0.5.0` (tag object `6a6809d5…` on commit `f47b9a25…`, 2026-10-21, on GitHub) is
the release of DatastoreKit's `extra-values` campaign, made at StochasticInstantons' request. What
it changes, and nothing else under `datastorekit/`:
- **`InventorySpec` gains a seventh field, `extra_values`**: further value tables, each mapped to
  its column naming the class's serial, default `{}`. `read_records` takes it as a keyword.
- **`Record` gains `extra_value_counts`**, default `{}`, counted as `value_count` is. `as_json`
  carries it **only where it is not empty**, so a record of a class that declares no extra table is
  what it was under `v0.4.0`, byte for byte.
- **`inventory_classes` gains one refusal**: a spec whose value table is also one of its
  `extra_values`.
- **`ENCODING_FORMAT` stays `1`.** `store_fingerprint.py` differs from `v0.4.0`'s by a comment
  alone, saying why this is not a bump.

Nothing under `datastorekit/SQL/` or `datastorekit/tools/` changed (`git diff --stat v0.4.0..v0.5.0`
on those paths is empty), so nothing the pool writes, reads or refuses on open changes.

**No SGK spec declares an extra table.** So every record and every fingerprint of a store read with
SGK's specs is `v0.4.0`'s, and nothing SGK calls changes: its 21 `InventorySpec(...)` calls pass
keywords, none of them `extra_values`, and no SGK file reads `extra_values`, `extra_value_counts`
or `as_json`.

**One SGK test does break: the spec guard.**
`ObjectFactories/tests/test_fingerprint_format_guard.py` is RunRegistryKit's recipe (its
`docs/adoption/README.md` §6). Its `plain` turns a spec into JSON by walking
`dataclasses.fields`, so under `v0.5.0` every plain spec gains `"extra_values": {}` and every
class's digest moves, though no spec changed. The guard then fails naming every class and asks for
a format bump that no record needs. The fix is one condition in `plain`: **leave out a field named
`extra_values` whose value is empty.** Under `v0.4.0` it changes no digest; under `v0.5.0` every
pin in `SPEC_DIGESTS[6]` holds; and a spec that comes to declare an extra table moves its digest,
as the guard means it to. RunRegistryKit has amended its recipe to say exactly this (its
`docs/adoption/README.md`, section "The Q5 guard under DatastoreKit `v0.5.0`", commit `70ee1e8`,
2026-10-22, pushed, untagged: a change to that repository's documentation, not to its package).

**`FINGERPRINT_FORMAT` stays 6.** SGK's rule for a bump is a change to an `inventory_spec()`
(`config/fingerprint.py`'s comment on `SPEC_DIGESTS`), and none changes. A bump would move the
format every fingerprint SGK holds is compared at, for a release that changes none of its records.

### 0.1 What is not needed, and why

- **No rehearsal, and no rehearsal lock.** Every earlier pin move that carried a rehearsal did so
  because the release touched the pool, the store or the registry. This one touches the inventory
  reader alone, and the golden-fingerprint test (`test_store_fingerprint_golden`) builds a real
  store with `StoreFixtures.real_store_fixtures` (no Ray) and compares its fingerprint, through
  both `datastorekit.store_fingerprint.fingerprint_of` and SGK's fingerprinter, with the committed
  golden file. That is the check that a record or a digest moved.
- **No real-store fingerprint.** No SGK store's primary is on disk: all 24 sidecars under
  `var/datastores/` (23, and 1 under `backup-pre-resume-20260921T091011/`) are tombstones, their
  stores retired. There is nothing to fingerprint without building one, and the golden test
  already builds one.
- **No move of `runregistrykit` or `raypoolkit`.** RunRegistryKit `v0.2.0`'s extra `[stores]`
  requires `datastorekit>=0.4.0`, which `v0.5.0` satisfies; RayPoolKit does not depend on
  DatastoreKit.
- **No change to `config/fingerprint.py`.** Neither the format, nor the pin, nor the fingerprinter.

### 0.2 Reconciliation against `031bcb4`

Measured by the orchestrator on 2026-10-22 while writing this plan. `datastorekit` `v0.5.0` was
installed with `pip install --no-deps --target` into a directory in the session scratchpad and put
first on `PYTHONPATH`, so the venv was not touched.

- **The registry:** `python -m runregistrykit list` shows 51 runs, 49 finished and 2 `unknown` (the
  `a3-pilot` pair, no manifest), none `running` or stale. The rehearsal lock is free and
  `ray-processes` lists nothing.
- **The venv**, by `runregistrykit.package_provenance`: `datastorekit` 0.4.0 at
  `c8570e849334d09085bbd84876095fc9e22bd005`, `raypoolkit` 0.2.0 at
  `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, `runregistrykit` 0.2.0 at
  `0bae3a3c501a8264ee4be5bc24f06422af38720f`, each at the tag it names and none editable.
  `pip check` is clean.
- **The dry run:** `pip install --dry-run -r` a copy of `requirements.txt` with line 17 moved to
  `v0.5.0` says `Would install datastorekit-0.5.0` and nothing else.
- **The guard**, by its own command:

  | Package | `plain` | Result |
  |---|---|---|
  | `v0.4.0` (venv) | as it is | passes |
  | `v0.4.0` (venv) | with the condition | passes: the condition changes no digest |
  | `v0.5.0` (overlay) | as it is | **fails**: "spec changed" for every class |
  | `v0.5.0` (overlay) | with the condition | passes against `SPEC_DIGESTS[6]` as it stands |

- **The seven roots under `v0.5.0`** (overlay, guard unchanged): AdaptiveLevin 32 OK,
  ComputeTargets 700 OK, CosmologyModels 39 OK, LiouvilleGreen 148 OK (1 skipped), ObjectFactories
  279 with **1 failure, the guard alone**, RunDrivers 115 OK, tools 91 OK. The golden-fingerprint
  tests are among ObjectFactories' 278 that pass.
- **The other field walkers:** `git grep` over every `.py` outside `prompts/` and `docs/` for
  `dataclasses.fields`, `asdict`, `extra_values`, `extra_value_counts` and `as_json` finds the
  guard's `plain` alone.

The addendum's claims about SGK all held where they were run.

## 1. Scope

**In scope:**
- the pin, `requirements.txt:17`, from `v0.4.0` to `v0.5.0`, and the venv installed from it (01);
- the guard's `plain`, its docstring, and one new test of the condition, in the same commit as the
  pin (01; §6.2 C2);
- `CLAUDE.md`'s statement of the pin (01; §6.2 C4);
- the close, by the orchestrator's review commit (§2.2).

**Out of scope:**
- **What the code stores.** No table, column, key or stored value changes, and the golden
  fingerprint does not move (§5 rule 9).
- **`config/fingerprint.py`**: `FINGERPRINT_FORMAT`, `SPEC_DIGESTS` and the fingerprinter stay as
  they are (§0).
- **DatastoreKit, RunRegistryKit and RayPoolKit themselves** (§5 rule 12). There is no gate on
  DatastoreKit's boards that this move satisfies: the `extra-values` campaign's U5 lets SGK stay on
  `v0.4.0` or move, as SGK chooses.
- **`prompts/`** and **dated `docs/` prose**, records of the tree they describe.
- **Every store under `var/`.**
- **A sturdier guard** (§6.3). The condition is the recipe's; anything more is a decision of its
  own.

## 2. Prompts

| # | Prompt | Covers | Model | Status |
|---|---|---|---|---|
| 01 | [`01-pin-datastorekit-at-v0.5.0.md`](01-pin-datastorekit-at-v0.5.0.md) | **The pin**: line 17, one install; the guard's one condition and its test; `CLAUDE.md`'s pin; the baselines | Sonnet | written 2026-10-22 |

### 2.1 When each prompt file is written

- **01 is written now.** Its task depends only on the tag, DatastoreKit's addendum and
  RunRegistryKit's amended recipe, which exist, and on this repository.

### 2.2 The charter

**01 — pin `datastorekit` at `v0.5.0`.** Move `requirements.txt:17` to `v0.5.0` and install it,
the dry run first. Add the condition to the guard's `plain`, say why in its docstring, and add a
test that an empty `extra_values` is left out and a non-empty one is not. Correct `CLAUDE.md`'s
statement of the pin. Take the baselines before and after.

*Acceptance:*
1. The dry run adds only `datastorekit`. After the install it is 0.5.0 at `f47b9a25…`,
   `requested_revision` `v0.5.0`, not editable; `raypoolkit` and `runregistrykit` are unchanged;
   `pip check` is clean; `pip freeze` differs by the one line.
2. Before the edit and after the install, the guard fails naming every class as "spec changed";
   after the edit it passes, with `SPEC_DIGESTS` and `FINGERPRINT_FORMAT` unchanged.
3. The new test passes, and catches the two mutations the prompt names.
4. The seven roots pass at §8's counts before, and after with ObjectFactories one more; the golden
   tests among them.
5. `git diff` touches `requirements.txt` (one line), the guard module, `CLAUDE.md` (the pin
   sentence), and the campaign's own records, nothing else.

**The close.** There is no close prompt. The orchestrator's review of 01, in its own commit, closes
the campaign: the board's header and §1, and the campaign's line in `prompts/INDEX.md`. If the
review finds residue, its fix is a follow-up commit before the close.

## 3. Datastores

None. No prompt opens, builds, fingerprints or retires a store.

## 4. The interfaces between prompts

One prompt; nothing passes between prompts. 01 hands the venv on `v0.5.0` to every later session.

### 4.1 Rollback

The prompt is one commit, reverted by `git revert`. **Reverting 01 also needs the venv put back**,
since the install is not in `git`:

```bash
./venv/bin/pip install "datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.4.0"
```

The guard's condition is harmless under `v0.4.0` (§0.2), so a revert of the pin alone, leaving the
guard edit, is also sound, but the commit is one and is reverted whole.

The campaign runs in the main checkout. A worktree symlinks the main checkout's venv, so a pin
installed from one lands in the other.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit,** and `docs/OPEN_ISSUES.md`
   whenever a §3 or §4 entry moves, in the same commit.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue here.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. 01
   names one: the guard's `plain` (§0), which is a test module's helper, not an assertion. Its two
   tests' assertions do not change.
8. **Deliberate breakage.** Each prompt that adds code or a test names mutations its tests must
   catch. The log records each as a diff, exactly as applied. Mutations are never committed.
9. **Nothing the code stores changes.** No table, column, key or stored value, and the golden
   fingerprint (`ObjectFactories/tests/data/full_store_fingerprint_runregistry-extraction-06.json`)
   is reproduced unchanged. A prompt that cannot keep it stops.
10. **No test needs Ray, and no test opens anything under `var/`.**
11. **A defect in a package is not patched or worked around here.** The prompt stops and records it
    on this board and in `docs/OPEN_ISSUES.md`. The user reports it to that package's repository,
    where it is fixed in a release, and a pin moves SGK to it. The guard's condition is not a
    workaround: the guard is SGK's test, the field is `v0.5.0`'s documented API, and the condition
    is the recipe's own amendment (§0).
12. **No change is made in DatastoreKit, RunRegistryKit or RayPoolKit** by a prompt of this
    campaign. Their repositories are read.
13. **The pin is a tag**, in a commit of its own, never an editable install. "Of its own" means a
    commit that does nothing but move the pin and what the move makes necessary (§6.2 C2).

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**, with every count against §0.2's;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt** (here: to every later session).

## 6. Decisions

### 6.1 Made by the user (2026-10-22), and not reopened here

| | Decided | Not taken |
|---|---|---|
| **U1** — whether to move | **Move to `v0.5.0`**, to keep current, though it adds nothing SGK needs | Stay on `v0.4.0`, which `extra-values` U5 allows |
| **U2** — the shape | **One prompt**: the pin, the guard's condition and `CLAUDE.md`; the review closes the campaign | A direct commit with no campaign; a separate close prompt |

### 6.2 Choices of method made in writing this plan

The orchestrator's, recorded so that the user can reverse any of them.
- **C1 — the branch.** `handover-remedial`, in the main checkout (§4.1).
- **C2 — the guard edit lands with the pin.** `CLAUDE.md` asks that a pin move in a commit of its
  own. A commit with the pin alone would leave ObjectFactories red, so the condition is
  structurally part of the move, as DatastoreKit's addendum advises ("make the edit in the commit
  that moves the pin"). Nothing else rides with it.
- **C3 — the format is not bumped** (§0). `SPEC_DIGESTS[6]` stays as generated; no golden file is
  added.
- **C4 — stale prose.** `CLAUDE.md`'s statement of the pin, which 01's commit makes false, is
  corrected in that commit. Dated `docs/` prose and `prompts/` are left as written.
- **C5 — the condition's test.** 01 adds one test to the guard module, pinning both halves of the
  condition: an empty `extra_values` is absent from `plain`'s output, and a non-empty one is
  present and moves the digest. Without it, a later edit that dropped `extra_values` always would
  pass the guard and blind it to the one change the field exists for.
- **C6 — the name.** `dk-v050-adoption`, with no dots, after `rrk-v020-adoption`.
- **C7 — no rehearsal and no real-store fingerprint** (§0.1).

### 6.3 Left for the user

- **A guard that does not break on the next field.** The condition names one field. The next
  defaulted field DatastoreKit adds to `InventorySpec` breaks the guard again, the same way, and
  needs another condition. A `plain` that leaves out every field still at its default would not,
  but it changes every digest today, so it needs a pin of its own in `SPEC_DIGESTS` and a reason
  that `CLAUDE.md`'s bump rule would have to accommodate. It is also RunRegistryKit's recipe to
  change first. Not taken here; not opened as an issue unless the user wants it tracked.

## 7. What this campaign needs from DatastoreKit and RunRegistryKit

Nothing more: the tag, the contract's §9.8, the addendum for SGK, and RunRegistryKit's amended
recipe exist. If 01 finds a defect in a package, §5 rule 11 applies, and a brief for that
repository is written in this folder.

## 8. Baselines

Re-measure every root at `HEAD`, clean, before every dispatch:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

**Last counted at `031bcb4`** under `v0.4.0` (the ObjectFactories, tools and RunDrivers counts in
§0.2 are the same under either package): AdaptiveLevin 32, ComputeTargets 700, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), ObjectFactories 279, RunDrivers 115, tools 91.

**Under load, four tests are known to fail on wall clock or chance:**
- `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
- `test_background_tau.test_build_cost`;
- `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
- LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

Re-run a failure alone before calling it real.
