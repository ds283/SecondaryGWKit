# Log — dk-v050-adoption prompt 01, pin `datastorekit` at `v0.5.0`

- **Subject:** [`01-pin-datastorekit-at-v0.5.0.md`](../01-pin-datastorekit-at-v0.5.0.md), board items K1, G1 and B1.
- **Commit:** this commit (parent `62de2c4`).
- **Result:** landed. K1, G1 and B1 pass; no stop condition of the prompt's §4 fired. All seven roots pass at README §8's
  counts before, and after with ObjectFactories at 280; no known flake fired, so none was re-run.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-22.

## What shipped

- **`requirements.txt:17`**, `@v0.4.0` to `@v0.5.0`. `git diff -U0` shows that line alone.
- **The venv**: `datastorekit` 0.4.0 at `c8570e84…` replaced by 0.5.0 at `f47b9a255e9ad58e0d0eaf85644f53b575884d31`, by the
  prompt's two `pip install` commands and no other. The install is not in the commit, and `git revert` does not undo it
  (README §4.1).
- **`ObjectFactories/tests/test_fingerprint_format_guard.py`**: `plain`'s dataclass branch skips a field named
  `extra_values` whose value is empty; `plain`'s docstring gains the sentence; the module docstring gains the paragraph
  naming RunRegistryKit `70ee1e8` and this campaign and prompt; `InventorySpec` is imported; `TestFormatGuard` gains
  `test_an_empty_extra_values_is_left_out` after `test_a_key_keeps_its_type`. `SPEC_DIGESTS`, `FINGERPRINT_FORMAT` and
  `config/fingerprint.py` are untouched. `black` leaves the file unchanged.
- **`CLAUDE.md`**: the pin sentence says `v0.5.0` "(moved to it on 2026-10-22 in the `dk-v050-adoption` campaign)"; the
  paragraph rewraps over lines 49-55. Nothing else moved.
- **`measure/out/`**: `01-pip.txt`, `01-guard.txt`, `01-roots.txt`.
- This log, and the board: header, the §1 row for 01, items K1, G1 and B1, the FP row of §2, `### 5.1 After prompt 01`.

## Deviations from the prompt

1. **`01-pip.txt` omits the 98 `Requirement already satisfied` lines of each pip run**, stated in place; every other line
   is kept, including each `Resolved ... to commit` line. **IMPLEMENTATION CHOICE** (the orchestrator allowed it).
2. **`01-roots.txt` records the load average before each root**, not once. **IMPLEMENTATION CHOICE.**
3. **The frozen baseline blocks are headed by `HEAD` `62de2c4`.** The prompt was written at `031bcb4`; nothing under code
   changed between them (`99199ca` and `62de2c4` touch `prompts/` only). **IMPLEMENTATION CHOICE.**
4. **The "after" roots ran with the working tree carrying the uncommitted edits** (the commit is the last step), so their
   header says so. **STRUCTURALLY REQUIRED.**
5. **The raw `grep -c "spec changed"` of the unedited guard's output is 22, not 21:** the closing advice line reads "A spec
   changed, so bump …". The 21 lines that name a class are `^  spec changed: <class>`; there are no "class added" or "class
   removed" lines. **STRUCTURALLY REQUIRED** (the message's own wording).
6. **No other deviation.**

## Verification performed

- **Gate, at the start:** tree clean at `62de2c4`; one worktree; `runregistrykit list` 51 runs, 49 finished, 2 unknown, none
  running or stale.
- **Baselines before** (`01-pip.txt`, `01-roots.txt`): `pip freeze`; `package_provenance`: `datastorekit` 0.4.0 at
  `c8570e84…`, `raypoolkit` 0.2.0 at `eaef0ce6…`, `runregistrykit` 0.2.0 at `0bae3a3c…`, none editable; the seven roots.
- **The dry run**, exit 0: `Would install datastorekit-0.5.0` and nothing else. It printed `Resolved … to commit 6a6809d5…`
  for DatastoreKit (the tag object), and the tag objects of the two other packages, `4e7eb153…` and `895e3c70…`, which it did
  not install.
- **The install**, exit 0: `Uninstalling datastorekit-0.4.0` (the replacement), `Successfully installed datastorekit-0.5.0`.
  Nothing else was installed, upgraded or removed.
- **`pip freeze`, before and after:** `diff` is the one line (17), `datastorekit @ …@c8570e84…` to
  `…@f47b9a255e9ad58e0d0eaf85644f53b575884d31`.
- **`pip check`:** `No broken requirements found.`
- **`package_provenance` after:** `datastorekit` 0.5.0, commit `f47b9a25…`, `requested_revision` `v0.5.0`, `editable` false;
  `raypoolkit` and `runregistrykit` identical to before.
- **`pip show`:** version 0.5.0, `Requires: ray, sqlalchemy`. **`direct_url.json`:** `commit_id` `f47b9a25…`,
  `requested_revision` `v0.5.0`, no `dir_info`. `importlib.metadata.version` prints `0.5.0`. `InventorySpec` has seven
  fields, `['leaves', 'parents', 'tags', 'values', 'validated', 'parent_sets', 'extra_values']`; `ENCODING_FORMAT` is `1`.
- **The guard, unedited, under `v0.5.0`** (`01-guard.txt`): of its 2 tests, `test_every_spec_is_pinned_for_the_current_format`
  FAILED with 21 lines `  spec changed: <class>` (`len(SPEC_DIGESTS[6])` is 21) and no "class added" or "class removed"
  line.
- **The guard, edited:** 3 tests, OK (`test_a_key_keeps_its_type`, `test_an_empty_extra_values_is_left_out`,
  `test_every_spec_is_pinned_for_the_current_format`), against `SPEC_DIGESTS[6]` as it stands.
- **The seven roots**, serially, foreground, README §8's command, load about 14 to 24 (the four known wall-clock tests did not
  fail):

  | Root | Expected | Before | After | Result |
  |---|---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | 32 | 32 | OK |
  | `ComputeTargets/tests` | 700 | 700 | 700 | OK |
  | `CosmologyModels/tests` | 39 | 39 | 39 | OK |
  | `LiouvilleGreen/tests` | 148 | 148 | 148 | OK (1 skipped) |
  | `ObjectFactories/tests` | 279 / 280 after | 279 | 280 | OK |
  | `RunDrivers/tests` | 115 | 115 | 115 | OK |
  | `tools/tests` | 91 | 91 | 91 | OK |

- **Golden fingerprint (rule 9):** `ObjectFactories.tests.test_store_fingerprint_golden`, 6 tests, all OK after the install,
  both in the root and alone: `test_the_config_names_the_fingerprinter`, `test_the_fingerprinter_takes_the_golden_fingerprint`,
  `test_the_new_golden_is_the_old_one_with_its_encoding`, `test_taken_is_outside_every_digest`, `test_the_golden_fingerprint`,
  `test_the_shape`. No record and no digest moved.
- **The addendum's claims (prompt §2.6), against what was measured:**
  - 21 `InventorySpec(...)` calls outside the guard module (`git grep -h`: 21 lines, `ObjectFactories/wavenumber.py` holds 2,
    the other 19 files one each): true. None passes `extra_values`: `git grep` for `extra_values`, `extra_value_counts`,
    `as_json`, `dataclasses.fields` and `asdict` over `*.py` outside `prompts/` and `docs/` finds the guard module alone:
    true.
  - The guard fails as the addendum says under `v0.5.0` unedited (21 classes "spec changed", message asking for a bump), and
    passes with the condition: true.
  - No claim found false.

## The deliberate-breakage record

Each applied to the edited guard module, the guard run with `-v`, then the module restored from a copy of the edited file
(`cmp` confirms identical; `git diff` shows the edited state only). Neither was committed.

**M1 — no condition.** `diff edited mutated`:

```diff
48d47
<             if not (f.name == "extra_values" and len(getattr(value, f.name)) == 0)
```

Result: FAILED (failures=2). `test_every_spec_is_pinned_for_the_current_format` fails with 21 lines `  spec changed: <class>`;
`test_an_empty_extra_values_is_left_out` fails on `assertNotIn`: `AssertionError: 'extra_values' unexpectedly found in {…
'parent_sets': {}, 'extra_values': {}}`; `test_a_key_keeps_its_type` passes. As predicted.

**M2 — the field always left out.** `diff edited mutated`:

```diff
48c48
<             if not (f.name == "extra_values" and len(getattr(value, f.name)) == 0)
---
>             if not (f.name == "extra_values")
```

Result: FAILED (failures=1). `test_an_empty_extra_values_is_left_out` fails on `assertIn`:
`AssertionError: 'extra_values' not found in {'leaves': [], … 'parent_sets': {}}` (line 116);
`test_every_spec_is_pinned_for_the_current_format` passes and `test_a_key_keeps_its_type` passes. As predicted: the new test
alone catches this mutation.

## Observations not acted on

- `pip` prints "A new release of pip is available: 26.2.1 -> 26.2.2" on each run. Not acted on.
- The condition names one field (README §6.3): the next defaulted field DatastoreKit adds to `InventorySpec` breaks the guard the
  same way. Left for the user, as the README says; no issue opened.

## State handed to the next prompt

- The venv holds `datastorekit` 0.5.0 at `f47b9a25…`; `raypoolkit` and `runregistrykit` unchanged. `requirements.txt:17` and
  `CLAUDE.md` say `v0.5.0`. There is no later prompt: the orchestrator's review closes the campaign.
- All seven roots pass (ObjectFactories at 280). `config/fingerprint.py` unchanged; `FINGERPRINT_FORMAT` stays 6.
  `docs/OPEN_ISSUES.md` unchanged (100 open).
- **The venv is ahead of `git revert`:** reverting 01 leaves `v0.5.0` installed (README §4.1).
