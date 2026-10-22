# Prompt 01 — pin `datastorekit` at `v0.5.0`

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**, **G1**, **B1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-22, with the plan, at `031bcb4`.
**Gate:**
- the planning commit has landed, and `git status` is clean;
- no commit after the planning commit touches code (anything outside `prompts/` and `docs/`);
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
- the venv holds `datastorekit` 0.4.0 at `c8570e849334d09085bbd84876095fc9e22bd005`,
  `raypoolkit` 0.2.0 at `eaef0ce60e2fc354bb32f0ad576010e6a83c1976` and `runregistrykit` 0.2.0 at
  `0bae3a3c501a8264ee4be5bc24f06422af38720f`, none editable;
- `git worktree list` shows the main checkout alone.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One line of `requirements.txt`, one condition and one test in a
test module, one sentence of `CLAUDE.md`, one install, the baselines and the roots.

**Read first:**

1. `CLAUDE.md`, "The datastore layer is `datastorekit`; …" (a pin is a tag, in a commit of its own;
   a package's defect is never patched or worked around here).
2. [`README.md`](README.md): §0, §0.1 and §0.2 (what changes, and what was measured), §2.2 (the
   charter), §4.1 (rollback), §5 (rules 7–13), §6.2 (C2–C5) and §8.
3. **DatastoreKit at the tag** (`/Users/ds283/Documents/Code/DatastoreKit`), read with
   `git -C <DK> show v0.5.0:<path>`:
   - `docs/adoption/secondarygwkit.md`, the section "Addendum: v0.5.0 (extra-values prompt 02,
     2026-10-21)", "The pin" and "What a move to `v0.5.0` changes for SGK";
   - `docs/client-contract.md` §9.8.
4. **RunRegistryKit's amended recipe** (`/Users/ds283/Documents/Code/RunRegistryKit`), at commit
   `70ee1e8`: `git -C <RRK> show 70ee1e8:docs/adoption/README.md`, the section "The Q5 guard under
   DatastoreKit `v0.5.0`". Its `plain` is the one this prompt writes.
5. `ObjectFactories/tests/test_fingerprint_format_guard.py` whole, and the comments on
   `FINGERPRINT_FORMAT` and `SPEC_DIGESTS` in `config/fingerprint.py`.
6. `rrk-v020-adoption`'s
   [`01-pin-runregistrykit-at-v0.2.0.md`](../rrk-v020-adoption/01-pin-runregistrykit-at-v0.2.0.md)
   and its log. This prompt is that one again for another package. Where this prompt is silent,
   that one governs.

---

## 1. What is wanted

`v0.5.0` adds `InventorySpec.extra_values`, which SGK does not use (README §0). **This prompt
installs it, makes the spec guard ignore the new field while it is empty, corrects the sentence of
`CLAUDE.md` that names the pin, and nothing else.**

The package is imported everywhere, so the install takes effect at once, before the commit. That is
why §2.1 takes the baselines **before** the install. Between the install (§2.3) and the guard edit
(§2.4) the ObjectFactories root is red by design; §2.4 shows that, then fixes it.

## 2. What to do

**2.1 The baselines, before.** Save each under `measure/out/`, every block headed by its command:
- `./venv/bin/pip freeze`, and
  `runregistrykit.package_provenance(("datastorekit", "raypoolkit", "runregistrykit"))`, printed as
  JSON, into `01-pip.txt`;
- the seven roots, by README §8's command, serially, in the foreground, into `01-roots.txt`. Expect
  §8's counts, all OK.

**2.2 The pin.** `requirements.txt:17` changes from

```text
datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.4.0
```

to

```text
datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.5.0
```

No other line of the file moves.

**2.3 The dry run, then the install.**
- `./venv/bin/pip install --dry-run -r requirements.txt`, saved whole in `01-pip.txt`. It must say
  `Would install datastorekit-0.5.0` and nothing else. The orchestrator measured exactly that on
  2026-10-22 (README §0.2).
- Then `./venv/bin/pip install -r requirements.txt`, saved whole.
- After it:
  - `pip freeze` again, whose diff against 2.1's must be the one `datastorekit` line, naming
    commit `f47b9a255e9ad58e0d0eaf85644f53b575884d31`;
  - `pip check`;
  - `package_provenance(...)` again. `datastorekit` must be 0.5.0 at `f47b9a25…`,
    `requested_revision` `v0.5.0` and `editable` `false`, and the other two entries must equal
    2.1's.
- **What was installed**, read:
  - `./venv/bin/pip show datastorekit`;
  - the distribution's `direct_url.json` (`vcs_info.commit_id` `f47b9a25…`, no
    `dir_info.editable`);
  - `importlib.metadata.version('datastorekit')`, which is `0.5.0`;
  - `./venv/bin/python -c "import dataclasses; from datastorekit.store_inventory import InventorySpec; from datastorekit.store_fingerprint import ENCODING_FORMAT; print([f.name for f in dataclasses.fields(InventorySpec)], ENCODING_FORMAT)"`.
    Seven fields, the last `extra_values`; `ENCODING_FORMAT` is `1`.

**2.4 The guard.**
- **First, unedited**, run
  `PYTHONPATH=. ./venv/bin/python -m unittest ObjectFactories.tests.test_fingerprint_format_guard`
  and save it into `01-guard.txt`. It must fail in `test_every_spec_is_pinned_for_the_current_format`
  with one "spec changed" line for each of the 21 classes of `SPEC_DIGESTS[6]`, and no "added" or
  "removed" line. Record the count. Any other shape of failure stops the prompt (§4).
- **The condition.** In `plain`, the dataclass branch keeps only the fields that are not an empty
  `extra_values`:

  ```python
  if dataclasses.is_dataclass(value):
      return {
          f.name: plain(getattr(value, f.name))
          for f in dataclasses.fields(value)
          if not (f.name == "extra_values" and len(getattr(value, f.name)) == 0)
      }
  ```

  The rest of `plain`, `spec_digests` and both existing tests are unchanged.
- **`plain`'s docstring** gains one sentence: an empty `extra_values` is left out, so that a spec
  declaring no extra table is what it was before DatastoreKit `v0.5.0` added the field.
- **The module docstring** gains a short paragraph after the one naming RunRegistryKit's recipe:
  since DatastoreKit `v0.5.0` the recipe carries one amendment, the condition above, given in that
  file's section "The Q5 guard under DatastoreKit `v0.5.0`" (RunRegistryKit `70ee1e8`); with it
  every pin in `SPEC_DIGESTS[6]` holds under `v0.5.0`, so the field is not a format change, while
  a spec that comes to declare an extra table moves its digest and is one. Name this campaign and
  prompt, in the form the docstring already uses.
- **The new test**, in `TestFormatGuard`, after `test_a_key_keeps_its_type`:

  ```python
  def test_an_empty_extra_values_is_left_out(self):
      empty = plain(InventorySpec())
      declared = plain(InventorySpec(extra_values={"t": "c"}))
      self.assertNotIn("extra_values", empty)
      self.assertIn("extra_values", declared)
      self.assertNotEqual(canonical_json(empty), canonical_json(declared))
  ```

  with `InventorySpec` imported beside `canonical_json` and `inventory_specs`. It touches no
  factory and no store. The orchestrator checked on 2026-10-22, under `v0.5.0`, that
  `InventorySpec()` takes no argument and that the condition gives exactly these three results.
- **Then, edited**, run the module again into `01-guard.txt`: three tests, OK.
- **`SPEC_DIGESTS` and `FINGERPRINT_FORMAT` do not change** (README §6.2 C3). If the edited guard
  does not pass against them as they stand, stop (§4): do not regenerate a pin.

**2.5 `CLAUDE.md`.** Its one statement of the pin, which the commit makes false (README §6.2 C4):
`:51`, "pinned in `requirements.txt` at `v0.4.0`" → "pinned in `requirements.txt` at `v0.5.0`".
Add one clause, in the form the RunRegistryKit sentence of the same section uses: "moved to
`v0.5.0` on <date> in the `dk-v050-adoption` campaign". Rewrap the paragraph to ~100 columns as
the file has it. Nothing else in `CLAUDE.md` changes.

**2.6 The baselines, after.**
- **The roots again**, into `01-roots.txt` beside 2.1's: §8's counts, with ObjectFactories one
  more (280) for the new test, all OK. `test_store_fingerprint_golden` passes among them: that is
  rule 9's check that no record and no digest moved. Name its tests' results in the log.
- **The addendum's claims.** Check the addendum's section "What a move to `v0.5.0` changes for
  SGK" against what you measured: the 21 `InventorySpec(...)` calls, none passing `extra_values`;
  no SGK file reading `extra_values`, `extra_value_counts` or `as_json`; the guard failing as it
  says, and passing with the condition. Record any claim that is false.

## 3. Acceptance

1. **The dry run adds only `datastorekit`.** After install it is 0.5.0 at `f47b9a25…`,
   `requested_revision` `v0.5.0`, and not editable. `raypoolkit` and `runregistrykit` are
   unchanged; `pip check` is clean; `pip freeze`'s diff is the one line.
2. **The guard fails unedited under `v0.5.0`, naming all 21 classes and nothing else, and passes
   edited**, with `config/fingerprint.py` unchanged.
3. **The new test passes, and catches both mutations of §3.1.**
4. **The seven roots pass at §8's counts before, and after with ObjectFactories at 280.** A failure
   among README §8's four known wall-clock tests is re-run alone and recorded. Any other failure,
   or any count that differs, stops the prompt (§4).
5. `requirements.txt` differs from `HEAD` by the one line, `CLAUDE.md` by §2.5's sentence, and
   `ObjectFactories/tests/test_fingerprint_format_guard.py` by §2.4's condition, docstrings, import
   and test. Nothing under `var/` is changed.

### 3.1 Deliberate breakage

Apply each, run the guard module, record the result and the diff exactly as applied, and revert:
- **M1 — no condition.** Restore `plain`'s dataclass branch to walking every field.
  `test_every_spec_is_pinned_for_the_current_format` must fail naming all 21 classes, and
  `test_an_empty_extra_values_is_left_out` must fail on `assertNotIn`.
- **M2 — the field always left out.** Drop `and len(getattr(value, f.name)) == 0` from the
  condition. `test_an_empty_extra_values_is_left_out` must fail on `assertIn`;
  `test_every_spec_is_pinned_for_the_current_format` passes, which is why the new test exists
  (README §6.2 C5).

## 4. Stop conditions — stop and ask

- The dry run would install, upgrade or remove anything but `datastorekit`.
- After install `datastorekit` is not at `f47b9a25…`, or is editable, or another package moved.
- The unedited guard fails other than by "spec changed" for exactly the 21 classes, or passes.
- The edited guard fails against `SPEC_DIGESTS[6]` as it stands.
- A root fails other than by README §8's known tests or (between §2.3 and §2.4 only) the guard; or
  a count differs from §8's; or a golden-fingerprint test fails.
- Anything suggests a record or a fingerprint of SGK's specs moved.

**If you stop after the install**, say so: the venv then holds `v0.5.0` while `requirements.txt`
may not be committed. Do not reinstall `v0.4.0` yourself; the orchestrator decides (README §4.1).

## 5. What this prompt does not do

- It changes no code outside the guard module, and nothing in `config/fingerprint.py`.
- It runs no build, starts no Ray, and does not take the rehearsal lock.
- It opens no store and runs no `runregistrykit store` command.
- It writes nothing under `var/`.
- It does not close the campaign: the review does (README §2.2).

## 6. The log and the board

**The files you may change, and nothing else:**
- `requirements.txt` (the one line), `CLAUDE.md` (§2.5) and
  `ObjectFactories/tests/test_fingerprint_format_guard.py` (§2.4);
- under `measure/out/`, `01-pip.txt`, `01-guard.txt` and `01-roots.txt`;
- the log, this campaign's board, and `docs/OPEN_ISSUES.md` only if §3 of the board gains an entry.

**The log**, at `logs/01-pin-datastorekit-at-v0.5.0.md`, in README §5.1's form.

**The board:**
- the header ("01 landed, awaiting review");
- the §1 row for 01, and items K1, G1 and B1;
- `### 5.1 After prompt 01 (<date>)`.

Run `./venv/bin/python -m black` on the guard module before committing.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
