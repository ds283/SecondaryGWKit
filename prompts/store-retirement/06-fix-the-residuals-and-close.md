# Prompt 06 — fix the three residuals in this campaign's own code, and leave it ready to close

**Campaign:** [`README.md`](README.md) · **Board items:** **R13**–**R16** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompts 01–05 have landed and been reviewed (05 in `0075f72`). The user decided on
2026-09-26 that the campaign fixes these three and then closes (README §6.4).
**Closes:** `[03-the-package-docstring-still-says-the-registry-deletes-nothing]`,
`[04-amend-calls-true-1-and-1-0-identical]` and `[04-no-test-amends-a-value-shaped-like-the-marker]`,
on the board's §4. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. The changes are one docstring paragraph, one comparison and
one test module. The care is in changing *only* the refusal decision, and in a marker test that
would fail if the markers were ever flattened.

**Written** 2026-09-26 at `0075f72`, against what 01–05 shipped. Where this text and the code or a
log disagree, the code and the log are right. Stop and say so.

**Read first:**

1. [`README.md`](README.md): §4 (prompt 04's names), §5, §6.2 **D0** and **D5**, and §6.4.
2. The board's §3 entries for the three issues above, and the orchestrator's review of prompt 04
   (`IMPLEMENTATION_STATE.md` §1), which opened two of them and holds the probes.
3. Prompt 04's log, `logs/04-amend-an-unknown-field.md`: "What shipped", the markers, and the
   deliberate-breakage record.
4. `RunRegistry/__init__.py`, the module docstring. `RunRegistry/stores.py`: `_amend_slot`,
   `_amend_slot_problems`, `_history_problems` and `amend_sidecar`, in full.
5. `RunRegistry/tests/test_store_amend.py`, in full: `AmendTestCase` and its helpers
   (`amend`, `refused_amend`, `sidecar`, `assertOnlySidecarChanged`), and
   `test_a_value_identical_after_a_json_round_trip`, which must still pass unmodified.

---

## 0. The rules this prompt runs under

README §5, unchanged. In particular:
- **Nothing under `var/` is opened**, not even read-only. Every test builds its stores in a
  temporary directory (rule 9). The orchestrator checks the real stores' end state itself.
- **Existing tests are not modified** (rule 7). The new tests go in a **new module**, so that
  `test_store_amend.py` stays byte-identical.
- **One commit**, holding the code, the tests, the log and the records.

---

## 1. What is wanted

Three things, each already specified by the issue it closes:

- **R13, the charter.** The `RunRegistry/__init__.py` module docstring says, twice, that the
  package "deletes nothing". Since prompt 03, `store retire` deletes a store's own files. D0's
  wording is what `CLAUDE.md` and the `stores.py` and `__main__.py` docstrings now say, and the
  package docstring must say it too.
- **R14, the guard.** `amend_sidecar`'s ninth refusal, a value "identical, after a JSON round
  trip" to the current one, tests `fields[field] == normalised` (`RunRegistry/stores.py:2110`). In
  Python, `True == 1 == 1.0` at any depth of nesting, so a real change is refused. The refusal
  must fire exactly when the field's JSON would not change, and at no other time.
- **R15, the test.** No committed test amends a field whose value has the marker's own shape. The
  wrapper design is right, and a probe showed it (review of 04). What is missing is a test that
  would fail if a later change flattened the markers.

Then **R16, the records.** The three issues move to §4, and the index loses their rows. The board
says that the campaign's work is done. **Closing the campaign is not this prompt's**: the
orchestrator records the closure, the user's decision of 2026-09-26, after its review (§4.4).

---

## 2. What to change

**2.1 The docstring (R13).** In `RunRegistry/__init__.py`, the module docstring's first paragraph.
Replace, from "It is a convention" to the end of the paragraph, with exactly this text, re-wrapped
to the file's width:

> It is a convention with a little code behind it, not a framework: it schedules nothing,
> supervises nothing and locks nothing. It also creates, adopts, copies and moves the stores it
> manages, with their `<stem>.manifest.json` sidecars (`RunRegistry.stores`), because the user
> decided that managing stores is part of managing the registry (`prompts/datastore-portability`
> README §6.5). It deletes nothing but a store's own files, and those only through `store retire`,
> which a person runs and which leaves the store's sidecar behind as its record
> (`prompts/store-retirement` README §6.2, D0). It never deletes a run directory, a sidecar or any
> other record.

Nothing else in the file changes. `list_runs`'s "Nothing here writes or deletes anything" is true
of that function, and stays.

**2.2 The comparison (R14).** In `amend_sidecar`, compare the current and the new value by their
**canonical JSON text**: `json.dumps(…, sort_keys=True)` of each side. Put that in a small private
helper or inline it; the choice is yours, and you log it.

Why this form, and not another:
- `sort_keys=True` is what the sidecar writer uses (`RunRegistry.write_json_atomic`,
  `json.dump(payload, handle, indent=2, sort_keys=True)`). So two values with the same canonical
  text write the same bytes for that field. Refusing exactly those is "nothing would change",
  which is what the refusal says.
- An object with its keys in another order is therefore **still refused**. `true`, `1` and `1.0`,
  and `-0.0` and `0.0`, are **accepted**, because their texts differ.
- `NaN`, which the writer emits, now compares equal to itself, so amending `NaN` to `NaN` is
  refused. Under `==` it was accepted, as a change that changed nothing. Say so in the log. No
  test is required for it.

Keep everything else about the refusal:
- It stays ninth, after the JSON-serialisability check.
- Its message still contains the fragment `identical, after a JSON round trip`, which the existing
  test asserts. You may add a clause saying how the comparison is made.
- **What is written does not change.** `new_value` is still `normalised`, and every other line of
  `amend_sidecar` is untouched. Only the decision to refuse moves.

Update the docstring's item 9 to say how the comparison is made. Change no other refusal, no other
function, and no other `==` in `stores.py`. If you find another place where JSON values are
compared with `==` and the difference could matter, record it (§6); do not change it.

**2.3 Nothing else.** `_amend_slot`, `_amend_slot_problems`, `_history_problems`, the command line
and every other operation stay exactly as they are.

---

## 3. Tests — a new module, `RunRegistry/tests/test_store_amend_residuals.py`

No Ray, nothing under `var/`, stores from the same placeholder fixture as `test_store_amend.py`.
Import `AmendTestCase`, `REASON` and whatever else you need from `RunRegistry.tests.test_store_amend`,
rather than copying them. Every call passes `runs_root`, and every command passes `--runs-root`,
as `AmendTestCase` does.

**R14, the comparison.**
1. **Accepted, with the type kept.** A field holding `1` amended to `True`; `1` to `1.0`; `True`
   to `1`; `{"k": 1}` to `{"k": True}`; `[1]` to `[True]`. For each:
   - the call returns, and `assertOnlySidecarChanged` holds;
   - the entry's `before` and `after` markers hold the old and the new value, **with their types**:
     `assertIs(type(…), bool)`, `float` or `int` as appropriate. `assertEqual` alone cannot tell
     `1` from `True`;
   - the sidecar's text on disk has `true` or `1.0` where the new value is.
2. **Still refused, with the tree unchanged** (`refused_amend`, fragment
   `identical, after a JSON round trip`): `1` to `1`; `{"a": 1, "b": 2}` to `{"b": 2, "a": 1}`.
3. **The command line.** `store amend … --field n --json true` on a field holding `1` exits 0 and
   prints `>> amended:`. The same command run again exits 1 and names the identical-value refusal.

**R15, the marker-shaped value.** One field, `m`, taken through a whole sequence:
4. **Add** `m` with the value `{"present": False}`. The entry's `before` is `{"present": False}`,
   because `m` was absent, and its `after` is `{"present": True, "value": {"present": False}}`.
5. **Replace** it with `{"present": True, "value": 1}`. `before` is exactly
   `{"present": True, "value": {"present": False}}`, and `after` is exactly
   `{"present": True, "value": {"present": True, "value": 1}}`.
6. **Remove** it. `before` is exactly `{"present": True, "value": {"present": True, "value": 1}}`,
   and `after` is `{"present": False}`.
7. **Read back.** The sidecar reads `ok`, with no problems, and `stores._history_problems` of its
   history is empty. A small walk over the three `amend` entries, which unwraps each marker one
   level, reconstructs the sequence *absent → `{"present": false}` →
   `{"present": true, "value": 1}` → absent*, and nothing else. `store show` on the store exits 0,
   with `problems: none`.

Assert each marker by `assertEqual` of the whole marker, so that a marker which lost its wrapper,
or gained a second, fails.

**Deliberate breakage.** Show that each makes the tests named against it fail, then restore it:
- (i) restore the `==` comparison. Tests 1 and 3 fail.
- (ii) compare `json.dumps(…)` without `sort_keys`. The reordered case in test 2 fails.
- (iii) make `_amend_slot` return the bare value, not the wrapper, when the field is present.
  Tests 5–7 fail. Name every other test, in this module and in `test_store_amend.py`, that fails
  too, and say why.

Put each in the log as a short diff, **exactly as applied**, for `git apply`. Name the tests that
failed, with counts. Mutations are never committed.

R13 has no mutation: it is prose. Quote the paragraph before and after in the log, and show
`grep -n "deletes nothing" RunRegistry/__init__.py` returning nothing.

---

## 4. Acceptance, and the records

1. **The tests.** §3's module exists, and its tests pass twice. Every existing test passes
   unmodified. `git diff` of `RunRegistry/tests/test_store_amend.py` is empty.
2. **The suites.** All six, from the repository root, against README §7's command:
   - `RunRegistry` is 190 plus your new tests;
   - the other five are at the orchestrator's pre-dispatch baselines.

   `black --check` is clean on every file you touched.
3. **The deliberate-breakage record**, (i)–(iii), with diffs and the tests that failed.
4. **The records (R16), in the same commit.** They are additive: nothing already on a board is
   rewritten, except the rows and statuses that this prompt's landing changes.
   - **This board, §3 to §4.** Move the three entries to §4, each keeping its text, its
     `**Assigned (2026-09-26):**` line, and a new `**Closed (date) by prompt 06.**` paragraph. That
     paragraph says what changed, which tests show it, and which mutation catches it. Follow
     `prompts/store-fingerprint/IMPLEMENTATION_STATE.md` §4 for the shape. Correct §3's opening
     paragraph, which counts the issues, and replace §4's "None yet."
   - **This board, the rest.** The §1 row for 06, items R13–R16, and a §5 "After prompt 06"
     column. The header's status is **"6 of 6 prompts landed. The campaign's work is done; it is
     closed by the user after the orchestrator's review of 06."** Do not write that it is closed.
   - **`docs/OPEN_ISSUES.md`.** Delete the three rows from §1.14, and add one sentence to §1.14's
     paragraph saying that prompt 06 closed them on this board's §4. Correct the count, 104 to 101,
     and set **Last updated** to the commit's date.
5. **The commit.** One, in `CLAUDE.md`'s form, staged by name, never with `git add -A`. For
   example: "Fix the three residuals of the store retirement campaign". The tree is clean
   afterwards, apart from untracked paths that are not this campaign's.

---

## 5. Stop conditions — stop and ask

- An existing test fails, or would pass only if it were modified.
- The comparison cannot be changed without changing what `amend_sidecar` writes, or any refusal
  but the ninth, or the message fragment the existing test asserts.
- A marker-shaped value does **not** read back unambiguously on the shipped code. That would mean
  the review of 04 was wrong, and the design is at fault, not just the test.
- Any step would need to open anything under `var/`.
- This prompt and a log disagree about a name, a field or a behaviour.

---

## 6. What this prompt does not do

- It does not touch the other four open issues:
  `[00-a-sigterm-pipeline-run-is-recorded-as-failed]`,
  `[00-a-launch-log-lives-outside-its-run-directory]`,
  `[00-a-copy-carries-its-sources-present-tense-fields]` and
  `[01-cross-filesystem-move-advice-says-delete-by-hand]`. They stay open, unassigned, in §3 and in
  the index, for their owners.
- It changes no other docstring, and no `CLAUDE.md` text. D0's wording is already in `CLAUDE.md`,
  and in the `stores.py` and `__main__.py` docstrings.
- It does not touch `ShardedPool`, `tools/`, `quadsource_atol_sweep.py`, the resolver's `!!`
  notice (log 05, "Observations" item 1), or anything under `var/`.
- It does not write the closure. The README's closure line, the board's "closed" status and the
  index's closure sentence are written by the orchestrator after its review, as for
  `store-fingerprint` (`42d4910`).
- Anything else you find is recorded in the log's "Observations not acted on", and opened in §3 if
  it is a defect, **without fixing it**.

---

## 7. The log

`logs/06-fix-the-residuals-and-close.md`, using README §5.1's template. In addition, it records:
- the docstring paragraph before and after, and the empty `grep`;
- the comparison before and after, as a diff, and where the helper lives if you made one;
- the `NaN` behaviour change (§2.2);
- the new tests, by name, against §3's numbers;
- the deliberate-breakage record;
- the six suites' counts, beside the pre-dispatch baselines;
- the records changed, with the index's count before and after.
