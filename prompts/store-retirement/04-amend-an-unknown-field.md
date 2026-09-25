# Prompt 04 — amend one unknown field of a sidecar, recording what it said before

**Campaign:** [`README.md`](README.md) · **Board item:** **R9** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 03 has landed. D5 was decided on 2026-09-25 (README §6.2).
**Closes:** nothing. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. The operation is small. The care is in making sure it can
touch nothing the registry interprets, and loses nothing it replaces.

**Read first:**

1. [`README.md`](README.md): §0, §4, §5, and §6.2 **D5** and **D6**.
2. [`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) §1 (the reference
   table) and §2.12.
3. Prompt 03's log, `logs/03-retire-a-store.md`: the history vocabulary and the tombstone
   refusals as shipped.
4. `RunRegistry/stores.py`, in full, and `RunRegistry/__main__.py`.
5. `RunRegistry/tests/store_fixtures.py` `a3_shaped`, which carries the live A3 sidecar's unknown
   fields, `backup` among them.

---

## 1. What is wanted

The registry carries every unknown field of a sidecar verbatim, and never interprets one
(`RunRegistry/stores.py` docstring). Some unknown fields make claims about the present. The live
A3 sidecar's `backup: {"path": …, "retained": true, …}` becomes false when prompt 05 retires the
backup (audit §1). What it should say then is a person's judgement. The registry's job (D5) is to
give that person a way to write it, **with no hand edit of a sidecar**, since only registry
operations write one (`datastore-portability` README §6.5 point 1). It must also make sure that
nothing is lost.

`amend_sidecar(primary, field, reason, *, value=…, remove=False) -> dict` and `python -m
RunRegistry store amend PRIMARY --field NAME (--json VALUE | --remove) --reason TEXT [--runs-root
DIR]` (README §4) replace or remove **one unknown field** of a registry sidecar, and append an
`amend` history entry that records the field, the reason, and the value before and after.

---

## 2. What to change

**2.1 The refusals.** Each is a `RuntimeError` naming the sidecar and the reason, and ending
"Nothing was written". Refuse:
- an empty or blank reason;
- both `value` and `remove`, or neither;
- a **known** field, any of `KNOWN_FIELDS`, `retired` included. The message names the operation
  that owns it: create, adopt, copy, move, fingerprint or retire;
- `remove` of a field that is absent;
- a `value` that is identical, after a JSON round trip, to the field's current value;
- a `value` that is not JSON-serialisable. On the command line, `--json` that does not parse;
- an absent, unreadable or legacy sidecar (create or adopt it first), or a registry sidecar with
  problems;
- a **tombstone**, complete or not (D6). A retired store's record is final;
- a store that a `running` run names, alive or stale. `Run.finish` may write the same sidecar, and
  two writers of one file must not race. This is the check copy, move, fingerprint and retire
  share.

Amend never opens the store's files. It reads and writes only the sidecar.

**2.2 The history entry.** `amend` becomes a history operation allowed after index 0 and before any
`retire`. Its entry has `HISTORY_KEYS` plus:
- `field`;
- `reason`;
- `before`: the old value, verbatim, or a marker that the field was absent;
- `after`: the new value, or a marker that it was removed.

Choose the markers so that no JSON value can be mistaken for one, and log the choice. `from` and
`to` have no paths to name for an amend. Decide what they hold, and make `_history_problems`
accept exactly that for `amend`, and require the extra keys for it and for nothing else. An
`amend` entry lacking any of its keys is a problem.

**2.3 The write.** Through `_update_sidecar`, the one in-place writer, which becomes its fifth
use, after adopt, the move's temporary sidecar, recording a fingerprint and a retirement's two
writes. Update the docstrings that count them: `_update_sidecar`'s own, and the `stores.py`
module docstring's list of in-place updates. *(Amended 2026-09-25, after prompt 03 landed: this
said "fourth", written before 03 added retirement's writes.)* Every other field must be
value-identical after the write. The fingerprint is not retaken, because the store's content has not changed.

**2.4 Copy and move** carry amended fields and `amend` entries like any others. Nothing in them
changes. A test shows it.

**2.5 The command line.** It prints the field, before and after, and the history entry. It exits
0 on an amendment and 1 on a refusal. `import RunRegistry` still loads neither `ray` nor
`sqlalchemy`.

**2.6 The docstrings.** Add `amend_sidecar` to the list of operations in the `stores.py` and
`__main__.py` docstrings. Say that it is the one way to change an unknown field, and that it never
touches a known one.

---

## 3. Tests — a new module in `RunRegistry/tests/`, no Ray, nothing under `var/`

1. **Replace** an unknown field of an `a3_shaped` sidecar.
   - The new value is in place.
   - The last history entry records `field`, `reason` and `before`, value-identical to the old
     value, nested structure included, and `after`.
   - Every other field is value-identical.
   - The store's files are unchanged by `tree_state`.
2. **Remove** an unknown field. The field is gone, and the entry records `before` and the removal
   marker.
3. **The live A3 shape.** Amend `backup`, from `"retained": true` to a value recording that the
   backup was retired, as prompt 05 will. Then `store show`. Both values are visible, the old one in
   history.
4. **Each refusal in §2.1**, the whole temporary tree unchanged. Every known field is covered, by
   iteration over `KNOWN_FIELDS`.
5. **The history rule.** An `amend` entry missing each of its keys in turn, an `amend` at index 0,
   and an `amend` after `retire` are each a problem. The extra keys on a `copy` entry are a problem.
6. **Copy and move** after an amendment carry the amended field and the `amend` entry.
7. **Two amendments of one field** leave two entries, each with its own `before`, so the full
   sequence of values can be read back from the history.
8. **The command line**: `--json`, `--remove`, bad JSON, and the exit codes.

**Deliberate breakage.** Show that each of these makes the tests written against it fail, then
restore it:
- (i) allow a known field;
- (ii) omit `before` from the entry;
- (iii) skip the running-run check;
- (iv) let the validator accept an `amend` with no `reason`;
- (v) allow an amendment of a tombstone.

Put each in the log as a short diff, **exactly as applied**, for `git apply`. Name the tests that
failed. Mutations are never committed.

---

## 4. Acceptance

1. §3's tests exist, need no Ray and open nothing under `var/`. Every existing test passes
   unmodified.
2. The deliberate-breakage record, (i)–(v), with diffs and the tests that failed.
3. The format table in the `stores.py` docstring includes `amend` and its keys, as shipped.
4. `black --check` clean. On the board: the §1 row for 04, item R9, and §5 baselines.
   `docs/OPEN_ISSUES.md` only if you open an issue. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- Amending an unknown field would need any known field to change, apart from `history`.
- An existing test would need to change.
- Any step would amend a sidecar under `var/`. That is prompt 05, and the live A3 sidecar is
  amended there by the user's command.

---

## 6. What this prompt does not do

- It does not amend any real sidecar.
- It does not change what `copy_store` carries (`[00-a-copy-carries-its-sources-present-tense-fields]`
  stays open), or touch any known field.
- It does not change `ShardedPool`, `retire_store`'s behaviour, or any run manifest.

---

## 7. The log and the board

`logs/04-amend-an-unknown-field.md`, using the template in README §5.1. In addition:
- the `amend` entry's shape, its markers and its `from` / `to` rule, as shipped;
- the deliberate-breakage record, with diffs.

`IMPLEMENTATION_STATE.md`: the §1 row for 04, item R9, and §5's baselines. Leave 05's held row as
it is.
