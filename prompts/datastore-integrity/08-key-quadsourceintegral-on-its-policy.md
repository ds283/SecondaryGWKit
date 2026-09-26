# Prompt 08 — `QuadSourceIntegral` is keyed on its `QuadSourcePolicy`

**Campaign:** [`README.md`](README.md) · **Board items:** **K3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 07 has landed; `git status` is clean. Decisions 6.1.2, D4, D5 and D6 are the
user's (README §6).
**Closes:** `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`, on the store-fingerprint
board's §4, with its index row deleted. **Opens:** anything out of scope that you find (§6),
**without fixing it**.
**Recommended model:** **Sonnet**. The change is mechanical and long: one new key column carried
through a compute target, its factory, `main.py`, an extraction script and their tests. The care is
in missing none of them.

**Read first:**

1. [`README.md`](README.md): §0, §1, §4, §5 (rules 7, 11, 12) and §6 — **6.1.2**, **D4**, **D5**,
   **D6**.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §2.5 and §2.6.
3. The store-fingerprint board's entry for `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`,
   with its **Assigned** line.
4. `MetadataConcepts/QuadSourcePolicy.py` and `Datastore/SQL/ObjectFactories/QuadSourcePolicy.py`;
   `MetadataConcepts/GkSourcePolicy.py` for comparison.
5. `ComputeTargets/QuadSourceIntegral.py`: the constructor and every place `policy` travels.
6. `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`: `register`, `build`, `store`,
   `read_batch` (as prompt 05 left it) and `inventory_records`.
7. `main.py`: where `QuadSource_policy_1pt5` and `_5pt0` are created and passed into
   `run_pipeline`, the `QuadSourceIntegral` existence query and the compute call. Find each by
   content; the audit's line numbers are `:981-982`, `:3061-3080`, `:3428-3440`, `:3672-3688`.
8. `extract_QuadSourceIntegral_data.py`, its policy lookup and its `read_batch` call.
9. [`logs/06-correct-the-foreign-keys.md`](logs/06-correct-the-foreign-keys.md) and
   [`logs/07-key-wkb-rows-on-their-numeric-row.md`](logs/07-key-wkb-rows-on-their-numeric-row.md):
   the D5 procedure as used twice.

---

## 1. What is wanted

`main.py` creates two `QuadSourcePolicy` rows and uses neither; `QuadSourceIntegral` keys on
`GkSourcePolicy` alone (audit §2.5). The user decided to keep `QuadSourcePolicy`, since it may gain
settings, and to treat it correctly: it joins `QuadSourceIntegral`'s key (6.1.2). Each integral is
keyed on **`QuadSource_policy_1pt5` only, for now** (D6), paired with the `GkSource_policy_1pt5`
every site already uses. No computation reads the policy's fields, so this changes the record and
no number.

---

## 2. What to change

**K3 — the key**, named `quad_policy_serial` (README §4):
- **Factory.** A column on `QuadSourceIntegral`, NOT NULL, indexed, with
  `sqla.ForeignKey("QuadSourcePolicy.serial")`. `QuadSourcePolicy` is replicated, so it is on every
  shard; since prompts 01–02 its serial is guaranteed the same on every shard. It joins `build()`'s
  filter, is written by `store()`, is selected, filtered on and passed back by `read_batch`, and is
  a key parent in `inventory_records` (`Parent("quad_policy_serial", "QuadSourcePolicy")`).
- **Compute target.** `QuadSourceIntegral`'s constructor takes the policy, required, beside the
  `GkSourcePolicy` it already takes; `store()` passes its serial. Choose the argument's name to sit
  beside the existing one without ambiguity (*prompt's choice*; say why).
- **`MetadataConcepts/QuadSourcePolicy.py`.** Its docstring says nothing reads the class. Amend it
  to say that `QuadSourceIntegral` is keyed on it, and that no computation reads its fields yet.
  Change nothing else in the class; its vocabulary is `handover` C2's
  (`[03-quadsource-policy-vocabulary-differs]`, out of scope).
- **`main.py`**, within README §5 rule 12: the existence query and the compute call pass
  `QuadSource_policy_1pt5`. `QuadSource_policy_5pt0` stays created and unused, as
  `[00-the-5pt0-policy-pair-is-created-and-never-used]` records; do not remove it. Run
  `substitute_main_py()` afterwards and show all five substitutions still match exactly once.
- **`extract_QuadSourceIntegral_data.py`.** It looks up the policy it reads under and passes it to
  `read_batch`. Look up `QuadSource_policy_1pt5`'s parameters as `main.py` creates them, and say in
  the log where the script now takes them from; do not hard-code a serial. `object_get` on a
  replicated class **inserts** a row when none matches, so a reader that looks a policy up can write
  to the store it reads. For `QuadSourcePolicy`, a store that lacks the policy is one the script
  cannot read: refuse, naming the parameters, rather than insert. The script's existing
  `GkSourcePolicy` lookup has the same behaviour; record it under "Observations not acted on" and
  open a §3 issue, but do not change it.
- **A store that predates the column** is refused with the message 07 established (D4).

**Existing tests that must change.** `ComputeTargets/tests/test_quadsource_integral.py` constructs
the target by keyword (the audit names `:1127`, `:2229`, `:2236`). With a required policy those
calls fail. README §5 rule 7 allows this edit because this prompt names it: add the policy argument
and change nothing else in those tests. List every call you changed in the log. Likewise the
store-inventory `IDENTITY` for `QuadSourceIntegral` and the real-store fixture's
`QuadSourceIntegral` row, which gain the new key.

**D5.** A new schema witness, `schema_at_datastore-integrity-08.json`, and a re-captured golden
fingerprint beside the old, each diff in the log. Neither old file overwritten.

---

## 3. Tests — beside the existing ones, no Ray, nothing under `var/`

1. **Keyed on the policy.** Two `QuadSourcePolicy` rows; a `QuadSourceIntegral` stored under the
   first is found under the first and missed under the second, all else equal.
2. **`read_batch` filters on it**, tagged and untagged.
3. **The constructor requires it.**
4. **The fingerprint's identity includes it**: two otherwise identical rows under the two policies
   are two records.
5. **`main.py`** passes `QuadSource_policy_1pt5` at both sites: an `ast` check, never an import.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) drop `quad_policy_serial` from `build()`'s filter;
- (ii) drop it from `read_batch`'s filter;
- (iii) pass `QuadSource_policy_5pt0` at the compute call in `main.py`;
- (iv) omit it from `inventory_records`.

Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. K3 in the tree; the tests of §3; no Ray, nothing under `var/`.
2. The witness diff and the fingerprint re-capture in the log.
3. `substitute_main_py()` shown to match all five substitutions exactly once.
4. Every existing test call changed, listed, each changed only by the added argument.
5. The deliberate-breakage record, (i)–(iv).
6. `[00-quadsourcepolicy-rows-are-referenced-by-nothing]` moved to the store-fingerprint board's §4
   with a closing note naming this commit and the user's decision; its index row deleted; the count
   and date corrected.
7. `black --check` clean. On the board: the §1 row, item K3, §3.1, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- Any code path computes something from a `QuadSourcePolicy`'s fields. The audit found none; if
  there is one, keying on it is no longer only a record, and D6 needs revisiting.
- `extract_QuadSourceIntegral_data.py` cannot find the policy's parameters without hard-coding them.
- A `substitute_main_py()` substitution stops matching exactly once.
- An existing test other than those named here would have to change.

---

## 6. What this prompt does not do

- It does not key `QuadSourceIntegral` on its source, policy data or Tk rows. That is 09.
- It does not remove or use the 5.0 policies, and does not change either policy class's
  vocabulary.
- It does not add `QuadSourcePolicy` to `QuadSource`'s or `OneLoopIntegral`'s key.

---

## 7. The log and the board

`logs/08-key-quadsourceintegral-on-its-policy.md`, using the template in README §5.1. In addition:
every site changed, in each file; the witness diff and fingerprint re-capture; the
`substitute_main_py()` output; where the extraction script takes the policy from.

`IMPLEMENTATION_STATE.md`: the §1 row for 08, item K3, §3.1 and §5.
