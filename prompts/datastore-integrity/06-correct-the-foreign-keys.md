# Prompt 06 — the foreign keys say what the columns hold, and a test holds them to it

**Campaign:** [`README.md`](README.md) · **Board items:** **S1**–**S5** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 03 has landed (S5 records what 03 made true); `git status` is clean. D5 is
decided (README §6.2).
**Closes:** `[00-quadsource-tq-serial-has-the-wrong-foreign-key]`, on the store-fingerprint board's
§4, with its index row deleted. **Opens:** anything out of scope that you find (§6), **without
fixing it**.
**Recommended model:** **Sonnet**. The edits are small. This is the campaign's **first deliberate
schema change**, so it also establishes the D5 procedure every later schema-changing prompt
follows; getting that procedure right matters more than the edits.

**Read first:**

1. [`README.md`](README.md): §0, §1, §4, §5 and §6.2 **D5** in full.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §1 (foreign keys
   are never enforced) and §2.1, §2.2, §2.6.
3. `docs/datastore-integrity-audit/fk_sweep_probe.py` and `tq_fk_probe.py`: the sweep and the
   enforcement probe this prompt turns into a test.
4. `Datastore/SQL/ObjectFactories/QuadSource.py` (`register`), `GkSourcePolicyData.py`
   (`register`), `wavenumber.py` (`sqla_wavenumber_exit_time_factory.register`) and
   `integration_metadata.py` (`register`); `ComputeTargets/QuadSource.py` around `Tq_serial` /
   `Tr_serial`.
5. `Datastore/SQL/schema.py` `build_schema`, including how `stepping_mode` is read.
6. `Datastore/tests/test_schema_builder.py`, `Datastore/tests/schema_description.py` and
   `Datastore/tests/data/schema_at_base.json`: the witness, and how it was captured.
7. `Datastore/tests/real_store_fixtures.py`, its undeclared-reference map and `references()`, and
   `RunRegistry/tests/test_store_fingerprint.py` with its golden
   `RunRegistry/tests/data/full_store_fingerprint.json`.
8. [`logs/03-exact-identity-lookups.md`](logs/03-exact-identity-lookups.md), "Observations not acted
   on": the two `stepping_mode` flags it left for this prompt.

---

## 1. What is wanted

Nothing enforces a foreign key in this datastore, so the schema's foreign keys are documentation,
and two of them are wrong (audit §2.1–§2.2):
- `QuadSource.Tq_serial` names `QuadSource.serial`, but holds a `TkNumericIntegration` serial;
- `GkSourcePolicyData.wavenumber_exit_serial` has none, though its referent is replicated.

Two registrations also say something prompt 03 made false: the exit time and `IntegrationSolver`
register `"stepping": "minimum"`, and both now match stepping exactly.

This prompt corrects all four, and adds the test that would have caught the first: a store written
the way production writes it must pass SQLite's own foreign-key check. It is also the first prompt
to change the schema on purpose, so it sets up the D5 witness procedure.

---

## 2. What to change

**S1 — `QuadSource`.**
- `Tq_serial`: `sqla.ForeignKey("TkNumericIntegration.serial")`. `Tq` is the Tk numeric row at
  `k = q`, and `QuadSource` shards on `q`, so it is always on the same shard (audit §2.1). Confirm
  that from `config/sharding.py` and `main.py` before you rely on it, and quote the evidence in the
  log.
- `Tr_serial`: no foreign key, as today. `Tr` is on `r`'s shard. Correct the comment beside it,
  which names `QuadSource.serial`, to name `TkNumericIntegration` and give the cross-shard reason.

**S2 — `GkSourcePolicyData.wavenumber_exit_serial`.** Add
`sqla.ForeignKey("wavenumber_exit_time.serial")`. The exit time is replicated, so it is on every
shard.

**S3 — the fixtures' map.** `real_store_fixtures.py`'s undeclared-reference map says
`{"Tr_serial": "QuadSource"}`. Correct it to `TkNumericIntegration`. Remove any entry S1 or S2
turns into a declared foreign key, since `references()` then finds it from the schema. README §5
rule 7 allows this edit to a test fixture because this prompt names it: the map describes the
schema, and the schema changes.

**S4 — the check.** A new test, no Ray, nothing under `var/`: every shard of a store written by the
real factories passes `PRAGMA foreign_key_check` with no row. `foreign_key_check` works whether or
not enforcement is on, so nothing about production's connections changes.
- Use the real-store fixture (`real_store_fixtures.py`), and also a store written through the real
  write path with the stand-in shards prompts 01–02 built, if their test helpers make that
  possible without copying them (*prompt's choice*; say which you used).
- Show the test fails on the tree before S1 (the `Tq_serial` row is a violation; `tq_fk_probe.py`
  shows it with enforcement on). That is this prompt's deliberate-breakage baseline.
- Cross-shard references (`Tr_serial`, `QuadSourceIntegral.source_serial`, `shard_keys`) have no
  declared foreign key and so are not checked. Say so in the test's docstring, naming each.

**S5 — the witness, under D5.**
- The exit-time and `IntegrationSolver` registrations become `"stepping": "exact"`, which
  `build_schema` accepts (`schema.py`). Check first that nothing reads `stepping_mode` but the
  schema description; if something does, stop (§5).
- Capture a **new** witness, `Datastore/tests/data/schema_at_datastore-integrity-06.json`, with the
  same `describe_schema` / `dumps` that captured `schema_at_base.json`. **Never overwrite
  `schema_at_base.json`.**
- Amend `test_schema_builder.py` so that its comparison is against the witness for the **current**
  schema design, named in one constant, and so that its docstring says so: `schema_at_base.json`
  witnessed a refactor that changed nothing; each deliberate schema change adds a witness beside it
  and repoints the constant; no witness is ever regenerated to make a test pass. README §5 rule 7
  allows this edit because D5 names it.
- Put the diff between the two witnesses in the log, **in full**. It is the reviewable statement of
  what this prompt changed in the schema, and must show exactly S1, S2 and the two `stepping_mode`
  values, and nothing else.
- **The golden fingerprint.** Run `RunRegistry/tests/test_store_fingerprint.py`. None of S1–S5 is
  expected to move it: a foreign key is not fingerprint identity. If it moves, stop (§5): the
  fingerprint would then depend on something this prompt did not think it did.

---

## 3. Verification

1. S4's test fails on the tree before S1, and passes after.
2. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) point `Tq_serial` back at `QuadSource.serial`;
   - (ii) give `Tr_serial` a foreign key to `TkNumericIntegration.serial` — S4 must fail on the
     fixture shard where a `QuadSource` has `q ≠ r`, because `Tr` is not there. If it does not, the
     fixture has no cross-shard pair, and adding one to it is part of S4;
   - (iii) repoint the witness constant at `schema_at_base.json`;
   - (iv) edit one byte of the new witness.

   Each in the log as a diff, exactly as applied, with the failing tests.
3. Every suite unchanged, except the tests this prompt names and adds.

---

## 4. Acceptance

1. S1–S5 in the tree. S4's test in a new module; no Ray, nothing under `var/`.
2. The two witnesses side by side, and their diff in the log, showing exactly S1, S2 and the two
   flags.
3. The golden fingerprint unchanged, with the test run quoted.
4. The deliberate-breakage record, (i)–(iv).
5. `[00-quadsource-tq-serial-has-the-wrong-foreign-key]` moved to the store-fingerprint board's §4
   with a closing note naming this commit; its index row deleted; the count and date corrected.
6. `black --check` clean. On the board: the §1 row, items S1–S5, §3.1, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- `Tq` is not guaranteed to be on `QuadSource`'s shard.
- Anything but the schema description reads `stepping_mode`.
- The golden fingerprint moves.
- S4 finds a violation that is not one of S1–S2: that is a new wrong foreign key, and whether the
  key or the data is wrong is not this prompt's to decide. Record it and stop.
- Any other existing test would have to change.

---

## 6. What this prompt does not do

- It does not add any column, or change any lookup key. Those are 07–09.
- It does not turn foreign-key enforcement on for production connections.
- It does not touch `_drop_order`; that follows the first foreign key that constrains a drop, in 07.

---

## 7. The log and the board

`logs/06-correct-the-foreign-keys.md`, using the template in README §5.1. In addition: the evidence
that `Tq` shares `QuadSource`'s shard; the witness diff in full; the fingerprint test run; which
store S4 checked, and what it cannot see.

`IMPLEMENTATION_STATE.md`: the §1 row for 06, items S1–S5, §3.1 and §5.
