# Prompt 05c — value reads key on their parent's full key, and refuse a second parent

**Campaign:** [`README.md`](README.md) · **Board items:** **P3**, **P4**, **P5**, and the close of
**P9** · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 05b landed and reviewed; `git status` and `git diff --cached` clean. This is the last of
the three prompts that make prompt 05. After it lands and is reviewed, 07 may follow once 06 has
also landed.
**Closes:** `[00-numeric-value-parent-lookup-omits-break-point-kind]` on the store-fingerprint
board's §4, with its index row deleted.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. Two of the read paths changed here are **live**:
`GkNumericValue.read_batch` and `GkWKBValue.read_batch`, both at `main.py`'s `GkSource` stage. The
change gives them a refusal they do not have today. It must refuse exactly when two parents could
be confused, and nowhere else.

**Read first:**

1. [`README.md`](README.md): §0.1, §2 (the 05 and 07 rows), §5, D1 and D2. D1's note on readers
   that hold only physical labels is the rule applied here: select by them, and **refuse** when
   more than one stored parent matches, naming the parents.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §3.3, §3.4, §3.11
   and §3.13.
3. `docs/datastore-integrity-audit/lookup_keys/probe_05_value_lookups_and_read_batch.py` and its
   `.out`: every case below, measured on the audit's tree. Do not edit them.
4. The parents' own lookups, the key you are matching: `sqla_TkNumericIntegration_factory.build`,
   `sqla_GkNumericIntegration_factory.build`, `sqla_GkWKBIntegration_factory.build` and
   `sqla_TkWKBIntegration_factory.build`. Then the value factories' `build`, `_build_impl_model`
   and `read_batch` in the same four files.
5. `main.py`'s `build_missing_GkSource`, which reads both live paths. Find it by name. It passes
   `model`, `z` (the response) and tags, the tolerance pair for `GkNumericValue` only, and **no**
   `z_source`. It then files each returned value under `_z_source.store_id`, so a second parent
   for one `z_source` silently overwrites the first.
6. The issue's entry on `prompts/store-fingerprint/IMPLEMENTATION_STATE.md`.
7. `07-key-wkb-rows-on-their-numeric-row.md` §1–§2, and `09-key-derived-rows-on-their-parents.md`
   K6. 07 replaces `z_init` in the WKB key by `numeric_serial`, and it edits the reads this prompt
   leaves. 09 relies on this prompt's refusal.
8. The board's reviews of 04–05b, for what the guard pins.

---

## 1. What is wanted

A value row is read by naming its parent's physical labels. The value reads name fewer of those
labels than the parent's own lookup does, so they can serve the values of a parent the caller did
not mean:
- **Numeric (P3).** `TkNumericValue._build_impl_model`, `GkNumericValue._build_impl_model` and
  `GkNumericValue.read_batch` omit `break_point_kind`, which the parents filter on.
  - They take `atol` and `rtol` as optional, and the parents require them.
  - Given two parents that differ only in break-point kind, both `build` paths raise, and
    **`read_batch` returns both parents' values** (probe_05).
  - `TkNumericValue.build` dispatches on `payload["z_init"]`, while its helper reads
    `payload["z_source"]`, so the dispatched call raises `KeyError`. Both of its messages call it
    `GkNumericValue`.
- **WKB (P4).** `GkWKBValue._build_impl_model` and `read_batch` omit `z_init`, which
  `GkWKBIntegration.build` filters on when it is given. With two parents at one `z_source` and
  different `z_init`, `read_batch` returns both and `build` raises (audit §3.4).
  **Found while writing this prompt, not in the audit:** `TkWKBValue._build_impl_model` and
  `read_batch` have the same omission against `TkWKBIntegration.build`, and there is no `z_source`
  to group by. P4 covers them: README §2's charter says "the value reads".
- **Labels (P5).** `GkNumericIntegration.build` labels `redshift.source` as
  `z_source_is_response`, and `GkNumericValue.read_batch` labels it as `z_response_is_response`.
  So both report a response flag equal to the source flag. `build` is live.

`break_point_kind` has no key form to choose, and it is simply added. For the rest, a value read
that holds only physical labels cannot always name one parent: `main.py` asks for every `z_source`
at once and knows no `z_init`. So D1's rule applies. **Select by the labels given, and refuse when
more than one parent matches**, naming the parents, rather than returning a mixture. The
refusal is per `z_source` for the Gk reads, and per `(model, k)` for the Tk WKB read, which has no
`z_source`.

---

## 2. What to change

**P3 — numeric.**
- `TkNumericValue._build_impl_model`, `GkNumericValue._build_impl_model` and
  `GkNumericValue.read_batch` filter the parent on
  `break_point_kind == <Target>.BREAK_POINT_KIND`. Read the constant from the compute target's
  single declaration, as the parents' `build()` does. A store without the column must fail with a
  refusal like the parents' (their `SQLAlchemyError` branch), never a raw `OperationalError`.
  *Your choice* whether to share one helper.
- `atol` and `rtol` become required (`payload["atol"]`) in the same three. `main.py` already passes
  both to `GkNumericValue.read_batch`; show that, and that no other caller omits them.
- `TkNumericValue`: one payload key, **`z_init`**, the parent's own name, in both `build` and
  `_build_impl_model`. Correct its two messages to say `TkNumericValue`.
- `GkNumericValue.read_batch` **refuses** when more than one parent contributes rows for one
  `z_source`, whether or not `z_source` was given.

**P4 — WKB.**
- `GkWKBValue._build_impl_model` and `read_batch`, and `TkWKBValue._build_impl_model` and
  `read_batch`, take an optional `z_init` and filter the parent on it **with the same expression
  its parent's `build()` uses**. That expression is the absolute `DEFAULT_FLOAT_PRECISION` match.
  Do not improve it here: 07 removes `z_init` from the key.
- `GkWKBValue.read_batch` refuses more than one parent per `z_source`, and `TkWKBValue.read_batch`
  more than one parent per `(model, k)`, whether or not `z_init` was given.

**The refusal.**
- Raise `sqlalchemy.exc.MultipleResultsFound`, the type the `build` paths already raise on the same
  condition.
- The message names the value class, `k`'s serial, the `z_source` serial where there is one, and
  every matching parent serial.
- Select the parent serial into the row query to decide it; do not issue a second query per row.
- A read that matches no parent returns `[]`, as today.

**P5 — labels.** `redshift_table.c.response` in both labels.

**P9 — close.** Add to the new test module a pin: the set of every factory function named
`read_batch` under `ObjectFactories/`, found by walking the sources as the guard does. It must equal
the five that have round-trip tests: `GkNumericValue`, `GkWKBValue` and `TkWKBValue` here, and
`QuadSourceIntegral` and `OneLoopIntegral` in 05b. Each entry names the test module that
round-trips it. A new `read_batch` then fails until someone writes its test.

**The guard's pins.** README §5 rule 7 allows edits to `Datastore/tests/test_factory_select_columns.py`,
named here, only to follow a change here that makes a pin fail, shown by the failing assertion in
the log.
- **Expected: none.** Filters add no reads, and the parent serial is read off rows the analyser
  already decides.
- Prefer a shape the analyser decides. An `EXPECTED_COVERAGE` entry may join if a new binding is
  unavoidable.
- No entry may leave `EXPECTED_COVERAGE`, and none may join `KNOWN_BLIND`. Either is a stop.

**`main.py` is not edited.** Nothing here needs it: the refusal is new behaviour only where two
parents already exist. If it turns out to need an edit, stop.

---

## 3. Tests — a new module `Datastore/tests/test_value_reads_parent_key.py`

No Ray, nothing under `var/`. The schema is built in memory and the calls go to the real factory
code, as in 05a and 05b.
1. **`break_point_kind`** (probe_05's design). Store two Gk numeric parents that differ only in the
   kind. `_build_impl_model` and `read_batch` serve only the parent whose kind is
   `GkNumericIntegration.BREAK_POINT_KIND`, and nothing raises. Do the same for Tk through
   `TkNumericValue.build` with `z_init`.
2. **Tolerances required.** Each of the three numeric reads, with `atol` or `rtol` omitted, raises.
3. **The live shape still works.** Call `GkNumericValue.read_batch` and `GkWKBValue.read_batch` with
   `main.py`'s payload: tags, `z`, no `z_source`, and the tolerance pair for numeric only. With one
   parent per `z_source` across three `z_source`s, every value comes back, each carrying its own
   `_z_source`.
4. **Refusal.** In each of the three `read_batch`es, two parents that both match: tags `{T}` and
   `{T, U}` queried with `[T]`, and for WKB also two `z_init`s with no `z_init` given. The read
   raises `MultipleResultsFound` naming both parent serials, with `z_source` given and without.
   Two parents at *different* `z_source`s do not refuse.
5. **`z_init` selects.** With the two-`z_init` store, giving one `z_init` serves only that parent,
   through `read_batch` and through `_build_impl_model`. This covers both `GkWKBValue` and
   `TkWKBValue`.
6. **Labels.** A redshift row stored with `source=True, response=False`, and one with the reverse,
   read back through `GkNumericIntegration.build` and `GkNumericValue.read_batch`. Each flag
   equals what was stored.
7. **The `read_batch` pin** (P9).

Tests 3–5 are P9's round trips for the three value `read_batch`es.

---

## 4. Verification

1. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) remove the `break_point_kind` filter from `GkNumericValue.read_batch`;
   - (ii) remove `GkWKBValue.read_batch`'s refusal;
   - (iii) remove the `z_init` filter from `TkWKBValue.read_batch`;
   - (iv) put `.source` back in one `…_is_response` label;
   - (v) make `atol` optional again in `TkNumericValue._build_impl_model`;
   - (vi) remove one entry from the `read_batch` pin.

   **Every diff in the log is `git diff` output, taken while the mutation is applied and pasted
   unaltered**, then replayed from the committed log: `git apply --check`, `git apply`, the tests,
   `git apply -R`.
2. **Callers.** A table of every caller of the six changed reads (`main.py`, `ComputeTargets/`,
   `extract_*.py`, `docs/`), each marked "unchanged" or with how its behaviour moves, and the
   evidence. Nothing outside the tests may newly refuse on a store with one parent per key.
3. All six suites, with README §7's command, run serially. Datastore rises by exactly the tests
   added, and the others match board §5.5. `black --check` is clean.

---

## 5. Acceptance

1. P3, P4 and P5 in the tree; §3's module. No Ray, no store, nothing under `var/`, and `main.py`
   unchanged.
2. The guard's pins unchanged, or moved only as §2 allows.
3. The deliberate-breakage record, (i)–(vi), replayed from the committed log, and the table of
   callers.
4. The issue closed on the store-fingerprint board's §4, with a closing note naming this commit by
   subject, and its index row deleted. The index's count and date are corrected.
5. The board: the §1 row for 05c, P3, P4 and P5, **P9 closed**, §3.1, and a new §5.6.
6. One commit, with the log at `logs/05c-key-the-value-reads-on-their-parent.md`.

---

## 6. Stop conditions — stop and ask the user

- `main.py` would have to change, or a caller outside the tests would newly refuse on a store with
  one parent per key.
- An existing test fails. That is a fixture with two parents, and whether the fixture or the
  refusal is wrong is the user's call.
- A guard pin would move in a way §2 forbids.
- A parent's own key turns out not to be what §1 says. Match the parent's `build()` as it is; do
  not change it.

---

## 7. What this prompt does not do

- It changes no parent's lookup, no schema, and no registration. The WKB key's `numeric_serial`
  is 07's, and keying derived rows on their parents is 09's.
- It does not make the `z_init` float match exact. 07 removes `z_init` from the key.
- It does not touch `GkSourceValue` or `QuadSourceValue`, whose reads already match their parents'
  keys, as the orchestrator checked while writing this prompt.
