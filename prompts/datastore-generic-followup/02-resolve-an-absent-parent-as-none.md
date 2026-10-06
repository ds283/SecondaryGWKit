# Prompt 02 — resolve an absent parent as `None`

**Campaign:** [`README.md`](README.md) · **Board item:** **V1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 01 landed and was reviewed. `git status` clean. Never in one checkout while another
prompt's change to `store_inventory.py` is uncommitted.
**Closes:** `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`, on `datastore-generic`'s
board. **Narrows:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. The change is three lines and a docstring sentence. Every
`resolve()` output it moves was measured with a rough version of the change (README §0.2).

**Read first:**

1. [`README.md`](README.md): §0.2 (item 2), §1, §4, §5 (rules 3, 4, 7 and 9) and §6.1 (decision 2).
2. `prompts/datastore-generic/IMPLEMENTATION_STATE.md`, §3.1: the entry
   `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`.
3. `Datastore/store_inventory.py`:
   - `Parent`, its `nullable` field and docstring (`:215-261`);
   - `StoreInventory.resolve` (`:445-479`) and `_resolve_reference` (`:481-492`);
   - in `read_records`, `_reference` (`:691-703`) and the loop that drops a row whose parent does
     not resolve (`:705-720`). That loop is why a `None` key parent means only "nullable and
     absent".
4. `Datastore/tests/test_inventory_declarations.py`, `TestResolve` (`:309-` to the class's end),
   where the new tests go.
5. The orchestrator's measurement: the `store_inventory.py` hunk of
   [`measure/rough_change.diff`](measure/rough_change.diff), and
   [`measure/scripts/r2_resolve.py`](measure/scripts/r2_resolve.py).

Line numbers are at `6de692a`. 01 does not touch `store_inventory.py`.

---

## 1. What is wanted

`StoreInventory.resolve(name, record)` replaces each parent reference in a record's key with the
parent's resolved key. A nullable key parent whose value is `None` is a row with no such parent.
`resolve()` renders it as `{"class": <class>, "digest": None, "unresolved": 0}`, the form of a
parent it failed to find. `_resolve_reference`, for a member of a parent set, returns `None`.

The user decided on 2026-10-06 that `resolve()` renders it as **`None`** (README §6.1).

- **V1.** In `resolve()`, a key parent field whose value is `None` resolves to `None`.

Measured at `6de692a` (README §0.2): over every record of `build_full_store` and
`build_real_store`, exactly 11 entries change. Each is a `numeric` field that goes from the
unresolved form to `None`: 2 in direct `GkWKBIntegration` records, and 3 each reached through
`GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral`.

**What does not change.** Every other `resolve()` entry; the fingerprint, the inventory and the
inventory report, none of which calls `resolve()`; and what the code writes.

---

## 2. What to change

**V1 — `StoreInventory.resolve`.**
- After the `field not in cls.parents` branch, a field whose value is `None` gives `None`, and the
  loop continues.
- A value that is not `None` and does not resolve to exactly one record keeps today's
  `{"class": …, "digest": …, "unresolved": …}`. In particular `TestResolve`'s case of a type the
  map does not hold is unchanged.
- The docstring says that an absent parent (a `None` value, which only a nullable parent can
  hold: `read_records` drops a row whose parent does not resolve) gives `None`, as a member does.

**Tests: two new methods in `test_inventory_declarations.TestResolve`.** No existing test changes.
- `test_an_absent_nullable_parent_resolves_to_none`. A hand-built inventory, as
  `test_a_hand_built_inventory` builds one, with a key parent whose value is `None` → `None`. Add
  a second key parent in the same record that does resolve, which must still give its resolved
  form.
- `test_the_full_store_resolves_no_absent_parent_as_unresolved`, on `build_full_store`:
  - walking `resolve()` of every record of every class, no mapping holding `"unresolved"` has
    `"digest": None`;
  - exactly 11 `numeric` fields are `None` (README §0.2);
  - assert first that the walk visited at least one record of `GkWKBIntegration`, so that the
    test cannot pass on an empty inventory.

---

## 3. Verification

1. **Every suite**, with counts before and after, each difference accounted for by test id:
   - Datastore: 01's count + 2. At 01's expected 732, that is **734** (`[a-f]` 158, `[g-o]` 183 + 2
     = 185, `[p-r]` 200, `[s-z]` 191);
   - every other suite unchanged.
2. **The measurement reproduced.** `measure/scripts/r2_resolve.py`, run at the parent and at this
   commit, differs in exactly the 11 entries of §1. Show the summary.
3. `test_layer_is_generic` passes. `black --check` is clean on every touched Python file.
4. **Deliberate breakage.** Each must make the named tests fail; then restore it.
   - (i) the new branch removed: both new tests;
   - (ii) the branch widened to give `None` for any reference that does not resolve to one record:
     `test_a_hand_built_inventory`, whose third child must stay unresolved;
   - (iii) the branch moved above `field not in cls.parents`, so that a `None` leaf skips the
     leaf path. Expected: nothing fails, because a `None` leaf gives `None` either way. Record it
     as a check that the order does not matter, not as a pin.

   Record each in the log as a diff, exactly as applied, with what failed. Check each with
   `git apply --check` and `-R --check`, in `bash`, and leave `git status` clean.

---

## 4. Acceptance

1. **V1** as §2 says, with the two tests.
2. Every suite has the counts of §3.1.
3. **The records**, in the same commit:
   - On `datastore-generic`'s board, `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`
     moves from §3.1 to §4 with a dated "Closed" line. It names this campaign, the change and the
     11 entries, and corrects the issue's hook, which said all 11 came through `GkSource`'s
     parent set.
   - `docs/OPEN_ISSUES.md`: its row leaves §1.19, the count goes down by one, and the date is
     corrected.
   - This board: the §1 row, V1, §5.

---

## 5. Stop conditions — stop and ask the user

- The measurement does not reproduce: any `resolve()` entry changes other than the 11.
- A test outside §2's plan fails.
- A `None` key parent is found on a parent that is not nullable. That would mean `read_records`
  keeps a row it should drop, which is a different defect.
- The change would open, write or touch any store under `var/`, or start Ray.

---

## 6. What this prompt does not do

- **Files it changes:** `Datastore/store_inventory.py` (`resolve` and its docstring);
  `Datastore/tests/test_inventory_declarations.py` (two new methods); the log, this board,
  `prompts/datastore-generic/IMPLEMENTATION_STATE.md` and `docs/OPEN_ISSUES.md`. Nothing else.
- It does not change `_resolve_reference`, `read_records`, the report or the fingerprint.

---

## 7. The log and the board

`logs/02-resolve-an-absent-parent-as-none.md`, using README §5.1. In addition, it gives the 11
entries before and after, and each breakage as a diff with what failed.

`IMPLEMENTATION_STATE.md`: the §1 row for 02, item V1, §5.
