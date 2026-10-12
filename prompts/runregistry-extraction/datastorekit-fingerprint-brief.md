# Brief: the store fingerprint's algorithm, and a version for it

**For:** the agent that plans a campaign in DatastoreKit (`/Users/ds283/Documents/Code/DatastoreKit`)
to release `datastorekit` with the store fingerprint's algorithm and a version covering it, under
DatastoreKit's `CLAUDE.md` and its campaign conventions. **Written:** 2026-10-12, from a Claude Code
session in SecondaryGWKit (SGK), with the user, while planning SGK's `runregistry-extraction`
campaign (`prompts/runregistry-extraction/` in SGK, README §6.1, D3).
**Trees:** DatastoreKit at `d88615e` (`v0.2.3` is `0de2b16`; `object-timestamp` is in progress,
its prompt 02 the release `v0.3.0`); SGK at `c8a7f14`.

**This brief is data to check, not a specification** (DatastoreKit `CLAUDE.md` invariant 7). The
user's ruling, in §1, is a ruling. Everything else is a finding with its citation, or a
recommendation marked **Advice:**. Where this brief and DatastoreKit's own records disagree, the
records are right, and the disagreement is worth reporting back. The campaign name, the decision
numbers, the version number and the shape of the campaign are DatastoreKit's to choose.

The reasoning behind the ruling is SGK's
[`docs/runregistry-extraction-audit.md`](../../docs/runregistry-extraction-audit.md) §3.

## 1. What is asked, and who ruled it

**The user's ruling (2026-10-12, SGK's D3).** SGK's `RunRegistry` moves to a repository of its own,
RunRegistryKit, shared by the three clients. Its store fingerprint splits by layer:
- **what goes into a fingerprint** stays in each client: its factories' `inventory_spec()`, its
  registry, and a format number, grouped together in one client module;
- **how a fingerprint is encoded and digested** is DatastoreKit's, **with a version of its own**;
- **when and where a fingerprint is taken and recorded** is RunRegistryKit's, which is handed a
  client-provided fingerprinter and treats the result as opaque.

**The change asked for:**
1. **The algorithm moves into DatastoreKit**, from SGK's `RunRegistry/stores.py:1164-1390` at
   `c8a7f14`, into a module of the package's own, beside `store_inventory`:
   - `fingerprint_of(inventory, taken=None)`;
   - `compare_fingerprints(recorded, current)`;
   - `listing_lines(inventory, fingerprint)`;
   - their helpers, `_sha256_lines` and `_tag_sets`.

   Where SGK's code reads the module constant `FINGERPRINT_FORMAT`, the package's takes **the
   client's format number as a parameter**. The package names no client format.
2. **A version for the encoding and the algorithm together**, recorded in every fingerprint
   **beside** the overall digest, not inside the object it hashes (§3). `compare_fingerprints`
   refuses to compare two fingerprints whose client format **or** package version differ, as it
   refuses a format mismatch today.
3. **A rule, in the client contract, for when that version is bumped**: any change to what a record
   or a canonical line contains, to how a digest is formed, or to what `compare_fingerprints` and
   `listing_lines` read.
4. **A tagged release** carrying all three, with a note for SGK (as the adoption checklists'
   addenda are), saying what SGK must do to adopt it.

## 2. Why: the encoding has no version

**A fingerprint is made in three layers** (audit §3.1, each re-read at `c8a7f14`):

| Layer | What it decides | Where today |
|---|---|---|
| A. Content | which classes, and each class's identity | the client's `inventory_spec()`s and registry |
| B. Encoding | a row → a `Record` → a canonical line: `canonical`, `canonical_json`, `reference_digest`, `Record.as_json`, the parent-set digest, the replicated-class choice, the problem names | `datastorekit/store_inventory.py` |
| C. Algorithm | SHA-256 per class and per tag set over canonical lines; the overall digest of `canonical_json({"fingerprint_format", "classes": {name: {"count", "digest"}}})`; problems and `taken` outside every digest; `compare_fingerprints`; `listing_lines` | SGK's `RunRegistry/stores.py` |

Layer C is a pure function of a `StoreInventory`: it reads nothing, names no client class, and
imports only `canonical_json` from layer B.

**The one version is the client's, and it has carried both jobs.** SGK's `FINGERPRINT_FORMAT` went
1 → 6 in six commits (audit §3.2). Every bump came from a change to what SGK puts in a key. Three of
them also extended layer B to carry it: `a171ced` added a `nullable` parent, `dbf2cf2` a
`cross_shard` parent, `3d7b606` the `ParentSet`. Those were SGK commits to SGK's own copy of the
layer, before the extraction.

**Layer B has no version.** Since format 6, `store_inventory.py` changed at `bc8bffc`, `6646b06`
and `4c34f13` in SGK, and moved here at extraction. `4c34f13` records that the fingerprint is
unchanged, and SGK's rehearsal digest `39809dca…` has reproduced at every pin move since. That
acceptance test, not a version, is what has protected it. A release that changed `canonical` would
change every digest while the client's format stayed the same, and `compare_fingerprints` would
report a content difference in every class rather than a format mismatch. With three clients, each
fingerprinting its own stores, that is worth closing in the package.

SGK records this as `[00-the-fingerprint-encoding-has-no-version-of-its-own]`, on
`prompts/runregistry-extraction/IMPLEMENTATION_STATE.md` §3. It is reported here as a DatastoreKit
defect. SGK's half closes when SGK records the version (its prompt 06).

## 3. The constraint: SGK's digest must not move

SGK's reference fingerprint is format 6, digest
`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21 classes. It is the acceptance
of every SGK rehearsal since `a3-v2-readiness` 07, and of `dk-and-rpk-migration`'s G2 records.
- So the object the overall digest hashes stays exactly `{"fingerprint_format": <client format>,
  "classes": {…}}`. SGK passes `6`; the package's new version goes **beside** the digest in the
  fingerprint's dictionary (SGK's audit calls it `inventory_format`; the name is yours).
- The per-class and per-tag-set digests and counts stay as they are.
- `problems` and `taken` stay outside every digest.

**How SGK checks it.** DatastoreKit's tests cannot name SGK. So SGK's R1 rehearsal (its prompt 04)
fingerprints each rehearsal store both ways, with SGK's own `fingerprint_of` and with the package's,
at format 6, and requires every digest and count to be equal. A difference there is reported back to
you as a defect.

**No reader of old fingerprints is asked for.** Every sidecar SGK holds is a tombstone, never
fingerprinted again (audit §3.4). SGK's standing rule is that old records are read at their own
commit, never by a version-aware reader. A fingerprint without the new version field need not be
accepted by the new `compare_fingerprints`.

## 4. What else SGK found, for you to weigh

- **Tests with no client.** The moved functions need tests here, on stores built by the neutral
  client (`datastorekit/tests/client/`, `real_store_fixtures.py`), with a golden fingerprint of the
  package's own. **Advice:** keep SGK's history comment (`RunRegistry/stores.py:194-205`) out of
  the package; it is SGK's format history, and moves to SGK's client module.
- **Advice: a fixture RunRegistryKit can use.** RunRegistryKit will need a small store that can be
  fingerprinted without any client (SGK's audit §4: 78 of its tests do). The package's tests are
  excluded from the distribution (`pyproject.toml`), so RunRegistryKit cannot import the neutral
  client. If you judge it in scope, a small supported fixture would spare RunRegistryKit writing
  its own. If not, RunRegistryKit writes its own.
- **`canonical_json` refuses a non-string key.** SGK's adoption will pin a digest of the canonical
  form of every `inventory_spec()` (its prompt 07). A polymorphic `Parent.types` maps a type
  value, not necessarily a string, to a class name, so SGK will stringify those keys itself. This
  needs nothing from you; it is noted in case you prefer to offer a canonical form of an
  `InventorySpec`.
- **Not asked now: more than one value table per spec.** StochasticInstantons'
  `GradientCoupledInstanton` has two child tables, and an `InventorySpec` has one `values` slot
  (`store_inventory.py:293-303`). SI's own prompt 10 settles which table takes the slot, and the user
  may then ask you for a second. That is a separate request, if it comes.
- **A fingerprint counts value rows; it does not hash them** (`Record.value_count`). By design, and
  not asked to change.

## 5. Who consumes the release

- **SGK**, by moving its pin from `v0.2.2` (its prompt 03), then calling the new function from its
  client module (prompt 06).
- **RunRegistryKit**, whose `[stores]` extra will require the release by name and minimum version,
  not by git URL, so that each client's own pin satisfies it.
- **ChamPBH and StochasticInstantons**, once their `dk-and-rpk-migration` campaigns have written
  their specs (ChamPBH 09; SI 10) and they adopt RunRegistryKit.

SGK's campaign waits on the tag (its gate E1). Please report the tag, its commit and the note for
SGK back to the user.
