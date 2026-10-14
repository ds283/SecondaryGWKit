# Log — runregistry-extraction prompt 03, pin `datastorekit` at `v0.4.0`

- **Subject:** [`03-pin-datastorekit-at-v0.4.0.md`](../03-pin-datastorekit-at-v0.4.0.md), board item K1.
- **Commit:** `5b81f2e` (parent `3b6ef6a`).
- **Result:** landed. K1 passes; no stop condition of the prompt's §4 fired. `ComputeTargets` shows its
  known wall-clock failure, which re-ran alone twice and failed both times (below).
- **Model:** Sonnet 5.5.

The orchestrator's facts for this prompt (its note's §0) were read first and are cited as "the note".

## What shipped

- **`requirements.txt:17`**, the one line: `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2`
  to `...@v0.4.0`. `git diff -U0` shows that line alone.
- **The venv**: `datastorekit` 0.2.2 (`6d8ee9ae…`) replaced by 0.4.0 at
  `c8570e849334d09085bbd84876095fc9e22bd005`, from the working tree, by the prompt's two `pip install`
  commands and no other. The install is not in the commit and `git revert` does not undo it (README §4.2).
- **[`measure/out/03-pip.txt`](../measure/out/03-pip.txt)**, **[`03-fingerprint.txt`](../measure/out/03-fingerprint.txt)**,
  **[`03-roots.txt`](../measure/out/03-roots.txt)**, **[`03-addenda.txt`](../measure/out/03-addenda.txt)**.
- This log, and the board: header, the §1 row for 03, item K1, `### 5.6 After prompt 03`.

## Deviations from the prompt

1. **`03-pip.txt` omits the `Requirement already satisfied` lines** of both pip runs (98 lines each),
   headed so. **IMPLEMENTATION CHOICE.** Every line that is not one of them is kept, including
   the two `Resolved ... to commit` lines. Nothing in the omitted lines can bear on "installs only
   `datastorekit`", which the `Would install` and `Successfully installed` lines say.
2. **`03-fingerprint.txt` records every store file with its SHA-256, size and `mtime_ns`** (twelve files:
   each store's primary, four shards, sidecar and manifest) by a small script
   in the scratchpad (`hashall.py`, `-I`), once before and once after, and shows the two listings equal.
   The after listing is stated, not repeated. The first attempt, with BSD `stat -f`, failed on this
   machine (a GNU `stat` is first on the path) and was re-run with Python before any read of a store;
   no store was touched by it. **IMPLEMENTATION CHOICE.**
3. **§2.3's `git grep -n 'store_fingerprint'` is not empty.** It gives two lines before and after the
   install, both the substring in `full_store_fingerprint`: a comment, `RunRegistry/stores.py:197`, and a
   golden file's name, `RunRegistry/tests/test_store_fingerprint.py:77`. The prompt's claim was meant
   for the module, so I also ran `git grep -nE 'datastorekit\.store_fingerprint|import store_fingerprint'`,
   which is empty before and after. **STRUCTURALLY REQUIRED** (the prompt's literal pattern cannot return an empty
   result on this tree; the intent is met). Both greps are in `03-pip.txt`.
4. **No other deviation.** Nothing outside §6's list was changed. `README.md`, `prompts/INDEX.md`,
   `orchestrator/` and `RunRegistry/` were not read for editing and not edited.

## Verification performed

- **The dry run (§2.3).** `pip install --dry-run -r requirements.txt`, exit 0: `Would install datastorekit-0.4.0`
  and nothing else. It resolved the tag to `8e7c1d77b3792b57b98983e84b99fc68ab88cb89` (the tag object, as board §5.5
  records). `raypoolkit` was resolved too and left alone, as already satisfied.
- **The install.** `Successfully installed datastorekit-0.4.0`, after `Uninstalling datastorekit-0.2.2`; no other package.
- **`pip freeze`, before and after (§2.1, §2.3).** `diff` is the one line, 17: `datastorekit @ ...@6d8ee9ae…` to `...@c8570e84…`.
- **`pip check`:** `No broken requirements found.`, exit 0.
- **`package_provenance()` before:**
  `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false, "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version": "0.2.2"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}}`.
  **After:** `datastorekit` `{"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false, "requested_revision": "v0.4.0", "url": ..., "version": "0.4.0"}`;
  `raypoolkit` byte-identical to before.
- **`ENCODING_FORMAT`.** `import datastorekit.store_fingerprint as m; print(m.ENCODING_FORMAT)` prints `1`. (The prompt's
  text says it prints `1`; it does.) That is the one import of the module.
- **SGK's own fingerprint of R0's stores (§2.4).** `python -m RunRegistry store fingerprint` on `runregistry-extraction-02-reference.sqlite`
  and `-02-resumed.sqlite`, neither with `--write` nor `--listing`: each prints format 6, digest
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21 classes (the 21 rows from `version` to `OneLoopIntegral`),
  `problems: none`, `recorded: matches ...`, exit 0. The twelve store files have the same SHA-256, size and mtime before and after.
- **The seven roots (§2.5)**, serially, load 19 to 75, against 02a's counts:

  | Root | 02a's count | Ran here | Result |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | 32 | OK |
  | `ComputeTargets/tests` | 693 | 693 | 1 failure, the known one |
  | `CosmologyModels/tests` | 39 | 39 | OK |
  | `LiouvilleGreen/tests` | 148 | 148 | OK (skipped=1) |
  | `ObjectFactories/tests` | 271 | 271 | OK |
  | `RunRegistry/tests` | 286 | 286 | OK |
  | `tools/tests` | 91 | 91 | OK |

  The failure is `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, README §8's, 0.147 s against 0.06 s in the root. Alone it
  failed twice more, 0.090 s and 0.118 s, at load about 68. It fails alone at this machine's load, as at 02a's review and as the note
  says; no code it runs has changed.
- **The addenda (§2.6).** [`03-addenda.txt`](../measure/out/03-addenda.txt) tables every item of the three. Summary:
  - **`v0.2.3`:** six items, none applies at the pin ("Nothing changes for SGK"); its pin item is done by this prompt's move
    to `v0.4.0`. SGK instantiates no factory: 38 classes name `SQLAFactoryBase`.
  - **`v0.3.0`:** five items, none applies at the pin ("Nothing changes for SGK but one key in `vars()`"); the 38
    `DatastoreObject.__init__` calls are there with one positional argument, and no class defines `timestamp`.
  - **`v0.4.0`:** the pin and the "nothing SGK calls" block are done here (the pin, the roots at 02a's counts, SGK's fingerprint
    `39809dca…`); the seven "at the call" items do not apply at the pin, by the closing table's own column. **Handed to 06:**
    C1 (ten calls of `fingerprint_of`, two with `taken` second positionally: `RunRegistry/stores.py:1500`,
    `test_store_fingerprint.py:378`), C2 (`test_the_shape`, `test_the_golden_fingerprint`, the six golden files), C3 (D6: retire R0's
    and R1's stores first), C4 (`parse_listing` unchanged), C5 (`listing_lines` needs the package's fingerprint), C6 (format 6 and its
    history to `config/fingerprint.py`, never `ENCODING_FORMAT`). **Handed to 04:** C7.
  - **The table's claims about SGK's tree**, checked at `HEAD`: every one holds. The ten `fingerprint_of` calls, the 20 calls of the
    three functions, the line numbers of the defs, `FINGERPRINT_FORMAT` at `:206`, the two live stores, 38 factory classes,
    38 `DatastoreObject.__init__` calls. One point of drift: the note's `stores.py:1499-1507` is a line early against the addendum's
    `:1500-1508`, which is right. Some secondary line cites are a line or two off; none false in substance.
- **`black`:** nothing here is Python; not run.
- **Scope:** `git diff --cached --name-only` is shown before the commit (below).

## The deliberate-breakage record

None. This prompt adds no code and no test (README §5 rule 8; acceptance 6).

## Observations not acted on

- **`git grep -n 'store_fingerprint'` matches `full_store_fingerprint`** (deviation 3). 04's prompt or 06 may want a pattern that names
  the module, as `datastorekit\.store_fingerprint` does, if it states "SGK does not import the module" as a check.
- **`pip install --dry-run` prints `Resolved ... to commit 8e7c1d77…`, the tag object, not the commit** `c8570e84…`.
  The installed `direct_url.json` and `pip freeze` record `c8570e84…`, as the prompt expects. This is pip's wording,
  not a defect; noted so that a later reader does not take the line for a mismatch.
- **A GNU `stat` is first on this shell's path**, so a BSD `stat -f` fails here (deviation 2). A script that hashes with Python is portable.
- **`DatastoreObject.__init__(self, None)`** is how 11 of the 38 calls read (store id `None` positionally), against the addendum's
  "pass `store_id` alone". Same thing; the addendum's wording is the keyword reading.
- **The `ComputeTargets` flake** failed alone twice at load about 68 (above); no new fact.
- Nothing opened, so `docs/OPEN_ISSUES.md` stays at 107.

## State handed to the next prompt

- **`requirements.txt:17`** pins `datastorekit` at `v0.4.0`, and the venv holds it: `c8570e84…`, not editable. `raypoolkit`
  is at `v0.2.0` as before; there is no `runregistrykit`. `pip freeze` differs from before by that one line.
- **R0's two stores are unchanged**, still live, each with a sidecar recording an SGK fingerprint at format 6, `39809dca…`, and
  SGK's own code reads them to that digest through the new pin. They are not retired (the orchestrator does that after 04's review,
  by D6).
- **04 runs R1 at this tree.** Its fingerprint-both-ways script may call `datastorekit.store_fingerprint` (the prompt-04 charter);
  the module's `ENCODING_FORMAT` is `1`. The lock is free, the registry shows nothing running, and nothing here took the lock or started Ray.
- **Rollback** (README §4.2): `git revert` of `5b81f2e`, then `pip install -r requirements.txt` from the reverted file to put
  `v0.2.2` back.
- **06's list** is in `03-addenda.txt`, "Handed to 06".
