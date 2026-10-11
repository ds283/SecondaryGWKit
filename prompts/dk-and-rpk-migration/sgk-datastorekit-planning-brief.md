# Brief: planning SGK's adoption of `datastorekit`

**For:** the agent that plans SecondaryGWKit's (SGK's) move from its own `Datastore` /
`ShardedPool` layer to the packaged `datastorekit`. **Written:** 2026-10-11, from DatastoreKit at
`460b012` (the `actor-names` campaign closed), and SGK read through `git` the same day. It is the
companion of [`sgk-raypoolkit-planning-brief.md`](sgk-raypoolkit-planning-brief.md), and has the
same shape.

This brief is an index. The facts are in the documents it points to; where the two disagree, they
are right and this brief is wrong.

## 1. Where things stand

| | State on 2026-10-11 |
|---|---|
| **The release to adopt** | `datastorekit` **`v0.2.2`**: tag object `c32d027` on commit `6d8ee9a`, at `https://github.com/ds283/DatastoreKit` (the tag is on `origin`). Python `>=3.12`, `ray>=2.43`, `sqlalchemy>=2.0.39,<2.1`; tested at 3.12.15 / 2.43.0 / 2.0.39 and 3.13.16 / 2.55.1 / 2.0.46. `v0.2.0` and `v0.2.1` remain tagged, but **every client adopts `v0.2.2`**: it fixes a defect all three earlier releases carry |
| **SGK** | Its `HEAD` is `b510bc9` on branch `handover-remedial`, with a clean tree (but for this folder) and no stash; **228 commits ahead of `main`** (`d6afa05`). Its venv is Python 3.12.15 with `ray==2.43.0` (`requirements.txt:74`) and `SQLAlchemy==2.0.39` (`requirements.txt:85`): the package's tested low end, exactly. No `datastorekit` in `requirements.txt` |
| **The freeze** | SGK's 17 layer files are byte-identical from the import commit `6f7f291` to `b510bc9` (`git diff --stat` over them is empty, re-checked today). The freeze is DatastoreKit's decision U3 and SGK's `CLAUDE.md:47-73` |
| **The checklist** | Measured at SGK `b510bc9` against `v0.2.0`, with addenda for `v0.2.1` and `v0.2.2`. Its import measure, re-run today at `b510bc9`, reproduces its appendix **byte for byte**: 233 import statements of the layer in 138 files |
| **SGK and RayPoolKit** | Not adopted yet either. RayPoolKit's latest tag is `v0.2.0`; its brief is beside this one |
| **The gate** | G2, on DatastoreKit's extraction board, reads "SGK adopted `v0.2.2`, fingerprint reproduced". It is open; DatastoreKit's campaigns closed without waiting for it |

## 2. What to read, in order

All of these are in DatastoreKit (`/Users/ds283/Documents/Code/DatastoreKit`) unless a path says
otherwise. **Read DatastoreKit and RayPoolKit; do not edit them.**

1. [`docs/adoption/secondarygwkit.md`](/Users/ds283/Documents/Code/DatastoreKit/docs/adoption/secondarygwkit.md),
   SGK's checklist and the plan's main input. Its ten items, at their lines in the file as it
   stands at `v0.2.2`:
   1. the client and its versions (`:44`);
   2. what is deleted: the 17 files, and what `tools/` keeps (`:59`);
   3. the 233 imports to rewrite, by group, with the `docs/` scripts that are live, dead or
      historical (`:84`);
   4. where the factories may go, and what each option touches (`:157`);
   5. what the factories change: their imports only (`:190`);
   6. the seven pool constructions, none of which changes (`:201`);
   7. the tests: the 26 modules the package now carries, the 18 that stay, the fixtures SGK keeps,
      and the tests that read deleted files (`:219`);
   8. why no SGK store is refused (`:296`);
   9. **the acceptance (G2)**, with the rehearsal command and the reference fingerprint (`:310`);
   10. what goes stale (`:344`).

   **Read it with both addenda** (`v0.2.1`, `:369`; `v0.2.2`, `:447`). Each ends with a table of
   the earlier statements it supersedes; where the body says `v0.2.0` or `v0.2.1`, read `v0.2.2`.
   A `datastorekit/` line in the body is `v0.2.0`'s; in an addendum it is that addendum's release.
   The appendix (`:537`) is the measuring script's output.
2. [`docs/adoption/README.md`](/Users/ds283/Documents/Code/DatastoreKit/docs/adoption/README.md),
   the rules for every client: the pin (§1), the module map (§2), what the wheel does not ship
   (§3), what a client supplies (§4), how a checklist is used (§5), how to re-measure (§6), and its
   two addenda, which list every behaviour that changed since `v0.2.0`.
3. [`README.md`](/Users/ds283/Documents/Code/DatastoreKit/README.md) (the package's): "Status",
   "Supported versions", "Installing" and "The tools".
4. [`docs/client-contract.md`](/Users/ds283/Documents/Code/DatastoreKit/docs/client-contract.md):
   §1–§7 are what SGK already supplies, unchanged; **§9** is every change after `v0.2.0`, each with
   the test that pins it.
5. [`PROVENANCE.md`](/Users/ds283/Documents/Code/DatastoreKit/PROVENANCE.md): how SGK's 17 files
   map onto the package, the differences allowed, and "The freeze".
6. [`docs/extraction-verification.md`](/Users/ds283/Documents/Code/DatastoreKit/docs/extraction-verification.md):
   §4 (the package under real Ray, at both ends, re-run after each `v0.2.2` change in §4.6–§4.7),
   §5 (the `v0.2.1` pin installed from GitHub), §6.3 (the four inherited issues) and **§9.3** (what
   is unverified: no client's suite has been run against the package, nor any multi-node cluster).
7. **RayPoolKit's brief and checklist**, for the order decision: the brief beside this one, and
   RayPoolKit's `docs/adoption/secondarygwkit.md` item 7 and `docs/adoption/README.md` §5.
8. **SGK itself**, at the commit the campaign starts from:
   - its `CLAUDE.md`: the freeze (`:47-73`), the run registry (`:75-102`), which governs the G2
     rehearsal, and the test command (`:106-109`);
   - `docs/a3-v2-readiness-verification.md`, which holds the reference fingerprint (`:25-27`) and
     the rehearsal command (§1.1, `:58-71`);
   - its `prompts/INDEX.md` and `docs/OPEN_ISSUES.md`, for anything in flight that touches the
     factories, `config/`, `RunRegistry/` or the tests.

## 3. Decisions for the plan to put to the user

1. **The order with RayPoolKit.** `datastorekit` first, `raypoolkit` first, or both in one
   campaign; all three are allowed. The two touch disjoint files but for `RayTools/RayWorkPool.py:8`
   (rewritten under RayPoolKit's decision U12 if DatastoreKit goes first) and
   `Datastore/tests/test_layer_is_generic.py` (deleted by this adoption; edited first by
   RayPoolKit's if that goes first). One consideration is this brief's own: **G2's rehearsal runs
   the pipeline, and so both packages.** A rehearsal run after both swaps that misses the
   fingerprint cannot say which package caused it, and the rehearsal is a long, person-run job. A
   rehearsal between the two swaps keeps the result attributable.
2. **The branch.** The checklist was measured on `handover-remedial`, not `main`. Settle which
   branch the adoption lands on, and re-measure there.
3. **Where the factories live** (checklist item 4): under `Datastore/` as namespace packages; under
   `Datastore/` with empty `__init__.py` files in place of the two deleted ones; or moved, which
   changes 98 factory imports and five paths that reach a factory by file.
4. **The tests that stay** (item 7):
   - how to change `test_inventory_retired.py`, which reads two deleted files (`:52`, `:75`);
   - where the one unported test, `test_the_report_renders_both_cosmology_types`, goes when
     `test_inventory_declarations` is deleted;
   - whether to keep `schema_description.py`, whose only importer is deleted;
   - whether SGK's `standin_pool.py` stands in `_ray_is_running` and `ray.kill`, as the package's
     does (`v0.2.2` addendum, "SGK's own stand-in"). Its tests pass either way.
5. **Which `docs/` scripts stay runnable** (item 3). Three are reached by SGK's tests or by G2 and
   must be rewritten: `docs/handover/quadsource_atol_sweep.py`, **the rehearsal's launcher**, and
   the two under `docs/source-remediation-verification/`. Ten statements in six files already fail
   at `b510bc9`; four more are live probes. The checklist advises leaving `prompts/`'s 51
   historical statements as they are; SGK's own rules govern.
6. **Which test roots the acceptance counts.** SGK has six: `AdaptiveLevin`, `ComputeTargets`,
   `CosmologyModels`, `Datastore`, `LiouvilleGreen` and `RunRegistry`, each `<root>/tests`.
   `CLAUDE.md:106-109` runs only `ComputeTargets/tests`. The adoption deletes 26 modules of
   `Datastore/tests` with 392 `def test_` lines, so its count falls by design: record each root's
   count before and after, and say which fall is expected.
7. **The G2 rehearsal**: the store, slug, purpose and campaign of the run (item 9), and when a
   person runs it.
8. **Run provenance** (item 10): `git_provenance()` records SGK's `HEAD` and nothing of the
   installed `datastorekit`. Whether to record its version is SGK's choice.
9. **The four inherited issues** (item 10; DatastoreKit's `docs/OPEN_ISSUES.md` §1.2). After the
   adoption the code they describe is DatastoreKit's. They are not fixed in SGK. How SGK's boards
   record the hand-over is SGK's choice; DatastoreKit's index changes only by a commit there.

## 4. Rules that come with the package

- **Pin the tag:** `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2` in
  `requirements.txt`, in a commit of its own. Never `v0.2.0` or `v0.2.1`, and never an editable
  install for a production run. SGK keeps its own Ray and SQLAlchemy pins.
- **Rewrite the module path only.** The names imported do not change (adoption README §2):
  `from Datastore import DatastoreObject` becomes `from datastorekit import DatastoreObject`, and
  `Datastore.SQL.ObjectFactories.base` becomes `datastorekit.SQL.factory_base`.
  `datastorekit.SQL.Datastore` is still the actor *class*; the module is reached by
  `importlib.import_module("datastorekit.SQL.Datastore")`. String targets (`import_module`,
  `mock.patch`) must be rewritten too.
- **The wheel ships no tests.** The stand-in pool, the fixtures and the neutral test client are not
  installed; SGK keeps its own copies, and nothing of SGK imports `datastorekit.tests`.
- **The tools move:** `python -m datastorekit.tools.sharded_store` and
  `python -m datastorekit.tools.shard_key_audit`. `tools/inventory_report.py` and
  `tools/__init__.py` stay in SGK.
- **The freeze holds until the adoption lands.** The 17 files are deleted by the adoption, not
  edited before it.
- **Nothing the layer writes changes**, and no message, refusal or exception changes but those
  contract §9 lists. A store the package refuses, or a fingerprint that differs, is reported to
  DatastoreKit as a defect, not worked around.
- **One pool per Ray session at a time.** The actors' names are fixed, so two pools open at once
  collide. A closed pool now releases its names, and its handles are dead. SGK opens one pool per
  process and uses none after its `with` block (`v0.2.2` addendum), so nothing in SGK changes.
- **A defect in the package is not patched in SGK.** Report it to DatastoreKit, where it is fixed
  in a new release that SGK then pins.
- **No change is made in DatastoreKit or RayPoolKit by SGK's campaign.** Recording G2 on
  DatastoreKit's extraction board (`prompts/extraction/IMPLEMENTATION_STATE.md`, the G2 row),
  once it holds, is a records commit in DatastoreKit, made there.

## 5. First steps

1. **Re-measure at the campaign's start commit**, from DatastoreKit's root with DatastoreKit's
   venv. The script reads SGK through `git` only:
   ```bash
   ./venv/bin/python docs/extraction/measure_client_imports.py sgk --commit <SHA>
   ```
   Compare with the checklist's appendix. At `b510bc9` the output is the appendix, byte for byte
   (re-run 2026-10-11). Otherwise the new figures govern, and any change in the 233 imports, the
   factory imports or the string and `import_module` sites goes into the plan.
2. **Check the freeze held:** `git -C <SGK> diff --stat 6f7f291 <SHA> --` over the 17 files of
   checklist item 2 is empty. If it is not, the checklist's premise has moved; find out why before
   planning.
3. **Record each test root's count** at the start commit (decision 6).
4. **Plan to checklist item 9**, the acceptance, read at `v0.2.2`:
   - the 17 layer files gone, and 0 imports of them outside `prompts/` and the `docs/` scripts the
     user chose to leave, by the script;
   - the pin in place, and SGK's venv installed from it. No one has yet installed `v0.2.2` from
     GitHub (`v0.2.1` was, verification §5); SGK's install is the first;
   - SGK's remaining suites passing, with each root's count against step 3;
   - **a rehearsal rebuild through the package reproducing the reference fingerprint**: format 6,
     digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes
     equal, taken with `python -m RunRegistry store fingerprint`, run by a person under SGK's
     run-registry rules.
