# Brief: planning SGK's adoption of `raypoolkit`

**For:** the agent that plans SecondaryGWKit's (SGK's) move from its own `RayTools/RayWorkPool.py`
to the packaged `raypoolkit`. **Written:** 2026-10-11, from RayPoolKit at `c9b0e11` (the extraction
campaign closed), and SGK and DatastoreKit read through `git` the same day.

This brief is an index. The facts are in the documents it points to; where the two disagree, they
are right and this brief is wrong.

## 1. Where things stand

| | State on 2026-10-11 |
|---|---|
| **The release to adopt** | `raypoolkit` **`v0.2.0`**: tag object `4e7eb15` on commit `eaef0ce`, at `https://github.com/ds283/RayPoolKit`. Python `>=3.12`, `ray>=2.43`; tested at 3.12.15 / 2.43.0 and 3.13.16 / 2.55.1 |
| **SGK** | Its `HEAD` is `b510bc9` on branch `handover-remedial`, with a clean tree and no stash. That branch is **228 commits ahead of SGK's `main`** (`d6afa05`, 2026-09-19). `RayTools/RayWorkPool.py` is blob `b4666fa` on both, frozen since 2026-10-10. Its venv is Python 3.12.15 with `ray==2.43.0` (`requirements.txt:74`), the package's tested low end |
| **SGK and DatastoreKit** | Not adopted yet: no `datastorekit` in `requirements.txt`. SGK's last commit froze its datastore layer for the move. DatastoreKit's latest tag is `v0.2.1` |
| **The checklist** | Measured at SGK `b510bc9`, which is still SGK's `HEAD`, so its figures hold today |

## 2. What to read, in order

All of these are in RayPoolKit (`/Users/ds283/Documents/Code/RayPoolKit`) unless a path says
otherwise. **Read RayPoolKit and DatastoreKit; do not edit them.**

1. [`docs/adoption/secondarygwkit.md`](/Users/ds283/Documents/Code/RayPoolKit/docs/adoption/secondarygwkit.md), SGK's checklist and the plan's main
   input. Its nine items:
   1. the versions and the pin;
   2. what is deleted;
   3. the 7 imports to rewrite;
   4. the 54 call sites, none of which must change;
   5. what changes in behaviour, with `main.py:3247` worked through;
   6. the tests: `test_pipeline_adoption.py`, the five `ComputeTargets/tests` modules and
      `test_layer_is_generic.py`, with what each order needs, by line;
   7. the order with DatastoreKit;
   8. **the acceptance (gate G2)**;
   9. what goes stale.

   Its appendix is the measuring script's output.
2. [`docs/adoption/README.md`](/Users/ds283/Documents/Code/RayPoolKit/docs/adoption/README.md), the rules for every client: the pin, the module map, what
   changes and what does not for a caller, the order with DatastoreKit (§5), how a checklist is
   used (§6) and how to re-measure (§7).
3. [`README.md`](/Users/ds283/Documents/Code/RayPoolKit/README.md) (the package's): "Which handlers a pool needs", "Changes in
   `v0.2.0`", the printed lines and the supported versions.
4. [`PROVENANCE.md`](/Users/ds283/Documents/Code/RayPoolKit/PROVENANCE.md): how SGK's file maps onto the package, and what changed
   "Since `v0.1.0`".
5. [`docs/extraction-verification.md`](/Users/ds283/Documents/Code/RayPoolKit/docs/extraction-verification.md): §4 (the pool under real Ray),
   §6 (the five faults fixed, with the test that guards each), and §10 (what is unverified: no
   client's suite has been run against the package, and the fault-1 fix is shown only by the
   suite's stand-in).
6. **DatastoreKit's SGK checklist at `v0.2.1`**, needed for the order decision. Read it as
   `git -C /Users/ds283/Documents/Code/DatastoreKit show v0.2.1:docs/adoption/secondarygwkit.md`
   (its `:92`, `:230-232` and `:704` concern `RayTools/`), or at
   <https://github.com/ds283/DatastoreKit/blob/v0.2.1/docs/adoption/secondarygwkit.md>.
7. **SGK itself**, at the commit the campaign starts from:
   - its `CLAUDE.md`, for its conventions, its test command (`:106-109`) and the `RayTools/`
     paragraph (`:69-73`), which goes stale;
   - its `prompts/INDEX.md` and open-issue index, for anything in flight that touches `main.py` or
     the tests.

## 3. Decisions for the plan to put to the user

1. **The order with DatastoreKit** (checklist item 7; adoption README §5): `raypoolkit` first,
   DatastoreKit first, or both in one campaign. All three are allowed. Decision U12 lets a
   DatastoreKit-first adoption rewrite the one `ShardedPool` import in `RayTools/RayWorkPool.py:8`.
   The order decides what happens to `Datastore/tests/test_layer_is_generic.py`.
2. **The branch.** The checklist was measured on `handover-remedial`, not `main`. Settle which
   branch the adoption lands on, and re-measure there.
3. **The 44 lookup-only sites' `store_handler=None, persist_handler=None`.** Under `v0.2.0` they
   may stay or go. The checklist advises keeping them, so that the adoption's diff is the imports
   alone.
4. **Whether to measure `main.py:3247`.** Its post handler now gets the persisted object, not the
   local copy. By reading, nothing it records changes (checklist item 5). If the user wants that
   measured, compare the "GkSourcePolicyData SUMMARY STATISTICS" output in a scoped run before
   and after.
5. **The stale prose.** The checklist advises leaving SGK's dated `docs/` and `prompts/` records as
   they are. SGK's own rules govern.

## 4. Rules that come with the package

- **Pin the tag:** `raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0` in
  `requirements.txt`, in a commit of its own. Never an editable install for a production run.
- **Keep the bare name:** `from raypoolkit import RayWorkPool`, so that every call stays spelled
  `RayWorkPool(...)`. Five SGK test modules depend on that spelling.
- **The freeze holds until the adoption lands.** SGK does not edit `RayTools/` before then (U5),
  except the one import U12 allows.
- **The printed lines are an interface.** SGK's run registry parses the banner, progress and
  completion lines, and `v0.2.0` prints them as SGK's copy does. If they differ, report it as a
  defect.
- **A defect in the package is not patched in SGK.** Report it to RayPoolKit, where it is fixed
  in a new release.
- **No change is made in RayPoolKit or DatastoreKit by SGK's campaign.** Recording G2 on
  RayPoolKit's extraction board, once it holds, is a records commit in RayPoolKit, made there.

## 5. First steps

1. **Re-measure at the campaign's start commit**, from RayPoolKit's root with RayPoolKit's venv.
   The script reads SGK through `git` only:
   ```bash
   ./venv/bin/python docs/adoption/measure_client_pools.py sgk --commit <SHA>
   ```
   Compare with the checklist's appendix. If the commit is still `b510bc9`, the output is the
   appendix, byte for byte. Otherwise the new figures govern, and any change in the 7 imports, the
   54 sites or the 7 test modules goes into the plan.
2. **Check the freeze held:** `git -C <SGK> rev-parse <SHA>:RayTools/RayWorkPool.py` is `b4666fa…`.
   If it is not, the checklist's premise has moved; find out why before planning.
3. **Plan to checklist item 8**, the acceptance:
   - `RayTools/` gone;
   - 0 imports of it, by the script;
   - the pin in place, and SGK's venv installed from it;
   - SGK's six test roots passing;
   - every site constructing, by the script, with as many calls as SGK expects.
