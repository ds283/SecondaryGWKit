"""What the references report of `store retire` counts as naming a store (a3-v2-readiness prompt 06,
closing `[10-store-retire-reports-every-tombstone-as-a-reference]`).

`RunRegistry.stores._find_references` takes the primary, every file in the plan and the store's
directory as targets. Every tombstone records `references.stores_root`, the directory its
retirement searched, so for a store directly in the stores root the directory is the stores root,
and each retirement listed every earlier tombstone as naming it. The stores root is therefore not a
target; a store's own subdirectory, the live A3 backup's shape, still is.

1. three stores directly in the stores root, retired in turn: no report lists an earlier tombstone,
   through a stores root that is a symlink too (the comparison is by `realpath`, not by text);
2. a sidecar that is not a tombstone, with a field naming the stores root: not reported, while a
   field of the same sidecar naming the primary is;
3. a store in a subdirectory, named by another sidecar's field by that subdirectory: reported, a
   directory it shares with another store included, and a store outside the stores root likewise.

`test_store_retire.TestRetirement.test_a_retirement`, unchanged, also requires the subdirectory
match. Stores are the real multi-shard fixture stores of `test_store_retire`, in a temporary
directory; no Ray is initialised and nothing under `var/` is opened.
"""

import os
from pathlib import Path

import RunRegistry
from RunRegistry import stores
from RunRegistry.tests import test_store_retire
from RunRegistry.tests.store_fixtures import write_sidecar_json
from RunRegistry.tests.test_store_command_line import REPO_ROOT

_repo_path = RunRegistry._repo_path


class ReferencesTestCase(test_store_retire.RetireTestCase):
    def a_store_in_the_root(self, stem) -> Path:
        """A full store directly in the stores root, which is where `datastores/` holds most."""
        primary = self.a_full_store("", stem=stem)
        self.assertEqual(os.path.realpath(primary.parent), self.stores_root)
        return primary

    def naming(self, sidecar_for, **fields) -> Path:
        """A registry store, directly in the stores root, whose sidecar gains ``notes`` holding
        ``fields``: the one place a sidecar of this module names something."""
        primary = self.a_registry_store("datastores", stem=sidecar_for)
        sidecar = self.sidecar(primary)
        sidecar["notes"] = fields
        write_sidecar_json(primary, sidecar)
        return primary

    def reported(self, result) -> dict:
        """``{sidecar path: its matching fields}`` of a retirement's report, nothing unreadable."""
        found = {}
        for entry in result["references"]["sidecars"]:
            self.assertNotIn("unreadable", entry, entry)
            found[entry["sidecar"]] = entry["fields"]
        return found


class TestTheStoresRootIsNotATarget(ReferencesTestCase):
    def assertNoEarlierTombstoneListed(self, earlier, result):
        """Each of ``earlier`` is a tombstone that records the stores root, and none is listed."""
        reported = self.reported(result)
        for primary in earlier:
            tombstone = self.sidecar(primary)
            # a precondition: the field that used to match is there, naming this very root
            recorded = tombstone["retired"]["references"]["stores_root"]
            self.assertEqual(
                os.path.realpath(os.path.join(REPO_ROOT, recorded)),
                os.path.realpath(self.stores_root),
            )
            self.assertNotIn(_repo_path(stores.sidecar_path(primary)), reported)
        self.assertEqual(reported, {})

    def test_three_stores_in_the_root_retired_in_turn(self):
        primaries = [self.a_store_in_the_root(f"s{i}") for i in range(3)]
        for n, primary in enumerate(primaries):
            result = self.retire(primary)
            self.assertNoEarlierTombstoneListed(primaries[:n], result)
            self.assertEqual(result["tombstone"]["references"]["sidecars"], [])

    def test_through_a_stores_root_that_is_a_symlink(self):
        """The root that is compared is the root that is searched: `os.walk` follows a symlinked
        root, so a comparison of the text of the two paths would not see them as one."""
        link = self.top / "link"
        link.symlink_to(self.stores_root, target_is_directory=True)
        primaries = [self.a_store_in_the_root(f"s{i}") for i in range(3)]
        # one retired by the real root, then two through the link, then the link again
        first = self.retire(primaries[0])
        self.assertNoEarlierTombstoneListed([], first)
        for n in (1, 2):
            result = stores.retire_store(
                primaries[n],
                test_store_retire.REASON,
                runs_root=self.root,
                stores_root=str(link),
            )
            self.assertNoEarlierTombstoneListed(primaries[:n], result)

    def test_through_a_primary_that_is_reached_by_a_symlink(self):
        link = self.top / "link"
        link.symlink_to(self.stores_root, target_is_directory=True)
        primaries = [self.a_store_in_the_root(f"s{i}") for i in range(2)]
        self.retire(primaries[0])
        result = self.retire(link / primaries[1].name)
        self.assertNoEarlierTombstoneListed(primaries[:1], result)

    def test_a_sidecar_that_is_not_a_tombstone_naming_the_root(self):
        primary = self.a_store_in_the_root("S")
        plain = self.naming(
            "plain",
            root=self.stores_root,
            relative=os.path.relpath(self.stores_root, REPO_ROOT),
        )
        self.assertEqual(self.reported(self.retire(primary)), {})
        # the sidecar was not rewritten, and is still a store
        self.assertTrue(stores.read_sidecar(plain).ok)

    def test_the_same_sidecar_is_reported_for_a_field_naming_the_primary(self):
        primary = self.a_store_in_the_root("S")
        plain = self.naming(
            "plain",
            root=self.stores_root,
            primary=str(primary),
            relative=os.path.relpath(self.stores_root, REPO_ROOT),
        )
        reported = self.reported(self.retire(primary))
        self.assertEqual(
            reported, {_repo_path(stores.sidecar_path(plain)): ["$.notes.primary"]}
        )


class TestADirectoryIsStillATarget(ReferencesTestCase):
    def test_a_store_in_a_subdirectory_named_by_the_subdirectory(self):
        """The live A3 backup's shape: another sidecar names the directory, by its path and by a
        path relative to the repository; and also names the stores root, which is not reported.
        """
        primary = self.a_full_store("S")
        named = self.naming(
            "named",
            directory=str(primary.parent),
            relative=os.path.relpath(primary.parent, REPO_ROOT),
            root=self.stores_root,
        )
        bystander = self.naming("bystander", root=self.stores_root)
        reported = self.reported(self.retire(primary))
        self.assertEqual(
            reported,
            {
                _repo_path(stores.sidecar_path(named)): [
                    "$.notes.directory",
                    "$.notes.relative",
                ]
            },
        )
        self.assertNotIn(_repo_path(stores.sidecar_path(bystander)), reported)

    def test_a_subdirectory_shared_with_another_store_is_still_a_match(self):
        primary = self.a_full_store("S")
        self.a_full_store("S", stem="other")
        named = self.naming("named", directory=str(primary.parent))
        reported = self.reported(self.retire(primary))
        self.assertEqual(
            reported, {_repo_path(stores.sidecar_path(named)): ["$.notes.directory"]}
        )

    def test_a_tombstone_in_the_root_does_not_name_a_subdirectory(self):
        """A store retired directly in the root leaves a tombstone naming the root; a store in a
        subdirectory, retired afterwards, does not find it."""
        in_root = self.a_store_in_the_root("R")
        self.retire(in_root)
        primary = self.a_full_store("S")
        self.assertEqual(self.reported(self.retire(primary)), {})

    def test_a_store_outside_the_stores_root_keeps_its_directory(self):
        elsewhere = self.a_full_store("E", inside=False)
        named = self.naming("named", directory=str(elsewhere.parent))
        reported = self.reported(self.retire(elsewhere))
        self.assertEqual(
            reported, {_repo_path(stores.sidecar_path(named)): ["$.notes.directory"]}
        )

    def test_a_store_in_the_root_is_still_found_by_what_names_it_itself(self):
        """The root is not a target; the primary and the copy's `copied_from` still match."""
        primary = self.a_store_in_the_root("S")
        copied = Path(self.stores_root) / "C" / "copy.sqlite"
        stores.copy_store(primary, copied, "a copy of S", runs_root=self.root)
        reported = self.reported(self.retire(primary))
        # the copy names the primary in several fields besides `store_id`, which is the one asked for
        self.assertEqual(list(reported), [_repo_path(stores.sidecar_path(copied))])
        (fields,) = reported.values()
        self.assertIn("$.copied_from.store_id", fields)
        self.assertIn("$.copied_from.datastore", fields)
