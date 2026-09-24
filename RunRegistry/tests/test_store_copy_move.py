"""The registry's copy and move of a store, with its sidecar. `prompts/datastore-portability`
prompt 03 §2 P11 and §3 tests 5-8.

`ShardedPool.copy_store` / `move_store` (prompt 02) move a store's files and know nothing about
its sidecar. `RunRegistry.stores.copy_store` / `move_store` call them, then carry the sidecar: a
copy is a new store with a new `store_id`, and a move keeps the id and renames the one sidecar.
Both refuse a store that any `running` run names. The interruption tests back the table in
`logs/03-the-registry-owns-the-store-sidecar.md`, one test per row.

Stores are placeholder stores in a temporary directory, and every "does it open" question is asked
through the constructor's read-and-check on an `object.__new__` instance. No Ray is initialised
and nothing under `var/` is touched.
"""

import errno
import os
import shutil
from pathlib import Path
from unittest import mock

import RunRegistry
from RunRegistry import stores
from RunRegistry.tests.store_fixtures import (
    LAYOUTS,
    StoreTestCase,
    load,
    write_sidecar_json,
)

from Datastore.SQL.ShardedPool import ShardedPool
from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import tree_state


def _fail_when(real, predicate, error=None):
    """``real``, except that a call whose (source, destination) satisfies ``predicate`` raises."""

    def fake(a, b, *args, **kwargs):
        if predicate(str(a), str(b)):
            raise error or OSError(errno.EIO, "injected failure", str(b))
        return real(a, b, *args, **kwargs)

    return fake


def _fail_on_call(n, real):
    """``real``, except that its ``n``th call raises."""
    calls = {"n": 0}

    def fake(a, b, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise OSError(errno.EIO, "injected failure", str(b))
        return real(a, b, *args, **kwargs)

    return fake


class TestCopy(StoreTestCase):
    def test_the_destination_is_a_new_store_and_the_source_is_untouched(self):
        src = self.a_registry_store("B")
        source = load(stores.sidecar_path(src))
        src_dir = tree_state(src.parent)
        dst = self.top / "C" / "pcopy.sqlite"

        fields = stores.copy_store(src, dst, "a copy, for a test", runs_root=self.root)

        self.assertEqual(load(stores.sidecar_path(dst)), fields)
        self.assertRegex(fields["store_id"], r"^[0-9a-f]{32}$")
        self.assertNotEqual(fields["store_id"], source["store_id"])
        self.assertEqual(
            fields["copied_from"],
            {"store_id": source["store_id"], "datastore": RunRegistry._repo_path(src)},
        )
        self.assertEqual(fields["history"][:-1], source["history"])
        entry = fields["history"][-1]
        self.assertEqual(entry["operation"], "copy")
        self.assertEqual(entry["from"], RunRegistry._repo_path(src))
        self.assertEqual(entry["to"], RunRegistry._repo_path(dst))
        self.assertEqual(
            (fields["datastore"], fields["name"]), ("pcopy.sqlite", "pcopy")
        )
        self.assertEqual(fields["purpose"], "a copy, for a test")
        self.assertTrue(fields["created"])
        self.assertTrue(stores.read_sidecar(dst).ok)

        # the source's sidecar and store files: byte-identical, unchanged mtimes
        self.assertEqual(tree_state(src.parent), src_dir)
        self.assertEqual(load(stores.sidecar_path(src)), source)

        # the destination reads its own shards
        self.assertOpens(dst)
        self.assertEqual(
            sorted(os.listdir(dst.parent)),
            sorted(
                ["pcopy.sqlite", "pcopy.manifest.json"]
                + [shard_file_name(dst, i) for i in range(3)]
            ),
        )

    def test_a_copy_of_a_copy_names_its_immediate_parent(self):
        src = self.a_registry_store("B")
        middle = self.top / "C" / "middle.sqlite"
        last = self.top / "D" / "last.sqlite"
        first = stores.read_sidecar(src).store_id
        second = stores.copy_store(src, middle, "middle", runs_root=self.root)
        third = stores.copy_store(middle, last, "last", runs_root=self.root)
        self.assertEqual(third["copied_from"]["store_id"], second["store_id"])
        self.assertEqual(
            third["copied_from"]["datastore"], RunRegistry._repo_path(middle)
        )
        self.assertEqual(
            [e["operation"] for e in third["history"]], ["create", "copy", "copy"]
        )
        self.assertEqual(len({first, second["store_id"], third["store_id"]}), 3)


class TestMove(StoreTestCase):
    def test_in_each_layout(self):
        for layout in LAYOUTS:
            with self.subTest(layout):
                self._clear()
                src = self.a_registry_store("B")
                source = load(stores.sidecar_path(src))
                dst = self.destination(layout)

                fields = stores.move_store(src, dst, runs_root=self.root)

                self.assertEqual(load(stores.sidecar_path(dst)), fields)
                self.assertEqual(fields["store_id"], source["store_id"])
                self.assertEqual(fields["history"][:-1], source["history"])
                entry = fields["history"][-1]
                self.assertEqual(
                    (entry["operation"], entry["from"], entry["to"]),
                    ("move", RunRegistry._repo_path(src), RunRegistry._repo_path(dst)),
                )
                self.assertEqual(
                    (fields["datastore"], fields["name"]), (dst.name, dst.stem)
                )
                for kept in ("purpose", "created"):
                    self.assertEqual(fields[kept], source[kept])
                self.assertTrue(stores.read_sidecar(dst).ok)
                self.assertEqual(stores.read_sidecar(src).kind, "absent")
                # nothing but the destination's sidecar: no .tmp, no .incomplete-move
                self.assertEqual(self.leftovers(), [stores.sidecar_path(dst)])
                self.assertOpens(dst)

    def _clear(self):
        for name in ("B", "C"):
            shutil.rmtree(self.top / name, ignore_errors=True)


class TestRefusals(StoreTestCase):
    """One test per refusal in P11. Each asserts that the error names the offending file or run
    and that nothing was created, changed or removed anywhere under the temporary directory.
    """

    def refused(self, operation, src, dst, *fragments):
        before = self.state()
        with self.assertRaises(RuntimeError) as refused:
            if operation == "copy":
                stores.copy_store(src, dst, "a copy", runs_root=self.root)
            else:
                stores.move_store(src, dst, runs_root=self.root)
        self.assertNamesAndNothingChanged(before, refused.exception, *fragments)
        return str(refused.exception)

    def test_a_source_sidecar_that_is_not_a_problem_free_registry_sidecar(self):
        primary = self.a_store("B")
        path = stores.sidecar_path(primary)
        dst = self.top / "C" / "c.sqlite"
        for label, arrange, fragment in (
            ("absent", lambda: None, "there is none"),
            (
                "legacy",
                lambda: write_sidecar_json(primary, {"name": "store", "purpose": "p"}),
                "legacy",
            ),
            ("unreadable", lambda: path.write_text("{torn"), "unreadable"),
            (
                "a registry sidecar with a problem",
                lambda: (
                    path.unlink(),
                    stores.create_sidecar(primary, "p"),
                    path.write_text(
                        path.read_text().replace('"store.sqlite"', '"other.sqlite"')
                    ),
                ),
                "does not name",
            ),
        ):
            for operation in ("copy", "move"):
                with self.subTest(label, operation=operation):
                    if operation == "copy":
                        arrange()
                    message = self.refused(operation, primary, dst, path, fragment)
                    self.assertIn("store adopt", message)

    def test_a_running_run_that_is_alive_names_the_source_by_path(self):
        src = self.a_registry_store()
        run = self.a_running_run("alive", src)
        for operation in ("copy", "move"):
            with self.subTest(operation):
                self.refused(
                    operation, src, self.top / "C" / "c.sqlite", run.id, "alive"
                )

    def test_a_running_run_that_is_stale_still_refuses(self):
        src = self.a_registry_store()
        run = self.a_stale_run("stale", src)
        self.assertEqual(RunRegistry.list_runs(root=self.root)[0]["liveness"], "stale")
        for operation in ("copy", "move"):
            with self.subTest(operation):
                self.refused(
                    operation,
                    src,
                    self.top / "C" / "c.sqlite",
                    run.id,
                    "stale",
                    "Run.finish",
                )

    def test_a_match_by_results_store_id_alone(self):
        """The run was begun against the store where it was. The store's directory was then moved
        by hand, which works since prompt 01, so its `results` path names somewhere else. Only
        the store_id says it is the same store."""
        old = self.a_registry_store("B")
        run = self.a_running_run("by-id", old)
        store_id = stores.read_sidecar(old).store_id
        self.assertEqual(run.manifest["results_store_id"], store_id)
        os.rename(self.top / "B", self.top / "B-moved-by-hand")
        src = self.top / "B-moved-by-hand" / "store.sqlite"
        self.assertEqual(stores.read_sidecar(src).store_id, store_id)
        self.assertNotEqual(os.path.realpath(run.results_path), os.path.realpath(src))

        for operation in ("copy", "move"):
            with self.subTest(operation):
                message = self.refused(
                    operation,
                    src,
                    self.top / "C" / "c.sqlite",
                    run.id,
                    "results_store_id",
                )
                self.assertNotIn("its results and", message)

    def test_a_running_run_that_names_the_destination(self):
        src = self.a_registry_store()
        dst = self.top / "C" / "c.sqlite"
        run = self.a_running_run("destination", dst)
        for operation in ("copy", "move"):
            with self.subTest(operation):
                self.refused(operation, src, dst, run.id)

    def test_a_run_begun_before_the_sidecar_is_matched_by_its_path(self):
        src = self.a_store()
        run = self.a_running_run("before-the-sidecar", src)
        self.assertIsNone(run.manifest["results_store_id"])
        stores.create_sidecar(src, "created after the run began")
        self.refused("copy", src, self.top / "C" / "c.sqlite", run.id, "by its results")

    def test_a_finished_run_does_not_refuse(self):
        src = self.a_registry_store()
        for state in ("done", "failed", "killed"):
            self.a_running_run(f"finished-{state}", src).finish(state)
        stores.copy_store(src, self.top / "C" / "c.sqlite", "p", runs_root=self.root)
        self.assertTrue(stores.read_sidecar(self.top / "C" / "c.sqlite").ok)
        stores.move_store(src, self.top / "D" / "d.sqlite", runs_root=self.root)
        self.assertTrue(stores.read_sidecar(self.top / "D" / "d.sqlite").ok)

    def test_a_runs_root_that_is_not_there(self):
        src = self.a_registry_store()
        before = self.state()
        with self.assertRaises(RuntimeError) as refused:
            stores.copy_store(
                src,
                self.top / "C" / "c.sqlite",
                "p",
                runs_root=str(self.top / "no-runs"),
            )
        self.assertNamesAndNothingChanged(
            before, refused.exception, "no-runs", "not a directory"
        )

    def test_the_destination_sidecar_names_exist(self):
        src = self.a_registry_store()
        dst = self.top / "C" / "c.sqlite"
        sidecar = stores.sidecar_path(dst)
        names = {
            "copy": [sidecar, Path(str(sidecar) + ".tmp")],
            "move": [
                sidecar,
                Path(str(sidecar) + ".tmp"),
                Path(str(sidecar) + stores.INCOMPLETE_MOVE_SUFFIX),
                Path(str(sidecar) + stores.INCOMPLETE_MOVE_SUFFIX + ".tmp"),
            ],
        }
        for operation, taken in names.items():
            for name in taken:
                with self.subTest(operation=operation, name=name.name):
                    name.parent.mkdir(exist_ok=True)
                    name.write_text("{}")
                    try:
                        self.refused(operation, src, dst, name, "already exist")
                    finally:
                        name.unlink()

    def test_a_copy_needs_its_own_purpose(self):
        src = self.a_registry_store()
        for purpose in ("", "  ", None):
            with self.subTest(purpose=purpose):
                before = self.state()
                with self.assertRaises(RuntimeError) as refused:
                    stores.copy_store(
                        src, self.top / "C" / "c.sqlite", purpose, runs_root=self.root
                    )
                self.assertNamesAndNothingChanged(before, refused.exception, "purpose")

    def test_prompt_02s_refusals_pass_through(self):
        src = self.a_registry_store()
        # a destination primary that exists, with no sidecar: prompt 02's refusal
        dst = self.top / "C" / "c.sqlite"
        dst.parent.mkdir()
        dst.write_text("somebody else's file")
        for operation in ("copy", "move"):
            with self.subTest(operation):
                message = self.refused(operation, src, dst, dst, "already exist")
                self.assertIn(f"Cannot {operation} sharded datastore", message)
                self.assertIn("Nothing was written", message)
                self.assertIn("No sidecar was written or moved", message)
        dst.unlink()
        # a hot journal beside the source
        journal = Path(str(src) + "-journal")
        journal.write_text("")
        message = self.refused("move", src, dst, journal, "not closed cleanly")
        self.assertIn("No sidecar was written or moved", message)


class TestInterruption(StoreTestCase):
    """The log's interruption table, one test method per row, every layout a subtest.

    In each row the step after the named one is made to fail, and the state is what a process
    that died there would leave: nothing is cleaned up. `assertProperty` checks prompt §2 P11's
    property at both store names, and each row checks what its line of the table says.
    """

    def _each_layout(self, operation, patcher, check):
        """Run ``operation`` in each layout under ``patcher``, which must make it fail, then call
        ``check(src, dst, the source's sidecar before, the error)`` inside that layout's subtest,
        so that a failed assertion is reported against its layout and the others still run.
        """
        for layout in LAYOUTS:
            with self.subTest(layout):
                for name in ("B", "C"):
                    shutil.rmtree(self.top / name, ignore_errors=True)
                src = self.a_registry_store("B")
                before = load(stores.sidecar_path(src))
                dst = self.destination(layout)
                with patcher(src, dst):
                    with self.assertRaises(RuntimeError) as failed:
                        if operation == "copy":
                            stores.copy_store(src, dst, "a copy", runs_root=self.root)
                        else:
                            stores.move_store(src, dst, runs_root=self.root)
                check(src, dst, before, str(failed.exception))

    # --- copy

    def test_C1_the_store_copy_fails_before_writing(self):
        patcher = lambda src, dst: mock.patch.object(
            ShardedPool, "copy_store", side_effect=RuntimeError("injected")
        )

        def check(src, dst, before, message):
            self.assertIn("No sidecar was written or moved", message)
            self.assertEqual(load(stores.sidecar_path(src)), before)
            self.assertOpens(src)
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertFalse(dst.exists())
            self.assertProperty(src, dst, {src: 1})

        self._each_layout("copy", patcher, check)

    def test_C2_the_store_copy_fails_part_way(self):
        real = shutil.copy2
        patcher = lambda src, dst: mock.patch("shutil.copy2", _fail_on_call(2, real))

        def check(src, dst, before, message):
            self.assertIn('failed at step "copy shard #1"', message)
            self.assertIn("No sidecar was written or moved", message)
            self.assertEqual(load(stores.sidecar_path(src)), before)
            self.assertOpens(src)
            # prompt 02's state C3: shards and no primary, which the constructor refuses
            self.assertFalse(os.path.lexists(dst))
            self.assertTrue(os.path.lexists(dst.parent / shard_file_name(dst, 0)))
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertProperty(src, dst, {src: 1})

        self._each_layout("copy", patcher, check)

    def test_C3_the_store_is_copied_and_the_sidecar_write_fails(self):
        patcher = lambda src, dst: mock.patch.object(
            stores, "write_json_atomic", side_effect=OSError(errno.EIO, "injected")
        )

        def check(src, dst, before, message):
            self.assertIn("write the destination's sidecar", message)
            self.assertIn(str(dst), message)
            self.assertOpens(dst)
            self.assertOpens(src)
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertFalse(os.path.lexists(str(stores.sidecar_path(dst)) + ".tmp"))
            self.assertEqual(load(stores.sidecar_path(src)), before)
            self.assertProperty(src, dst, {src: 1})

        self._each_layout("copy", patcher, check)

    def test_C4_the_sidecar_is_written_to_its_tmp_name_and_the_replace_fails(self):
        real = os.replace
        patcher = lambda src, dst: mock.patch(
            "os.replace",
            _fail_when(real, lambda a, b: b.endswith(".manifest.json")),
        )

        def check(src, dst, before, message):
            tmp = str(stores.sidecar_path(dst)) + ".tmp"
            self.assertIn(tmp, message)  # the failure lists the .tmp file
            self.assertTrue(os.path.exists(tmp))  # not a sidecar name, so not reachable
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertOpens(dst)
            self.assertEqual(load(stores.sidecar_path(src)), before)
            self.assertProperty(src, dst, {src: 1})
            # and a second attempt is refused: the .tmp name is taken, and the store is there
            with self.assertRaises(RuntimeError):
                stores.copy_store(src, dst, "again", runs_root=self.root)

        self._each_layout("copy", patcher, check)

    def test_C5_complete(self):
        for layout in LAYOUTS:
            with self.subTest(layout):
                for name in ("B", "C"):
                    shutil.rmtree(self.top / name, ignore_errors=True)
                src = self.a_registry_store("B")
                dst = self.destination(layout)
                stores.copy_store(src, dst, "a copy", runs_root=self.root)
                self.assertProperty(src, dst, {src: 1, dst: 2})
                self.assertOpens(src)
                self.assertOpens(dst)

    # --- move

    def test_M1_the_store_move_fails_before_writing(self):
        patcher = lambda src, dst: mock.patch.object(
            ShardedPool, "move_store", side_effect=RuntimeError("injected")
        )

        def check(src, dst, before, message):
            self.assertIn("No sidecar was written or moved", message)
            self.assertEqual(load(stores.sidecar_path(src)), before)
            self.assertOpens(src)
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertProperty(src, dst, {src: 1})

        self._each_layout("move", patcher, check)

    def test_M2_the_store_move_fails_after_a_shard_is_renamed(self):
        real = os.rename
        patcher = lambda src, dst: mock.patch("os.rename", _fail_on_call(2, real))

        def check(src, dst, before, message):
            self.assertIn('failed at step "rename shard #1"', message)
            # the source sidecar still describes the store beside it, which prompt 02's check
            # refuses because shard #0 has gone
            self.assertEqual(load(stores.sidecar_path(src)), before)
            self.assertTrue(stores.read_sidecar(src).ok)
            self.assertRefusedStore(src)
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertProperty(src, dst, {src: 1})

        self._each_layout("move", patcher, check)

    def test_M3_the_store_move_fails_after_the_primary_is_renamed(self):
        patcher = lambda src, dst: mock.patch.object(
            ShardedPool,
            "_write_shard_records",
            side_effect=RuntimeError("injected before the rewrite"),
        )

        def check(src, dst, before, message):
            self.assertIn("rewrite the destination primary's shards rows", message)
            reading = stores.read_sidecar(src)
            self.assertEqual(reading.fields, before)
            self.assertIn("does not exist", "; ".join(reading.problems))  # orphaned
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertProperty(src, dst, {})

        self._each_layout("move", patcher, check)

    def test_M4_the_store_is_moved_and_the_rename_to_the_temporary_name_fails(self):
        real = os.rename
        patcher = lambda src, dst: mock.patch(
            "os.rename",
            _fail_when(real, lambda a, b: b.endswith(stores.INCOMPLETE_MOVE_SUFFIX)),
        )

        def check(src, dst, before, message):
            self.assertIn("rename the source's sidecar to its temporary name", message)
            self.assertOpens(dst)
            reading = stores.read_sidecar(src)
            self.assertEqual(reading.fields, before)
            self.assertIn("does not exist", "; ".join(reading.problems))
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertEqual(self.leftovers(), [stores.sidecar_path(src)])
            self.assertProperty(src, dst, {})

        self._each_layout("move", patcher, check)

    def test_M5_the_sidecar_is_at_its_temporary_name_and_the_update_fails(self):
        patcher = lambda src, dst: mock.patch.object(
            stores, "_update_sidecar", side_effect=RuntimeError("injected")
        )

        def check(src, dst, before, message):
            # first: an un-updated sidecar never sits under a sidecar name. In the "new
            # directory, same stem" layout it would pass every check the reader makes and still
            # lack its move entry, which is why the update happens under a temporary name
            self.assertProperty(src, dst, {})
            pending = Path(
                str(stores.sidecar_path(dst)) + stores.INCOMPLETE_MOVE_SUFFIX
            )
            self.assertIn('"update the temporary sidecar"', message)
            self.assertIn(str(pending), message)
            self.assertEqual(stores.read_sidecar(src).kind, "absent")
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertEqual(load(pending), before)  # not updated, and not reachable
            self.assertEqual(self.leftovers(), [pending])
            self.assertOpens(dst)

        self._each_layout("move", patcher, check)

    def test_M6_the_update_is_written_to_its_tmp_name_and_the_replace_fails(self):
        real = os.replace
        patcher = lambda src, dst: mock.patch(
            "os.replace",
            _fail_when(real, lambda a, b: b.endswith(stores.INCOMPLETE_MOVE_SUFFIX)),
        )

        def check(src, dst, before, message):
            pending = Path(
                str(stores.sidecar_path(dst)) + stores.INCOMPLETE_MOVE_SUFFIX
            )
            tmp = Path(str(pending) + ".tmp")
            self.assertEqual(load(pending), before)
            self.assertEqual(len(load(tmp)["history"]), 2)
            self.assertEqual(sorted(self.leftovers()), sorted([pending, tmp]))
            self.assertEqual(stores.read_sidecar(src).kind, "absent")
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertProperty(src, dst, {})

        self._each_layout("move", patcher, check)

    def test_M7_the_sidecar_is_updated_and_the_final_rename_fails(self):
        real = os.rename
        patcher = lambda src, dst: mock.patch(
            "os.rename",
            _fail_when(real, lambda a, b: a.endswith(stores.INCOMPLETE_MOVE_SUFFIX)),
        )

        def check(src, dst, before, message):
            pending = Path(
                str(stores.sidecar_path(dst)) + stores.INCOMPLETE_MOVE_SUFFIX
            )
            self.assertIn("rename the temporary sidecar", message)
            updated = load(pending)
            self.assertEqual(updated["store_id"], before["store_id"])
            self.assertEqual(
                [e["operation"] for e in updated["history"]], ["create", "move"]
            )
            self.assertEqual(self.leftovers(), [pending])
            self.assertEqual(stores.read_sidecar(src).kind, "absent")
            self.assertEqual(stores.read_sidecar(dst).kind, "absent")
            self.assertProperty(src, dst, {})
            # a person finishes the move by renaming it, and it then describes the store
            os.rename(pending, stores.sidecar_path(dst))
            self.assertDescribes(dst, before["store_id"], 2)

        self._each_layout("move", patcher, check)

    def test_M8_complete(self):
        for layout in LAYOUTS:
            with self.subTest(layout):
                for name in ("B", "C"):
                    shutil.rmtree(self.top / name, ignore_errors=True)
                src = self.a_registry_store("B")
                store_id = stores.read_sidecar(src).store_id
                dst = self.destination(layout)
                stores.move_store(src, dst, runs_root=self.root)
                self.assertProperty(src, dst, {dst: 2})
                self.assertDescribes(dst, store_id, 2)
                self.assertEqual(self.leftovers(), [stores.sidecar_path(dst)])
