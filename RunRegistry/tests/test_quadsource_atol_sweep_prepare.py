"""`docs/handover/quadsource_atol_sweep.py` `prepare()` and `assert_store_is_self_consistent()`,
for `store-retirement` prompt 02 (**R4**, **R5**).

`prepare()` now makes the sweep store with one `RunRegistry.stores.copy_store` call, so that a
tombstone left behind by a future `store retire` is never silently overwritten
(`docs/store-retirement-audit.md` §2.6). `assert_store_is_self_consistent` now compares each
`shards` row, serial by serial, through `Datastore.shard_paths.resolve_shard_path`, which is what
`ShardedPool` itself has read every record through since `datastore-portability` prompt 01, and
what unblocks `--build --resume` of a store built since then.

The script is imported by path with `importlib.util`; its module-level imports are light
(`argparse`, `json`, `numpy`, the standard library), so this costs nothing. `BASELINE_STORE` and
`SWEEP_STORE` are monkeypatched onto a fresh copy of the module, at stores in one temporary
directory per test, built with `Datastore/tests/shard_store_fixtures.py` (no Ray, no real
``ShardedPool`` actors). `RunRegistry.stores.DEFAULT_ROOT` is monkeypatched at the same store, so
that the running-run check `copy_store` makes reads a temporary runs root and never
``var/runs/``. Nothing here calls `ShardedPool` through Ray, and nothing here opens anything
under `var/`.
"""

import importlib.util
from pathlib import Path
from unittest import mock

import RunRegistry
from RunRegistry import stores
from RunRegistry.tests.store_fixtures import StoreTestCase

from Datastore.shard_paths import shard_file_name
from Datastore.tests.shard_store_fixtures import (
    stored_records,
    tree_state,
    write_legacy_primary,
    write_new_store,
    write_placeholder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "docs" / "handover" / "quadsource_atol_sweep.py"

# the purpose text `prepare()` passes to `copy_store`, copied verbatim from the script so that a
# test failure here means the script's text changed, not that this test guessed wrong
SWEEP_PURPOSE = (
    "Working copy of the A3 baseline store for "
    "docs/handover/quadsource_atol_sweep.py. Disposable: every row that is not "
    "at the production tolerance pair belongs to a sweep and nothing else reads "
    "it. The comparator is handover-A3-baseline-lambdacdm, not this."
)


def _load_script():
    """A fresh import of the script, isolated from every other test's monkeypatching."""
    spec = importlib.util.spec_from_file_location(
        "_quadsource_atol_sweep_under_test", str(SCRIPT_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SweepPrepareTestCase(StoreTestCase):
    """One temporary directory (`self.top`), a runs root inside it (`self.root`), and a fresh
    import of the script with `BASELINE_STORE` / `SWEEP_STORE` pointed into that directory and
    `RunRegistry.stores.DEFAULT_ROOT` pointed at the temporary runs root."""

    def setUp(self):
        super().setUp()
        self.sweep = _load_script()
        self.baseline = self.top / "A" / "baseline.sqlite"
        self.sweep_store = self.top / "B" / "sweep.sqlite"
        self.sweep.BASELINE_STORE = self.baseline
        self.sweep.SWEEP_STORE = self.sweep_store
        patcher = mock.patch.object(stores, "DEFAULT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- fixtures, at the script's own shard count -----------------------------------------

    def shard_paths_of(self, primary: Path):
        return [
            primary.parent / shard_file_name(primary, i)
            for i in range(self.sweep.SHARDS)
        ]

    def a_baseline(self) -> Path:
        write_new_store(self.baseline, shards=self.sweep.SHARDS)
        stores.create_sidecar(
            self.baseline, "baseline store for a test, in a temporary directory"
        )
        return self.baseline

    def a_tombstone_at_sweep_name(self) -> Path:
        """A sidecar at the sweep name with no primary and no shard: what a retired store looks
        like to today's reader (`docs/store-retirement-audit.md` §2.1)."""
        write_new_store(self.sweep_store, shards=self.sweep.SHARDS)
        stores.create_sidecar(
            self.sweep_store, "a store that has since been retired, in a test"
        )
        for path in [self.sweep_store] + self.shard_paths_of(self.sweep_store):
            path.unlink()
        return stores.sidecar_path(self.sweep_store)

    def an_existing_sweep_store(self) -> Path:
        write_new_store(self.sweep_store, shards=self.sweep.SHARDS)
        stores.create_sidecar(self.sweep_store, "an existing sweep store, in a test")
        return self.sweep_store

    def dir_state(self, name: str):
        return tree_state(self.top / name)

    # --- test 1 -----------------------------------------------------------------------------

    def test_a_fresh_sweep_name_is_made_through_the_registry(self):
        baseline = self.a_baseline()
        baseline_id = stores.read_sidecar(baseline).fields["store_id"]
        before = self.dir_state("A")

        self.sweep.prepare()

        self.assertEqual(self.dir_state("A"), before, "prepare() touched the source")

        records = stored_records(self.sweep_store)
        self.assertEqual(
            records,
            {i: shard_file_name(self.sweep_store, i) for i in range(self.sweep.SHARDS)},
            "the sweep store's shards rows are not bare names",
        )

        reading = stores.read_sidecar(self.sweep_store)
        self.assertTrue(reading.ok, (reading.kind, reading.problems))
        self.assertNotEqual(reading.fields["store_id"], baseline_id)
        self.assertEqual(
            reading.fields["copied_from"],
            {"store_id": baseline_id, "datastore": RunRegistry._repo_path(baseline)},
        )
        self.assertEqual(reading.fields["purpose"], SWEEP_PURPOSE)
        self.assertEqual(reading.fields["history"][-1]["operation"], "copy")

        # the self-consistency check passes on what prepare() just made
        self.sweep.assert_store_is_self_consistent(self.sweep_store)

    # --- test 2 -------------------------------------------------------------------------------

    def test_a_tombstone_at_the_sweep_name_is_never_overwritten(self):
        self.a_baseline()
        sidecar_path = self.a_tombstone_at_sweep_name()
        sidecar_before = sidecar_path.read_bytes()
        baseline_before = self.dir_state("A")

        with self.assertRaises(RuntimeError) as ctx:
            self.sweep.prepare()
        self.assertIn("already exist", str(ctx.exception))
        self.assertEqual(
            sidecar_path.read_bytes(), sidecar_before, "the tombstone was rewritten"
        )
        self.assertFalse(
            self.sweep_store.exists(), "a store file was created at a tombstone"
        )
        self.assertEqual(
            [p.name for p in self.shard_paths_of(self.sweep_store) if p.exists()],
            [],
            "a shard file was created at a tombstone",
        )
        self.assertEqual(self.dir_state("A"), baseline_before)

        # again, with force=True: refused before anything is read
        with self.assertRaises(RuntimeError) as ctx2:
            self.sweep.prepare(force=True)
        self.assertIn("no longer means", str(ctx2.exception))
        self.assertEqual(sidecar_path.read_bytes(), sidecar_before)
        self.assertFalse(self.sweep_store.exists())
        self.assertEqual(self.dir_state("A"), baseline_before)

    # --- test 3 -------------------------------------------------------------------------------

    def test_an_existing_sweep_store_is_never_replaced(self):
        self.a_baseline()
        self.an_existing_sweep_store()
        before = self.state()

        with self.assertRaises(RuntimeError) as ctx:
            self.sweep.prepare()
        self.assertIn("already exist", str(ctx.exception))
        self.assertEqual(self.state(), before)

        with self.assertRaises(RuntimeError) as ctx2:
            self.sweep.prepare(force=True)
        self.assertIn("no longer means", str(ctx2.exception))
        self.assertEqual(self.state(), before)

    # --- test 4 -------------------------------------------------------------------------------

    def test_the_check_accepts_bare_names(self):
        primary = self.top / "C" / "store.sqlite"
        write_new_store(primary, shards=self.sweep.SHARDS)
        self.sweep.assert_store_is_self_consistent(primary)  # must not raise

    def test_the_check_accepts_a_legacy_store_naming_its_own_siblings(self):
        primary = self.top / "D" / "store.sqlite"
        shards = self.shard_paths_of(primary)
        for path in shards:
            write_placeholder(path)
        write_legacy_primary(
            primary, {i: str(p.resolve()) for i, p in enumerate(shards)}
        )
        self.sweep.assert_store_is_self_consistent(primary)  # must not raise

    def test_the_check_accepts_the_backups_shape(self):
        """A legacy primary whose absolute records name the same file names in another existing
        directory: the resolver reads *this* store's own siblings, not the directory named in the
        stored string, which is exactly the backup's shape
        (`docs/store-retirement-audit.md` §2.7): the backup's primary holds the **live** store's
        absolute shard paths, byte for byte, and a correct reader must still delete only the
        backup's own four files, never the live store's."""
        live = self.top / "X" / "store.sqlite"
        write_new_store(live, shards=self.sweep.SHARDS)  # "another existing directory"

        backup = self.top / "Y" / "store.sqlite"
        backup_shards = self.shard_paths_of(backup)
        for path in backup_shards:
            write_placeholder(path)
        records = {
            i: str((live.parent / shard_file_name(live, i)).resolve())
            for i in range(self.sweep.SHARDS)
        }
        write_legacy_primary(backup, records)

        self.sweep.assert_store_is_self_consistent(backup)  # must not raise

    def test_the_check_refuses_a_renamed_record(self):
        primary = self.top / "E" / "store.sqlite"
        shards = self.shard_paths_of(primary)
        for path in shards:
            write_placeholder(path)
        records = {i: str(p.resolve()) for i, p in enumerate(shards)}
        records[0] = str((primary.parent / "not-the-right-name.sqlite").resolve())
        write_legacy_primary(primary, records)

        with self.assertRaises(RuntimeError) as ctx:
            self.sweep.assert_store_is_self_consistent(primary)
        message = str(ctx.exception)
        self.assertIn("serial 0", message)
        self.assertIn("not-the-right-name.sqlite", message)
        self.assertIn(str(shards[0]), message)

    def test_the_check_refuses_a_missing_serial(self):
        primary = self.top / "F" / "store.sqlite"
        shards = self.shard_paths_of(primary)
        for path in shards:
            write_placeholder(path)
        records = {i: str(p.resolve()) for i, p in enumerate(shards)}
        del records[self.sweep.SHARDS - 1]
        write_legacy_primary(primary, records)

        with self.assertRaises(RuntimeError) as ctx:
            self.sweep.assert_store_is_self_consistent(primary)
        self.assertIn(
            f"missing serial(s): [{self.sweep.SHARDS - 1}]", str(ctx.exception)
        )

    def test_the_check_refuses_an_extra_serial(self):
        primary = self.top / "G" / "store.sqlite"
        shards = self.shard_paths_of(primary)
        for path in shards:
            write_placeholder(path)
        extra = primary.parent / shard_file_name(primary, self.sweep.SHARDS)
        write_placeholder(extra)
        records = {i: str(p.resolve()) for i, p in enumerate(shards)}
        records[self.sweep.SHARDS] = str(extra.resolve())
        write_legacy_primary(primary, records)

        with self.assertRaises(RuntimeError) as ctx:
            self.sweep.assert_store_is_self_consistent(primary)
        self.assertIn(f"extra serial(s)", str(ctx.exception))
        self.assertIn(f"[{self.sweep.SHARDS}]", str(ctx.exception))

    def test_the_check_refuses_two_swapped_serials(self):
        """Deliberate-breakage case (iv): serials 0 and 1 hold each other's record. The *set* of
        resolved paths is exactly the expected set, so a check that compared sets rather than
        serials would wrongly accept this."""
        primary = self.top / "H" / "store.sqlite"
        shards = self.shard_paths_of(primary)
        for path in shards:
            write_placeholder(path)
        records = {i: str(p.resolve()) for i, p in enumerate(shards)}
        records[0], records[1] = records[1], records[0]
        write_legacy_primary(primary, records)

        with self.assertRaises(RuntimeError) as ctx:
            self.sweep.assert_store_is_self_consistent(primary)
        message = str(ctx.exception)
        self.assertIn("serial 0", message)
        self.assertIn("serial 1", message)

    # --- test 5 -------------------------------------------------------------------------------

    def test_the_build_resume_path_accepts_a_store_written_by_write_new_store(self):
        """What `run_build`'s pre-resume check runs against: a four-shard store written by
        `write_new_store`, the fixtures' stand-in for what the pipeline itself writes. It refused
        before this prompt (`docs/store-retirement-audit.md` §2.6, probed); this is the after. No
        build is run here."""
        primary = self.top / "I" / "handover-A3-baseline-lambdacdm-v2.sqlite"
        write_new_store(primary, shards=self.sweep.SHARDS)
        self.sweep.assert_store_is_self_consistent(primary)  # must not raise


if __name__ == "__main__":
    import unittest

    unittest.main()
