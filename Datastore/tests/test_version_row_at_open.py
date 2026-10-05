"""
The version row, written through the recorded write (prompts/a3-v2-readiness, prompt 02, W1–W2).

``Datastore.__init__`` no longer looks up or inserts the version row. ``ShardedPool.__init__``,
once every actor exists, reads the label's row from the shard files without writing anything,
writes it once through ``_get_impl_replicated_table`` (and so through ``_replicated_write``, under
a ``replication_in_flight`` record) only when no shard holds it, and then calls ``set_version`` on
every actor. Until then an actor refuses every insert into a class with a ``version`` column.
These tests drive the real ``ShardedPool``, ``Datastore``, factories and broker on stand-in shards
(``Datastore.tests.standin_pool``):

1. a new store: the row is written once, through the recorded write, and no actor constructor
   inserts it; every versioned row stored afterwards carries its serial, on every shard;
2. a new label on an existing store (audit V1 case 3), interrupted after the controlling shard's
   get commits, before it commits, and on one replica only: the next open completes it;
3. a new store's first open interrupted at the version write, after every shard file exists: the
   next open succeeds and writes the row;
4. ``_insert`` before ``set_version`` raises, names the class, takes no serial and writes nothing;
5. with no record, a ``version`` table that differs across shards still refuses;
6. a ``prune`` record never reaches the get/store/validate licence, which admits the version row
   under a ``get`` record;
7. no Ray is initialised.

A clean reopen under a label the store holds writes nothing: that is
``test_reconcile_at_open.TestCleanStoreUntouched`` and the second-reopen check of
``TestKillAndReopen``, unchanged. Every store is built in a temporary directory; nothing under
``var/`` is opened.
"""

import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

import ray
import sqlalchemy as sqla

from config.datastore import factories as _factories
from Datastore.SQL.schema import build_schema
from Datastore.replication import ReplicatedDivergence
from Datastore.tests import standin_pool as sp
from config.sharding import (
    replicated_tables,
    sharded_tables,
    shard_key_type,
    shard_key_wavenumber_store_id,
)


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_version_row_at_open")


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


def execute(path: Path, *statements):
    conn = sqlite3.connect(path)
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def version_rows(path: Path):
    return sp._read(path, "SELECT serial, label FROM version ORDER BY serial")


class _VersionTestCase(unittest.TestCase):
    """Stand-in shards in a temporary directory, three per store."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self._stores = 0

        self.cluster = sp.StandinCluster()
        active = self.cluster.active()
        active.__enter__()
        self.addCleanup(active.__exit__, None, None, None)
        self.addCleanup(self.close)
        self.units = sp.make_units()

    def tearDown(self):
        self.assertFalse(ray.is_initialized())

    def new_primary(self) -> Path:
        self._stores += 1
        return self.root / f"store-{self._stores}" / "store.sqlite"

    def open(self, primary: Path, label: str = "L1", **kwargs):
        """Open (or create) ``primary`` under ``label``; return what the constructor printed.
        Faults and hooks set before the call stay set."""
        self.close_pool()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pool = sp.sp_mod.ShardedPool(
                version_label=label,
                db_name=primary,
                ShardKeyType=shard_key_type,
                ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
                replicated_tables=replicated_tables,
                sharded_tables=sharded_tables,
                shards=3,
                factories=_factories,
                **kwargs,
            )
        self.cluster.pool = pool
        self.pool = pool
        self.files = dict(pool._shard_db_files)
        return out.getvalue()

    def interrupted_open(self, primary: Path, label: str, faults, controller=None):
        """Open ``primary`` under ``label`` with ``faults`` [(shard, method, class, when)] and the
        controlling shard pinned, and require the open to be killed."""
        self.close()
        for sid, method, cls_name, when in faults:
            self.cluster.fault(sid, method, cls_name, when)
        pin = (
            self.cluster.pin_controller(controller)
            if controller is not None
            else contextlib.nullcontext()
        )
        try:
            with pin, quiet(), self.assertRaises(sp.StandinActorDied):
                self.open(primary, label)
        finally:
            self.cluster.clear_faults()
        self.pool = None

    def close_pool(self):
        if self.cluster.pool is not None:
            self.cluster.close_pool()
        self.pool = None

    def close(self):
        self.cluster.clear_faults()
        self.cluster.hooks.clear()
        self.close_pool()

    def shard_files(self, primary: Path):
        directory = primary.parent
        return {
            i: directory / f"store-shard{i:04d}.sqlite"
            for i in range(3)
            if (directory / f"store-shard{i:04d}.sqlite").exists()
        }

    def primary_records(self, primary: Path):
        rows = sp._read(
            primary,
            'SELECT operation, class_name, controller_shard, store_id FROM "replication_in_flight"',
        )
        return [tuple(r) for r in rows]

    def version_gets(self):
        return [
            c
            for c in self.cluster.calls
            if c["method"] == "object_get" and c["class"] == "version"
        ]


# ------------------------------------------------------------------------------------------------
# 1. a new store
# ------------------------------------------------------------------------------------------------


class TestNewStore(_VersionTestCase):
    def test_the_row_is_written_once_through_the_recorded_write(self):
        primary = self.new_primary()
        seen = []

        def hook(shard, method, cls_name, when):
            if method == "object_get" and cls_name == "version" and when == "before":
                seen.append((shard, self.primary_records(primary)))

        self.cluster.hooks.append(hook)
        self.open(primary, "L1")
        self.cluster.hooks.clear()

        # one controller call, then one call per replica, each made under the get's record
        self.assertEqual(3, len(seen), seen)
        self.assertEqual({0, 1, 2}, {shard for shard, _ in seen})
        controller = seen[0][0]
        for shard, records in seen:
            self.assertEqual(1, len(records), (shard, records))
            operation, cls_name, record_controller, _ = records[0]
            self.assertEqual(
                ("get", "version", controller), (operation, cls_name, record_controller)
            )
        self.assertEqual(3, len(self.version_gets()))

        # the row is identical on every shard, and is the pool's
        for sid, path in self.files.items():
            self.assertEqual([(1, "L1")], version_rows(path), sid)
        self.assertEqual(1, self.pool._version.store_id)
        self.assertEqual("L1", self.pool._version.label)
        self.assertEqual([], self.primary_records(primary))

        # every actor was given the serial, once
        set_calls = sorted(
            c["shard"] for c in self.cluster.calls if c["method"] == "set_version"
        )
        self.assertEqual([0, 1, 2], set_calls)

    def test_no_actor_constructor_inserts_the_row(self):
        """With the pool's version get killed before it runs on every shard, every shard file and
        every table exists, and no shard holds a version row: nothing but the pool's recorded
        write inserts it. The record the pool committed is left."""
        primary = self.new_primary()
        self.interrupted_open(
            primary,
            "L1",
            [(sid, "object_get", "version", "before") for sid in range(3)],
        )
        files = self.shard_files(primary)
        self.assertEqual([0, 1, 2], sorted(files))
        for sid, path in files.items():
            self.assertEqual([], version_rows(path), sid)
        records = self.primary_records(primary)
        self.assertEqual(1, len(records))
        self.assertEqual(("get", "version"), records[0][:2])

    def test_versioned_rows_carry_the_serial_on_every_shard(self):
        primary = self.new_primary()
        self.open(primary, "L1")
        cosmology = sp.StandinCosmology(self.units)
        atol = sp.get_tolerance(self.pool, 1.0e-10)
        rtol = sp.get_tolerance(self.pool, 1.0e-9)
        k = sp.get_wavenumber(self.pool, 0.5, self.units)
        exit_time = ray.get(
            self.pool.object_store(sp.make_exit_time(k, cosmology, atol, rtol))
        )
        self.policy("policy-1", 1.5)
        tag = ray.get(self.pool.object_get("store_tag", label="run-A"))
        ray.get(
            self.pool.object_store(
                sp.make_background_model(self.pool, cosmology, [tag])
            )
        )
        # a sharded versioned row, on its wavenumber's shard
        ray.get(self.pool.object_store(sp.make_policy_data(exit_time)))
        self.close()

        # reopened under a second label: the row is written once more, and new rows carry it
        self.open(primary, "L2")
        self.assertEqual(2, self.pool._version.store_id)
        self.policy("policy-2", 2.5)
        self.close()

        for sid, path in self.files.items():
            self.assertEqual([(1, "L1"), (2, "L2")], version_rows(path), sid)
            self.assertEqual(
                [("policy-1", 1), ("policy-2", 2)],
                sp._read(
                    path, "SELECT label, version FROM GkSourcePolicy ORDER BY serial"
                ),
                sid,
            )
            for table in ("wavenumber_exit_time", "BackgroundModel"):
                self.assertEqual(
                    [(1,)],
                    sp._read(path, f'SELECT version FROM "{table}"'),
                    (table, sid),
                )
        held = {
            sid: sp._read(path, "SELECT version FROM GkSourcePolicyData")
            for sid, path in self.files.items()
        }
        self.assertEqual([[(1,)]], [rows for rows in held.values() if rows])

    def policy(self, label, threshold):
        return ray.get(
            self.pool.object_get(
                "GkSourcePolicy",
                label=label,
                Levin_threshold=threshold,
                numeric_policy="maximize-numeric",
            )
        )


# ------------------------------------------------------------------------------------------------
# 2. a new label on an existing store (audit V1 case 3)
# ------------------------------------------------------------------------------------------------


class TestNewLabelInterrupted(_VersionTestCase):
    def existing_store(self) -> Path:
        primary = self.new_primary()
        self.open(primary, "L1")
        self.shard_ids = list(self.pool._shards.keys())
        self.close()
        return primary

    def assertCompleted(self, primary, copied_to):
        """Reopen under L2 with no fault: the record is cleared, the copies are the expected
        ones, every shard holds both rows, and a second reopen writes nothing."""
        printed = self.open(primary, "L2")
        rec = self.pool.reconciliation
        self.assertEqual("get", rec["record"]["operation"])
        self.assertEqual("version", rec["record"]["class_name"])
        self.assertTrue(rec["cleared"])
        self.assertEqual(
            sorted(copied_to),
            sorted(
                a["shard"]
                for a in rec["repaired"]
                if a["action"] == "copied" and a["class_name"] == "version"
            ),
        )
        self.assertEqual(
            [], [a for a in rec["repaired"] if a["class_name"] != "version"]
        )
        self.assertIn("the replication_in_flight record was cleared", printed)
        self.assertEqual(2, self.pool._version.store_id)
        self.close()
        for sid, path in self.files.items():
            self.assertEqual([(1, "L1"), (2, "L2")], version_rows(path), sid)
        self.assertEqual([], self.primary_records(primary))

        checksums = sp.store_checksums(primary)
        self.open(primary, "L2")
        self.assertIsNone(self.pool.reconciliation["record"])
        self.close()
        self.assertEqual(checksums, sp.store_checksums(primary))

    def test_a_fault_after_the_controller_commits(self):
        primary = self.existing_store()
        controller = self.shard_ids[1]
        self.interrupted_open(
            primary,
            "L2",
            [(controller, "object_get", "version", "after")],
            controller=controller,
        )
        files = self.shard_files(primary)
        for sid, path in files.items():
            expected = [(1, "L1"), (2, "L2")] if sid == controller else [(1, "L1")]
            self.assertEqual(expected, version_rows(path), sid)
        # killed at _replicated_write's step 2: the record names the controller, no store_id
        self.assertEqual(
            [("get", "version", controller, None)], self.primary_records(primary)
        )
        self.assertCompleted(primary, [s for s in files if s != controller])

    def test_a_fault_before_the_controller_commits(self):
        primary = self.existing_store()
        controller = self.shard_ids[1]
        checksums = {
            p: c for p, c in sp.store_checksums(primary).items() if p != "store.sqlite"
        }
        self.interrupted_open(
            primary,
            "L2",
            [(controller, "object_get", "version", "before")],
            controller=controller,
        )
        # the shards are untouched; only the primary's record was written
        self.assertEqual(
            checksums,
            {
                p: c
                for p, c in sp.store_checksums(primary).items()
                if p != "store.sqlite"
            },
        )
        self.assertEqual(
            [("get", "version", controller, None)], self.primary_records(primary)
        )

        # the next open clears the record over identical shards, then writes the row itself
        self.cluster.calls.clear()
        self.open(primary, "L2")
        rec = self.pool.reconciliation
        self.assertTrue(rec["cleared"])
        self.assertEqual([], rec["repaired"])
        self.assertEqual(3, len(self.version_gets()))
        self.assertEqual(2, self.pool._version.store_id)
        self.close()
        for sid, path in self.files.items():
            self.assertEqual([(1, "L1"), (2, "L2")], version_rows(path), sid)
        self.assertEqual([], self.primary_records(primary))

    def test_a_fault_on_one_replica_only(self):
        primary = self.existing_store()
        controller = self.shard_ids[1]
        replica = [s for s in self.shard_ids if s != controller][0]
        self.interrupted_open(
            primary,
            "L2",
            [(replica, "object_get", "version", "before")],
            controller=controller,
        )
        files = self.shard_files(primary)
        for sid, path in files.items():
            expected = [(1, "L1")] if sid == replica else [(1, "L1"), (2, "L2")]
            self.assertEqual(expected, version_rows(path), sid)
        # killed after step 3: the record names the controller and the serial
        self.assertEqual(
            [("get", "version", controller, 2)], self.primary_records(primary)
        )
        self.assertCompleted(primary, [replica])

    def test_a_label_no_shard_holds_is_written_through_the_recorded_write(self):
        """The store is consistent: every shard lacks L2. Its open writes the row as a new
        store's does, and the reopen under L1 that follows finds L1 and writes nothing.
        """
        primary = self.existing_store()
        self.cluster.calls.clear()
        self.open(primary, "L2")
        self.assertIsNone(self.pool.reconciliation["record"])
        self.assertEqual(3, len(self.version_gets()))
        self.close()
        checksums = sp.store_checksums(primary)
        self.cluster.calls.clear()
        self.open(primary, "L1")
        self.assertEqual(1, self.pool._version.store_id)
        self.assertEqual([], self.version_gets())
        self.close()
        self.assertEqual(checksums, sp.store_checksums(primary))


# ------------------------------------------------------------------------------------------------
# 3. a new store's first open, interrupted at the version write
# ------------------------------------------------------------------------------------------------


class TestNewStoreInterrupted(_VersionTestCase):
    def test_interrupted_at_the_version_write(self):
        for when in ("before", "after"):
            with self.subTest(when=when):
                primary = self.new_primary()
                # a new store's shards are 0, 1 and 2, built 2, 0, 1: pin shard 0
                self.interrupted_open(
                    primary,
                    "L1",
                    [(0, "object_get", "version", when)],
                    controller=0,
                )
                files = self.shard_files(primary)
                self.assertEqual([0, 1, 2], sorted(files))
                for sid, path in files.items():
                    held = when == "after" and sid == 0
                    self.assertEqual(
                        [(1, "L1")] if held else [], version_rows(path), sid
                    )

                self.open(primary, "L1")
                rec = self.pool.reconciliation
                self.assertTrue(rec["cleared"])
                self.assertEqual(1, self.pool._version.store_id)
                self.close()
                for sid, path in self.files.items():
                    self.assertEqual([(1, "L1")], version_rows(path), (when, sid))
                self.assertEqual([], self.primary_records(primary))


# ------------------------------------------------------------------------------------------------
# 4. _insert before set_version
# ------------------------------------------------------------------------------------------------


class TestInsertBeforeSetVersion(_VersionTestCase):
    # GkSourcePolicy has a version column (its factory registers "version": True)
    POLICY = dict(
        label="standin-policy", Levin_threshold=1.5, numeric_policy="maximize-numeric"
    )

    def bare_actor(self):
        """A store made and closed through the pool, and an actor on its shard 0 built directly,
        as the pool builds one, with no set_version."""
        primary = self.new_primary()
        self.open(primary, "L1")
        self.close()
        broker = sp.sp_mod.SerialPoolBroker.options(name="SerialPoolBroker").remote(
            name="SerialPoolBroker"
        )
        with quiet():
            actor = sp.sp_mod.Datastore.options(name="shard0000-store").remote(
                version_label="L1",
                db_name=self.files[0],
                replicated_tables=list(replicated_tables),
                factories=_factories,
                my_name="shard0000-store",
                serial_broker=broker,
            )
        return primary, actor

    def test_a_versioned_insert_raises_names_the_class_and_writes_nothing(self):
        primary, actor = self.bare_actor()
        checksums = sp.store_checksums(primary)
        self.cluster.calls.clear()
        with quiet(), self.assertRaises(RuntimeError) as raised:
            ray.get(actor.object_get.remote("GkSourcePolicy", **self.POLICY))
        self.assertIn('"GkSourcePolicy"', str(raised.exception))
        self.assertIn("set_version", str(raised.exception))
        self.assertEqual(checksums, sp.store_checksums(primary))
        # the refusal comes before the serial lease: the broker was never called
        self.assertEqual(
            [(0, "object_get", "GkSourcePolicy")],
            [(c["shard"], c["method"], c["class"]) for c in self.cluster.calls],
        )

    def test_an_unversioned_insert_is_not_refused_and_set_version_admits_the_rest(self):
        primary, actor = self.bare_actor()
        tol = ray.get(actor.object_get.remote("tolerance", tol=1.0e-5))
        self.assertIsNotNone(tol.store_id)

        ray.get(actor.set_version.remote(1))
        policy = ray.get(actor.object_get.remote("GkSourcePolicy", **self.POLICY))
        self.assertEqual(
            [(policy.store_id, 1)],
            sp._read(self.files[0], "SELECT serial, version FROM GkSourcePolicy"),
        )

        # the same serial again is harmless; another is refused
        ray.get(actor.set_version.remote(1))
        with self.assertRaises(RuntimeError):
            ray.get(actor.set_version.remote(2))
        with self.assertRaises(TypeError):
            ray.get(actor.set_version.remote("1"))


# ------------------------------------------------------------------------------------------------
# 5. with no record, a version table that differs still refuses
# ------------------------------------------------------------------------------------------------


class TestVersionDifferenceWithNoRecord(_VersionTestCase):
    def test_refused_naming_the_class_and_nothing_written(self):
        primary = self.new_primary()
        self.open(primary, "L1")
        self.close()
        execute(self.files[2], "INSERT INTO version (serial, label) VALUES (2, 'L2')")
        checksums = sp.store_checksums(primary)
        with quiet(), self.assertRaises(ReplicatedDivergence) as raised:
            self.open(primary, "L2")
        self.assertIsNone(raised.exception.record)
        self.assertIn('class "version"', str(raised.exception))
        self.cluster.pool = None
        self.assertEqual(checksums, sp.store_checksums(primary))


# ------------------------------------------------------------------------------------------------
# 6. the licence: a get record of the version row is admitted, a prune record is not
# ------------------------------------------------------------------------------------------------


class TestTheLicenceAndTheVersionRow(_VersionTestCase):
    def test_a_prune_record_is_refused_where_a_get_record_is_admitted(self):
        """The state an interrupted version get leaves (the controller holds L2, the replicas do
        not), planned under its own get record and under the same record as a prune. The
        get licence admits the copy; the prune record, even naming a shard, is refused.
        """
        primary = self.new_primary()
        self.open(primary, "L1")
        shard_ids = list(self.pool._shards.keys())
        # any pool serves for the planner's methods, which read the files they are given
        pool = self.pool
        self.close()
        controller = shard_ids[1]
        self.interrupted_open(
            primary,
            "L2",
            [(controller, "object_get", "version", "after")],
            controller=controller,
        )
        record = dict(
            zip(
                sp.table_columns(primary, "replication_in_flight"),
                sp._read(primary, 'SELECT * FROM "replication_in_flight"')[0],
            )
        )

        built = build_schema(sqla.MetaData(), _factories)
        specs = pool._replicated_table_specs(built)
        files = self.shard_files(primary)
        engines = {sid: pool._reconcile_engine(path) for sid, path in files.items()}
        conns = {sid: e.connect() for sid, e in engines.items()}
        try:
            snapshots = {
                sid: pool._read_replicated_tables(conns[sid], specs) for sid in files
            }
            for c in conns.values():
                c.rollback()
            differences = pool._replicated_differences(specs, snapshots)
            self.assertTrue(differences)
            self.assertTrue(all(d["table"] == "version" for d in differences))

            # the record read back: started is a string; the planner reads it only to compare
            # timestamps, which version has none of
            plan, refusals = pool._plan_replicated_repair(
                record, specs, snapshots, differences, conns
            )
            self.assertEqual([], refusals)
            self.assertEqual(
                {("version", s) for s in files if s != controller},
                set(plan["copies"].keys()),
            )

            for named in (controller, None):
                prune = {
                    **record,
                    "operation": "prune",
                    "controller_shard": named,
                }
                plan, refusals = pool._plan_replicated_repair(
                    prune, specs, snapshots, differences, conns
                )
                self.assertIsNone(plan, named)
                self.assertEqual(1, len(refusals), named)
                self.assertIn("does not name an operation", refusals[0]["detail"])
        finally:
            for c in conns.values():
                c.close()
            for e in engines.values():
                e.dispose()


if __name__ == "__main__":
    unittest.main()
