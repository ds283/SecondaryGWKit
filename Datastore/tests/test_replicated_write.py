"""
The checked replicated write (prompts/datastore-integrity, prompt 01).

A replicated object is written on a controlling shard, chosen at random per call, and copied to
every other shard in separate transactions. These tests drive the real ``ShardedPool``,
``Datastore``, factories and broker on stand-in shards (``Datastore.tests.standin_pool``), and
check the invariant the prompt establishes: at every instant of every replicated write, either
every shard holds the same replicated rows, or the primary's ``replication_in_flight`` table names
the operation, the class and the controlling shard whose rows are right.

1. a clean write leaves row-identical copies on every shard, timestamps and value serials
   included, and no record;
2. the record is set across the window between the controller's commit and the last replica's,
   on every path, and during every shard's call;
3. a replica that answers with a different serial raises ``ReplicationMismatch``, naming both,
   and the record stays set;
4. a replica's store of ``wavenumber_exit_time`` and ``BackgroundModel`` inserts or verifies by
   serial: a replay writes nothing, a differing row or the key under another serial raises;
5. a write over a set record refuses, and writes nothing to any shard;
6. a sharded store still stamps ``datetime.now()``, and writes no record;
7. no Ray is initialised.

Every store is built in a temporary directory. Nothing under ``var/`` is opened.
"""

import contextlib
import io
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import ray

from Datastore.replication import ReplicationInFlight, ReplicationMismatch
from Datastore.tests import standin_pool as sp
from config.sharding import replicated_tables

# every table a replicated write touches: the replicated classes and BackgroundModel's tag table
REPLICATED = list(replicated_tables) + ["BackgroundModel_tags"]


@contextlib.contextmanager
def quiet():
    """Swallow what the actor code prints when a store raises (utilities.WallclockTimer prints
    the exception and its traceback on the way out)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_replicated_write")


class _TickingDatetime(datetime):
    """The Datastore module's ``datetime``, with ``now()`` a new value on every call, so that two
    shards stamping their own time can never agree by accident."""

    _tick = [0]

    @classmethod
    def now(cls, tz=None):
        cls._tick[0] += 1
        return datetime(2030, 1, 1) + timedelta(seconds=cls._tick[0])


FIXED_NOW = datetime(2001, 2, 3, 4, 5, 6, 789000)


class _FixedDatetime(datetime):
    """The Datastore module's ``datetime``, with ``now()`` a sentinel."""

    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW


class _PoolTestCase(unittest.TestCase):
    """A fresh three-shard store in a temporary directory, on stand-in shards, per test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self._stores = 0

        self.cluster = sp.StandinCluster()
        active = self.cluster.active()
        active.__enter__()
        self.addCleanup(active.__exit__, None, None, None)
        self.addCleanup(self._close)

        self.units = sp.make_units()
        self.cosmology = sp.StandinCosmology(self.units)
        self.fresh()

    def fresh(self):
        """Close the open pool, if any, and open a new, empty store beside it."""
        self._close()
        self._stores += 1
        self.primary = self.root / f"store-{self._stores}" / "store.sqlite"
        self.pool = self.cluster.open_pool(self.primary)
        self.shard_ids = list(self.pool._shards.keys())
        # the middle shard controls, so that replicas come before and after it
        self.cluster.controller = self.shard_ids[1]
        self.controller = self.cluster.controller
        self.replicas = self.cluster.replica_ids()

    def _close(self):
        self.cluster.clear_faults()
        self.cluster.hooks.clear()
        if self.cluster.pool is not None:
            self.cluster.close_pool()

    def tearDown(self):
        self.assertFalse(ray.is_initialized())

    # -- building blocks --------------------------------------------------------------------

    def exit_time(self, k_inv_Mpc=0.5, z_exit=1.0e6):
        atol = sp.get_tolerance(self.pool, 1.0e-10)
        rtol = sp.get_tolerance(self.pool, 1.0e-9)
        k = sp.get_wavenumber(self.pool, k_inv_Mpc, self.units)
        return sp.make_exit_time(k, self.cosmology, atol, rtol, z_exit=z_exit)

    def tags(self, *labels):
        return [ray.get(self.pool.object_get("store_tag", label=l)) for l in labels]

    def background(self, tags=("run-A",), scale=1.0):
        return sp.make_background_model(
            self.pool, self.cosmology, self.tags(*tags), scale=scale
        )

    def stored_background(self, **kwargs):
        return ray.get(self.pool.object_store(self.background(**kwargs)))

    def records(self):
        return sp.in_flight_records(self.pool)

    def assertRecord(self, operation, class_name, store_id=None):
        records = self.records()
        self.assertEqual(1, len(records), records)
        record = records[0]
        self.assertEqual(operation, record["operation"])
        self.assertEqual(class_name, record["class_name"])
        self.assertEqual(self.controller, record["controller_shard"])
        if store_id is not None:
            self.assertEqual(store_id, record["store_id"])
        self.assertIsNotNone(record["started"])
        return record

    def assertNoRecord(self):
        self.assertEqual([], self.records())

    def serials(self, table, where="", params=()):
        """table -> {shard: sorted serials}."""
        out = {}
        for sid, path in sorted(self.pool._shard_db_files.items()):
            out[sid] = sorted(
                r[0]
                for r in sp._read(path, f'SELECT serial FROM "{table}" {where}', params)
            )
        return out


# ------------------------------------------------------------------------------------------------
# 1. a clean write
# ------------------------------------------------------------------------------------------------


class TestCleanWrite(_PoolTestCase):
    def test_every_copy_is_row_identical(self):
        """
        Every column of every copy of every replicated row is the same on every shard, including
        the timestamp and BackgroundModelValue's serials, after a get-inserted class (tolerance,
        and a vectorized get of redshift), a stored class (wavenumber_exit_time) and a
        BackgroundModel with tags and values, stored and validated. Each shard's own clock is made
        to tick on every call, so only a timestamp the driver chose can agree.
        """
        with mock.patch.object(sp.ds_mod, "datetime", _TickingDatetime):
            sp.get_tolerance(self.pool, 1.0e-6)
            sp.get_redshifts(self.pool, [5.0, 50.0])
            exit_time = ray.get(self.pool.object_store(self.exit_time()))
            model = self.stored_background(tags=("run-A", "grid-B"))
            self.assertTrue(ray.get(self.pool.object_validate(model)))

        self.assertNoRecord()

        written = {
            "tolerance",
            "redshift",
            "wavenumber",
            "wavenumber_exit_time",
            "store_tag",
            "IntegrationSolver",
            "BackgroundModel",
            "BackgroundModel_tags",
            "BackgroundModelValue",
        }
        for table in REPLICATED:
            rows = sp.shard_rows(self.pool, table)
            reference = rows[self.controller]
            for sid, shard_rows in rows.items():
                self.assertEqual(reference, shard_rows, f"{table}: shard {sid} differs")
            if table in written:
                self.assertGreater(len(reference), 0, table)

        # the timestamps are the driver's, not the shards' ticking clocks
        for table in ("tolerance", "wavenumber_exit_time", "BackgroundModel"):
            path = self.pool._shard_db_files[self.controller]
            for (stamp,) in sp._read(path, f'SELECT timestamp FROM "{table}"'):
                self.assertFalse(str(stamp).startswith("2030-"), (table, stamp))

        # the value rows carry the controller's serials on every shard
        value_serials = sorted(v.store_id for v in model.values)
        for sid, serials in self.serials(
            "BackgroundModelValue", "WHERE model_serial = ?", (model.store_id,)
        ).items():
            self.assertEqual(value_serials, serials, f"shard {sid}")
        self.assertEqual(
            {sid: [exit_time.store_id] for sid in self.shard_ids},
            self.serials("wavenumber_exit_time"),
        )

    def test_a_get_that_inserts_nothing_replicates_nothing(self):
        sp.get_tolerance(self.pool, 1.0e-6)
        self.cluster.calls.clear()
        sp.get_tolerance(self.pool, 1.0e-6)
        gets = [c for c in self.cluster.calls if c["method"] == "object_get"]
        self.assertEqual([self.controller], [c["shard"] for c in gets])
        self.assertNoRecord()

    def test_an_existing_store_gains_an_empty_record_table_when_opened(self):
        self.cluster.close_pool()
        conn = sqlite3.connect(self.primary)
        conn.execute('DROP TABLE "replication_in_flight"')
        conn.commit()
        conn.close()

        self.pool = self.cluster.open_pool(self.primary)
        self.assertNoRecord()
        sp.get_tolerance(self.pool, 1.0e-6)
        self.assertNoRecord()


# ------------------------------------------------------------------------------------------------
# 2. the record is set across the window
# ------------------------------------------------------------------------------------------------


class TestRecordAcrossTheWindow(_PoolTestCase):
    def _fail_replica(self, replica, method, cls_name, write):
        """Kill ``replica``'s call before it runs; return what the write raised."""
        self.cluster.fault(replica, method, cls_name, "before")
        with self.assertRaises(sp.StandinActorDied):
            write()
        self.cluster.clear_faults()

    def test_scalar_get(self):
        for replica in self.replicas:
            with self.subTest(replica=replica):
                self.fresh()
                self._fail_replica(
                    replica,
                    "object_get",
                    "tolerance",
                    lambda: sp.get_tolerance(self.pool, 1.0e-6),
                )
                record = self.assertRecord("get", "tolerance")
                serials = self.serials("tolerance")
                self.assertEqual(1, len(serials[self.controller]))
                self.assertEqual([], serials[replica])
                self.assertEqual(serials[self.controller][0], record["store_id"])

    def test_vectorized_get(self):
        for replica in self.replicas:
            with self.subTest(replica=replica):
                self.fresh()
                self._fail_replica(
                    replica,
                    "object_get",
                    "redshift",
                    lambda: sp.get_redshifts(self.pool, [5.0, 50.0]),
                )
                self.assertRecord("get", "redshift")
                serials = self.serials("redshift")
                self.assertEqual(2, len(serials[self.controller]))
                self.assertEqual([], serials[replica])

    def test_store_of_an_exit_time(self):
        for replica in self.replicas:
            with self.subTest(replica=replica):
                self.fresh()
                exit_time = self.exit_time()
                self._fail_replica(
                    replica,
                    "object_store",
                    "wavenumber_exit_time",
                    lambda: self.pool.object_store(exit_time),
                )
                record = self.assertRecord("store", "wavenumber_exit_time")
                serials = self.serials("wavenumber_exit_time")
                self.assertEqual([record["store_id"]], serials[self.controller])
                self.assertEqual([], serials[replica])

    def test_store_of_a_background_model(self):
        for replica in self.replicas:
            with self.subTest(replica=replica):
                self.fresh()
                model = self.background()
                self._fail_replica(
                    replica,
                    "object_store",
                    "BackgroundModel",
                    lambda: self.pool.object_store(model),
                )
                record = self.assertRecord("store", "BackgroundModel")
                serials = self.serials("BackgroundModel")
                self.assertEqual([record["store_id"]], serials[self.controller])
                self.assertEqual([], serials[replica])
                self.assertEqual([], self.serials("BackgroundModelValue")[replica])

    def test_validate_of_a_background_model(self):
        for replica in self.replicas:
            with self.subTest(replica=replica):
                self.fresh()
                model = self.stored_background()
                self._fail_replica(
                    replica,
                    "object_validate",
                    "BackgroundModel",
                    lambda: self.pool.object_validate(model),
                )
                self.assertRecord("validate", "BackgroundModel", model.store_id)
                validated = self.serials("BackgroundModel", "WHERE validated = 1")
                self.assertEqual([model.store_id], validated[self.controller])
                self.assertEqual([], validated[replica])

    def test_a_controller_that_dies_after_its_commit_leaves_the_record(self):
        """
        The controlling shard commits and then dies before its answer reaches the driver. The
        record must already be on disk: a record written after the controller's call would not
        exist here, and the controller's row would be on one shard only, with nothing saying so.
        """
        cases = [
            ("object_get", "tolerance", lambda: sp.get_tolerance(self.pool, 1.0e-6)),
            (
                "object_store",
                "wavenumber_exit_time",
                lambda: self.pool.object_store(self.exit_time()),
            ),
            (
                "object_store",
                "BackgroundModel",
                lambda: self.pool.object_store(self.background()),
            ),
        ]
        for method, cls_name, write in cases:
            with self.subTest(cls_name=cls_name):
                self.fresh()
                self.cluster.fault(self.controller, method, cls_name, "after")
                with self.assertRaises(sp.StandinActorDied):
                    write()
                self.cluster.clear_faults()
                operation = "get" if method == "object_get" else "store"
                self.assertRecord(operation, cls_name)
                serials = self.serials(cls_name)
                self.assertEqual(1, len(serials[self.controller]))
                for replica in self.replicas:
                    self.assertEqual([], serials[replica])

    def test_the_record_is_on_disk_during_every_shards_call(self):
        """While each shard's call of a replicated write runs, the primary holds the record, for
        the controller and every replica, on every path."""
        seen = []

        def hook(shard_id, method, cls_name, phase):
            if method in ("object_get", "object_store", "object_validate"):
                seen.append((shard_id, method, cls_name, phase, len(self.records())))

        model = self.background()
        exit_time = self.exit_time()
        self.cluster.hooks.append(hook)
        sp.get_tolerance(self.pool, 1.0e-6)
        sp.get_redshifts(self.pool, [5.0, 50.0])
        self.pool.object_store(exit_time)
        model = ray.get(self.pool.object_store(model))
        ray.get(self.pool.object_validate(model))
        self.cluster.hooks.clear()

        for cls_name, method in [
            ("tolerance", "object_get"),
            ("redshift", "object_get"),
            ("wavenumber_exit_time", "object_store"),
            ("BackgroundModel", "object_store"),
            ("BackgroundModel", "object_validate"),
        ]:
            calls = [s for s in seen if s[1] == method and s[2] == cls_name]
            self.assertEqual(
                sorted(self.shard_ids * 2),
                sorted(s[0] for s in calls),
                (cls_name, method),
            )
            for call in calls:
                self.assertEqual(1, call[4], call)
        self.assertNoRecord()


# ------------------------------------------------------------------------------------------------
# 3. a replica's different serial is loud
# ------------------------------------------------------------------------------------------------


class TestReplicaSerialMismatch(_PoolTestCase):
    def test_scalar_get(self):
        """
        The split of the audit probe's step 4, by hand: one replica holds the key under a serial
        of its own. The controller, which lacks it, inserts it under a new serial and the replica
        answers with its own.
        """
        replica = self.replicas[0]
        own = ray.get(
            self.pool._shards[replica].object_get.remote("tolerance", tol=1.0e-6)
        )

        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            sp.get_tolerance(self.pool, 1.0e-6)

        e = raised.exception
        record = self.assertRecord("get", "tolerance")
        self.assertEqual(replica, e.shard)
        self.assertEqual("tolerance", e.class_name)
        self.assertEqual(record["store_id"], e.controller_serial)
        self.assertEqual(own.store_id, e.replica_serial)
        self.assertNotEqual(e.controller_serial, e.replica_serial)
        self.assertIn(str(e.controller_serial), str(e))
        self.assertIn(str(e.replica_serial), str(e))

    def test_vectorized_get(self):
        replica = self.replicas[1]
        own = ray.get(
            self.pool._shards[replica].object_get.remote(
                "redshift", z=50.0, is_source=True, is_response=False
            )
        )

        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            sp.get_redshifts(self.pool, [5.0, 50.0])

        e = raised.exception
        self.assertRecord("get", "redshift")
        self.assertEqual(replica, e.shard)
        self.assertEqual("redshift", e.class_name)
        self.assertEqual(own.store_id, e.replica_serial)
        controller_serials = self.serials("redshift", "WHERE z = 50.0")[self.controller]
        self.assertEqual(controller_serials, [e.controller_serial])

    def test_store_over_a_replica_holding_the_key_under_another_serial(self):
        """The same split for a stored class: the replica's store refuses, naming its shard."""
        replica = self.replicas[0]
        exit_time = self.exit_time()
        # the replica alone holds this exit time, under a serial of its own
        own = ray.get(self.pool._shards[replica].object_store.remote(exit_time))

        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            self.pool.object_store(exit_time)

        e = raised.exception
        record = self.assertRecord("store", "wavenumber_exit_time")
        self.assertEqual(record["store_id"], e.controller_serial)
        self.assertEqual(own.store_id, e.replica_serial)
        self.assertIn(f"shard{replica:04d}", str(e.shard))

    def test_the_exception_pickles_with_its_fields(self):
        import pickle

        e = ReplicationMismatch(2, "tolerance", 7, 3, detail="why")
        f = pickle.loads(pickle.dumps(e))
        self.assertEqual(
            (2, "tolerance", 7, 3, "why"),
            (f.shard, f.class_name, f.controller_serial, f.replica_serial, f.detail),
        )
        self.assertEqual(str(e), str(f))


# ------------------------------------------------------------------------------------------------
# 4. replica stores insert or verify
# ------------------------------------------------------------------------------------------------


class TestIdempotentReplicaStores(_PoolTestCase):
    def replay(self, shard_id, obj):
        return ray.get(self.pool._shards[shard_id].object_store.remote(obj))

    def test_exit_time_replay_writes_nothing_and_returns_the_row(self):
        exit_time = ray.get(self.pool.object_store(self.exit_time()))
        before = sp.shard_snapshot(self.pool, REPLICATED)
        for sid in self.shard_ids:
            again = self.replay(sid, exit_time)
            self.assertEqual(exit_time.store_id, again.store_id)
        self.assertEqual(before, sp.shard_snapshot(self.pool, REPLICATED))

    def test_exit_time_differing_row_raises(self):
        exit_time = ray.get(self.pool.object_store(self.exit_time()))
        exit_time._z_exit = exit_time.z_exit * 2.0
        before = sp.shard_snapshot(self.pool, REPLICATED)
        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            self.replay(self.replicas[0], exit_time)
        self.assertIn("z_exit", str(raised.exception))
        self.assertEqual(before, sp.shard_snapshot(self.pool, REPLICATED))

    def test_exit_time_key_under_another_serial_raises(self):
        exit_time = ray.get(self.pool.object_store(self.exit_time()))
        original = exit_time.store_id
        exit_time._my_id = original + 1000
        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            self.replay(self.replicas[0], exit_time)
        self.assertEqual(original + 1000, raised.exception.controller_serial)
        self.assertEqual(original, raised.exception.replica_serial)

    def test_background_replay_writes_nothing_and_returns_the_row(self):
        model = self.stored_background()
        before = sp.shard_snapshot(self.pool, REPLICATED)
        for sid in self.shard_ids:
            again = self.replay(sid, model)
            self.assertEqual(model.store_id, again.store_id)
            self.assertEqual(
                [v.store_id for v in model.values], [v.store_id for v in again.values]
            )
        self.assertEqual(before, sp.shard_snapshot(self.pool, REPLICATED))

    def test_background_differing_row_raises(self):
        model = self.stored_background()
        cases = {
            "a value row": lambda m: setattr(
                m.values[1], "_Hubble", m.values[1].Hubble * 2.0
            ),
            "the model row": lambda m: setattr(m, "_label", "another-label"),
            "the tags": lambda m: m.tags.append(self.tags("run-Z")[0]),
        }
        for what, change in cases.items():
            with self.subTest(what=what):
                changed = ray.get(standin_copy(model))
                change(changed)
                before = sp.shard_snapshot(self.pool, REPLICATED)
                with quiet(), self.assertRaises(ReplicationMismatch):
                    self.replay(self.replicas[0], changed)
                self.assertEqual(before, sp.shard_snapshot(self.pool, REPLICATED))

    def test_background_key_under_another_serial_raises(self):
        model = self.stored_background()
        self.assertTrue(ray.get(self.pool.object_validate(model)))

        # the same model, carrying serials this store has never used
        other = self.background()
        other._my_id = model.store_id + 1000
        for i, value in enumerate(other.values):
            value._my_id = model.store_id + 2000 + i
        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            self.replay(self.replicas[0], other)
        self.assertEqual(model.store_id, raised.exception.replica_serial)

    def test_background_unvalidated_row_of_the_same_key_does_not_block(self):
        """
        A store interrupted before its validation leaves an unvalidated row on every shard, which
        build() does not serve, and a later store of the same model is written beside it. A
        replica's store does not take that row for the key held under another serial.
        """
        model = self.stored_background()  # stored, never validated

        again = ray.get(self.pool.object_store(self.background()))
        self.assertNotEqual(model.store_id, again.store_id)
        self.assertNoRecord()
        for sid, serials in self.serials("BackgroundModel").items():
            self.assertEqual(sorted([model.store_id, again.store_id]), serials, sid)


def standin_copy(obj):
    """A stand-in reference to a pickled copy of ``obj``."""
    return sp.StandinRef(result=obj)


# ------------------------------------------------------------------------------------------------
# 5. a write over a set record refuses
# ------------------------------------------------------------------------------------------------


class TestWriteOverASetRecord(_PoolTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.stored_background()
        self.exit = self.exit_time(k_inv_Mpc=2.0)
        self.other_model = self.background(tags=("x",))
        # an interrupted write: the record is left set
        self.cluster.fault(self.replicas[0], "object_get", "tolerance", "before")
        with self.assertRaises(sp.StandinActorDied):
            sp.get_tolerance(self.pool, 1.0e-6)
        self.cluster.clear_faults()
        self.record = self.assertRecord("get", "tolerance")

    def assertRefused(self, write):
        before = sp.shard_snapshot(self.pool, REPLICATED)
        self.cluster.calls.clear()
        with self.assertRaises(ReplicationInFlight) as raised:
            write()
        self.assertEqual(before, sp.shard_snapshot(self.pool, REPLICATED))
        self.assertEqual([], self.cluster.calls)
        self.assertEqual([self.record], self.records())
        self.assertIn("tolerance", str(raised.exception))

    def test_every_path_refuses(self):
        self.assertRefused(lambda: sp.get_tolerance(self.pool, 1.0e-4))
        self.assertRefused(lambda: sp.get_redshifts(self.pool, [7.0]))
        self.assertRefused(lambda: self.pool.object_store(self.exit))
        self.assertRefused(lambda: self.pool.object_store(self.other_model))
        self.assertRefused(lambda: self.pool.object_validate(self.model))

    def test_a_reopened_pool_refuses(self):
        self.cluster.close_pool()
        self.pool = self.cluster.open_pool(self.primary)
        self.assertRefused(lambda: sp.get_tolerance(self.pool, 1.0e-4))

    def test_overlapping_writes_are_a_programming_error(self):
        conn = sqlite3.connect(self.primary)
        conn.execute('DELETE FROM "replication_in_flight"')
        conn.commit()
        conn.close()
        before = sp.shard_snapshot(self.pool, REPLICATED)
        self.pool._replication_active = True
        with self.assertRaises(RuntimeError) as raised:
            sp.get_tolerance(self.pool, 1.0e-4)
        self.assertNotIsInstance(raised.exception, ReplicationInFlight)
        self.assertIn("overlap", str(raised.exception))
        self.assertNoRecord()
        self.assertEqual(before, sp.shard_snapshot(self.pool, REPLICATED))


# ------------------------------------------------------------------------------------------------
# 6. sharded writes are unchanged
# ------------------------------------------------------------------------------------------------


class TestShardedWritesUnchanged(_PoolTestCase):
    def test_a_sharded_store_stamps_its_own_time_and_writes_no_record(self):
        exit_time = ray.get(self.pool.object_store(self.exit_time()))
        policy_data = sp.make_policy_data(exit_time)
        shard = self.pool._shard_keys[exit_time.k.store_id]

        seen = []
        self.cluster.hooks.append(
            lambda sid, method, cls, phase: (
                seen.append(len(self.records())) if method == "object_store" else None
            )
        )
        self.cluster.calls.clear()
        with mock.patch.object(sp.ds_mod, "datetime", _FixedDatetime):
            stored = ray.get(self.pool.object_store(policy_data))
        self.cluster.hooks.clear()

        # one call, on the wavenumber's shard, with no timestamp override, and no record at any
        # point
        stores = [c for c in self.cluster.calls if c["method"] == "object_store"]
        self.assertEqual(1, len(stores))
        self.assertEqual(shard, stores[0]["shard"])
        self.assertEqual("GkSourcePolicyData", stores[0]["class"])
        self.assertNotIn("insert_timestamp", stores[0]["kwargs"])
        self.assertEqual([0, 0], seen)
        self.assertNoRecord()

        rows = sp._read(
            self.pool._shard_db_files[shard],
            'SELECT serial, timestamp FROM "GkSourcePolicyData"',
        )
        self.assertEqual([stored.store_id], [r[0] for r in rows])
        self.assertEqual(str(FIXED_NOW), rows[0][1])

    def test_a_replicated_row_does_not_take_the_shards_time(self):
        with mock.patch.object(sp.ds_mod, "datetime", _FixedDatetime):
            sp.get_tolerance(self.pool, 1.0e-6)
        for sid, rows in sp.shard_rows(self.pool, "tolerance").items():
            path = self.pool._shard_db_files[sid]
            (stamp,) = sp._read(path, 'SELECT timestamp FROM "tolerance"')[0]
            self.assertNotEqual(str(FIXED_NOW), stamp)


if __name__ == "__main__":
    unittest.main()
