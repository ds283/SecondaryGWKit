"""
The prune at open of a replicated class, by the pool, under a record (prompts/a3-v2-readiness,
prompt 02, W3; the user's decision U1).

Under ``prune_unvalidated=True``, ``ShardedPool.__init__`` prunes every replicated class whose
factory prunes at startup (``BackgroundModel``, with its tags and values) after the check at open
and before the broker or any actor exists: a ``prune`` record is committed on the primary, each
shard is pruned by the factory's own ``validate_on_startup`` in its own transaction and checked,
and the record is cleared last. The check at open completes an interrupted prune by running it
again on every shard, whatever the open's ``prune_unvalidated`` is. Actors prune sharded classes
only. These tests drive the real ``ShardedPool``, ``Datastore``, factories and broker on stand-in
shards (``Datastore.tests.standin_pool``):

1. an uninterrupted prune: every shard identical, no record left, the record set while each shard
   was pruned; a validated model, and every sharded row, untouched; a prune with nothing to
   prune writes nothing;
2. faulted after shard k's prune, for each k, and before the first shard: the next open, with
   ``prune_unvalidated`` false, completes it and clears the record;
3. a prune that fails on one shard inside the factory (which catches its own ``SQLAlchemyError``):
   the record stays, the open raises, a completion that fails the same way writes nothing, and
   the next open completes it;
4. the two kinds of record stay apart: a ``prune`` record over a difference in another class
   refuses; a ``get`` record of ``BackgroundModel`` over shards that differ is repaired, not
   pruned;
5. no actor deletes a replicated row, even when the pool's prune has not run;
6. no Ray is initialised.

Every store is built in a temporary directory; nothing under ``var/`` is opened.
"""

import contextlib
import io
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ray
import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.replication import ReplicatedDivergence
from Datastore.tests import standin_pool as sp
from config.sharding import replicated_tables, sharded_tables

# every table the check compares: the replicated classes and BackgroundModel's tag table
REPLICATED = list(replicated_tables) + ["BackgroundModel_tags"]
UNIT = ("BackgroundModel", "BackgroundModel_tags", "BackgroundModelValue")


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_prune_at_open")


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


class SimulatedDeath(Exception):
    """The process killed inside the pool's prune."""


class _PruneTestCase(unittest.TestCase):
    """
    A three-shard store holding a validated BackgroundModel (A, tag run-A), an unvalidated one
    (B, tag run-B, its values scaled), an exit time and a sharded GkSourcePolicyData row.
    """

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
        self.cosmology = sp.StandinCosmology(self.units)

    def tearDown(self):
        self.assertFalse(ray.is_initialized())

    def build(self, unvalidated=True):
        """A new store, closed: model A validated, model B (if ``unvalidated``) not."""
        self.close()
        self._stores += 1
        self.primary = self.root / f"store-{self._stores}" / "store.sqlite"
        self.pool = self.cluster.open_pool(self.primary)
        self.shard_ids = sorted(self.pool._shards.keys())
        self.files = dict(self.pool._shard_db_files)

        atol = sp.get_tolerance(self.pool, 1.0e-10)
        rtol = sp.get_tolerance(self.pool, 1.0e-9)
        k = sp.get_wavenumber(self.pool, 0.5, self.units)
        exit_time = ray.get(
            self.pool.object_store(sp.make_exit_time(k, self.cosmology, atol, rtol))
        )
        ray.get(self.pool.object_store(sp.make_policy_data(exit_time)))

        self.model_a = self.store_model("run-A", 1.0)
        self.assertTrue(ray.get(self.pool.object_validate(self.model_a)))
        self.model_b = self.store_model("run-B", 2.0) if unvalidated else None
        self.close()

    def store_model(self, tag, scale):
        tags = [ray.get(self.pool.object_get("store_tag", label=tag))]
        return ray.get(
            self.pool.object_store(
                sp.make_background_model(self.pool, self.cosmology, tags, scale=scale)
            )
        )

    def open(self, **kwargs):
        self.close()
        self.pool, printed = self.cluster.open_pool_output(self.primary, **kwargs)
        return printed

    def close(self):
        self.cluster.clear_faults()
        self.cluster.hooks.clear()
        if self.cluster.pool is not None:
            self.cluster.close_pool()
        self.pool = None

    # -- reading the closed store ------------------------------------------------------------

    def records(self):
        rows = sp._read(
            self.primary,
            'SELECT operation, class_name, controller_shard, store_id FROM "replication_in_flight"',
        )
        return [tuple(r) for r in rows]

    def models(self, sid):
        return sp._read(
            self.files[sid],
            'SELECT serial, validated FROM "BackgroundModel" ORDER BY serial',
        )

    def unit_rows(self, sid, serial):
        return {
            "BackgroundModel": sp._read(
                self.files[sid],
                'SELECT * FROM "BackgroundModel" WHERE serial = ?',
                (serial,),
            ),
            "BackgroundModel_tags": sp._read(
                self.files[sid],
                'SELECT * FROM "BackgroundModel_tags" WHERE model_serial = ?',
                (serial,),
            ),
            "BackgroundModelValue": sp._read(
                self.files[sid],
                'SELECT * FROM "BackgroundModelValue" WHERE model_serial = ? ORDER BY serial',
                (serial,),
            ),
        }

    def sharded_rows(self):
        return {
            (table, sid): sp._read(path, f'SELECT * FROM "{table}" ORDER BY 1')
            for table in sharded_tables
            for sid, path in self.files.items()
        }

    def replicated_rows(self, exclude=UNIT):
        out = {}
        for table in REPLICATED:
            if table in exclude:
                continue
            for sid, path in self.files.items():
                out[(table, sid)] = sp._read(
                    path, f'SELECT * FROM "{table}" ORDER BY 1'
                )
        return out

    def assertIdentical(self):
        for table in REPLICATED:
            rows = {
                sid: sp._read(path, f'SELECT * FROM "{table}" ORDER BY 1')
                for sid, path in self.files.items()
            }
            first = rows[self.shard_ids[0]]
            for sid in self.shard_ids:
                self.assertEqual(first, rows[sid], f"{table}: shard {sid}")

    def assertPruned(self, a_rows=None):
        """B is gone from every shard, as a unit; A is there, validated, unchanged."""
        for sid in self.shard_ids:
            self.assertEqual([(self.model_a.store_id, 1)], self.models(sid), sid)
            gone = self.unit_rows(sid, self.model_b.store_id)
            self.assertEqual({t: [] for t in UNIT}, gone, sid)
            if a_rows is not None:
                self.assertEqual(
                    a_rows[sid], self.unit_rows(sid, self.model_a.store_id), sid
                )
        self.assertIdentical()

    # -- faults inside the pool's prune ------------------------------------------------------

    @contextlib.contextmanager
    def die_in_prune(self, after=None, before=None):
        """Kill the open inside the pool's prune: after the ``after``-th shard's prune has
        committed (0-based, in the order the pool prunes them), or before the ``before``-th.
        """
        original = sp.sp_mod.ShardedPool._prune_shard_and_commit
        calls = {"n": 0}

        def wrapped(pool, *args, **kwargs):
            n = calls["n"]
            calls["n"] += 1
            if before is not None and n == before:
                raise SimulatedDeath(f"killed before shard prune {n}")
            result = original(pool, *args, **kwargs)
            if after is not None and n == after:
                raise SimulatedDeath(f"killed after shard prune {n}")
            return result

        with mock.patch.object(
            sp.sp_mod.ShardedPool, "_prune_shard_and_commit", wrapped
        ):
            yield calls


# ------------------------------------------------------------------------------------------------
# 1. the uninterrupted prune
# ------------------------------------------------------------------------------------------------


class TestUninterruptedPrune(_PruneTestCase):
    def test_every_shard_identical_and_no_record_left(self):
        self.build()
        a_rows = {
            sid: self.unit_rows(sid, self.model_a.store_id) for sid in self.shard_ids
        }
        sharded = self.sharded_rows()
        others = self.replicated_rows()

        # the record is set, naming the prune and no controller, while each shard is pruned
        seen = []
        original = sp.sp_mod.ShardedPool._prune_shard_and_commit

        def watching(pool, cls_name, sid, *args, **kwargs):
            seen.append((sid, self.records()))
            return original(pool, cls_name, sid, *args, **kwargs)

        with mock.patch.object(
            sp.sp_mod.ShardedPool, "_prune_shard_and_commit", watching
        ):
            printed = self.open(prune_unvalidated=True)
        pruned = self.pool.pruned
        self.close()

        self.assertEqual(
            [
                (sid, [("prune", "BackgroundModel", None, None)])
                for sid in self.shard_ids
            ],
            seen,
        )
        self.assertEqual([], self.records())
        self.assertPruned(a_rows)
        self.assertEqual(sharded, self.sharded_rows())
        self.assertEqual(others, self.replicated_rows())

        self.assertEqual(
            {(t, sid) for t in UNIT for sid in self.shard_ids},
            {(a["class_name"], a["shard"]) for a in pruned},
        )
        self.assertIn('pruned by the pool under a "prune" record', printed)

    def test_a_prune_with_nothing_to_prune_writes_nothing(self):
        self.build(unvalidated=False)
        checksums = sp.store_checksums(self.primary)
        self.open(prune_unvalidated=True)
        self.assertEqual([], self.pool.pruned)
        self.close()
        self.assertEqual(checksums, sp.store_checksums(self.primary))

    def test_the_set_of_classes_is_read_from_the_factories(self):
        from Datastore.SQL.schema import build_schema

        built = build_schema(sqla.MetaData(), _factories)
        self.assertEqual(
            ["BackgroundModel"],
            sp.sp_mod.ShardedPool._replicated_prune_classes(replicated_tables, built),
        )
        # a replicated list naming a sharded class that prunes at startup would include it
        self.assertEqual(
            ["BackgroundModel", "GkSource"],
            sp.sp_mod.ShardedPool._replicated_prune_classes(
                ["version", "BackgroundModel", "GkSource"], built
            ),
        )


# ------------------------------------------------------------------------------------------------
# 2. an interrupted prune, completed by the next open
# ------------------------------------------------------------------------------------------------


class TestInterruptedPrune(_PruneTestCase):
    def assertCompleted(self, done_before):
        """Reopen with prune_unvalidated false: the prune is completed on the shards it had not
        reached, the record cleared, and a second reopen writes nothing."""
        printed = self.open(prune_unvalidated=False)
        rec = self.pool.reconciliation
        self.close()
        self.assertEqual("prune", rec["record"]["operation"])
        self.assertEqual("BackgroundModel", rec["record"]["class_name"])
        self.assertIsNone(rec["controller"])
        self.assertTrue(rec["cleared"])
        self.assertEqual(
            {
                (t, sid)
                for t in UNIT
                for sid in self.shard_ids
                if sid not in done_before
            },
            {(a["class_name"], a["shard"]) for a in rec["repaired"]},
        )
        self.assertIn("was opened with a prune in flight", printed)
        self.assertEqual([], self.records())
        self.assertPruned(self.a_rows)

        checksums = sp.store_checksums(self.primary)
        self.open()
        self.assertIsNone(self.pool.reconciliation["record"])
        self.close()
        self.assertEqual(checksums, sp.store_checksums(self.primary))

    def test_faulted_after_each_shard(self):
        for k in range(3):
            with self.subTest(after=k):
                self.build()
                self.a_rows = {
                    sid: self.unit_rows(sid, self.model_a.store_id)
                    for sid in self.shard_ids
                }
                with self.die_in_prune(after=k), quiet(), self.assertRaises(
                    SimulatedDeath
                ):
                    self.open(prune_unvalidated=True)
                self.cluster.pool = None

                self.assertEqual(
                    [("prune", "BackgroundModel", None, None)], self.records()
                )
                done = self.shard_ids[: k + 1]
                for sid in self.shard_ids:
                    models = [s for s, _ in self.models(sid)]
                    if sid in done:
                        self.assertNotIn(self.model_b.store_id, models, (k, sid))
                    else:
                        self.assertIn(self.model_b.store_id, models, (k, sid))

                self.assertCompleted(done)

    def test_faulted_before_the_first_shard(self):
        self.build()
        self.a_rows = {
            sid: self.unit_rows(sid, self.model_a.store_id) for sid in self.shard_ids
        }
        shards_before = {
            name: c
            for name, c in sp.store_checksums(self.primary).items()
            if name != "store.sqlite"
        }
        with self.die_in_prune(before=0), quiet(), self.assertRaises(SimulatedDeath):
            self.open(prune_unvalidated=True)
        self.cluster.pool = None
        self.assertEqual([("prune", "BackgroundModel", None, None)], self.records())
        self.assertEqual(
            shards_before,
            {
                name: c
                for name, c in sp.store_checksums(self.primary).items()
                if name != "store.sqlite"
            },
        )
        self.assertCompleted([])


# ------------------------------------------------------------------------------------------------
# 3. a prune that fails inside the factory
# ------------------------------------------------------------------------------------------------


class TestPruneFailsInTheFactory(_PruneTestCase):
    TRIGGER = (
        'CREATE TRIGGER standin_refuse BEFORE DELETE ON "BackgroundModel" '
        "BEGIN SELECT RAISE(ABORT, 'standin: delete refused'); END"
    )

    def test_the_record_stays_and_the_next_open_completes_it(self):
        """The factory catches the SQLAlchemyError its delete raises and returns normally; the
        pool sees the unvalidated row still there, rolls the shard back and raises."""
        self.build()
        self.a_rows = {
            sid: self.unit_rows(sid, self.model_a.store_id) for sid in self.shard_ids
        }
        failing = self.shard_ids[1]
        b_rows = self.unit_rows(failing, self.model_b.store_id)
        execute(self.files[failing], self.TRIGGER)

        with quiet(), self.assertRaises(RuntimeError) as raised:
            self.open(prune_unvalidated=True)
        self.cluster.pool = None
        self.assertIn("did not take", str(raised.exception))
        self.assertIn(f"shard {failing}", str(raised.exception))
        self.assertEqual([("prune", "BackgroundModel", None, None)], self.records())
        # the shard before it was pruned and committed, the failing shard rolled back whole
        # (its values and tags were deleted before the model's delete failed), the one after
        # never reached
        self.assertNotIn(
            self.model_b.store_id, [s for s, _ in self.models(self.shard_ids[0])]
        )
        self.assertEqual(b_rows, self.unit_rows(failing, self.model_b.store_id))
        self.assertIn(
            self.model_b.store_id, [s for s, _ in self.models(self.shard_ids[2])]
        )

        # a completion that fails the same way rolls every shard back and writes nothing
        checksums = sp.store_checksums(self.primary)
        with quiet(), self.assertRaises(RuntimeError):
            self.open()
        self.cluster.pool = None
        self.assertEqual(checksums, sp.store_checksums(self.primary))

        # the fault cleared, the next open completes the prune
        execute(self.files[failing], "DROP TRIGGER standin_refuse")
        self.open()
        rec = self.pool.reconciliation
        self.close()
        self.assertTrue(rec["cleared"])
        self.assertEqual(
            {(t, sid) for t in UNIT for sid in self.shard_ids[1:]},
            {(a["class_name"], a["shard"]) for a in rec["repaired"]},
        )
        self.assertEqual([], self.records())
        self.assertPruned(self.a_rows)


# ------------------------------------------------------------------------------------------------
# 4. the two kinds of record stay apart
# ------------------------------------------------------------------------------------------------


class TestRecordsStayApart(_PruneTestCase):
    def test_a_prune_record_over_a_difference_in_another_class_refuses(self):
        self.build()
        with self.die_in_prune(after=0), quiet(), self.assertRaises(SimulatedDeath):
            self.open(prune_unvalidated=True)
        self.cluster.pool = None
        # a difference the prune does not explain: a tolerance row gone from one shard
        execute(
            self.files[self.shard_ids[2]],
            "DELETE FROM tolerance WHERE serial = (SELECT max(serial) FROM tolerance)",
        )
        checksums = sp.store_checksums(self.primary)
        with quiet(), self.assertRaises(ReplicatedDivergence) as raised:
            self.open()
        self.cluster.pool = None
        e = raised.exception
        self.assertEqual("prune", e.record["operation"])
        self.assertTrue(e.after_repair)
        self.assertEqual({"tolerance"}, {d["class_name"] for d in e.differences})
        # every shard rolled back, the record left
        self.assertEqual(checksums, sp.store_checksums(self.primary))
        self.assertEqual([("prune", "BackgroundModel", None, None)], self.records())

    def test_a_prune_record_naming_a_class_that_does_not_prune_refuses(self):
        self.build(unvalidated=False)
        execute(
            self.primary,
            "INSERT INTO replication_in_flight (operation, class_name, controller_shard, "
            "store_id, started) VALUES ('prune', 'tolerance', NULL, NULL, '2026-10-03 00:00:00')",
        )
        checksums = sp.store_checksums(self.primary)
        with quiet(), self.assertRaises(ReplicatedDivergence) as raised:
            self.open()
        self.cluster.pool = None
        self.assertIn("prune of a class", str(raised.exception))
        self.assertEqual(checksums, sp.store_checksums(self.primary))

    def test_a_get_record_of_a_background_model_is_repaired_not_pruned(self):
        """An interrupted store of the unvalidated model B, whose replica never ran, with its
        record then read as a get: the check at open copies B to that replica, from the
        controller, and deletes nothing."""
        self.build(unvalidated=False)
        self.pool = self.cluster.open_pool(self.primary)
        shard_ids = list(self.pool._shards.keys())
        self.cluster.controller = shard_ids[1]
        replica = self.cluster.replica_ids()[1]
        model = sp.make_background_model(
            self.pool,
            self.cosmology,
            [ray.get(self.pool.object_get("store_tag", label="run-B"))],
            scale=2.0,
        )
        self.cluster.fault(replica, "object_store", "BackgroundModel", "before")
        with quiet(), self.assertRaises(sp.StandinActorDied):
            ray.get(self.pool.object_store(model))
        self.close()
        self.cluster.controller = None
        execute(self.primary, "UPDATE replication_in_flight SET operation = 'get'")
        held = [s for s, _ in self.models(shard_ids[1])]
        serial_b = max(held)
        self.assertNotIn(serial_b, [s for s, _ in self.models(replica)])

        self.open(prune_unvalidated=False)
        rec = self.pool.reconciliation
        self.close()
        self.assertEqual("get", rec["record"]["operation"])
        self.assertTrue(rec["cleared"])
        self.assertEqual(
            {("copied", t, replica) for t in UNIT},
            {(a["action"], a["class_name"], a["shard"]) for a in rec["repaired"]},
        )
        for sid in self.shard_ids:
            self.assertEqual(
                [(self.model_a.store_id, 1), (serial_b, 0)], self.models(sid), sid
            )
        self.assertIdentical()


# ------------------------------------------------------------------------------------------------
# 5. no actor deletes a replicated row
# ------------------------------------------------------------------------------------------------


class TestNoActorDeletesAReplicatedRow(_PruneTestCase):
    def test_under_prune_unvalidated(self):
        """With the pool's own prune stood down, so that the unvalidated model B is still there
        when the actors start: each actor's _validate_on_startup issues no DELETE on a replicated
        table, asks the BackgroundModel factory to report only, and asks a sharded class's
        factory to prune. B is still on every shard afterwards."""
        self.build()
        deletes = []
        prunes = []
        original_validate = sp.DatastoreClass._validate_on_startup

        def validate(actor):
            def on_execute(conn, cursor, statement, parameters, context, executemany):
                if statement.lstrip().upper().startswith("DELETE"):
                    deletes.append((actor._my_name, statement))

            sqla.event.listen(actor._engine, "before_cursor_execute", on_execute)
            try:
                return original_validate(actor)
            finally:
                sqla.event.remove(actor._engine, "before_cursor_execute", on_execute)

        patches = []
        for cls_name in ("BackgroundModel", "TkNumericIntegration"):
            factory = _factories[cls_name]
            original = factory.validate_on_startup

            def recording(conn, table, tables, prune=False, _o=original, _c=cls_name):
                prunes.append((_c, prune))
                return _o(conn, table, tables, prune)

            patches.append(mock.patch.object(factory, "validate_on_startup", recording))

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(sp.DatastoreClass, "_validate_on_startup", validate)
            )
            stack.enter_context(
                mock.patch.object(
                    sp.sp_mod.ShardedPool,
                    "_prune_replicated_tables",
                    lambda pool: [],
                )
            )
            for p in patches:
                stack.enter_context(p)
            with quiet():
                self.open(prune_unvalidated=True)
        self.close()

        tables = [
            re.match(r'\s*DELETE\s+FROM\s+"?(\w+)"?', s, re.IGNORECASE).group(1)
            for _, s in deletes
        ]
        self.assertEqual([], [t for t in tables if t in set(REPLICATED)])
        self.assertEqual(
            {("BackgroundModel", False), ("TkNumericIntegration", True)}, set(prunes)
        )
        self.assertEqual(6, len(prunes))
        for sid in self.shard_ids:
            self.assertIn(self.model_b.store_id, [s for s, _ in self.models(sid)])
        self.assertIdentical()


if __name__ == "__main__":
    unittest.main()
