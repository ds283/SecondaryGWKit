"""
The check at open (prompts/datastore-integrity, prompt 02).

``ShardedPool``'s constructor, for an existing store, runs ``_reconcile_replicated_tables`` after
``_check_shard_files`` and before any actor exists. It rolls back hot journals, compares every
replicated table across the shards, repairs an interrupted replication from the in-flight record's
controlling shard, and refuses to open, with ``ReplicatedDivergence``, on any other difference.
These tests drive the real ``ShardedPool``, ``Datastore``, factories and broker on stand-in shards
(``Datastore.tests.standin_pool``):

1. kill and reopen, at every point of prompt 01's commit-point table, on every replicated path:
   every shard identical afterwards, the record empty, what was copied listed; a second reopen
   repairs nothing;
2. a serial split refuses, with and without a record, naming both serials, writing nothing; so
   does every other state a ``ReplicationMismatch`` leaves with the record set (a vectorized get,
   a store, a validate);
3. with no record, a missing row, an extra row, a flag, a ``validated`` difference and a
   value-count difference each refuse, writing nothing;
4. with the record set, a row the controller lacks refuses, and so do a difference in another
   class, a row the recorded write did not insert and a partial ``BackgroundModel``;
5. a hot journal on one shard is rolled back, then compared, then opened; one on the primary (a
   crash inside the record's own transaction) is rolled back before the record is read;
6. an interrupted ``BackgroundModel`` validate or store, reopened with ``prune_unvalidated`` true
   and false: every shard ends identical;
7. a clean store is untouched: no shard file and not the primary changes;
8. no Ray is initialised.

Every store is built in a temporary directory. Nothing under ``var/`` is opened.
"""

import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import ray

from Datastore.replication import ReplicatedDivergence, ReplicationMismatch
from Datastore.tests import standin_pool as sp
from config.sharding import replicated_tables

# every table the check compares: the replicated classes and BackgroundModel's tag table
REPLICATED = list(replicated_tables) + ["BackgroundModel_tags"]

# the key each compared table's rows are listed under in reconciliation["repaired"]
KEYS = {"BackgroundModel_tags": ("model_serial", "tag_serial")}


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_reconcile_at_open")


@contextlib.contextmanager
def quiet():
    """Swallow what the actor code prints when a call raises."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


def execute(path: Path, *statements):
    """Run raw SQL on one closed store file: the hand edits that build a state the write path
    cannot."""
    conn = sqlite3.connect(path)
    try:
        for statement in statements:
            if isinstance(statement, tuple):
                conn.execute(*statement)
            else:
                conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


class _ReconcileTestCase(unittest.TestCase):
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
        self.addCleanup(self.close)

        self.units = sp.make_units()
        self.cosmology = sp.StandinCosmology(self.units)
        self.fresh()

    def tearDown(self):
        self.assertFalse(ray.is_initialized())

    def fresh(self, **kwargs):
        """Close the open pool, if any, and open a new, empty store in its own directory."""
        self.close()
        self._stores += 1
        self.primary = self.root / f"store-{self._stores}" / "store.sqlite"
        self.pool = self.cluster.open_pool(self.primary, **kwargs)
        self.shard_ids = list(self.pool._shards.keys())
        self.files = dict(self.pool._shard_db_files)
        # the middle shard controls, so that replicas come before and after it
        self.cluster.controller = self.shard_ids[1]
        self.controller = self.cluster.controller
        self.replicas = self.cluster.replica_ids()
        self.roles = {
            "controller": self.controller,
            "r0": self.replicas[0],
            "r1": self.replicas[1],
        }

    def close(self):
        self.cluster.clear_faults()
        self.cluster.hooks.clear()
        if self.cluster.pool is not None:
            self.cluster.close_pool()
        self.pool = None

    def reopen(self, **kwargs):
        """Close the pool and open the store again; return what the constructor printed."""
        self.close()
        self.pool, printed = self.cluster.open_pool_output(self.primary, **kwargs)
        return printed

    def refused_reopen(self, **kwargs) -> ReplicatedDivergence:
        """Close the pool, open the store again, and require the open to be refused before any
        actor is created, with no file of the store changed."""
        self.close()
        before = sp.store_checksums(self.primary)
        constructed = []
        original = sp._Options.remote

        def counting(options, *args, **kw):
            constructed.append(options._name)
            return original(options, *args, **kw)

        with mock.patch.object(sp._Options, "remote", counting):
            with self.assertRaises(ReplicatedDivergence) as raised:
                self.cluster.open_pool_output(self.primary, **kwargs)
        self.assertEqual([], constructed, "an actor was created before the refusal")
        self.assertEqual(before, sp.store_checksums(self.primary), "a file changed")
        return raised.exception

    # -- building blocks --------------------------------------------------------------------

    def exit_time(self, k_inv_Mpc=0.5, z_exit=1.0e6):
        atol = sp.get_tolerance(self.pool, 1.0e-10)
        rtol = sp.get_tolerance(self.pool, 1.0e-9)
        k = sp.get_wavenumber(self.pool, k_inv_Mpc, self.units)
        return sp.make_exit_time(k, self.cosmology, atol, rtol, z_exit=z_exit)

    def tags(self, *labels):
        return [ray.get(self.pool.object_get("store_tag", label=l)) for l in labels]

    def background(self, tags=("run-A",), scale=1.0, zs=(1.0e4, 1.0e2, 1.0)):
        return sp.make_background_model(
            self.pool, self.cosmology, self.tags(*tags), zs=zs, scale=scale
        )

    def stored_background(self, **kwargs):
        return ray.get(self.pool.object_store(self.background(**kwargs)))

    def interrupt(self, faults, write):
        """Inject ``faults`` [(role, method, class, "before" | "after")], run ``write``, which
        must be killed, and clear the faults."""
        for role, method, cls_name, when in faults:
            self.cluster.fault(self.roles[role], method, cls_name, when)
        with quiet(), self.assertRaises(sp.StandinActorDied):
            write()
        self.cluster.clear_faults()

    def records(self):
        return sp.in_flight_records(self.pool) if self.pool is not None else []

    def primary_records(self):
        """The record table of the closed store, read mode=ro."""
        return sp._read(self.primary, 'SELECT * FROM "replication_in_flight"')

    def keys_of(self, table, sid):
        columns = KEYS.get(table, ("serial",))
        rows = sp._read(self.files[sid], f'SELECT {", ".join(columns)} FROM "{table}"')
        return {tuple(r) for r in rows}

    def all_keys(self):
        return {
            (table, sid): self.keys_of(table, sid)
            for table in REPLICATED
            for sid in self.shard_ids
        }

    def assertIdentical(self):
        """Every column of every row of every compared table is the same on every shard."""
        for table in REPLICATED:
            rows = sp.shard_rows(self.pool, table)
            for sid in self.shard_ids:
                self.assertEqual(
                    rows[self.controller], rows[sid], f"{table}: shard {sid}"
                )

    def assertNoRecord(self):
        self.assertEqual([], self.records())


# ------------------------------------------------------------------------------------------------
# 1. kill and reopen, at every point of the commit-point table
# ------------------------------------------------------------------------------------------------


class TestKillAndReopen(_ReconcileTestCase):
    """
    Each case is a row of log 01's commit-point table ("After"): the faults that stop the write
    at that point, and what the reopen must write, as (action, class, replica role). A fault
    "before" on a replica means that replica's call never ran; "after" means it ran and
    committed and then the actor, or the driver, died.
    """

    def run_case(self, prepare, write, faults, expected, prune=False):
        self.fresh()
        objects = prepare()
        self.interrupt(faults, lambda: write(objects))
        records = self.records()
        self.assertEqual(1, len(records), "the write left no record")
        record = records[0]
        self.assertEqual(self.controller, record["controller_shard"])
        before = self.all_keys()
        controller_rows = {
            t: sp.shard_rows(self.pool, t)[self.controller] for t in REPLICATED
        }

        printed = self.reopen(prune_unvalidated=prune)
        rec = self.pool.reconciliation

        # the record is empty, and the reconciliation says what it was and cleared it
        self.assertNoRecord()
        self.assertTrue(rec["cleared"])
        self.assertEqual(record["operation"], rec["record"]["operation"])
        self.assertEqual(record["class_name"], rec["record"]["class_name"])
        self.assertEqual(self.controller, rec["controller"])

        # every shard identical, and the controller's rows unchanged except a recomputed flag
        self.assertIdentical()
        if not any(a == "validated recomputed" for a, _, _ in expected):
            for t in REPLICATED:
                self.assertEqual(
                    controller_rows[t],
                    sp.shard_rows(self.pool, t)[self.controller],
                    f"{t}: the controller's rows changed",
                )

        # what was written, from the controller, and each copy's keys exactly what was missing
        roles = {sid: role for role, sid in self.roles.items()}
        done = {
            (a["action"], a["class_name"], roles[a["shard"]]) for a in rec["repaired"]
        }
        self.assertEqual(set(expected), done)
        for a in rec["repaired"]:
            if a["action"] in ("copied", "flags set"):
                self.assertEqual(self.controller, a["from"])
            if a["action"] == "copied":
                table, sid = a["class_name"], a["shard"]
                missing = before[(table, self.controller)] - before[(table, sid)]
                self.assertEqual(sorted(missing), a["keys"], (table, sid))

        # printed: the record, one line per class and shard written, the clearing
        self.assertIn("was opened with a replicated write in flight", printed)
        lines = [l for l in printed.splitlines() if l.startswith(">>   ")]
        self.assertEqual(
            len(rec["repaired"]) + (0 if rec["repaired"] else 1) + 1, len(lines)
        )
        self.assertIn("the replication_in_flight record was cleared", printed)

        # a second reopen repairs nothing and writes nothing
        self.close()
        checksums = sp.store_checksums(self.primary)
        printed = self.reopen()
        self.assertIsNone(self.pool.reconciliation["record"])
        self.assertEqual([], self.pool.reconciliation["repaired"])
        self.assertNotIn("in flight", printed)
        self.close()
        self.assertEqual(checksums, sp.store_checksums(self.primary))

    # -- get ---------------------------------------------------------------------------------

    def test_scalar_get(self):
        write = lambda _: sp.get_tolerance(self.pool, 1.0e-6)
        cls = "tolerance"
        cases = {
            "P1-C": ([("controller", "object_get", cls, "before")], []),
            "C-P2": (
                [("controller", "object_get", cls, "after")],
                [("copied", cls, "r0"), ("copied", cls, "r1")],
            ),
            "P2-R1": (
                [
                    ("r0", "object_get", cls, "before"),
                    ("r1", "object_get", cls, "before"),
                ],
                [("copied", cls, "r0"), ("copied", cls, "r1")],
            ),
            "Ri-Ri+1, r0 lacks": (
                [("r0", "object_get", cls, "before")],
                [("copied", cls, "r0")],
            ),
            "Ri-Ri+1, r1 lacks": (
                [("r1", "object_get", cls, "before")],
                [("copied", cls, "r1")],
            ),
            "Rn-P3": ([("r1", "object_get", cls, "after")], []),
        }
        for point, (faults, expected) in cases.items():
            with self.subTest(point=point):
                self.run_case(lambda: None, write, faults, expected)

    def test_scalar_get_of_the_shard_key(self):
        """A wavenumber: after the repair the new key has no shard assignment, and the next get
        assigns it (log 01: _assign_shard_keys self-heals)."""
        cls = "wavenumber"
        write = lambda _: sp.get_wavenumber(self.pool, 0.25, self.units)
        self.run_case(
            lambda: None,
            write,
            [("r0", "object_get", cls, "before")],
            [("copied", cls, "r0")],
        )
        self.reopen()
        k = sp.get_wavenumber(self.pool, 0.25, self.units)
        self.assertIn(k.store_id, self.pool._shard_keys)

    def test_scalar_get_that_turns_a_flag_on(self):
        cls = "wavenumber"

        def prepare():
            sp.get_wavenumber(self.pool, 0.25, self.units)

        def write(_):
            ray.get(
                self.pool.object_get(
                    cls,
                    k_inv_Mpc=0.25,
                    units=self.units,
                    is_source=False,
                    is_response=True,
                )
            )

        self.run_case(
            prepare,
            write,
            [("r1", "object_get", cls, "before")],
            [("flags set", cls, "r1")],
        )

    def test_vectorized_get(self):
        cls = "redshift"
        write = lambda _: sp.get_redshifts(self.pool, [5.0, 50.0])
        cases = {
            "P1-C": ([("controller", "object_get", cls, "before")], []),
            "C-R1": (
                [("controller", "object_get", cls, "after")],
                [("copied", cls, "r0"), ("copied", cls, "r1")],
            ),
            "Ri-Ri+1, r0 lacks": (
                [("r0", "object_get", cls, "before")],
                [("copied", cls, "r0")],
            ),
            "Ri-Ri+1, r1 lacks": (
                [("r1", "object_get", cls, "before")],
                [("copied", cls, "r1")],
            ),
            "Rn-P3": ([("r0", "object_get", cls, "after")], []),
        }
        for point, (faults, expected) in cases.items():
            with self.subTest(point=point):
                self.run_case(lambda: None, write, faults, expected)

    def test_vectorized_get_that_inserts_and_turns_a_flag_on(self):
        cls = "redshift"

        def prepare():
            sp.get_redshifts(self.pool, [5.0])

        def write(_):
            ray.get(
                self.pool.object_get(
                    cls,
                    payload_data=[
                        {"z": 5.0, "is_source": False, "is_response": True},
                        {"z": 7.0, "is_source": True, "is_response": False},
                    ],
                )
            )

        self.run_case(
            prepare,
            write,
            [("r0", "object_get", cls, "before")],
            [("copied", cls, "r0"), ("flags set", cls, "r0")],
        )

    def test_get_that_inserts_nothing(self):
        cls = "tolerance"
        prepare = lambda: sp.get_tolerance(self.pool, 1.0e-6)
        write = lambda _: sp.get_tolerance(self.pool, 1.0e-6)
        for point, when in (("P1-C", "before"), ("C-P3", "after")):
            with self.subTest(point=point):
                self.run_case(
                    prepare, write, [("controller", "object_get", cls, when)], []
                )

    # -- store -------------------------------------------------------------------------------

    def _store_cases(self, cls):
        return {
            "P1-C": ([("controller", "object_store", cls, "before")], []),
            "C-P2": ([("controller", "object_store", cls, "after")], ["r0", "r1"]),
            "P2-R1": (
                [
                    ("r0", "object_store", cls, "before"),
                    ("r1", "object_store", cls, "before"),
                ],
                ["r0", "r1"],
            ),
            "Ri-Ri+1, r0 lacks": ([("r0", "object_store", cls, "before")], ["r0"]),
            "Ri-Ri+1, r1 lacks": ([("r1", "object_store", cls, "before")], ["r1"]),
            "Rn-P3": ([("r1", "object_store", cls, "after")], []),
        }

    def test_store_of_an_exit_time(self):
        cls = "wavenumber_exit_time"
        prepare = lambda: self.exit_time()
        write = lambda exit_time: self.pool.object_store(exit_time)
        for point, (faults, lacking) in self._store_cases(cls).items():
            with self.subTest(point=point):
                expected = [("copied", cls, role) for role in lacking]
                self.run_case(prepare, write, faults, expected)

    def test_store_of_a_background_model(self):
        cls = "BackgroundModel"
        prepare = lambda: self.background(tags=("run-A", "grid-B"))
        write = lambda model: self.pool.object_store(model)
        unit = ("BackgroundModel", "BackgroundModel_tags", "BackgroundModelValue")
        for point, (faults, lacking) in self._store_cases(cls).items():
            with self.subTest(point=point):
                expected = [("copied", t, role) for role in lacking for t in unit]
                self.run_case(prepare, write, faults, expected)

    def test_store_of_a_second_background_model_beside_the_first(self):
        """A second model whose row equals the first's in every column but its serial (they
        differ in their tags and values) is copied as its own unit, not taken for the first.
        """
        cls = "BackgroundModel"

        def prepare():
            self.assertTrue(
                ray.get(self.pool.object_validate(self.stored_background()))
            )
            return self.background(tags=("run-B",), scale=2.0)

        write = lambda model: self.pool.object_store(model)
        unit = ("BackgroundModel", "BackgroundModel_tags", "BackgroundModelValue")
        self.run_case(
            prepare,
            write,
            [("r0", "object_store", cls, "before")],
            [("copied", t, "r0") for t in unit],
        )

    # -- validate ----------------------------------------------------------------------------

    def test_validate_of_a_background_model(self):
        cls = "BackgroundModel"
        prepare = lambda: self.stored_background()
        write = lambda model: self.pool.object_validate(model)
        cases = {
            "P1-C": ([("controller", "object_validate", cls, "before")], []),
            "C-R1": (
                [
                    ("r0", "object_validate", cls, "before"),
                    ("r1", "object_validate", cls, "before"),
                ],
                [
                    ("validated recomputed", cls, "r0"),
                    ("validated recomputed", cls, "r1"),
                ],
            ),
            "Ri-Ri+1, r0 lacks": (
                [("r0", "object_validate", cls, "before")],
                [("validated recomputed", cls, "r0")],
            ),
            "Ri-Ri+1, r1 lacks": (
                [("r1", "object_validate", cls, "before")],
                [("validated recomputed", cls, "r1")],
            ),
            "Rn-P3": ([("r1", "object_validate", cls, "after")], []),
        }
        for point, (faults, expected) in cases.items():
            with self.subTest(point=point):
                self.run_case(prepare, write, faults, expected)
                validated = sp._read(
                    self.files[self.controller],
                    'SELECT validated FROM "BackgroundModel"',
                )
                self.assertEqual([(0 if point == "P1-C" else 1,)], validated)

    def test_validate_whose_controller_counts_short(self):
        """The controller's outcome False is not replicated; a failure at P1-C or C-P3 leaves
        the record over identical shards, which is cleared."""
        cls = "BackgroundModel"

        def prepare():
            model = self.stored_background()
            self.close()
            # the model's value rows short on every shard alike, so every copy counts short
            for sid in self.shard_ids:
                execute(
                    self.files[sid],
                    "DELETE FROM BackgroundModelValue WHERE serial = "
                    "(SELECT max(serial) FROM BackgroundModelValue)",
                )
            self.pool = self.cluster.open_pool(self.primary)
            return model

        write = lambda model: self.pool.object_validate(model)
        for point, when in (("P1-C", "before"), ("C-P3", "after")):
            with self.subTest(point=point):
                self.run_case(
                    prepare, write, [("controller", "object_validate", cls, when)], []
                )

    # -- after the record is cleared -----------------------------------------------------------

    def test_between_the_record_cleared_and_the_shard_key(self):
        """P3-K: the shards agree and there is no record; the reopen writes nothing, and the next
        get of the wavenumber assigns its shard key."""

        def dies(pool, obj):
            raise sp.StandinActorDied("killed before _assign_shard_keys committed")

        with mock.patch.object(sp.sp_mod.ShardedPool, "_assign_shard_keys", dies):
            with self.assertRaises(sp.StandinActorDied):
                sp.get_wavenumber(self.pool, 0.25, self.units)
        self.assertNoRecord()
        self.close()
        checksums = sp.store_checksums(self.primary)
        self.reopen()
        self.assertIsNone(self.pool.reconciliation["record"])
        self.assertEqual([], self.pool.reconciliation["repaired"])
        self.assertEqual({}, self.pool._shard_keys)
        k = sp.get_wavenumber(self.pool, 0.25, self.units)
        self.assertIn(k.store_id, self.pool._shard_keys)
        self.close()
        self.assertNotEqual(checksums, sp.store_checksums(self.primary))


# ------------------------------------------------------------------------------------------------
# 2. a serial split refuses
# ------------------------------------------------------------------------------------------------


class TestSerialSplit(_ReconcileTestCase):
    """The audit probe's step 4, built as test_replicated_write builds it: replica r0 holds the
    tolerance under a serial of its own, and a replicated get then inserts it on the controller
    under another, which r1 receives and r0 answers with its own."""

    def setUp(self):
        super().setUp()
        replica = self.replicas[0]
        self.own = ray.get(
            self.pool._shards[replica].object_get.remote("tolerance", tol=1.0e-6)
        ).store_id
        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            sp.get_tolerance(self.pool, 1.0e-6)
        self.new = raised.exception.controller_serial
        self.assertNotEqual(self.own, self.new)

    def assertNamesTheSplit(self, e: ReplicatedDivergence):
        text = str(e)
        self.assertIn("serial split", text)
        self.assertIn(f"serial {self.own}", text)
        self.assertIn(f"serial {self.new}", text)
        self.assertIn('class "tolerance"', text)
        self.assertIn("Nothing was repaired", text)
        self.assertIn("needs a person, or regeneration", text)

    def test_with_the_record_set(self):
        self.assertEqual(1, len(self.records()))
        e = self.refused_reopen()
        self.assertNamesTheSplit(e)
        self.assertIsNotNone(e.record)
        self.assertEqual(1, len(self.primary_records()))

    def test_with_no_record(self):
        self.close()
        execute(self.primary, 'DELETE FROM "replication_in_flight"')
        e = self.refused_reopen()
        self.assertNamesTheSplit(e)
        self.assertIsNone(e.record)
        self.assertIn("there is no replication_in_flight record", str(e))

    def test_the_refusal_pickles_with_its_fields(self):
        import pickle

        e = self.refused_reopen()
        f = pickle.loads(pickle.dumps(e))
        self.assertEqual(str(e), str(f))
        self.assertEqual(e.differences, f.differences)
        self.assertEqual(e.record, f.record)


class TestMismatchLeftByTheCheck(_ReconcileTestCase):
    """The other states a ReplicationMismatch leaves at Rn-P3 with the record set: whatever the
    mismatch reports differs, and the reopen refuses it."""

    def test_a_vectorized_get(self):
        own = ray.get(
            self.pool._shards[self.replicas[1]].object_get.remote(
                "redshift", z=50.0, is_source=True, is_response=False
            )
        ).store_id
        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            sp.get_redshifts(self.pool, [5.0, 50.0])
        new = raised.exception.controller_serial
        e = self.refused_reopen()
        self.assertIn("serial split", str(e))
        self.assertIn(f"serial {own}", str(e))
        self.assertIn(f"serial {new}", str(e))

    def test_a_store(self):
        exit_time = self.exit_time()
        own = ray.get(
            self.pool._shards[self.replicas[0]].object_store.remote(exit_time)
        ).store_id
        with quiet(), self.assertRaises(ReplicationMismatch) as raised:
            self.pool.object_store(exit_time)
        new = raised.exception.controller_serial
        e = self.refused_reopen()
        self.assertIn('class "wavenumber_exit_time"', str(e))
        self.assertIn(f"serial {own}", str(e))
        self.assertIn(f"serial {new}", str(e))

    def test_a_validate(self):
        """r0 counts short (one of its value rows removed by hand while the pool is open), so its
        validation outcome differs from the controller's and the check raises."""
        model = self.stored_background()
        execute(
            self.files[self.replicas[0]],
            "DELETE FROM BackgroundModelValue WHERE serial = "
            "(SELECT max(serial) FROM BackgroundModelValue)",
        )
        with quiet(), self.assertRaises(ReplicationMismatch):
            self.pool.object_validate(model)
        self.assertEqual(1, len(self.records()))
        e = self.refused_reopen()
        self.assertIn('class "BackgroundModelValue"', str(e))
        self.assertIn("a validate writes no row", str(e))


# ------------------------------------------------------------------------------------------------
# 3. no record: any difference refuses
# ------------------------------------------------------------------------------------------------


class TestNoRecordRefuses(_ReconcileTestCase):
    def setUp(self):
        super().setUp()
        sp.get_tolerance(self.pool, 1.0e-6)
        sp.get_redshifts(self.pool, [5.0, 50.0])
        self.exit = ray.get(self.pool.object_store(self.exit_time()))
        self.model = self.stored_background()
        self.assertTrue(ray.get(self.pool.object_validate(self.model)))
        self.assertNoRecord()
        self.close()

    def refuse_after(self, *statements, shard_role="r0"):
        execute(self.files[self.roles[shard_role]], *statements)
        e = self.refused_reopen()
        self.assertIsNone(e.record)
        self.assertIn("there is no replication_in_flight record", str(e))
        self.assertIn("Nothing was repaired", str(e))
        return e

    def test_a_missing_row(self):
        e = self.refuse_after("DELETE FROM tolerance WHERE log10_tol = -6.0")
        self.assertEqual(["tolerance"], [d["class_name"] for d in e.differences])
        self.assertEqual([self.replicas[0]], e.differences[0]["shard"])

    def test_an_extra_row(self):
        e = self.refuse_after(
            "INSERT INTO tolerance (serial, timestamp, log10_tol) "
            "VALUES (9999, '2001-01-01 00:00:00.000000', -3.0)"
        )
        self.assertIn("serial 9999", str(e))

    def test_a_flag(self):
        e = self.refuse_after("UPDATE redshift SET response = 1 WHERE z = 5.0")
        self.assertIn("response", str(e))
        self.assertEqual(["redshift"], [d["class_name"] for d in e.differences])

    def test_a_validated_difference(self):
        e = self.refuse_after("UPDATE BackgroundModel SET validated = 0")
        self.assertIn("validated", str(e))

    def test_a_value_count_difference(self):
        e = self.refuse_after(
            "DELETE FROM BackgroundModelValue WHERE serial = "
            "(SELECT max(serial) FROM BackgroundModelValue)"
        )
        self.assertEqual(
            ["BackgroundModelValue"], [d["class_name"] for d in e.differences]
        )

    def test_a_store_written_before_the_record_existed(self):
        """A store without the replication_in_flight table reads as no record; a difference
        refuses, and the primary is not given the table."""
        execute(self.primary, 'DROP TABLE "replication_in_flight"')
        self.refuse_after("DELETE FROM tolerance WHERE log10_tol = -6.0")
        tables = sp._read(
            self.primary,
            "SELECT name FROM sqlite_master WHERE name = 'replication_in_flight'",
        )
        self.assertEqual([], tables)

    def test_a_difference_in_a_sharded_table_is_not_looked_at(self):
        """Sharded rows live on one shard; a hand edit to one is neither repaired nor refused."""
        self.pool = self.cluster.open_pool(self.primary)
        stored = ray.get(self.pool.object_store(sp.make_policy_data(self.exit)))
        shard = self.pool._shard_keys[self.exit.k.store_id]
        self.close()
        execute(
            self.files[shard],
            f"UPDATE GkSourcePolicyData SET quality = 'edited' WHERE serial = {stored.store_id}",
        )
        checksums = sp.store_checksums(self.primary)
        self.reopen()
        self.assertEqual([], self.pool.reconciliation["repaired"])
        self.assertNotIn("GkSourcePolicyData", self.pool.reconciliation["compared"])
        self.close()
        self.assertEqual(checksums, sp.store_checksums(self.primary))


# ------------------------------------------------------------------------------------------------
# 4. with the record set: what the record does not explain refuses
# ------------------------------------------------------------------------------------------------


class TestRecordDoesNotExplain(_ReconcileTestCase):
    def interrupted_tolerance_get(self):
        """The record left by a tolerance get whose replica r0 died: the controller and r1 hold
        the row, r0 lacks it."""
        self.interrupt(
            [("r0", "object_get", "tolerance", "before")],
            lambda: sp.get_tolerance(self.pool, 1.0e-6),
        )
        self.assertEqual(1, len(self.records()))
        self.close()

    def assertRefusedWithTheRecord(self, *phrases):
        e = self.refused_reopen()
        self.assertIsNotNone(e.record)
        self.assertEqual(1, len(self.primary_records()), "the record was touched")
        for phrase in phrases:
            self.assertIn(phrase, str(e))
        return e

    def test_a_row_the_controller_lacks(self):
        self.interrupted_tolerance_get()
        execute(
            self.files[self.replicas[1]],
            "INSERT INTO tolerance (serial, timestamp, log10_tol) "
            "VALUES (9999, '2001-01-01 00:00:00.000000', -3.0)",
        )
        self.assertRefusedWithTheRecord(
            "serial 9999", f"controlling shard {self.controller} lacks this row"
        )

    def test_a_record_naming_another_class(self):
        sp.get_redshifts(self.pool, [5.0])
        self.interrupted_tolerance_get()
        execute(self.files[self.replicas[1]], "DELETE FROM redshift WHERE z = 5.0")
        self.assertRefusedWithTheRecord(
            'class "redshift"', 'the record names "tolerance"'
        )

    def test_a_row_the_recorded_write_did_not_insert(self):
        """The controller holds a row of the record's class that r0 lacks, but its timestamp is
        not the record's start: an older difference, which the record does not license.
        """
        sp.get_tolerance(self.pool, 1.0e-4)
        self.close()
        execute(
            self.files[self.replicas[0]], "DELETE FROM tolerance WHERE log10_tol = -4.0"
        )
        # the record of a later get of the class, written by hand over that older difference
        execute(
            self.primary,
            (
                'INSERT INTO "replication_in_flight" (operation, class_name, controller_shard, '
                "store_id, started) VALUES ('get', 'tolerance', ?, NULL, ?)",
                (self.controller, "2030-01-01 00:00:00.000000"),
            ),
        )
        self.assertRefusedWithTheRecord("is not the recorded write's start")

    def test_a_flag_the_controller_lacks(self):
        sp.get_redshifts(self.pool, [5.0])
        self.interrupt(
            [("controller", "object_get", "redshift", "before")],
            lambda: sp.get_redshifts(self.pool, [5.0]),
        )
        self.close()
        execute(self.files[self.replicas[0]], "UPDATE redshift SET response = 1")
        self.assertRefusedWithTheRecord("response")

    def test_part_of_a_background_model(self):
        """A store of a second model interrupted before r0: r0 lacks the whole unit, which is
        licensed, and also one value row of the first model, which is not."""
        self.stored_background()
        second = self.background(tags=("run-B",), scale=2.0)
        self.interrupt(
            [("r0", "object_store", "BackgroundModel", "before")],
            lambda: self.pool.object_store(second),
        )
        self.close()
        execute(
            self.files[self.replicas[0]],
            "DELETE FROM BackgroundModelValue WHERE serial = "
            "(SELECT min(serial) FROM BackgroundModelValue)",
        )
        self.assertRefusedWithTheRecord('class "BackgroundModelValue"')

    def test_a_recomputed_validated_that_disagrees(self):
        """A validate record whose controller holds validated=True over value rows that are
        short on every shard alike: the factory's rule, run on each shard's own rows, gives False,
        so nothing is turned on and the open refuses. A copied flag would have agreed.
        """
        model = self.stored_background()
        self.close()
        for sid in self.shard_ids:
            execute(
                self.files[sid],
                "DELETE FROM BackgroundModelValue WHERE serial = "
                "(SELECT max(serial) FROM BackgroundModelValue)",
            )
        execute(self.files[self.controller], "UPDATE BackgroundModel SET validated = 1")
        execute(
            self.primary,
            (
                'INSERT INTO "replication_in_flight" (operation, class_name, controller_shard, '
                "store_id, started) VALUES ('validate', 'BackgroundModel', ?, ?, ?)",
                (self.controller, model.store_id, "2030-01-01 00:00:00.000000"),
            ),
        )
        with quiet():
            self.assertRefusedWithTheRecord('class "BackgroundModel"', "validated")

    def test_a_shard_key_the_repair_does_not_supply(self):
        """The repair itself is licensed, but the comparison made after it still finds the
        primary assigning a shard key to a wavenumber no shard holds: the repair is rolled back,
        and the record is left set."""
        self.interrupted_tolerance_get()
        execute(
            self.primary,
            'INSERT INTO "shard_keys" (key_serial, shard_id) VALUES (9999, 0)',
        )
        e = self.assertRefusedWithTheRecord("key_serial 9999", "rolled back")
        self.assertTrue(e.after_repair)


# ------------------------------------------------------------------------------------------------
# 5. a hot journal
# ------------------------------------------------------------------------------------------------


class TestHotJournal(_ReconcileTestCase):
    def test_a_hot_journal_on_one_shard_is_rolled_back_then_compared(self):
        sp.get_tolerance(self.pool, 1.0e-6)
        self.close()
        shard = self.files[self.replicas[0]]
        journal = Path(str(shard) + "-journal")

        # hot_journal_probe.py's method: a write transaction large enough to spill to the file,
        # in a child killed before it commits
        # the child only opens the shard, writes and exits: whatever threads this process has
        # are not touched, so fork()'s multi-threading DeprecationWarning is silenced here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            try:
                c = sqlite3.connect(shard)
                c.execute("PRAGMA cache_size = 1")
                c.execute("BEGIN IMMEDIATE")
                for i in range(2, 5000):
                    c.execute(
                        "INSERT INTO tolerance (serial, timestamp, log10_tol) VALUES (?, ?, ?)",
                        (100000 + i, "2001-01-01 00:00:00.000000", -float(i)),
                    )
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        self.assertTrue(journal.exists(), "the child left no journal")

        self.reopen()
        self.assertFalse(journal.exists())
        self.assertEqual([str(shard)], self.pool.reconciliation["hot_journals"])
        self.assertEqual([], self.pool.reconciliation["repaired"])
        self.assertIdentical()
        self.assertEqual(1, len(self.keys_of("tolerance", self.replicas[0])))

    def test_a_hot_journal_on_the_primary_is_rolled_back_before_the_record_is_read(
        self,
    ):
        """A crash inside the record's own transaction (P1): the primary's journal is rolled back
        by the constructor's first read of it, the record it was writing is gone, and the store
        opens as a clean one."""
        sp.get_tolerance(self.pool, 1.0e-6)
        self.close()
        primary = Path(str(self.primary))
        journal = Path(str(primary) + "-journal")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            try:
                c = sqlite3.connect(primary)
                c.execute("PRAGMA cache_size = 1")
                c.execute("BEGIN IMMEDIATE")
                for i in range(5000):
                    c.execute(
                        'INSERT INTO "replication_in_flight" (operation, class_name, '
                        "controller_shard, store_id, started) VALUES (?, ?, ?, NULL, ?)",
                        ("get", "x" * 200, 0, "2001-01-01 00:00:00.000000"),
                    )
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        self.assertTrue(journal.exists(), "the child left no journal")

        self.reopen()
        self.assertFalse(journal.exists())
        self.assertEqual(
            [str(self.pool.primary)], self.pool.reconciliation["hot_journals"]
        )
        self.assertIsNone(self.pool.reconciliation["record"])
        self.assertNoRecord()
        self.assertIdentical()


# ------------------------------------------------------------------------------------------------
# 6. pruning after repair
# ------------------------------------------------------------------------------------------------


class TestPruningAfterRepair(_ReconcileTestCase):
    """prune_unvalidated runs after the check: for a replicated class in the pool, under a prune
    record, before any actor exists (prompts/a3-v2-readiness prompt 02), and for a sharded class
    inside each actor; drop_actions run inside each actor. Every shard is identical by then, so
    they act identically."""

    def test_an_interrupted_validate(self):
        for prune in (True, False):
            with self.subTest(prune=prune):
                self.fresh()
                model = self.stored_background()
                self.interrupt(
                    [("r0", "object_validate", "BackgroundModel", "before")],
                    lambda: self.pool.object_validate(model),
                )
                with quiet():
                    self.reopen(prune_unvalidated=prune)
                self.assertNoRecord()
                self.assertIdentical()
                for sid in self.shard_ids:
                    self.assertEqual(
                        [(model.store_id, 1)],
                        sp._read(
                            self.files[sid],
                            'SELECT serial, validated FROM "BackgroundModel"',
                        ),
                    )

    def test_an_interrupted_store(self):
        """The copied model is unvalidated everywhere: pruned on every shard, or kept on every
        shard, and never on some. The drop actions include gk-source-policy-records, which drops
        the replicated GkSourcePolicy table: on every shard alike."""
        for prune in (True, False):
            with self.subTest(prune=prune):
                self.fresh()
                ray.get(
                    self.pool.object_get(
                        "GkSourcePolicy",
                        label="standin-policy",
                        Levin_threshold=1.5,
                        numeric_policy="maximize-numeric",
                    )
                )
                model = self.background()
                self.interrupt(
                    [("r1", "object_store", "BackgroundModel", "before")],
                    lambda: self.pool.object_store(model),
                )
                with quiet():
                    self.reopen(
                        prune_unvalidated=prune,
                        drop_actions=["gk-source", "gk-source-policy-records"],
                    )
                self.assertNoRecord()
                self.assertIdentical()
                counts = {
                    sid: sp._read(
                        self.files[sid], 'SELECT count(*) FROM "BackgroundModel"'
                    )[0][0]
                    for sid in self.shard_ids
                }
                self.assertEqual(
                    {sid: 0 if prune else 1 for sid in self.shard_ids}, counts
                )
                for sid in self.shard_ids:
                    self.assertEqual(
                        [(0,)],
                        sp._read(
                            self.files[sid], 'SELECT count(*) FROM "GkSourcePolicy"'
                        ),
                    )


# ------------------------------------------------------------------------------------------------
# 7. a clean store is untouched
# ------------------------------------------------------------------------------------------------


class TestCleanStoreUntouched(_ReconcileTestCase):
    def test_opening_a_clean_store_writes_nothing(self):
        sp.get_tolerance(self.pool, 1.0e-6)
        sp.get_redshifts(self.pool, [5.0, 50.0])
        exit_time = ray.get(self.pool.object_store(self.exit_time()))
        model = self.stored_background(tags=("run-A", "grid-B"))
        self.assertTrue(ray.get(self.pool.object_validate(model)))
        ray.get(self.pool.object_store(sp.make_policy_data(exit_time)))
        self.close()

        checksums = sp.store_checksums(self.primary)
        printed = self.reopen()
        rec = self.pool.reconciliation
        self.close()
        self.assertEqual(checksums, sp.store_checksums(self.primary))

        self.assertIsNone(rec["record"])
        self.assertEqual([], rec["repaired"])
        self.assertEqual([], rec["hot_journals"])
        self.assertFalse(rec["cleared"])
        self.assertEqual(REPLICATED, rec["compared"])
        self.assertNotIn(">>   ", printed)

    def test_a_new_store_has_no_reconciliation(self):
        self.assertIsNone(self.pool.reconciliation)


if __name__ == "__main__":
    unittest.main()
