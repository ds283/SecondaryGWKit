"""M7: every refusal and printed line whose source text the change touches, by scenario.

usage (from the tree, PYTHONPATH=. PYTHONHASHSEED=0): python m7_messages.py <out.json>

Scenarios are built as the existing tests build them (the reconcile and prune test cases'
fixtures), plus four the tests do not reach: a shard holding part of a model it lacks, the
pool's ReplicationMismatch detail on owned serials, an orphan tag in the inventory, and a prune
that deletes outside its unit. Paths and datetimes are normalised as in M2.
"""
import contextlib
import io
import json
import random
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import ray

from Datastore.replication import ReadOnlyMiss, ReplicatedDivergence, ReplicationMismatch
from Datastore.tests import standin_pool as sp
from Datastore.tests.test_prune_at_open import _PruneTestCase, execute as p_execute
from Datastore.tests.test_reconcile_at_open import (
    TestRecordDoesNotExplain,
    _ReconcileTestCase,
    execute,
    quiet,
)

_TMP = re.compile(r"(/private)?/(var/folders|tmp)/[^\s\"',)\]]*")
_DT = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?")
_DTREPR = re.compile(r"datetime\.datetime\([0-9, ]+\)")


def norm(text):
    return _DT.sub("<datetime>", _DTREPR.sub("<datetime>", _TMP.sub("<tmp>", text)))


messages = {}


def record(name, text):
    messages[name] = norm(text)


class M7Reconcile(TestRecordDoesNotExplain):
    def setUp(self):
        random.seed(20261006)
        super().setUp()

    def test_unit_member_of_a_model_the_shard_holds(self):
        # as test_part_of_the_recorded_background_model
        model = self.background()
        self.interrupt(
            [("r1", "object_store", "BackgroundModel", "after")],
            lambda: self.pool.object_store(model),
        )
        (rec,) = self.records()
        self.close()
        execute(
            self.files[self.replicas[0]],
            (
                "DELETE FROM BackgroundModelValue WHERE serial = (SELECT max(serial) FROM "
                "BackgroundModelValue WHERE model_serial = ?)",
                (rec["store_id"],),
            ),
        )
        record("unit: not part of a model it lacks", str(self.refused_reopen()))

    def test_unit_part_of_a_model_the_shard_lacks(self):
        # a second model's store interrupted before r0; r0 then given one tag row and one value
        # row of it, copied verbatim from the controller
        self.stored_background()
        second = self.background(tags=("run-B",), scale=2.0)
        self.interrupt(
            [("r0", "object_store", "BackgroundModel", "before")],
            lambda: self.pool.object_store(second),
        )
        (rec,) = self.records()
        self.close()
        serial = rec["store_id"]
        src = sqlite3.connect(self.files[self.controller])
        tag = src.execute(
            "SELECT * FROM BackgroundModel_tags WHERE model_serial = ?", (serial,)
        ).fetchone()
        value = src.execute(
            "SELECT * FROM BackgroundModelValue WHERE model_serial = ? ORDER BY serial LIMIT 1",
            (serial,),
        ).fetchone()
        src.close()
        execute(
            self.files[self.replicas[0]],
            ("INSERT INTO BackgroundModel_tags VALUES (" + ",".join("?" * len(tag)) + ")", tag),
            (
                "INSERT INTO BackgroundModelValue VALUES (" + ",".join("?" * len(value)) + ")",
                value,
            ),
        )
        e = self.refused_reopen()
        record("unit: holds part of a model it lacks", str(e))
        record(
            "unit: holds part of a model it lacks (order of refusals)",
            json.dumps([d["key"] + " | " + d["detail"] for d in e.differences]),
        )

    def test_flag_refusal(self):
        sp.get_redshifts(self.pool, [5.0])
        self.interrupt(
            [("controller", "object_get", "redshift", "before")],
            lambda: sp.get_redshifts(self.pool, [5.0]),
        )
        self.close()
        execute(self.files[self.replicas[0]], "UPDATE redshift SET response = 1")
        record("flag-or-validated refusal", str(self.refused_reopen()))

    def test_recompute_refusal(self):
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            e = self.refused_reopen()
        record("recompute refusal", str(e))
        record("recompute refusal: printed", out.getvalue())

    def test_shard_key_difference(self):
        self.interrupted_tolerance_get()
        execute(
            self.primary,
            'INSERT INTO "shard_keys" (key_serial, shard_id) VALUES (9999, 0)',
        )
        record("shard_key difference", str(self.refused_reopen()))

    def test_recomputed_validated_printed(self):
        # an interrupted validate repaired: the printed "recomputed validated" lines
        model = self.stored_background()
        self.close()
        for sid in self.shard_ids:
            if sid != self.controller:
                execute(self.files[sid], "UPDATE BackgroundModel SET validated = 0")
        execute(self.files[self.controller], "UPDATE BackgroundModel SET validated = 1")
        execute(
            self.primary,
            (
                'INSERT INTO "replication_in_flight" (operation, class_name, controller_shard, '
                "store_id, started) VALUES ('validate', 'BackgroundModel', ?, ?, ?)",
                (self.controller, model.store_id, "2030-01-01 00:00:00.000000"),
            ),
        )
        printed = self.reopen()
        record("validate repaired: printed", printed)
        record(
            "validate repaired: actions",
            json.dumps(self.pool.reconciliation["repaired"], default=str),
        )

    def test_version_read_only_miss(self):
        self.close()
        from config.sharding import (
            replicated_tables,
            sharded_tables,
            shard_key_type,
            shard_key_wavenumber_store_id,
        )
        from config.datastore import factories

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sp.sp_mod.ShardedPool(
                    version_label="not-held",
                    db_name=self.primary,
                    ShardKeyType=shard_key_type,
                    ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
                    replicated_tables=replicated_tables,
                    sharded_tables=sharded_tables,
                    shards=3,
                    read_only=True,
                    factories=factories,
                )
        except ReadOnlyMiss as e:
            record("version ReadOnlyMiss", str(e))
            record("version ReadOnlyMiss: fields", repr((e.class_name, e.payload)))


class M7Prune(_PruneTestCase):
    def setUp(self):
        random.seed(20261006)
        super().setUp()

    def test_a_prune_that_deletes_outside_its_unit(self):
        self.build()
        failing = self.shard_ids[1]
        p_execute(
            self.files[failing],
            'CREATE TRIGGER standin_spill AFTER DELETE ON "BackgroundModel" BEGIN '
            "DELETE FROM tolerance WHERE serial = (SELECT min(serial) FROM tolerance); END",
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.open(prune_unvalidated=True)
        except RuntimeError as e:
            record("prune outside its unit", str(e))
        self.cluster.pool = None


class M7Mismatch(unittest.TestCase):
    def test_owned_serials_detail(self):
        from config.datastore import factories

        pool = object.__new__(sp.sp_mod.ShardedPool)
        pool._factories = dict(factories)
        captured = {}

        def fake_write(operation, cls_name, submit_controller, replication, store_id=None):
            captured["replication"] = replication
            return None, None

        pool._replicated_write = fake_write

        class Row:
            def __init__(self, serial):
                self._my_id = serial

        class BackgroundModel:
            def __init__(self, serial, values):
                self._my_id = serial
                self._values = [Row(v) for v in values]

        pool._store_impl_replicated_table("BackgroundModel", BackgroundModel(7, [1, 2]))
        plan = captured["replication"](0, BackgroundModel(7, [1, 2]))
        for name, answer in (
            ("ReplicationMismatch: owned serials differ", BackgroundModel(7, [1, 3])),
            ("ReplicationMismatch: no owned rows", type("BackgroundModel", (), {"_my_id": 7})()),
        ):
            try:
                plan.check(2, answer)
            except ReplicationMismatch as e:
                record(name, str(e))
        plan.check(2, BackgroundModel(7, [1, 2]))
        record("ReplicationMismatch: equal answer", "accepted")


class M7Inventory(unittest.TestCase):
    def test_orphan_tag(self):
        from config.datastore import factories
        from Datastore.store_inventory import read_inventory
        from Datastore.tests.real_store_fixtures import build_full_store

        with tempfile.TemporaryDirectory() as d:
            store = build_full_store(Path(d))
            shards = sorted(p for p in Path(d).glob("*shard*.sqlite"))
            for path in shards:
                conn = sqlite3.connect(path)
                conn.execute(
                    "DELETE FROM store_tag WHERE serial = (SELECT min(tag_serial) FROM "
                    "BackgroundModel_tags)"
                )
                conn.commit()
                conn.close()
            inv = read_inventory(store.primary, factories)
            problems = [p for c in inv.classes.values() for p in c.problems]
            record("inventory orphan-tag problems", "\n".join(problems))


suite = unittest.TestSuite()
loader = unittest.TestLoader()
for cls in (M7Reconcile, M7Prune, M7Mismatch, M7Inventory):
    for name in loader.getTestCaseNames(cls):
        if name.startswith("test_") and name in cls.__dict__:
            suite.addTest(cls(name))
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
with open(sys.argv[1], "w") as f:
    json.dump(messages, f, indent=1, sort_keys=True)
print("scenarios recorded:", len(messages), file=sys.stderr)
