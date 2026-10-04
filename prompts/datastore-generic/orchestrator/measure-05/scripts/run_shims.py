"""Diagnostic only: confirm the proposed one-line test changes are sufficient, by patching module
globals from outside (no test file is edited)."""

import dataclasses
import sys
import unittest

import sqlalchemy as sqla

from Datastore.SQL.schema import StoreSchemaMismatch
import Datastore.store_inventory as inv

# 1. the primary without replication_in_flight: OperationalError -> StoreSchemaMismatch
import Datastore.tests.test_reconcile_at_open as rao

rao.OperationalError = StoreSchemaMismatch

# 2. the WKB value reads: the precondition column numeric_serial -> rho_gauss_order
import Datastore.tests.test_database_errors_reach_the_caller as dercc

dercc.NUMERIC_SERIAL = dercc.RHO_GAUSS_ORDER


# 3. ShardContext built by hand with absent_tables / absent_columns: the two arguments dropped
def _ctx(**kw):
    kw.pop("absent_tables", None)
    kw.pop("absent_columns", None)
    return inv.ShardContext(**kw)


import Datastore.tests.test_quadsource_parent_key as qpk
import Datastore.tests.test_wkb_numeric_parent_key as wnpk

qpk.ShardContext = _ctx
wnpk.ShardContext = _ctx

# 4. test_store_reader's helpers without the absent filters (fixture constants supplied so it imports)
import Datastore.tests.real_store_fixtures as f

for k, v in dict(
    OLD_MISSING_TABLE="x",
    OLD_MISSING_TABLE_SHARD=1,
    OLD_MISSING_COLUMN_TABLE="x",
    OLD_MISSING_COLUMN="x",
    OLD_MISSING_COLUMN_SHARD=0,
    build_old_store=None,
).items():
    setattr(f, k, v)
import Datastore.tests.test_store_reader as tsr


def read_every_table(store):
    counts = {}
    for shard in store.shards:
        counts[shard.serial] = {}
        with shard.engine.connect() as conn:
            for name, table in shard.tables.items():
                counts[shard.serial][name] = len(
                    conn.execute(sqla.select(table)).fetchall()
                )
    return counts


tsr.read_every_table = read_every_table
tsr._CHILD = tsr._CHILD.replace(
    "                if name in shard.absent_tables:\n                    continue\n",
    "",
)

names = [
    "Datastore.tests.test_reconcile_at_open.TestNoRecordRefuses.test_a_primary_without_the_record_table_is_refused",
    "Datastore.tests.test_database_errors_reach_the_caller.TestTheWKBValueReads",
    "Datastore.tests.test_quadsource_parent_key.TestCrossShardParentUnit",
    "Datastore.tests.test_wkb_numeric_parent_key.TestNullableParent",
    "Datastore.tests.test_store_reader.TestReaderReads",
    "Datastore.tests.test_store_reader.TestReaderNeverWrites",
    "Datastore.tests.test_store_reader.TestNoRay",
]
suite = unittest.defaultTestLoader.loadTestsFromNames(names)
r = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() else 1)
