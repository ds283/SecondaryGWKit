"""Diagnostic only: run test_store_reader with the deleted fixtures supplied from outside, so that
its tests are measured one by one rather than as one import error. The test file is not edited.
"""

import sys
import unittest

import Datastore.tests.real_store_fixtures as f

f.OLD_MISSING_TABLE = "OneLoopIntegral_tags"
f.OLD_MISSING_TABLE_SHARD = 1
f.OLD_MISSING_COLUMN_TABLE = "TkNumericIntegration"
f.OLD_MISSING_COLUMN = "stop_Tprime"
f.OLD_MISSING_COLUMN_SHARD = 0


def build_old_store(directory, stem="old-store"):
    return f.build_real_store(
        directory,
        stem=stem,
        missing_tables={f.OLD_MISSING_TABLE_SHARD: [f.OLD_MISSING_TABLE]},
        missing_columns={
            f.OLD_MISSING_COLUMN_SHARD: {
                f.OLD_MISSING_COLUMN_TABLE: [f.OLD_MISSING_COLUMN]
            }
        },
    )


f.build_old_store = build_old_store
mods = sys.argv[1:] or ["Datastore.tests.test_store_reader"]
suite = unittest.defaultTestLoader.loadTestsFromNames(mods)
r = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if r.wasSuccessful() else 1)
