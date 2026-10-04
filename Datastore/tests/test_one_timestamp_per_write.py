"""
One timestamp per replicated write, on every path that writes a replicated row
(prompts/datastore-generic, prompt 03, R2).

The check at open compares every column of every replicated table, ``timestamp`` included. That is
sound only if every path that writes a replicated row gives every shard's copy the same timestamp.
Each test here runs an existing test of one such path with the Datastore module's ``datetime``
replaced by ``test_replicated_write._TickingDatetime``, whose ``now()`` is a new value on every
call: a shard that stamped its own time would stamp a value no other shard holds, and a value in
2030, which no driver's clock gives. Each existing test already asserts full-row identity
(``SELECT *``, timestamp included) on every shard afterwards; here that assertion also requires
that no replicated row on any shard carries a shard's tick.

1. the clean write (get, store, validate): ``test_replicated_write.TestCleanWrite.
   test_every_copy_is_row_identical``, which already runs under the ticking clock; not repeated;
2. the repair at open of an interrupted get, store or validate, at every point of the
   commit-point table: ``test_reconcile_at_open.TestKillAndReopen``, and the same repair followed
   by the prune at open, ``TestPruningAfterRepair``;
3. the completion at open of an interrupted prune: ``test_prune_at_open.TestInterruptedPrune``;
4. the prune at open: ``test_prune_at_open.TestUninterruptedPrune``;
5. the version row's write, on a new store, under a new label, and its repair after an
   interrupted write: ``test_version_row_at_open``'s ``TestNewStore``, ``TestNewLabelInterrupted``
   and ``TestNewStoreInterrupted``. The version table has no ``timestamp`` column, and its rows
   are compared there by ``(serial, label)``, which is the whole row; that is asserted here.
   Those tests compare only the version rows, so each also ends here by comparing every
   replicated table of the last store it opened, whole rows, on every shard.

No Ray is initialised; every store is built in a temporary directory, and nothing under ``var/``
is opened.
"""

import unittest
from unittest import mock

import ray

from Datastore.tests import standin_pool as sp
from Datastore.tests import test_prune_at_open as prune_at_open
from Datastore.tests import test_reconcile_at_open as reconcile_at_open
from Datastore.tests import test_replicated_write as replicated_write
from Datastore.tests import test_version_row_at_open as version_row_at_open

# every table the check compares: the replicated classes and BackgroundModel's tag table
REPLICATED = reconcile_at_open.REPLICATED

# _TickingDatetime's now() starts here and counts up in seconds
SHARD_CLOCK_YEAR = "2030-"


def tearDownModule():
    if ray.is_initialized():
        raise AssertionError("Ray was initialised by test_one_timestamp_per_write")


class _TickingClock:
    """
    Every shard's clock ticks on every call, from before the test's store is built until it ends,
    and full-row identity also requires that no replicated row on any shard carries a tick.
    """

    def setUp(self):
        patcher = mock.patch.object(
            sp.ds_mod, "datetime", replicated_write._TickingDatetime
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()

    def assertNoShardClock(self, files):
        for table in REPLICATED:
            for sid, path in sorted(files.items()):
                if "timestamp" not in sp.table_columns(path, table):
                    continue
                for (stamp,) in sp._read(path, f'SELECT timestamp FROM "{table}"'):
                    self.assertFalse(
                        str(stamp).startswith(SHARD_CLOCK_YEAR),
                        f"{table}: shard {sid} holds a row stamped by its own clock ({stamp})",
                    )

    def assertIdentical(self):
        super().assertIdentical()
        self.assertNoShardClock(self.files)


# ------------------------------------------------------------------------------------------------
# 2. the repair at open
# ------------------------------------------------------------------------------------------------


class TestRepairAtOpen(_TickingClock, reconcile_at_open.TestKillAndReopen):
    pass


class TestRepairThenPruneAtOpen(
    _TickingClock, reconcile_at_open.TestPruningAfterRepair
):
    pass


# ------------------------------------------------------------------------------------------------
# 3. the completion of an interrupted prune, and 4. the prune at open
# ------------------------------------------------------------------------------------------------


class TestCompletionOfAnInterruptedPrune(
    _TickingClock, prune_at_open.TestInterruptedPrune
):
    pass


class TestPruneAtOpen(_TickingClock, prune_at_open.TestUninterruptedPrune):
    pass


# ------------------------------------------------------------------------------------------------
# 5. the version row
# ------------------------------------------------------------------------------------------------


class _VersionTickingClock(_TickingClock):
    """test_version_row_at_open compares the version rows only, so each test here ends by
    comparing every replicated table of the last store it opened, whole rows, on every shard.
    """

    def tearDown(self):
        files = getattr(self, "files", None)
        if files:
            for table in REPLICATED:
                rows = {}
                for sid, path in sorted(files.items()):
                    order = ", ".join(f'"{c}"' for c in sp.table_columns(path, table))
                    rows[sid] = sp._read(
                        path, f'SELECT * FROM "{table}" ORDER BY {order}'
                    )
                first = next(iter(rows.values()))
                for sid, shard_rows in rows.items():
                    self.assertEqual(first, shard_rows, f"{table}: shard {sid}")
            self.assertNoShardClock(files)
        super().tearDown()


class TestVersionRowOfANewStore(_VersionTickingClock, version_row_at_open.TestNewStore):
    def test_the_version_row_is_its_serial_and_label(self):
        """The version table's columns are serial and label: no timestamp, and the
        ``(serial, label)`` comparisons of test_version_row_at_open compare whole rows.
        """
        primary = self.new_primary()
        self.open(primary, "L1")
        self.close()
        for sid, path in self.files.items():
            self.assertEqual(
                ["serial", "label"], sp.table_columns(path, "version"), sid
            )


class TestVersionRowOfANewLabel(
    _VersionTickingClock, version_row_at_open.TestNewLabelInterrupted
):
    pass


class TestVersionRowOfAnInterruptedNewStore(
    _VersionTickingClock, version_row_at_open.TestNewStoreInterrupted
):
    pass


if __name__ == "__main__":
    unittest.main()
