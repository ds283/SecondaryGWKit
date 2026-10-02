"""
M1 and M2 (a3-v2-readiness audit 00): what a pool writes at open, and whether every lookup the
readers make can be served from files opened ``mode=ro``.

No Ray, nothing under ``var/``. Stores are built by ``reader_writes_probe.build_full_store`` (beside
this file) on stand-in shards (``Datastore/tests/standin_pool.py``) in a temporary directory, which
is deleted at the end. The instruments are ``reader_writes_probe``'s: every file's SQLite change
counter, every table's row count, and every file's SHA-256, before and after.

**Part M1 — what the open writes today.** A read-write ``ShardedPool`` opened on a copy of the full
store, clean and then in each state that makes its constructor write: ``prune_unvalidated=True``
over an unvalidated ``BackgroundModel``; ``drop_actions``; a primary without the
``replication_in_flight`` table; a shard missing a declared table; a version label the store has
not seen; a path where no store exists. Each first shows the known write, so that the clean open's
"no file changed" is a count.

**Part M2 — the same store read-only.** Every engine is opened with
``Datastore.store_reader.read_only_url`` (``mode=ro``): ``ShardedPool._create_engine`` and
``Datastore._create_engine`` are patched for the duration, and nothing else is changed.

- M2a: the pool as it is, read-only: does it open, and what does the first replicated lookup do?
- M2b: every replicated lookup sent to one shard's actor, with no in-flight record and no
  replication (what a read-only pool would do in place of ``_replicated_write``), and the whole of
  ``reader_writes_probe.qsi_sequence`` run through it on the full store;
- M2c: a lookup that misses, on that pool: what SQLite raises;
- M2d: journals. ``store_reader.open_read_only`` against an empty ``-journal`` beside a shard; the
  read-only pool against the same; and a *hot* journal, made by a child process that dies inside
  a write transaction after its page cache has spilled, read ``mode=ro`` and opened read-write;
- M2e: the version row found by label with a ``mode=ro`` select, without inserting it.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/readonly_open_probe.py \
        > docs/a3-v2-readiness/readonly_open_probe.txt 2>&1
"""

import contextlib
import importlib.util
import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest import mock

import ray
import sqlalchemy as sqla

from Datastore.store_reader import open_read_only, read_only_url
from Datastore.tests import standin_pool as sp
from config.sharding import read_table_config

_spec = importlib.util.spec_from_file_location(
    "reader_writes_probe", Path(__file__).with_name("reader_writes_probe.py")
)
rw = importlib.util.module_from_spec(_spec)
sys.modules["reader_writes_probe"] = rw
_spec.loader.exec_module(rw)


def show(label, before, after, extra=""):
    print(f"- {label}: {rw.describe_delta(rw.state_delta(before, after))}{extra}")


def exc_text(e: BaseException) -> str:
    chain = []
    while e is not None:
        chain.append(
            f"{type(e).__module__}.{type(e).__name__}: {str(e).splitlines()[0]}"
        )
        e = e.__cause__ or e.__context__
    return " <- ".join(chain)


def raw_state(directory: Path) -> dict:
    """SHA-256 and change counter of every file, with no SQL: a store with a hot journal cannot
    be read mode=ro, so the row counts of reader_writes_probe.file_state are not available.
    """
    import hashlib

    out = {}
    for p in sorted(directory.iterdir()):
        if p.is_file():
            entry = {"sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12]}
            if p.suffix == ".sqlite":
                entry["counter"] = rw.change_counter(p)
            out[p.name] = entry
    return out


def raw_delta(before: dict, after: dict) -> str:
    parts = []
    for name in sorted(set(before) | set(after)):
        if name not in before:
            parts.append(f"{name}: appeared")
        elif name not in after:
            parts.append(f"{name}: went")
        elif before[name] != after[name]:
            b, a = before[name], after[name]
            bits = ["changed"]
            if "counter" in b:
                bits.append(f"counter {b['counter']} -> {a['counter']}")
            parts.append(f"{name}: " + ", ".join(bits))
    return "; ".join(parts) if parts else "no file changed"


# ------------------------------------------------------------------------------------------------
# read-only engines
# ------------------------------------------------------------------------------------------------


def _ro_datastore_engine(self):
    connect_args = {}
    if self._timeout is not None:
        connect_args["timeout"] = self._timeout
    self._engine = sqla.create_engine(
        read_only_url(self._db_file), future=True, connect_args=connect_args
    )
    self._metadata = sqla.MetaData()
    self._inspector = sqla.inspect(self._engine)


_original_pool_engine = sp.sp_mod.ShardedPool._create_engine


def _ro_pool_engine(self):
    _original_pool_engine(self)
    self._engine.dispose()
    self._engine = sqla.create_engine(read_only_url(self._primary_file), future=True)


@contextlib.contextmanager
def read_only_engines():
    with mock.patch.object(
        sp.DatastoreClass, "_create_engine", _ro_datastore_engine
    ), mock.patch.object(sp.sp_mod.ShardedPool, "_create_engine", _ro_pool_engine):
        yield


def _direct_replicated_write(
    self, operation, cls_name, submit_controller, replication, store_id=None
):
    """In place of ShardedPool._replicated_write: one shard (the lowest id), no record, no
    replication, no timestamp override."""
    shard = self._shards[min(self._shards.keys())]
    ref = submit_controller(shard, None)
    return ref, ray.get(ref)


@contextlib.contextmanager
def direct_replicated_reads():
    with mock.patch.object(
        sp.sp_mod.ShardedPool, "_replicated_write", _direct_replicated_write
    ):
        yield


# ------------------------------------------------------------------------------------------------


HOT_JOURNAL_CHILD = r"""
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("PRAGMA cache_size=2")
conn.execute("BEGIN IMMEDIATE")
conn.execute("CREATE TABLE probe_spill (x BLOB)")
for i in range(400):
    conn.execute("INSERT INTO probe_spill VALUES (randomblob(4000))")
conn.execute("UPDATE tolerance SET log10_tol = log10_tol")
os._exit(0)
"""


def main():
    t0 = time.monotonic()
    print("# readonly_open_probe.py (a3-v2-readiness audit 00, M1 and M2)")
    print(f"python {sys.version.split()[0]}, sqlite {sqlite3.sqlite_version}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cluster = sp.StandinCluster()
        with cluster.active():
            full = root / "full"
            facts = rw.build_full_store(full / "store.sqlite", cluster)
            print(f"full store: {facts}")

            # ==================================================================================
            print("\n## M1 - what a read-write open writes today")

            def open_and_close(d, label="open, then close", **kw):
                before = rw.file_state(d)
                with rw.quiet():
                    pool = cluster.open_pool(
                        d / "store.sqlite", read_table_config=read_table_config, **kw
                    )
                    cluster.close_pool(pool)
                show(label, before, rw.file_state(d))

            d = root / "m1-clean"
            rw.copy_store(full, d)
            open_and_close(
                d, "clean full store, prune_unvalidated=False (the readers')"
            )

            d = root / "m1-prune"
            rw.copy_store(full, d)
            rw.delete_rows(d, ["UPDATE BackgroundModel SET validated = 0"])
            open_and_close(
                d,
                "an unvalidated BackgroundModel, prune_unvalidated=False",
                prune_unvalidated=False,
            )
            open_and_close(
                d,
                "the same store, prune_unvalidated=True (main.py's default)",
                prune_unvalidated=True,
            )

            d = root / "m1-drop"
            rw.copy_store(full, d)
            open_and_close(
                d,
                "drop_actions=['gk-source-policy']",
                drop_actions=["gk-source-policy"],
            )

            d = root / "m1-no-inflight-table"
            rw.copy_store(full, d)
            conn = sqlite3.connect(str(d / "store.sqlite"))
            conn.execute("DROP TABLE replication_in_flight")
            conn.commit()
            conn.close()
            open_and_close(d, "a primary with no replication_in_flight table")

            d = root / "m1-missing-table"
            rw.copy_store(full, d)
            conn = sqlite3.connect(str(d / "store-shard0001.sqlite"))
            conn.execute("DROP TABLE OneLoopIntegral_tags")
            conn.execute("DROP TABLE OneLoopIntegral")
            conn.commit()
            conn.close()
            open_and_close(d, "shard 1 missing the OneLoopIntegral tables")

            d = root / "m1-new-label"
            rw.copy_store(full, d)
            before = rw.file_state(d)
            with rw.quiet():
                from config.sharding import (
                    replicated_tables,
                    sharded_tables,
                    shard_key_type,
                    shard_key_wavenumber_store_id,
                )

                pool = sp.sp_mod.ShardedPool(
                    version_label="a-label-this-store-has-not-seen",
                    db_name=d / "store.sqlite",
                    ShardKeyType=shard_key_type,
                    ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
                    replicated_tables=replicated_tables,
                    sharded_tables=sharded_tables,
                    shards=3,
                )
                cluster.pool = pool
                cluster.close_pool(pool)
            show("a version label the store has not seen", before, rw.file_state(d))

            d = root / "m1-nonexistent"
            d.mkdir()
            before = rw.file_state(d)
            with rw.quiet():
                pool = cluster.open_pool(d / "store.sqlite")
                cluster.close_pool(pool)
            show("a path where no store exists", before, rw.file_state(d))

            # ==================================================================================
            print("\n## M2a - the pool as it is, every engine mode=ro")
            d = root / "m2a"
            rw.copy_store(full, d)
            before = rw.file_state(d)
            with read_only_engines():
                try:
                    with rw.quiet():
                        pool = cluster.open_pool(
                            d / "store.sqlite", read_table_config=read_table_config
                        )
                    print("- the constructor returned")
                except Exception as e:
                    pool = None
                    print(f"- the constructor raised: {exc_text(e)}")
                show("files after the open", before, rw.file_state(d))
                if pool is not None:
                    try:
                        ray.get(pool.object_get("tolerance", tol=1.0e-10))
                        print("- object_get tolerance (a hit) returned")
                    except Exception as e:
                        print(f"- object_get tolerance (a hit) raised: {exc_text(e)}")
                    try:
                        ray.get(pool.read_table("wavenumber", units=sp.make_units()))
                        print("- read_table wavenumber returned")
                    except Exception as e:
                        print(f"- read_table wavenumber raised: {exc_text(e)}")
                    with rw.quiet():
                        cluster.close_pool(pool)
            show("files after close", before, rw.file_state(d))

            # ==================================================================================
            print(
                "\n## M2b - mode=ro, replicated lookups sent to one shard's actor, no record, "
                "the reader's whole sequence"
            )
            d = root / "m2b"
            rw.copy_store(full, d)
            log = rw.ReplicatedWriteLog()
            rec = rw.Recorder(d, log)
            before = rw.file_state(d)
            with read_only_engines(), direct_replicated_reads():
                with rec.step("ShardedPool open (constructor), mode=ro"):
                    with rw.quiet():
                        pool = cluster.open_pool(
                            d / "store.sqlite", read_table_config=read_table_config
                        )
                try:
                    with rw.quiet():
                        rw.qsi_sequence(pool, rec, sp.make_units())
                except RuntimeError as e:
                    print(
                        f"  (the sequence stopped where the script would exit 1: {e})"
                    )

                # ------------------------------------------------------------------------------
                print("\n## M2c - a lookup that misses, on the same pool")
                try:
                    ray.get(pool.object_get("tolerance", tol=1.0e-99))
                    print("- object_get tolerance tol=1e-99 returned (it should not)")
                except Exception as e:
                    print(f"- object_get tolerance tol=1e-99 raised: {exc_text(e)}")
                try:
                    ray.get(
                        pool.object_get(
                            "GkSourcePolicy",
                            label="probe",
                            Levin_threshold=9.0,
                            numeric_policy="maximize-WKB",
                        )
                    )
                    print("- object_get GkSourcePolicy (9.0) returned (it should not)")
                except Exception as e:
                    print(f"- object_get GkSourcePolicy (9.0) raised: {exc_text(e)}")
                with rec.step("ShardedPool close (__exit__)"):
                    with rw.quiet():
                        cluster.close_pool(pool)
            rec.print("M2b - full store, mode=ro")
            show("M2b+M2c, every file, open to close", before, rw.file_state(d))

            # ==================================================================================
            print("\n## M2d - journals")
            d = root / "m2d-empty"
            rw.copy_store(full, d)
            journal = d / "store-shard0001.sqlite-journal"
            journal.write_bytes(b"")
            try:
                with open_read_only(d / "store.sqlite"):
                    print(
                        "- store_reader.open_read_only, empty journal beside shard 1: opened"
                    )
            except Exception as e:
                print(
                    f"- store_reader.open_read_only, empty journal beside shard 1: refused: {exc_text(e)}"
                )
            before = rw.file_state(d)
            with read_only_engines(), direct_replicated_reads():
                try:
                    with rw.quiet():
                        pool = cluster.open_pool(
                            d / "store.sqlite", read_table_config=read_table_config
                        )
                        ray.get(pool.object_get("tolerance", tol=1.0e-10))
                        cluster.close_pool(pool)
                    print(
                        "- the mode=ro pool, same empty journal: opened and served a lookup"
                    )
                except Exception as e:
                    print(
                        f"- the mode=ro pool, same empty journal: raised {exc_text(e)}"
                    )
            show("files", before, rw.file_state(d))

            d = root / "m2d-hot"
            rw.copy_store(full, d)
            shard1 = d / "store-shard0001.sqlite"
            child = subprocess.run(
                [sys.executable, "-c", HOT_JOURNAL_CHILD, str(shard1)],
                capture_output=True,
                text=True,
            )
            journal = d / "store-shard0001.sqlite-journal"
            print(
                f"- a child died inside a write transaction on shard 1 (exit {child.returncode}): "
                f"journal exists={journal.exists()}, size={journal.stat().st_size if journal.exists() else None}, "
                f"first bytes={journal.read_bytes()[:8].hex() if journal.exists() else None}"
            )
            for copy in ("m2d-hot-ro-pool", "m2d-hot-rw-pool"):
                rw.copy_store(d, root / copy)
            before = raw_state(d)
            eng = sqla.create_engine(read_only_url(shard1), future=True)
            try:
                with eng.connect() as conn:
                    n = conn.exec_driver_sql("SELECT count(*) FROM tolerance").scalar()
                print(
                    f"- a mode=ro read of shard 1 (SQLAlchemy) returned {n} tolerance rows"
                )
            except Exception as e:
                print(f"- a mode=ro read of shard 1 (SQLAlchemy) raised: {exc_text(e)}")
            finally:
                eng.dispose()
            try:
                c = sqlite3.connect(f"{shard1.as_uri()}?mode=ro", uri=True)
                c.execute("SELECT count(*) FROM tolerance").fetchone()
                c.close()
                print("- a mode=ro read of shard 1 (sqlite3): returned")
            except sqlite3.Error as e:
                print(
                    f"- a mode=ro read of shard 1 (sqlite3) raised: {type(e).__name__}: {e} "
                    f"(sqlite_errorcode={getattr(e, 'sqlite_errorcode', None)}, "
                    f"sqlite_errorname={getattr(e, 'sqlite_errorname', None)})"
                )
            try:
                with open_read_only(d / "store.sqlite"):
                    print("- store_reader.open_read_only: opened")
            except Exception as e:
                print(f"- store_reader.open_read_only: refused: {exc_text(e)}")
            print(
                f"- files after the read-only attempts: {raw_delta(before, raw_state(d))}"
            )

            d2 = root / "m2d-hot-ro-pool"
            before = raw_state(d2)
            with read_only_engines(), direct_replicated_reads():
                try:
                    with rw.quiet():
                        pool = cluster.open_pool(
                            d2 / "store.sqlite", read_table_config=read_table_config
                        )
                        record = pool.reconciliation
                        ray.get(pool.object_get("tolerance", tol=1.0e-10))
                        cluster.close_pool(pool)
                    print(
                        "- the M2b pool (engines mode=ro; the check at open's own connect left "
                        f"as it is): opened and served a lookup; hot_journals={record['hot_journals']}"
                    )
                except Exception as e:
                    print(f"- the M2b pool: raised {exc_text(e)}")
            print(f"- files after it: {raw_delta(before, raw_state(d2))}")

            d3 = root / "m2d-hot-rw-pool"
            before = raw_state(d3)
            try:
                with rw.quiet():
                    pool, printed = cluster.open_pool_output(
                        d3 / "store.sqlite", read_table_config=read_table_config
                    )
                    record = pool.reconciliation
                    cluster.close_pool(pool)
                print(
                    "- today's read-write open: opened; reconciliation "
                    f"hot_journals={record['hot_journals']}"
                )
            except Exception as e:
                print(f"- today's read-write open: raised {exc_text(e)}")
            print(f"- files after it: {raw_delta(before, raw_state(d3))}")

            # ==================================================================================
            print("\n## M2e - the version row, found by label mode=ro")
            for sid, path in enumerate(rw.shard_files(full)):
                for label in ("standin", "2025.1.1"):
                    rows = sp._read(
                        path,
                        "SELECT serial, label FROM version WHERE label = ?",
                        (label,),
                    )
                    print(f"- {path.name}: label {label!r}: {rows}")

        print(f"\nray.is_initialized() = {ray.is_initialized()}")
    print(f"wall time {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
