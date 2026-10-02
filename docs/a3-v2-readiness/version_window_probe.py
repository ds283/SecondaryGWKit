"""
V1 (a3-v2-readiness audit 00): where a crash in ``ShardedPool.__init__`` leaves the replicated
``version`` row, and what the next open does with it.

No Ray, nothing under ``var/``: a real ``ShardedPool`` and the real ``Datastore`` actor code on
stand-in shards (``Datastore/tests/standin_pool.py``), in a temporary directory deleted at the end.
A crash is simulated by making a ``Datastore`` constructor raise at one of two points, for every
actor after the first (the first actor is the one built without ``version_serial``):

- ``start``: before the constructor does anything, so that the actor's file is never created;
- ``version``: inside the constructor's ``object_get("version", serial=...)``, after its file and
  every table exist, before the version row is inserted.

Three cases for the version row: a new store's first open crashing at ``start`` and at ``version``, and an existing
store opened under a label it has never seen, crashing at ``version`` (log 02's observation 1, which
was demonstrated by a scratch script that was not committed). After each crash the probe lists the
store's files and every shard's ``version`` rows (read ``mode=ro``), then opens the store again
with no fault and reports what the constructor does.

A fourth case measures the same window for the other replicated write a constructor makes: the
prune at open (``prune_unvalidated=True``, main.py's default) of an unvalidated ``BackgroundModel``,
a replicated class, which each actor deletes from its own shard in ``_validate_on_startup``.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/version_window_probe.py \
        > docs/a3-v2-readiness/version_window_probe.txt 2>&1
"""

import contextlib
import io
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import ray

from Datastore.tests import standin_pool as sp
from config.sharding import (
    replicated_tables,
    sharded_tables,
    shard_key_type,
    shard_key_wavenumber_store_id,
)


class SimulatedCrash(Exception):
    pass


_original_init = sp.DatastoreClass.__init__
_original_get = sp.DatastoreClass.object_get


@contextlib.contextmanager
def crash_later_actors(point):
    def init(self, *args, **kwargs):
        if point == "start" and kwargs.get("version_serial") is not None:
            raise SimulatedCrash(
                f"constructor of {kwargs.get('my_name')} killed at start"
            )
        return _original_init(self, *args, **kwargs)

    def object_get(self, ObjectClass, *args, **kwargs):
        if point == "version" and ObjectClass == "version" and "serial" in kwargs:
            raise SimulatedCrash(
                f"constructor of {self._my_name} killed before inserting the version row"
            )
        return _original_get(self, ObjectClass, *args, **kwargs)

    with mock.patch.object(sp.DatastoreClass, "__init__", init), mock.patch.object(
        sp.DatastoreClass, "object_get", object_get
    ):
        yield


def open_pool(primary, label, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pool = sp.sp_mod.ShardedPool(
            **kwargs,
            version_label=label,
            db_name=primary,
            ShardKeyType=shard_key_type,
            ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
            replicated_tables=replicated_tables,
            sharded_tables=sharded_tables,
            shards=3,
        )
    return pool, out.getvalue()


def close(pool):
    with contextlib.redirect_stdout(io.StringIO()):
        pool.__exit__(None, None, None)


def describe(directory: Path):
    names = sorted(p.name for p in directory.iterdir())
    print(f"  files: {names}")
    for p in sorted(directory.glob("store-shard*.sqlite")):
        try:
            rows = sp._read(p, "SELECT serial, label FROM version ORDER BY serial")
        except Exception as e:
            rows = f"unreadable: {type(e).__name__}: {e}"
        print(f"  {p.name}: version rows {rows}")


def describe_models(directory: Path):
    for p in sorted(directory.glob("store-shard*.sqlite")):
        rows = sp._read(
            p, "SELECT serial, validated FROM BackgroundModel ORDER BY serial"
        )
        print(f"  {p.name}: BackgroundModel (serial, validated) {rows}")


def reopen(primary, label):
    try:
        pool, printed = open_pool(primary, label)
        print(
            f"  next open under {label!r}: opened; version serial {pool._version.store_id}"
        )
        close(pool)
        describe(primary.parent)
    except Exception as e:
        print(f"  next open under {label!r}: raised {type(e).__name__}: {e}")


def main():
    t0 = time.monotonic()
    print("# version_window_probe.py (a3-v2-readiness audit 00, V1)")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cluster = sp.StandinCluster()
        with cluster.active():
            for n, (title, point, existing) in enumerate(
                [
                    (
                        "a new store's first open, later actors killed at start",
                        "start",
                        False,
                    ),
                    (
                        "a new store's first open, later actors killed before the version insert",
                        "version",
                        False,
                    ),
                    (
                        "an existing store (label L1) opened under L2, later actors killed before "
                        "the version insert",
                        "version",
                        True,
                    ),
                ],
                start=1,
            ):
                print(f"\n## case {n}: {title}")
                primary = root / f"case{n}" / "store.sqlite"
                label = "L1"
                if existing:
                    pool, _ = open_pool(primary, "L1")
                    close(pool)
                    print("  store created and closed under L1")
                    describe(primary.parent)
                    label = "L2"
                with crash_later_actors(point):
                    try:
                        pool, _ = open_pool(primary, label)
                        print(f"  open under {label!r} returned (no crash happened)")
                        close(pool)
                    except SimulatedCrash as e:
                        print(f"  open under {label!r}: crashed: {e}")
                describe(primary.parent)
                reopen(primary, label)

            # ---- case 4: the same window for the prune of a replicated row at open
            print(
                "\n## case 4: an existing store holding an unvalidated (replicated) BackgroundModel, "
                "opened with prune_unvalidated=True, later actors killed at start"
            )
            primary = root / "case4" / "store.sqlite"
            pool, _ = open_pool(primary, "L1")
            cluster.pool = pool
            units = sp.make_units()
            with contextlib.redirect_stdout(io.StringIO()):
                tag = ray.get(pool.object_get("store_tag", label="Run_probe"))
                model = ray.get(
                    pool.object_store(
                        sp.make_background_model(
                            pool, sp.StandinCosmology(units), [tag]
                        )
                    )
                )
            close(pool)
            print(
                f"  stored BackgroundModel serial {model.store_id}, not validated; store closed"
            )
            describe_models(primary.parent)
            with crash_later_actors("start"):
                try:
                    pool, _ = open_pool(primary, "L1", prune_unvalidated=True)
                    print("  open returned (no crash happened)")
                    close(pool)
                except SimulatedCrash as e:
                    print(f"  open with prune_unvalidated=True: crashed: {e}")
            describe_models(primary.parent)
            try:
                pool, _ = open_pool(primary, "L1", prune_unvalidated=True)
                print("  next open with prune_unvalidated=True: opened")
                close(pool)
                describe_models(primary.parent)
            except Exception as e:
                print(
                    f"  next open with prune_unvalidated=True: raised {type(e).__name__}: {e}"
                )
        print(f"\nray.is_initialized() = {ray.is_initialized()}")
    print(f"wall time {time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
