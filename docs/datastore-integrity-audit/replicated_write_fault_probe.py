"""
Probe: drive the real ShardedPool and the real Datastore actor code with no Ray, inject a fault
between the controlling shard's commit and the replicas', and look at what is left on disk and at
what a resumed pool does with it.

How it runs without Ray. ``ray.get`` is replaced by the identity. ``ShardedPool``'s references to
the ``Datastore`` and ``SerialPoolBroker`` actor classes are replaced by stand-ins whose
``.options(...).remote(...)`` builds the *undecorated* class (``__ray_metadata__.modified_class``)
in-process and wraps it so that ``handle.method.remote(...)`` calls the method at once. Arguments
and results are pickled through, as Ray would, so that no two "actors" share an object. The
controller choice (``random.randrange``) is pinned per call so a scenario is deterministic.

Everything is written to a temporary directory under this directory and deleted afterwards.

Run from the repository root:
    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_write_fault_probe.py
"""

import contextlib
import importlib
import io
import pickle
import sqlite3
import tempfile
from pathlib import Path

import ray

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------------
# the stand-ins
# --------------------------------------------------------------------------------------------

ray.get = lambda x, *args, **kwargs: x  # every "ref" below is already its value

sp_mod = importlib.import_module("datastorekit.SQL.ShardedPool")
ds_mod = importlib.import_module("datastorekit.SQL.Datastore")
broker_mod = importlib.import_module("datastorekit.SQL.SerialPoolBroker")

DatastoreClass = ds_mod.Datastore.__ray_metadata__.modified_class
BrokerClass = broker_mod.SerialPoolBroker.__ray_metadata__.modified_class


class ActorDied(Exception):
    """Stands in for a RayActorError, or for the driver being killed at this point."""


# faults[(actor_name, method, cls_name)] = True -> that call raises instead of running
FAULTS = {}


def _copy(x):
    return pickle.loads(pickle.dumps(x))


class _Method:
    def __init__(self, handle, name):
        self._handle = handle
        self._name = name

    def remote(self, *args, **kwargs):
        cls_name = None
        if len(args) > 0 and isinstance(args[0], str):
            cls_name = args[0]
        elif len(args) > 0:
            cls_name = type(args[0]).__name__
        if FAULTS.get((self._handle.name, self._name, cls_name)):
            raise ActorDied(f"{self._handle.name}.{self._name}({cls_name}) killed")
        fn = getattr(self._handle.obj, self._name)
        return _copy(fn(*_copy(args), **_copy(kwargs)))


class Handle:
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name

    def __getattr__(self, name):
        return _Method(self, name)


class _Options:
    def __init__(self, cls, name):
        self._cls = cls
        self._name = name

    def remote(self, *args, **kwargs):
        if self._cls is BrokerClass:
            return Handle(self._cls(*args, **kwargs), self._name)
        # the Datastore actor receives the broker handle and calls it as broker.method.remote
        return Handle(self._cls(*args, **kwargs), self._name)


class FakeActorClass:
    def __init__(self, cls):
        self._cls = cls

    def options(self, name=None, **_):
        return _Options(self._cls, name)


sp_mod.Datastore = FakeActorClass(DatastoreClass)
sp_mod.SerialPoolBroker = FakeActorClass(BrokerClass)

# pin the controller: CONTROLLER[0] is the index into the shard-id list randrange draws from
CONTROLLER = [0]
sp_mod.random.randrange = lambda n: CONTROLLER[0] % n

from config.sharding import (  # noqa: E402
    replicated_tables,
    sharded_tables,
    shard_key_type,
    shard_key_wavenumber_store_id,
)


def open_pool(primary: Path, shards: int = 3):
    with contextlib.redirect_stdout(io.StringIO()):
        return sp_mod.ShardedPool(
            version_label="probe",
            db_name=primary,
            ShardKeyType=shard_key_type,
            ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
            replicated_tables=replicated_tables,
            sharded_tables=sharded_tables,
            shards=shards,
        )


def close_pool(pool):
    with contextlib.redirect_stdout(io.StringIO()):
        pool.__exit__(None, None, None)


def per_shard(pool, sql):
    out = {}
    for sid, path in sorted(pool._shard_db_files.items()):
        c = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)
        try:
            out[sid] = c.execute(sql).fetchall()
        finally:
            c.close()
    return out


def controller_index_for(pool, shard_id):
    """The randrange value that makes ``shard_id`` the controller (ShardedPool swaps entry i to
    the end and pops it)."""
    return list(pool._shards.keys()).index(shard_id)


# --------------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------------


def main():
    from datastorekit.store_inventory import read_inventory

    with tempfile.TemporaryDirectory(dir=HERE, prefix="fault-") as tmp:
        primary = Path(tmp) / "store.sqlite"

        # 1. a clean replicated get
        pool = open_pool(primary)
        ids = list(pool._shards.keys())
        print("shard ids in pool order:", ids)
        CONTROLLER[0] = 0
        t = pool.object_get("tolerance", tol=1e-8)
        print("1. clean get: controller", ids[0], "returned serial", t.store_id)
        print(
            "   per shard:", per_shard(pool, "SELECT serial, log10_tol FROM tolerance")
        )

        # 2. the controller commits a new tolerance; the replica on shard ids[2] dies before its
        #    commit. The driver sees the exception, as it would a RayActorError
        CONTROLLER[0] = controller_index_for(pool, ids[0])
        FAULTS[(f"shard{ids[2]:04d}-store", "object_get", "tolerance")] = True
        try:
            pool.object_get("tolerance", tol=1e-6)
        except ActorDied as e:
            print("2. replica fault:", e)
        FAULTS.clear()
        print(
            "   per shard:", per_shard(pool, "SELECT serial, log10_tol FROM tolerance")
        )
        close_pool(pool)

        # 3. resume: a new pool on the same files. The same get, with the controller a shard that
        #    holds the row -> nothing replicates, the missing copy stays missing
        pool = open_pool(primary)
        CONTROLLER[0] = controller_index_for(pool, ids[0])
        t = pool.object_get("tolerance", tol=1e-6)
        print("3. resume, controller holds the row: serial", t.store_id)
        print(
            "   per shard:", per_shard(pool, "SELECT serial, log10_tol FROM tolerance")
        )

        # 4. the same get, with the controller the shard that lacks the row -> it inserts under a
        #    new serial and replicates that; the shards holding the old serial find the key and
        #    answer with their own serial, which ShardedPool discards
        CONTROLLER[0] = controller_index_for(pool, ids[2])
        t = pool.object_get("tolerance", tol=1e-6)
        print(
            "4. resume, controller lacks the row: driver now holds serial", t.store_id
        )
        print(
            "   per shard:", per_shard(pool, "SELECT serial, log10_tol FROM tolerance")
        )

        # 5. a wavenumber (the shard key): the same sequence, then the shard-key table
        FAULTS[(f"shard{ids[2]:04d}-store", "object_get", "wavenumber")] = True
        CONTROLLER[0] = controller_index_for(pool, ids[0])
        from Units import Mpc_units

        units = Mpc_units()
        payload = {
            "k_inv_Mpc": 0.5,
            "units": units,
            "is_source": True,
            "is_response": False,
        }
        try:
            pool.object_get("wavenumber", **payload)
        except ActorDied as e:
            print("5. replica fault:", e)
        FAULTS.clear()
        CONTROLLER[0] = controller_index_for(pool, ids[2])
        k = pool.object_get("wavenumber", **payload)
        print("   resume with lacking controller: driver holds k serial", k.store_id)
        print(
            "   per shard:", per_shard(pool, "SELECT serial, k_inv_Mpc FROM wavenumber")
        )
        print("   primary shard_keys:", dict(pool._shard_keys))
        close_pool(pool)

        # 6. what the structured inventory says about this store
        inv = read_inventory(primary)
        print("6. inventory problems:", inv.problems if inv.problems else "(none)")


if __name__ == "__main__":
    main()
