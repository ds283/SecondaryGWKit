"""
A real ``ShardedPool`` on stand-in shards: no Ray, nothing under ``var/``.

The pattern is ``docs/datastore-integrity-audit/replicated_write_fault_probe.py`` (campaign
``prompts/datastore-integrity`` README §5 rule 9), made reusable so that later prompts import it
rather than copy it. What is real: ``ShardedPool``, the ``Datastore`` actor code, every factory,
the ``SerialPoolBroker`` and SQLite on disk. What is stood in:

- the actor classes. ``ShardedPool``'s references to ``Datastore`` and ``SerialPoolBroker`` are
  replaced by stand-ins whose ``.options(...).remote(...)`` builds the *undecorated* class
  (``__ray_metadata__.modified_class``) in-process. ``handle.method.remote(...)`` runs the method
  at once, with its arguments and its result pickled through, as Ray would, so that no two
  "actors" share an object;
- ``ray.get``. ``.remote()`` returns a ``StandinRef`` holding the pickled result, or the exception
  the call raised. ``ray.get`` of a list reads every reference and raises the first error, so an
  exception reaches the driver only after every call it submitted has run, as with Ray;
- the controlling shard. ``ShardedPool`` draws it with ``random.randrange``; ``controller`` pins
  it by shard id, or leaves the draw random when ``None``.

Faults are injected per shard, method and class: ``"before"`` raises ``StandinActorDied`` without
running the call (an actor that died before committing), ``"after"`` runs the call, commits, and
then raises (an actor, or the driver, that died after the commit). Hooks see every actor call
before and after it runs.

Every patch is scoped by ``StandinCluster.active()`` with ``unittest.mock.patch``. Stores are built
in directories the caller supplies, which must be temporary.

This module is not a test module (no ``test_`` prefix); test modules import it.
"""

import contextlib
import importlib
import io
import pickle
import random as _random
import sqlite3
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from unittest import mock

import ray

sp_mod = importlib.import_module("Datastore.SQL.ShardedPool")
ds_mod = importlib.import_module("Datastore.SQL.Datastore")
broker_mod = importlib.import_module("Datastore.SQL.SerialPoolBroker")

DatastoreClass = ds_mod.Datastore.__ray_metadata__.modified_class
BrokerClass = broker_mod.SerialPoolBroker.__ray_metadata__.modified_class


class StandinActorDied(Exception):
    """Stands in for a RayActorError, or for the driver being killed at this point."""


def _copy(x):
    return pickle.loads(pickle.dumps(x))


class StandinRef:
    """What ``.remote()`` returns: the call's pickled result, or the exception it raised."""

    def __init__(self, result=None, error: Optional[BaseException] = None):
        self._result = pickle.dumps(result) if error is None else None
        self._error = error
        if error is not None:
            # an exception reaches the driver pickled, as under Ray, when it can be
            try:
                self._error = pickle.loads(pickle.dumps(error))
            except Exception:
                pass

    def get(self):
        if self._error is not None:
            raise self._error
        # a fresh copy on every get, as Ray deserializes one
        return pickle.loads(self._result)


def standin_get(refs, *args, **kwargs):
    """``ray.get``: a list is read in order and raises its first error; anything else that is not
    a ``StandinRef`` is returned as it is."""
    if isinstance(refs, (list, tuple)):
        return [standin_get(r) for r in refs]
    if isinstance(refs, StandinRef):
        return refs.get()
    return refs


def _class_name(args) -> Optional[str]:
    if len(args) == 0:
        return None
    if isinstance(args[0], str):
        return args[0]
    return type(args[0]).__name__


class _Method:
    def __init__(self, handle: "Handle", name: str):
        self._handle = handle
        self._name = name

    def remote(self, *args, **kwargs):
        handle = self._handle
        cluster = handle.cluster
        cls_name = _class_name(args)
        call = (handle.shard_id, self._name, cls_name)

        cluster.calls.append(
            {
                "shard": handle.shard_id,
                "method": self._name,
                "class": cls_name,
                "kwargs": sorted(kwargs.keys()),
                "insert_timestamp": kwargs.get("insert_timestamp"),
            }
        )

        mode = cluster.faults.get(call)
        if mode == "before":
            return StandinRef(
                error=StandinActorDied(
                    f"{handle.name}.{self._name}({cls_name}) killed before it ran"
                )
            )

        for hook in list(cluster.hooks):
            hook(handle.shard_id, self._name, cls_name, "before")

        fn = getattr(handle.obj, self._name)
        try:
            result = fn(*_copy(args), **_copy(kwargs))
        except Exception as e:
            return StandinRef(error=e)

        for hook in list(cluster.hooks):
            hook(handle.shard_id, self._name, cls_name, "after")

        if mode == "after":
            return StandinRef(
                error=StandinActorDied(
                    f"{handle.name}.{self._name}({cls_name}) killed after it ran"
                )
            )

        return StandinRef(result=result)


class Handle:
    """A stand-in actor handle: ``handle.method.remote(...)`` calls ``obj.method`` at once."""

    def __init__(
        self, obj, name: str, cluster: "StandinCluster", shard_id: Optional[int]
    ):
        self.obj = obj
        self.name = name
        self.cluster = cluster
        self.shard_id = shard_id

    def __getattr__(self, name):
        return _Method(self, name)


class _Options:
    def __init__(self, cls, name, cluster):
        self._cls = cls
        self._name = name
        self._cluster = cluster

    def remote(self, *args, **kwargs):
        shard_id = None
        if self._name is not None and self._name.startswith("shard"):
            shard_id = int(self._name[len("shard") : len("shard") + 4])
        # constructor arguments are passed as they are: the Datastore actor receives the broker's
        # handle, which must stay the one broker
        return Handle(self._cls(*args, **kwargs), self._name, self._cluster, shard_id)


class _StandinActorClass:
    def __init__(self, cls, cluster):
        self._cls = cls
        self._cluster = cluster

    def options(self, name=None, **_):
        return _Options(self._cls, name, self._cluster)


class _StandinRandom:
    """ShardedPool's ``random``: ``randrange`` returns the index of the pinned controller."""

    def __init__(self, cluster: "StandinCluster"):
        self._cluster = cluster

    def randrange(self, n):
        cluster = self._cluster
        # added by a3-v2-readiness prompt 02: inside StandinCluster.pin_controller, the draw is
        # resolved against the pool making the write, which may still be under construction
        writing = getattr(cluster, "_writing_pool", None)
        if (
            getattr(cluster, "open_controller", None) is not None
            and writing is not None
        ):
            return list(writing._shards.keys()).index(cluster.open_controller)
        if cluster.controller is None or cluster.pool is None:
            return _random.randrange(n)
        # ShardedPool swaps entry i of list(self._shards.keys()) to the end and pops it
        return list(cluster.pool._shards.keys()).index(cluster.controller)


class StandinCluster:
    """
    The stand-in actors of one or more pools, their faults, hooks and call log.

    Use ``with cluster.active():`` around everything that touches a pool.
    """

    def __init__(self):
        # (shard_id, method, class name) -> "before" | "after"
        self.faults: Dict[Tuple[Optional[int], str, Optional[str]], str] = {}
        # hook(shard_id, method, class name, "before" | "after")
        self.hooks: List[Callable] = []
        self.calls: List[dict] = []
        # the shard id to make the controlling shard, or None for a random draw
        self.controller: Optional[int] = None
        self.pool = None
        # added by a3-v2-readiness prompt 02 (pin_controller): the shard id pinned inside the
        # block, and the pool whose replicated write is running
        self.open_controller: Optional[int] = None
        self._writing_pool = None

    @contextlib.contextmanager
    def active(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ray, "get", standin_get))
            stack.enter_context(
                mock.patch.object(
                    sp_mod, "Datastore", _StandinActorClass(DatastoreClass, self)
                )
            )
            stack.enter_context(
                mock.patch.object(
                    sp_mod, "SerialPoolBroker", _StandinActorClass(BrokerClass, self)
                )
            )
            stack.enter_context(
                mock.patch.object(sp_mod, "random", _StandinRandom(self))
            )
            yield self

    @contextlib.contextmanager
    def pin_controller(self, shard_id: int):
        """
        Added by a3-v2-readiness prompt 02. Inside the block, the controlling shard of every
        replicated write is ``shard_id``, resolved against the pool making the write rather than
        against ``self.pool``. A write made inside ``ShardedPool``'s constructor (the version
        row, since that prompt) has no ``self.pool`` to resolve against: ``open_pool`` sets it
        only once the constructor returns. ``controller`` and its behaviour are unchanged.
        """
        original = sp_mod.ShardedPool._replicated_write
        cluster = self

        def wrapped(pool, *args, **kwargs):
            previous = getattr(cluster, "_writing_pool", None)
            cluster._writing_pool = pool
            try:
                return original(pool, *args, **kwargs)
            finally:
                cluster._writing_pool = previous

        previous_pin = getattr(self, "open_controller", None)
        self.open_controller = shard_id
        try:
            with mock.patch.object(sp_mod.ShardedPool, "_replicated_write", wrapped):
                yield self
        finally:
            self.open_controller = previous_pin

    def fault(self, shard_id: int, method: str, cls_name: str, when: str = "before"):
        if when not in ("before", "after"):
            raise ValueError(f'unknown fault mode "{when}"')
        self.faults[(shard_id, method, cls_name)] = when

    def clear_faults(self):
        self.faults.clear()

    def open_pool(self, primary: Path, shards: int = 3, **kwargs):
        """Open (or create) a pool on ``primary`` with the production table lists."""
        from config.sharding import (
            replicated_tables,
            sharded_tables,
            shard_key_type,
            shard_key_wavenumber_store_id,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            pool = sp_mod.ShardedPool(
                version_label="standin",
                db_name=primary,
                ShardKeyType=shard_key_type,
                ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
                replicated_tables=replicated_tables,
                sharded_tables=sharded_tables,
                shards=shards,
                **kwargs,
            )
        self.pool = pool
        return pool

    def open_pool_output(self, primary: Path, shards: int = 3, **kwargs):
        """
        As ``open_pool``, but return ``(pool, printed)``: what the constructor printed, which
        ``open_pool`` discards (prompt 02: the check at open prints what it repaired). If the
        constructor raises, what it printed before raising is kept as ``self.last_output``.
        """
        from config.sharding import (
            replicated_tables,
            sharded_tables,
            shard_key_type,
            shard_key_wavenumber_store_id,
        )

        out = io.StringIO()
        self.last_output = ""
        try:
            with contextlib.redirect_stdout(out):
                pool = sp_mod.ShardedPool(
                    version_label="standin",
                    db_name=primary,
                    ShardKeyType=shard_key_type,
                    ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
                    replicated_tables=replicated_tables,
                    sharded_tables=sharded_tables,
                    shards=shards,
                    **kwargs,
                )
        finally:
            self.last_output = out.getvalue()
        self.pool = pool
        return pool, out.getvalue()

    def close_pool(self, pool=None):
        pool = pool if pool is not None else self.pool
        with contextlib.redirect_stdout(io.StringIO()):
            pool.__exit__(None, None, None)
        if pool is self.pool:
            self.pool = None

    def replica_ids(self, pool=None) -> List[int]:
        """The shard ids other than the pinned controller, in the order they are submitted."""
        pool = pool if pool is not None else self.pool
        shard_ids = list(pool._shards.keys())
        i = shard_ids.index(self.controller)
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_ids.pop()
        return shard_ids


# ------------------------------------------------------------------------------------------------
# reading a store's files directly, mode=ro
# ------------------------------------------------------------------------------------------------


def _read(path: Path, sql: str, params=()) -> List[tuple]:
    conn = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def table_columns(path: Path, table: str) -> List[str]:
    return [row[1] for row in _read(path, f'PRAGMA table_info("{table}")')]


def shard_rows(pool, table: str) -> Dict[int, List[tuple]]:
    """Every column of every row of ``table`` on each shard, in a canonical order."""
    out = {}
    for sid, path in sorted(pool._shard_db_files.items()):
        columns = table_columns(path, table)
        order = ", ".join(f'"{c}"' for c in columns)
        out[sid] = _read(path, f'SELECT * FROM "{table}" ORDER BY {order}')
    return out


def shard_snapshot(pool, tables) -> Dict[str, Dict[int, List[tuple]]]:
    return {table: shard_rows(pool, table) for table in tables}


def store_checksums(primary: Path) -> Dict[str, str]:
    """
    The SHA-256 of every file in the directory of the primary ``primary``, by file name: the
    primary, its shards and any journal beside them (prompt 02: a check that writes nothing leaves
    every one unchanged, and creates no file). A stand-in store is the only thing in its
    directory; the caller makes sure of that.
    """
    import hashlib

    directory = Path(primary).parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def in_flight_records(pool) -> List[dict]:
    """The primary's ``replication_in_flight`` rows, as dicts."""
    path = pool.primary
    columns = table_columns(path, "replication_in_flight")
    return [
        dict(zip(columns, row))
        for row in _read(path, 'SELECT * FROM "replication_in_flight"')
    ]


# ------------------------------------------------------------------------------------------------
# objects for the two stored replicated classes, and one sharded class
# ------------------------------------------------------------------------------------------------


class StandinSerial:
    """Anything a factory reads only a ``store_id`` from."""

    def __init__(self, store_id: int):
        self.store_id = store_id
        self._my_id = store_id


class StandinCosmology:
    """What the exit-time and background factories read of a cosmology: its type, row and units.
    No foreign key names the cosmology, so any serial serves."""

    def __init__(self, units, type_id: int = 0, store_id: int = 1):
        self.units = units
        self.type_id = type_id
        self.store_id = store_id


def make_units():
    from Units import Mpc_units

    return Mpc_units()


def get_wavenumber(pool, k_inv_Mpc: float, units):
    return ray.get(
        pool.object_get(
            "wavenumber",
            k_inv_Mpc=k_inv_Mpc,
            units=units,
            is_source=True,
            is_response=False,
        )
    )


def get_tolerance(pool, tol: float):
    return ray.get(pool.object_get("tolerance", tol=tol))


def make_exit_time(k, cosmology, atol, rtol, z_exit: float = 1.0e6, stepping: int = 0):
    """A computed, unstored ``wavenumber_exit_time``, as ``compute()`` would leave it."""
    from CosmologyConcepts import wavenumber_exit_time
    from CosmologyConcepts.wavenumber import (
        WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
        WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
    )

    payload = {
        "store_id": None,
        "z_exit": z_exit,
        "compute_time": 0.25,
        "stepping": stepping,
    }
    for n in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        payload[f"z_exit_suph_e{n}"] = z_exit * (1.0 + 0.5 * n)
    for n in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        payload[f"z_exit_subh_e{n}"] = z_exit / (1.0 + 0.5 * n)
    return wavenumber_exit_time(
        payload=payload, k=k, cosmology=cosmology, atol=atol, rtol=rtol
    )


def get_redshifts(pool, zs):
    return ray.get(
        pool.object_get(
            "redshift",
            payload_data=[
                {"z": z, "is_source": True, "is_response": False} for z in zs
            ],
        )
    )


def make_background_model(pool, cosmology, tags, zs=(1.0e4, 1.0e2, 1.0), scale=1.0):
    """
    A computed, unstored ``BackgroundModel`` with ``tags`` and one value row per redshift in
    ``zs``, its redshifts and solver got through ``pool`` (both replicated). ``scale`` changes
    every value, for a model that differs.
    """
    from ComputeTargets import BackgroundModel, BackgroundModelValue
    from CosmologyConcepts import redshift_array
    from Quadrature.integration_metadata import IntegrationData

    units = cosmology.units
    GeV = units.GeV
    Mpc = units.Mpc
    z_objects = get_redshifts(pool, zs)
    solver = ray.get(
        pool.object_get("IntegrationSolver", label="standin-solver", stepping=0)
    )
    values = [
        BackgroundModelValue(
            store_id=None,
            z=z,
            Hubble=scale * (1.0 + z.z) * 1.0e-42 * GeV,
            wBackground=1.0 / 3.0,
            wPerturbations=0.3,
            rho=scale * 1.0e-40 * pow(GeV, 4.0),
            tau=scale * 1.0e3 / (1.0 + z.z) * Mpc,
            T_photon=scale * 2.0e-13 * (1.0 + z.z) * GeV,
            d_lnH_dz=0.5,
            d2_lnH_dz2=-0.25,
            d3_lnH_dz3=0.125,
            d_wPerturbations_dz=1.0e-3,
            d2_wPerturbations_dz2=-1.0e-4,
            tau_lo=1.0e-19 * Mpc,
            cs_tau=scale * 5.0e2 / (1.0 + z.z) * Mpc,
            cs_tau_lo=1.0e-20 * Mpc,
            friction_F=-float(i),
        )
        for i, z in enumerate(z_objects)
    ]
    return BackgroundModel(
        payload={
            "store_id": None,
            "data": IntegrationData(
                compute_time=1.5,
                compute_steps=10,
                RHS_evaluations=20,
                mean_RHS_time=0.01,
                max_RHS_time=0.02,
                min_RHS_time=0.005,
            ),
            "solver": solver,
            "values": values,
            "tau_gauss_order": 4,
            "cs_tau_gauss_order": 4,
            "friction_F_gauss_order": 4,
        },
        solver_labels={},
        cosmology=cosmology,
        z_sample=redshift_array(z_objects),
        label="standin-background",
        tags=tags,
    )


def make_policy_data(exit_time, source_serial: int = 1, policy_serial: int = 1):
    """A ``GkSourcePolicyData`` (a sharded class, sharded on its exit time's wavenumber), as its
    compute would leave it, over stand-in source and policy serials."""
    from ComputeTargets import GkSourcePolicyData

    return GkSourcePolicyData(
        payload={
            "store_id": None,
            "type": "numeric",
            "quality": "complete",
            "crossover_z": None,
            "Levin_z": None,
            "metadata": {},
        },
        source=StandinSerial(source_serial),
        policy=StandinSerial(policy_serial),
        k=exit_time,
    )
