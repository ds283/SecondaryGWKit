"""
levin-phase-floor prompt 03: the Ray-free harness shared by the fixture proof, rebuild.py and the
registered run.

Route (a) of prompt step 3: a real ``datastorekit`` ``ShardedPool(read_only=True)`` whose actors
are stood in by ``StoreFixtures.standin_pool`` (in-process, every call's arguments and result
pickled through as Ray would).  Two further stand-ins complete the picture, and are the only
things this module adds to ``standin_pool``:

- ``ray.put``.  ``ModelProxy``, ``GkSourceProxy`` and ``BesselPhaseProxy`` call it in their
  constructors.  The stand-in serialises the object with ``ray.cloudpickle`` (what Ray's object
  store uses) and returns a ``StandinRef`` subclass, so that ``standin_pool``'s ``ray.get``
  stand-in deserialises a fresh copy on every get, as Ray does.
- ``compute_QuadSource_integral.remote``.  ``QuadSourceIntegral.compute()`` dispatches its work
  with it.  The stand-in runs the undecorated production function
  (``compute_QuadSource_integral._function``) in this process, on argument copies made with
  ``ray.cloudpickle`` as Ray would ship them, and returns its result as a ref.

- ``standin_pool``'s own serialisation.  Its ``StandinRef`` and ``_copy`` use the standard
  ``pickle``, which cannot serialise every object a lookup of A3 v2 returns (a ``GkSource``'s
  ``NumericData``, a class built inside a function); Ray uses ``ray.cloudpickle``.  Inside
  ``standin_ray()`` both names in ``StoreFixtures.standin_pool`` are patched to their cloudpickle
  equivalents, so that actor results and arguments travel as they would under Ray.

Nothing here writes to a store: the pool is read-only (``object_store``/``object_validate`` raise
``ReadOnlyWrite``; a miss that would insert raises ``ReadOnlyMiss``).  Nothing here changes
production code: every substitution is a ``unittest.mock.patch`` of a module attribute, scoped by
a context manager.
"""

import contextlib
import importlib
import io
from pathlib import Path
from unittest import mock

import ray
import ray.cloudpickle as cloudpickle

import StoreFixtures.standin_pool as standin_pool_mod
from StoreFixtures.standin_pool import StandinCluster, StandinRef

sp_mod = importlib.import_module("datastorekit.SQL.ShardedPool")
QSI_MOD = importlib.import_module("ComputeTargets.QuadSourceIntegral")

# main.py's VERSION_LABEL (main.py:322); the store's version row (A3 v2 holds only this label)
VERSION_LABEL = "2025.1.1"


class CloudpickleRef(StandinRef):
    """A ``StandinRef`` serialised with ``ray.cloudpickle``, as Ray's object store does."""

    def __init__(self, result=None, error=None):
        self._error = error
        self._result = cloudpickle.dumps(result) if error is None else None

    def get(self):
        if self._error is not None:
            raise self._error
        return cloudpickle.loads(self._result)


def standin_put(obj, *args, **kwargs):
    return CloudpickleRef(result=obj)


def _ray_copy(x):
    return cloudpickle.loads(cloudpickle.dumps(x))


class _StandinRemoteFunction:
    """``compute_QuadSource_integral`` with ``.remote`` run in-process on Ray-style copies."""

    def __init__(self, remote_function):
        self._function = remote_function._function

    def remote(self, *args, **kwargs):
        try:
            result = self._function(*_ray_copy(args), **_ray_copy(kwargs))
        except Exception as e:  # reaches the caller at ray.get, as under Ray
            return CloudpickleRef(error=e)
        return CloudpickleRef(result=result)


@contextlib.contextmanager
def standin_ray():
    """Everything that needs Ray, stood in: the pool's actors, ray.get, ray.put and the
    compute task.  Ray itself is never initialised."""
    if ray.is_initialized():
        raise RuntimeError("harness: Ray is initialised; route (a) must not use it")
    cluster = StandinCluster()
    with cluster.active(), mock.patch.object(
        standin_pool_mod, "StandinRef", CloudpickleRef
    ), mock.patch.object(standin_pool_mod, "_copy", _ray_copy):
        with mock.patch.object(ray, "put", standin_put):
            with mock.patch.object(
                QSI_MOD,
                "compute_QuadSource_integral",
                _StandinRemoteFunction(QSI_MOD.compute_QuadSource_integral),
            ):
                yield cluster
    if ray.is_initialized():
        raise RuntimeError("harness: Ray was initialised during the run")


def open_readonly_pool(primary: Path, cluster: StandinCluster, timeout: int = 60):
    """``ShardedPool(read_only=True)`` as extract_QuadSourceIntegral_data.py:1318-1329 builds
    it, on the stand-in actors.  Returns ``(pool, printed)``."""
    from config.datastore import factories
    from config.sharding import (
        read_table_config,
        replicated_tables,
        shard_key_type,
        shard_key_wavenumber_store_id,
        sharded_tables,
    )

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pool = sp_mod.ShardedPool(
            version_label=VERSION_LABEL,
            db_name=str(primary),
            ShardKeyType=shard_key_type,
            ShardKeyStoreIdGetter=shard_key_wavenumber_store_id,
            replicated_tables=replicated_tables,
            sharded_tables=sharded_tables,
            timeout=timeout,
            profile_agent=None,
            job_name="levin-phase-floor-03",
            read_only=True,
            read_table_config=read_table_config,
            factories=factories,
        )
    cluster.pool = pool
    return pool, out.getvalue()


def close_pool(pool, cluster: StandinCluster):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pool.__exit__(None, None, None)
    cluster.pool = None
    return out.getvalue()


# ---------------------------------------------------------------------------------------------
# pickling what the wrappers keep
# ---------------------------------------------------------------------------------------------
#
# A PhaseGroup's f_sin/f_cos close over build_phase_groups' local ``terms_at``, which is wrapped
# in functools.lru_cache; ray.cloudpickle cannot pickle an lru_cache wrapper of a local function.
# The reducer below pickles such a wrapper as the function it wraps (by value, as cloudpickle
# pickles any local function) plus its maxsize, and rebuilds an empty cache of the same size on
# load. The cache holds only values of a pure function of its argument, so an empty one computes
# exactly what a full one returns.

import functools  # noqa: E402


def _rebuild_lru_cache(wrapped, maxsize, typed):
    return functools.lru_cache(maxsize=maxsize, typed=typed)(wrapped)


class _GroupPickler(cloudpickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, functools._lru_cache_wrapper) and "<locals>" in getattr(
            obj, "__qualname__", ""
        ):
            params = obj.cache_parameters()
            return _rebuild_lru_cache, (
                obj.__wrapped__,
                params["maxsize"],
                params["typed"],
            )
        return super().reducer_override(obj)


def dump_record(record, path):
    """Pickle ``record`` to ``path`` with ray.cloudpickle and the lru_cache reducer."""
    with open(path, "wb") as f:
        _GroupPickler(f, protocol=5).dump(record)


def load_record(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)
