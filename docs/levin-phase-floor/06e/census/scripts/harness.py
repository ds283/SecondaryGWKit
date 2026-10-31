"""
levin-phase-floor prompt 06e section 5 (copied from prompt 06c's census scripts/harness.py, a record
that is not edited; its copy is kept in state/scripts_06c_reference/). Changed here, marked "prompt
06e", only in the ShardedPool's ``job_name``, which Part B's dispatch brief names
``levin-phase-floor-06e``. 06c's text follows.

levin-phase-floor prompt 06c section 5 (copied from prompt 06b's census scripts/harness.py, a record
that is not edited). Changed here, marked "prompt 06c", only in the ShardedPool's ``job_name``,
which Part B's dispatch brief names ``levin-phase-floor-06c``. 06b's text follows.

levin-phase-floor prompt 06b section 5 (copied from prompt 06a's scripts/harness.py, itself 06's
unchanged; a record, not edited). Changed here, marked "prompt 06b", only in the ShardedPool's
``job_name``, which Part B's dispatch brief names ``levin-phase-floor-06b``. datastorekit 0.5.0's
ShardedPool stores it (``self._job_name``) and reads it nowhere (out/cmd/11). 06's text follows.

levin-phase-floor prompt 06 section 6 (copied from prompt 05a's scripts/harness.py,
docs/levin-phase-floor/05a/wider/scripts/harness.py, itself 05's unchanged; a record, not edited).
Changed here, each change marked "prompt 06": this paragraph; and, for the
producers re-run in-process (prompt 06 section 6: "the re-run of the producers ... with the
``ray.remote`` stood in as 05's harness stood in ``QuadSource``'s"; rerun.py), four more stand-ins
inside ``standin_ray()``, each the same ``_StandinRemoteFunction`` as 05's two: the
``WKB_phase_function.remote`` that ``GkWKBIntegration.compute()`` and
``TkWKBIntegration.compute()`` dispatch with (patched at each producer module's attribute, which is
where each calls it through), ``assemble_GkSource_values.remote`` (``GkSource.compute()``) and
``apply_GkSource_policy.remote`` (``GkSourcePolicyData.compute()``). Each runs the function's own
body in-process on Ray-style copies of its arguments. In a tree whose producers are not re-run
they are never called.

(05's text follows, unchanged.)

levin-phase-floor prompt 05 section 6 (copied from prompt 04's scripts/harness.py,
docs/levin-phase-floor/04/scripts/harness.py, sha256 a2692689..., a record that is not edited).
Changed here in exactly four places (counting the ``QS_MOD`` import), each marked "prompt 05": this paragraph; the job_name below;
and two more stand-ins inside ``standin_ray()`` that recomputing ``QuadSource`` in-process needs
(prompt 05 section 6): ``ComputeTargets.QuadSource.compute_quad_source.remote``, which
``QuadSource.compute()`` dispatches its work with, run in-process exactly as
``compute_QuadSource_integral.remote`` is below; and ``ray.wait``, which ``QuadSource.store()``
calls (with ``timeout=0``) to ask whether that work has resolved. In-process every reference is
resolved when it is returned, so the stand-in returns every reference as ready. The Ray-free
harness shared by the gate L0 and the registered run.

(Prompt 04's text follows, unchanged.)

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
QS_MOD = importlib.import_module("ComputeTargets.QuadSource")  # prompt 05
# prompt 06: the producers' modules, for the re-run (rerun.py)
GKW_MOD = importlib.import_module("ComputeTargets.GkWKBIntegration")
TKW_MOD = importlib.import_module("ComputeTargets.TkWKBIntegration")
GKS_MOD = importlib.import_module("ComputeTargets.GkSource")
GPD_MOD = importlib.import_module("ComputeTargets.GkSourcePolicyData")

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


def standin_wait(refs, *args, **kwargs):
    """prompt 05: ``ray.wait`` in-process. Every reference here is resolved when it is returned
    (the work ran inside ``.remote``), so all are ready and none is pending."""
    refs = list(refs)
    for ref in refs:
        if not isinstance(ref, StandinRef):
            raise TypeError(f"harness: ray.wait given a {type(ref).__name__}")
    return refs, []


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
            ), mock.patch.object(  # prompt 05: QuadSource recomputed in-process
                QS_MOD,
                "compute_quad_source",
                _StandinRemoteFunction(QS_MOD.compute_quad_source),
            ), mock.patch.object(
                ray, "wait", standin_wait
            ), mock.patch.object(  # prompt 06: the producers re-run in-process
                GKW_MOD,
                "WKB_phase_function",
                _StandinRemoteFunction(GKW_MOD.WKB_phase_function),
            ), mock.patch.object(
                TKW_MOD,
                "WKB_phase_function",
                _StandinRemoteFunction(TKW_MOD.WKB_phase_function),
            ), mock.patch.object(
                GKS_MOD,
                "assemble_GkSource_values",
                _StandinRemoteFunction(GKS_MOD.assemble_GkSource_values),
            ), mock.patch.object(
                GPD_MOD,
                "apply_GkSource_policy",
                _StandinRemoteFunction(GPD_MOD.apply_GkSource_policy),
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
            job_name="levin-phase-floor-06e",  # prompt 06e (the dispatch brief); stored, never read
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
