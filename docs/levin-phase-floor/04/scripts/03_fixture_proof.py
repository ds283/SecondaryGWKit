"""
levin-phase-floor prompt 03, step 3 (a): prove the Ray-free route on a fixture store before it
touches A3 v2.

A store is built by ``StoreFixtures.real_store_fixtures.build_real_store`` in
``var/levin-phase-floor-03/state/fixture_proof_store/`` (kept, so that the review can re-hash it).
It is then opened with ``ShardedPool(read_only=True)`` on ``standin_pool``'s in-process actors
(scripts/harness.py), and the lookups a ``QuadSourceIntegral``'s inputs are found by are made
through the pool's own ``object_get`` / ``read_table`` with main.py's keyword shapes:

- ``tolerance`` (replicated), as main.py:4253-4264;
- ``wavenumber`` (replicated), as main.py's ``convert_to_wavenumbers``;
- ``store_tag`` (replicated), as main.py:1162-1190;
- ``wavenumber_exit_time`` (replicated), as main.py:1042-1051;
- ``TkNumericIntegration`` (sharded), a ``QuadSourceIntegral`` input (``Tq_numeric``), with the
  keyword set main.py:3895-3916 uses, values populated;
- ``read_table("redshift")``.

Pass conditions: the pool opens read-only; the ``TkNumericIntegration`` comes back available
with the fixture's serial and its three values; a write (``object_store``) is refused; a lookup
that misses a replicated row raises rather than inserting; and every file in the fixture's
directory is byte-identical (name list, SHA-256, size, mtime_ns) before and after.

Writes only its transcript (stdout) and the fixture directory.
"""

import hashlib
import shutil
import sys
from pathlib import Path

import ray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import close_pool, open_readonly_pool, standin_ray  # noqa: E402
import copy  # noqa: E402

from StoreFixtures.real_store_fixtures import (  # noqa: E402
    REPLICATED_ROWS,
    build_real_store,
    file_state,
)
from StoreFixtures.standin_pool import StandinCosmology, make_units  # noqa: E402


class _ModelProxy:
    """What the TkNumericIntegration factory reads of a model proxy: its serial and units."""

    def __init__(self, store_id, units):
        self.store_id = store_id
        self.units = units
        self.available = True


def main():
    print(
        f"# fixture_proof.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
    )
    print(
        f"# harness.py sha256 {hashlib.sha256((HERE / 'harness.py').read_bytes()).hexdigest()}"
    )
    directory = HERE.parent / "state" / "fixture_proof_store"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    # The default fixture's exit-time rows carry z_exit only; TkNumericIntegration's constructor
    # reads z_exit_suph_e3 (ComputeTargets/TkNumericIntegration.py:152). The efold columns are
    # supplied through build_real_store's own `replicated` argument, the fixture's documented way
    # of adding to its rows: suph_e(n) = z_exit (1 + n/2), subh_e(n) = z_exit / (1 + n/2), the
    # form standin_pool.make_exit_time uses. Nothing else differs from the default store.
    replicated = copy.deepcopy(REPLICATED_ROWS)
    for row in replicated["wavenumber_exit_time"]:
        for n in (1, 2, 3, 4, 5):
            row[f"z_exit_suph_e{n}"] = row["z_exit"] * (1.0 + 0.5 * n)
        for n in (1, 2, 3, 4, 5, 6):
            row[f"z_exit_subh_e{n}"] = row["z_exit"] / (1.0 + 0.5 * n)
    store = build_real_store(directory, replicated=replicated)
    print(f"built fixture store: primary {store.primary}")
    listing0, state0 = file_state(directory)
    for name in listing0:
        print(
            f"  before  {state0[name][0]}  {state0[name][1]:>8}  {state0[name][2]}  {name}"
        )

    failures = []
    with standin_ray() as cluster:
        pool, printed = open_readonly_pool(store.primary, cluster)
        print("pool constructor printed:", printed.strip())
        print("pool._read_only =", pool._read_only)
        try:
            units = make_units()
            atol, rtol = ray.get(
                [
                    pool.object_get("tolerance", tol=1.0e-5),
                    pool.object_get("tolerance", tol=1.0e-7),
                ]
            )
            print(
                f"tolerance: atol store_id={atol.store_id} tol={atol.tol}; rtol store_id={rtol.store_id} tol={rtol.tol}"
            )
            (k_wn,) = ray.get(
                pool.object_get(
                    "wavenumber",
                    payload_data=[
                        {
                            "k_inv_Mpc": 0.1,
                            "units": units,
                            "is_source": True,
                            "is_response": False,
                        }
                    ],
                )
            )
            print(f"wavenumber: store_id={k_wn.store_id} k_inv_Mpc={k_wn.k_inv_Mpc}")
            tags = ray.get(
                [
                    pool.object_get("store_tag", label="Run_fixture"),
                    pool.object_get("store_tag", label="grid-A"),
                ]
            )
            print(f"store_tag: {[(t.store_id, t.label) for t in tags]}")
            cosmology = StandinCosmology(units, type_id=0, store_id=1)
            k_exit = ray.get(
                pool.object_get(
                    "wavenumber_exit_time",
                    k=k_wn,
                    cosmology=cosmology,
                    atol=atol,
                    rtol=rtol,
                )
            )
            print(
                f"wavenumber_exit_time: store_id={k_exit.store_id} available={k_exit.available} z_exit={k_exit.z_exit}"
            )
            Tk = ray.get(
                pool.object_get(
                    "TkNumericIntegration",
                    model=_ModelProxy(1, units),
                    z_sample=None,
                    k=k_exit,
                    z_init=None,
                    atol=atol,
                    rtol=rtol,
                    tags=tags,
                )
            )
            values = Tk.values
            print(
                f"TkNumericIntegration: store_id={Tk.store_id} available={Tk.available} "
                f"values={[(v.z.store_id, v.T, v.Tprime) for v in values]}"
            )
            if not (Tk.available and Tk.store_id == 1 and len(values) == 3):
                failures.append(
                    "TkNumericIntegration lookup did not return the fixture row"
                )
            zs = ray.get(pool.read_table("redshift", is_source=True))
            print(
                f"read_table(redshift, is_source=True): {[(z.store_id, z.z) for z in zs]}"
            )

            # a write is refused
            try:
                pool.object_store(Tk)
                failures.append("object_store was not refused")
            except Exception as e:
                print(f"object_store refused: {type(e).__name__}: {e}")
            # a replicated miss raises rather than inserting
            try:
                ray.get(pool.object_get("tolerance", tol=1.0e-3))
                failures.append("a tolerance miss did not raise")
            except Exception as e:
                print(f"tolerance miss refused: {type(e).__name__}: {str(e)[:200]}")
        finally:
            print("pool close printed:", close_pool(pool, cluster).strip())
        print(
            f"actor calls made: {len(cluster.calls)}; methods {sorted({c['method'] for c in cluster.calls})}"
        )

    listing1, state1 = file_state(directory)
    for name in listing1:
        print(
            f"  after   {state1[name][0]}  {state1[name][1]:>8}  {state1[name][2]}  {name}"
        )
    if listing0 != listing1:
        failures.append(f"directory listing changed: {listing0} -> {listing1}")
    if state0 != state1:
        failures.append("a fixture file changed")
    print("ray.is_initialized() =", ray.is_initialized())
    if failures:
        print("FIXTURE PROOF FAILED:", failures)
        sys.exit(1)
    print(
        "FIXTURE PROOF PASSED: opens read-only, object_get returns a QuadSourceIntegral input, files byte-identical"
    )


if __name__ == "__main__":
    main()
