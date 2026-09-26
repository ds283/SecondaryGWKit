"""
Probes for the inequality keys:
  [02-exit-time-lookup-runs-inside-the-subhorizon-loop]
  [02-wavenumber-exit-time-tolerance-is-an-inequality-key]
  [03-integrationsolver-stepping-minimum-lookup]

Calls the real ``sqla_wavenumber_exit_time_factory.build`` and ``sqla_IntegrationSolver_factory.build``
against an in-memory store.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.exc import MultipleResultsFound  # noqa: E402

from common import RecordingConn, StandInCosmology, Store, UNITS  # noqa: E402
from CosmologyConcepts import wavenumber  # noqa: E402
from CosmologyConcepts.wavenumber import (  # noqa: E402
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
)
from Datastore.SQL.ObjectFactories.integration_metadata import (  # noqa: E402
    sqla_IntegrationSolver_factory,
)
from Datastore.SQL.ObjectFactories.wavenumber import (  # noqa: E402
    sqla_wavenumber_exit_time_factory,
)
from MetadataConcepts import tolerance  # noqa: E402

TOL = {-12.0: 1, -10.0: 2, -9.0: 3, -8.0: 4}


def tol(log10):
    return tolerance(store_id=TOL[log10], log10_tol=log10)


def exit_row(s, serial, atol, rtol, stepping=0, z_exit=1.0e4):
    row = dict(
        serial=serial,
        stepping=stepping,
        wavenumber_serial=1,
        cosmology_type=0,
        cosmology_serial=1,
        atol_serial=TOL[atol],
        rtol_serial=TOL[rtol],
        compute_time=0.1,
        z_exit=z_exit,
    )
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        row[f"z_exit_suph_e{o}"] = 1.0
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        row[f"z_exit_subh_e{o}"] = 1.0
    s.insert("wavenumber_exit_time", **row)


def fresh(rows):
    s = Store()
    s.insert("version", serial=1, label="probe")
    for log10, serial in TOL.items():
        s.insert("tolerance", serial=serial, log10_tol=log10)
    s.insert("wavenumber", serial=1, k_inv_Mpc=1.0e5, source=True, response=True)
    for r in rows:
        exit_row(s, *r)
    s.conn.commit()
    return s


def lookup(s, atol, rtol, stepping=None):
    conn = RecordingConn(s)
    payload = dict(
        k=wavenumber(store_id=1, k_inv_Mpc=1.0e5, units=UNITS),
        cosmology=StandInCosmology(),
        atol=tol(atol),
        rtol=tol(rtol),
    )
    if stepping is not None:
        payload["stepping"] = stepping
    try:
        obj = sqla_wavenumber_exit_time_factory.build(
            payload, conn, s.tables["wavenumber_exit_time"], None, s.tables, None
        )
        out = (
            f"store_id={obj.store_id if obj.available else None} available={obj.available} "
            f"obj.atol.log10={obj._atol.log10_tol} obj.rtol.log10={obj._rtol.log10_tol}"
            + (
                f" stepping={obj._stepping} z_exit={obj._z_exit}"
                if obj.available
                else ""
            )
        )
    except MultipleResultsFound:
        out = "MultipleResultsFound raised"
    return out, conn


def main():
    print("=" * 90)
    print("[02-exit-time-lookup-runs-inside-the-subhorizon-loop]")
    print("=" * 90)
    s = fresh([(1, -10.0, -9.0)])
    out, conn = lookup(s, -10.0, -9.0)
    print(f"one stored row, exact request -> {out}")
    print(f"statements executed by one build(): {len(conn.statements)}")
    for i, sql in enumerate(conn.statements):
        n_subh = sum(
            f"z_exit_subh_e{o}" in sql for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS
        )
        print(f"   statement {i + 1}: selects {n_subh} of 6 z_exit_subh_e* columns")

    print("\n" + "=" * 90)
    print(
        "[02-wavenumber-exit-time-tolerance-is-an-inequality-key]  (stored row: atol 1e-10, rtol 1e-9)"
    )
    print("=" * 90)
    for req in [(-10.0, -9.0), (-10.0, -8.0), (-8.0, -8.0), (-12.0, -9.0)]:
        out, _ = lookup(s, *req)
        print(f"request log10 (atol, rtol)={req}: {out}")

    print(
        "\nTwo stored rows at the same k: (1e-10, 1e-9) serial 1 and (1e-12, 1e-9) serial 2"
    )
    s2 = fresh([(1, -10.0, -9.0, 0, 1.0e4), (2, -12.0, -9.0, 0, 1.00001e4)])
    for req in [(-10.0, -9.0), (-12.0, -9.0), (-8.0, -8.0)]:
        out, _ = lookup(s2, *req)
        print(f"request log10 (atol, rtol)={req}: {out}")

    print("\nStepping (stored row stepping=1):")
    s3 = fresh([(1, -10.0, -9.0, 1)])
    for st in (0, 1, 2):
        out, _ = lookup(s3, -10.0, -9.0, stepping=st)
        print(f"request stepping={st}: {out}")

    print("\n" + "=" * 90)
    print("[03-integrationsolver-stepping-minimum-lookup]")
    print("=" * 90)
    for stored in (
        [(1, "cumulative-GL", 4)],
        [(1, "cumulative-GL", 8), (2, "cumulative-GL", 4)],
    ):
        s4 = Store()
        for serial, label, stepping in stored:
            s4.insert(
                "IntegrationSolver", serial=serial, label=label, stepping=stepping
            )
        s4.conn.commit()
        print(f"stored {[(sr, st) for sr, _, st in stored]}  (serial, stepping)")
        for req in (4, 6, 8):
            n_before = s4.conn.execute(
                s4.tables["IntegrationSolver"].select()
            ).fetchall()
            obj = sqla_IntegrationSolver_factory.build(
                {"label": "cumulative-GL", "stepping": req},
                s4.conn,
                s4.tables["IntegrationSolver"],
                s4.inserters["IntegrationSolver"],
                s4.tables,
                s4.inserters,
            )
            n_after = s4.conn.execute(
                s4.tables["IntegrationSolver"].select()
            ).fetchall()
            actual = {r.serial: r.stepping for r in n_after}.get(obj.store_id)
            print(
                f"   request stepping={req}: served serial={obj.store_id} "
                f"(row's stepping={actual}), object reports stepping={obj.stepping}, "
                f"new row inserted={len(n_after) > len(n_before)}"
            )


if __name__ == "__main__":
    main()
