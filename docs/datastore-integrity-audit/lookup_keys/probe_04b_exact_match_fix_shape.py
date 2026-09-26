"""
Check the proposed exact-match fix shapes for
  [02-wavenumber-exit-time-tolerance-is-an-inequality-key]   (user decision 2026-09-25: exact match)
  [02-exit-time-lookup-runs-inside-the-subhorizon-loop]
  [03-integrationsolver-stepping-minimum-lookup]
without editing any tracked file: the factory sources are patched **in memory** and executed into
throwaway namespaces, then run against the same cases as probe_04.

Exit time, the shape the other keyed types use (e.g. TkNumericIntegration.py:246-247):
    table.c.atol_serial == target_atol.store_id,
    table.c.rtol_serial == target_rtol.store_id,
    table.c.stepping == target_stepping,
  no ORDER BY, and the execute dedented out of the sub-horizon loop.
IntegrationSolver:
    table.c.label == label, table.c.stepping == stepping, and .one_or_none() instead of .scalar()
"""

import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import probe_04_exit_time_and_solver as p4  # noqa: E402
from common import RecordingConn, StandInCosmology, Store, UNITS  # noqa: E402
from CosmologyConcepts import wavenumber  # noqa: E402
from sqlalchemy.exc import MultipleResultsFound  # noqa: E402

FACT = ROOT / "Datastore" / "SQL" / "ObjectFactories"


def patch(path, edits, name):
    text = path.read_text()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    mod = types.ModuleType(name)
    exec(compile(text, f"{path} [patched in memory]", "exec"), mod.__dict__)
    return mod


EXIT_EDITS = [
    (
        """                        atol_table.c.log10_tol - target_atol.log10_tol
                        <= DEFAULT_FLOAT_PRECISION,
                        rtol_table.c.log10_tol - target_rtol.log10_tol
                        <= DEFAULT_FLOAT_PRECISION,
                        table.c.stepping >= target_stepping,
                    )
                )
                .order_by(atol_table.c.log10_tol.desc(), rtol_table.c.log10_tol.desc())
""",
        """                        table.c.atol_serial == target_atol.store_id,
                        table.c.rtol_serial == target_rtol.store_id,
                        table.c.stepping == target_stepping,
                    )
                )
""",
    ),
    (
        """                query = query.add_columns(table.c[f"z_exit_subh_e{z_offset}"])

                row_data = conn.execute(query).one_or_none()""",
        """                query = query.add_columns(table.c[f"z_exit_subh_e{z_offset}"])

            row_data = conn.execute(query).one_or_none()""",
    ),
]

SOLVER_EDITS = [
    (
        """        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.and_(table.c.label == label, table.c.stepping >= stepping)
            )
        ).scalar()""",
        """        row = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.and_(table.c.label == label, table.c.stepping == stepping)
            )
        ).one_or_none()
        store_id = None if row is None else row.serial""",
    )
]


def main():
    wn = patch(FACT / "wavenumber.py", EXIT_EDITS, "patched_wavenumber")
    im = patch(
        FACT / "integration_metadata.py", SOLVER_EDITS, "patched_integration_metadata"
    )

    def lookup(s, atol, rtol, stepping=None):
        conn = RecordingConn(s)
        payload = dict(
            k=wavenumber(store_id=1, k_inv_Mpc=1.0e5, units=UNITS),
            cosmology=StandInCosmology(),
            atol=p4.tol(atol),
            rtol=p4.tol(rtol),
        )
        if stepping is not None:
            payload["stepping"] = stepping
        try:
            obj = wn.sqla_wavenumber_exit_time_factory.build(
                payload, conn, s.tables["wavenumber_exit_time"], None, s.tables, None
            )
            out = f"available={obj.available}" + (
                f" store_id={obj.store_id} z_exit={obj._z_exit}"
                if obj.available
                else ""
            )
        except MultipleResultsFound:
            out = "MultipleResultsFound raised"
        return out, conn

    print("exit time, patched: one stored row (1e-10, 1e-9)")
    s = p4.fresh([(1, -10.0, -9.0)])
    out, conn = lookup(s, -10.0, -9.0)
    print(f"   exact request: {out}; statements executed={len(conn.statements)}")
    for req in [(-10.0, -8.0), (-8.0, -8.0), (-12.0, -9.0)]:
        print(f"   request {req}: {lookup(s, *req)[0]}")
    print(
        "exit time, patched: two stored rows (1e-10, 1e-9) serial 1 and (1e-12, 1e-9) serial 2"
    )
    s2 = p4.fresh([(1, -10.0, -9.0, 0, 1.0e4), (2, -12.0, -9.0, 0, 1.00001e4)])
    for req in [(-10.0, -9.0), (-12.0, -9.0), (-8.0, -8.0)]:
        print(f"   request {req}: {lookup(s2, *req)[0]}")
    print("exit time, patched: stored stepping=1")
    s3 = p4.fresh([(1, -10.0, -9.0, 1)])
    for st in (0, 1, 2):
        print(f"   request stepping={st}: {lookup(s3, -10.0, -9.0, stepping=st)[0]}")

    print(
        "IntegrationSolver, patched: stored (serial 1, stepping 8), (serial 2, stepping 4)"
    )
    s4 = Store()
    s4.insert("IntegrationSolver", serial=1, label="cumulative-GL", stepping=8)
    s4.insert("IntegrationSolver", serial=2, label="cumulative-GL", stepping=4)
    s4.conn.commit()
    for req in (4, 6, 8):
        obj = im.sqla_IntegrationSolver_factory.build(
            {"label": "cumulative-GL", "stepping": req},
            s4.conn,
            s4.tables["IntegrationSolver"],
            s4.inserters["IntegrationSolver"],
            s4.tables,
            s4.inserters,
        )
        print(
            f"   request {req}: served serial={obj.store_id} new_insert={getattr(obj, '_new_insert', False)}"
        )


if __name__ == "__main__":
    main()
