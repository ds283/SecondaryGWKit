"""
Probes for [00-oneloop-lookup-joins-the-wrong-tag-table] and two smaller lookup findings:

  * OneLoopIntegral.build with a tag: joins QuadSourceIntegral_tags (the wrong table). Shown both
    ways it goes wrong: a miss on a row that is tagged, and a false hit when a QuadSourceIntegral
    tag row happens to carry the same serial.
  * redshift.build at z = 0: the relative filter divides by z, SQLite returns NULL, the lookup
    never matches, and every call inserts a new row.
  * .scalar() lookups (tolerance, store_tag, version, IntegrationSolver, GkSourcePolicy,
    LambdaCDM, QCD_Cosmology) take the first row and never detect a duplicate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import Ref, RecordingConn, StandInModelProxy, Store  # noqa: E402
from ObjectFactories.OneLoopIntegral import (  # noqa: E402
    sqla_OneLoopIntegral_factory,
)
from ObjectFactories.redshift import sqla_redshift_factory  # noqa: E402
from ObjectFactories.tolerance import sqla_tolerance_factory  # noqa: E402
from MetadataConcepts import store_tag, tolerance  # noqa: E402


def oneloop_store(qsi_tag_rows, oli_tag_rows):
    s = Store()
    s.insert("version", serial=1, label="probe")
    s.insert("store_tag", serial=1, label="T")
    s.insert(
        "OneLoopIntegral",
        serial=5,
        label="oli",
        model_serial=1,
        wavenumber_exit_serial=11,
        atol_serial=1,
        rtol_serial=2,
        z_response_serial=1,
        value=1.25,
    )
    for parent in oli_tag_rows:
        s.insert("OneLoopIntegral_tags", parent_serial=parent, tag_serial=1)
    for parent in qsi_tag_rows:
        s.insert("QuadSourceIntegral_tags", parent_serial=parent, tag_serial=1)
    s.conn.commit()
    return s


def oneloop_lookup(s):
    conn = RecordingConn(s)
    try:
        obj = _oneloop_build(s, conn)
        found = f"found={obj.available}"
    except Exception as e:  # noqa: BLE001
        found = (
            f"a row matched, and constructing the object raised {type(e).__name__}: {e}"
        )
    sql = conn.statements[-1]
    return found, sql[sql.find("JOIN") : sql.find("WHERE")].strip()


def _oneloop_build(s, conn):
    return sqla_OneLoopIntegral_factory.build(
        dict(
            model=StandInModelProxy(),
            k=Ref(11),
            z_response=Ref(1),
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-8.0),
            tags=[store_tag(store_id=1, label="T")],
        ),
        conn,
        s.tables["OneLoopIntegral"],
        None,
        s.tables,
        None,
    )


def main():
    print("=" * 90)
    print(
        "[00-oneloop-lookup-joins-the-wrong-tag-table]  OneLoopIntegral serial 5, requested tag T"
    )
    print("=" * 90)
    cases = [
        ("tagged in OneLoopIntegral_tags only (correctly tagged)", [], [5]),
        (
            "tagged correctly AND a QuadSourceIntegral with serial 5 carries tag T",
            [5],
            [5],
        ),
        ("not tagged; a QuadSourceIntegral with serial 5 carries tag T", [5], []),
    ]
    for label, qsi, oli in cases:
        found, join = oneloop_lookup(oneloop_store(qsi, oli))
        print(f"{label}: {found}")
        print(f"   join: {join}")

    print("\n" + "=" * 90)
    print("redshift.build at z = 0")
    print("=" * 90)
    s = Store()
    t = s.tables["redshift"]
    for i in range(3):
        obj = sqla_redshift_factory.build(
            dict(z=0.0, is_source=False, is_response=True),
            s.conn,
            t,
            s.inserters["redshift"],
            s.tables,
            s.inserters,
        )
        rows = s.conn.execute(t.select()).fetchall()
        print(
            f"call {i + 1}: store_id={obj.store_id} new_insert={getattr(obj, '_new_insert', False)} rows in table={len(rows)}"
        )
    print(
        "SQLite value of (z - 0.0)/0.0:",
        s.conn.exec_driver_sql("select (0.0 - 0.0)/0.0").scalar(),
    )

    print("\n" + "=" * 90)
    print(
        ".scalar() lookups with a duplicate present (tolerance, two rows at log10 -10)"
    )
    print("=" * 90)
    s = Store()
    s.insert("tolerance", serial=7, log10_tol=-10.0)
    s.insert("tolerance", serial=3, log10_tol=-10.0)
    s.conn.commit()
    obj = sqla_tolerance_factory.build(
        dict(log10_tol=-10.0),
        s.conn,
        s.tables["tolerance"],
        s.inserters["tolerance"],
        s.tables,
        s.inserters,
    )
    print(f"served store_id={obj.store_id}; no error raised")


if __name__ == "__main__":
    main()
