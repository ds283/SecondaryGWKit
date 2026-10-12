"""
Probe for [00-tagged-read-batch-joins-an-unselected-alias].

Calls the real ``sqla_QuadSourceIntegral_factory.read_batch`` (and the module-level
``OneLoopIntegral.read_batch``) against an in-memory store holding three QuadSourceIntegral rows
with the same key and different tag sets, and reports for 0, 1 and 2 requested tags: the compiled
SQL, any warning, whether it executes, and which rows come back. Then runs the same query with the
tag join written the way every other factory writes it (join against the base table's column).

Store contents:
  QSI serial 1: tags {A, B}
  QSI serial 2: tags {A}
  QSI serial 3: no tags
Expected: tags=[] -> {1,2,3}; tags=[A] -> {1,2}; tags=[A,B] -> {1}.
"""

import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sqlalchemy as sqla  # noqa: E402
from sqlalchemy import and_  # noqa: E402

from common import Ref, RecordingConn, StandInModelProxy, Store  # noqa: E402
from CosmologyConcepts import wavenumber, wavenumber_exit_time  # noqa: E402
from CosmologyConcepts.wavenumber import (  # noqa: E402
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
)
from ObjectFactories import OneLoopIntegral as OLI_module  # noqa: E402
from ObjectFactories.QuadSourceIntegral import (  # noqa: E402
    sqla_QuadSourceIntegral_factory,
)
from MetadataConcepts import store_tag, tolerance  # noqa: E402
from common import UNITS  # noqa: E402


def populate(s: Store):
    s.insert("version", serial=1, label="probe")
    s.insert("tolerance", serial=1, log10_tol=-10.0)
    s.insert("tolerance", serial=2, log10_tol=-9.0)
    s.insert("store_tag", serial=1, label="A")
    s.insert("store_tag", serial=2, label="B")
    s.insert("redshift", serial=1, z=10.0, source=False, response=True)
    s.insert("redshift", serial=2, z=1.0e5, source=True, response=False)
    # wavenumbers 1..3 (k, q, r) with serials deliberately different from their exit serials
    for serial, k in ((1, 1.0e5), (2, 2.0e5), (3, 3.0e5)):
        s.insert("wavenumber", serial=serial, k_inv_Mpc=k, source=True, response=True)
    for exit_serial, k_serial in ((11, 1), (12, 2), (13, 3)):
        row = dict(
            serial=exit_serial,
            stepping=0,
            wavenumber_serial=k_serial,
            cosmology_type=0,
            cosmology_serial=1,
            atol_serial=1,
            rtol_serial=2,
            compute_time=0.1,
            z_exit=1.0e4 * k_serial,
        )
        for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
            row[f"z_exit_suph_e{o}"] = 1.0
        for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
            row[f"z_exit_subh_e{o}"] = 1.0
        s.insert("wavenumber_exit_time", **row)

    for serial in (1, 2, 3):
        s.insert(
            "QuadSourceIntegral",
            serial=serial,
            label=f"qsi-{serial}",
            model_serial=1,
            k_wavenumber_exit_serial=11,
            q_wavenumber_exit_serial=12,
            r_wavenumber_exit_serial=13,
            atol_serial=1,
            rtol_serial=2,
            policy_serial=1,
            source_serial=100 + serial,
            data_serial=200 + serial,
            z_response_serial=1,
            z_source_max_serial=2,
            b=0.0,
            total=float(serial),
            numeric_quad=0.0,
            WKB_Levin=0.0,
        )
    for parent, tag in ((1, 1), (1, 2), (2, 1)):
        s.insert("QuadSourceIntegral_tags", parent_serial=parent, tag_serial=tag)

    for serial in (1, 2, 3):
        s.insert(
            "OneLoopIntegral",
            serial=serial,
            label=f"oli-{serial}",
            model_serial=1,
            wavenumber_exit_serial=11,
            atol_serial=1,
            rtol_serial=2,
            z_response_serial=1,
            value=float(serial),
        )
    for parent, tag in ((1, 1), (1, 2), (2, 1)):
        s.insert("OneLoopIntegral_tags", parent_serial=parent, tag_serial=tag)
    s.conn.commit()


def exit_time(serial, k_serial, k):
    payload = {"store_id": serial, "z_exit": 1.0, "compute_time": 0.0, "stepping": 0}
    for o in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
        payload[f"z_exit_suph_e{o}"] = 1.0
    for o in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
        payload[f"z_exit_subh_e{o}"] = 1.0
    return wavenumber_exit_time(
        payload=payload,
        k=wavenumber(store_id=k_serial, k_inv_Mpc=k, units=UNITS),
        cosmology=StandInModelProxy().cosmology,
        atol=tolerance(store_id=1, log10_tol=-10.0),
        rtol=tolerance(store_id=2, log10_tol=-9.0),
    )


def main():
    s = Store()
    populate(s)
    tags = {
        "A": store_tag(store_id=1, label="A"),
        "B": store_tag(store_id=2, label="B"),
    }
    k_exit = exit_time(11, 1, 1.0e5)
    q_exit = exit_time(12, 2, 2.0e5)
    r_exit = exit_time(13, 3, 3.0e5)

    expected = {(): {1, 2, 3}, ("A",): {1, 2}, ("A", "B"): {1}}

    print("=" * 100)
    print("QuadSourceIntegral.read_batch (the real factory method)")
    print("=" * 100)
    for names, want in expected.items():
        conn = RecordingConn(s)
        payload = dict(
            model=StandInModelProxy(),
            policy=Ref(1),
            k=k_exit,
            q=q_exit,
            r=r_exit,
            z_response=None,
            z_source_max=None,
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-9.0),
            tags=[tags[n] for n in names],
        )
        print(f"\n--- tags={list(names)}  expected serials={sorted(want)}")
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                objs = sqla_QuadSourceIntegral_factory.read_batch(
                    payload, conn, s.tables["QuadSourceIntegral"], s.tables
                )
            got = sorted(o.store_id for o in objs)
            print(f"executed: yes; returned serials={got}; correct={set(got) == want}")
            q_ids = sorted({o._q_exit.store_id for o in objs})
            print(
                f"returned q_exit.store_id values={q_ids} "
                f"(the stored q_wavenumber_exit_serial is 12; the q wavenumber serial is 2)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"executed: NO -> {type(e).__name__}: {e}".splitlines()[0])
        for w in caught:
            print(f"warning (building query): {w.category.__name__}: {w.message}")
        for w in conn.warnings:
            print(f"warning (executing): {w}")
        tail = conn.statements[-1] if conn.statements else "<none>"
        print("compiled SQL, tag joins and WHERE:")
        joins = re.findall(
            r'JOIN "QuadSourceIntegral_tags" AS \S+ ON [^J]*?(?= JOIN| WHERE|$)',
            tail.replace("\n", " "),
        )
        for j in joins:
            print("   " + j)
        print("   " + tail[tail.find("WHERE") :].replace("\n", " "))

    # the same query, with the tag join written against the base table as every other factory does
    print("\n" + "=" * 100)
    print("Control: identical query, tag join ON tab.c.parent_serial == table.c.serial")
    print("=" * 100)
    table = s.tables["QuadSourceIntegral"]
    for names, want in expected.items():
        q = sqla.select(table.c.serial).filter(
            table.c.model_serial == 1,
            table.c.policy_serial == 1,
            table.c.k_wavenumber_exit_serial == 11,
            table.c.atol_serial == 1,
            table.c.rtol_serial == 2,
        )
        for i, n in enumerate(names):
            tab = s.tables["QuadSourceIntegral_tags"].alias(f"tag_{i}")
            q = q.join(
                tab,
                and_(
                    tab.c.parent_serial == table.c.serial,
                    tab.c.tag_serial == tags[n].store_id,
                ),
            )
        got = sorted(r.serial for r in s.conn.execute(q))
        print(f"tags={list(names)} -> {got}; correct={set(got) == want}")
        print("   " + s.compile(q)[s.compile(q).find("FROM") :].replace("\n", " "))

    print("\n" + "=" * 100)
    print("OneLoopIntegral: module-level read_batch (not a factory method)")
    print("=" * 100)
    from ObjectFactories.OneLoopIntegral import (
        sqla_OneLoopIntegral_factory,
    )

    print(
        "sqla_OneLoopIntegral_factory has read_batch attribute:",
        hasattr(sqla_OneLoopIntegral_factory, "read_batch"),
    )
    for names, want in list(expected.items()) + [(None, {1, 2, 3})]:
        conn = RecordingConn(s)
        payload = dict(
            model=StandInModelProxy(),
            k=k_exit,
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-9.0),
        )
        if names is not None:
            payload["tags"] = [tags[n] for n in names]
        label = "omitted" if names is None else list(names)
        try:
            ret = OLI_module.read_batch(
                payload, conn, s.tables["OneLoopIntegral"], s.tables
            )
            print(f"tags={label}: executed; return value={ret!r}")
        except Exception as e:  # noqa: BLE001
            print(f"tags={label}: {type(e).__name__}: {e}".splitlines()[0])
        for w in conn.warnings:
            print(f"   warning (executing): {w}")
        if conn.statements:
            tail = conn.statements[-1]
            print("   " + tail[tail.find("FROM") :].replace("\n", " "))


if __name__ == "__main__":
    main()
