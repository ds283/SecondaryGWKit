"""
Probes for the value-table read paths: [00-numeric-value-parent-lookup-omits-break-point-kind]
and the new findings in the value factories' _build_impl_* helpers and read_batch methods.

Every call is to the real factory code, against an in-memory store. Each case prints what the code
does; the report says what it should do.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ComputeTargets.phase_residual as phase_residual  # noqa: E402
from common import RecordingConn, StandInModelProxy, Store  # noqa: E402
from ComputeTargets import GkNumericIntegration, TkNumericIntegration  # noqa: E402
from CosmologyConcepts import redshift  # noqa: E402
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (  # noqa: E402
    sqla_GkNumericValue_factory,
)
from Datastore.SQL.ObjectFactories.GkSource import (
    sqla_GkSourceValue_factory,
)  # noqa: E402
from Datastore.SQL.ObjectFactories.GkWKBIntegration import (  # noqa: E402
    sqla_GkWKBValue_factory,
)
from Datastore.SQL.ObjectFactories.QuadSource import (  # noqa: E402
    sqla_QuadSourceValue_factory,
)
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (  # noqa: E402
    sqla_TkNumericValue_factory,
)
from Datastore.SQL.ObjectFactories.TkWKBIntegration import (  # noqa: E402
    sqla_TkWKBValue_factory,
)
from common import Ref  # noqa: E402
from MetadataConcepts import tolerance  # noqa: E402

GK_KIND = GkNumericIntegration.BREAK_POINT_KIND
TK_KIND = TkNumericIntegration.BREAK_POINT_KIND
OTHER = {"discontinuity": "all", "all": "discontinuity"}


def z(serial, value, is_source=False, is_response=False):
    return redshift(
        store_id=serial, z=value, is_source=is_source, is_response=is_response
    )


def base_store():
    s = Store()
    s.insert("version", serial=1, label="probe")
    s.insert("tolerance", serial=1, log10_tol=-10.0)
    s.insert("tolerance", serial=2, log10_tol=-8.0)
    s.insert("IntegrationSolver", serial=1, label="solver", stepping=0)
    # z_source (serial 1) is source-only; z_response (serial 2) is response-only, so a flag read
    # off the wrong column is visible
    s.insert("redshift", serial=1, z=1.0e5, source=True, response=False)
    s.insert("redshift", serial=2, z=10.0, source=False, response=True)
    s.insert("redshift", serial=3, z=1.0, source=False, response=True)
    s.insert("store_tag", serial=1, label="T")
    return s


def run(label, fn):
    try:
        print(f"{label}: {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"{label}: {type(e).__name__}: {str(e).splitlines()[0]}")


def gk_numeric_parent(s, serial, kind):
    s.insert(
        "GkNumericIntegration",
        serial=serial,
        label=f"gk-{serial}",
        wavenumber_exit_serial=11,
        model_serial=1,
        atol_serial=1,
        rtol_serial=2,
        break_point_kind=kind,
        solver_serial=1,
        z_source_serial=1,
        z_min_serial=3,
        z_samples=1,
        validated=True,
    )
    s.insert("GkNumeric_tags", integration_serial=serial, tag_serial=1)
    s.insert(
        "GkNumericValue",
        serial=10 * serial,
        integration_serial=serial,
        z_serial=2,
        G=float(serial),
        Gprime=0.0,
    )


def tk_numeric_parent(s, serial, kind):
    s.insert(
        "TkNumericIntegration",
        serial=serial,
        label=f"tk-{serial}",
        wavenumber_exit_serial=11,
        model_serial=1,
        atol_serial=1,
        rtol_serial=2,
        break_point_kind=kind,
        solver_serial=1,
        z_init_serial=1,
        z_min_serial=3,
        z_samples=1,
        validated=True,
    )
    s.insert("TkNumeric_tags", integration_serial=serial, tag_serial=1)
    s.insert(
        "TkNumericValue",
        serial=10 * serial,
        integration_serial=serial,
        z_serial=2,
        T=float(serial),
        Tprime=0.0,
    )


def gk_wkb_parent(s, serial, z_init):
    s.insert(
        "GkWKBIntegration",
        serial=serial,
        label=f"gkwkb-{serial}",
        wavenumber_exit_serial=11,
        model_serial=1,
        rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
        solver_serial=1,
        z_source_serial=1,
        z_min_serial=3,
        z_samples=1,
        z_init=z_init,
        sin_coeff=0.0,
        cos_coeff=1.0,
        G_init=0.0,
        Gprime_init=1.0,
        validated=True,
    )
    s.insert("GkWKB_tags", wkb_serial=serial, tag_serial=1)
    s.insert(
        "GkWKBValue",
        serial=10 * serial,
        wkb_serial=serial,
        z_serial=2,
        H_ratio=1.0,
        theta_mod_2pi=0.5,
        theta_div_2pi=3,
        omega_WKB_sq=1.0,
        WKB_criterion=0.1,
        G_WKB=float(serial),
    )


def tk_wkb_parent(s, serial):
    s.insert(
        "TkWKBIntegration",
        serial=serial,
        label=f"tkwkb-{serial}",
        wavenumber_exit_serial=11,
        model_serial=1,
        rho_gauss_order=phase_residual.RHO_GAUSS_ORDER,
        phase_solver_serial=1,
        friction_solver_serial=1,
        z_min_serial=3,
        z_samples=1,
        z_init=1.0e4,
        sin_coeff=0.0,
        cos_coeff=1.0,
        T_init=1.0,
        Tprime_init=0.0,
        validated=True,
    )
    s.insert("TkWKB_tags", wkb_serial=serial, tag_serial=1)
    s.insert(
        "TkWKBValue",
        serial=10 * serial,
        wkb_serial=serial,
        z_serial=2,
        H_ratio=1.0,
        theta_mod_2pi=0.5,
        theta_div_2pi=3,
        friction=0.25,
        omega_WKB_sq=1.0,
        WKB_criterion=0.1,
        T_WKB=1.0,
    )


def main():
    tag = Ref(1)
    model = StandInModelProxy()
    k = Ref(11)
    atol = tolerance(store_id=1, log10_tol=-10.0)
    rtol = tolerance(store_id=2, log10_tol=-8.0)

    print("=" * 96)
    print("[00-numeric-value-parent-lookup-omits-break-point-kind]")
    print(
        f"GkNumericIntegration.BREAK_POINT_KIND={GK_KIND!r}  TkNumericIntegration={TK_KIND!r}"
    )
    print("=" * 96)
    for n_parents in (1, 2):
        s = base_store()
        gk_numeric_parent(s, 1, GK_KIND)
        tk_numeric_parent(s, 1, TK_KIND)
        if n_parents == 2:
            gk_numeric_parent(s, 2, OTHER[GK_KIND])
            tk_numeric_parent(s, 2, OTHER[TK_KIND])
        s.conn.commit()
        print(
            f"\n-- store with {n_parents} parent(s) differing only in break_point_kind"
        )
        gk_payload = dict(
            z=z(2, 10.0),
            model=model,
            k=k,
            z_source=z(1, 1.0e5),
            atol=atol,
            rtol=rtol,
            tags=[tag],
        )
        run(
            "GkNumericValue build (model path)",
            lambda: f"G={sqla_GkNumericValue_factory.build(gk_payload, s.conn, s.tables['GkNumericValue'], None, s.tables, None).G}",
        )
        run(
            "GkNumericValue.read_batch",
            lambda: [
                (o.store_id, o.G)
                for o in sqla_GkNumericValue_factory.read_batch(
                    dict(model=model, k=k, atol=atol, rtol=rtol, tags=[tag]),
                    s.conn,
                    s.tables["GkNumericValue"],
                    s.tables,
                )
            ],
        )
        tk_payload = dict(
            z=z(2, 10.0),
            model=model,
            k=k,
            z_init=z(1, 1.0e5),
            atol=atol,
            rtol=rtol,
            tags=[tag],
        )
        run(
            "TkNumericValue build, payload with z_init (what build() dispatches on)",
            lambda: f"T={sqla_TkNumericValue_factory.build(tk_payload, s.conn, s.tables['TkNumericValue'], None, s.tables, None).T}",
        )
        tk_payload2 = dict(tk_payload, z_source=z(1, 1.0e5))
        run(
            "TkNumericValue build, payload with z_init AND z_source",
            lambda: f"T={sqla_TkNumericValue_factory.build(tk_payload2, s.conn, s.tables['TkNumericValue'], None, s.tables, None).T}",
        )

    print("\n-- tags omitted (payload.get('tags', None) then 'for tag in tags')")
    s = base_store()
    gk_numeric_parent(s, 1, GK_KIND)
    s.conn.commit()
    run(
        "GkNumericValue.read_batch without tags",
        lambda: sqla_GkNumericValue_factory.read_batch(
            dict(model=model, k=k), s.conn, s.tables["GkNumericValue"], s.tables
        ),
    )

    print("\n" + "=" * 96)
    print(
        "GkNumericValue.read_batch: flags of the returned z_response (row serial 2: source=False, response=True)"
    )
    print("=" * 96)
    objs = sqla_GkNumericValue_factory.read_batch(
        dict(model=model, k=k, atol=atol, rtol=rtol, tags=[tag]),
        s.conn,
        s.tables["GkNumericValue"],
        s.tables,
    )
    for o in objs:
        print(
            f"z_response.is_source={o.z.is_source} z_response.is_response={o.z.is_response}"
        )

    print("\n" + "=" * 96)
    print(
        "GkWKBValue: two GkWKBIntegration parents, same (model, k, z_source), z_init 1e4 and 2e4"
    )
    print("=" * 96)
    s = base_store()
    gk_wkb_parent(s, 1, 1.0e4)
    gk_wkb_parent(s, 2, 2.0e4)
    s.conn.commit()
    run(
        "GkWKBValue.read_batch",
        lambda: [
            (o.store_id, o.z_init)
            for o in sqla_GkWKBValue_factory.read_batch(
                dict(model=model, k=k, tags=[tag]),
                s.conn,
                s.tables["GkWKBValue"],
                s.tables,
            )
        ],
    )
    run(
        "GkWKBValue build (model path)",
        lambda: sqla_GkWKBValue_factory.build(
            dict(z=z(2, 10.0), model=model, k=k, z_source=z(1, 1.0e5), tags=[tag]),
            s.conn,
            s.tables["GkWKBValue"],
            None,
            s.tables,
            None,
        ).store_id,
    )
    conn = RecordingConn(s)
    run(
        "GkWKBValue build (serial path, existing row)",
        lambda: sqla_GkWKBValue_factory.build(
            dict(z=z(2, 10.0), wkb_serial=1),
            conn,
            s.tables["GkWKBValue"],
            None,
            s.tables,
            None,
        ).store_id,
    )
    if conn.statements:
        sql = conn.statements[-1]
        print("   its SQL (FROM onward): " + sql[sql.find("FROM") :].replace("\n", " "))

    print("\n" + "=" * 96)
    print("TkWKBValue")
    print("=" * 96)
    s = base_store()
    tk_wkb_parent(s, 1)
    s.conn.commit()
    run(
        "TkWKBValue.read_batch",
        lambda: sqla_TkWKBValue_factory.read_batch(
            dict(model=model, k=k, tags=[tag]), s.conn, s.tables["TkWKBValue"], s.tables
        ),
    )
    run(
        "TkWKBValue build (serial path, existing row)",
        lambda: sqla_TkWKBValue_factory.build(
            dict(z=z(2, 10.0), wkb_serial=1),
            s.conn,
            s.tables["TkWKBValue"],
            None,
            s.tables,
            None,
        ).store_id,
    )
    run(
        "TkWKBValue build (model path)",
        lambda: sqla_TkWKBValue_factory.build(
            dict(z=z(2, 10.0), model=model, k=k, tags=[tag]),
            s.conn,
            s.tables["TkWKBValue"],
            None,
            s.tables,
            None,
        ).friction,
    )

    print("\n" + "=" * 96)
    print("GkSourceValue / QuadSourceValue helpers")
    print("=" * 96)
    s = base_store()
    s.conn.commit()
    run(
        "GkSourceValue build (serial path)",
        lambda: sqla_GkSourceValue_factory.build(
            dict(z_source=z(1, 1.0e5), parent_serial=1),
            s.conn,
            s.tables["GkSourceValue"],
            None,
            s.tables,
            None,
        ),
    )
    run(
        "QuadSourceValue build (model path)",
        lambda: sqla_QuadSourceValue_factory.build(
            dict(z=z(2, 10.0), model=model, q=Ref(11), r=Ref(12), tags=[]),
            s.conn,
            s.tables["QuadSourceValue"],
            None,
            s.tables,
            None,
        ),
    )


if __name__ == "__main__":
    main()
