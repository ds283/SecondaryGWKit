"""Read-only probe, in-memory SQLite only: with PRAGMA foreign_keys=ON, does the declared
QuadSource.Tq_serial -> QuadSource.serial constraint reject a row whose Tq_serial is a genuine
TkNumericIntegration serial (as ComputeTargets/QuadSource.py:473 writes)? And does the
retargeted constraint (-> TkNumericIntegration.serial) accept it?"""

import sqlalchemy as sqla

from config.datastore import factories as _factories
from datastorekit.SQL.schema import build_schema


def attempt(retarget):
    # seed the parents with foreign_keys OFF, then switch it ON for the QuadSource insert
    md = sqla.MetaData()
    import ObjectFactories.QuadSource as QS

    factories = dict(_factories)
    if retarget:
        orig = QS.sqla_QuadSource_factory.register

        def patched():
            reg = orig()
            reg["columns"] = [
                (
                    sqla.Column(
                        "Tq_serial",
                        sqla.Integer,
                        sqla.ForeignKey("TkNumericIntegration.serial"),
                        index=True,
                        nullable=False,
                    )
                    if c.name == "Tq_serial"
                    else c
                )
                for c in reg["columns"]
            ]
            return reg

        class F(QS.sqla_QuadSource_factory):
            register = staticmethod(patched)

        factories["QuadSource"] = F
    built = build_schema(md, factories)
    eng = sqla.create_engine("sqlite://")
    md.create_all(eng)
    t = built.tables
    with eng.connect() as c:
        c.exec_driver_sql("PRAGMA foreign_keys=OFF")
        # parents of QuadSource: BackgroundModel 1, wavenumber_exit_time 1 and 2,
        # TkNumericIntegration 7 (q) -- inserted raw, bypassing their own FKs and NOT NULLs
        c.exec_driver_sql("PRAGMA ignore_check_constraints=ON")
        for name, serial in [
            ("BackgroundModel", 1),
            ("wavenumber_exit_time", 1),
            ("wavenumber_exit_time", 2),
            ("TkNumericIntegration", 7),
        ]:
            cols = [
                col
                for col in t[name].columns
                if not col.nullable and col.name != "serial"
            ]
            names = ",".join(["serial"] + [f'"{col.name}"' for col in cols])
            vals = ",".join([str(serial)] + ["0"] * len(cols))
            c.exec_driver_sql(f'INSERT INTO "{name}" ({names}) VALUES ({vals})')
        c.commit()
        c.exec_driver_sql("PRAGMA foreign_keys=ON")
        try:
            c.execute(
                t["QuadSource"].insert(),
                {
                    "serial": 100,
                    "model_serial": 1,
                    "q_wavenumber_exit_serial": 1,
                    "r_wavenumber_exit_serial": 2,
                    "Tq_serial": 7,
                    "Tr_serial": 8,
                    "z_samples": 0,
                    "validated": False,
                },
            )
            c.commit()
            return "accepted"
        except sqla.exc.IntegrityError as e:
            return f"REJECTED: {e.orig}"


print(
    "declared FK (QuadSource.serial), Tq_serial=7 is a TkNumericIntegration serial ->",
    attempt(False),
)
print("retargeted FK (TkNumericIntegration.serial), same row ->", attempt(True))
