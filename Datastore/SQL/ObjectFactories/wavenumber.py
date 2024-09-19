import sqlalchemy as sqla

from CosmologyConcepts import wavenumber, wavenumber_exit_time
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import tolerance
from Units.base import UnitsLike
from defaults import (
    DEFAULT_FLOAT_PRECISION,
)


class sqla_wavenumber_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("k_inv_Mpc", sqla.Float(64))],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        k_inv_Mpc: float = payload["k_inv_Mpc"]
        units: UnitsLike = payload["units"]

        # query for this wavenumber in the datastore
        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.func.abs(table.c.k_inv_Mpc - k_inv_Mpc) < DEFAULT_FLOAT_PRECISION
            )
        ).scalar()

        # if not present, create a new id using the provided inserter
        if store_id is None:
            store_id = inserter(conn, {"k_inv_Mpc": k_inv_Mpc})

        # return constructed object
        return wavenumber(store_id=store_id, k_inv_Mpc=k_inv_Mpc, units=units)

    @staticmethod
    def read_table(conn, table, units: UnitsLike):
        # query for all wavenumber records in the table
        rows = conn.execute(
            sqla.select(
                table.c.serial,
                table.c.k_inv_Mpc,
            ).order_by(table.c.k_inv_Mpc)
        )

        return [
            wavenumber(store_id=row.serial, k_inv_Mpc=row.k_inv_Mpc, units=units)
            for row in rows
        ]


class sqla_wavenumber_exit_time_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        # Does not set up a foreign key constraint for the cosmology object.
        # The issue is that this is polymorphic, because we have different implementations of the CosmologyBase concept.
        # Rather than try to deal with this using SQLAlchemy-level polymorphism, we handle the polymorphism ourselves
        # and just skip foreign key constraints here
        return {
            "version": True,
            "timestamp": True,
            "stepping": "minimum",
            "columns": [
                sqla.Column(
                    "wavenumber_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber.serial"),
                    nullable=False,
                ),
                sqla.Column("cosmology_type", sqla.Integer, nullable=False),
                sqla.Column("cosmology_serial", sqla.Integer, nullable=False),
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "rtol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("z_exit", sqla.Float(64)),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        target_atol = payload["atol"]
        target_rtol = payload["rtol"]

        k = payload["k"]
        cosmology = payload["cosmology"]

        target_stepping = payload.get("stepping", 0)

        # Note that if we just have .select_from(BASE_TABLE), we cannot access any columns from the joined tables (at least using SQLite)
        # see: https://stackoverflow.com/questions/68137220/getting-columns-of-the-joined-table-using-sqlalchemy-core

        # order by descending values of abs and relative tolerances, so that we get the best computed value we hold
        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        row_data = conn.execute(
            sqla.select(
                table.c.serial,
                table.c.stepping,
                table.c.atol_serial,
                table.c.rtol_serial,
                table.c.compute_time,
                atol_table.c.log10_tol.label("log10_atol"),
                rtol_table.c.log10_tol.label("log10_rtol"),
                table.c.z_exit,
            )
            .select_from(
                table.join(atol_table, atol_table.c.serial == table.c.atol_serial).join(
                    rtol_table, rtol_table.c.serial == table.c.rtol_serial
                )
            )
            .filter(
                sqla.and_(
                    table.c.wavenumber_serial == k.store_id,
                    table.c.cosmology_type == cosmology.type_id,
                    table.c.cosmology_serial == cosmology.store_id,
                    atol_table.c.log10_tol - target_atol.log10_tol
                    <= DEFAULT_FLOAT_PRECISION,
                    rtol_table.c.log10_tol - target_rtol.log10_tol
                    <= DEFAULT_FLOAT_PRECISION,
                    table.c.stepping >= target_stepping,
                )
            )
            .order_by(atol_table.c.log10_tol.desc(), rtol_table.c.log10_tol.desc())
        ).one_or_none()

        if row_data is None:
            # build and return an unpopulated object
            return wavenumber_exit_time(
                payload=None,
                k=k,
                cosmology=cosmology,
                atol=target_atol,
                rtol=target_rtol,
            )

        store_id = row_data.serial
        z_exit = row_data.z_exit
        compute_time = row_data.compute_time
        stepping = row_data.stepping

        atol = tolerance(store_id=row_data.atol_serial, log10_tol=row_data.log10_atol)
        rtol = tolerance(store_id=row_data.rtol_serial, log10_tol=row_data.log10_rtol)

        return wavenumber_exit_time(
            payload={
                "store_id": store_id,
                "z_exit": z_exit,
                "compute_time": compute_time,
                "stepping": stepping,
            },
            k=k,
            cosmology=cosmology,
            atol=atol,
            rtol=rtol,
        )

    @staticmethod
    def store(
        obj: wavenumber_exit_time,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        # now serialize the computed value in the database
        store_id = inserter(
            conn,
            {
                "stepping": obj.stepping,
                "wavenumber_serial": obj.k.store_id,
                "cosmology_type": obj.cosmology.type_id,
                "cosmology_serial": obj.cosmology.store_id,
                "atol_serial": obj._atol.store_id,
                "rtol_serial": obj._rtol.store_id,
                "compute_time": obj.compute_time,
                "z_exit": obj.z_exit,
            },
        )

        # set store_id on behalf of constructed object
        obj._my_id = store_id

        return obj
