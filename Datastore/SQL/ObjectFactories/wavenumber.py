import sqlalchemy as sqla

from CosmologyConcepts import wavenumber, wavenumber_exit_time, tolerance
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
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
    async def build(
        payload,
        engine,
        conn,
        table,
        full_query,
        serial_query,
        inserter,
        tables,
        inserters,
    ):
        k_inv_Mpc = payload["k_inv_Mpc"]
        units = payload["units"]

        # query for this wavenumber in the datastore
        ref = await conn.execute(
            serial_query.filter(
                sqla.func.abs(table.c.k_inv_Mpc - k_inv_Mpc) < DEFAULT_FLOAT_PRECISION
            )
        )
        store_id = ref.scalar()

        # if not present, create a new id using the provided inserter
        if store_id is None:
            store_id = await inserter(conn, {"k_inv_Mpc": k_inv_Mpc})

        # return constructed object
        return wavenumber(store_id=store_id, k_inv_Mpc=k_inv_Mpc, units=units)


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
    async def build(
        payload,
        engine,
        conn,
        table,
        full_query,
        serial_query,
        inserter,
        tables,
        inserters,
    ):
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

        ref = await conn.execute(
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
        )
        row_data = ref.one_or_none()

        if row_data is not None:
            store_id = row_data.serial
            z_exit = row_data.z_exit
            compute_time = row_data.compute_time
            stepping = row_data.stepping

            atol = tolerance(
                store_id=row_data.atol_serial, log10_tol=row_data.log10_atol
            )
            rtol = tolerance(
                store_id=row_data.rtol_serial, log10_tol=row_data.log10_rtol
            )

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

        else:
            obj = wavenumber_exit_time(
                payload=None,
                k=k,
                cosmology=cosmology,
                atol=target_atol,
                rtol=target_rtol,
            )

            # asynchronously compute the horizon-exit time and await the result
            # (this yields control so that other actor services can run while the result completes)
            data = await obj.compute()

            # read the result of the computation, and use to populate the constructed object
            res = await obj.store()
            if res is None:
                raise RuntimeError("compute() did not generate a running future")
            if res is False:
                raise RuntimeError(
                    "await compute() returned, but the future was not resolved"
                )

            # now serialize the computed value in the database
            store_id = await inserter(
                conn,
                {
                    "stepping": obj.stepping,
                    "wavenumber_serial": k.store_id,
                    "cosmology_type": cosmology.type_id,
                    "cosmology_serial": cosmology.store_id,
                    "atol_serial": target_atol.store_id,
                    "rtol_serial": target_rtol.store_id,
                    "compute_time": obj.compute_time,
                    "z_exit": obj.z_exit,
                },
            )

            # set store_id on behalf of constructed object
            obj._my_id = store_id

            return obj
