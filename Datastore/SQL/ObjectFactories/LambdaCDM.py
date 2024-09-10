import sqlalchemy as sqla

from CosmologyModels.LambdaCDM import LambdaCDM
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


class sqla_LambdaCDM_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [
                sqla.Column("name", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column("omega_m", sqla.Float(64)),
                sqla.Column("omega_cc", sqla.Float(64)),
                sqla.Column("h", sqla.Float(64)),
                sqla.Column("f_baryon", sqla.Float(64)),
                sqla.Column("T_CMB_Kelvin", sqla.Float(64)),
                sqla.Column("Neff", sqla.Float(64)),
            ],
        }

    @staticmethod
    async def build(
        engine, conn, table, full_query, serial_query, tables, inserter, payload
    ):
        params = payload["params"]
        units = payload["units"]

        name = params.name
        omega_m = params.omega_m
        omega_cc = params.omega_cc
        h = params.h
        f_baryon = params.f_baryon
        T_CMB_Kelvin = params.T_CMB_Kelvin
        Neff = params.Neff

        ref = await conn.execute(
            full_query.filter(
                sqla.and_(
                    sqla.func.abs(table.c.omega_m - omega_m) < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.omega_cc - omega_cc)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.h - h) < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.f_baryon - f_baryon)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.T_CMB_Kelvin - T_CMB_Kelvin)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.Neff - Neff) < DEFAULT_FLOAT_PRECISION,
                )
            )
        )
        row_data = ref.one_or_none()

        # if not present, create a new id using the provided inserter
        if row_data is None:
            store_id = await inserter(
                conn,
                {
                    "name": name,
                    "omega_m": omega_m,
                    "omega_cc": omega_cc,
                    "h": h,
                    "f_baryon": f_baryon,
                    "T_CMB_Kelvin": T_CMB_Kelvin,
                    "Neff": Neff,
                },
            )
        else:
            store_id = row_data.serial

        return LambdaCDM(
            store_id=store_id,
            units=units,
            params=params,
        )
