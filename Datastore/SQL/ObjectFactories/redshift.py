import sqlalchemy as sqla

from CosmologyConcepts import redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_FLOAT_PRECISION


class sqla_redshift_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("z", sqla.Float(64))],
        }

    @staticmethod
    async def build(
        engine, conn, table, full_query, serial_query, tables, inserter, payload
    ):
        z = payload["z"]

        # query for this redshift in the datastore
        ref = await conn.execute(
            serial_query.filter(sqla.func.abs(table.c.z - z) < DEFAULT_FLOAT_PRECISION)
        )
        store_id = ref.scalar()

        # if not present, create a new id using the provided inserter
        if store_id is None:
            store_id = await inserter(conn, {"z": z})

        # return constructed object
        return redshift(store_id=store_id, z=z)
