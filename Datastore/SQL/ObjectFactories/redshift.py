import sqlalchemy as sqla

from CosmologyConcepts import redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_REDSHIFT_RELATIVE_PRECISION


class sqla_redshift_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("z", sqla.Float(64), index=True)],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        # query for this redshift in the datastore
        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.func.abs((table.c.z - z) / z) < DEFAULT_REDSHIFT_RELATIVE_PRECISION
            )
        ).scalar()

        # if not present, create a new id using the provided inserter
        if store_id is None:
            store_id = inserter(conn, {"z": z})

        # return constructed object
        return redshift(store_id=store_id, z=z)
