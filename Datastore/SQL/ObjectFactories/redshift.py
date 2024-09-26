import sqlalchemy as sqla

from CosmologyConcepts import redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_REDSHIFT_RELATIVE_PRECISION


class sqla_redshift_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
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
            insert_data = {"z": z}
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(conn, insert_data)

        # return constructed object
        return redshift(store_id=store_id, z=z)

    @staticmethod
    def read_table(conn, table):
        # query for all redshift records in the table
        rows = conn.execute(
            sqla.select(
                table.c.serial,
                table.c.z,
            ).order_by(table.c.z)
        )

        return [redshift(store_id=row.serial, z=row.z) for row in rows]
