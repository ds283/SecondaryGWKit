from typing import Optional

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
            "columns": [
                sqla.Column("z", sqla.Float(64), index=True),
                sqla.Column("source", sqla.Boolean, default=False),
                sqla.Column("response", sqla.Boolean, default=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]
        is_source = payload["is_source"]
        is_response = payload["is_response"]

        # query for this redshift in the datastore
        query = sqla.select(
            table.c.serial,
            table.c.source,
            table.c.response,
        ).filter(
            sqla.func.abs((table.c.z - z) / z) < DEFAULT_REDSHIFT_RELATIVE_PRECISION
        )
        row_data = conn.execute(query).one_or_none()

        # if not present, create a new id using the provided inserter
        if row_data is None:
            insert_data = {"z": z, "source": is_source, "response": is_response}
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(conn, insert_data)
            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial
            attribute_set = {"_deserialized": True}

            new_is_source = is_source or row_data.source
            new_is_response = is_response or row_data.response
            if (new_is_source and not row_data.source) or (
                new_is_response and not row_data.response
            ):
                conn.execute(
                    sqla.update(table)
                    .where(table.c.serial == store_id)
                    .values(source=new_is_source, response=new_is_response)
                )
                attribute_set["_updated"] = True

        # return constructed object
        obj = redshift(
            store_id=store_id, z=z, is_source=is_source, is_response=is_response
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def read_table(
        conn,
        table,
        is_source: Optional[bool] = None,
        is_response: Optional[bool] = None,
    ):
        # query for all redshift records in the table
        query = sqla.select(
            table.c.serial,
            table.c.z,
            table.c.source,
            table.c.response,
        )

        if is_source is not None:
            query = query.filter(table.c.source == is_source)
        if is_response is not None:
            query = query.filter(table.c.response == is_response)

        rows = conn.execute(query.order_by(table.c.z))

        return [
            redshift(
                store_id=row.serial,
                z=row.z,
                is_source=row.source,
                is_response=row.response,
            )
            for row in rows
        ]
