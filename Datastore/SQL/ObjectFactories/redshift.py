from typing import Optional

import sqlalchemy as sqla
from sqlalchemy import and_

from ComputeTargets import ModelProxy, BackgroundModel
from CosmologyConcepts import redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from config.defaults import DEFAULT_REDSHIFT_RELATIVE_PRECISION


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
            # the pool's check at open may set these from the controlling shard after a get
            "monotone_flags": ("source", "response"),
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]
        is_source = payload["is_source"]
        is_response = payload["is_response"]

        # query for this redshift in the datastore. z = 0 cannot use the relative match below: it
        # divides by z, and SQLite's 0/0 is NULL, which matches nothing, so every call at z = 0
        # would insert a fresh row (prompts/datastore-integrity, prompt 03, L4, audit §3.9).
        # z == 0 is matched by equality instead; every other z keeps the relative match unchanged.
        z_filter = (
            table.c.z == 0.0
            if z == 0.0
            else sqla.func.abs((table.c.z - z) / z)
            < DEFAULT_REDSHIFT_RELATIVE_PRECISION
        )
        query = sqla.select(
            table.c.serial,
            table.c.source,
            table.c.response,
        ).filter(z_filter)
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

        # return the constructed object
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
        tables,
        is_source: Optional[bool] = None,
        is_response: Optional[bool] = None,
        model_proxy: Optional[ModelProxy] = None,
    ):
        # query for all redshift records in the table
        query = sqla.select(
            table.c.serial,
            table.c.z,
            table.c.source,
            table.c.response,
        )

        if model_proxy is not None:
            bgv_table = tables["BackgroundModelValue"]
            model: BackgroundModel = model_proxy.get()

            # force a join to the BackgroundModelValue table, on redshift serial and matching model number
            # the net result will filter out only those redshifts that are used for the intended model
            query = query.join(
                bgv_table,
                and_(
                    bgv_table.c.model_serial == model.store_id,
                    bgv_table.c.z_serial == table.c.serial,
                ),
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

    @staticmethod
    def inventory_records(conn, table, tables, context):
        # the physical identity of a redshift row (store-fingerprint prompt 02): z as stored, which is
        # what build() matches (relative 1e-7). The source/response flags accumulate by OR and are
        # not identity
        from Datastore.store_inventory import read_records

        return read_records(conn, table, tables, context, leaves=("z",))
