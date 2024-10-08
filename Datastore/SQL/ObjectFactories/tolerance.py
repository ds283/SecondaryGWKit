import sqlalchemy as sqla
from math import log10

from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import tolerance
from defaults import DEFAULT_FLOAT_PRECISION


class sqla_tolerance_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("log10_tol", sqla.Float(64))],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        log10_tol = payload.get("log10_tol", None)
        if log10_tol is None:
            tol = payload.get("tol", None)
            if tol is None:
                raise KeyError("Missing expected arguments 'log10_tol' or 'tol")
            log10_tol = log10(tol)

        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.func.abs(table.c.log10_tol - log10_tol) < DEFAULT_FLOAT_PRECISION
            )
        ).scalar()

        if store_id is None:
            insert_data = {"log10_tol": log10_tol}
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(conn, insert_data)

            attribute_set = {"_new_insert": True}
        else:
            attribute_set = {"_deserialized": True}

        obj = tolerance(store_id=store_id, log10_tol=log10_tol)
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj
