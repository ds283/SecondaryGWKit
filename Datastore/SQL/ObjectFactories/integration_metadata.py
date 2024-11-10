import sqlalchemy as sqla

from AdaptiveLevin.integration_metadata import IntegrationSolver
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_STRING_LENGTH


class sqla_IntegrationSolver_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "stepping": "minimum",
            "timestamp": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label = payload["label"]
        stepping = payload["stepping"]
        if stepping < 0:
            stepping = 0

        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.and_(table.c.label == label, table.c.stepping >= stepping)
            )
        ).scalar()

        if store_id is None:
            insert_data = {"label": label, "stepping": stepping}
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(conn, insert_data)

            attribute_set = {"_new_insert": True}
        else:
            attribute_set = {"_deserialized": True}

        # return constructed object
        obj = IntegrationSolver(store_id=store_id, label=label, stepping=stepping)
        for key, value in attribute_set.items():
            setattr(obj, key, value)

        return obj
