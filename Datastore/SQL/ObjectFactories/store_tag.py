import sqlalchemy as sqla

from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from defaults import DEFAULT_STRING_LENGTH


class sqla_store_tag_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH))],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label = payload["label"]

        store_id = conn.execute(
            sqla.select(table.c.serial).filter(table.c.label == label)
        ).scalar()

        if store_id is None:
            insert_data = {"label": label}
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(conn, insert_data)

        return store_tag(store_id=store_id, label=label)
