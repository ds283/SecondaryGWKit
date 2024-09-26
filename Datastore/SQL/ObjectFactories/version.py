import sqlalchemy as sqla

from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from defaults import DEFAULT_STRING_LENGTH


class sqla_version_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": False,
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

            attribute_set = {"_new_insert": True}
        else:
            attribute_set = {"_deserialized": True}

        obj = store_tag(store_id=store_id, label=label)
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj
