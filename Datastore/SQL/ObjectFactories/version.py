import sqlalchemy as sqla

from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from defaults import DEFAULT_STRING_LENGTH


class sqla_version_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
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
            store_id = inserter(conn, {"label": label})

        return store_tag(store_id=store_id, label=label)
