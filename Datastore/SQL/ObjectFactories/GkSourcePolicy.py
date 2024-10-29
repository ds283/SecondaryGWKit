import sqlalchemy as sqla

from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import GkSourcePolicy
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_GkSourcePolicy_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH), nullable=True),
                sqla.Column("Levin_threshold", sqla.Float(64), nullable=False),
                sqla.Column(
                    "numeric_policy", sqla.String(DEFAULT_STRING_LENGTH), nullable=False
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        Levin_threshold = payload["Levin_threshold"]
        numeric_policy = payload["numeric_policy"]

        label = payload.get("label", None)

        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.func.abs(table.c.Levin_threshold - Levin_threshold)
                < DEFAULT_FLOAT_PRECISION,
                table.c.numeric_policy == numeric_policy,
            )
        ).scalar()

        if store_id is None:
            store_id = inserter(
                conn,
                {
                    "label": label,
                    "Levin_threshold": Levin_threshold,
                    "numeric_policy": numeric_policy,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            attribute_set = {"_deserialized": True}

        obj = GkSourcePolicy(
            store_id=store_id, numeric_policy=numeric_policy, label=label
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj
