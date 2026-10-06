import sqlalchemy as sqla

from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import QuadSourcePolicy
from config.defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_QuadSourcePolicy_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": True,
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

        # scalar_one_or_none() raises MultipleResultsFound on a duplicate rather than silently
        # serving the first row scalar() would have returned (prompts/datastore-integrity,
        # prompt 03, L3)
        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.func.abs(table.c.Levin_threshold - Levin_threshold)
                < DEFAULT_FLOAT_PRECISION,
                table.c.numeric_policy == numeric_policy,
            )
        ).scalar_one_or_none()

        if store_id is None:
            insert_data = {
                "label": label,
                "Levin_threshold": Levin_threshold,
                "numeric_policy": numeric_policy,
            }
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(conn, insert_data)

            attribute_set = {"_new_insert": True}
        else:
            attribute_set = {"_deserialized": True}

        obj = QuadSourcePolicy(
            store_id=store_id,
            Levin_threshold=Levin_threshold,
            numeric_policy=numeric_policy,
            label=label,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def read_table(conn, table, Levin_threshold, numeric_policy):
        """
        The rows that build() would match, without build()'s insert-on-miss (prompts/
        datastore-integrity, prompt 08). A reader that must find a QuadSourcePolicy but must not
        create one uses this: ShardedPool.read_table reads one shard and writes nothing, where
        object_get on a replicated class commits an in-flight record on the primary first.

        The match is build()'s own -- Levin_threshold within DEFAULT_FLOAT_PRECISION and
        numeric_policy equal -- so two readers cannot disagree about which row is "the" policy.
        Every match is returned, in serial order, so that the caller can refuse more than one.
        """
        rows = conn.execute(
            sqla.select(
                table.c.serial,
                table.c.label,
                table.c.Levin_threshold,
                table.c.numeric_policy,
            )
            .filter(
                sqla.func.abs(table.c.Levin_threshold - Levin_threshold)
                < DEFAULT_FLOAT_PRECISION,
                table.c.numeric_policy == numeric_policy,
            )
            .order_by(table.c.serial)
        )

        return [
            QuadSourcePolicy(
                store_id=row.serial,
                Levin_threshold=row.Levin_threshold,
                numeric_policy=row.numeric_policy,
                label=row.label,
            )
            for row in rows
        ]

    @staticmethod
    def inventory_spec():
        # the physical identity of a QuadSourcePolicy row (store-fingerprint prompt 02): the two
        # columns build() matches. The label is descriptive and is not identity
        from Datastore.store_inventory import InventorySpec

        return InventorySpec(
            leaves=("Levin_threshold", "numeric_policy"),
        )
