import sqlalchemy as sqla

from datastorekit.SQL.factory_base import SQLAFactoryBase
from Quadrature.integration_metadata import IntegrationSolver
from config.defaults import DEFAULT_STRING_LENGTH


class sqla_IntegrationSolver_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "stepping": "exact",
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

        # exact match on stepping (prompts/datastore-integrity, prompt 03, L2): the requested
        # stepping is the row's identity alongside the label, not a floor an existing row must
        # clear -- and scalar_one_or_none() raises MultipleResultsFound on a duplicate rather than
        # silently serving the first row scalar() would have returned
        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.and_(table.c.label == label, table.c.stepping == stepping)
            )
        ).scalar_one_or_none()

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

    @staticmethod
    def inventory_spec():
        # the physical identity of an IntegrationSolver row (store-fingerprint prompt 02): its label,
        # which is this class's identity, and the stepping build() filters on
        from datastorekit.store_inventory import InventorySpec

        return InventorySpec(leaves=("label", "stepping"))
