import sqlalchemy as sqla

from ComputeTargets import IntegrationSolver
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_STRING_LENGTH


class sqla_IntegrationSolver_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "stepping": "minimum",
            "timestamp": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
            ],
        }

    @staticmethod
    async def build(
        engine, conn, table, full_query, serial_query, tables, inserter, payload
    ):
        label = payload["label"]
        stepping = payload["stepping"]
        if stepping < 0:
            stepping = 0

        ref = await conn.execute(
            serial_query.filter(
                sqla.and_(table.c.label == label, table.c.stepping >= stepping)
            )
        )
        store_id = ref.scalar()

        if store_id is None:
            store_id = await inserter(conn, {"label": label, "stepping": stepping})

        # return constructed object
        return IntegrationSolver(store_id=store_id, label=label, stepping=stepping)
