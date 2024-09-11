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
    def build(payload, engine, table, inserter, tables, inserters):
        label = payload["label"]
        stepping = payload["stepping"]
        if stepping < 0:
            stepping = 0

        with engine.begin() as conn:
            store_id = conn.execute(
                sqla.select(table.c.serial).filter(
                    sqla.and_(table.c.label == label, table.c.stepping >= stepping)
                )
            ).scalar()

        if store_id is None:
            with engine.begin() as conn:
                store_id = inserter(conn, {"label": label, "stepping": stepping})
                conn.commit()

        # return constructed object
        return IntegrationSolver(store_id=store_id, label=label, stepping=stepping)