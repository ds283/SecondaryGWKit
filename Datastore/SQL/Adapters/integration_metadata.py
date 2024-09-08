import sqlalchemy as sqla

from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


class sqla_IntegrationSolver_adapter:
    def __init__(self):
        pass

    @property
    def stepping(self) -> int:
        return self._stepping

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "stepping": "minimum",
            "timestamp": False,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
            ],
        }

    def build_query(self, table, query):
        return query.filter(table.c.label == self._label)
