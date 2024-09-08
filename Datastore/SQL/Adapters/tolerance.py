import sqlalchemy as sqla

from defaults import DEFAULT_FLOAT_PRECISION


class sqla_tolerance_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("log10_tol", sqla.Float(64))],
        }

    def build_query(self, table, query):
        return query.filter(
            sqla.func.abs(table.c.log10_tol - self.log10_tol) < DEFAULT_FLOAT_PRECISION
        )
