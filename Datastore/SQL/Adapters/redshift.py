import sqlalchemy as sqla

from defaults import DEFAULT_FLOAT_PRECISION


class sqla_redshift_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("z", sqla.Float(64))],
        }

    def build_query(self, table, query):
        return query.filter(sqla.func.abs(table.c.z - self.z) < DEFAULT_FLOAT_PRECISION)
