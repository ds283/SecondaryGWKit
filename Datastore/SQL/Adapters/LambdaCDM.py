import sqlalchemy as sqla

from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


class sqla_LambdaCDM_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [
                sqla.Column("name", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column("omega_m", sqla.Float(64)),
                sqla.Column("omega_cc", sqla.Float(64)),
                sqla.Column("h", sqla.Float(64)),
                sqla.Column("f_baryon", sqla.Float(64)),
                sqla.Column("T_CMB_Kelvin", sqla.Float(64)),
                sqla.Column("Neff", sqla.Float(64)),
            ],
        }

    def build_query(self, table, query):
        return query.filter(
            sqla.and_(
                sqla.func.abs(table.c.omega_m - self.omega_m) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.omega_cc - self.omega_cc)
                < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.h - self.h) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.f_baryon - self.f_baryon)
                < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.T_CMB_Kelvin - self.T_CMB_Kelvin)
                < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.Neff - self.Neff) < DEFAULT_FLOAT_PRECISION,
            )
        )
