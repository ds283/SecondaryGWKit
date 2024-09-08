import sqlalchemy as sqla

from defaults import DEFAULT_FLOAT_PRECISION


class sqla_wavenumber_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "timestamp": True,
            "columns": [sqla.Column("k_inv_Mpc", sqla.Float(64))],
        }

    def build_query(self, table, query):
        return query.filter(
            sqla.func.abs(table.c.k_inv_Mpc - self.k_inv_Mpc) < DEFAULT_FLOAT_PRECISION
        )


class sqla_wavenumber_exit_time_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        # Does not set up a foreign key constraint for the cosmology object.
        # The problem is that this is polymorphic, because we have different implementations of the CosmologyBase concept.
        # Rather than try to deal with this using SQLAlchemy-level polymorphism, we handle the polymorphism ourselves
        # and just skip foreign key constraints here
        return {
            "version": True,
            "timestamp": True,
            "stepping": "minimum",
            "columns": [
                sqla.Column(
                    "wavenumber_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber.serial"),
                    nullable=False,
                ),
                sqla.Column("cosmology_type", sqla.Integer, nullable=False),
                sqla.Column("cosmology_serial", sqla.Integer, nullable=False),
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "rtol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column("time", sqla.Float(64)),
                sqla.Column("z_exit", sqla.Float(64)),
            ],
        }

    @property
    def stepping(self):
        # stepping 0: initial implementation using solve_ivp and an event handler to determine when ln(k/aH) = 0
        return 0

    def build_query(self, table, query):
        # query for an existing record that at least matches the specified tolerances
        # notice we have to replace the .select_from() specifier that has been pre-populated by the DataStore object
        # If we just have .select_from(BASE_TABLE), we cannot access any columns from the joined tables (at least using SQLite)
        # see: https://stackoverflow.com/questions/68137220/getting-columns-of-the-joined-table-using-sqlalchemy-core

        # order by descending values of abs and relative tolerances, so that we get the best computed value we hold
        query = (
            sqla.select(
                table.c.serial,
                table.c.version,
                table.c.stepping,
                table.c.timestamp,
                table.c.wavenumber_serial,
                table.c.cosmology_type,
                table.c.cosmology_serial,
                table.c.atol_serial,
                table.c.rtol_serial,
                table.c.time,
                self._atol_table.c.log10_tol.label("log10_atol"),
                self._rtol_table.c.log10_tol.label("log10_rtol"),
                table.c.z_exit,
            )
            .select_from(
                table.join(
                    self._atol_table, self._atol_table.c.serial == table.c.atol_serial
                ).join(
                    self._rtol_table, self._rtol_table.c.serial == table.c.rtol_serial
                )
            )
            .filter(
                sqla.and_(
                    table.c.wavenumber_serial == self.k.store_id,
                    table.c.cosmology_type == self.cosmology.type_id,
                    table.c.cosmology_serial == self.cosmology.store_id,
                    self._atol_table.c.log10_tol - self._target_atol.log10_tol
                    <= DEFAULT_FLOAT_PRECISION,
                    self._rtol_table.c.log10_tol - self._target_rtol.log10_tol
                    <= DEFAULT_FLOAT_PRECISION,
                )
            )
            .order_by(self._atol_table.c.log10_tol.desc())
            .order_by(self._rtol_table.c.log10_tol.desc())
        )
        return query
