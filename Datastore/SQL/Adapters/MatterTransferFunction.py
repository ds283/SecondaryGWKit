import sqlalchemy as sqla

from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_MatterTransferFunctionIntegration_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "version": True,
            "stepping": False,
            "timestamp": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
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
                sqla.Column(
                    "solver_serial",
                    sqla.Integer,
                    sqla.ForeignKey("IntegrationSolver.serial"),
                    nullable=False,
                ),
                sqla.Column("time", sqla.Float(64)),
                sqla.Column("steps", sqla.Integer),
            ],
        }

    def build_query(self, table, query):
        return query.filter(
            sqla.and_(
                table.c.wavenumber_serial == self._k.store_id,
                table.c.cosmology_type == self._cosmology.type_id,
                table.c.cosmology_serial == self._cosmology.store_id,
                table.c.label == self._label,
                table.c.atol_serial == self._atol.store_id,
                table.c.rtol_serial == self._rtol.store_id,
            )
        )


class sqla_MatterTransferFunctionValue_adapter:
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "defer_insert": True,
            "version": False,
            "timestamp": False,
            "stepping": False,
            "columns": [
                sqla.Column(
                    "integration_serial",
                    sqla.Integer,
                    sqla.ForeignKey("MatterTransferFunctionIntegration.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "value",
                    sqla.Float(64),
                ),
            ],
        }

    def build_query(self, table, query):
        query = (
            sqla.select(
                table.c.serial,
                table.c.integration_serial,
                table.c.z_serial,
                table.c.value,
                self._atol_table.c.log10_tol.label("log10_atol"),
                self._rtol_table.c.log10_tol.label("log10_rtol"),
                self._solver_table.c.label.label("solver_label"),
                self._solver_table.c.stepping.label("solver_stepping"),
                self._integration_table.c.timestamp,
                self._integration_table.c.version,
            )
            .select_from(
                table.join(
                    self._integration_table,
                    self._integration_table.c.serial == table.c.integration_serial,
                )
                .join(
                    self._solver_table,
                    self._solver_table.c.serial
                    == self._integration_table.c.solver_serial,
                )
                .join(
                    self._atol_table,
                    self._atol_table.c.serial == self._integration_table.c.atol_serial,
                )
                .join(
                    self._rtol_table,
                    self._rtol_table.c.serial == self._integration_table.c.rtol_serial,
                )
            )
            .filter(
                sqla.and_(
                    table.c.z_serial == self._z.store_id,
                    self._integration_table.c.wavenumber_serial == self._k.store_id,
                    self._integration_table.c.cosmology_type == self._cosmology.type_id,
                    self._integration_table.c.cosmology_serial
                    == self._cosmology.store_id,
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
