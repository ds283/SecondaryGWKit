from math import fabs

import sqlalchemy as sqla

from ComputeTargets import (
    IntegrationSolver,
    MatterTransferFunctionIntegration,
    MatterTransferFunctionValue,
)
from CosmologyConcepts import tolerance, wavenumber, redshift_array, redshift
from CosmologyModels.base import CosmologyBase
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_MatterTransferFunctionIntegration_factory(SQLAFactoryBase):
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
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("compute_steps", sqla.Integer),
            ],
        }

    @staticmethod
    async def build(
        engine, conn, table, full_query, serial_query, tables, inserter, payload
    ):
        label: str = payload["label"]

        target_atol: tolerance = payload["atol"]
        target_rtol: tolerance = payload["rtol"]

        k: wavenumber = payload["k"]
        cosmology: CosmologyBase = payload["cosmology"]
        z_samples: redshift_array = payload["z_samples"]
        z_init: redshift = payload["z_init"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        solver_table = tables["IntegrationSolver"]

        # we treat z_samples
        ref = await conn.execute(
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.compute_steps,
                table.c.solver_serial,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
                atol_table.c.log10_tol.label("log10_atol"),
                rtol_table.c.log10_tol.label("log10_rtol"),
            )
            .select_from(
                table.join(solver_table, solver_table.c.serial == table.c.solver_serial)
            )
            .filter(
                table.c.wavenumber_serial == k.store_id,
                table.c.cosmology_type == cosmology.type_id,
                table.c.cosmology_serial == cosmology.store_id,
                table.c.label == label,
                table.c.z_init == z_init,
                table.c.atol_serial == target_atol.store_id,
                table.c.rtol_serial == target_rtol.store_id,
            )
        )
        row_data = ref.one_or_none()

        if row_data is not None:
            store_id = row_data["serial"]
            compute_time = row_data["compute_time"]
            compute_steps = row_data["compute_steps"]
            solver_label = row_data["solver_label"]
            solver_stepping = row_data["solver_stepping"]

            solver = IntegrationSolver(
                store_id=row_data["solver_serial"],
                label=solver_label,
                stepping=solver_stepping,
            )

            atol = tolerance(
                store_id=row_data["atol_serial"], log10_tol=row_data["log10_atol"]
            )
            rtol = tolerance(
                store_id=row_data["rtol_serial"], log10_tol=row_data["log10_rtol"]
            )

            # read out sample values associated with this integration
            value_table = tables["MatterTransferFunctionValue"]
            redshift_table = tables["redshift"]
            sample_rows = await conn.execute(
                sqla.select(
                    value_table.c.serial,
                    value_table.c.z_serial,
                    redshift_table.c.z,
                    value_table.c.value,
                )
                .select_from(
                    value_table.join(
                        redshift_table,
                        redshift_table.c.serial == value_table.c.z_serial,
                    )
                )
                .filter(value_table.c.integration_serial == store_id)
                .order_by(redshift_table.c.z)
            )

            z_points = []
            values = []
            for row in sample_rows:
                z_value = redshift(store_id=row["z_serial"], z=row["z"])
                z_points.append(z_value)
                values.append(
                    MatterTransferFunctionValue(
                        store_id=row["store_id"], z=z_value, value=row["value"]
                    )
                )

            z_samples = redshift_array(z_samples)

            return MatterTransferFunctionIntegration(
                payload={
                    "store_id": store_id,
                    "compute_time": compute_time,
                    "compute_steps": compute_steps,
                    "solver": solver,
                    "z_samples": z_samples,
                    "values": values,
                },
                k=k,
                cosmology=cosmology,
                z_init=z_init,
                atol=atol,
                rtol=rtol,
            )
        else:
            obj = MatterTransferFunctionIntegration(
                payload=None,
                k=k,
                cosmology=cosmology,
                z_init=z_init,
                z_samples=z_samples,
                atol=target_atol,
                rtol=target_rtol,
            )

            return obj


class sqla_MatterTransferFunctionValue_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
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

    @staticmethod
    async def build(
        engine, conn, table, full_query, serial_query, tables, inserter, payload
    ):
        integration_serial = payload["integration_serial"]
        z = payload["z"]
        target_value = payload["value"]

        ref = await conn.execute(
            full_query.filter(
                table.c.integration_serial == integration_serial,
                table.c.z_serial == z.store_id,
            )
        )
        row_data = ref.one_or_none()

        if row_data is None:
            store_id = await inserter(
                conn,
                {
                    "integration_serial": integration_serial,
                    "z_serial": z.store_id,
                    "value": target_value,
                },
            )
            value = target_value
        else:
            store_id = row_data["serial"]
            value = row_data["value"]

            if fabs(value - target_value) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored value (integration={integration_serial}, z={z.store_id}) = {value} differs from target value = {target_value}"
                )

        return MatterTransferFunctionValue(store_id=store_id, z=z, value=value)
