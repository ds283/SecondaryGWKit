from math import fabs

import sqlalchemy as sqla

from ComputeTargets import (
    IntegrationSolver,
    MatterTransferFunctionIntegration,
    MatterTransferFunctionValue,
    MatterTransferFunctionContainer,
)
from CosmologyConcepts import tolerance, wavenumber, redshift_array, redshift
from CosmologyModels.base import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from Datastore.SQL.ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
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
                sqla.Column(
                    "z_init_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    nullable=False,
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("compute_steps", sqla.Integer),
            ],
        }

    @staticmethod
    async def build(
        payload,
        engine,
        conn,
        table,
        full_query,
        serial_query,
        inserter,
        tables,
        inserters,
    ):
        label: str = payload["label"]

        target_atol: tolerance = payload["atol"]
        target_rtol: tolerance = payload["rtol"]

        k: wavenumber = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_init: redshift = payload["z_init"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        solver_table = tables["IntegrationSolver"]

        # we treat z_sample as a target rather than a selection criterion;
        # later, the actual set of z_sample stored with this integration is read in
        # and used to populate the z_sample field of the constructed object
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
                .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
            )
            .filter(
                table.c.wavenumber_serial == k.store_id,
                table.c.cosmology_type == cosmology.type_id,
                table.c.cosmology_serial == cosmology.store_id,
                table.c.label == label,
                table.c.z_init_serial == z_init.store_id,
                table.c.atol_serial == target_atol.store_id,
                table.c.rtol_serial == target_rtol.store_id,
            )
        )
        row_data = ref.one_or_none()

        if row_data is not None:
            store_id = row_data.serial
            compute_time = row_data.compute_time
            compute_steps = row_data.compute_steps
            solver_label = row_data.solver_label
            solver_stepping = row_data.solver_stepping

            solver = IntegrationSolver(
                store_id=row_data.solver_serial,
                label=solver_label,
                stepping=solver_stepping,
            )

            atol = tolerance(
                store_id=row_data.atol_serial, log10_tol=row_data.log10_atol
            )
            rtol = tolerance(
                store_id=row_data.rtol_serial, log10_tol=row_data.log10_rtol
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

            z_sample = redshift_array(z_sample)

            return MatterTransferFunctionIntegration(
                payload={
                    "store_id": store_id,
                    "compute_time": compute_time,
                    "compute_steps": compute_steps,
                    "solver": solver,
                    "z_sample": z_sample,
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
                label=label,
                k=k,
                cosmology=cosmology,
                z_init=z_init,
                z_sample=z_sample,
                atol=target_atol,
                rtol=target_rtol,
            )

            # asynchronously perform the computation and await the result
            # (this yields control so that other actor services can run while the result completes)
            data = await obj.compute()

            # read the result of the computation, and use to populate the constructed object
            res = await obj.store()
            if res is None:
                raise RuntimeError("compute() did not generate a running future")
            if res is False:
                raise RuntimeError(
                    "await compute() returned, but the future was not resolved"
                )

            # first query for the solver ID, which is needed to serialize the main integration record
            solver_table = tables["IntegrationSolver"]
            solver_serial_query = sqla.select(solver_table.c.serial)
            solver_inserter = inserters["IntegrationSolver"]

            solver = await sqla_IntegrationSolver_factory.build(
                {"label": obj._solver.label, "stepping": obj._solver.stepping},
                engine,
                conn,
                solver_table,
                None,
                solver_serial_query,
                solver_inserter,
                tables,
                inserters,
            )
            obj._solver._my_id = solver.store_id

            # now serialize the record of the integration
            store_id = await inserter(
                conn,
                {
                    "label": label,
                    "wavenumber_serial": k.store_id,
                    "cosmology_type": cosmology.type_id,
                    "cosmology_serial": cosmology.store_id,
                    "atol_serial": target_atol.store_id,
                    "rtol_serial": target_rtol.store_id,
                    "solver_serial": solver.store_id,
                    "z_init_serial": z_init.store_id,
                    "compute_time": obj.compute_time,
                    "compute_steps": obj.compute_steps,
                },
            )

            # set store_id on behalf of the MatterTransferFunctionIntegration instance
            obj._my_id = store_id

            # now serialize the sampled output points
            value_inserter = inserters["MatterTransferFunctionValue"]
            for value in obj.values:
                value: MatterTransferFunctionValue
                value_id = await value_inserter(
                    conn,
                    {
                        "integration_serial": store_id,
                        "z_serial": value.z_serial,
                        "value": value.value,
                    },
                )

                # set store_id on behalf of the MatterTransferFunctionValue instance
                value._my_id = value_id

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
        payload,
        engine,
        conn,
        table,
        full_query,
        serial_query,
        inserter,
        tables,
        inserters,
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
            store_id = row_data.serial
            value = row_data.value

            if fabs(value - target_value) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored value (integration={integration_serial}, z={z.store_id}) = {value} differs from target value = {target_value}"
                )

        return MatterTransferFunctionValue(store_id=store_id, z=z, value=value)


class sqla_MatterTransferFunctionContainer_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return None

    @staticmethod
    async def build(
        payload,
        engine,
        conn,
        table,
        full_query,
        serial_query,
        inserter,
        tables,
        inserters,
    ):
        target_atol: tolerance = payload["atol"]
        target_rtol: tolerance = payload["rtol"]

        k: wavenumber = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_init: redshift = payload["z_init"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        values_table = tables["MatterTransferFunctionValue"].alias("value")
        integration_table = tables["MatterTransferFunctionIntegration"].alias(
            "integration"
        )
        redshift_table = tables["redshift"].alias("redshift")
        solver_table = tables["IntegrationSolver"].alias("solver")

        z_sample_serials = [z.store_id for z in z_sample]

        # query for stored samples of the matter transfer function at the required
        # redshift sample points, belonging to an integration with the right cosmology,
        # k-value, and toleranecs.
        # We group values by their z serial number and then order by log10 of each tolerance.
        # Then we pick the first value from each group. This gets us the value computed using
        # the best tolerance, no matter which integration it came from, so it is possible
        # that the outputs here come from a heterogeneous group of integrations
        row_data = await conn.execute(
            sqla.select(
                values_table.c.serial,
                values_table.c.integration_serial,
                values_table.c.z_serial,
                redshift_table.c.z,
                sqla.func.first_value(values_table.c.value)
                .over(
                    partition_by=values_table.c.z_serial,
                    order_by=[atol_table.c.log10_tol, rtol_table.c.log10_tol],
                )
                .label("value"),
                integration_table.c.solver_serial,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
                atol_table.c.log10_tol.label("log10_atol"),
                rtol_table.c.log10_tol.label("log10_rtol"),
            )
            .select_from(
                values_table.join(
                    integration_table,
                    integration_table.c.serial == values_table.c.integration_serial,
                )
                .join(
                    atol_table,
                    atol_table.c.serial == integration_table.c.atol_serial,
                )
                .join(
                    rtol_table,
                    rtol_table.c.serial == integration_table.c.rtol_serial,
                )
                .join(
                    redshift_table,
                    redshift_table.c.serial == values_table.c.z_serial,
                )
                .join(
                    solver_table,
                    solver_table.c.serial == integration_table.c.solver_serial,
                )
            )
            .filter(
                values_table.c.z_serial.in_(z_sample_serials),
                integration_table.c.wavenumber_serial == k.store_id,
                integration_table.c.cosmology_type == cosmology.type_id,
                integration_table.c.cosmology_serial == cosmology.store_id,
                integration_table.c.z_init_serial == z_init.store_id,
                atol_table.c.log10_tol - target_atol.log10_tol
                <= DEFAULT_FLOAT_PRECISION,
                rtol_table.c.log10_tol - target_rtol.log10_tol
                <= DEFAULT_FLOAT_PRECISION,
            )
            .order_by(redshift_table.c.z.desc())
        )

        value_set = {}
        for row in row_data:
            z_value = redshift(store_id=row.z_serial, z=row.z)
            value = MatterTransferFunctionValue(
                store_id=row.serial, z=z_value, value=row.value
            )

            # embed information about provenance
            value._integration_serial = row.integration_serial
            value._log10_atol = row.log10_atol
            value._log10_rtol = row.log10_rtol
            value._atol = pow(10.0, value._log10_atol)
            value._rtol = pow(10.0, value._log10_rtol)
            value._solver_serial = row.solver_serial
            value._solver_label = row.solver_label
            value._solver_stepping = row.solver_stepping

            value_set[z_value.store_id] = value

        obj = MatterTransferFunctionContainer(
            payload={"values": value_set},
            cosmology=cosmology,
            k=k,
            z_init=z_init,
            z_sample=z_sample,
            target_atol=target_atol,
            target_rtol=target_rtol,
        )

        return obj
