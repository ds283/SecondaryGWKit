from math import fabs

import sqlalchemy as sqla

from ComputeTargets import (
    TensorGreenFunctionIntegration,
    TensorGreenFunctionValue,
    TensorGreenFunctionContainer,
    IntegrationSolver,
)
from CosmologyConcepts import wavenumber, redshift_array, redshift
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import tolerance
from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH
from .integration_metadata import sqla_IntegrationSolver_factory


class sqla_TensorGreenFunctionIntegration_factory(SQLAFactoryBase):
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
                    "z_source_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    nullable=False,
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("compute_steps", sqla.Integer),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label: str = payload["label"]

        target_atol: tolerance = payload["atol"]
        target_rtol: tolerance = payload["rtol"]

        k: wavenumber = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_source: redshift = payload["z_source"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        solver_table = tables["IntegrationSolver"]

        # we treat z_sample as a target rather than a selection criterion;
        # later, the actual set of z_sample stored with this integration is read in
        # and used to populate the z_sample field of the constructed object
        row_data = conn.execute(
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
                table.c.z_source_serial == z_source.store_id,
                table.c.atol_serial == target_atol.store_id,
                table.c.rtol_serial == target_rtol.store_id,
            )
        ).one_or_none()

        if row_data is None:
            # build and return an unpopulated object
            return TensorGreenFunctionIntegration(
                payload=None,
                label=label,
                k=k,
                cosmology=cosmology,
                z_source=z_source,
                z_sample=z_sample,
                atol=target_atol,
                rtol=target_rtol,
            )

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

        atol = tolerance(store_id=row_data.atol_serial, log10_tol=row_data.log10_atol)
        rtol = tolerance(store_id=row_data.rtol_serial, log10_tol=row_data.log10_rtol)

        # read out sample values associated with this integration
        value_table = tables["TensorGreenFunctionValue"]
        redshift_table = tables["redshift"]

        sample_rows = conn.execute(
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
                TensorGreenFunctionValue(
                    store_id=row["store_id"], z=z_value, value=row["value"]
                )
            )

        z_sample = redshift_array(z_sample)

        return TensorGreenFunctionIntegration(
            payload={
                "store_id": store_id,
                "compute_time": compute_time,
                "compute_steps": compute_steps,
                "solver": solver,
                "values": values,
            },
            k=k,
            cosmology=cosmology,
            z_source=z_source,
            z_sample=z_sample,
            atol=atol,
            rtol=rtol,
        )

    @staticmethod
    def store(
        obj: TensorGreenFunctionIntegration,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        # first query for the solver ID, which is needed to serialize the main integration record
        solver_table = tables["IntegrationSolver"]
        solver_inserter = inserters["IntegrationSolver"]

        solver = sqla_IntegrationSolver_factory.build(
            {"label": obj._solver.label, "stepping": obj._solver.stepping},
            conn,
            solver_table,
            solver_inserter,
            tables,
            inserters,
        )

        # replace the store_id in our IntegrationSolver instance with the version obtained from the
        # datastore
        obj._solver = solver.store_id

        # now serialize the record of the integration
        store_id = inserter(
            conn,
            {
                "label": obj.label,
                "wavenumber_serial": obj.k.store_id,
                "cosmology_type": obj.cosmology.type_id,
                "cosmology_serial": obj.cosmology.store_id,
                "atol_serial": obj._atol.store_id,
                "rtol_serial": obj._rtol.store_id,
                "solver_serial": solver.store_id,
                "z_source_serial": obj.z_source.store_id,
                "compute_time": obj.compute_time,
                "compute_steps": obj.compute_steps,
            },
        )

        # set store_id on behalf of the TensorGreenFunctionIntegration instance
        obj._my_id = store_id

        # now serialize the sampled output points
        value_inserter = inserters["TensorGreenFunctionValue"]
        for value in obj.values:
            value: TensorGreenFunctionValue
            value_id = value_inserter(
                conn,
                {
                    "integration_serial": store_id,
                    "z_serial": value.z_serial,
                    "value": value.value,
                },
            )

            # set store_id on behalf of the TensorGreenFunctionValue instance
            value._my_id = value_id

        return obj


class sqla_TensorGreenFunctionValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("TensorGreenFunctionIntegration.serial"),
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
    def build(payload, conn, table, inserter, tables, inserters):
        integration_serial = payload["integration_serial"]
        z = payload["z"]
        target_value = payload["value"]

        row_data = conn.execute(
            sqla.select(table.c.serial, table.c.value).filter(
                table.c.integration_serial == integration_serial,
                table.c.z_serial == z.store_id,
            )
        ).one_or_none()

        if row_data is None:
            store_id = inserter(
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

        return TensorGreenFunctionValue(store_id=store_id, z=z, value=value)


class sqla_TensorGreenFunctionContainer_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return None

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        target_atol: tolerance = payload["atol"]
        target_rtol: tolerance = payload["rtol"]

        k: wavenumber = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_source: redshift = payload["z_source"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        values_table = tables["TensorGreenFunctionValue"].alias("value")
        integration_table = tables["TensorGreenFunctionIntegration"].alias(
            "integration"
        )
        redshift_table = tables["redshift"].alias("redshift")
        solver_table = tables["IntegrationSolver"].alias("solver")

        z_sample_serials = [z.store_id for z in z_sample]

        # query for stored samples of the tensor Green's function at the required
        # response redshift samples, belonging to an integration with the right cosmology,
        # source redshift, k-value, and tolerances.
        # We group values by their z serial number and then order by log10 of each tolerance.
        # Then we pick the first value from each group. This gets us the value computed using
        # the best tolerance, no matter which integration it came from, so it is possible
        # that the outputs here come from a heterogeneous group of integrations
        row_data = conn.execute(
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
                integration_table.c.z_source_serial == z_source.store_id,
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
            value = TensorGreenFunctionValue(
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

        obj = TensorGreenFunctionContainer(
            payload={"values": value_set},
            cosmology=cosmology,
            k=k,
            z_source=z_source,
            z_sample=z_sample,
            target_atol=target_atol,
            target_rtol=target_rtol,
        )

        return obj
