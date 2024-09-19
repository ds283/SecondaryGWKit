from math import fabs
from typing import List, Optional

import sqlalchemy as sqla
from sqlalchemy import and_

from ComputeTargets import (
    IntegrationSolver,
    MatterTransferFunctionIntegration,
    MatterTransferFunctionValue,
)
from CosmologyConcepts import redshift_array, redshift, wavenumber_exit_time
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from Datastore.SQL.ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_MatterTransferFunctionTagAssociation_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def generate_columns():
        return {
            "serial": False,
            "version": False,
            "stepping": False,
            "timestamp": True,
            "columns": [
                sqla.Column(
                    "integration_serial",
                    sqla.Integer,
                    sqla.ForeignKey("MatterTransferFunctionIntegration.serial"),
                    nullable=False,
                    primary_key=True,
                ),
                sqla.Column(
                    "tag_serial",
                    sqla.Integer,
                    sqla.ForeignKey("store_tag.serial"),
                    nullable=False,
                    primary_key=True,
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    def add_tag(
        conn, inserter, integration: MatterTransferFunctionIntegration, tag: store_tag
    ):
        inserter(
            conn,
            {
                "integration_serial": integration.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(
        conn, table, integration: MatterTransferFunctionIntegration, tag: store_tag
    ):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.integration_serial == integration.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


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
                    "wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
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
                sqla.Column(
                    "z_min_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "z_samples",
                    sqla.Integer,
                    nullable=False,
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("compute_steps", sqla.Integer),
                sqla.Column("RHS_evaluations", sqla.Integer),
                sqla.Column("mean_RHS_time", sqla.Float(64)),
                sqla.Column("max_RHS_time", sqla.Float(64)),
                sqla.Column("min_RHS_time", sqla.Float(64)),
                sqla.Column("validated", sqla.Boolean, default=False, nullable=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        solver_labels = payload.get("solver_labels")

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        k_exit: wavenumber_exit_time = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_init: redshift = payload["z_init"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")

        solver_table = tables["IntegrationSolver"]

        tag_table = tables["MatterTransferFunctionIntegration_tags"]

        # we treat z_sample as a target rather than a selection criterion
        # later, the actual set of z_sample stored with this integration is read in
        # and used to populate the z_sample field of the constructed object

        # note that we query only for validated data
        query = (
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.compute_steps,
                table.c.RHS_evaluations,
                table.c.mean_RHS_time,
                table.c.max_RHS_time,
                table.c.min_RHS_time,
                table.c.solver_serial,
                table.c.label,
                table.c.z_samples,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
            )
            .select_from(
                table.join(solver_table, solver_table.c.serial == table.c.solver_serial)
                .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
            )
            .filter(
                table.c.validated == True,
                table.c.wavenumber_exit_serial == k_exit.store_id,
                table.c.cosmology_type == cosmology.type_id,
                table.c.cosmology_serial == cosmology.store_id,
                table.c.z_init_serial == z_init.store_id,
                table.c.atol_serial == atol.store_id,
                table.c.rtol_serial == rtol.store_id,
            )
        )

        # require that the integration we search for has the specified list of tags
        count = 0
        for tag in tags:
            tag: store_tag
            tab = tag_table.alias(f"tag_{count}")
            count += 1
            query = query.join(
                tab,
                and_(
                    tab.c.integration_serial == table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        row_data = conn.execute(query).one_or_none()

        if row_data is None:
            # build and return an unpopulated object
            return MatterTransferFunctionIntegration(
                payload=None,
                solver_labels=solver_labels,
                cosmology=cosmology,
                label=label,
                k=k_exit,
                z_sample=z_sample,
                z_init=z_init,
                atol=atol,
                rtol=rtol,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label

        compute_time = row_data.compute_time
        compute_steps = row_data.compute_steps
        RHS_evaluations = row_data.RHS_evaluations
        mean_RHS_time = row_data.mean_RHS_time
        max_RHS_time = row_data.max_RHS_time
        min_RHS_time = row_data.min_RHS_time

        solver_label = row_data.solver_label
        solver_stepping = row_data.solver_stepping
        num_expected_samples = row_data.z_samples

        solver = IntegrationSolver(
            store_id=row_data.solver_serial,
            label=solver_label,
            stepping=solver_stepping,
        )

        # read out sample values associated with this integration
        value_table = tables["MatterTransferFunctionValue"]
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
            z_value = redshift(store_id=row.z_serial, z=row.z)
            z_points.append(z_value)
            values.append(
                MatterTransferFunctionValue(
                    store_id=row.serial, z=z_value, value=row.value
                )
            )

        z_sample = redshift_array(z_sample)
        if len(z_sample) != num_expected_samples:
            raise RuntimeError(
                f'Fewer z-samples than expected were recovered from the validated transfer function "{store_label}"'
            )

        return MatterTransferFunctionIntegration(
            payload={
                "store_id": store_id,
                "compute_time": compute_time,
                "compute_steps": compute_steps,
                "RHS_evaluations": RHS_evaluations,
                "mean_RHS_time": mean_RHS_time,
                "max_RHS_time": max_RHS_time,
                "min_RHS_time": min_RHS_time,
                "solver": solver,
                "values": values,
            },
            cosmology=cosmology,
            label=store_label,
            k=k,
            z_sample=z_sample,
            z_init=z_init,
            atol=atol,
            rtol=rtol,
            tags=tags,
        )

    @staticmethod
    def store(
        obj: MatterTransferFunctionIntegration,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        # now serialize the record of the integration; on the store step, we clear the validated flag.
        # Validation should happen in a separate step *after* the store transaction has completed.
        store_id = inserter(
            conn,
            {
                "label": obj.label,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "cosmology_type": obj.cosmology.type_id,
                "cosmology_serial": obj.cosmology.store_id,
                "atol_serial": obj._atol.store_id,
                "rtol_serial": obj._rtol.store_id,
                "solver_serial": obj.solver.store_id,
                "z_init_serial": obj.z_init.store_id,
                "z_min_serial": obj.z_sample.min.store_id,
                "z_samples": len(obj.z_sample),
                "compute_time": obj.compute_time,
                "compute_steps": obj.compute_steps,
                "RHS_evaluations": obj.RHS_evaluations,
                "mean_RHS_time": obj.mean_RHS_time,
                "max_RHS_time": obj.max_RHS_time,
                "min_RHS_time": obj.min_RHS_time,
                "validated": False,
            },
        )

        # set store_id on behalf of the MatterTransferFunctionIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["MatterTransferFunctionIntegration_tags"]
        for tag in obj.tags:
            sqla_MatterTransferFunctionTagAssociation_factory.add_tag(
                conn, tag_inserter, obj, tag
            )

        # now serialize the sampled output points
        value_inserter = inserters["MatterTransferFunctionValue"]
        for value in obj.values:
            value: MatterTransferFunctionValue
            value_id = value_inserter(
                conn,
                {
                    "integration_serial": store_id,
                    "z_serial": value.z.store_id,
                    "value": value.value,
                },
            )

            # set store_id on behalf of the MatterTransferFunctionValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: MatterTransferFunctionIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in MatterTransferFunctionIntegration corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["MatterTransferFunctionValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.integration_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: matter transfer function "{obj.label}" did not validate after serialization'
            )

        conn.execute(
            sqla.update(table)
            .where(table.c.serial == obj.store_id)
            .values(validated=validated)
        )

        return validated


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
                    nullable=False,
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
                    f"Stored matter transfer function value (integration={integration_serial}, z={z.z}) = {value} differs from target value = {target_value}"
                )

        return MatterTransferFunctionValue(store_id=store_id, z=z, value=value)
