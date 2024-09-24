from typing import List, Optional

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_, or_

from ComputeTargets import (
    TensorGreenFunctionIntegration,
    TensorGreenFunctionValue,
    IntegrationSolver,
)
from CosmologyConcepts import redshift_array, redshift, wavenumber_exit_time
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


class sqla_TensorGreenFunctionTagAssociation_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "serial": False,
            "version": False,
            "stepping": False,
            "timestamp": True,
            "columns": [
                sqla.Column(
                    "integration_serial",
                    sqla.Integer,
                    sqla.ForeignKey("TensorGreenFunctionIntegration.serial"),
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
        conn, inserter, integration: TensorGreenFunctionIntegration, tag: store_tag
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
        conn, table, integration: TensorGreenFunctionIntegration, tag: store_tag
    ):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.integration_serial == integration.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_TensorGreenFunctionIntegration_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": True,
            "stepping": False,
            "timestamp": True,
            "validate_on_startup": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column(
                    "wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("cosmology_type", sqla.Integer, index=True, nullable=False),
                sqla.Column(
                    "cosmology_serial", sqla.Integer, index=True, nullable=False
                ),
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "rtol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "solver_serial",
                    sqla.Integer,
                    sqla.ForeignKey("IntegrationSolver.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_source_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_min_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
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
                sqla.Column("has_unresolved_osc", sqla.Boolean),
                sqla.Column("unresolved_z", sqla.Float(64)),
                sqla.Column("unresolved_efolds_subh", sqla.Float(64)),
                sqla.Column("init_efolds_suph", sqla.Float(64)),
                sqla.Column("validated", sqla.Boolean, default=False, nullable=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        solver_labels = payload["solver_labels"]
        delta_logz = payload.get("delta_logz", None)

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        k_exit: wavenumber_exit_time = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_source: redshift = payload["z_source"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        tag_table = tables["TensorGreenFunctionIntegration_tags"]
        redshift_table = tables["redshift"]

        # we treat z_sample as a target rather than a selection criterion;
        # later, the actual set of z_sample stored with this integration is read in
        # and used to populate the z_sample field of the constructed object

        # notice that we query only for validated data
        query = (
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.compute_steps,
                table.c.RHS_evaluations,
                table.c.mean_RHS_time,
                table.c.max_RHS_time,
                table.c.min_RHS_time,
                table.c.has_unresolved_osc,
                table.c.unresolved_z,
                table.c.unresolved_efolds_subh,
                table.c.init_efolds_suph,
                table.c.solver_serial,
                table.c.label,
                table.c.z_source_serial,
                redshift_table.c.z.label("z_source"),
                table.c.z_samples,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
                atol_table.c.log10_tol.label("log10_atol"),
                rtol_table.c.log10_tol.label("log10_rtol"),
            )
            .select_from(
                table.join(solver_table, solver_table.c.serial == table.c.solver_serial)
                .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
                .join(
                    redshift_table, redshift_table.c.serial == table.c.z_source_serial
                )
            )
            .filter(
                table.c.validated == True,
                table.c.wavenumber_exit_serial == k_exit.store_id,
                table.c.cosmology_type == cosmology.type_id,
                table.c.cosmology_serial == cosmology.store_id,
                table.c.atol_serial == atol.store_id,
                table.c.rtol_serial == rtol.store_id,
            )
        )

        if z_source is not None:
            query = query.filter(
                table.c.z_source_serial == z_source.store_id,
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
            return TensorGreenFunctionIntegration(
                payload=None,
                solver_labels=solver_labels,
                label=label,
                k=k_exit,
                cosmology=cosmology,
                atol=atol,
                rtol=rtol,
                z_source=z_source,
                z_sample=z_sample,
                tags=tags,
                delta_logz=delta_logz,
            )

        store_id = row_data.serial
        store_label = row_data.label

        compute_time = row_data.compute_time
        compute_steps = row_data.compute_steps
        RHS_evaluations = row_data.RHS_evaluations
        mean_RHS_time = row_data.mean_RHS_time
        max_RHS_time = row_data.max_RHS_time
        min_RHS_time = row_data.min_RHS_time

        has_unresolved_osc = row_data.has_unresolved_osc
        unresolved_z = row_data.unresolved_z
        unresolved_efolds_subh = row_data.unresolved_efolds_subh

        init_efolds_suph = row_data.init_efolds_suph

        solver_label = row_data.solver_label
        solver_stepping = row_data.solver_stepping
        num_expected_samples = row_data.z_samples

        z_source_serial = row_data.z_source_serial
        z_source_value = row_data.z_source

        solver = IntegrationSolver(
            store_id=row_data.solver_serial,
            label=solver_label,
            stepping=solver_stepping,
        )

        # read out sample values associated with this integration
        value_table = tables["TensorGreenFunctionValue"]

        sample_rows = conn.execute(
            sqla.select(
                value_table.c.serial,
                value_table.c.z_serial,
                redshift_table.c.z,
                value_table.c.G,
                value_table.c.Gprime,
                value_table.c.analytic_G,
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
                TensorGreenFunctionValue(
                    store_id=row.serial,
                    z=z_value,
                    G=row.G,
                    Gprime=row.Gprime,
                    analytic_G=row.analytic_G,
                )
            )
        imported_z_sample = redshift_array(z_points)

        if num_expected_samples is not None:
            if len(imported_z_sample) != num_expected_samples:
                raise RuntimeError(
                    f'Fewer z-samples than expected were recovered from the validated tensor Green function "{store_label}"'
                )

        return TensorGreenFunctionIntegration(
            payload={
                "store_id": store_id,
                "compute_time": compute_time,
                "compute_steps": compute_steps,
                "RHS_evaluations": RHS_evaluations,
                "mean_RHS_time": mean_RHS_time,
                "max_RHS_time": max_RHS_time,
                "min_RHS_time": min_RHS_time,
                "has_unresolved_osc": has_unresolved_osc,
                "unresolved_z": unresolved_z,
                "unresolved_efolds_subh": unresolved_efolds_subh,
                "init_efolds_suph": init_efolds_suph,
                "solver": solver,
                "values": values,
            },
            solver_labels=solver_labels,
            k=k_exit,
            cosmology=cosmology,
            label=store_label,
            atol=atol,
            rtol=rtol,
            z_source=redshift(store_id=z_source_serial, z=z_source_value),
            z_sample=imported_z_sample,
            tags=tags,
            delta_logz=delta_logz,
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
                "z_source_serial": obj.z_source.store_id,
                "z_min_serial": obj.z_sample.min.store_id,
                "z_samples": len(obj.z_sample),
                "compute_time": obj.compute_time,
                "compute_steps": obj.compute_steps,
                "RHS_evaluations": obj.RHS_evaluations,
                "mean_RHS_time": obj.mean_RHS_time,
                "max_RHS_time": obj.max_RHS_time,
                "min_RHS_time": obj.min_RHS_time,
                "has_unresolved_osc": obj.has_unresolved_osc,
                "unresolved_z": obj.unresolved_z,
                "unresolved_efolds_subh": obj.unresolved_efolds_subh,
                "init_efolds_suph": obj.init_efolds_suph,
                "validated": False,
            },
        )

        # set store_id on behalf of the TensorGreenFunctionIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["TensorGreenFunctionIntegration_tags"]
        for tag in obj.tags:
            sqla_TensorGreenFunctionTagAssociation_factory.add_tag(
                conn, tag_inserter, obj, tag
            )

        # now serialize the sampled output points
        value_inserter = inserters["TensorGreenFunctionValue"]
        for value in obj.values:
            value: TensorGreenFunctionValue
            value_id = value_inserter(
                conn,
                {
                    "integration_serial": store_id,
                    "z_serial": value.z.store_id,
                    "G": value.G,
                    "Gprime": value.Gprime,
                    "analytic_G": value.analytic_G,
                },
            )

            # set store_id on behalf of the TensorGreenFunctionValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: TensorGreenFunctionIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in TensorGreenFunctionIntegration corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["TensorGreenFunctionValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.integration_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: tensor Green function "{obj.label}" did not validate after serialization'
            )

        conn.execute(
            sqla.update(table)
            .where(table.c.serial == obj.store_id)
            .values(validated=validated)
        )

        return validated

    @staticmethod
    def validate_on_startup(conn, table, tables):
        # query the datastore for any integrations that are not validated

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["TensorGreenFunctionValue"]

        # bake results into a list so that we can close this query; we are going to want to run
        # another one as we process the rows from this one
        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_samples,
                    wavenumber_table.c.k_inv_Mpc.label("k_inv_Mpc"),
                    solver_table.c.label.label("solver_label"),
                    atol_table.c.log10_tol.label("log10_atol"),
                    rtol_table.c.log10_tol.label("log10_rtol"),
                    redshift_table.c.z.label("z_source"),
                )
                .select_from(
                    table.join(
                        solver_table, solver_table.c.serial == table.c.solver_serial
                    )
                    .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                    .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
                    .join(
                        redshift_table,
                        redshift_table.c.serial == table.c.z_source_serial,
                    )
                    .join(
                        wavenumber_exit_table,
                        wavenumber_exit_table.c.serial
                        == table.c.wavenumber_exit_serial,
                    )
                    .join(
                        wavenumber_table,
                        wavenumber_table.c.serial
                        == wavenumber_exit_table.c.wavenumber_serial,
                    )
                )
                .filter(or_(table.c.validated == False, table.c.validated == None))
            )
        )

        if len(not_validated) == 0:
            return []

        msgs = [
            ">> Tensor Green function",
            "     The following unvalidated integrations were detected in the datastore:",
        ]
        for integration in not_validated:
            msgs.append(
                f'       -- "{integration.label}" (store_id={integration.serial}) for k={integration.k_inv_Mpc:.5g}/Mpc and z_source={integration.z_source:.5g} (log10_atol={integration.log10_atol}, log10_rtol={integration.log10_rtol})'
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.integration_serial == integration.serial,
                )
            ).scalar()
            msgs.append(
                f"          contains {rows} z-sample values expected={integration.z_samples}"
            )

        return msgs


class sqla_TensorGreenFunctionValue_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
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
                    "G",
                    sqla.Float(64),
                    nullable=False,
                ),
                sqla.Column(
                    "Gprime",
                    sqla.Float(64),
                    nullable=False,
                ),
                sqla.Column(
                    "analytic_G",
                    sqla.Float(64),
                    nullable=True,
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        integration_serial = payload["integration_serial"]
        z = payload["z"]
        G = payload["G"]
        Gprime = payload["Gprime"]
        analytic_G = payload.get("analytic_G", None)

        row_data = conn.execute(
            sqla.select(
                table.c.serial,
                table.c.G,
                table.c.Gprime,
                table.c.analytic_G,
            ).filter(
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
                    "G": G,
                    "Gprime": Gprime,
                    "analytic_G": analytic_G,
                },
            )
        else:
            store_id = row_data.serial
            analytic_G = row_data.analytic_G

            if fabs(row_data.G - G) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor Green function value (integration={integration_serial}, z={z.store_id}) = {row_data.G} differs from expected value = {G}"
                )
            if fabs(row_data.Gprime - Gprime) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor Green function derivative (integration={integration_serial}, z={z.store_id}) = {row_data.Gprime} differs from expected value = {Gprime}"
                )

        return TensorGreenFunctionValue(
            store_id=store_id, z=z, G=G, Gprime=Gprime, analytic_G=analytic_G
        )
