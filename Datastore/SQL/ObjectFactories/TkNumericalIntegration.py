from operator import or_
from typing import List, Optional

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import (
    IntegrationSolver,
    TkNumericalIntegration,
    TkNumericalValue,
)
from CosmologyConcepts import redshift_array, redshift, wavenumber_exit_time
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_TkNumericalTagAssociation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("TkNumericalIntegration.serial"),
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
    def add_tag(conn, inserter, integration: TkNumericalIntegration, tag: store_tag):
        inserter(
            conn,
            {
                "integration_serial": integration.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, integration: TkNumericalIntegration, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.integration_serial == integration.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_TkNumericalIntegration_factory(SQLAFactoryBase):
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
                    "z_init_serial",
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

        solver_labels = payload.get("solver_labels")
        delta_logz = payload.get("delta_logz", None)

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        k_exit: wavenumber_exit_time = payload["k"]
        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_init: redshift = payload["z_init"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        tag_table = tables["MatterTransferFunctionIntegration_tags"]
        solver_table = tables["IntegrationSolver"]
        redshift_table = tables["redshift"]

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
                table.c.has_unresolved_osc,
                table.c.unresolved_z,
                table.c.unresolved_efolds_subh,
                table.c.init_efolds_suph,
                table.c.solver_serial,
                table.c.label,
                table.c.z_init_serial,
                redshift_table.c.z.label("z_init"),
                table.c.z_samples,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
            )
            .select_from(
                table.join(solver_table, solver_table.c.serial == table.c.solver_serial)
                .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
                .join(redshift_table, redshift_table.c.serial == table.c.z_init_serial)
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

        if z_init is not None:
            query = query.filter(
                table.c.z_init_serial == z_init.store_id,
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

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! Database error: multiple results found when querying for TkNumericalIntegration"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return TkNumericalIntegration(
                payload=None,
                solver_labels=solver_labels,
                cosmology=cosmology,
                label=label,
                k=k_exit,
                atol=atol,
                rtol=rtol,
                z_sample=z_sample,
                z_init=z_init,
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

        z_init_serial = row_data.z_init_serial
        z_init_value = row_data.z_init

        solver = IntegrationSolver(
            store_id=row_data.solver_serial,
            label=solver_label,
            stepping=solver_stepping,
        )

        # read out sample values associated with this integration
        value_table = tables["TkNumericalValue"]

        sample_rows = conn.execute(
            sqla.select(
                value_table.c.serial,
                value_table.c.z_serial,
                redshift_table.c.z,
                value_table.c.T,
                value_table.c.Tprime,
                value_table.c.analytic_T,
                value_table.c.analytic_Tprime,
            )
            .select_from(
                value_table.join(
                    redshift_table,
                    redshift_table.c.serial == value_table.c.z_serial,
                )
            )
            .filter(value_table.c.integration_serial == store_id)
            .order_by(redshift_table.c.z.desc())
        )

        z_points = []
        values = []
        for row in sample_rows:
            z_value = redshift(store_id=row.z_serial, z=row.z)
            z_points.append(z_value)
            values.append(
                TkNumericalValue(
                    store_id=row.serial,
                    z=z_value,
                    T=row.T,
                    Tprime=row.Tprime,
                    analytic_T=row.analytic_T,
                    analytic_Tprime=row.analytic_Tprime,
                )
            )
        imported_z_sample = redshift_array(z_points)

        if num_expected_samples is not None:
            if len(imported_z_sample) != num_expected_samples:
                raise RuntimeError(
                    f'Fewer z-samples than expected were recovered from the validated transfer function "{store_label}"'
                )

        obj = TkNumericalIntegration(
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
            cosmology=cosmology,
            label=store_label,
            k=k_exit,
            atol=atol,
            rtol=rtol,
            z_sample=imported_z_sample,
            z_init=redshift(store_id=z_init_serial, z=z_init_value),
            tags=tags,
            delta_logz=delta_logz,
        )
        obj._deserialized = True
        return obj

    @staticmethod
    def store(
        obj: TkNumericalIntegration,
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
                "z_samples": len(obj.values),
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

        # set store_id on behalf of the TkNumericalIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["MatterTransferFunctionIntegration_tags"]
        for tag in obj.tags:
            sqla_TkNumericalTagAssociation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sampled output points
        # TODO: this is undesirable, because there are two ways a TkNumericalValue can be serialized:
        #  directly, or using the logic here as part of a TkNumericalIntegration. We need to be careful to
        #  keep the logic in sync. It would be better to have a single serialization point for TkNumericalValue.
        value_inserter = inserters["TkNumericalValue"]
        for value in obj.values:
            value: TkNumericalValue
            value_id = value_inserter(
                conn,
                {
                    "integration_serial": store_id,
                    "z_serial": value.z.store_id,
                    "T": value.T,
                    "Tprime": value.Tprime,
                    "analytic_T": value.analytic_T,
                    "analytic_Tprime": value.analytic_Tprime,
                },
            )

            # set store_id on behalf of the TkNumericalValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: TkNumericalIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in TkNumericalIntegration corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["TkNumericalValue"]
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

    @staticmethod
    def validate_on_startup(conn, table, tables):
        # query the datastore for any integrations that are not validated

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["TkNumericalValue"]

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
                    redshift_table.c.z.label("z_init"),
                )
                .select_from(
                    table.join(
                        solver_table, solver_table.c.serial == table.c.solver_serial
                    )
                    .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                    .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
                    .join(
                        redshift_table, redshift_table.c.serial == table.c.z_init_serial
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
            ">> Matter transfer function",
            "     The following unvalidated integrations were detected in the datastore:",
        ]
        for integration in not_validated:
            msgs.append(
                f'       -- "{integration.label}" (store_id={integration.serial}) for k={integration.k_inv_Mpc:.5g}/Mpc and z_init={integration.z_init:.5g} (log10_atol={integration.log10_atol}, log10_rtol={integration.log10_rtol})'
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.integration_serial == integration.serial,
                )
            ).scalar()
            msgs.append(
                f"          contains {rows} z-sample values | expected={integration.z_samples}"
            )

        return msgs


class sqla_TkNumericalValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("TkNumericalIntegration.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "T",
                    sqla.Float(64),
                    nullable=False,
                ),
                sqla.Column("Tprime", sqla.Float(64), nullable=False),
                sqla.Column("analytic_T", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Tprime", sqla.Float(64), nullable=True),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        integration_serial = payload["integration_serial"]
        z = payload["z"]
        T = payload["T"]
        Tprime = payload["Tprime"]

        analytic_T = payload.get("analytic_T", None)
        analytic_Tprime = payload.get("analytic_Tprime", None)

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.T,
                    table.c.Tprime,
                    table.c.analytic_T,
                    table.c.analytic_Tprime,
                ).filter(
                    table.c.integration_serial == integration_serial,
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! Database error: multiple results found when querying for TkNumericalValue"
            )
            raise e

        if row_data is None:
            store_id = inserter(
                conn,
                {
                    "integration_serial": integration_serial,
                    "z_serial": z.store_id,
                    "T": T,
                    "Tprime": Tprime,
                    "analytic_T": analytic_T,
                    "analytic_Tprime": analytic_Tprime,
                },
            )
        else:
            store_id = row_data.serial
            analytic_T = row_data.analytic_T
            analytic_Tprime = row_data.analytic_Tprime

            if fabs(row_data.T - T) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored matter transfer function value (integration={integration_serial}, z={z.z}) = {row_data.T} differs from expected value = {T}"
                )
            if fabs(row_data.Tprime - Tprime) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored matter transfer function derivative (integration={integration_serial}, z={z.z}) = {row_data.Tprime} differs from expected value = {Tprime}"
                )

        return TkNumericalValue(
            store_id=store_id,
            z=z,
            T=T,
            Tprime=Tprime,
            analytic_T=analytic_T,
            analytic_Tprime=analytic_Tprime,
        )
