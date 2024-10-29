from typing import List, Optional

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    GkNumericalIntegration,
    GkNumericalValue,
    IntegrationSolver,
    IntegrationData,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import redshift_array, redshift, wavenumber_exit_time
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import tolerance, store_tag
from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


class sqla_GkNumericalTagAssociation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("GkNumericalIntegration.serial"),
                    index=True,
                    nullable=False,
                    primary_key=True,
                ),
                sqla.Column(
                    "tag_serial",
                    sqla.Integer,
                    sqla.ForeignKey("store_tag.serial"),
                    index=True,
                    nullable=False,
                    primary_key=True,
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    def add_tag(conn, inserter, integration: GkNumericalIntegration, tag: store_tag):
        inserter(
            conn,
            {
                "integration_serial": integration.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, integration: GkNumericalIntegration, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.integration_serial == integration.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_GkNumericalIntegration_factory(SQLAFactoryBase):
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
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH), nullable=True),
                sqla.Column(
                    "wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "model_serial",
                    sqla.Integer,
                    sqla.ForeignKey("BackgroundModel.serial"),
                    index=True,
                    nullable=False,
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
                sqla.Column("stop_deltaz_subh", sqla.Float(64)),
                sqla.Column("stop_G", sqla.Float(64)),
                sqla.Column("stop_Gprime", sqla.Float(64)),
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
        model_proxy: ModelProxy = payload["model"]
        z_sample: redshift_array = payload["z_sample"]
        z_source: redshift = payload.get("z_source", None)

        mode: str = payload.get("mode", None)

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        tag_table = tables["GkNumerical_tags"]
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
                table.c.stop_deltaz_subh,
                table.c.stop_G,
                table.c.stop_Gprime,
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
                table.c.model_serial == model_proxy.store_id,
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

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkNumericalIntegration.build(): multiple results found when querying for GkNumericalIntegration"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return GkNumericalIntegration(
                payload=None,
                solver_labels=solver_labels,
                label=label,
                k=k_exit,
                model=model_proxy,
                atol=atol,
                rtol=rtol,
                z_source=z_source,
                z_sample=z_sample,
                tags=tags,
                delta_logz=delta_logz,
                mode=mode,
            )

        store_id = row_data.serial
        store_label = row_data.label

        num_expected_samples = row_data.z_samples

        do_not_populate = payload.get("_do_not_populate", False)
        if not do_not_populate:
            # read out sample values associated with this integration
            value_table = tables["GkNumericalValue"]

            sample_rows = conn.execute(
                sqla.select(
                    value_table.c.serial,
                    value_table.c.z_serial,
                    redshift_table.c.z,
                    value_table.c.G,
                    value_table.c.Gprime,
                    value_table.c.analytic_G_rad,
                    value_table.c.analytic_Gprime_rad,
                    value_table.c.analytic_G_w,
                    value_table.c.analytic_Gprime_w,
                    value_table.c.omega_WKB_sq,
                    value_table.c.WKB_criterion,
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
                    GkNumericalValue(
                        store_id=row.serial,
                        z=z_value,
                        G=row.G,
                        Gprime=row.Gprime,
                        analytic_G_rad=row.analytic_G_rad,
                        analytic_Gprime_rad=row.analytic_Gprime_rad,
                        analytic_G_w=row.analytic_G_w,
                        analytic_Gprime_w=row.analytic_Gprime_w,
                        omega_WKB_sq=row.omega_WKB_sq,
                        WKB_criterion=row.WKB_criterion,
                    )
                )
            imported_z_sample = redshift_array(z_points)

            if num_expected_samples is not None:
                if len(imported_z_sample) != num_expected_samples:
                    raise RuntimeError(
                        f'Fewer z-samples than expected were recovered from the validated tensor Green function "{store_label}"'
                    )

            attributes = {"_deserialized": True}
        else:
            values = None
            imported_z_sample = None

            attributes = {"_do_not_populate": True, "_deserialized": True}

        obj = GkNumericalIntegration(
            payload={
                "store_id": store_id,
                "data": IntegrationData(
                    compute_time=row_data.compute_time,
                    compute_steps=row_data.compute_steps,
                    RHS_evaluations=row_data.RHS_evaluations,
                    mean_RHS_time=row_data.mean_RHS_time,
                    max_RHS_time=row_data.max_RHS_time,
                    min_RHS_time=row_data.min_RHS_time,
                ),
                "has_unresolved_osc": (row_data.has_unresolved_osc),
                "unresolved_z": (row_data.unresolved_z),
                "unresolved_efolds_subh": (row_data.unresolved_efolds_subh),
                "init_efolds_suph": (row_data.init_efolds_suph),
                "stop_deltaz_subh": (row_data.stop_deltaz_subh),
                "stop_G": (row_data.stop_G),
                "stop_Gprime": (row_data.stop_Gprime),
                "solver": (
                    IntegrationSolver(
                        store_id=row_data.solver_serial,
                        label=(row_data.solver_label),
                        stepping=(row_data.solver_stepping),
                    )
                ),
                "values": values,
            },
            solver_labels=solver_labels,
            k=k_exit,
            model=model_proxy,
            label=store_label,
            atol=atol,
            rtol=rtol,
            z_source=redshift(
                store_id=(row_data.z_source_serial), z=(row_data.z_source)
            ),
            z_sample=imported_z_sample,
            tags=tags,
            delta_logz=delta_logz,
            # no need to pass the mode argument, which is only needed/relevant for unpopulated GkNumericalIntegration instances
        )
        for key, value in attributes.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def store(
        obj: GkNumericalIntegration,
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
                "model_serial": obj.model_proxy.store_id,
                "atol_serial": obj._atol.store_id,
                "rtol_serial": obj._rtol.store_id,
                "solver_serial": obj.solver.store_id,
                "z_source_serial": obj.z_source.store_id,
                "z_min_serial": obj.z_sample.min.store_id,
                "z_samples": len(obj.values),
                "compute_time": obj.data.compute_time,
                "compute_steps": obj.data.compute_steps,
                "RHS_evaluations": obj.data.RHS_evaluations,
                "mean_RHS_time": obj.data.mean_RHS_time,
                "max_RHS_time": obj.data.max_RHS_time,
                "min_RHS_time": obj.data.min_RHS_time,
                "has_unresolved_osc": obj.has_unresolved_osc,
                "unresolved_z": obj.unresolved_z,
                "unresolved_efolds_subh": obj.unresolved_efolds_subh,
                "init_efolds_suph": obj.init_efolds_suph,
                "stop_deltaz_subh": obj.stop_deltaz_subh,
                "stop_G": obj.stop_G,
                "stop_Gprime": obj.stop_Gprime,
                "validated": False,
            },
        )

        # set store_id on behalf of the GkNumericalIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["GkNumerical_tags"]
        for tag in obj.tags:
            sqla_GkNumericalTagAssociation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sampled output points
        value_inserter = inserters["GkNumericalValue"]
        for value in obj.values:
            value: GkNumericalValue
            value_id = value_inserter(
                conn,
                {
                    "integration_serial": store_id,
                    "z_serial": value.z.store_id,
                    "G": value.G,
                    "Gprime": value.Gprime,
                    "analytic_G_rad": value.analytic_G_rad,
                    "analytic_Gprime_rad": value.analytic_Gprime_rad,
                    "analytic_G_w": value.analytic_G_w,
                    "analytic_Gprime_w": value.analytic_Gprime_w,
                    "omega_WKB_sq": value.omega_WKB_sq,
                    "WKB_criterion": value.WKB_criterion,
                },
            )

            # set store_id on behalf of the GkNumericalValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: GkNumericalIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in GkNumericalIntegration corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["GkNumericalValue"]
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
    def validate_on_startup(conn, table, tables, prune=False):
        # query the datastore for any integrations that are not validated

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["GkNumericalValue"]
        tags_table = tables["GkNumerical_tags"]

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
                f"          contains {rows} z-sample values | expected={integration.z_samples}"
            )

        if prune:
            invalid_serials = [nv.serial for nv in not_validated]
            try:
                conn.execute(
                    sqla.delete(value_table).where(
                        value_table.c.integration_serial.in_(invalid_serials)
                    )
                )
                conn.execute(
                    sqla.delete(tags_table).where(
                        tags_table.c.integration_serial.in_(invalid_serials)
                    )
                )
                conn.execute(
                    sqla.delete(table).where(table.c.serial.in_(invalid_serials))
                )
            except SQLAlchemyError:
                msgs.append(
                    f"!!        DATABASE ERROR encountered when pruning these values"
                )
                pass
            else:
                msgs.append(
                    f"     ** Note: these values have been pruned from the datastore."
                )

        return msgs


class sqla_GkNumericalValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("GkNumericalIntegration.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("G", sqla.Float(64), nullable=False),
                sqla.Column("Gprime", sqla.Float(64), nullable=False),
                sqla.Column("analytic_G_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Gprime_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_G_w", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Gprime_w", sqla.Float(64), nullable=True),
                sqla.Column("omega_WKB_sq", sqla.Float(64), nullable=True),
                sqla.Column("WKB_criterion", sqla.Float(64), nullable=True),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        integration_serial = payload.get("integration_serial", None)

        model_proxy: Optional[ModelProxy] = payload.get("model", None)
        k: Optional[wavenumber_exit_time] = payload.get("k", None)
        z_source: Optional[redshift] = payload.get("z_source", None)

        has_serial = all([integration_serial is not None])
        has_model = all([model_proxy is not None, k is not None, z_source is not None])

        if all([has_serial, has_model]):
            print(
                "## GkNumericalValue.build(): both an integration serial number and a (model, wavenumber, z_source) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "GkNumericalValue.build(): at least one of an integration serial number and a (model, wavenumber, z_source) set must be supplied."
            )

        if has_serial:
            return sqla_GkNumericalValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_GkNumericalValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        integration_serial: int = payload["integration_serial"]

        G: float = payload["G"]
        Gprime: float = payload["Gprime"]

        analytic_G_rad: Optional[float] = payload.get("analytic_G_rad", None)
        analytic_Gprime_rad: Optional[float] = payload.get("analytic_Gprime_rad", None)

        analytic_G_w: Optional[float] = payload.get("analytic_G_w", None)
        analytic_Gprime_w: Optional[float] = payload.get("analytic_Gprime_w", None)

        omega_WKB_sq: Optional[float] = payload.get("omega_WKB_sq", None)
        WKB_criterion: Optional[float] = payload.get("WKB_criterion", None)

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.G,
                    table.c.Gprime,
                    table.c.analytic_G_rad,
                    table.c.analytic_Gprime_rad,
                    table.c.analytic_G_w,
                    table.c.analytic_Gprime_w,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                ).filter(
                    table.c.integration_serial == integration_serial,
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkNumericalValue.build(): multiple results found when querying for GkNumericalValue"
            )
            raise e

        if row_data is None:
            store_id = inserter(
                conn,
                {
                    "integration_serial": integration_serial,
                    "z_serial": z.store_id,
                    "G": G,
                    "Gprime": Gprime,
                    "analytic_G_rad": analytic_G_rad,
                    "analytic_Gprime_rad": analytic_Gprime_rad,
                    "analytic_G_w": analytic_G_w,
                    "analytic_Gprime_w": analytic_Gprime_w,
                    "omega_WKB_sq": omega_WKB_sq,
                    "WKB_criterion": WKB_criterion,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial
            analytic_G_rad = row_data.analytic_G_rad
            analytic_Gprime_rad = row_data.analytic_Gprime_rad
            analytic_G_w = row_data.analytic_G_w
            analytic_Gprime_w = row_data.analytic_Gprime_w
            omega_WKB_sq = row_data.omega_WKB_sq
            WKB_criterion = row_data.WKB_criterion

            if G is not None and fabs(row_data.G - G) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"GkNumericalValue.build(): Stored tensor Green function value (integration={integration_serial}, z={z.store_id}) = {row_data.G} differs from expected value = {G}"
                )
            if (
                Gprime is not None
                and fabs(row_data.Gprime - Gprime) > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"GkNumericalValue.build(): Stored tensor Green function derivative (integration={integration_serial}, z={z.store_id}) = {row_data.Gprime} differs from expected value = {Gprime}"
                )

            G = row_data.G
            Gprime = row_data.Gprime

            attribute_set = {"_deserialized": True}

        obj = GkNumericalValue(
            store_id=store_id,
            z=z,
            G=G,
            Gprime=Gprime,
            analytic_G_rad=analytic_G_rad,
            analytic_Gprime_rad=analytic_Gprime_rad,
            analytic_G_w=analytic_G_w,
            analytic_Gprime_w=analytic_Gprime_w,
            omega_WKB_sq=omega_WKB_sq,
            WKB_criterion=WKB_criterion,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _build_impl_model(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        model_proxy: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]
        z_source: redshift = payload["z_source"]

        atol: Optional[tolerance] = payload.get("atol", None)
        rtol: Optional[tolerance] = payload.get("rtol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        integration_table = tables["GkNumericalIntegration"]

        try:
            # TODO: benchmarking suggests this query is indistinguishable from filtering directly on the
            #  GkNumericalIntegration serial number (if we only knew what it was), so this may be about
            #  as good as we can do. But it is still slow. For production use, should look at how this
            #  can be improved.
            integration_query = sqla.select(integration_table.c.serial).filter(
                integration_table.c.model_serial == model_proxy.store_id,
                integration_table.c.wavenumber_exit_serial == k.store_id,
                integration_table.c.z_source_serial == z_source.store_id,
                integration_table.c.validated == True,
            )

            if atol is not None:
                integration_query = integration_query.filter(
                    integration_table.c.atol_serial == atol.store_id
                )

            if rtol is not None:
                integration_query = integration_query.filter(
                    integration_table.c.rtol_serial == rtol.store_id
                )

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["GkNumerical_tags"].alias(f"tag_{count}")
                count += 1
                integration_query = integration_query.join(
                    tab,
                    and_(
                        tab.c.integration_serial == integration_table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            subquery = integration_query.subquery()

            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.G,
                    table.c.Gprime,
                    table.c.analytic_G_rad,
                    table.c.analytic_Gprime_rad,
                    table.c.analytic_G_w,
                    table.c.analytic_Gprime_w,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                )
                .select_from(
                    subquery.join(
                        table, table.c.integration_serial == subquery.c.serial
                    )
                )
                .filter(
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkNumericalValue.build(): multiple results found when querying for GkNumericalValue"
            )
            raise e

        if row_data is None:
            # return empty object
            obj = GkNumericalValue(store_id=None, z=z, G=None, Gprime=None)
            obj._k_exit = k
            obj._z_source = z_source
            return obj

        obj = GkNumericalValue(
            store_id=row_data.serial,
            z=z,
            G=row_data.G,
            Gprime=row_data.Gprime,
            analytic_G_rad=row_data.analytic_G_rad,
            analytic_Gprime_rad=row_data.analytic_Gprime_rad,
            analytic_G_w=row_data.analytic_G_w,
            analytic_Gprime_w=row_data.analytic_Gprime_w,
            omega_WKB_sq=row_data.omega_WKB_sq,
            WKB_criterion=row_data.WKB_criterion,
        )
        obj._deserialized = True
        obj._k_exit = k
        obj._z_source = z_source
        return obj

    @staticmethod
    def read_batch(payload, conn, table, tables):
        model_proxy: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        atol: Optional[tolerance] = payload.get("atol", None)
        rtol: Optional[tolerance] = payload.get("rtol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        z_response: Optional[redshift] = payload.get("z", None)
        z_source: Optional[redshift] = payload.get("z_source", None)

        integration_table = tables["GkNumericalIntegration"]
        redshift_table = tables["redshift"]

        integration_query = (
            sqla.select(
                integration_table.c.serial,
                integration_table.c.z_source_serial,
                redshift_table.c.z.label("z_source"),
            )
            .filter(
                integration_table.c.model_serial == model_proxy.store_id,
                integration_table.c.wavenumber_exit_serial == k.store_id,
                integration_table.c.validated == True,
            )
            .select_from(
                integration_table.join(
                    redshift_table,
                    redshift_table.c.serial == integration_table.c.z_source_serial,
                )
            )
        )

        if z_source is not None:
            integration_query = integration_query.filter(
                integration_table.c.z_source_serial == z_source.store_id
            )

        if atol is not None:
            integration_query = integration_query.filter(
                integration_table.c.atol_serial == atol.store_id
            )

        if rtol is not None:
            integration_query = integration_query.filter(
                integration_table.c.rtol_serial == rtol.store_id
            )

        count = 0
        for tag in tags:
            tag: store_tag
            tab = tables["GkNumerical_tags"].alias(f"tag_{count}")
            count += 1
            integration_query = integration_query.join(
                tab,
                and_(
                    tab.c.integration_serial == integration_table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        subquery = integration_query.subquery()

        row_query = sqla.select(
            table.c.serial,
            table.c.G,
            table.c.Gprime,
            table.c.analytic_G_rad,
            table.c.analytic_Gprime_rad,
            table.c.analytic_G_w,
            table.c.analytic_Gprime_w,
            table.c.omega_WKB_sq,
            table.c.WKB_criterion,
            subquery.c.z_source_serial,
            subquery.c.z_source,
            redshift_table.c.z.label("z_response"),
            table.c.z_serial.label("z_response_serial"),
        ).select_from(
            subquery.join(table, table.c.integration_serial == subquery.c.serial).join(
                redshift_table, redshift_table.c.serial == table.c.z_serial
            )
        )

        if z_response is not None:
            row_query = row_query.filter(table.c.z_serial == z_response.store_id)

        row_data = conn.execute(row_query)

        def make_obj(row):
            obj = GkNumericalValue(
                store_id=row.serial,
                z=redshift(store_id=row.z_response_serial, z=row.z_response),
                G=row.G,
                Gprime=row.Gprime,
                analytic_G_rad=row.analytic_G_rad,
                analytic_Gprime_rad=row.analytic_Gprime_rad,
                analytic_G_w=row.analytic_G_w,
                analytic_Gprime_w=row.analytic_Gprime_w,
                omega_WKB_sq=row.omega_WKB_sq,
                WKB_criterion=row.WKB_criterion,
            )
            obj._deserialized = True
            obj._k_exit = k
            obj._z_source = redshift(store_id=row.z_source_serial, z=row.z_source)

            return obj

        objects = [make_obj(row) for row in row_data]
        return objects
