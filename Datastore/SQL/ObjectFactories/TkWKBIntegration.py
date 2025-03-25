import json
from math import fabs
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    TkWKBIntegration,
    TkWKBValue,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift_array, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_TkWKBTagAssociation_factory(SQLAFactoryBase):
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
                    "wkb_serial",
                    sqla.Integer,
                    sqla.ForeignKey("TkWKBIntegration.serial"),
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
    def add_tag(conn, inserter, wkb: TkWKBIntegration, tag: store_tag):
        inserter(
            conn,
            {
                "wkb_serial": wkb.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, wkb: TkWKBIntegration, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.wkb_serial == wkb.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_TkWKBIntegration_factory(SQLAFactoryBase):
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
                    "phase_solver_serial",
                    sqla.Integer,
                    sqla.ForeignKey("IntegrationSolver.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "friction_solver_serial",
                    sqla.Integer,
                    sqla.ForeignKey("IntegrationSolver.serial"),
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
                sqla.Column("z_samples", sqla.Integer, nullable=False),
                sqla.Column("z_init", sqla.Float(64), nullable=False),
                sqla.Column("sin_coeff", sqla.Float(64), nullable=False),
                sqla.Column("cos_coeff", sqla.Float(64), nullable=False),
                sqla.Column("T_init", sqla.Float(64), nullable=False),
                sqla.Column("Tprime_init", sqla.Float(64), nullable=False),
                sqla.Column("stage_1_compute_time", sqla.Float(64)),
                sqla.Column("stage_1_compute_steps", sqla.Integer),
                sqla.Column("stage_1_RHS_evaluations", sqla.Integer),
                sqla.Column("stage_1_mean_RHS_time", sqla.Float(64)),
                sqla.Column("stage_1_max_RHS_time", sqla.Float(64)),
                sqla.Column("stage_1_min_RHS_time", sqla.Float(64)),
                sqla.Column("stage_2_compute_time", sqla.Float(64)),
                sqla.Column("stage_2_compute_steps", sqla.Integer),
                sqla.Column("stage_2_RHS_evaluations", sqla.Integer),
                sqla.Column("stage_2_mean_RHS_time", sqla.Float(64)),
                sqla.Column("stage_2_max_RHS_time", sqla.Float(64)),
                sqla.Column("stage_2_min_RHS_time", sqla.Float(64)),
                sqla.Column("friction_compute_time", sqla.Float(64)),
                sqla.Column("friction_compute_steps", sqla.Integer),
                sqla.Column("friction_RHS_evaluations", sqla.Integer),
                sqla.Column("friction_mean_RHS_time", sqla.Float(64)),
                sqla.Column("friction_max_RHS_time", sqla.Float(64)),
                sqla.Column("friction_min_RHS_time", sqla.Float(64)),
                sqla.Column("has_WKB_violation", sqla.Boolean),
                sqla.Column("WKB_violation_z", sqla.Float(64)),
                sqla.Column("WKB_violation_efolds_subh", sqla.Float(64)),
                sqla.Column("init_efolds_subh", sqla.Float(64)),
                sqla.Column("validated", sqla.Boolean, default=False, nullable=False),
                sqla.Column(
                    "metadata", sqla.String(DEFAULT_STRING_LENGTH), nullable=True
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        solver_labels = payload["solver_labels"]

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        k_exit: wavenumber_exit_time = payload["k"]
        model_proxy: ModelProxy = payload["model"]
        z_sample: redshift_array = payload["z_sample"]

        z_init: Optional[float] = payload.get("z_init", None)
        T_init: Optional[float] = payload.get("T_init", 1.0)
        Tprime_init: Optional[float] = payload.get("Tprime_init", 0.0)

        phase_solver_table = tables["IntegrationSolver"].alias("phase_solver")
        friction_solver_table = tables["IntegrationSolver"].alias("friction_solver")
        tag_table = tables["TkWKB_tags"]

        # notice that we query only for validated data
        query = (
            sqla.select(
                table.c.serial,
                table.c.sin_coeff,
                table.c.cos_coeff,
                table.c.stage_1_compute_time,
                table.c.stage_1_compute_steps,
                table.c.stage_1_RHS_evaluations,
                table.c.stage_1_mean_RHS_time,
                table.c.stage_1_max_RHS_time,
                table.c.stage_1_min_RHS_time,
                table.c.stage_2_compute_time,
                table.c.stage_2_compute_steps,
                table.c.stage_2_RHS_evaluations,
                table.c.stage_2_mean_RHS_time,
                table.c.stage_2_max_RHS_time,
                table.c.stage_2_min_RHS_time,
                table.c.friction_compute_time,
                table.c.friction_compute_steps,
                table.c.friction_RHS_evaluations,
                table.c.friction_mean_RHS_time,
                table.c.friction_max_RHS_time,
                table.c.friction_min_RHS_time,
                table.c.has_WKB_violation,
                table.c.WKB_violation_z,
                table.c.WKB_violation_efolds_subh,
                table.c.init_efolds_subh,
                table.c.metadata,
                table.c.label,
                table.c.z_samples,
                table.c.z_init,
                table.c.T_init,
                table.c.Tprime_init,
                table.c.phase_solver_serial,
                phase_solver_table.c.label.label("phase_solver_label"),
                phase_solver_table.c.stepping.label("phase_solver_stepping"),
                table.c.friction_solver_serial,
                friction_solver_table.c.label.label("friction_solver_label"),
                friction_solver_table.c.stepping.label("friction_solver_stepping"),
            )
            .select_from(
                table.join(
                    phase_solver_table,
                    phase_solver_table.c.serial == table.c.phase_solver_serial,
                ).join(
                    friction_solver_table,
                    friction_solver_table.c.serial == table.c.friction_solver_serial,
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

        if z_init is not None:
            query = query.filter(
                sqla.func.abs(table.c.z_init - z_init) < DEFAULT_FLOAT_PRECISION
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
                    tab.c.wkb_serial == table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! TkWKBIntegration.build(): multiple results found when querying for TkWKBIntegration"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return TkWKBIntegration(
                payload=None,
                solver_labels=solver_labels,
                label=label,
                k=k_exit,
                model=model_proxy,
                atol=atol,
                rtol=rtol,
                z_init=z_init,
                T_init=T_init,
                Tprime_init=Tprime_init,
                z_sample=z_sample,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label

        sin_coeff = row_data.sin_coeff
        cos_coeff = row_data.cos_coeff

        z_init = row_data.z_init
        T_init = row_data.T_init
        Tprime_init = row_data.Tprime_init

        do_not_populate = payload.get("_do_not_populate", False)
        if not do_not_populate:
            # read out sample values associated with this integration
            value_table = tables["TkWKBValue"]
            redshift_table = tables["redshift"]

            sample_rows = conn.execute(
                sqla.select(
                    value_table.c.serial,
                    value_table.c.z_serial,
                    redshift_table.c.z,
                    redshift_table.c.source.label("z_is_source"),
                    redshift_table.c.response.label("z_is_response"),
                    value_table.c.H_ratio,
                    value_table.c.theta_mod_2pi,
                    value_table.c.theta_div_2pi,
                    value_table.c.friction,
                    value_table.c.T_WKB,
                    value_table.c.omega_WKB_sq,
                    value_table.c.WKB_criterion,
                    value_table.c.analytic_T_rad,
                    value_table.c.analytic_Tprime_rad,
                    value_table.c.analytic_T_w,
                    value_table.c.analytic_Tprime_w,
                )
                .select_from(
                    value_table.join(
                        redshift_table,
                        redshift_table.c.serial == value_table.c.z_serial,
                    )
                )
                .filter(value_table.c.wkb_serial == store_id)
                .order_by(redshift_table.c.z.desc())
            )

            z_points = []
            values = []
            for row in sample_rows:
                z_value = redshift(
                    store_id=row.z_serial,
                    z=row.z,
                    is_source=row.z_is_source,
                    is_response=row.z_is_response,
                )
                z_points.append(z_value)
                values.append(
                    TkWKBValue(
                        store_id=row.serial,
                        z=z_value,
                        H_ratio=row.H_ratio,
                        theta_mod_2pi=row.theta_mod_2pi,
                        theta_div_2pi=row.theta_div_2pi,
                        friction=row.friction,
                        omega_WKB_sq=row.omega_WKB_sq,
                        WKB_criterion=row.WKB_criterion,
                        T_WKB=row.T_WKB,
                        analytic_T_rad=row.analytic_T_rad,
                        analytic_Tprime_rad=row.analytic_Tprime_rad,
                        analytic_T_w=row.analytic_T_w,
                        analytic_Tprime_w=row.analytic_Tprime_w,
                        sin_coeff=sin_coeff,
                        cos_coeff=cos_coeff,
                        z_init=z_init,
                    )
                )
            imported_z_sample = redshift_array(z_points)

            if row_data.z_samples is not None:
                if len(imported_z_sample) != row_data.z_samples:
                    raise RuntimeError(
                        f'Fewer z-samples than expected were recovered from the validated WKB tensor Green function "{store_label}"'
                    )

            attributes = {"_deserialized": True}
        else:
            values = None
            imported_z_sample = None

            attributes = {"_do_not_populate": True, "_deserialized": True}

        obj = TkWKBIntegration(
            payload={
                "store_id": store_id,
                "sin_coeff": sin_coeff,
                "cos_coeff": cos_coeff,
                "stage_1_data": IntegrationData(
                    compute_time=row_data.stage_1_compute_time,
                    compute_steps=row_data.stage_1_compute_steps,
                    RHS_evaluations=row_data.stage_1_RHS_evaluations,
                    mean_RHS_time=row_data.stage_1_mean_RHS_time,
                    max_RHS_time=row_data.stage_1_max_RHS_time,
                    min_RHS_time=row_data.stage_1_min_RHS_time,
                ),
                "stage_2_data": IntegrationData(
                    compute_time=row_data.stage_2_compute_time,
                    compute_steps=row_data.stage_2_compute_steps,
                    RHS_evaluations=row_data.stage_2_RHS_evaluations,
                    mean_RHS_time=row_data.stage_2_mean_RHS_time,
                    max_RHS_time=row_data.stage_2_max_RHS_time,
                    min_RHS_time=row_data.stage_2_min_RHS_time,
                ),
                "friction_data": IntegrationData(
                    compute_time=row_data.friction_compute_time,
                    compute_steps=row_data.friction_compute_steps,
                    RHS_evaluations=row_data.friction_RHS_evaluations,
                    mean_RHS_time=row_data.friction_mean_RHS_time,
                    max_RHS_time=row_data.friction_max_RHS_time,
                    min_RHS_time=row_data.friction_min_RHS_time,
                ),
                "has_WKB_violation": row_data.has_WKB_violation,
                "WKB_violation_z": row_data.WKB_violation_z,
                "WKB_violation_efolds_subh": row_data.WKB_violation_efolds_subh,
                "init_efolds_subh": row_data.init_efolds_subh,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
                "phase_solver": (
                    IntegrationSolver(
                        store_id=row_data.phase_solver_serial,
                        label=(row_data.phase_solver_label),
                        stepping=(row_data.phase_solver_stepping),
                    )
                ),
                "friction_solver": (
                    IntegrationSolver(
                        store_id=row_data.friction_solver_serial,
                        label=(row_data.friction_solver_label),
                        stepping=(row_data.friction_solver_stepping),
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
            z_init=z_init,
            T_init=T_init,
            Tprime_init=Tprime_init,
            z_sample=imported_z_sample,
            tags=tags,
        )
        for key, value in attributes.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def store(
        obj: TkWKBIntegration,
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
                "phase_solver_serial": obj.phase_solver.store_id,
                "friction_solver_serial": obj.friction_solver.store_id,
                "z_min_serial": obj.z_sample.min.store_id,
                "z_samples": len(obj.values),
                "z_init": obj.z_init,
                "T_init": obj.T_init,
                "Tprime_init": obj.Tprime_init,
                "sin_coeff": obj.sin_coeff,
                "cos_coeff": obj.cos_coeff,
                "stage_1_compute_time": (
                    obj.stage_1_data.compute_time
                    if obj.stage_1_data is not None
                    else None
                ),
                "stage_1_compute_steps": (
                    obj.stage_1_data.compute_steps
                    if obj.stage_1_data is not None
                    else None
                ),
                "stage_1_RHS_evaluations": (
                    obj.stage_1_data.RHS_evaluations
                    if obj.stage_1_data is not None
                    else None
                ),
                "stage_1_mean_RHS_time": (
                    obj.stage_1_data.mean_RHS_time
                    if obj.stage_1_data is not None
                    else None
                ),
                "stage_1_max_RHS_time": (
                    obj.stage_1_data.max_RHS_time
                    if obj.stage_1_data is not None
                    else None
                ),
                "stage_1_min_RHS_time": (
                    obj.stage_1_data.min_RHS_time
                    if obj.stage_1_data is not None
                    else None
                ),
                "stage_2_compute_time": (
                    obj.stage_2_data.compute_time
                    if obj.stage_2_data is not None
                    else None
                ),
                "stage_2_compute_steps": (
                    obj.stage_2_data.compute_steps
                    if obj.stage_2_data is not None
                    else None
                ),
                "stage_2_RHS_evaluations": (
                    obj.stage_2_data.RHS_evaluations
                    if obj.stage_2_data is not None
                    else None
                ),
                "stage_2_mean_RHS_time": (
                    obj.stage_2_data.mean_RHS_time
                    if obj.stage_2_data is not None
                    else None
                ),
                "stage_2_max_RHS_time": (
                    obj.stage_2_data.max_RHS_time
                    if obj.stage_2_data is not None
                    else None
                ),
                "stage_2_min_RHS_time": (
                    obj.stage_2_data.min_RHS_time
                    if obj.stage_2_data is not None
                    else None
                ),
                "friction_compute_time": (
                    obj.friction_data.compute_time
                    if obj.friction_data is not None
                    else None
                ),
                "friction_compute_steps": (
                    obj.friction_data.compute_steps
                    if obj.friction_data is not None
                    else None
                ),
                "friction_RHS_evaluations": (
                    obj.friction_data.RHS_evaluations
                    if obj.friction_data is not None
                    else None
                ),
                "friction_mean_RHS_time": (
                    obj.friction_data.mean_RHS_time
                    if obj.friction_data is not None
                    else None
                ),
                "friction_max_RHS_time": (
                    obj.friction_data.max_RHS_time
                    if obj.friction_data is not None
                    else None
                ),
                "friction_min_RHS_time": (
                    obj.friction_data.min_RHS_time
                    if obj.friction_data is not None
                    else None
                ),
                "has_WKB_violation": obj.has_WKB_violation,
                "WKB_violation_z": obj.WKB_violation_z,
                "WKB_violation_efolds_subh": obj.WKB_violation_efolds_subh,
                "init_efolds_subh": obj._init_efolds_subh,
                "metadata": (
                    json.dumps(obj.metadata) if obj._metadata is not None else None
                ),
                "validated": False,
            },
        )

        # set store_id on behalf of the TkWKBIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["TkWKB_tags"]
        for tag in obj.tags:
            sqla_TkWKBTagAssociation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sampled output points
        value_inserter = inserters["TkWKBValue"]
        for value in obj.values:
            value: TkWKBValue
            value_id = value_inserter(
                conn,
                {
                    "wkb_serial": store_id,
                    "z_serial": value.z.store_id,
                    "H_ratio": value.H_ratio,
                    "theta_mod_2pi": value.theta_mod_2pi,
                    "theta_div_2pi": value.theta_div_2pi,
                    "friction": value.friction,
                    "T_WKB": value.T_WKB,
                    "omega_WKB_sq": value.omega_WKB_sq,
                    "WKB_criterion": value.WKB_criterion,
                    "analytic_T_rad": value.analytic_T_rad,
                    "analytic_Tprime_rad": value.analytic_Tprime_rad,
                    "analytic_T_w": value.analytic_T_w,
                    "analytic_Tprime_w": value.analytic_Tprime_w,
                },
            )

            # set store_id on behalf of the TkWKBValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: TkWKBIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in TkWKBIntegration corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["TkWKBValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.wkb_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: WKB tensor Green function "{obj.label}" did not validate after serialization'
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
        phase_solver_table = tables["IntegrationSolver"].alias("phase_solver")
        friction_solver_table = tables["IntegrationSolver"].alias("friction_solver")
        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["TkWKBValue"]
        tags_table = tables["TkWKB_tags"]

        # bake results into a list so that we can close this query; we are going to want to run
        # another one as we process the rows from this one
        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_samples,
                    wavenumber_table.c.k_inv_Mpc.label("k_inv_Mpc"),
                    phase_solver_table.c.label.label("phase_solver_label"),
                    friction_solver_table.c.label.label("friction_solver_label"),
                    atol_table.c.log10_tol.label("log10_atol"),
                    rtol_table.c.log10_tol.label("log10_rtol"),
                )
                .select_from(
                    table.join(
                        phase_solver_table,
                        phase_solver_table.c.serial == table.c.phase_solver_serial,
                    )
                    .join(
                        friction_solver_table,
                        friction_solver_table.c.serial
                        == table.c.friction_solver_serial,
                    )
                    .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                    .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
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
            ">> WKB tensor Green function",
            "     The following unvalidated integrations were detected in the datastore:",
        ]
        for integration in not_validated:
            msgs.append(
                f'       -- "{integration.label}" (store_id={integration.serial}) for k={integration.k_inv_Mpc:.5g}/Mpc (log10_atol={integration.log10_atol}, log10_rtol={integration.log10_rtol})'
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.wkb_serial == integration.serial,
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
                        value_table.c.wkb_serial.in_(invalid_serials)
                    )
                )
                conn.execute(
                    sqla.delete(tags_table).where(
                        tags_table.c.wkb_serial.in_(invalid_serials)
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


class sqla_TkWKBValue_factory(SQLAFactoryBase):
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
                    "wkb_serial",
                    sqla.Integer,
                    sqla.ForeignKey("TkWKBIntegration.serial"),
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
                sqla.Column("H_ratio", sqla.Float(64), nullable=False),
                sqla.Column("theta_mod_2pi", sqla.Float(64), nullable=False),
                sqla.Column("theta_div_2pi", sqla.Integer, nullable=False),
                sqla.Column("friction", sqla.Float(64), nullable=True),
                sqla.Column("omega_WKB_sq", sqla.Float(64), nullable=True),
                sqla.Column("WKB_criterion", sqla.Float(64), nullable=True),
                sqla.Column("T_WKB", sqla.Float(64), nullable=True),
                sqla.Column("analytic_T_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Tprime_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_T_w", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Tprime_w", sqla.Float(64), nullable=True),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        wkb_serial = payload.get("wkb_serial", None)

        model_proxy: Optional[ModelProxy] = payload.get("model", None)
        k: Optional[wavenumber_exit_time] = payload.get("k", None)

        has_serial = all([wkb_serial is not None])
        has_model = all([model_proxy is not None, k is not None])

        if all([has_serial, has_model]):
            print(
                "## TkWKBValue.build(): both an WKB integration serial number and a (model, wavenumber) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "TkWKBValue.build(): at least one of a WKB integration serial number and a (model, wavenumber) set must be supplied."
            )

        if has_serial:
            return sqla_TkWKBValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_TkWKBValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        wkb_serial: int = payload["wkb_serial"]

        H_ratio: Optional[float] = payload.get("H_ratio", None)
        theta_mod_2pi: Optional[float] = payload.get("theta_mod_2pi", None)
        theta_div_2pi: Optional[int] = payload.get("theta_div_2pi", None)
        friction: Optional[float] = payload.get("friction", None)
        omega_WKB_sq: Optional[float] = payload.get("omega_WKB_sq", None)
        WKB_criterion: Optional[float] = payload.get("WKB_criterion", None)
        T_WKB: Optional[float] = payload.get("T_WKB", None)
        has_data = all(
            [
                H_ratio is not None,
                theta_mod_2pi is not None,
                theta_div_2pi is not None,
                friction is not None,
                omega_WKB_sq is not None,
                WKB_criterion is not None,
                T_WKB is not None,
            ]
        )

        analytic_T_rad: Optional[float] = payload.get("analytic_T_rad", None)
        analytic_Tprime_rad: Optional[float] = payload.get("analytic_Tprime_rad", None)

        analytic_T_w: Optional[float] = payload.get("analytic_T_w", None)
        analytic_Tprime_w: Optional[float] = payload.get("analytic_Tprime_w", None)

        sin_coeff: Optional[float] = payload.get("sin_coeff", None)
        cos_coeff: Optional[float] = payload.get("cos_coeff", None)
        z_init: Optional[float] = payload.get("z_init", None)

        wkb_table = tables["TkWKBIntegration"]

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.H_ratio,
                    table.c.theta_mod_2pi,
                    table.c.theta_div_2pi,
                    table.c.friction,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                    table.c.T_WKB,
                    table.c.analytic_T_rad,
                    table.c.analytic_Tprime_rad,
                    table.c.analytic_T_w,
                    table.c.analytic_Tprime_w,
                    wkb_table.c.sin_coeff,
                    wkb_table.c.cos_coeff,
                    wkb_table.c.z_init,
                )
                .filter(
                    table.c.wkb_serial == wkb_serial,
                    table.c.z_serial == z.store_id,
                )
                .join(wkb_table, wkb_table.c.serial == table.c.wkb_serial)
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! TkWKBValue.build(): multiple results found when querying for TkWKBValue"
            )
            raise e

        if row_data is None:
            if not has_data:
                raise (
                    "TkWKBValue().build(): result was not found in datastore, but a data payload was not provided"
                )

            store_id = inserter(
                conn,
                {
                    "wkb_serial": wkb_serial,
                    "z_serial": z.store_id,
                    "H_ratio": H_ratio,
                    "theta_mod_2pi": theta_mod_2pi,
                    "theta_div_2pi": theta_div_2pi,
                    "friction": friction,
                    "omega_WKB_sq": omega_WKB_sq,
                    "WKB_criterion": WKB_criterion,
                    "T_WKB": T_WKB,
                    "analytic_T_rad": analytic_T_rad,
                    "analytic_Tprime_rad": analytic_Tprime_rad,
                    "analytic_T_w": analytic_T_w,
                    "analytic_Tprime_w": analytic_Tprime_w,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial
            omega_WKB_sq = row_data.omega_WKB_sq
            WKB_criterion = row_data.WKB_criterio

            T_WKB = row_data.T_WKB

            analytic_T_rad = row_data.analytic_T_rad
            analytic_Tprime_rad = row_data.analytic_Tprime_rad
            analytic_T_w = row_data.analytic_T_w
            analytic_Tprime_w = row_data.analytic_Tprime_w

            sin_coeff = row_data.sin_coeff
            cos_coeff = row_data.cos_coeff
            z_init = row_data.z_init

            # we choose H_ratio and theta to test because these are the critical data to reconstruct
            # T_WKB; everything else, such as omega_WKB_sq, is optional
            if (
                H_ratio is not None
                and fabs(row_data.H_ratio - H_ratio) > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"TkWKBValue.build(): Stored WKB H_ratio ratio (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.H_ratio} differs from expected value = {H_ratio}"
                )
            if (
                theta_mod_2pi is not None
                and fabs(row_data.theta_mod_2pi - theta_mod_2pi)
                > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"TkWKBValue.build(): Stored WKB theta mod 2pi (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.theta_mod_2pi} differs from expected value = {theta_mod_2pi}"
                )
            if theta_div_2pi is not None and row_data.theta_div_2pi != theta_div_2pi:
                raise ValueError(
                    f"TkWKBValue.build(): Stored WKB theta div 2pi (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.theta_div_2pi} differs from expected value = {theta_div_2pi}"
                )
            if (
                friction is not None
                and fabs(row_data.friction - friction) > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"TkWKBValue.build(): Stored WKB friction (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.theta_mod_2pi} differs from expected value = {theta_mod_2pi}"
                )

            H_ratio = row_data.H_ratio
            theta_mod_2pi = row_data.theta_mod_2pi
            theta_div_2pi = row_data.theta_div_2pi
            friction = row_data.friction

            attribute_set = {"_deserialized": True}

        obj = TkWKBValue(
            store_id=store_id,
            z=z,
            H_ratio=H_ratio,
            theta_mod_2pi=theta_mod_2pi,
            theta_div_2pi=theta_div_2pi,
            friction=friction,
            omega_WKB_sq=omega_WKB_sq,
            WKB_criterion=WKB_criterion,
            T_WKB=T_WKB,
            analytic_T_rad=analytic_T_rad,
            analytic_Tprime_rad=analytic_Tprime_rad,
            analytic_T_w=analytic_T_w,
            analytic_Tprime_w=analytic_Tprime_w,
            sin_coeff=sin_coeff,
            cos_coeff=cos_coeff,
            z_init=z_init,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _build_impl_model(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        model_payload: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        atol: Optional[tolerance] = payload.get("atol", None)
        rtol: Optional[tolerance] = payload.get("rtol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        wkb_table = tables["TkWKBIntegration"]

        try:
            # TODO: benchmarking suggests this query is indistinguishable from filtering directly on the
            #  TkWKBIntegration serial number (if we only knew what it was), so this may be about
            #  as good as we can do. But it is still slow. For production use, should look at how this
            #  can be improved.
            wkb_query = sqla.select(
                wkb_table.c.serial,
                wkb_table.c.sin_coeff,
                wkb_table.c.cos_coeff,
                wkb_table.c.z_init,
            ).filter(
                wkb_table.c.model_serial == model_payload.store_id,
                wkb_table.c.wavenumber_exit_serial == k.store_id,
                wkb_table.c.validated == True,
            )

            if atol is not None:
                wkb_query = wkb_query.filter(wkb_table.c.atol_serial == atol.store_id)

            if rtol is not None:
                wkb_query = wkb_query.filter(wkb_table.c.rtol_serial == rtol.store_id)

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["TkWKB_tags"].alias(f"tag_{count}")
                count += 1
                wkb_query = wkb_query.join(
                    tab,
                    and_(
                        tab.c.wkb_serial == wkb_table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            subquery = wkb_query.subquery()

            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.H_ratio,
                    table.c.theta_mod_2pi,
                    table.c.theta_div_2pi,
                    table.c.friction,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                    table.c.T_WKB,
                    table.c.analytic_T_rad,
                    table.c.analytic_Tprime_rad,
                    table.c.analytic_T_w,
                    table.c.analytic_Tprime_w,
                    subquery.c.sin_coeff,
                    subquery.c.cos_coeff,
                    subquery.c.z_init,
                )
                .select_from(
                    subquery.join(table, table.c.wkb_serial == subquery.c.serial)
                )
                .filter(
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! TkWKBValue.build(): multiple results found when querying for TkWKBValue"
            )
            raise e

        if row_data is None:
            # return empty object
            obj = TkWKBValue(
                store_id=None,
                z=z,
                H_ratio=None,
                theta_mod_2pi=None,
                theta_div_2pi=None,
                friction=None,
            )
            obj._k_exit = k
            return obj

        obj = TkWKBValue(
            store_id=row_data.serial,
            z=z,
            H_ratio=row_data.H_ratio,
            theta_mod_2pi=row_data.theta_mod_2pi,
            theta_div_2pi=row_data.theta_div_2pi,
            friction=row_data.friction,
            omega_WKB_sq=row_data.omega_WKB_sq,
            WKB_criterion=row_data.WKB_criterion,
            T_WKB=row_data.T_WKB,
            analytic_T_rad=row_data.analytic_T_rad,
            analytic_Tprime_rad=row_data.analytic_Tprime_rad,
            analytic_T_w=row_data.analytic_T_w,
            analytic_Tprime_w=row_data.analytic_Tprime_w,
            sin_coeff=row_data.sin_coeff,
            cos_coeff=row_data.cos_coeff,
            z_init=row_data.z_init,
        )
        obj._deserialized = True
        obj._k_exit = k
        return obj

    @staticmethod
    def read_batch(payload, conn, table, tables):
        model_proxy: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        atol: Optional[tolerance] = payload.get("atol", None)
        rtol: Optional[tolerance] = payload.get("rtol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        z: Optional[redshift] = payload.get("z", None)

        wkb_table = tables["TkWKBIntegration"]
        redshift_table = tables["redshift"]

        wkb_query = sqla.select(
            wkb_table.c.serial,
            wkb_table.c.sin_coeff,
            wkb_table.c.cos_coeff,
            wkb_table.c.z_init,
        ).filter(
            wkb_table.c.model_serial == model_proxy.store_id,
            wkb_table.c.wavenumber_exit_serial == k.store_id,
            wkb_table.c.validated == True,
        )

        if atol is not None:
            wkb_query = wkb_query.filter(wkb_table.c.atol_serial == atol.store_id)

        if rtol is not None:
            wkb_query = wkb_query.filter(wkb_table.c.rtol_serial == rtol.store_id)

        count = 0
        for tag in tags:
            tag: store_tag
            tab = tables["TkWKB_tags"].alias(f"tag_{count}")
            count += 1
            wkb_query = wkb_query.join(
                tab,
                and_(
                    tab.c.wkb_serial == wkb_table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        subquery = wkb_query.subquery()

        row_query = sqla.select(
            table.c.serial,
            table.c.H_ratio,
            table.c.theta_mod_2pi,
            table.c.theta_div_2pi,
            table.c.friction,
            table.c.omega_WKB_sq,
            table.c.WKB_criterion,
            table.c.T_WKB,
            table.c.analytic_T_rad,
            table.c.analytic_Tprime_rad,
            table.c.analytic_T_w,
            table.c.analytic_Tprime_w,
            subquery.c.sin_coeff,
            subquery.c.cos_coeff,
            subquery.c.z_init,
            redshift_table.c.z.label("z"),
            table.c.z_serial.label("z_serial"),
            redshift_table.c.source.label("z_is_source"),
            redshift_table.c.response.label("z_is_response"),
        ).select_from(
            subquery.join(table, table.c.wkb_serial == subquery.c.serial).join(
                redshift_table, redshift_table.c.serial == table.c.z_serial
            )
        )

        if z is not None:
            row_query = row_query.filter(table.c.z_serial == z.store_id)

        row_data = conn.execute(row_query)

        def make_obj(row):
            obj = TkWKBValue(
                store_id=row.serial,
                z=redshift(
                    store_id=row.z_serial,
                    z=row.z,
                    is_source=row.z_is_source,
                    is_response=row.z_is_response,
                ),
                H_ratio=row.H_ratio,
                theta_mod_2pi=row.theta_mod_2pi,
                theta_div_2pi=row.theta_div_2pi,
                omega_WKB_sq=row.omega_WKB_sq,
                WKB_criterion=row.WKB_criterion,
                T_WKB=row.T_WKB,
                analytic_T_rad=row.analytic_T_rad,
                analytic_Tprime_rad=row.analytic_Tprime_rad,
                analytic_T_w=row.analytic_T_w,
                analytic_Tprime_w=row.analytic_Tprime_w,
                sin_coeff=row.sin_coeff,
                cos_coeff=row.cos_coeff,
                z_init=row.z_init,
            )
            obj._deserialized = True
            obj._k_exit = k

            return obj

        objects = [make_obj(row) for row in row_data]
        return objects
