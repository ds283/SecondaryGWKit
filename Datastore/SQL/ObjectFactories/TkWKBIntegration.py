"""
Datastore factories for TkWKBIntegration and its per-redshift TkWKBValue rows.

SCHEMA NOTE (prompts/tolerance-convergence, prompt 05). "atol_serial" and "rtol_serial" are gone
and one integer column, "rho_gauss_order", stands where they stood in the lookup key.

The pair described nothing. The WKB phase is built from the background model's conformal-time (or
sound-horizon) table and a Gauss-Legendre residual table
(Quadrature/integrators/WKB_phase_function.py), neither of which has a tolerance; the pair was
kept only because it was in the key, as ComputeTargets/TkWKBIntegration.py's own comment said. What sets the
accuracy is RHO_GAUSS_ORDER (ComputeTargets/phase_residual.py), which was a module constant in no
column at all -- so moving it left the key unmoved and the pipeline went on serving the row it had.
Prompt 04 measured it (docs/tolerance-convergence/ORDER-AUDIT.md) and the user settled the
replacement on 2026-09-18 (prompts/tolerance-convergence/README.md §7 D3).

RESIDUAL_WKB_REGION_MARGIN is deliberately *not* a column beside it: prompt 04 swept it from 0.05
to 0.9 and found the residual a producer reads bit-identical throughout, so it bounds where the
residual may be evaluated rather than how accurately, and it cannot produce two rows that differ.

SCHEMA NOTE (prompts/tolerance-convergence, prompt 05b). No column changes here, but what fills
"rho_gauss_order" does. Prompt 05 wrote the value of the module constant, read through an accessor
that re-read it on every call; that reports what the module currently says rather than what the
object is. Since 05b the compute path carries the residual table's own order out of
WKB_phase_function's payload, and build() below *selects* the column and hands it to the
constructor, so a rehydrated object reports its row. build() still *filters* on the current module
constant -- a row computed at another order is a different row, not a miss to repair (README §7
D10).

SCHEMA NOTE (prompts/datastore-integrity, prompt 07). A TkWKBIntegration starts from its
TkNumericIntegration's stop point: z_init = z_exit - stop_deltaz_subh, T_init = stop_T and
Tprime_init = stop_Tprime (main.py's Tk WKB stage). The row now records which numeric row that
was, in "numeric_serial", NOT NULL with a foreign key to TkNumericIntegration.serial (both shard
on k, config/sharding.py, so the parent is always on this shard), and build() filters on it. The
initial data are functions of the parent, so they left the key: they stay stored, as payload, and
the absolute DEFAULT_FLOAT_PRECISION match on z_init (docs/datastore-integrity-audit.md §3.13) is
gone. store() refuses a row whose initial data are not its parent's stop values bit for bit.

A lookup names its parent ("numeric_serial"), or asks for the labels-only reader mode
("labels_only": True), which D1 (prompts/datastore-integrity README §6.2) allows a reader holding
only physical labels: it refuses when more than one stored row matches, naming the parents that
distinguish them. A lookup that does neither raises.
"""

import json
import struct
from math import fabs
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    TkWKBIntegration,
    TkWKBValue,
)
import ComputeTargets.phase_residual as phase_residual
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift_array, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from config.defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


def _same_bits(a, b) -> bool:
    """Whether two stored floats are the same float, bit for bit. None is never the same as
    anything: an initial datum that is missing cannot have been derived from its parent.
    """
    if a is None or b is None:
        return False
    return struct.pack("<d", float(a)) == struct.pack("<d", float(b))


def _numeric_parent_filter(column, numeric_serial):
    """The key's parent term: a TkWKBIntegration is always seeded from a TkNumericIntegration,
    so the parent is a serial and nothing else (there is no pure-WKB transfer function).
    """
    if isinstance(numeric_serial, bool) or not isinstance(numeric_serial, int):
        raise TypeError(
            f"TkWKBIntegration: a numeric parent is a TkNumericIntegration serial, not "
            f"{numeric_serial!r}"
        )
    return column == numeric_serial


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
                # the TkNumericIntegration whose stop point this row starts from (prompts/
                # datastore-integrity, prompt 07). Both tables shard on k, so the parent is on
                # this shard and the foreign key can be declared. Part of the lookup key
                sqla.Column(
                    "numeric_serial",
                    sqla.Integer,
                    sqla.ForeignKey("TkNumericIntegration.serial"),
                    index=True,
                    nullable=False,
                ),
                # the Gauss-Legendre order of the phase residual table rho (module docstring
                # above). This is the accuracy parameter of this object: an integer, not a
                # tolerance, and the only one it has.
                sqla.Column(
                    "rho_gauss_order", sqla.Integer, index=True, nullable=False
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

        k_exit: wavenumber_exit_time = payload["k"]
        model_proxy: ModelProxy = payload["model"]
        z_sample: redshift_array = payload["z_sample"]

        # payload, not key: functions of the numeric parent (module docstring)
        z_init: Optional[float] = payload.get("z_init", None)
        T_init: Optional[float] = payload.get("T_init", 1.0)
        Tprime_init: Optional[float] = payload.get("Tprime_init", 0.0)

        # the key's parent term, or the reader mode D1 allows; never neither, and never both
        numeric_serial: Optional[int] = payload.get("numeric_serial", None)
        labels_only: bool = payload.get("labels_only", False)
        if labels_only and numeric_serial is not None:
            raise ValueError(
                "TkWKBIntegration.build(): both a numeric parent and labels_only were given; a "
                "keyed lookup names its parent, and the reader mode names none"
            )
        if not labels_only and numeric_serial is None:
            raise RuntimeError(
                "TkWKBIntegration.build(): no numeric parent was given. A TkWKBIntegration is "
                "keyed on the TkNumericIntegration it starts from (numeric_serial); a reader "
                "holding only physical labels must ask for labels_only=True explicitly"
            )

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
                # selected, not merely filtered on: the object must report the order its own row
                # records rather than the current module constant
                # (prompts/tolerance-convergence, prompt 05b)
                table.c.rho_gauss_order,
                table.c.label,
                table.c.numeric_serial,
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
                # the accuracy half of the key. Read from the single declaration of the order at
                # call time, never inlined as a literal: a literal would stop tracking the
                # constant the moment a later prompt moved it, which is the exact failure this
                # key exists to prevent
                table.c.rho_gauss_order == phase_residual.RHO_GAUSS_ORDER,
            )
        )

        # the numeric parent (prompt 07). z_init is not filtered on: it is a function of the
        # parent, and the absolute DEFAULT_FLOAT_PRECISION match it used to have was bit equality
        # at z ~ 1e9-1e12 (docs/datastore-integrity-audit.md §3.13)
        if not labels_only:
            query = query.filter(
                _numeric_parent_filter(table.c.numeric_serial, numeric_serial)
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
            if labels_only:
                # D1: the physical labels may select, but more than one stored row is refused,
                # naming the parents that distinguish them, rather than one of them chosen
                candidates = sorted(
                    conn.execute(
                        sqla.select(query.subquery().c.numeric_serial)
                    ).scalars()
                )
                if len(candidates) > 1:
                    raise MultipleResultsFound(
                        f"TkWKBIntegration.build(): labels_only lookup matched "
                        f"{len(candidates)} stored rows (model serial={model_proxy.store_id}, k "
                        f"serial={k_exit.store_id}), with numeric_serial values {candidates}. "
                        f"Refusing to choose one; name the numeric parent."
                    )

            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! TkWKBIntegration.build(): multiple results found when querying for TkWKBIntegration"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object. A labels-only miss has no parent to give
            # it, and says so: it can report absence to a reader and cannot be computed
            return TkWKBIntegration(
                payload=None,
                solver_labels=solver_labels,
                label=label,
                k=k_exit,
                model=model_proxy,
                numeric_serial=numeric_serial,
                labels_only=labels_only,
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
                # the row's own order, which is the order this object's phase residual was
                # tabulated at. build() filters on the current module constant, so this is that
                # constant today; it is read off the row all the same, because what the object
                # reports must be a property of the object and not of the module
                "rho_gauss_order": row_data.rho_gauss_order,
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
            numeric_serial=row_data.numeric_serial,
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
        # no row is written whose initial data are not its numeric parent's stop values
        sqla_TkWKBIntegration_factory._check_numeric_parent(obj, conn, tables)

        store_id = inserter(
            conn,
            {
                "label": obj.label,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "model_serial": obj.model_proxy.store_id,
                "numeric_serial": obj.numeric_serial,
                # the order the residual table this object's phase came from was actually
                # built at: since prompt 05b the accessor reports the object's own order --
                # carried out of the table by WKB_phase_function on the compute path, read off
                # the row on the rehydration path -- rather than re-reading the module constant.
                # build() goes on filtering on that constant, so a row written at another order
                # is simply a different row (README §7 D10).
                "rho_gauss_order": obj.rho_gauss_order,
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
    def _check_numeric_parent(obj: TkWKBIntegration, conn, tables):
        """
        The store-time consistency check (prompts/datastore-integrity, prompt 07). The row's
        initial data are payload, not key, so nothing else would notice if they were not the
        parent's: main.py derives z_init = z_exit - stop_deltaz_subh, T_init = stop_T and
        Tprime_init = stop_Tprime from the TkNumericIntegration it looked up, and here the parent
        row is read on this shard -- with its exit time, which is replicated -- and the same
        expression recomputed. Every value must agree bit for bit, and the parent must be for this
        object's k and model. It lives in store(), which every write goes through, so no caller
        can skip it. A mismatch raises naming both values.
        """
        numeric_serial = obj.numeric_serial
        if numeric_serial is None:
            raise RuntimeError(
                "TkWKBIntegration.store(): the object has no numeric parent. A "
                "TkWKBIntegration always starts from a TkNumericIntegration's stop point"
            )

        numeric_table = tables["TkNumericIntegration"]
        exit_table = tables["wavenumber_exit_time"]

        parent = conn.execute(
            sqla.select(
                numeric_table.c.serial,
                numeric_table.c.wavenumber_exit_serial,
                numeric_table.c.model_serial,
                numeric_table.c.stop_deltaz_subh,
                numeric_table.c.stop_T,
                numeric_table.c.stop_Tprime,
                exit_table.c.z_exit,
            )
            .select_from(
                numeric_table.join(
                    exit_table,
                    exit_table.c.serial == numeric_table.c.wavenumber_exit_serial,
                    isouter=True,
                )
            )
            .filter(numeric_table.c.serial == numeric_serial)
        ).one_or_none()

        where = f"TkWKBIntegration.store() (numeric_serial={numeric_serial})"
        if parent is None:
            raise RuntimeError(
                f"{where}: no TkNumericIntegration with that serial is on this shard"
            )
        if parent.wavenumber_exit_serial != obj._k_exit.store_id:
            raise RuntimeError(
                f"{where}: the parent is for wavenumber exit serial "
                f"{parent.wavenumber_exit_serial}, this object for {obj._k_exit.store_id}"
            )
        if parent.model_serial != obj.model_proxy.store_id:
            raise RuntimeError(
                f"{where}: the parent is for model serial {parent.model_serial}, this object "
                f"for {obj.model_proxy.store_id}"
            )
        if parent.z_exit is None or parent.stop_deltaz_subh is None:
            raise RuntimeError(
                f"{where}: the parent's stop point cannot be recomputed (z_exit="
                f"{parent.z_exit!r}, stop_deltaz_subh={parent.stop_deltaz_subh!r})"
            )

        expected_z_init = parent.z_exit - parent.stop_deltaz_subh
        for name, stored, expected, derivation in (
            ("z_init", obj.z_init, expected_z_init, "z_exit - stop_deltaz_subh"),
            ("T_init", obj.T_init, parent.stop_T, "stop_T"),
            ("Tprime_init", obj.Tprime_init, parent.stop_Tprime, "stop_Tprime"),
        ):
            if not _same_bits(stored, expected):
                raise ValueError(
                    f"{where}: {name}={stored!r} is not the parent's {derivation}="
                    f"{expected!r}. A WKB row must start from its numeric parent's stop values, "
                    f"bit for bit."
                )

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
                    table.c.rho_gauss_order,
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
                f'       -- "{integration.label}" (store_id={integration.serial}) for k={integration.k_inv_Mpc:.5g}/Mpc (N_rho={integration.rho_gauss_order})'
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

    @staticmethod
    def inventory_records(conn, table, tables, context):
        # the physical identity of a TkWKBIntegration row (store-fingerprint prompt 02): what build()
        # filters on -- the model, the wavenumber exit, rho_gauss_order and, since
        # prompts/datastore-integrity prompt 07, the numeric parent (same shard, so a key parent)
        # -- with its tags, its validated flag and its TkWKBValue count. z_init left the key with
        # prompt 07: it is a function of the parent. The solvers are not in the lookup and are not
        # identity
        from Datastore.store_inventory import Parent, read_records

        return read_records(
            conn,
            table,
            tables,
            context,
            leaves=("rho_gauss_order",),
            parents={
                "model": Parent("model_serial", "BackgroundModel"),
                "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
                "numeric": Parent("numeric_serial", "TkNumericIntegration"),
            },
            tags=("TkWKB_tags", "wkb_serial"),
            values=("TkWKBValue", "wkb_serial"),
            validated="validated",
        )


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
                raise RuntimeError(
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
            WKB_criterion = row_data.WKB_criterion

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
        numeric_serial: Optional[int] = payload.get("numeric_serial", None)

        tags: List[store_tag] = payload.get("tags", [])

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

            wkb_query = wkb_query.filter(
                wkb_table.c.rho_gauss_order == phase_residual.RHO_GAUSS_ORDER
            )

            # the numeric parent is in the parent's key when it is given (prompt 07, in place of
            # 05c's z_init). With none, the read spans every parent, and a second one is refused
            # below
            if numeric_serial is not None:
                wkb_query = wkb_query.filter(
                    _numeric_parent_filter(wkb_table.c.numeric_serial, numeric_serial)
                )

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

        tags: List[store_tag] = payload.get("tags", [])

        z: Optional[redshift] = payload.get("z", None)
        numeric_serial: Optional[int] = payload.get("numeric_serial", None)

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

        wkb_query = wkb_query.filter(
            wkb_table.c.rho_gauss_order == phase_residual.RHO_GAUSS_ORDER
        )

        # the numeric parent is in the parent's key when it is given (prompt 07, in place of 05c's
        # z_init). With none, the read spans every parent, and a second one is refused below
        if numeric_serial is not None:
            wkb_query = wkb_query.filter(
                _numeric_parent_filter(wkb_table.c.numeric_serial, numeric_serial)
            )

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
            # the parent serial, selected so that a second parent for one (model, k) can be
            # refused from these rows, without a second query
            table.c.wkb_serial,
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

        row_data = conn.execute(row_query).all()

        # D1 (prompts/datastore-integrity README §6.2): a read that holds only physical labels may
        # select by them, but must refuse when more than one stored parent matches, rather than
        # return a mixture. A Tk WKB row has no z_source, so the refusal is per (model, k), which
        # every row of this read shares (prompt 05c)
        #
        # a for-loop rather than a set comprehension: the read-back guard
        # (test_factory_select_columns.py) walks a comprehension's element before its target, so
        # it would not see this read of row.wkb_serial
        parents = set()
        for row in row_data:
            parents.add(row.wkb_serial)
        if len(parents) > 1:
            raise MultipleResultsFound(
                f"TkWKBValue.read_batch(): {len(parents)} TkWKBIntegration parents match one key "
                f"(model serial={model_proxy.store_id}, k serial={k.store_id}): parent serials "
                f"{sorted(parents)}. Refusing to return values from more than one parent."
            )

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
                friction=row.friction,
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
