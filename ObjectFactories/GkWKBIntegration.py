"""
Datastore factories for GkWKBIntegration and its per-redshift GkWKBValue rows.

SCHEMA NOTE (prompts/tolerance-convergence, prompt 05). "atol_serial" and "rtol_serial" are gone
and one integer column, "rho_gauss_order", stands where they stood in the lookup key.

The pair described nothing. The WKB phase is built from the background model's conformal-time (or
sound-horizon) table and a Gauss-Legendre residual table
(Quadrature/integrators/WKB_phase_function.py), neither of which has a tolerance; the pair was
kept only because it was in the key, as ComputeTargets/GkWKBIntegration.py's own comment said. What sets the
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

SCHEMA NOTE (prompts/datastore-integrity, prompt 07). A GkWKBIntegration either starts from its
GkNumericIntegration's stop point -- z_init = z_exit - stop_deltaz_subh, G_init = stop_G,
Gprime_init = stop_Gprime -- or is *pure WKB*, started at z_source with G_init = 0 and
Gprime_init = 1 (main.py's Gk WKB stage chooses which, by z_source against the geometric mean of
the 3- and 4-e-fold points and by whether a numeric row exists). The row now records its parent
in "numeric_serial", with a foreign key to GkNumericIntegration.serial (both shard on k,
config/sharding.py), and **NULL means pure WKB**. build() filters on it. The initial data are
functions of the parent (or of z_source), so they left the key; they stay stored, as payload, and
the absolute DEFAULT_FLOAT_PRECISION match on z_init (docs/datastore-integrity-audit.md §3.13) is
gone. store() refuses a row whose initial data are not what its parent, or the pure-WKB branch,
gives bit for bit.

A lookup names its parent ("numeric_serial": a serial, or GkWKBIntegration.PURE_WKB for a
pure-WKB row), or asks for the labels-only reader mode ("labels_only": True), which D1
(prompts/datastore-integrity README §6.2) allows a reader holding only physical labels: it refuses
when more than one stored row matches, naming the parents (a NULL as pure WKB). A lookup that does
neither raises -- a missing parent is never read as "pure WKB" by omission.
"""

import json
import struct
from math import fabs
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    GkWKBIntegration,
    GkWKBValue,
)
import ComputeTargets.phase_residual as phase_residual
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift_array, redshift
from datastorekit.SQL.factory_base import SQLAFactoryBase
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
    """The key's parent term: a GkNumericIntegration serial, or the explicit pure-WKB marker,
    which matches the rows that have no numeric parent (NULL) and nothing else."""
    if numeric_serial == GkWKBIntegration.PURE_WKB:
        return column.is_(None)
    if isinstance(numeric_serial, bool) or not isinstance(numeric_serial, int):
        raise TypeError(
            f"GkWKBIntegration: a numeric parent is a GkNumericIntegration serial or "
            f"GkWKBIntegration.PURE_WKB, not {numeric_serial!r}"
        )
    return column == numeric_serial


def _parent_name(numeric_serial) -> str:
    return "pure WKB" if numeric_serial is None else str(numeric_serial)


class sqla_GkWKBTagAssociation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("GkWKBIntegration.serial"),
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
    def add_tag(conn, inserter, wkb: GkWKBIntegration, tag: store_tag):
        inserter(
            conn,
            {
                "wkb_serial": wkb.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, wkb: GkWKBIntegration, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.wkb_serial == wkb.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_GkWKBIntegration_factory(SQLAFactoryBase):
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
                # the GkNumericIntegration whose stop point this row starts from, or NULL for a
                # pure-WKB row started at z_source with G = 0, G' = 1 (prompts/datastore-integrity,
                # prompt 07). Both tables shard on k, so the parent is on this shard and the
                # foreign key can be declared. Part of the lookup key
                sqla.Column(
                    "numeric_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkNumericIntegration.serial"),
                    index=True,
                    nullable=True,
                ),
                # the Gauss-Legendre order of the phase residual table rho (module docstring
                # above). This is the accuracy parameter of this object: an integer, not a
                # tolerance, and the only one it has.
                sqla.Column(
                    "rho_gauss_order", sqla.Integer, index=True, nullable=False
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
                sqla.Column("z_samples", sqla.Integer, nullable=False),
                sqla.Column("z_init", sqla.Float(64), nullable=False),
                sqla.Column("sin_coeff", sqla.Float(64), nullable=False),
                sqla.Column("cos_coeff", sqla.Float(64), nullable=False),
                sqla.Column("G_init", sqla.Float(64), nullable=False),
                sqla.Column("Gprime_init", sqla.Float(64), nullable=False),
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
        z_source: redshift = payload.get("z_source", None)

        # payload, not key: functions of the numeric parent, or of z_source (module docstring)
        z_init: Optional[float] = payload.get("z_init", None)
        G_init: Optional[float] = payload.get("G_init", 0.0)
        Gprime_init: Optional[float] = payload.get("Gprime_init", 1.0)

        # the key's parent term (a serial, or the pure-WKB marker), or the reader mode D1 allows;
        # never neither, and never both
        numeric_serial = payload.get("numeric_serial", None)
        labels_only: bool = payload.get("labels_only", False)
        if labels_only and numeric_serial is not None:
            raise ValueError(
                "GkWKBIntegration.build(): both a numeric parent and labels_only were given; a "
                "keyed lookup names its parent, and the reader mode names none"
            )
        if not labels_only and numeric_serial is None:
            raise RuntimeError(
                "GkWKBIntegration.build(): no numeric parent was given. A GkWKBIntegration is "
                "keyed on the GkNumericIntegration it starts from (numeric_serial), or on "
                "numeric_serial=GkWKBIntegration.PURE_WKB for a pure-WKB row; a reader holding "
                "only physical labels must ask for labels_only=True explicitly"
            )

        solver_table = tables["IntegrationSolver"]
        tag_table = tables["GkWKB_tags"]
        redshift_table = tables["redshift"]

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
                table.c.has_WKB_violation,
                table.c.WKB_violation_z,
                table.c.WKB_violation_efolds_subh,
                table.c.init_efolds_subh,
                table.c.metadata,
                # selected, not merely filtered on: the object must report the order its own row
                # records rather than the current module constant
                # (prompts/tolerance-convergence, prompt 05b)
                table.c.rho_gauss_order,
                table.c.solver_serial,
                table.c.label,
                table.c.numeric_serial,
                table.c.z_source_serial,
                redshift_table.c.z.label("z_source"),
                redshift_table.c.source.label("z_source_is_source"),
                redshift_table.c.response.label("z_source_is_response"),
                table.c.z_samples,
                table.c.z_init,
                table.c.G_init,
                table.c.Gprime_init,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
            )
            .select_from(
                table.join(
                    solver_table, solver_table.c.serial == table.c.solver_serial
                ).join(
                    redshift_table, redshift_table.c.serial == table.c.z_source_serial
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

        if z_source is not None:
            query = query.filter(
                table.c.z_source_serial == z_source.store_id,
            )

        # the numeric parent, or NULL for pure WKB (prompt 07). z_init is not filtered on: it is a
        # function of the parent, and the absolute DEFAULT_FLOAT_PRECISION match it used to have
        # was bit equality at z ~ 1e9-1e12 (docs/datastore-integrity-audit.md §3.13)
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
                # naming the parents that distinguish them (a NULL as pure WKB), rather than one
                # of them chosen
                candidates = list(
                    conn.execute(
                        sqla.select(query.subquery().c.numeric_serial)
                    ).scalars()
                )
                if len(candidates) > 1:
                    named = sorted(_parent_name(c) for c in candidates)
                    raise MultipleResultsFound(
                        f"GkWKBIntegration.build(): labels_only lookup matched "
                        f"{len(candidates)} stored rows (model serial={model_proxy.store_id}, k "
                        f"serial={k_exit.store_id}, z_source serial="
                        f"{z_source.store_id if z_source is not None else None}), with "
                        f"numeric_serial values {named}. Refusing to choose one; name the "
                        f"numeric parent, or GkWKBIntegration.PURE_WKB."
                    )

            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkWKBIntegration.build(): multiple results found when querying for GkWKBIntegration"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object: seeded from the parent, pure WKB for the
            # marker, and for a labels-only miss neither -- it can report absence to a reader and
            # cannot be computed
            return GkWKBIntegration(
                payload=None,
                solver_labels=solver_labels,
                label=label,
                k=k_exit,
                model=model_proxy,
                numeric_serial=(
                    None
                    if numeric_serial == GkWKBIntegration.PURE_WKB
                    else numeric_serial
                ),
                labels_only=labels_only,
                z_source=z_source,
                z_init=z_init,
                G_init=G_init,
                Gprime_init=Gprime_init,
                z_sample=z_sample,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label

        sin_coeff = row_data.sin_coeff
        cos_coeff = row_data.cos_coeff

        z_init = row_data.z_init
        G_init = row_data.G_init
        Gprime_init = row_data.Gprime_init

        do_not_populate = payload.get("_do_not_populate", False)
        if not do_not_populate:
            # read out sample values associated with this integration
            value_table = tables["GkWKBValue"]

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
                    value_table.c.G_WKB,
                    value_table.c.omega_WKB_sq,
                    value_table.c.WKB_criterion,
                    value_table.c.analytic_G_rad,
                    value_table.c.analytic_Gprime_rad,
                    value_table.c.analytic_G_w,
                    value_table.c.analytic_Gprime_w,
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
                    GkWKBValue(
                        store_id=row.serial,
                        z=z_value,
                        H_ratio=row.H_ratio,
                        theta_mod_2pi=row.theta_mod_2pi,
                        theta_div_2pi=row.theta_div_2pi,
                        omega_WKB_sq=row.omega_WKB_sq,
                        WKB_criterion=row.WKB_criterion,
                        G_WKB=row.G_WKB,
                        analytic_G_rad=row.analytic_G_rad,
                        analytic_Gprime_rad=row.analytic_Gprime_rad,
                        analytic_G_w=row.analytic_G_w,
                        analytic_Gprime_w=row.analytic_Gprime_w,
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

        obj = GkWKBIntegration(
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
            numeric_serial=row_data.numeric_serial,
            z_source=redshift(
                store_id=row_data.z_source_serial,
                z=row_data.z_source,
                is_source=row_data.z_source_is_source,
                is_response=row_data.z_source_is_response,
            ),
            z_init=z_init,
            G_init=G_init,
            Gprime_init=Gprime_init,
            z_sample=imported_z_sample,
            tags=tags,
        )
        for key, value in attributes.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def store(
        obj: GkWKBIntegration,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        # no row is written whose initial data are not what its parent, or the pure-WKB branch,
        # gives
        sqla_GkWKBIntegration_factory._check_numeric_parent(obj, conn, tables)

        store_id = inserter(
            conn,
            {
                "label": obj.label,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "model_serial": obj.model_proxy.store_id,
                # NULL for a pure-WKB row
                "numeric_serial": obj.numeric_serial,
                # the order the residual table this object's phase came from was actually
                # built at: since prompt 05b the accessor reports the object's own order --
                # carried out of the table by WKB_phase_function on the compute path, read off
                # the row on the rehydration path -- rather than re-reading the module constant.
                # build() goes on filtering on that constant, so a row written at another order
                # is simply a different row (README §7 D10).
                "rho_gauss_order": obj.rho_gauss_order,
                "solver_serial": obj.solver.store_id,
                "z_source_serial": obj.z_source.store_id,
                "z_min_serial": obj.z_sample.min.store_id,
                "z_samples": len(obj.values),
                "z_init": obj.z_init,
                "G_init": obj.G_init,
                "Gprime_init": obj.Gprime_init,
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

        # set store_id on behalf of the GkWKBIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["GkWKB_tags"]
        for tag in obj.tags:
            sqla_GkWKBTagAssociation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sampled output points
        value_inserter = inserters["GkWKBValue"]
        for value in obj.values:
            value: GkWKBValue
            value_id = value_inserter(
                conn,
                {
                    "wkb_serial": store_id,
                    "z_serial": value.z.store_id,
                    "H_ratio": value.H_ratio,
                    "theta_mod_2pi": value.theta_mod_2pi,
                    "theta_div_2pi": value.theta_div_2pi,
                    "G_WKB": value.G_WKB,
                    "omega_WKB_sq": value.omega_WKB_sq,
                    "WKB_criterion": value.WKB_criterion,
                    "analytic_G_rad": value.analytic_G_rad,
                    "analytic_Gprime_rad": value.analytic_Gprime_rad,
                    "analytic_G_w": value.analytic_G_w,
                    "analytic_Gprime_w": value.analytic_Gprime_w,
                },
            )

            # set store_id on behalf of the GkWKBValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def _check_numeric_parent(obj: GkWKBIntegration, conn, tables):
        """
        The store-time consistency check (prompts/datastore-integrity, prompt 07). The row's
        initial data are payload, not key, so nothing else would notice if they were not what its
        parent gives. main.py seeds a row from the GkNumericIntegration it looked up -- z_init =
        z_exit - stop_deltaz_subh, G_init = stop_G, Gprime_init = stop_Gprime -- or starts it pure
        WKB at z_source with G_init = 0, Gprime_init = 1. Here the parent row is read on this shard,
        with its exit time (replicated), and the expression recomputed; for a pure-WKB row the
        z_source row is read instead. Every value must agree bit for bit, and a parent must be for
        this object's k, model and z_source. It lives in store(), which every write goes through, so
        no caller can skip it. A mismatch raises naming both values.
        """
        numeric_serial = obj.numeric_serial
        where = (
            f"GkWKBIntegration.store() (numeric_serial={_parent_name(numeric_serial)}, "
            f"z_source serial={obj.z_source.store_id})"
        )

        if numeric_serial is None:
            # pure WKB: no parent row, so the initial data are the branch's own constants and the
            # source redshift
            redshift_table = tables["redshift"]
            z_source_row = conn.execute(
                sqla.select(redshift_table.c.serial, redshift_table.c.z).filter(
                    redshift_table.c.serial == obj.z_source.store_id
                )
            ).one_or_none()
            if z_source_row is None:
                raise RuntimeError(
                    f"{where}: the z_source redshift is not in the store"
                )
            if obj.G_init != 0.0 or obj.Gprime_init != 1.0:
                raise ValueError(
                    f"{where}: a pure-WKB row starts from G_init=0, Gprime_init=1, but this one "
                    f"has G_init={obj.G_init!r}, Gprime_init={obj.Gprime_init!r}"
                )
            if not _same_bits(obj.z_init, z_source_row.z):
                raise ValueError(
                    f"{where}: z_init={obj.z_init!r} is not the pure-WKB row's z_source="
                    f"{z_source_row.z!r}"
                )
            return

        numeric_table = tables["GkNumericIntegration"]
        exit_table = tables["wavenumber_exit_time"]

        parent = conn.execute(
            sqla.select(
                numeric_table.c.serial,
                numeric_table.c.wavenumber_exit_serial,
                numeric_table.c.model_serial,
                numeric_table.c.z_source_serial,
                numeric_table.c.stop_deltaz_subh,
                numeric_table.c.stop_G,
                numeric_table.c.stop_Gprime,
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

        if parent is None:
            raise RuntimeError(
                f"{where}: no GkNumericIntegration with that serial is on this shard"
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
        if parent.z_source_serial != obj.z_source.store_id:
            raise RuntimeError(
                f"{where}: the parent is for z_source serial {parent.z_source_serial}, this "
                f"object for {obj.z_source.store_id}"
            )
        if parent.z_exit is None or parent.stop_deltaz_subh is None:
            raise RuntimeError(
                f"{where}: the parent's stop point cannot be recomputed (z_exit="
                f"{parent.z_exit!r}, stop_deltaz_subh={parent.stop_deltaz_subh!r})"
            )

        expected_z_init = parent.z_exit - parent.stop_deltaz_subh
        for name, stored, expected, derivation in (
            ("z_init", obj.z_init, expected_z_init, "z_exit - stop_deltaz_subh"),
            ("G_init", obj.G_init, parent.stop_G, "stop_G"),
            ("Gprime_init", obj.Gprime_init, parent.stop_Gprime, "stop_Gprime"),
        ):
            if not _same_bits(stored, expected):
                raise ValueError(
                    f"{where}: {name}={stored!r} is not the parent's {derivation}="
                    f"{expected!r}. A WKB row must start from its numeric parent's stop values, "
                    f"bit for bit."
                )

    @staticmethod
    def validate(
        obj: GkWKBIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in GkWKBIntegration corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["GkWKBValue"]
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

        solver_table = tables["IntegrationSolver"]
        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["GkWKBValue"]
        tags_table = tables["GkWKB_tags"]

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
                    table.c.rho_gauss_order,
                    redshift_table.c.z.label("z_source"),
                )
                .select_from(
                    table.join(
                        solver_table, solver_table.c.serial == table.c.solver_serial
                    )
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
            ">> WKB tensor Green function",
            "     The following unvalidated integrations were detected in the datastore:",
        ]
        for integration in not_validated:
            msgs.append(
                f'       -- "{integration.label}" (store_id={integration.serial}) for k={integration.k_inv_Mpc:.5g}/Mpc and z_source={integration.z_source:.5g} (N_rho={integration.rho_gauss_order})'
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
    def inventory_spec():
        # the physical identity of a GkWKBIntegration row (store-fingerprint prompt 02): what build()
        # filters on -- the model, the wavenumber exit, rho_gauss_order, z_source (optional in the
        # lookup, always in the key) and, since prompts/datastore-integrity prompt 07, the numeric
        # parent (same shard, so a key parent; nullable, and a pure-WKB row's key field is None)
        # -- with its tags, its validated flag and its GkWKBValue count. z_init left the key with
        # prompt 07: it is a function of the parent, or of z_source. The solver is not in the
        # lookup and is not identity
        from datastorekit.store_inventory import InventorySpec, Parent

        return InventorySpec(
            leaves=("rho_gauss_order",),
            parents={
                "model": Parent("model_serial", "BackgroundModel"),
                "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
                "z_source": Parent("z_source_serial", "redshift"),
                "numeric": Parent(
                    "numeric_serial", "GkNumericIntegration", nullable=True
                ),
            },
            tags=("GkWKB_tags", "wkb_serial"),
            values=("GkWKBValue", "wkb_serial"),
            validated="validated",
        )


class sqla_GkWKBValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("GkWKBIntegration.serial"),
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
                sqla.Column("omega_WKB_sq", sqla.Float(64), nullable=True),
                sqla.Column("WKB_criterion", sqla.Float(64), nullable=True),
                sqla.Column("G_WKB", sqla.Float(64), nullable=True),
                sqla.Column("analytic_G_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Gprime_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_G_w", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Gprime_w", sqla.Float(64), nullable=True),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        wkb_serial = payload.get("wkb_serial", None)

        model_proxy: Optional[ModelProxy] = payload.get("model", None)
        k: Optional[wavenumber_exit_time] = payload.get("k", None)
        z_source: Optional[redshift] = payload.get("z_source", None)

        has_serial = all([wkb_serial is not None])
        has_model = all([model_proxy is not None, k is not None, z_source is not None])

        if all([has_serial, has_model]):
            print(
                "## GkWKBValue.build(): both an WKB integration serial number and a (model, wavenumber, z_source) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "GkWKBValue.build(): at least one of a WKB integration serial number and a (model, wavenumber, z_source) set must be supplied."
            )

        if has_serial:
            return sqla_GkWKBValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_GkWKBValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        wkb_serial: int = payload["wkb_serial"]

        H_ratio: Optional[float] = payload.get("H_ratio", None)
        theta_mod_2pi: Optional[float] = payload.get("theta_mod_2pi", None)
        theta_div_2pi: Optional[int] = payload.get("theta_div_2pi", None)
        omega_WKB_sq: Optional[float] = payload.get("omega_WKB_sq", None)
        WKB_criterion: Optional[float] = payload.get("WKB_criterion", None)
        G_WKB: Optional[float] = payload.get("G_WKB", None)
        has_data = all(
            [
                H_ratio is not None,
                theta_mod_2pi is not None,
                theta_div_2pi is not None,
                omega_WKB_sq is not None,
                WKB_criterion is not None,
                G_WKB is not None,
            ]
        )

        analytic_G_rad: Optional[float] = payload.get("analytic_G_rad", None)
        analytic_Gprime_rad: Optional[float] = payload.get("analytic_Gprime_rad", None)

        analytic_G_w: Optional[float] = payload.get("analytic_G_w", None)
        analytic_Gprime_w: Optional[float] = payload.get("analytic_Gprime_w", None)

        sin_coeff: Optional[float] = payload.get("sin_coeff", None)
        cos_coeff: Optional[float] = payload.get("cos_coeff", None)
        z_init: Optional[float] = payload.get("z_init", None)

        wkb_table = tables["GkWKBIntegration"]

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.H_ratio,
                    table.c.theta_mod_2pi,
                    table.c.theta_div_2pi,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                    table.c.G_WKB,
                    table.c.analytic_G_rad,
                    table.c.analytic_Gprime_rad,
                    table.c.analytic_G_w,
                    table.c.analytic_Gprime_w,
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
                f"!! GkWKBValue.build(): multiple results found when querying for GkWKBValue"
            )
            raise e

        if row_data is None:
            if not has_data:
                raise RuntimeError(
                    "GkWKBValue().build(): result was not found in datastore, but a data payload was not provided"
                )

            store_id = inserter(
                conn,
                {
                    "wkb_serial": wkb_serial,
                    "z_serial": z.store_id,
                    "H_ratio": H_ratio,
                    "theta_mod_2pi": theta_mod_2pi,
                    "theta_div_2pi": theta_div_2pi,
                    "omega_WKB_sq": omega_WKB_sq,
                    "WKB_criterion": WKB_criterion,
                    "G_WKB": G_WKB,
                    "analytic_G_rad": analytic_G_rad,
                    "analytic_Gprime_rad": analytic_Gprime_rad,
                    "analytic_G_w": analytic_G_w,
                    "analytic_Gprime_w": analytic_Gprime_w,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial
            omega_WKB_sq = row_data.omega_WKB_sq
            WKB_criterion = row_data.WKB_criterion

            G_WKB = row_data.G_WKB

            analytic_G_rad = row_data.analytic_G_rad
            analytic_Gprime_rad = row_data.analytic_Gprime_rad
            analytic_G_w = row_data.analytic_G_w
            analytic_Gprime_w = row_data.analytic_Gprime_w

            sin_coeff = row_data.sin_coeff
            cos_coeff = row_data.cos_coeff
            z_init = row_data.z_init

            # we choose H_ratio and theta to test because these are the critical data to reconstruct
            # G_WKB; everything else, such as omega_WKB_sq, is optional
            if (
                H_ratio is not None
                and fabs(row_data.H_ratio - H_ratio) > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"GkWKBValue.build(): Stored H_ratio ratio (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.H_ratio} differs from expected value = {H_ratio}"
                )
            if (
                theta_mod_2pi is not None
                and fabs(row_data.theta_mod_2pi - theta_mod_2pi)
                > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"GkWKBValue.build(): Stored theta mod 2pi (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.theta_mod_2pi} differs from expected value = {theta_mod_2pi}"
                )
            if theta_div_2pi is not None and row_data.theta_div_2pi != theta_div_2pi:
                raise ValueError(
                    f"GkWKBValue.build(): Stored WKB theta div 2pi (WKB store_id={wkb_serial}, z={z.store_id}) = {row_data.theta_div_2pi} differs from expected value = {theta_div_2pi}"
                )

            H_ratio = row_data.H_ratio
            theta_mod_2pi = row_data.theta_mod_2pi
            theta_div_2pi = row_data.theta_div_2pi

            attribute_set = {"_deserialized": True}

        obj = GkWKBValue(
            store_id=store_id,
            z=z,
            H_ratio=H_ratio,
            theta_mod_2pi=theta_mod_2pi,
            theta_div_2pi=theta_div_2pi,
            omega_WKB_sq=omega_WKB_sq,
            WKB_criterion=WKB_criterion,
            G_WKB=G_WKB,
            analytic_G_rad=analytic_G_rad,
            analytic_Gprime_rad=analytic_Gprime_rad,
            analytic_G_w=analytic_G_w,
            analytic_Gprime_w=analytic_Gprime_w,
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
        z_source: redshift = payload["z_source"]
        numeric_serial = payload.get("numeric_serial", None)

        tags: List[store_tag] = payload.get("tags", [])

        wkb_table = tables["GkWKBIntegration"]

        try:
            # TODO: benchmarking suggests this query is indistinguishable from filtering directly on the
            #  GkWKBIntegration serial number (if we only knew what it was), so this may be about
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
                wkb_table.c.z_source_serial == z_source.store_id,
                wkb_table.c.validated == True,
            )

            wkb_query = wkb_query.filter(
                wkb_table.c.rho_gauss_order == phase_residual.RHO_GAUSS_ORDER
            )

            # the numeric parent (or the pure-WKB marker) is in the parent's key when it is given
            # (prompt 07, in place of 05c's z_init). With none, the read spans every parent
            if numeric_serial is not None:
                wkb_query = wkb_query.filter(
                    _numeric_parent_filter(wkb_table.c.numeric_serial, numeric_serial)
                )

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["GkWKB_tags"].alias(f"tag_{count}")
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
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                    table.c.G_WKB,
                    table.c.analytic_G_rad,
                    table.c.analytic_Gprime_rad,
                    table.c.analytic_G_w,
                    table.c.analytic_Gprime_w,
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
                f"!! GkWKBValue.build(): multiple results found when querying for GkWKBValue"
            )
            raise e

        if row_data is None:
            # return empty object
            obj = GkWKBValue(
                store_id=None, z=z, H_ratio=None, theta_mod_2pi=None, theta_div_2pi=None
            )
            obj._k_exit = k
            obj._z_source = z_source
            return obj

        obj = GkWKBValue(
            store_id=row_data.serial,
            z=z,
            H_ratio=row_data.H_ratio,
            theta_mod_2pi=row_data.theta_mod_2pi,
            theta_div_2pi=row_data.theta_div_2pi,
            omega_WKB_sq=row_data.omega_WKB_sq,
            WKB_criterion=row_data.WKB_criterion,
            G_WKB=row_data.G_WKB,
            analytic_G_rad=row_data.analytic_G_rad,
            analytic_Gprime_rad=row_data.analytic_Gprime_rad,
            analytic_G_w=row_data.analytic_G_w,
            analytic_Gprime_w=row_data.analytic_Gprime_w,
            sin_coeff=row_data.sin_coeff,
            cos_coeff=row_data.cos_coeff,
            z_init=row_data.z_init,
        )
        obj._deserialized = True
        obj._k_exit = k
        obj._z_source = z_source
        return obj

    @staticmethod
    def read_batch(payload, conn, table, tables):
        model_proxy: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        tags: List[store_tag] = payload.get("tags", [])

        z_response: Optional[redshift] = payload.get("z", None)
        z_source: Optional[redshift] = payload.get("z_source", None)
        numeric_serial = payload.get("numeric_serial", None)

        # the serial-only form (prompts/datastore-integrity, prompt 09b): no values, only the
        # (z_source serial, GkWKBIntegration serial) pairs, from which main.py forms the parent
        # set a GkSource is keyed on before it asks whether that GkSource exists
        parents_only: bool = payload.get("parents_only", False)

        wkb_table = tables["GkWKBIntegration"]
        redshift_table = tables["redshift"]

        wkb_query = (
            sqla.select(
                wkb_table.c.serial,
                wkb_table.c.sin_coeff,
                wkb_table.c.cos_coeff,
                wkb_table.c.z_init,
                wkb_table.c.z_source_serial,
                redshift_table.c.z.label("z_source"),
                redshift_table.c.source.label("z_source_is_source"),
                redshift_table.c.response.label("z_source_is_response"),
            )
            .select_from(
                wkb_table.join(
                    redshift_table,
                    redshift_table.c.serial == wkb_table.c.z_source_serial,
                )
            )
            .filter(
                wkb_table.c.model_serial == model_proxy.store_id,
                wkb_table.c.wavenumber_exit_serial == k.store_id,
                wkb_table.c.validated == True,
            )
        )

        if z_source is not None:
            wkb_query = wkb_query.filter(
                wkb_table.c.z_source_serial == z_source.store_id
            )

        wkb_query = wkb_query.filter(
            wkb_table.c.rho_gauss_order == phase_residual.RHO_GAUSS_ORDER
        )

        # the numeric parent (or the pure-WKB marker) is in the parent's key when it is given
        # (prompt 07, in place of 05c's z_init). With none, the read spans every parent, and a
        # second one for a z_source is refused below
        if numeric_serial is not None:
            wkb_query = wkb_query.filter(
                _numeric_parent_filter(wkb_table.c.numeric_serial, numeric_serial)
            )

        count = 0
        for tag in tags:
            tag: store_tag
            tab = tables["GkWKB_tags"].alias(f"tag_{count}")
            count += 1
            wkb_query = wkb_query.join(
                tab,
                and_(
                    tab.c.wkb_serial == wkb_table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        subquery = wkb_query.subquery()

        if parents_only:
            row_query = (
                sqla.select(subquery.c.z_source_serial, table.c.wkb_serial)
                .distinct()
                .select_from(
                    subquery.join(table, table.c.wkb_serial == subquery.c.serial)
                )
            )
        else:
            row_query = sqla.select(
                table.c.serial,
                # the parent serial, selected so that a second parent for one z_source can be
                # refused from these rows, without a second query, and handed back on each value
                # (prompt 09b)
                table.c.wkb_serial,
                table.c.H_ratio,
                table.c.theta_mod_2pi,
                table.c.theta_div_2pi,
                table.c.omega_WKB_sq,
                table.c.WKB_criterion,
                table.c.G_WKB,
                table.c.analytic_G_rad,
                table.c.analytic_Gprime_rad,
                table.c.analytic_G_w,
                table.c.analytic_Gprime_w,
                subquery.c.sin_coeff,
                subquery.c.cos_coeff,
                subquery.c.z_init,
                subquery.c.z_source_serial,
                subquery.c.z_source,
                subquery.c.z_source_is_source,
                subquery.c.z_source_is_response,
                redshift_table.c.z.label("z_response"),
                table.c.z_serial.label("z_response_serial"),
                redshift_table.c.source.label("z_response_is_source"),
                redshift_table.c.response.label("z_response_is_response"),
            ).select_from(
                subquery.join(table, table.c.wkb_serial == subquery.c.serial).join(
                    redshift_table, redshift_table.c.serial == table.c.z_serial
                )
            )

        if z_response is not None:
            row_query = row_query.filter(table.c.z_serial == z_response.store_id)

        row_data = conn.execute(row_query).all()

        # D1 (prompts/datastore-integrity README §6.2): a read that holds only physical labels may
        # select by them, but must refuse when more than one stored parent matches, rather than
        # return a mixture. The caller files each value under its z_source, so two parents for one
        # z_source would silently overwrite one another (prompt 05c)
        parents_by_z_source = {}
        for row in row_data:
            parents_by_z_source.setdefault(row.z_source_serial, set()).add(
                row.wkb_serial
            )
        for z_source_serial, parents in parents_by_z_source.items():
            if len(parents) > 1:
                raise MultipleResultsFound(
                    f"GkWKBValue.read_batch(): {len(parents)} GkWKBIntegration parents match one "
                    f"key (k serial={k.store_id}, z_source serial={z_source_serial}): parent "
                    f"serials {sorted(parents)}. Refusing to return values from more than one "
                    f"parent."
                )

        # the serial-only form hands back the pairs, and no value (prompt 09b)
        if parents_only:
            pairs = set()
            for row in row_data:
                pairs.add((row.z_source_serial, row.wkb_serial))
            return sorted(pairs)

        def make_obj(row):
            obj = GkWKBValue(
                store_id=row.serial,
                z=redshift(
                    store_id=row.z_response_serial,
                    z=row.z_response,
                    is_source=row.z_response_is_source,
                    is_response=row.z_response_is_response,
                ),
                H_ratio=row.H_ratio,
                theta_mod_2pi=row.theta_mod_2pi,
                theta_div_2pi=row.theta_div_2pi,
                omega_WKB_sq=row.omega_WKB_sq,
                WKB_criterion=row.WKB_criterion,
                G_WKB=row.G_WKB,
                analytic_G_rad=row.analytic_G_rad,
                analytic_Gprime_rad=row.analytic_Gprime_rad,
                analytic_G_w=row.analytic_G_w,
                analytic_Gprime_w=row.analytic_Gprime_w,
                sin_coeff=row.sin_coeff,
                cos_coeff=row.cos_coeff,
                z_init=row.z_init,
            )
            obj._deserialized = True
            obj._k_exit = k
            obj._z_source = redshift(
                store_id=row.z_source_serial,
                z=row.z_source,
                is_source=row.z_source_is_source,
                is_response=row.z_source_is_response,
            )
            # the GkWKBIntegration this value was read from, so that a caller can form the parent
            # set of what it assembles from it (prompt 09b)
            obj._parent_serial = row.wkb_serial

            return obj

        objects = [make_obj(row) for row in row_data]
        return objects
