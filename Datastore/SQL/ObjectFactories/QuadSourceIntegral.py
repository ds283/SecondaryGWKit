import json
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import (
    QuadSourceIntegral,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber
from CosmologyConcepts.wavenumber import (
    WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
    WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
)
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance, GkSourcePolicy
from Quadrature.integration_metadata import IntegrationData, LevinData
from defaults import DEFAULT_STRING_LENGTH


class sqla_QuadSourceIntegralTagAssociation_factory(SQLAFactoryBase):
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
                    "parent_serial",
                    sqla.Integer,
                    sqla.ForeignKey("QuadSourceIntegral.serial"),
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
    def add_tag(conn, inserter, source: QuadSourceIntegral, tag: store_tag):
        inserter(
            conn,
            {
                "parent_serial": source.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, source: QuadSourceIntegral, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.parent_serial == source.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": True,
            "stepping": False,
            "timestamp": True,
            "validate_on_startup": False,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column(
                    "model_serial",
                    sqla.Integer,
                    sqla.ForeignKey("BackgroundModel.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "k_wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "q_wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "r_wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
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
                    "policy_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkSourcePolicy.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "source_serial",
                    sqla.Integer,  # Don't impose foreign key. QuadSource instance may not be held on this shard.
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "data_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkSourcePolicyData.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_response_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_source_max_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("total", sqla.Float(64), nullable=False),
                sqla.Column("numeric_quad", sqla.Float(64), nullable=False),
                sqla.Column("WKB_quad", sqla.Float(64), nullable=False),
                sqla.Column("WKB_Levin", sqla.Float(64), nullable=False),
                sqla.Column("analytic_rad", sqla.Float(64), nullable=True),
                sqla.Column("eta_source_max", sqla.Float(64), nullable=True),
                sqla.Column("eta_response", sqla.Float(64), nullable=True),
                sqla.Column("numeric_quad_compute_time", sqla.Float(64), nullable=True),
                sqla.Column("numeric_quad_compute_steps", sqla.Integer, nullable=True),
                sqla.Column(
                    "numeric_quad_RHS_evaluations", sqla.Integer, nullable=True
                ),
                sqla.Column(
                    "numeric_quad_mean_RHS_time", sqla.Float(64), nullable=True
                ),
                sqla.Column("numeric_quad_max_RHS_time", sqla.Float(64), nullable=True),
                sqla.Column("numeric_quad_min_RHS_time", sqla.Float(64), nullable=True),
                sqla.Column("WKB_quad_compute_time", sqla.Float(64), nullable=True),
                sqla.Column("WKB_quad_compute_steps", sqla.Integer, nullable=True),
                sqla.Column("WKB_quad_RHS_evaluations", sqla.Integer, nullable=True),
                sqla.Column("WKB_quad_mean_RHS_time", sqla.Float(64), nullable=True),
                sqla.Column("WKB_quad_max_RHS_time", sqla.Float(64), nullable=True),
                sqla.Column("WKB_quad_min_RHS_time", sqla.Float(64), nullable=True),
                sqla.Column("WKB_Levin_num_regions", sqla.Integer, nullable=True),
                sqla.Column("WKB_Levin_evaluations", sqla.Integer, nullable=True),
                sqla.Column("WKB_Levin_simple_regions", sqla.Integer, nullable=True),
                sqla.Column("WKB_Levin_SVD_errors", sqla.Integer, nullable=True),
                sqla.Column("WKB_Levin_order_changes", sqla.Integer, nullable=True),
                sqla.Column("WKB_Levin_elapsed", sqla.Float(64), nullable=True),
                sqla.Column("WKB_phase_spline_chunks", sqla.Integer, nullable=True),
                sqla.Column("compute_time", sqla.Float(64), nullable=True),
                sqla.Column("analytic_compute_time", sqla.Float(64), nullable=True),
                sqla.Column(
                    "metadata", sqla.String(DEFAULT_STRING_LENGTH), nullable=True
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        model_proxy: ModelProxy = payload["model"]
        policy: GkSourcePolicy = payload["policy"]

        k: wavenumber_exit_time = payload["k"]
        q: wavenumber_exit_time = payload["q"]
        r: wavenumber_exit_time = payload["r"]
        z_response: redshift = payload["z_response"]
        z_source_max: redshift = payload["z_source_max"]

        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        tag_table = tables["QuadSourceIntegral_tags"]

        query = sqla.select(
            table.c.serial,
            table.c.compute_time,
            table.c.analytic_compute_time,
            table.c.label,
            table.c.metadata,
            table.c.source_serial,
            table.c.total,
            table.c.numeric_quad,
            table.c.WKB_quad,
            table.c.WKB_Levin,
            table.c.analytic_rad,
            table.c.eta_source_max,
            table.c.eta_response,
            table.c.data_serial,
            table.c.numeric_quad_compute_time,
            table.c.numeric_quad_compute_steps,
            table.c.numeric_quad_RHS_evaluations,
            table.c.numeric_quad_mean_RHS_time,
            table.c.numeric_quad_max_RHS_time,
            table.c.numeric_quad_min_RHS_time,
            table.c.WKB_quad_compute_time,
            table.c.WKB_quad_compute_steps,
            table.c.WKB_quad_RHS_evaluations,
            table.c.WKB_quad_mean_RHS_time,
            table.c.WKB_quad_max_RHS_time,
            table.c.WKB_quad_min_RHS_time,
            table.c.WKB_Levin_num_regions,
            table.c.WKB_Levin_evaluations,
            table.c.WKB_Levin_simple_regions,
            table.c.WKB_Levin_SVD_errors,
            table.c.WKB_Levin_order_changes,
            table.c.WKB_Levin_elapsed,
            table.c.WKB_phase_spline_chunks,
        ).filter(
            table.c.model_serial == model_proxy.store_id,
            table.c.policy_serial == policy.store_id,
            table.c.k_wavenumber_exit_serial == k.store_id,
            table.c.q_wavenumber_exit_serial == q.store_id,
            table.c.r_wavenumber_exit_serial == r.store_id,
            table.c.z_response_serial == z_response.store_id,
            table.c.z_source_max_serial == z_source_max.store_id,
            table.c.atol_serial == atol.store_id,
            table.c.rtol_serial == rtol.store_id,
        )

        # require that the queried calculation has any specified tags
        count = 0
        for tag in tags:
            tag: store_tag
            tab = tag_table.alias(f"tag_{count}")
            count += 1
            query = query.join(
                tab,
                and_(
                    tab.c.parent_serial == table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! QuadSourceIntegral.build(): multiple results found when querying for QuadSourceIntegral (k={k.k.k_inv_Mpc:.5g}/Mpc, store_id={k.store_id} | q={q.k.k_inv_Mpc:.5g}/Mpc, store_id={q.store_id} | r={r.k.k_inv_Mpc:.5g}/Mpc, store_id={r.store_id})"
            )
            raise e

        if row_data is None:
            return QuadSourceIntegral(
                payload=None,
                model=model_proxy,
                policy=policy,
                z_response=z_response,
                z_source_max=z_source_max,
                k=k,
                q=q,
                r=r,
                atol=atol,
                rtol=rtol,
                label=label,
                tags=tags,
            )

        obj = QuadSourceIntegral(
            payload={
                "store_id": row_data.serial,
                "total": row_data.total,
                "numeric_quad": row_data.numeric_quad,
                "WKB_quad": row_data.WKB_quad,
                "WKB_Levin": row_data.WKB_Levin,
                "analytic_rad": row_data.analytic_rad,
                "eta_source_max": row_data.eta_source_max,
                "eta_response": row_data.eta_response,
                "numeric_quad_data": IntegrationData(
                    compute_time=row_data.numeric_quad_compute_time,
                    compute_steps=row_data.numeric_quad_compute_steps,
                    RHS_evaluations=row_data.numeric_quad_RHS_evaluations,
                    mean_RHS_time=row_data.numeric_quad_mean_RHS_time,
                    max_RHS_time=row_data.numeric_quad_max_RHS_time,
                    min_RHS_time=row_data.numeric_quad_min_RHS_time,
                ),
                "WKB_quad_data": IntegrationData(
                    compute_time=row_data.WKB_quad_compute_time,
                    compute_steps=row_data.WKB_quad_compute_steps,
                    RHS_evaluations=row_data.WKB_quad_RHS_evaluations,
                    mean_RHS_time=row_data.WKB_quad_mean_RHS_time,
                    max_RHS_time=row_data.WKB_quad_max_RHS_time,
                    min_RHS_time=row_data.WKB_quad_min_RHS_time,
                ),
                "WKB_Levin_data": LevinData(
                    num_regions=row_data.WKB_Levin_num_regions,
                    evaluations=row_data.WKB_Levin_evaluations,
                    num_simple_regions=row_data.WKB_Levin_simple_regions,
                    num_SVD_errors=row_data.WKB_Levin_SVD_errors,
                    num_order_changes=row_data.WKB_Levin_order_changes,
                    elapsed=row_data.WKB_Levin_elapsed,
                ),
                "WKB_phase_spline_chunks": row_data.WKB_phase_spline_chunks,
                "compute_time": row_data.compute_time,
                "analytic_compute_time": row_data.analytic_compute_time,
                "data_serial": row_data.data_serial,
                "source_serial": row_data.source_serial,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
            },
            model=model_proxy,
            policy=policy,
            z_response=z_response,
            z_source_max=z_source_max,
            k=k,
            q=q,
            r=r,
            atol=atol,
            rtol=rtol,
            label=row_data.label,
            tags=tags,
        )
        obj._deserialized = True
        return obj

    @staticmethod
    def store(
        obj: QuadSourceIntegral,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        numeric_quad_data = obj.numeric_quad_data
        WKB_quad_data = obj.WKB_quad_data
        WKB_Levin_data = obj.WKB_Levin_data

        try:
            store_id = inserter(
                conn,
                {
                    "label": obj.label,
                    "model_serial": obj._model_proxy.store_id,
                    "z_response_serial": obj._z_response.store_id,
                    "z_source_max_serial": obj._z_source_max.store_id,
                    "k_wavenumber_exit_serial": obj._k_exit.store_id,
                    "q_wavenumber_exit_serial": obj._q_exit.store_id,
                    "r_wavenumber_exit_serial": obj._r_exit.store_id,
                    "atol_serial": obj._atol.store_id,
                    "rtol_serial": obj._rtol.store_id,
                    "policy_serial": obj._policy.store_id,
                    "source_serial": obj._source_serial,
                    "data_serial": obj._data_serial,
                    "compute_time": obj.compute_time,
                    "analytic_compute_time": obj.analytic_compute_time,
                    "total": obj._total,
                    "numeric_quad": obj._numeric_quad,
                    "WKB_quad": obj._WKB_quad,
                    "WKB_Levin": obj._WKB_Levin,
                    "analytic_rad": obj._analytic_rad,
                    "eta_source_max": obj._eta_source_max,
                    "eta_response": obj._eta_response,
                    "numeric_quad_compute_time": (
                        numeric_quad_data.compute_time
                        if numeric_quad_data is not None
                        else None
                    ),
                    "numeric_quad_compute_steps": (
                        numeric_quad_data.compute_steps
                        if numeric_quad_data is not None
                        else None
                    ),
                    "numeric_quad_RHS_evaluations": (
                        numeric_quad_data.RHS_evaluations
                        if numeric_quad_data is not None
                        else None
                    ),
                    "numeric_quad_mean_RHS_time": (
                        numeric_quad_data.mean_RHS_time
                        if numeric_quad_data is not None
                        else None
                    ),
                    "numeric_quad_max_RHS_time": (
                        numeric_quad_data.max_RHS_time
                        if numeric_quad_data is not None
                        else None
                    ),
                    "numeric_quad_min_RHS_time": (
                        numeric_quad_data.min_RHS_time
                        if numeric_quad_data is not None
                        else None
                    ),
                    "WKB_quad_compute_time": (
                        WKB_quad_data.compute_time
                        if WKB_quad_data is not None
                        else None
                    ),
                    "WKB_quad_compute_steps": (
                        WKB_quad_data.compute_steps
                        if WKB_quad_data is not None
                        else None
                    ),
                    "WKB_quad_RHS_evaluations": (
                        WKB_quad_data.RHS_evaluations
                        if WKB_quad_data is not None
                        else None
                    ),
                    "WKB_quad_mean_RHS_time": (
                        WKB_quad_data.mean_RHS_time
                        if WKB_quad_data is not None
                        else None
                    ),
                    "WKB_quad_max_RHS_time": (
                        WKB_quad_data.max_RHS_time
                        if WKB_quad_data is not None
                        else None
                    ),
                    "WKB_quad_min_RHS_time": (
                        WKB_quad_data.min_RHS_time
                        if WKB_quad_data is not None
                        else None
                    ),
                    "WKB_Levin_num_regions": (
                        WKB_Levin_data.num_regions
                        if WKB_Levin_data is not None
                        else None
                    ),
                    "WKB_Levin_evaluations": (
                        WKB_Levin_data.evaluations
                        if WKB_Levin_data is not None
                        else None
                    ),
                    "WKB_Levin_simple_regions": (
                        WKB_Levin_data.num_simple_regions
                        if WKB_Levin_data is not None
                        else None
                    ),
                    "WKB_Levin_SVD_errors": (
                        WKB_Levin_data.num_SVD_errors
                        if WKB_Levin_data is not None
                        else None
                    ),
                    "WKB_Levin_order_changes": (
                        WKB_Levin_data.num_order_changes
                        if WKB_Levin_data is not None
                        else None
                    ),
                    "WKB_Levin_elapsed": (
                        WKB_Levin_data.elapsed if WKB_Levin_data is not None else None
                    ),
                    "WKB_phase_spline_chunks": obj._WKB_phase_spline_chunks,
                    "metadata": (
                        json.dumps(obj.metadata) if obj._metadata is not None else None
                    ),
                },
            )
        except TypeError as e:
            print(obj._metadata)
            raise e

        # set store_id on behalf of the QuadSourceIntegration instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["QuadSourceIntegral_tags"]
        for tag in obj.tags:
            sqla_QuadSourceIntegralTagAssociation_factory.add_tag(
                conn, tag_inserter, obj, tag
            )

        return obj

    @staticmethod
    def read_batch(payload, conn, table, tables):
        model_proxy: ModelProxy = payload["model"]
        policy: GkSourcePolicy = payload["policy"]

        k: wavenumber_exit_time = payload["k"]
        q: Optional[wavenumber_exit_time] = payload.get("q", None)
        r: Optional[wavenumber_exit_time] = payload.get("r", None)

        z_response: Optional[redshift] = payload.get("z_response", None)
        z_source_max: Optional[redshift] = payload.get("z_source_max", None)

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        tags: List[store_tag] = payload.get("tags", [])

        z_response_tab = tables["redshift"].alias("z_resp_tab")
        z_source_max_tab = tables["redshift"].alias("z_src_max_tab")
        q_exit_tab = tables["wavenumber_exit_time"].alias("q_exit")
        r_exit_tab = tables["wavenumber_exit_time"].alias("r_exit")
        q_wavenumber_tab = tables["wavenumber"].alias("q_tab")
        r_wavenumber_tab = tables["wavenumber"].alias("r_tab")
        q_atol_tab = tables["tolerance"].alias("q_atol")
        q_rtol_tab = tables["tolerance"].alias("q_rtol")
        r_atol_tab = tables["tolerance"].alias("r_atol")
        r_rtol_tab = tables["tolerance"].alias("r_rtol")

        query = (
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.analytic_compute_time,
                table.c.label,
                table.c.metadata,
                table.c.source_serial,
                table.c.data_serial,
                table.c.total,
                table.c.numeric_quad,
                table.c.WKB_quad,
                table.c.WKB_Levin,
                table.c.analytic_rad,
                table.c.eta_source_max,
                table.c.eta_response,
                table.c.numeric_quad_compute_time,
                table.c.numeric_quad_compute_steps,
                table.c.numeric_quad_RHS_evaluations,
                table.c.numeric_quad_mean_RHS_time,
                table.c.numeric_quad_max_RHS_time,
                table.c.numeric_quad_min_RHS_time,
                table.c.WKB_quad_compute_time,
                table.c.WKB_quad_compute_steps,
                table.c.WKB_quad_RHS_evaluations,
                table.c.WKB_quad_mean_RHS_time,
                table.c.WKB_quad_max_RHS_time,
                table.c.WKB_quad_min_RHS_time,
                table.c.WKB_Levin_num_regions,
                table.c.WKB_Levin_evaluations,
                table.c.WKB_Levin_simple_regions,
                table.c.WKB_Levin_SVD_errors,
                table.c.WKB_Levin_order_changes,
                table.c.WKB_Levin_elapsed,
                table.c.WKB_phase_spline_chunks,
                table.c.z_response_serial,
                z_response_tab.c.z.label("z_response"),
                z_response_tab.c.source.label("z_response_is_source"),
                z_response_tab.c.response.label("z_response_is_response"),
                table.c.z_source_max_serial,
                z_source_max_tab.c.z.label("z_source_max"),
                z_source_max_tab.c.source.label("z_source_max_is_source"),
                z_source_max_tab.c.response.label("z_source_max_is_response"),
                q_wavenumber_tab.c.serial.label("q_serial"),
                q_wavenumber_tab.c.k_inv_Mpc.label("q_inv_Mpc"),
                q_wavenumber_tab.c.source.label("q_is_source"),
                q_wavenumber_tab.c.response.label("q_is_response"),
                r_wavenumber_tab.c.serial.label("r_serial"),
                r_wavenumber_tab.c.k_inv_Mpc.label("r_inv_Mpc"),
                r_wavenumber_tab.c.source.label("r_is_source"),
                r_wavenumber_tab.c.response.label("r_is_response"),
                q_exit_tab.c.stepping.label("q_stepping"),
                q_exit_tab.c.atol_serial.label("q_atol_serial"),
                q_exit_tab.c.rtol_serial.label("q_rtol_serial"),
                q_exit_tab.c.compute_time.label("q_compute_time"),
                q_atol_tab.c.log10_tol.label("q_log10_atol"),
                q_rtol_tab.c.log10_tol.label("q_log10_rtol"),
                q_exit_tab.c.z_exit.label("q_z_exit"),
                r_exit_tab.c.stepping.label("r_stepping"),
                r_exit_tab.c.atol_serial.label("r_atol_serial"),
                r_exit_tab.c.rtol_serial.label("r_rtol_serial"),
                r_exit_tab.c.compute_time.label("r_compute_time"),
                r_atol_tab.c.log10_tol.label("r_log10_atol"),
                r_rtol_tab.c.log10_tol.label("r_log10_rtol"),
                r_exit_tab.c.z_exit.label("r_z_exit"),
            )
            .select_from(
                table.join(
                    z_response_tab,
                    z_response_tab.c.serial == table.c.z_response_serial,
                )
                .join(
                    z_source_max_tab,
                    z_source_max_tab.c.serial == table.c.z_source_max_serial,
                )
                .join(
                    q_exit_tab, q_exit_tab.c.serial == table.c.q_wavenumber_exit_serial
                )
                .join(
                    r_exit_tab, r_exit_tab.c.serial == table.c.r_wavenumber_exit_serial
                )
                .join(
                    q_wavenumber_tab,
                    q_wavenumber_tab.c.serial == q_exit_tab.c.wavenumber_serial,
                )
                .join(
                    r_wavenumber_tab,
                    r_wavenumber_tab.c.serial == r_exit_tab.c.wavenumber_serial,
                )
                .join(q_atol_tab, q_atol_tab.c.serial == q_exit_tab.c.atol_serial)
                .join(q_rtol_tab, q_rtol_tab.c.serial == q_exit_tab.c.rtol_serial)
                .join(r_atol_tab, r_atol_tab.c.serial == r_exit_tab.c.atol_serial)
                .join(r_rtol_tab, r_rtol_tab.c.serial == r_exit_tab.c.rtol_serial)
            )
            .filter(
                table.c.model_serial == model_proxy.store_id,
                table.c.policy_serial == policy.store_id,
                table.c.k_wavenumber_exit_serial == k.store_id,
                table.c.atol_serial == atol.store_id,
                table.c.rtol_serial == rtol.store_id,
            )
        )

        for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
            query = query.add_columns(
                q_exit_tab.c[f"z_exit_suph_e{z_offset}"].label(
                    f"q_z_exit_suph_e{z_offset}"
                ),
                r_exit_tab.c[f"z_exit_suph_e{z_offset}"].label(
                    f"r_z_exit_suph_e{z_offset}"
                ),
            )
        for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
            query = query.add_columns(
                q_exit_tab.c[f"z_exit_subh_e{z_offset}"].label(
                    f"q_z_exit_subh_e{z_offset}"
                ),
                r_exit_tab.c[f"z_exit_subh_e{z_offset}"].label(
                    f"r_z_exit_subh_e{z_offset}"
                ),
            )

        if q is not None:
            query = query.filter(
                table.c.q_wavenumber_exit_serial == q.store_id,
            )

        if r is not None:
            query = query.filter(
                table.c.r_wavenumber_exit_serial == r.store_id,
            )

        if z_response is not None:
            query = query.filter(
                table.c.z_response_serial == z_response.store_id,
            )

        if z_source_max is not None:
            query = query.filter(
                table.c.z_source_max_serial == z_source_max.store_id,
            )

        count = 0
        for tag in tags:
            tag: store_tag
            tab = tables["QuadSourceIntegral_tags"].alias(f"tag_{count}")
            count += 1
            query = query.join(
                tab,
                and_(
                    tab.c.parent_serial == query.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        row_data = conn.execute(query)

        def make_object(row):
            q: wavenumber = wavenumber(
                store_id=row.q_serial,
                k_inv_Mpc=row.q_inv_Mpc,
                units=model_proxy.units,
                is_source=row.q_is_source,
                is_response=row.q_is_response,
            )
            r: wavenumber = wavenumber(
                store_id=row.r_serial,
                k_inv_Mpc=row.r_inv_Mpc,
                units=model_proxy.units,
                is_source=row.r_is_source,
                is_response=row.r_is_response,
            )

            q_exit_payload = {
                "store_id": row.q_serial,
                "z_exit": row.q_z_exit,
                "compute_time": row.q_compute_time,
                "stepping": row.q_stepping,
            }
            r_exit_payload = {
                "store_id": row.r_serial,
                "z_exit": row.r_z_exit,
                "compute_time": row.r_compute_time,
                "stepping": row.r_stepping,
            }
            for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                q_exit_payload[f"z_exit_suph_e{z_offset}"] = row._mapping[
                    f"q_z_exit_suph_e{z_offset}"
                ]
                r_exit_payload[f"z_exit_suph_e{z_offset}"] = row._mapping[
                    f"r_z_exit_suph_e{z_offset}"
                ]
            for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                q_exit_payload[f"z_exit_subh_e{z_offset}"] = row._mapping[
                    f"q_z_exit_subh_e{z_offset}"
                ]
                r_exit_payload[f"z_exit_subh_e{z_offset}"] = row._mapping[
                    f"r_z_exit_subh_e{z_offset}"
                ]

            q_exit: wavenumber_exit_time = wavenumber_exit_time(
                payload=q_exit_payload,
                k=q,
                cosmology=model_proxy.cosmology,
                atol=tolerance(store_id=row.q_atol_serial, log10_tol=row.q_log10_atol),
                rtol=tolerance(store_id=row.q_rtol_serial, log10_tol=row.q_log10_rtol),
            )
            r_exit: wavenumber_exit_time = wavenumber_exit_time(
                payload=r_exit_payload,
                k=r,
                cosmology=model_proxy.cosmology,
                atol=tolerance(store_id=row.r_atol_serial, log10_tol=row.r_log10_atol),
                rtol=tolerance(store_id=row.r_rtol_serial, log10_tol=row.r_log10_rtol),
            )

            obj = QuadSourceIntegral(
                payload={
                    "store_id": row.serial,
                    "total": row.total,
                    "numeric_quad": row.numeric_quad,
                    "WKB_quad": row.WKB_quad,
                    "WKB_Levin": row.WKB_Levin,
                    "analytic_rad": row.analytic_rad,
                    "eta_source_max": row.eta_source_max,
                    "eta_response": row.eta_response,
                    "numeric_quad_data": IntegrationData(
                        compute_time=row.numeric_quad_compute_time,
                        compute_steps=row.numeric_quad_compute_steps,
                        RHS_evaluations=row.numeric_quad_RHS_evaluations,
                        mean_RHS_time=row.numeric_quad_mean_RHS_time,
                        max_RHS_time=row.numeric_quad_max_RHS_time,
                        min_RHS_time=row.numeric_quad_min_RHS_time,
                    ),
                    "WKB_quad_data": IntegrationData(
                        compute_time=row.WKB_quad_compute_time,
                        compute_steps=row.WKB_quad_compute_steps,
                        RHS_evaluations=row.WKB_quad_RHS_evaluations,
                        mean_RHS_time=row.WKB_quad_mean_RHS_time,
                        max_RHS_time=row.WKB_quad_max_RHS_time,
                        min_RHS_time=row.WKB_quad_min_RHS_time,
                    ),
                    "WKB_Levin_data": LevinData(
                        num_regions=row.WKB_Levin_num_regions,
                        evaluations=row.WKB_Levin_evaluations,
                        num_simple_regions=row.WKB_Levin_simple_regions,
                        num_SVD_errors=row.WKB_Levin_SVD_errors,
                        num_order_changes=row.WKB_Levin_order_changes,
                        elapsed=row.WKB_Levin_elapsed,
                    ),
                    "WKB_phase_spline_chunks": row.WKB_phase_spline_chunks,
                    "compute_time": row.compute_time,
                    "analytic_compute_time": row.analytic_compute_time,
                    "source_serial": row.source_serial,
                    "data_serial": row.data_serial,
                    "metadata": (
                        json.loads(row.metadata) if row.metadata is not None else None
                    ),
                },
                model=model_proxy,
                policy=policy,
                z_response=redshift(
                    store_id=row.z_response_serial,
                    z=row.z_response,
                    is_source=row.z_response_is_source,
                    is_response=row.z_response_is_response,
                ),
                z_source_max=redshift(
                    store_id=row.z_source_max_serial,
                    z=row.z_source_max,
                    is_source=row.z_source_max_is_source,
                    is_response=row.z_source_max_is_response,
                ),
                k=k,
                q=q_exit,
                r=r_exit,
                atol=atol,
                rtol=rtol,
                label=row.label,
                tags=tags,
            )
            obj._deserialized = True
            return obj

        objects = [make_object(row) for row in row_data]
        return objects
