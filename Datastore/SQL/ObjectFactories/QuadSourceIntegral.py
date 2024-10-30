import json
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import (
    QuadSourceIntegral,
    IntegrationData,
    LevinData,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance, GkSourcePolicy
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
                    sqla.Integer,  # Don't impose foreign key. q instances may not be held on this shard. We shard by q.
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "r_wavenumber_exit_serial",
                    sqla.Integer,  # Don't impose foreign key. r instances may not be held on this shard. We shard by q.
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "tol_serial",
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
                sqla.Column(
                    "WKB_Levin_elapsed",
                    sqla.Float(64),
                    nullable=True,
                ),
                sqla.Column(
                    "compute_time",
                    sqla.Float(64),
                    nullable=True,
                ),
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

        tol: tolerance = payload["tol"]

        tag_table = tables["QuadSourceIntegral_tags"]
        tol_table = tables["tolerance"].alias("tol")

        query = (
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.label,
                table.c.metadata,
                table.c.source_serial,
                table.c.total,
                table.c.numeric_quad,
                table.c.WKB_quad,
                table.c.WKB_Levin,
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
                table.c.WKB_Levin_elapsed,
                tol_table.c.log10_tol,
            )
            .select_from(
                table.join(tol_table, tol_table.c.serial == table.c.tol_serial)
            )
            .filter(
                table.c.model_serial == model_proxy.store_id,
                table.c.policy_serial == policy.store_id,
                table.c.k_wavenumber_exit_serial == k.store_id,
                table.c.q_wavenumber_exit_serial == q.store_id,
                table.c.r_wavenumber_exit_serial == r.store_id,
                table.c.z_response_serial == z_response.store_id,
                table.c.z_source_max_serial == z_source_max.store_id,
                table.c.tol_serial == tol.store_id,
            )
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
                tol=tol,
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
                    elapsed=row_data.WKB_Levin_elapsed,
                ),
                "compute_time": row_data.compute_time,
                "Gk_serial": row_data.Gk_serial,
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
            tol=tol,
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
                "tol_serial": obj._tol.store_id,
                "policy_serial": obj._policy.store_id,
                "source_serial": obj._source_serial,
                "data_serial": obj._data_serial,
                "compute_time": obj.compute_time,
                "total": obj._total,
                "numeric_quad": obj._numeric_quad,
                "WKB_quad": obj._WKB_quad,
                "WKB_Levin": obj._WKB_Levin,
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
                    WKB_quad_data.compute_time if WKB_quad_data is not None else None
                ),
                "WKB_quad_compute_steps": (
                    WKB_quad_data.compute_steps if WKB_quad_data is not None else None
                ),
                "WKB_quad_RHS_evaluations": (
                    WKB_quad_data.RHS_evaluations if WKB_quad_data is not None else None
                ),
                "WKB_quad_mean_RHS_time": (
                    WKB_quad_data.mean_RHS_time if WKB_quad_data is not None else None
                ),
                "WKB_quad_max_RHS_time": (
                    WKB_quad_data.max_RHS_time if WKB_quad_data is not None else None
                ),
                "WKB_quad_min_RHS_time": (
                    WKB_quad_data.min_RHS_time if WKB_quad_data is not None else None
                ),
                "WKB_Levin_num_regions": (
                    WKB_Levin_data.num_regions if WKB_Levin_data is not None else None
                ),
                "WKB_Levin_evaluations": (
                    WKB_Levin_data.evaluations if WKB_Levin_data is not None else None
                ),
                "WKB_Levin_elapsed": (
                    WKB_Levin_data.elapsed if WKB_Levin_data is not None else None
                ),
                "metadata": (
                    json.dumps(obj.metadata) if obj._metadata is not None else None
                ),
            },
        )

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

        q: Optional[wavenumber_exit_time] = payload.get("k", None)
        r: Optional[wavenumber_exit_time] = payload.get("k", None)
        z_response: Optional[redshift] = payload.get("z_response", None)
        z_source_max: Optional[redshift] = payload.get("z_source_max", None)

        tol: Optional[tolerance] = payload.get("tol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        tol_table = tables["tolerance"]

        q_exit_table = tables["wavenumber_exit_time"].alias("q_exit_tab")
        r_exit_table = tables["wavenumber_exit_time"].alias("r_exit_tab")
        q_table = tables["wavenumber"].alias("q_tab")
        r_table = tables["wavenumber"].alias("r_tab")
        z_response_table = tables["redshift"].alias("z_resp_tab")
        z_source_max_table = tables["redshift"].alias("z_src_max_tab")

        query = (
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.label,
                table.c.metadata,
                table.c.source_serial,
                table.c.data_serial,
                table.c.total,
                table.c.numeric_quad,
                table.c.WKB_quad,
                table.c.WKB_Levin,
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
                table.c.WKB_Levin_elapsed,
                tol_table.c.log10_tol,
                table.c.z_response_serial,
                z_response_table.c.z.label("z_response"),
                table.c.z_source_max_serial,
                z_source_max_table.c.z.label("z_source_max"),
            )
            .select_from(
                table.join(
                    z_response_table,
                    z_response_table.c.serial == table.c.z_response_serial,
                )
                .join(
                    z_source_max_table,
                    z_source_max_table.c.serial == table.c.z_source_max_serial,
                )
                .join(
                    q_exit_table,
                    q_exit_table.c.serial == table.q_wavenumber_exit_serial,
                )
                .join(
                    r_exit_table,
                    r_exit_table.c.serial == table.r_wavenumber_exit_serial,
                )
                .join(q_table, q_table.c.serial == q_exit_table.c.wavenumber_serial)
                .join(r_table, r_table.c.serial == r_exit_table.c.wavenumber_serial)
                .join(tol_table, tol_table.c.serial == table.c.tol_serial)
            )
            .filter(
                table.c.model_serial == model_proxy.store_id,
                table.c.policy_serial == policy.store_id,
                table.c.k_wavenumber_exit_serial == k.store_id,
            )
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

        if tol is not None:
            query = query.filter(
                table.c.tol_serial == tol.store_id,
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

        # TODO: not yet complete

        def make_object(row_data):
            obj = QuadSourceIntegral(
                payload={
                    "store_id": row_data.serial,
                    "total": row_data.total,
                    "numeric_quad": row_data.numeric_quad,
                    "WKB_quad": row_data.WKB_quad,
                    "WKB_Levin": row_data.WKB_Levin,
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
                        elapsed=row_data.WKB_Levin_elapsed,
                    ),
                    "compute_time": row_data.compute_time,
                    "source_serial": row_data.source_serial,
                    "data_serial": row_data.data_serial,
                    "metadata": (
                        json.loads(row_data.metadata)
                        if row_data.metadata is not None
                        else None
                    ),
                },
                model=model_proxy,
                policy=policy,
                z_response=row_data.z_response,
                z_source_max=row_data.z_source_max,
                k=k,
                q=q,
                r=r,
                tol=tol,
                label=row_data.label,
                tags=tags,
            )
            obj._deserialized = True
            return obj
