import json
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import OneLoopIntegral
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_STRING_LENGTH


class sqla_OneLoopIntegralTagAssociation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("OneLoopIntegral.serial"),
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
    def add_tag(conn, inserter, source: OneLoopIntegral, tag: store_tag):
        inserter(
            conn,
            {
                "parent_serial": source.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, source: OneLoopIntegral, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.parent_serial == source.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_OneLoopIntegral_factory(SQLAFactoryBase):
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
                    "wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber_exit_time.serial"),
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
                    "z_response_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("value", sqla.Float(64), nullable=False),
                sqla.Column("compute_time", sqla.Float(64), nullable=True),
                sqla.Column(
                    "metadata", sqla.String(DEFAULT_STRING_LENGTH), nullable=True
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        model_proxy: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]
        z_response: redshift = payload["z_response"]

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
                table.c.value,
                tol_table.c.log10_tol,
            )
            .select_from(
                table.join(tol_table, tol_table.c.serial == table.c.tol_serial)
            )
            .filter(
                table.c.model_serial == model_proxy.store_id,
                table.c.wavenumber_exit_serial == k.store_id,
                table.c.z_response_serial == z_response.store_id,
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
                f"!! OneLoopIntegral.build(): multiple results found when querying for OneLoopIntegral (k={k.k.k_inv_Mpc:.5g}/Mpc, store_id={k.store_id})"
            )
            raise e

        if row_data is None:
            return OneLoopIntegral(
                payload=None,
                model=model_proxy,
                z_response=z_response,
                k=k,
                tol=tol,
                label=label,
                tags=tags,
            )

        obj = OneLoopIntegral(
            payload={
                "store_id": row_data.serial,
                "compute_time": row_data.compute_time,
                "value": row_data.value,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
            },
            model=model_proxy,
            z_response=z_response,
            k=k,
            tol=tol,
            label=row_data.label,
            tags=tags,
        )
        obj._deserialized = True
        return obj

    @staticmethod
    def store(
        obj: OneLoopIntegral,
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
                "model_serial": obj.model_proxy.store_id,
                "z_response_serial": obj.z_response.store_id,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "tol_serial": obj._tol.store_id,
                "value": obj.value,
                "metadata": (
                    json.dumps(obj.metadata) if obj._metadata is not None else None
                ),
                "validated": False,
            },
        )

        # set store_id on behalf of the OneLoopIntegral instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["OneLoopIntegral_tags"]
        for tag in obj.tags:
            sqla_OneLoopIntegralTagAssociation_factory.add_tag(
                conn, tag_inserter, obj, tag
            )

        return obj


def read_batch(payload, conn, table, tables):
    model_proxy: ModelProxy = payload["model"]

    k: wavenumber_exit_time = payload.get("k", None)
    z_response: redshift = payload.get("z_response", None)

    tol: Optional[tolerance] = payload.get("tol", None)
    tags: Optional[List[store_tag]] = payload.get("tags", None)

    tol_table = tables["tolerance"].alias("tol")
    k_exit_table = tables["wavenumber_exit_time"].alias("k_exit_tab")
    k_table = tables["wavenumber"].alias("k_tab")
    redshift_table = tables["redshift"]

    query = (
        sqla.select(
            table.c.serial,
            table.c.compute_time,
            table.c.label,
            table.c.metadata,
            table.c.value,
            tol_table.c.log10_tol,
            table.c.z_response_serial,
            redshift_table.c.z.label("z_response"),
        )
        .select_from(
            table.join(tol_table, tol_table.c.serial == table.c.tol_serial)
            .join(redshift_table, redshift_table.c.serial == table.c.z_response_serial)
            .join(
                k_exit_table,
                k_exit_table.c.serial == table.c.wavenumber_exit_serial,
            )
            .join(k_table, k_table.c.serial == k_exit_table.c.wavenumber_serial)
        )
        .filter(
            table.c.model_serial == model_proxy.store_id,
        )
    )

    if k is not None:
        query = query.filter(table.c.wavenumber_exit_serial == k.store_id)

    if z_response is not None:
        query = query.filter(table.c.z_response_serial == z_response.store_id)

    if tol is not None:
        query = query.filter(
            table.c.tol_serial == tol.store_id,
        )

    count = 0
    for tag in tags:
        tag: store_tag
        tab = tables["OneLoopIntegral_tags"].alias(f"tag_{count}")
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
        obj = OneLoopIntegral(
            payload={
                "store_id": row_data.serial,
                "compute_time": row_data.compute_time,
                "value": row_data.value,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
            },
            model=model_proxy,
            z_response=z_response,
            k=k,
            tol=tol,
            label=row_data.label,
            tags=tags,
        )
        obj._deserialized = True
        return obj
