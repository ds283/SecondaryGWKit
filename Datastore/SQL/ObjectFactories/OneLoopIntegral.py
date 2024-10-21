import json
from typing import Optional, List

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import OneLoopIntegral, BackgroundModel, OneLoopIntegralValue
from CosmologyConcepts import redshift_array, wavenumber_exit_time, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


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
            "validate_on_startup": True,
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
                    "source_serial",
                    sqla.Integer,
                    sqla.ForeignKey("QuadSourceIntegral.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_response_samples",
                    sqla.Integer,
                    nullable=False,
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("validated", sqla.Boolean, default=False, nullable=False),
                sqla.Column(
                    "metadata", sqla.String(DEFAULT_STRING_LENGTH), nullable=True
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        z_response_sample: redshift_array = payload["z_response_sample"]

        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        tol: tolerance = payload["tol"]

        model: BackgroundModel = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        tag_table = tables["OneLoopIntegral_tags"]
        redshift_table = tables["redshift"]

        tol_table = tables["tolerance"].alias("tol")

        query = (
            sqla.select(
                table.c.serial,
                table.c.compute_time,
                table.c.z_response_samples,
                table.c.label,
                table.c.metadata,
                table.c.source_serial,
                tol_table.c.log10_tol,
            )
            .select_from(
                table.join(tol_table, tol_table.c.serial == table.c.tol_serial)
            )
            .filter(
                table.c.validated == True,
                table.c.model_serial == model.store_id,
                table.c.wavenumber_exit_serial == k.store_id,
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
                f"!! OneLoopIntegral.build(): multiple results found when querying for OneLoopIntegral (k={k.k.k_inv_Mpc:.5g}/Mpc, store_id={q.store_id})"
            )
            raise e

        if row_data is None:
            return OneLoopIntegral(
                payload=None,
                model=model,
                z_response_sample=z_response_sample,
                k=k,
                tol=tol,
                label=label,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label
        compute_time = row_data.compute_time

        num_expected_samples = row_data.z_response_samples

        if payload is None or not payload.get("_do_not_populate", False):
            value_table = tables["OneLoopIntegralValue"]

            sample_rows = conn.execute(
                sqla.select(
                    value_table.c.serial,
                    value_table.c.z_response_serial,
                    redshift_table.c.z.label("z_response"),
                    value_table.c.total,
                )
                .select_from(
                    value_table.join(
                        redshift_table,
                        redshift_table.c.serial == value_table.c.z_response_serial,
                    )
                )
                .filter(value_table.c.parent_serial == store_id)
                .order_by(redshift_table.c.z.desc())
            )

            z_response_points = []
            values = []
            for row in sample_rows:
                z_response_value = redshift(
                    store_id=row.z_response_serial, z=row.z_response
                )
                z_response_points.append(z_response_value)
                values.append(
                    OneLoopIntegralValue(
                        store_id=row.serial,
                        z_response=z_response_value,
                        total=row.total,
                    )
                )
            imported_z_sample = redshift_array(z_response_points)

            if num_expected_samples is not None:
                if len(imported_z_sample) != num_expected_samples:
                    raise RuntimeError(
                        f'Fewer z-samples than expected were recovered from the validated 1-loop sample "{store_label}"'
                    )

            attributes = {"_deserialized": True}
        else:
            values = None
            imported_z_sample = None

            attributes = {"_do_not_populate": True, "_deserialized": True}

        obj = OneLoopIntegral(
            payload={
                "store_id": store_id,
                "compute_time": compute_time,
                "values": values,
                "source_serial": row_data.source_serial,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
            },
            model=model,
            z_response_sample=imported_z_sample,
            k=k,
            tol=tol,
            label=store_label,
            tags=tags,
        )
        for key, value in attributes.items():
            setattr(obj, key, value)
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
                "model_serial": obj.model.store_id,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "tol_serial": obj._tol.store_id,
                "z_response_samples": len(obj.values),
                "source_serial": obj._source_serial,
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

        # now serialize the sampled output points
        value_inserter = inserters["OneLoopIntegralValue"]
        for value in obj.values:
            value: OneLoopIntegralValue
            value_id = value_inserter(
                conn,
                {
                    "parent_serial": store_id,
                    "z_response_serial": value.z.store_id,
                    "total": value.total,
                },
            )

            # set store_id on behalf of the OneLoopIntegralValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: OneLoopIntegral,
        conn,
        table,
        tables,
    ):
        # query the row in OneLoopIntegralValue corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_response_samples).filter(
                table.c.serial == obj.store_id
            )
        ).scalar()

        value_table = tables["OneLoopIntegralValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.parent_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: quadratic source integral "{obj.label}" did not validate after serialization'
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

        tol_table = tables["tolerance"].alias("tol")
        k_exit_table = tables["wavenumber_exit_time"].alias("k_exit_table")
        k_table = tables["wavenumber"].alias("k_table")

        value_table = tables["OneLoopIntegralValue"]
        tags_table = tables["OneLoopIntegral_tags"]

        # bake results into a list so that we can close this query; we are going to want to run
        # another one as we process the rows from this one
        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_response_samples,
                    k_table.c.k_inv_Mpc.label("k_inv_Mpc"),
                    tol_table.c.log10_tol,
                )
                .select_from(
                    table.join(tol_table, tol_table.c.serial == table.c.tol_serial)
                    .join(
                        k_exit_table,
                        k_exit_table.c.serial == table.c.wavenumber_exit_serial,
                    )
                    .join(
                        k_table,
                        k_table.c.serial == k_exit_table.c.wavenumber_serial,
                    )
                )
                .filter(or_(table.c.validated == False, table.c.validated == None))
            )
        )

        if len(not_validated) == 0:
            return []

        msgs = [
            ">> 1-loop power spectrum integrals",
            "     The following unvalidated 1-loop power spectrum integrals were detected in the datastore:",
        ]
        for calc in not_validated:
            msgs.append(
                f'       -- "{calc.label}" (store_id={calc.serial}) for k={calc.k_inv_Mpc:.5g}/Mpc (log10_tol={calc.log10_atol})'
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.parent_serial == calc.serial,
                )
            ).scalar()
            msgs.append(
                f"          contains {rows} z-sample values | expected={calc.z_response_samples}"
            )

        if prune:
            invalid_serials = [nv.serial for nv in not_validated]
            try:
                conn.execute(
                    sqla.delete(value_table).where(
                        value_table.c.parent_serial.in_(invalid_serials)
                    )
                )
                conn.execute(
                    sqla.delete(tags_table).where(
                        tags_table.c.parent_serial.in_(invalid_serials)
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


class sqla_OneLoopIntegralValue_factory(SQLAFactoryBase):
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
                    "parent_serial",
                    sqla.Integer,
                    sqla.ForeignKey("OneLoopIntegral.serial"),
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
                sqla.Column("total", sqla.Float(64), nullable=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        parent_serial = payload.get("parent_serial", None)

        model: Optional[BackgroundModel] = payload.get("model", None)
        k: Optional[wavenumber_exit_time] = payload.get("k", None)
        z_response = payload.get("z_response", None)

        has_serial = all([parent_serial is not None])
        has_model = all(
            [
                model is not None,
                k is not None,
                z_response is not None,
            ]
        )

        if all([has_serial, has_model]):
            print(
                "## OneLoopIntegralValue.build(): both an OneLoopIntegral serial number and a (model, k, z_response) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "OneLoopIntegralValue.build(): at least one of a OneLoopIntegral serial number and a (model, k, z_response) set must be supplied."
            )

        if has_serial:
            return sqla_OneLoopIntegralValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_OneLoopIntegralValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        z_response = payload["z_response"]

        parent_serial: int = payload["parent_serial"]

        total: Optional[float] = payload.get("total", None)
        has_data = all(
            [
                total is not None,
            ]
        )

        parent_table = tables["OneLoopIntegral"]

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.total,
                    parent_table.c.source_serial,
                )
                .filter(
                    table.c.parent_serial == parent_serial,
                    table.c.z_repsonse_serial == z_response.store_id,
                )
                .join(parent_table, parent_table.c.serial == table.c.parent_serial)
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! OneLoopIntegralValue.build(): multiple results found when querying for OneLoopIntegralValue"
            )
            raise e

        if row_data is None:
            if not has_data:
                raise (
                    "OneLoopIntegralValue().build(): result was not found in datastore, but a data payload was not provided"
                )

            store_id = inserter(
                conn,
                {
                    "parent_serial": parent_serial,
                    "z_response_serial": z_response.store_id,
                    "total": total,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial

            if (
                total is not None
                and fabs(row_data.total - total) > DEFAULT_FLOAT_PRECISION
            ):
                raise ValueError(
                    f"OneLoopIntegralValue.build(): Stored quadratic source integral value (OneLoopIntegral store_id={parent_serial}, z_response={z_response.store_id}) = {row_data.total} differs from expected value = {total}"
                )

            total = row_data.total

            attribute_set = {
                "_deserialized": True,
                "_source_serial": row_data.source_serial,
            }

        obj = OneLoopIntegralValue(
            store_id=store_id,
            z_response=z_response,
            total=total,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _build_impl_model(payload, conn, table, inserter, tables, inserters):
        z_response = payload["z_response"]

        model: BackgroundModel = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        tol: Optional[tolerance] = payload.get("tol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        parent_table = tables["OneLoopIntegral"]

        try:
            parent_query = sqla.select(
                parent_table.c.serial,
            ).filter(
                parent_table.c.model_serial == model.store_id,
                parent_table.c.wavenumber_exit_serial == k.store_id,
                parent_table.c.validated == True,
            )

            if tol is not None:
                parent_query = parent_query.filter(
                    parent_table.c.tol_serial == tol.store_id
                )

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["OneLoopIntegral_tags"].alias(f"tag_{count}")
                count += 1
                parent_query = parent_query.join(
                    tab,
                    and_(
                        tab.c.parent_serial == parent_table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            subquery = parent_query.subquery()

            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.total,
                    subquery.c.source_serial,
                )
                .select_from(
                    subquery.join(table, table.c.parent_serial == subquery.c.serial)
                )
                .filter(
                    table.c.z_response_serial == z_response.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! OneLoopIntegralValue.build(): multiple results found when querying for OneLoopIntegralValue"
            )
            raise e

        if row_data is None:
            # return empty object
            obj = OneLoopIntegralValue(
                store_id=None,
                z_response=z_response,
                total=None,
            )
            obj._k_exit = k
            return obj

        obj = OneLoopIntegralValue(
            store_id=row_data.serial,
            z_response=z_response,
            total=row_data.total,
        )
        obj._deserialized = True
        obj._k_exit = k
        obj._source_serial = row_data.source_serial
        return obj

    @staticmethod
    def read_batch(payload, conn, table, tables):
        model: BackgroundModel = payload["model"]
        k: wavenumber_exit_time = payload["k"]

        tol: Optional[tolerance] = payload.get("tol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        parent_table = tables["OneLoopIntegral"]
        redshift_table = tables["redshift"]

        parent_query = sqla.select(
            parent_table.c.serial,
            parent_table.c.source_serial,
        ).filter(
            parent_table.c.model_serial == model.store_id,
            parent_table.c.wavenumber_exit_serial == k.store_id,
            parent_table.c.validated == True,
        )

        if tol is not None:
            parent_query = parent_query.filter(
                parent_table.c.tol_serial == tol.store_id
            )

        count = 0
        for tag in tags:
            tag: store_tag
            tab = tables["OneLoopIntegral_tags"].alias(f"tag_{count}")
            count += 1
            parent_query = parent_query.join(
                tab,
                and_(
                    tab.c.parent_serial == parent_table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        subquery = parent_query.subquery()

        row_query = sqla.select(
            table.c.serial,
            redshift_table.c.z.label("z_response"),
            table.c.z_serial.label("z_response_serial"),
            table.c.total,
            subquery.c.source_serial,
        ).select_from(
            subquery.join(table, table.c.parent_serial == subquery.c.serial).join(
                redshift_table, redshift_table.c.serial == table.c.z_response_serial
            )
        )

        row_data = conn.execute(row_query)

        def make_obj(row):
            obj = OneLoopIntegralValue(
                store_id=row.serial,
                z_response=redshift(store_id=row.z_response_serial, z=row.z_response),
                total=row.total,
            )
            obj._deserialized = True
            obj._k_exit = k
            obj._source_serial = row.source_serial

            return obj

        objects = [make_obj(row) for row in row_data]
        return objects
