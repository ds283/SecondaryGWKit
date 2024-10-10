from typing import Optional, List

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import TkNumericalIntegration, BackgroundModel
from ComputeTargets.TensorSource import TensorSource, TensorSourceValue
from CosmologyConcepts import redshift, redshift_array, wavenumber_exit_time
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_TensorSourceTagAssocation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("TensorSource.serial"),
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
    def add_tag(conn, inserter, source: TensorSource, tag: store_tag):
        inserter(
            conn,
            {
                "parent_serial": source.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, source: TensorSource, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.parent_serial == source.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_TensorSource_factory(SQLAFactoryBase):
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
                    "Tq_serial",
                    sqla.Integer,
                    sqla.ForeignKey("TensorSource.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "Tr_serial",
                    sqla.Integer,
                    # sqla.ForeignKey("TensorSource.serial"),
                    # index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_samples",
                    sqla.Integer,
                    nullable=False,
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("validated", sqla.Boolean, default=False, nullable=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        z_sample: redshift_array = payload["z_sample"]

        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        Tq: TkNumericalIntegration = payload["Tq"]
        Tr: TkNumericalIntegration = payload["Tr"]

        tag_table = tables["TensorSource_tags"]
        redshift_table = tables["redshift"]

        query = sqla.select(
            table.c.serial,
            table.c.compute_time,
            table.c.z_samples,
            table.c.label,
        ).filter(
            table.c.validated == True,
            table.c.Tq_serial == Tq.store_id,
            table.c.Tr_serial == Tr.store_id,
        )

        # require that the tensor source calculation we search for has the specified list of tags
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
                f"!! TensorSource.build(): multiple results found when querying for TensorSource (Tq_id={Tq.store_id}, Tr_id={Tr.store_id})"
            )
            raise e

        if row_data is None:
            return TensorSource(
                payload=None, z_sample=z_sample, Tq=Tq, Tr=Tr, label=label, tags=tags
            )

        store_id = row_data.serial
        store_label = row_data.label
        compute_time = row_data.compute_time

        num_expected_samples = row_data.z_samples

        value_table = tables["TensorSourceValue"]

        sample_rows = conn.execute(
            sqla.select(
                value_table.c.serial,
                value_table.c.z_serial,
                redshift_table.c.z,
                value_table.c.source_term,
                value_table.c.undiff_part,
                value_table.c.diff_part,
                value_table.c.analytic_source_term,
                value_table.c.analytic_undiff_part,
                value_table.c.analytic_diff_part,
            )
            .select_from(
                value_table.join(
                    redshift_table,
                    redshift_table.c.serial == value_table.c.z_serial,
                )
            )
            .filter(value_table.c.parent_serial == store_id)
            .order_by(redshift_table.c.z.desc())
        )

        z_points = []
        values = []
        for row in sample_rows:
            z_value = redshift(store_id=row.z_serial, z=row.z)
            z_points.append(z_value)
            values.append(
                TensorSourceValue(
                    store_id=row.serial,
                    z=z_value,
                    source_term=row.source_term,
                    undiff_part=row.undiff_part,
                    diff_part=row.diff_part,
                    analytic_source_term=row.analytic_source_term,
                    analytic_undiff_part=row.analytic_undiff_part,
                    analytic_diff_part=row.analytic_diff_part,
                )
            )
        imported_z_sample = redshift_array(z_points)

        if num_expected_samples is not None:
            if len(imported_z_sample) != num_expected_samples:
                raise RuntimeError(
                    f'Fewer z-samples than expected were recovered from the validated tensor source "{store_label}"'
                )

        obj = TensorSource(
            payload={
                "store_id": store_id,
                "compute_time": compute_time,
                "values": values,
            },
            z_sample=imported_z_sample,
            Tq=Tq,
            Tr=Tr,
            label=store_label,
            tags=tags,
        )
        obj._deserialized = True
        return obj

    @staticmethod
    def store(
        obj: TensorSource,
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
                "Tq_serial": obj.Tq.store_id,
                "Tr_serial": obj.Tr.store_id,
                "z_samples": len(obj.values),
                "compute_time": obj.compute_time,
                "validated": False,
            },
        )

        # set store_id on behalf of supplied TensorSource
        obj._my_id = store_id

        # add any tags
        tag_inserter = inserters["TensorSource_tags"]
        for tag in obj.tags:
            sqla_TensorSourceTagAssocation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sample points
        value_inserter = inserters["TensorSourceValue"]
        for value in obj.values:
            value: TensorSourceValue
            value_id = value_inserter(
                conn,
                {
                    "parent_serial": store_id,
                    "z_serial": value.z.store_id,
                    "source_term": value.source_term,
                    "undiff_part": value.undiff_part,
                    "diff_part": value.diff_part,
                    "analytic_source_term": value.analytic_source_term,
                    "analytic_undiff_part": value.analytic_undiff_part,
                    "analytic_diff_part": value.analytic_diff_part,
                },
            )

            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: TensorSource,
        conn,
        table,
        tables,
    ):
        # query the row in TensorSource corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["TensorSourceValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.parent_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: tensor source "{obj.label}" did not validate after serialization'
            )

        conn.execute(
            sqla.update(table)
            .where(table.c.serial == obj.store_id)
            .values(validated=validated)
        )

        return validated

    @staticmethod
    def validate_on_startup(conn, table, tables, prune=False):
        # query the datastore for any tensor source computations that are not validated

        Tq_table = tables["TkNumericalIntegration"].alias("Tq")
        Tr_table = tables["TkNumericalIntegration"].alias("Tr")
        value_table = tables["TensorSourceValue"]
        tags_table = tables["TensorSource_tags"]

        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_samples,
                    Tq_table.c.label.label("Tq_label"),
                    Tq_table.c.serial.label("Tq_serial"),
                    Tr_table.c.label.label("Tr_label"),
                    Tr_table.c.serial.label("Tr_serial"),
                )
                .select_from(
                    table.join(Tq_table, Tq_table.c.serial == table.c.Tq_serial).join(
                        Tr_table, Tr_table.c.serial == table.c.Tr_serial
                    )
                )
                .filter(or_(table.c.validated == False, table.c.validated == None))
            )
        )

        if len(not_validated) == 0:
            return []

        msgs = [
            ">> Tensor source term",
            "     The following unvalidated tensor source terms were detected in the datastore:",
        ]
        for calc in not_validated:
            msgs.append(
                f'       -- "{calc.label}" (store_id={calc.serial}) for Tq="{calc.Tq_label}" (store_id={calc.Tq_serial}), Tr="{calc.Tr_label}" (store_id={calc.Tr_serial})'
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.parent_serial == calc.serial
                )
            ).scalar()
            msgs.append(
                f"          contains {rows} z-sample values | expected={calc.z_samples}"
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


class sqla_TensorSourceValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("TensorSource.serial"),
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
                sqla.Column("source_term", sqla.Float(64), nullable=False),
                sqla.Column("undiff_part", sqla.Float(64), nullable=False),
                sqla.Column("diff_part", sqla.Float(64), nullable=False),
                sqla.Column("analytic_source_term", sqla.Float(64)),
                sqla.Column("analytic_undiff_part", sqla.Float(64)),
                sqla.Column("analytic_diff_part", sqla.Float(64)),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        parent_serial: Optional[int] = payload.get("parent_serial", None)

        model: Optional[BackgroundModel] = payload.get("model", None)
        q: Optional[wavenumber_exit_time] = payload.get("q", None)
        r: Optional[wavenumber_exit_time] = payload.get("r", None)

        has_serial = all([parent_serial is not None])
        has_model = all([model is not None, q is not None, r is not None])

        if all([has_serial, has_model]):
            print(
                "## TensorSourceValue.build(): both an parent serial number and a (model, q, r) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "TensorSourceValue.build(): at least one of a parent serial number and a (model, q, r) set must be supplied."
            )

        if has_serial:
            return sqla_TensorSourceValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_TensorSourceValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        parent_serial = payload["parent_serial"]
        z = payload["z"]

        source_term = payload["source_term"]
        undiff_part = payload["undiff_part"]
        diff_part = payload["diff_part"]

        analytic_source_term = payload.get("analytic_source_term", None)
        analytic_undiff_part = payload.get("analytic_undiff_part", None)
        analytic_diff_part = payload.get("analytic_diff_part", None)

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.source_term,
                    table.c.undiff_part,
                    table.c.diff_part,
                    table.c.analytic_source_term,
                    table.c.analytic_undiff_part,
                    table.c.analytic_diff_part,
                ).filter(
                    table.c.parent_serial == parent_serial,
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! TensorSourceValue.build(): multiple results found when querying for TensorSourceValue"
            )
            raise e

        if row_data is None:
            store_id = inserter(
                conn,
                {
                    "parent_serial": parent_serial,
                    "z_serial": z.store_id,
                    "source_term": source_term,
                    "undiff_part": undiff_part,
                    "diff_part": diff_part,
                    "analytic_source_term": analytic_source_term,
                    "analytic_undiff_part": analytic_undiff_part,
                    "analytic_diff_part": analytic_diff_part,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial

            analytic_source_term = row_data.analytic_source_term
            analytic_undiff_part = row_data.analytic_undiff_part
            analytic_diff_part = row_data.analytic_diff_part

            if fabs(row_data.source_term - source_term) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor source term value (calculation={parent_serial}, z={z.z}) = {row_data.source_term} differs from expected vlalue = {source_term}"
                )
            if fabs(row_data.undiff_part - undiff_part) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor source term undifferentiated part (calculation={parent_serial}, z={z.z}) = {row_data.undiff_part} differs from expected vlalue = {undiff_part}"
                )
            if fabs(row_data.diff_part - diff_part) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor source term differentiated part (calculation={parent_serial}, z={z.z}) = {row_data.diff_part} differs from expected vlalue = {diff_part}"
                )

            attribute_set = {"_deserialized": True}

        obj = TensorSourceValue(
            store_id=store_id,
            z=z,
            source_term=source_term,
            undiff_part=undiff_part,
            diff_part=diff_part,
            analytic_source_term=analytic_source_term,
            analytic_undiff_part=analytic_undiff_part,
            analytic_diff_part=analytic_diff_part,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _build_impl_model(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        model: BackgroundModel = payload["model"]
        q: wavenumber_exit_time = payload["q"]
        r: wavenumber_exit_time = payload["r"]

        atol: Optional[tolerance] = payload.get("atol", None)
        rtol: Optional[tolerance] = payload.get("rtol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        tensorsource_table = tables["TensorSource"]
        Tq_table = tables["TkNumericalIntegration"].alias("Tk_table")
        Tr_table = tables["TkNumericalIntegration"].alias("Tr_table")

        try:
            tensorsource_query = (
                sqla.select(tensorsource_table.c.serial)
                .select_from(
                    tensorsource_table.join(
                        Tq_table, Tq_table.serial == tensorsource_table.c.Tq_serial
                    ).join(Tr_table, Tr_table.serial == tensorsource_table.c.Tr_serial)
                )
                .filter(
                    Tq_table.c.model_serial == model.store_id,
                    Tr_table.c.model_serial == model.store_id,
                    Tq_table.c.wavenumber_exit_serial == q.store_id,
                    Tr_table.c.wavenumber_exit_serial == r.store_id,
                    tensorsource_table.c.validated == True,
                )
            )

            if atol is not None:
                tensorsource_query = tensorsource_query.filter(
                    Tq_table.c.atol_serial == atol.store_id,
                    Tr_table.c.atol_serial == atol.store_id,
                )

            if rtol is not None:
                tensorsource_query = tensorsource_query.filter(
                    Tq_table.c.rtol_serial == rtol.store_id,
                    Tr_table.c.rtol_serial == rtol.store_id,
                )

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["TensorSource_tags"].alias(f"tag_{count}")
                count += 1
                tensorsource_query = tensorsource_query.join(
                    tab,
                    and_(
                        tab.c.parent_serial == tensorsource_table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            subquery = tensorsource_query.subquery()

            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.source_term,
                    table.c.undiff_part,
                    table.c.diff_part,
                    table.c.analytic_source_term,
                    table.c.analytic_undiff_part,
                    table.c.analytic_diff_part,
                )
                .select_from(
                    subquery.join(table, table.c.parent_serial == subquery.c.serial)
                )
                .filter(
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! TensorSourceValue.build(): multiple results found when querying for TensorSourceValue"
            )
            raise e

        if row_data is None:
            # return empty object
            return TensorSourceValue(
                store_id=None, z=z, source=None, undiff_part=None, diff_part=None
            )

        obj = TensorSourceValue(
            store_id=row_data.serial,
            z=z,
            source_term=row_data.source_term,
            undiff_part=row_data.undiff_part,
            diff_part=row_data.diff_part,
            analytic_source_term=row_data.analytic_source_term,
            analytic_undiff_part=row_data.analytic_undiff_part,
            analytic_diff_part=row_data.analytic_diff_part,
        )
        obj._deserialized = True
        obj._q_exit = q
        obj._r_exit = r
        return obj
