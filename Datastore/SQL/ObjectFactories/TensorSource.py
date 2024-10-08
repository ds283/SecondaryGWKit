from typing import Optional, List

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import TkNumericalIntegration
from ComputeTargets.TensorSource import TensorSource, TensorSourceValue
from CosmologyConcepts import redshift, redshift_array
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
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
    def validate_on_startup(conn, table, tables):
        # query the datastore for any tensor source computations that are not validated

        Tq_table = tables["TkNumericalIntegration"].alias("Tq")
        Tr_table = tables["TkNumericalIntegration"].alias("Tr")
        value_table = tables["TensorSourceValue"]

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

        return TensorSourceValue(
            store_id=store_id,
            z=z,
            source_term=source_term,
            undiff_part=undiff_part,
            diff_part=diff_part,
            analytic_source_term=analytic_source_term,
            analytic_undiff_part=analytic_undiff_part,
            analytic_diff_part=analytic_diff_part,
        )
