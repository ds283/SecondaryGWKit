from math import fabs
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    QuadSource,
    QuadSourceValue,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import redshift, redshift_array, wavenumber_exit_time
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from config.defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


def _serial_argument(where: str, name: str, value) -> int:
    """The serial a lookup names one of its parents by: an integer, never a bool or an object."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{where}: {name} must be the integer serial of a TkNumericIntegration row, not "
            f"{value!r}"
        )
    return value


class sqla_QuadSourceTagAssocation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("QuadSource.serial"),
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
    def add_tag(conn, inserter, source: QuadSource, tag: store_tag):
        inserter(
            conn,
            {
                "parent_serial": source.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, source: QuadSource, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.parent_serial == source.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_QuadSource_factory(SQLAFactoryBase):
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
                    "q_wavenumber_exit_serial",
                    sqla.Integer,
                    sqla.ForeignKey(
                        "wavenumber_exit_time.serial"
                    ),  # q instances will be held on this shard, because we shard by q.
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
                    "Tq_serial",
                    sqla.Integer,
                    sqla.ForeignKey(
                        "TkNumericIntegration.serial"
                    ),  # Tq is the Tk numeric row at k = q, which is held on this shard, because we shard by q.
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "Tr_serial",
                    sqla.Integer,
                    # No foreign key: Tr is the Tk numeric row at k = r, which is held on r's shard,
                    # not necessarily this one (we shard by q), and a foreign key cannot cross a
                    # shard. Serials are unique across shards, so the serial still names the row.
                    # Indexed because it is part of the lookup key, beside Tq_serial
                    # (prompts/datastore-integrity, prompt 09a)
                    index=True,
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

        model_proxy: ModelProxy = payload["model"]
        q: wavenumber_exit_time = payload["q"]
        r: wavenumber_exit_time = payload["r"]

        # the key's parent terms, or the reader mode D1 allows; never neither, and never both.
        # A QuadSource is computed from exactly two rows, the TkNumericIntegrations for q and r, so
        # a lookup names both by serial (prompts/datastore-integrity, prompt 09a). The parents are
        # named by serial rather than by object: the serial is all the query needs, it is what the
        # compute target carries from construction, and Tr is on another shard
        Tq_serial: Optional[int] = payload.get("Tq_serial", None)
        Tr_serial: Optional[int] = payload.get("Tr_serial", None)
        labels_only: bool = payload.get("labels_only", False)
        if labels_only and (Tq_serial is not None or Tr_serial is not None):
            raise ValueError(
                "QuadSource.build(): a transfer-function parent and labels_only were both given; "
                "a keyed lookup names its parents, and the reader mode names none"
            )
        if not labels_only and (Tq_serial is None or Tr_serial is None):
            raise RuntimeError(
                "QuadSource.build(): the lookup names neither both transfer-function parents nor "
                "the reader mode. A QuadSource is keyed on the two TkNumericIntegrations it is "
                "computed from (Tq_serial and Tr_serial); a reader holding only physical labels "
                "must ask for labels_only=True explicitly"
            )
        if not labels_only:
            Tq_serial = _serial_argument("QuadSource.build()", "Tq_serial", Tq_serial)
            Tr_serial = _serial_argument("QuadSource.build()", "Tr_serial", Tr_serial)

        tag_table = tables["QuadSource_tags"]
        redshift_table = tables["redshift"]

        query = sqla.select(
            table.c.serial,
            table.c.compute_time,
            table.c.z_samples,
            table.c.label,
            table.c.Tq_serial,
            table.c.Tr_serial,
        ).filter(
            table.c.validated == True,
            table.c.model_serial == model_proxy.store_id,
            table.c.q_wavenumber_exit_serial == q.store_id,
            table.c.r_wavenumber_exit_serial == r.store_id,
        )

        # the two parents (prompt 09a). Without them a change to the Tk rows under the same tags
        # would serve the old source as a match
        if not labels_only:
            query = query.filter(
                table.c.Tq_serial == Tq_serial,
                table.c.Tr_serial == Tr_serial,
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
            if labels_only:
                # D1: the physical labels may select, but more than one stored row is refused,
                # naming the parents that distinguish them, rather than one of them chosen
                candidates = sorted(
                    (row.Tq_serial, row.Tr_serial) for row in conn.execute(query)
                )
                if len(candidates) > 1:
                    named = "; ".join(
                        f"Tq_serial={Tq}, Tr_serial={Tr}" for Tq, Tr in candidates
                    )
                    raise MultipleResultsFound(
                        f"QuadSource.build(): labels_only lookup matched {len(candidates)} stored "
                        f"rows (model serial={model_proxy.store_id}, q serial={q.store_id}, r "
                        f"serial={r.store_id}): {named}. Refusing to choose one; name the parents."
                    )

            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! QuadSource.build(): multiple results found when querying for QuadSource (q={q.k.k_inv_Mpc:.5g}/Mpc, store_id={q.store_id} | r={r.k.k_inv_Mpc:.5g}/Mpc, store_id={r.store_id})"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object. A labels-only miss has no parents to give
            # it, and says so: it can report absence to a reader and cannot be computed
            return QuadSource(
                payload=None,
                model=model_proxy,
                z_sample=z_sample,
                q=q,
                r=r,
                Tq_serial=Tq_serial,
                Tr_serial=Tr_serial,
                labels_only=labels_only,
                label=label,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label
        compute_time = row_data.compute_time

        num_expected_samples = row_data.z_samples

        do_not_populate = payload.get("_do_not_populate", False)
        if not do_not_populate:
            value_table = tables["QuadSourceValue"]

            sample_rows = conn.execute(
                sqla.select(
                    value_table.c.serial,
                    value_table.c.z_serial,
                    redshift_table.c.z,
                    redshift_table.c.source.label("z_is_source"),
                    redshift_table.c.response.label("z_is_response"),
                    value_table.c.source,
                    value_table.c.undiff,
                    value_table.c.diff,
                    value_table.c.analytic_source_rad,
                    value_table.c.analytic_undiff_rad,
                    value_table.c.analytic_diff_rad,
                    value_table.c.analytic_source_w,
                    value_table.c.analytic_undiff_w,
                    value_table.c.analytic_diff_w,
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
                z_value = redshift(
                    store_id=row.z_serial,
                    z=row.z,
                    is_source=row.z_is_source,
                    is_response=row.z_is_response,
                )
                z_points.append(z_value)
                values.append(
                    QuadSourceValue(
                        store_id=row.serial,
                        z=z_value,
                        source=row.source,
                        undiff=row.undiff,
                        diff=row.diff,
                        analytic_source_rad=row.analytic_source_rad,
                        analytic_undiff_rad=row.analytic_undiff_rad,
                        analytic_diff_rad=row.analytic_diff_rad,
                        analytic_source_w=row.analytic_source_w,
                        analytic_undiff_w=row.analytic_undiff_w,
                        analytic_diff_w=row.analytic_diff_w,
                    )
                )
            imported_z_sample = redshift_array(z_points)

            if num_expected_samples is not None:
                if len(imported_z_sample) != num_expected_samples:
                    raise RuntimeError(
                        f'Fewer z-samples than expected were recovered from the validated tensor source "{store_label}"'
                    )

            attributes = {"_deserialized": True}
        else:
            values = None
            imported_z_sample = None

            attributes = {"_do_not_populate": True, "_deserialized": True}

        obj = QuadSource(
            payload={
                "store_id": store_id,
                "compute_time": compute_time,
                "values": values,
                "Tq_serial": row_data.Tq_serial,
                "Tr_serial": row_data.Tr_serial,
            },
            model=model_proxy,
            z_sample=imported_z_sample,
            q=q,
            r=r,
            label=store_label,
            tags=tags,
        )
        for key, value in attributes.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def store(
        obj: QuadSource,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        if obj._Tq_serial is None or obj._Tr_serial is None:
            raise ValueError(
                "QuadSource.store(): the object carries no Tq_serial and Tr_serial, so it was "
                "not keyed on the TkNumericIntegrations it would be computed from"
            )

        store_id = inserter(
            conn,
            {
                "label": obj.label,
                "model_serial": obj.model_proxy.store_id,
                "q_wavenumber_exit_serial": obj._q_exit.store_id,
                "r_wavenumber_exit_serial": obj._r_exit.store_id,
                "Tq_serial": obj._Tq_serial,
                "Tr_serial": obj._Tr_serial,
                "z_samples": len(obj.values),
                "compute_time": obj.compute_time,
                "validated": False,
            },
        )

        # set store_id on behalf of supplied QuadSource
        obj._my_id = store_id

        # add any tags
        tag_inserter = inserters["QuadSource_tags"]
        for tag in obj.tags:
            sqla_QuadSourceTagAssocation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sample points
        value_inserter = inserters["QuadSourceValue"]
        for value in obj.values:
            value: QuadSourceValue
            value_id = value_inserter(
                conn,
                {
                    "parent_serial": store_id,
                    "z_serial": value.z.store_id,
                    "source": value.source,
                    "undiff": value.undiff,
                    "diff": value.diff,
                    "analytic_source_rad": value.analytic_source_rad,
                    "analytic_undiff_rad": value.analytic_undiff_rad,
                    "analytic_diff_rad": value.analytic_diff_rad,
                    "analytic_source_w": value.analytic_source_w,
                    "analytic_undiff_w": value.analytic_undiff_w,
                    "analytic_diff_w": value.analytic_diff_w,
                },
            )

            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: QuadSource,
        conn,
        table,
        tables,
    ):
        # query the row in QuadSource corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["QuadSourceValue"]
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

        Tq_table = tables["TkNumericIntegration"].alias("Tq")
        Tr_table = tables["TkNumericIntegration"].alias("Tr")
        value_table = tables["QuadSourceValue"]
        tags_table = tables["QuadSource_tags"]

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

    @staticmethod
    def inventory_records(conn, table, tables, context):
        # the physical identity of a QuadSource row (store-fingerprint prompt 02): what build()
        # filters on -- the model, the q and r wavenumber exits, and the two TkNumericIntegrations
        # it is computed from -- with its tags, its validated flag and its QuadSourceValue count.
        # Tq is on this row's shard, because both are sharded by q. Tr is the Tk row at k = r, on
        # r's shard, so it is a cross-shard parent: it resolves against the TkNumericIntegration
        # records of every shard, to that record's own reference digest and never its serial
        # (prompts/datastore-integrity, prompt 09a)
        from Datastore.store_inventory import Parent, read_records

        return read_records(
            conn,
            table,
            tables,
            context,
            parents={
                "model": Parent("model_serial", "BackgroundModel"),
                "q": Parent("q_wavenumber_exit_serial", "wavenumber_exit_time"),
                "r": Parent("r_wavenumber_exit_serial", "wavenumber_exit_time"),
                "Tq": Parent("Tq_serial", "TkNumericIntegration"),
                "Tr": Parent("Tr_serial", "TkNumericIntegration", cross_shard=True),
            },
            tags=("QuadSource_tags", "parent_serial"),
            values=("QuadSourceValue", "parent_serial"),
            validated=True,
        )


class sqla_QuadSourceValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("QuadSource.serial"),
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
                sqla.Column("source", sqla.Float(64), nullable=False),
                sqla.Column("undiff", sqla.Float(64), nullable=False),
                sqla.Column("diff", sqla.Float(64), nullable=False),
                sqla.Column("analytic_source_rad", sqla.Float(64)),
                sqla.Column("analytic_undiff_rad", sqla.Float(64)),
                sqla.Column("analytic_diff_rad", sqla.Float(64)),
                sqla.Column("analytic_source_w", sqla.Float(64)),
                sqla.Column("analytic_undiff_w", sqla.Float(64)),
                sqla.Column("analytic_diff_w", sqla.Float(64)),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        parent_serial: Optional[int] = payload.get("parent_serial", None)

        model_proxy: Optional[ModelProxy] = payload.get("model", None)
        q: Optional[wavenumber_exit_time] = payload.get("q", None)
        r: Optional[wavenumber_exit_time] = payload.get("r", None)

        has_serial = all([parent_serial is not None])
        has_model = all([model_proxy is not None, q is not None, r is not None])

        if all([has_serial, has_model]):
            print(
                "## QuadSourceValue.build(): both an parent serial number and a (model, q, r) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "QuadSourceValue.build(): at least one of a parent serial number and a (model, q, r) set must be supplied."
            )

        if has_serial:
            return sqla_QuadSourceValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_QuadSourceValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        parent_serial = payload["parent_serial"]
        z = payload["z"]

        source = payload["source"]
        undiff = payload["undiff"]
        diff = payload["diff"]

        analytic_source_rad = payload.get("analytic_source_rad", None)
        analytic_undiff_rad = payload.get("analytic_undiff_rad", None)
        analytic_diff_rad = payload.get("analytic_diff_rad", None)

        analytic_source_w = payload.get("analytic_source_w", None)
        analytic_undiff_w = payload.get("analytic_undiff_w", None)
        analytic_diff_w = payload.get("analytic_diff_w", None)

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.source,
                    table.c.undiff,
                    table.c.diff,
                    table.c.analytic_source_rad,
                    table.c.analytic_undiff_rad,
                    table.c.analytic_diff_rad,
                    table.c.analytic_source_w,
                    table.c.analytic_undiff_w,
                    table.c.analytic_diff_w,
                ).filter(
                    table.c.parent_serial == parent_serial,
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! QuadSourceValue.build(): multiple results found when querying for QuadSourceValue"
            )
            raise e

        if row_data is None:
            store_id = inserter(
                conn,
                {
                    "parent_serial": parent_serial,
                    "z_serial": z.store_id,
                    "source": source,
                    "undiff": undiff,
                    "diff": diff,
                    "analytic_source_rad": analytic_source_rad,
                    "analytic_undiff_rad": analytic_undiff_rad,
                    "analytic_diff_rad": analytic_diff_rad,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial

            analytic_source_rad = row_data.analytic_source_rad
            analytic_undiff_rad = row_data.analytic_undiff_rad
            analytic_diff_rad = row_data.analytic_diff_rad

            analytic_source_w = row_data.analytic_source_w
            analytic_undiff_w = row_data.analytic_undiff_w
            analytic_diff_w = row_data.analytic_diff_w

            if fabs(row_data.source - source) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor source term value (calculation={parent_serial}, z={z.z}) = {row_data.source} differs from expected vlalue = {source}"
                )
            if fabs(row_data.undiff - undiff) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor source term undifferentiated part (calculation={parent_serial}, z={z.z}) = {row_data.undiff} differs from expected vlalue = {undiff}"
                )
            if fabs(row_data.diff - diff) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored tensor source term differentiated part (calculation={parent_serial}, z={z.z}) = {row_data.diff} differs from expected vlalue = {diff}"
                )

            attribute_set = {"_deserialized": True}

        obj = QuadSourceValue(
            store_id=store_id,
            z=z,
            source=source,
            undiff=undiff,
            diff=diff,
            analytic_source_rad=analytic_source_rad,
            analytic_undiff_rad=analytic_undiff_rad,
            analytic_diff_rad=analytic_diff_rad,
            analytic_source_w=analytic_source_w,
            analytic_undiff_w=analytic_undiff_w,
            analytic_diff_w=analytic_diff_w,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _build_impl_model(payload, conn, table, inserter, tables, inserters):
        z = payload["z"]

        model_proxy: ModelProxy = payload["model"]
        q: wavenumber_exit_time = payload["q"]
        r: wavenumber_exit_time = payload["r"]

        tags: List[store_tag] = payload.get("tags", [])

        quadsource_table = tables["QuadSource"]

        try:
            quadsource_query = sqla.select(quadsource_table.c.serial).filter(
                quadsource_table.c.model_serial == model_proxy.store_id,
                quadsource_table.c.q_wavenumber_exit_serial == q.store_id,
                quadsource_table.c.r_wavenumber_exit_serial == r.store_id,
                quadsource_table.c.validated == True,
            )

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["QuadSource_tags"].alias(f"tag_{count}")
                count += 1
                quadsource_query = quadsource_query.join(
                    tab,
                    and_(
                        tab.c.parent_serial == quadsource_table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            subquery = quadsource_query.subquery()

            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.source,
                    table.c.undiff,
                    table.c.diff,
                    table.c.analytic_source_rad,
                    table.c.analytic_undiff_rad,
                    table.c.analytic_diff_rad,
                    table.c.analytic_source_w,
                    table.c.analytic_undiff_w,
                    table.c.analytic_diff_w,
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
                f"!! QuadSourceValue.build(): multiple results found when querying for QuadSourceValue"
            )
            raise e

        if row_data is None:
            # return empty object
            obj = QuadSourceValue(
                store_id=None, z=z, source=None, undiff=None, diff=None
            )
            obj._model_proxy = model_proxy
            obj._q_exit = q
            obj._r_exit = r
            return obj

        obj = QuadSourceValue(
            store_id=row_data.serial,
            z=z,
            source=row_data.source,
            undiff=row_data.undiff,
            diff=row_data.diff,
            analytic_source_rad=row_data.analytic_source_rad,
            analytic_undiff_rad=row_data.analytic_undiff_rad,
            analytic_diff_rad=row_data.analytic_diff_rad,
            analytic_source_w=row_data.analytic_source_w,
            analytic_undiff_w=row_data.analytic_undiff_w,
            analytic_diff_w=row_data.analytic_diff_w,
        )
        obj._deserialized = True
        obj._model_proxy = model_proxy
        obj._q_exit = q
        obj._r_exit = r
        return obj
