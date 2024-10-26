import json
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    GkWKBIntegration,
    GkSource,
    GkSourceValue,
)
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift_array, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_STRING_LENGTH

_quality_serialize = {
    "complete": 0,
    "acceptable": 1,
    "marginal": 2,
    "minimal": 3,
    "incomplete": 4,
}

_quality_deserialize = {
    0: "complete",
    1: "acceptable",
    2: "marginal",
    3: "minimal",
    4: "incomplete",
}

_type_serialize = {
    "numeric": 0,
    "WKB": 1,
    "mixed": 2,
}

_type_deserialize = {
    0: "numeric",
    1: "WKB",
    2: "mixed",
}


class sqla_GkSourceTagAssociation_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("GkSource.serial"),
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
    def add_tag(conn, inserter, Gk: GkSource, tag: store_tag):
        inserter(
            conn,
            {
                "parent_serial": Gk.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, Gk: GkSource, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.parent_serial == Gk.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_GkSource_factory(SQLAFactoryBase):
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
                    "z_response_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_max_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("z_samples", sqla.Integer, nullable=False),
                sqla.Column(
                    "numerical_smallest_z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=True,
                ),
                sqla.Column(
                    "primary_WKB_largest_z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=True,
                ),
                sqla.Column("type", sqla.SmallInteger, nullable=False),
                sqla.Column("quality", sqla.SmallInteger, nullable=False),
                sqla.Column(
                    "crossover_z",
                    sqla.Float(64),
                    nullable=True,
                ),
                sqla.Column("Levin_z", sqla.Float(65), nullable=True),
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

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        k_exit: wavenumber_exit_time = payload["k"]
        model_proxy: ModelProxy = payload["model_proxy"]
        z_sample: redshift_array = payload["z_sample"]
        z_response: redshift = payload["z_response"]

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        tag_table = tables["GkSource_tags"]
        redshift_table = tables["redshift"]

        smallest_numerical_z_table = redshift_table.alias("smallest_numerical")
        largest_WKB_z_table = redshift_table.alias("largest_WKB")

        # notice that we query only for validated data
        query = (
            sqla.select(
                table.c.serial,
                table.c.label,
                table.c.z_response_serial,
                redshift_table.c.z.label("z_response"),
                table.c.z_samples,
                atol_table.c.log10_tol.label("log10_atol"),
                rtol_table.c.log10_tol.label("log10_rtol"),
                table.c.numerical_smallest_z_serial,
                smallest_numerical_z_table.c.z.label("numerical_smallest_z"),
                table.c.primary_WKB_largest_z_serial,
                largest_WKB_z_table.c.z.label("primary_WKB_largest_z"),
                table.c.type,
                table.c.quality,
                table.c.crossover_z,
                table.c.Levin_z,
                table.c.metadata,
            )
            .select_from(
                table.join(atol_table, atol_table.c.serial == table.c.atol_serial)
                .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
                .join(
                    redshift_table, redshift_table.c.serial == table.c.z_response_serial
                )
                .outerjoin(
                    smallest_numerical_z_table,
                    smallest_numerical_z_table.c.serial
                    == table.c.numerical_smallest_z_serial,
                )
                .outerjoin(
                    largest_WKB_z_table,
                    largest_WKB_z_table.c.serial
                    == table.c.primary_WKB_largest_z_serial,
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

        if z_response is not None:
            query = query.filter(
                table.c.z_response_serial == z_response.store_id,
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
                    tab.c.parent_serial == table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkWKBIntegration.build(): multiple results found when querying for GkWKBIntegration"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return GkSource(
                payload=None,
                label=label,
                k=k_exit,
                model=model_proxy,
                atol=atol,
                rtol=rtol,
                z_response=z_response,
                z_sample=z_sample,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label

        numerical_smallest_z = None
        if row_data.numerical_smallest_z_serial is not None:
            numerical_smallest_z = redshift(
                store_id=row_data.numerical_smallest_z_serial,
                z=row_data.numerical_smallest_z,
            )

        primary_WKB_largest_z = None
        if row_data.primary_WKB_largest_z_serial is not None:
            primary_WKB_largest_z = redshift(
                store_id=row_data.primary_WKB_largest_z_serial,
                z=row_data.primary_WKB_largest_z,
            )

        num_expected_samples = row_data.z_samples

        do_not_populate = payload.get("_do_not_populate", False)
        if not do_not_populate:
            # read out sample values associated with this integration
            value_table = tables["GkSourceValue"]

            sample_rows = conn.execute(
                sqla.select(
                    value_table.c.serial,
                    value_table.c.z_source_serial,
                    redshift_table.c.z.label("z_source"),
                    value_table.c.G,
                    value_table.c.Gprime,
                    value_table.c.H_ratio,
                    value_table.c.theta_mod_2pi,
                    value_table.c.theta_div_2pi,
                    value_table.c.raw_theta_div_2pi,
                    value_table.c.G_WKB,
                    value_table.c.new_G_WKB,
                    value_table.c.abs_G_WKB_err,
                    value_table.c.rel_G_WKB_err,
                    value_table.c.sin_coeff,
                    value_table.c.cos_coeff,
                    value_table.c.z_init,
                    value_table.c.omega_WKB_sq,
                    value_table.c.WKB_criterion,
                    value_table.c.analytic_G,
                    value_table.c.analytic_Gprime,
                )
                .select_from(
                    value_table.join(
                        redshift_table,
                        redshift_table.c.serial == value_table.c.z_source_serial,
                    )
                )
                .filter(value_table.c.parent_serial == store_id)
                .order_by(redshift_table.c.z.desc())
            )

            z_points = []
            values = []
            for row in sample_rows:
                z_value = redshift(store_id=row.z_source_serial, z=row.z_source)
                z_points.append(z_value)
                values.append(
                    GkSourceValue(
                        store_id=row.serial,
                        z_source=z_value,
                        G=row.G,
                        Gprime=row.Gprime,
                        H_ratio=row.H_ratio,
                        theta_mod_2pi=row.theta_mod_2pi,
                        theta_div_2pi=row.theta_div_2pi,
                        raw_theta_div_2pi=row.raw_theta_div_2pi,
                        sin_coeff=row.sin_coeff,
                        cos_coeff=row.cos_coeff,
                        z_init=row.z_init,
                        G_WKB=row.G_WKB,
                        new_G_WKB=row.new_G_WKB,
                        abs_G_WKB_err=row.abs_G_WKB_err,
                        rel_G_WKB_err=row.rel_G_WKB_err,
                        omega_WKB_sq=row.omega_WKB_sq,
                        WKB_criterion=row.WKB_criterion,
                        analytic_G=row.analytic_G,
                        analytic_Gprime=row.analytic_Gprime,
                    )
                )
            imported_z_sample = redshift_array(z_points)

            if num_expected_samples is not None:
                if len(imported_z_sample) != num_expected_samples:
                    raise RuntimeError(
                        f'Fewer z-samples than expected were recovered from the validated WKB tensor Green function "{store_label}"'
                    )

            attributes = {"_deserialized": True}
        else:
            values = None
            imported_z_sample = None

            attributes = {"_do_not_populate": True, "_deserialized": True}

        obj = GkSource(
            payload={
                "store_id": store_id,
                "values": values,
                "numerical_smallest_z": numerical_smallest_z,
                "primary_WKB_largest_z": primary_WKB_largest_z,
                "type": (
                    _type_deserialize[row_data.type]
                    if row_data.type is not None
                    else None
                ),
                "quality": (
                    _quality_deserialize[row_data.quality]
                    if row_data.quality is not None
                    else None
                ),
                "crossover_z": row_data.crossover_z,
                "Levin_z": row_data.Levin_z,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
            },
            k=k_exit,
            model=model_proxy,
            label=store_label,
            atol=atol,
            rtol=rtol,
            z_response=redshift(
                store_id=(row_data.z_response_serial), z=(row_data.z_response)
            ),
            z_sample=imported_z_sample,
            tags=tags,
        )
        for key, value in attributes.items():
            setattr(obj, key, value)

        # obj_pickled = cloudpickle.dumps(obj)
        # model_pickled = cloudpickle.dumps(model_proxy)
        # sample_pickled = cloudpickle.dumps(imported_z_sample)
        # print(
        #     f"** deserialized GkSource object (store_id={store_id}) with cloudpickle size={humanize.naturalsize(len(obj_pickled), binary=True)}, model_proxy size={humanize.naturalsize(len(model_pickled), binary=True)}, zsample size={humanize.naturalsize(len(sample_pickled), binary=True)}) | _do_not_populate={payload.get("_do_not_populate", False)}"
        # )

        return obj

    @staticmethod
    def store(
        obj: GkSource,
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
                "z_response_serial": obj.z_response.store_id,
                "z_max_serial": obj.z_sample.max.store_id,
                "z_samples": len(obj.values),
                "numerical_smallest_z_serial": (
                    obj.numerical_smallest_z.store_id
                    if obj.numerical_smallest_z is not None
                    else None
                ),
                "primary_WKB_largest_z_serial": (
                    obj.primary_WKB_largest_z.store_id
                    if obj.primary_WKB_largest_z is not None
                    else None
                ),
                "type": _type_serialize[obj._type] if obj._type is not None else None,
                "quality": (
                    _quality_serialize[obj._quality]
                    if obj._quality is not None
                    else None
                ),
                "crossover_z": obj._crossover_z,
                "Levin_z": obj._Levin_z,
                "metadata": (
                    json.dumps(obj._metadata) if obj._metadata is not None else None
                ),
                "validated": False,
            },
        )

        # set store_id on behalf of the GkSource instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["GkSource_tags"]
        for tag in obj.tags:
            sqla_GkSourceTagAssociation_factory.add_tag(conn, tag_inserter, obj, tag)

        # now serialize the sampled output points
        value_inserter = inserters["GkSourceValue"]
        for value in obj.values:
            value: GkSourceValue
            value_id = value_inserter(
                conn,
                {
                    "parent_serial": store_id,
                    "z_source_serial": value.z_source.store_id,
                    "G": value._numeric_data.G,
                    "Gprime": value._numeric_data.Gprime,
                    "H_ratio": value._WKB_data.H_ratio,
                    "theta_mod_2pi": value._WKB_data.theta_mod_2pi,
                    "theta_div_2pi": value._WKB_data.theta_div_2pi,
                    "raw_theta_div_2pi": value._WKB_data.raw_theta_div_2pi,
                    "G_WKB": value._WKB_data.G_WKB,
                    "new_G_WKB": value._WKB_data.new_G_WKB,
                    "abs_G_WKB_err": value._WKB_data.abs_G_WKB_err,
                    "rel_G_WKB_err": value._WKB_data.rel_G_WKB_err,
                    "sin_coeff": value._WKB_data.sin_coeff,
                    "cos_coeff": value._WKB_data.cos_coeff,
                    "z_init": value.WKB.z_init,
                    "omega_WKB_sq": value.omega_WKB_sq,
                    "WKB_criterion": value.WKB_criterion,
                    "analytic_G": value.analytic_G,
                    "analytic_Gprime": value.analytic_Gprime,
                },
            )

            # set store_id on behalf of the GkSourceValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: GkWKBIntegration,
        conn,
        table,
        tables,
    ):
        # query the row in GkSource corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["GkSourceValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.parent_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: GkSource object "{obj.label}" did not validate after serialization'
            )

        conn.execute(
            sqla.update(table)
            .where(table.c.serial == obj.store_id)
            .values(validated=validated)
        )

        return validated

    @staticmethod
    def validate_on_startup(conn, table, tables, prune=False):
        # query the datastore for any GkSource objects that are not validated

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["GkSourceValue"]
        tags_table = tables["GkSource_tags"]

        # bake results into a list so that we can close this query; we are going to want to run
        # another one as we process the rows from this one
        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_samples,
                    wavenumber_table.c.k_inv_Mpc.label("k_inv_Mpc"),
                    atol_table.c.log10_tol.label("log10_atol"),
                    rtol_table.c.log10_tol.label("log10_rtol"),
                    redshift_table.c.z.label("z_response"),
                )
                .select_from(
                    table.join(atol_table, atol_table.c.serial == table.c.atol_serial)
                    .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
                    .join(
                        redshift_table,
                        redshift_table.c.serial == table.c.z_response_serial,
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
            ">> GkSource objects",
            "     The following unvalidated GkSource objects were detected in the datastore:",
        ]
        for obj in not_validated:
            msgs.append(
                f'       -- "{obj.label}" (store_id={obj.serial}) for k={obj.k_inv_Mpc:.5g}/Mpc and z_response={obj.z_response:.5g} (log10_atol={obj.log10_atol}, log10_rtol={obj.log10_rtol})'
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.parent_serial == obj.serial,
                )
            ).scalar()
            msgs.append(
                f"          contains {rows} z-sample values | expected={obj.z_samples}"
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


class sqla_GkSourceValue_factory(SQLAFactoryBase):
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
                    sqla.ForeignKey("GkSource.serial"),
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
                sqla.Column("G", sqla.Float(64), nullable=True),
                sqla.Column("Gprime", sqla.Float(64), nullable=True),
                sqla.Column("H_ratio", sqla.Float(64), nullable=True),
                sqla.Column("theta_mod_2pi", sqla.Float(64), nullable=True),
                sqla.Column("theta_div_2pi", sqla.Integer, nullable=True),
                sqla.Column("raw_theta_div_2pi", sqla.Integer, nullable=True),
                sqla.Column("G_WKB", sqla.Float(64), nullable=True),
                sqla.Column("new_G_WKB", sqla.Float(64), nullable=True),
                sqla.Column("abs_G_WKB_err", sqla.Float(64), nullable=True),
                sqla.Column("rel_G_WKB_err", sqla.Float(64), nullable=True),
                sqla.Column("sin_coeff", sqla.Float(64), nullable=True),
                sqla.Column("cos_coeff", sqla.Float(64), nullable=True),
                sqla.Column("z_init", sqla.Float(64), nullable=True),
                sqla.Column("omega_WKB_sq", sqla.Float(64), nullable=True),
                sqla.Column("WKB_criterion", sqla.Float(64), nullable=True),
                sqla.Column("analytic_G", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Gprime", sqla.Float(64), nullable=True),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        parent_serial = payload.get("parent_serial", None)

        model_proxy: Optional[ModelProxy] = payload.get("model", None)
        k: Optional[wavenumber_exit_time] = payload.get("k", None)
        z_source: Optional[redshift] = payload.get("z_source", None)

        has_serial = all([parent_serial is not None])
        has_model = all([model_proxy is not None, k is not None, z_source is not None])

        if all([has_serial, has_model]):
            print(
                "## GkSourceValue.build(): both an source serial number and a (model, wavenumber, z_source) set were queried. Only the serial number will be used."
            )

        if not any([has_serial, has_model]):
            raise RuntimeError(
                "GkSourceValue.build(): at least one of a source serial number and a (model, wavenumber, z_source) set must be supplied."
            )

        if has_serial:
            return sqla_GkSourceValue_factory._build_impl_serial(
                payload, conn, table, inserter, tables, inserters
            )

        return sqla_GkSourceValue_factory._build_impl_model(
            payload, conn, table, inserter, tables, inserters
        )

    @staticmethod
    def _build_impl_serial(payload, conn, table, inserter, tables, inserters):
        z_source = payload["z_source"]

        parent_serial: int = payload["parent_serial"]

        G = payload.get("G", None)
        Gprime = payload.get("Gprime", None)

        H_ratio = payload.get("H_ratio", None)
        theta_mod_2pi = payload.get("theta_mod_2pi", None)
        theta_div_2pi = payload.get("theta_div_2pi", None)
        raw_theta_div_2pi = payload.get("raw_theta_div_2pi", None)

        G_WKB = payload.get("G_WKB", None)
        new_G_WKB = payload.get("new_G_WKB", None)
        abs_G_WKB_err = payload.get("abs_G_WKB_err", None)
        rel_G_WKB_err = payload.get("rel_G_WKB_err", None)

        sin_coeff = payload.get("sin_coeff", None)
        cos_coeff = payload.get("cos_coeff", None)
        z_init = payload.get("z_init", None)

        omega_WKB_sq = payload.get("omega_WKB_sq", None)
        WKB_criterion = payload.get("WKB_criterion", None)

        analytic_G: Optional[float] = payload.get("analytic_G", None)
        analytic_Gprime: Optional[float] = payload.get("analytic_Gprime", None)

        has_numerical = all([G is not None, Gprime is not None])
        has_WKB = all(
            [
                G_WKB is not None,
                H_ratio is not None,
                theta_mod_2pi is not None,
                theta_div_2pi is not None,
                raw_theta_div_2pi is not None,
                omega_WKB_sq is not None,
            ]
        )

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.G,
                    table.c.Gprime,
                    table.c.H_ratio,
                    table.c.theta_mod_2pi,
                    table.c.theta_div_2pi,
                    table.c.raw_theta_div_2pi,
                    table.c.G_WKB,
                    table.c.new_G_WKB,
                    table.c.abs_G_WKB_err,
                    table.c.rel_G_WKB_err,
                    table.c.sin_coeff,
                    table.c.cos_coeff,
                    table.c.z_init,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                    table.c.analytic_G,
                    table.c.analytic_Gprime,
                ).filter(
                    table.c.parent_serial == parent_serial,
                    table.c.z_serial == z_source.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkSourceValue.build(): multiple results found when querying for GkSourceValue"
            )
            raise e

        if row_data is None:
            if not has_numerical and not has_WKB:
                raise (
                    "GkSourceValue().build(): result was not found in datastore, but a data payload was not provided"
                )

            store_id = inserter(
                conn,
                {
                    "parent_serial": parent_serial,
                    "z_source_serial": z_source.store_id,
                    "G": G,
                    "Gprime": Gprime,
                    "H_ratio": H_ratio,
                    "theta_mod_2pi": theta_mod_2pi,
                    "theta_div_2pi": theta_div_2pi,
                    "raw_theta_div_2pi": raw_theta_div_2pi,
                    "G_WKB": G_WKB,
                    "new_G_WKB": new_G_WKB,
                    "abs_G_WKB_err": abs_G_WKB_err,
                    "rel_G_WKB_err": rel_G_WKB_err,
                    "sin_coeff": sin_coeff,
                    "cos_coeff": cos_coeff,
                    "z_init": z_init,
                    "omega_WKB_sq": omega_WKB_sq,
                    "WKB_criterion": WKB_criterion,
                    "analytic_G": analytic_G,
                    "analytic_Gprime": analytic_Gprime,
                },
            )

            attribute_set = {"_new_insert": True}
        else:
            store_id = row_data.serial

            G = row_data.G
            Gprime = row_data.Gprime

            H_ratio = row_data.H_ratio
            theta_mod_2pi = row_data.theta_mod_2pi
            theta_div_2pi = row_data.theta_div_2pi
            raw_theta_div_2pi = row_data.raw_theta_div_2pi

            G_WKB = row_data.G_WKB
            new_G_WKB = row_data.new_G_WKB
            abs_G_WKB_err = row_data.abs_G_WKB_err
            rel_G_WKB_err = row_data.rel_G_WKB_err

            sin_coeff = row_data.sin_coeff
            cos_coeff = row_data.cos_coeff
            z_init = row_data.z_init

            omega_WKB_sq = row_data.omega_WKB_sq
            WKB_criterion = row_data.WKB_criterion
            analytic_G = row_data.analytic_G
            analytic_Gprime = row_data.analytic_Gprime

            attribute_set = {"_deserialized": True}

        obj = GkSourceValue(
            store_id=store_id,
            z_source=z_source,
            G=G,
            Gprime=Gprime,
            H_ratio=H_ratio,
            theta_mod_2pi=theta_mod_2pi,
            theta_div_2pi=theta_div_2pi,
            raw_theta_div_2pi=raw_theta_div_2pi,
            G_WKB=G_WKB,
            new_G_WKB=new_G_WKB,
            abs_G_WKB_err=abs_G_WKB_err,
            rel_G_WKB_err=rel_G_WKB_err,
            sin_coeff=sin_coeff,
            cos_coeff=cos_coeff,
            z_init=z_init,
            omega_WKB_sq=omega_WKB_sq,
            WKB_criterion=WKB_criterion,
            analytic_G=analytic_G,
            analytic_Gprime=analytic_Gprime,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _build_impl_model(payload, conn, table, inserter, tables, inserters):
        z_source = payload["z_source"]

        model_proxy: ModelProxy = payload["model"]
        k: wavenumber_exit_time = payload["k"]
        z_response: redshift = payload["z_response"]

        atol: Optional[tolerance] = payload.get("atol", None)
        rtol: Optional[tolerance] = payload.get("rtol", None)
        tags: Optional[List[store_tag]] = payload.get("tags", None)

        source_table = tables["GkSource"]

        try:
            # TODO: benchmarking suggests this query is indistinguishable from filtering directly on the
            #  GkSource serial number (if we only knew what it was), so this may be about
            #  as good as we can do. But it is still slow. For production use, should look at how this
            #  can be improved.
            source_query = sqla.select(source_table.c.serial).filter(
                source_table.c.model_serial == model_proxy.store_id,
                source_table.c.wavenumber_exit_serial == k.store_id,
                source_table.c.z_response_serial == z_response.store_id,
                source_table.c.validated == True,
            )

            if atol is not None:
                source_query = source_query.filter(
                    source_table.c.atol_serial == atol.store_id
                )

            if rtol is not None:
                source_query = source_query.filter(
                    source_table.c.rtol_serial == rtol.store_id
                )

            count = 0
            for tag in tags:
                tag: store_tag
                tab = tables["GkSource_tags"].alias(f"tag_{count}")
                count += 1
                source_query = source_query.join(
                    tab,
                    and_(
                        tab.c.parent_serial == source_table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            subquery = source_query.subquery()

            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.G,
                    table.c.Gprime,
                    table.c.H_ratio,
                    table.c.theta_mod_2pi,
                    table.c.theta_div_2pi,
                    table.c.raw_theta_div_2pi,
                    table.c.G_WKB,
                    table.c.new_G_WKB,
                    table.c.abs_G_WKB_err,
                    table.c.rel_G_WKB_err,
                    table.c.sin_coeff,
                    table.c.cos_coeff,
                    table.c.z_init,
                    table.c.omega_WKB_sq,
                    table.c.WKB_criterion,
                    table.c.analytic_G,
                    table.c.analytic_Gprime,
                )
                .select_from(
                    subquery.join(table, table.c.parent_serial == subquery.c.serial)
                )
                .filter(
                    table.c.z_source_serial == z_source.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkSourceValue.build(): multiple results found when querying for GkSourceValue"
            )
            raise e

        if row_data is None:
            # return empty object
            obj = GkSourceValue(store_id=None, z_source=z_source)
            obj._k_exit = k
            obj._z_response = z_response
            return obj

        obj = GkSourceValue(
            store_id=row_data.serial,
            z_source=z_source,
            G=row_data.G,
            Gprime=row_data.Gprime,
            H_ratio=row_data.H_ratio,
            theta_mod_2pi=row_data.theta_mod_2pi,
            theta_div_2pi=row_data.theta_div_2pi,
            raw_theta_div_2pi=row_data.raw_theta_div_2pi,
            G_WKB=row_data.G_WKB,
            new_G_WKB=row_data.new_G_WKB,
            abs_G_WKB_err=row_data.abs_G_WKB_err,
            rel_G_WKB_err=row_data.rel_G_WKB_err,
            sin_coeff=row_data.sin_coeff,
            cos_coeff=row_data.cos_coeff,
            z_init=row_data.z_init,
            omega_WKB_sq=row_data.omega_WKB_sq,
            WKB_criterion=row_data.WKB_criterion,
            analytic_G=row_data.analytic_G,
            analytic_Gprime=row_data.analytic_Gprime,
        )
        obj._deserialized = True
        obj._k_exit = k
        obj._z_response = z_response
        return obj
