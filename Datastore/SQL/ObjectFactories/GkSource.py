"""
Datastore factories for GkSource and its per-redshift GkSourceValue rows.

SCHEMA NOTE (prompts/tolerance-convergence, prompt 05). "atol_serial" and "rtol_serial" are gone,
and nothing stands in their place.

They described nothing and there is no order to put there either: compute() calls
assemble_GkSource_values, which stitches the stored numeric and WKB results together and
integrates nothing. GkSource has no quadrature, no solver and therefore no accuracy parameter of
its own -- its accuracy is entirely that of the GkNumericIntegration and GkWKBIntegration rows it
assembles, each of which carries its own. The user settled the drop on 2026-09-18
(prompts/tolerance-convergence/README.md §7 D3).

SCHEMA NOTE (prompts/datastore-integrity, prompt 09b). A GkSource for (k, z_response) is
assembled, for each z_source of its grid, from the value at z_response of one GkNumericIntegration,
one GkWKBIntegration, or both (ComputeTargets/GkSource.py, assemble_GkSource_values). Those rows
are its parents, and they are a set. They are recorded in a new table, **GkSource_parents**, on
the GkSource's own shard (every Gk row shards on k, config/sharding.py): one row per z_source,
(gk_source_serial, z_source_serial, numeric_serial, wkb_serial), each with its foreign key and the
last two nullable, written in the same transaction as the GkSource row. It is the record of exactly
what was assembled. A new key column, **parents_digest** on GkSource, is the SHA-256 of that set
(GkSource_parents_digest, below), and build() filters on it, so a lookup says "assembled from
exactly this set" in one comparison. store() computes the digest from the rows it writes to
GkSource_parents, never from anything else, and refuses an empty set.

A lookup names its parent set ("parents"), or asks for the reader mode ("labels_only": True) that
D1 (prompts/datastore-integrity README §6.2) allows a reader holding only physical labels: it
refuses when more than one stored GkSource matches, naming their digests and the z_sources at
which their parents differ. A lookup that does neither raises.
"""

import hashlib
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
from ComputeTargets.GkSource import GkSource_parent_set, GkSource_parents_differ
from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber_exit_time, redshift_array, redshift
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from config.defaults import DEFAULT_STRING_LENGTH


def GkSource_parents_digest(members) -> str:
    """
    The GkSource key column "parents_digest" (prompts/datastore-integrity, prompt 09b): SHA-256
    (hex) over the sorted (z_source_serial, numeric_serial, wkb_serial) members of a parent set,
    each serial an integer or None, as compact JSON. build() and store() both call this, and no
    other code forms the digest.
    """
    canonical = [list(m) for m in GkSource_parent_set(members)]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_parent_sets(conn, tables, serials) -> dict:
    """GkSource serial -> the parent set its GkSource_parents rows record, for ``serials``."""
    parents_table = tables["GkSource_parents"]
    found = {serial: [] for serial in serials}
    for member in conn.execute(
        sqla.select(
            parents_table.c.gk_source_serial,
            parents_table.c.z_source_serial,
            parents_table.c.numeric_serial,
            parents_table.c.wkb_serial,
        ).filter(parents_table.c.gk_source_serial.in_(list(serials)))
    ):
        found[member.gk_source_serial].append(
            (member.z_source_serial, member.numeric_serial, member.wkb_serial)
        )
    return {
        serial: tuple(sorted(members, key=lambda m: m[0]))
        for serial, members in found.items()
    }


def _describe_matches(conn, tables, candidates) -> str:
    """
    What distinguishes several stored GkSources that match one set of physical labels (D1): each
    one's parents_digest, and the z_source serials at which their parent sets differ.
    ``candidates`` is a list of (serial, parents_digest).
    """
    sets = _read_parent_sets(conn, tables, [serial for serial, _ in candidates])
    serials = sorted(sets)
    differ = set()
    for other in serials[1:]:
        differ.update(GkSource_parents_differ(sets[serials[0]], sets[other]))
    named = "; ".join(
        f"serial={serial}, parents_digest={digest}"
        for serial, digest in sorted(candidates)
    )
    return (
        f"{named}. Their parent sets differ at {len(differ)} z_source serial(s): "
        f"{sorted(differ)[:10]}"
    )


class sqla_GkSourceParents_factory(SQLAFactoryBase):
    """
    GkSource_parents (prompts/datastore-integrity, prompt 09b): the record of exactly which Gk
    rows a GkSource was assembled from, one row per z_source. It has no lookup of its own: its rows
    are written by sqla_GkSource_factory.store(), read by build()'s reader mode and by the
    inventory, and never built as objects. The table is on the GkSource's shard, as every row it
    names is, so each reference is a declared foreign key.
    """

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
                    "gk_source_serial",
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
                # NULL: no numeric value was assembled at this z_source
                sqla.Column(
                    "numeric_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkNumericIntegration.serial"),
                    index=True,
                    nullable=True,
                ),
                # NULL: no WKB value was assembled at this z_source
                sqla.Column(
                    "wkb_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkWKBIntegration.serial"),
                    index=True,
                    nullable=True,
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        raise NotImplementedError


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
                # no accuracy parameter: this object assembles, it does not integrate (module
                # docstring above). The atol_serial / rtol_serial pair that used to key it was
                # removed by prompts/tolerance-convergence prompt 05 and nothing replaces it.
                sqla.Column(
                    "z_response_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                # the digest of the parent set GkSource_parents records for this row (module
                # docstring; prompts/datastore-integrity, prompt 09b). Part of the lookup key
                sqla.Column(
                    "parents_digest",
                    sqla.String(64),
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
                    "numeric_smallest_z_serial",
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

        k_exit: wavenumber_exit_time = payload["k"]
        model_proxy: ModelProxy = payload["model"]
        z_sample: redshift_array = payload["z_sample"]
        z_response: redshift = payload["z_response"]

        # the key's parent set (module docstring), or the reader mode D1 allows; never neither,
        # and never both
        parents = payload.get("parents", None)
        labels_only: bool = payload.get("labels_only", False)
        if labels_only and parents is not None:
            raise ValueError(
                "GkSource.build(): both a parent set and labels_only were given; a keyed lookup "
                "names its parents, and the reader mode names none"
            )
        if not labels_only and parents is None:
            raise RuntimeError(
                "GkSource.build(): no parent set was given. A GkSource is keyed on the Gk "
                "numeric and WKB rows it is assembled from (parents=...); a reader holding only "
                "physical labels must ask for labels_only=True explicitly"
            )
        if parents is not None:
            parents = GkSource_parent_set(parents)

        tag_table = tables["GkSource_tags"]
        redshift_table = tables["redshift"]

        smallest_numeric_z_table = redshift_table.alias("smallest_numeric")
        largest_WKB_z_table = redshift_table.alias("largest_WKB")

        # notice that we query only for validated data
        query = (
            sqla.select(
                table.c.serial,
                table.c.label,
                table.c.z_response_serial,
                redshift_table.c.z.label("z_response"),
                table.c.z_samples,
                table.c.numeric_smallest_z_serial,
                smallest_numeric_z_table.c.z.label("numeric_smallest_z"),
                smallest_numeric_z_table.c.source.label("nsz_is_source"),
                smallest_numeric_z_table.c.response.label("nsz_is_response"),
                table.c.primary_WKB_largest_z_serial,
                largest_WKB_z_table.c.z.label("primary_WKB_largest_z"),
                largest_WKB_z_table.c.source.label("plz_is_source"),
                largest_WKB_z_table.c.response.label("plz_is_response"),
                table.c.metadata,
                table.c.parents_digest,
            )
            .select_from(
                table.join(
                    redshift_table, redshift_table.c.serial == table.c.z_response_serial
                )
                .outerjoin(
                    smallest_numeric_z_table,
                    smallest_numeric_z_table.c.serial
                    == table.c.numeric_smallest_z_serial,
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
            )
        )

        if z_response is not None:
            query = query.filter(
                table.c.z_response_serial == z_response.store_id,
            )

        # the parent set, through its digest (prompt 09b)
        if not labels_only:
            query = query.filter(
                table.c.parents_digest == GkSource_parents_digest(parents)
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
            if labels_only:
                # D1: the physical labels may select, but more than one stored row is refused,
                # naming what distinguishes them, rather than one of them chosen
                candidates = []
                for candidate in conn.execute(query):
                    candidates.append((candidate.serial, candidate.parents_digest))
                if len(candidates) > 1:
                    raise MultipleResultsFound(
                        f"GkSource.build(): labels_only lookup matched {len(candidates)} stored "
                        f"rows (model serial={model_proxy.store_id}, k serial={k_exit.store_id}, "
                        f"z_response serial="
                        f"{z_response.store_id if z_response is not None else None}): "
                        f"{_describe_matches(conn, tables, candidates)}. Refusing to choose one; "
                        f"name the parent set."
                    )

            row_data = conn.execute(query).one_or_none()

            # the reader mode's one match carries the parent set its row records
            if labels_only and row_data is not None:
                parents = _read_parent_sets(conn, tables, [row_data.serial])[
                    row_data.serial
                ]
        except MultipleResultsFound as e:
            print(
                f"!! GkSource.build(): multiple results found when querying for GkSource"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return GkSource(
                payload=None,
                label=label,
                k=k_exit,
                model=model_proxy,
                z_response=z_response,
                z_sample=z_sample,
                tags=tags,
                parents=parents,
                labels_only=labels_only,
            )

        store_id = row_data.serial
        store_label = row_data.label

        numeric_smallest_z = None
        if row_data.numeric_smallest_z_serial is not None:
            numeric_smallest_z = redshift(
                store_id=row_data.numeric_smallest_z_serial,
                z=row_data.numeric_smallest_z,
                is_source=row_data.nsz_is_source,
                is_response=row_data.nsz_is_response,
            )

        primary_WKB_largest_z = None
        if row_data.primary_WKB_largest_z_serial is not None:
            primary_WKB_largest_z = redshift(
                store_id=row_data.primary_WKB_largest_z_serial,
                z=row_data.primary_WKB_largest_z,
                is_source=row_data.plz_is_source,
                is_response=row_data.plz_is_response,
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
                    redshift_table.c.source.label("z_source_is_source"),
                    redshift_table.c.response.label("z_source_is_response"),
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
                    value_table.c.analytic_G_rad,
                    value_table.c.analytic_Gprime_rad,
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
                z_value = redshift(
                    store_id=row.z_source_serial,
                    z=row.z_source,
                    is_source=row.z_source_is_source,
                    is_response=row.z_source_is_response,
                )
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
                        analytic_G_rad=row.analytic_G_rad,
                        analytic_Gprime_rad=row.analytic_Gprime_rad,
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
                "numeric_smallest_z": numeric_smallest_z,
                "primary_WKB_largest_z": primary_WKB_largest_z,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else None
                ),
            },
            k=k_exit,
            model=model_proxy,
            label=store_label,
            z_response=redshift(
                store_id=(row_data.z_response_serial), z=(row_data.z_response)
            ),
            z_sample=imported_z_sample,
            tags=tags,
            parents=parents,
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
        # the record of exactly what was assembled (prompt 09b): the members compute()'s
        # assemble_GkSource_values reports it read, never the set the lookup asked for. An object
        # with none is refused before anything is written (the user's decision, 2026-09-30)
        assembled = obj.assembled_parents
        if assembled is None or len(assembled) == 0:
            raise ValueError(
                "GkSource.store(): refusing to write a GkSource with an empty parent set "
                f"(k serial={obj._k_exit.store_id}, z_response serial="
                f"{obj.z_response.store_id}); a GkSource is assembled from at least one Gk "
                "numeric or WKB row, and records which"
            )
        parent_rows = [
            {"z_source_serial": z_source, "numeric_serial": numeric, "wkb_serial": wkb}
            for z_source, numeric, wkb in GkSource_parent_set(assembled)
        ]

        # the key column is computed from the rows written below, and from nothing else
        parents_digest = GkSource_parents_digest(
            (
                member["z_source_serial"],
                member["numeric_serial"],
                member["wkb_serial"],
            )
            for member in parent_rows
        )

        store_id = inserter(
            conn,
            {
                "label": obj.label,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "model_serial": obj.model_proxy.store_id,
                "z_response_serial": obj.z_response.store_id,
                "parents_digest": parents_digest,
                "z_max_serial": obj.z_sample.max.store_id,
                "z_samples": len(obj.values),
                "numeric_smallest_z_serial": (
                    obj.numeric_smallest_z.store_id
                    if obj.numeric_smallest_z is not None
                    else None
                ),
                "primary_WKB_largest_z_serial": (
                    obj.primary_WKB_largest_z.store_id
                    if obj.primary_WKB_largest_z is not None
                    else None
                ),
                "metadata": (
                    json.dumps(obj._metadata) if obj._metadata is not None else None
                ),
                "validated": False,
            },
        )

        # set store_id on behalf of the GkSource instance
        obj._my_id = store_id

        # the parent set, one row per z_source, in the same transaction as the GkSource row
        parents_inserter = inserters["GkSource_parents"]
        for member in parent_rows:
            parents_inserter(
                conn,
                {
                    "gk_source_serial": store_id,
                    "z_source_serial": member["z_source_serial"],
                    "numeric_serial": member["numeric_serial"],
                    "wkb_serial": member["wkb_serial"],
                },
            )

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
                    "analytic_G_rad": value.analytic_G_rad,
                    "analytic_Gprime_rad": value.analytic_Gprime_rad,
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

        redshift_table = tables["redshift"]
        wavenumber_exit_table = tables["wavenumber_exit_time"]
        wavenumber_table = tables["wavenumber"]
        value_table = tables["GkSourceValue"]
        tags_table = tables["GkSource_tags"]
        parents_table = tables["GkSource_parents"]

        # bake results into a list so that we can close this query; we are going to want to run
        # another one as we process the rows from this one
        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_samples,
                    wavenumber_table.c.k_inv_Mpc.label("k_inv_Mpc"),
                    redshift_table.c.z.label("z_response"),
                )
                .select_from(
                    table.join(
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
                f'       -- "{obj.label}" (store_id={obj.serial}) for k={obj.k_inv_Mpc:.5g}/Mpc and z_response={obj.z_response:.5g}'
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
                # a pruned GkSource's parent-set rows go with it (prompt 09b), or they would be
                # orphans naming a row that is no longer there
                conn.execute(
                    sqla.delete(parents_table).where(
                        parents_table.c.gk_source_serial.in_(invalid_serials)
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
        # the physical identity of a GkSource row (store-fingerprint prompt 02): what build() filters
        # on -- the model, the wavenumber exit and z_response (optional in the lookup, always in
        # the key) and, since prompts/datastore-integrity prompt 09b, its parent set -- with its
        # tags, its validated flag and its GkSourceValue count. The parent set is resolved
        # through GkSource_parents to the parents' own identities: for each member its z_source
        # and the reference digests of its numeric and WKB rows, or None. parents_digest is not
        # read: it is formed from serials, which are never identity
        from Datastore.store_inventory import InventorySpec, Parent, ParentSet

        return InventorySpec(
            parents={
                "model": Parent("model_serial", "BackgroundModel"),
                "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
                "z_response": Parent("z_response_serial", "redshift"),
            },
            parent_sets={
                "parents": ParentSet(
                    table="GkSource_parents",
                    owner="gk_source_serial",
                    members={
                        "z_source": Parent("z_source_serial", "redshift"),
                        "numeric": Parent(
                            "numeric_serial", "GkNumericIntegration", nullable=True
                        ),
                        "wkb": Parent("wkb_serial", "GkWKBIntegration", nullable=True),
                    },
                )
            },
            tags=("GkSource_tags", "parent_serial"),
            values=("GkSourceValue", "parent_serial"),
            validated="validated",
        )


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
                sqla.Column("analytic_G_rad", sqla.Float(64), nullable=True),
                sqla.Column("analytic_Gprime_rad", sqla.Float(64), nullable=True),
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

        analytic_G: Optional[float] = payload.get("analytic_G_rad", None)
        analytic_Gprime: Optional[float] = payload.get("analytic_Gprime_rad", None)

        has_numeric = all([G is not None, Gprime is not None])
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
                    table.c.analytic_G_rad,
                    table.c.analytic_Gprime_rad,
                ).filter(
                    table.c.parent_serial == parent_serial,
                    table.c.z_source_serial == z_source.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkSourceValue.build(): multiple results found when querying for GkSourceValue"
            )
            raise e

        if row_data is None:
            if not has_numeric and not has_WKB:
                raise RuntimeError(
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
                    "analytic_G_rad": analytic_G,
                    "analytic_Gprime_rad": analytic_Gprime,
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
            analytic_G = row_data.analytic_G_rad
            analytic_Gprime = row_data.analytic_Gprime_rad

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
            analytic_G_rad=analytic_G,
            analytic_Gprime_rad=analytic_Gprime,
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

        tags: List[store_tag] = payload.get("tags", [])

        source_table = tables["GkSource"]

        try:
            # TODO: benchmarking suggests this query is indistinguishable from filtering directly on the
            #  GkSource serial number (if we only knew what it was), so this may be about
            #  as good as we can do. But it is still slow. For production use, should look at how this
            #  can be improved.
            # a read by physical labels (D1, prompts/datastore-integrity README §6.2): its
            # parents_digest is selected so that more than one matching GkSource can be refused,
            # naming what distinguishes them, rather than their values joined together (the
            # user's decision on prompt 09b, 2026-09-30)
            source_query = sqla.select(
                source_table.c.serial, source_table.c.parents_digest
            ).filter(
                source_table.c.model_serial == model_proxy.store_id,
                source_table.c.wavenumber_exit_serial == k.store_id,
                source_table.c.z_response_serial == z_response.store_id,
                source_table.c.validated == True,
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

            candidates = []
            for candidate in conn.execute(source_query):
                candidates.append((candidate.serial, candidate.parents_digest))
            if len(candidates) > 1:
                raise MultipleResultsFound(
                    f"GkSourceValue.build(): {len(candidates)} stored GkSources match one set of "
                    f"physical labels (model serial={model_proxy.store_id}, k serial="
                    f"{k.store_id}, z_response serial={z_response.store_id}): "
                    f"{_describe_matches(conn, tables, candidates)}. Refusing to read a value "
                    f"from more than one of them."
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
                    table.c.analytic_G_rad,
                    table.c.analytic_Gprime_rad,
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
            analytic_G_rad=row_data.analytic_G_rad,
            analytic_Gprime_rad=row_data.analytic_Gprime_rad,
        )
        obj._deserialized = True
        obj._k_exit = k
        obj._z_response = z_response
        return obj
