"""
Datastore factory for QCD_Cosmology.

SCHEMA NOTE (prompts/qcd-background-audit, prompt 03). The lookup key carries a
"T_z_representation" column beside the seven parameter values and log10_max_z. It holds
LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION, which identifies how T(z) is represented -- the
node solve's tolerance, the quantity splined, the node count, the spline order, the segmentation,
and the break-point set -- none of which the parameter values can see.

Without it, a datastore written before a representation change is served back under the same
serial: the same cosmology row matches, every BackgroundModel keyed on that serial deserialises,
and its tau, cs_tau and friction_F limbs are the old background's, with no exception and no column
that differs. A datastore whose QCD_Cosmology table lacks the column predates prompt 03 and its
QCD half must be regenerated; there is no migration, because the rows record no representation
and there is no defensible value to assume for them. Datastore._ensure_tables() creates missing
tables but never alters an existing one, so the mismatch surfaces when SQLite executes build()'s
select ("no such column"); build() below catches that and raises with this message rather than
letting a SQLAlchemy error a reader cannot interpret escape.
"""

from math import log10

import sqlalchemy as sqla
from sqlalchemy.exc import SQLAlchemyError

from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from config.defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


class sqla_QCDCosmology_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": True,
            "columns": [
                sqla.Column("name", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column("omega_m", sqla.Float(64)),
                sqla.Column("omega_cc", sqla.Float(64)),
                sqla.Column("h", sqla.Float(64)),
                sqla.Column("f_baryon", sqla.Float(64)),
                sqla.Column("T_CMB_Kelvin", sqla.Float(64)),
                sqla.Column("Neff", sqla.Float(64)),
                sqla.Column("log10_max_z", sqla.Float(64)),
                # the identity of the T(z) representation (module docstring above). An integer
                # identifier, not a measured quantity: it is compared for equality and
                # DEFAULT_FLOAT_PRECISION has no business near it.
                sqla.Column(
                    "T_z_representation", sqla.Integer, index=True, nullable=False
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        params = payload["params"]
        units = payload["units"]

        max_z = payload["max_z"]

        log10_max_z = log10(1.0 + max_z)

        name = params.name
        omega_m = params.omega_m
        omega_cc = params.omega_cc
        h = params.h
        f_baryon = params.f_baryon
        T_CMB_Kelvin = params.T_CMB_Kelvin
        Neff = params.Neff

        # the representation is read from its single declaration on the model class, never written
        # out as a literal here: a literal would stop tracking the constant the moment a later
        # prompt bumps it, which is the exact failure this key exists to prevent.
        T_z_representation = QCD_Cosmology.T_Z_REPRESENTATION_VERSION

        query = sqla.select(table.c.serial).filter(
            sqla.and_(
                sqla.func.abs(table.c.omega_m - omega_m) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.omega_cc - omega_cc) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.h - h) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.f_baryon - f_baryon) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.T_CMB_Kelvin - T_CMB_Kelvin)
                < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.Neff - Neff) < DEFAULT_FLOAT_PRECISION,
                sqla.func.abs(table.c.log10_max_z - log10_max_z)
                < DEFAULT_FLOAT_PRECISION,
                # an equality, alongside -- not instead of -- the parameter key that was already
                # here: a row built under a different T(z) representation is a different
                # background and must miss.
                table.c.T_z_representation == T_z_representation,
            )
        )

        try:
            store_id = conn.execute(query).scalar()
        except SQLAlchemyError as e:
            # a datastore written before prompt 03 has no T_z_representation column, and its rows
            # record no representation at all. There is no defensible default to supply, so this
            # fails loudly rather than silently matching -- the defect being repaired is precisely
            # a stale row that looked like a hit. This is the pattern prompts 03 and 04 of
            # prompts/GkTk-remedial set at
            # Datastore/SQL/ObjectFactories/BackgroundModel.py:300-309.
            if "T_z_representation" in str(e):
                raise RuntimeError(
                    "QCD_Cosmology.build(): the QCD_Cosmology table has no "
                    '"T_z_representation" column. This datastore predates the T(z) '
                    "representation becoming part of the cosmology lookup key "
                    "(prompts/qcd-background-audit, prompt 03) and its QCD half must be "
                    "regenerated; there is no migration. Its rows record no representation, so "
                    "the BackgroundModel, Gk and Tk rows hanging from them cannot be told apart "
                    "from rows computed against the background now in the code."
                ) from e
            raise

        # if not present, create a new id using the provided inserter
        if store_id is None:
            insert_data = {
                "name": name,
                "omega_m": omega_m,
                "omega_cc": omega_cc,
                "h": h,
                "f_baryon": f_baryon,
                "T_CMB_Kelvin": T_CMB_Kelvin,
                "Neff": Neff,
                "log10_max_z": log10_max_z,
                "T_z_representation": T_z_representation,
            }
            if "serial" in payload:
                insert_data["serial"] = payload["serial"]
            store_id = inserter(
                conn,
                insert_data,
            )

            attribute_set = {"_new_insert": True}
        else:
            attribute_set = {"_deserialized": True}

        obj = QCD_Cosmology(store_id=store_id, units=units, params=params, max_z=max_z)
        for key, value in attribute_set.items():
            setattr(obj, key, value)

        return obj

    @staticmethod
    def inventory(conn, table, tables, *args, **kwargs):
        earliest_timestamp = conn.execute(
            sqla.select(sqla.func.min(table.c.timestamp))
        ).scalar()
        latest_timestamp = conn.execute(
            sqla.select(sqla.func.max(table.c.timestamp))
        ).scalar()

        # a small configuration table -- a label per row (name plus the
        # distinguishing cosmological parameters) is more useful than a raw
        # value list
        #
        # T_z_representation is placed immediately before log10_max_z rather than at the end: the
        # two of them describe how this row's background was computed and over what range, while
        # name/omega_m/omega_cc/h are the cosmology's physical parameters that precede them. Since
        # prompts/qcd-background-audit prompt 03 it is part of the lookup key (module docstring
        # above): two rows can share every physical parameter and log10_max_z and still be
        # different backgrounds, and without this the inventory renders them as indistinguishable
        # duplicates ([03-qcd-inventory-does-not-report-the-representation]).
        values = [
            {
                "name": row.name,
                "omega_m": row.omega_m,
                "omega_cc": row.omega_cc,
                "h": row.h,
                "T_z_representation": row.T_z_representation,
                "log10_max_z": row.log10_max_z,
            }
            for row in conn.execute(
                sqla.select(
                    table.c.name,
                    table.c.omega_m,
                    table.c.omega_cc,
                    table.c.h,
                    table.c.T_z_representation,
                    table.c.log10_max_z,
                ).order_by(table.c.name)
            )
        ]

        return {
            "earliest_timestamp": earliest_timestamp,
            "latest_timestamp": latest_timestamp,
            "values": values,
        }
