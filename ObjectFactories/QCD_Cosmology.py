"""
Datastore factory for QCD_Cosmology.

SCHEMA NOTE (prompts/qcd-background-audit, prompt 03). The lookup key carries a
"T_z_representation" column beside the seven parameter values and log10_max_z. It holds
LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION, which identifies how T(z) is represented -- the
node solve's tolerance, the quantity splined, the node count, the spline order, the segmentation,
and the break-point set -- none of which the parameter values can see.

Without it, a row computed under one representation is served back, under the same serial, to a
build() that asks for another: the same cosmology row matches, every BackgroundModel keyed on that
serial deserialises, and its tau, cs_tau and friction_F limbs are the other representation's, with
no exception and no column that differs.
"""

from math import log10

import sqlalchemy as sqla

from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from datastorekit.SQL.factory_base import SQLAFactoryBase
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

        # scalar_one_or_none() raises MultipleResultsFound on a duplicate rather than silently
        # serving the first row scalar() would have returned (prompts/datastore-integrity, prompt
        # 03, L3). Any database error, that one included, reaches the caller as itself.
        store_id = conn.execute(query).scalar_one_or_none()

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
    def inventory_spec():
        # the physical identity of a QCD_Cosmology row (store-fingerprint prompt 02): the six
        # parameters, log10_max_z and T_z_representation, which build() matches. The name is
        # descriptive and is not identity
        from datastorekit.store_inventory import InventorySpec

        return InventorySpec(
            leaves=(
                "omega_m",
                "omega_cc",
                "h",
                "f_baryon",
                "T_CMB_Kelvin",
                "Neff",
                "log10_max_z",
                "T_z_representation",
            ),
        )
