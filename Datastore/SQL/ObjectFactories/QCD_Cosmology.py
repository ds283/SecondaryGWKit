from math import log10

import sqlalchemy as sqla

from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from defaults import DEFAULT_FLOAT_PRECISION, DEFAULT_STRING_LENGTH


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
                sqla.Column("log10_min_z", sqla.Float(64)),
                sqla.Column("log10_max_z", sqla.Float(64)),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        params = payload["params"]
        units = payload["units"]

        min_z = payload["min_z"]
        max_z = payload["max_z"]

        log10_min_z = log10(1.0 + min_z)
        log10_max_z = log10(1.0 + max_z)

        name = params.name
        omega_m = params.omega_m
        omega_cc = params.omega_cc
        h = params.h
        f_baryon = params.f_baryon
        T_CMB_Kelvin = params.T_CMB_Kelvin
        Neff = params.Neff

        store_id = conn.execute(
            sqla.select(table.c.serial).filter(
                sqla.and_(
                    sqla.func.abs(table.c.omega_m - omega_m) < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.omega_cc - omega_cc)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.h - h) < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.f_baryon - f_baryon)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.T_CMB_Kelvin - T_CMB_Kelvin)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.Neff - Neff) < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.log10_min_z - log10_min_z)
                    < DEFAULT_FLOAT_PRECISION,
                    sqla.func.abs(table.c.log10_max_z - log10_max_z)
                    < DEFAULT_FLOAT_PRECISION,
                )
            )
        ).scalar()

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
                "log10_min_z": log10_min_z,
                "log10_max_z": log10_max_z,
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

        obj = QCD_Cosmology(
            store_id=store_id,
            units=units,
            params=params,
            min_z=min_z,
            max_z=max_z,
        )
        for key, value in attribute_set.items():
            setattr(obj, key, value)

        return obj
