import ray
from ray.actor import ActorHandle

import sqlalchemy as sqla
from sqlalchemy import func, and_
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class LambdaCDM:
    def __init__(self, store: ActorHandle, params, units):
        """
        Construct a datastore-backed object representing a simple LambdaCDM cosmology
        :param store: handle to datastore actor
        :param params: parameter block for the LambdaCDM model (e.g. Planck2018)
        :param units: units block (e.g. Mpc-based units)
        """
        self._store: ActorHandle = store

        self._params = params
        self._units = units

        # unpack details of the parameter block so we can access them without extensive nesting
        self.name = params.name

        self.omega_cc = params.omega_cc
        self.omega_m = params.omega_m
        self.f_baryon = params.f_baryon
        self.h = params.h
        self.T_CMB_Kelvin = params.T_CMB_Kelvin
        self.Neff = params.Neff

        # derived dimensionful quantities, expressed in whatever system of units we require
        self.H0 = 100.0 * params.h * units.Kilometre
        self.T_CMB = params.T_CMB_Kelvin * units.Kelvin

        # request our own unique id from the datastore
        self._my_id = ray.get(self._store.query.remote(self))
        print(f'LambdaCDM object "{self.name}" constructed with id={self._my_id}')

    @staticmethod
    def generate_columns():
        return {
            "version": False,
            "columns": [
                sqla.Column("name", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column("omega_m", sqla.Float(64)),
                sqla.Column("omega_cc", sqla.Float(64)),
                sqla.Column("h", sqla.Float(64)),
                sqla.Column("f_baryon", sqla.Float(64)),
                sqla.Column("T_CMB_Kelvin", sqla.Float(64)),
                sqla.Column("Neff", sqla.Float(64)),
            ],
        }

    def build_query(self, table, query):
        return query.filter(
            and_(
                func.abs(table.c.omega_m - self.omega_m) < DEFAULT_FLOAT_PRECISION,
                func.abs(table.c.omega_cc - self.omega_cc) < DEFAULT_FLOAT_PRECISION,
                func.abs(table.c.h - self.h) < DEFAULT_FLOAT_PRECISION,
                func.abs(table.c.f_baryon - self.f_baryon) < DEFAULT_FLOAT_PRECISION,
                func.abs(table.c.T_CMB_Kelvin - self.T_CMB_Kelvin)
                < DEFAULT_FLOAT_PRECISION,
                func.abs(table.c.Neff - self.Neff) < DEFAULT_FLOAT_PRECISION,
            )
        )

    def build_payload(self):
        return {
            "name": self.name,
            "omega_m": self.omega_m,
            "omega_cc": self.omega_cc,
            "h": self.h,
            "f_baryon": self.f_baryon,
            "T_CMB_Kelvin": self.T_CMB_Kelvin,
            "Neff": self.Neff,
        }
