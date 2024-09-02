from typing import Iterable

import ray
from ray.actor import ActorHandle

import sqlalchemy as sqla
from sqlalchemy import func, and_

from CosmologyModels.base import CosmologyBase
from Datastore import DatastoreObject
from defaults import DEFAULT_FLOAT_PRECISION
from utilities import find_horizon_exit_time, check_units


class wavenumber(DatastoreObject):
    def __init__(self, store: ActorHandle, k_inv_Mpc: float, units):
        """
        Construct a datastore-backed object representing a wavenumber, e.g.,
        used to sample a transfer function or power spectrum
        :param store: handle to datastore actor
        :param k_value: wavenumber, measured in 1/Mpc
        :param units: units block (e.g. Mpc-based units)
        """
        DatastoreObject.__init__(self, store)

        # units are available for inspection
        self.units = units

        self.k_inv_Mpc = k_inv_Mpc
        self.k = k_inv_Mpc / units.Mpc

        # request our own unique id from the datastore
        self._my_id = ray.get(self._store.query.remote(self))

    def __float__(self):
        """
        Silently cast to float. Returns dimensionful wavenumber.
        :return:
        """
        return self.k

    @staticmethod
    def generate_columns():
        return {"version": False, "columns": [sqla.Column("k_inv_Mpc", sqla.Float(64))]}

    def build_query(self, table, query):
        return query.filter(
            func.abs(table.c.k_inv_Mpc - self.k_inv_Mpc) < DEFAULT_FLOAT_PRECISION
        )

    def build_payload(self):
        return {"k_inv_Mpc": self.k_inv_Mpc}


class wavenumber_array:
    def __init__(self, store: ActorHandle, k_inv_Mpc_array: Iterable, units):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        :param store: handle to datastore actor
        :param k_inv_Mpc_array: array of wavenumbers, measured in 1/Mpc
        :param units: units block
        """
        self._store: ActorHandle = store

        # build array
        self._k_array = [
            wavenumber(store, k_inv_Mpc, units) for k_inv_Mpc in k_inv_Mpc_array
        ]

    def __iter__(self):
        for k in self._k_array:
            yield k

    def __getitem__(self, key):
        return self._k_array[key]


class wavenumber_exit_time(DatastoreObject):
    def __init__(self, store: ActorHandle, k: wavenumber, cosmology: CosmologyBase):
        """
        Construct a datastore-backed object representing the horizon exit time
        for a mode of wavenumber k
        :param store: handle to datastore actor
        :param k: wavenumber object
        :param cosmology: cosmology object satisfying the CosmologyBase concept
        """
        DatastoreObject.__init__(self, store)
        check_units(k, cosmology)

        self.k = k
        self.cosmology = cosmology

        self._z_exit = None

        # request our own unique id from the datastore
        data = ray.get(self._store.query.remote(self, serial_only=False))
        self._my_id = data["store_id"]
        self._z_exit = data["data"]["z_exit"]

    @staticmethod
    def generate_columns():
        # not yet setting up foreign key links for k_serial and cosmology_serial
        # for cosmology_serial, the problem is that we have different providers of the CosmologyBase concept.
        # It's not yet clear what is the best way to deal with this
        return {
            "version": True,
            "stepping": True,
            "columns": [
                sqla.Column("wavenumber_serial", sqla.Integer, nullable=False),
                sqla.Column("cosmology_serial", sqla.Integer, nullable=False),
                sqla.Column("z_exit", sqla.Float(64)),
            ],
        }

    @property
    def stepping(self):
        # stepping 0: initial implementation using solve_ivp and an event handler to determine when ln(k/aH) = 0
        return 0

    def build_query(self, table, query):
        return query.filter(
            and_(
                table.c.k_serial == self.k.store_id,
                table.c.cosmology_serial == self.cosmology.store_id,
            )
        )

    def build_payload(self):
        self._z_exit = find_horizon_exit_time(self.cosmology, self.k)

        return {
            "wavenumber_serial": self.k.store_id,
            "cosmology_serial": self.cosmology.store_id,
            "z_exit": self._z_exit,
        }

    @property
    def z_exit(self):
        return self._z_exit
