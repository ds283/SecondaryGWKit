from typing import Iterable

import ray
from ray.actor import ActorHandle

import sqlalchemy as sqla
from sqlalchemy import func

from defaults import DEFAULT_FLOAT_PRECISION


class wavenumber:
    def __init__(self, store: ActorHandle, k_inv_Mpc: float, units):
        """
        Construct a datastore-backed object representing a wavenumber, e.g.,
        used to sample a transfer function or power spectrum
        :param store: handle to datastore actor
        :param k_value: wavenumber, measured in 1/Mpc
        :param units: units block (e.g. Mpc-based units)
        """
        self._store: ActorHandle = store

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
