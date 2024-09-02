from typing import Iterable

import ray
from ray.actor import ActorHandle

import sqlalchemy as sqla
from sqlalchemy import func

from Datastore import DatastoreObject
from defaults import DEFAULT_FLOAT_PRECISION


class redshift(DatastoreObject):
    def __init__(self, store: ActorHandle, z: float):
        """
        Construct a datastore-backed object representing a redshift,
        e.g., used to sample a transfer function or power spectrum
        :param store: handle to datastore actor
        :param z: redshift value
        """
        DatastoreObject.__init__(self, store)

        self.z = z

        # request our own unique id from the datastore
        self._my_id = ray.get(self._store.query.remote(self))

    def __float__(self):
        """
        Silently cast to float. Returns numerical value.
        :return:
        """
        return self.z

    @staticmethod
    def generate_columns():
        return {"version": False, "columns": [sqla.Column("z", sqla.Float(64))]}

    def build_query(self, table, query):
        return query.filter(func.abs(table.c.z - self.z) < DEFAULT_FLOAT_PRECISION)

    def build_payload(self):
        return {"z": self.z}


class redshift_array:
    def __init__(self, store: ActorHandle, z_array: Iterable):
        """
        Construct a datastore-backed object representing an array of redshifts
        :param store: handle to datastore actor
        :param z_array: array of redshift value
        """
        self._store: ActorHandle = store

        # build array
        self._z_array = [redshift(store, z) for z in z_array]

    def __iter__(self):
        for z in self._z_array:
            yield z

    def __getitem__(self, key):
        return self._z_array[key]
