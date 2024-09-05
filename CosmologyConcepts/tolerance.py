import ray
from ray.actor import ActorHandle

from math import log10, pow

import sqlalchemy as sqla
from sqlalchemy import func

from Datastore import DatastoreObject
from defaults import DEFAULT_FLOAT_PRECISION


class tolerance(DatastoreObject):
    def __init__(self, store: ActorHandle, **kwargs):
        """
        Construct a datastore-backed object representing a tolerance (absolute or relative).
        Effectively tokenizes a floating point number to an integer.
        :param store: handle to datastore actor
        :param tol: tolerance
        """
        DatastoreObject.__init__(self, store)

        if "log10_tol" in kwargs:
            log10_tol = kwargs["log10_tol"]
            self.log10_tol = log10_tol
            self.tol = pow(10.0, log10_tol)
        elif "tol" in kwargs:
            tol = kwargs["tol"]
            self.tol = tol
            self.log10_tol = log10(tol)
        else:
            raise RuntimeError(
                'Neither "tol" nor "log10_tol" was supplied to tolerance() constructor'
            )

        # request our own unique id from the datastore, unless a store id was provided, in which case we circumvent going out to the database
        self._my_id = ray.get(self._store.query.remote(self))

    def __float__(self):
        """
        Cast to float.
        :return:
        """
        return self.tol

    @staticmethod
    def generate_columns():
        return {"version": False, "timestamp": True, "columns": [sqla.Column("log10_tol", sqla.Float(64))]}

    def build_query(self, table, query):
        return query.filter(
            func.abs(table.c.log10_tol - self.log10_tol) < DEFAULT_FLOAT_PRECISION
        )

    def build_storage_payload(self):
        return {"log10_tol": self.log10_tol}
