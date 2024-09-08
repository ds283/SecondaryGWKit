import ray
from ray.actor import ActorHandle

import sqlalchemy as sqla
from sqlalchemy import func, and_
from Datastore import DatastoreObject, Datastore
from defaults import DEFAULT_STRING_LENGTH


class IntegrationSolver(DatastoreObject):
    def __init__(self, store: ActorHandle, label: str, stepping: int):
        """
        Construct a datastore-backed object representing a named
        integration strategy (such as "solve_ivp+RK45")
        :param store:
        :param label:
        """
        DatastoreObject.__init__(self, store)

        self._label = label
        self._stepping = stepping if stepping >= 0 else 0

        # request our own unique id from the datastore
        self._my_id = ray.get(self._store.query.remote(self))

    def build_storage_payload(self):
        return {"label": self._label}
