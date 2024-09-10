from Datastore import DatastoreObject


class IntegrationSolver(DatastoreObject):
    def __init__(self, store_id: int, label: str, stepping: int):
        """
        Construct a datastore-backed object representing a named
        integration strategy (such as "solve_ivp+RK45")
        :param store_id: unique Datastore id. Should not be None

        :param label:
        """
        DatastoreObject.__init__(self, store_id)

        self._label = label
        self._stepping = stepping if stepping >= 0 else 0

    @property
    def stepping(self) -> int:
        return self._stepping
