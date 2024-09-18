from Datastore import DatastoreObject


class version(DatastoreObject):
    def __init__(self, store_id: int, label: str):
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def __str__(self):
        return self._label
