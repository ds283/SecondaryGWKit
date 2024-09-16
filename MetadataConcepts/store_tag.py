from Datastore import DatastoreObject


class store_tag(DatastoreObject):
    def __init__(self, store_id: int, label: str):
        """
        Construct a Datastore-backed object that is used to tag other objects held in the Datastore.
        The initial use case is to identify integrations that have sufficiently dense z-sampling
        to be used for particular purposes - we need a way to tell an integration of one type from
        another.
        """
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def __str__(self):
        return self._label
