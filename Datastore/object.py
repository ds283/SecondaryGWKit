from ray.actor import ActorHandle


class DatastoreObject:
    """
    Represent an object that can be serialized in a datastore
    """

    def __init__(self, store: ActorHandle):
        self._store = store
        self._my_id = None

    @property
    def store_id(self) -> int:
        if self._my_id is None:
            raise RuntimeError("Attempt to read datastore id before it has been set")

        return self._my_id
