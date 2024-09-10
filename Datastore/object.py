from typing import Optional

from ray.actor import ActorHandle


class DatastoreObject:
    """
    Represent an object that can be serialized in a datastore
    """

    def __init__(self, store_id: Optional[int]):
        self._my_id = store_id

    @property
    def store_id(self) -> int:
        if self._my_id is None:
            raise RuntimeError("Attempt to read datastore id before it has been set")

        return self._my_id
