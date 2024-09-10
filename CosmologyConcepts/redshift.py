from typing import Iterable

from Datastore import DatastoreObject


class redshift(DatastoreObject):
    def __init__(self, store_id: int, z: float):
        """
        Represents a redshift,
        e.g., used to sample a transfer function or power spectrum
        :param store_id: unique Datastore id. Should not be None
        :param z: redshift value
        """
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        self.z = z

    def __float__(self):
        """
        Cast to float. Returns numerical value.
        :return:
        """
        return self.z


class redshift_array:
    def __init__(self, z_array: Iterable[redshift]):
        """
        Construct a datastore-backed object representing an array of redshifts
        :param store_id: unique Datastore id. Should not be None
        :param z_array: array of redshift value
        """
        # store array
        self._z_array = z_array

    def __iter__(self):
        for z in self._z_array:
            yield z

    def __getitem__(self, key):
        return self._z_array[key]

    def as_list(self) -> list[float]:
        return [float(z) for z in self._z_array]
