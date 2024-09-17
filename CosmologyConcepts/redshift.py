from typing import Iterable, Self

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
        return float(self.z)


class redshift_array:
    def __init__(self, z_array: Iterable[redshift]):
        """
        Reppresents an array of redshifts
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

    def __len__(self):
        return len(self._z_array)

    def as_list(self) -> list[float]:
        return [float(z) for z in self._z_array]

    @property
    def max(self) -> redshift:
        z_max = None
        z_max_item = None

        for z in self._z_array:
            if z_max is None or z.z > z_max:
                z_max = z.z
                z_max_item = z

        return z_max_item

    @property
    def min(self) -> redshift:
        z_min = None
        z_min_item = None

        for z in self._z_array:
            if z_min is None or z.z < z_min:
                z_min = z.z
                z_min_item = z

        return z_min_item

    def truncate(self, max_z) -> Self:
        if isinstance(max_z, redshift):
            max_z_value = max_z.z
        else:
            max_z_value = float(max_z)

        new_z_array = [z for z in self._z_array if z.z <= max_z_value]
        return redshift_array(z_array=new_z_array)
