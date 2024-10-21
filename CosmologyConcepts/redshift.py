from functools import total_ordering
from typing import Iterable, Self

from Datastore import DatastoreObject
from defaults import DEFAULT_FLOAT_PRECISION


@total_ordering
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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.store_id == other.store_id

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.z < other.z

    def __hash__(self):
        return ("redshift", self.store_id).__hash__()


class redshift_array:
    def __init__(self, z_array: Iterable[redshift]):
        """
        Reppresents an array of redshifts
        :param store_id: unique Datastore id. Should not be None
        :param z_array: array of redshift value
        """
        # store array, sorted in descending order of redshift;
        # the conversion to set ensure that we remove any duplicates
        self._z_array = sorted(set(z_array), key=lambda x: x.z, reverse=True)

        # sort into descending order of redshift

    def __iter__(self):
        for z in self._z_array:
            yield z

    def __getitem__(self, key):
        return self._z_array[key]

    def __len__(self):
        return len(self._z_array)

    def __eq__(self, other):
        return not self.__ne__(other)

    def __ne__(self, other):
        if len(self._z_array) != len(other._z_array):
            return True

        if any(za != zb for za, zb in zip(self._z_array, other._z_array)):
            return True

        return False

    def __add__(self, other):
        full_set = set(self._z_array)
        full_set.update(set(other._z_array))
        return redshift_array(full_set)

    def as_float_list(self) -> list[float]:
        return [float(z) for z in self._z_array]

    @property
    def max(self) -> redshift:
        return self._z_array[0]

    @property
    def min(self) -> redshift:
        return self._z_array[-1]

    def truncate(self, z_limit, keep: str = "lower") -> Self:
        if keep == "lower":
            return self._truncate_lower(z_limit)
        if keep == "higher":
            return self._truncate_higher(z_limit)
        if keep == "lower-strict":
            return self._truncate_lower_strict(z_limit)
        if keep == "higher-strict":
            return self._truncate_higher_strict(z_limit)

        raise ValueError(f'Unknown truncation mode "{keep}')

    def _truncate_lower(self, max_z) -> Self:
        if isinstance(max_z, redshift):
            return redshift_array(
                z_array=[
                    z
                    for z in self._z_array
                    if z.z <= max_z.z + DEFAULT_FLOAT_PRECISION
                    or z.store_id == max_z.store_id
                ]
            )

        return redshift_array(
            z_array=[z for z in self._z_array if z.z <= max_z + DEFAULT_FLOAT_PRECISION]
        )

    def _truncate_higher(self, min_z) -> Self:
        if isinstance(min_z, redshift):
            return redshift_array(
                z_array=[
                    z
                    for z in self._z_array
                    if z.z >= min_z.z - DEFAULT_FLOAT_PRECISION
                    or z.store_id == min_z.store_id
                ]
            )

        return redshift_array(
            z_array=[z for z in self._z_array if z.z >= min_z - DEFAULT_FLOAT_PRECISION]
        )

    def _truncate_lower_strict(self, max_z) -> Self:
        if isinstance(max_z, redshift):
            return redshift_array(
                z_array=[
                    z
                    for z in self._z_array
                    if z.z < max_z.z - DEFAULT_FLOAT_PRECISION
                    and z.store_id != max_z.store_id
                ]
            )

        return redshift_array(
            z_array=[z for z in self._z_array if z.z < max_z - DEFAULT_FLOAT_PRECISION]
        )

    def _truncate_higher_strict(self, min_z) -> Self:
        if isinstance(min_z, redshift):
            return redshift_array(
                z_array=[
                    z
                    for z in self._z_array
                    if z.z > min_z.z + DEFAULT_FLOAT_PRECISION
                    and z.store_id != min_z.store_id
                ]
            )

        return redshift_array(
            z_array=[z for z in self._z_array if z.z > min_z + DEFAULT_FLOAT_PRECISION]
        )
