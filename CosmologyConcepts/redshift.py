import hashlib
import struct
from functools import total_ordering
from typing import Iterable, Optional, Self

from Datastore import DatastoreObject
from config.defaults import DEFAULT_FLOAT_PRECISION

# Number of hexadecimal characters in the digest that identifies a sample grid. Four bytes is
# enough that the handful of grids a datastore ever holds will not collide (two grids collide
# with probability ~1e-9 each), and short enough that the tag it goes into stays readable.
REDSHIFT_GRID_DIGEST_CHARS = 8


def redshift_grid_digest(
    z_values: Iterable, chars: int = REDSHIFT_GRID_DIGEST_CHARS
) -> str:
    """
    A short hex digest of a redshift grid, computed over the **exact bits** of its values in the
    order given.

    This exists because ``main.py``'s grid tags were ``SourceRedshiftGrid_{len}`` and
    ``ResponseRedshiftGrid_{len}``, which label *size only* (audit section 7). That was harmless
    while the grid was a pure function of ``(z_init, z_end, samples_per_log10z)``, since two runs
    with the same length then had the same grid; it stops being harmless the moment the grid also
    depends on what the cosmology declares, because two different grids of equal length then
    collide in the datastore and objects computed on one are silently served for the other.

    Digesting the values themselves, rather than the construction parameters, is deliberate: it
    cannot fall out of step with the grid the way a parameter list can, and it distinguishes two
    grids that differ in *any* sample, including in the protected set alone.

    :param z_values: the grid, as floats or as objects castable to float, in grid order
    :param chars: length of the returned digest
    """
    digest = hashlib.blake2b(digest_size=(chars + 1) // 2)
    for z in z_values:
        digest.update(struct.pack("<d", float(z)))
    return digest.hexdigest()[:chars]


@total_ordering
class redshift(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: float,
        is_source: bool = False,
        is_response: bool = False,
    ):
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

        self.is_source = is_source
        self.is_response = is_response

    def __float__(self):
        """
        Cast to float. Returns numerical value.
        :return:
        """
        return float(self.z)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.store_id == other.store_id

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

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
        if keep == "lower-include":
            return self._truncate_lower_include(z_limit)
        if keep == "higher-include":
            return self._truncate_higher_include(z_limit)

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

    def _truncate_lower_include(self, max_z) -> Self:
        include_array = []
        included_endpoint = False
        for z in self._z_array:
            if isinstance(max_z, redshift):
                if (
                    z.z <= max_z.z + DEFAULT_FLOAT_PRECISION
                    or z.store_id == max_z.store_id
                ):
                    include_array.append(z)
                elif not included_endpoint:
                    include_array.append(z)
                    included_endpoint = True
            else:
                if z.z <= max_z + DEFAULT_FLOAT_PRECISION:
                    include_array.append(z)
                elif not included_endpoint:
                    include_array.append(z)
                    included_endpoint = True

        return redshift_array(include_array)

    def _truncate_higher_include(self, max_z) -> Self:
        include_array = []
        included_endpoint = False
        for z in self._z_array:
            if isinstance(max_z, redshift):
                if (
                    z.z >= max_z.z - DEFAULT_FLOAT_PRECISION
                    or z.store_id == max_z.store_id
                ):
                    include_array.append(z)
                elif not included_endpoint:
                    include_array.append(z)
                    included_endpoint = True
            else:
                if z.z >= max_z - DEFAULT_FLOAT_PRECISION:
                    include_array.append(z)
                elif not included_endpoint:
                    include_array.append(z)
                    included_endpoint = True

        return redshift_array(include_array)

    def winnow(self, sparseness: int, protect: Optional[Iterable] = None) -> Self:
        """
        Decimate the array by keeping every ``sparseness``-th element, anchored at the lowest
        redshift, and then restoring every element of ``protect``.

        The stride is unchanged: ``self._z_array`` is descending, so ``[::-sparseness]`` walks
        upwards from the smallest z, which is what keeps the response grid reaching the bottom of
        the source grid. What is new is ``protect`` -- the points the cosmology asked to have
        resolved (``CosmologyConcepts.wavenumber.SourceGrid.protected_z``), which a blind stride
        would drop with probability ``1 - 1/sparseness``. Matching is on ``store_id``, not on the
        redshift value, so nothing here compares two recovered redshifts for equality
        (CLAUDE.md's redshift rule), and the result is a subset of ``self`` by construction:
        every element of ``protect`` that is not already in this array is ignored.

        :param sparseness: keep one element in ``sparseness``
        :param protect: redshifts that must be kept whether or not the stride lands on them
        """
        sparseness = int(sparseness)
        if sparseness <= 0:
            raise ValueError("sparseness must be greater than zero")

        kept = list(self._z_array[::-sparseness])
        if protect is not None:
            protected_ids = {z.store_id for z in protect}
            if len(protected_ids) > 0:
                kept.extend(z for z in self._z_array if z.store_id in protected_ids)

        # redshift_array's constructor de-duplicates (redshift hashes on store_id) and re-sorts
        return redshift_array(z_array=kept)

    def digest(self, chars: int = REDSHIFT_GRID_DIGEST_CHARS) -> str:
        """
        A short hex digest identifying this grid; see :func:`redshift_grid_digest`.
        """
        return redshift_grid_digest(self._z_array, chars=chars)


def check_zsample(A, B):
    A_sample: redshift_array = A if isinstance(A, redshift_array) else A.z_sample
    B_sample: redshift_array = B if isinstance(B, redshift_array) else B.z_sample

    if A_sample != B_sample:
        raise RuntimeError("Redshift sample grids are not equal")
