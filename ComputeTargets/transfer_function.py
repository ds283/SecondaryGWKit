from math import log10

import numpy as np
from ray.actor import ActorHandle

from CosmologyConcepts import redshift_array, wavenumber
from CosmologyConcepts.wavenumber import wavenumber_exit_time
from CosmologyModels.base import CosmologyBase
from Datastore import DatastoreObject


class MatterTransferFunction(DatastoreObject):
    """
    Encapsulates a sample of the matter transfer function, labelled by a wavenumber k,
    and sampled over a range of redshifts
    """

    def __init__(
        self,
        store: ActorHandle,
        cosmology: CosmologyBase,
        k: wavenumber,
        z_samples: redshift_array,
    ):
        DatastoreObject.__init__(self, store)
        self._cosmology: CosmologyBase = cosmology

        # cache wavenumber and z-sample array
        self._k = k
        self._z_samples = z_samples

        # query datastore to find out whether the necessary sample values are available,
        # and schedule asynchronous tasks to compute any that are missing.

    @classmethod
    def populate_z_samples(
        cls,
        store: ActorHandle,
        cosmology: CosmologyBase,
        k: wavenumber,
        outside_horizon_efolds: float = 10.0,
        samples_per_log10z: int = 100,
        z_end: float = 0.1,
    ):
        """
        Determine the approximate starting redshift, if we wish to begin the calculation when
        the mode k is a fixed number of e-folds outside the horizon, and returns
        a MatterTransferFunction instance populated with an appropriate z-sample
        array. The array using a fixed number of sample points per log10 interval of redshift.

        The calculation of the initial redshift assumes that horizon crossing occurs during
        :param store: handle to datastore actor
        :param cosmology: cosmology models satisfying the CosmologyBase concept
        :param k: the wavenumber to sample
        :param outside_horizon_efolds: number of superhorizon efolds required in the initial condition
        :param samples_per_log10z: number of points to sample per log10 interval of redshift
        :return: MatterTransferFunction instance
        """
        # find horizon-crossing time for wavenumber z
        z_exit = wavenumber_exit_time(store, k, cosmology)

        # add on any specified number of superhorizon e-folds
        if outside_horizon_efolds is not None:
            z_init = z_exit.z_exit + outside_horizon_efolds

        print(
            f"horizon-crossing time +{outside_horizon_efolds} for wavenumber k = {k.k_inv_Mpc}/Mpc is z_exit = {z_init}"
        )

        # now we want to sample to transfer function between z_init and z = 0, using the specified
        # number of redshift sample points
        num_z_samples = int(round(samples_per_log10z * log10(z_init) + 0.5, 0))
        z_samples = redshift_array(
            store,
            np.logspace(log10(z_init), log10(z_end), num=num_z_samples),
        )

        print(
            f"-- using N = {num_z_samples} redshift sample points to represent transfer function"
        )

        return cls(store, cosmology, k, z_samples)
