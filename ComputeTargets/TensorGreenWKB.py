from typing import Optional, List

from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from utilities import check_units


def WKB_omegaEff_sq(cosmology: BaseCosmology, k: float, z: float):
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    H = cosmology.Hubble(z)
    eps = cosmology.epsilon(z)
    epsPrime = cosmology.d_epsilon_dz(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    A = k_over_H_2
    B = -epsPrime / 2.0 / one_plus_z
    C = (3.0 * eps / 2.0 - eps * eps / 4.0 - 2.0) / one_plus_z_2

    return A + B + C


def WKB_d_ln_omegaEffPrime_dz(cosmology: BaseCosmology, k: float, z: float):
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z
    one_plus_z_3 = one_plus_z_2 * one_plus_z

    eps = cosmology.epsilon(z)
    epsPrime = cosmology.d_epsilon_dz(z)
    epsPrimePrime = cosmology.d2_epsilon_dz2(z)

    omega_eff_sq = WKB_omegaEff_sq(cosmology, k, z)

    A = -epsPrimePrime / 2.0 / one_plus_z
    B = (2.0 * epsPrimePrime - eps * epsPrime / 2.0) / one_plus_z_2
    C = (3.0 * eps - eps * eps / 2.0 - 4.0) / one_plus_z_3

    numerator = A + B + C
    denominator = 2.0 * omega_eff_sq

    return numerator / denominator


class TensorGreenWKB(DatastoreObject):
    """
    Encapsulates all sample points produced for a calculation of the WKB
    phase associated with the tensor Green's function
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        cosmology: BaseCosmology,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_source: Optional[redshift] = None,
        z_sample: Optional[redshift_array] = None,
        G_init: Optional[float] = 0.0,
        Gprime_init: Optional[float] = 1.0,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, cosmology)

        self._solver_labels = solver_labels
        self._z_sample = z_sample

        self._G_init = G_init
        self._Gprime_init = Gprime_init

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None
            self._compute_steps = None
            self._RHS_evaluations = None
            self._mean_RHS_time = None
            self._max_RHS_time = None
            self._min_RHS_time = None

            self._init_efolds_suph = None

            self._solver = None

            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]
            self._compute_steps = payload["compute_steps"]
            self._RHS_evaluations = payload["RHS_evaluations"]
            self._mean_RHS_time = payload["mean_RHS_time"]
            self._max_RHS_time = payload["max_RHS_time"]
            self._min_RHS_time = payload["min_RHS_time"]

            self._init_efolds_subh = payload["init_efolds_suph"]

            self._solver = payload["solver"]

            self._values = payload["values"]
