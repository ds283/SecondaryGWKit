from abc import ABC, abstractmethod

from Datastore import DatastoreObject
from Units.base import UnitsLike


class BaseCosmology(DatastoreObject, ABC):
    def __init__(self, store_id: int):
        DatastoreObject.__init__(self, store_id)
        # no constructor for ABC

    @property
    @abstractmethod
    def type_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def units(self) -> UnitsLike:
        raise NotImplementedError

    @property
    @abstractmethod
    def H0(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def rho(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def Hubble(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def T_photon(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def wBackground(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def wPerturbations(self, z: float) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def z_matter_radiation_equality(self) -> float:
        """
        The redshift at which this cosmology's matter and radiation energy densities are equal,
        **as this cosmology computes it**.

        The model is authoritative for this number and no consumer may compute it instead. In
        particular a consumer must not fall back on the radiation-domination closed form
        1 + z_eq = Omega_m/Omega_r: Omega_r is a *present-day* density parameter, so that form is
        exact only while rho_r ~ (1+z)^4 holds all the way from today back to equality. For a
        cosmology with no equation of state that holds by construction and the closed form is the
        answer; for one whose relativistic content changes -- entropy injection below z_eq, a
        decaying species, extra relativistic degrees of freedom appearing late -- it does not, and
        it fails silently and by much more than rounding.

        **A subclass for which the closed form is not exact must not return it.** This declares
        the obligation; it does not police it. A subclass that chooses to answer incoherently has
        made its own problem.

        :return: the matter-radiation equality redshift
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def z_matter_lambda_equality(self) -> float:
        """
        The redshift at which this cosmology's matter and cosmological-constant energy densities
        are equal, **as this cosmology computes it**.

        The same contract as :meth:`z_matter_radiation_equality`, and the same prohibition on a
        consumer computing it instead. This pair is the easier of the two -- rho_m/rho_Lambda is
        rho_m0 (1+z)^3 over a constant, with no temperature dependence at all, so
        (Omega_Lambda/Omega_m)^(1/3) - 1 is exact on any equation of state -- but a subclass that
        departs from that form is still the authority on its own answer, and nothing here assumes
        which form it uses.

        :return: the matter-Lambda equality redshift
        """
        raise NotImplementedError


def check_cosmology(A, B):
    """
    Check that object A and B are defined with the same cosmology
    Assumes that both provide a .cosmology property that returns a BaseCosmology object
    :param A:
    :param B:
    :return:
    """
    A_cosmology: BaseCosmology = A if isinstance(A, BaseCosmology) else A.cosmology
    B_cosmology: BaseCosmology = B if isinstance(A, BaseCosmology) else B.cosmology

    if A_cosmology.store_id != B_cosmology.store_id:
        raise RuntimeError("Cosmology store_ids are different")
