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
    def d_lnH_dz(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def d2_lnH_dz2(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def d3_lnH_dz3(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def wBackground(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def wPerturbations(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def d_wPerturbations_dz(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def d2_wPerturbations_dz2(self, z: float) -> float:
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
