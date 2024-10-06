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
